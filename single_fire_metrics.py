#!/usr/bin/env python3
"""Generate per-fire VIIRS intensity/severity metrics and visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from rasterstats import zonal_stats


def duration_category(duration_days: float) -> str:
    """Classify fires into short vs long duration groups."""
    if pd.isna(duration_days):
        return "unknown"
    return "short" if duration_days < 7 else "long"


def get_cbi_path(cbi_dir: Path, fire_date: str) -> Path:
    year = pd.to_datetime(fire_date).year
    cbi_path = cbi_dir / f"{year}_pnw_firedv2_bccbi_epsg5070.tif"
    if not cbi_path.exists():
        raise FileNotFoundError(f"No CBI raster for year {year}: {cbi_path}")
    return cbi_path


def compute_single_fire_metrics(
    viirs_gpkg: Path,
    fire_id: int,
    fire_name: str,
    fire_date: str,
    cbi_dir: Path,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Build synchronized VIIRS intensity and CBI severity metrics."""
    viirs = gpd.read_file(viirs_gpkg)
    viirs_singlefire = viirs.loc[viirs["id"] == fire_id].copy()

    if viirs_singlefire.empty:
        raise ValueError(f"No VIIRS records found for id={fire_id}")

    summary = pd.DataFrame(
        [
            {
                "fire_name": fire_name,
                "fire_id": fire_id,
                "fire_date": fire_date,
                "n_cells": len(viirs_singlefire),
                "frp_norm_csum_mean": viirs_singlefire["frp_norm_csum"].mean(),
                "frp_norm_csum_median": viirs_singlefire["frp_norm_csum"].median(),
                "frp_norm_csum_min": viirs_singlefire["frp_norm_csum"].min(),
                "frp_norm_csum_max": viirs_singlefire["frp_norm_csum"].max(),
                "obs_duration_mean": viirs_singlefire["obs_duration"].mean(),
                "obs_duration_median": viirs_singlefire["obs_duration"].median(),
                "obs_duration_min": viirs_singlefire["obs_duration"].min(),
                "obs_duration_max": viirs_singlefire["obs_duration"].max(),
            }
        ]
    )

    cbi_path = get_cbi_path(cbi_dir, fire_date)
    with rasterio.open(cbi_path) as src:
        cbi_array = src.read(1, masked=True).filled(np.nan)
        cbi_transform = src.transform

    downsample_stats = zonal_stats(
        viirs_singlefire,
        cbi_array,
        affine=cbi_transform,
        stats=["mean", "median", "min", "max", "percentile_95", "std"],
        nodata=np.nan,
        all_touched=True,
    )

    viirs_singlefire["cbi_mean"] = [s["mean"] for s in downsample_stats]
    viirs_singlefire["cbi_stdev"] = [s["std"] for s in downsample_stats]
    viirs_singlefire["cbi_median"] = [s["median"] for s in downsample_stats]
    viirs_singlefire["cbi_min"] = [s["min"] for s in downsample_stats]
    viirs_singlefire["cbi_max"] = [s["max"] for s in downsample_stats]
    viirs_singlefire["cbi_95"] = [s["percentile_95"] for s in downsample_stats]

    viirs_singlefire["log_cfrp"] = np.log(viirs_singlefire["frp_norm_csum"] + 1)
    viirs_singlefire["duration_group"] = viirs_singlefire["obs_duration"].apply(duration_category)

    bins = [0, 500, 1000, 2000, viirs_singlefire["frp_norm_csum"].max()]
    labels = ["<500", "500-1000", "1000-2000", "2000+"]
    viirs_singlefire["intensity_class"] = pd.cut(
        viirs_singlefire["frp_norm_csum"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return viirs_singlefire, summary


def make_figures(df: gpd.GeoDataFrame, out_dir: Path, fire_name: str) -> None:
    """Create requested plots for synchronized intensity/severity data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    df.plot(column="log_cfrp", ax=axes[0], cmap="viridis", legend=True, linewidth=0)
    axes[0].set_title("log_cfrp")
    axes[0].set_axis_off()

    df.plot(column="cbi_mean", ax=axes[1], cmap="magma", legend=True, linewidth=0, vmin=0, vmax=3)
    axes[1].set_title("cbi_mean")
    axes[1].set_axis_off()

    df.plot(column="obs_duration", ax=axes[2], cmap="plasma", legend=True, linewidth=0)
    axes[2].set_title("obs_duration")
    axes[2].set_axis_off()

    maps_path = out_dir / f"{fire_name}_maps_logcfrp_cbi_duration.png"
    fig.savefig(maps_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df, x="log_cfrp", y="cbi_mean", hue="duration_group", alpha=0.6, ax=ax)
    ax.set_title("log_cfrp vs cbi_mean by duration")
    scatter_path = out_dir / f"{fire_name}_scatter_logcfrp_vs_cbimean.png"
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    gdf_clean = df.dropna(subset=["cbi_mean", "frp_norm_csum", "duration_group", "intensity_class"])
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=gdf_clean,
        x="intensity_class",
        y="cbi_mean",
        hue="duration_group",
        palette="viridis",
        ax=ax,
    )
    ax.set_ylim(0, 3)
    ax.set_xlabel("cFRP intensity class")
    ax.set_ylabel("cbi_mean")
    ax.set_title("CBI by cFRP class grouped by duration")
    boxplot_path = out_dir / f"{fire_name}_boxplot_cfrpclass_cbimean_duration.png"
    fig.savefig(boxplot_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create single-fire behavior metrics and plots.")
    parser.add_argument("--viirs-gpkg", type=Path, required=True)
    parser.add_argument("--fire-name", required=True)
    parser.add_argument("--fire-date", required=True, help="Fire date (YYYY-MM-DD). Year selects CBI raster.")
    parser.add_argument("--fire-id", type=int, required=True)
    parser.add_argument("--cbi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


args = parse_args()
gdf, summary = compute_single_fire_metrics(
    viirs_gpkg=args.viirs_gpkg,
    fire_id=args.fire_id,
    fire_name=args.fire_name,
    fire_date=args.fire_date,
    cbi_dir=args.cbi_dir,
)

args.output_dir.mkdir(parents=True, exist_ok=True)
safe_fire_name = args.fire_name.replace(" ", "_")
gdf.to_file(args.output_dir / f"{safe_fire_name}_viirs_singlefire_metrics.gpkg", driver="GPKG")
summary.to_csv(args.output_dir / f"{safe_fire_name}_summary_stats.csv", index=False)

make_figures(gdf, args.output_dir, safe_fire_name)

print("✅ Wrote outputs:")
print(f"  - {args.output_dir / f'{safe_fire_name}_viirs_singlefire_metrics.gpkg'}")
print(f"  - {args.output_dir / f'{safe_fire_name}_summary_stats.csv'}")
