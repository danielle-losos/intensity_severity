#!/usr/bin/env python3
"""Create annual CBI rasters resampled to 375 m in EPSG:5070."""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject


DEFAULT_INPUT = "merged_pnw_firedv2_bccbi.tif"
DEFAULT_START_YEAR = 2012
DEFAULT_END_YEAR = 2024
DEFAULT_CRS = "EPSG:5070"
DEFAULT_RES = 375


def _build_band_lookup(src: rasterio.DatasetReader) -> dict[int, int]:
    """Map year to source band index using band descriptions like year_2012."""
    lookup: dict[int, int] = {}
    for i, name in enumerate(src.descriptions, start=1):
        if not name:
            continue
        if name.startswith("year_"):
            try:
                year = int(name.split("_")[1])
                lookup[year] = i
            except (IndexError, ValueError):
                continue
    return lookup


def create_annual_cbi_rasters(
    input_raster: Path,
    output_dir: Path,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    target_crs: str = DEFAULT_CRS,
    target_res: int = DEFAULT_RES,
) -> None:
    """Export one CBI raster per year at the requested CRS/resolution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_raster) as src:
        band_lookup = _build_band_lookup(src)

        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_res,
        )

        for year in range(start_year, end_year + 1):
            if year not in band_lookup:
                print(f"⚠️ Year {year} not found in source band descriptions; skipping.")
                continue

            band_index = band_lookup[year]
            cbi_data = src.read(band_index)
            out_path = output_dir / f"{year}_pnw_firedv2_bccbi_epsg5070.tif"

            meta = src.meta.copy()
            meta.update(
                {
                    "count": 1,
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "nodata": src.nodata,
                }
            )

            with rasterio.open(out_path, "w", **meta) as dst:
                reproject(
                    source=cbi_data,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                )
                dst.set_band_description(1, f"CBI_{year}")

            print(f"✅ Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create annual CBI rasters for years 2012-2024 from "
            "merged_pnw_firedv2_bccbi.tif at 375 m in EPSG:5070"
        )
    )
    parser.add_argument("--input-raster", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--target-crs", default=DEFAULT_CRS)
    parser.add_argument("--target-res", type=int, default=DEFAULT_RES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_annual_cbi_rasters(
        input_raster=args.input_raster,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        target_crs=args.target_crs,
        target_res=args.target_res,
    )


if __name__ == "__main__":
    main()
