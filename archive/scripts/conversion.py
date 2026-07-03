"""
Convert the IberFire NetCDF dataset to Zarr format for faster preprocessing.

This is a ONE-TIME (per configuration) conversion that may take a long time
for a ~730 GB dataset, but once done, Zarr will generally allow much faster
and more flexible preprocessing.

Typical usage (from project root):

    python scripts/conversion.py \\
        --netcdf-path data/bronze/IberFire.nc \\
        --zarr-path  data/silver/IberFire.zarr \\
        --force

You can omit the flags to use the default paths shown above.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import shutil

import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

# Input NetCDF file (raw IberFire datacube)
DEFAULT_NETCDF_PATH = Path("data/bronze/IberFire.nc")

# Output Zarr directory (shared silver dataset)
DEFAULT_ZARR_PATH = Path("data/silver/IberFire.zarr")

# Chunking strategy (optimized for per-day full-image access)
# time=1, full spatial dims → one chunk per day, matching U-Net access pattern.
CHUNKS: Dict[str, int] = {
    "time": 1,
    "y": -1,  # -1 means "full dimension" in xarray chunking
    "x": -1,
}

# Compression settings
# zstd with moderate level provides a good speed/size trade-off for local SSD.
COMPRESSOR = Blosc(
    cname="zstd",   # zstd: good compression + fast decompression
    clevel=3,       # level 3: balanced speed/compression
    shuffle=2,      # bit-shuffle: better for floating point
)


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------

@dataclass
class ConversionConfig:
    """Configuration for a single NetCDF → Zarr conversion run."""

    netcdf_path: Path
    zarr_path: Path
    force: bool
    verify: bool


def parse_args(argv: list[str] | None = None) -> ConversionConfig:
    """Parse command-line arguments into a ConversionConfig."""
    parser = argparse.ArgumentParser(
        description="Convert IberFire NetCDF dataset to Zarr format.",
    )
    parser.add_argument(
        "--netcdf-path",
        type=Path,
        default=DEFAULT_NETCDF_PATH,
        help=f"Path to input NetCDF file (default: {DEFAULT_NETCDF_PATH}).",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=DEFAULT_ZARR_PATH,
        help=f"Path to output Zarr directory (default: {DEFAULT_ZARR_PATH}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Zarr directory without prompting.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip final verification step (open & basic checks).",
    )

    args = parser.parse_args(argv)

    return ConversionConfig(
        netcdf_path=args.netcdf_path,
        zarr_path=args.zarr_path,
        force=bool(args.force),
        verify=not bool(args.no_verify),
    )


# ---------------------------------------------------------------------------
# Core conversion steps
# ---------------------------------------------------------------------------

def ensure_input_exists(path: Path) -> None:
    """Ensure the input NetCDF file exists, otherwise exit with an error."""
    if not path.exists():
        print(f"ERROR: Input NetCDF file not found: {path}")
        print(f"       Current working directory: {Path.cwd()}")
        sys.exit(1)


def prepare_output(path: Path, force: bool) -> None:
    """
    Ensure output directory is ready to use.

    If it exists and force is False, abort with a clear error.
    If it exists and force is True, delete it recursively.
    """
    if not path.exists():
        return

    if not force:
        print(
            f"ERROR: Output Zarr directory already exists: {path}\n"
            f"       Use --force to overwrite it."
        )
        sys.exit(1)

    print(f"Removing existing Zarr directory: {path}")
    shutil.rmtree(path)


def open_netcdf(path: Path) -> xr.Dataset:
    """Open the input NetCDF dataset and log basic info."""
    print("Step 1/4: Opening NetCDF dataset...")
    start = time.time()
    try:
        ds = xr.open_dataset(
            path,
            engine="h5netcdf",
            decode_times=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR opening NetCDF: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"  Opened in {elapsed:.1f}s")
    print(f"  Variables : {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Estimated size in memory: {ds.nbytes / 1e9:.1f} GB\n")
    return ds


def downcast_floats(ds: xr.Dataset) -> xr.Dataset:
    """
    Downcast float64 → float32 to save space, leaving ints/bools as-is.

    This is a safe optimization for the IberFire dataset that reduces disk
    usage without materially affecting model performance.
    """
    print("Step 2/4: Downcasting float64 variables to float32 (if any)...")
    converted = 0
    for var in ds.data_vars:
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")
            converted += 1

    if converted > 0:
        print(f"  Downcasting complete: converted {converted} variable(s).\n")
    else:
        print("  No float64 variables found. Skipping downcasting.\n")
    return ds


def rechunk_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Rechunk dataset according to CHUNKS.

    CHUNKS is chosen to match the main access pattern used in this project:
    per-day, full-spatial U-Net training (time=1, y/x full).
    """
    print(f"Step 3/4: Rechunking with {CHUNKS}...")
    start = time.time()
    try:
        ds_rechunked = ds.chunk(CHUNKS)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR rechunking: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"  Rechunked in {elapsed:.1f}s")

    first_var = list(ds_rechunked.data_vars)[0]
    da = ds_rechunked[first_var]
    chunks_info = getattr(da.data, "chunks", None)
    if chunks_info is not None:
        print(f"  Example chunks for '{first_var}': {chunks_info}\n")
    else:
        print("  (No chunk information available for first variable)\n")
    return ds_rechunked


def write_zarr(ds: xr.Dataset, zarr_path: Path) -> None:
    """Write the rechunked dataset to a Zarr store with compression."""
    print("Step 4/4: Writing to Zarr (this is the slow part)...")
    print("  This may take a long time depending on disk and CPU.\n")
    start = time.time()

    # Prepare encoding for all variables
    encoding = {var: {"compressor": COMPRESSOR} for var in ds.data_vars}

    try:
        with ProgressBar():
            ds.to_zarr(
                zarr_path,
                mode="w",
                encoding=encoding,
                consolidated=True,  # create consolidated metadata for faster opens
            )
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR writing Zarr: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n  Written in {elapsed / 60:.1f} minutes ({elapsed / 3600:.2f} hours)")


def verify_zarr(zarr_path: Path) -> None:
    """Lightweight verification that the Zarr store is readable and consistent."""
    print("\nVerification: Opening Zarr store for basic checks...")
    try:
        ds_zarr = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not open written Zarr store: {e}")
        return

    print("  Zarr opens successfully.")
    print(f"  Variables : {list(ds_zarr.data_vars)}")
    print(f"  Dimensions: {dict(ds_zarr.dims)}")

    var = list(ds_zarr.data_vars)[0]
    da = ds_zarr[var]
    print(f"  Sample check ({var}): shape = {da.shape}, dtype = {da.dtype}")


def directory_size_gb(path: Path) -> float:
    """Compute total size in GB of all files under a directory."""
    if not path.exists():
        return 0.0
    total_bytes = sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    )
    return total_bytes / 1e9


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run a complete NetCDF → Zarr conversion according to CLI arguments."""
    print("NetCDF -> Zarr Conversion")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg = parse_args(argv)

    print(f"Input NetCDF : {cfg.netcdf_path}")
    print(f"Output Zarr  : {cfg.zarr_path}")
    print(f"Chunks       : {CHUNKS}")
    print(f"Compressor   : {COMPRESSOR}\n")

    ensure_input_exists(cfg.netcdf_path)
    prepare_output(cfg.zarr_path, force=cfg.force)

    ds = open_netcdf(cfg.netcdf_path)
    ds = downcast_floats(ds)
    ds_rechunked = rechunk_dataset(ds)
    write_zarr(ds_rechunked, cfg.zarr_path)

    if cfg.verify:
        verify_zarr(cfg.zarr_path)

    print("\nSize comparison:")
    netcdf_size_gb = cfg.netcdf_path.stat().st_size / 1e9
    zarr_size_gb = directory_size_gb(cfg.zarr_path)
    if netcdf_size_gb > 0:
        ratio = zarr_size_gb / netcdf_size_gb * 100
    else:
        ratio = 0.0
    print(f"  NetCDF: {netcdf_size_gb:.1f} GB")
    print(f"  Zarr  : {zarr_size_gb:.1f} GB ({ratio:.1f}% of original)")

    print("\nCONVERSION COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Zarr may be incomplete.")
        print("You can safely delete the partially written store and restart.")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"\n\nUNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)