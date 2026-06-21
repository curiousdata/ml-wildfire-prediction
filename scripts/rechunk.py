"""Rechunk a Zarr store to new chunk sizes (writes a derived copy).

Typical use: build a TRAINING-optimized cube with large time chunks (e.g. --time 200) so a block of
consecutive days reads as one chunk per variable instead of one-per-day — the access pattern GBT training
wants. Spatial dims default to a single full chunk (-1). Static vars (no time dim) ignore --time.

Example (FGDC gold -> 200-day time blocks, lz4 for fast read-back):
  python scripts/rechunk.py -i data/gold/FireGuard_coarse4.zarr -o data/gold/FireGuard_coarse4_t200.zarr \\
      --time 200 --cname lz4 --overwrite
"""
import argparse
import shutil
import time
from pathlib import Path

import xarray as xr
from numcodecs import Blosc
from dask.diagnostics import ProgressBar


def main():
    ap = argparse.ArgumentParser(description="Rechunk a Zarr store to new chunk sizes.")
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--output", required=True, type=Path)
    ap.add_argument("--time", required=True, type=int, help="time chunk (days per chunk)")
    ap.add_argument("--y", type=int, default=-1, help="y chunk (-1 = full extent; default)")
    ap.add_argument("--x", type=int, default=-1, help="x chunk (-1 = full extent; default)")
    ap.add_argument("--cname", default="zstd", help="Blosc codec: zstd (default) | lz4 (faster read-back)")
    ap.add_argument("--clevel", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    chunks = {"time": args.time, "y": args.y, "x": args.x}
    compressor = Blosc(cname=args.cname, clevel=args.clevel, shuffle=2)

    if not args.input.exists():
        raise FileNotFoundError(f"Source Zarr not found: {args.input}")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists — pass --overwrite.")
        shutil.rmtree(args.output)

    print(f"Source: {args.input}\nTarget: {args.output}\nChunks: {chunks}\nCodec:  {compressor}\n")

    t0 = time.time()
    ds = xr.open_zarr(args.input, consolidated=True)
    print(f"opened {dict(ds.sizes)}, {len(ds.data_vars)} vars in {time.time() - t0:.1f}s")

    ds = ds.chunk(chunks)
    encoding = {v: {"compressor": compressor} for v in ds.data_vars}

    t1 = time.time()
    with ProgressBar():
        # zarr_format=2: zarr 3.x defaults to v3, which rejects the numcodecs.Blosc compressor (CLAUDE.md).
        ds.to_zarr(args.output, mode="w", encoding=encoding, consolidated=True, zarr_format=2)
    print(f"written in {(time.time() - t1) / 60:.1f} min")

    check = xr.open_zarr(args.output, consolidated=True)
    sample = list(check.data_vars)[0]
    print(f"verify: {sample} chunks = {check[sample].data.chunks}")


if __name__ == "__main__":
    main()
