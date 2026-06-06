"""Benchmark DataLoader num_workers on Mac/MPS for the fire dataset (one-off check).

macOS uses 'spawn' for workers -> each re-imports this module and the zarr-backed dataset
must pickle/re-open per worker. This measures whether num_workers>0 actually helps or
(as observed) hurts. Run: python scripts/bench_workers.py
"""
import sys
import time
from pathlib import Path

import xarray as xr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import RegimeIberFireDataset
from src.data.features import build_segmentation_features

CUBE = "data/gold/IberFire_coarse4.zarr"
STATS = "stats/coarse4_norm_stats_train.json"


def main():
    feats = build_segmentation_features(xr.open_zarr(CUBE, consolidated=True).data_vars)
    ds = RegimeIberFireDataset(zarr_path=CUBE, time_start="2015-01-01", time_end="2016-12-31",
                               feature_vars=feats, label_var="is_fire", lead_time=1,
                               compute_stats=False, stats_path=STATS, mode="all")
    N = 15
    for nw in (0, 2, 4):
        try:
            dl = DataLoader(ds, batch_size=2, num_workers=nw, shuffle=True,
                            persistent_workers=(nw > 0))
            it = iter(dl)
            next(it)  # warmup (absorb worker spawn)
            t = time.time()
            for k, _ in enumerate(it):
                if k + 1 >= N:
                    break
            dt = time.time() - t
            print(f"  num_workers={nw}: {dt:.2f}s / {N} batches -> {N/dt:.2f} batch/s", flush=True)
        except Exception as e:
            print(f"  num_workers={nw}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
