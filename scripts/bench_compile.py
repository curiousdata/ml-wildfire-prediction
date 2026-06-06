"""Benchmark torch.compile modes vs eager on the v5 wide-deep U-Net (MPS, batch 8).

torch.compile has historically HUNG for hours on MPS, so each mode runs in an isolated
subprocess with a hard timeout (this script re-invokes itself with --single MODE). A hang is
killed and reported as HANG/TIMEOUT instead of blocking. Measures first-step time (includes
compilation) and steady-state median step time (forward + backward + optimizer step).
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODES = ["eager", "aot_eager", "default", "reduce-overhead"]
TIMEOUT = 180          # per-mode wall-clock budget; a compile slower than this is unusable here
B, C, H, W = 8, 146, 230, 297   # match v5 training (batch 8, 4 km grid)
N_STEPS = 8


def run_single(mode: str) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models.cnn import build_wide_deep_unet

    dev = torch.device("mps")
    torch.manual_seed(0)
    model = build_wide_deep_unet(in_channels=C, encoder_name="resnet34",
                                 encoder_weights=None, norm="group", wide_dropout=0.1).to(dev)
    if mode == "eager":
        run = model
    elif mode == "aot_eager":
        run = torch.compile(model, backend="aot_eager")
    elif mode == "default":
        run = torch.compile(model)
    elif mode == "reduce-overhead":
        run = torch.compile(model, mode="reduce-overhead")
    else:
        raise ValueError(mode)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(B, C, H, W, device=dev)
    y = torch.zeros(B, 1, H, W, device=dev)

    def sync():
        if hasattr(torch, "mps"):
            torch.mps.synchronize()

    def step():
        opt.zero_grad()
        loss = lossf(run(x), y)
        loss.backward()
        opt.step()
        sync()

    t0 = time.time(); step(); t_first = time.time() - t0   # includes compile
    times = []
    for _ in range(N_STEPS):
        t = time.time(); step(); times.append(time.time() - t)
    print(f"RESULT mode={mode} first_step={t_first:.1f}s steady_median={statistics.median(times):.3f}s "
          f"steady_min={min(times):.3f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single")
    args = ap.parse_args()
    if args.single:
        run_single(args.single)
        return
    print(f"benchmarking {MODES} on wide-deep U-Net, batch {B}, {H}x{W}, timeout {TIMEOUT}s/mode", flush=True)
    for mode in MODES:
        print(f"--- mode={mode} ---", flush=True)
        try:
            r = subprocess.run([sys.executable, __file__, "--single", mode],
                               timeout=TIMEOUT, capture_output=True, text=True)
            lines = (r.stdout + r.stderr).strip().splitlines()
            res = [ln for ln in lines if ln.startswith("RESULT")]
            if res:
                print(res[0], flush=True)
            else:
                tail = " | ".join(lines[-4:]) if lines else "no output"
                print(f"RESULT mode={mode} FAILED (rc={r.returncode}): {tail[:300]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"RESULT mode={mode} HANG/TIMEOUT (killed at {TIMEOUT}s) — compile never finished", flush=True)


if __name__ == "__main__":
    main()
