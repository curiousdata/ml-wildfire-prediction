"""Backtest the LIVE serving pipeline vs the offline cube ceiling — the honest 'normal accuracy?' check.

The live feed never sees the true cube slice: it seasonally WARM-STARTS the features it can't refresh
(vegetation, soil, RH/wind/pressure aggregates) from a nearest-day-of-year slice of ANOTHER year, then
overwrites what it CAN refresh live — temperature + antecedent dryness from Open-Meteo, and fire from EFFIS
(the model's training source). This script measures how much accuracy that costs.

For each of a set of in-cube fire-season dates t (so we have true t+1 fire):
  * CUBE  prediction = GBT on the true cube features at t            (the offline ceiling, ~0.63 new-ign AP)
  * LIVE  prediction = GBT on build_live_slice(t):  seasonal warm-start (nearest-DOY, DIFFERENT year)
                       + Open-Meteo temp + Open-Meteo 90-day dryness + cube-truth fire
                       (fire_source='cube' because live EFFIS can't reproduce a past day's fires; the model
                        fire source IS EFFIS, so the cube's EFFIS-at-t is the faithful stand-in)
Both are scored against the SAME true t+1 label under the SAME cube regime (EFFIS dist_to_fire at t), so the
only thing that varies is the live feature assembly. The CUBE−LIVE gap is the real serving degradation from
(a) seasonally warm-starting un-refreshable features + (b) Open-Meteo weather/dryness approximation.

Output: reports/backtest_live.json (per-day + aggregate cube-vs-live regime metrics + prediction agreement).

CLI: --start YYYY-MM-DD --end YYYY-MM-DD --stride N  (default: 2024 fire season, every 10th day)
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import scripts.train as T
import scripts.daily_job as DJ
import scripts.live_slice as LS

log = logging.getLogger("backtest_live")


def pick_warmstart_idx(date, ds, dates_years):
    """Nearest day-of-year slice from a DIFFERENT calendar year (so veg/soil/aggregates are genuinely
    borrowed, as in real serving where the cube doesn't contain 'today'). Mirrors run_live but excludes the
    target year. Returns (base_idx, delta_doy)."""
    import datetime as _dt
    import pandas as pd
    tgt = _dt.date.fromisoformat(date)
    tgt_doy = tgt.timetuple().tm_yday
    cdoy = np.array([pd.Timestamp(ds.get_time_value(i)).dayofyear for i in range(len(ds))])
    circ = np.minimum(np.abs(cdoy - tgt_doy), 366 - np.abs(cdoy - tgt_doy))
    other_year = np.array([y != tgt.year for y in dates_years])
    circ = np.where(other_year, circ, 10_000)              # forbid same-year slices
    base_idx = int(np.where(circ == circ.min())[0][-1])    # latest year among ties
    return base_idx, int(circ.min())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    a = sys.argv
    start = a[a.index("--start") + 1] if "--start" in a else "2024-06-01"
    end = a[a.index("--end") + 1] if "--end" in a else "2024-09-30"
    stride = int(a[a.index("--stride") + 1]) if "--stride" in a else 10
    weather = a[a.index("--weather") + 1] if "--weather" in a else "full"   # "full" | "temp"

    feats, ds, gbt, calib, ccaa = DJ._load()
    stats = json.loads(Path(T.STATS).read_text())
    dates = [str(ds.get_time_value(i))[:10] for i in range(len(ds))]
    years = [int(d[:4]) for d in dates]
    window = [i for i, d in enumerate(dates) if start <= d <= end and i + 1 < len(ds)]
    sel = window[::stride]
    log.info(f"backtest {len(sel)} dates in [{start}..{end}] stride {stride} | weather={weather} | {len(feats)} features")

    pc, pl, ys, rs = [], [], [], []          # accumulate land-cell cube-prob, live-prob, truth(t+1), regime(t)
    per_day = []
    for idx in sel:
        date = dates[idx]; target = dates[idx + 1]
        Xc, _, rc = ds[idx]
        reg = rc[0].numpy().astype(int)
        land = reg.ravel() > 0
        prob_cube = LS.predict(gbt, calib, Xc.numpy(), reg).ravel()
        base_idx, ddoy = pick_warmstart_idx(date, ds, years)
        try:
            Xn, regL, _, refreshed = LS.build_live_slice(date, feats, ds, stats, base_idx, source="archive",
                                                         use_firms=False, fire_source="cube", weather=weather)
            prob_live = LS.predict(gbt, calib, Xn, regL).ravel()
        except Exception as e:
            log.warning(f"{date}: live slice failed ({type(e).__name__}: {e}) — skip"); continue
        truth = (ds.ds["is_fire"].sel(time=target).values > 0.5).astype(np.float32).ravel()
        pc.append(prob_cube[land]); pl.append(prob_live[land]); ys.append(truth[land]); rs.append(reg.ravel()[land])
        corr = float(np.corrcoef(prob_cube[land], prob_live[land])[0, 1])
        per_day.append({"date": date, "target": target, "warmstart_idx_date": dates[base_idx],
                        "warmstart_ddoy": ddoy, "pred_corr": corr, "n_fire_t1": int(truth[land].sum()),
                        "refreshed": refreshed})
        log.info(f"{date}->{target}: warm-start {dates[base_idx]} (Δdoy {ddoy}) corr={corr:.4f} "
                 f"t+1 fire cells={int(truth[land].sum())} refreshed={refreshed}")

    if not pc:
        raise SystemExit("no days scored")
    pc, pl, ys, rs = (np.concatenate(x) for x in (pc, pl, ys, rs))
    m_cube = T.regime_metrics(pc, ys, rs)
    m_live = T.regime_metrics(pl, ys, rs)
    agree = float(np.corrcoef(pc, pl)[0, 1])

    def fmt(tag, m):
        log.info(f"{tag:<6} new-ign AP={m['new_ignition_ap']:.4f} spread={m['spread_ap']:.4f} "
                 f"overall={m['overall_ap']:.4f} prec@K={m['prec_at_k']:.4f} roc={m['roc']:.4f}")
    log.info(f"=== AGGREGATE over {len(per_day)} days, {int(ys.sum())} t+1 fire cells (pooled prediction corr {agree:.4f}) ===")
    fmt("CUBE", m_cube); fmt("LIVE", m_live)
    for k in ("new_ignition_ap", "spread_ap", "overall_ap", "prec_at_k", "roc"):
        log.info(f"  Δ {k}: {m_live[k]-m_cube[k]:+.4f}")

    reports = T.project_root / "reports"; reports.mkdir(exist_ok=True)
    (reports / "backtest_live.json").write_text(json.dumps({
        "window": {"start": start, "end": end, "stride": stride, "weather": weather, "n_days": len(per_day)},
        "n_fire_cells_t1": int(ys.sum()), "pooled_pred_corr": agree,
        "cube": m_cube, "live": m_live,
        "delta": {k: m_live[k] - m_cube[k] for k in m_cube if isinstance(m_cube[k], float)},
        "per_day": per_day,
        "note": "LIVE = seasonal warm-start (other-year nearest DOY) + Open-Meteo temp/dryness + cube-truth "
                "fire; scored vs true t+1 under cube regime. Gap = real serving degradation (warm-start + "
                "weather approximation); fire held at training source (EFFIS), so this isolates the v1-fixable "
                "part — the structural fire-source question is the FGDC track.",
    }, indent=2))
    log.info("wrote reports/backtest_live.json")


if __name__ == "__main__":
    main()
