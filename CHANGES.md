# CHANGES

A running devlog of substantive changes — *what* changed and *why it mattered*, newest first.
Written to double as a narrative for presenting the project: each entry is a beat in the story,
not just a diff. (The forward-looking plan and open decisions live in `ROADMAP.md`; per-agent
working notes in `CLAUDE.md`.)

---

## 2026-06-05 — Recovery, cleanup, and the pivot to a data-first rebuild

A consolidation day: took a working-but-messy research repo, paid down its debt, and re-pointed it
at a sharper plan before resuming feature work.

### Planning reconstituted
- Recovered the working **ROADMAP** (plan of record: the four "species" of work, open decisions,
  phased sequence) and per-agent guidance, so direction is explicit rather than tribal.

### Storage triage — reclaimed ~107 GB
- Removed an orphaned 105 GB intermediate rechunk, a stale coarsened cube, superseded model
  checkpoints, and MLflow logging bloat — none of it referenced by the active pipeline. The 1 km
  *silver* cube (the source of truth for the rebuild) and the shipped model were kept.

### Sprint 1 — killed train/serve drift *(runtime-verified)*
The training script and the serving app had each hard-coded their own ~120-feature list and their
own model construction — a latent bug where train and serve could silently disagree on channel
order. Consolidated to a single source of truth:
- **`src/data/features.py`** — one canonical, ordered `FEATURE_VARS` (116), imported by both train
  and serve. Channel order is load-bearing for the shipped checkpoint, so it was verified
  byte-for-byte against both old inline lists.
- **`src/models/cnn.py::build_unet`** — one model factory; removed the duplicated `smp.Unet(...)`.
- **Headless by default** — replaced interactive `input()` prompts (dataset + `coarsen.py`) with
  parameters/flags, so training and preprocessing run unattended (CI, Docker, background).
- **Verified end-to-end on real data/hardware**: `build_unet(116)` loads the shipped model with
  `strict=True` (0 missing/unexpected keys); the dataset yields correct `(116,28,37)` samples from
  the real cube. (The original refactor had been compile-checked only, on a machine without the data.)

### Strategic direction set
- **Resolution → go fine, data-first.** Rejected staying at 32 km (caps the ceiling at "fire
  somewhere in ~1000 km²") *and* native 1 km (over-resolved vs. the label's and inputs' true
  precision). Target a 2–8 km sweep, chosen empirically where skill plateaus.
- **Horizon stays t+1** (next-day) — the framing with operational lead-time value, and the one that
  makes "yesterday's fire" a causal predictor rather than leakage.
- **Sequence**: fix/engineer features on the 1 km cube → measurement harness → predictive-potential
  analysis at fine res → pick resolution → re-coarsen → retrain anew.

### Repo slimmed
- Deleted the dead `catalonia-wildfire-mvp/` FastAPI+Streamlit stack (it referenced data and a model
  format that no longer exist, and re-duplicated the feature list) and its orphaned TorchScript
  export script. The Streamlit **monolith** is now the sole serving path; README + Copilot rules
  updated to match.

### Data audit — conventions pinned against source
Confirmed against the cube metadata, the IberFire paper (arXiv:2505.00837), and the author's repo:
- **Wind direction** is the meteorological "from" convention (degrees clockwise from north) — the
  detail that, if assumed wrong, would flip the planned upwind-exposure feature by 180°.
- **Units**: `t2m` °C, `RH` %, `wind` m/s, `precip` mm, `pressure` hPa.
- **`is_fire`** is from **EFFIS burned-area polygons** (fires > 5 ha, stamped across each event's
  start–end dates) — *not* raw active-fire pixels. Implication: the label is persistence-dominated
  by construction and has a 5 ha floor, so evaluation must split **new-ignition vs. continuation**.

### Feature engineering begun (on the 1 km silver cube)
New module **`src/data/feature_engineering.py`** — pure, slice-validated derived features:
- **`wind_to_uv`** — reconstructs (u, v) from speed + direction. Round-trip validated against the
  stored direction: **max error 0.000°** over 645,864 cells (no 180° flip). Unblocks the upwind /
  distance-to-fire feature family.
- **`VPD` (mean & peak) + `HDW`** — vapour-pressure deficit (the air's drying power) and the
  Hot-Dry-Windy index. Validated: textbook `e_s`, VPD ≥ 0, peak > mean, and a sharp fire-season
  signal (VPD_peak 4.82 kPa summer vs 0.58 winter; HDW 17.9 vs 1.2). corr(VPD_peak, FWI)=+0.74 —
  related but distinct, so their incremental value over the existing FWI is a question for analysis.
