# CHANGES

A running devlog of substantive changes — *what* changed and *why it mattered*, newest first.
Written to double as a narrative for presenting the project: each entry is a beat in the story,
not just a diff. (The forward-looking plan and open decisions live in `ROADMAP.md`; per-agent
working notes in `CLAUDE.md`.)

**Rule:** every bug found or fixed gets logged under a "Bugs found & fixed" subsection of the
current dated entry — what was wrong, its impact, and the fix (or that it's flagged, not yet fixed).

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
- **`day_of_year_sincos`** — cyclic seasonality encoding (the model's only prior calendar signal was
  the sparse `is_holiday`). Validated: on the unit circle, Dec-31↔Jan-1 adjacency restored (no seam).
- **`day_of_week_sincos`** — weekly human-ignition rhythm (weekday/weekend), orthogonal to season and
  weather. Validated: equal spacing incl. the Sun→Mon wrap. *Skipped `month` deliberately* — redundant
  with day-of-year (coarser, and reintroduces the Dec→Jan seam). Clean calendar set is now
  `{day-of-year, day-of-week, is_holiday}`: annual + weekly + holiday, no overlap.
- **Antecedent dryness** — `rolling_sum_time` (trailing 7/30/90-day precip sums) + `days_since_rain`
  (dry-spell length, which recovers the recency/ordering a sum discards). Causal (backward windows),
  no label leakage. Validated synthetic-exact vs. pandas, and real-data dry-day threshold calibrated to
  the hourly-mean precip units (`<1/24 mm` ⇒ 46% of land-days dry). Overlaps FWI's drought codes —
  incremental value is a step-3 question.

### §E spatial fire-context (on the coarse fire mask)
- **`dist_to_fire`** (km to nearest fire cell) + **`fire_upwind_exposure`** = (W·d)/|d|² — the
  hand-engineered advection of yesterday's fire (>0 downwind of a nearby fire, the highest-ROI spatial
  idea). Resolution-coupled → computed post-coarsen and appended to the 4 km cube (`add_fire_context.py`).
  Geometry validated synthetic (incl. the y-axis-decreasing sign) + real (`dist==0` iff fire cell). Both
  favour *continuation* → read via the §A new-ignition split. Cube now 271 vars.

### Data pipeline — rewrote `coarsen.py` (silver → gold) + built the provisional 4 km cube
- **Semantic pooling** replaces the old mean-everything: `*_max`→max, `*_min`→min, label `is_fire`→max,
  CLC/aspect one-hots→fractional composition, `AutonomousCommunities`→mode; engineered features computed
  inline; calendar stored as `(time,)`. Dropped non-features (`x_index`/`y_index`/`x/y_coordinate`,
  `is_near_fire`). **Map georeferencing preserved** — verified the coarse `x`/`y` block-mean coords match
  the convention the Streamlit map already renders (`coarse x[0] == mean(silver x[0:F])`).
- Provisional **4 km analysis cube** (230×297, 269 vars) built (23m50s, 22 GB) from the validated silver
  cube and QC-passed (engineered features sane, LST clipped, map coords preserved). **Fire positive rate
  0.0042%** — extreme imbalance at fine res, confirming tiling + class-balanced crop sampling will be essential.

### Data validation — audited everything before silver (the conversion was never rigorously checked)
- NetCDF→Zarr conversion confirmed **faithful**: perfect daily time axis (6241 steps, no gaps/dupes),
  correct CRS/coords, **zero** fill-leaks / all-NaN / corruption across 261 vars, physically sane ranges,
  and real Spanish fire seasonality (summer + NW spring). **No re-conversion needed.**

### Bugs found & fixed
- **`wind_direction` circular averaging** *(found → fixed)*. The old coarsen mean-pooled compass degrees,
  so 350° + 10° averaged to 180° — the exact opposite direction. **Impact:** corrupts any wind-direction
  use, and would have made the planned upwind-exposure feature point backwards. **Fix:** decompose to u/v
  *before* pooling (`coarsen.py`); raw degrees dropped from the gold cube.
- **LST cloud/edge artifacts** *(found → fixed)*. 0.07% of LST cells are physically impossible
  (156 K = −117 °C, 409 K = 136 °C) from its multi-source satellite origin. **Impact:** 4×4 mean-pooling
  ingests the garbage (one 156 K outlier drags a cell mean ~8 K). **Fix:** clip to [250, 340] K before pooling.
- **zarr v3 vs `numcodecs.Blosc`** *(found via smoke test → fixed)*. zarr 3.1.5 defaults to format v3, which
  rejects the Blosc compressor object (`Expected a BytesBytesCodec`). **Impact:** any write via the old
  encoding pattern fails — latent in `conversion.py` and the old `coarsen.py` too. **Fix:** write `zarr_format=2`
  (matches the existing cubes).
- **Old coarsen mean-pooled *everything*** *(found → fixed)*. Averaging `_max`/`_min` statistics, categorical
  region codes, and grid indices produces meaningless values. **Fix:** semantic + special-case pooling (above).
- **Stale serving default** *(found → fixed earlier)*. `app.py` defaulted to a nonexistent
  `IberFire_coarse8_time1.zarr`; realigned to `coarse32` (only worked before because compose overrode it).
- **LST multi-source inhomogeneity** *(found → flagged, NOT fixed)*. LST is stitched from ERA5 skin-temp →
  CLMS v1 → CLMS v2 (breakpoints 2010-06-20, 2021-01-19); train (2008–22) and val (2023–24) draw partly from
  different instruments → built-in distribution shift for that feature. Flagged in ROADMAP §B for the
  predictive-potential analysis to decide LST's fate.
