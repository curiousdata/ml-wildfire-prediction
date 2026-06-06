# CHANGES

A running devlog of substantive changes — *what* changed and *why it mattered*, newest first.
Written to double as a narrative for presenting the project: each entry is a beat in the story,
not just a diff. (The forward-looking plan and open decisions live in `ROADMAP.md`; per-agent
working notes in `CLAUDE.md`.)

**Rule:** every bug found or fixed gets logged under a "Bugs found & fixed" subsection of the
current dated entry — what was wrong, its impact, and the fix (or that it's flagged, not yet fixed).

---

## 2026-06-06 — Training pipeline rebuilt (single-head regime-aware U-Net)

Fresh training pipeline on the 4 km cube (provisional resolution), validated end-to-end on MPS.

- **Stats** (`scripts/compute_norm_stats.py`): per-feature mean/std over land cells, TRAIN split only
  (no leakage). 284 vars (`is_fire`/`AutonomousCommunities` excluded). → `stats/coarse4_norm_stats_train.json`.
- **Dataset** (`RegimeIberFireDataset`): inherits `BaseIberFireDataset` via a behavior-preserving refactor
  (`_raw_feature`/`_build_X`/`_build_y` extracted; base output unchanged). Adds calendar `(time,)` broadcast,
  a per-pixel **regime_code** (0=sea, 1=ignition, 2=spread, from `dist_to_fire(t)`), and a fire-day
  `WeightedRandomSampler`. **C = 146** (`build_segmentation_features`: year-resolved CLC/popdens, `is_fire(t)`
  added as a feature, `is_sea`/`AutonomousCommunities` excluded — leakage/redundancy calls, not pruning).
- **Loss** (`RegimeLogitAdjustedBCE`): `α·L_ignition + (1−α)·L_spread`, **per-regime** logit adjustment.
  Real-data priors confirm the rationale — ignition adj **−10.2** (0.004% pos) vs spread **−1.8** (14% pos);
  one global adjustment would badly over-boost spread. α = **0.6** (gentle lean to the hard regime).
- **Model** (`build_unet(norm="group")`): BatchNorm→GroupNorm for small-batch safety on ~14 GB; BN stays the
  default so the shipped `resnet34_v9` still loads.
- **train.py** (deep rewrite): 3-way temporal split (touched-once test), GroupNorm U-Net, regime loss with
  priors from train, MLflow **params + metrics only** (no duplicate model artifact), single `.pth`, early-stop
  on **val new-ignition AP** (bar ≈ 0.50), MPS device.

### num_workers finding (Mac/MPS) — measure the real loop, not the micro-benchmark
Isolated DataLoader benchmark said workers help (nw=4 → 9× loading throughput, with `persistent_workers`).
But in the **actual training loop** nw=0 is **3× faster** than nw=4 (13 s vs 40 s/epoch): the MPS main process
is CPU-bound dispatching kernels + backward, and worker processes contend for the few M-series cores and pay
IPC to ship 40 MB full-image tensors. **Default `num_workers=0`**, confirmed by training-context measurement
(the loading micro-benchmark was a red herring; this also matches prior hands-on experience).

## 2026-06-06 — Ignition vs. continuation: the two-regime analysis

`scripts/measurement_floor.py --new-ignition` / `--continuation` split the label by whether a fire is within
~1 cell at day t. Reports: `reports/measurement_floor_newign.json`, `..._contin.json`.

**New ignition (far from fire; base 5.4%):** FWI 0.054 (ROC 0.47), LogReg 0.386, HistGBT **0.499**. Drivers:
fuel/land-cover (`CLC_2018_27` #1, `scrub_proportion` #6, LAI, NDVI) + human access (`dist_to_roads_stdev` #2,
`popdens_2020` #4) dominate; then fire-proneness, seasonal dryness (`precip_sum_90d`, `kbdi`), season, terrain.

**Continuation / spread (near fire; base 14.5%):** FWI 0.189 (ROC 0.60), LogReg 0.581, HistGBT **0.606**. Drivers:
`dist_to_fire` dominates (proximity), then weather (`t2m_max`, `LST`, `RH`), **wind/spread geometry**
(`fire_upwind_exposure` #11, `wind_v_atmaxspeed`), dryness (`kbdi`, `precip_sum_30d`), fire history, season.

### ⭐ Main insight — the EFFIS label is TWO physical processes with different drivers
| | New ignition (far) | Continuation / spread (near) |
|---|---|---|
| base rate / HistGBT AP | 5.4% / **0.50** | 14.5% / **0.61** |
| FWI ROC | 0.47 (worse-than-random) | 0.60 (weakly useful) |
| dominant drivers | **fuel + human access** (land cover, roads, popdens) + dryness/season | **proximity + weather** (t2m/LST/RH hot-dry) + **wind** (upwind exposure, wind_v) + dryness |
| the process | *where/whether a fire STARTS* — human-fuel-static | *whether a nearby fire REACHES here* — meteorological-dynamic |

- **FWI is a fire-*weather* (spread-conditions) index** — weakly useful for spread, useless for ignition.
- **Our engineered features split exactly as designed:** `fire_upwind_exposure` + wind → *spread*; WUI/roads/fuel →
  *ignition*; fire-history / drought-memory / calendar → both. The upwind channel pays off in the regime it was built for.
- **Implication:** one label, two regimes. Evaluate them separately (done); consider *modelling* them separately
  (two heads / conditioning) — a single blended loss optimises the easy spread regime at the expense of the hard,
  valuable ignition regime. The headline metric to beat remains **new-ignition AP ≈ 0.50**.
- Caveats unchanged: point-wise GBT (not the spatial U-Net); permutation importance masks correlated groups.

## 2026-06-06 — Group A features materialized + measurement floor

### Group A SOTA gap-fill — all materialized onto the 4 km cube (now 285 vars)
Functions in `src/data/feature_engineering.py`; materialized via `scripts/add_engineered_features.py`
(incremental appends). All slice-validated before materialization.
- **Fire-weather / fuel / veg:** `emc_peak` (1-hr dead-fuel moisture, Simard), `ffwi` (Fosberg FWI), `fvc` (fractional vegetation cover).
- **Drought:** `kbdi` (Keetch-Byram; `daily_rain≈2.9×tp`, calibrated to AEMET ~640 mm/yr — treated as a *relative* predictor), `spi_90d` (standardized 90-day precip anomaly).
- **Greenness:** `ndvi_anomaly`, `lai_anomaly` (z-score vs day-of-year climatology).
- **Terrain:** `tpi`, `terrain_curvature`, `aspect_southness`/`aspect_eastness` (continuous orientation from the aspect one-hots — an HLI substitute that needs no latitude).
- **Fire history:** `time_since_last_fire`, `burn_frequency_365d` (reuse the dryness/rolling helpers on `is_fire`).
- **Human:** `dist_to_urban` (WUI proxy: distance to CLC_2018 artificial>0.5).
- **`hli`** (McCune-Keon Heat Load Index) — added after `pyproj` was installed (terrain solar load; slope + reconstructed aspect + latitude). Kept per the no-GBT-pruning principle (varies by region).
- **Still deferred:** TWI (needs a flow-accumulation lib — now *justified* by the region-varying principle, worth investing in later).

### Measurement floor (master unblocker) — `scripts/measurement_floor.py`
3-way temporal split (train 2008–18 / val 2019–21 / **touched-once test** 2022–24), features at t → `is_fire` at t+1.
Baselines: FWI-alone, logistic regression, HistGradientBoosting. PR-AUC + ROC reported **overall and split
new-ignition vs continuation** (continuation = fire within ~1 cell at t). Permutation importance → pruning
shortlist. Report: `reports/measurement_floor.json`.

**Results (test, base rate 6.25%):** FWI-alone PR-AUC **0.063** (ROC 0.475 — *worse than random* at pixel-level
next-day); LogReg **0.626**; HistGBT **0.789**. **Headline — new-ignition vs continuation:** continuation AP
**0.98** (trivial persistence) vs new-ignition AP **0.32**. The blended 0.79 massively overstates real value;
**the honest bar the U-Net must beat is new-ignition AP ≈ 0.32.** FWI-alone confirms "data-driven ≫ FWI."

**Feature ranking (143 features):** `dist_to_fire` dominates (0.42, persistence); then human activity
(`popdens_2020`, `dist_to_roads_stdev`), land cover (CLC scrub / class 27 / 24), **our fire-history**
(`time_since_last_fire` #5, `burn_frequency_365d` #19), calendar (`doy_sin` #8), drought memory
(`precip_sum_90d` #11, `kbdi` #15, `ndvi_anomaly` #18). **100/143 features near-useless** (top-15 = 92% of
importance). The instantaneous fire-weather/fuel indices (VPD_peak #130, **EMC #143 last**, FFWI #76, FVC,
SPI #136) ranked ~0/negative — **redundant** given raw weather + FWI (permutation importance masks correlated
features). Terrain orientation (`aspect_southness` #83, `tpi` #117, `curvature` #106) ~0.

**Caveats:** pixel-level tabular model (NOT the spatial U-Net — terrain/neighbourhood may matter more spatially);
permutation importance under-ranks *correlated* groups (so "redundant" ≠ "no signal"); a new-ignition-specific
importance pass (excluding `dist_to_fire`) is the high-value follow-up.

**Methodological correction — DON'T prune features on GBT importance.** The point-wise GBT is blind to spatial
structure *except* the relations we hand-engineered (and `dist_to_fire`, a spatial feature, ranked #1). For the
*segmentation* model — the project's core strength — a feature that looks dead to the GBT can still feed the U-Net
spatial context. **Operating principle: at the feature stage, keep any metric that varies by time or region; let
in-model (U-Net) ablation at the target resolution prune later (§C).** The floor is the *bar to beat*
(new-ignition AP 0.32) + a sanity check, NOT a kill list.

**HLI / solar-load decision — REVERSED to ADD.** Initially skipped on GBT importance (aspect ranked ~0); but per
the principle above that's the wrong basis. HLI varies by region (terrain + latitude) and `pyproj` is installed →
built `heat_load_index` (McCune-Keon) and materialized as `hli`.

### Bugs found & fixed
- **`time_since_last_fire` timedelta decoding** *(found via FutureWarning → fixed)*. A `units:"days"` attr made
  xarray decode the variable as `timedelta64` instead of float32 — downstream numeric code would have read
  nanosecond counts. **Fix:** dropped the `units` attr (kept a plain description); now loads as float32.
- **`aspect_southness`/`aspect_eastness` wrong bearings** *(found → fixed)*. Assumed aspect class 1 = North
  centered at 0°, but `aspect_1` = "0–45°" (center 22.5°), so the orientation features were rotated 22.5°.
  Harmless to the floor (aspect ranked ~0) but incorrect. **Fix:** sector centers 22.5 + k·45; recomputed (and
  the same correct reconstruction feeds `hli`).
- **Overnight session halted on denied writes** *(not a code bug)* — after a likely software-update/permission
  reset, unattended `Write`s were rejected, so the measurement floor wasn't built until the morning. Data
  materialization had already completed cleanly.

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
- **KBDI recursion error** *(found via validation → fixed)*. `net_rain` went negative on the first dry day
  after a wet spell (cum_wet reset made `new_excess < prev_excess`), spuriously *adding* drought deficit
  (cold-check Q hit 54.9 where it should be 0). **Fix:** zero `net_rain` on dry days. Now cold-check Q=0.
- **Precipitation units ambiguity** *(found → RESOLVED)*. The paper confirms `total_precipitation_mean` is the
  **mean of hourly ERA5-Land `total_precipitation`** (a *cumulative* field), m→mm — NOT a daily sum. So `×24`
  (≈8665 mm/yr, absurd) was wrong. Empirically the land mean is 437 mm/yr at `×2` with the correct wet-NW
  (595) / dry-SE (237) pattern; matching AEMET's ~640 needs **`daily_total ≈ 2.9×tp`** (the cumulative-mean
  factor is timing-dependent, ~2–3, so it's an *approximate* daily-rainfall proxy). **Impact:** KBDI uses
  `2.9×tp` and is treated as a relative predictor; the relative precip features (`precip_sum_*`,
  `days_since_rain`) and SPI (standardized) are unaffected. *(Aside: `pyproj` is missing from the local venv —
  used national-mean calibration, not per-city.)*
