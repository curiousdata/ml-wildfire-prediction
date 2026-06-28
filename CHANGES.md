# CHANGES

A running devlog of substantive changes — *what* changed and *why it mattered*, newest first.
Written to double as a narrative for presenting the project: each entry is a beat in the story,
not just a diff. (The forward-looking plan and open decisions live in `ROADMAP.md`; per-agent
working notes in `CLAUDE.md`.)

**Rule:** every bug found or fixed gets logged under a "Bugs found & fixed" subsection of the
current dated entry — what was wrong, its impact, and the fix (or that it's flagged, not yet fixed).

---

## 2026-06-28 — `fgdc-serving`: architecture settled — batch / speed / serve along a data-VINTAGE axis

Worked the Lambda mapping with the user until it clicked. The classic batch/speed split is about *compute
latency over identical data*; **ours is a data-VINTAGE gradient** — a calendar day's data MATURES
forecast→ERA5T→final. That reframes everything into **three tiers, one cube, two writers** (see the
`lambda-architecture-fgdc` memory):

- **BATCH** (monthly; the TRUE batch — **not built**): best data only, **final ERA5 + FIRMS SP** (~2–3 mo lag).
  Re-fetches and **overwrites** the cube from the `final_watermark` seam to the final edge, and **recomputes the
  recursive engineered FORWARD from the seam** (overwriting old raw re-propagates kbdi/precip_sum_*/anomalies).
- **SPEED** (weekly/daily; = today's `batch_job.py`, a misnomer): **ERA5T** preliminary reanalysis (~5 d lag),
  **appends** newly-settled days to the cube. Built + validated end-to-end (first real 1-day run 2026-06-21).
- **SERVE** (on-demand; **ephemeral**): forecast + FIRMS NRT + the cube-tail *seed* (the bundle) → run
  `compute_edge_engineered` over the forecast band → t+1 features → predict. **Writes nothing to the cube.**

**Three concrete deltas this locks in:**
1. **Data-driven settle edge.** The only freshness limiter is *weather reanalysis* — ERA5T lags ~5 d, while fire
   (NRT) and carried veg reach today. So push the seam to the **last available ERA5T day** (query it; ~5 d), not a
   hardcoded `WATERMARK_DAYS=7`.
2. **`final_watermark` seam + monthly final-reanalysis batch (the missing TRUE batch).** A single cube-attr date
   marks `≤seam`=final · `(seam, cube_edge]`=ERA5T · `>cube_edge`=forecast(serve-only). It's the batch↔speed
   contract AND it bounds the batch's forward recompute. Append-only speed means the cube currently holds *ERA5T
   permanently* with no path to final — this batch is what upgrades it.
3. **Serve is ephemeral → retires Option C.** Earlier we'd planned to write *provisional forecast rows* into gold
   and overwrite-on-settle (Option C, `extend_cube`). The cleaner model: **never persist the forecast edge** —
   serve computes it just-in-time and discards it. The cube is then *always* authoritative-settled; the whole
   provisional-overwrite/atomicity class of problems vanishes. `extend_cube` becomes the ephemeral serve engine,
   reusing `update_edge`'s `compute_edge_engineered`.

Naming follow-through (deferred): weekly `batch_job` is really the **speed** tier; the monthly final job is the
**batch**. Also reviewed the incremental engine and **fixed its one real bug** (non-atomic edge write — see below)
before it goes on a schedule.

**Serving = progressive refinement (decided 2026-06-28).** Measured the question "how much fire is in hand by
morning?" — the **night pass (~01:30 UTC, available ~morning) alone captures ~65% of a day's fire cells** (pooled
65.8%, median 64%, IQR 58–73%; 62% of detections; 25 summer-2025 days, FIRMS SP). Surprise: the night pass has
*more* raw detections than the afternoon (VIIRS night thermal contrast + persistent large/ag fires). So the serve
tier **always predicts t+1 = tomorrow**, starting from a **preliminary** morning prediction (night-pass-only fire;
weather is forecast = available) and **re-running in the evening** when the afternoon pass completes `t` — display
and prediction sharpen. `daily_job.latest_complete_fire_date()` becomes a *preliminary-vs-final* flag, not a
today-vs-tomorrow switch. The ~35% the night pass misses skew small/daytime/human-ignited → softer on the
**new-ignition** regime, robust on **spread** (whose `dist_to_fire` keys off the big persistent fires the night pass
catches). Built `extend_cube.serve_edge` (ephemeral: forecast + cube-tail seed → `compute_edge_engineered`, NO cube
write) as the serve engine; live test pending an Open-Meteo quota reset (was 429-rate-limited; fetch now batched).

## 2026-06-27 — `fgdc-serving`: monthly batch job + the silver-rebuild path made real (static preserved, regridder fixed)

Goal: a **monthly job rolled out before Jul 15** that keeps the cube current with settled data. Designed the
two-cadence Lambda split (see the `fgdc-extend-cadence` memory): **monthly batch** mutates silver with *settled*
ERA5/VIIRS/MODIS (the only thing that touches silver), **daily** writes only the provisional gold edge (Option C,
deferred). Built **`scripts/batch_job.py`** on the clean model *bronze is the source of truth → top up bronze
with settled days, then rebuild silver→gold→engineered from it* (no append-mode/provisional logic — the back half
is whole-cube anyway since the engineered features are causal/recursive). Per-feed watermarks: weather ERA5 ~5 d
(margin 7), fire VIIRS ARCHIVE/SP where final + NRT for the recent ≤60-day edge, veg MODIS graceful-NaN.

Validated the feeds (ingest-only, non-destructive): **20/20 weather+fire+veg** for 2026-06-01..06-20, MODIS data
physically sane (NDVI [-0.2,0.99], LAI [0,7], FAPAR [0,1], LST 280–325 K). Then measured the silver rebuild on a
51-day **temp-store** slice (real silver untouched): **0.54 s/day → full 5285-day rebuild ≈ 47 min.**

Added a compact, verified **`## Module map (FGDC v2)`** to CLAUDE.md (real entry points for `src/data/ingest/` +
`scripts/`) after repeatedly re-deriving interfaces; pointer saved as the `fgdc-module-map` memory.

**Bugs found & fixed:**
- **Silver rebuild was hard-blocked by the deleted v1 cube.** `build_silver._load_static` read `grid.V1_CUBE`
  (deleted 2026-06-26) → any rebuild would crash on static-load. *Impact:* the monthly job could not rebuild silver
  at all; worse, the 221 1 km static layers survived **only** inside the 150 GB silver, so a mid-rebuild failure
  could have lost them. *Fix:* `build_silver.extract_static()` preserves them into `data/silver/FireGuard_static.zarr`
  (44 MB, byte-identical to the refined-from-v1 static, incl. all masks); `_load_static` now reads that store
  (V1_CUBE demoted to legacy fallback). Static-load runs *before* the rmtree, so the failure mode was at least
  non-destructive.
- **`build_silver` assumed a single native weather grid for all days.** It built ONE regridder from `dates[0]` and
  applied it to every day → `IndexError` the moment a day's native grid differed. Surfaced because the new ingest's
  bbox (`SPAIN_BBOX = -9.5,35.5,4.5,44.0`, 1995 pts) is **narrower than the historical backfill** (`-10,35.25,5,44.5`,
  2318 pts). *Impact:* the full rebuild would have crashed in chunk 2 (caught on the temp slice, not live silver).
  *Fix:* regridder cached **per native grid** (`_weather_regridder`, keyed by coord signature; only ~2 grids exist).
  Verified seam-continuous across 05-31→06-01 (t2m 21.7→22.1, NDVI 0.563→0.563, no jump). *Secondary, flagged not
  fixed:* the `SPAIN_BBOX` narrowing is a latent regression — harmless now that regridding is per-grid, but worth
  realigning to the historical bbox so the backfill stops proliferating grids.
- **Engineered stage ran the producer AFTER the consumer (ordering bug).** The clean from-scratch baseline
  exposed it: `add_engineered_features`'s `spi_90d` reads `precip_sum_90d`, but `precip_sum_{7,30,90,180,365}d` is
  *produced* by `add_fire_context` — and `batch_job`/`extend_cube` ran `add_engineered` FIRST → `KeyError:
  'precip_sum_90d'` after ~80 min (silver+coarsen had already succeeded). *Impact:* the `coarsen --overwrite`
  rebuilt gold before the failure, so gold was left **raw + 3 engineered (251/278 vars), serving features
  incomplete** — recoverable (silver complete, bronze intact), needs a gold re-run. *Fix:* swapped the order to
  **`add_fire_context` → `add_engineered_features`** in both `batch_job.py` and `extend_cube.py` (+ CLAUDE.md
  pipeline order). `add_fire_context` depends only on raw vars, so the order is acyclic.
- **Incremental gold edge was non-atomic (found in code review, fixed).** `update_edge.update_gold_edge` appended new
  rows with NaN engineered, then computed + region-wrote engineered in a *separate* step — a crash between the two
  (OOM/teardown/kill, all of which happened) would leave permanent NaN-engineered edge rows that the date-based
  currency check (`new = silver days > gold_last`) mistakes for complete → serving reads NaN `dist_to_fire`, regime
  classifier collapses. *Fix:* build a VIRTUAL extended cube (lazy zarr history + in-memory new raw + NaN
  placeholders), compute engineered from it, then append COMPLETE rows in ONE write — a crash now leaves only
  fewer-DAY rows the date check self-heals. Re-verified bit-identical (`--test`/`--e2e`). Also merged the two
  near-duplicate `_causal_anomaly_edge*` fns into one (review #4).

**Live-serving label semantics (settled this session).** The `is_fire[t]` label is the **whole-UTC-day union of
VIIRS-SNPP detections** (conf ≥ nominal) with `acq_date == t` — i.e. BOTH SNPP passes that fall on UTC day `t`
(~01:30 UTC night + ~13:30 UTC afternoon), not a single pass (`ingest_fire.py` filters `acq_date == wd`). "Day" =
UTC calendar day. Consequence for serving: `is_fire[t]` (and `dist_to_fire[t]`, `time_since_last_fire[t]`) is
**complete only after `t`'s ~13:30 UTC afternoon pass settles in FIRMS (~3 h → ~16:30 UTC)**. So scoring before
that has only a *partial* `t` (night pass only) — a label-definition mismatch, not just staleness. Live rule:
issue date `t` = latest UTC date whose afternoon SNPP pass has settled; predict `t+1`; before settle, latest
complete `t` is yesterday (a same-day nowcast). A 5 am score predicts *today*; an evening score predicts
*tomorrow* — the horizon is data-relative. **Baked a `latest_complete_fire_date()` gate into `daily_job`.**

**Objectives logged (not yet built):**
- **NOAA-20 VIIRS pooling — experiment.** `ingest_fire` is SNPP-only; pool NOAA-20 (VNP14IMG, 2018+) for ~2×
  detection density / more passes. Test next-day AP lift + the train/serve-coverage seam (SNPP-only pre-2018).
- **2/3-day union target — secondary objective.** Rolling-OR label P(fire within {2,3} d). Already **proven to
  lift AP** (less sparse, higher-skill than the spiky t+1). New rationale from the label-settle analysis: a
  multi-day union is **robust to the ≤1-day completeness lag** — predicting "fire within 2–3 d of `t`" still
  covers today+tomorrow even when forced onto a 1-day-stale `t`, so it stays **true forecasting on a stale
  label**. Dual win (skill + operational robustness); serve a risk curve alongside the t+1 head.

## 2026-06-26 — `fgdc-forecast-features`: forecast weather PROVEN (+CAPE), calendar DROPPED, baseline floors, v1 cubes deleted

Branch goal: push *past* v1 by adding next-day signal. Net — **one lever proven (t+1 forecast weather), one
dropped (calendar/human-activity), and the non-ML baseline floor finally established.** Production integration
of the forecast feature is **deferred** (proven but modest; the full GEFS backfill isn't worth +0.0054 now).

**Detection-lag framing (the prerequisite reasoning).** The regime split already separates nowcasting (spread —
fire adjacent at t) from forecasting (new-ignition). VIIRS overpass timing means afternoon ignitions are first
seen at t+1, so part of "next-day fire" is half-nowcast; the model's real value lives in new-ignition. The
right test order: the **perfect-foresight ceiling** (reanalysis t+1) first, then a real forecast trained
hindcast-honestly as a **complement** (keep observed t-weather AND add the errorful t+1 forecast, so the GBT
learns to weight by reliability — substitute would strand the t-observation).

**Calendar / holiday / HDW / VPD → DROPPED (flat).** Materialized `doy/dow` sincos (t & t+1),
`is_holiday_{national,regional}` (region via the `holidays` lib + `AutonomousCommunities`, t & t+1), and the
dead `hdw`/`vpd_peak` functions. Bundled (147) and isolated (143, holiday/dow only) both flat-to-negative
(new-ign +0.0015 = noise; prec@K *down*). Human-ignition timing is already proxied by
`popdens`/`dist_to_roads`/`dist_to_urban`; the fire-weather couplings are collinear with `ffwi`. Reverted to the
135-feature production set; vars kept in the cube. (ABLATIONS 2026-06-23.)

**Baseline floor panel (the ML-justification we never had).** `scripts/baseline_panel.py` ranks the val by
single non-ML scores through the same `regime_metrics`: fire-weather index (`ffwi`/`kbdi`/`hdw`) ≈ **0.07
new-ign, ROC ~0.55** (near-random as a per-cell ranker); per-cell×doy **climatology 0.33** (the real floor);
**persistence** 0.99 spread / 0.12 new-ign (exposing the 0.98 spread AP as **near-nowcast**); **logistic 0.56**;
**GBT 0.6215** (reproduces the trained number → harness self-validates). Headline: the GBT beats the operational
index ~8× and the best non-ML floor ~2× on new-ignition; spread is *not* where the model earns its keep.

**Forecast weather → PROVEN (KEEP), production DEFERRED.** Ceiling (`train_gbt_fgdc --weather-lead/--complement`):
perfect t+1 reanalysis lifts new-ign **+0.008** (t+1 subsumes t under perfect foresight). Real test: built a
**GEFSv12-reforecast d+1 ingester** (`src/data/ingest/ingest_weather_gefs.py` — control member, `.idx`
byte-range to fetch only the 8 d+1 messages ≈ 10% of each GRIB, stream-process-and-delete → ~0.6 GB persistent;
RH-from-spfh, native tmax/tmin, soil dropped, CAPE free), backfilled the **2016-2019 slice (1461 days)**, ran
the complement on an internal split (train 2016-2018 / val 2019, `scripts/train_gbt_fc1_slice.py`). **+both =
+0.0054 new-ign / +0.0054 prec@K** — real, coherent (all metrics up), ≈ **2/3 of the +0.008 ceiling**, matching
the measured d+1 forecast skill (temp corr 0.95, debiased MAE ~1 °C, over 174 days). Split: **weather drives it
(+0.0045)**; **CAPE alone +0.0031 but largely redundant** with weather (combined ≪ sum; CAPE's incremental over
weather +0.0009). (ABLATIONS 2026-06-26.) **Verdict: proven, banked, kept in bronze. Production fill (full
2012-2026 backfill incl. the un-validated operational 2020-2026 bucket → materialize → retrain → live GRIB
serving) is deferred — not worth ~30 backfill rounds for +0.0054.**

**v1 IberFire cubes deleted (−131 GB; 59 → 190 GB free).** `silver/IberFire.zarr` (87) +
`gold/IberFire_coarse4.zarr` (29) + `_dyn` (15). Full historical reference captured first in the
`iberfire-v1-reference` memory (cube structure, 286 vars, feature engineering, GBT-vs-U-Net, the 0.10
fire-source bug). **NB:** the v1 cube was the FGDC static-feature source (`build_silver._load_static` →
`grid.V1_CUBE`); static is already baked into the FGDC gold cubes, but that codepath (+ `load_masks_from_v1`,
`ingest_static`, `ingest_veg --validate`) needs repointing if silver is ever rebuilt.

### Bugs found & fixed
- **★ Kernel panic from parallel backfill (memory exhaustion).** Launched 4 simultaneous GEFS backfill processes;
  each holds ~36 global GRIB fields (~33 MB ea) in memory → ~5 GB across 4 procs + OS → swap exhausted → macOS
  watchdog rebooted the machine. **Fix: single process only** (one proc at workers=4 ≈ 1.2 GB ran 100 days
  cleanly); the 10-min harness cap means the slice backfills in ~10 resumable rounds (skip-existing).
- **GEFS `apcp` accumulation**: per-bucket, not cumulative-from-init → sum the d+1 buckets for the daily total.
- **GEFS wind file holds 10 m AND 100 m** → `filter_by_keys={level:10}` on read (else cfgrib u10/u100 clash).
- **Slice-trainer val OOM risk**: full-prevalence val matrix ~14 GB → per-day eval + 100-day block-read
  (~28× faster, memory-bounded).
- **`train_gbt_fgdc` stray `lt_w` NameError** (misplaced boundary-guard snippet at function scope) — deleted;
  the `--weather-lead` substitute/complement plumbing validated (lead=0 reproduces 0.6215 exactly).

## 2026-06-21 — FGDC v2 baseline complete: full cube → engineered features → model MATCHES v1 (the A/B)

Headline: the from-scratch operational-source rebuild now has a trained production model that **matches
IberFire v1 on next-day new-ignition skill — on operational, self-sourced feeds with zero train/serve gap.**

**Full backfill + cube.** All three dynamic feeds complete over 2012-01-01→2026-05-31 (5265 days): weather
(EDH ERA5 reanalysis, native 0.25° stored + regridded on read), fire (FIRMS VIIRS), veg (MODIS/MPC). Built
silver (BitRound-12, 150 GB) → 4 km gold (5265 days). Full-span ablation (ABLATIONS.md 2026-06-18):
**fire_context dominant (+0.042)** — its earlier negative was a single-window artifact; human/terrain
confirmed; raw weather ~0 marginal (the engineered-weather + regime tests resolved that, below).

**P4 engineered features materialized.** `add_fire_context.py` (+ multi-scale `precip_sum_{7,30,90,180,365}d`
rolling windows via a cumsum-diff) and `add_engineered_features.py` (kbdi, spi_90d, ndvi/lai anomalies, ffwi,
time_since_last_fire, burn_frequency_365d) run on the FGDC gold (both now `--cube`-parametrized) → 266 vars.
P4-A inline ablation: the engineered ignition features lift new-ign AP **0.483 → 0.547**, with
**`time_since_last_fire` the single biggest driver** (engineered weather is real but largely redundant with
fire-memory).

**P5 — feature set + production model + the IberFire A/B.**
- `src/data/features_fireguard.py` — the frozen, leak-free, fixed-order **135-feature** contract (excludes the
  `is_fire` label, masks, region id, and stale CLC 2006/2012 editions; the v1 `features.py` analog).
- `scripts/train_gbt_fgdc.py` — trains the production GBT on the enriched cube; reports v1-comparable regime
  metrics (next-day horizon, matched 15:1 prevalence).
- **A/B RESULT (held-out recent ~20%, ≈2023→2026):**

  | regime | FGDC v2 | v1 bar |
  |---|---|---|
  | **new-ignition AP** | **0.622** | ~0.63 |
  | **spread AP** | **0.984** | ~0.98 |
  | overall / prec@K / ROC | 0.749 / 0.319 / 0.932 | — |

  FGDC **matches v1** on the hard ignition regime *and* spread — trained entirely on operational sources.
  Directional caveat: FGDC label = VIIRS active-fire vs v1 EFFIS burned-area; window = held-out recent slice,
  not v1's exact test set.

**Precompute speedup.** `scripts/rechunk.py` (rewritten as a CLI) → `FireGuard_coarse4_t200.zarr` (200-day time
chunks, lz4); `train_gbt_fgdc` block-reads it → train-matrix build **34 min → 73 s (~28×)**, metrics
bit-identical. (A v1-style channel-stack was tried and dropped — rechunking keeps *named* vars, so no
fp16/clip/channel-index plan.)

**Bugs found & fixed.**
- `train_gbt_fgdc` val regime briefly `(d2f≤6).astype(int8)` → 0/1, which `regime_metrics` reads as
  non-land/spread — silently dropping the new-ignition cells. Fixed to `np.where(…, 2, 1)` (2=spread, 1=ignition).
- float16 overflow (`fire_upwind_exposure` → inf near fires, `1/|d|²` blow-up): the rechunk path avoids it
  (native dtype); capping `fire_upwind_exposure` at source is a follow-up.

Closes the FGDC build → baseline-model arc. Next phase (new branch): t+1 forecast features (hindcast-archive
discipline) + holiday/dow, Optuna, refit-on-all, calibration, the live serving loop (P6), and the external
benchmark vs the operational fire-danger index. See ROADMAP for the 2–3-month plan.

---

## 2026-06-14 — FGDC adopts Lambda architecture (+ light weather backfill)

**Architecture decision.** The FGDC (v2) data pipeline adopts **Lambda architecture** as its primary
vocabulary — the speed-layer / batch-layer split — because a train/serve gap at the live edge is *inherent*
(ERA5 reanalysis does not exist for "today"), so the question is how to *manage* it, not avoid it:

- **Batch layer** — immutable, append-only **master dataset** of best-grade observations (EDH ERA5
  *reanalysis*, finalized VIIRS, MODIS) → recomputes the **batch view** = the training cube (1 km → 4 km +
  engineered). Cadence: **monthly** on reanalysis finalization. Training-grade. The 2012→present backfill is
  this layer's one-time seed.
- **Speed layer** — freshest-source trailing-edge slices (Open-Meteo *forecast* / FIRMS NRT) + live
  predictions for the last ~7 days; transient (superseded once batch finalizes those dates). Cadence:
  **daily**. Operational-grade.
- **Serving layer** — the merged risk product the Space reads: `batch view if date ≤ watermark else speed
  view`; watermark = last date the batch layer has finalized.

One codepath: `reconcile(start, end, layer="batch"|"speed")` — **same feature transform + grid**, only the
source *vintage* differs → daily HF job = speed, monthly EDH job = batch, Space = serving. On HF the layers
map to: HF Dataset (store, git = lineage) · scheduled Jobs (engine) · the Space (UI). **Medallion
(bronze/silver/gold) is retained as the within-batch refinement axis** (orthogonal to Lambda's latency axis).
Physical paths are **adopt-forward** (not renamed mid-backfill). Discipline carried from v1's failure: the
trailing-edge forecast-vs-reanalysis skew is small + bounded but **must be measured** on the overlap — v1's
sin was an *unmeasured* gap (EFFIS-trained vs FIRMS-served, 0.10 corr), not a gap per se.

**Light weather backfill (batch-layer seed).** Two optimizations made the multi-year ERA5 pull cheap enough
to run: (1) `fetch_openmeteo.make_regridder` precomputes the source→cube interpolation once (barycentric +
nearest-fill) — bit-exact vs per-call `griddata`, ~38× faster/field; (2) switched the weather source from the
public ARCO Zarr (whole-globe chunks → ~3.7 TB for 13 yr) to **Earth Data Hub (DestinE)** ERA5, chunked for
time analysis (4320 h × 64 × 64) → a Spain slice pulls ~GBs; store opened once, 6-month block reads aligned
to the time-chunk, 16-worker fetch. EDH values match ARCO to ~3 decimals; ~25× faster/month. CDS time-series
was evaluated and rejected (point-only — not viable for a gridded cube).

**Bugs found & fixed.**
- **Weather bronze filled the disk (~370 GB) + non-atomic writes left corrupt partials.** The backfill stored
  weather *upsampled to 1 km* (~71 MB/day, ~99.8% redundant interpolation from the ~2318 ERA5 points) and
  hit ENOSPC at ~2021; ENOSPC mid-`savez` left truncated npz that skip-existing treated as done. **Fix:** store
  native 0.25° per-point vectors + coords (~118 KB/day, ~600× smaller), regrid native→1 km on read in
  `build_silver` (values bit-identical); `atomic_savez` (temp→rename, name ends `.npz`) + `_qc_native` gate so
  a crash never leaves a "present" corrupt partition.
- **Veg backfill crashed: `UnboundLocalError: 'e'`.** In `ingest_veg.build_range`, `except Exception as e:`
  deletes `e` at block end (Python 3), but `e` was also the end-date passed to `_search()` on the next
  collection → the first composite skip wiped the end-date and aborted the whole run. **Fix:** renamed the
  exception var to `exc`.
- **Veg MPC 403s (expired SAS signatures).** The client signed assets once at search time (`pc.sign_inplace`),
  but slow multi-composite chunks read tiles >1 h later, after the token expired → HTTP 403. **Fix:** sign each
  tile href FRESH right before reading (`pc.sign` in `_mosaic_reproject`); search no longer signs. Validated:
  NDVI vs v1 corr 0.9909 on 2012-07-15.

---

## 2026-06-10 — FGDC P2 vegetation complete + ablation practice established

**Vegetation (P2) done & validated, keyless.** `ingest_veg.py` pulls MODIS NDVI/EVI (13A1), LAI/FAPAR (15A2H),
LST (11A2) from **Microsoft Planetary Computer** (no Earthdata login) → mosaic Spain tiles → reproject_match
to the 1 km grid → composite→daily interp. NDVI vs v1 corr **0.9918**. Folded into `build_silver` (261 vars
now: 27 dynamic + 234 static) → coarsen → train. Deployment decision recorded (HF Space + GH-Actions engine,
deferred). giga-spatial assessed & rejected (DGGS-only, no projected-grid/temporal model).

**Ablation practice established (user-requested).** New committed **`ABLATIONS.md`** registry + reusable
`scripts/fgdc_ablation.py` (leave-one-group-out + target-horizon, identical-setup with/without, AP+ROC).
Rule: every major feature/source/target change gets a documented with/vs/without entry before it's "kept".
First clean results (Aug-2016 slice, directional):
- **Multi-horizon target (user idea): KEEP (strong)** — val AP 0.026 (1d) → 0.148 (3d) → 0.226 (7d).
- **Leave-one-group-out:** **human features dominate (+0.030 AP)** — validates the GHS-POP/BUILT investment;
  vegetation modest (+0.003) on a single month (expected; re-test across seasons on the backfill).
- Honest correction: the earlier "veg lift 0.028→0.077" was a *conflated* comparison (different horizon/
  code); the clean ablation shows veg adds +0.003 on this slice — exactly the trap the practice catches.

**P3 started — GHS-POP population + temporal interpolation.** `ingest_static.py` downloads GHS-POP (and
GHS-BUILT-S) R2023A 1 km global from JRC, clips Spain, reprojects Mollweide→1 km EPSG:3035. **Temporal
interpolation** built (`interp_to_date` linear-between-editions vs `nearest_to_date` step-snap baseline — the
two arms of the user-prioritized interpolation ablation). GHS-POP validated: mean **84 ppl/km²**, sensible
2015→2020 growth (83.6→85.0). Agreement with v1's WorldPop is **moderate (log-corr 0.48)** — expected, they're
different dasymetric products (GHS disaggregates census onto built-up; WorldPop RF-smooths); GHS-POP is chosen
for forward-continuity to 2030 (WorldPop stops 2020), not v1 parity. **The interpolation ablation needs a
MULTI-YEAR span** (population barely moves within one month) → queued for the full backfill, not the slice.
Next P3: wire GHS-POP into build_silver (replace v1's stale popdens), add GHS-BUILT-S as an ablation
candidate, inherit CORINE/OSM/Natura2000 from v1, compute calendar.

**Sample backfill (2015–2018) — operational finding: Open-Meteo quota.** Fire (1461 days ✓) + GHS-POP/BUILT
(✓) done; veg running. **Weather hit a 429 rate-limit at ~120 days** — not volume (a 4 yr backfill is only
~250–700 requests) but the free-tier daily/hourly quota was already spent by the day's many validate/dev
runs. Mitigations: `backfill_range` chunk default 30→60 days (fewer, larger range requests); the backfill is
resumable (skips existing days) so it's re-run when quota refreshes. Strategy note: the eventual full
2012→present weather pull should run once on a fresh quota (or the scheduled engine accumulates it forward).

### Bugs found & fixed
- **MODIS QA fill contamination** *(found → fixed)*. MOD15A2H fill DN (249–255) survived scaling → FAPAR 1.20
  (>1!), LAI 10.2. Fixed by masking each product to its valid DN range before scaling → FAPAR ≤1, LAI ≤7.
- **reproject_match nodata = 0** *(found → fixed)*. Reprojection filled uncovered areas with 0, not NaN →
  20% of LST = 0 K (mean 248 K, median fine at 310). Fixed with `write_nodata(np.nan)` on the masked tiles +
  `nodata=np.nan` through merge/reproject; LST mean back to **311.8 K**. Also dropped `nan_to_num` in the
  trainer so HistGBT handles veg cloud-gaps natively instead of seeing a misleading 0.

## 2026-06-08 — Fire Guard Datacube (FGDC): recollect the cube from operational, same-source providers

**The strategic pivot behind the live track.** Every live-serving hack so far (persistence cascade, archive
antecedents, the whole "Known live-serving inconsistencies" list) is a symptom of ONE root cause: the v1
cube was downloaded as a frozen Zenodo NetCDF, so its sources and our live feeds are *different products*.
Trained-on-EFFIS vs served-FIRMS fire dropped prediction-corr to 0.10. The fix is structural — **rebuild
the cube from the reliable, append-daily providers we actually serve from, using the same code path for the
historical backfill and the daily append.** Train and serve become one store; warm-starts retire.

Named the **Fire Guard Datacube (FGDC)** (slug `fireguard`) — successor to IberFire v1 but **non-destructive**:
v1 cube/model/feature-order/app all stay; FGDC is additive under `src/data/ingest/` and
`data/{bronze,silver,gold}/fireguard/`, cutover is a reversible config repoint.

**Decisions (evidence-backed; full research + citations in the session and the plan):**
- **Resolution** — collect at **1 km native** (EPSG:3035), coarsen to 4 km as now.
- **Fire** — **VIIRS 375 m active-fire (FIRMS VNP14IMG)** for BOTH label and fire-context features (label =
  ≥1 next-day detection at confidence ≥ nominal; no ha threshold). VIIRS is a markedly more *learnable*
  next-day target than MODIS active-fire (MOD14 "highly stochastic"; VNP14 "a much better option" —
  Karlsson et al. 2025, [arXiv:2503.08580](https://arxiv.org/abs/2503.08580)), and one source for
  label+features gives train=serve identity. EFFIS burned-area kept as an **offline aux eval layer** only.
- **Span** — **2012 → present, rolling** (VIIRS era). Lose 2007–2011; gain a consistent, append-daily label.
- **Vegetation** — **MODIS MOD13/MCD15 + VIIRS VNP13/VNP15** spine (same lineage, long overlap → seam frozen
  in the past, harmonizable). NDVI primary + derived FVC + NDWI/NDMI. Rejected Sentinel (1 km↔10 m break at
  the serve edge, 2015 start) and CLMS (forward platform churn + 1 km→300 m seam).

**Reference corrections (this entry's housekeeping).** The IberFire dataset is
[arXiv:2505.00837](https://arxiv.org/abs/2505.00837) — **Erzibengoa**, Gómez-Omella & Goienetxea (2025);
1 km × 1 km × 1 day, Dec 2007–Dec 2024, **120 features in 8 categories** (the cube materialises more once
one-hots expand). Weather backbone = ERA5-Land (Muñoz-Sabater et al. 2021, ESSD 13:4349); VIIRS 375 m fire =
Schroeder et al. 2014 (RSE).

**P0 done.** `src/data/ingest/grid.py` — canonical 1 km EPSG:3035 grid, **aligned to v1**: 920×1188, origins
(2674734.3466, 2492195.9911) ±1000 m; `--verify` confirms block-mean ×4 reproduces v1's coarse4 x/y centres
exactly (so FGDC gold shares v1's grid → clean per-cell A/B). Provisional masks refined ×4 from v1 (P3
re-derives from CORINE/DEM).

**P1 vertical slice — GATE PASSED.** Built + validated the dynamic ingesters and proved the full pipeline on
recollected Aug-2016 data:
- `ingest_weather.py` — **uniform ERA5** (era5_land lacks pressure/wind/precip via Open-Meteo) hourly→daily
  (t2m/RH/pressure/wind→u,v/precip/soil), regrid to 1 km; `backfill_range` fetches a whole window in one
  request/batch. Validated vs v1 (2015-07-11): t2m corr 0.95, RH 0.99, wind 0.90. Fixes: Open-Meteo wind is
  km/h→÷3.6 m/s; `_get` now retries on timeouts.
- `ingest_fire.py` — VIIRS_SNPP_SP archive → 1 km is_fire (conf≥nominal). Gotcha: FIRMS area API day_range
  caps at **5** (not 10); SP covers 2012-01-20→2026-04-27, NRT tiles after. Aug-2016: 31 days, 45–65 cells/d.
- `build_silver.py` → `silver/FireGuard.zarr` (256 vars = 22 dynamic + **234 static inherited from v1**,
  refined ×4 — lossless at 4 km gold, static doesn't drift). `coarsen_fgdc.py` → `gold/FireGuard_coarse4.zarr`
  (230×297; is_fire/*_max→max, *_min→min, else mean — FGDC stores wind u/v directly so v1's coarsen.py is
  left untouched).
- `smoke_train.py` — next-day GBT on the slice: 933,630 rows, **855 positives (0.092%)**, **ROC-AUC 0.847**
  with no vegetation yet — sane signal, machinery proven.

**P3 population decision:** source population from **GHS-POP (R2023A)** (100 m, 1975–2030 incl. projections)
rather than re-inheriting v1's WorldPop, which stops at 2020 → stale for the FGDC's 2021→ live edge; `popdens`
is a top-2/3 ignition driver, so its forward-staleness matters. Add **GHS-BUILT-S** (built-up surface) as an
ablation candidate (WUI/structure exposure beyond `popdens`/CLC); keep only if importance earns it. Skip SMOD.

**giga-spatial: not adopted** — assumes H3/S2/Mercator DGGS not arbitrary EPSG:3035 projected grids, no
temporal model, 4/8 sources uncovered, AGPL+GEE. Only its WorldPop/GHSL/GADM downloaders worth referencing.

**P2 vegetation — keyless via Microsoft Planetary Computer (plan refinement).** `ingest_veg.py` pulls MODIS
**modis-13A1-061** (NDVI/EVI, 500 m 16-day) from MPC's STAC — **no NASA Earthdata login** (MPC is
anonymous-read + SAS-signed COGs), a better fit for the FGDC keyless principle than the planned earthaccess
route. Pipeline: search Spain sinusoidal tiles (h17/h18 × v04/v05) → mosaic → reproject_match onto the 1 km
EPSG:3035 grid → linear composite→daily interp. **Validated: FGDC MODIS NDVI vs v1 NDVI (2016-08-14) MAE
0.032, corr 0.9918** (v1 used CGLS; the ~+0.03 cross-sensor bias is harmonizable). MODIS covers 2012→~2026;
VIIRS VNP13/15 forward-bridge is a later task (not on MPC; harmonize on the overlap). LAI/FAPAR (15A2H) +
NDWI/NDMI (09A1) follow the same pattern.
Fixes: pin `PROJ_DATA` to rasterio's bundled proj_data (shell leaks a broken conda PROJ_DATA; pyproj's db is
version-incompatible with rasterio's GDAL); per-tile full-read-with-retry + GDAL_HTTP_MAX_RETRY for transient
MPC COG `TIFFReadEncodedTile` partial reads.

Next: fold veg into the slice + re-train; full backfill 2012→present; P3 static (GHS-POP/BUILT) + P4 parity.

## 2026-06-07 — Live antecedent dryness (A.1) — done & validated; + a FIRMS-vs-EFFIS fire mismatch found

`live_slice.py` now computes the antecedent-dryness features LIVE: fetch the last 90 days of Open-Meteo
precip+temp in one range request per batch (`fetch_grid_range`), regrid to a daily stack, and recompute
`precip_sum_7/30/90d`, `total_precipitation_mean`, `kbdi` with the SAME `feature_engineering` functions the
cube used (precip /24 to match the cube's hourly-mean units; KBDI seeded q0 from the cube's seasonal value).
Wired into `daily_job --mode live`.

**Validated (isolated, FIRMS off, cube date 2024-07-15): live-slice prediction vs cube MAE 0.00013, corr
0.997** — i.e., the live antecedent dryness keeps predictions essentially identical to the cube. At the
FEATURE level the precip sums show pattern corr 0.83–0.93 but a magnitude bias (Open-Meteo default ERA5 vs
the cube's ERA5-Land precip — and ERA5-Land precip is NOT exposed by Open-Meteo, returns null), yet that
bias WASHES OUT in the prediction (normalized features + GBT precip-robustness, per the shift test).
`days_since_rain` is excluded from the live overwrite (poor transfer, corr 0.16) — see the
**Known live-serving inconsistencies** subsection below for the mechanism and fix path.

### Known live-serving inconsistencies (solvable later — written down so they are not forgotten)
Each is a place where the *live* feature value diverges from how the cube built it. None silently dropped;
all either warm-started (seasonal cube value) or flagged. Measured on cube date 2024-07-15
(`live_slice.py --validate-dryness`), live vs cube:

1. **`days_since_rain` — live value is noise; warm-started instead (corr 0.16, MAE 9.3 days).** It's a
   *consecutive-day threshold counter* (`#days with precip < DRY_DAY_THRESHOLD_MM`). The live precip is
   Open-Meteo **ERA5** (ERA5-Land daily precip is not exposed → returns null); the cube used **ERA5-Land**.
   The two differ in *magnitude*, which flips the dry/wet classification on borderline days, and because the
   feature is a *consecutive* counter those flips compound (a reset/extension propagates the whole run).
   Contrast the integrating antecedents, which absorb the same magnitude bias: `precip_sum_7/30/90d`
   corr **0.84 / 0.83 / 0.93**, `kbdi` corr **0.70** — all kept live. So `days_since_rain` is NOT recomputed
   live; it keeps the cube's seasonal (day-of-year) value, which is at least correct climatology, just not
   today's actual dry-spell length. **Fix path:** (a) source ERA5-Land daily precip so the hard threshold
   behaves (no Open-Meteo route today), or (b) accumulate the daily-job's own precip series forward and count
   dry days on that self-consistent stream. Both wait on the live pipeline maturing.
2. **Precip magnitude bias on the sums/KBDI (ERA5 vs ERA5-Land).** Pattern is right (corr 0.83–0.93) but
   magnitude is biased (e.g. `precip_sum_90d` MAE 71 mm on cube-mean 77; `kbdi` MAE 97 on mean 58). It WASHES
   OUT in the prediction (normalized features + GBT precip-robustness → pred-corr 0.997), so it is accepted
   for now, but it is a real feature-level inconsistency. **Fix path:** same ERA5-Land sourcing as (1).
3. **Model fire is EFFIS-consistent but warm-started while the EFFIS WFS is down** (see fire-mismatch note
   below). Live fire definition matches training; today's actual perimeters are pending endpoint recovery.
4. **⚠️ PHANTOM SPREAD-RISK while EFFIS is down (surfaced 2026-06-07 on the present-day forecast run).** The
   fire warm-start pulls the *nearest day-of-year* cube slice — for a 2026-06-06 prediction that's
   **2024-06-05**, which had real fires. So `dist_to_fire`/regime import *last year's* fire geography:
   the live 2026-06-06 run produced **13 spread-regime cells and a peak risk of 82%**, and ALL 4 cells ≥20%
   were those spread cells — i.e. the entire HIGH headline came from cells that "think" a fire is burning
   next door, inherited from 2024. (The per-day occlusion attribution caught it: spread's #1 driver was
   `dist_to_fire`, drop 0.51.) The 40,951 ignition cells were correctly low. Mitigated in the app by a
   **degraded-input banner** + the warm-start flag; the real fix is live EFFIS, or — decision pending —
   deriving the model's `dist_to_fire` from **today's FIRMS** hotspots when EFFIS is down (current fire
   geography, definition-mismatched) vs. treating "no live burned-area" as **no-fire/all-ignition** vs.
   keeping the (worst) 2024 warm-start.
   **✅ FIXED (2026-06-07, persistence cascade — user's idea, the weather-forecasting persistence baseline).**
   Fire features now NEVER warm-start from the seasonal cube. `build_live_slice` uses a recency cascade:
   (1) **live EFFIS today** → cached to `data/serving_store/effis_cache/`; (2) **most-recent CACHED EFFIS**
   (persisted from a prior run) → used and shown dated ("fire as of {date}"); (3) **cold-start, no cache →
   NO-FIRE / all-ignition** (`is_fire=0`, `dist_to_fire=max`, every land cell ignition) — never invent fire.
   FIRMS stays display-only. Verified on the 2026-06-07 live run: regime = **40,964 ignition / 0 spread**,
   **peak risk 82% → 2.6%**, 0 cells ≥20% — the phantom is gone. The app shows a degraded-input banner naming
   the exact fire tier. (Variant 2 — autoregressive: predict today's label, feed it to predict tomorrow —
   noted as a future enhancement; it compounds model error into an input and needs separate validation.)
5. **Antecedent dryness fails on the present-day FORECAST path (ValueError → warm-start).** Open-Meteo's
   *forecast* endpoint returns the 91 date slots but **all-NaN daily precip/temp** for past days, so
   `regrid_to_cube` gets no valid points and raises. **✅ FIXED (2026-06-07):** `live_antecedent_dryness`
   now ALWAYS fetches the 90-day stack from the **archive** (ERA5, which has the daily precip history) and
   **drops all-NaN tail days** that fall inside archive's ~5-day latency, computing antecedents over the
   available history (a 90-day sum tolerates a few-day-stale tail; raises only if <7 valid days). Verified on
   the 2026-06-07 live run: `precip_sum_7/30/90d`, `kbdi`, `antecedent-dryness` all refreshed live.

**Finding — fire-feature train/serve mismatch (next priority).** With FIRMS enabled the same validation
dropped to pred-corr 0.10: FIRMS *active-fire* (375 m hotspots) ≠ the cube's EFFIS *burned-area >5 ha* that
the model trained on, so `dist_to_fire`/regime differ and predictions shift hard. The fire-history features
are the impactful live signal AND the most mismatched. Fix options: source **EFFIS NRT burned-area** (matches
the training definition) instead of FIRMS active-fire; or recalibrate/retrain on FIRMS-derived fire features.

**FIXED (decision + implementation).** FIRMS isn't worth relearning — its latency/resolution edge only helps
the *display*, not the model — so: **model fire features ← EFFIS burned-area (matches training); FIRMS ←
display only.** `fetch_effis.py` (open WFS `ercc.ba` → transform 4326→3035 → `rasterio.features.rasterize`
→ `dist_to_fire`; rasterize chain validated offline). `build_live_slice(fire_source="effis")` overwrites
is_fire/dist_to_fire/regime from EFFIS, **falling back to the cube's EFFIS-consistent warm-start if the
EFFIS endpoint is down** (NOT FIRMS); FIRMS only sets the display `today_fire`. Validated: decoupling FIRMS
from the model restores **pred-corr 0.10 → 0.997**. ⚠️ EFFIS open WFS backend (ies-ows) is currently
returning an OracleSpatial error (server-side, transient) → live model-fire is warm-started for now; it
switches to live EFFIS automatically when their endpoint recovers (retry + fallback in place).

## 2026-06-07 — LIVE MVP: end-to-end same-day prediction on real feeds (Open-Meteo + FIRMS)

`daily_job.py --mode live --date D` now produces a genuine live next-day prediction from REAL data:
- **weather** ← Open-Meteo (keyless, gridded→regrid; `fetch_openmeteo.py`),
- **today's fire** ← FIRMS NRT (key works — fetched 67 real detections for Spain 2026-06-04;
  `fetch_firms.py` → rasterise → `dist_to_fire` → ignition/spread regime),
- **everything else** warm-started from the cube slice with the nearest **day-of-year** (seasonal match),
- → GBT (+isotonic calibrator) → per-region alerts + grid + feature-stats into `data/serving_store/`.
Validated assembly (live-slice vs cube prediction for a cube date: MAE 0.00002, corr 0.999). `live_slice.py`
builds the slice; `.env` keys loaded via dotenv.

**Honest status — plumbing proven, values not yet trustworthy.** Only temperature + fire are live-refreshed;
antecedent dryness (`precip_sum_*`, `kbdi`), RH/pressure/wind aggregates, vegetation, and
time_since_last_fire are warm-started (now seasonally matched, but still last-year values). For trustworthy
live risk these must come live too — the full path is: Open-Meteo HOURLY → IberFire's exact aggregation for
RH/pressure/wind/u-v; a rolling precip/fire history (the accumulating daily-job store provides it over time)
for antecedents; CLMS NRT (10-day) for vegetation. That's "IberFire-v2 live" — the remaining build.

Rate-limit notes: Open-Meteo free tier limits/min → small URL-safe batches + Retry-After backoff; live grid
at 0.5° (~5 requests). FIRMS NRT covers recent dates only (use forecast/recent for live; archive for backfill).

## 2026-06-07 — Weather feed: pivot AEMET → Open-Meteo (keyless, gridded, ERA5-based — validated)

AEMET's OpenData key signup is unreliable (key emailed, page hangs) — so switched the weather feed to
**Open-Meteo** (`scripts/fetch_openmeteo.py`), which is strictly better here:
- **keyless + free**, no signup; **already gridded** (query a lat/lon grid → bilinear regrid to the cube;
  no scattered-station IDW); **ERA5-based** → matches the model's training distribution (less shift than
  AEMET stations); **forecast API** (live "today") + **archive API** (ERA5, for backfill / IberFire-v2).
- **Validated end-to-end against ground truth:** fetched 522 grid points for 2024-07-15, regridded
  t2m_mean → vs the cube's own t2m_mean: **MAE 0.83 °C, corr 0.95** (≈ the AEMET gridding budget, but
  keyless + no station interpolation + ERA5-native). This IS a cube-compatible live weather slice.
- Notes: small batches (URL-length limit) + backoff (free-tier rate limit); 0.25° grid tightens the MAE
  vs the 0.5° demo; pressure/RH need hourly→daily aggregation when wiring the full slice.
- `fetch_aemet.py` retained as the station-based alternative (+ the upstream shift validation), not the path.

**⇒ The live weather feed is unblocked NOW (no key). With FIRMS (fire) keyed, the live MVP is essentially
unblocked — remaining: wire Open-Meteo + FIRMS into `daily_job.py --mode live` (the feature-slice builder).**

## 2026-06-07 — Live-data track: replay dashboard + AEMET/FIRMS feed prototypes + the ERA5↔AEMET shift budget

Scoped the real-time path (ROADMAP §F) after reading the IberFire authors' own pipeline
([github.com/JulenErcibengoaTekniker/IberFire](https://github.com/JulenErcibengoaTekniker/IberFire)).

- **Replay dashboard** (`docker/monolith/app_live.py`): one map, three layers for a chosen "today" — today's
  fires `is_fire(t)`, tomorrow's IGNITION risk (warm, regime 1), tomorrow's SPREAD risk (cool, regime 2),
  all calibrated — plus a per-Autonomous-Community ALERT panel (real INE names) and a replay date-clock
  (honestly labelled; no faked liveness). Toggles stripped (reproject + scale always on).
- **AEMET feed prototype** (`scripts/fetch_aemet.py`): OpenData API client + `normalize_aemet` (mirrors
  upstream `process_aemet_station_data`: PRECIPITACION Ip→0/÷24, DIR×10, etc.) + **IDW station→grid** (the
  piece upstream never did — they only validated point-wise). `--demo` on cube data: gridding 250 synthetic
  stations → 40,964 cells, t2m **MAE ≈ 0.9 °C** (corr 0.94) — the station-sparsity error budget.
- **FIRMS feed prototype** (`scripts/fetch_firms.py`): active-fire CSV API + rasterise-to-grid +
  `dist_to_fire` (EDT). `--demo` round-trips the chain exactly (corr 1.000).
- **Distribution-shift budget (key §F finding)** — aggregated the authors' 758-station ERA5-vs-AEMET MAEs:
  **temperature swaps cleanly** (TMEDIA/TMAX norm-MAE ~0.04, ~1–2 °C), wind speed MED; **precipitation is
  the risk** (norm-MAE 0.23, ~0.7 mm/h, *and* hardest to grid) — and precip drives the antecedent-dryness
  features (`precip_sum_90d/7d`, `kbdi`) that are TOP new-ignition predictors. Pressure's high MAE (~9 hPa)
  is likely a fixable altitude-reference offset. **⇒ the live feed will be solid on temperature, degraded on
  precip-driven dryness — that's exactly what the §F backtest (AEMET-fed vs ERA5-fed) must measure.**

- **Shift-sensitivity test** (`scripts/shift_sensitivity.py`) — perturbed TEST meteo features by the measured
  AEMET shift, re-scored the GBT: temperature **+0.002**, wind **+0.002**, precip **−0.003** (antecedent
  *sums* average out the noisy daily precip!), pressure **−0.023**, ALL-combined **−0.021** (0.633→0.612,
  pressure-dominated — and pressure is modelled as random noise when it's really a correctable altitude
  offset, so overstated). **⇒ VERDICT: the live AEMET feed is VIABLE** — even with the full measured shift the
  model holds new-ign ~0.61 (vs the U-Net's 0.22), and the only real contributor (pressure) is fixable.

**Status:** both dynamic feeds are coded + validated on cube data; going live needs free API keys
(AEMET_API_KEY, FIRMS_MAP_KEY) + the AEMET-vs-ERA5 backtest → recalibrate (our isotonic calibrator likely
needs re-fitting on the live distribution) → drift monitoring. CLMS vegetation (10-day cadence) not yet built.

## 2026-06-07 — PIVOT: point-wise GBT beats the U-Net decisively; spatial learning doesn't help. Segmentation shelved.

The biggest finding of the project, and a reversal. After v6 (regularized wide-deep) failed to close the
v5 val→test gap (test new-ign 0.19 ≈ v5's 0.22 — so the gap was NOT overfitting), we benchmarked a
point-wise **HistGBT on the IDENTICAL eval** (same 146 features, same cells, same `regime_metrics`;
`scripts/gbt_compare.py`). Result, held-out TEST (2022-24, matched 15:1 prevalence):

| TEST new-ign AP | spread | prec@K | ROC |
|---|---|---|---|
| **GBT (point-wise): 0.633** | 0.997 | 0.453 | 0.974 |
| U-Net v5 (wide-deep): 0.216 | 0.985 | 0.267 | 0.868 |
| U-Net v6 (+reg): 0.191 | 0.983 | 0.337 | 0.854 |

**The point-wise GBT crushes the spatial U-Net (~3× on the valuable new-ignition regime), matches/beats it
on spread, and generalizes cleanly (val 0.65 → test 0.63, NO val→test gap).** The gap we'd been fighting
was a U-Net pathology, not a data limit; there is no ceiling problem — new ignitions ARE highly predictable.

**Then we dug before pivoting (`scripts/dig.py`, `scripts/dig_spatial.py`):**
- **GBT trustworthy (no leakage):** new-ignition drivers are distributed, legitimate physics — popdens &
  dist_to_roads (human access), time_since_last_fire & dist_to_fire (fire history), precip_90d/7d
  (dryness), CLC scrub/forest (fuel), doy/t2m/slope (season/weather/terrain).
- **WHY the U-Net fails — its spatial branch is net-NEGATIVE.** v5 branch ablation on TEST: deep-only
  new-ign 0.166 | wide-only 0.222 | full 0.216. The deep spatial branch is the *worst* part and drags the
  full model *below* the point-wise wide branch alone — its smoothed logits corrupt the spiky point-wise
  signal. (And the shallow wide MLP, 0.222, ≪ GBT's 0.63: trees > shallow NN on tabular point-wise data.)
- **Does spatial EVER help? No.** Gave the strong learner maximal hand-crafted spatial context (3×3 + 5×5
  neighbourhood means of all 146 features → 438): TEST new-ign 0.638 vs 0.633 = **+0.005 (noise)**. Two
  independent lines — learned (U-Net, net-negative) and hand-crafted (aggregates, negligible) — agree.

**Precise conclusion (NOT "spatial doesn't matter"):** spatial AND temporal structure are *essential* — and
the thorough **feature engineering already captures them as per-cell scalars**, which is precisely why the
GBT is so strong. Its top new-ignition drivers ARE engineered spatial features (`dist_to_fire`,
`dist_to_roads_stdev`, `elevation_stdev`) and temporal ones (`time_since_last_fire`, `precip_sum_90d/7d`,
`doy_sin`). So the finding is: **once spatial/temporal complexity is well-engineered into the features,
adding *learned* spatial processing (CNN) or neighbourhood pooling on top is REDUNDANT** — the +0.005 from
GBT+neighbourhood-means is re-deriving signal that's already there. Worse than redundant for the CNN: its
lossy downsampling + conv-smoothness actively *corrupt* the spiky point-wise target (deep branch
net-negative). **The feature engineering is the load-bearing contribution; the model should learn
point-wise over those rich features. Do NOT drop the engineered spatial/temporal features — they ARE the
signal.**

**DECISION (user, 2026-06-07): adopt the point-wise GBT as the model; shelve the segmentation U-Net.**
Forward: persist a production GBT, calibration, feature parsimony (cluster + day-level stability importance),
and wire GBT into the map app (faster + more accurate than the CPU U-Net). The U-Net work (v4-v6,
wide-deep, focal/regime loss, the whole `build_unet` path) stays in the repo + CHANGES as the documented
road that led here — the apples-to-apples comparison is exactly what made the pivot defensible.

## 2026-06-06 — v5 wide-and-deep CONFIRMS the diagnosis: point-wise branch lifts ignition (interim)

Built the wide-and-deep variant (`WideDeepUNet` in `src/models/cnn.py` + `--wide` in train.py): the v4
deep U-Net (unchanged) plus a **zero-initialized point-wise 1×1 branch** (per-pixel MLP, 146→128→64→1,
27.5K params = 0.11% of the model), fused **additively** on logits. Zero-init verified — an untrained
WideDeepUNet is bit-for-bit the deep baseline, so it can only *add* signal. Same config as v4 (focal-mass
loss, lr 5e-5, α 0.6, oversample 3, GroupNorm) except **batch 8** (see swap note below).

**Result (interim, through epoch 7; v5 still training, best @ epoch 6):**

| metric | v5 wide-deep | v4 deep-only | GBT floor |
|---|---|---|---|
| new-ign AP | **~0.37 plateau** (best 0.386) | ~0.32 (best 0.358) | 0.50 |
| **prec@K** (R-precision) | **~0.32** | ~0.10 | — |
| spread AP | 0.98–0.99 (retained) | 0.99 | ~0.98 |
| val_blend (best) | **0.626** | 0.611 | — |

**The diagnosis holds.** Adding a downsample-free, single-cell-receptive-field pathway lifted ignition
above the deep-only ceiling — new-ign broke v4's entire range by epoch 4 and settled ~0.37 (≈75% of the
GBT floor, up from ~64%). The lift is **modest on full-curve AP (+~0.05) but large on top-rank precision
(prec@K ~3×)** — consistent with the mechanism: the point-wise branch sharpens the *confident top* of the
ranking (the obvious, feature-driven ignitions), while the **stochastic ignition tail** (human/lightning
triggers absent from the features) still caps full-curve AP near 0.37, well short of 0.50. That residual
is likely **partly irreducible**, not an architecture gap. For an operational risk map, prec@K ("of the
cells we'd flag, how many burn") is the more meaningful metric — and it tripled.

**Next:** let v5 finish; run per-regime feature importance (cluster + day-level n/2 stability) on the wide
branch (its point-wise attributions are GBT-comparable) to learn ignition drivers; decide v6 from there.

**FINAL (early-stopped epoch 35, best ckpt epoch 10).** Val new-ign peaked ~0.40 (ep11), then late epochs
**overfit hard** (val spread collapsed 0.98→0.65 by ep30+). **TEST (touched once, best ckpt): new-ign
0.216, spread 0.985, prec@K 0.267, ROC 0.868.** The sobering part: **val new-ign 0.40 → test 0.22** — a
large generalization gap. Spread transfers perfectly (0.98 val=test); **ignition does not**. So on the
held-out test the wide-deep does NOT beat the GBT new-ign floor (0.22 < 0.50) — the val gains were partly
overfitting + harder test years (2022–24, incl. Spain's extreme 2022 season). **The bottleneck is now
GENERALIZATION, not architecture.** → v6 attacks it with strong regularization (wide_dropout 0.30,
decoder_dropout 0.20, weight_decay 1e-2, patience 8).

### Bugs found & fixed
- **Wide branch re-triggered hard swap at batch 16** *(found → fixed)*. The 1×1 branch is tiny in params
  but runs at FULL 230×297 resolution; its 128/64-channel activations (~840 MB at batch 16) tipped the
  already-edge memory into active disk swap (5484 swapouts/s, 939 s/epoch). Fixed by dropping to **batch 8**
  (halves all activations, free under GroupNorm) → swapouts 0, ~830 s/epoch.

## 2026-06-06 — torch.compile on MPS: hangs on our 2.9.1, but 2.12's Metal Inductor backend works

Benchmarked compile modes on our stack and looked into whether newer PyTorch changes the picture
(`scripts/bench_compile.py`, subprocess-isolated with per-mode timeouts so a hang is killed, not blocking).

**On installed torch 2.9.1 (wide-deep model, batch 8, MPS):**

| mode | result |
|---|---|
| eager | 1.40 s/step |
| aot_eager | 1.43 s/step (no gain — traces autograd, runs eager, no fusion) |
| default (inductor) | **HANG** (killed at 180 s) |
| reduce-overhead | **HANG** (killed at 180 s) |

Cause: 2.9.1's Inductor has **no Metal codegen backend** — its targets are Triton (CUDA) and C++ (CPU),
so on MPS it slides into a CPU-compile/autotune path that stalls on macOS. `aot_eager` works only because
it skips Inductor entirely (and so gives no speedup). **Verdict: eager is correct on 2.9.1.**

**Did newer PyTorch change it? Yes.** Latest stable is **2.12 (May 13, 2026)**; it added a Metal Inductor
backend (`torch/_inductor/codegen/mps.py`, `MetalKernel`). Tested in an isolated `/tmp` venv (torch 2.12,
small conv+GroupNorm+GELU probe, MPS):

| mode | first (compile) | steady median | steady min |
|---|---|---|---|
| eager | 3.4 s | 44.7 ms | 42.9 ms |
| default (inductor) | 3.6 s | 53.2 ms (noisy) | 29.5 ms |
| reduce-overhead | 2.5 s | **35.1 ms (~21% faster)** | 31.4 ms |

**Inductor no longer hangs on 2.12** — it compiles in seconds and runs; `reduce-overhead` is ~21% faster
than eager on the probe. Caveats: small probe (not the real model), measured under concurrent v5 GPU
contention, and the Metal backend is still prototype (codegen bugs on larger graphs). **Action (deferred):**
after the wide-deep experiment, do a controlled migration test — install repo deps in the 2.12 venv, run the
*real* `build_wide_deep_unet` through the timeout-guarded bench with training stopped; if it compiles cleanly
and reduce-overhead beats eager, migrate to 2.12 for ~20% faster training. Not mid-experiment (keeps the
v4-vs-v5 comparison on one stack; reproducibility).

**RESOLVED (2026-06-07, real-model test on 2.12).** Installed full repo deps into the 2.12 venv and benched
the ACTUAL wide-deep model (batch 8, 230×297): eager 1.377s, aot_eager 1.403s, default 1.543s (compiled in
19.7s — **works, no hang, no codegen bug**), reduce-overhead 1.482s. So the Metal Inductor backend is now
functional on the real model — but **slower than eager**: the probe's ~21% gain was a small-model artifact;
the real conv-heavy U-Net is already eager-optimal on MPS and inductor adds overhead. **Decision: do NOT
upgrade — stay on 2.9.1 eager. The compile path offers no speedup for this model on this hardware.**
Compile question closed. (Epoch cost stays ~13 min, which caps overnight experiment throughput — the real
limiter, not compile.)

## 2026-06-06 — v4 baseline established: spread solved, ignition plateaus below GBT (architecture-bound)

Ran `seg_coarse4_focal_v4` (focal-mass loss, lr 5e-5, batch 16, α 0.6, oversample 3, GroupNorm,
corrected eval) for 14 epochs and stopped it — it had clearly converged into a plateau. This is the
clean, honestly-measured **baseline** for the next-day fire U-Net.

**Results (val, AP @ matched 15:1 prevalence, per-regime adjustment applied — comparable to GBT):**
- best checkpoint = **epoch 5**: val blend **0.611**, **new-ign AP 0.358**, **spread AP 0.991**, ROC 0.909, prec@K 0.11.
- spread is **solved**: 0.96–0.99 throughout, at/above the GBT floor (~0.98).
- new-ign **plateaus ~0.32** (range 0.27–0.36 across 14 epochs, peak 0.358 @ epoch 5), **well below the
  GBT floor of 0.50**, while train loss kept falling (6.2 → 0.66, ignition 9.6 → 0.75). Converged
  ceiling + mild overfitting — NOT undertraining.

**Diagnosis — inductive-bias mismatch (not capacity / time / gradient budget).** The decisive clue is
the regime asymmetry: the U-Net **matches GBT on the spatial regime (spread 0.99)** but **underperforms
the point-wise GBT on ignition (0.32 vs 0.50)**. New ignition is a near-point-wise, spatially *sparse*
event (a single cell lights up from its local fuel/weather/ignition-source conditions); the U-Net's two
core priors fight that — (1) the resnet34 **encoder's internal** 32× stride (an architecture property, NOT
the 4 km data coarsening: it downsamples the 230×297 grid to a ~8×10 bottleneck = ~128 km per deepest
cell, before the decoder upsamples back) blurs the per-cell signal GBT keys on, and (2) convolutional
smoothness spreads probability over neighborhoods, killing precision-at-top on a spiky target. Those same priors are *assets* for spread, hence 0.99 there. Ruled out: capacity (would
hurt both regimes), training time (new-ign peaked @ epoch 5 then drifted; loss already low), alpha
(ignition isn't gradient-starved — its loss descends fine).

**Test set deliberately NOT touched** — reserved for a single head-to-head against the planned
improvement, to keep the touched-once discipline.

**Next: wide-and-deep v5.** Add a "wide" point-wise branch (stacked 1×1 convs = per-pixel MLP, the
GBT-style pathway) fused **additively** with the existing U-Net logits, with the wide head **zero-
initialized** so training starts bit-for-bit at the v4 baseline and can only add. Regularize the wide
branch (Dropout2d) since a per-pixel MLP on 146 features can overfit the few ignition events. Expected:
≥ GBT on ignition (the wide branch guarantees the point-wise pathway), > 0.50 where spatial context
helps; spread retained via the unchanged deep branch. Loss / dataset / eval all unchanged — drop-in
model swap behind the `build_unet` factory.

## 2026-06-06 — The model was fine; the EVAL was broken (twice). Two metric bugs, not model bugs

`focal_v3` (focal-mass loss, lr 5e-5, batch 16) trained and looked like it was *failing*: val
new-ign AP ~0.0006 (vs a "GBT floor" of 0.50 — ~1000× short), spread AP drifting to **0.0000**, and
val ROC sliding **0.846 → 0.715** over 4 epochs. I read that as overfitting + spread starvation and
stopped to investigate. **Both readings were wrong — the model was healthy; the eval lied twice.**

**Bug A — prevalence mismatch (the 1000× "gap").** AP is acutely prevalence-dependent (its random
baseline *is* the prevalence). `train.py` scored AP against **all** land cells (true ~0.005 % new-ign
prevalence), while the GBT measurement floor subsampled negatives to `neg_ratio=15` (~6 % prevalence).
The two prevalences differ ~1200×, which fully accounts for the ~1000× AP "gap". In lift-over-random
terms the U-Net (~12×) was already on par with / ahead of the GBT (~8×). The bar was never comparable.

**Bug B — missing inference adjustment (the real one).** The loss trains the model to produce logits
that are correct *after* a per-regime logit adjustment (ignition −10.90, spread −2.67). But
`evaluate()` scored **raw** logits, with no adjustment — so cross-regime ranking was garbage. Applying
the adjustment at inference (legitimate: regime is known from `dist_to_fire(t)`) on the *same* epoch-4
checkpoint:

| metric (matched 15:1 prevalence) | raw logits | + per-regime adj | GBT floor |
|---|---|---|---|
| spread AP | 0.068 | **0.991** | ~0.98 |
| new-ign AP | 0.318 | 0.309 | 0.50 |
| prec@K (R-precision) | 0.002 | **0.096** | — |
| ROC | 0.715 | **0.899** | — |

Spread wasn't starved — it had learned to **GBT level (0.99)**. And the ROC "decline" wasn't
overfitting: as the model trained it leaned *more* on the adjustment, so raw-logit ranking decayed
while true performance *improved*. We were watching a broken gauge fall while the engine was fine.

**Fixes (eval only — the model/loss/training config were left unchanged):**
- `evaluate()` now applies the per-regime adjustment before sigmoid (threads `adj_ign`/`adj_spr` from
  the priors). **Serving must do the same** — the monolith's raw `sigmoid(model(X))` will need the
  per-regime adjustment once a seg model ships (flagged for the serving path).
- `regime_ap` → `regime_metrics`: reports **prevalence-matched AP** (negatives subsampled 15:1, same
  seed each epoch → comparable across epochs AND to the GBT floor), **precision@K** (R-precision,
  operational), and full-prevalence ROC. Early-stop blend now keys off the matched, adjusted APs.
- Relaunched as `seg_coarse4_focal_v4` (identical training config; corrected eval + meaningful early stop).

### Bugs found & fixed
- **AP measured at a different prevalence than the GBT floor** *(found → fixed)*. Full-image negatives
  vs the floor's 15:1 subsample → ~1000× incomparable AP; misread as the model failing. Fixed with a
  prevalence-matched AP in `regime_metrics`.
- **`evaluate()` scored raw logits without the per-regime adjustment** *(found → fixed)*. The model is
  trained to be correct post-adjustment; raw scoring tanked spread AP (0.99→0.07) and ROC (0.90→0.71)
  and faked a ROC "decline". Fixed by applying the adjustment at inference. Impact: two training runs
  (v2 diagnosis aside, v3) were judged failures when the model was actually performing well.

## 2026-06-06 — First training run collapsed; focal loss is the fix (diagnosed, not guessed)

The first real run (logit-adjusted BCE, lr 3e-5) **failed to learn** — val AP ≈ 0 in *both* regimes and
val ROC < 0.5 (worse than chance). Rather than tweak blindly, I ran an **overfit-a-tiny-batch diagnostic**
(`scripts/diag_overfit.py`): grab 8 fire-containing days (13 fire pixels total) and try to *memorize* them.
A model that can't overfit 13 pixels has a broken loss/gradient, not a capacity or data problem. The test
A/B's four recipes and reports train AP + mean predicted prob on fire vs no-fire cells:

| recipe | train AP | ROC | meanP fire vs no-fire |
|---|---|---|---|
| current (logit-adj BCE, lr 3e-5) | **0.00** | 0.75 | 0.668 vs 0.634 |
| lr 1e-4 | **0.00** | 0.72 | 0.648 vs 0.620 |
| **focal γ=2 (lr 1e-4)** | **0.77** | **1.00** | **1.000 vs 0.661** |
| tempered adjustment (lr 1e-4) | 0.00 | 0.64 | 0.482 vs 0.529 (anti-ranked!) |

**Conclusion (conclusive):** plain logit-adjusted BCE *cannot overfit even 13 fire pixels* — fire and
no-fire probs end up nearly equal (0.668 vs 0.634). The mean BCE over ~99.99 % negative cells dilutes the
rare-positive gradient to nothing; the logit adjustment **shifts the decision boundary but does not change
the gradient magnitude**, so it can't fix dilution. Higher lr alone doesn't help (it's dilution, not speed);
*tempering* the adjustment makes it worse (anti-ranks). **Focal (γ=2) is the cure**: down-weighting easy,
confident negatives by `(1−p_t)^γ` lets the rare positives dominate — it overfits the batch (AP 0.77, ROC
1.00, fire cells → prob 1.0). A unit test on synthetic data confirms the mechanism: focal lifts the
positive:negative gradient ratio from **40:1 to 3447:1** (the easy-negative gradient drops ~80×).

**Fix shipped:**
- `RegimeLogitAdjustedBCE` gained a `focal_gamma` knob (default 0 = unchanged; training uses **2**): applies
  `(1−p_t)^γ` modulation on top of the per-regime logit adjustment, keeping both orthogonal knobs (adjustment
  = boundary/prior, focal = un-dilute gradient, α = regime budget).
- `train.py`: `--focal-gamma` (default 2.0), lr default → **1e-4**, loss logged as `RegimeFocalLogitAdjustedBCE`.
- Relaunched as `seg_coarse4_focal_v2` (batch 32, α 0.6, oversample 3, GroupNorm, blend early-stop, 300/patience 25),
  wrapped in `caffeinate -i` (the prior run died when the laptop slept).

**...which exposed a SECOND bug (the dilution one level up).** `focal_v2` trained 9 epochs and *diverged*:
train loss fell steadily (0.034→0.010) but val ROC drifted **down** (0.78→0.69) and AP stayed pinned at the
floor — the overfitting signature, not pre-climb stagnation. The tell was in the loss components: the
**ignition term was frozen at 0.0004 from epoch 1** while spread fell 0.083→0.024. The model learned *only*
spread and ignored ignition entirely. (Confirmed the val metric was real, not degenerate: the strided val
subset holds 761 new-ign + 603 spread positive pixels.)

**Root cause:** the regime reduction averaged each regime's focal loss over its **total cell count** (~40k for
ignition), re-diluting the handful of positives that focal had just rescued — the *same* dilution, one level
up. So `L_ignition ≈ (few positives × ~1)/40,000 ≈ 1e-4`, and α=0.6 just scaled a near-zero number; the spread
term out-gradiented ignition ~150:1 every step. The overfit diagnostic still passed earlier only because 120
repeats of one fixed batch let a tiny gradient accumulate. This is exactly what RetinaNet guards against by
normalizing focal loss by **#positives**, not #anchors — I'd divided by #cells and reintroduced it.

**Fix #2:** normalize each regime by its **focal-weight mass** (`Σ focal_weight`, detached, floored), i.e. a
focal-*weighted* mean of BCE — positives + hard negatives dominate, easy negatives drop out, loss stays O(1)
so lr/α are unchanged in scale. (With `focal_gamma=0` the mass reduces to the cell count → old behavior, back-
compat.) The lr-sweep overfit diagnostic (now exercising the **real** `RegimeLogitAdjustedBCE`) confirms it:

| recipe (real loss, focal-mass norm) | loss 0→120 | train AP | new-ign | ROC |
|---|---|---|---|---|
| lr 5e-5 | 2.46 → 0.009 | **1.000** | **1.000** | **1.000** |
| lr 2e-5 | 2.46 → 0.038 | 1.000 | 1.000 | 1.000 |
| lr 1e-5 | 2.46 → 0.124 | 1.000 | 1.000 | 1.000 |
| lr 5e-6 | 2.46 → 2.026 | 0.094 | 0.094 | (too slow) |

Note the *new* normalization overfits to **AP 1.000** where the buggy one capped at 0.770 — the dilution was
hurting even the overfit test. **Relaunched as `seg_coarse4_focal_v3`: lr 5e-5, batch 16, focal-mass loss.**

**Batch 32 → 16 (swap fix).** `focal_v2` ran the machine into paging (23 % free + ~106k pageouts during; 81 %
free after stop), and the ~600 s epochs were likely part swap-stall. The biggest controllable consumer is
U-Net activation memory on MPS, which scales with batch. Because we chose **GroupNorm** (batch-independent),
halving the batch is statistically **free** — so batch 16 cuts the dominant unified-memory cost with no
normalization penalty, and should stop the swap.

### Bugs found & fixed
- **Loss collapse under extreme imbalance** *(found → fixed)*. Mean-reduced regime BCE diluted the
  rare-positive (ignition: 0.004 % pos) gradient so far that the model converged to all-negative — val AP ≈ 0,
  ROC < 0.5. Impact: the entire first training run learned nothing. Fixed by adding focal down-weighting
  (γ=2), proven necessary by the overfit diagnostic above (the logit adjustment alone shifts the boundary but
  does not undo dilution).
- **Regime-mean re-dilution (focal normalized by cell count, not positives)** *(found → fixed)*. The per-regime
  reduction divided each regime's focal loss by its total cell count (~40k), re-diluting the positives focal had
  rescued → ignition loss frozen at 1e-4, `focal_v2` diverged (train↓ / val ROC↓, AP floored) over 9 epochs.
  Impact: the second training run learned spread only, never ignition. Fixed by normalizing by the detached
  focal-weight mass (RetinaNet-style #positives normalization); lr lowered 1e-4 → 5e-5 for the new loss scale,
  batch 32 → 16 (free under GroupNorm) to end the swapping.

## 2026-06-06 — I/O optimization: training is now GPU-bound (~4× faster epochs)

**Bottleneck found:** data loading, not compute — at batch 32 / `num_workers=0`, ~12.5 s/step loading
(48 dynamic features = 48 separate compressed zarr chunk reads/sample) vs ~4 s/step MPS compute → **GPU
only ~24 % utilized.**

**Fix — a training-optimized dynamic-feature stack + thread prefetch:**
- `scripts/build_training_array.py` → `data/gold/IberFire_coarse4_dyn.zarr`: the 48 *dynamic* features
  pre-stacked into `(time, channel, y, x)`, **float16, pre-normalized** (train stats), **Blosc-lz4**,
  chunked `(1, C, y, x)`. So a day = **1 contiguous read + 1 decompress** (statics stay RAM-cached,
  calendar broadcast). Built via `_build_X`, so values are identical to the live pipeline. + a precomputed
  `regime` array. ~14 GB.
- `StackedRegimeIberFireDataset` (datasets.py): reads the stack for dynamic channels, assembles cached
  statics + broadcast calendar — same `(X, y, regime)` contract.
- **Thread prefetcher** in `train.py` (`--use-stack`): overlaps loading with compute *without* multiprocess
  workers (GIL released during zarr I/O; avoids the worker CPU-contention measured earlier).

**Result:** loading **13.7 → 1.5 s/step (9×)**; since 1.5 s < 4 s compute, the step is compute-bound →
**GPU ~100 %**, epoch **~16 → ~4 min**, startup 5 min → ~51 s.

**float16 verified safe:** model trains in fp32 (only stored *inputs* are fp16); round-trip error max 0.004 /
mean 8e-5 on normalized features — far below the data's own noise floor. Full-X parity vs the reference loader
confirmed (max|Δ|≈0.008, y/regime exact).

### Bugs found & fixed
- **xarray append-mode drops group attrs** *(found → fixed)*. `to_zarr(mode="a", append_dim=...)` didn't
  persist the `dyn_features` attr written on the first block, so the loader `KeyError`'d. Fixed by reading the
  channel names from the stored `channel` *coordinate* (survives appends) instead of a group attr.

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
