# ABLATIONS

A committed registry of **ablation studies** for the Fire Guard Datacube (FGDC) and models. Every major
change — a new feature group, a new data source, a target-definition change, or a representation choice like
temporal interpolation — earns an entry here **before it's accepted as "kept"**.

**Why:** academic rigor (we only keep what demonstrably helps) and presentation (a clean "we tested X, here's
the lift" table tells the story far better than a list of features). It also guards against the parsimony
trap — our feature set is already rich (v1: 146→93 cost almost nothing), so additions must *earn* their place.

**Method (the rule):** toggle ONE thing, hold everything else fixed — same days, same train/val split, same
model, same metric. Reproduce with `scripts/fgdc_ablation.py` (`--groups` for leave-one-feature-group-out,
`--horizons` for target-definition). Metric = **average precision** (the rare-event metric that matters for
~0.1%-positive fire) + ROC-AUC, on a chronological 80/20 split. HistGBT handles NaN natively.

**Entry template:**
> ### <date> — <change>
> - **Idea:** what the change is.
> - **What it solves / hypothesis:** the imperfection or signal it targets.
> - **Setup:** cube/window, target, model, split, metric, what's toggled.
> - **Result:** with-vs-without table (Δ metric).
> - **Verdict:** keep / drop / needs-more-data — and why.
> - **Caveats.**

> ✅ The **2026-06-18 FULL-SPAN entry** (2012→2026-05, 5265 days) is now the **authoritative** group/horizon
> result. The earlier slice/summer entries below are kept as the *method-development* record (and to show how
> the `fire_context` sign flipped once enough seasons were present) — but they are **superseded** for final
> magnitudes by the full-span run.

---

## Planned / in-progress
- **Temporal interpolation of slow layers** (GHS-POP / built-up / CLC proportions: step-snap vs
  linear-interp between editions) — *to run when P3 lands; user-prioritized.* Hypothesis: removing Jan-1
  step discontinuities removes a calendar artifact the model can key on and better reflects gradual change.
- ~~Re-run all entries on the full 2012→present backfill~~ **DONE 2026-06-18** (see the full-span entry below).
- ~~Regime-split / new-ignition-restricted ablation~~ **DONE** (full-span entry 2026-06-18): RAW features →
  new-ign AP **0.483**, spread **0.968**. Confirmed the blended metric was hiding the hard ignition regime.
- ~~P4 engineered fire-weather~~ **DONE** — materialized into the cube; **new-ign 0.483 → 0.547 (inline P4-A)
  → 0.622 (production model, 135-feature `features_fireguard`)**, spread 0.984. **FGDC matches v1's ~0.63 bar.**
  Lift concentrated in `time_since_last_fire`; engineered drought/fire-weather is real but cross-correlated
  with fire-memory (recoverable in leave-one-group-out). See CHANGES.md 2026-06-21.
- ~~t+1 forecast features (does tomorrow's-weather beat t-only?)~~ **DONE 2026-06-26** — yes, +0.0054 new-ign
  (entry above); production fill deferred.
- ~~external benchmark vs EFFIS/FWI~~ **DONE 2026-06-26** — the baseline-floor panel (`baseline_panel.py`):
  GBT ~8× the FWI-index, ~2× per-cell×doy climatology on new-ignition; persistence shows spread is near-nowcast.
- **Still open** (next phase): CLC multi-edition + temporal interpolation (P3); Optuna on the frozen feature set
  (then refit-on-all + isotonic calibration); the **forecast-weather production fill** (full GEFS backfill →
  materialize → retrain) if/when scoped as a deliberate push.

---

### 2026-06-26 — GEFS d+1 forecast weather (+CAPE) complement → **KEEP (proven; real but modest)**
- **Idea:** add tomorrow's *forecast* of the weather block (the GEFS run issued at t, valid t+1) as **extra**
  channels alongside the observed t-weather (complement, not substitute). 19 forecast-weather + 2 CAPE = 21
  `*_fc1` channels, regridded from a GEFSv12-reforecast bronze (control member, idx byte-range, stream-delete).
- **Setup:** 2016–2019 reforecast slice; internal split **train 2016–2018 / val 2019** (365 val days, 7,265 pos);
  complement = 135 base + fc; HistGBT; regime AP at matched prevalence. `scripts/train_gbt_fc1_slice.py`,
  `src/data/ingest/ingest_weather_gefs.py`.
- **Result (four-way):**

  | variant | new-ign | Δ | spread | overall | prec@K | Δ | roc |
  |---|---|---|---|---|---|---|---|
  | baseline (135) | 0.6308 | — | 0.9779 | 0.7485 | 0.3436 | — | 0.9292 |
  | +CAPE (2) | 0.6339 | +0.0031 | 0.9781 | 0.7504 | 0.3447 | +0.0011 | 0.9305 |
  | +weather (19) | 0.6353 | +0.0045 | 0.9777 | 0.7526 | 0.3484 | +0.0048 | 0.9318 |
  | **+both (21)** | **0.6363** | **+0.0054** | 0.9782 | 0.7533 | **0.3489** | **+0.0054** | 0.9318 |
- **Verdict — KEEP.** Real, **coherent** lift (all metrics up together — contrast the calendar block's flat
  +0.0015 with mixed signs). +both = **+0.0054 new-ign / +0.0054 prec@K ≈ 2/3 of the perfect-foresight ceiling**
  (+0.008, measured earlier on the main val by time-shifting reanalysis), exactly as predicted by the high d+1
  forecast skill (temp **corr 0.95, debiased MAE ~1 °C** over 174 days).
- **Split:** **weather is the primary driver** (+0.0045, and ~all the prec@K gain). **CAPE alone is a real but
  smaller signal (+0.0031 new-ign)** and **largely redundant with weather** — combined (+0.0054) ≪ individual
  sum (+0.0076); CAPE's *incremental* value over weather is only +0.0009. CAPE helps ranking (AP) more than
  the operational top-K.
- **Caveats:** (a) GEFS forecast is a *different model* than the cube's ERA5 t-weather, but its bias is a
  constant offset the GBT absorbs (debiased MAE confirms). (b) reforecast-era slice (2016–2019) ≠ production
  val (2023–2026); physics carries. (c) **Bronze covers only 2016–2019** — a *live production* feature still
  needs the full-span backfill (2012–2019 reforecast + operational 2020–2026) → materialize → retrain.
- Finding banked; features kept (`data/bronze/fireguard/weather_fc1`). See [[fgdc-build-state]].

### 2026-06-23 — Calendar / holidays + HDW / VPD (human-ignition proxies + fire-weather couplings) → **DROP (all 12 — holidays/dow flat in isolation too)**
- **Idea:** append 12 features to the production set — `vpd_peak`, `hdw` (instantaneous fire-weather couplings
  at day *t*) + a 10-channel calendar block: `doy_sin/cos` (target day *t+1*), `dow_sin/cos` at *t* and *t+1*,
  and `is_holiday_{national,regional}` at *t* and *t+1* (regional via the `holidays` lib subdiv keyed off
  `AutonomousCommunities`; national/regional kept non-redundant).
- **Hypothesis:** human-caused ignition follows a weekly/holiday rhythm the cube didn't encode (the only
  *genuinely new* signal); HDW/VPD add hot-dry-windy couplings. The `t` vs `t+1` pairing was meant to span the
  nowcast ("ignited on today's holiday, detected tomorrow") vs forecast ("ignites on tomorrow's holiday") cases.
- **Setup:** `FireGuard_coarse4` (5265 days, 2012→2026-05), target `is_fire(t+1)`, horizon=1, HistGBT (400 iters),
  chronological 80/20, TRAIN neg 30:1 / VAL full-prevalence, regime split @ 6 km, AP at matched 15:1 — **identical
  to the production 135-feature run, toggling ONLY the +12 features** (135 → 147).
- **Result:** (held-out val, n_pos 14,673)

  | metric | 135 | 147 | Δ |
  |---|---|---|---|
  | **new-ignition AP** | 0.6215 | 0.6230 | **+0.0015** |
  | spread AP | 0.9835 | 0.9831 | −0.0004 |
  | overall AP | 0.7484 | 0.7492 | +0.0008 |
  | prec@K | 0.3230 | 0.3217 | −0.0013 |
  | ROC | 0.9316 | 0.9314 | −0.0002 |
- **Verdict — DROP all 12.** The bundle (147) was flat (new-ign +0.0015; prec@K/ROC down). The **isolated
  re-test (2026-06-24): 135 + the 8 holiday/dow channels = 143 features** (redundant `vpd_peak`/`hdw`/`doy_*`
  dropped) is flat-to-negative too:

  | metric | 135 | 143 (+holiday/dow) | Δ |
  |---|---|---|---|
  | new-ignition AP | 0.6215 | 0.6214 | −0.0001 |
  | spread AP | 0.9835 | 0.9831 | −0.0004 |
  | prec@K | 0.3230 | 0.3170 | −0.0060 |
  | ROC | 0.9316 | 0.9315 | −0.0001 |

  Even isolated from the redundant fire-weather/seasonality, the human-ignition signal adds nothing — new-ign is
  dead flat and prec@K drops 0.006 (8 dead channels mildly *hurt* the operational top-K). **Hypothesis rejected:**
  weekly/holiday ignition timing is already fully proxied by `popdens`/`dist_to_roads`/`dist_to_urban` (and the
  VIIRS active-fire label, spanning multi-day burns, smears the weekly pulse). Production stays 135
  (`gbt_fireguard_135.joblib`; flat 147 kept as `_147`, 143 as `_143holdow`). **All 12 vars remain in the CUBE**
  (free on disk) for a possible future calendar × forecast-weather *interaction* test, but are **dropped from the
  production feature set** → `FGDC_FEATURE_VARS` trims back to 135.
- **Why flat:** redundancy. (a) `hdw`=VPD×wind and `vpd_peak` are collinear with `ffwi` + the raw t2m/RH/wind
  channels already in the set. (b) `doy` seasonality is already carried by the weather itself and by the seasonal
  anomalies (`spi_90d`, `ndvi_anomaly`, which are *built* from doy climatology). (c) holidays/dow were the only
  new signal and it's weak — human-ignition timing is already proxied by `popdens`/`dist_to_roads`/`dist_to_urban`.
  Consistent with the **baseline panel** (same session): a linear logistic was only 0.06 behind the GBT on
  new-ignition AP — **the engineered features already carry the signal, so more *correlated* features don't help.**
- **Strategic read:** the headroom is **not** in human-activity proxies but in the **physical t+1 driver** the cube
  genuinely lacks — a *forecast of tomorrow's weather* (new information), not a calendar flag (redundant). This
  reframes the branch toward the forecast-weather build; the ceiling experiment (perfect-foresight t+1 reanalysis)
  is the next test of whether that headroom exists.
- **Caveats:** the features are kept in the cube for a later **calendar × forecast-weather interaction** test
  (holiday flags may matter more once paired with tomorrow's weather); the weekly ignition pulse may also be
  partly smeared by the VIIRS active-fire label spanning multi-day burns.

---

### 2026-06-18 — FULL-SPAN re-run (2012-01-01 → 2026-05-31, 5265 days) — the trustworthy verdict
**The full-backfill re-run the slice/summer entries were waiting for.** All three feeds complete over 14.5
years — EDH ERA5 weather (native 0.25°, regridded on read) + FIRMS VIIRS fire + MODIS/MPC veg → silver
(BitRound-12 compressed, 150 GB) → 4 km gold (5265 days) → ablation. **Supersedes the slice/summer entries
for the group magnitudes.**
- **Method note (new):** TRAIN negatives subsampled to 30:1 (all positives kept) so the ~50 M+ rows don't OOM;
  VAL kept at FULL prevalence → val AP stays comparable to the prior full-prevalence entries. Same HistGBT,
  chronological 80/20, within-3d default. (train_rows 6.07 M, val_rows 32.77 M, val pos 35,754.)
- **Horizon (full feats):** AP 0.150 (1d) → **0.169 (3d)** → 0.152 (7d); ROC 0.910 → 0.878 → 0.848. **3d is the
  sweet spot**; on full data 1d is now competitive (vs the ~9× gap on the 31-day slice) and 7d falls off.
- **Leave-one-group-out (3d, full AP 0.169, ROC 0.878):**

  | dropped group | #feats | ΔAP |
  |---|---|---|
  | **fire_context** (dist_to_fire) | 1 | **+0.042** (dominant) |
  | human (popdens, dist_to_roads, artificial) | 4 | **+0.020** |
  | terrain (elevation, slope) | 2 | **+0.014** |
  | fuel_cover (CLC forest/scrub) | 2 | +0.005 |
  | weather (13 RAW: t2m/RH/pressure/wind/precip/soil) | 13 | +0.001 |
  | soil_moisture | 2 | −0.003 |
  | vegetation (NDVI/EVI/LAI/FAPAR/LST) | 5 | −0.005 |
- **Verdict — `fire_context` is the dominant driver; its prior negative was a WINDOW ARTIFACT.** On the
  single-window runs it scored NEGATIVE (−0.020 summer, −0.005 slice) and was flagged "surprising, needs a
  closer look." On 14.5 years it is **+0.042, the #1 group** — recency/proximity of fire is the strongest
  signal once enough seasons are present. Artifact confirmed and resolved. **human (+0.020) and terrain
  (+0.014) robustly confirmed**; fuel_cover modest-positive.
- **Weather & vegetation are still ~0 marginal even on full multi-year/multi-season data — but this is NOT a
  drop signal**, for two specific reasons:
  1. **RAW weather only.** The 13 weather vars are raw daily aggregates — NOT the ENGINEERED fire-weather that
     carried weather in v1 (`precip_sum_90d` drought-memory was v1's *top* weather driver; VPD/HDW/FFWI/KBDI).
     Those are **P4**, absent from this cube. So "raw weather +0.001 marginal" is *expected*; the real weather
     test is the engineered drought/fire-weather memory.
  2. **Blended target.** Leave-one-GROUP-out measures *marginal* value, and within-3d blends ignition + spread.
     Weather's conditional value is on the spread/extreme-danger regime, washed out in the blend (and weather
     is cross-correlated with fire_context/soil/veg → a tree recovers it). The **regime-split** is the test that
     should expose it.
- **Regime split + FIRST IberFire A/B** (same full-span held-out val; ignition vs spread at the **6 km**
  threshold = v1's `regime_dist_cells=1.5`; AP at **matched prevalence** — negatives subsampled 15:1 per
  regime, exactly v1's `regime_metrics` recipe; reuses the full-feature GBT's val predictions):

  | regime | FGDC (raw, pre-P4) | v1 bar |
  |---|---|---|
  | **spread** | **0.968** | ~0.98 |
  | **new-ignition** | **0.483** | **0.63** |
  | ROC (all) | 0.878 | — |
  - **Spread matched** (0.968 ≈ v1's ~0.98) — the easy regime (fire already adjacent) is essentially solved,
    same as v1.
  - **New-ignition 0.483 vs 0.63 — but this is FGDC-RAW vs v1-ENGINEERED, not like-for-like.** The cube has
    only 29 raw vars + inline `dist_to_fire`; it LACKS the engineered features that carry v1's ignition skill
    (`time_since_last_fire`, `precip_sum_90d` drought-memory, VPD/HDW/FFWI/KBDI, `burn_frequency`, calendar).
    Reaching **~77 % of v1's new-ign AP on raw features alone, with zero train/serve gap** (everything
    self-sourced) is a strong starting point; the **0.147 gap is the P4-engineering gap, not a source gap.**
  - **Closes the blended-metric puzzle:** weather scored +0.001 marginal in the blend, yet the split exposes a
    hard ignition regime with real headroom — and v1's evidence says engineered *weather memory* is what lifts
    it. So the blend WAS hiding weather's conditional value, as hypothesized.
  - **Caveats:** label differs (FGDC VIIRS active-fire vs v1 EFFIS burned-area); the val is the held-out recent
    ~20 % (≈ 2023→2026), not v1's exact 2022–2024 test slice. Method is apples-to-apples (same threshold +
    matched prevalence) even where the label isn't.
- **Verdict → P4 is the validated lever.** FGDC's operational sources already deliver a working model
  (spread matched, ignition ~77 % on raw); the path to v1-parity on new ignitions is **P4 engineering**
  (drought-memory / fire-weather / time-since-fire / FRP-weighted fire_context), not more sources. **No group dropped.**
- **Caveat:** absolute AP (0.169) is below the summer-2015 slice (0.214) because the full span averages in many
  low-fire winters (pos-rate 0.109% vs the summer slice's higher rate); full-prevalence AP is prevalence-
  sensitive, so **ROC 0.878 (prevalence-independent) is the honest cross-entry comparison**. Artifact JSON:
  `reports/fgdc_ablation_full.json`.

### 2026-06-13 — Summer-2015 re-judgment (Jun–Sep, 122 days, ARCO ERA5 weather)
**The test the Jan–Jul window couldn't do.** Weather sourced from the public **ARCO-ERA5 Zarr** (no CDS
account; full hourly fidelity), + VIIRS fire + MODIS veg + GHS + v1 static → silver → coarsen → ablation.
Full pipeline validated end-to-end on the new weather source (full AP 0.214, ROC 0.902, 6,481 pos / 3.7 M rows;
drivers sane, human dominant).
- **Horizon (full feats):** AP 0.133 (1d) → **0.214 (3d)** → 0.231 (7d); ROC 0.923 → 0.902 → 0.893. Multi-horizon
  win holds; same shape as Jan–Jul.
- **Leave-one-group-out (3d):** human **+0.066** (dominant), fuel_cover **+0.021**, terrain **+0.013**,
  soil −0.006, weather −0.012, vegetation −0.013, fire_context −0.020.
- **Verdict — the "low-season artifact" hypothesis is NOT supported.** Even in peak fire season, weather and
  vegetation show flat-to-slightly-**negative** *marginal* ΔAP — same sign as Jan–Jul. So the earlier
  explanation ("negative because it's winter") was too optimistic. **But this is still NOT a drop signal:**
  (1) magnitudes are tiny (≤0.02 AP) on a single summer / single split — within noise; (2) leave-one-GROUP-out
  measures *marginal* value, and weather/veg/soil/fire-context are heavily **cross-correlated** (drought shows
  in weather AND soil AND veg-anomaly; `dist_to_fire` correlates with burn-history) → a tree recovers the
  signal from the rest, so redundant groups score ~0 even when individually predictive; (3) the target is
  **blended** over all cells (dominated by human-accessible WUI), which washes out weather's *conditional*
  value on extreme-danger / spread-regime days (the v1 measurement-floor found weather+wind+proximity carry
  the SPREAD regime). **fire_context −0.020** (dropping `dist_to_fire` raised AP) is surprising and flagged for
  a closer look — likely redundancy with burn-history features under the within-3d target.
- **Next (for a trustworthy verdict):** the **full multi-year backfill** (statistical power across many
  seasons) + a **regime-split / new-ignition-restricted** ablation where weather & fire-context should show
  their conditional value. No group dropped.

### 2026-06-10 (update) — Multi-month re-run (Jan–Jul 2015, 184 days, all feeds)
First multi-month run (vs the 31-day slice): fire + GHS-POP/BUILT + 2015 weather/veg, chunked-built cube.
- **Horizon:** AP 0.150 (1d) → **0.257 (3d)** → 0.253 (7d); ROC 0.919 → 0.883 → 0.854. **AP peaks at 3d then
  plateaus**, ROC monotonically falls → multi-horizon win holds, but **~3d is the sweet spot**, not "longer
  is better." (Next-day AP also rose 0.026→0.150 vs the slice — more data + GHS + veg lift base skill.)
- **Leave-one-group-out (3d, full AP 0.257):** human **+0.070** (dominant, ↑ from slice's +0.030), terrain
  **+0.022**, fuel_cover −0.003, fire_context −0.010, soil −0.011, weather −0.011, **vegetation −0.033**.
- **Verdict:** human-features dominance is now robust; terrain confirmed; **GHS-POP/BUILT investment
  validated**. **⚠️ Seasonal caveat:** this window is mostly LOW fire season (Jan–Jul) — veg (green spring,
  non-discriminative) and weather (no summer drought/heat signal) show *negative* ΔAP, which is a
  **window artifact, NOT a reason to drop them**. They must be re-judged on **summer-inclusive** data (veg
  backfill is filling toward summer 2015 now). The slice/H1 entries below are superseded by this for the
  group numbers; all get a final re-run on the full multi-year backfill.
- **Scale note:** validated `build_silver`'s new INCREMENTAL chunked write — 184 days built with bounded
  memory (the old all-in-RAM path would need ~22 GB).

### 2026-06-10 — Target horizon: next-1d vs within-3d vs within-7d
- **Idea:** predict "fire within N days" (forward rolling-OR of the daily VIIRS label) instead of only t+1.
- **What it solves:** next-day ignition is the sparsest, most *stochastic* target (exact ignition day ≈ noise);
  a horizon target is denser, smooths timing noise, and matches operational multi-day outlooks.
- **Setup:** FGDC gold (Aug-2016 slice, 4 km), HistGBT, full features, chronological 80/20, val AP/ROC-AUC.
  Only the label horizon is toggled (`fgdc_ablation.py --horizons`).

  | target | pos-rate | val AP | val ROC-AUC |
  |---|---|---|---|
  | within 1d | 0.092% | 0.026 | 0.874 |
  | within 3d | 0.221% | **0.148** | 0.845 |
  | within 7d | 0.450% | **0.226** | 0.898 |
- **Verdict: KEEP (strong).** AP rises ~9× from 1d→7d. Adopt **multi-horizon** in production (multi-output;
  retain t+1 so users get a risk *curve*). Re-confirm magnitudes on the full backfill.
- **Caveats:** 31-day slice, small val set — directional. Longer horizons are inherently denser (easier AP);
  the operational choice trades horizon length vs actionability, not just AP.

### 2026-06-10 — Leave-one-feature-group-out (within-3d target)
- **Idea:** drop one feature group at a time; measure the AP it was contributing.
- **What it solves:** tells us which families carry signal (guards the parsimony trap — additions must earn
  their place) and where to invest data effort.
- **Setup:** as above; `fgdc_ablation.py --groups`; full model **AP 0.148, ROC-AUC 0.845**.

  | dropped group | #feats | ΔAP (full − without) |
  |---|---|---|
  | human (popdens, dist_to_roads, artificial) | 2 | **+0.030** |
  | fuel_cover (CLC forest/scrub) | 2 | +0.011 |
  | vegetation (NDVI/EVI/LAI/FAPAR/LST) | 5 | +0.003 |
  | terrain (elevation, slope) | 2 | −0.003 |
  | fire_context (dist_to_fire) | 1 | −0.005 |
  | soil_moisture | 2 | −0.005 |
  | weather (t2m/RH/pressure/wind/precip) | 13 | −0.011 |
- **Verdict:** **human features dominate** on this slice (+0.030) — confirms human-ignition is the key driver
  and validates the GHS-POP/GHS-BUILT investment (P3). **Vegetation modest (+0.003)** — *expected* in a single
  August month (greenness ~static); MUST re-test across seasons on the backfill before judging. Negative ΔAP
  for weather/terrain/soil/fire_context is **tiny-slice noise** (one month, dominated by where-people-are),
  NOT evidence they're useless — weather/drought/fire-proximity are seasonal signals a 31-day window can't
  show. **No group dropped yet** — decision deferred to the full-backfill re-run.
- **Caveats:** the headline caveat applies doubly here — a one-month window structurally can't reveal
  seasonal (weather/veg/drought) signal. This entry establishes the *method*; the backfill gives the verdict.
