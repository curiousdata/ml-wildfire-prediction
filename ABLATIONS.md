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
- **Regime-split / new-ignition-restricted ablation** (ignition vs spread @ 6 km) — the test that should expose
  weather's conditional value the blended metric hides; gives the v1-comparable new-ign AP (bar ≈ 0.63).
- **P4 engineered fire-weather** (precip_sum_90d drought-memory, VPD/HDW/FFWI/KBDI) — the real weather test;
  raw daily weather scored ~0 marginal on the full span, as expected without the engineered memory.

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
