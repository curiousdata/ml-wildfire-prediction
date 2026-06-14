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

> ⚠️ Current entries run on the **Aug-2016 dev slice** (31 days) — numbers are **directional**, meant to
> establish method + sign of effect. Each will be **re-run on the full 2012→present backfill** for final
> magnitudes before any production decision; this file is updated in place when that happens.

---

## Planned / in-progress
- **Temporal interpolation of slow layers** (GHS-POP / built-up / CLC proportions: step-snap vs
  linear-interp between editions) — *to run when P3 lands; user-prioritized.* Hypothesis: removing Jan-1
  step discontinuities removes a calendar artifact the model can key on and better reflects gradual change.
- **Re-run all entries below on the full 2012→present backfill** for final magnitudes (slice numbers are
  directional only).

---

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
