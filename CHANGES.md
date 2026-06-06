# CHANGES

A running devlog of substantive changes — *what* changed and *why it mattered*, newest first.
Written to double as a narrative for presenting the project: each entry is a beat in the story,
not just a diff. (The forward-looking plan and open decisions live in `ROADMAP.md`; per-agent
working notes in `CLAUDE.md`.)

**Rule:** every bug found or fixed gets logged under a "Bugs found & fixed" subsection of the
current dated entry — what was wrong, its impact, and the fix (or that it's flagged, not yet fixed).

---

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
