# BACKLOG.md

Forward-work register for FireGuard (FGDC v2). Consolidates ~2 months of ideas that were parked while the
2 km production cutover shipped. **This supersedes ROADMAP.md's forward plan** (ROADMAP is now historical — see
the note at the bottom).

**How this relates to the other docs:** `CLAUDE.md` = how the repo works now · `CHANGES.md` = dated narrative
devlog (what happened) · `ABLATIONS.md` = with/without experiment results · this file = what's *next* and why it's
parked. Most items have a fuller write-up in a `~/.claude/.../memory/*.md` note (linked as `[name]`).

**Status legend:** 🟢 shipped (kept for context) · 🔵 proven-but-deferred (measured positive, not adopted) ·
⚪ idea / not started · 🔒 gated on a prereq · 🎯 has a concrete gating experiment defined.

Golden rules that constrain everything below: **train = serve by construction** (source-vintage changes are
all-or-nothing retrains — the 0.10-fire-source-bug class); `FGDC_FEATURE_VARS` is **append-only + load-bearing
order** (any feature change = deliberate retrain); **gate every feature/source/target change on an ABLATIONS
entry**; discussion-first (design before code).

---

## A. Data pipeline & resolution — the current strategic line

- **⚪🔒 Out-of-core data pipeline** `[data-pipeline-streaming]` — **own branch, current.** Formalize streaming
  before going finer than 2 km (1 km = 16× cells, 500 m = 64× — the ad-hoc time-subset / drop-rebuild / block-scale
  hacks won't hold). Two halves: **READ-streaming** (resurrect the archived U-Net `_dyn` pre-stack idea, GBT-flavored:
  land-cells-only, raw-not-normalized, fp16-free-for-HistGBT, + a shared `day_matrix(t)→[cell,feat]` reader to kill
  the 4 duplicated block-loads in train/calibrate/serve/ablation) + **COMPUTE-streaming** (tiling + halos / dask
  `map_overlap` / `update_edge` as the canonical incremental builder; also **fix `build_features --overwrite`** — the
  xarray "var exists, encoding provided" bug that forces the drop-rebuild dance). Prereq for 1 km.
- **⚪ CLC temporal-interpolation revival** — **designed this session, standing by for go.** Closes the logged CLC
  bug (land cover is frozen at the 2018 edition for all dates 2012→2026; popdens/built_s are already time-interpolated).
  DECIDED approach (user): **read-time resolution via a single shared resolver** both `train_gbt` and `serve` import
  (no cube materialization → no train/serve drift); **linear-interp the 19 CLC proportions** between editions by date,
  **nearest-edition the 44 one-hots**. Deliberate retrain re-freezes feature order. Gate on an ablation (expect small
  — served era ≈ 2018 — value is correctness/consistency). `[feature-enrichment-branch]` (bug note).
- **🟢→⚪ Finer resolution: 2 km shipped → 1 km → 500 m** `[finer-resolution-branch]` — 4 km→**2 km is LIVE**;
  next milestones 1 km then a **500 m hard-stop** (just above the 375 m VIIRS label). Value = **feature fidelity**
  (DEM 30 m, dist_to_roads/fire, fuel/veg, WUI), **not** label density (cells grow by area, far faster than sensors
  add signal → label gets sparser). Measured 4→2 km: **ignition AP tie but +40% relative localization, spread AP
  +0.006, tighter alerts** (a tie AP at 4× smaller cells is an *operational* win — ¼ the search box). Judge finer
  steps on **per-regime skill + localization + calibration**, not AP (AP mechanically drops at finer res). 1 km needs
  the streaming branch + SSD. Reopens CNN-for-spread. `[unet-resolution-hypothesis]`, `[two-model-fire-architecture]`.
- **⚪ update_edge canonical-math consolidation (#2/#3)** `[update-edge-review]` — `update_edge.py` hand-copies the
  causal-anomaly math (`feature_engineering.seasonal_anomaly(causal=True)`) and the precip cumsum-window block. Real
  refactors but bit-identity-risky → require the full `--test`/`--e2e` re-verify loop. Deferred; folds into the
  streaming branch. (#1 non-atomic edge write + #4/#5 already FIXED.)

## B. Model & features

- **🔵 Forecast weather (GEFS d+1) production fill** — **proven +0.0054 new-ign/prec@K** (≈⅔ of the perfect-foresight
  ceiling; weather drives it, CAPE redundant) on the 2016–2019 slice, bronze kept. **Deferred:** a live feature needs
  the full 2012–2026 backfill incl. the un-validated operational `noaa-gefs-pds` bucket → materialize → retrain → live
  GRIB serving. Not worth +0.0054 right now. Next step if resumed: validate that bucket. (CLAUDE.md, ABLATIONS 2026-06-26.)
- **⚪ Lightning / CAPE — SIZED, aggregate-capped** `[feature-enrichment-branch]` — the one ignition mechanism the
  model entirely lacks (natural ignition). But **measured**: forecast CAPE = +0.0031 standalone, **+0.0009 incremental
  over existing weather** (largely redundant). Capped by construction: Spain fires are ~85–96% human-caused →
  lightning touches only the small natural slice → aggregate new-ign AP is the wrong yardstick. Do NOT build as a
  global-AP bet. If pursued: **CAPE ≠ dry lightning** (a real strike feed / MTG Lightning Imager is more direct, +
  an ESA-narrative asset), and the honest test is a **regime-subset eval** (interior/NE + Jun–Sep) on the CAPE bronze
  already on disk — no new fetch.
- **⚪ Human-distance features: trails / power lines / campsites** `[feature-enrichment-branch]` — `dist_to_trails`
  (OSM `highway=path/track`) is the plausible one — a genuinely missing mechanism (recreation in wildland). `dist_to_
  powerlines` + campsites likely land in the **saturated** human-distance family (dist_to_city/pop_access already
  ablated ≈0). Distance-form, near-static. Cheap to test, gate each on ablation; low prior except trails. Meta-finding:
  **cube-derived engineered features have plateaued for new-ignition** — the levers are new orthogonal DATA, more data,
  or a different target, not more re-encodings.
- **🔵 4-/6-pass VIIRS label — train-side retrain** `[noaa20-signal-density]`, `[fire-source-multisensor]` — NOAA-20
  (4-pass) + NOAA-21 (6-pass) are **shipped at SERVE** (proven +0.013 new-ign via denser `dist_to_fire`, zero retrain).
  Train-side retrain on the stitched multi-sat label **ties on ranking AP** (+0.0135 prec@K only — density benefit is
  delivered at serve, feature→risk mapping is density-invariant) → **low priority**, only if top-K alert precision
  becomes the product goal. Pick = train all-years on the auto-stitched label (crop-2024 is data-starved, decisively
  worse). "Source ≫ years."
- **⚪🎯 Sentinel-3 SLSTR fire source** `[fire-source-multisensor]` — the real research item: 2 more platforms
  (EUMETSAT), fills the ~10:00/22:00 slots VIIRS's orbit misses. Cost: 1 km + different algorithm → cross-sensor
  heterogeneity (label *meaning* drifts). Needs the same density-AND-skill ablation harness as NOAA-20, per source.
- **⚪ ROADMAP P4 fuel/fire-weather shortlist (still open):** FRP-weighted fire_context (FIRMS returns FRP per
  detection — weight dist_to_fire/recency by intensity; free, strengthens the #1 group); **dead fuel moisture (1/10/
  100-hr, NFDRS from our ERA5)** — the input the 2025 Europe ignition literature flags that we lack; **live fuel
  moisture (LFMC)** (SMAP/VIIRS-ML products, new source); **longer fire history** (MODIS MCD64A1 burned-area 2000+ to
  extend burn-frequency before the VIIRS-2012 start). Each earns its place via an ABLATIONS entry. (Engineered
  fire-weather — precip_sum_90d/VPD/HDW/KBDI/FFWI/SPI — is now DONE in `build_features`.)
- **⚪ Optuna HPO** `[optuna-friend-machine]` — on the FINAL frozen feature set (tuning before freeze = re-tuning).
  Marginal on v1 (+0.006); do once, last among modelling steps. Sized for the friend's M3 Max (36 GB/14 CPU) weekend.
- **⚪ Refit-on-all + drift/monitoring loop** — after tuning + the held-out A/B is recorded, refit the frozen config on
  ALL data (matters for an operational model); **always report the held-out number, never the all-data model's.**
  Ongoing eval shifts to a **rolling backtest** (logged served preds vs finalized truth) + a champion/challenger
  gated-promotion + PSI/KS drift triggers. (ROADMAP items 3, 5.)

## C. Architecture & research bets (all gated on the 1 km cube)

- **⚪🎯 Two-model architecture: ignition forecaster + spread nowcaster** `[two-model-fire-architecture]` — escalates
  the regime split. Only pays off if the spread model is **structurally different** (finer res + directional +
  sub-daily, maybe CNN) — two GBTs on a dist_to_fire partition would underperform the joint model. **Gating experiment:**
  re-measure downwind/upwind next-day fire asymmetry at 1 km (it's grid-suppressed: 4 km=1.00×, 2 km=1.06× rising to
  1.28× at 8–10 km, monotonic with resolution). If it climbs >~1.5× in resolved bands → spread-specialist justified;
  if flat → keep joint. One measurement decides it.
- **⚪🎯 CNN-for-spread at 1 km** `[unet-resolution-hypothesis]` — does the shelved U-Net's advantage revive at finer
  res? Nuanced: **likely yes for spread** (coherent front = spatial structure a tree can't use) + continuous targets
  (FRP/burned-fraction), **likely no for new-ignition** (point event, sparser target worsens the spiky-corruption that
  sank the U-Net). Test: CNN vs GBT × {4,2,1 km} × {ignition,spread} + a continuous-target variant. U-Net code
  (`build_unet`, `archive/scripts/train.py`) still exists, no weights.
- **⚪🎯 Geostationary nowcast fusion (MTG-FCI as a raw thermal FEATURE)** `[geostationary-nowcast-fusion]` — MTG
  ~10–15 min 3.8 µm anomaly as **direct combustion evidence**, fused (model-prior × live-thermal → each suppresses the
  other's false alarms) into a real-time nowcast head. NOT a label (measured: 60–80% of geo-unique detections are
  intraday spread, not ignitions) and NOT a next-day feature. Value = temporal + thermal-direct, not spatial (2 km →
  upsampled). = the spread half of the two-model architecture. **Gate:** a cheap retrospective lead-time study on the
  MTG archive (does the anomaly appear before VIIRS confirmation, at what false-alarm cost). Product pivot to real-time.
- **⚪🔒 ERA5 → ERA6 weather migration** `[era6-weather-migration]` — ERA6 ~14 km (vs ERA5 ~31 km), production started
  2026-03, first 20-yr batch ~end-2027. Source-vintage migration → all-or-nothing retrain. Modest standalone (raw
  weather ≈0 marginal) but **couple it with the finer-res rebuild**: 14 km halves the weather-imposed resolution
  ceiling, and its better precip helps the load-bearing drought features (kbdi/spi_90d/precip_sum). Gate: Open-Meteo/EDH
  exposing ERA6 + confirm 2012+ coverage. Timelines align with the streaming/finer-res maturity.

## D. Product, serving & deployment

- **⚪ FireGuard API + separate models + separate risk layers** `[fireguard-product-layers]` — one architecture: the
  **API** (thin read layer over the existing HF Dataset — grid / region / per-cell drivers / `latest`) is the spine;
  **separate models** (ignition/spread, §C) feed **selectable layers** (ignition / spread / multi-horizon t+1,3d,7d /
  smoke). Turns the demo into an **integrable service** = directly the ESA "service viability" criterion `[esa-wildfire-
  funding]`.
- **⚪ Binary "commit map" display mode** `[fireguard-product-layers]` — "where fire appears if the model must guess
  now." Real as a **top-K / target-recall** layer, NOT a 0.50 threshold (served probs top out ~0.024; a 0.50 cutoff
  predicts nothing, ever). Empirical operating points banked: `>base-rate` flags ~half of Spain @ 0.96 recall / 0.06%
  precision; top-K = 29%/29%. No threshold is both high-recall and high-precision (intrinsic AP). Frame operationally,
  not "fire will be here."
- **⚪ Control-Center map refinements** `[control-center-interactivity]` — 🟡 small serve-side: **spread-direction
  arrows** (needs serve to publish wind_u/v) + **smoke/air-quality gray layer** `[air-quality-smoke-layer]` (keyless
  Open-Meteo AQ — display/nowcast layer, NOT a t+1 feature). Cluster refinements: **per-cluster local attribution**
  (serve-side occlusion, replaces day-level regime causes) + waterfall clustering (only if the scipy watershed's
  over-segmentation bites). Re-anchor ramp/tiers to prevalence on any resolution change.
- **⚪ Air-quality / smoke layer** `[air-quality-smoke-layer]` — keyless Open-Meteo AQ (CAMS PM2.5/PM10/AOD/dust) →
  gray RGBA overlay via the existing `risk_rgba`/`fire_rgba` compositing. Situational awareness; smoke is a
  *consequence* of fire → display layer, not a feature.
- **⚪ Deploy: Ship B (cloud) + Raspberry-Pi node** — **Ship B** `[hf-deploy-plan]` = always-on cloud serve on HF Jobs
  + a serialized seed **bundle** (365-d raw tail + carry-forward kbdi/counters + per-doy climatology + static + model),
  so the cloud job needn't the 76 GB cube — backlog until traction. **Pi node** `[raspberry-pi-server]` = 1–2 Pi 5 +
  SSD (~$190/node) as a self-contained always-on FGDC node (incremental batch + serve + push; torch is off the live
  path → ARM-fine; avoids the whole-cube RAM freeze). Middle path between Ship A (laptop-dependent) and Ship B (cloud
  cost). One-time full baseline on the laptop, copy silver over.
- **⚪ Batch layer (final-watermark overwrite)** `[fgdc-extend-cadence]`, `[lambda-architecture-fgdc]` — the monthly
  best-grade (EDH ERA5 / finalized VIIRS) master job that overwrites behind a `final_watermark` seam. The speed layer
  (weekly ERA5T append + ephemeral serve) is built; the batch overwrite path is **not built yet**. (ROADMAP item 6.)
- **⚪ Portugal → full Iberia** `[portugal-iberia-expansion]` — model is Spain-only; FIRMS already covers Iberia so
  fire shows over Portugal with no prediction (interim: Control Center masks fire to Spain). Real fix = extend
  grid/masks (need a Portugal region layer — no `AutonomousCommunities` equivalent) + re-ingest static/veg over the
  wider extent + retrain. Fire/weather sources are already Iberia-wide.

## E. Real-time latency (only for a nowcast product pivot)

- **⚪ Cut fire latency below FIRMS ~3 h** `[realtime-viirs-directbroadcast]` — routes: EUMETSAT **EARS-VIIRS** (~1–1.5 h,
  no hardware, but ships L1/SDR → you run detection = source-gap risk) → own **Direct Broadcast + CSPP** (~1–2 min,
  ~$100–300k X-band dish; FIRMS-algorithm-identical so train=serve holds) → **geostationary MTG-FCI** (10 min, §C).
  **Key scoping:** the 3 h lag is NOT binding for next-day t+1 (the day's label is a UTC union complete by evening;
  progressive refinement handles it). Pays off ONLY for a real-time nowcast/active-spread product. Barcelona has no
  X-band EOS asset; realistic = EARS or Ground-Station-as-a-Service.

## F. Funding & publication (external, deadline-bound)

- **🎯 ESA Wildfires — Preparedness call** `[esa-wildfire-funding]` — **opens 1 Sept 2026** (webinar 30 Sept).
  FireGuard fits the risk-assessment track; strong on space-data value + product maturity. **Critical-path gaps are
  non-technical:** a named operational customer (civil-protection / *bomberos* / a Comunidad Autónoma) + Spanish
  delegation (CDTI) authorization + a legal entity (SME — raises the co-funding rate to 80%). Response call closed
  2 June (missed) — that was the spread-nowcaster's home.
- **⚪ External benchmark + dataset publication + docs** (ROADMAP items 8–10) — the credibility gate for a paper is an
  **A/B vs the operational fire-danger index (EFFIS / FWI)** on the shared window, not the v1↔FGDC self-comparison.
  Then publish the FGDC dataset; fix scientific references across markdowns (Erzibengoa spelling / Zenodo id; cite
  ERA5-Land = Muñoz-Sabater 2021 ESSD 13:4349, VIIRS 375 m = Schroeder 2014 RSE).

---

## Divergences from ROADMAP.md (why it's now historical)

ROADMAP's newest section is 2026-06-21; the bottom two-thirds is U-Net/v1 history and its "open decisions" D1–D4 are
resolved. What shipped since, **outside** the written plan:
- **2 km production cutover + go-live** — Space RUNNING at 2 km, live 4×/day serve → HF Dataset, calibrator shipped.
  ROADMAP still frames 4 km gold.
- **v1 cube + U-Net artifacts DELETED** (2026-06-26, 131 GB) — ROADMAP item 7 said "gate strictly behind a proven
  cutover"; done, and the U-Net code is now under `archive/`.
- **Calibration shipped** (isotonic, true-prevalence) — ROADMAP item 4 listed it as "currently missing."
- **NOAA-20/21 multi-sat serve, weekly/speed Lambda layers, update_edge incremental engine, band-cache serve** — none
  were in the plan; all live.
- **The resolution/streaming/two-model/nowcast research program** — entirely post-ROADMAP; captured in §A–C above.

**Recommendation on ROADMAP.md:** its forward plan is superseded by this file; its history is preserved in CHANGES.md
+ the `iberfire-v1-reference` memory; its "four species / measurement-harness" framing is done. → **safe to delete**,
or trim to a one-line pointer here. Deferred to the user (not deleting a checked-in doc without a go).
