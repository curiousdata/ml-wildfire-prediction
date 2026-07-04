# FireGuard — next-day wildfire risk for Spain

**FireGuard** is an end-to-end ecosystem for predicting *tomorrow's* wildfire risk across mainland Spain and
the Balearic Islands: it recollects its own analysis-ready datacube from operational satellite and reanalysis
feeds, engineers the physical drivers of fire, learns a calibrated per-cell next-day risk model, and serves it
through a live control center — all from the *same* data sources at train and serve time.

**Live control center:** [curiousdata-fireguard.hf.space](https://curiousdata-fireguard.hf.space) ·
**Model:** calibrated point-wise gradient-boosted trees · **Grid:** 4 km × daily × Spain, 2012→ ·
**License:** Apache-2.0

---

## The ecosystem

FireGuard is several coordinated pieces, not one script:

| Component | What it is |
|---|---|
| **FGDC — Fire Guard Datacube** | The analysis-ready data product: a medallion cube (bronze → silver → gold) **recollected from operational, append-daily providers** (VIIRS active fire, ERA5/ERA5-Land weather, MODIS vegetation, human-settlement & terrain layers), coarsened to a 4 km working grid with **135 engineered features** and a max-pooled `is_fire` label. |
| **FG Pipeline** | The data-engineering engine that builds and maintains FGDC — the medallion transforms plus a **Lambda architecture along a data-*vintage* axis** (forecast → ERA5T → final): a weekly *speed* refresh that appends newly-settled days, an incremental edge engine that recomputes only the new days' features (bit-identical to a whole-cube pass), and an ephemeral *serve* path. |
| **FG Forecaster** | The model: a **point-wise gradient-boosted tree** (`HistGradientBoostingClassifier`) that maps a cell's features at day *t* to a calibrated probability of fire at *t+1*, with metrics reported separately for **new ignitions** vs **fire spread**. |
| **FG Control Center** | The live operational monitor — a map-first Hugging Face Space that renders tomorrow's risk as a heat glow on a dark schematic base, with **hover-able danger areas** (aggregate probability + plain-language causes), today's active fire, and per-day drivers. |

Two supporting layers sit inside the pipeline: **FG Live Edge** (fetch today's feeds → build the feature vector in
memory → predict t+1 → publish predictions to a Hugging Face Dataset the Control Center reads) and **FG Lab** (an
ablation-driven evidence base — every feature/source/target change is tested with/without and logged to
`ABLATIONS.md`).

## How it works

The task is per-cell next-day classification: **features at day *t* → fire at *t+1***. The `is_fire` label is
**max-pooled** on coarsening to preserve the rare positives.

The production model is a **point-wise gradient-boosted tree**. This is the result of a deliberate pivot: an
earlier semantic-segmentation U-Net is retained as the documented prior approach, but on an identical held-out
evaluation the point-wise GBT predicts **new ignitions ≈ 3× better** (test new-ignition average precision ≈ 0.63
vs 0.19–0.22) and generalizes with no validation→test gap. Spatial and temporal structure remain essential — but
they are captured by **feature engineering** (distance to active fire, time since last fire, antecedent-precip
sums, drought memory), not by a learned spatial architecture, which proved redundant on top of those features.
Probabilities are isotonic-calibrated to true prevalence.

## Data & pipeline

A medallion pipeline flows **ingest (→ bronze) → `build_silver` (1 km Zarr) → `coarsen` (4 km gold) →
`build_features` (engineered)**, then `train_gbt` / `calibrate`. Engineered fire-weather, drought, fuel, terrain,
human-exposure and calendar features (VPD, EMC, FFWI, HDW, KBDI, SPI, distance-to-fire, burn history, holidays…)
are derived on the gold cube in one ordered pass. The feature set is defined once in **`src/data/features.py`** —
the single source of truth shared by training and serving.

Because FGDC is recollected from the **same operational feeds it serves from**, there is no train/serve source
mismatch by construction. The Live Edge fetches same-day inputs (Open-Meteo/ERA5 weather, NASA FIRMS active fire),
runs the calibrated model, and publishes tomorrow's risk for the Control Center to render.

## Repository layout

```
src/data/            features.py (feature set), feature_engineering.py, fetch.py (Open-Meteo+FIRMS),
                     regions.py, metrics.py, ingest/ (grid, build_silver, coarsen, ingest_{weather,fire,veg,static})
scripts/             build_features.py · train_gbt.py · calibrate.py · serve.py · serve_engine.py
                     push_predictions.py · weekly_update.py · update_edge.py · rechunk.py · experiments/
models/              gbt_fireguard.{joblib,meta.json} + calibrator + calibration report
space/               FG Control Center — deployable Hugging Face Space (reads the predictions Dataset)
docker/monolith/     local Streamlit build of the monitor + Dockerfile
archive/             retired v1 (IberFire) code + models, kept for lineage
```

## Reproduce & develop

Requires Python 3.11+.

```bash
git clone https://github.com/curiousdata/ml-wildfire-prediction.git
cd ml-wildfire-prediction
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

- **Train the model:** `python scripts/train_gbt.py` → `models/gbt_fireguard.joblib` (+ metadata, val regime metrics).
- **Calibrate:** `python scripts/calibrate.py` (isotonic → `gbt_fireguard.calibrator.joblib`).
- **Build engineered features on a gold cube:** `python scripts/build_features.py --cube FireGuard --factor 4`.
- **Weekly settled-data refresh:** `python scripts/weekly_update.py` (ingest → silver → gold → features).
- **Serve tomorrow's risk & publish:** `python scripts/serve.py --mode live && python scripts/push_predictions.py`.
- **Run the monitor locally:** `docker-compose up --build` → http://localhost:8501.
- **Baselines / ablations:** `scripts/experiments/` (`baseline_panel.py`, `fgdc_ablation.py`).

Experiment history is tracked with MLflow (`mlruns/`, local). The forward plan lives in `ROADMAP.md`; the
development log in `CHANGES.md`.

## Roadmap

FGDC v2 is the shipped, serving system. Active/queued directions: adopt **dual-satellite (S-NPP + NOAA-20) fire**
(measured +57% label density → a consistent next-day-skill lift), forecast-weather complement features, and a
real-time nowcast path. See `ROADMAP.md` and `CHANGES.md`.

## References

### Scientific literature

**Datasets & remote sensing**
- Erzibengoa, J., Gómez-Omella, M., & Goienetxea, I. (2025). *IberFire — a spatio-temporal dataset for wildfire
  risk assessment in Spain.* arXiv:2505.00837. *(The v1 cube FireGuard succeeds and re-derives.)*
- Karlsson, L. et al. (2025). *VIIRS vs MODIS active fire for next-day prediction.* arXiv:2503.08580.
  *(Motivates VIIRS 375 m as the label source.)*
- Muñoz-Sabater, J. et al. (2021). *ERA5-Land: a state-of-the-art global reanalysis dataset for land
  applications.* Earth System Science Data 13, 4349–4383.
- Schroeder, W. et al. (2014). *The New VIIRS 375 m active fire detection data product.* Remote Sensing of
  Environment 143, 85–96.

**Fire-danger & environmental indices** (implemented in `src/data/feature_engineering.py`)
- Keetch, J. J., & Byram, G. M. (1968). *A drought index for forest fire control.* USDA Forest Service. — KBDI
- Fosberg, M. A. (1978). *Weather in wildland fire management: the Fire Weather Index.* — FFWI
- Srock, A. F. et al. (2018). *The Hot-Dry-Windy Index.* Atmosphere 9(7):279. — HDW
- McKee, T. B., Doesken, N. J., & Kleist, J. (1993). *The relationship of drought frequency and duration to time
  scales.* — SPI (used as the antecedent-precip anomaly)
- McCune, B., & Keon, D. (2002). *Equations for potential annual direct incident radiation and heat load.*
  Journal of Vegetation Science 13, 603–606. — Heat Load Index
- Carlson, T. N., & Ripley, D. A. (1997). *On the relation between NDVI, fractional vegetation cover, and leaf
  area index.* Remote Sensing of Environment 62, 241–252. — FVC
- Weiss, A. (2001). *Topographic Position and landforms analysis.* — TPI

**Machine learning**
- Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics.
- Ke, G. et al. (2017). *LightGBM: a highly efficient gradient boosting decision tree.* NeurIPS. *(Histogram GBT,
  the basis of scikit-learn's `HistGradientBoostingClassifier`.)*
- Zadrozny, B., & Elkan, C. (2002). *Transforming classifier scores into accurate multiclass probability
  estimates.* KDD. *(Isotonic calibration.)*
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: convolutional networks for biomedical image
  segmentation.* MICCAI. *(The shelved prior approach.)*

### Data sources

- **ERA5 / ERA5-Land** reanalysis — Copernicus Climate Data Store (ECMWF), accessed via **Open-Meteo** (archive +
  forecast APIs) and directly for the batch master.
- **NASA FIRMS** active fire — VIIRS S-NPP (VNP14IMG) and NOAA-20 (VJ114).
- **MODIS** vegetation & land-surface temperature (NDVI/EVI, LAI/FAPAR, LST) via the **Microsoft Planetary
  Computer**.
- **Copernicus EFFIS** burned-area perimeters (auxiliary/offline).
- **NOAA GEFS** reforecast — forecast-weather features (experimental).
- **JRC Global Human Settlement Layer** — GHS-POP (population), GHS-BUILT-S (built-up).
- **CORINE Land Cover** (Copernicus) · **Copernicus DEM** (elevation) · **OpenStreetMap** via Geofabrik
  (roads/railways/waterways) · **Natura 2000** protected areas.
- **CartoDB (dark_matter) / OpenStreetMap** — dark schematic basemap tiles for the Control Center.

### Software & frameworks

- **Data:** NumPy, pandas, xarray, Zarr, numcodecs, Dask.
- **ML & tracking:** scikit-learn (`HistGradientBoostingClassifier`, `IsotonicRegression`), MLflow.
- **Geospatial:** pyproj, rasterio, SciPy (interpolation, distance transforms, spatial).
- **Data access:** requests, cdsapi, earthaccess, planetary-computer, `huggingface_hub`; python-holidays.
- **App & deploy:** Streamlit, folium, Pillow; Docker / docker-compose; Hugging Face Spaces + Datasets.
- **Legacy (shelved U-Net):** PyTorch, segmentation-models-pytorch.

## Citation

If you use this repository, cite it via [`CITATION.cff`](CITATION.cff), and **cite the IberFire dataset** it
builds on:

> Erzibengoa, J., Gómez-Omella, M., & Goienetxea, I. (2025). *IberFire — a detailed creation of a spatio-temporal
> dataset for wildfire risk assessment in Spain.* arXiv:2505.00837. Dataset: https://zenodo.org/records/15225886

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Retain the attribution in [`NOTICE`](NOTICE), which records the
required citations to IberFire and the underlying open-data sources.
