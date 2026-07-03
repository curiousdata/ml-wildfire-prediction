# ML Wildfire Prediction — Spain

Next-day wildfire-occurrence modelling for mainland Spain and the Balearic Islands, built on the
**IberFire** datacube. The system predicts, for each cell, the probability of fire *tomorrow* from
conditions *today*, calibrates that probability into an operational risk product, and serves it through a
live monitor.

**Live demo:** [Fire Guard control center](https://curiousdata-fireguard.hf.space) ·
**Model:** point-wise gradient-boosted trees · **License:** Apache-2.0

---

## Overview

The task is framed as per-cell next-day classification: features at day *t* → fire at *t+1*. The dataset is
the published **IberFire** cube (1 km × 1 km × 1 day, EPSG:3035, Dec 2007–Dec 2024, 120 features across eight
categories), coarsened to a 4 km working grid where the rare `is_fire` label is **max-pooled** to preserve
positives.

The production model is a **point-wise gradient-boosted tree** (scikit-learn `HistGradientBoostingClassifier`).
This is the result of a deliberate pivot: an earlier semantic-segmentation U-Net is retained as the documented
prior approach, but on an identical held-out evaluation the point-wise GBT predicts **new ignitions ≈ 3× better**
(test new-ignition average precision ≈ 0.63 vs 0.19–0.22) and generalizes with no validation→test gap. Spatial
and temporal structure remain essential — but they are captured by **feature engineering** (e.g. distance to
active fire, time since last fire, antecedent-precipitation sums), not by a learned spatial architecture, which
proved redundant on top of those features. Probabilities are isotonic-calibrated, and metrics are reported
separately for *new ignitions* vs *fire spread*, since a blended score otherwise hides the operationally
valuable case.

## Data & pipeline

A medallion pipeline takes raw IberFire NetCDF → **silver** (1 km Zarr) → **gold** (coarsened 4 km Zarr;
features mean-pooled, label max-pooled). Engineered fire-weather and fire-context features (VPD, HDW, FFWI,
KBDI, distance-to-fire, burn history, calendar terms) are derived during/after coarsening. The feature set is
defined once in `src/data/features.py` — the single source of truth shared by training and serving.

The live monitor fetches same-day inputs from operational feeds (Open-Meteo / ERA5 weather, NASA FIRMS active
fire), runs the calibrated model, and renders tomorrow's risk over fresh satellite imagery, with the dominant
risk drivers attributed per day.

## Repository layout

```
src/data/            feature set (features.py), datasets, feature engineering, ingest/ (FGDC)
src/models/          model definitions (legacy U-Net factory)
scripts/             build_features.py · train_gbt.py · calibrate.py · serve.py · serve_engine.py
                     push_predictions.py · weekly_update.py · update_edge.py · rechunk.py · experiments/
docker/monolith/     Streamlit monitor (app_live.py) + Dockerfile
space/               deployable Hugging Face Space (read-only renderer)
models/              gbt_coarse4.* (production GBT + calibrator + metadata)
reports/             evaluation artifacts (measurement floor, tuning, live backtest)
```

## Reproduce & develop

Requires Python 3.11+. The gold cube and model artifacts ship via Git LFS.

```bash
git clone https://github.com/curiousdata/ml-wildfire-prediction.git
cd ml-wildfire-prediction && git lfs pull
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

- **Run the monitor locally:** `docker-compose up --build` → http://localhost:8501
- **Train the production model:** `python scripts/train_gbt.py` (trains on the gold cube, writes
  `models/gbt_coarse4.joblib` + metadata and reports val/test regime metrics).
- **Calibrate:** `python scripts/calibrate_gbt.py` · **tune:** `python scripts/tune_gbt.py`.
- **Benchmark vs baselines (FWI, logistic regression) and the U-Net:** `scripts/measurement_floor.py`,
  `scripts/gbt_compare.py`.

Experiment history is tracked with MLflow (`mlruns/`, local).

## Roadmap

The current cube is the published IberFire download, whose sources differ from the live feeds we serve from —
a train/serve gap that currently requires fallbacks at inference. The active line of work is the **Fire Guard
Datacube (FGDC)**: a from-scratch recollection of the cube from operational, append-daily providers (VIIRS
active fire, ERA5 via Copernicus/Open-Meteo, MODIS vegetation) so that training and serving use the *same*
sources by construction. See `CHANGES.md` for the development log.

## Citation

If you use this repository, please cite it via [`CITATION.cff`](CITATION.cff), and **cite the IberFire
dataset** it is built on:

> Erzibengoa, J., Gómez-Omella, M., & Goienetxea, I. (2025). *IberFire — a detailed creation of a
> spatio-temporal dataset for wildfire risk assessment in Spain.* arXiv:2505.00837.
> Dataset: https://zenodo.org/records/15225886

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Per the license, retain the attribution in [`NOTICE`](NOTICE),
which records the required citation to IberFire and the underlying open-data sources (ERA5/ERA5-Land, NASA
FIRMS, MODIS, EFFIS, CORINE, GHS-POP).

## Acknowledgments

Built on the **IberFire** dataset (Erzibengoa et al., 2025). Weather from ERA5/ERA5-Land
(Muñoz-Sabater et al., 2021) via Open-Meteo and the Copernicus Climate Data Store; active fire from NASA FIRMS
VIIRS (Schroeder et al., 2014); vegetation/LST from MODIS via the Microsoft Planetary Computer; burned area
from Copernicus EFFIS. Built with scikit-learn, xarray/Zarr, Streamlit, and MLflow.
