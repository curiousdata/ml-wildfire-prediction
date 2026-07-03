# Copilot Instructions for Catalonia Wildfire Prediction

## Project Overview

This is a machine learning project for predicting wildfire risks in Catalonia using deep learning models trained on the **IberFire dataset**. The project combines:
- Deep learning models (U-Net with various encoders) for wildfire risk prediction
- Data processing pipelines (NetCDF to Zarr conversion)
- A web-based Streamlit application (model inference + map visualization) — the "monolith" at `docker/monolith/`
- MLflow integration for experiment tracking

## Tech Stack

### Core Technologies
- **Python 3.13+**: Primary programming language
- **PyTorch**: Deep learning framework for model training and inference
- **Streamlit**: Web application for model inference + map visualization (the monolith)
- **Docker & Docker Compose**: Containerization and orchestration

### Key Libraries
- **segmentation_models_pytorch**: Pre-trained segmentation model architectures
- **xarray & Zarr**: Multi-dimensional array processing and efficient data storage
- **MLflow**: Experiment tracking and model management
- **NumPy, Pandas**: Data manipulation
- **scikit-learn**: Metrics and utilities

### Data Formats
- **NetCDF**: Raw input data format
- **Zarr**: Optimized storage format for efficient data access
- **PyTorch `.pth`**: Model weights format (the monolith loads `.pth` directly via the `build_unet` factory)

## Repository Structure

```
ml-wildfire-prediction/
├── .github/                     # GitHub configuration and workflows
├── data/                        # Datasets (NetCDF, Zarr formats)
├── docker/monolith/             # Streamlit web app (model inference + map viz)
│   └── app.py
├── notebooks/                   # Jupyter notebooks for EDA and experiments
├── scripts/                     # Training and data processing scripts
│   ├── train.py                 # U-Net model training
│   ├── conversion.py            # NetCDF → Zarr conversion
│   ├── coarsen.py              # Spatial resolution adjustment
│   └── rechunk.py              # Data chunking optimization
├── src/                         # Core modules
│   ├── data/                    # Dataset classes + canonical FEATURE_VARS
│   │   ├── datasets.py
│   │   └── features.py
│   └── models/                  # Model architectures (build_unet factory)
│       └── cnn.py
└── docker-compose.yml           # Monolith application configuration
```

## Development Guidelines

### Code Style
- Follow PEP 8 conventions for Python code
- Use type hints where appropriate (project uses `from __future__ import annotations`)
- Keep imports organized (standard library, third-party, local)
- Use descriptive variable names that reflect the domain (e.g., `zarr_path`, `model_name`, `fire_days`)

### Project-Specific Conventions
- **Path handling**: Use `pathlib.Path` for file paths, not string concatenation
- **Project root**: Scripts use `Path(__file__).resolve().parents[1]` to find project root
- **Imports**: Add project root to `sys.path` for absolute imports: `from src.data.datasets import ...`
- **Model naming**: Model files end with `.pth` extension
- **Data paths**: Default Zarr data location is `data/gold/IberFire_coarse32.zarr`

### MLflow Integration
- Use MLflow for experiment tracking in training scripts
- Set experiment name: `mlflow.set_experiment("iberfire_unet_experiments")`
- Log parameters, metrics, and models consistently
- MLflow server runs on `http://localhost:5001` by default

### Deep Learning Specifics
- Models are U-Net architectures from `segmentation_models_pytorch`
- Support for different encoders (e.g., resnet34, efficientnet)
- Handle class imbalance using logit-adjusted loss (based on Long-tail learning research)
- Training uses GPU acceleration (CUDA or MPS)
- Metrics: ROC-AUC and Average Precision

### Docker & Deployment
- Use `docker-compose.yml` from project root for the monolith application
- Single service (`monolith`) — a Streamlit app on port 8501, built from `docker/monolith/Dockerfile`
- It loads the `.pth` model directly and reads the gold Zarr; no separate backend/API

## Building and Running

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow server (optional, for experiment tracking)
mlflow server --host 0.0.0.0 --port 5001
```

### Training
```bash
# Train a model
python scripts/train.py --model_name resnet34_v8 --epochs 50
```

### Data Processing
```bash
# Convert NetCDF to Zarr
python scripts/conversion.py

# Reduce spatial resolution
python scripts/coarsen.py

# Optimize chunking
python scripts/rechunk.py
```

### Running the Application
```bash
# Start the monolith Streamlit app with Docker
docker-compose up --build

# Access the app at http://localhost:8501
```

## Testing
- The project does not currently have a formal test suite
- Validation is done through MLflow metrics during training
- Manual testing of the web application is performed via Docker

## Dataset Information

### IberFire Dataset
- **Provider**: Julen Ercibengoa Calvo
- **Coverage**: Spain with 1km spatial resolution
- **Temporal Resolution**: Daily
- **Format**: NetCDF (raw), Zarr (processed)
- **Variables**: Environmental and meteorological data including:
  - Temperature
  - Precipitation
  - Wind speed/direction
  - Vegetation indices
  - Topography

## Common Tasks

### Adding a New Model
1. Define model architecture in `src/models/` (or use segmentation_models_pytorch)
2. Update training script in `scripts/train.py` to support new model
3. Log model parameters and metrics to MLflow
4. Save model weights as `.pth` file

### Adding New Data Processing Scripts
1. Place script in `scripts/` directory
2. Use `pathlib.Path` for path handling
3. Add project root to `sys.path` if importing from `src/`
4. Document input/output data formats

### Modifying the Web Application
- **App**: Edit `docker/monolith/app.py` (Streamlit UI + inference + map rendering)
- **Rebuild**: Run `docker-compose up --build` to test changes

## Important Notes

- GPU is recommended for training (CUDA or MPS)
- Data files are large and stored locally in `data/` directory
- The project uses coarsened data (32x spatial resolution) for faster experimentation
- Model inference in the web app loads the `.pth` weights directly (via `src/models/cnn.py::build_unet`)
- MLflow tracks all experiments and model versions

## Research Context

This project implements techniques from academic research:
- **Long-tail learning via logit adjustment**: Used for addressing class imbalance in wildfire prediction
  - Paper: Menon et al. (2020), arXiv:2007.07314
  - Applied through custom loss functions during training
