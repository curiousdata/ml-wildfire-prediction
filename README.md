# ML Wildfire Prediction

A machine learning project for predicting wildfire risks in Spain using deep learning models trained on the **IberFire dataset**, a comprehensive datacube containing environmental, meteorological, and geographical data for Spain.

## Overview

This project uses convolutional neural networks (CNNs), specifically **U-Net architecture with various encoders**, to perform spatial segmentation and predict wildfire risks in Catalonia. The system includes:
- Data processing pipelines to convert NetCDF data to Zarr format
- U-Net based models with different encoders (ResNet34, etc.) for wildfire prediction
- A web-based MVP application for visualizing predictions
- MLflow integration for experiment tracking

## Project Structure

```
ml-wildfire-prediction/
├── data/                        # Datasets (NetCDF, Zarr formats)
├── docker/monolith/             # Streamlit web app (model inference + map visualization)
├── notebooks/                   # Jupyter notebooks for EDA and experiments
├── scripts/                     # Training and data processing scripts
│   ├── train.py                 # U-Net model training
│   ├── conversion.py            # Data format conversion
│   └── coarsen.py               # Spatial resolution adjustment
├── src/                         # Core modules
│   ├── data/                    # Dataset classes
│   └── models/                  # Model architectures
├── docker-compose.yml           # Monolith application configuration
└── requirements.txt             # Python dependencies
```

## Quick Start

This section is divided into three parts based on your use case:
1. **Running the Application** - For all users who just want to try the wildfire prediction app
2. **Training the Model (beta)** - For ML practitioners who want to train the model on the existing dataset
3. **Full Experimentation (beta)** - For advanced devs and ML engineers who want to experiment with data processing, feature engineering, and model architecture

### Compute Requirements
1. For just starting the application, there are none. There is a trained model already in the repo, and the amount of CPU work is minimal.
2. If you chose the training mode - your training will be faster if you have a powerful machine. You might need 10-11 GB of RAM for effective data streaming. Fastest training will be on machines with physical GPU like that of NVIDIA or Apple Silicon.
3. For full experimentation mode with raw NetCDF dataset, it is preferrable that you have a powerful machine with a lot of CPU, GPU and RAM resource. Dataset's original creators recommend using a machine with at least 128 GB RAM. However, if for some reason you don't have the aforementioned machine, don't let that stop you - this project was successfully created using a single 2020 MacBook Air M1. 

### Storage Requirements
**Important:** The gold dataset and latest model are managed via Git LFS. Ensure you have enough storage: the dataset and model are approximately **1 GB in total**.
In case you're considering downloading the original dataset, make sure you have **at least 30 GB** available. 

---

### Part 1: Running the Application

If you just want to start the wildfire prediction application:

**Prerequisites:**
- Docker and Docker Compose
- Git LFS (for downloading the dataset and model)

**Steps:**
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/curiousdata/ml-wildfire-prediction.git
   cd ml-wildfire-prediction
   ```
3. Prepare large files:
   
   - Make sure LFS is installed:
   ```bash
   git lfs install
   ```
   - Pull model and dataset files (should have started downloading automatically, but to be sure):
   ```bash
   git lfs pull
   ```
   - Unpack the gold dataset (because it's sharded, we had to archive it into one file for LFS support):
   ```bash
   tar -xzf data/gold/IberFire_coarse32.zarr.tar.gz -C data/gold
   ```
   You only have to do it once.
   
4. Start the application:
   ```bash
   docker-compose up --build
   ```
5. Access the application at [http://localhost:8501](http://localhost:8501)

The archived gold dataset (`data/gold/IberFire_coarse32.zarr.tar.gz`) and the latest model (`models/resnet34_v9.pth`) are managed by Git LFS and will be downloaded automatically when you clone the repository. When you run `git lfs pull`, you make sure this download is finished before moving on.

---

### Part 2: Training the Model (beta)
Note: this is still under development, so expect less smooth experience. Proceed with caution.

If you want to train the U-Net model on the existing gold dataset:

**Prerequisites:**
- Python 3.11+
- Physical GPU with support of either CUDA or MPS drivers (recommended) 
- The gold dataset (automatically available via Git LFS)

**Steps:**
1. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start MLflow UI in a separate terminal (for tracking model metrics):**
   ```bash
   mlflow server --host 0.0.0.0 --port 5001
   ```
   Access MLflow UI at [http://localhost:5001](http://localhost:5001)

3. **Train the model:**
   ```bash
   python scripts/train.py --model_name 'my_model' --epochs 50
   ```
   You can get creative with the names, too.

   You can also change the model's backbone from the default *resnet34* by passing `--encoder_name` parameter like so:
   ```bash
   python scripts/train.py --model_name 'my_model' --epochs 50 --encoder_name 'resnet50'
   ```
   This uses SMP's model registry. See the full list of available encoders [here](https://smp.readthedocs.io/en/latest/encoders.html)

   If you want to change the default learning rate of *3e-5*, use `-lr` or `--learning_rate` argument.


The training script uses the gold dataset (`data/gold/IberFire_coarse32.zarr`) and logs all metrics to MLflow for experiment tracking.

---

### Part 3: Full Experimentation (beta)
Note: this is still under development, so expect less smooth experience. Proceed with caution.

If you want to experiment with the dataset, coarsening, feature engineering, etc.:

**Prerequisites:**
- Python 3.11+
- Significant disk space (~20-50 GB for raw NetCDF data)
- Computer connected to power (overnight processing recommended)

**Steps:**
1. **Download the original NetCDF dataset from Zenodo:**
   - Visit the IberFire dataset page on Zenodo: https://zenodo.org/records/15798999
   - Download the NetCDF file
   - Place it in `data/bronze/`

2. **Convert NetCDF to Zarr format:**
   ```bash
   python scripts/conversion.py
   ```
   **Note:** This conversion is best done overnight with your computer connected to power. In the future, this dataset will be published in Zarr format directly.

3. **Customize chunking and compression (optional):**
   The **conversion script** already converts the dataset with chunks optimized for access pattern of this project (1 image at a time fed into U-Net). You are free to change the chunking and compression settings with this helper script:
   ```bash
   python scripts/rechunk.py
   ```
   Don't forget to actually change the values to the ones you need.

4. **Apply coarsening and max pooling to the target:**
   ```bash
   # Use default coarsening factor of 32
   python scripts/coarsen.py
   
   # Or specify a custom coarsening factor
   python scripts/coarsen.py --factor 16
   ```
   That will create a coarser version of dataset and save it in data/gold. 
   Coarser versions are faster to train and have improved class balance (target feature is max-pooled).
   The `--factor` argument controls the spatial coarsening factor (default: 32).

5. After processing, follow **Part 2** to train your models with the new dataset configurations.

---

## Data Processing

The project includes scripts for processing the IberFire dataset:
- `conversion.py`: Convert NetCDF to Zarr format for efficient data access
- `coarsen.py`: Reduce spatial resolution for faster experimentation and less severe imbalance (supports `--factor` argument to customize coarsening)
- `rechunk.py`: Change data chunking 

## Future Plans

We are actively working to improve wildfire prediction capabilities. Future directions include:

- **Bigger Models**: Experimenting with larger U-Net architectures and more sophisticated encoders for improved prediction accuracy
- **Finer Resolution**: Training models on higher spatial resolution data to capture more detailed fire risk patterns
- **Real-Time Data Ingestion**: Implementing a pipeline for real-time data ingestion to predict fires for tomorrow based on current conditions

If you're interested in contributing to any of these areas, please see the Contributing section below.

## Contributing

Interested in contributing to this project? We welcome contributions from the community! Here are some guidelines to get started:

### How to Contribute

1. **Fork the repository** and create a new branch for your feature or bug fix
2. **Make your changes** following the existing code style and conventions
3. **Test your changes** thoroughly to ensure they work as expected
4. **Write clear commit messages** that describe what your changes do
5. **Submit a pull request** with a detailed description of your changes and the problem they solve

### Contribution Ideas

- Work on **real-time data ingestion** for tomorrow's fire predictions
- Experiment with **bigger models** and different architectures
- Improve **spatial resolution** by training on finer-grained data
- Add new features to the web application
- Improve documentation or fix bugs
- Create tutorials or example notebooks

### Questions or Discussions

If you have questions, ideas, or want to discuss potential contributions, feel free to:
- Open an issue on GitHub
- Reach out at **vladimv.morozov@gmail.com**

We appreciate all contributions, whether it's code, documentation, bug reports, or feature suggestions!

## Acknowledgments

This project builds upon the work of several researchers and open-source projects:

### Dataset
- **IberFire Dataset**: Provided by **Julen Ercibengoa Calvo** (julen.ercibengoa@gmail.com, julen.ercibengoa@tekniker.es). The IberFire datacube contains environmental and meteorological data for Spain with 1km spatial resolution and daily temporal resolution.

### Libraries and Frameworks
- **PyTorch**: Deep learning framework
- **segmentation_models_pytorch**: Pre-trained segmentation models
- **MLflow**: Experiment tracking and model management
- **xarray & Zarr**: Multi-dimensional array processing and storage
- **Streamlit**: Web application framework

### Research Papers
- **Long-tail learning via logit adjustment**: Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2020). arXiv preprint arXiv:2007.07314. https://arxiv.org/abs/2007.07314
  - Used for addressing class imbalance in the wildfire prediction model through logit-adjusted loss

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

