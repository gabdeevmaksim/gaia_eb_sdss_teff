# Gaia Eclipsing Binary Temperature Prediction Pipeline

Production-ready ML pipeline for predicting effective temperatures of eclipsing binary stars using Gaia, Pan-STARRS, SDSS, and 2MASS photometry.

**Deployment Branch** - This is the production-ready branch with Docker containerization and automated dataset downloads. For development work, see the `main` branch.

## Features

- **5 Complete Pipelines**: Data processing, ML training, prediction, validation, complete workflow
- **30+ Pre-configured Models**: YAML-driven model configurations for different photometric combinations
- **Best-of-Three Ensemble**: 263K mean uncertainty (18% improvement vs single model)
- **Docker Support**: Training and prediction containers with auto-downloads
- **HuggingFace Integration**: Automatic dataset and model downloads from HuggingFace Hub
- **Production Ready**: Logging, error handling, versioned models, reproducible results

## Dataset

**Full Catalog**: 2.1M eclipsing binaries with Teff predictions (97.2% coverage)
- HuggingFace: [YOUR_ORG/gaia-eb-teff-datasets](https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets)
- 196MB FITS file with quality flags (A=Gaia, B=ML<300K, C=ML<500K, D=ML≥500K)

**Pre-trained Models** (1.2-2GB each):
- **gaia_teff_corrected_log** (RECOMMENDED): 556K MAE, R²=0.640
- **gaia_2mass_ir**: Optical + infrared, 765K MAE
- **gaia_all_colors_teff_log**: Log-transformed, 557K MAE

See [DATASET_ACCESS.md](DATASET_ACCESS.md) for download instructions.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository (deploy branch)
git clone https://github.com/YOUR_ORG/gaia-eb-teff.git -b deploy
cd gaia-eb-teff

# Setup environment
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Train a model
docker-compose up train

# Run predictions
docker-compose up predict

# Generate validation plots
docker-compose up validate
```

See [DOCKER_USAGE.md](DOCKER_USAGE.md) for detailed Docker instructions.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/gaia-eb-teff.git -b deploy
cd gaia-eb-teff

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires HF_TOKEN)
export HF_TOKEN=your_huggingface_token
python scripts/download_datasets.py --datasets training

# Train a model
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# Run predictions
python pipeline.py --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# Validate model
python pipeline.py --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml
```

## Pipeline Usage

### 1. Training Pipeline

Train a new model using configuration files:

```bash
# Using Docker
docker-compose run train --ml --ml-config config/models/gaia_teff_corrected_log.yaml

# Local
python pipeline.py --ml --ml-config config/models/gaia_teff_corrected_log.yaml
```

**Output**: Model saved to `models/` with timestamped filename, metadata, summary, and test predictions.

### 2. Prediction Pipeline

Generate predictions for new data:

```bash
# Using Docker (auto-downloads model)
docker-compose run predict --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml

# Local
python pipeline.py --predict --pred-config config/prediction/predict_gaia_colors_teff.yaml
```

**Output**: Predictions saved to `data/processed/` as parquet file.

### 3. Validation Pipeline

Generate validation plots and metrics:

```bash
# Using Docker
docker-compose run validate --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml

# Local
python pipeline.py --validate --val-config config/validation/validate_gaia_teff_corrected_log.yaml
```

**Output**: Plots saved to `reports/figures/`, metrics saved to `reports/`.

### 4. Complete Pipeline

Run all pipelines sequentially:

```bash
python pipeline.py --all
```

## Configuration

All pipelines are configured via YAML files in `config/`:

- **`config/models/*.yaml`**: Model training configurations (30 files)
  - Define features, hyperparameters, target transformations
  - Example: `gaia_teff_corrected_log.yaml`

- **`config/prediction/*.yaml`**: Prediction configurations (18 files)
  - Specify model, input data, preprocessing
  - Supports wildcard model matching

- **`config/validation/*.yaml`**: Validation configurations (20 files)
  - Define plots, metrics, output locations

- **`config/config.yaml`**: Central configuration
  - Paths, datasets, default parameters

See [docs/CONFIGURABLE_PIPELINE.md](docs/CONFIGURABLE_PIPELINE.md) for configuration details.

## Model Performance

| Model | MAE (K) | RMSE (K) | R² | Within 10% |
|-------|---------|----------|-----|------------|
| Gaia Teff Corrected Log (BEST) | 557 | 1021 | 0.640 | 68.5% |
| Gaia + 2MASS IR | 765 | 1168 | 0.315 | 43.4% |
| Gaia All Colors Log | 557 | 1021 | 0.640 | 68.5% |
| Ensemble PanSTARRS | 720 | 1184 | 0.297 | 53.0% |

## Directory Structure

```
deploy/
├── pipeline.py              # Master orchestrator
├── requirements.txt         # Full dependencies
├── requirements-docker.txt  # Minimal for containers
├── Dockerfile               # Prediction container
├── Dockerfile.train         # Training container
├── docker-compose.yml       # Multi-service orchestration
├── docker-entrypoint.sh     # Container startup script
├── .env.example             # Environment template
│
├── src/                     # Source modules
│   ├── config/              # Settings management
│   ├── data/                # Data loading, caching
│   ├── features/            # Feature engineering
│   ├── visualization/       # Plotting utilities
│   └── pipeline/            # Pipeline implementations
│
├── config/                  # Configuration
│   ├── config.yaml          # Central config
│   ├── models/              # Training configs (30)
│   ├── prediction/          # Prediction configs (18)
│   └── validation/          # Validation configs (20)
│
├── docs/                    # Documentation
│   ├── PIPELINES.md
│   ├── CONFIGURABLE_PIPELINE.md
│   ├── CONFIGURATION.md
│   └── ...
│
└── scripts/                 # Essential scripts
    ├── download_datasets.py
    └── upload_to_huggingface.py
```

## Documentation

- **[DOCKER_USAGE.md](DOCKER_USAGE.md)**: Complete Docker guide (building, running, troubleshooting)
- **[DATASET_ACCESS.md](DATASET_ACCESS.md)**: Dataset download methods and structure
- **[docs/PIPELINES.md](docs/PIPELINES.md)**: Pipeline architecture and usage
- **[docs/CONFIGURABLE_PIPELINE.md](docs/CONFIGURABLE_PIPELINE.md)**: Model configuration guide
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**: Configuration system API

## Requirements

**Python**: 3.9+

**Core Dependencies**:
- numpy, pandas, scikit-learn, joblib
- polars, pyarrow (data formats)
- astropy (FITS files)
- pyyaml (configuration)
- huggingface_hub (dataset downloads)

**Optional** (for validation plots):
- matplotlib, seaborn

**Docker**: For containerized deployment (recommended)

## Environment Variables

Create `.env` file from `.env.example`:

```bash
# Required for dataset/model downloads
HF_TOKEN=your_huggingface_token

# Optional for specific model
MODEL_NAME=rf_gaia_teff_corrected_log_20251126_130144

# Optional: Kaggle credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

## Examples

### Example 1: Train Custom Model

```bash
# Create custom config: config/models/my_custom_model.yaml
# Then train:
python pipeline.py --ml --ml-config config/models/my_custom_model.yaml
```

### Example 2: Batch Predictions

```bash
# Prepare your input data as parquet file
# Create prediction config
# Run predictions
python pipeline.py --predict --pred-config config/prediction/my_predictions.yaml
```

### Example 3: Model Comparison

```bash
# Train multiple models
for config in config/models/*.yaml; do
    python pipeline.py --ml --ml-config $config
done

# Validate all
for config in config/validation/*.yaml; do
    python pipeline.py --validate --val-config $config
done
```

## Troubleshooting

### Issue: HuggingFace download fails

**Solution**: Ensure `HF_TOKEN` is set and valid:
```bash
export HF_TOKEN=your_token
# Or login via CLI:
huggingface-cli login
```

### Issue: Out of memory during training

**Solution**: Reduce `n_estimators` or `max_depth` in model config, or use Docker with memory limits:
```bash
docker run --memory=8g --memory-swap=16g ...
```

### Issue: Model not found

**Solution**: Check model registry and download:
```bash
python scripts/download_datasets.py --model gaia_teff_corrected_log
```

## Citation

If you use this pipeline or dataset, please cite:

```bibtex
@article{your_paper,
  title={Effective Temperature Predictions for Eclipsing Binary Stars},
  author={Your Name},
  journal={Journal},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

This is the deployment branch. For development contributions, please work on the `main` branch and submit pull requests there.

## Support

- **Issues**: https://github.com/YOUR_ORG/gaia-eb-teff/issues
- **Discussions**: https://github.com/YOUR_ORG/gaia-eb-teff/discussions
- **Email**: your.email@example.com

## Acknowledgments

- **Gaia Mission**: ESA's Gaia satellite (DR3)
- **Pan-STARRS**: Panoramic Survey Telescope and Rapid Response System
- **SDSS**: Sloan Digital Sky Survey
- **2MASS**: Two Micron All Sky Survey
