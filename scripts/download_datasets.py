#!/usr/bin/env python3
"""
Download datasets and models from HuggingFace Hub.

This script is called automatically by Docker containers on first run,
but can also be used manually for local setup.

Usage:
    # Download training datasets
    python scripts/download_datasets.py --datasets training

    # Download catalog
    python scripts/download_datasets.py --datasets catalog

    # Download specific model
    python scripts/download_datasets.py --model rf_gaia_teff_corrected_log_20251126_130144

    # Download everything
    python scripts/download_datasets.py --datasets all --model all
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

import yaml


# HuggingFace repositories
HF_DATASET_REPO = "Dedulek/gaia-eb-teff-datasets"
HF_MODEL_REPO = "Dedulek/gaia-eb-teff-models"


def download_from_huggingface(dataset_name: str, output_dir: str = "data/processed"):
    """
    Download dataset from HuggingFace Hub.

    Parameters
    ----------
    dataset_name : str
        Dataset to download: 'training', 'catalog', 'prediction', or 'all'
    output_dir : str
        Output directory for downloaded files
    """
    print(f"Downloading {dataset_name} datasets from HuggingFace...")
    print(f"Repository: {HF_DATASET_REPO}")

    # Define dataset files
    datasets = {
        'training': [
            'photometry/eb_unified_photometry.parquet',
            'photometry/eb_unified_photometry_SUMMARY.txt',
        ],
        'catalog': [
            'catalogs/stars_types_with_best_predictions.fits',
            'catalogs/stars_types_with_best_predictions_DESCRIPTION.txt',
        ],
        'correction': [
            'correction/teff_correction_coeffs_deg2.pkl',
        ],
        'all': [
            'photometry/*.parquet',
            'photometry/*.txt',
            'catalogs/*.fits',
            'catalogs/*.txt',
            'correction/*.pkl',
        ]
    }

    files = datasets.get(dataset_name, [dataset_name])

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download files
    downloaded_count = 0
    for file_pattern in files:
        try:
            print(f"  Downloading: {file_pattern}")

            # Use snapshot_download for wildcards, hf_hub_download for specific files
            if '*' in file_pattern:
                snapshot_download(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    allow_patterns=[file_pattern],
                    local_dir=output_path,
                    local_dir_use_symlinks=False
                )
            else:
                downloaded_file = hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename=file_pattern,
                    local_dir=output_path,
                    repo_type="dataset",
                    local_dir_use_symlinks=False
                )
                print(f"    ✓ Downloaded to: {downloaded_file}")

                # Move file to output_path root if it was downloaded with subdirectory
                from pathlib import Path as PathLib
                downloaded_path = PathLib(downloaded_file)
                target_path = output_path / downloaded_path.name
                if downloaded_path != target_path and downloaded_path.exists():
                    import shutil
                    shutil.move(str(downloaded_path), str(target_path))
                    print(f"    → Moved to: {target_path}")
                    # Clean up empty subdirectory if it exists
                    if downloaded_path.parent != output_path and downloaded_path.parent.exists():
                        try:
                            downloaded_path.parent.rmdir()
                        except:
                            pass

            downloaded_count += 1

        except Exception as e:
            print(f"    ✗ Error downloading {file_pattern}: {e}")

    print(f"\nDownloaded {downloaded_count} file(s) to {output_path}")
    return downloaded_count > 0


def download_model(model_name: str, output_dir: str = "models"):
    """
    Download trained model from HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        Model name (from model_registry.yaml) or 'all' to download all models
    output_dir : str
        Output directory for downloaded models
    """
    print(f"Downloading model: {model_name}")
    print(f"Repository: {HF_MODEL_REPO}")

    # Load model registry if it exists
    registry_path = Path("config/models/model_registry.yaml")
    if registry_path.exists():
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
    else:
        print(f"Warning: Model registry not found at {registry_path}")
        print("Attempting to download model file directly...")
        registry = {'models': {}}

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download all models if requested
    if model_name == 'all':
        if not registry.get('models'):
            print("Error: No models defined in registry")
            return False

        for model_key in registry['models'].keys():
            download_model(model_key, output_dir)
        return True

    # Download specific model
    if model_name in registry.get('models', {}):
        model_info = registry['models'][model_name]
        model_file = model_info['file']

        # Download model file (.pkl)
        try:
            print(f"  Downloading: {model_file}")
            downloaded_pkl = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=model_file,
                local_dir=output_path,
                repo_type="model",
                local_dir_use_symlinks=False
            )
            print(f"    ✓ Downloaded to: {downloaded_pkl}")

            # Download metadata file (.json)
            metadata_file = model_file.replace('.pkl', '_metadata.json')
            print(f"  Downloading: {metadata_file}")
            downloaded_json = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=metadata_file,
                local_dir=output_path,
                repo_type="model",
                local_dir_use_symlinks=False
            )
            print(f"    ✓ Downloaded to: {downloaded_json}")

            print(f"\n✓ Model downloaded successfully: {model_name}")
            return True

        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            return False
    else:
        # Try downloading directly if not in registry
        print(f"Model '{model_name}' not found in registry. Trying direct download...")
        try:
            downloaded_file = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=f"{model_name}.pkl",
                local_dir=output_path,
                repo_type="model",
                local_dir_use_symlinks=False
            )
            print(f"✓ Downloaded to: {downloaded_file}")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--datasets',
        choices=['training', 'catalog', 'correction', 'all'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--model',
        help='Model name to download (or "all" for all models)'
    )
    parser.add_argument(
        '--output-data',
        default='data/processed',
        help='Output directory for datasets (default: data/processed)'
    )
    parser.add_argument(
        '--output-models',
        default='models',
        help='Output directory for models (default: models)'
    )

    args = parser.parse_args()

    if not args.datasets and not args.model:
        parser.print_help()
        print("\nError: Specify --datasets and/or --model")
        sys.exit(1)

    # Check for HF_TOKEN
    if not os.getenv('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")
        print("Public datasets will work, but private datasets will fail.")
        print("Set HF_TOKEN with: export HF_TOKEN=your_token")

    success = True

    # Download datasets
    if args.datasets:
        # Correction coefficients go to data/ root, others to data/processed
        output_dir = "data" if args.datasets == 'correction' else args.output_data
        if not download_from_huggingface(args.datasets, output_dir):
            success = False

    # Download models
    if args.model:
        if not download_model(args.model, args.output_models):
            success = False

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
