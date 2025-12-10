#!/usr/bin/env python3
"""
Upload datasets and models to HuggingFace Hub.

This script uploads the full catalog, training datasets, and pre-trained models
to HuggingFace Hub for public access and easy distribution.

Prerequisites:
    - Install huggingface_hub: pip install huggingface_hub
    - Login to HuggingFace: huggingface-cli login
    - Or set HF_TOKEN environment variable

Usage:
    # Upload training datasets
    python scripts/upload_to_huggingface.py --datasets training

    # Upload catalog
    python scripts/upload_to_huggingface.py --datasets catalog

    # Upload specific model
    python scripts/upload_to_huggingface.py --model rf_gaia_teff_corrected_log_20251126_130144

    # Upload everything (WARNING: Large upload!)
    python scripts/upload_to_huggingface.py --datasets all --models all
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

import yaml


# TODO: Replace with your actual HuggingFace organization/username
HF_DATASET_REPO = "YOUR_ORG/gaia-eb-teff-datasets"
HF_MODEL_REPO = "YOUR_ORG/gaia-eb-teff-models"


def create_repositories():
    """Create HuggingFace repositories if they don't exist."""
    api = HfApi()

    print("Creating HuggingFace repositories (if they don't exist)...")

    try:
        # Create dataset repository
        create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            exist_ok=True,
            private=False  # Set to True for private datasets
        )
        print(f"  ✓ Dataset repository: https://huggingface.co/datasets/{HF_DATASET_REPO}")
    except Exception as e:
        print(f"  ✗ Error creating dataset repo: {e}")

    try:
        # Create model repository
        create_repo(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            exist_ok=True,
            private=False  # Set to True for private models
        )
        print(f"  ✓ Model repository: https://huggingface.co/models/{HF_MODEL_REPO}")
    except Exception as e:
        print(f"  ✗ Error creating model repo: {e}")


def upload_datasets(dataset_type: str):
    """
    Upload datasets to HuggingFace Hub.

    Parameters
    ----------
    dataset_type : str
        Type of dataset to upload: 'training', 'catalog', 'prediction', or 'all'
    """
    api = HfApi()

    datasets = {
        'training': {
            'folder': 'data/processed',
            'patterns': ['gaia_all_colors_train*.parquet'],
            'path_in_repo': 'training'
        },
        'catalog': {
            'folder': 'data/processed',
            'patterns': ['stars_types_with_best_predictions*'],
            'path_in_repo': 'catalogs'
        },
        'prediction': {
            'folder': 'data/processed',
            'patterns': ['gaia_all_colors_predict.parquet', 'eb_2mass_photometry.parquet'],
            'path_in_repo': 'prediction'
        }
    }

    if dataset_type == 'all':
        for dt in ['training', 'catalog', 'prediction']:
            upload_datasets(dt)
        return

    if dataset_type not in datasets:
        print(f"Error: Unknown dataset type: {dataset_type}")
        return

    dataset_info = datasets[dataset_type]
    folder = Path(dataset_info['folder'])

    if not folder.exists():
        print(f"Error: Data folder not found: {folder}")
        return

    print(f"\nUploading {dataset_type} datasets to {HF_DATASET_REPO}...")

    # Upload files matching patterns
    for pattern in dataset_info['patterns']:
        matching_files = list(folder.glob(pattern))

        if not matching_files:
            print(f"  ⚠ No files matching pattern: {pattern}")
            continue

        for file_path in matching_files:
            try:
                print(f"  Uploading: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")

                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"{dataset_info['path_in_repo']}/{file_path.name}",
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset"
                )

                print(f"    ✓ Uploaded successfully")

            except Exception as e:
                print(f"    ✗ Error: {e}")

    print(f"\n✓ {dataset_type.capitalize()} datasets uploaded")


def upload_models(model_name: str):
    """
    Upload trained models to HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        Model name (from model_registry.yaml) or 'all' to upload all models
    """
    api = HfApi()

    # Load model registry
    registry_path = Path("config/models/model_registry.yaml")
    if not registry_path.exists():
        print(f"Error: Model registry not found at {registry_path}")
        print("Create model_registry.yaml before uploading models")
        return

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    models_dir = Path("models")
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return

    # Upload all models if requested
    if model_name == 'all':
        for model_key in registry.get('models', {}).keys():
            upload_models(model_key)

        # Also upload model registry
        try:
            print("\nUploading model registry...")
            api.upload_file(
                path_or_fileobj=str(registry_path),
                path_in_repo="model_registry.yaml",
                repo_id=HF_MODEL_REPO,
                repo_type="model"
            )
            print("  ✓ Model registry uploaded")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        return

    # Upload specific model
    if model_name not in registry.get('models', {}):
        print(f"Error: Model '{model_name}' not found in registry")
        return

    model_info = registry['models'][model_name]
    model_file = model_info['file']
    model_path = models_dir / model_file

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"\nUploading model: {model_name}")
    print(f"  File: {model_file} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    try:
        # Upload .pkl file
        print(f"  Uploading {model_file}...")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_file,
            repo_id=HF_MODEL_REPO,
            repo_type="model"
        )
        print(f"    ✓ Uploaded successfully")

        # Upload metadata file
        metadata_file = model_file.replace('.pkl', '_metadata.json')
        metadata_path = models_dir / metadata_file

        if metadata_path.exists():
            print(f"  Uploading {metadata_file}...")
            api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo=metadata_file,
                repo_id=HF_MODEL_REPO,
                repo_type="model"
            )
            print(f"    ✓ Uploaded successfully")

        # Upload summary file if exists
        summary_file = model_file.replace('.pkl', '_SUMMARY.txt')
        summary_path = models_dir / summary_file

        if summary_path.exists():
            print(f"  Uploading {summary_file}...")
            api.upload_file(
                path_or_fileobj=str(summary_path),
                path_in_repo=summary_file,
                repo_id=HF_MODEL_REPO,
                repo_type="model"
            )
            print(f"    ✓ Uploaded successfully")

        print(f"\n✓ Model uploaded successfully: {model_name}")

    except Exception as e:
        print(f"✗ Error uploading model: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload datasets and models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--datasets',
        choices=['training', 'catalog', 'prediction', 'all'],
        help='Dataset type to upload'
    )
    parser.add_argument(
        '--models',
        help='Model name to upload (or "all" for all models)'
    )
    parser.add_argument(
        '--create-repos',
        action='store_true',
        help='Create HuggingFace repositories if they don\'t exist'
    )

    args = parser.parse_args()

    if not args.datasets and not args.models and not args.create_repos:
        parser.print_help()
        print("\nError: Specify --datasets, --models, or --create-repos")
        sys.exit(1)

    # Check for authentication
    if not os.getenv('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")
        print("You should be logged in with: huggingface-cli login")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Create repositories if requested
    if args.create_repos:
        create_repositories()

    # Upload datasets
    if args.datasets:
        upload_datasets(args.datasets)

    # Upload models
    if args.models:
        upload_models(args.models)

    print("\n✓ Upload complete!")
    print(f"\nDataset repository: https://huggingface.co/datasets/{HF_DATASET_REPO}")
    print(f"Model repository: https://huggingface.co/models/{HF_MODEL_REPO}")


if __name__ == '__main__':
    main()
