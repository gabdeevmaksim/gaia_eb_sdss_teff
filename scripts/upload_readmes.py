#!/usr/bin/env python3
"""Upload README files to HuggingFace repositories."""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)


def main():
    # Initialize API
    api = HfApi()

    # Check for HF_TOKEN
    if not os.getenv('HF_TOKEN'):
        print("Warning: HF_TOKEN environment variable not set.")
        print("Set with: export HF_TOKEN=your_token")
        sys.exit(1)

    # Upload dataset README
    print("Uploading dataset README...")
    try:
        api.upload_file(
            path_or_fileobj="HF_DATASET_README.md",
            path_in_repo="README.md",
            repo_id="Dedulek/gaia-eb-teff-datasets",
            repo_type="dataset",
            commit_message="Add comprehensive dataset documentation"
        )
        print("  ✓ Dataset README uploaded")
    except Exception as e:
        print(f"  ✗ Error uploading dataset README: {e}")

    # Upload model README
    print("\nUploading model README...")
    try:
        api.upload_file(
            path_or_fileobj="HF_MODEL_README.md",
            path_in_repo="README.md",
            repo_id="Dedulek/gaia-eb-teff-models",
            repo_type="model",
            commit_message="Add comprehensive model documentation"
        )
        print("  ✓ Model README uploaded")
    except Exception as e:
        print(f"  ✗ Error uploading model README: {e}")

    print("\n✓ All README files uploaded successfully!")
    print(f"\nDataset: https://huggingface.co/datasets/Dedulek/gaia-eb-teff-datasets")
    print(f"Models: https://huggingface.co/Dedulek/gaia-eb-teff-models")


if __name__ == '__main__':
    main()
