#!/usr/bin/env python3
"""
Example script demonstrating the configuration system.

This shows how to use the centralized configuration to:
- Access data paths
- Get dataset file paths
- Read configuration parameters
- Ensure directories exist
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def main():
    print("=" * 70)
    print("CONFIGURATION SYSTEM EXAMPLE")
    print("=" * 70)

    # Get configuration instance (singleton)
    config = get_config()

    print(f"\n1. PROJECT ROOT")
    print(f"   Auto-detected project root: {config.project_root}")

    print(f"\n2. DATA DIRECTORIES")
    print(f"   Raw data:       {config.get_path('raw')}")
    print(f"   Processed data: {config.get_path('processed')}")
    print(f"   External data:  {config.get_path('external')}")
    print(f"   Models:         {config.get_path('models')}")

    print(f"\n3. DATASET PATHS")
    print(f"   EB Catalog (raw):        {config.get_dataset_path('eb_catalog', 'raw')}")
    print(f"   EB Catalog (parquet):    {config.get_dataset_path('eb_catalog_parquet', 'processed')}")
    print(f"   ML Training Data:        {config.get_dataset_path('ml_training_clean', 'processed')}")

    print(f"\n4. PROCESSING PARAMETERS")
    print(f"   Missing value:    {config.get('processing', 'missing_value')}")
    print(f"   Color threshold:  {config.get('processing', 'color_threshold')}")

    print(f"\n5. ML PARAMETERS")
    print(f"   Test size:        {config.get('ml', 'test_size')}")
    print(f"   Random state:     {config.get('ml', 'random_state')}")
    print(f"   RF estimators:    {config.get('ml', 'rf_n_estimators')}")
    print(f"   RF max depth:     {config.get('ml', 'rf_max_depth')}")

    print(f"\n6. TEMPERATURE COEFFICIENTS")
    gr_coef = config.get('temperature', 'gr_coefficients')
    print(f"   g-r: A={gr_coef['A']}, B={gr_coef['B']}")

    ri_coef = config.get('temperature', 'ri_coefficients')
    print(f"   r-i: A={ri_coef['A']}, B={ri_coef['B']}")

    iz_coef = config.get('temperature', 'iz_coefficients')
    print(f"   i-z: A={iz_coef['A']}, B={iz_coef['B']}")

    print(f"\n7. CREATING DIRECTORIES")
    # Create reports directory if it doesn't exist
    reports_dir = config.get_path('reports', ensure_exists=True)
    print(f"   Ensured exists:   {reports_dir}")

    print(f"\n8. RELATIVE PATHS (for display)")
    catalog_path = config.get_dataset_path('eb_catalog', 'raw')
    print(f"   Absolute: {catalog_path}")
    print(f"   Relative: {catalog_path.relative_to(config.project_root)}")

    print(f"\n9. ALL CONFIGURATION")
    print(f"   Available via config.get_all():")
    all_config = config.get_all()
    print(f"   - Top-level keys: {list(all_config.keys())}")

    print("\n" + "=" * 70)
    print("✓ Configuration system working correctly!")
    print("=" * 70)


if __name__ == '__main__':
    main()
