# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an astronomical data science project for analyzing eclipsing binary stars using Gaia, Pan-STARRS, and SDSS data. The project focuses on effective temperature (Teff) analysis of eclipsing binaries with a ~1.2 million row catalog dataset.

## Environment Setup

**Virtual Environment**: Use `.venv/` directory (not `venv/`)
```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies**:
- `astropy` - Astronomical data handling, ECSV format support
- `polars` - High-performance data analysis (preferred over pandas for large datasets)
- `pandas` - Alternative data analysis (available but prefer polars)
- `pyarrow` - Parquet format support for efficient data storage
- `healpy` - HEALPix sky mapping
- `aplpy` - Astronomical plotting
- `reproject` - Astronomical image reprojection
- `numpy` - Numerical computing foundation
- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning algorithms
- `jupyter`/`notebook` - Interactive analysis environment

## Data Architecture

**Data Flow**: ECSV → Parquet → Analysis
- Raw data in `data/raw/` (ECSV format, ~226MB, 1.2M rows)
- Processed data in `data/processed/` (Parquet format for ~5-10x faster loading)
- External data in `data/external/` for third-party datasets
- Interim data in `data/interim/` for intermediate processing steps
- Cache system in `data/cache/` for expensive computations

**Primary Dataset**: `data/raw/eb_panstarrs_with_param-result.ecsv`
- Eclipsing binary catalog with Pan-STARRS photometry
- Use `scripts/convert_ecsv_to_parquet.py` to convert for performance

**Current Processed Files**:
- `data/processed/eb_catalog.parquet` - Main catalog (82MB)
- `data/processed/eb_catalog_with_pm.parquet` - Catalog with proper motion (92MB)
- `data/processed/original_ext_source_id.csv` - Extracted source IDs (23MB)
- `data/processed/gaia_eb_panstarrs_phot_with_temperatures.parquet` - Pan-STARRS photometry with effective temperatures (1.17M objects)
- `data/processed/gaia_eb_sdss_teff.parquet` - SDSS photometry with effective temperatures
- `data/processed/gaia_eb_colors_temperatures.parquet` - Multi-band colors (B-V, V-K) and temperatures

## Core Scripts and Usage

**Data Conversion**:
```bash
# Convert ECSV to Parquet for faster loading
python scripts/convert_ecsv_to_parquet.py data/raw/eb_panstarrs_with_param-result.ecsv
```

**Data Extraction**:
```bash
# Extract original_ext_source_id column to CSV
python scripts/extract_original_ext_source_id.py
```

**Pan-STARRS Photometry Processing**:
```bash
# Extract duplicate Pan-STARRS entries
python scripts/extract_panstarrs_duplicates.py

# Merge duplicate measurements (fast version)
python scripts/merge_panstarrs_duplicates_fast.py

# Clean photometry data and filter for magnitude pairs
python scripts/clean_panstarrs_photometry.py

# Calculate effective temperatures from colors
python scripts/calculate_temperatures.py

# Add new colors (B-V, V-K) and temperatures to existing dataset
python scripts/add_new_colors.py
```

**Jupyter Analysis**:
```bash
# Start Jupyter for interactive analysis
jupyter lab notebooks/

# Analysis notebooks:
# - eclipsing_binary_analysis.ipynb - Main analysis notebook
# - sdss_temperature_analysis.ipynb - SDSS temperature analysis
# - bv_vk_temperature_analysis.ipynb - B-V and V-K color-temperature analysis
# - color_quality_analysis.ipynb - Color data quality assessment
```

## Code Architecture

**Module Structure**:
- `src/data/` - Data loading and caching utilities
  - `load_data.py` - Multi-format data loader (ECSV, Parquet, CSV)
  - `cache_manager.py` - Caching system for expensive computations
- `src/visualization/` - Plotting and visualization functions
  - `plots.py` - Astronomical plots, sky maps, histograms
- `src/features/` - Feature engineering (placeholder)
- `src/models/` - Machine learning models (placeholder)

**Data Loading Pattern**:
```python
from src.data.load_data import load_eb_catalog

# Load as Polars DataFrame (recommended for performance)
data = load_eb_catalog('data/processed/eb_catalog.parquet', convert_to='polars')

# Load as Astropy Table (preserves astronomical metadata)
table = load_eb_catalog('data/raw/eb_catalog.ecsv', convert_to='astropy')
```

**Cache Usage**:
```python
from src.data.cache_manager import CacheManager
cache = CacheManager()  # Uses data/cache/ by default
```

## Performance Considerations

**Large Dataset Handling**:
- Always prefer Parquet over ECSV for repeated analysis
- Use Polars over Pandas for operations on the full dataset
- Implement caching for expensive astronomical computations
- Use astropy.Table only when astronomical metadata is required

**Data Processing Pipeline**:
1. Load raw ECSV with astropy (preserves metadata)
2. Convert to Polars for analysis (performance)
3. Cache intermediate results
4. Save processed data as Parquet

## Astronomical Specifics

**Coordinate Systems**: Use `astropy.coordinates.SkyCoord` for astronomical coordinates
**Sky Visualization**: HEALPix maps with `healpy` for all-sky distributions
**Photometric Data**: Pan-STARRS photometry accessed via MAST API
**File Formats**: ECSV (Enhanced CSV) is the astronomical standard for tabular data

## Development Workflow

**Working with Large Data**:
- Test on subsets before processing full catalog
- Use `data.sample(n=1000)` (Polars) or `data[:1000]` (astropy) for development
- Monitor memory usage when working with full dataset

**Notebook Development**:
- Primary analysis in `notebooks/eclipsing_binary_analysis.ipynb`
- Keep notebooks focused on exploration; move production code to `src/`

## Pan-STARRS Data Processing

**Photometry Pipeline**:
- Raw Pan-STARRS data contains duplicate measurements for many sources
- Duplicates are merged using weighted averages based on photometric errors
- Final dataset includes colors (g-r, r-i, i-z) and effective temperatures
- Filter encoding system tracks which photometric bands are available per object
- Temperature calculations use empirical color-temperature relations with quality cuts

**Data Quality**:
- Missing measurements encoded as -999.0
- Color constraints (>= -0.5) applied before temperature calculations
- Filter encoding: "gr", "ri", "griz" etc. indicates available photometric bands
- ~95% of objects have all three temperature estimates (Te_gr, Te_ri, Te_iz)

## Directory Structure

```
├── scripts/                    # Data processing scripts
│   ├── convert_ecsv_to_parquet.py         # ECSV → Parquet conversion
│   ├── extract_original_ext_source_id.py  # Source ID extraction
│   ├── extract_panstarrs_duplicates.py    # Find duplicate Pan-STARRS measurements
│   ├── merge_panstarrs_duplicates_fast.py # Merge duplicates with weighted averages
│   ├── clean_panstarrs_photometry.py      # Clean and filter photometry data
│   ├── calculate_temperatures.py          # Calculate effective temperatures
│   ├── add_new_colors.py                  # Add B-V, V-K colors and temperatures
│   └── add_colors_and_temperatures.py     # Alternative color/temperature script
├── src/                        # Source code modules
│   ├── data/                   # Data handling utilities
│   │   ├── load_data.py       # Multi-format data loader
│   │   └── cache_manager.py   # Caching system
│   ├── visualization/          # Plotting and visualization
│   │   └── plots.py           # Astronomical plots, sky maps
│   ├── features/              # Feature engineering (placeholder)
│   └── models/                # Machine learning models (placeholder)
├── notebooks/                  # Jupyter notebooks
│   ├── eclipsing_binary_analysis.ipynb     # Main analysis notebook
│   ├── sdss_temperature_analysis.ipynb     # SDSS temperature analysis
│   ├── bv_vk_temperature_analysis.ipynb    # B-V, V-K color analysis
│   └── color_quality_analysis.ipynb        # Color data quality assessment
├── data/                       # Data storage
│   ├── raw/                   # Original ECSV files
│   ├── processed/             # Converted Parquet files
│   ├── external/              # Third-party datasets
│   └── interim/               # Intermediate processing
├── config/                     # Configuration files
├── docs/                       # Documentation
├── reports/                    # Generated reports
│   ├── figures/               # Generated plots
│   └── presentations/         # Presentation materials
└── tests/                      # Unit tests
```