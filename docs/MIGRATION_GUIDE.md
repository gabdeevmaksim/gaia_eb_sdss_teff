# Migration Guide: Hardcoded Paths → Configuration System

## What Changed

The project now uses a centralized configuration system to eliminate hardcoded paths and make the codebase portable.

### ✅ Completed Changes

1. **Created configuration system:**
   - `config/config.yaml` - Central configuration file
   - `src/config/settings.py` - Configuration API
   - `src/config/__init__.py` - Module interface

2. **Updated scripts:**
   - `scripts/calculate_temperatures.py` ✓
   - `scripts/clean_panstarrs_photometry.py` ✓
   - `scripts/add_gaia_colors_to_ml_data.py` ✓

3. **Documentation:**
   - `docs/CONFIGURATION.md` - Configuration system guide
   - `examples/configuration_example.py` - Usage example

4. **Dependencies:**
   - Added `pyyaml` to `requirements.txt`

### 🔄 Scripts Remaining to Update

The following scripts still use hardcoded paths and need to be updated:

- `scripts/convert_ecsv_to_parquet.py`
- `scripts/extract_original_ext_source_id.py`
- `scripts/extract_panstarrs_duplicates.py`
- `scripts/merge_panstarrs_duplicates.py`
- `scripts/merge_panstarrs_duplicates_fast.py`
- `scripts/add_new_colors.py`
- `scripts/add_colors_and_temperatures.py`

### 📋 Migration Pattern

To update a script, follow this pattern:

**1. Add imports:**
```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
```

**2. Replace hardcoded paths:**
```python
# OLD:
data_dir = Path("data/processed")
input_file = data_dir / "input.parquet"

# NEW:
config = get_config()
input_file = config.get_dataset_path('input_data', 'processed')
```

**3. Replace hardcoded values:**
```python
# OLD:
missing_val = -999.0

# NEW:
missing_val = config.get('processing', 'missing_value')
```

**4. Update output:**
```python
# Show relative paths in output
print(f"Input: {input_file.relative_to(config.project_root)}")
```

## How to Use Configuration System

### Basic Usage

```python
from src.config import get_config

config = get_config()

# Get paths
data_dir = config.get_path('processed')

# Get dataset paths
catalog = config.get_dataset_path('eb_catalog', 'raw')

# Get parameters
missing_val = config.get('processing', 'missing_value')
```

See `docs/CONFIGURATION.md` for full API documentation.

### Testing

Run the example to verify the system works:

```bash
python examples/configuration_example.py
```

## Benefits

✅ **Portability**: Works on any machine without path changes
✅ **Maintainability**: Single file (`config/config.yaml`) to update
✅ **Flexibility**: Easy to reorganize directory structure
✅ **Testing**: Can use different configs for testing
✅ **Documentation**: Self-documenting configuration

## Adding New Datasets/Paths

1. Edit `config/config.yaml`:
```yaml
datasets:
  my_new_data: filename.parquet
```

2. Use in scripts:
```python
path = config.get_dataset_path('my_new_data', 'processed')
```

## Next Steps for Full Migration

1. Update remaining scripts (list above)
2. Update notebooks to use configuration
3. Add configuration to data loading utilities (`src/data/load_data.py`)
4. Create tests for configuration system
5. Update CLAUDE.md to document configuration usage

## Questions?

See:
- `docs/CONFIGURATION.md` - Full documentation
- `examples/configuration_example.py` - Working example
- `src/config/settings.py` - Implementation
