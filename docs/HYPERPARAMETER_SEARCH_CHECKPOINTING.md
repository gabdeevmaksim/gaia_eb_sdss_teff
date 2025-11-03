# Hyperparameter Search Checkpointing

This document explains how to use the checkpointing feature in hyperparameter search scripts.

## Overview

The hyperparameter search script (`scripts/hyperparam_search_gaia_g_bprp.py`) includes automatic checkpointing to save progress and allow resuming from interruptions.

## How Checkpointing Works

1. **Automatic Saving**: Progress is saved after each iteration completes
2. **Resume on Restart**: If the script is interrupted, it automatically resumes from the last checkpoint
3. **Deduplication**: Already-tested parameter combinations are tracked and skipped
4. **Auto-Cleanup**: Checkpoint is deleted after successful completion
5. **Resource Management**: Uses only 8 CPU cores (half of 16 available) to prevent overloading

## Usage

### Starting a New Search

```bash
python scripts/hyperparam_search_gaia_g_bprp.py
```

### Resuming After Interruption

Simply run the same command - the script will automatically detect and resume from the checkpoint:

```bash
python scripts/hyperparam_search_gaia_g_bprp.py
```

Output will show:
```
4. Resuming from checkpoint...
   Found X completed iterations
   Checkpoint saved at: YYYY-MM-DDTHH:MM:SS
   Total iterations: 50
   Remaining iterations: Y
```

### Starting Fresh (Ignore Existing Checkpoint)

```bash
rm models/checkpoints/hyperparam_search_gaia_g_bprp.pkl
python scripts/hyperparam_search_gaia_g_bprp.py
```

## Checkpoint Location

Checkpoints are stored in:
```
models/checkpoints/hyperparam_search_gaia_g_bprp.pkl
```

## Checkpoint Contents

Each checkpoint contains:
- **results**: List of all completed iterations with parameters and CV scores
- **tested_params**: Set of parameter combinations already tested (prevents duplicates)
- **timestamp**: When the checkpoint was last saved (ISO format)

## Resource Usage

The script is configured to use **8 CPU cores** (half of the 16 available on your laptop) to avoid consuming all system resources. This is set via the `N_JOBS = 8` parameter.

## Inspecting Checkpoint Progress

To check progress without running the script:

```python
import pickle
from pathlib import Path

checkpoint_path = Path('models/checkpoints/hyperparam_search_gaia_g_bprp.pkl')

if checkpoint_path.exists():
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Completed iterations: {len(data['results'])}/{50}")
    print(f"Last saved: {data['timestamp']}")
    print(f"\nBest result so far:")
    best = min(data['results'], key=lambda x: x['mean_cv_score'])
    print(f"  CV MAE: {-best['mean_cv_score']:.1f} K")
    print(f"  Parameters: {best['params']}")
else:
    print("No checkpoint found")
```

## Search Parameters

- **Iterations**: 50 random parameter combinations
- **Cross-validation folds**: 5
- **Scoring metric**: Mean Absolute Error (MAE)
- **CPU cores**: 8 (half of available)

## Hyperparameter Search Space

```python
{
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
```

## Expected Runtime

- **Per iteration**: ~2-5 minutes (depends on parameters)
- **Total**: ~2-4 hours for 50 iterations
- **Resumable**: Can be interrupted and resumed at any time

## Output Files

After completion, the script generates:

1. **Model file**: `models/rf_gaia_g_bprp_optimized_YYYYMMDD_HHMMSS.pkl`
2. **Search results**: `models/rf_gaia_g_bprp_optimized_YYYYMMDD_HHMMSS_search_results.json`
3. **Test predictions**: `models/rf_gaia_g_bprp_optimized_YYYYMMDD_HHMMSS_test_predictions.parquet`
4. **Metadata**: `models/rf_gaia_g_bprp_optimized_YYYYMMDD_HHMMSS_metadata.json`
5. **Summary**: `models/rf_gaia_g_bprp_optimized_YYYYMMDD_HHMMSS_SUMMARY.txt`

The checkpoint file is automatically deleted after successful completion.

## Troubleshooting

### Checkpoint is corrupted

```bash
rm models/checkpoints/hyperparam_search_gaia_g_bprp.pkl
python scripts/hyperparam_search_gaia_g_bprp.py
```

### Want to change search parameters

If you modify the search space or number of iterations, it's recommended to start fresh:

```bash
rm models/checkpoints/hyperparam_search_gaia_g_bprp.pkl
python scripts/hyperparam_search_gaia_g_bprp.py
```

### Running out of memory

The script uses 8 cores by default. If memory is still an issue, edit the script and reduce `N_JOBS`:

```python
N_JOBS = 4  # Use only 4 cores instead of 8
```

## Notes

- The script uses the same random state for data splitting, ensuring reproducibility
- Parameter combinations are randomly generated but deterministic (based on iteration number)
- Duplicate parameter combinations are automatically skipped
- All iterations use the same train/test split for fair comparison
