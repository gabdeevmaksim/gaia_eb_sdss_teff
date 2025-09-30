"""
Data loading utilities for eclipsing binary analysis.
"""

import os
from pathlib import Path
from astropy.table import Table
import polars as pl
import pandas as pd
from typing import Union, Optional


def load_eb_catalog(
    file_path: Union[str, Path], 
    format: str = 'auto',
    convert_to: str = 'polars'
) -> Union[Table, pl.DataFrame, pd.DataFrame]:
    """
    Load eclipsing binary catalog data.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the catalog file
    format : str, default 'auto'
        File format ('auto', 'parquet', 'ecsv', 'csv', etc.)
        If 'auto', determines format from file extension
    convert_to : str, default 'polars'
        Output format: 'astropy', 'polars', or 'pandas'
        
    Returns
    -------
    data : Table, DataFrame, or Polars DataFrame
        Loaded catalog data
    """
    file_path = Path(file_path)
    print(f"Loading catalog from: {file_path}")
    
    # Auto-detect format from extension
    if format == 'auto':
        if file_path.suffix.lower() == '.parquet':
            format = 'parquet'
        elif file_path.suffix.lower() == '.ecsv':
            format = 'ascii.ecsv'
        elif file_path.suffix.lower() == '.csv':
            format = 'ascii.csv'
        else:
            format = 'ascii'
    
    # Load data efficiently based on format and desired output
    if format == 'parquet' and convert_to == 'polars':
        # Direct Parquet -> Polars (fastest)
        data = pl.read_parquet(file_path)
        print(f"Successfully loaded {data.height:,} rows with {data.width} columns")
        return data
    elif format == 'parquet' and convert_to == 'pandas':
        # Direct Parquet -> Pandas
        data = pd.read_parquet(file_path)
        print(f"Successfully loaded {len(data):,} rows with {len(data.columns)} columns")
        return data
    else:
        # Load with astropy for other formats/outputs
        table = Table.read(file_path, format=format)
        print(f"Successfully loaded {len(table):,} rows with {len(table.colnames)} columns")
        
        if convert_to == 'astropy':
            return table
        elif convert_to == 'polars':
            # Convert to polars for efficient analysis
            df = table.to_pandas()
            return pl.from_pandas(df)
        elif convert_to == 'pandas':
            return table.to_pandas()
        else:
            raise ValueError("convert_to must be 'astropy', 'polars', or 'pandas'")


def get_data_info(data: Union[Table, pl.DataFrame, pd.DataFrame]) -> dict:
    """
    Get basic information about the loaded data.
    
    Parameters
    ----------
    data : Table, DataFrame, or Polars DataFrame
        The data to analyze
        
    Returns
    -------
    info : dict
        Dictionary with data information
    """
    if isinstance(data, Table):
        return {
            'n_rows': len(data),
            'n_columns': len(data.colnames),
            'columns': data.colnames,
            'dtypes': [str(data[col].dtype) for col in data.colnames]
        }
    elif isinstance(data, pl.DataFrame):
        return {
            'n_rows': data.height,
            'n_columns': data.width,
            'columns': data.columns,
            'dtypes': [str(dtype) for dtype in data.dtypes]
        }
    elif isinstance(data, pd.DataFrame):
        return {
            'n_rows': len(data),
            'n_columns': len(data.columns),
            'columns': list(data.columns),
            'dtypes': [str(dtype) for dtype in data.dtypes]
        }
