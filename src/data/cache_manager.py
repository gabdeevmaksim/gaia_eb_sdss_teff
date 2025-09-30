"""
Cache management for expensive computations and analysis results.
"""

import pickle
import json
from pathlib import Path
import hashlib
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import polars as pl


class CacheManager:
    """
    Manages caching of expensive computations and analysis results.
    """
    
    def __init__(self, cache_dir: Union[str, Path] = None):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        cache_dir : str or Path, optional
            Directory for cache files. If None, uses data/cache/
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / 'data' / 'cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_data_hash(self, data: pl.DataFrame) -> str:
        """Generate hash for data to detect changes."""
        # Use shape and a sample of data for hashing
        sample_data = data.head(1000).to_pandas().to_string()
        content = f"{data.shape}_{sample_data}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_cache_key(self, operation: str, data_hash: str, **kwargs) -> str:
        """Generate cache key for operation."""
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{operation}_{data_hash}_{hashlib.md5(params_str.encode()).hexdigest()[:8]}"
    
    def get(self, operation: str, data: pl.DataFrame, **kwargs) -> Optional[Any]:
        """
        Get cached result if available and valid.
        
        Parameters
        ----------
        operation : str
            Name of the operation (e.g., 'histograms', 'sky_plot_data')
        data : pl.DataFrame
            The data being processed
        **kwargs
            Additional parameters for the operation
            
        Returns
        -------
        result : Any or None
            Cached result if available, None otherwise
        """
        data_hash = self._get_data_hash(data)
        cache_key = self._get_cache_key(operation, data_hash, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists() and cache_key in self.metadata:
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Update access time
                self.metadata[cache_key]['last_accessed'] = time.time()
                self._save_metadata()
                
                print(f"📦 Loaded cached result for {operation}")
                return result
            except Exception as e:
                print(f"⚠️ Error loading cache for {operation}: {e}")
                # Remove corrupted cache
                cache_file.unlink(missing_ok=True)
                if cache_key in self.metadata:
                    del self.metadata[cache_key]
                    self._save_metadata()
        
        return None
    
    def set(self, operation: str, data: pl.DataFrame, result: Any, **kwargs):
        """
        Cache computation result.
        
        Parameters
        ----------
        operation : str
            Name of the operation
        data : pl.DataFrame
            The data being processed
        result : Any
            Result to cache
        **kwargs
            Additional parameters for the operation
        """
        data_hash = self._get_data_hash(data)
        cache_key = self._get_cache_key(operation, data_hash, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'operation': operation,
                'data_shape': data.shape,
                'created': time.time(),
                'last_accessed': time.time(),
                'file_size': cache_file.stat().st_size,
                'parameters': kwargs
            }
            self._save_metadata()
            
            print(f"💾 Cached result for {operation}")
        except Exception as e:
            print(f"⚠️ Error caching result for {operation}: {e}")
    
    def invalidate(self, operation: str = None):
        """
        Invalidate cache entries.
        
        Parameters
        ----------
        operation : str, optional
            If provided, only invalidate caches for this operation.
            If None, invalidate all caches.
        """
        to_remove = []
        
        for cache_key, metadata in self.metadata.items():
            if operation is None or metadata['operation'] == operation:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                cache_file.unlink(missing_ok=True)
                to_remove.append(cache_key)
        
        for key in to_remove:
            del self.metadata[key]
        
        self._save_metadata()
        print(f"🗑️ Invalidated {len(to_remove)} cache entries")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached items."""
        total_size = sum(item['file_size'] for item in self.metadata.values())
        
        info = {
            'total_items': len(self.metadata),
            'total_size_mb': total_size / (1024**2),
            'operations': {}
        }
        
        for metadata in self.metadata.values():
            op = metadata['operation']
            if op not in info['operations']:
                info['operations'][op] = {'count': 0, 'size_mb': 0}
            info['operations'][op]['count'] += 1
            info['operations'][op]['size_mb'] += metadata['file_size'] / (1024**2)
        
        return info


def compute_histogram_data(data: pl.DataFrame, columns: list, bins: int = 50) -> Dict:
    """
    Compute histogram data for specified columns.
    
    Parameters
    ----------
    data : pl.DataFrame
        The data
    columns : list
        Column names to compute histograms for
    bins : int
        Number of bins
        
    Returns
    -------
    hist_data : dict
        Dictionary with histogram data for each column
    """
    hist_data = {}
    
    for col in columns:
        if col in data.columns:
            values = data[col].drop_nulls().to_numpy()
            if len(values) > 0:
                hist, bin_edges = np.histogram(values, bins=bins)
                hist_data[col] = {
                    'hist': hist,
                    'bin_edges': bin_edges,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'total_count': len(data)
                }
    
    return hist_data


def compute_sky_plot_data(data: pl.DataFrame, coord_system: str = 'galactic', 
                         value_col: str = None, sample_size: int = None) -> Dict:
    """
    Compute sky plot data for visualization.
    
    Parameters
    ----------
    data : pl.DataFrame
        The data with coordinates
    coord_system : str
        'galactic' or 'equatorial'
    value_col : str, optional
        Column for color coding
    sample_size : int, optional
        Subsample data for faster plotting
        
    Returns
    -------
    plot_data : dict
        Dictionary with plot data
    """
    if coord_system == 'galactic':
        lon_col, lat_col = 'l', 'b'
    else:
        lon_col, lat_col = 'ra', 'dec'
    
    # Filter valid data
    if value_col and value_col in data.columns:
        valid_data = data.filter(
            (pl.col(lon_col).is_not_null()) & 
            (pl.col(lat_col).is_not_null()) & 
            (pl.col(value_col).is_not_null())
        )
    else:
        valid_data = data.filter(
            (pl.col(lon_col).is_not_null()) & 
            (pl.col(lat_col).is_not_null())
        )
    
    # Subsample if requested
    if sample_size and len(valid_data) > sample_size:
        valid_data = valid_data.sample(sample_size, seed=42)
    
    plot_data = {
        'lon': valid_data[lon_col].to_numpy(),
        'lat': valid_data[lat_col].to_numpy(),
        'coord_system': coord_system,
        'count': len(valid_data)
    }
    
    if value_col and value_col in valid_data.columns:
        plot_data['values'] = valid_data[value_col].to_numpy()
        plot_data['value_col'] = value_col
    
    return plot_data

