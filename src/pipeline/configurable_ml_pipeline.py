"""
Configurable machine learning training pipeline for temperature prediction.

This pipeline uses YAML configuration files to define model variants,
eliminating the need for separate training scripts for each model.

Usage:
    # Train using model config
    pipeline = ConfigurableMLPipeline('config/models/gaia_2mass_ir.yaml')
    context = pipeline.run()

    # Or use CLI
    python pipeline.py --ml-config config/models/gaia_2mass_ir.yaml
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
import joblib
import json
import yaml
import logging

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import Pipeline, PipelineStep
from ..features import engineer_all_features
from ..config import get_config


class LoadModelConfigStep(PipelineStep):
    """Load model configuration from YAML file."""

    def __init__(self, config_path: str):
        super().__init__("Load Model Configuration")
        self.config_path = config_path

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")

        with open(config_file, 'r') as f:
            model_config = yaml.safe_load(f)

        self.logger.info(f"Loaded model config: {config_file.name}")
        self.logger.info(f"Model: {model_config['model']['name']}")
        self.logger.info(f"Description: {model_config['model']['description']}")

        context['model_config'] = model_config
        return context


class LoadDataFromConfigStep(PipelineStep):
    """Load training data based on model configuration."""

    def __init__(self):
        super().__init__("Load Training Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model_config = context['model_config']
        config = context['config']

        # Get data source
        source_file = model_config['data']['source_file']

        # Check if source_location is specified (raw/processed), default to processed
        source_location = model_config['data'].get('source_location', 'processed')
        data_dir = Path(config.get_path(source_location))
        data_path = data_dir / source_file

        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        # Load data based on format
        if data_path.suffix == '.parquet':
            df = pl.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pl.read_csv(data_path)
        elif data_path.suffix == '.fits':
            # Load FITS file using astropy, convert to polars
            from astropy.table import Table
            table = Table.read(str(data_path))
            df = pl.from_pandas(table.to_pandas())
            self.logger.info(f"Loaded FITS file from {source_location}/")
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        self.logger.info(f"Loaded {len(df):,} samples from {source_file}")

        context['raw_data'] = df
        return context


class PreprocessDataStep(PipelineStep):
    """Preprocess data based on configuration."""

    def __init__(self):
        super().__init__("Preprocess Data")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['raw_data']
        model_config = context['model_config']
        config = context['config']

        preprocessing = model_config.get('preprocessing', {})
        target = model_config['data']['target']
        features = model_config['data'].get('features', [])

        # Step 1: Apply value filters from config (e.g., temperature range, color limits)
        filters = preprocessing.get('filters', {})
        if filters:
            self.logger.info(f"Applying {len(filters)} value filters from config")
            initial_count = len(df)

            for col, (min_val, max_val) in filters.items():
                if col in df.columns:
                    before = len(df)
                    df = df.filter((pl.col(col) >= min_val) & (pl.col(col) <= max_val))
                    removed = before - len(df)
                    self.logger.info(f"  {col}: [{min_val}, {max_val}] → removed {removed:,} rows")

            self.logger.info(f"After value filters: {len(df):,} samples ({len(df)/initial_count*100:.1f}% retained)")

        # Step 2: Drop rows with missing/NaN values
        if preprocessing.get('drop_missing', True):
            before = len(df)
            # Drop nulls in target and features
            cols_to_check = [target] + features
            cols_to_check = [c for c in cols_to_check if c in df.columns]

            df = df.drop_nulls(subset=cols_to_check)
            removed = before - len(df)
            self.logger.info(f"Dropped {removed:,} rows with missing values")

        self.logger.info(f"After preprocessing: {len(df):,} samples")

        context['clean_data'] = df
        return context


class EngineerFeaturesFromConfigStep(PipelineStep):
    """Engineer features if enabled in configuration."""

    def __init__(self):
        super().__init__("Engineer Features")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['clean_data']
        model_config = context['model_config']

        feature_config = model_config.get('feature_engineering', {})

        if feature_config.get('enabled', False):
            self.logger.info("Feature engineering enabled")

            # Convert to pandas for feature engineering
            df_pd = df.to_pandas()

            color_cols = feature_config.get('color_cols', [])
            mag_cols = feature_config.get('mag_cols', [])

            self.logger.info(f"Color columns: {color_cols}")
            self.logger.info(f"Magnitude columns: {mag_cols}")

            # Engineer features
            df_features = engineer_all_features(
                df_pd,
                color_cols=color_cols,
                mag_cols=mag_cols
            )

            self.logger.info(f"Engineered features: {df_features.shape[1]} columns "
                           f"(from {df.shape[1]})")

            # Convert back to polars
            df_result = pl.from_pandas(df_features)
        else:
            self.logger.info("Feature engineering disabled - using raw features")
            df_result = df

            # Apply custom features from config if specified
            custom_features = feature_config.get('custom_features', {})
            if custom_features:
                self.logger.info(f"Adding {len(custom_features)} custom features from config")
                df_pd = df_result.to_pandas()

                for feature_name, feature_expr in custom_features.items():
                    try:
                        df_pd[feature_name] = df_pd.eval(feature_expr)
                        self.logger.info(f"  Added: {feature_name} = {feature_expr}")
                    except Exception as e:
                        self.logger.warning(f"  Failed to add {feature_name}: {e}")

                df_result = pl.from_pandas(df_pd)
                self.logger.info(f"Features after custom additions: {df_result.shape[1]} columns")

        context['feature_data'] = df_result
        return context


class PrepareTrainTestFromConfigStep(PipelineStep):
    """Prepare train/test split based on configuration."""

    def __init__(self):
        super().__init__("Prepare Train/Test Split")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['feature_data']
        model_config = context['model_config']

        # Get target and features
        target = model_config['data']['target']
        exclude_cols = model_config['data'].get('exclude_columns', [])
        specified_features = model_config['data'].get('features')

        # Determine feature columns
        if specified_features:
            # Use specified features
            feature_cols = [c for c in specified_features if c in df.columns]
            self.logger.info(f"Using {len(feature_cols)} specified features")
        else:
            # Auto-detect: all columns except target and excluded
            all_exclude = [target] + exclude_cols
            feature_cols = [c for c in df.columns if c not in all_exclude]
            self.logger.info(f"Auto-detected {len(feature_cols)} features")

        # Extract features and target
        X = df.select(feature_cols).to_pandas()
        y = df[target].to_pandas()

        # Apply target transformation if specified
        target_transform = model_config.get('target_transform', 'none')
        if target_transform == 'log':
            self.logger.info("Applying log transformation to target variable")
            y_original = y.copy()
            y = np.log10(y)
            self.logger.info(f"  Original target range: [{y_original.min():.0f}, {y_original.max():.0f}]")
            self.logger.info(f"  Transformed target range: [{y.min():.4f}, {y.max():.4f}]")
            context['y_original_train_full'] = y_original  # Store for reference
            context['target_transform'] = 'log'
        elif target_transform == 'log2':
            self.logger.info("Applying log2 transformation to target variable")
            y_original = y.copy()
            y = np.log2(y)
            self.logger.info(f"  Original target range: [{y_original.min():.0f}, {y_original.max():.0f}]")
            self.logger.info(f"  Transformed target range: [{y.min():.4f}, {y.max():.4f}]")
            context['y_original_train_full'] = y_original
            context['target_transform'] = 'log2'
        elif target_transform == 'ln':
            self.logger.info("Applying natural log transformation to target variable")
            y_original = y.copy()
            y = np.log(y)
            self.logger.info(f"  Original target range: [{y_original.min():.0f}, {y_original.max():.0f}]")
            self.logger.info(f"  Transformed target range: [{y.min():.4f}, {y.max():.4f}]")
            context['y_original_train_full'] = y_original
            context['target_transform'] = 'ln'
        else:
            context['target_transform'] = 'none'

        # Check for sample weights
        preprocessing = model_config.get('preprocessing', {})
        sample_weight = None

        if preprocessing.get('use_sample_weights', False):
            weight_col = preprocessing.get('weight_column', 'sample_weight')
            if weight_col in df.columns:
                sample_weight = df[weight_col].to_pandas().values
                self.logger.info(f"Using sample weights from column: {weight_col}")
                self.logger.info(f"  Weight range: [{sample_weight.min():.3f}, {sample_weight.max():.3f}]")
            else:
                self.logger.warning(f"Sample weight column '{weight_col}' not found, ignoring")

        # Train/test split
        training_config = model_config.get('training', {})
        test_size = training_config.get('test_size', 0.2)
        random_state = training_config.get('random_state', 42)

        if sample_weight is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, sample_weight,
                test_size=test_size,
                random_state=random_state
            )
            context['sample_weights_train'] = w_train
            context['sample_weights_test'] = w_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )

        self.logger.info(f"Train: {len(X_train):,} samples")
        self.logger.info(f"Test:  {len(X_test):,} samples")
        self.logger.info(f"Features: {len(feature_cols)}")

        # Log top features for inspection
        if len(feature_cols) <= 10:
            self.logger.info(f"Feature list: {feature_cols}")
        else:
            self.logger.info(f"First 5 features: {feature_cols[:5]}")

        context['X_train'] = X_train
        context['X_test'] = X_test
        context['y_train'] = y_train
        context['y_test'] = y_test
        context['feature_cols'] = feature_cols

        return context


class AddClusteringFeaturesStep(PipelineStep):
    """Add cluster membership probabilities as features.

    Supports multiple clustering methods:
    - gmm: Gaussian Mixture Model
    - bayesian_gmm: Bayesian GMM (auto-selects number of clusters)
    - kmeans: KMeans with soft distance-based probabilities
    """

    def __init__(self):
        super().__init__("Add Clustering Features")

    def _fit_gmm(self, X_train_scaled, clustering_config, random_state):
        """Fit Gaussian Mixture Model."""
        n_clusters = clustering_config.get('n_clusters', 5)

        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=clustering_config.get('covariance_type', 'full'),
            n_init=clustering_config.get('n_init', 10),
            max_iter=clustering_config.get('max_iter', 200),
            reg_covar=clustering_config.get('reg_covar', 1e-6),
            random_state=random_state
        )
        model.fit(X_train_scaled)

        self.logger.info(f"GMM converged: {model.converged_}")
        self.logger.info(f"GMM iterations: {model.n_iter_}")

        return model, n_clusters

    def _fit_bayesian_gmm(self, X_train_scaled, clustering_config, random_state):
        """Fit Bayesian Gaussian Mixture Model (auto-selects clusters)."""
        n_clusters = clustering_config.get('n_clusters', 10)  # Max clusters

        model = BayesianGaussianMixture(
            n_components=n_clusters,
            covariance_type=clustering_config.get('covariance_type', 'full'),
            n_init=clustering_config.get('n_init', 5),
            max_iter=clustering_config.get('max_iter', 200),
            reg_covar=clustering_config.get('reg_covar', 1e-6),
            weight_concentration_prior_type=clustering_config.get('weight_prior', 'dirichlet_process'),
            random_state=random_state
        )
        model.fit(X_train_scaled)

        self.logger.info(f"Bayesian GMM converged: {model.converged_}")
        self.logger.info(f"Bayesian GMM iterations: {model.n_iter_}")

        # Count effective clusters (weight > 0.01)
        effective_clusters = (model.weights_ > 0.01).sum()
        self.logger.info(f"Effective clusters (weight > 1%): {effective_clusters} / {n_clusters}")

        for i, w in enumerate(model.weights_):
            if w > 0.01:
                self.logger.info(f"  Cluster {i}: weight = {w:.4f}")

        return model, n_clusters

    def _fit_kmeans(self, X_train_scaled, clustering_config, random_state):
        """Fit KMeans with soft distance-based probabilities."""
        n_clusters = clustering_config.get('n_clusters', 5)

        model = KMeans(
            n_clusters=n_clusters,
            n_init=clustering_config.get('n_init', 10),
            max_iter=clustering_config.get('max_iter', 300),
            random_state=random_state
        )
        model.fit(X_train_scaled)

        self.logger.info(f"KMeans inertia: {model.inertia_:.2f}")
        self.logger.info(f"KMeans iterations: {model.n_iter_}")

        return model, n_clusters

    def _get_kmeans_probabilities(self, model, X_scaled):
        """Convert KMeans distances to soft probabilities."""
        # Get distances to all centroids
        distances = model.transform(X_scaled)

        # Convert distances to similarity (inverse distance)
        # Add small epsilon to avoid division by zero
        similarities = 1.0 / (distances + 1e-8)

        # Normalize to probabilities
        probs = similarities / similarities.sum(axis=1, keepdims=True)

        return probs

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model_config = context['model_config']
        clustering_config = model_config.get('clustering', {})

        if not clustering_config.get('enabled', False):
            self.logger.info("Clustering features disabled - skipping")
            return context

        method = clustering_config.get('method', 'gmm')
        random_state = model_config.get('training', {}).get('random_state', 42)

        self.logger.info(f"Adding clustering features using method: {method}")

        X_train = context['X_train']
        X_test = context['X_test']
        feature_cols = context['feature_cols']

        # Standardize features for clustering
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        X_test_scaled = scaler.transform(X_test.astype(np.float64))

        # Fit clustering model based on method
        if method == 'gmm':
            model, n_clusters = self._fit_gmm(X_train_scaled, clustering_config, random_state)
            train_probs = model.predict_proba(X_train_scaled)
            test_probs = model.predict_proba(X_test_scaled)

        elif method == 'bayesian_gmm':
            model, n_clusters = self._fit_bayesian_gmm(X_train_scaled, clustering_config, random_state)
            train_probs = model.predict_proba(X_train_scaled)
            test_probs = model.predict_proba(X_test_scaled)

        elif method == 'kmeans':
            model, n_clusters = self._fit_kmeans(X_train_scaled, clustering_config, random_state)
            train_probs = self._get_kmeans_probabilities(model, X_train_scaled)
            test_probs = self._get_kmeans_probabilities(model, X_test_scaled)

        else:
            raise ValueError(f"Unknown clustering method: {method}. "
                           f"Supported: gmm, bayesian_gmm, kmeans")

        # Add probability features to dataframes
        prob_col_names = [f'cluster_prob_{i}' for i in range(n_clusters)]

        X_train_enh = np.hstack([X_train.values, train_probs])
        X_test_enh = np.hstack([X_test.values, test_probs])

        feature_cols_enh = list(feature_cols) + prob_col_names

        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_enh, columns=feature_cols_enh)
        X_test_df = pd.DataFrame(X_test_enh, columns=feature_cols_enh)

        self.logger.info(f"Added {n_clusters} cluster probability features")
        self.logger.info(f"Total features: {len(feature_cols_enh)} (was {len(feature_cols)})")

        # Analyze clusters by target value
        train_labels = train_probs.argmax(axis=1)
        y_train = context['y_train']
        if hasattr(y_train, 'values'):
            y_train_arr = y_train.values
        else:
            y_train_arr = y_train

        self.logger.info("Cluster statistics:")
        for i in range(n_clusters):
            mask = train_labels == i
            n_obj = mask.sum()
            if n_obj > 0:
                target_mean = y_train_arr[mask].mean()
                target_std = y_train_arr[mask].std()
                self.logger.info(f"  Cluster {i}: {n_obj:,} objects ({n_obj/len(train_labels)*100:.1f}%) "
                               f"- Target mean: {target_mean:.1f} +/- {target_std:.1f}")

        # Store models for saving
        context['clustering_model'] = model
        context['clustering_scaler'] = scaler
        context['clustering_method'] = method
        context['n_clusters'] = n_clusters

        # Update context
        context['X_train'] = X_train_df
        context['X_test'] = X_test_df
        context['feature_cols'] = feature_cols_enh

        return context


class TrainModelFromConfigStep(PipelineStep):
    """Train Random Forest model using configuration."""

    def __init__(self):
        super().__init__("Train Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context['X_train']
        y_train = context['y_train']
        model_config = context['model_config']

        # Get hyperparameters
        hyperparams = model_config.get('hyperparameters', {})

        # Handle max_features (can be string or number)
        max_features = hyperparams.get('max_features', 'log2')
        if isinstance(max_features, str) and max_features.isdigit():
            max_features = int(max_features)

        model_params = {
            'n_estimators': hyperparams.get('n_estimators', 300),
            'max_depth': hyperparams.get('max_depth', 20),
            'min_samples_split': hyperparams.get('min_samples_split', 5),
            'min_samples_leaf': hyperparams.get('min_samples_leaf', 4),
            'max_features': max_features,
            'random_state': hyperparams.get('random_state', 42),
            'n_jobs': hyperparams.get('n_jobs', -1),
            'verbose': 0
        }

        self.logger.info("Training Random Forest with parameters:")
        for param, value in model_params.items():
            if param != 'verbose':
                self.logger.info(f"  {param}: {value}")

        # Create and train model
        model = RandomForestRegressor(**model_params)

        # Use sample weights if available
        sample_weights = context.get('sample_weights_train')
        if sample_weights is not None:
            self.logger.info("Training with sample weights")
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        self.logger.info("✓ Model trained successfully")

        context['model'] = model
        context['hyperparameters'] = model_params
        return context


class EvaluateModelFromConfigStep(PipelineStep):
    """Evaluate model performance."""

    def __init__(self):
        super().__init__("Evaluate Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context['model']
        X_train = context['X_train']
        X_test = context['X_test']
        y_train = context['y_train']
        y_test = context['y_test']

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Inverse transform predictions if target was transformed
        target_transform = context.get('target_transform', 'none')
        if target_transform == 'log':
            self.logger.info("Inverse transforming predictions (10^y)")
            y_pred_train = 10 ** y_pred_train
            y_pred_test = 10 ** y_pred_test
            y_train = 10 ** y_train
            y_test = 10 ** y_test
        elif target_transform == 'log2':
            self.logger.info("Inverse transforming predictions (2^y)")
            y_pred_train = 2 ** y_pred_train
            y_pred_test = 2 ** y_pred_test
            y_train = 2 ** y_train
            y_test = 2 ** y_test
        elif target_transform == 'ln':
            self.logger.info("Inverse transforming predictions (e^y)")
            y_pred_train = np.exp(y_pred_train)
            y_pred_test = np.exp(y_pred_test)
            y_train = np.exp(y_train)
            y_test = np.exp(y_test)

        # Calculate metrics (on original scale)
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)

        # Update context with inverse-transformed values for saving
        context['y_train'] = y_train
        context['y_test'] = y_test

        self.logger.info("Performance Metrics:")
        self.logger.info("  TRAIN SET:")
        self.logger.info(f"    MAE:  {train_metrics['mae']:.0f} K")
        self.logger.info(f"    RMSE: {train_metrics['rmse']:.0f} K")
        self.logger.info(f"    R²:   {train_metrics['r2']:.4f}")
        self.logger.info(f"    Within 10%: {train_metrics['within_10_percent']:.1f}%")

        self.logger.info("  TEST SET:")
        self.logger.info(f"    MAE:  {test_metrics['mae']:.0f} K")
        self.logger.info(f"    RMSE: {test_metrics['rmse']:.0f} K")
        self.logger.info(f"    R²:   {test_metrics['r2']:.4f}")
        self.logger.info(f"    Within 10%: {test_metrics['within_10_percent']:.1f}%")

        # Feature importance
        feature_importance = dict(zip(
            context['feature_cols'],
            model.feature_importances_
        ))

        # Sort by importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info("Top 10 Features by Importance:")
        for feature, importance in top_features:
            self.logger.info(f"    {feature}: {importance:.4f}")

        context['y_pred_train'] = y_pred_train
        context['y_pred_test'] = y_pred_test
        context['train_metrics'] = train_metrics
        context['test_metrics'] = test_metrics
        context['feature_importance'] = feature_importance

        return context

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Relative error
        relative_errors = np.abs((y_true - y_pred) / y_true)
        mean_rel_error = np.mean(relative_errors) * 100

        # Percentage within thresholds
        within_5 = (relative_errors <= 0.05).sum() / len(y_true) * 100
        within_10 = (relative_errors <= 0.10).sum() / len(y_true) * 100
        within_20 = (relative_errors <= 0.20).sum() / len(y_true) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mean_relative_error': float(mean_rel_error),
            'within_5_percent': float(within_5),
            'within_10_percent': float(within_10),
            'within_20_percent': float(within_20),
            'n_samples': len(y_true)
        }


class SaveModelFromConfigStep(PipelineStep):
    """Save trained model and artifacts."""

    def __init__(self):
        super().__init__("Save Model")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context['model']
        model_config = context['model_config']
        test_metrics = context['test_metrics']
        train_metrics = context['train_metrics']
        feature_cols = context['feature_cols']
        feature_importance = context['feature_importance']
        config = context['config']

        # Create model ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_prefix = model_config['model']['id_prefix']
        model_id = f"{id_prefix}_{timestamp}"

        models_dir = Path(config.get_path('models', ensure_exists=True))

        # Save model
        model_file = models_dir / f"{model_id}.pkl"
        joblib.dump(model, model_file)
        self.logger.info(f"✓ Saved model: {model_file.name}")

        # Save metadata
        metadata = {
            'model_id': model_id,
            'timestamp': timestamp,
            'model_name': model_config['model']['name'],
            'description': model_config['model']['description'],
            'model_type': 'RandomForestRegressor',
            'n_features': len(feature_cols),
            'features': feature_cols,
            'data_source': model_config['data']['source_file'],
            'target': model_config['data']['target'],
            'target_transform': context.get('target_transform', 'none'),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'hyperparameters': context['hyperparameters'],
            'config_file': str(context.get('config_file', 'unknown')),
            'data_config': model_config['data']  # Save full data config for validation
        }

        metadata_file = models_dir / f"{model_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"✓ Saved metadata: {metadata_file.name}")

        # Save summary text file
        summary_file = models_dir / f"{model_id}_SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Model: {model_config['model']['name']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Model ID: {model_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Description: {model_config['model']['description']}\n\n")

            f.write(f"Data Source: {model_config['data']['source_file']}\n")
            f.write(f"Target: {model_config['data']['target']}\n")
            target_transform = context.get('target_transform', 'none')
            if target_transform != 'none':
                f.write(f"Target Transform: {target_transform}\n")
            f.write(f"Features: {len(feature_cols)}\n\n")

            f.write("TRAIN SET PERFORMANCE:\n")
            f.write(f"  MAE:  {train_metrics['mae']:.1f} K\n")
            f.write(f"  RMSE: {train_metrics['rmse']:.1f} K\n")
            f.write(f"  R²:   {train_metrics['r2']:.4f}\n")
            f.write(f"  Within 10%: {train_metrics['within_10_percent']:.1f}%\n\n")

            f.write("TEST SET PERFORMANCE:\n")
            f.write(f"  MAE:  {test_metrics['mae']:.1f} K\n")
            f.write(f"  RMSE: {test_metrics['rmse']:.1f} K\n")
            f.write(f"  R²:   {test_metrics['r2']:.4f}\n")
            f.write(f"  Within 10%: {test_metrics['within_10_percent']:.1f}%\n\n")

            f.write("TOP 10 FEATURES:\n")
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(top_features, 1):
                f.write(f"  {i:2d}. {feature:30s} {importance:.4f}\n")

        self.logger.info(f"✓ Saved summary: {summary_file.name}")

        # Save predictions with original features (for color-temp plots)
        # Note: y_test and y_pred_test are already inverse-transformed if needed
        predictions_df = pd.DataFrame({
            'y_true': context['y_test'],
            'y_pred': context['y_pred_test']
        })

        # Add original color features to predictions for validation plots
        # Get the original features from X_test (before any feature engineering)
        X_test_df = context['X_test']
        if isinstance(X_test_df, pd.DataFrame):
            # Try to add raw feature columns (colors) that might be useful for plotting
            data_features = model_config['data'].get('features', [])
            for feat in data_features:
                if feat in X_test_df.columns:
                    predictions_df[feat] = X_test_df[feat].values

        pred_file = models_dir / f"{model_id}_test_predictions.parquet"
        predictions_df.to_parquet(pred_file)
        self.logger.info(f"✓ Saved predictions: {pred_file.name}")

        # Save clustering models if they were used
        if 'clustering_model' in context:
            method = context.get('clustering_method', 'gmm')
            cluster_file = models_dir / f"{model_id}_clustering_{method}.pkl"
            scaler_file = models_dir / f"{model_id}_clustering_scaler.pkl"

            joblib.dump(context['clustering_model'], cluster_file)
            joblib.dump(context['clustering_scaler'], scaler_file)

            self.logger.info(f"✓ Saved clustering model ({method}): {cluster_file.name}")
            self.logger.info(f"✓ Saved clustering scaler: {scaler_file.name}")

            # Update metadata with clustering info
            metadata['clustering'] = {
                'method': method,
                'n_clusters': context['n_clusters'],
                'model_file': str(cluster_file.name),
                'scaler_file': str(scaler_file.name)
            }

            # Re-save metadata with clustering info
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        context['model_id'] = model_id
        context['model_file'] = str(model_file)
        context['metadata_file'] = str(metadata_file)
        context['summary_file'] = str(summary_file)

        return context


class ConfigurableMLPipeline(Pipeline):
    """
    Configurable ML training pipeline that uses YAML configuration files.

    This eliminates the need for separate training scripts for each model variant.
    All model variations can be trained using different config files.

    Usage
    -----
    >>> # Train using config file
    >>> pipeline = ConfigurableMLPipeline('config/models/gaia_2mass_ir.yaml')
    >>> context = pipeline.run()
    >>> print(f"Model saved: {context['model_id']}")

    >>> # Or from CLI
    >>> python pipeline.py --ml-config config/models/gaia_2mass_ir.yaml

    Parameters
    ----------
    model_config_path : str
        Path to model configuration YAML file
    """

    def __init__(self, model_config_path: str):
        # Store config path for later use
        self.model_config_path = model_config_path

        steps = [
            LoadModelConfigStep(model_config_path),
            LoadDataFromConfigStep(),
            PreprocessDataStep(),
            EngineerFeaturesFromConfigStep(),
            PrepareTrainTestFromConfigStep(),
            AddClusteringFeaturesStep(),
            TrainModelFromConfigStep(),
            EvaluateModelFromConfigStep(),
            SaveModelFromConfigStep(),
        ]

        # Extract model name from config for pipeline name
        config_file = Path(model_config_path)
        pipeline_name = f"ML Training Pipeline ({config_file.stem})"

        super().__init__(pipeline_name, steps)

    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run the pipeline with initial context."""
        context = {
            'config': get_config(),
            'config_file': self.model_config_path
        }
        return super().run(context)
