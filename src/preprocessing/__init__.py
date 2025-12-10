# ==============================================================================
# src/preprocessing/__init__.py
# ==============================================================================
"""
Preprocessing package for data cleaning and preparation.

Contains:
    - DataPreprocessor: Handles missing values, outliers, validation
    - FeatureEngineer: Creates CDI smoothing, volatility, lagged features
    - Convenience functions for quick data loading
"""

from .preprocessing import (
    DataPreprocessor, 
    FeatureEngineer,
    load_and_preprocess,
    create_features_from_raw
)

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer',
    'load_and_preprocess',
    'create_features_from_raw'
]
