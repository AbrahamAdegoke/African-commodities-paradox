# ==============================================================================
# src/models/__init__.py
# ==============================================================================
"""
Models package for African Commodities Paradox project.

This package contains regression models for predicting GDP growth volatility
based on commodity dependence and macroeconomic indicators.
"""

from .ridge_regression import RidgeRegressionModel
from .gradient_boosting import GradientBoostingModel

__all__ = [
    'RidgeRegressionModel',
    'GradientBoostingModel'
]