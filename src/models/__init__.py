# ==============================================================================
# src/models/__init__.py
# ==============================================================================
"""
Models package for African Commodities Paradox project.

This package contains regression models for:
1. Predicting GDP growth volatility (main objective)
2. Forecasting next-year GDP growth (stretch goal)

Models:
    - RidgeRegressionModel: Baseline linear model with L2 regularization
    - GradientBoostingModel: Non-linear model capturing interactions
    - GDPGrowthForecaster: Stretch goal - forecast GDP growth at t+1
"""

from .ridge_regression import RidgeRegressionModel
from .gradient_boosting import GradientBoostingModel
from .gdp_forecaster import GDPGrowthForecaster

__all__ = [
    'RidgeRegressionModel',
    'GradientBoostingModel',
    'GDPGrowthForecaster'
]
