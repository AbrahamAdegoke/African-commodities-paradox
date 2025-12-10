# ==============================================================================
# src/evaluation/__init__.py
# ==============================================================================
"""
Evaluation package for model assessment.

Contains:
    - ModelEvaluator: Metrics computation (R², RMSE, MAE)
    - SHAPAnalyzer: SHAP-based feature importance and interpretability
    - Visualization tools for model comparison
"""

from .metrics import ModelEvaluator, calculate_prediction_intervals

# Import SHAP analyzer if shap is available
try:
    from .shap_analysis import SHAPAnalyzer, analyze_with_shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

__all__ = [
    'ModelEvaluator',
    'calculate_prediction_intervals'
]

if SHAP_AVAILABLE:
    __all__.extend(['SHAPAnalyzer', 'analyze_with_shap'])
