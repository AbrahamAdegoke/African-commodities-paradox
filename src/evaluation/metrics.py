"""
Model Evaluation Metrics Module

Computes comprehensive evaluation metrics for regression models:
- R² (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Additional diagnostic metrics

Author: Abraham Adegoke
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error
)
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Methods:
        - evaluate: Compute all metrics
        - compare_models: Compare multiple models
        - generate_report: Create formatted evaluation report
    """
    
    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for logging
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # R² Score (coefficient of determination)
        # 1.0 = perfect fit, 0.0 = baseline (mean), <0 = worse than baseline
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Root Mean Squared Error
        # Lower is better, same units as target
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Error
        # Lower is better, more robust to outliers than RMSE
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        # Interpretable as average % error
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics['mape'] = np.nan
        
        # Adjusted R² (penalizes model complexity)
        n = len(y_true)
        p = 1  # number of predictors (will be updated externally if needed)
        metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        
        logger.info(f"\n{model_name} Evaluation:")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    @staticmethod
    def compare_models(
        results: Dict[str, Dict[str, float]],
        metric: str = 'r2'
    ) -> pd.DataFrame:
        """
        Compare multiple models based on metrics.
        
        Args:
            results: Dictionary of {model_name: metrics_dict}
            metric: Primary metric for ranking ('r2', 'rmse', 'mae')
        
        Returns:
            DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(results).T
        
        # Sort by primary metric
        ascending = False if metric == 'r2' else True  # Higher R² is better, lower RMSE/MAE is better
        comparison_df = comparison_df.sort_values(metric, ascending=ascending)
        
        # Add rank column
        comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df
    
    @staticmethod
    def generate_report(
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report for multiple models.
        
        Args:
            y_true: True target values
            predictions_dict: Dictionary of {model_name: predictions}
            save_path: Optional path to save report
        
        Returns:
            Formatted report string
        """
        report = "\n" + "=" * 80 + "\n"
        report += "MODEL EVALUATION REPORT - AFRICAN COMMODITIES PARADOX\n"
        report += "=" * 80 + "\n\n"
        
        # Evaluate each model
        results = {}
        for model_name, y_pred in predictions_dict.items():
            results[model_name] = ModelEvaluator.evaluate(y_true, y_pred, model_name)
        
        # Create comparison table
        comparison_df = ModelEvaluator.compare_models(results)
        
        report += "PERFORMANCE COMPARISON (sorted by R²):\n"
        report += "-" * 80 + "\n"
        report += comparison_df[['r2', 'rmse', 'mae', 'mape', 'rank']].to_string()
        report += "\n\n"
        
        # Best model analysis
        best_model = comparison_df.index[0]
        report += f"BEST MODEL: {best_model}\n"
        report += "-" * 80 + "\n"
        best_metrics = results[best_model]
        report += f"  R² Score: {best_metrics['r2']:.4f}\n"
        report += f"  RMSE: {best_metrics['rmse']:.4f}\n"
        report += f"  MAE: {best_metrics['mae']:.4f}\n"
        report += f"  MAPE: {best_metrics['mape']:.2f}%\n"
        report += f"  Mean Residual: {best_metrics['mean_residual']:.4f}\n"
        report += f"  Std Residual: {best_metrics['std_residual']:.4f}\n"
        report += f"  Max Error: {best_metrics['max_error']:.4f}\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += "INTERPRETATION:\n"
        report += "-" * 80 + "\n"
        report += f"R² = {best_metrics['r2']:.4f} means the model explains "
        report += f"{best_metrics['r2']*100:.2f}% of variance in GDP volatility.\n"
        report += f"\nOn average, predictions are off by {best_metrics['mae']:.4f} "
        report += f"(log volatility units).\n"
        report += "=" * 80 + "\n\n"
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report


def calculate_prediction_intervals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    confidence: float = 0.95
) -> tuple:
    """
    Calculate prediction intervals for regression.
    
    Args:
        y_pred: Predicted values
        residuals: Residuals (y_true - y_pred)
        confidence: Confidence level (default 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Calculate standard error
    std_residuals = np.std(residuals)
    
    # Get z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate intervals
    margin = z_score * std_residuals
    lower_bound = y_pred - margin
    upper_bound = y_pred + margin
    
    return lower_bound, upper_bound


# Example usage
if __name__ == "__main__":
    print("Testing Model Evaluator...")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randn(n_samples)
    
    # Model 1: Good predictions
    y_pred_ridge = y_true + np.random.randn(n_samples) * 0.3
    
    # Model 2: Better predictions
    y_pred_gbr = y_true + np.random.randn(n_samples) * 0.2
    
    # Evaluate
    evaluator = ModelEvaluator()
    
    predictions = {
        'Ridge Regression': y_pred_ridge,
        'Gradient Boosting': y_pred_gbr
    }
    
    report = evaluator.generate_report(
        y_true, 
        predictions,
        save_path='test_evaluation_report.txt'
    )
    
    print(report)
    
    print("\n✓ Model Evaluator test completed successfully!")