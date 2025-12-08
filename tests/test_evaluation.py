"""
Unit tests for model evaluation module.

Tests evaluation metrics and model comparison functionality.

Author: Abraham Adegoke
Date: December 2025
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.metrics import ModelEvaluator, calculate_prediction_intervals


@pytest.fixture
def perfect_predictions():
    """Create perfect predictions for testing."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true.copy()  # Perfect predictions
    return y_true, y_pred


@pytest.fixture
def noisy_predictions():
    """Create noisy predictions for testing."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.3  # Add noise
    return y_true, y_pred


@pytest.fixture
def multiple_model_predictions():
    """Create predictions from multiple models."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    
    predictions = {
        'Model A': y_true + np.random.randn(100) * 0.2,  # Better model
        'Model B': y_true + np.random.randn(100) * 0.5,  # Worse model
        'Model C': y_true + np.random.randn(100) * 0.3   # Medium model
    }
    
    return y_true, predictions


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""
    
    def test_evaluate_perfect_predictions(self, perfect_predictions):
        """Test evaluation with perfect predictions."""
        y_true, y_pred = perfect_predictions
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred, "Perfect Model")
        
        # Perfect predictions should have R²=1, RMSE=0, MAE=0
        assert metrics['r2'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mean_residual'] == pytest.approx(0.0, abs=1e-10)
    
    def test_evaluate_noisy_predictions(self, noisy_predictions):
        """Test evaluation with noisy predictions."""
        y_true, y_pred = noisy_predictions
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred, "Noisy Model")
        
        # Should have reasonable but imperfect metrics
        assert 0.0 < metrics['r2'] < 1.0
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['rmse'] >= metrics['mae']  # RMSE should be >= MAE
    
    def test_evaluate_metric_types(self, noisy_predictions):
        """Test that all expected metrics are computed."""
        y_true, y_pred = noisy_predictions
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        # Check all expected metrics are present
        expected_metrics = [
            'r2', 'rmse', 'mae', 'mape', 'adj_r2',
            'mean_residual', 'std_residual', 'max_error'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(metrics[metric]) or metric == 'mape'  # MAPE can be NaN
            assert not np.isinf(metrics[metric])
    
    def test_evaluate_negative_r2(self):
        """Test that R² can be negative for very poor predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])  # Reversed - very bad
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        # Should have negative R² (worse than mean baseline)
        assert metrics['r2'] < 0
    
    def test_compare_models(self, multiple_model_predictions):
        """Test model comparison functionality."""
        y_true, predictions = multiple_model_predictions
        
        # Evaluate all models
        results = {}
        for model_name, y_pred in predictions.items():
            results[model_name] = ModelEvaluator.evaluate(y_true, y_pred, model_name)
        
        # Compare models
        comparison_df = ModelEvaluator.compare_models(results, metric='r2')
        
        # Check comparison DataFrame
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'rank' in comparison_df.columns
        assert len(comparison_df) == len(predictions)
        
        # Check sorting (higher R² should rank first)
        assert comparison_df['rank'].iloc[0] == 1
        assert comparison_df['r2'].iloc[0] >= comparison_df['r2'].iloc[1]
    
    def test_compare_models_by_rmse(self, multiple_model_predictions):
        """Test model comparison by RMSE."""
        y_true, predictions = multiple_model_predictions
        
        results = {}
        for model_name, y_pred in predictions.items():
            results[model_name] = ModelEvaluator.evaluate(y_true, y_pred, model_name)
        
        # Compare by RMSE (lower is better)
        comparison_df = ModelEvaluator.compare_models(results, metric='rmse')
        
        # Check sorting (lower RMSE should rank first)
        assert comparison_df['rmse'].iloc[0] <= comparison_df['rmse'].iloc[1]
    
    def test_generate_report(self, multiple_model_predictions, tmp_path):
        """Test report generation."""
        y_true, predictions = multiple_model_predictions
        
        # Generate report
        report_path = tmp_path / "test_report.txt"
        report = ModelEvaluator.generate_report(
            y_true,
            predictions,
            save_path=str(report_path)
        )
        
        # Check report content
        assert isinstance(report, str)
        assert 'MODEL EVALUATION REPORT' in report
        assert 'PERFORMANCE COMPARISON' in report
        assert 'BEST MODEL' in report
        
        # Check that all models are mentioned
        for model_name in predictions.keys():
            assert model_name in report
        
        # Check report file was saved
        assert report_path.exists()
    
    def test_generate_report_interpretation(self, multiple_model_predictions):
        """Test that report includes interpretation."""
        y_true, predictions = multiple_model_predictions
        
        report = ModelEvaluator.generate_report(y_true, predictions)
        
        # Should include interpretation section
        assert 'INTERPRETATION' in report
        assert 'explains' in report.lower()
        assert 'variance' in report.lower()


class TestPredictionIntervals:
    """Test prediction interval calculation."""
    
    def test_calculate_prediction_intervals_symmetric(self):
        """Test prediction intervals are symmetric around predictions."""
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
        
        lower, upper = calculate_prediction_intervals(y_pred, residuals, confidence=0.95)
        
        # Check bounds
        assert len(lower) == len(y_pred)
        assert len(upper) == len(y_pred)
        assert (lower < y_pred).all()
        assert (upper > y_pred).all()
        
        # Check symmetry
        margin = upper - y_pred
        assert np.allclose(margin, y_pred - lower)
    
    def test_prediction_intervals_confidence_level(self):
        """Test that higher confidence gives wider intervals."""
        y_pred = np.ones(100)
        residuals = np.random.randn(100) * 0.5
        
        lower_90, upper_90 = calculate_prediction_intervals(y_pred, residuals, confidence=0.90)
        lower_99, upper_99 = calculate_prediction_intervals(y_pred, residuals, confidence=0.99)
        
        # 99% intervals should be wider than 90%
        width_90 = (upper_90 - lower_90).mean()
        width_99 = (upper_99 - lower_99).mean()
        
        assert width_99 > width_90


class TestMetricsProperties:
    """Test mathematical properties of metrics."""
    
    def test_rmse_mae_relationship(self):
        """Test that RMSE >= MAE always holds."""
        np.random.seed(42)
        
        for _ in range(10):
            y_true = np.random.randn(100)
            y_pred = y_true + np.random.randn(100) * np.random.uniform(0.1, 1.0)
            
            metrics = ModelEvaluator.evaluate(y_true, y_pred)
            
            # RMSE should always be >= MAE
            assert metrics['rmse'] >= metrics['mae']
    
    def test_r2_bounds(self):
        """Test that R² is unbounded below but bounded at 1 above."""
        # Perfect predictions: R² = 1
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred_perfect = y_true.copy()
        
        metrics_perfect = ModelEvaluator.evaluate(y_true, y_pred_perfect)
        assert metrics_perfect['r2'] == pytest.approx(1.0)
        
        # Constant predictions (mean): R² = 0
        y_pred_mean = np.full_like(y_true, y_true.mean(), dtype=float)
        metrics_mean = ModelEvaluator.evaluate(y_true, y_pred_mean)
        assert metrics_mean['r2'] == pytest.approx(0.0, abs=1e-10)
        
        # Bad predictions: R² < 0
        y_pred_bad = -y_true  # Inverted
        metrics_bad = ModelEvaluator.evaluate(y_true, y_pred_bad)
        assert metrics_bad['r2'] < 0
    
    def test_mae_scale_invariance(self):
        """Test MAE behavior with scaled predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Constant offset
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        # MAE should be approximately 0.1
        assert metrics['mae'] == pytest.approx(0.1, abs=1e-10)
    
    def test_residuals_sum_to_zero_ols(self):
        """Test that mean residual is close to zero for unbiased predictions."""
        np.random.seed(42)
        y_true = np.random.randn(1000)
        y_pred = y_true + np.random.randn(1000) * 0.3  # Unbiased noise
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        # Mean residual should be close to zero for large sample
        assert abs(metrics['mean_residual']) < 0.1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_prediction(self):
        """Test evaluation with single prediction."""
        y_true = np.array([1.0])
        y_pred = np.array([1.1])
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        # Should compute metrics without error
        assert metrics['mae'] == 0.1
        assert metrics['rmse'] == 0.1
    
    def test_identical_predictions(self):
        """Test when all predictions are identical."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # All same (mean)
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        # R² should be 0 (predicting mean)
        assert metrics['r2'] == pytest.approx(0.0, abs=1e-10)
    
    def test_zero_variance_target(self):
        """Test when target has zero variance."""
        y_true = np.array([5, 5, 5, 5, 5])  # Constant
        y_pred = np.array([5, 5, 5, 5, 5])
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred)
        
        # Perfect predictions on constant target
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])