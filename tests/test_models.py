"""
Unit tests for machine learning models.

Tests Ridge Regression and Gradient Boosting models.

Author: Abraham Adegoke
Date: December 2025
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.ridge_regression import RidgeRegressionModel
from models.gradient_boosting import GradientBoostingModel


@pytest.fixture
def synthetic_data():
    """Create synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'cdi_smooth_lag1': np.random.uniform(0, 100, n_samples),
        'inflation_lag1': np.random.uniform(0, 20, n_samples),
        'trade_openness_lag1': np.random.uniform(20, 100, n_samples),
        'investment_lag1': np.random.uniform(10, 40, n_samples)
    })
    
    # True relationship with some noise
    y = (
        0.01 * X['cdi_smooth_lag1'] + 
        0.02 * X['inflation_lag1'] - 
        0.01 * X['investment_lag1'] + 
        np.random.randn(n_samples) * 0.3
    )
    
    return X, y


class TestRidgeRegressionModel:
    """Test suite for Ridge Regression model."""
    
    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        model = RidgeRegressionModel()
        assert model.alphas is not None
        assert model.cv == 5
        assert model.scale_features == True
        
        # Custom alphas
        custom_alphas = np.array([0.1, 1.0, 10.0])
        model_custom = RidgeRegressionModel(alphas=custom_alphas, cv=3)
        assert len(model_custom.alphas) == 3
        assert model_custom.cv == 3
    
    def test_model_fit(self, synthetic_data):
        """Test model fitting."""
        X, y = synthetic_data
        
        model = RidgeRegressionModel(cv=3)
        model.fit(X, y)
        
        # Check that model is fitted
        assert model.model is not None
        assert model.best_alpha is not None
        assert model.cv_scores is not None
        assert len(model.cv_scores) == 3
        
        # Check feature names stored
        assert model.feature_names == list(X.columns)
    
    def test_model_predict(self, synthetic_data):
        """Test model predictions."""
        X, y = synthetic_data
        
        model = RidgeRegressionModel(cv=3)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check predictions
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
    
    def test_model_coefficients(self, synthetic_data):
        """Test coefficient extraction and interpretation."""
        X, y = synthetic_data
        
        model = RidgeRegressionModel(cv=3)
        model.fit(X, y)
        
        # Get coefficients
        coef_df = model.get_coefficients()
        
        # Check DataFrame structure
        assert isinstance(coef_df, pd.DataFrame)
        assert 'feature' in coef_df.columns
        assert 'coefficient' in coef_df.columns
        assert len(coef_df) == X.shape[1]
        
        # Check interpretation
        interpretation = model.interpret_coefficients()
        assert isinstance(interpretation, str)
        assert 'cdi_smooth_lag1' in interpretation
    
    def test_model_save_load(self, synthetic_data, tmp_path):
        """Test model saving and loading."""
        X, y = synthetic_data
        
        # Train model
        model = RidgeRegressionModel(cv=3)
        model.fit(X, y)
        
        # Save model
        model_path = tmp_path / "test_ridge.pkl"
        model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = RidgeRegressionModel.load(str(model_path))
        
        # Check loaded model
        assert loaded_model.best_alpha == model.best_alpha
        assert loaded_model.feature_names == model.feature_names
        
        # Check predictions match
        pred_original = model.predict(X[:10])
        pred_loaded = loaded_model.predict(X[:10])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)
    
    def test_model_without_scaling(self, synthetic_data):
        """Test model without feature scaling."""
        X, y = synthetic_data
        
        model = RidgeRegressionModel(scale_features=False, cv=3)
        model.fit(X, y)
        
        assert model.scaler is None
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_cross_validation_scores(self, synthetic_data):
        """Test cross-validation score distribution."""
        X, y = synthetic_data
        
        model = RidgeRegressionModel(cv=5)
        model.fit(X, y)
        
        # CV scores should be reasonable
        assert model.cv_scores.mean() > -1.0  # R² shouldn't be too negative
        assert model.cv_scores.std() < 1.0  # Scores should be consistent


class TestGradientBoostingModel:
    """Test suite for Gradient Boosting model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = GradientBoostingModel(cv=3)
        assert model.cv == 3
        assert model.n_jobs == -1
        assert model.param_grid is not None
    
    def test_model_fit_small_grid(self, synthetic_data):
        """Test model fitting with small parameter grid."""
        X, y = synthetic_data
        
        # Small grid for fast testing
        small_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3],
            'learning_rate': [0.1]
        }
        
        model = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        model.fit(X, y)
        
        # Check that model is fitted
        assert model.model is not None
        assert model.best_params is not None
        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == X.shape[1]
    
    def test_model_predict(self, synthetic_data):
        """Test model predictions."""
        X, y = synthetic_data
        
        small_grid = {
            'n_estimators': [50],
            'max_depth': [3],
            'learning_rate': [0.1]
        }
        
        model = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
    
    def test_feature_importance(self, synthetic_data):
        """Test feature importance extraction."""
        X, y = synthetic_data
        
        small_grid = {
            'n_estimators': [50],
            'max_depth': [3],
            'learning_rate': [0.1]
        }
        
        model = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        model.fit(X, y)
        
        # Get feature importance
        importance_df = model.get_feature_importance()
        
        # Check DataFrame structure
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'importance_pct' in importance_df.columns
        assert len(importance_df) == X.shape[1]
        
        # Importance should sum to ~1.0
        assert importance_df['importance'].sum() == pytest.approx(1.0, rel=0.01)
        
        # Percentages should sum to ~100
        assert importance_df['importance_pct'].sum() == pytest.approx(100.0, rel=0.01)
    
    def test_feature_importance_interpretation(self, synthetic_data):
        """Test feature importance interpretation."""
        X, y = synthetic_data
        
        small_grid = {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}
        
        model = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        model.fit(X, y)
        
        interpretation = model.interpret_feature_importance()
        
        assert isinstance(interpretation, str)
        assert 'cdi_smooth_lag1' in interpretation
        assert 'Importance' in interpretation
    
    def test_model_save_load(self, synthetic_data, tmp_path):
        """Test model saving and loading."""
        X, y = synthetic_data
        
        small_grid = {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}
        
        model = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        model.fit(X, y)
        
        # Save model
        model_path = tmp_path / "test_gbr.pkl"
        model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = GradientBoostingModel.load(str(model_path))
        
        # Check loaded model
        assert loaded_model.best_params == model.best_params
        assert loaded_model.feature_names == model.feature_names
        
        # Check predictions match
        pred_original = model.predict(X[:10])
        pred_loaded = loaded_model.predict(X[:10])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=5)
    
    def test_hyperparameter_search_results(self, synthetic_data):
        """Test hyperparameter search results."""
        X, y = synthetic_data
        
        small_grid = {
            'n_estimators': [50, 100],
            'max_depth': [2, 3],
            'learning_rate': [0.05, 0.1]
        }
        
        model = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        model.fit(X, y)
        
        # Get search results
        results_df = model.get_hyperparameter_search_results()
        
        assert isinstance(results_df, pd.DataFrame)
        assert 'params' in results_df.columns
        assert 'mean_test_score' in results_df.columns
        assert len(results_df) > 0


class TestModelComparison:
    """Test model comparison functionality."""
    
    def test_ridge_vs_gbr_performance(self, synthetic_data):
        """Test that both models can fit the data."""
        X, y = synthetic_data
        
        # Train both models
        ridge = RidgeRegressionModel(cv=3)
        ridge.fit(X, y)
        
        small_grid = {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}
        gbr = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        gbr.fit(X, y)
        
        # Get predictions
        pred_ridge = ridge.predict(X)
        pred_gbr = gbr.predict(X)
        
        # Both should have reasonable predictions
        from sklearn.metrics import r2_score
        r2_ridge = r2_score(y, pred_ridge)
        r2_gbr = r2_score(y, pred_gbr)
        
        assert r2_ridge > 0  # Should explain some variance
        assert r2_gbr > 0
        
        # GBR typically performs better on training data
        # But we're not enforcing that in tests as it depends on data


class TestModelRobustness:
    """Test model robustness to edge cases."""
    
    def test_model_with_correlated_features(self):
        """Test models with highly correlated features."""
        np.random.seed(42)
        n = 200
        
        # Create correlated features
        X = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
        })
        X['feature3'] = X['feature1'] * 0.9 + np.random.randn(n) * 0.1  # Highly correlated
        X['feature4'] = X['feature2'] * 0.8 + np.random.randn(n) * 0.2
        
        y = X['feature1'] + 0.5 * X['feature2'] + np.random.randn(n) * 0.3
        
        # Ridge should handle correlated features well
        model = RidgeRegressionModel(cv=3)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert not np.isnan(predictions).any()
    
    def test_model_with_outliers(self):
        """Test model robustness to outliers."""
        np.random.seed(42)
        n = 200
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
        })
        
        y = X['feature1'] + X['feature2'] + np.random.randn(n) * 0.3
        
        # Add outliers
        y.iloc[0] = 100
        y.iloc[1] = -100
        
        # Models should still fit without errors
        ridge = RidgeRegressionModel(cv=3)
        ridge.fit(X, y)
        
        small_grid = {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}
        gbr = GradientBoostingModel(param_grid=small_grid, cv=3, n_jobs=1)
        gbr.fit(X, y)
        
        # Should produce predictions
        assert ridge.predict(X) is not None
        assert gbr.predict(X) is not None


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])