"""
Gradient Boosting Regressor Model

Implements Gradient Boosting for capturing non-linear relationships between
commodity dependence and GDP growth volatility.

This advanced model complements the Ridge baseline per the project proposal.

Author: Abraham Adegoke
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Optional, Tuple
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientBoostingModel:
    """
    Gradient Boosting Regressor with hyperparameter tuning.
    
    Features:
        - Captures non-linear relationships and interactions
        - Automatic hyperparameter optimization via GridSearch
        - Feature importance ranking
        - Robust to outliers and multicollinearity
    
    Attributes:
        model: Trained Gradient Boosting model
        feature_names: List of feature column names
        best_params: Optimal hyperparameters
        feature_importances: Feature importance scores
    """
    
    def __init__(
        self,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1
    ):
        """
        Initialize Gradient Boosting model.
        
        Args:
            param_grid: Hyperparameter grid for tuning (None = default)
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        if param_grid is None:
            # Default hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10],
                'subsample': [0.8, 1.0]
            }
        
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        
        # Initialize components
        self.model = None
        self.grid_search = None
        self.feature_names = None
        self.best_params = None
        self.feature_importances_ = None
        self.cv_scores = None
        
        logger.info(f"Gradient Boosting initialized with {cv}-fold CV")
        logger.info(f"Hyperparameter grid: {param_grid}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GradientBoostingModel':
        """
        Fit Gradient Boosting model with hyperparameter tuning.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        
        Returns:
            self (fitted model)
        """
        logger.info("=" * 70)
        logger.info("TRAINING GRADIENT BOOSTING MODEL")
        logger.info("=" * 70)
        
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        # Base model
        base_model = GradientBoostingRegressor(
            random_state=42,
            loss='squared_error',
            verbose=0
        )
        
        # Grid search for hyperparameter tuning
        logger.info(f"Performing GridSearchCV with {self.cv} folds...")
        logger.info("This may take a few minutes...")
        
        self.grid_search = GridSearchCV(
            base_model,
            self.param_grid,
            cv=self.cv,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        self.grid_search.fit(X, y)
        
        # Extract best model
        self.model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        self.feature_importances_ = self.model.feature_importances_
        
        # Calculate cross-validation R² scores with best params
        self.cv_scores = cross_val_score(
            self.model, X, y,
            cv=self.cv,
            scoring='r2',
            n_jobs=self.n_jobs
        )
        
        logger.info(f"✓ Model trained successfully!")
        logger.info(f"  Best parameters: {self.best_params}")
        logger.info(f"  Best CV MSE: {-self.grid_search.best_score_:.4f}")
        logger.info(f"  CV R² scores: {self.cv_scores}")
        logger.info(f"  Mean CV R²: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, sort: bool = True) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            sort: Whether to sort by importance (descending)
        
        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_,
            'importance_pct': self.feature_importances_ / self.feature_importances_.sum() * 100
        })
        
        if sort:
            importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def interpret_feature_importance(self) -> str:
        """
        Generate human-readable interpretation of feature importance.
        
        Returns:
            String with economic interpretation
        """
        importance_df = self.get_feature_importance()
        
        interpretation = "\n" + "=" * 70 + "\n"
        interpretation += "GRADIENT BOOSTING FEATURE IMPORTANCE\n"
        interpretation += "=" * 70 + "\n\n"
        
        interpretation += "Feature Importance (sorted by importance):\n"
        interpretation += "-" * 70 + "\n"
        
        for idx, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            importance_pct = row['importance_pct']
            
            interpretation += f"\n{feature}:\n"
            interpretation += f"  Importance: {importance:.6f} ({importance_pct:.2f}%)\n"
            interpretation += f"  Rank: #{idx + 1}\n"
        
        interpretation += "\n" + "=" * 70 + "\n"
        interpretation += "NOTE: Higher importance = stronger predictive power\n"
        interpretation += "GBR captures both linear and non-linear relationships\n"
        interpretation += "=" * 70 + "\n"
        
        return interpretation
    
    def get_hyperparameter_search_results(self) -> pd.DataFrame:
        """
        Get detailed results from hyperparameter search.
        
        Returns:
            DataFrame with all tested hyperparameter combinations and scores
        """
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        
        # Sort by mean test score (best first)
        results_df = results_df.sort_values('mean_test_score', ascending=False)
        
        # Select relevant columns
        cols = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        return results_df[cols].head(10)
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (e.g., 'models/gbr_model.pkl')
        """
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'feature_importances': self.feature_importances_,
            'cv_scores': self.cv_scores
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GradientBoostingModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            Loaded GradientBoostingModel instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.best_params = model_data['best_params']
        instance.feature_importances_ = model_data['feature_importances']
        instance.cv_scores = model_data['cv_scores']
        
        logger.info(f"✓ Model loaded from {filepath}")
        return instance


# Example usage and testing
if __name__ == "__main__":
    print("Testing Gradient Boosting Model...")
    print("=" * 70)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=['cdi_smooth_lag1', 'inflation_lag1', 'trade_openness_lag1', 'investment_lag1']
    )
    
    # Non-linear relationship with interactions
    y = (
        0.5 * X['cdi_smooth_lag1'] + 
        0.2 * X['inflation_lag1'] - 
        0.3 * X['investment_lag1'] +
        0.1 * X['cdi_smooth_lag1'] * X['inflation_lag1'] +  # Interaction
        np.random.randn(n_samples) * 0.5
    )
    
    # Train model with smaller grid for testing
    test_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1]
    }
    
    model = GradientBoostingModel(param_grid=test_grid, cv=3)
    model.fit(X, y)
    
    # Get feature importance
    print("\n" + model.interpret_feature_importance())
    
    # Make predictions
    predictions = model.predict(X[:10])
    print(f"\nSample predictions: {predictions}")
    
    # Hyperparameter search results
    print("\nTop 5 hyperparameter combinations:")
    print(model.get_hyperparameter_search_results())
    
    # Save and load
    model.save('test_gbr_model.pkl')
    loaded_model = GradientBoostingModel.load('test_gbr_model.pkl')
    
    print("\n✓ Gradient Boosting test completed successfully!")