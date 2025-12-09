"""
Ridge Linear Regression Model

Implements Ridge Regression (L2 regularization) for predicting GDP growth volatility
as a function of commodity dependence and macroeconomic indicators.

This serves as the baseline interpretable model per the project proposal.

Author: Abraham Adegoke
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, Optional, List
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RidgeRegressionModel:
    """
    Ridge Regression model with automatic hyperparameter tuning.
    
    Features:
        - L2 regularization to handle multicollinearity
        - Cross-validation for alpha selection
        - Standardized features for interpretability
        - Coefficient analysis for economic insights
    
    Attributes:
        model: Trained Ridge regression model
        scaler: StandardScaler for feature normalization
        feature_names: List of feature column names
        best_alpha: Optimal regularization parameter
        
    Example:
        >>> model = RidgeRegressionModel(cv=5)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> print(model.interpret_coefficients())
    """
    
    def __init__(
        self, 
        alphas: Optional[np.ndarray] = None,
        cv: int = 5,
        scale_features: bool = True
    ):
        """
        Initialize Ridge Regression model.
        
        Args:
            alphas: Array of alpha values to test (None = default range)
            cv: Number of cross-validation folds
            scale_features: Whether to standardize features
        """
        if alphas is None:
            # Default alpha range: from very weak to very strong regularization
            alphas = np.logspace(-2, 3, 50)
        
        self.alphas = alphas
        self.cv = cv
        self.scale_features = scale_features
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names = None
        self.best_alpha = None
        self.cv_scores = None
        
        logger.info(f"Ridge Regression initialized with {len(alphas)} alpha values, {cv}-fold CV")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RidgeRegressionModel':
        """
        Fit Ridge Regression model with cross-validation for alpha selection.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        
        Returns:
            self (fitted model)
        """
        logger.info("=" * 70)
        logger.info("TRAINING RIDGE REGRESSION MODEL")
        logger.info("=" * 70)
        
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        # Scale features if requested
        if self.scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Use RidgeCV for automatic alpha selection via cross-validation
        logger.info(f"Testing {len(self.alphas)} alpha values with {self.cv}-fold CV...")
        
        self.model = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
            scoring='neg_mean_squared_error'
        )
        self.model.fit(X_scaled, y)
        self.best_alpha = self.model.alpha_
        
        # Calculate cross-validation R² scores
        self.cv_scores = cross_val_score(
            Ridge(alpha=self.best_alpha),
            X_scaled, y,
            cv=self.cv,
            scoring='r2'
        )
        
        logger.info(f"✓ Model trained successfully!")
        logger.info(f"  Best alpha: {self.best_alpha:.4f}")
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
        
        if self.scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with feature names.
        
        Returns:
            DataFrame with features and their coefficients
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on absolute coefficient values.
        
        Returns:
            DataFrame with features and their importance scores (normalized)
        """
        coef_df = self.get_coefficients()
        
        # Normalize to get importance percentages
        total_abs = coef_df['abs_coefficient'].sum()
        coef_df['importance'] = coef_df['abs_coefficient'] / total_abs
        coef_df['importance_pct'] = coef_df['importance'] * 100
        
        return coef_df[['feature', 'importance', 'importance_pct']].sort_values(
            'importance', ascending=False
        )
    
    def interpret_coefficients(self) -> str:
        """
        Generate human-readable interpretation of coefficients.
        
        Returns:
            String with economic interpretation
        """
        coef_df = self.get_coefficients()
        
        interpretation = "\n" + "=" * 70 + "\n"
        interpretation += "RIDGE REGRESSION COEFFICIENTS INTERPRETATION\n"
        interpretation += "=" * 70 + "\n\n"
        interpretation += f"Intercept: {self.model.intercept_:.4f}\n"
        interpretation += f"Best Alpha (L2 penalty): {self.best_alpha:.4f}\n\n"
        
        interpretation += "Feature Importance (sorted by absolute coefficient):\n"
        interpretation += "-" * 70 + "\n"
        
        for _, row in coef_df.iterrows():
            feature = row['feature']
            coef = row['coefficient']
            direction = "increases" if coef > 0 else "decreases"
            
            interpretation += f"\n{feature}:\n"
            interpretation += f"  Coefficient: {coef:+.6f}\n"
            interpretation += f"  Interpretation: A 1-unit increase in {feature} {direction}\n"
            interpretation += f"                  GDP growth volatility by {abs(coef):.6f} (log scale)\n"
        
        interpretation += "\n" + "=" * 70 + "\n"
        interpretation += "NOTE: Coefficients are for standardized features\n"
        interpretation += "Positive coef = increases volatility (destabilizing)\n"
        interpretation += "Negative coef = decreases volatility (stabilizing)\n"
        interpretation += "=" * 70 + "\n"
        
        return interpretation
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (e.g., 'models/ridge_model.pkl')
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_alpha': self.best_alpha,
            'cv_scores': self.cv_scores,
            'scale_features': self.scale_features
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RidgeRegressionModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            Loaded RidgeRegressionModel instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(scale_features=model_data['scale_features'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.best_alpha = model_data['best_alpha']
        instance.cv_scores = model_data['cv_scores']
        
        logger.info(f"✓ Model loaded from {filepath}")
        return instance


# Example usage and testing
if __name__ == "__main__":
    print("Testing Ridge Regression Model...")
    print("=" * 70)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=['cdi_smooth_lag1', 'inflation_lag1', 'trade_openness_lag1', 'investment_lag1']
    )
    
    # True relationship: volatility increases with CDI, decreases with investment
    y = (
        0.5 * X['cdi_smooth_lag1'] + 
        0.2 * X['inflation_lag1'] - 
        0.3 * X['investment_lag1'] + 
        np.random.randn(n_samples) * 0.5
    )
    
    # Train model
    model = RidgeRegressionModel()
    model.fit(X, y)
    
    # Get coefficients
    print("\n" + model.interpret_coefficients())
    
    # Make predictions
    predictions = model.predict(X[:10])
    print(f"\nSample predictions: {predictions}")
    
    # Feature importance
    print("\nFeature Importance:")
    print(model.get_feature_importance())
    
    # Save and load
    model.save('test_ridge_model.pkl')
    loaded_model = RidgeRegressionModel.load('test_ridge_model.pkl')
    
    print("\n✓ Ridge Regression test completed successfully!")