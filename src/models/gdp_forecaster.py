"""
GDP Growth Forecaster Module

Predicts future GDP growth using structural macroeconomic indicators.
This implements the stretch goal from the project proposal.

Author: Abraham Adegoke
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Tuple, List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDPForecaster:
    """
    Forecasts GDP growth using lagged macroeconomic indicators.
    
    Model: GDP_growth(t+1) = f(CDI_t, Inflation_t, Trade_t, Investment_t)
    """
    
    def __init__(
    self,
    n_estimators: int = 50,  # ← Réduit de 200 à 50
    max_depth: int = 3,      # ← Réduit de 4 à 3
    learning_rate: float = 0.1,  # ← Augmente de 0.05 à 0.1
    random_state: int = 42
):
        """
        Initialize GDP forecaster.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            random_state: Random seed for reproducibility
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            subsample=0.8,
            min_samples_split=10
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        logger.info(f"GDPForecaster initialized with {n_estimators} estimators")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for GDP forecasting.
        
        Creates target variable (GDP growth at t+1) and features (indicators at t).
        
        Args:
            df: DataFrame with columns: country, year, cdi_smooth, inflation,
                trade_openness, investment, gdp_growth
        
        Returns:
            X: Features (indicators at time t)
            y: Target (GDP growth at time t+1)
        """
        logger.info("Preparing data for GDP forecasting...")
        
        # Sort by country and year
        df = df.sort_values(['country', 'year']).copy()
        
        # Create target: next year's GDP growth
        df['gdp_growth_next'] = df.groupby('country')['gdp_growth'].shift(-1)
        
        # Features: current year indicators
        feature_cols = ['cdi_smooth', 'inflation', 'trade_openness', 'investment']
        
        # Remove rows with missing values
        df_clean = df[feature_cols + ['gdp_growth_next']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['gdp_growth_next']
        
        self.feature_names = feature_cols
        
        logger.info(f"Data prepared: {len(X)} observations, {len(feature_cols)} features")
        logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        return X, y
    
    def fit(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Train the GDP forecasting model.
        
        Args:
            X: Feature matrix
            y: Target vector (GDP growth at t+1)
            cv: Cross-validation folds
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("="*70)
        logger.info("TRAINING GDP FORECASTER")
        logger.info("="*70)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        logger.info(f"Running {cv}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=cv, scoring='r2', n_jobs=-1
        )
        
        logger.info(f"Cross-validation R² scores: {cv_scores}")
        logger.info(f"Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Train final model
        logger.info("Training final model on full dataset...")
        self.model.fit(X_scaled, y)
        
        # Training performance
        train_score = self.model.score(X_scaled, y)
        logger.info(f"Training R²: {train_score:.4f}")
        
        self.is_fitted = True
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_r2': cv_scores.mean(),
            'std_cv_r2': cv_scores.std(),
            'train_r2': train_score
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict GDP growth for next year.
        
        Args:
            X: Feature matrix with current year indicators
        
        Returns:
            Predicted GDP growth for next year
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
            confidence: Confidence level (e.g., 0.95 for 95% CI)
        
        Returns:
            predictions: Point predictions
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        # Point predictions
        predictions = self.predict(X)
        
        # Estimate prediction intervals using residual standard error
        # This is an approximation; for exact intervals, use quantile regression
        
        # Use training residuals to estimate uncertainty
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from individual trees (for variance estimation)
        tree_predictions = np.array([
            tree.predict(X_scaled)
            for tree in self.model.estimators_.ravel()
        ])
        
        # Standard deviation across trees
        std_predictions = tree_predictions.std(axis=0)
        
        # Z-score for confidence level
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence) / 2)
        
        # Confidence intervals
        margin = z_score * std_predictions
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        return predictions, lower_bound, upper_bound
    
    def predict_multi_year(
        self,
        initial_features: pd.DataFrame,
        years: int = 5,
        scenario: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Predict GDP growth for multiple years ahead.
        
        Note: Uncertainty increases with prediction horizon.
        
        Args:
            initial_features: Current year features (1 row)
            years: Number of years to predict
            scenario: Optional dict with feature adjustments
                     e.g., {'cdi_smooth': -10, 'inflation': -5}
        
        Returns:
            DataFrame with predictions for each year
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Predicting {years} years ahead...")
        
        results = []
        current_features = initial_features.copy()
        
        for year_ahead in range(1, years + 1):
            # Apply scenario adjustments
            if scenario:
                for feature, adjustment in scenario.items():
                    if feature in current_features.columns:
                        current_features[feature] += adjustment
            
            # Predict
            pred, lower, upper = self.predict_with_confidence(current_features)
            
            results.append({
                'year_ahead': year_ahead,
                'predicted_gdp_growth': pred[0],
                'lower_95': lower[0],
                'upper_95': upper[0],
                'uncertainty': upper[0] - lower[0]
            })
            
            # Update features for next iteration (simple persistence model)
            # In reality, features would evolve based on predicted GDP growth
            # This is a simplification
        
        predictions_df = pd.DataFrame(results)
        
        logger.info(f"Multi-year predictions complete")
        logger.info(f"Year 1: {predictions_df.iloc[0]['predicted_gdp_growth']:.2f}%")
        logger.info(f"Year {years}: {predictions_df.iloc[-1]['predicted_gdp_growth']:.2f}%")
        
        return predictions_df
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance for GDP forecasting.
        
        Returns:
            DataFrame with features ranked by importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance * 100
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GDPForecaster':
        """Load model from disk."""
        data = joblib.load(filepath)
        
        forecaster = cls()
        forecaster.model = data['model']
        forecaster.scaler = data['scaler']
        forecaster.feature_names = data['feature_names']
        forecaster.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return forecaster


def train_gdp_forecaster(data_path: str, output_path: str = 'results/gdp_forecaster.pkl'):
    """
    Train GDP forecaster on prepared data.
    
    Args:
        data_path: Path to features_ready.csv
        output_path: Path to save trained model
    
    Returns:
        Trained GDPForecaster instance
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize forecaster
    forecaster = GDPForecaster()
    
    # Prepare data
    X, y = forecaster.prepare_data(df)
    
    # Train
    metrics = forecaster.fit(X, y, cv=5)
    
    # Feature importance
    importance = forecaster.feature_importance()
    logger.info("\nFeature Importance for GDP Forecasting:")
    logger.info(importance.to_string())
    
    # Save model
    forecaster.save(output_path)
    
    return forecaster, metrics


# Example usage
if __name__ == "__main__":
    # Train forecaster
    forecaster, metrics = train_gdp_forecaster('data/processed/features_ready.csv')
    
    # Example prediction for Nigeria
    nigeria_features = pd.DataFrame({
        'cdi_smooth': [92.0],
        'inflation': [15.0],
        'trade_openness': [45.0],
        'investment': [20.0]
    })
    
    # Single year prediction
    pred = forecaster.predict(nigeria_features)
    print(f"\nNigeria GDP growth prediction (next year): {pred[0]:.2f}%")
    
    # With confidence interval
    pred, lower, upper = forecaster.predict_with_confidence(nigeria_features)
    print(f"95% Confidence Interval: [{lower[0]:.2f}%, {upper[0]:.2f}%]")
    
    # Multi-year prediction
    multi_year = forecaster.predict_multi_year(nigeria_features, years=5)
    print("\nMulti-year predictions:")
    print(multi_year)
    
    # Scenario: What if Nigeria reduces CDI by 10% per year?
    scenario = {'cdi_smooth': -2, 'inflation': -1}  # Gradual improvements
    multi_year_scenario = forecaster.predict_multi_year(
        nigeria_features, years=5, scenario=scenario
    )
    print("\nScenario (diversification + inflation control):")
    print(multi_year_scenario)