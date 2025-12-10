"""
GDP Growth Forecasting Module (Stretch Goal)

Implements models to forecast next-year GDP growth based on current-year
macroeconomic and structural indicators.

Model: GDP_Growth(i, t+1) = f(CDI, Inflation, Governance, Openness, Investment)

Author: Abraham Adegoke
Date: December 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Optional, Tuple, List, Any
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDPGrowthForecaster:
    """
    Forecasts next-year GDP growth based on current macroeconomic indicators.
    
    This is the stretch goal implementation from the project proposal:
    Predict GDP_Growth(t+1) using indicators at time t.
    
    Features:
        - Multiple model options (Ridge, GBR, Random Forest)
        - Time-series aware cross-validation
        - Feature importance analysis
        - Confidence intervals for predictions
    
    Example:
        >>> forecaster = GDPGrowthForecaster(model_type='gbr')
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
        >>> print(forecaster.get_forecast_report())
    """
    
    def __init__(
        self,
        model_type: str = 'gbr',
        cv_folds: int = 5,
        scale_features: bool = True
    ):
        """
        Initialize GDP Growth Forecaster.
        
        Args:
            model_type: Type of model ('ridge', 'gbr', 'rf')
            cv_folds: Number of cross-validation folds
            scale_features: Whether to standardize features
        """
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.scale_features = scale_features
        
        self.model = None
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names = None
        self.cv_scores = None
        self.is_fitted = False
        
        logger.info(f"GDP Growth Forecaster initialized")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  CV folds: {cv_folds}")
    
    def _create_model(self):
        """Create the forecasting model based on model_type."""
        if self.model_type == 'ridge':
            return RidgeCV(
                alphas=np.logspace(-2, 3, 50),
                cv=self.cv_folds
            )
        elif self.model_type == 'gbr':
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=10,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GDPGrowthForecaster':
        """
        Fit the forecasting model.
        
        Args:
            X: Features at time t (n_samples, n_features)
            y: GDP growth at time t+1 (n_samples,)
            
        Returns:
            self (fitted forecaster)
        """
        logger.info("=" * 70)
        logger.info("TRAINING GDP GROWTH FORECASTER")
        logger.info("=" * 70)
        
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        # Scale features if requested
        if self.scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Cross-validation with time-series aware splits
        logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        self.cv_scores = cross_val_score(
            self._create_model(),
            X_scaled, y,
            cv=tscv,
            scoring='r2'
        )
        
        self.is_fitted = True
        
        logger.info(f"✓ Model trained successfully!")
        logger.info(f"  CV R² scores: {self.cv_scores}")
        logger.info(f"  Mean CV R²: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Forecast GDP growth for next year.
        
        Args:
            X: Current year features
            
        Returns:
            Predicted GDP growth for next year
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        
        return self.model.predict(X_scaled)
    
    def predict_with_confidence(
        self, 
        X: pd.DataFrame, 
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast with confidence intervals.
        
        Only available for ensemble models (GBR, RF).
        
        Args:
            X: Current year features
            confidence: Confidence level (default 95%)
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(X)
        
        # For ensemble models, we can estimate prediction intervals
        if hasattr(self.model, 'estimators_'):
            if self.scale_features:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values if isinstance(X, pd.DataFrame) else X
            
            # Get predictions from each tree
            if hasattr(self.model, 'staged_predict'):
                # For GBR, use staged predictions
                all_preds = np.array([
                    pred for pred in self.model.staged_predict(X_scaled)
                ])
                std = all_preds.std(axis=0)
            else:
                # For RF, use individual tree predictions
                all_preds = np.array([
                    tree.predict(X_scaled) for tree in self.model.estimators_
                ])
                std = all_preds.std(axis=0)
            
            # Calculate confidence intervals
            from scipy import stats
            z = stats.norm.ppf((1 + confidence) / 2)
            lower = predictions - z * std
            upper = predictions + z * std
            
            return predictions, lower, upper
        else:
            # For non-ensemble models, return NaN for intervals
            logger.warning("Confidence intervals only available for ensemble models")
            nan_array = np.full_like(predictions, np.nan)
            return predictions, nan_array, nan_array
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance for the forecasting model.
        
        Returns:
            DataFrame with features and their importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate forecasting performance.
        
        Args:
            X_test: Test features
            y_test: Actual GDP growth
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'mean_cv_r2': self.cv_scores.mean() if self.cv_scores is not None else np.nan,
            'std_cv_r2': self.cv_scores.std() if self.cv_scores is not None else np.nan
        }
        
        return metrics
    
    def get_forecast_report(
        self, 
        X_test: Optional[pd.DataFrame] = None, 
        y_test: Optional[pd.Series] = None
    ) -> str:
        """
        Generate comprehensive forecasting report.
        
        Args:
            X_test: Test features (optional)
            y_test: Actual values (optional)
            
        Returns:
            Report string
        """
        report = "\n" + "=" * 80 + "\n"
        report += "GDP GROWTH FORECASTING REPORT\n"
        report += "Stretch Goal: Forecast Next-Year GDP Growth\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Model Type: {self.model_type.upper()}\n"
        report += f"Features: {len(self.feature_names)}\n"
        report += f"Feature Scaling: {self.scale_features}\n\n"
        
        # Cross-validation results
        report += "CROSS-VALIDATION RESULTS:\n"
        report += "-" * 80 + "\n"
        if self.cv_scores is not None:
            report += f"CV R² Scores: {self.cv_scores}\n"
            report += f"Mean CV R²: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})\n\n"
        
        # Feature importance
        importance_df = self.get_feature_importance()
        if not importance_df.empty:
            report += "FEATURE IMPORTANCE:\n"
            report += "-" * 80 + "\n"
            report += importance_df.to_string(index=False)
            report += "\n\n"
        
        # Test set evaluation
        if X_test is not None and y_test is not None:
            metrics = self.evaluate(X_test, y_test)
            
            report += "TEST SET PERFORMANCE:\n"
            report += "-" * 80 + "\n"
            report += f"R² Score: {metrics['r2']:.4f}\n"
            report += f"RMSE: {metrics['rmse']:.4f} percentage points\n"
            report += f"MAE: {metrics['mae']:.4f} percentage points\n\n"
            
            # Interpretation
            report += "INTERPRETATION:\n"
            report += "-" * 80 + "\n"
            report += f"The model explains {metrics['r2']*100:.1f}% of variance in next-year GDP growth.\n"
            report += f"On average, forecasts are off by {metrics['mae']:.2f} percentage points.\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def save(self, filepath: str):
        """Save forecaster to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'cv_scores': self.cv_scores,
            'scale_features': self.scale_features
        }
        joblib.dump(model_data, filepath)
        logger.info(f"✓ Forecaster saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GDPGrowthForecaster':
        """Load forecaster from disk."""
        model_data = joblib.load(filepath)
        
        instance = cls(
            model_type=model_data['model_type'],
            scale_features=model_data['scale_features']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.cv_scores = model_data['cv_scores']
        instance.is_fitted = True
        
        logger.info(f"✓ Forecaster loaded from {filepath}")
        return instance


def prepare_forecast_data(
    df: pd.DataFrame,
    target_col: str = 'gdp_growth'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for GDP growth forecasting.
    
    Creates features at time t to predict GDP growth at time t+1.
    
    Args:
        df: DataFrame with country, year, and indicators
        target_col: Column name for GDP growth
        
    Returns:
        Tuple of (X_features, y_target)
    """
    logger.info("Preparing data for GDP growth forecasting...")
    
    df = df.sort_values(['country', 'year']).copy()
    
    # Create target: GDP growth at t+1
    df['gdp_growth_next'] = df.groupby('country')[target_col].shift(-1)
    
    # Feature columns (current year values)
    feature_cols = [
        'cdi_smooth',           # Commodity dependence
        'inflation',            # Current inflation
        'trade_openness',       # Trade openness
        'investment',           # Investment
        'governance_index',     # Governance quality
        'exchange_rate_volatility',  # Exchange rate volatility
        'gdp_growth'            # Current GDP growth (momentum)
    ]
    
    # Filter to available columns
    available_features = [c for c in feature_cols if c in df.columns]
    
    # Remove rows with missing target
    df_forecast = df.dropna(subset=['gdp_growth_next'])
    
    # Remove rows with missing features
    df_forecast = df_forecast.dropna(subset=available_features)
    
    X = df_forecast[available_features]
    y = df_forecast['gdp_growth_next']
    
    logger.info(f"✓ Forecast data prepared")
    logger.info(f"  Observations: {len(X)}")
    logger.info(f"  Features: {available_features}")
    
    return X, y


def run_forecast_analysis(
    df: pd.DataFrame,
    test_size: float = 0.2,
    model_type: str = 'gbr',
    save_dir: str = 'results/forecast'
) -> Dict[str, Any]:
    """
    Run complete GDP growth forecasting analysis.
    
    Args:
        df: DataFrame with all indicators
        test_size: Fraction of data for testing
        model_type: Model type ('ridge', 'gbr', 'rf')
        save_dir: Directory to save results
        
    Returns:
        Dictionary with forecaster, metrics, and report
    """
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("GDP GROWTH FORECASTING ANALYSIS (STRETCH GOAL)")
    logger.info("=" * 80)
    
    # Prepare data
    X, y = prepare_forecast_data(df)
    
    # Train/test split (chronological for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} | Test set: {len(X_test)}")
    
    # Train forecaster
    forecaster = GDPGrowthForecaster(model_type=model_type)
    forecaster.fit(X_train, y_train)
    
    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    
    # Generate report
    report = forecaster.get_forecast_report(X_test, y_test)
    print(report)
    
    # Save report
    with open(f'{save_dir}/forecast_report.txt', 'w') as f:
        f.write(report)
    
    # Save forecaster
    forecaster.save(f'{save_dir}/gdp_forecaster.pkl')
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    predictions = forecaster.predict(X_test)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual GDP Growth (%)')
    plt.ylabel('Predicted GDP Growth (%)')
    plt.title(f'GDP Growth Forecast\nR² = {metrics["r2"]:.4f}')
    
    # Plot 2: Feature Importance
    plt.subplot(1, 2, 2)
    importance_df = forecaster.get_feature_importance()
    plt.barh(importance_df['feature'], importance_df['importance_pct'])
    plt.xlabel('Importance (%)')
    plt.title('Feature Importance for GDP Forecasting')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/forecast_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Results saved to {save_dir}/")
    
    return {
        'forecaster': forecaster,
        'metrics': metrics,
        'report': report,
        'predictions': predictions,
        'X_test': X_test,
        'y_test': y_test
    }


# Example usage
if __name__ == "__main__":
    print("Testing GDP Growth Forecaster...")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'country': np.repeat(['NGA', 'ZAF', 'KEN', 'GHA', 'EGY'], n // 5),
        'year': np.tile(range(2000, 2000 + n // 5), 5),
        'cdi_smooth': np.random.uniform(20, 80, n),
        'inflation': np.random.uniform(2, 20, n),
        'trade_openness': np.random.uniform(30, 80, n),
        'investment': np.random.uniform(15, 35, n),
        'governance_index': np.random.uniform(-1, 1, n),
        'exchange_rate_volatility': np.random.uniform(1, 10, n),
        'gdp_growth': np.random.uniform(-2, 10, n)
    })
    
    # Run analysis
    results = run_forecast_analysis(df, model_type='gbr', save_dir='test_forecast')
    
    print("\n✓ GDP Growth Forecaster test completed!")
