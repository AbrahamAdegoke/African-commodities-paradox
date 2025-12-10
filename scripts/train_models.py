"""
Model Training Script

Trains Ridge Regression and Gradient Boosting models on the processed data,
evaluates performance, generates SHAP analysis, and optionally runs the
stretch goal (GDP growth forecasting).

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --include-shap --include-forecast

Author: Abraham Adegoke
Date: December 2025
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import custom modules
from preprocessing.preprocessing import load_and_preprocess
from models.ridge_regression import RidgeRegressionModel
from models.gradient_boosting import GradientBoostingModel
from evaluation.metrics import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print training banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          AFRICAN COMMODITIES PARADOX - MODEL TRAINING           ║
    ║                                                                  ║
    ║     Ridge Regression + Gradient Boosting + SHAP Analysis        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_data(data_path: str) -> tuple:
    """Load and preprocess data for modeling."""
    logger.info(f"Loading data from: {data_path}")
    
    try:
        X, y, features = load_and_preprocess(
            data_path, 
            strategy='impute',
            include_governance=True,
            include_exchange_rate=True
        )
    except Exception as e:
        logger.warning(f"Advanced preprocessing failed: {e}")
        df = pd.read_csv(data_path)
        
        features = ['cdi_smooth_lag1', 'inflation_lag1', 'trade_openness_lag1', 'investment_lag1']
        
        if 'governance_index_lag1' in df.columns:
            features.append('governance_index_lag1')
        if 'exchange_rate_volatility_lag1' in df.columns:
            features.append('exchange_rate_volatility_lag1')
        
        features = [f for f in features if f in df.columns]
        df_model = df[features + ['log_gdp_volatility']].dropna()
        X = df_model[features]
        y = df_model['log_gdp_volatility']
    
    logger.info(f"✓ Data loaded: {X.shape[0]} observations, {X.shape[1]} features")
    return X, y, list(X.columns)


def train_ridge_model(X_train, y_train, X_test, y_test, output_dir):
    """Train Ridge Regression model."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING RIDGE REGRESSION")
    logger.info("=" * 70)
    
    ridge_model = RidgeRegressionModel(cv=5, scale_features=True)
    ridge_model.fit(X_train, y_train)
    
    y_test_pred = ridge_model.predict(X_test)
    
    logger.info("\nTest Set Performance:")
    test_metrics = ModelEvaluator.evaluate(y_test, y_test_pred, "Ridge (Test)")
    logger.info(ridge_model.interpret_coefficients())
    
    model_path = Path(output_dir) / 'models' / 'ridge_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ridge_model.save(str(model_path))
    
    return ridge_model, y_test_pred, test_metrics


def train_gbr_model(X_train, y_train, X_test, y_test, output_dir):
    """Train Gradient Boosting model."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING GRADIENT BOOSTING REGRESSOR")
    logger.info("=" * 70)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [10, 20],
        'subsample': [0.8, 1.0]
    }
    
    gbr_model = GradientBoostingModel(param_grid=param_grid, cv=5, n_jobs=-1)
    gbr_model.fit(X_train, y_train)
    
    y_test_pred = gbr_model.predict(X_test)
    
    logger.info("\nTest Set Performance:")
    test_metrics = ModelEvaluator.evaluate(y_test, y_test_pred, "GBR (Test)")
    logger.info(gbr_model.interpret_feature_importance())
    
    model_path = Path(output_dir) / 'models' / 'gbr_model.pkl'
    gbr_model.save(str(model_path))
    
    return gbr_model, y_test_pred, test_metrics


def run_shap_analysis(gbr_model, X_train, X_test, output_dir):
    """Run SHAP analysis on the Gradient Boosting model."""
    logger.info("\n" + "=" * 70)
    logger.info("SHAP ANALYSIS")
    logger.info("=" * 70)
    
    try:
        from evaluation.shap_analysis import analyze_with_shap
        results = analyze_with_shap(gbr_model.model, X_train, X_test, save_dir=f'{output_dir}/shap')
        return results
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
        return None
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        return None


def run_gdp_forecast(data_path, output_dir):
    """Run GDP growth forecasting (stretch goal)."""
    logger.info("\n" + "=" * 70)
    logger.info("STRETCH GOAL: GDP GROWTH FORECASTING")
    logger.info("=" * 70)
    
    try:
        from models.gdp_forecaster import run_forecast_analysis
        df = pd.read_csv(data_path)
        
        required = ['country', 'year', 'gdp_growth']
        if not all(col in df.columns for col in required):
            logger.warning("Missing required columns for GDP forecasting")
            return None
        
        results = run_forecast_analysis(df, test_size=0.2, model_type='gbr', save_dir=f'{output_dir}/forecast')
        return results
    except Exception as e:
        logger.warning(f"GDP forecasting failed: {e}")
        return None


def create_visualizations(y_test, predictions_dict, feature_importance_dict, output_dir):
    """Create and save evaluation visualizations."""
    logger.info("\nCreating visualizations...")
    
    output_dir = Path(output_dir) / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style('whitegrid')
    
    # Predictions vs Actual
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        axes[idx].scatter(y_test, y_pred, alpha=0.5)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[idx].set_xlabel('Actual Log GDP Volatility')
        axes[idx].set_ylabel('Predicted Log GDP Volatility')
        axes[idx].set_title(f'{model_name}: Predictions vs Actual')
        
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        axes[idx].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[idx].transAxes,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    if feature_importance_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, importance_df in feature_importance_dict.items():
            importance_df = importance_df.sort_values('importance', ascending=True)
            ax.barh(importance_df['feature'], importance_df['importance_pct'])
            ax.set_xlabel('Importance (%)')
            ax.set_title(f'{model_name} Feature Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✓ Visualizations saved to {output_dir}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train models for African Commodities Paradox')
    parser.add_argument('--data-path', type=str, default='data/processed/features_ready.csv')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--include-shap', action='store_true', help='Include SHAP analysis')
    parser.add_argument('--include-forecast', action='store_true', help='Include GDP forecasting (stretch goal)')
    
    args = parser.parse_args()
    
    print_banner()
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    X, y, features = load_data(args.data_path)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_seed)
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train models
    ridge_model, ridge_pred, _ = train_ridge_model(X_train, y_train, X_test, y_test, args.output_dir)
    gbr_model, gbr_pred, _ = train_gbr_model(X_train, y_train, X_test, y_test, args.output_dir)
    
    # Generate report
    predictions = {'Ridge Regression': ridge_pred, 'Gradient Boosting': gbr_pred}
    report = ModelEvaluator.generate_report(y_test, predictions, save_path=f'{args.output_dir}/evaluation_report.txt')
    print(report)
    
    # Visualizations
    feature_importance = {'Gradient Boosting': gbr_model.get_feature_importance()}
    create_visualizations(y_test, predictions, feature_importance, args.output_dir)
    
    # Optional: SHAP analysis
    if args.include_shap:
        run_shap_analysis(gbr_model, X_train, X_test, args.output_dir)
    
    # Optional: GDP forecasting (stretch goal)
    if args.include_forecast:
        run_gdp_forecast(args.data_path, args.output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
