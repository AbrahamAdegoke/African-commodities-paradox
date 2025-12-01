"""
Model Training Script

Trains Ridge Regression and Gradient Boosting models on the processed data,
evaluates performance, and saves results.

This is the main CLI script for the modeling phase.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --data-path data/processed/features_ready.csv
    python scripts/train_models.py --test-size 0.3 --cv-folds 10

Author: Abraham Adegoke
Date: November 2025
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

# Import custom modules (will be created)
# For now, we'll use placeholder imports
try:
    from preprocessing.preprocessing import load_and_preprocess
    from models.ridge_regression import RidgeRegressionModel
    from models.gradient_boosting import GradientBoostingModel
    from evaluation.metrics import ModelEvaluator
except ImportError:
    logger.warning("Some modules not found, using fallbacks")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print training banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          AFRICAN COMMODITIES PARADOX - MODEL TRAINING           ║
    ║                                                                  ║
    ║     Ridge Regression + Gradient Boosting Regressor              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_data(data_path: str) -> tuple:
    """
    Load and preprocess data for modeling.
    
    Args:
        data_path: Path to processed features CSV
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Load using preprocessing module
    try:
        X, y, features = load_and_preprocess(data_path, strategy='impute')
    except:
        # Fallback: manual loading
        df = pd.read_csv(data_path)
        
        features = [
            'cdi_smooth_lag1',
            'inflation_lag1',
            'trade_openness_lag1',
            'investment_lag1'
        ]
        
        df_model = df[features + ['log_gdp_volatility']].dropna()
        X = df_model[features]
        y = df_model['log_gdp_volatility']
    
    logger.info(f"✓ Data loaded: {X.shape[0]} observations, {X.shape[1]} features")
    
    return X, y, features


def train_ridge_model(X_train, y_train, X_test, y_test, output_dir):
    """Train Ridge Regression model."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING RIDGE REGRESSION")
    logger.info("=" * 70)
    
    # Train model
    ridge_model = RidgeRegressionModel(cv=5, scale_features=True)
    ridge_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = ridge_model.predict(X_train)
    y_test_pred = ridge_model.predict(X_test)
    
    # Evaluate
    logger.info("\nTrain Set Performance:")
    train_metrics = ModelEvaluator.evaluate(y_train, y_train_pred, "Ridge (Train)")
    
    logger.info("\nTest Set Performance:")
    test_metrics = ModelEvaluator.evaluate(y_test, y_test_pred, "Ridge (Test)")
    
    # Coefficients
    logger.info(ridge_model.interpret_coefficients())
    
    # Save model
    model_path = Path(output_dir) / 'ridge_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ridge_model.save(str(model_path))
    
    return ridge_model, y_test_pred, test_metrics


def train_gbr_model(X_train, y_train, X_test, y_test, output_dir):
    """Train Gradient Boosting model."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING GRADIENT BOOSTING REGRESSOR")
    logger.info("=" * 70)
    
    # Reduced parameter grid for faster training
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [10, 20],
        'subsample': [0.8, 1.0]
    }
    
    # Train model
    gbr_model = GradientBoostingModel(param_grid=param_grid, cv=5, n_jobs=-1)
    gbr_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = gbr_model.predict(X_train)
    y_test_pred = gbr_model.predict(X_test)
    
    # Evaluate
    logger.info("\nTrain Set Performance:")
    train_metrics = ModelEvaluator.evaluate(y_train, y_train_pred, "GBR (Train)")
    
    logger.info("\nTest Set Performance:")
    test_metrics = ModelEvaluator.evaluate(y_test, y_test_pred, "GBR (Test)")
    
    # Feature importance
    logger.info(gbr_model.interpret_feature_importance())
    
    # Save model
    model_path = Path(output_dir) / 'gbr_model.pkl'
    gbr_model.save(str(model_path))
    
    return gbr_model, y_test_pred, test_metrics


def create_visualizations(
    y_test, 
    predictions_dict,
    feature_importance_dict,
    output_dir
):
    """Create and save evaluation visualizations."""
    logger.info("\nCreating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    
    # 1. Predictions vs Actual (both models)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        axes[idx].scatter(y_test, y_pred, alpha=0.5)
        axes[idx].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 
                       'r--', lw=2)
        axes[idx].set_xlabel('Actual Log GDP Volatility')
        axes[idx].set_ylabel('Predicted Log GDP Volatility')
        axes[idx].set_title(f'{model_name}: Predictions vs Actual')
        
        # Add R² to plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        axes[idx].text(0.05, 0.95, f'R² = {r2:.4f}', 
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: predictions_vs_actual.png")
    plt.close()
    
    # 2. Feature importance comparison
    if feature_importance_dict:
        fig, axes = plt.subplots(1, len(feature_importance_dict), 
                                figsize=(7*len(feature_importance_dict), 5))
        
        if len(feature_importance_dict) == 1:
            axes = [axes]
        
        for idx, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
            importance_df = importance_df.sort_values('importance', ascending=True)
            axes[idx].barh(importance_df['feature'], importance_df['importance'])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name} Feature Importance')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: feature_importance.png")
        plt.close()
    
    # 3. Residuals plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        residuals = y_test - y_pred
        axes[idx].scatter(y_pred, residuals, alpha=0.5)
        axes[idx].axhline(y=0, color='r', linestyle='--')
        axes[idx].set_xlabel('Predicted Values')
        axes[idx].set_ylabel('Residuals')
        axes[idx].set_title(f'{model_name}: Residual Plot')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals_plot.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: residuals_plot.png")
    plt.close()


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train Ridge and GBR models for African Commodities Paradox'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/features_ready.csv',
        help='Path to processed features CSV'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for models and results'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration:")
    logger.info(f"  Data: {args.data_path}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info(f"  CV folds: {args.cv_folds}")
    logger.info(f"  Random seed: {args.random_seed}")
    
    # Load data
    X, y, features = load_data(args.data_path)
    
    # Train/test split
    logger.info(f"\nSplitting data: {int((1-args.test_size)*100)}% train, {int(args.test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size,
        random_state=args.random_seed
    )
    
    logger.info(f"  Train set: {X_train.shape[0]} observations")
    logger.info(f"  Test set: {X_test.shape[0]} observations")
    
    # Train models
    ridge_model, ridge_pred, ridge_metrics = train_ridge_model(
        X_train, y_train, X_test, y_test, args.output_dir
    )
    
    gbr_model, gbr_pred, gbr_metrics = train_gbr_model(
        X_train, y_train, X_test, y_test, args.output_dir
    )
    
    # Generate comparison report
    predictions = {
        'Ridge Regression': ridge_pred,
        'Gradient Boosting': gbr_pred
    }
    
    report = ModelEvaluator.generate_report(
        y_test,
        predictions,
        save_path=f'{args.output_dir}/evaluation_report.txt'
    )
    
    print(report)
    
    # Create visualizations
    feature_importance = {
        'Gradient Boosting': gbr_model.get_feature_importance()
    }
    
    create_visualizations(
        y_test,
        predictions,
        feature_importance,
        f'{args.output_dir}/figures'
    )
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"\n📁 Output files saved in: {args.output_dir}/")
    logger.info(f"   - Models: ridge_model.pkl, gbr_model.pkl")
    logger.info(f"   - Report: evaluation_report.txt")
    logger.info(f"   - Figures: figures/")
    logger.info("\n💡 Next steps:")
    logger.info("   1. Review evaluation report")
    logger.info("   2. Analyze feature importance")
    logger.info("   3. Write technical report with insights")


if __name__ == "__main__":
    main()