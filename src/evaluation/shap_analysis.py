"""
SHAP Values and Model Interpretability Module

Provides SHAP-based feature importance analysis for understanding
which factors drive GDP growth volatility predictions.

Author: Abraham Adegoke
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP-based model interpretability for the African Commodities Paradox project.
    
    Provides:
        - SHAP values calculation for any sklearn model
        - Feature importance plots
        - Summary plots
        - Dependence plots
        - Force plots for individual predictions
    
    Example:
        >>> analyzer = SHAPAnalyzer(model, X_train)
        >>> shap_values = analyzer.compute_shap_values(X_test)
        >>> analyzer.plot_summary(X_test)
    """
    
    def __init__(self, model: Any, X_background: pd.DataFrame):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained sklearn model with predict method
            X_background: Background dataset for SHAP explainer (usually X_train)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.X_background = X_background
        self.feature_names = list(X_background.columns)
        self.explainer = None
        self.shap_values = None
        
        # Initialize explainer based on model type
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__
        
        logger.info(f"Initializing SHAP explainer for {model_type}...")
        
        # Use TreeExplainer for tree-based models (faster)
        if hasattr(self.model, 'estimators_') or 'Gradient' in model_type or 'Forest' in model_type:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("  ✓ Using TreeExplainer (fast)")
            except:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    shap.sample(self.X_background, 100)
                )
                logger.info("  ✓ Using KernelExplainer (slower but universal)")
        else:
            # Use KernelExplainer for other models (Ridge, etc.)
            # Sample background data for efficiency
            background = shap.sample(self.X_background, min(100, len(self.X_background)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            logger.info("  ✓ Using KernelExplainer")
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Feature matrix to explain
            
        Returns:
            Array of SHAP values (n_samples, n_features)
        """
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        logger.info(f"  ✓ SHAP values computed: shape {self.shap_values.shape}")
        
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Returns:
            DataFrame with features ranked by importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # Mean absolute SHAP value per feature
        importance = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df
    
    def plot_summary(
        self, 
        X: pd.DataFrame, 
        plot_type: str = 'bar',
        save_path: Optional[str] = None,
        max_display: int = 10
    ):
        """
        Create SHAP summary plot.
        
        Args:
            X: Feature matrix
            plot_type: 'bar' for bar plot, 'dot' for beeswarm plot
            save_path: Optional path to save figure
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, 
                X, 
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Feature Importance (Mean |SHAP value|)')
        else:
            shap.summary_plot(
                self.shap_values, 
                X, 
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Summary Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_dependence(
        self, 
        feature: str, 
        X: pd.DataFrame,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP dependence plot for a specific feature.
        
        Shows how the model output varies with a feature value.
        
        Args:
            feature: Feature name to analyze
            X: Feature matrix
            interaction_feature: Optional feature for interaction coloring
            save_path: Optional path to save figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        
        if interaction_feature:
            shap.dependence_plot(
                feature, 
                self.shap_values, 
                X,
                feature_names=self.feature_names,
                interaction_index=interaction_feature,
                show=False
            )
        else:
            shap.dependence_plot(
                feature, 
                self.shap_values, 
                X,
                feature_names=self.feature_names,
                show=False
            )
        
        plt.title(f'SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Dependence plot saved to {save_path}")
        
        plt.show()
    
    def plot_force(
        self, 
        X: pd.DataFrame, 
        index: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP force plot for a single prediction.
        
        Shows how each feature contributes to pushing the prediction
        away from the base value.
        
        Args:
            X: Feature matrix
            index: Index of sample to explain
            save_path: Optional path to save figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get expected value (base value)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = 0
        
        # Create force plot
        shap.force_plot(
            base_value,
            self.shap_values[index],
            X.iloc[index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot (Sample {index})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Force plot saved to {save_path}")
        
        plt.show()
    
    def interpret_prediction(
        self, 
        X: pd.DataFrame, 
        index: int = 0
    ) -> str:
        """
        Generate human-readable interpretation of a single prediction.
        
        Args:
            X: Feature matrix
            index: Index of sample to interpret
            
        Returns:
            String with interpretation
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        sample_shap = self.shap_values[index]
        sample_features = X.iloc[index]
        
        # Get expected value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = 0
        
        prediction = base_value + sample_shap.sum()
        
        # Sort features by absolute SHAP value
        feature_contributions = list(zip(self.feature_names, sample_shap, sample_features))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        interpretation = "\n" + "=" * 70 + "\n"
        interpretation += f"PREDICTION INTERPRETATION (Sample {index})\n"
        interpretation += "=" * 70 + "\n\n"
        interpretation += f"Base value (average prediction): {base_value:.4f}\n"
        interpretation += f"Final prediction: {prediction:.4f}\n\n"
        interpretation += "Feature contributions (sorted by importance):\n"
        interpretation += "-" * 70 + "\n"
        
        for feature, shap_val, feat_val in feature_contributions:
            direction = "↑" if shap_val > 0 else "↓"
            interpretation += f"\n{feature}:\n"
            interpretation += f"  Value: {feat_val:.4f}\n"
            interpretation += f"  SHAP contribution: {shap_val:+.4f} {direction}\n"
        
        interpretation += "\n" + "=" * 70 + "\n"
        interpretation += "Interpretation:\n"
        interpretation += "-" * 70 + "\n"
        
        # Top 3 drivers
        top_3 = feature_contributions[:3]
        interpretation += "Top 3 factors driving this prediction:\n"
        for i, (feature, shap_val, _) in enumerate(top_3, 1):
            effect = "increases" if shap_val > 0 else "decreases"
            interpretation += f"  {i}. {feature} {effect} volatility by {abs(shap_val):.4f}\n"
        
        interpretation += "=" * 70 + "\n"
        
        return interpretation
    
    def generate_shap_report(
        self, 
        X: pd.DataFrame,
        save_dir: str = 'results/shap'
    ) -> str:
        """
        Generate comprehensive SHAP analysis report.
        
        Args:
            X: Feature matrix to analyze
            save_dir: Directory to save plots
            
        Returns:
            Report string
        """
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Compute SHAP values
        self.compute_shap_values(X)
        
        # Get feature importance
        importance_df = self.get_feature_importance()
        
        # Generate report
        report = "\n" + "=" * 80 + "\n"
        report += "SHAP ANALYSIS REPORT - AFRICAN COMMODITIES PARADOX\n"
        report += "=" * 80 + "\n\n"
        
        report += "FEATURE IMPORTANCE (based on mean |SHAP value|):\n"
        report += "-" * 80 + "\n"
        report += importance_df.to_string(index=False)
        report += "\n\n"
        
        # Interpretation
        report += "KEY FINDINGS:\n"
        report += "-" * 80 + "\n"
        
        top_feature = importance_df.iloc[0]
        report += f"\n1. Most important predictor: {top_feature['feature']}\n"
        report += f"   - Contributes {top_feature['importance_pct']:.1f}% to model predictions\n"
        
        if 'cdi' in top_feature['feature'].lower():
            report += "   - This confirms commodity dependence is a key driver of volatility\n"
        
        report += "\n2. Feature ranking:\n"
        for i, row in importance_df.iterrows():
            report += f"   {i+1}. {row['feature']}: {row['importance_pct']:.1f}%\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        # Save plots
        logger.info("Generating SHAP plots...")
        
        # Summary bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, X, 
            feature_names=self.feature_names,
            plot_type='bar', 
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary dot plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, X, 
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_summary_dot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ SHAP plots saved to {save_dir}/")
        
        report += f"\nPlots saved to: {save_dir}/\n"
        report += "  - shap_importance_bar.png\n"
        report += "  - shap_summary_dot.png\n"
        report += "=" * 80 + "\n"
        
        # Save report
        with open(f'{save_dir}/shap_report.txt', 'w') as f:
            f.write(report)
        
        return report


def analyze_with_shap(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_dir: str = 'results/shap'
) -> Dict[str, Any]:
    """
    Convenience function to run complete SHAP analysis.
    
    Args:
        model: Trained model
        X_train: Training features (for background)
        X_test: Test features (to explain)
        save_dir: Directory to save results
        
    Returns:
        Dictionary with SHAP values, importance, and report
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Skipping SHAP analysis.")
        return {}
    
    analyzer = SHAPAnalyzer(model, X_train)
    
    # Compute SHAP values
    shap_values = analyzer.compute_shap_values(X_test)
    
    # Get importance
    importance_df = analyzer.get_feature_importance()
    
    # Generate report
    report = analyzer.generate_shap_report(X_test, save_dir)
    
    print(report)
    
    return {
        'shap_values': shap_values,
        'importance': importance_df,
        'report': report,
        'analyzer': analyzer
    }


# Example usage
if __name__ == "__main__":
    print("Testing SHAP Analyzer...")
    print("=" * 70)
    
    if not SHAP_AVAILABLE:
        print("⚠ SHAP not installed. Install with: pip install shap")
    else:
        # Generate synthetic data for testing
        from sklearn.ensemble import GradientBoostingRegressor
        
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'cdi_smooth_lag1': np.random.uniform(0, 100, n_samples),
            'inflation_lag1': np.random.uniform(0, 20, n_samples),
            'trade_openness_lag1': np.random.uniform(20, 100, n_samples),
            'investment_lag1': np.random.uniform(10, 40, n_samples)
        })
        
        y = 0.02 * X['cdi_smooth_lag1'] + 0.05 * X['inflation_lag1'] - 0.01 * X['investment_lag1'] + np.random.randn(n_samples) * 0.3
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # SHAP analysis
        results = analyze_with_shap(model, X, X.iloc[:50], save_dir='test_shap')
        
        print("\n✓ SHAP Analyzer test completed!")
