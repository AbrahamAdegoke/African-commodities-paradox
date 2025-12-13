"""
PCA Analysis Module

Implements Principal Component Analysis for dimensionality reduction
and visualization of African countries' economic characteristics.

Key Questions Addressed:
- What are the main dimensions that explain economic differences between countries?
- How do countries relate to each other in reduced dimensional space?
- Is there a "commodity dependence axis" vs a "governance axis"?

Author: Abraham Adegoke
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAAnalyzer:
    """
    Principal Component Analysis for understanding the main dimensions
    of economic variation across African countries.
    
    Key insight: PCA can reveal whether CDI and Governance form independent
    axes or are correlated, helping understand the commodities paradox.
    
    Example:
        >>> analyzer = PCAAnalyzer(n_components=2)
        >>> analyzer.fit(df)
        >>> analyzer.plot_biplot()
    """
    
    def __init__(
        self,
        n_components: int = 2,
        features: Optional[List[str]] = None
    ):
        """
        Initialize PCA analyzer.
        
        Args:
            n_components: Number of principal components to compute
            features: List of features to include in PCA
        """
        self.n_components = n_components
        
        # Default features for PCA
        self.features = features or [
            'cdi_smooth',
            'gdp_volatility',
            'governance_index',
            'inflation',
            'investment',
            'trade_openness'
        ]
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        
        self.country_data = None
        self.X_scaled = None
        self.X_pca = None
        self.loadings = None
        self.explained_variance = None
        
        logger.info(f"PCA Analyzer initialized")
        logger.info(f"  Components: {n_components}")
        logger.info(f"  Features: {self.features}")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for PCA by aggregating to country level.
        
        Args:
            df: Raw data with country-year observations
            
        Returns:
            DataFrame with one row per country
        """
        logger.info("Preparing data for PCA...")
        
        # Filter to available features
        available_features = [f for f in self.features if f in df.columns]
        
        if len(available_features) < 2:
            raise ValueError(f"Not enough features. Found: {available_features}")
        
        self.features = available_features
        logger.info(f"  Using features: {self.features}")
        
        # Aggregate to country level
        agg_dict = {f: 'mean' for f in self.features}
        
        if 'country_name' in df.columns:
            country_data = df.groupby('country').agg({
                **agg_dict,
                'country_name': 'first'
            }).reset_index()
        else:
            country_data = df.groupby('country').agg(agg_dict).reset_index()
        
        # Drop rows with missing values
        country_data = country_data.dropna(subset=self.features)
        
        logger.info(f"  Countries: {len(country_data)}")
        
        self.country_data = country_data
        return country_data
    
    def fit(self, df: pd.DataFrame) -> 'PCAAnalyzer':
        """
        Fit PCA on the data.
        
        Args:
            df: Raw data
            
        Returns:
            self
        """
        logger.info("=" * 70)
        logger.info("PRINCIPAL COMPONENT ANALYSIS")
        logger.info("=" * 70)
        
        # Prepare data
        country_data = self.prepare_data(df)
        
        # Scale features
        X = country_data[self.features].values
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        # Store results
        self.explained_variance = self.pca.explained_variance_ratio_
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.features
        )
        
        # Add PCA coordinates to country data
        for i in range(self.n_components):
            self.country_data[f'PC{i+1}'] = self.X_pca[:, i]
        
        logger.info(f"✓ PCA fitted successfully!")
        logger.info(f"\nExplained Variance Ratio:")
        for i, var in enumerate(self.explained_variance):
            logger.info(f"  PC{i+1}: {var:.1%}")
        logger.info(f"  Total: {sum(self.explained_variance):.1%}")
        
        logger.info(f"\nFeature Loadings (contribution to each PC):")
        logger.info(f"\n{self.loadings.round(3).to_string()}")
        
        return self
    
    def get_component_interpretation(self) -> Dict[str, Dict]:
        """
        Interpret what each principal component represents.
        
        Returns:
            Dictionary with interpretation for each component
        """
        if self.loadings is None:
            raise ValueError("Must fit PCA first")
        
        interpretations = {}
        
        for i in range(self.n_components):
            pc_name = f'PC{i+1}'
            loadings = self.loadings[pc_name]
            
            # Get top positive and negative contributors
            sorted_loadings = loadings.sort_values()
            
            top_positive = sorted_loadings.tail(2)
            top_negative = sorted_loadings.head(2)
            
            # Generate interpretation
            pos_features = [f"{feat} ({val:.2f})" for feat, val in top_positive.items() if val > 0.3]
            neg_features = [f"{feat} ({val:.2f})" for feat, val in top_negative.items() if val < -0.3]
            
            interpretations[pc_name] = {
                'variance_explained': self.explained_variance[i],
                'top_positive_loadings': dict(top_positive),
                'top_negative_loadings': dict(top_negative),
                'positive_features': pos_features,
                'negative_features': neg_features,
                'interpretation': self._generate_pc_interpretation(loadings)
            }
        
        return interpretations
    
    def _generate_pc_interpretation(self, loadings: pd.Series) -> str:
        """Generate human-readable interpretation of a component."""
        # Find dominant features
        abs_loadings = loadings.abs().sort_values(ascending=False)
        top_features = abs_loadings.head(3).index.tolist()
        
        interpretations = {
            'cdi_smooth': 'commodity dependence',
            'cdi_raw': 'commodity dependence',
            'gdp_volatility': 'economic instability',
            'log_gdp_volatility': 'economic instability',
            'governance_index': 'governance quality',
            'inflation': 'price stability',
            'investment': 'capital formation',
            'trade_openness': 'trade integration'
        }
        
        # Build interpretation
        parts = []
        for feat in top_features[:2]:
            if feat in interpretations:
                direction = "high" if loadings[feat] > 0 else "low"
                parts.append(f"{direction} {interpretations[feat]}")
        
        if parts:
            return f"This component captures {' and '.join(parts)}"
        return "Mixed characteristics"
    
    def plot_scree(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot scree plot showing explained variance by component.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.explained_variance is None:
            raise ValueError("Must fit PCA first")
        
        # Get full explained variance for all possible components
        pca_full = PCA().fit(self.X_scaled)
        cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        n_show = min(10, len(pca_full.explained_variance_ratio_))
        axes[0].bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show], 
                   color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[0].set_title('Scree Plot', fontsize=14)
        axes[0].set_xticks(range(1, n_show + 1))
        
        # Cumulative variance
        axes[1].plot(range(1, n_show + 1), cumulative_var[:n_show], 'bo-', linewidth=2, markersize=8)
        axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1].set_title('Cumulative Explained Variance', fontsize=14)
        axes[1].set_xticks(range(1, n_show + 1))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Scree plot saved to {save_path}")
        
        return fig
    
    def plot_biplot(
        self,
        pc_x: int = 1,
        pc_y: int = 2,
        color_by: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a biplot showing countries and feature vectors.
        
        Args:
            pc_x: Principal component for x-axis (1-indexed)
            pc_y: Principal component for y-axis (1-indexed)
            color_by: Column to use for coloring points
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.X_pca is None:
            raise ValueError("Must fit PCA first")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get data
        x_idx, y_idx = pc_x - 1, pc_y - 1
        xs = self.X_pca[:, x_idx]
        ys = self.X_pca[:, y_idx]
        
        # Color by cluster or feature if specified
        if color_by and color_by in self.country_data.columns:
            colors = self.country_data[color_by]
            scatter = ax.scatter(xs, ys, c=colors, cmap='RdYlGn', s=150, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(xs, ys, c='steelblue', s=150, alpha=0.7)
        
        # Add country labels
        labels = self.country_data['country_name'] if 'country_name' in self.country_data.columns else self.country_data['country']
        for i, label in enumerate(labels):
            if len(str(label)) > 12:
                label = self.country_data['country'].iloc[i]
            ax.annotate(
                label,
                (xs[i], ys[i]),
                fontsize=9,
                alpha=0.8,
                ha='center',
                va='bottom'
            )
        
        # Add feature vectors (loadings)
        scale = max(abs(xs).max(), abs(ys).max()) * 0.8
        
        for i, feature in enumerate(self.features):
            ax.arrow(
                0, 0,
                self.loadings.iloc[i, x_idx] * scale,
                self.loadings.iloc[i, y_idx] * scale,
                head_width=0.05 * scale,
                head_length=0.03 * scale,
                fc='red',
                ec='red',
                alpha=0.8
            )
            ax.text(
                self.loadings.iloc[i, x_idx] * scale * 1.1,
                self.loadings.iloc[i, y_idx] * scale * 1.1,
                feature.replace('_', ' ').title(),
                fontsize=10,
                color='red',
                fontweight='bold'
            )
        
        # Labels
        var_x = self.explained_variance[x_idx]
        var_y = self.explained_variance[y_idx]
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1%} variance)', fontsize=12)
        ax.set_title('PCA Biplot: African Countries\n(Red arrows = feature directions)', fontsize=14)
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Biplot saved to {save_path}")
        
        return fig
    
    def plot_loadings_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of feature loadings on each component.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.loadings is None:
            raise ValueError("Must fit PCA first")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            self.loadings,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Loading'}
        )
        
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('PCA Feature Loadings\n(How each feature contributes to each component)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Loadings heatmap saved to {save_path}")
        
        return fig
    
    def plot_2d_projection(
        self,
        cluster_labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot countries in 2D PCA space, optionally colored by cluster.
        
        Args:
            cluster_labels: Optional cluster labels for coloring
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.X_pca is None:
            raise ValueError("Must fit PCA first")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if cluster_labels is not None:
            scatter = ax.scatter(
                self.X_pca[:, 0],
                self.X_pca[:, 1],
                c=cluster_labels,
                cmap='Set1',
                s=150,
                alpha=0.7
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(
                self.X_pca[:, 0],
                self.X_pca[:, 1],
                c='steelblue',
                s=150,
                alpha=0.7
            )
        
        # Add labels
        labels = self.country_data['country_name'] if 'country_name' in self.country_data.columns else self.country_data['country']
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (self.X_pca[i, 0], self.X_pca[i, 1]),
                fontsize=8,
                alpha=0.8
            )
        
        var1 = self.explained_variance[0]
        var2 = self.explained_variance[1]
        ax.set_xlabel(f'PC1 ({var1:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var2:.1%} variance)', fontsize=12)
        ax.set_title('African Countries in PCA Space', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 2D projection saved to {save_path}")
        
        return fig
    
    def get_pca_summary_report(self) -> str:
        """Generate a text summary of PCA analysis."""
        if self.loadings is None:
            return "PCA not fitted yet."
        
        report = "\n" + "=" * 80 + "\n"
        report += "PCA ANALYSIS REPORT - AFRICAN COMMODITIES PARADOX\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Features analyzed: {len(self.features)}\n"
        report += f"Countries: {len(self.country_data)}\n"
        report += f"Components extracted: {self.n_components}\n\n"
        
        report += "-" * 80 + "\n"
        report += "EXPLAINED VARIANCE\n"
        report += "-" * 80 + "\n"
        
        total_var = 0
        for i, var in enumerate(self.explained_variance):
            total_var += var
            report += f"PC{i+1}: {var:.1%} (cumulative: {total_var:.1%})\n"
        
        report += f"\nTotal variance explained by {self.n_components} components: {total_var:.1%}\n\n"
        
        # Component interpretations
        interpretations = self.get_component_interpretation()
        
        report += "-" * 80 + "\n"
        report += "COMPONENT INTERPRETATIONS\n"
        report += "-" * 80 + "\n"
        
        for pc_name, interp in interpretations.items():
            report += f"\n{pc_name} ({interp['variance_explained']:.1%} variance):\n"
            report += f"  {interp['interpretation']}\n"
            
            if interp['positive_features']:
                report += f"  High: {', '.join(interp['positive_features'])}\n"
            if interp['negative_features']:
                report += f"  Low: {', '.join(interp['negative_features'])}\n"
        
        report += "\n" + "-" * 80 + "\n"
        report += "FEATURE LOADINGS\n"
        report += "-" * 80 + "\n"
        report += f"\n{self.loadings.round(3).to_string()}\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += "KEY INSIGHTS\n"
        report += "-" * 80 + "\n"
        
        # Check for interesting patterns
        if 'cdi_smooth' in self.features and 'governance_index' in self.features:
            cdi_pc1 = self.loadings.loc['cdi_smooth', 'PC1'] if 'cdi_smooth' in self.loadings.index else 0
            gov_pc1 = self.loadings.loc['governance_index', 'PC1'] if 'governance_index' in self.loadings.index else 0
            
            if np.sign(cdi_pc1) != np.sign(gov_pc1):
                report += "\n• CDI and Governance load OPPOSITELY on PC1\n"
                report += "  → This suggests an inverse relationship between commodity\n"
                report += "    dependence and governance quality!\n"
            else:
                report += "\n• CDI and Governance load in the SAME direction on PC1\n"
                report += "  → This may indicate that both improve/decline together.\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report


# Convenience function
def analyze_with_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Convenience function to run complete PCA analysis.
    
    Args:
        df: Raw data
        n_components: Number of components
        save_dir: Directory to save outputs
        
    Returns:
        Dictionary with analyzer and results
    """
    from pathlib import Path
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = PCAAnalyzer(n_components=n_components)
    analyzer.fit(df)
    
    # Generate plots
    if save_dir:
        analyzer.plot_scree(save_path=f'{save_dir}/pca_scree.png')
        analyzer.plot_biplot(save_path=f'{save_dir}/pca_biplot.png')
        analyzer.plot_loadings_heatmap(save_path=f'{save_dir}/pca_loadings.png')
    
    # Generate report
    report = analyzer.get_pca_summary_report()
    print(report)
    
    if save_dir:
        with open(f'{save_dir}/pca_report.txt', 'w') as f:
            f.write(report)
    
    return {
        'analyzer': analyzer,
        'loadings': analyzer.loadings,
        'explained_variance': analyzer.explained_variance,
        'country_pca': analyzer.country_data,
        'report': report
    }


# Example usage
if __name__ == "__main__":
    print("Testing PCA Module...")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 50
    
    df = pd.DataFrame({
        'country': [f'C{i:02d}' for i in range(n)],
        'country_name': [f'Country {i}' for i in range(n)],
        'year': np.repeat(2020, n),
        'cdi_smooth': np.random.uniform(10, 90, n),
        'gdp_volatility': np.random.uniform(0.5, 4, n),
        'governance_index': np.random.uniform(-2, 1, n),
        'inflation': np.random.uniform(2, 20, n),
        'investment': np.random.uniform(10, 40, n),
        'trade_openness': np.random.uniform(30, 100, n)
    })
    
    # Run analysis
    results = analyze_with_pca(df, n_components=3, save_dir='test_pca')
    
    print("\n✓ PCA test completed!")
