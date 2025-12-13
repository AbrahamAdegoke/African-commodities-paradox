"""
Clustering Analysis Module

Implements unsupervised learning techniques to discover patterns
and group African countries based on economic characteristics.

Techniques:
- k-Means Clustering: Partition countries into distinct groups
- Hierarchical Clustering: Build a tree of country similarities

Author: Abraham Adegoke
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CountryClusterAnalyzer:
    """
    Analyzes African countries using clustering techniques to discover
    patterns in commodity dependence and economic volatility.
    
    This addresses a key research question:
    "Are there different TYPES of countries affected differently by 
    the commodities paradox?"
    
    Example:
        >>> analyzer = CountryClusterAnalyzer(n_clusters=3)
        >>> analyzer.fit(df)
        >>> clusters = analyzer.get_cluster_profiles()
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize the cluster analyzer.
        
        Args:
            n_clusters: Number of clusters for k-Means
            features: List of features to use for clustering
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Default features for clustering
        self.features = features or [
            'cdi_smooth',
            'gdp_volatility', 
            'governance_index',
            'inflation',
            'investment',
            'trade_openness'
        ]
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.hierarchical = None
        self.linkage_matrix = None
        
        self.df_clustered = None
        self.cluster_profiles = None
        self.country_data = None
        
        logger.info(f"CountryClusterAnalyzer initialized")
        logger.info(f"  Clusters: {n_clusters}")
        logger.info(f"  Features: {self.features}")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for clustering by aggregating to country level.
        
        Args:
            df: Raw data with country-year observations
            
        Returns:
            DataFrame with one row per country, averaged features
        """
        logger.info("Preparing data for clustering...")
        
        # Filter to available features
        available_features = [f for f in self.features if f in df.columns]
        
        if len(available_features) < 3:
            raise ValueError(f"Not enough features available. Found: {available_features}")
        
        self.features = available_features
        logger.info(f"  Using features: {self.features}")
        
        # Aggregate to country level (mean of all years)
        agg_dict = {f: 'mean' for f in self.features}
        
        # Add country name if available
        if 'country_name' in df.columns:
            country_data = df.groupby('country').agg({
                **agg_dict,
                'country_name': 'first'
            }).reset_index()
        else:
            country_data = df.groupby('country').agg(agg_dict).reset_index()
        
        # Drop rows with missing values
        initial_count = len(country_data)
        country_data = country_data.dropna(subset=self.features)
        final_count = len(country_data)
        
        logger.info(f"  Countries: {final_count} (dropped {initial_count - final_count} with missing data)")
        
        self.country_data = country_data
        return country_data
    
    def fit_kmeans(self, df: pd.DataFrame) -> 'CountryClusterAnalyzer':
        """
        Fit k-Means clustering on country data.
        
        Args:
            df: Raw data (will be aggregated to country level)
            
        Returns:
            self
        """
        logger.info("=" * 70)
        logger.info("K-MEANS CLUSTERING")
        logger.info("=" * 70)
        
        # Prepare data
        country_data = self.prepare_data(df)
        
        # Scale features
        X = country_data[self.features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit k-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        country_data['cluster'] = clusters
        self.df_clustered = country_data
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        calinski = calinski_harabasz_score(X_scaled, clusters)
        
        logger.info(f"✓ k-Means fitted successfully!")
        logger.info(f"  Silhouette Score: {silhouette:.3f} (higher is better, max=1)")
        logger.info(f"  Calinski-Harabasz Score: {calinski:.1f} (higher is better)")
        
        # Generate cluster profiles
        self._generate_cluster_profiles()
        
        return self
    
    def fit_hierarchical(self, df: pd.DataFrame, method: str = 'ward') -> 'CountryClusterAnalyzer':
        """
        Fit Hierarchical (Agglomerative) Clustering.
        
        Args:
            df: Raw data (will be aggregated to country level)
            method: Linkage method ('ward', 'complete', 'average', 'single')
            
        Returns:
            self
        """
        logger.info("=" * 70)
        logger.info("HIERARCHICAL CLUSTERING")
        logger.info("=" * 70)
        
        # Prepare data if not already done
        if self.country_data is None:
            self.prepare_data(df)
        
        # Scale features
        X = self.country_data[self.features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute linkage matrix for dendrogram
        self.linkage_matrix = linkage(X_scaled, method=method)
        
        # Fit Agglomerative Clustering
        self.hierarchical = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=method
        )
        clusters = self.hierarchical.fit_predict(X_scaled)
        
        # Add hierarchical cluster labels
        self.country_data['cluster_hierarchical'] = clusters
        
        logger.info(f"✓ Hierarchical Clustering fitted successfully!")
        logger.info(f"  Method: {method}")
        logger.info(f"  Clusters: {self.n_clusters}")
        
        return self
    
    def _generate_cluster_profiles(self):
        """Generate descriptive profiles for each cluster."""
        if self.df_clustered is None:
            return
        
        profiles = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.df_clustered[self.df_clustered['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'n_countries': len(cluster_data),
                'countries': cluster_data['country'].tolist()
            }
            
            # Add mean values for each feature
            for feature in self.features:
                profile[f'avg_{feature}'] = cluster_data[feature].mean()
            
            # Add country names if available
            if 'country_name' in cluster_data.columns:
                profile['country_names'] = cluster_data['country_name'].tolist()
            
            profiles.append(profile)
        
        self.cluster_profiles = profiles
        
        # Assign descriptive names based on characteristics
        self._assign_cluster_names()
    
    def _assign_cluster_names(self):
        """Assign descriptive names to clusters based on their characteristics."""
        if not self.cluster_profiles:
            return
        
        for profile in self.cluster_profiles:
            cdi = profile.get('avg_cdi_smooth', profile.get('avg_cdi_raw', 50))
            vol = profile.get('avg_gdp_volatility', profile.get('avg_log_gdp_volatility', 0))
            gov = profile.get('avg_governance_index', 0)
            
            # Assign name based on characteristics
            if cdi > 60 and gov < -0.3:
                profile['cluster_name'] = "🔴 Resource Curse"
                profile['description'] = "High commodity dependence + Weak governance = High volatility"
            elif cdi > 60 and gov >= -0.3:
                profile['cluster_name'] = "🟡 Managed Resource Wealth"
                profile['description'] = "High commodity dependence but better governance"
            elif cdi <= 60 and vol < 2:
                profile['cluster_name'] = "🟢 Diversified & Stable"
                profile['description'] = "Lower commodity dependence, more stable growth"
            else:
                profile['cluster_name'] = "🟠 Transitional"
                profile['description'] = "Mixed characteristics"
    
    def find_optimal_clusters(
        self, 
        df: pd.DataFrame, 
        max_clusters: int = 10
    ) -> Dict[str, List[float]]:
        """
        Find optimal number of clusters using elbow method and silhouette scores.
        
        Args:
            df: Raw data
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Dictionary with inertia and silhouette scores for each k
        """
        logger.info("Finding optimal number of clusters...")
        
        # Prepare data
        if self.country_data is None:
            self.prepare_data(df)
        
        X = self.country_data[self.features].values
        X_scaled = self.scaler.fit_transform(X)
        
        results = {
            'k': [],
            'inertia': [],
            'silhouette': []
        }
        
        for k in range(2, min(max_clusters + 1, len(X_scaled))):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            results['k'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette'].append(silhouette_score(X_scaled, labels))
        
        # Find optimal k (highest silhouette)
        optimal_k = results['k'][np.argmax(results['silhouette'])]
        logger.info(f"  Optimal k (by silhouette): {optimal_k}")
        
        return results
    
    def get_cluster_profiles(self) -> List[Dict]:
        """Return cluster profiles with statistics and country lists."""
        return self.cluster_profiles
    
    def get_country_cluster(self, country_code: str) -> Optional[Dict]:
        """
        Get cluster information for a specific country.
        
        Args:
            country_code: ISO3 country code
            
        Returns:
            Dictionary with cluster info or None if not found
        """
        if self.df_clustered is None:
            return None
        
        country_row = self.df_clustered[self.df_clustered['country'] == country_code]
        
        if len(country_row) == 0:
            return None
        
        cluster_id = country_row['cluster'].values[0]
        profile = self.cluster_profiles[cluster_id]
        
        return {
            'country': country_code,
            'cluster_id': cluster_id,
            'cluster_name': profile.get('cluster_name', f'Cluster {cluster_id}'),
            'cluster_description': profile.get('description', ''),
            'similar_countries': [c for c in profile['countries'] if c != country_code]
        }
    
    def plot_clusters(
        self, 
        x_feature: str = 'cdi_smooth',
        y_feature: str = 'gdp_volatility',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot clusters in 2D feature space.
        
        Args:
            x_feature: Feature for x-axis
            y_feature: Feature for y-axis
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.df_clustered is None:
            raise ValueError("Must fit clustering first")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use available features
        if x_feature not in self.df_clustered.columns:
            x_feature = self.features[0]
        if y_feature not in self.df_clustered.columns:
            y_feature = self.features[1] if len(self.features) > 1 else self.features[0]
        
        # Plot each cluster
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.df_clustered[self.df_clustered['cluster'] == cluster_id]
            profile = self.cluster_profiles[cluster_id]
            
            ax.scatter(
                cluster_data[x_feature],
                cluster_data[y_feature],
                c=colors[cluster_id % len(colors)],
                s=150,
                alpha=0.7,
                label=f"{profile.get('cluster_name', f'Cluster {cluster_id}')} (n={len(cluster_data)})"
            )
            
            # Add country labels
            for _, row in cluster_data.iterrows():
                label = row.get('country_name', row['country'])
                if len(label) > 10:
                    label = row['country']
                ax.annotate(
                    label,
                    (row[x_feature], row[y_feature]),
                    fontsize=8,
                    alpha=0.8
                )
        
        ax.set_xlabel(x_feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_feature.replace('_', ' ').title(), fontsize=12)
        ax.set_title('African Countries: Clustering by Economic Characteristics', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Cluster plot saved to {save_path}")
        
        return fig
    
    def plot_dendrogram(
        self,
        save_path: Optional[str] = None,
        truncate_mode: str = 'level',
        p: int = 5
    ) -> plt.Figure:
        """
        Plot hierarchical clustering dendrogram.
        
        Args:
            save_path: Optional path to save figure
            truncate_mode: How to truncate dendrogram ('level', 'lastp', None)
            p: Parameter for truncation
            
        Returns:
            Matplotlib figure
        """
        if self.linkage_matrix is None:
            raise ValueError("Must fit hierarchical clustering first")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get labels
        if 'country_name' in self.country_data.columns:
            labels = self.country_data['country_name'].values
        else:
            labels = self.country_data['country'].values
        
        # Plot dendrogram
        dendrogram(
            self.linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=9,
            ax=ax,
            truncate_mode=truncate_mode if len(labels) > 20 else None,
            p=p,
            color_threshold=0.7 * max(self.linkage_matrix[:, 2])
        )
        
        ax.set_xlabel('Countries', fontsize=12)
        ax.set_ylabel('Distance (Dissimilarity)', fontsize=12)
        ax.set_title('Hierarchical Clustering: African Countries Dendrogram', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Dendrogram saved to {save_path}")
        
        return fig
    
    def plot_elbow(
        self,
        df: pd.DataFrame,
        max_clusters: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot elbow curve and silhouette scores.
        
        Args:
            df: Raw data
            max_clusters: Maximum clusters to try
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        results = self.find_optimal_clusters(df, max_clusters)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(results['k'], results['inertia'], 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(results['k'], results['silhouette'], 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Mark optimal k
        optimal_idx = np.argmax(results['silhouette'])
        axes[1].axvline(x=results['k'][optimal_idx], color='r', linestyle='--', 
                       label=f'Optimal k={results["k"][optimal_idx]}')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Elbow plot saved to {save_path}")
        
        return fig
    
    def get_cluster_summary_report(self) -> str:
        """Generate a text summary report of cluster analysis."""
        if not self.cluster_profiles:
            return "No clustering performed yet."
        
        report = "\n" + "=" * 80 + "\n"
        report += "CLUSTER ANALYSIS REPORT - AFRICAN COMMODITIES PARADOX\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Number of clusters: {self.n_clusters}\n"
        report += f"Features used: {', '.join(self.features)}\n"
        report += f"Total countries analyzed: {len(self.df_clustered)}\n\n"
        
        for profile in self.cluster_profiles:
            report += "-" * 80 + "\n"
            report += f"\n{profile.get('cluster_name', f'Cluster {profile['cluster_id']}')} "
            report += f"({profile['n_countries']} countries)\n"
            report += "-" * 80 + "\n"
            report += f"Description: {profile.get('description', 'N/A')}\n\n"
            
            report += "Average Characteristics:\n"
            for feature in self.features:
                avg_val = profile.get(f'avg_{feature}', 'N/A')
                if isinstance(avg_val, float):
                    report += f"  - {feature}: {avg_val:.2f}\n"
            
            report += f"\nCountries: {', '.join(profile.get('country_names', profile['countries']))}\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += "KEY INSIGHTS:\n"
        report += "-" * 80 + "\n"
        
        # Find the "resource curse" cluster
        resource_curse = [p for p in self.cluster_profiles if 'Curse' in p.get('cluster_name', '')]
        stable = [p for p in self.cluster_profiles if 'Stable' in p.get('cluster_name', '')]
        
        if resource_curse:
            rc = resource_curse[0]
            report += f"\n1. RESOURCE CURSE CLUSTER ({rc['n_countries']} countries):\n"
            report += f"   These countries exhibit the classic commodities paradox.\n"
            report += f"   High CDI + Poor governance leads to economic instability.\n"
        
        if stable:
            st = stable[0]
            report += f"\n2. STABLE CLUSTER ({st['n_countries']} countries):\n"
            report += f"   These countries have escaped the resource curse through\n"
            report += f"   diversification and/or better governance.\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report


# Convenience function
def cluster_african_countries(
    df: pd.DataFrame,
    n_clusters: int = 3,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Convenience function to run complete clustering analysis.
    
    Args:
        df: Raw data
        n_clusters: Number of clusters
        save_dir: Directory to save outputs
        
    Returns:
        Dictionary with analyzer, profiles, and report
    """
    from pathlib import Path
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = CountryClusterAnalyzer(n_clusters=n_clusters)
    
    # Fit both clustering methods
    analyzer.fit_kmeans(df)
    analyzer.fit_hierarchical(df)
    
    # Generate plots
    if save_dir:
        analyzer.plot_clusters(save_path=f'{save_dir}/cluster_scatter.png')
        analyzer.plot_dendrogram(save_path=f'{save_dir}/dendrogram.png')
        analyzer.plot_elbow(df, save_path=f'{save_dir}/elbow_plot.png')
    
    # Generate report
    report = analyzer.get_cluster_summary_report()
    print(report)
    
    if save_dir:
        with open(f'{save_dir}/cluster_report.txt', 'w') as f:
            f.write(report)
    
    return {
        'analyzer': analyzer,
        'profiles': analyzer.get_cluster_profiles(),
        'report': report,
        'clustered_data': analyzer.df_clustered
    }


# Example usage
if __name__ == "__main__":
    print("Testing Clustering Module...")
    print("=" * 70)
    
    # Generate synthetic data for testing
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
    results = cluster_african_countries(df, n_clusters=3, save_dir='test_clustering')
    
    print("\n✓ Clustering test completed!")
