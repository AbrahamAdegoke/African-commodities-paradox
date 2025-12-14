"""
Tests for Clustering Analysis Module

Tests k-Means clustering, hierarchical clustering, and related functionality.

Author: Abraham Adegoke
Date: December 2025
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.clustering import CountryClusterAnalyzer, cluster_african_countries


class TestCountryClusterAnalyzer:
    """Test cases for CountryClusterAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_countries = 20
        n_years = 5
        
        data = []
        for i in range(n_countries):
            for year in range(2018, 2018 + n_years):
                data.append({
                    'country': f'C{i:02d}',
                    'country_name': f'Country {i}',
                    'year': year,
                    'cdi_smooth': np.random.uniform(20, 80),
                    'gdp_volatility': np.random.uniform(0.5, 4),
                    'governance_index': np.random.uniform(-1.5, 1),
                    'inflation': np.random.uniform(2, 15),
                    'investment': np.random.uniform(15, 35),
                    'trade_openness': np.random.uniform(30, 90)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CountryClusterAnalyzer(n_clusters=3)
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = CountryClusterAnalyzer(n_clusters=4)
        
        assert analyzer.n_clusters == 4
        assert analyzer.random_state == 42
        assert len(analyzer.features) > 0
    
    def test_init_custom_features(self):
        """Test initialization with custom features."""
        custom_features = ['cdi_smooth', 'inflation']
        analyzer = CountryClusterAnalyzer(n_clusters=3, features=custom_features)
        
        assert analyzer.features == custom_features
    
    def test_prepare_data(self, analyzer, sample_data):
        """Test data preparation."""
        country_data = analyzer.prepare_data(sample_data)
        
        # Should aggregate to country level
        assert len(country_data) == 20  # 20 unique countries
        assert 'country' in country_data.columns
    
    def test_prepare_data_filters_available_features(self, sample_data):
        """Test that prepare_data filters to available features."""
        # Create analyzer with some non-existent features but enough valid ones
        analyzer = CountryClusterAnalyzer(
            features=['cdi_smooth', 'inflation', 'investment', 'nonexistent_feature']
        )
        
        country_data = analyzer.prepare_data(sample_data)
        
        # Should only use available features
        assert 'nonexistent_feature' not in analyzer.features
        assert 'cdi_smooth' in analyzer.features
        assert len(analyzer.features) >= 3
    
    def test_prepare_data_insufficient_features(self, analyzer):
        """Test error when not enough features available."""
        df = pd.DataFrame({
            'country': ['A', 'B'],
            'year': [2020, 2020],
            'only_one_feature': [1, 2]
        })
        
        with pytest.raises(ValueError, match="Not enough features"):
            analyzer.prepare_data(df)
    
    def test_fit_kmeans(self, analyzer, sample_data):
        """Test k-means fitting."""
        analyzer.fit_kmeans(sample_data)
        
        assert analyzer.kmeans is not None
        assert analyzer.df_clustered is not None
        assert 'cluster' in analyzer.df_clustered.columns
        assert len(analyzer.df_clustered['cluster'].unique()) == 3
    
    def test_fit_kmeans_cluster_range(self, analyzer, sample_data):
        """Test that cluster labels are in valid range."""
        analyzer.fit_kmeans(sample_data)
        
        clusters = analyzer.df_clustered['cluster'].unique()
        assert all(0 <= c < analyzer.n_clusters for c in clusters)
    
    def test_fit_hierarchical(self, analyzer, sample_data):
        """Test hierarchical clustering fitting."""
        analyzer.fit_kmeans(sample_data)  # Need to prepare data first
        analyzer.fit_hierarchical(sample_data)
        
        assert analyzer.hierarchical is not None
        assert analyzer.linkage_matrix is not None
        assert 'cluster_hierarchical' in analyzer.country_data.columns
    
    def test_fit_hierarchical_methods(self, sample_data):
        """Test different linkage methods."""
        for method in ['ward', 'complete', 'average']:
            analyzer = CountryClusterAnalyzer(n_clusters=3)
            analyzer.fit_kmeans(sample_data)
            analyzer.fit_hierarchical(sample_data, method=method)
            
            assert analyzer.linkage_matrix is not None
    
    def test_cluster_profiles_generated(self, analyzer, sample_data):
        """Test that cluster profiles are generated."""
        analyzer.fit_kmeans(sample_data)
        
        profiles = analyzer.get_cluster_profiles()
        
        assert profiles is not None
        assert len(profiles) == 3
        assert all('cluster_id' in p for p in profiles)
        assert all('n_countries' in p for p in profiles)
        assert all('countries' in p for p in profiles)
    
    def test_cluster_profiles_have_names(self, analyzer, sample_data):
        """Test that cluster profiles have descriptive names."""
        analyzer.fit_kmeans(sample_data)
        
        profiles = analyzer.get_cluster_profiles()
        
        # At least some profiles should have names assigned
        assert any('cluster_name' in p for p in profiles)
    
    def test_get_country_cluster(self, analyzer, sample_data):
        """Test getting cluster info for a specific country."""
        analyzer.fit_kmeans(sample_data)
        
        result = analyzer.get_country_cluster('C00')
        
        assert result is not None
        assert 'country' in result
        assert 'cluster_id' in result
        assert 'similar_countries' in result
    
    def test_get_country_cluster_not_found(self, analyzer, sample_data):
        """Test getting cluster info for non-existent country."""
        analyzer.fit_kmeans(sample_data)
        
        result = analyzer.get_country_cluster('NONEXISTENT')
        
        assert result is None
    
    def test_find_optimal_clusters(self, analyzer, sample_data):
        """Test finding optimal number of clusters."""
        results = analyzer.find_optimal_clusters(sample_data, max_clusters=6)
        
        assert 'k' in results
        assert 'inertia' in results
        assert 'silhouette' in results
        assert len(results['k']) == 5  # k from 2 to 6
    
    def test_find_optimal_clusters_silhouette_range(self, analyzer, sample_data):
        """Test that silhouette scores are in valid range."""
        results = analyzer.find_optimal_clusters(sample_data, max_clusters=5)
        
        for score in results['silhouette']:
            assert -1 <= score <= 1
    
    def test_get_cluster_summary_report(self, analyzer, sample_data):
        """Test summary report generation."""
        analyzer.fit_kmeans(sample_data)
        
        report = analyzer.get_cluster_summary_report()
        
        assert isinstance(report, str)
        assert 'CLUSTER ANALYSIS REPORT' in report
        assert 'countries' in report.lower()
    
    def test_plot_clusters_returns_figure(self, analyzer, sample_data):
        """Test that plot_clusters returns a matplotlib figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit_kmeans(sample_data)
        fig = analyzer.plot_clusters()
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_dendrogram_returns_figure(self, analyzer, sample_data):
        """Test that plot_dendrogram returns a matplotlib figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit_kmeans(sample_data)
        analyzer.fit_hierarchical(sample_data)
        fig = analyzer.plot_dendrogram()
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_elbow_returns_figure(self, analyzer, sample_data):
        """Test that plot_elbow returns a matplotlib figure."""
        import matplotlib.pyplot as plt
        
        fig = analyzer.plot_elbow(sample_data, max_clusters=5)
        
        assert fig is not None
        plt.close(fig)


class TestClusterAfricanCountries:
    """Test cases for convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 30
        
        return pd.DataFrame({
            'country': [f'C{i:02d}' for i in range(n)],
            'country_name': [f'Country {i}' for i in range(n)],
            'year': [2020] * n,
            'cdi_smooth': np.random.uniform(20, 80, n),
            'gdp_volatility': np.random.uniform(0.5, 4, n),
            'governance_index': np.random.uniform(-1.5, 1, n),
            'inflation': np.random.uniform(2, 15, n),
            'investment': np.random.uniform(15, 35, n),
            'trade_openness': np.random.uniform(30, 90, n)
        })
    
    def test_cluster_african_countries_returns_dict(self, sample_data):
        """Test that function returns expected dictionary."""
        results = cluster_african_countries(sample_data, n_clusters=3)
        
        assert isinstance(results, dict)
        assert 'analyzer' in results
        assert 'profiles' in results
        assert 'report' in results
        assert 'clustered_data' in results
    
    def test_cluster_african_countries_with_save_dir(self, sample_data, tmp_path):
        """Test saving outputs to directory."""
        save_dir = str(tmp_path / 'cluster_output')
        
        results = cluster_african_countries(
            sample_data, 
            n_clusters=3, 
            save_dir=save_dir
        )
        
        assert Path(save_dir).exists()


class TestClusteringEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            'country': ['A', 'B', 'C'],
            'year': [2020, 2020, 2020],
            'cdi_smooth': [10, 50, 90],
            'gdp_volatility': [1, 2, 3],
            'governance_index': [-1, 0, 1]
        })
        
        analyzer = CountryClusterAnalyzer(n_clusters=2)
        analyzer.fit_kmeans(df)
        
        assert len(analyzer.df_clustered) == 3
    
    def test_missing_values_handled(self):
        """Test that missing values are handled."""
        df = pd.DataFrame({
            'country': ['A', 'B', 'C', 'D', 'E', 'F'],
            'year': [2020, 2020, 2020, 2020, 2020, 2020],
            'cdi_smooth': [10, 50, np.nan, 90, 30, 70],
            'gdp_volatility': [1, 2, 3, 4, 5, 6],
            'governance_index': [-1, np.nan, 0, 1, 0.5, -0.5]
        })
        
        analyzer = CountryClusterAnalyzer(n_clusters=2)
        analyzer.fit_kmeans(df)
        
        # Should have dropped rows with NaN, keeping at least 4
        assert len(analyzer.df_clustered) <= 6
        assert len(analyzer.df_clustered) >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
