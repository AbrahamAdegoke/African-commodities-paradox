"""
Tests for PCA Analysis Module

Tests Principal Component Analysis functionality for dimensionality reduction
and visualization of African countries' economic characteristics.

Author: Abraham Adegoke
Date: December 2025
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.pca_analysis import PCAAnalyzer, analyze_with_pca


class TestPCAAnalyzer:
    """Test cases for PCAAnalyzer class."""
    
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
        return PCAAnalyzer(n_components=2)
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = PCAAnalyzer(n_components=3)
        
        assert analyzer.n_components == 3
        assert len(analyzer.features) > 0
    
    def test_init_custom_features(self):
        """Test initialization with custom features."""
        custom_features = ['cdi_smooth', 'inflation', 'investment']
        analyzer = PCAAnalyzer(n_components=2, features=custom_features)
        
        assert analyzer.features == custom_features
    
    def test_prepare_data(self, analyzer, sample_data):
        """Test data preparation."""
        country_data = analyzer.prepare_data(sample_data)
        
        # Should aggregate to country level
        assert len(country_data) == 20
        assert 'country' in country_data.columns
    
    def test_prepare_data_filters_features(self, sample_data):
        """Test that prepare_data filters to available features."""
        analyzer = PCAAnalyzer(
            features=['cdi_smooth', 'inflation', 'nonexistent_feature']
        )
        
        country_data = analyzer.prepare_data(sample_data)
        
        assert 'nonexistent_feature' not in analyzer.features
        assert 'cdi_smooth' in analyzer.features
        assert 'inflation' in analyzer.features
    
    def test_prepare_data_insufficient_features(self):
        """Test error when not enough features."""
        df = pd.DataFrame({
            'country': ['A', 'B'],
            'year': [2020, 2020],
            'only_one': [1, 2]
        })
        
        analyzer = PCAAnalyzer()
        
        with pytest.raises(ValueError, match="Not enough features"):
            analyzer.prepare_data(df)
    
    def test_fit(self, analyzer, sample_data):
        """Test PCA fitting."""
        analyzer.fit(sample_data)
        
        assert analyzer.X_pca is not None
        assert analyzer.loadings is not None
        assert analyzer.explained_variance is not None
    
    def test_fit_creates_pca_coordinates(self, analyzer, sample_data):
        """Test that fit creates PC columns in data."""
        analyzer.fit(sample_data)
        
        assert 'PC1' in analyzer.country_data.columns
        assert 'PC2' in analyzer.country_data.columns
    
    def test_explained_variance_sums_correctly(self, analyzer, sample_data):
        """Test that explained variance is valid."""
        analyzer.fit(sample_data)
        
        # Each component should explain 0-100%
        assert all(0 <= v <= 1 for v in analyzer.explained_variance)
        
        # Total should not exceed 100%
        assert sum(analyzer.explained_variance) <= 1.0
    
    def test_loadings_shape(self, analyzer, sample_data):
        """Test loadings matrix shape."""
        analyzer.fit(sample_data)
        
        n_features = len(analyzer.features)
        n_components = analyzer.n_components
        
        assert analyzer.loadings.shape == (n_features, n_components)
    
    def test_loadings_columns(self, analyzer, sample_data):
        """Test loadings DataFrame columns."""
        analyzer.fit(sample_data)
        
        assert 'PC1' in analyzer.loadings.columns
        assert 'PC2' in analyzer.loadings.columns
    
    def test_get_component_interpretation(self, analyzer, sample_data):
        """Test component interpretation."""
        analyzer.fit(sample_data)
        
        interpretations = analyzer.get_component_interpretation()
        
        assert 'PC1' in interpretations
        assert 'PC2' in interpretations
        assert 'variance_explained' in interpretations['PC1']
        assert 'interpretation' in interpretations['PC1']
    
    def test_pca_coordinates_shape(self, analyzer, sample_data):
        """Test PCA transformed coordinates shape."""
        analyzer.fit(sample_data)
        
        n_countries = len(analyzer.country_data)
        n_components = analyzer.n_components
        
        assert analyzer.X_pca.shape == (n_countries, n_components)
    
    def test_get_pca_summary_report(self, analyzer, sample_data):
        """Test summary report generation."""
        analyzer.fit(sample_data)
        
        report = analyzer.get_pca_summary_report()
        
        assert isinstance(report, str)
        assert 'PCA ANALYSIS REPORT' in report
        assert 'EXPLAINED VARIANCE' in report
    
    def test_plot_scree_returns_figure(self, analyzer, sample_data):
        """Test that plot_scree returns a figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data)
        fig = analyzer.plot_scree()
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_biplot_returns_figure(self, analyzer, sample_data):
        """Test that plot_biplot returns a figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data)
        fig = analyzer.plot_biplot()
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_loadings_heatmap_returns_figure(self, analyzer, sample_data):
        """Test that plot_loadings_heatmap returns a figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data)
        fig = analyzer.plot_loadings_heatmap()
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_2d_projection_returns_figure(self, analyzer, sample_data):
        """Test that plot_2d_projection returns a figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data)
        fig = analyzer.plot_2d_projection()
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_2d_projection_with_clusters(self, analyzer, sample_data):
        """Test 2D projection with cluster labels."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data)
        
        # Create dummy cluster labels
        cluster_labels = np.random.randint(0, 3, len(analyzer.country_data))
        
        fig = analyzer.plot_2d_projection(cluster_labels=cluster_labels)
        
        assert fig is not None
        plt.close(fig)


class TestAnalyzeWithPca:
    """Test cases for convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 25
        
        return pd.DataFrame({
            'country': [f'C{i:02d}' for i in range(n)],
            'country_name': [f'Country {i}' for i in range(n)],
            'year': [2020] * n,
            'cdi_smooth': np.random.uniform(20, 80, n),
            'gdp_volatility': np.random.uniform(0.5, 4, n),
            'governance_index': np.random.uniform(-1.5, 1, n),
            'inflation': np.random.uniform(2, 15, n)
        })
    
    def test_analyze_with_pca_returns_dict(self, sample_data):
        """Test that function returns expected dictionary."""
        results = analyze_with_pca(sample_data, n_components=2)
        
        assert isinstance(results, dict)
        assert 'analyzer' in results
        assert 'loadings' in results
        assert 'explained_variance' in results
        assert 'country_pca' in results
        assert 'report' in results
    
    def test_analyze_with_pca_with_save_dir(self, sample_data, tmp_path):
        """Test saving outputs to directory."""
        save_dir = str(tmp_path / 'pca_output')
        
        results = analyze_with_pca(
            sample_data,
            n_components=2,
            save_dir=save_dir
        )
        
        assert Path(save_dir).exists()


class TestPCAEdgeCases:
    """Test edge cases and error handling."""
    
    def test_more_components_than_features(self):
        """Test requesting more components than features - should use max available."""
        df = pd.DataFrame({
            'country': ['A', 'B', 'C', 'D', 'E'],
            'year': [2020] * 5,
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6]
        })
        
        # Request 2 components with 2 features - should work
        analyzer = PCAAnalyzer(n_components=2, features=['feature1', 'feature2'])
        analyzer.fit(df)
        
        # Should get 2 components
        assert analyzer.X_pca.shape[1] == 2
    
    def test_small_dataset(self):
        """Test with small dataset."""
        df = pd.DataFrame({
            'country': ['A', 'B', 'C'],
            'year': [2020, 2020, 2020],
            'cdi_smooth': [10, 50, 90],
            'gdp_volatility': [1, 2, 3],
            'governance_index': [-1, 0, 1]
        })
        
        analyzer = PCAAnalyzer(n_components=2)
        analyzer.fit(df)
        
        assert analyzer.X_pca is not None
        assert len(analyzer.country_data) == 3
    
    def test_missing_values_dropped(self):
        """Test that rows with missing values are dropped."""
        df = pd.DataFrame({
            'country': ['A', 'B', 'C', 'D'],
            'year': [2020, 2020, 2020, 2020],
            'cdi_smooth': [10, 50, np.nan, 90],
            'gdp_volatility': [1, 2, 3, 4]
        })
        
        analyzer = PCAAnalyzer(n_components=2, features=['cdi_smooth', 'gdp_volatility'])
        analyzer.fit(df)
        
        # Should have dropped the row with NaN
        assert len(analyzer.country_data) == 3
    
    def test_pca_before_fit_raises_error(self):
        """Test that methods raise error before fit."""
        analyzer = PCAAnalyzer()
        
        with pytest.raises(ValueError, match="Must fit PCA first"):
            analyzer.plot_scree()
    
    def test_loadings_index_matches_features(self):
        """Test that loadings index matches feature names."""
        np.random.seed(42)
        df = pd.DataFrame({
            'country': ['A', 'B', 'C', 'D', 'E'],
            'year': [2020] * 5,
            'cdi_smooth': np.random.rand(5),
            'inflation': np.random.rand(5),
            'investment': np.random.rand(5)
        })
        
        features = ['cdi_smooth', 'inflation', 'investment']
        analyzer = PCAAnalyzer(n_components=2, features=features)
        analyzer.fit(df)
        
        assert list(analyzer.loadings.index) == features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
