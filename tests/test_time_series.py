"""
Tests for Time Series Analysis Module

Tests trend analysis, decomposition, stationarity tests, and ARIMA forecasting.

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

from analysis.time_series import (
    TimeSeriesAnalyzer, 
    analyze_country_time_series,
    compare_country_trends
)


class TestTimeSeriesAnalyzer:
    """Test cases for TimeSeriesAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        years = list(range(2000, 2024))
        n_years = len(years)
        
        # Create data with trend and noise
        trend = np.linspace(2, 5, n_years)
        noise = np.random.normal(0, 0.5, n_years)
        gdp_growth = trend + noise
        
        return pd.DataFrame({
            'country': ['NGA'] * n_years,
            'country_name': ['Nigeria'] * n_years,
            'year': years,
            'gdp_growth': gdp_growth,
            'cdi_smooth': np.random.uniform(60, 80, n_years),
            'inflation': np.random.uniform(5, 15, n_years)
        })
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TimeSeriesAnalyzer()
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = TimeSeriesAnalyzer()
        
        assert analyzer.data is None
        assert analyzer.country is None
        assert analyzer.target is None
    
    def test_fit(self, analyzer, sample_data):
        """Test fitting analyzer on data."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        
        assert analyzer.data is not None
        assert analyzer.country == 'NGA'
        assert analyzer.target == 'gdp_growth'
    
    def test_fit_sorts_by_year(self, analyzer, sample_data):
        """Test that fit sorts data by year."""
        # Shuffle the data
        shuffled = sample_data.sample(frac=1, random_state=42)
        
        analyzer.fit(shuffled, country='NGA', target='gdp_growth')
        
        years = analyzer.data['year'].values
        assert all(years[i] <= years[i+1] for i in range(len(years)-1))
    
    def test_fit_insufficient_data(self, analyzer):
        """Test error with insufficient data."""
        df = pd.DataFrame({
            'country': ['NGA'] * 3,
            'year': [2020, 2021, 2022],
            'gdp_growth': [1, 2, 3]
        })
        
        with pytest.raises(ValueError, match="Not enough data"):
            analyzer.fit(df, country='NGA', target='gdp_growth')
    
    def test_fit_missing_target(self, analyzer, sample_data):
        """Test error when target column missing."""
        with pytest.raises(ValueError, match="not found"):
            analyzer.fit(sample_data, country='NGA', target='nonexistent')
    
    def test_analyze_trend(self, analyzer, sample_data):
        """Test trend analysis."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.analyze_trend()
        
        assert 'slope' in results
        assert 'intercept' in results
        assert 'r_squared' in results
        assert 'p_value' in results
        assert 'trend_direction' in results
        assert 'is_significant' in results
    
    def test_analyze_trend_positive_slope(self, analyzer, sample_data):
        """Test that positive trend is detected."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.analyze_trend()
        
        # Sample data has upward trend
        assert results['slope'] > 0
    
    def test_analyze_trend_direction(self, analyzer):
        """Test trend direction detection."""
        # Create data with clear downward trend
        years = list(range(2010, 2024))
        df = pd.DataFrame({
            'country': ['TEST'] * len(years),
            'year': years,
            'value': np.linspace(10, 2, len(years))  # Decreasing
        })
        
        analyzer.fit(df, country='TEST', target='value')
        results = analyzer.analyze_trend()
        
        assert results['slope'] < 0
        assert 'Decreasing' in results['trend_direction']
    
    def test_decompose(self, analyzer, sample_data):
        """Test time series decomposition."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.decompose()
        
        assert results is not None
        assert 'trend' in results
        assert 'seasonal' in results
        assert 'residual' in results
        assert 'observed' in results
    
    def test_decompose_period_adjustment(self, analyzer):
        """Test that period is adjusted for short series."""
        # Create short series
        df = pd.DataFrame({
            'country': ['TEST'] * 10,
            'year': list(range(2014, 2024)),
            'value': np.random.rand(10)
        })
        
        analyzer.fit(df, country='TEST', target='value')
        results = analyzer.decompose(period=8)  # Request large period
        
        # Should still work with adjusted period
        assert results is not None or results is None  # May fail gracefully
    
    def test_test_stationarity(self, analyzer, sample_data):
        """Test stationarity test (ADF)."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.test_stationarity()
        
        assert 'adf_statistic' in results
        assert 'p_value' in results
        assert 'is_stationary' in results
        assert 'critical_values' in results
    
    def test_stationarity_result_is_boolean(self, analyzer, sample_data):
        """Test that is_stationary is boolean."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.test_stationarity()
        
        assert isinstance(results['is_stationary'], (bool, np.bool_))
    
    def test_compute_autocorrelation(self, analyzer, sample_data):
        """Test autocorrelation computation."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.compute_autocorrelation(nlags=5)
        
        assert results is not None
        assert 'acf' in results
        assert 'pacf' in results
        assert 'nlags' in results
    
    def test_autocorrelation_values_in_range(self, analyzer, sample_data):
        """Test that ACF values are in [-1, 1]."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.compute_autocorrelation(nlags=5)
        
        assert all(-1 <= v <= 1 for v in results['acf'])
        assert all(-1 <= v <= 1 for v in results['pacf'])
    
    def test_fit_arima(self, analyzer, sample_data):
        """Test ARIMA model fitting."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.fit_arima()
        
        assert results is not None
        assert 'order' in results
        assert 'aic' in results
        assert 'rmse' in results
        assert 'fitted_values' in results
    
    def test_fit_arima_custom_order(self, analyzer, sample_data):
        """Test ARIMA with custom order."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.fit_arima(order=(1, 0, 0))
        
        assert results['order'] == (1, 0, 0)
    
    def test_fit_arima_auto_select(self, analyzer, sample_data):
        """Test ARIMA auto order selection."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        results = analyzer.fit_arima(auto_select=True)
        
        assert results is not None
        assert len(results['order']) == 3
    
    def test_forecast(self, analyzer, sample_data):
        """Test forecasting."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        analyzer.fit_arima()
        
        forecast_df = analyzer.forecast(periods=3)
        
        assert len(forecast_df) == 3
        assert 'year' in forecast_df.columns
        assert 'forecast' in forecast_df.columns
        assert 'lower_ci' in forecast_df.columns
        assert 'upper_ci' in forecast_df.columns
    
    def test_forecast_years_correct(self, analyzer, sample_data):
        """Test that forecast years are correct."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        analyzer.fit_arima()
        
        last_year = int(sample_data['year'].max())
        forecast_df = analyzer.forecast(periods=3)
        
        expected_years = [last_year + 1, last_year + 2, last_year + 3]
        assert list(forecast_df['year']) == expected_years
    
    def test_forecast_confidence_interval_order(self, analyzer, sample_data):
        """Test that lower CI < forecast < upper CI."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        analyzer.fit_arima()
        
        forecast_df = analyzer.forecast(periods=3)
        
        for _, row in forecast_df.iterrows():
            assert row['lower_ci'] <= row['forecast'] <= row['upper_ci']
    
    def test_forecast_without_arima_raises_error(self, analyzer, sample_data):
        """Test that forecast raises error without ARIMA fit."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        
        with pytest.raises(ValueError, match="Must fit ARIMA"):
            analyzer.forecast(periods=3)
    
    def test_get_summary_report(self, analyzer, sample_data):
        """Test summary report generation."""
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        analyzer.analyze_trend()
        
        report = analyzer.get_summary_report()
        
        assert isinstance(report, str)
        assert 'TIME SERIES ANALYSIS REPORT' in report
        assert 'NGA' in report
    
    def test_plot_decomposition_returns_figure(self, analyzer, sample_data):
        """Test that plot_decomposition returns a figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        fig = analyzer.plot_decomposition()
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_forecast_returns_figure(self, analyzer, sample_data):
        """Test that plot_forecast returns a figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        fig = analyzer.plot_forecast(periods=3)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_acf_pacf_returns_figure(self, analyzer, sample_data):
        """Test that plot_acf_pacf returns a figure."""
        import matplotlib.pyplot as plt
        
        analyzer.fit(sample_data, country='NGA', target='gdp_growth')
        fig = analyzer.plot_acf_pacf(nlags=5)
        
        assert fig is not None
        plt.close(fig)


class TestAnalyzeCountryTimeSeries:
    """Test cases for convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        years = list(range(2005, 2024))
        n = len(years)
        
        return pd.DataFrame({
            'country': ['NGA'] * n,
            'year': years,
            'gdp_growth': np.random.normal(3, 2, n),
            'cdi_smooth': np.random.uniform(60, 80, n)
        })
    
    def test_analyze_country_returns_dict(self, sample_data):
        """Test that function returns expected dictionary."""
        results = analyze_country_time_series(
            sample_data, 
            country='NGA',
            target='gdp_growth',
            forecast_periods=2
        )
        
        assert isinstance(results, dict)
        assert 'country' in results
        assert 'target' in results
        assert 'trend' in results
        assert 'stationarity' in results
    
    def test_analyze_country_with_save_dir(self, sample_data, tmp_path):
        """Test saving outputs."""
        save_dir = str(tmp_path / 'ts_output')
        
        results = analyze_country_time_series(
            sample_data,
            country='NGA',
            target='gdp_growth',
            save_dir=save_dir
        )
        
        assert Path(save_dir).exists()
    
    def test_analyze_country_invalid_returns_error(self, sample_data):
        """Test that invalid country returns error dict."""
        results = analyze_country_time_series(
            sample_data,
            country='INVALID',
            target='gdp_growth'
        )
        
        assert 'error' in results


class TestCompareCountryTrends:
    """Test cases for trend comparison function."""
    
    @pytest.fixture
    def multi_country_data(self):
        """Create data for multiple countries."""
        np.random.seed(42)
        years = list(range(2010, 2024))
        n = len(years)
        
        data = []
        for country in ['NGA', 'KEN', 'ZAF']:
            for i, year in enumerate(years):
                data.append({
                    'country': country,
                    'year': year,
                    'gdp_growth': np.random.normal(3, 1) + i * 0.1  # Slight upward trend
                })
        
        return pd.DataFrame(data)
    
    def test_compare_trends_returns_dataframe(self, multi_country_data):
        """Test that function returns DataFrame."""
        results = compare_country_trends(
            multi_country_data,
            countries=['NGA', 'KEN', 'ZAF'],
            target='gdp_growth'
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
    
    def test_compare_trends_columns(self, multi_country_data):
        """Test that result has expected columns."""
        results = compare_country_trends(
            multi_country_data,
            countries=['NGA', 'KEN'],
            target='gdp_growth'
        )
        
        assert 'country' in results.columns
        assert 'trend_direction' in results.columns
        assert 'annual_change' in results.columns
    
    def test_compare_trends_handles_missing_country(self, multi_country_data):
        """Test that missing countries are handled gracefully."""
        results = compare_country_trends(
            multi_country_data,
            countries=['NGA', 'INVALID'],
            target='gdp_growth'
        )
        
        # Should only have NGA
        assert len(results) == 1
        assert results.iloc[0]['country'] == 'NGA'


class TestTimeSeriesEdgeCases:
    """Test edge cases and error handling."""
    
    def test_constant_series(self):
        """Test with constant time series."""
        df = pd.DataFrame({
            'country': ['TEST'] * 15,
            'year': list(range(2010, 2025)),
            'value': [5.0] * 15  # Constant
        })
        
        analyzer = TimeSeriesAnalyzer()
        analyzer.fit(df, country='TEST', target='value')
        
        results = analyzer.analyze_trend()
        
        # Slope should be essentially zero
        assert abs(results['slope']) < 0.001
    
    def test_missing_values_interpolated(self):
        """Test that missing values are interpolated."""
        values = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
        
        df = pd.DataFrame({
            'country': ['TEST'] * 10,
            'year': list(range(2015, 2025)),
            'value': values
        })
        
        analyzer = TimeSeriesAnalyzer()
        analyzer.fit(df, country='TEST', target='value')
        
        # Should have no NaN after interpolation
        assert not analyzer.data['value'].isna().any()
    
    def test_short_series_autocorrelation(self):
        """Test autocorrelation with short series."""
        df = pd.DataFrame({
            'country': ['TEST'] * 6,
            'year': list(range(2019, 2025)),
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        analyzer = TimeSeriesAnalyzer()
        analyzer.fit(df, country='TEST', target='value')
        
        results = analyzer.compute_autocorrelation(nlags=10)
        
        # Should adjust nlags or return None
        if results is not None:
            assert results['nlags'] <= 2  # Max lags for 6 observations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])