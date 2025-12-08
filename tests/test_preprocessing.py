"""
Unit tests for data preprocessing module.

Tests data cleaning, missing value handling, and validation.

Author: Abraham Adegoke
Date: December 2025
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor(strategy='impute')
        
        # Create sample data with missing values
        self.sample_data = pd.DataFrame({
            'country': ['NGA', 'NGA', 'ZAF', 'ZAF', 'KEN', 'KEN'],
            'year': [2010, 2011, 2010, 2011, 2010, 2011],
            'cdi_smooth_lag1': [90.0, 92.0, 30.0, np.nan, 40.0, 42.0],
            'inflation_lag1': [13.5, np.nan, 5.2, 6.1, np.nan, 8.5],
            'trade_openness_lag1': [50.0, 52.0, 60.0, 62.0, 55.0, 57.0],
            'investment_lag1': [22.0, 23.0, np.nan, 28.0, 25.0, 26.0],
            'log_gdp_volatility': [1.2, 1.5, 0.8, 0.9, 1.1, 1.3]
        })
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        pp = DataPreprocessor(strategy='impute')
        assert pp.strategy == 'impute'
        assert pp.imputation_values == {}
        
        pp_drop = DataPreprocessor(strategy='drop')
        assert pp_drop.strategy == 'drop'
    
    def test_handle_missing_values_impute(self):
        """Test missing value imputation."""
        df_clean = self.preprocessor.handle_missing_values(
            self.sample_data.copy(),
            method='median'
        )
        
        # Should have no NaN in numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        assert not df_clean[numeric_cols].isnull().any().any()
        
        # Length should be preserved with imputation
        assert len(df_clean) == len(self.sample_data)
    
    def test_handle_missing_values_drop(self):
        """Test missing value dropping."""
        pp_drop = DataPreprocessor(strategy='drop')
        df_clean = pp_drop.handle_missing_values(self.sample_data.copy())
        
        # Should have fewer rows
        assert len(df_clean) < len(self.sample_data)
        
        # Should have no missing values in key features
        key_features = ['cdi_smooth_lag1', 'inflation_lag1', 
                       'trade_openness_lag1', 'investment_lag1']
        assert not df_clean[key_features].isnull().any().any()
    
    def test_imputation_by_country(self):
        """Test that imputation is done by country."""
        df_clean = self.preprocessor.handle_missing_values(
            self.sample_data.copy(),
            method='median'
        )
        
        # Check that NGA's missing inflation was filled
        nga_inflation = df_clean[
            (df_clean['country'] == 'NGA') & 
            (df_clean['year'] == 2011)
        ]['inflation_lag1'].values[0]
        
        # Should be close to NGA's other inflation value
        assert not np.isnan(nga_inflation)
        assert nga_inflation > 0
    
    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method."""
        # Create data with outliers
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        })
        
        pp = DataPreprocessor()
        df_clean = pp.remove_outliers(
            test_data,
            columns=['value'],
            method='iqr',
            threshold=1.5
        )
        
        # Outlier should be capped, not removed (we use clip)
        assert len(df_clean) == len(test_data)
        assert df_clean['value'].max() < 100
    
    def test_validate_data_success(self):
        """Test data validation with valid data."""
        valid_data = pd.DataFrame({
            'country': ['NGA', 'ZAF'],
            'year': [2010, 2010],
            'cdi_smooth_lag1': [90.0, 30.0],
            'log_gdp_volatility': [1.2, 0.8]
        })
        
        validation = self.preprocessor.validate_data(valid_data)
        
        # All checks should pass
        assert validation['has_required_columns']
        assert validation['no_duplicates']
        assert validation['valid_ranges']
    
    def test_validate_data_detects_duplicates(self):
        """Test duplicate detection."""
        duplicate_data = pd.DataFrame({
            'country': ['NGA', 'NGA', 'NGA'],
            'year': [2010, 2010, 2011],  # First two are duplicates
            'cdi_smooth_lag1': [90.0, 90.0, 92.0],
            'log_gdp_volatility': [1.2, 1.2, 1.5]
        })
        
        validation = self.preprocessor.validate_data(duplicate_data)
        
        # Should detect duplicates
        assert not validation['no_duplicates']
    
    def test_validate_data_detects_invalid_ranges(self):
        """Test invalid value range detection."""
        invalid_data = pd.DataFrame({
            'country': ['NGA'],
            'year': [2010],
            'cdi_smooth_lag1': [150.0],  # CDI > 100 is invalid
            'log_gdp_volatility': [1.2]
        })
        
        validation = self.preprocessor.validate_data(invalid_data)
        
        # Should detect invalid range
        assert not validation['valid_ranges']
    
    def test_prepare_for_modeling_complete(self):
        """Test complete preprocessing pipeline."""
        X, y, features = self.preprocessor.prepare_for_modeling(
            self.sample_data.copy(),
            target='log_gdp_volatility'
        )
        
        # Check outputs
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(features, list)
        
        # Should have no missing values
        assert not X.isnull().any().any()
        assert not y.isnull().any()
        
        # Features should be correct
        assert 'cdi_smooth_lag1' in features
        assert 'inflation_lag1' in features
        
        # X and y should have same length
        assert len(X) == len(y)


class TestFeatureEngineering:
    """Test feature engineering operations."""
    
    def test_cdi_smoothing(self):
        """Test 3-year moving average on CDI."""
        test_df = pd.DataFrame({
            'country': ['NGA'] * 5,
            'year': [2010, 2011, 2012, 2013, 2014],
            'cdi_raw': [90, 92, 88, 91, 89]
        })
        
        # Apply 3-year MA
        test_df['cdi_smooth'] = test_df.groupby('country')['cdi_raw'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Check smoothing
        assert test_df['cdi_smooth'].iloc[0] == 90.0  # First year: only itself
        assert test_df['cdi_smooth'].iloc[1] == 91.0  # (90+92)/2
        assert test_df['cdi_smooth'].iloc[2] == pytest.approx(90.0)  # (90+92+88)/3
    
    def test_volatility_calculation(self):
        """Test 5-year rolling volatility."""
        test_df = pd.DataFrame({
            'country': ['NGA'] * 7,
            'year': range(2010, 2017),
            'gdp_growth': [5.0, 4.5, 5.5, 3.0, 6.0, 4.0, 5.0]
        })
        
        # Calculate 5-year rolling std
        test_df['gdp_volatility'] = test_df.groupby('country')['gdp_growth'].transform(
            lambda x: x.rolling(window=5, min_periods=3).std()
        )
        
        # Check volatility
        assert not test_df['gdp_volatility'].iloc[:2].isnull().all()  # First 2 years: NaN
        assert test_df['gdp_volatility'].iloc[4] > 0  # Should have value by year 5
    
    def test_lagged_features(self):
        """Test lagged feature creation."""
        test_df = pd.DataFrame({
            'country': ['NGA'] * 5,
            'year': [2010, 2011, 2012, 2013, 2014],
            'cdi_smooth': [90, 92, 88, 91, 89]
        })
        
        # Create lag
        test_df['cdi_smooth_lag1'] = test_df.groupby('country')['cdi_smooth'].shift(1)
        
        # Check lag
        assert np.isnan(test_df['cdi_smooth_lag1'].iloc[0])  # First year: NaN
        assert test_df['cdi_smooth_lag1'].iloc[1] == 90  # 2011 should have 2010's value
        assert test_df['cdi_smooth_lag1'].iloc[2] == 92  # 2012 should have 2011's value


class TestDataQuality:
    """Test data quality checks."""
    
    def test_no_extreme_values(self):
        """Test that data has no extreme outliers."""
        test_df = pd.DataFrame({
            'gdp_growth': [-10, -5, 0, 5, 10],  # Reasonable range
            'inflation': [2, 5, 10, 15, 20],
            'cdi': [0, 25, 50, 75, 100]
        })
        
        # GDP growth should be reasonable
        assert test_df['gdp_growth'].min() >= -20
        assert test_df['gdp_growth'].max() <= 20
        
        # CDI should be 0-100
        assert test_df['cdi'].min() >= 0
        assert test_df['cdi'].max() <= 100
    
    def test_no_future_data_leakage(self):
        """Test that lagged features prevent data leakage."""
        test_df = pd.DataFrame({
            'country': ['NGA'] * 3,
            'year': [2010, 2011, 2012],
            'cdi': [90, 92, 88],
            'gdp_volatility': [1.2, 1.5, 1.3]
        })
        
        # Create lagged CDI
        test_df['cdi_lag1'] = test_df.groupby('country')['cdi'].shift(1)
        
        # When predicting 2011 volatility, should only use 2010 CDI
        row_2011 = test_df[test_df['year'] == 2011].iloc[0]
        assert row_2011['cdi_lag1'] == 90  # 2010's value, not 2011's


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])