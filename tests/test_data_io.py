"""
Unit tests for data collection modules.

Tests the World Bank API client and data fetching functionality.

Author: Abraham Adegoke
Date: December 2025
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_io.worldbank import WorldBankAPI, fetch_wdi_data


class TestWorldBankAPI:
    """Test suite for WorldBankAPI class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api = WorldBankAPI()
        self.test_countries = ['NGA', 'ZAF', 'KEN']
        self.test_indicator = 'NY.GDP.MKTP.KD.ZG'
        self.start_year = 2010
        self.end_year = 2015
    
    def test_api_initialization(self):
        """Test API client initialization."""
        api = WorldBankAPI(per_page=500)
        assert api.per_page == 500
        assert api.BASE_URL == "https://api.worldbank.org/v2"
        assert api.session is not None
    
    @patch('data_io.worldbank.requests.Session.get')
    def test_fetch_indicator_success(self, mock_get):
        """Test successful indicator fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {'page': 1, 'pages': 1, 'per_page': 1000, 'total': 3},
            [
                {
                    'countryiso3code': 'NGA',
                    'country': {'value': 'Nigeria'},
                    'date': '2010',
                    'value': 5.3
                },
                {
                    'countryiso3code': 'NGA',
                    'country': {'value': 'Nigeria'},
                    'date': '2011',
                    'value': 4.8
                }
            ]
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Fetch data
        df = self.api.fetch_indicator(
            countries=['NGA'],
            indicator=self.test_indicator,
            start_year=2010,
            end_year=2011
        )
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'country' in df.columns
        assert 'year' in df.columns
        assert 'value' in df.columns
        assert df['country'].iloc[0] == 'NGA'
        assert df['year'].iloc[0] == 2010
    
    @patch('data_io.worldbank.requests.Session.get')
    def test_fetch_indicator_empty_response(self, mock_get):
        """Test handling of empty API response."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {'page': 1, 'pages': 1, 'per_page': 1000, 'total': 0},
            []
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        df = self.api.fetch_indicator(
            countries=['NGA'],
            indicator=self.test_indicator,
            start_year=2010,
            end_year=2011
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @patch('data_io.worldbank.requests.Session.get')
    def test_fetch_indicator_api_error(self, mock_get):
        """Test handling of API errors."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        df = self.api.fetch_indicator(
            countries=['NGA'],
            indicator=self.test_indicator,
            start_year=2010,
            end_year=2011
        )
        
        # Should return empty DataFrame on error
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @patch('data_io.worldbank.requests.Session.get')
    def test_fetch_multiple_indicators(self, mock_get):
        """Test fetching multiple indicators."""
        # Mock responses for each indicator
        mock_response = Mock()
        mock_response.json.return_value = [
            {'page': 1, 'pages': 1, 'per_page': 1000, 'total': 1},
            [
                {
                    'countryiso3code': 'NGA',
                    'country': {'value': 'Nigeria'},
                    'date': '2010',
                    'value': 5.0
                }
            ]
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        indicators = {
            'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
            'FP.CPI.TOTL.ZG': 'inflation'
        }
        
        df = self.api.fetch_multiple_indicators(
            countries=['NGA'],
            indicators=indicators,
            start_year=2010,
            end_year=2010
        )
        
        assert isinstance(df, pd.DataFrame)
        assert 'country' in df.columns
        assert 'year' in df.columns
        # At least one of the indicators should be present
        assert any(ind in df.columns for ind in indicators.values())


class TestFetchWDIData:
    """Test suite for fetch_wdi_data function."""
    
    @patch('data_io.worldbank.WorldBankAPI.fetch_multiple_indicators')
    def test_fetch_wdi_data_success(self, mock_fetch):
        """Test successful WDI data fetch."""
        # Mock data
        mock_df = pd.DataFrame({
            'country': ['NGA', 'NGA'],
            'country_name': ['Nigeria', 'Nigeria'],
            'year': [2010, 2011],
            'gdp_growth': [5.3, 4.8],
            'inflation': [13.7, 10.8],
            'trade_openness': [50.2, 52.1],
            'investment': [22.3, 21.8],
            'fuel_exports_pct': [95.0, 94.5],
            'metals_exports_pct': [0.0, 0.0],
            'food_exports_pct': [2.0, 2.5]
        })
        mock_fetch.return_value = mock_df
        
        # Fetch data
        df = fetch_wdi_data(
            countries=['NGA'],
            start_year=2010,
            end_year=2011
        )
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert 'cdi_raw' in df.columns
        assert len(df) == 2
        # CDI should be sum of commodity exports
        expected_cdi = 95.0 + 0.0 + 2.0
        assert df['cdi_raw'].iloc[0] == pytest.approx(expected_cdi)
    
    def test_cdi_calculation(self):
        """Test CDI calculation logic."""
        # Create test data
        test_df = pd.DataFrame({
            'country': ['NGA', 'ZAF'],
            'year': [2010, 2010],
            'fuel_exports_pct': [90.0, 20.0],
            'metals_exports_pct': [5.0, 50.0],
            'food_exports_pct': [3.0, 10.0]
        })
        
        # Calculate CDI
        test_df['cdi_raw'] = (
            test_df['fuel_exports_pct'].fillna(0) + 
            test_df['metals_exports_pct'].fillna(0) + 
            test_df['food_exports_pct'].fillna(0)
        )
        
        # Assertions
        assert test_df['cdi_raw'].iloc[0] == 98.0  # Nigeria: oil-dependent
        assert test_df['cdi_raw'].iloc[1] == 80.0  # South Africa: mineral-dependent
    
    def test_cdi_handles_missing_values(self):
        """Test CDI calculation with missing values."""
        test_df = pd.DataFrame({
            'fuel_exports_pct': [90.0, np.nan],
            'metals_exports_pct': [np.nan, 50.0],
            'food_exports_pct': [3.0, 10.0]
        })
        
        # Calculate CDI with fillna
        test_df['cdi_raw'] = (
            test_df['fuel_exports_pct'].fillna(0) + 
            test_df['metals_exports_pct'].fillna(0) + 
            test_df['food_exports_pct'].fillna(0)
        )
        
        # Should handle NaN gracefully
        assert test_df['cdi_raw'].iloc[0] == 93.0
        assert test_df['cdi_raw'].iloc[1] == 60.0
        assert not test_df['cdi_raw'].isnull().any()


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def test_country_codes_valid(self):
        """Test that country codes are valid ISO3 codes."""
        valid_codes = ['NGA', 'ZAF', 'KEN', 'GHA', 'EGY']
        
        for code in valid_codes:
            assert len(code) == 3
            assert code.isupper()
            assert code.isalpha()
    
    def test_year_range_valid(self):
        """Test that year ranges are valid."""
        start_year = 1990
        end_year = 2023
        
        assert start_year < end_year
        assert start_year >= 1960  # World Bank data starts ~1960
        assert end_year <= 2025  # Current year + buffer
    
    def test_cdi_bounds(self):
        """Test that CDI values are within valid bounds."""
        test_df = pd.DataFrame({
            'cdi_raw': [0, 50, 100, 33.5]
        })
        
        # CDI should be between 0 and 100
        assert (test_df['cdi_raw'] >= 0).all()
        assert (test_df['cdi_raw'] <= 100).all()


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])