# ==============================================================================
# src/analysis/__init__.py
# ==============================================================================
"""
Analysis package for unsupervised learning and time series techniques.

Contains:
    - CountryClusterAnalyzer: k-Means and Hierarchical Clustering
    - PCAAnalyzer: Principal Component Analysis
    - TimeSeriesAnalyzer: Trend, Decomposition, ARIMA
    
These modules address key questions:
    - Are there different TYPES of countries affected by the commodities paradox?
    - What are the main dimensions of economic variation?
    - Which countries are similar to each other?
    - What are the temporal trends and forecasts?
"""

from .clustering import CountryClusterAnalyzer, cluster_african_countries
from .pca_analysis import PCAAnalyzer, analyze_with_pca
from .time_series import TimeSeriesAnalyzer, analyze_country_time_series, compare_country_trends

__all__ = [
    'CountryClusterAnalyzer',
    'cluster_african_countries',
    'PCAAnalyzer', 
    'analyze_with_pca',
    'TimeSeriesAnalyzer',
    'analyze_country_time_series',
    'compare_country_trends'
]
