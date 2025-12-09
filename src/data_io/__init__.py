# ==============================================================================
# src/data_io/__init__.py
# ==============================================================================
"""
Data I/O package for African Commodities Paradox project.

This package contains modules for fetching economic data from various sources:
- World Bank WDI API
- (Future) UNCTAD commodity statistics
- (Future) World Governance Indicators
"""

from .worldbank import WorldBankAPI, fetch_wdi_data

__all__ = [
    'WorldBankAPI',
    'fetch_wdi_data'
]