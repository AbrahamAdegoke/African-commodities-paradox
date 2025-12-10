"""
World Bank Data Collector Module

This module fetches economic indicators from the World Bank WDI API
for African countries to analyze the commodities paradox.

Includes:
- Core macroeconomic indicators (GDP growth, inflation, trade, investment)
- Commodity export indicators (fuel, metals, food, agriculture)
- World Governance Indicators (WGI)
- Exchange rate indicators

Author: Abraham Adegoke
Date: November 2025
"""

import pandas as pd
import requests
from typing import List, Dict, Optional, Union
import time
import logging
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorldBankAPI:
    """
    Client for interacting with World Bank WDI API v2.
    
    Base URL: https://api.worldbank.org/v2/
    Documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
    
    Attributes:
        per_page (int): Number of records per API request (max 1000)
        session (requests.Session): Reusable HTTP session
        
    Example:
        >>> api = WorldBankAPI()
        >>> df = api.fetch_indicator(['NGA', 'ZAF'], 'NY.GDP.MKTP.KD.ZG', 2010, 2020)
    """
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    def __init__(self, per_page: int = 1000):
        """
        Initialize World Bank API client.
        
        Args:
            per_page: Number of records per API request (max 1000)
        """
        self.per_page = per_page
        self.session = requests.Session()
    
    def fetch_indicator(
        self, 
        countries: List[str], 
        indicator: str, 
        start_year: int, 
        end_year: int
    ) -> pd.DataFrame:
        """
        Fetch a single indicator for multiple countries.
        
        Args:
            countries: List of ISO3 country codes (e.g., ['NGA', 'KEN', 'ZAF'])
            indicator: World Bank indicator code (e.g., 'NY.GDP.MKTP.KD.ZG')
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            DataFrame with columns: country, country_name, year, indicator_code, value
        """
        logger.info(f"Fetching {indicator} for {len(countries)} countries ({start_year}-{end_year})")
        
        all_data = []
        
        for country in countries:
            try:
                url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}"
                params = {
                    "date": f"{start_year}:{end_year}",
                    "format": "json",
                    "per_page": self.per_page
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # World Bank API returns [metadata, data]
                if len(data) > 1 and data[1]:
                    for record in data[1]:
                        all_data.append({
                            'country': record['countryiso3code'],
                            'country_name': record['country']['value'],
                            'year': int(record['date']),
                            'indicator_code': indicator,
                            'value': record['value']
                        })
                
                # Rate limiting: be nice to the API
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error fetching {indicator} for {country}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        logger.info(f"✓ Fetched {len(df)} records for {indicator}")
        
        return df
    
    def fetch_multiple_indicators(
        self,
        countries: List[str],
        indicators: Dict[str, str],
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Fetch multiple indicators and merge into a single DataFrame.
        
        Args:
            countries: List of ISO3 country codes
            indicators: Dict mapping indicator codes to friendly names
                       e.g., {'NY.GDP.MKTP.KD.ZG': 'gdp_growth'}
            start_year: Start year
            end_year: End year
            
        Returns:
            Wide-format DataFrame with columns: country, country_name, year, [indicator_names]
        """
        all_dfs = []
        
        for indicator_code, indicator_name in indicators.items():
            df = self.fetch_indicator(countries, indicator_code, start_year, end_year)
            
            if not df.empty:
                # Rename value column to indicator name
                df = df.rename(columns={'value': indicator_name})
                df = df[['country', 'country_name', 'year', indicator_name]]
                all_dfs.append(df)
        
        # Merge all indicators
        if not all_dfs:
            logger.warning("No data fetched!")
            return pd.DataFrame()
        
        merged_df = all_dfs[0]
        for df in all_dfs[1:]:
            merged_df = merged_df.merge(
                df,
                on=['country', 'country_name', 'year'],
                how='outer'
            )
        
        logger.info(f"✓ Merged {len(indicators)} indicators into single DataFrame")
        logger.info(f"  Shape: {merged_df.shape}")
        logger.info(f"  Years: {merged_df['year'].min()}-{merged_df['year'].max()}")
        
        return merged_df


def fetch_wdi(
    countries: List[str],
    indicators: Dict[str, str],
    start_year: int = 2000,
    end_year: int = 2023
) -> pd.DataFrame:
    """
    Fetch custom World Bank indicators.
    
    This is a lower-level function that allows fetching any set of indicators.
    For the standard project indicators, use fetch_wdi_data() instead.
    
    Args:
        countries: List of ISO3 country codes
        indicators: Dict mapping WB indicator codes to column names
        start_year: Start year (default: 2000)
        end_year: End year (default: 2023)
        
    Returns:
        DataFrame with country, year, and indicator columns
        
    Example:
        >>> indicators = {
        ...     'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
        ...     'FP.CPI.TOTL.ZG': 'inflation'
        ... }
        >>> df = fetch_wdi(['NGA', 'ZAF'], indicators, 2010, 2020)
    """
    api = WorldBankAPI()
    return api.fetch_multiple_indicators(
        countries=countries,
        indicators=indicators,
        start_year=start_year,
        end_year=end_year
    )


def fetch_wdi_data(
    countries: List[str],
    start_year: int = 1990,
    end_year: int = 2023,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Main function to fetch all WDI indicators needed for the project.
    
    This is the high-level function that fetches all indicators required
    for the African Commodities Paradox analysis, including:
    - Core macroeconomic indicators
    - Commodity export indicators
    - World Governance Indicators
    - Exchange rate indicators
    
    Args:
        countries: List of ISO3 country codes
        start_year: Start year (default: 1990)
        end_year: End year (default: 2023)
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with all indicators and calculated CDI
        
    Example:
        >>> df = fetch_wdi_data(['NGA', 'ZAF', 'KEN'], 2000, 2023)
        >>> print(df['cdi_raw'].describe())
    """
    
    # Define ALL indicators needed for the project
    indicators = {
        # ===========================================
        # CORE MACROECONOMIC INDICATORS
        # ===========================================
        'NY.GDP.MKTP.KD.ZG': 'gdp_growth',              # GDP growth (annual %)
        'FP.CPI.TOTL.ZG': 'inflation',                  # Inflation, consumer prices (annual %)
        'NE.TRD.GNFS.ZS': 'trade_openness',             # Trade (% of GDP)
        'NE.GDI.TOTL.ZS': 'investment',                 # Gross capital formation (% of GDP)
        
        # ===========================================
        # COMMODITY EXPORT INDICATORS (for CDI)
        # ===========================================
        'TX.VAL.FUEL.ZS.UN': 'fuel_exports_pct',        # Fuel exports (% of merchandise exports)
        'TX.VAL.MMTL.ZS.UN': 'metals_exports_pct',      # Ores and metals exports (% of merch exports)
        'TX.VAL.AGRI.ZS.UN': 'agri_exports_pct',        # Agricultural raw materials exports
        'TX.VAL.FOOD.ZS.UN': 'food_exports_pct',        # Food exports (% of merchandise exports)
        
        # ===========================================
        # EXCHANGE RATE INDICATORS
        # ===========================================
        'PA.NUS.FCRF': 'exchange_rate',                 # Official exchange rate (LCU per US$)
        'PX.REX.REER': 'real_eff_exchange_rate',        # Real effective exchange rate index
        
        # ===========================================
        # WORLD GOVERNANCE INDICATORS (WGI)
        # ===========================================
        'CC.EST': 'control_corruption',                 # Control of Corruption: Estimate
        'GE.EST': 'govt_effectiveness',                 # Government Effectiveness: Estimate
        'PV.EST': 'political_stability',                # Political Stability: Estimate
        'RQ.EST': 'regulatory_quality',                 # Regulatory Quality: Estimate
        'RL.EST': 'rule_of_law',                        # Rule of Law: Estimate
        'VA.EST': 'voice_accountability',               # Voice and Accountability: Estimate
        
        # ===========================================
        # ADDITIONAL USEFUL INDICATORS
        # ===========================================
        'NE.EXP.GNFS.ZS': 'exports_gdp',                # Exports of goods and services (% of GDP)
        'NE.IMP.GNFS.ZS': 'imports_gdp',                # Imports of goods and services (% of GDP)
        'BN.CAB.XOKA.GD.ZS': 'current_account_gdp',     # Current account balance (% of GDP)
        'DT.DOD.DECT.GN.ZS': 'external_debt_gni',       # External debt stocks (% of GNI)
    }
    
    # Initialize API client
    api = WorldBankAPI()
    
    logger.info("=" * 70)
    logger.info("FETCHING WORLD BANK DATA")
    logger.info("=" * 70)
    logger.info(f"Countries: {len(countries)}")
    logger.info(f"Indicators: {len(indicators)}")
    logger.info(f"Period: {start_year}-{end_year}")
    logger.info("=" * 70)
    
    # Fetch all indicators
    df = api.fetch_multiple_indicators(
        countries=countries,
        indicators=indicators,
        start_year=start_year,
        end_year=end_year
    )
    
    if df.empty:
        logger.warning("No data returned from API")
        return df
    
    # ===========================================
    # CALCULATE COMMODITY DEPENDENCE INDEX (CDI)
    # ===========================================
    logger.info("\n📊 Calculating Commodity Dependence Index (CDI)...")
    
    commodity_cols = ['fuel_exports_pct', 'metals_exports_pct', 'agri_exports_pct', 'food_exports_pct']
    
    # Initialize CDI
    df['cdi_raw'] = 0.0
    
    # Sum available commodity export percentages
    available_components = []
    for col in commodity_cols:
        if col in df.columns:
            df['cdi_raw'] += df[col].fillna(0)
            available_components.append(col.replace('_exports_pct', '').replace('_pct', ''))
    
    # Cap CDI at 100%
    df['cdi_raw'] = df['cdi_raw'].clip(upper=100)
    
    logger.info(f"  CDI components: {', '.join(available_components)}")
    logger.info(f"  CDI range: {df['cdi_raw'].min():.1f}% - {df['cdi_raw'].max():.1f}%")
    
    # ===========================================
    # CALCULATE EXCHANGE RATE VOLATILITY
    # ===========================================
    logger.info("\n📈 Calculating Exchange Rate Volatility...")
    
    df = df.sort_values(['country', 'year'])
    
    if 'exchange_rate' in df.columns:
        # Calculate year-over-year % change in exchange rate
        df['exchange_rate_change'] = df.groupby('country')['exchange_rate'].pct_change() * 100
        
        # Calculate 3-year rolling volatility of exchange rate changes
        df['exchange_rate_volatility'] = df.groupby('country')['exchange_rate_change'].transform(
            lambda x: x.rolling(window=3, min_periods=2).std()
        )
        logger.info("  ✓ Exchange rate volatility calculated (3-year rolling std)")
    else:
        logger.warning("  ⚠ Exchange rate data not available")
    
    # ===========================================
    # CALCULATE COMPOSITE GOVERNANCE INDEX
    # ===========================================
    logger.info("\n🏛️  Calculating Composite Governance Index...")
    
    governance_cols = [
        'control_corruption', 'govt_effectiveness', 'political_stability',
        'regulatory_quality', 'rule_of_law', 'voice_accountability'
    ]
    
    available_gov = [col for col in governance_cols if col in df.columns]
    
    if available_gov:
        # WGI scores range from -2.5 (weak) to 2.5 (strong)
        # Calculate average of available governance indicators
        df['governance_index'] = df[available_gov].mean(axis=1)
        logger.info(f"  ✓ Governance index calculated from {len(available_gov)} indicators")
        logger.info(f"  Governance range: {df['governance_index'].min():.2f} to {df['governance_index'].max():.2f}")
    else:
        logger.warning("  ⚠ Governance indicators not available")
    
    # Sort by country and year
    df = df.sort_values(['country', 'year']).reset_index(drop=True)
    
    # ===========================================
    # SUMMARY
    # ===========================================
    logger.info("\n" + "=" * 70)
    logger.info("DATA FETCH COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Countries: {df['country'].nunique()}")
    logger.info(f"Years: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"Columns: {len(df.columns)}")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Data saved to {output_path}")
    
    return df


def get_available_indicators() -> Dict[str, str]:
    """
    Return dictionary of available indicators for reference.
    
    Returns:
        Dict mapping indicator codes to descriptions
    """
    return {
        # Core macro
        'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
        'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',
        'NE.TRD.GNFS.ZS': 'Trade (% of GDP)',
        'NE.GDI.TOTL.ZS': 'Gross capital formation (% of GDP)',
        
        # Commodity exports
        'TX.VAL.FUEL.ZS.UN': 'Fuel exports (% of merchandise exports)',
        'TX.VAL.MMTL.ZS.UN': 'Ores and metals exports (% of merchandise exports)',
        'TX.VAL.AGRI.ZS.UN': 'Agricultural raw materials exports (% of merchandise exports)',
        'TX.VAL.FOOD.ZS.UN': 'Food exports (% of merchandise exports)',
        
        # Exchange rate
        'PA.NUS.FCRF': 'Official exchange rate (LCU per US$)',
        'PX.REX.REER': 'Real effective exchange rate index',
        
        # Governance (WGI)
        'CC.EST': 'Control of Corruption: Estimate (-2.5 to 2.5)',
        'GE.EST': 'Government Effectiveness: Estimate (-2.5 to 2.5)',
        'PV.EST': 'Political Stability: Estimate (-2.5 to 2.5)',
        'RQ.EST': 'Regulatory Quality: Estimate (-2.5 to 2.5)',
        'RL.EST': 'Rule of Law: Estimate (-2.5 to 2.5)',
        'VA.EST': 'Voice and Accountability: Estimate (-2.5 to 2.5)',
        
        # Other
        'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)',
        'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',
        'BN.CAB.XOKA.GD.ZS': 'Current account balance (% of GDP)',
        'DT.DOD.DECT.GN.ZS': 'External debt stocks (% of GNI)',
    }


# Example usage and test
if __name__ == "__main__":
    # Test with a few African countries
    test_countries = ['NGA', 'ZAF', 'EGY', 'KEN', 'GHA']
    
    print("🌍 Testing World Bank Data Collector...")
    print(f"Countries: {test_countries}")
    print(f"Period: 2010-2023")
    print("-" * 60)
    
    df = fetch_wdi_data(
        countries=test_countries,
        start_year=2010,
        end_year=2023,
        output_path='data/raw/worldbank_wdi.csv'
    )
    
    print("\n📊 Sample Data:")
    print(df.head(10))
    
    print("\n📈 Data Summary:")
    print(f"Total records: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n✅ Missing values per column:")
    print(df.isnull().sum())
    
    print("\n🔥 CDI Statistics:")
    print(df['cdi_raw'].describe())
    
    if 'governance_index' in df.columns:
        print("\n🏛️ Governance Index Statistics:")
        print(df['governance_index'].describe())
    
    if 'exchange_rate_volatility' in df.columns:
        print("\n💱 Exchange Rate Volatility Statistics:")
        print(df['exchange_rate_volatility'].describe())
