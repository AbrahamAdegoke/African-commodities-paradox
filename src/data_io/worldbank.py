"""
World Bank Data Collector Module

This module fetches economic indicators from the World Bank WDI API
for African countries to analyze the commodities paradox.

Author: Abraham Adegoke
Date: November 2025
"""

import pandas as pd
import requests
from typing import List, Dict, Optional
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorldBankAPI:
    """
    Client for interacting with World Bank WDI API v2.
    
    Base URL: https://api.worldbank.org/v2/
    Documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
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
                logger.error(f"Error fetching {indicator} for {country}: {e}")
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


def fetch_wdi_data(
    countries: List[str],
    start_year: int = 1990,
    end_year: int = 2023,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Main function to fetch all WDI indicators needed for the project.
    
    Indicators fetched:
    - GDP growth (annual %)
    - Inflation, consumer prices (annual %)
    - Trade openness ((Exports + Imports) / GDP)
    - Gross capital formation (% of GDP) - proxy for investment
    - Fuel exports (% of merchandise exports)
    - Ores and metals exports (% of merchandise exports)
    - Food exports (% of merchandise exports)
    
    Args:
        countries: List of ISO3 country codes
        start_year: Start year (default: 1990)
        end_year: End year (default: 2023)
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with all indicators
    """
    
    # Define indicators
    indicators = {
        # Core macroeconomic indicators
        'NY.GDP.MKTP.KD.ZG': 'gdp_growth',              # GDP growth (annual %)
        'FP.CPI.TOTL.ZG': 'inflation',                  # Inflation, consumer prices (annual %)
        'NE.TRD.GNFS.ZS': 'trade_openness',             # Trade (% of GDP)
        'NE.GDI.TOTL.ZS': 'investment',                 # Gross capital formation (% of GDP)
        
        # Commodity export indicators (for CDI calculation)
        'TX.VAL.FUEL.ZS.UN': 'fuel_exports_pct',        # Fuel exports (% of merchandise exports)
        'TX.VAL.MMLS.ZS.UN': 'metals_exports_pct',      # Ores and metals exports (% of merch exports)
        'TX.VAL.FOOD.ZS.UN': 'food_exports_pct',        # Food exports (% of merchandise exports)
        
        # Additional useful indicators
        'NE.EXP.GNFS.ZS': 'exports_gdp',                # Exports of goods and services (% of GDP)
        'NE.IMP.GNFS.ZS': 'imports_gdp',                # Imports of goods and services (% of GDP)
    }
    
    # Initialize API client
    api = WorldBankAPI()
    
    # Fetch all indicators
    df = api.fetch_multiple_indicators(
        countries=countries,
        indicators=indicators,
        start_year=start_year,
        end_year=end_year
    )
    
    # Calculate CDI (Commodity Dependence Index)
    # CDI = sum of fuel + metals + food exports as % of total merchandise exports
    # Handle missing columns gracefully (some indicators may return no data)
    fuel = df['fuel_exports_pct'].fillna(0) if 'fuel_exports_pct' in df.columns else 0
    metals = df['metals_exports_pct'].fillna(0) if 'metals_exports_pct' in df.columns else 0
    food = df['food_exports_pct'].fillna(0) if 'food_exports_pct' in df.columns else 0
    
    df['cdi_raw'] = fuel + metals + food
    
    # Log which commodity components are available
    available_components = []
    if 'fuel_exports_pct' in df.columns:
        available_components.append('fuel')
    if 'metals_exports_pct' in df.columns:
        available_components.append('metals')
    if 'food_exports_pct' in df.columns:
        available_components.append('food')
    
    logger.info(f"CDI components available: {', '.join(available_components)}")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Data saved to {output_path}")
    
    return df


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