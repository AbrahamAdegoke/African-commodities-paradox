"""
Data Download Script

This script orchestrates the download of all data sources needed for the project:
- World Bank WDI indicators
- (Future: UNCTAD commodity data)
- (Future: World Governance Indicators)

Run this script to fetch all raw data.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --subset high_quality  # Use subset of countries
    python scripts/download_data.py --start-year 2000 --end-year 2023

Author: Abraham Adegoke
Date: November 2025
"""

import sys
from pathlib import Path
import argparse
import yaml
import logging

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data_io.worldbank import fetch_wdi_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_countries(config_path: str = 'configs/countries.yaml', subset: str = 'all_countries') -> list:
    """
    Load country list from YAML config.
    
    Args:
        config_path: Path to countries.yaml
        subset: Which country group to use (e.g., 'all_countries', 'high_quality_data')
        
    Returns:
        List of ISO3 country codes
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if subset not in config:
        logger.warning(f"Subset '{subset}' not found, using 'all_countries'")
        subset = 'all_countries'
    
    countries = config[subset]
    logger.info(f"✓ Loaded {len(countries)} countries from '{subset}' group")
    
    return countries


def download_all_data(
    countries: list,
    start_year: int = 1990,
    end_year: int = 2023,
    output_dir: str = 'data/raw'
):
    """
    Download all data sources.
    
    Args:
        countries: List of ISO3 country codes
        start_year: Start year for data
        end_year: End year for data
        output_dir: Directory to save raw data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("🌍 AFRICAN COMMODITIES PARADOX - DATA DOWNLOAD")
    logger.info("=" * 70)
    logger.info(f"Countries: {len(countries)}")
    logger.info(f"Period: {start_year}-{end_year}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)
    
    # 1. Download World Bank WDI data
    logger.info("\n📊 Step 1/3: Downloading World Bank WDI data...")
    wdi_path = output_path / 'worldbank_wdi.csv'
    
    try:
        wdi_df = fetch_wdi_data(
            countries=countries,
            start_year=start_year,
            end_year=end_year,
            output_path=str(wdi_path)
        )
        logger.info(f"✅ World Bank WDI data downloaded: {wdi_path}")
        logger.info(f"   Records: {len(wdi_df)}")
        logger.info(f"   Columns: {list(wdi_df.columns)}")
    except Exception as e:
        logger.error(f"❌ Failed to download WDI data: {e}")
        return False
    
    # 2. Future: Download UNCTAD commodity data
    logger.info("\n📦 Step 2/3: UNCTAD commodity data (TODO)")
    logger.info("   → Will be implemented in src/data_io/unctad.py")
    
    # 3. Future: Download World Governance Indicators
    logger.info("\n🏛️  Step 3/3: World Governance Indicators (TODO)")
    logger.info("   → Will be implemented in src/data_io/wgi.py")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ DATA DOWNLOAD COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\n📁 Downloaded files in: {output_dir}/")
    logger.info(f"   - worldbank_wdi.csv ({len(wdi_df)} records)")
    logger.info("\n💡 Next steps:")
    logger.info("   1. Run: python scripts/preprocess_data.py")
    logger.info("   2. Explore data in notebooks/01_data_exploration.ipynb")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download all data for African Commodities Paradox project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined subset
  python scripts/download_data.py --subset oil_exporters
  
  # Custom list of countries
  python scripts/download_data.py --countries NGA,ZAF,KEN,GHA
  
  # All African countries
  python scripts/download_data.py --subset all_countries --start-year 1990 --end-year 2023
        """
    )
    
    # Country selection (mutually exclusive)
    country_group = parser.add_mutually_exclusive_group()
    country_group.add_argument(
        '--subset',
        type=str,
        default='high_quality_data',
        help='Country subset from configs/countries.yaml (default: high_quality_data)'
    )
    country_group.add_argument(
        '--countries',
        type=str,
        help='Comma-separated list of ISO3 country codes (e.g., NGA,ZAF,KEN)'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=1990,
        help='Start year (default: 1990)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2023,
        help='End year (default: 2023)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for raw data (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    # Load countries based on input
    if args.countries:
        # Custom list provided
        countries = [c.strip().upper() for c in args.countries.split(',')]
        logger.info(f"✓ Using custom country list: {countries}")
    else:
        # Use subset from YAML
        countries = load_countries(subset=args.subset)
    
    # Download data
    success = download_all_data(
        countries=countries,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir
    )
    
    if success:
        logger.info("\n🎉 All done! Data is ready for preprocessing.")
        sys.exit(0)
    else:
        logger.error("\n❌ Data download failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()