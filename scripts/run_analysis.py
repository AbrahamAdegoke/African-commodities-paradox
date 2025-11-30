"""
Complete Analysis Pipeline

This script runs the entire analysis pipeline from data download to model evaluation.
It orchestrates all steps of the African Commodities Paradox project.

Usage:
    # Basic usage with default settings
    python scripts/run_analysis.py
    
    # Custom countries and time period
    python scripts/run_analysis.py --countries NGA,ZAF,KEN --start-year 2000 --end-year 2023
    
    # Use a predefined subset
    python scripts/run_analysis.py --subset oil_exporters --start-year 1990
    
    # Skip data download (if already downloaded)
    python scripts/run_analysis.py --skip-download

Author: Abraham Adegoke
Date: November 2025
"""

import sys
from pathlib import Path
import argparse
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        AFRICAN COMMODITIES PARADOX ANALYSIS PIPELINE            ║
    ║                                                                  ║
    ║        A Data-Driven Tool for Economic Volatility Analysis      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_step(step_name: str, step_func, *args, **kwargs):
    """
    Run a pipeline step with timing and error handling.
    
    Args:
        step_name: Name of the step for logging
        step_func: Function to execute
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = step_func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"✅ {step_name} completed in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ {step_name} failed after {elapsed:.2f}s: {e}")
        return False


def step_download_data(countries, start_year, end_year, output_dir):
    """Step 1: Download data from World Bank."""
    import yaml
    
    # Add src to path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))
    
    from data_io.worldbank import fetch_wdi_data
    
    logger.info(f"📊 Downloading data for {len(countries)} countries ({start_year}-{end_year})")
    
    wdi_df = fetch_wdi_data(
        countries=countries,
        start_year=start_year,
        end_year=end_year,
        output_path=f'{output_dir}/worldbank_wdi.csv'
    )
    
    logger.info(f"✓ Downloaded {len(wdi_df)} records")
    return wdi_df


def step_preprocess_data(input_path, output_dir):
    """Step 2: Preprocess and clean data."""
    import pandas as pd
    
    logger.info(f"🧹 Preprocessing data from {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Basic cleaning (this will be expanded in preprocessing module)
    logger.info(f"  Initial shape: {df.shape}")
    
    # Remove rows with all NaN values in key indicators
    key_cols = ['gdp_growth', 'inflation', 'trade_openness', 'investment', 'cdi_raw']
    df_clean = df.dropna(subset=key_cols, how='all')
    
    logger.info(f"  After cleaning: {df_clean.shape}")
    logger.info(f"  Removed {len(df) - len(df_clean)} rows with missing data")
    
    # Save cleaned data
    output_path = f'{output_dir}/cleaned_data.csv'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    logger.info(f"✓ Saved cleaned data to {output_path}")
    return df_clean


def step_feature_engineering(input_path, output_dir):
    """Step 3: Engineer features (CDI smoothing, volatility calculation)."""
    import pandas as pd
    import numpy as np
    
    logger.info(f"⚙️  Engineering features from {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Sort by country and year
    df = df.sort_values(['country', 'year'])
    
    # 1. Apply 3-year moving average to CDI
    logger.info("  → Applying 3-year MA to CDI")
    df['cdi_smooth'] = df.groupby('country')['cdi_raw'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # 2. Calculate 5-year rolling volatility of GDP growth
    logger.info("  → Calculating 5-year GDP growth volatility")
    df['gdp_volatility'] = df.groupby('country')['gdp_growth'].transform(
        lambda x: x.rolling(window=5, min_periods=3).std()
    )
    
    # Log-transform volatility (as per proposal)
    df['log_gdp_volatility'] = np.log(df['gdp_volatility'] + 0.01)  # Add small constant to avoid log(0)
    
    # 3. Create lagged features (t-1)
    logger.info("  → Creating lagged features (t-1)")
    lag_features = ['cdi_smooth', 'inflation', 'trade_openness', 'investment']
    
    for feature in lag_features:
        df[f'{feature}_lag1'] = df.groupby('country')[feature].shift(1)
    
    # Drop rows with NaN in target variable
    df_features = df.dropna(subset=['log_gdp_volatility'])
    
    logger.info(f"  Final feature set shape: {df_features.shape}")
    logger.info(f"  Features: {df_features.columns.tolist()}")
    
    # Save feature-engineered data
    output_path = f'{output_dir}/features_ready.csv'
    df_features.to_csv(output_path, index=False)
    
    logger.info(f"✓ Saved feature-ready data to {output_path}")
    return df_features


def step_summary_stats(df):
    """Print summary statistics."""
    import pandas as pd
    
    logger.info("\n📈 SUMMARY STATISTICS")
    logger.info("="*70)
    
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Countries: {df['country'].nunique()}")
    logger.info(f"Years: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"Total observations: {len(df)}")
    
    logger.info("\n📊 Key Variables:")
    key_vars = ['cdi_smooth', 'gdp_growth', 'log_gdp_volatility', 'inflation', 'investment']
    stats = df[key_vars].describe()
    print(stats)
    
    logger.info("\n🌍 Countries in dataset:")
    countries = df.groupby('country')['year'].count().sort_values(ascending=False)
    print(countries)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Run complete African Commodities Paradox analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with default settings
  python scripts/run_analysis.py
  
  # Analyze specific countries
  python scripts/run_analysis.py --countries NGA,ZAF,KEN,GHA,EGY
  
  # Use predefined subset
  python scripts/run_analysis.py --subset oil_exporters --start-year 1995
  
  # Skip data download (use existing data)
  python scripts/run_analysis.py --skip-download
        """
    )
    
    # Country selection
    country_group = parser.add_mutually_exclusive_group()
    country_group.add_argument(
        '--subset',
        type=str,
        default='high_quality_data',
        help='Country subset from configs/countries.yaml'
    )
    country_group.add_argument(
        '--countries',
        type=str,
        help='Comma-separated ISO3 country codes (e.g., NGA,ZAF,KEN)'
    )
    
    # Time period
    parser.add_argument('--start-year', type=int, default=1990, help='Start year')
    parser.add_argument('--end-year', type=int, default=2023, help='End year')
    
    # Pipeline options
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip data download step (use existing data)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine countries
    if args.countries:
        countries = [c.strip().upper() for c in args.countries.split(',')]
        logger.info(f"Countries: {', '.join(countries)}")
    else:
        import yaml
        with open('configs/countries.yaml', 'r') as f:
            config = yaml.safe_load(f)
        countries = config.get(args.subset, config['high_quality_data'])
        logger.info(f"Subset: {args.subset} ({len(countries)} countries)")
    
    logger.info(f"Time period: {args.start_year} - {args.end_year}")
    
    # Pipeline execution
    pipeline_start = time.time()
    
    # Step 1: Download data (optional)
    if not args.skip_download:
        success = run_step(
            "1. Data Download",
            step_download_data,
            countries, args.start_year, args.end_year, f'{args.output_dir}/raw'
        )
        if not success:
            logger.error("Pipeline failed at data download step")
            sys.exit(1)
    else:
        logger.info("⏭️  Skipping data download (using existing data)")
    
    # Step 2: Preprocessing
    success = run_step(
        "2. Data Preprocessing",
        step_preprocess_data,
        f'{args.output_dir}/raw/worldbank_wdi.csv',
        f'{args.output_dir}/processed'
    )
    if not success:
        logger.error("Pipeline failed at preprocessing step")
        sys.exit(1)
    
    # Step 3: Feature Engineering
    df_features = None
    success = run_step(
        "3. Feature Engineering",
        step_feature_engineering,
        f'{args.output_dir}/processed/cleaned_data.csv',
        f'{args.output_dir}/processed'
    )
    if success:
        import pandas as pd
        df_features = pd.read_csv(f'{args.output_dir}/processed/features_ready.csv')
    else:
        logger.error("Pipeline failed at feature engineering step")
        sys.exit(1)
    
    # Step 4: Summary statistics
    run_step("4. Summary Statistics", step_summary_stats, df_features)
    
    # Pipeline complete
    total_time = time.time() - pipeline_start
    
    logger.info("\n" + "="*70)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"\n📁 Output files:")
    logger.info(f"  → Raw data: {args.output_dir}/raw/worldbank_wdi.csv")
    logger.info(f"  → Cleaned data: {args.output_dir}/processed/cleaned_data.csv")
    logger.info(f"  → Features ready: {args.output_dir}/processed/features_ready.csv")
    logger.info(f"\n💡 Next steps:")
    logger.info(f"  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    logger.info(f"  2. Train models: python scripts/train_models.py")
    logger.info(f"  3. Evaluate: python scripts/evaluate_models.py")


if __name__ == "__main__":
    main()