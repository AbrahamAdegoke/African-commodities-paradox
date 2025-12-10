"""
Data Preprocessing Script

Transforms raw World Bank data into model-ready features including:
- CDI smoothing (3-year moving average)
- GDP growth volatility (5-year rolling std)
- Governance index
- Exchange rate volatility
- Lagged features (t-1)

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --input data/raw/worldbank_wdi.csv --output data/processed/features_ready.csv

Author: Abraham Adegoke
Date: December 2025
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print preprocessing banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          AFRICAN COMMODITIES PARADOX - DATA PREPROCESSING       ║
    ║                                                                  ║
    ║     Creating features: CDI, Volatility, Governance, Lags        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features needed for modeling.
    
    Args:
        df: Raw data with country, year, and indicators
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    df = df.sort_values(['country', 'year']).copy()
    
    # =========================================
    # 1. CDI Smoothing (3-year moving average)
    # =========================================
    logger.info("\n📊 Creating smoothed CDI (3-year MA)...")
    
    if 'cdi_raw' in df.columns:
        df['cdi_smooth'] = df.groupby('country')['cdi_raw'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        logger.info(f"  ✓ cdi_smooth created: range {df['cdi_smooth'].min():.1f}% - {df['cdi_smooth'].max():.1f}%")
    else:
        logger.warning("  ⚠ cdi_raw not found - cannot create cdi_smooth")
    
    # =========================================
    # 2. GDP Growth Volatility (5-year rolling std)
    # =========================================
    logger.info("\n📈 Creating GDP growth volatility (5-year rolling std)...")
    
    if 'gdp_growth' in df.columns:
        df['gdp_volatility'] = df.groupby('country')['gdp_growth'].transform(
            lambda x: x.rolling(window=5, min_periods=3).std()
        )
        # Log-transform volatility (as per proposal)
        df['log_gdp_volatility'] = np.log(df['gdp_volatility'] + 0.01)
        logger.info(f"  ✓ gdp_volatility created: range {df['gdp_volatility'].min():.2f} - {df['gdp_volatility'].max():.2f}")
        logger.info(f"  ✓ log_gdp_volatility created")
    else:
        logger.warning("  ⚠ gdp_growth not found - cannot create volatility")
    
    # =========================================
    # 3. Governance Index (if not already present)
    # =========================================
    logger.info("\n🏛️  Processing Governance Index...")
    
    governance_cols = [
        'control_corruption', 'govt_effectiveness', 'political_stability',
        'regulatory_quality', 'rule_of_law', 'voice_accountability'
    ]
    
    available_gov = [col for col in governance_cols if col in df.columns]
    
    if available_gov and 'governance_index' not in df.columns:
        df['governance_index'] = df[available_gov].mean(axis=1)
        logger.info(f"  ✓ governance_index created from {len(available_gov)} indicators")
    elif 'governance_index' in df.columns:
        logger.info(f"  ✓ governance_index already exists: range {df['governance_index'].min():.2f} to {df['governance_index'].max():.2f}")
    else:
        logger.warning("  ⚠ No governance indicators available")
    
    # =========================================
    # 4. Exchange Rate Volatility (if not already present)
    # =========================================
    logger.info("\n💱 Processing Exchange Rate Volatility...")
    
    if 'exchange_rate_volatility' in df.columns:
        logger.info(f"  ✓ exchange_rate_volatility already exists")
    elif 'exchange_rate' in df.columns:
        df['exchange_rate_change'] = df.groupby('country')['exchange_rate'].pct_change() * 100
        df['exchange_rate_volatility'] = df.groupby('country')['exchange_rate_change'].transform(
            lambda x: x.rolling(window=3, min_periods=2).std()
        )
        logger.info(f"  ✓ exchange_rate_volatility created")
    else:
        logger.warning("  ⚠ exchange_rate not found")
    
    # =========================================
    # 5. Create Lagged Features (t-1)
    # =========================================
    logger.info("\n🔄 Creating lagged features (t-1)...")
    
    features_to_lag = [
        'cdi_smooth',
        'inflation',
        'trade_openness',
        'investment',
        'governance_index',
        'exchange_rate_volatility'
    ]
    
    lagged_count = 0
    for feature in features_to_lag:
        if feature in df.columns:
            df[f'{feature}_lag1'] = df.groupby('country')[feature].shift(1)
            lagged_count += 1
            logger.info(f"  ✓ {feature}_lag1 created")
    
    logger.info(f"\n  Total lagged features created: {lagged_count}")
    
    return df


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess data for African Commodities Paradox')
    parser.add_argument('--input', type=str, default='data/raw/worldbank_wdi.csv',
                        help='Input raw data CSV')
    parser.add_argument('--output', type=str, default='data/processed/features_ready.csv',
                        help='Output processed data CSV')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Load raw data
    logger.info(f"Loading raw data from: {args.input}")
    
    try:
        df = pd.read_csv(args.input)
        logger.info(f"  ✓ Loaded {len(df)} records, {len(df.columns)} columns")
        logger.info(f"  Countries: {df['country'].nunique()}")
        logger.info(f"  Years: {df['year'].min()} - {df['year'].max()}")
    except FileNotFoundError:
        logger.error(f"File not found: {args.input}")
        logger.error("Please run: python scripts/download_data.py --subset all_countries --start-year 1990")
        sys.exit(1)
    
    # Create features
    df_features = create_features(df)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 70)
    
    # Check which features are available
    expected_features = [
        'cdi_smooth_lag1', 'inflation_lag1', 'trade_openness_lag1', 
        'investment_lag1', 'governance_index_lag1', 'exchange_rate_volatility_lag1',
        'log_gdp_volatility'
    ]
    
    available = [f for f in expected_features if f in df_features.columns]
    missing = [f for f in expected_features if f not in df_features.columns]
    
    logger.info(f"\n✅ Available features ({len(available)}):")
    for f in available:
        non_null = df_features[f].notna().sum()
        logger.info(f"   - {f}: {non_null} non-null values")
    
    if missing:
        logger.info(f"\n⚠️  Missing features ({len(missing)}):")
        for f in missing:
            logger.info(f"   - {f}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Features saved to: {args.output}")
    logger.info(f"  Total records: {len(df_features)}")
    logger.info(f"  Total columns: {len(df_features.columns)}")
    
    # Count usable observations (non-null target and key features)
    key_features = ['cdi_smooth_lag1', 'inflation_lag1', 'trade_openness_lag1', 
                    'investment_lag1', 'log_gdp_volatility']
    key_features = [f for f in key_features if f in df_features.columns]
    
    usable = df_features[key_features].dropna()
    logger.info(f"  Usable observations (no missing in key features): {len(usable)}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info("\n💡 Next step:")
    logger.info("   python scripts/train_models.py --include-shap --include-forecast")


if __name__ == "__main__":
    main()