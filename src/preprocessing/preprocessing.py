"""
Advanced Data Preprocessing Module

Handles missing values, outliers, feature engineering, and data validation 
for the African Commodities Paradox project.

Includes:
- CDI smoothing (3-year moving average)
- GDP growth volatility calculation (5-year rolling std)
- Exchange rate volatility
- Governance index processing
- Lagged features creation

Author: Abraham Adegoke
Date: November 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles preprocessing of economic data for modeling.
    
    Methods:
        - handle_missing_values: Impute or drop missing data
        - remove_outliers: Detect and handle statistical outliers
        - validate_data: Check data quality and integrity
        - prepare_for_modeling: Complete preprocessing pipeline
    """
    
    def __init__(self, strategy: str = 'impute'):
        """
        Initialize preprocessor.
        
        Args:
            strategy: Strategy for missing values ('impute' or 'drop')
        """
        self.strategy = strategy
        self.imputation_values = {}
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        method: str = 'median'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            method: Imputation method ('median', 'mean', 'forward_fill')
        
        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values using strategy: {self.strategy}, method: {method}")
        
        df_clean = df.copy()
        
        if self.strategy == 'drop':
            # Drop rows with any missing values in key features
            key_features = ['cdi_smooth_lag1', 'inflation_lag1', 
                           'trade_openness_lag1', 'investment_lag1', 
                           'log_gdp_volatility']
            initial_len = len(df_clean)
            df_clean = df_clean.dropna(subset=key_features)
            logger.info(f"Dropped {initial_len - len(df_clean)} rows with missing values")
            
        elif self.strategy == 'impute':
            # Impute missing values by country (country-specific median/mean)
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    if method == 'median':
                        # Group by country and fill with country median
                        df_clean[col] = df_clean.groupby('country')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                        # If still NaN (country has all NaN), use global median
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        
                    elif method == 'mean':
                        df_clean[col] = df_clean.groupby('country')[col].transform(
                            lambda x: x.fillna(x.mean())
                        )
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                        
                    elif method == 'forward_fill':
                        # Forward fill within each country
                        df_clean = df_clean.sort_values(['country', 'year'])
                        df_clean[col] = df_clean.groupby('country')[col].ffill()
                        df_clean[col] = df_clean.groupby('country')[col].bfill()
                    
                    # Store imputation value for reporting
                    self.imputation_values[col] = df_clean[col].median()
        
        # Final check: drop any remaining rows with NaN in target
        if 'log_gdp_volatility' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['log_gdp_volatility'])
        
        logger.info(f"Final dataset shape after handling missing values: {df_clean.shape}")
        
        return df_clean
    
    def remove_outliers(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove or cap outliers in specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (None = all numeric)
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Removing outliers using {method} method")
        
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        initial_len = len(df_clean)
        
        for col in columns:
            if col in df_clean.columns:
                if method == 'iqr':
                    # Interquartile Range method
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Cap outliers instead of removing
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    
                elif method == 'zscore':
                    # Z-score method
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < threshold]
        
        logger.info(f"Outliers handled. Rows affected: {initial_len - len(df_clean)}")
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data quality and integrity.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data quality...")
        
        validation_results = {}
        
        # Check 1: Required columns present
        required_cols = ['country', 'year', 'log_gdp_volatility']
        validation_results['has_required_columns'] = all(col in df.columns for col in required_cols)
        
        # Check 2: No duplicate country-year pairs
        validation_results['no_duplicates'] = not df.duplicated(subset=['country', 'year']).any()
        
        # Check 3: Reasonable value ranges
        checks = []
        if 'log_gdp_volatility' in df.columns:
            checks.append((df['log_gdp_volatility'] > -10).all())
            checks.append((df['log_gdp_volatility'] < 10).all())
        if 'cdi_smooth_lag1' in df.columns:
            checks.append((df['cdi_smooth_lag1'] >= 0).all())
            checks.append((df['cdi_smooth_lag1'] <= 100).all())
        validation_results['valid_ranges'] = all(checks) if checks else True
        
        # Check 4: Sufficient observations per country
        country_counts = df.groupby('country').size()
        validation_results['sufficient_data'] = (country_counts >= 3).all()
        
        # Check 5: No extreme missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        validation_results['acceptable_missing'] = (missing_pct < 50).all()
        
        # Log results
        for check, passed in validation_results.items():
            status = "✓" if passed else "✗"
            logger.info(f"{status} {check}: {passed}")
        
        return validation_results
    
    def prepare_for_modeling(
        self, 
        df: pd.DataFrame,
        target: str = 'log_gdp_volatility',
        feature_cols: Optional[List[str]] = None,
        include_governance: bool = True,
        include_exchange_rate: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Complete preprocessing pipeline for modeling.
        
        Args:
            df: Input DataFrame
            target: Target variable column name
            feature_cols: Feature columns (None = auto-select)
            include_governance: Include governance index in features
            include_exchange_rate: Include exchange rate volatility in features
        
        Returns:
            Tuple of (X_features, y_target, feature_names)
        """
        logger.info("=" * 70)
        logger.info("PREPARING DATA FOR MODELING")
        logger.info("=" * 70)
        
        # Step 1: Handle missing values
        df_clean = self.handle_missing_values(df, method='median')
        
        # Step 2: Define features
        if feature_cols is None:
            feature_cols = [
                'cdi_smooth_lag1',
                'inflation_lag1',
                'trade_openness_lag1',
                'investment_lag1'
            ]
            
            # Add governance if available and requested
            if include_governance and 'governance_index_lag1' in df_clean.columns:
                feature_cols.append('governance_index_lag1')
            
            # Add exchange rate volatility if available and requested
            if include_exchange_rate and 'exchange_rate_volatility_lag1' in df_clean.columns:
                feature_cols.append('exchange_rate_volatility_lag1')
        
        # Filter to only available features
        feature_cols = [f for f in feature_cols if f in df_clean.columns]
        
        # Step 3: Validate
        validation = self.validate_data(df_clean)
        if not all(validation.values()):
            logger.warning("Some validation checks failed!")
        
        # Step 4: Extract features and target
        # Remove rows with any NaN in features
        df_model = df_clean[feature_cols + [target]].dropna()
        
        X = df_model[feature_cols]
        y = df_model[target]
        
        logger.info(f"\n✓ Preprocessing complete!")
        logger.info(f"  Final dataset shape: {df_model.shape}")
        logger.info(f"  Features: {feature_cols}")
        logger.info(f"  Target: {target}")
        logger.info(f"  Total observations: {len(X)}")
        logger.info(f"  Missing values in X: {X.isnull().sum().sum()}")
        logger.info(f"  Missing values in y: {y.isnull().sum()}")
        
        return X, y, feature_cols


class FeatureEngineer:
    """
    Handles feature engineering for the African Commodities Paradox project.
    
    Creates:
        - CDI smoothing (3-year moving average)
        - GDP growth volatility (5-year rolling std)
        - Lagged features (t-1)
        - Governance composite index
        - Exchange rate volatility
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features needed for modeling.
        
        Args:
            df: Raw data DataFrame with country, year, and indicators
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 70)
        
        df = df.sort_values(['country', 'year']).copy()
        
        # 1. Smooth CDI with 3-year moving average
        df = self._create_cdi_smooth(df)
        
        # 2. Calculate GDP growth volatility
        df = self._create_gdp_volatility(df)
        
        # 3. Create governance composite index (if not already present)
        df = self._create_governance_index(df)
        
        # 4. Create lagged features
        df = self._create_lagged_features(df)
        
        logger.info(f"\n✓ Feature engineering complete!")
        logger.info(f"  Final shape: {df.shape}")
        logger.info(f"  New columns: {[c for c in df.columns if 'lag1' in c or 'smooth' in c or 'volatility' in c]}")
        
        return df
    
    def _create_cdi_smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply 3-year moving average to CDI."""
        logger.info("📊 Creating smoothed CDI (3-year MA)...")
        
        if 'cdi_raw' in df.columns:
            df['cdi_smooth'] = df.groupby('country')['cdi_raw'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            logger.info("  ✓ cdi_smooth created")
        else:
            logger.warning("  ⚠ cdi_raw not found")
        
        return df
    
    def _create_gdp_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 5-year rolling volatility of GDP growth."""
        logger.info("📈 Creating GDP growth volatility (5-year rolling std)...")
        
        if 'gdp_growth' in df.columns:
            df['gdp_volatility'] = df.groupby('country')['gdp_growth'].transform(
                lambda x: x.rolling(window=5, min_periods=3).std()
            )
            # Log-transform volatility (as per proposal)
            # Add small constant to avoid log(0)
            df['log_gdp_volatility'] = np.log(df['gdp_volatility'] + 0.01)
            logger.info("  ✓ gdp_volatility and log_gdp_volatility created")
        else:
            logger.warning("  ⚠ gdp_growth not found")
        
        return df
    
    def _create_governance_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite governance index from WGI indicators."""
        logger.info("🏛️  Creating governance index...")
        
        governance_cols = [
            'control_corruption', 'govt_effectiveness', 'political_stability',
            'regulatory_quality', 'rule_of_law', 'voice_accountability'
        ]
        
        available_gov = [col for col in governance_cols if col in df.columns]
        
        if available_gov and 'governance_index' not in df.columns:
            df['governance_index'] = df[available_gov].mean(axis=1)
            logger.info(f"  ✓ governance_index created from {len(available_gov)} indicators")
        elif 'governance_index' in df.columns:
            logger.info("  ✓ governance_index already exists")
        else:
            logger.warning("  ⚠ No governance indicators available")
        
        return df
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features (t-1) to avoid simultaneity."""
        logger.info("🔄 Creating lagged features (t-1)...")
        
        # Features to lag
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
        
        logger.info(f"  ✓ Created {lagged_count} lagged features")
        
        return df


def load_and_preprocess(
    data_path: str = 'data/processed/features_ready.csv',
    strategy: str = 'impute',
    include_governance: bool = True,
    include_exchange_rate: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Convenience function to load and preprocess data in one step.
    
    Args:
        data_path: Path to feature-ready CSV
        strategy: Missing value strategy ('impute' or 'drop')
        include_governance: Include governance index in features
        include_exchange_rate: Include exchange rate volatility in features
    
    Returns:
        Tuple of (X_features, y_target, feature_names)
    """
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    preprocessor = DataPreprocessor(strategy=strategy)
    X, y, features = preprocessor.prepare_for_modeling(
        df,
        include_governance=include_governance,
        include_exchange_rate=include_exchange_rate
    )
    
    return X, y, features


def create_features_from_raw(
    raw_data_path: str = 'data/raw/worldbank_wdi.csv',
    output_path: str = 'data/processed/features_ready.csv'
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline from raw data.
    
    Args:
        raw_data_path: Path to raw World Bank data
        output_path: Path to save feature-ready data
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Create all features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Save
    df_features.to_csv(output_path, index=False)
    logger.info(f"✓ Features saved to: {output_path}")
    
    return df_features


# Example usage
if __name__ == "__main__":
    # Test preprocessing
    print("Testing Data Preprocessor...")
    print("=" * 70)
    
    # Test with sample data
    try:
        X, y, features = load_and_preprocess('data/processed/features_ready.csv')
        
        print("\n✓ Preprocessing test successful!")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Features: {features}")
        print(f"\n  X summary:\n{X.describe()}")
        print(f"\n  y summary:\n{y.describe()}")
    except FileNotFoundError:
        print("⚠ No data file found. Run the data download first.")
