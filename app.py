"""
African Commodities Paradox - Interactive Web Application

A Streamlit web interface for analyzing commodity dependence and economic volatility
in African countries. Provides interactive visualizations, GDP forecasting, and
unsupervised learning analysis (clustering and PCA).

Usage:
    streamlit run app.py

Author: Abraham Adegoke
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.gradient_boosting import GradientBoostingModel
from models.gdp_forecaster import GDPGrowthForecaster
from preprocessing.preprocessing import DataPreprocessor

# Unsupervised Learning imports
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats

# Time Series imports (with error handling)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="African Commodities Paradox Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Country mapping (ISO3 to full names)
COUNTRY_NAMES_STATIC = {
    'DZA': 'Algeria', 'EGY': 'Egypt', 'LBY': 'Libya', 'MAR': 'Morocco', 'TUN': 'Tunisia',
    'BEN': 'Benin', 'BFA': 'Burkina Faso', 'CPV': 'Cape Verde', 'CIV': "Côte d'Ivoire",
    'GMB': 'Gambia', 'GHA': 'Ghana', 'GIN': 'Guinea', 'GNB': 'Guinea-Bissau',
    'LBR': 'Liberia', 'MLI': 'Mali', 'MRT': 'Mauritania', 'NER': 'Niger',
    'NGA': 'Nigeria', 'SEN': 'Senegal', 'SLE': 'Sierra Leone', 'TGO': 'Togo',
    'AGO': 'Angola', 'CMR': 'Cameroon', 'CAF': 'Central African Republic',
    'TCD': 'Chad', 'COG': 'Congo Republic', 'COD': 'DR Congo',
    'GNQ': 'Equatorial Guinea', 'GAB': 'Gabon',
    'BDI': 'Burundi', 'COM': 'Comoros', 'DJI': 'Djibouti', 'ERI': 'Eritrea',
    'ETH': 'Ethiopia', 'KEN': 'Kenya', 'MDG': 'Madagascar', 'MWI': 'Malawi',
    'MUS': 'Mauritius', 'MOZ': 'Mozambique', 'RWA': 'Rwanda', 'SYC': 'Seychelles',
    'SOM': 'Somalia', 'SSD': 'South Sudan', 'TZA': 'Tanzania', 'UGA': 'Uganda',
    'BWA': 'Botswana', 'LSO': 'Lesotho', 'NAM': 'Namibia', 'ZAF': 'South Africa',
    'SWZ': 'Eswatini', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
}


# Countries with known data quality issues (conflicts, new nations, etc.)
COUNTRIES_WITH_CAUTION = {
    'SSD': {'name': 'South Sudan', 'reason': 'Civil war since 2013, nation created in 2011'},
    'SOM': {'name': 'Somalia', 'reason': 'Ongoing conflict, limited government control'},
    'ERI': {'name': 'Eritrea', 'reason': 'Limited data availability, authoritarian regime'},
    'LBY': {'name': 'Libya', 'reason': 'Civil war since 2011'},
    'CAF': {'name': 'Central African Republic', 'reason': 'Ongoing armed conflict'},
    'TCD': {'name': 'Chad', 'reason': 'Political instability, conflict zones'},
    'COD': {'name': 'DR Congo', 'reason': 'Ongoing conflicts in eastern regions'},
    'SDN': {'name': 'Sudan', 'reason': 'Civil war, political instability'},
    'GNQ': {'name': 'Equatorial Guinea', 'reason': 'Limited data transparency'},
}


def assess_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assess data quality for each country.
    
    Returns DataFrame with quality metrics per country.
    """
    quality_metrics = []
    
    key_columns = ['gdp_growth', 'cdi_smooth', 'cdi_raw', 'inflation', 
                   'investment', 'governance_index', 'trade_openness']
    available_cols = [c for c in key_columns if c in df.columns]
    
    for country in df['country'].unique():
        country_df = df[df['country'] == country]
        
        # Count years of data
        n_years = len(country_df)
        year_range = f"{int(country_df['year'].min())}-{int(country_df['year'].max())}"
        
        # Calculate missing data percentage
        if available_cols:
            missing_pct = country_df[available_cols].isnull().mean().mean() * 100
        else:
            missing_pct = 0
        
        # Check if country has known issues
        has_caution = country in COUNTRIES_WITH_CAUTION
        caution_reason = COUNTRIES_WITH_CAUTION.get(country, {}).get('reason', '')
        
        # Calculate quality score (0-100)
        quality_score = 100
        quality_score -= max(0, (10 - n_years) * 5)  # Penalize < 10 years
        quality_score -= missing_pct  # Penalize missing data
        if has_caution:
            quality_score -= 20  # Penalize known issues
        quality_score = max(0, min(100, quality_score))
        
        # Determine quality level
        if quality_score >= 70:
            quality_level = "Good"
        elif quality_score >= 50:
            quality_level = "Moderate"
        else:
            quality_level = "Low"
        
        quality_metrics.append({
            'country': country,
            'n_years': n_years,
            'year_range': year_range,
            'missing_pct': round(missing_pct, 1),
            'has_caution': has_caution,
            'caution_reason': caution_reason,
            'quality_score': round(quality_score, 0),
            'quality_level': quality_level
        })
    
    return pd.DataFrame(quality_metrics)


@st.cache_data
def load_data():
    """Load processed data and create country mappings."""
    try:
        df = pd.read_csv('data/processed/features_ready.csv')
        
        # Create country mappings from actual data
        unique_countries = df['country'].unique()
        
        # Use static names if available, otherwise use ISO code
        country_names = {
            code: COUNTRY_NAMES_STATIC.get(code, code)
            for code in unique_countries
        }
        
        # Add country_name if it exists in the dataframe
        if 'country_name' in df.columns:
            for _, row in df[['country', 'country_name']].drop_duplicates().iterrows():
                if pd.notna(row['country_name']):
                    country_names[row['country']] = row['country_name']
        
        country_codes = {v: k for k, v in country_names.items()}
        
        return df, country_names, country_codes
        
    except FileNotFoundError:
        st.error("❌ Data not found. Please run: python scripts/run_analysis.py")
        return None, {}, {}


@st.cache_resource
def load_models():
    """Load trained models."""
    volatility_model = None
    gdp_model = None
    
    try:
        volatility_model = GradientBoostingModel.load('results/models/gbr_model.pkl')
    except:
        try:
            volatility_model = GradientBoostingModel.load('results/gbr_model.pkl')
        except:
            pass
    
    try:
        gdp_model = GDPGrowthForecaster.load('results/forecast/gdp_forecaster.pkl')
    except:
        try:
            gdp_model = GDPGrowthForecaster.load('results/gdp_forecaster.pkl')
        except:
            pass
            
    return volatility_model, gdp_model


def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 African Commodities Paradox Analyzer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Analyze the relationship between commodity dependence and economic volatility in African countries.**
    
    This tool uses machine learning to predict GDP growth volatility and forecast future economic performance.
    """)
    
    # Load data and models
    df, country_names, country_codes = load_data()
    
    if df is None or not country_names:
        st.error("❌ Could not load data or country mappings")
        st.stop()
    
    volatility_model, gdp_model = load_models()
    
    # Sidebar - User inputs
    st.sidebar.title("⚙️ Analysis Settings")
    
    # Analysis mode - WITH UNSUPERVISED LEARNING AND TIME SERIES
    mode = st.sidebar.radio(
        "Select Analysis Mode:",
        ["📊 Country Analysis", "🔮 GDP Forecasting", "📈 Time Series Analysis", "🔬 Unsupervised Analysis", "🌍 Regional Comparison"]
    )
    
    # Country selection (not needed for Unsupervised Analysis)
    if mode == "🌍 Regional Comparison":
        selected_countries = st.sidebar.multiselect(
            "Select countries to compare:",
            options=sorted(list(country_names.values())),
            default=[list(country_names.values())[0]] if country_names else []
        )
        selected_country = selected_countries[0] if selected_countries else None
    elif mode == "🔬 Unsupervised Analysis":
        selected_country = None
        selected_countries = []
    else:
        selected_country = st.sidebar.selectbox(
            "Select a country:",
            options=sorted(list(country_names.values())),
            index=0
        )
        selected_countries = [selected_country]
    
    # Year range
    st.sidebar.subheader("📅 Time Period")
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    
    year_range = st.sidebar.slider(
        "Select year range:",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, 2000), max_year)
    )
    
    # Filter data (for modes that need it)
    if mode != "🔬 Unsupervised Analysis":
        country_codes_list = []
        for c in selected_countries:
            if c in country_codes:
                country_codes_list.append(country_codes[c])
        
        if not country_codes_list:
            st.error("No valid countries selected")
            st.stop()
        
        df_filtered = df[
            (df['country'].isin(country_codes_list)) &
            (df['year'] >= year_range[0]) &
            (df['year'] <= year_range[1])
        ]
    else:
        # For unsupervised analysis, use all data in year range
        df_filtered = df[
            (df['year'] >= year_range[0]) &
            (df['year'] <= year_range[1])
        ]
    
    # Main content based on mode
    if mode == "📊 Country Analysis":
        show_country_analysis(df_filtered, selected_country, volatility_model, country_codes)
    
    elif mode == "🔮 GDP Forecasting":
        show_gdp_forecasting(df_filtered, selected_country, gdp_model, country_codes)
    
    elif mode == "📈 Time Series Analysis":
        show_time_series_analysis(df_filtered, selected_country, country_codes, country_names)
    
    elif mode == "🔬 Unsupervised Analysis":
        show_unsupervised_analysis(df_filtered, country_names, country_codes)
    
    elif mode == "🌍 Regional Comparison":
        show_regional_comparison(df_filtered, selected_countries, country_names)


def show_country_analysis(df, country_name, model, country_codes):
    """Show detailed analysis for a single country."""
    st.header(f"📊 Analysis: {country_name}")
    
    country_code = country_codes.get(country_name)
    if not country_code:
        st.error(f"❌ Country code not found for {country_name}")
        return
    
    df_country = df[df['country'] == country_code].copy()
    
    if len(df_country) == 0:
        st.warning(f"No data available for {country_name}")
        return
    
    # ============================================
    # DATA QUALITY WARNING
    # ============================================
    # Check if country has known issues
    if country_code in COUNTRIES_WITH_CAUTION:
        caution_info = COUNTRIES_WITH_CAUTION[country_code]
        st.warning(f"""
        **Data Quality Warning for {country_name}**
        
        {caution_info['reason']}
        
        Results for this country should be interpreted with caution.
        """)
    
    # Check data availability
    n_years = len(df_country)
    key_cols = ['gdp_growth', 'cdi_smooth', 'inflation', 'governance_index']
    available_key_cols = [c for c in key_cols if c in df_country.columns]
    missing_pct = df_country[available_key_cols].isnull().mean().mean() * 100 if available_key_cols else 0
    
    if n_years < 10 or missing_pct > 30:
        st.info(f"""
        ℹ️ **Data Availability**: {n_years} years of data ({int(df_country['year'].min())}-{int(df_country['year'].max())}), 
        {missing_pct:.0f}% missing values in key indicators.
        """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'cdi_smooth' in df_country.columns:
            avg_cdi = df_country['cdi_smooth'].mean()
            st.metric("Average CDI", f"{avg_cdi:.1f}%")
        elif 'cdi_raw' in df_country.columns:
            avg_cdi = df_country['cdi_raw'].mean()
            st.metric("Average CDI", f"{avg_cdi:.1f}%")
    
    with col2:
        if 'log_gdp_volatility' in df_country.columns:
            avg_volatility = np.exp(df_country['log_gdp_volatility'].mean())
            st.metric("Avg Volatility", f"{avg_volatility:.2f}")
        elif 'gdp_volatility' in df_country.columns:
            avg_volatility = df_country['gdp_volatility'].mean()
            st.metric("Avg Volatility", f"{avg_volatility:.2f}")
    
    with col3:
        if 'inflation' in df_country.columns:
            avg_inflation = df_country['inflation'].mean()
            st.metric("Avg Inflation", f"{avg_inflation:.1f}%")
    
    with col4:
        if 'investment' in df_country.columns:
            avg_investment = df_country['investment'].mean()
            st.metric("Avg Investment", f"{avg_investment:.1f}% GDP")
    
    st.markdown("---")
    
    # Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Commodity Dependence Over Time")
        
        cdi_col = 'cdi_smooth' if 'cdi_smooth' in df_country.columns else 'cdi_raw'
        if cdi_col in df_country.columns:
            fig_cdi = px.line(
                df_country,
                x='year',
                y=cdi_col,
                title=f'CDI Trend: {country_name}',
                labels={cdi_col: 'CDI (%)', 'year': 'Year'}
            )
            fig_cdi.update_traces(line_color='#1f77b4', line_width=3)
            fig_cdi.add_hline(y=50, line_dash="dash", line_color="red", 
                             annotation_text="High Dependence Threshold")
            st.plotly_chart(fig_cdi, use_container_width=True)
    
    with col_right:
        st.subheader("📉 GDP Growth Volatility")
        
        if 'log_gdp_volatility' in df_country.columns:
            df_country['volatility'] = np.exp(df_country['log_gdp_volatility'])
        elif 'gdp_volatility' in df_country.columns:
            df_country['volatility'] = df_country['gdp_volatility']
        
        if 'volatility' in df_country.columns:
            fig_vol = px.line(
                df_country,
                x='year',
                y='volatility',
                title=f'Economic Volatility: {country_name}',
                labels={'volatility': 'Volatility', 'year': 'Year'}
            )
            fig_vol.update_traces(line_color='#ff7f0e', line_width=3)
            st.plotly_chart(fig_vol, use_container_width=True)
    
    # GDP Growth over time
    st.subheader("📊 GDP Growth Over Time")
    if 'gdp_growth' in df_country.columns:
        fig_gdp = px.bar(
            df_country,
            x='year',
            y='gdp_growth',
            title=f'Annual GDP Growth: {country_name}',
            labels={'gdp_growth': 'GDP Growth (%)', 'year': 'Year'},
            color='gdp_growth',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_gdp.add_hline(y=0, line_dash="solid", line_color="black")
        st.plotly_chart(fig_gdp, use_container_width=True)
    
    # Model prediction
    if model is not None:
        st.markdown("---")
        st.subheader("🤖 Volatility Prediction")
        
        latest_data = df_country.iloc[-1]
        
        # Build features based on what's available - handle NaN values
        feature_dict = {}
        
        # Helper function to get value with fallback
        def get_value(col_lag, col_current, default):
            if col_lag in df_country.columns:
                val = latest_data.get(col_lag)
                if pd.notna(val):
                    return val
            if col_current in df_country.columns:
                val = latest_data.get(col_current)
                if pd.notna(val):
                    return val
                # Try median of column
                median_val = df_country[col_current].median()
                if pd.notna(median_val):
                    return median_val
            return default
        
        feature_dict['cdi_smooth_lag1'] = [get_value('cdi_smooth_lag1', 'cdi_smooth', 50.0)]
        feature_dict['inflation_lag1'] = [get_value('inflation_lag1', 'inflation', 5.0)]
        feature_dict['trade_openness_lag1'] = [get_value('trade_openness_lag1', 'trade_openness', 50.0)]
        feature_dict['investment_lag1'] = [get_value('investment_lag1', 'investment', 20.0)]
        
        # Add governance if available
        if 'governance_index_lag1' in df_country.columns or 'governance_index' in df_country.columns:
            feature_dict['governance_index_lag1'] = [get_value('governance_index_lag1', 'governance_index', 0.0)]
        
        # Add exchange rate volatility if available
        if 'exchange_rate_volatility_lag1' in df_country.columns or 'exchange_rate_volatility' in df_country.columns:
            feature_dict['exchange_rate_volatility_lag1'] = [get_value('exchange_rate_volatility_lag1', 'exchange_rate_volatility', 5.0)]
        
        features = pd.DataFrame(feature_dict)
        
        # Final safety: fill any remaining NaN
        features = features.fillna(0)
        
        try:
            prediction = model.predict(features)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Predicted Volatility (next year):** {np.exp(prediction):.2f}")
                
                if np.exp(prediction) > 2.0:
                    st.warning("HIGH volatility expected")
                elif np.exp(prediction) > 1.0:
                    st.info("ℹ️ MODERATE volatility expected")
                else:
                    st.success("LOW volatility expected")
            
            with col2:
                st.write("**Key Risk Factors:**")
                
                risk_factors = []
                cdi_val = latest_data.get('cdi_smooth', latest_data.get('cdi_raw', 0))
                if pd.notna(cdi_val) and cdi_val > 70:
                    risk_factors.append("- High commodity dependence")
                inflation_val = latest_data.get('inflation', 0)
                if pd.notna(inflation_val) and inflation_val > 10:
                    risk_factors.append("- High inflation")
                investment_val = latest_data.get('investment', 0)
                if pd.notna(investment_val) and investment_val < 20:
                    risk_factors.append("Low investment")
                governance_val = latest_data.get('governance_index', 0)
                if pd.notna(governance_val) and governance_val < -0.5:
                    risk_factors.append("- Weak governance")
                
                if not risk_factors:
                    st.success("No major risk factors identified")
                else:
                    for factor in risk_factors:
                        st.write(factor)
        
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def show_gdp_forecasting(df, country_name, gdp_model, country_codes):
    """Show GDP growth forecasting for next year."""
    st.header(f"🔮 GDP Growth Forecasting: {country_name}")
    
    country_code = country_codes.get(country_name)
    if not country_code:
        st.error(f"❌ Country code not found for {country_name}")
        return
    
    df_country = df[df['country'] == country_code].copy()
    
    if len(df_country) == 0:
        st.warning(f"No data available for {country_name}")
        return
    
    # Show historical GDP growth
    st.subheader("📊 Historical GDP Growth")
    
    if 'gdp_growth' in df_country.columns:
        fig_hist = px.bar(
            df_country,
            x='year',
            y='gdp_growth',
            title=f'Historical GDP Growth: {country_name}',
            labels={'gdp_growth': 'GDP Growth (%)', 'year': 'Year'},
            color='gdp_growth',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_hist.add_hline(y=0, line_dash="solid", line_color="black")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Key statistics
    st.subheader("📈 GDP Growth Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'gdp_growth' in df_country.columns:
        with col1:
            avg_growth = df_country['gdp_growth'].mean()
            st.metric("Average Growth", f"{avg_growth:.2f}%")
        
        with col2:
            max_growth = df_country['gdp_growth'].max()
            st.metric("Max Growth", f"{max_growth:.2f}%")
        
        with col3:
            min_growth = df_country['gdp_growth'].min()
            st.metric("Min Growth", f"{min_growth:.2f}%")
        
        with col4:
            std_growth = df_country['gdp_growth'].std()
            st.metric("Volatility (Std)", f"{std_growth:.2f}%")
    
    st.markdown("---")
    
    # GDP Forecasting Section
    st.subheader("🔮 Forecast Next Year's GDP Growth")
    
    # Get latest data
    latest_data = df_country.iloc[-1]
    latest_year = int(latest_data['year'])
    
    st.info(f"📅 Forecasting GDP growth for **{latest_year + 1}** based on {latest_year} data")
    
    # Show current indicators
    st.write("**Current Economic Indicators:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cdi_val = latest_data.get('cdi_smooth', latest_data.get('cdi_raw', None))
        if cdi_val is not None and pd.notna(cdi_val):
            st.metric("CDI (Commodity Dependence)", f"{cdi_val:.1f}%")
        
        inflation_val = latest_data.get('inflation', None)
        if inflation_val is not None and pd.notna(inflation_val):
            st.metric("Inflation", f"{inflation_val:.1f}%")
    
    with col2:
        trade_val = latest_data.get('trade_openness', None)
        if trade_val is not None and pd.notna(trade_val):
            st.metric("Trade Openness", f"{trade_val:.1f}%")
        
        investment_val = latest_data.get('investment', None)
        if investment_val is not None and pd.notna(investment_val):
            st.metric("Investment (% GDP)", f"{investment_val:.1f}%")
    
    with col3:
        governance_val = latest_data.get('governance_index', None)
        if governance_val is not None and pd.notna(governance_val):
            st.metric("Governance Index", f"{governance_val:.2f}")
        
        current_gdp = latest_data.get('gdp_growth', None)
        if current_gdp is not None and pd.notna(current_gdp):
            st.metric(f"GDP Growth ({latest_year})", f"{current_gdp:.2f}%")
    
    st.markdown("---")
    
    # Make prediction
    if gdp_model is not None:
        st.subheader("🤖 Model Prediction")
        
        # Prepare features - use median as fallback for NaN values
        feature_dict = {}
        
        if 'cdi_smooth' in df_country.columns:
            val = latest_data['cdi_smooth']
            feature_dict['cdi_smooth'] = [val if pd.notna(val) else df_country['cdi_smooth'].median()]
        if 'inflation' in df_country.columns:
            val = latest_data['inflation']
            feature_dict['inflation'] = [val if pd.notna(val) else df_country['inflation'].median()]
        if 'trade_openness' in df_country.columns:
            val = latest_data['trade_openness']
            feature_dict['trade_openness'] = [val if pd.notna(val) else df_country['trade_openness'].median()]
        if 'investment' in df_country.columns:
            val = latest_data['investment']
            feature_dict['investment'] = [val if pd.notna(val) else df_country['investment'].median()]
        if 'governance_index' in df_country.columns:
            val = latest_data['governance_index']
            feature_dict['governance_index'] = [val if pd.notna(val) else df_country['governance_index'].median()]
        if 'exchange_rate_volatility' in df_country.columns:
            val = latest_data['exchange_rate_volatility']
            feature_dict['exchange_rate_volatility'] = [val if pd.notna(val) else df_country['exchange_rate_volatility'].median()]
        if 'gdp_growth' in df_country.columns:
            val = latest_data['gdp_growth']
            feature_dict['gdp_growth'] = [val if pd.notna(val) else df_country['gdp_growth'].median()]
        
        features = pd.DataFrame(feature_dict)
        
        # Fill any remaining NaN with 0
        features = features.fillna(0)
        
        try:
            prediction = gdp_model.predict(features)[0]
            
            # Display prediction
            col1, col2 = st.columns(2)
            
            with col1:
                color = 'green' if prediction > 0 else 'red'
                st.markdown(f"""
                <div style="background-color: #e6f3ff; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: #1f77b4;">Predicted GDP Growth ({latest_year + 1})</h2>
                    <h1 style="color: {color}; font-size: 3rem;">{prediction:.2f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Interpretation
                st.write("**Interpretation:**")
                
                if prediction > 5:
                    st.success("🚀 **Strong growth** expected! Economy likely to expand significantly.")
                elif prediction > 2:
                    st.success("**Moderate growth** expected. Positive economic outlook.")
                elif prediction > 0:
                    st.info("**Slow growth** expected. Economy growing but slowly.")
                elif prediction > -2:
                    st.warning("**Slight contraction** possible. Economic headwinds ahead.")
                else:
                    st.error("**Recession risk**. Significant economic challenges expected.")
                
                # Comparison with historical average
                if 'gdp_growth' in df_country.columns:
                    hist_avg = df_country['gdp_growth'].mean()
                    diff = prediction - hist_avg
                    
                    if diff > 0:
                        st.write(f"This is **{diff:.1f}pp above** the historical average ({hist_avg:.1f}%)")
                    else:
                        st.write(f"📉 This is **{abs(diff):.1f}pp below** the historical average ({hist_avg:.1f}%)")
            
            # Visualization: Historical + Forecast
            st.subheader("📊 Historical Data + Forecast")
            
            if 'gdp_growth' in df_country.columns:
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'year': [latest_year + 1],
                    'gdp_growth': [prediction],
                    'type': ['Forecast']
                })
                
                hist_df = df_country[['year', 'gdp_growth']].copy()
                hist_df['type'] = 'Historical'
                
                combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
                
                fig = px.bar(
                    combined_df,
                    x='year',
                    y='gdp_growth',
                    color='type',
                    title=f'{country_name}: GDP Growth (Historical + Forecast)',
                    labels={'gdp_growth': 'GDP Growth (%)', 'year': 'Year'},
                    color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#ff7f0e'}
                )
                fig.add_hline(y=0, line_dash="solid", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("The model may not have the required features. Try retraining with: `python scripts/train_models.py --include-forecast`")
    
    else:
        st.warning("GDP Forecasting model not available.")
        st.info("""
        To enable GDP forecasting, train the model with:
        ```
        python scripts/train_models.py --include-forecast
        ```
        """)
        
        # Show simple statistical forecast instead
        st.subheader("📊 Simple Statistical Forecast")
        
        if 'gdp_growth' in df_country.columns:
            # Simple moving average forecast
            recent_growth = df_country['gdp_growth'].tail(5).mean()
            
            st.metric(
                f"Estimated GDP Growth ({latest_year + 1})",
                f"{recent_growth:.2f}%",
                help="Based on 5-year moving average"
            )
            
            st.caption("⚠️ This is a simple statistical estimate, not a machine learning prediction.")


def show_time_series_analysis(df, country_name, country_codes, country_names):
    """Show time series analysis for a country."""
    from scipy import stats
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')
    
    st.header(f"📈 Time Series Analysis: {country_name}")
    
    st.markdown("""
    **Analyze temporal patterns and forecast future values using time series techniques.**
    
    This analysis includes:
    - **Trend Analysis**: Is the variable increasing or decreasing over time?
    - **Decomposition**: Separate trend, seasonality, and noise
    - **Stationarity Test**: Is the series stable over time?
    - **ARIMA Forecasting**: Predict future values
    """)
    
    country_code = country_codes.get(country_name)
    if not country_code:
        st.error(f"❌ Country code not found for {country_name}")
        return
    
    df_country = df[df['country'] == country_code].copy()
    df_country = df_country.sort_values('year')
    
    if len(df_country) < 5:
        st.warning(f"⚠️ Not enough data for {country_name}. Need at least 5 years of data.")
        return
    
    # Sidebar options
    st.sidebar.subheader("📈 Time Series Settings")
    
    # Select target variable
    numeric_cols = df_country.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['year', 'cluster', 'PC1', 'PC2']
    target_options = [c for c in numeric_cols if c not in exclude_cols and df_country[c].notna().sum() > 5]
    
    if not target_options:
        st.error("No suitable variables for time series analysis")
        return
    
    default_target = 'gdp_growth' if 'gdp_growth' in target_options else target_options[0]
    target = st.sidebar.selectbox(
        "Variable to analyze:",
        target_options,
        index=target_options.index(default_target) if default_target in target_options else 0
    )
    
    forecast_periods = st.sidebar.slider("Forecast periods:", 1, 5, 3)
    
    # Prepare data
    ts_data = df_country[['year', target]].dropna()
    
    if len(ts_data) < 5:
        st.warning(f"⚠️ Not enough non-null data for {target}")
        return
    
    years = ts_data['year'].values
    values = ts_data[target].values
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Trend Analysis", "🔄 Decomposition", "📉 Autocorrelation", "🔮 ARIMA Forecast"])
    
    # ===========================================
    # TAB 1: TREND ANALYSIS
    # ===========================================
    with tab1:
        st.subheader("📊 Trend Analysis")
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
        trend_line = slope * years + intercept
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            direction = "📈 Increasing" if slope > 0 else "📉 Decreasing"
            st.metric("Trend Direction", direction)
        
        with col2:
            st.metric("Annual Change", f"{slope:.3f}")
        
        with col3:
            st.metric("R² (Fit)", f"{r_value**2:.3f}")
        
        with col4:
            significant = "Yes" if p_value < 0.05 else "No"
            st.metric("Significant?", significant)
        
        # Plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=values,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f'{country_name}: {target.replace("_", " ").title()} with Trend Line',
            xaxis_title='Year',
            yaxis_title=target.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        total_change = slope * (years[-1] - years[0])
        st.info(f"""
        **Interpretation:**
        - The {target.replace('_', ' ')} {'increased' if slope > 0 else 'decreased'} by approximately **{abs(slope):.3f} per year**
        - Total change over the period: **{total_change:.2f}**
        - This trend is {'statistically significant (p < 0.05)' if p_value < 0.05 else 'not statistically significant (p ≥ 0.05)'}
        """)
    
    # ===========================================
    # TAB 2: DECOMPOSITION
    # ===========================================
    with tab2:
        st.subheader("🔄 Time Series Decomposition")
        
        st.markdown("""
        Decomposition separates the time series into:
        - **Trend**: Long-term direction
        - **Seasonal**: Repeating patterns (cycles)
        - **Residual**: Random noise
        """)
        
        # Need enough data for decomposition
        if len(ts_data) >= 8:
            try:
                # Create time series with year index
                ts = pd.Series(values, index=pd.to_datetime(years, format='%Y'))
                
                # Decompose
                period = min(4, len(ts) // 2)
                decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')
                
                # Plot decomposition
                fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                
                axes[0].plot(years, values, 'b-')
                axes[0].set_title('Observed')
                axes[0].set_ylabel(target)
                
                axes[1].plot(years, decomposition.trend, 'g-')
                axes[1].set_title('Trend')
                axes[1].set_ylabel('Trend')
                
                axes[2].plot(years, decomposition.seasonal, 'orange')
                axes[2].set_title('Seasonal (Cyclical)')
                axes[2].set_ylabel('Seasonal')
                
                axes[3].plot(years, decomposition.resid, 'r-')
                axes[3].set_title('Residual (Noise)')
                axes[3].set_ylabel('Residual')
                axes[3].set_xlabel('Year')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Variance explained
                total_var = np.var(values)
                trend_var = np.nanvar(decomposition.trend)
                seasonal_var = np.nanvar(decomposition.seasonal)
                resid_var = np.nanvar(decomposition.resid)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Trend Variance", f"{trend_var/total_var*100:.1f}%")
                col2.metric("Seasonal Variance", f"{seasonal_var/total_var*100:.1f}%")
                col3.metric("Residual Variance", f"{resid_var/total_var*100:.1f}%")
                
            except Exception as e:
                st.warning(f"⚠️ Decomposition failed: {e}")
        else:
            st.warning("⚠️ Need at least 8 years of data for decomposition")
    
    # ===========================================
    # TAB 3: AUTOCORRELATION
    # ===========================================
    with tab3:
        st.subheader("📉 Autocorrelation Analysis")
        
        st.markdown("""
        Autocorrelation measures how correlated a value is with its past values.
        - **ACF** (Autocorrelation Function): Correlation at different lags
        - **PACF** (Partial ACF): Direct correlation at each lag
        
        This helps us understand:
        - How persistent are economic shocks?
        - How many past values influence the current value?
        """)
        
        # Stationarity test
        st.write("### Stationarity Test (Augmented Dickey-Fuller)")
        
        try:
            adf_result = adfuller(values)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
                st.metric("P-value", f"{adf_result[1]:.4f}")
            
            with col2:
                is_stationary = adf_result[1] < 0.05
                if is_stationary:
                    st.success("Series is STATIONARY")
                    st.write("The series does not have a unit root - it's stable over time.")
                else:
                    st.warning("⚠️ Series is NON-STATIONARY")
                    st.write("The series may have a trend or changing variance over time.")
        except Exception as e:
            st.warning(f"⚠️ Stationarity test failed: {e}")
        
        # ACF/PACF plots
        st.write("### ACF and PACF Plots")
        
        max_lags = min(10, len(values) // 2 - 1)
        
        if max_lags >= 2:
            try:
                acf_values = acf(values, nlags=max_lags)
                pacf_values = pacf(values, nlags=max_lags)
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                
                # ACF
                axes[0].bar(range(len(acf_values)), acf_values, color='steelblue')
                axes[0].axhline(y=1.96/np.sqrt(len(values)), color='r', linestyle='--', label='95% CI')
                axes[0].axhline(y=-1.96/np.sqrt(len(values)), color='r', linestyle='--')
                axes[0].set_title('Autocorrelation Function (ACF)')
                axes[0].set_xlabel('Lag')
                axes[0].set_ylabel('Correlation')
                
                # PACF
                axes[1].bar(range(len(pacf_values)), pacf_values, color='darkorange')
                axes[1].axhline(y=1.96/np.sqrt(len(values)), color='r', linestyle='--', label='95% CI')
                axes[1].axhline(y=-1.96/np.sqrt(len(values)), color='r', linestyle='--')
                axes[1].set_title('Partial Autocorrelation Function (PACF)')
                axes[1].set_xlabel('Lag')
                axes[1].set_ylabel('Correlation')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Find significant lags
                ci = 1.96 / np.sqrt(len(values))
                sig_acf = [i for i, v in enumerate(acf_values) if abs(v) > ci and i > 0]
                sig_pacf = [i for i, v in enumerate(pacf_values) if abs(v) > ci and i > 0]
                
                st.info(f"""
                **Interpretation:**
                - Significant ACF lags: {sig_acf if sig_acf else 'None'} - suggests MA order
                - Significant PACF lags: {sig_pacf if sig_pacf else 'None'} - suggests AR order
                - Bars outside the red dashed lines are statistically significant
                """)
                
            except Exception as e:
                st.warning(f"⚠️ ACF/PACF calculation failed: {e}")
        else:
            st.warning("⚠️ Not enough data for ACF/PACF analysis")
    
    # ===========================================
    # TAB 4: ARIMA FORECAST
    # ===========================================
    with tab4:
        st.subheader("🔮 ARIMA Forecast")
        
        st.markdown("""
        **ARIMA** (AutoRegressive Integrated Moving Average) is a classic time series forecasting method.
        
        It combines:
        - **AR (p)**: Past values influence current value
        - **I (d)**: Differencing to make series stationary
        - **MA (q)**: Past errors influence current value
        """)
        
        if len(values) < 10:
            st.warning("⚠️ ARIMA requires at least 10 data points for reliable forecasting")
            return
        
        # Auto-select ARIMA order
        st.write("### Model Selection")
        
        # Simple order selection
        try:
            adf_result = adfuller(values)
            d = 0 if adf_result[1] < 0.05 else 1
        except:
            d = 0
        
        # Try a few orders and pick best AIC
        best_aic = np.inf
        best_order = (1, d, 1)
        
        orders_to_try = [(0,d,1), (1,d,0), (1,d,1), (2,d,1), (1,d,2)]
        
        for order in orders_to_try:
            try:
                model = ARIMA(values, order=order)
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = order
            except:
                continue
        
        st.write(f"**Selected Order**: ARIMA{best_order} (AIC: {best_aic:.2f})")
        
        # Fit final model
        try:
            model = ARIMA(values, order=best_order)
            fitted = model.fit()
            
            # In-sample metrics
            fitted_values = fitted.fittedvalues
            
            col1, col2, col3 = st.columns(3)
            
            rmse = np.sqrt(np.mean((values[1:] - fitted_values[1:])**2))
            mae = np.mean(np.abs(values[1:] - fitted_values[1:]))
            
            col1.metric("RMSE", f"{rmse:.3f}")
            col2.metric("MAE", f"{mae:.3f}")
            col3.metric("AIC", f"{fitted.aic:.2f}")
            
            # Forecast
            st.write("### Forecast")
            
            forecast_result = fitted.get_forecast(steps=forecast_periods)
            forecast_mean = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            last_year = int(years[-1])
            forecast_years = [last_year + i + 1 for i in range(forecast_periods)]
            
            # Plot
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=list(years), y=list(values),
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Fitted values
            fig.add_trace(go.Scatter(
                x=list(years), y=list(fitted_values),
                mode='lines',
                name='Fitted',
                line=dict(color='green', dash='dot', width=1)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_years, y=list(forecast_mean),
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=10)
            ))
            
            # Confidence interval
            ci_lower = forecast_ci.iloc[:, 0].values if hasattr(forecast_ci, 'iloc') else forecast_ci[:, 0]
            ci_upper = forecast_ci.iloc[:, 1].values if hasattr(forecast_ci, 'iloc') else forecast_ci[:, 1]
            
            fig.add_trace(go.Scatter(
                x=forecast_years + forecast_years[::-1],
                y=list(ci_upper) + list(ci_lower)[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,0,0,0)'),
                name='95% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f'ARIMA Forecast: {country_name} - {target.replace("_", " ").title()}',
                xaxis_title='Year',
                yaxis_title=target.replace('_', ' ').title(),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.write("### Forecast Values")
            
            forecast_values = forecast_mean.values if hasattr(forecast_mean, 'values') else forecast_mean
            
            forecast_df = pd.DataFrame({
                'Year': forecast_years,
                'Forecast': forecast_values,
                'Lower 95% CI': ci_lower,
                'Upper 95% CI': ci_upper
            })
            
            st.dataframe(forecast_df.style.format({
                'Forecast': '{:.2f}',
                'Lower 95% CI': '{:.2f}',
                'Upper 95% CI': '{:.2f}'
            }), use_container_width=True)
            
            # Interpretation
            first_forecast = forecast_values[0] if hasattr(forecast_values, '__getitem__') else forecast_values
            first_lower = ci_lower[0] if hasattr(ci_lower, '__getitem__') else ci_lower
            first_upper = ci_upper[0] if hasattr(ci_upper, '__getitem__') else ci_upper
            
            st.info(f"""
            **Forecast Interpretation:**
            - The model predicts {target.replace('_', ' ')} for {country_name} will be approximately **{first_forecast:.2f}** in {forecast_years[0]}
            - The 95% confidence interval ranges from **{first_lower:.2f}** to **{first_upper:.2f}**
            - Wider confidence intervals indicate more uncertainty
            """)
            
        except Exception as e:
            st.error(f"❌ ARIMA forecasting failed: {e}")
            st.info("Try selecting a different variable or country with more data.")


def show_unsupervised_analysis(df, country_names, country_codes):
    """Show unsupervised learning analysis: Clustering and PCA."""
    
    st.header("🔬 Unsupervised Learning Analysis")
    
    st.markdown("""
    **Discover hidden patterns in African economies using Machine Learning.**
    
    This analysis uses:
    - **k-Means Clustering**: Group similar countries together
    - **Hierarchical Clustering**: Build a tree of country similarities  
    - **PCA**: Reduce dimensions and visualize relationships
    """)
    
    # Sidebar options
    st.sidebar.subheader("🔬 Clustering Settings")
    n_clusters = st.sidebar.slider("Number of clusters:", 2, 6, 3)
    
    # Features for clustering
    available_features = []
    feature_options = ['cdi_smooth', 'cdi_raw', 'gdp_volatility', 'log_gdp_volatility',
                       'governance_index', 'inflation', 'investment', 'trade_openness',
                       'exchange_rate_volatility']
    
    for f in feature_options:
        if f in df.columns:
            available_features.append(f)
    
    if len(available_features) < 2:
        st.error("Not enough features available for clustering. Need at least 2 features.")
        return
    
    # Select features
    default_features = [f for f in ['cdi_smooth', 'governance_index', 'inflation', 'investment'] 
                       if f in available_features]
    if len(default_features) < 2:
        default_features = available_features[:min(4, len(available_features))]
    
    selected_features = st.sidebar.multiselect(
        "Features for clustering:",
        available_features,
        default=default_features
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for clustering")
        return
    
    # Prepare data - aggregate to country level
    agg_dict = {f: 'mean' for f in selected_features}
    
    country_data = df.groupby('country').agg(agg_dict).reset_index()
    country_data = country_data.dropna(subset=selected_features)
    
    # Add country names
    country_data['country_name'] = country_data['country'].map(country_names)
    country_data['country_name'] = country_data['country_name'].fillna(country_data['country'])
    
    if len(country_data) < n_clusters:
        st.error(f"Not enough countries with complete data. Found {len(country_data)}, need at least {n_clusters}")
        return
    
    st.success(f"✓ Analyzing {len(country_data)} countries with {len(selected_features)} features")
    
    # Scale data
    scaler = StandardScaler()
    X = country_data[selected_features].values
    X_scaled = scaler.fit_transform(X)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 K-Means Clustering", "🌳 Hierarchical Clustering", "📉 PCA Analysis", "⚠️ Data Quality"])
    
    # ===========================================
    # TAB 1: K-MEANS CLUSTERING
    # ===========================================
    with tab1:
        st.subheader("📊 K-Means Clustering")
        
        # Fit k-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        country_data['cluster'] = clusters
        
        # Assign cluster names based on characteristics
        cluster_names = {}
        for c in range(n_clusters):
            cluster_df = country_data[country_data['cluster'] == c]
            
            # Get average CDI and governance for this cluster
            avg_cdi = cluster_df['cdi_smooth'].mean() if 'cdi_smooth' in cluster_df.columns else cluster_df.get('cdi_raw', pd.Series([50])).mean()
            avg_gov = cluster_df['governance_index'].mean() if 'governance_index' in cluster_df.columns else 0
            
            if avg_cdi > 60 and avg_gov < -0.3:
                cluster_names[c] = "Resource Curse"
            elif avg_cdi > 60 and avg_gov >= -0.3:
                cluster_names[c] = "🟡 Managed Resources"
            elif avg_cdi <= 40:
                cluster_names[c] = "🟢 Diversified Economy"
            else:
                cluster_names[c] = f"🟠 Cluster {c+1}"
        
        country_data['cluster_name'] = country_data['cluster'].map(cluster_names)
        
        # Display cluster summary
        st.write("### Cluster Summary")
        
        for c in range(n_clusters):
            cluster_df = country_data[country_data['cluster'] == c]
            
            with st.expander(f"{cluster_names[c]} ({len(cluster_df)} countries)", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Average Characteristics:**")
                    for feat in selected_features:
                        avg_val = cluster_df[feat].mean()
                        st.write(f"• {feat.replace('_', ' ').title()}: {avg_val:.2f}")
                
                with col2:
                    st.write("**Countries:**")
                    countries_list = cluster_df['country_name'].tolist()
                    st.write(", ".join(countries_list))
        
        # Scatter plot
        st.write("### Cluster Visualization")
        
        # Choose features for x and y axes
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis:", selected_features, index=0, key="kmeans_x")
        with col2:
            y_feature = st.selectbox("Y-axis:", selected_features, 
                                    index=min(1, len(selected_features)-1), key="kmeans_y")
        
        fig = px.scatter(
            country_data,
            x=x_feature,
            y=y_feature,
            color='cluster_name',
            hover_name='country_name',
            size_max=15,
            title=f'K-Means Clustering: {x_feature} vs {y_feature}',
            labels={x_feature: x_feature.replace('_', ' ').title(),
                   y_feature: y_feature.replace('_', ' ').title()},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_traces(marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insight
        st.info("""
        💡 **Key Insight**: Countries in the "Resource Curse" cluster typically have 
        high commodity dependence AND weak governance. This confirms that the 
        commodities paradox is not universal - it primarily affects countries 
        with institutional weaknesses.
        """)
    
    # ===========================================
    # TAB 2: HIERARCHICAL CLUSTERING
    # ===========================================
    with tab2:
        st.subheader("🌳 Hierarchical Clustering")
        
        st.markdown("""
        The dendrogram shows how countries group together based on similarity.
        Countries that merge at lower heights are more similar.
        """)
        
        # Compute linkage
        linkage_matrix = linkage(X_scaled, method='ward')
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(14, 8))
        
        labels = country_data['country_name'].values
        
        dendrogram(
            linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=9,
            ax=ax,
            color_threshold=0.7 * max(linkage_matrix[:, 2])
        )
        
        ax.set_xlabel('Countries', fontsize=12)
        ax.set_ylabel('Distance (Dissimilarity)', fontsize=12)
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        💡 **How to read the dendrogram**: 
        - Countries that merge at **lower heights** are more similar
        - The **horizontal line** where branches merge indicates the distance at which countries are grouped
        - Look for distinct clusters that merge at high distances - these represent fundamentally different types of economies
        """)
    
    # ===========================================
    # TAB 3: PCA ANALYSIS
    # ===========================================
    with tab3:
        st.subheader("📉 Principal Component Analysis (PCA)")
        
        st.markdown("""
        PCA reduces multiple economic features into 2 main dimensions, 
        revealing the underlying structure of the data.
        """)
        
        # Fit PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        country_data['PC1'] = X_pca[:, 0]
        country_data['PC2'] = X_pca[:, 1]
        
        # Explained variance
        var1 = pca.explained_variance_ratio_[0]
        var2 = pca.explained_variance_ratio_[1]
        total_var = var1 + var2
        
        col1, col2, col3 = st.columns(3)
        col1.metric("PC1 Variance", f"{var1:.1%}")
        col2.metric("PC2 Variance", f"{var2:.1%}")
        col3.metric("Total Explained", f"{total_var:.1%}")
        
        # Loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=selected_features
        )
        
        st.write("### Feature Loadings")
        st.write("How each feature contributes to each principal component:")
        
        # Heatmap of loadings
        fig_loadings = px.imshow(
            loadings.values,
            x=['PC1', 'PC2'],
            y=[f.replace('_', ' ').title() for f in selected_features],
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='PCA Feature Loadings'
        )
        fig_loadings.update_layout(height=400)
        st.plotly_chart(fig_loadings, use_container_width=True)
        
        # Interpret components
        st.write("### Component Interpretation")
        
        for i, pc in enumerate(['PC1', 'PC2']):
            top_pos = loadings[pc].nlargest(2)
            top_neg = loadings[pc].nsmallest(2)
            
            pos_str = ", ".join([f"{feat.replace('_', ' ')} (+{val:.2f})" for feat, val in top_pos.items()])
            neg_str = ", ".join([f"{feat.replace('_', ' ')} ({val:.2f})" for feat, val in top_neg.items()])
            
            st.write(f"**{pc}**: High values indicate {pos_str}; Low values indicate {neg_str}")
        
        # 2D scatter plot
        st.write("### Countries in PCA Space")
        
        color_option = st.radio(
            "Color by:",
            ["Cluster", "CDI", "Governance"],
            horizontal=True,
            key="pca_color"
        )
        
        if color_option == "Cluster":
            color_col = 'cluster_name'
            fig_pca = px.scatter(
                country_data,
                x='PC1',
                y='PC2',
                color=color_col,
                hover_name='country_name',
                title='Countries in PCA Space (colored by cluster)',
                labels={'PC1': f'PC1 ({var1:.1%})', 'PC2': f'PC2 ({var2:.1%})'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        else:
            color_col = 'cdi_smooth' if color_option == "CDI" else 'governance_index'
            if color_col not in country_data.columns:
                color_col = selected_features[0]
            
            fig_pca = px.scatter(
                country_data,
                x='PC1',
                y='PC2',
                color=color_col,
                hover_name='country_name',
                title=f'Countries in PCA Space (colored by {color_option})',
                labels={'PC1': f'PC1 ({var1:.1%})', 'PC2': f'PC2 ({var2:.1%})'},
                color_continuous_scale='RdYlGn' if color_option == "Governance" else 'RdYlGn_r'
            )
        
        fig_pca.update_traces(marker=dict(size=12))
        
        # Add country labels
        for idx, row in country_data.iterrows():
            fig_pca.add_annotation(
                x=row['PC1'],
                y=row['PC2'],
                text=str(row['country_name'])[:10],
                showarrow=False,
                font=dict(size=8),
                yshift=10
            )
        
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.info("""
        💡 **Key Insight from PCA**: 
        - If CDI and Governance point in **opposite directions**, this confirms they are inversely related
        - Countries in the **same region** of the plot have similar economic profiles
        - The **distance** between countries shows how different their economies are
        """)
    
    # ===========================================
    # TAB 4: DATA QUALITY
    # ===========================================
    with tab4:
        st.subheader("⚠️ Data Quality Assessment")
        
        st.markdown("""
        **Understanding the reliability of our analysis.**
        
        Not all countries have the same data quality. Factors affecting reliability include:
        - Number of years with available data
        - Percentage of missing values
        - Known conflicts or political instability
        """)
        
        # Calculate quality metrics
        quality_df = assess_data_quality(df)
        quality_df['country_name'] = quality_df['country'].map(country_names)
        quality_df['country_name'] = quality_df['country_name'].fillna(quality_df['country'])
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        good_count = len(quality_df[quality_df['quality_level'].str.contains('Good')])
        moderate_count = len(quality_df[quality_df['quality_level'].str.contains('Moderate')])
        low_count = len(quality_df[quality_df['quality_level'].str.contains('Low')])
        
        col1.metric("Good Quality", f"{good_count} countries")
        col2.metric("Moderate Quality", f"{moderate_count} countries")
        col3.metric("Low Quality", f"{low_count} countries")
        
        # Countries with caution warnings
        st.write("### ⚠️ Countries Requiring Caution")
        
        caution_countries = quality_df[quality_df['has_caution'] == True]
        
        if len(caution_countries) > 0:
            for _, row in caution_countries.iterrows():
                with st.expander(f"🔴 {row['country_name']} ({row['country']})"):
                    st.write(f"**Reason:** {row['caution_reason']}")
                    st.write(f"**Data available:** {row['n_years']} years ({row['year_range']})")
                    st.write(f"**Missing data:** {row['missing_pct']}%")
                    st.write(f"**Quality score:** {row['quality_score']}/100")
        else:
            st.success("No countries with specific caution warnings in the current dataset.")
        
        # Full quality table
        st.write("### 📋 Full Data Quality Report")
        
        display_df = quality_df[['country_name', 'n_years', 'year_range', 
                                  'missing_pct', 'quality_score', 'quality_level']].copy()
        display_df.columns = ['Country', 'Years of Data', 'Period', 
                              'Missing %', 'Quality Score', 'Quality Level']
        display_df = display_df.sort_values('Quality Score', ascending=False)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Interpretation guide
        st.write("### 📖 How to Interpret")
        
        st.markdown("""
        | Quality Level | Score | Interpretation |
        |---------------|-------|----------------|
        | Good | 70-100 | Results are reliable |
        | Moderate | 50-69 | Results should be cross-checked |
        | Low | 0-49 | Results may be misleading - use with caution |
        
        **Factors reducing quality score:**
        - Less than 10 years of data: -5 points per missing year
        - Missing values in key indicators: -1 point per % missing
        - Known conflicts/instability: -20 points
        """)
    
    # ===========================================
    # KEY FINDINGS
    # ===========================================
    st.markdown("---")
    st.header("🎯 Key Findings from Unsupervised Analysis")
    
    # Calculate insights
    resource_curse_countries = country_data[country_data['cluster_name'].str.contains('Curse', na=False)]
    stable_countries = country_data[country_data['cluster_name'].str.contains('Diversified|Green', na=False)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Resource Curse Pattern")
        if len(resource_curse_countries) > 0:
            st.write(f"**{len(resource_curse_countries)} countries** show the classic paradox:")
            st.write("• High commodity dependence")
            st.write("• Weak governance")
            st.write("• High economic volatility")
            st.write(f"\n*Examples: {', '.join(resource_curse_countries['country_name'].head(5).tolist())}*")
        else:
            st.write("No clear resource curse pattern identified with current clustering.")
    
    with col2:
        st.markdown("### 🟢 Escaped the Paradox")
        if len(stable_countries) > 0:
            st.write(f"**{len(stable_countries)} countries** have stable economies:")
            st.write("• Diversified exports OR")
            st.write("• Strong governance despite resources")
            st.write("• Lower economic volatility")
            st.write(f"\n*Examples: {', '.join(stable_countries['country_name'].head(5).tolist())}*")
        else:
            st.write("Stable economies identified in other clusters.")
    
    st.success("""
    **Conclusion**: The African Commodities Paradox is NOT universal. 
    Clustering reveals that **governance quality** is often the key differentiator 
    between countries that suffer from the resource curse and those that don't.
    """)
    
    # Warning about data quality
    st.markdown("---")
    st.warning("""
    ⚠️ **Important Limitations**
    
    These results should be interpreted with caution for countries affected by:
    - **Armed conflicts** (South Sudan, Libya, CAR, Somalia)
    - **Limited data availability** (newer nations, data gaps)
    - **Political instability** affecting data reliability
    
    Check the **Data Quality** tab for a detailed assessment of each country's data reliability.
    """)


def show_regional_comparison(df, countries, country_names):
    """Show comparison across multiple countries."""
    st.header("🌍 Regional Comparison")
    
    if len(countries) == 0:
        st.warning("Please select at least one country")
        return
    
    # Get country codes
    country_codes_reverse = {v: k for k, v in country_names.items()}
    country_codes_list = [country_codes_reverse.get(c) for c in countries if c in country_codes_reverse]
    
    # Filter data
    df_filtered = df[df['country'].isin(country_codes_list)].copy()
    
    # Add country names
    df_filtered['country_name'] = df_filtered['country'].map(country_names)
    
    # Determine which columns to use
    cdi_col = 'cdi_smooth' if 'cdi_smooth' in df_filtered.columns else 'cdi_raw'
    vol_col = 'gdp_volatility' if 'gdp_volatility' in df_filtered.columns else 'log_gdp_volatility'
    
    # Average metrics by country
    st.subheader("📊 Average Metrics by Country")
    
    agg_dict = {}
    if cdi_col in df_filtered.columns:
        agg_dict[cdi_col] = 'mean'
    if vol_col in df_filtered.columns:
        agg_dict[vol_col] = 'mean'
    if 'inflation' in df_filtered.columns:
        agg_dict['inflation'] = 'mean'
    if 'investment' in df_filtered.columns:
        agg_dict['investment'] = 'mean'
    if 'gdp_growth' in df_filtered.columns:
        agg_dict['gdp_growth'] = 'mean'
    
    if agg_dict:
        summary = df_filtered.groupby('country_name').agg(agg_dict).round(2)
        st.dataframe(summary, use_container_width=True)
    
    # Comparative visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 CDI Comparison")
        
        if cdi_col in df_filtered.columns:
            avg_cdi = df_filtered.groupby('country_name')[cdi_col].mean().reset_index()
            fig_cdi = px.bar(
                avg_cdi,
                x='country_name',
                y=cdi_col,
                color=cdi_col,
                color_continuous_scale='Reds',
                title='Average Commodity Dependence Index'
            )
            st.plotly_chart(fig_cdi, use_container_width=True)
    
    with col2:
        st.subheader("📈 GDP Growth Comparison")
        
        if 'gdp_growth' in df_filtered.columns:
            avg_gdp = df_filtered.groupby('country_name')['gdp_growth'].mean().reset_index()
            fig_gdp = px.bar(
                avg_gdp,
                x='country_name',
                y='gdp_growth',
                color='gdp_growth',
                color_continuous_scale=['red', 'yellow', 'green'],
                title='Average GDP Growth'
            )
            st.plotly_chart(fig_gdp, use_container_width=True)


# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>African Commodities Paradox Analyzer</strong></p>
        <p>Author: Abraham Adegoke | HEC Lausanne | December 2025</p>
        <p>Powered by Machine Learning (Gradient Boosting, K-Means, PCA, ARIMA) | Data: World Bank WDI (1990-2023)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    show_footer()