"""
African Commodities Paradox - Interactive Web Application

A Streamlit web interface for analyzing commodity dependence and economic volatility
in African countries. Provides interactive visualizations and GDP forecasting.

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.gradient_boosting import GradientBoostingModel
from models.gdp_forecaster import GDPForecaster
from preprocessing.preprocessing import DataPreprocessor

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

# Country mapping (ISO3 to full names) - Will be populated from data
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

# Will be dynamically populated from actual data
COUNTRY_NAMES = {}
COUNTRY_CODES = {}


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
    try:
        volatility_model = GradientBoostingModel.load('results/gbr_model.pkl')
        
        # Try to load GDP forecaster (might not exist yet)
        try:
            gdp_model = GDPForecaster.load('results/gdp_forecaster.pkl')
        except:
            gdp_model = None
            
        return volatility_model, gdp_model
    except FileNotFoundError:
        st.error("❌ Models not found. Please run: python scripts/train_models.py")
        return None, None


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
    
    # Analysis mode
    mode = st.sidebar.radio(
        "Select Analysis Mode:",
        ["📊 Country Analysis", "🌍 Regional Comparison"]
    )
    
    # Country selection
    if mode == "🌍 Regional Comparison":
        selected_countries = st.sidebar.multiselect(
            "Select countries to compare:",
            options=sorted(list(country_names.values())),
            default=[list(country_names.values())[0]] if country_names else []
        )
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
        value=(max(min_year, 2010), max_year)
    )
    
    # Filter data - handle case where country not in mapping
    country_codes_list = []
    for c in selected_countries:
        if c in country_codes:
            country_codes_list.append(country_codes[c])
        else:
            st.warning(f"⚠️ Country '{c}' not found in data")
    
    if not country_codes_list:
        st.error("❌ No valid countries selected")
        st.stop()
    
    df_filtered = df[
        (df['country'].isin(country_codes_list)) &
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]
    
    # Main content based on mode
    if mode == "📊 Country Analysis":
        show_country_analysis(df_filtered, selected_country, volatility_model, country_codes)
    
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_cdi = df_country['cdi_smooth'].mean()
        st.metric("Average CDI", f"{avg_cdi:.1f}%")
    
    with col2:
        avg_volatility = np.exp(df_country['log_gdp_volatility'].mean())
        st.metric("Avg Volatility", f"{avg_volatility:.2f}")
    
    with col3:
        avg_inflation = df_country['inflation'].mean()
        st.metric("Avg Inflation", f"{avg_inflation:.1f}%")
    
    with col4:
        avg_investment = df_country['investment'].mean()
        st.metric("Avg Investment", f"{avg_investment:.1f}% GDP")
    
    st.markdown("---")
    
    # Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Commodity Dependence Over Time")
        
        fig_cdi = px.line(
            df_country,
            x='year',
            y='cdi_smooth',
            title=f'CDI Trend: {country_name}',
            labels={'cdi_smooth': 'CDI (%)', 'year': 'Year'}
        )
        fig_cdi.update_traces(line_color='#1f77b4', line_width=3)
        fig_cdi.add_hline(y=50, line_dash="dash", line_color="red", 
                         annotation_text="High Dependence Threshold")
        st.plotly_chart(fig_cdi, use_container_width=True)
    
    with col_right:
        st.subheader("📉 GDP Growth Volatility")
        
        df_country['volatility'] = np.exp(df_country['log_gdp_volatility'])
        
        fig_vol = px.line(
            df_country,
            x='year',
            y='volatility',
            title=f'Economic Volatility: {country_name}',
            labels={'volatility': 'Volatility', 'year': 'Year'}
        )
        fig_vol.update_traces(line_color='#ff7f0e', line_width=3)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Scatter plot: CDI vs Volatility
    st.subheader("🔍 CDI vs Volatility Relationship")
    
    # Calculate volatility if not present
    if 'volatility' not in df_country.columns:
        if 'log_gdp_volatility' in df_country.columns:
            df_country['volatility'] = np.exp(df_country['log_gdp_volatility'])
        elif 'gdp_volatility' in df_country.columns:
            df_country['volatility'] = df_country['gdp_volatility']
        else:
            st.warning("⚠️ Volatility data not available for scatter plot")
            return
    
    # Remove rows with missing values for plotting
    df_plot = df_country[['cdi_smooth', 'volatility', 'year', 'investment', 'inflation']].dropna()
    
    if len(df_plot) == 0:
        st.warning("⚠️ Not enough data for scatter plot")
        return
    
    fig_scatter = px.scatter(
        df_plot,
        x='cdi_smooth',
        y='volatility',
        size='investment',
        color='year',
        hover_data=['year', 'inflation'],
        title=f'{country_name}: Commodity Dependence vs Economic Volatility',
        labels={'cdi_smooth': 'CDI (%)', 'volatility': 'GDP Volatility'},
        color_continuous_scale='Viridis'
    )
    
    # Add trendline if enough data points
    if len(df_plot) >= 3:
        try:
            z = np.polyfit(df_plot['cdi_smooth'], df_plot['volatility'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_plot['cdi_smooth'].min(), 
                                  df_plot['cdi_smooth'].max(), 100)
            
            fig_scatter.add_trace(
                go.Scatter(x=x_trend, y=p(x_trend), mode='lines', 
                          name='Trend', line=dict(color='red', dash='dash'))
            )
        except:
            pass  # Skip trendline if it fails
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Model prediction
    if model is not None:
        st.markdown("---")
        st.subheader("🤖 Volatility Prediction")
        
        # Get latest data
        latest_data = df_country.iloc[-1]
        
        features = pd.DataFrame({
            'cdi_smooth_lag1': [latest_data['cdi_smooth']],
            'inflation_lag1': [latest_data['inflation']],
            'trade_openness_lag1': [latest_data['trade_openness']],
            'investment_lag1': [latest_data['investment']]
        })
        
        try:
            prediction = model.predict(features)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Predicted Volatility (next year):** {np.exp(prediction):.2f}")
                
                # Interpretation
                if np.exp(prediction) > 2.0:
                    st.warning("⚠️ HIGH volatility expected")
                elif np.exp(prediction) > 1.0:
                    st.info("ℹ️ MODERATE volatility expected")
                else:
                    st.success("✅ LOW volatility expected")
            
            with col2:
                # Feature contribution (simplified)
                st.write("**Key Risk Factors:**")
                
                risk_factors = []
                if latest_data['cdi_smooth'] > 70:
                    risk_factors.append("🔴 High commodity dependence")
                if latest_data['inflation'] > 10:
                    risk_factors.append("🔴 High inflation")
                if latest_data['investment'] < 20:
                    risk_factors.append("🔴 Low investment")
                if latest_data['trade_openness'] > 80:
                    risk_factors.append("🟡 High trade exposure")
                
                if not risk_factors:
                    st.success("✅ No major risk factors identified")
                else:
                    for factor in risk_factors:
                        st.write(factor)
        
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def show_gdp_forecasting(df, country_name, gdp_model):
    """Show GDP growth forecasting."""
    st.header(f"🔮 GDP Growth Forecasting: {country_name}")
    
    if gdp_model is None:
        st.warning("⚠️ GDP Forecasting model not available yet. Train it first with:")
        st.code("python src/models/gdp_forecaster.py")
        st.info("💡 For now, this is a placeholder. The model will predict GDP growth for upcoming years.")
        return
    
    country_code = COUNTRY_CODES[country_name]
    df_country = df[df['country'] == country_code].copy()
    
    if len(df_country) == 0:
        st.warning(f"No data available for {country_name}")
        return
    
    # Forecast settings
    st.sidebar.subheader("🔮 Forecast Settings")
    
    forecast_years = st.sidebar.slider("Years to forecast:", 1, 10, 5)
    
    st.sidebar.subheader("📊 Scenario Analysis")
    scenario_mode = st.sidebar.checkbox("Enable scenario analysis")
    
    scenario = {}
    if scenario_mode:
        st.sidebar.write("Adjust parameters (annual change):")
        scenario['cdi_smooth'] = st.sidebar.slider("CDI change per year:", -10.0, 10.0, 0.0)
        scenario['inflation'] = st.sidebar.slider("Inflation change per year:", -5.0, 5.0, 0.0)
        scenario['investment'] = st.sidebar.slider("Investment change per year:", -5.0, 5.0, 0.0)
    
    # Get latest features
    latest_data = df_country.iloc[-1]
    
    initial_features = pd.DataFrame({
        'cdi_smooth': [latest_data['cdi_smooth']],
        'inflation': [latest_data['inflation']],
        'trade_openness': [latest_data['trade_openness']],
        'investment': [latest_data['investment']]
    })
    
    # Make predictions
    try:
        if scenario_mode and scenario:
            predictions = gdp_model.predict_multi_year(
                initial_features, years=forecast_years, scenario=scenario
            )
        else:
            predictions = gdp_model.predict_multi_year(
                initial_features, years=forecast_years
            )
        
        # Display predictions
        st.subheader("📈 GDP Growth Forecast")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            next_year_gdp = predictions.iloc[0]['predicted_gdp_growth']
            st.metric("Next Year GDP Growth", f"{next_year_gdp:.2f}%")
        
        with col2:
            avg_forecast = predictions['predicted_gdp_growth'].mean()
            st.metric(f"Avg Growth ({forecast_years}Y)", f"{avg_forecast:.2f}%")
        
        with col3:
            uncertainty = predictions['uncertainty'].mean()
            st.metric("Avg Uncertainty", f"±{uncertainty:.2f}%")
        
        # Forecast chart
        predictions['year'] = latest_data['year'] + predictions['year_ahead']
        
        fig_forecast = go.Figure()
        
        # Add prediction line
        fig_forecast.add_trace(
            go.Scatter(
                x=predictions['year'],
                y=predictions['predicted_gdp_growth'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#1f77b4', width=3)
            )
        )
        
        # Add confidence interval
        fig_forecast.add_trace(
            go.Scatter(
                x=predictions['year'].tolist() + predictions['year'].tolist()[::-1],
                y=predictions['upper_95'].tolist() + predictions['lower_95'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            )
        )
        
        fig_forecast.update_layout(
            title=f'{country_name}: GDP Growth Forecast ({forecast_years} years)',
            xaxis_title='Year',
            yaxis_title='GDP Growth (%)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Predictions table
        st.subheader("📊 Detailed Forecast")
        
        display_predictions = predictions[[
            'year', 'predicted_gdp_growth', 'lower_95', 'upper_95', 'uncertainty'
        ]].copy()
        
        display_predictions.columns = [
            'Year', 'Forecast (%)', 'Lower 95% (%)', 'Upper 95% (%)', 'Uncertainty (±%)'
        ]
        
        st.dataframe(
            display_predictions.style.format({
                'Forecast (%)': '{:.2f}',
                'Lower 95% (%)': '{:.2f}',
                'Upper 95% (%)': '{:.2f}',
                'Uncertainty (±%)': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Download button
        csv = display_predictions.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv,
            file_name=f"{country_name}_gdp_forecast.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Forecasting failed: {e}")


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
    
    # Average metrics by country
    st.subheader("📊 Average Metrics by Country")
    
    summary = df_filtered.groupby('country_name').agg({
        'cdi_smooth': 'mean',
        'gdp_volatility': 'mean',
        'inflation': 'mean',
        'investment': 'mean'
    }).round(2)
    
    summary.columns = ['Avg CDI (%)', 'Avg Volatility', 'Avg Inflation (%)', 'Avg Investment (% GDP)']
    
    st.dataframe(summary, use_container_width=True)
    
    # Comparative visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 CDI Comparison")
        
        fig_cdi = px.bar(
            summary.reset_index(),
            x='country_name',
            y='Avg CDI (%)',
            color='Avg CDI (%)',
            color_continuous_scale='Reds',
            title='Average Commodity Dependence Index'
        )
        st.plotly_chart(fig_cdi, use_container_width=True)
    
    with col2:
        st.subheader("📉 Volatility Comparison")
        
        fig_vol = px.bar(
            summary.reset_index(),
            x='country_name',
            y='Avg Volatility',
            color='Avg Volatility',
            color_continuous_scale='Oranges',
            title='Average GDP Growth Volatility'
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Scatter plot: CDI vs Volatility for all countries
    st.subheader("🔍 CDI vs Volatility (All Countries)")
    
    # Prepare data for plotting - remove NaN values
    plot_cols = ['cdi_smooth', 'gdp_volatility', 'country_name', 'investment', 'year', 'inflation']
    df_plot = df_filtered[plot_cols].dropna()
    
    if len(df_plot) == 0:
        st.warning("⚠️ Not enough data for scatter plot")
    else:
        fig_scatter = px.scatter(
            df_plot,
            x='cdi_smooth',
            y='gdp_volatility',
            color='country_name',
            size='investment',
            hover_data=['year', 'inflation'],
            title='Commodity Dependence vs Economic Volatility',
            labels={'cdi_smooth': 'CDI (%)', 'gdp_volatility': 'Volatility'}
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)


# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>African Commodities Paradox Analyzer</strong></p>
        <p>Author: Abraham Adegoke | HEC Lausanne | December 2025</p>
        <p>Powered by Machine Learning (Gradient Boosting) | Data: World Bank WDI (1992-2023)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    show_footer()