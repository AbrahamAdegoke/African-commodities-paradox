"""
Time Series Analysis Module

Implements time series techniques to analyze temporal patterns
in African economic data and improve GDP forecasting.

Techniques:
- Trend Analysis: Detect long-term trends in CDI, GDP, etc.
- Seasonal Decomposition: Separate trend, seasonality, and residuals
- Autocorrelation Analysis: Measure temporal dependencies
- ARIMA Forecasting: Improved GDP predictions using time series models

Author: Abraham Adegoke
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import warnings
import logging

# Statistical imports
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """
    Analyzes temporal patterns in economic data for African countries.
    
    This addresses key questions:
    - Is there a trend in commodity dependence over time?
    - Are economic cycles predictable?
    - Can we forecast GDP growth using time series methods?
    
    Example:
        >>> analyzer = TimeSeriesAnalyzer()
        >>> analyzer.fit(df, country='NGA', target='gdp_growth')
        >>> forecast = analyzer.forecast(periods=3)
    """
    
    def __init__(self):
        """Initialize the Time Series Analyzer."""
        self.data = None
        self.country = None
        self.target = None
        self.decomposition = None
        self.arima_model = None
        self.arima_fitted = None
        self.trend_results = None
        
        logger.info("TimeSeriesAnalyzer initialized")
    
    def fit(
        self, 
        df: pd.DataFrame, 
        country: str, 
        target: str = 'gdp_growth'
    ) -> 'TimeSeriesAnalyzer':
        """
        Fit the analyzer on a specific country's time series.
        
        Args:
            df: DataFrame with country, year, and target columns
            country: ISO3 country code
            target: Target variable to analyze
            
        Returns:
            self
        """
        logger.info(f"Fitting TimeSeriesAnalyzer for {country}, target={target}")
        
        self.country = country
        self.target = target
        
        # Filter and sort data
        country_data = df[df['country'] == country].copy()
        country_data = country_data.sort_values('year')
        
        if len(country_data) < 5:
            raise ValueError(f"Not enough data for {country}. Need at least 5 years.")
        
        # Check if target exists
        if target not in country_data.columns:
            raise ValueError(f"Target '{target}' not found in data")
        
        # Handle missing values with interpolation
        country_data[target] = country_data[target].interpolate(method='linear')
        country_data = country_data.dropna(subset=[target])
        
        self.data = country_data
        
        logger.info(f"  Data prepared: {len(self.data)} observations")
        logger.info(f"  Period: {int(self.data['year'].min())}-{int(self.data['year'].max())}")
        
        return self
    
    def analyze_trend(self) -> Dict:
        """
        Analyze the long-term trend in the target variable.
        
        Returns:
            Dictionary with trend statistics
        """
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        
        logger.info("Analyzing trend...")
        
        years = self.data['year'].values
        values = self.data[self.target].values
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
        
        # Calculate trend line
        trend_line = slope * years + intercept
        
        # Determine trend direction
        if p_value < 0.05:
            if slope > 0:
                trend_direction = "Increasing"
            else:
                trend_direction = "Decreasing"
        else:
            trend_direction = "No significant trend"
        
        # Calculate annual change
        annual_change = slope
        total_change = slope * (years[-1] - years[0])
        
        self.trend_results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'trend_direction': trend_direction,
            'annual_change': annual_change,
            'total_change': total_change,
            'trend_line': trend_line,
            'is_significant': p_value < 0.05
        }
        
        logger.info(f"  Trend: {trend_direction}")
        logger.info(f"  Slope: {slope:.4f} per year")
        logger.info(f"  P-value: {p_value:.4f}")
        
        return self.trend_results
    
    def decompose(self, period: int = None) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            period: Seasonal period (default: auto-detect or 4)
            
        Returns:
            Dictionary with decomposition results
        """
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        
        logger.info("Decomposing time series...")
        
        # Prepare time series
        ts = self.data.set_index('year')[self.target]
        
        # Auto-detect period or use default
        if period is None:
            period = min(4, len(ts) // 2)  # Default to 4 or half the data length
        
        if len(ts) < 2 * period:
            logger.warning(f"Not enough data for period={period}, using period=2")
            period = 2
        
        try:
            # Perform decomposition
            self.decomposition = seasonal_decompose(
                ts, 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            results = {
                'trend': self.decomposition.trend,
                'seasonal': self.decomposition.seasonal,
                'residual': self.decomposition.resid,
                'observed': self.decomposition.observed,
                'period': period
            }
            
            logger.info(f"  Decomposition complete (period={period})")
            
            return results
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return None
    
    def test_stationarity(self) -> Dict:
        """
        Test if the time series is stationary using Augmented Dickey-Fuller test.
        
        Stationarity is important for ARIMA modeling.
        
        Returns:
            Dictionary with test results
        """
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        
        logger.info("Testing stationarity (ADF test)...")
        
        ts = self.data[self.target].values
        
        # Perform ADF test
        adf_result = adfuller(ts, autolag='AIC')
        
        results = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'used_lag': adf_result[2],
            'n_obs': adf_result[3],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        if results['is_stationary']:
            logger.info(f"  Series IS stationary (p={results['p_value']:.4f})")
        else:
            logger.info(f"  Series is NOT stationary (p={results['p_value']:.4f})")
        
        return results
    
    def compute_autocorrelation(self, nlags: int = 10) -> Dict:
        """
        Compute autocorrelation and partial autocorrelation functions.
        
        This helps understand temporal dependencies in the data.
        
        Args:
            nlags: Number of lags to compute
            
        Returns:
            Dictionary with ACF and PACF values
        """
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        
        logger.info(f"Computing autocorrelation (lags={nlags})...")
        
        ts = self.data[self.target].values
        
        # Adjust nlags if necessary
        max_lags = len(ts) // 2 - 1
        nlags = min(nlags, max_lags)
        
        if nlags < 2:
            logger.warning("Not enough data for autocorrelation analysis")
            return None
        
        # Compute ACF and PACF
        acf_values = acf(ts, nlags=nlags)
        pacf_values = pacf(ts, nlags=nlags)
        
        # Find significant lags (outside 95% confidence interval)
        confidence_interval = 1.96 / np.sqrt(len(ts))
        significant_acf_lags = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
        significant_pacf_lags = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]
        
        results = {
            'acf': acf_values,
            'pacf': pacf_values,
            'nlags': nlags,
            'confidence_interval': confidence_interval,
            'significant_acf_lags': significant_acf_lags,
            'significant_pacf_lags': significant_pacf_lags
        }
        
        logger.info(f"  Significant ACF lags: {significant_acf_lags}")
        logger.info(f"  Significant PACF lags: {significant_pacf_lags}")
        
        return results
    
    def fit_arima(
        self, 
        order: Tuple[int, int, int] = None,
        auto_select: bool = True
    ) -> Dict:
        """
        Fit an ARIMA model for forecasting.
        
        Args:
            order: (p, d, q) order for ARIMA. If None, auto-select.
            auto_select: Whether to automatically select best order
            
        Returns:
            Dictionary with model results
        """
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        
        logger.info("Fitting ARIMA model...")
        
        ts = self.data[self.target].values
        
        # Auto-select order if not provided
        if order is None and auto_select:
            order = self._select_arima_order(ts)
        elif order is None:
            order = (1, 0, 1)  # Default order
        
        logger.info(f"  Using order: {order}")
        
        try:
            # Fit ARIMA model
            self.arima_model = ARIMA(ts, order=order)
            self.arima_fitted = self.arima_model.fit()
            
            # Get fitted values
            fitted_values = self.arima_fitted.fittedvalues
            
            # Calculate in-sample metrics
            mse = mean_squared_error(ts[1:], fitted_values[1:])  # Skip first value
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(ts[1:], fitted_values[1:])
            
            # Calculate R² (comparing to mean baseline)
            ss_res = np.sum((ts[1:] - fitted_values[1:]) ** 2)
            ss_tot = np.sum((ts[1:] - np.mean(ts[1:])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results = {
                'order': order,
                'aic': self.arima_fitted.aic,
                'bic': self.arima_fitted.bic,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'fitted_values': fitted_values,
                'residuals': self.arima_fitted.resid
            }
            
            logger.info(f"  AIC: {results['aic']:.2f}")
            logger.info(f"  RMSE: {results['rmse']:.4f}")
            logger.info(f"  R²: {results['r2']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            return None
    
    def _select_arima_order(
        self, 
        ts: np.ndarray, 
        max_p: int = 3, 
        max_d: int = 2, 
        max_q: int = 3
    ) -> Tuple[int, int, int]:
        """Auto-select ARIMA order using AIC."""
        
        best_aic = np.inf
        best_order = (1, 0, 1)
        
        # Test stationarity to determine d
        adf_result = adfuller(ts)
        d = 0 if adf_result[1] < 0.05 else 1
        
        # Grid search for p and q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        
                except:
                    continue
        
        logger.info(f"  Best order selected: {best_order} (AIC={best_aic:.2f})")
        return best_order
    
    def forecast(self, periods: int = 3) -> pd.DataFrame:
        """
        Forecast future values using the fitted ARIMA model.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self.arima_fitted is None:
            raise ValueError("Must fit ARIMA model first")
        
        logger.info(f"Forecasting {periods} periods ahead...")
        
        # Get forecast
        forecast_result = self.arima_fitted.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        # Create forecast DataFrame
        last_year = int(self.data['year'].max())
        forecast_years = [last_year + i + 1 for i in range(periods)]
        
        forecast_df = pd.DataFrame({
            'year': forecast_years,
            'forecast': forecast_mean.values,
            'lower_ci': forecast_ci.iloc[:, 0].values,
            'upper_ci': forecast_ci.iloc[:, 1].values
        })
        
        logger.info(f"  Forecast for {forecast_years}:")
        for _, row in forecast_df.iterrows():
            logger.info(f"    {int(row['year'])}: {row['forecast']:.2f} [{row['lower_ci']:.2f}, {row['upper_ci']:.2f}]")
        
        return forecast_df
    
    def plot_decomposition(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot time series decomposition."""
        if self.decomposition is None:
            self.decompose()
        
        if self.decomposition is None:
            return None
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        self.decomposition.observed.plot(ax=axes[0], title='Observed')
        axes[0].set_ylabel(self.target)
        
        self.decomposition.trend.plot(ax=axes[1], title='Trend')
        axes[1].set_ylabel('Trend')
        
        self.decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        axes[2].set_ylabel('Seasonal')
        
        self.decomposition.resid.plot(ax=axes[3], title='Residual')
        axes[3].set_ylabel('Residual')
        
        plt.suptitle(f'Time Series Decomposition: {self.country} - {self.target}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_forecast(self, periods: int = 3, save_path: Optional[str] = None) -> plt.Figure:
        """Plot historical data with forecast."""
        if self.arima_fitted is None:
            self.fit_arima()
        
        forecast_df = self.forecast(periods)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(self.data['year'], self.data[self.target], 
                'b-o', label='Historical', linewidth=2, markersize=6)
        
        # Plot forecast
        ax.plot(forecast_df['year'], forecast_df['forecast'], 
                'r--o', label='Forecast', linewidth=2, markersize=8)
        
        # Plot confidence interval
        ax.fill_between(
            forecast_df['year'],
            forecast_df['lower_ci'],
            forecast_df['upper_ci'],
            color='red',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(self.target.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'ARIMA Forecast: {self.country} - {self.target}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_acf_pacf(self, nlags: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """Plot ACF and PACF."""
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        
        ts = self.data[self.target].values
        max_lags = min(nlags, len(ts) // 2 - 1)
        
        if max_lags < 2:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        plot_acf(ts, lags=max_lags, ax=axes[0])
        axes[0].set_title(f'Autocorrelation Function (ACF): {self.country}')
        
        plot_pacf(ts, lags=max_lags, ax=axes[1])
        axes[1].set_title(f'Partial Autocorrelation Function (PACF): {self.country}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if self.data is None:
            return "No analysis performed yet."
        
        report = "\n" + "=" * 80 + "\n"
        report += f"TIME SERIES ANALYSIS REPORT: {self.country}\n"
        report += f"Target Variable: {self.target}\n"
        report += "=" * 80 + "\n\n"
        
        # Data summary
        report += "DATA SUMMARY\n"
        report += "-" * 40 + "\n"
        report += f"Period: {int(self.data['year'].min())} - {int(self.data['year'].max())}\n"
        report += f"Observations: {len(self.data)}\n"
        report += f"Mean: {self.data[self.target].mean():.2f}\n"
        report += f"Std Dev: {self.data[self.target].std():.2f}\n"
        report += f"Min: {self.data[self.target].min():.2f}\n"
        report += f"Max: {self.data[self.target].max():.2f}\n\n"
        
        # Trend analysis
        if self.trend_results:
            report += "TREND ANALYSIS\n"
            report += "-" * 40 + "\n"
            report += f"Direction: {self.trend_results['trend_direction']}\n"
            report += f"Annual change: {self.trend_results['annual_change']:.4f}\n"
            report += f"Total change: {self.trend_results['total_change']:.2f}\n"
            report += f"R²: {self.trend_results['r_squared']:.4f}\n"
            report += f"P-value: {self.trend_results['p_value']:.4f}\n"
            report += f"Statistically significant: {'Yes' if self.trend_results['is_significant'] else 'No'}\n\n"
        
        # ARIMA results
        if self.arima_fitted:
            report += "ARIMA MODEL\n"
            report += "-" * 40 + "\n"
            report += f"Order: {self.arima_model.order}\n"
            report += f"AIC: {self.arima_fitted.aic:.2f}\n"
            report += f"BIC: {self.arima_fitted.bic:.2f}\n\n"
        
        report += "=" * 80 + "\n"
        
        return report


def analyze_country_time_series(
    df: pd.DataFrame,
    country: str,
    target: str = 'gdp_growth',
    forecast_periods: int = 3,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Convenience function to run complete time series analysis for a country.
    
    Args:
        df: Data with country, year, and target columns
        country: ISO3 country code
        target: Target variable
        forecast_periods: Number of periods to forecast
        save_dir: Directory to save outputs
        
    Returns:
        Dictionary with all analysis results
    """
    from pathlib import Path
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = TimeSeriesAnalyzer()
    
    try:
        analyzer.fit(df, country, target)
    except ValueError as e:
        logger.error(f"Could not analyze {country}: {e}")
        return {'error': str(e)}
    
    results = {
        'country': country,
        'target': target,
        'data': analyzer.data
    }
    
    # Trend analysis
    results['trend'] = analyzer.analyze_trend()
    
    # Stationarity test
    results['stationarity'] = analyzer.test_stationarity()
    
    # Autocorrelation
    results['autocorrelation'] = analyzer.compute_autocorrelation()
    
    # Decomposition
    results['decomposition'] = analyzer.decompose()
    
    # ARIMA
    results['arima'] = analyzer.fit_arima()
    
    # Forecast
    if results['arima']:
        results['forecast'] = analyzer.forecast(forecast_periods)
    
    # Generate report
    results['report'] = analyzer.get_summary_report()
    print(results['report'])
    
    # Save plots
    if save_dir:
        analyzer.plot_decomposition(f'{save_dir}/decomposition_{country}.png')
        analyzer.plot_forecast(forecast_periods, f'{save_dir}/forecast_{country}.png')
        analyzer.plot_acf_pacf(save_path=f'{save_dir}/acf_pacf_{country}.png')
        
        with open(f'{save_dir}/report_{country}.txt', 'w') as f:
            f.write(results['report'])
    
    results['analyzer'] = analyzer
    
    return results


def compare_country_trends(
    df: pd.DataFrame,
    countries: List[str],
    target: str = 'gdp_growth'
) -> pd.DataFrame:
    """
    Compare trends across multiple countries.
    
    Args:
        df: Data
        countries: List of ISO3 country codes
        target: Target variable
        
    Returns:
        DataFrame with trend comparison
    """
    comparisons = []
    
    for country in countries:
        try:
            analyzer = TimeSeriesAnalyzer()
            analyzer.fit(df, country, target)
            trend = analyzer.analyze_trend()
            
            comparisons.append({
                'country': country,
                'trend_direction': trend['trend_direction'],
                'annual_change': trend['annual_change'],
                'total_change': trend['total_change'],
                'r_squared': trend['r_squared'],
                'p_value': trend['p_value'],
                'is_significant': trend['is_significant']
            })
        except Exception as e:
            logger.warning(f"Could not analyze {country}: {e}")
            continue
    
    return pd.DataFrame(comparisons)


# Example usage
if __name__ == "__main__":
    print("Testing Time Series Module...")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    years = list(range(2000, 2024))
    n_years = len(years)
    
    # Create synthetic GDP growth with trend and noise
    trend = np.linspace(3, 5, n_years)  # Slight upward trend
    seasonal = 0.5 * np.sin(np.linspace(0, 4 * np.pi, n_years))  # Cycles
    noise = np.random.normal(0, 1, n_years)
    
    gdp_growth = trend + seasonal + noise
    
    df = pd.DataFrame({
        'country': ['NGA'] * n_years,
        'year': years,
        'gdp_growth': gdp_growth,
        'cdi_smooth': np.random.uniform(60, 80, n_years)
    })
    
    # Run analysis
    results = analyze_country_time_series(
        df, 
        country='NGA', 
        target='gdp_growth',
        forecast_periods=3,
        save_dir='test_timeseries'
    )
    
    print("\n✓ Time Series test completed!")
