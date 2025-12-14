"""
Time Series Analysis Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import warnings
import logging

from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    def __init__(self):
        self.data = None
        self.country = None
        self.target = None
        self.decomposition = None
        self.arima_model = None
        self.arima_fitted = None
        self.trend_results = None
        logger.info("TimeSeriesAnalyzer initialized")
    
    def fit(self, df: pd.DataFrame, country: str, target: str = 'gdp_growth') -> 'TimeSeriesAnalyzer':
        logger.info(f"Fitting TimeSeriesAnalyzer for {country}, target={target}")
        self.country = country
        self.target = target
        country_data = df[df['country'] == country].copy()
        country_data = country_data.sort_values('year')
        if len(country_data) < 5:
            raise ValueError(f"Not enough data for {country}. Need at least 5 years.")
        if target not in country_data.columns:
            raise ValueError(f"Target '{target}' not found in data")
        country_data[target] = country_data[target].interpolate(method='linear')
        country_data = country_data.dropna(subset=[target])
        self.data = country_data
        logger.info(f"  Data prepared: {len(self.data)} observations")
        return self
    
    def analyze_trend(self) -> Dict:
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        logger.info("Analyzing trend...")
        years = self.data['year'].values
        values = self.data[self.target].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
        trend_line = slope * years + intercept
        if p_value < 0.05:
            trend_direction = "Increasing" if slope > 0 else "Decreasing"
        else:
            trend_direction = "No significant trend"
        self.trend_results = {
            'slope': slope, 'intercept': intercept, 'r_squared': r_value ** 2,
            'p_value': p_value, 'std_err': std_err, 'trend_direction': trend_direction,
            'annual_change': slope, 'total_change': slope * (years[-1] - years[0]),
            'trend_line': trend_line, 'is_significant': p_value < 0.05
        }
        return self.trend_results
    
    def decompose(self, period: int = None) -> Dict:
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        ts = self.data.set_index('year')[self.target]
        if period is None:
            period = min(4, len(ts) // 2)
        if len(ts) < 2 * period:
            period = 2
        try:
            self.decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')
            return {'trend': self.decomposition.trend, 'seasonal': self.decomposition.seasonal,
                    'residual': self.decomposition.resid, 'observed': self.decomposition.observed, 'period': period}
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return None
    
    def test_stationarity(self) -> Dict:
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        ts = self.data[self.target].values
        adf_result = adfuller(ts, autolag='AIC')
        return {'adf_statistic': adf_result[0], 'p_value': adf_result[1], 'used_lag': adf_result[2],
                'n_obs': adf_result[3], 'critical_values': adf_result[4], 'is_stationary': adf_result[1] < 0.05}
    
    def compute_autocorrelation(self, nlags: int = 10) -> Dict:
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        ts = self.data[self.target].values
        max_lags = len(ts) // 2 - 1
        nlags = min(nlags, max_lags)
        if nlags < 2:
            return None
        acf_values = acf(ts, nlags=nlags)
        pacf_values = pacf(ts, nlags=nlags)
        ci = 1.96 / np.sqrt(len(ts))
        return {'acf': acf_values, 'pacf': pacf_values, 'nlags': nlags, 'confidence_interval': ci,
                'significant_acf_lags': [i for i, v in enumerate(acf_values) if abs(v) > ci and i > 0],
                'significant_pacf_lags': [i for i, v in enumerate(pacf_values) if abs(v) > ci and i > 0]}
    
    def fit_arima(self, order: Tuple[int, int, int] = None, auto_select: bool = True) -> Dict:
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        ts = self.data[self.target].values
        if order is None and auto_select:
            order = self._select_arima_order(ts)
        elif order is None:
            order = (1, 0, 1)
        try:
            self.arima_model = ARIMA(ts, order=order)
            self.arima_fitted = self.arima_model.fit()
            fitted_values = self.arima_fitted.fittedvalues
            mse = mean_squared_error(ts[1:], fitted_values[1:])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(ts[1:], fitted_values[1:])
            ss_res = np.sum((ts[1:] - fitted_values[1:]) ** 2)
            ss_tot = np.sum((ts[1:] - np.mean(ts[1:])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            return {'order': order, 'aic': self.arima_fitted.aic, 'bic': self.arima_fitted.bic,
                    'rmse': rmse, 'mae': mae, 'r2': r2, 'fitted_values': fitted_values, 'residuals': self.arima_fitted.resid}
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            return None
    
    def _select_arima_order(self, ts: np.ndarray, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        best_aic = np.inf
        best_order = (1, 0, 1)
        adf_result = adfuller(ts)
        d = 0 if adf_result[1] < 0.05 else 1
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
        return best_order
    
    def forecast(self, periods: int = 3) -> pd.DataFrame:
        if self.arima_fitted is None:
            raise ValueError("Must fit ARIMA model first")
        forecast_result = self.arima_fitted.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        last_year = int(self.data['year'].max())
        forecast_years = [last_year + i + 1 for i in range(periods)]
        
        # Handle numpy array vs pandas
        if isinstance(forecast_mean, np.ndarray):
            forecast_values = forecast_mean
        else:
            forecast_values = forecast_mean.values
        
        if isinstance(forecast_ci, np.ndarray):
            lower_ci = forecast_ci[:, 0]
            upper_ci = forecast_ci[:, 1]
        else:
            lower_ci = forecast_ci.iloc[:, 0].values
            upper_ci = forecast_ci.iloc[:, 1].values
        
        return pd.DataFrame({'year': forecast_years, 'forecast': forecast_values, 'lower_ci': lower_ci, 'upper_ci': upper_ci})
    
    def plot_decomposition(self, save_path: Optional[str] = None) -> plt.Figure:
        if self.decomposition is None:
            self.decompose()
        if self.decomposition is None:
            return None
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        self.decomposition.observed.plot(ax=axes[0], title='Observed')
        self.decomposition.trend.plot(ax=axes[1], title='Trend')
        self.decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        self.decomposition.resid.plot(ax=axes[3], title='Residual')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_forecast(self, periods: int = 3, save_path: Optional[str] = None) -> plt.Figure:
        if self.arima_fitted is None:
            self.fit_arima()
        forecast_df = self.forecast(periods)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data['year'], self.data[self.target], 'b-o', label='Historical', linewidth=2)
        ax.plot(forecast_df['year'], forecast_df['forecast'], 'r--o', label='Forecast', linewidth=2)
        ax.fill_between(forecast_df['year'], forecast_df['lower_ci'], forecast_df['upper_ci'], color='red', alpha=0.2)
        ax.set_xlabel('Year')
        ax.set_ylabel(self.target)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_acf_pacf(self, nlags: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        if self.data is None:
            raise ValueError("Must fit analyzer first")
        ts = self.data[self.target].values
        max_lags = min(nlags, len(ts) // 2 - 1)
        if max_lags < 2:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(ts, lags=max_lags, ax=axes[0])
        plot_pacf(ts, lags=max_lags, ax=axes[1])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def get_summary_report(self) -> str:
        if self.data is None:
            return "No analysis performed yet."
        report = f"\nTIME SERIES ANALYSIS REPORT: {self.country}\n"
        report += f"Target: {self.target}\n"
        report += f"Period: {int(self.data['year'].min())} - {int(self.data['year'].max())}\n"
        report += f"Observations: {len(self.data)}\n"
        if self.trend_results:
            report += f"Trend: {self.trend_results['trend_direction']}\n"
        return report


def analyze_country_time_series(df: pd.DataFrame, country: str, target: str = 'gdp_growth',
                                 forecast_periods: int = 3, save_dir: Optional[str] = None) -> Dict:
    from pathlib import Path
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    analyzer = TimeSeriesAnalyzer()
    try:
        analyzer.fit(df, country, target)
    except ValueError as e:
        return {'error': str(e)}
    results = {'country': country, 'target': target, 'data': analyzer.data}
    results['trend'] = analyzer.analyze_trend()
    results['stationarity'] = analyzer.test_stationarity()
    results['autocorrelation'] = analyzer.compute_autocorrelation()
    results['decomposition'] = analyzer.decompose()
    results['arima'] = analyzer.fit_arima()
    if results['arima']:
        results['forecast'] = analyzer.forecast(forecast_periods)
    results['report'] = analyzer.get_summary_report()
    results['analyzer'] = analyzer
    return results


def compare_country_trends(df: pd.DataFrame, countries: List[str], target: str = 'gdp_growth') -> pd.DataFrame:
    comparisons = []
    for country in countries:
        try:
            analyzer = TimeSeriesAnalyzer()
            analyzer.fit(df, country, target)
            trend = analyzer.analyze_trend()
            comparisons.append({'country': country, 'trend_direction': trend['trend_direction'],
                               'annual_change': trend['annual_change'], 'total_change': trend['total_change'],
                               'r_squared': trend['r_squared'], 'p_value': trend['p_value'], 'is_significant': trend['is_significant']})
        except:
            continue
    return pd.DataFrame(comparisons)
