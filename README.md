# African Commodities Paradox: A Data-Driven Analysis

**Analyzing the relationship between commodity dependence and economic volatility across 52 African countries (1990-2023)**

A machine learning project investigating why resource-rich African economies often experience higher GDP growth volatility—the "African Commodities Paradox."

**Author:** Abraham Adegoke  
**Institution:** HEC Lausanne  
**Course:** Advanced Programming (Fall 2025)

---

## Project Overview

### The Problem

Many African economies rely heavily on commodity exports (oil, minerals, agricultural products), yet this dependence often leads to unstable and volatile economic growth. This project builds a data-driven framework to:

1. **Quantify** commodity dependence using a custom Commodity Dependence Index (CDI)
2. **Predict** GDP growth volatility using machine learning
3. **Identify** country clusters with different economic profiles
4. **Analyze** temporal trends and forecast future growth

### Key Research Questions

- Does commodity dependence increase GDP growth volatility?
- Which factors (governance, inflation, trade openness, investment) are the strongest predictors?
- Are there distinct groups of African economies with different risk profiles?
- Can good governance overcome the "resource curse"?

---

## Key Results

### Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Gradient Boosting** | **0.434** | 3.95 | 1.81 |
| Ridge Regression | 0.074 | 5.05 | 2.66 |

### Clustering Analysis (K-Means, k=3)

| Cluster | Countries | CDI | Governance | Volatility | Profile |
|---------|-----------|-----|------------|------------|---------|
| 0 | Botswana, Mauritius, Tunisia... | 30% | +0.19 | 3.25% | "Escaped Paradox" |
| 1 | Libya, South Sudan, Zimbabwe... | 17% | -1.28 | 12.44% | "Fragile States" |
| 2 | Nigeria, Kenya, Ghana... (33 countries) | 52% | -0.70 | 3.00% | "Typical Africa" |

### PCA Results

- 3 components explain **73.2%** of variance
- PC1 represents "Instability vs Stability" axis

### Main Finding

**The commodities paradox exists but is conditional.** High CDI countries grow 0.24% slower on average, but **governance is the key factor**. The "Fragile States" cluster proves this: despite LOWER commodity dependence, they have 4x MORE volatility due to poor governance. **Botswana** demonstrates that good institutions can overcome the resource curse.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/AbrahamAdegoke/African-commodities-paradox.git
cd African-commodities-paradox

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run the complete analysis pipeline
python main.py
```

Expected output:
- Data loading (1768 observations, 52 countries)
- Supervised learning (Ridge R²=0.074, Gradient Boosting R²=0.434)
- Unsupervised learning (3 clusters, PCA 73.2% variance)
- Key findings and insights

### Interactive Dashboard

```bash
# Launch the Streamlit web application
streamlit run app.py
```

Features:
- Country-by-country analysis
- GDP volatility predictions
- Cluster visualization
- Time series analysis and forecasting

### Running Tests

```bash
# Run all 141 tests with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

Current status: **141 tests passing, 88% coverage**

---

## Project Structure

```
african-commodities-paradox/
├── main.py                     # Main entry point (run this!)
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── project_report.pdf          # Technical report (10 pages)
├── README.md                   # This file
├── PROPOSAL.md                 # Project proposal
├── AI_USAGE.md                 # AI tools disclosure
│
├── src/                        # Source code
│   ├── data_io/
│   │   └── worldbank.py        # World Bank API client
│   ├── preprocessing/
│   │   └── preprocessing.py    # Data cleaning, feature engineering
│   ├── models/
│   │   ├── ridge_regression.py # Ridge with cross-validation
│   │   └── gradient_boosting.py# GB with GridSearchCV
│   ├── analysis/
│   │   ├── clustering.py       # K-Means, Hierarchical clustering
│   │   ├── pca_analysis.py     # Principal Component Analysis
│   │   └── time_series.py      # ARIMA, trend analysis
│   └── evaluation/
│       └── metrics.py          # R², RMSE, MAE, SHAP
│
├── tests/                      # Unit tests (141 tests, 88% coverage)
│   ├── test_clustering.py
│   ├── test_pca.py
│   ├── test_time_series.py
│   ├── test_models.py
│   ├── test_preprocessing.py
│   ├── test_data_io.py
│   └── test_evaluation.py
│
├── data/
│   ├── raw/                    # Raw World Bank data
│   └── processed/              # Feature-engineered data
│
├── configs/
│   └── countries.yaml          # List of African countries
│
├── notebooks/
│   └── 00_quickstart.ipynb     # Interactive exploration
│
├── results/                    # Model outputs and figures
│   └── figures/
│
└── scripts/                    # Pipeline scripts
    ├── download_data.py
    ├── preprocess_data.py
    └── train_models.py
```

---

## Methodology

### 1. Commodity Dependence Index (CDI)

```
CDI = Fuel Exports (%) + Metals Exports (%) + Agri Exports (%) + Food Exports (%)
```

- Each component is already expressed as % of total exports (World Bank data)
- Smoothed with 3-year moving average to reduce noise
- Example: Nigeria CDI = 97% (91.6% fuel + 1.2% metals + 0.4% agri + 3.8% food)

### 2. GDP Volatility (Target Variable)

```
GDP_volatility = Rolling standard deviation of GDP growth (5-year window)
```

### 3. Machine Learning Models

**Supervised Learning:**
- Ridge Regression: L2 regularization, 50 alphas tested with 5-fold CV
- Gradient Boosting: 216 hyperparameter combinations via GridSearchCV

**Unsupervised Learning:**
- K-Means Clustering (k=3): Silhouette score = 0.31
- Hierarchical Clustering: Ward linkage
- PCA: 3 components, 73.2% variance explained

**Time Series:**
- Trend Analysis: Linear regression on GDP growth
- Stationarity Testing: Augmented Dickey-Fuller
- ARIMA Forecasting: Auto-selected order

---

## Data Sources

| Source | Indicators | Period | Countries |
|--------|------------|--------|-----------|
| World Bank WDI | GDP growth, inflation, trade, investment, exports | 1990-2023 | 52 |
| World Governance Indicators | Government effectiveness | 1996-2023 | 52 |

---

## Key Insights

1. **The Paradox is Real but Modest**: Low-CDI countries grow 0.24% faster than high-CDI countries

2. **Governance Matters More Than Resources**: The "Fragile States" cluster has LOW CDI but HIGH volatility due to poor governance (-1.28)

3. **The Botswana Exception**: High diamond resources + good governance (+0.67) = stable 4% growth

4. **Three Development Paths**:
   - "Escaped Paradox": Good institutions overcome resource dependence
   - "Fragile States": Institutional failure causes instability regardless of resources
   - "Typical Africa": Moderate outcomes, room for improvement

---

## Technologies Used

- **Data Collection**: wbgapi, requests
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn (Ridge, Gradient Boosting, K-Means, PCA)
- **Time Series**: statsmodels (ARIMA, ADF test)
- **Visualization**: matplotlib, seaborn, plotly
- **Web Application**: Streamlit
- **Model Interpretation**: SHAP
- **Testing**: pytest, pytest-cov

---

## Contact

**Abraham Adegoke**  
HEC Lausanne  
GitHub: [@AbrahamAdegoke](https://github.com/AbrahamAdegoke)

---

## Acknowledgments

- **Prof. Simon Scheidegger** - Course instructor
- **Anna Smirnova** - Teaching assistant
- **World Bank** - Open data access
- **Claude (Anthropic)** - AI assistance (see [AI_USAGE.md](AI_USAGE.md))

---

**Last Updated:** December 2025