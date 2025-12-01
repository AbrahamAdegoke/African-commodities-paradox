
# 🌍 African Commodities Paradox: A Data-Driven Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Analyzing the relationship between commodity dependence and economic volatility across 51 African countries (1992-2023)**

A machine learning project investigating why resource-rich African economies experience higher GDP growth volatility—the "African Commodities Paradox."

**Author:** Abraham Adegoke  
**Institution:** HEC Lausanne  
**Course:** Advanced Programming (Fall 2025)

---

## 📊 Project Overview

### The Problem

Many African economies rely heavily on commodity exports (oil, minerals, agricultural products), yet this dependence often leads to **unstable and volatile economic growth**. This project builds a data-driven framework to:

1. **Quantify** commodity dependence using a custom Commodity Dependence Index (CDI)
2. **Predict** GDP growth volatility using machine learning
3. **Identify** which structural factors amplify economic instability

### Key Research Questions

- Does commodity dependence increase GDP growth volatility?
- Which factors (inflation, trade openness, investment, governance) are the strongest predictors?
- Can we predict economic volatility using structural indicators?

---

## 🎯 Key Results

### Model Performance

| Model | R² Score | RMSE | MAE | Interpretation |
|-------|----------|------|-----|----------------|
| **Gradient Boosting** 🏆 | **0.274** | 0.755 | 0.575 | Explains 27.4% of volatility variance |
| Ridge Regression | 0.029 | 0.872 | 0.663 | Baseline (linear model) |

### Feature Importance (Gradient Boosting)

1. **Inflation (28%)** - Strongest predictor of volatility
2. **Trade Openness (27%)** - Exposure to external shocks
3. **Investment (26%)** - Stabilizing factor
4. **Commodity Dependence (19%)** - Significant structural vulnerability

### Main Finding

**GDP growth volatility in Africa is multi-factorial**: While commodity dependence contributes significantly (19%), it interacts with inflation, trade exposure, and investment patterns. The paradox is best understood as a **complex interaction** between resource dependence and macroeconomic vulnerabilities.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/AbrahamAdegoke/African-commodities-paradox.git
cd African-commodities-paradox

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

#### Option 1: Complete Pipeline (Recommended)

```bash
# Download data, preprocess, and create features for all African countries
python scripts/run_analysis.py --subset all_countries --start-year 1990 --end-year 2023

# Train machine learning models
python scripts/train_models.py

# View results
cat results/evaluation_report.txt
open results/figures/feature_importance.png
```

#### Option 2: Step-by-Step

```bash
# 1. Download World Bank data
python scripts/download_data.py --subset high_quality_data --start-year 2000 --end-year 2023

# 2. Train models
python scripts/train_models.py --test-size 0.2 --cv-folds 5

# 3. Explore results
jupyter notebook notebooks/00_quickstart.ipynb
```

#### Option 3: Custom Country Selection

```bash
# Analyze specific countries
python scripts/run_analysis.py --countries NGA,ZAF,KEN,GHA,EGY --start-year 2000
```

---

## 📁 Project Structure

```
african-commodities-paradox/
│
├── README.md                   # This file
├── PROPOSAL.md                 # Project proposal
├── AI_USAGE.md                 # AI tools disclosure
├── requirements.txt            # Python dependencies
│
├── configs/                    # Configuration files
│   └── countries.yaml          # List of African countries by category
│
├── data/                       # Data files (not committed if large)
│   ├── raw/                    # Raw data from World Bank
│   └── processed/              # Cleaned and feature-engineered data
│
├── src/                        # Source code
│   ├── data_io/                # Data collection modules
│   │   └── worldbank.py        # World Bank API client
│   ├── preprocessing/          # Data cleaning and validation
│   │   └── preprocessing.py
│   ├── features/               # Feature engineering (TODO)
│   ├── models/                 # Machine learning models
│   │   ├── ridge_regression.py
│   │   └── gradient_boosting.py
│   ├── evaluation/             # Model evaluation
│   │   └── metrics.py
│   └── utils/                  # Utility functions
│
├── scripts/                    # Executable scripts
│   ├── download_data.py        # Download raw data
│   ├── run_analysis.py         # Complete pipeline
│   └── train_models.py         # Train ML models
│
├── notebooks/                  # Jupyter notebooks
│   └── 00_quickstart.ipynb     # Interactive demo
│
├── tests/                      # Unit tests
│   ├── test_data_io.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── results/                    # Analysis outputs
│   ├── figures/                # Visualizations
│   ├── models/                 # Saved models (.pkl)
│   └── evaluation_report.txt   # Performance metrics
│
└── docs/                       # Documentation
    └── technical_report.pdf    # Final report (10 pages)
```

---

## 🔬 Methodology

### 1. Commodity Dependence Index (CDI)

$$\text{CDI}_{i,t} = \frac{\text{Fuel Exports}_{i,t} + \text{Metals Exports}_{i,t} + \text{Food Exports}_{i,t}}{\text{Total Merchandise Exports}_{i,t}} \times 100$$

- Smoothed with 3-year moving average to reduce noise
- Lagged by 1 year to predict future volatility

### 2. GDP Growth Volatility

$$\text{Volatility}_{i,t} = \ln(\text{std}(\text{GDP Growth}_{i,t-4:t}))$$

- 5-year rolling standard deviation of GDP growth
- Log-transformed to stabilize variance

### 3. Machine Learning Models

**Ridge Regression (Baseline)**
- L2 regularization to handle multicollinearity
- Cross-validation for alpha selection
- Interpretable linear coefficients

**Gradient Boosting Regressor (Advanced)**
- Captures non-linear relationships
- GridSearchCV for hyperparameter tuning
- Feature importance via SHAP values

### 4. Evaluation Metrics

- **R²**: Proportion of variance explained
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **5-fold Cross-Validation**: Robust performance estimation

---

## 📊 Data Sources

| Source | Indicators | Years | Countries |
|--------|------------|-------|-----------|
| [World Bank WDI](https://databank.worldbank.org/) | GDP growth, inflation, trade, investment, commodity exports | 1990-2023 | 51 African countries |
| [UNCTAD](https://unctadstat.unctad.org/) | Commodity trade data (future) | 1995-2023 | - |
| [World Governance Indicators](https://info.worldbank.org/governance/wgi/) | Governance quality (future) | 1996-2023 | - |

---

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

### Adding a New Model

1. Create model class in `src/models/your_model.py`
2. Inherit from base model interface
3. Implement `fit()`, `predict()`, and `get_feature_importance()`
4. Add tests in `tests/test_models.py`
5. Update `train_models.py` to include new model

---

## 📈 Extending the Analysis

### Add More Countries

Edit `configs/countries.yaml` to include additional countries:

```yaml
custom_analysis:
  - ETH  # Ethiopia
  - TZA  # Tanzania
  - UGA  # Uganda
```

Then run:
```bash
python scripts/run_analysis.py --subset custom_analysis
```

### Add New Indicators

Modify `src/data_io/worldbank.py` to fetch additional World Bank indicators:

```python
indicators = {
    'NY.GDP.PCAP.KD': 'gdp_per_capita',
    'SE.XPD.TOTL.GD.ZS': 'education_spending',
    # Add your indicators here
}
```

### Experiment with Models

Try different algorithms in `scripts/train_models.py`:

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Add to training pipeline
rf_model = RandomForestRegressor(n_estimators=200)
xgb_model = XGBRegressor(n_estimators=200)
```

---

## 📚 References

### Academic Literature

1. Venables, A. J. (2016). *Using natural resources for development: why has it proven so difficult?* Journal of Economic Perspectives, 30(1), 161-184.
2. Ross, M. L. (2019). *What do we know about export diversification in oil-producing countries?* The Extractive Industries and Society, 6(3), 792-806.
3. Collier, P., & Goderis, B. (2012). *Commodity prices and growth: An empirical investigation*. European Economic Review, 56(6), 1241-1260.

### Data Sources

- World Bank World Development Indicators (WDI)
- UNCTAD Commodity Statistics
- IMF International Financial Statistics

---

## 🤝 Contributing

This is an academic project for HEC Lausanne's Advanced Programming course. Contributions are not currently accepted, but feedback is welcome!

---

## 📧 Contact

**Abraham Adegoke**  
HEC Lausanne  
Email: [your.email@unil.ch]  
GitHub: [@AbrahamAdegoke](https://github.com/AbrahamAdegoke)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Prof. Simon Scheidegger** - Course instructor and advisor
- **Anna Smirnova** - Teaching assistant
- **World Bank** - Open data access
- **Claude (Anthropic)** - AI assistance for code development (see [AI_USAGE.md](AI_USAGE.md))

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{adegoke2025commodities,
  author = {Adegoke, Abraham},
  title = {African Commodities Paradox: A Data-Driven Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AbrahamAdegoke/African-commodities-paradox}
}
```

---

**Last Updated:** December 2025  
**Version:** 1.0.0