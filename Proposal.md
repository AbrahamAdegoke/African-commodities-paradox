# The African Commodities Paradox: A Data-Driven Tool Analysis

**Student:** ADEGOKE Adekounle Abraham  
**Category:** Data Analysis & Visualization  
**Course:** Advanced Programming (Fall 2025) - HEC Lausanne

---

## Problem Statement and Motivation

Many African economies rely heavily on the export of primary commodities such as oil, minerals, and agricultural products. While this dependence provides substantial foreign exchange revenues, it also exposes these economies to global price shocks, terms-of-trade fluctuations, and external crises. The result is often unstable and unpredictable economic performance—a phenomenon widely known as the **African Commodities Paradox**, where resource abundance coexists with weak and volatile growth.

This project develops a data-driven analytical framework to examine and quantify the African commodities paradox. Using machine learning techniques, it aims to predict the volatility of GDP growth as a function of commodity dependence and key macroeconomic indicators (inflation, exchange-rate volatility, trade openness, governance, and investment). By linking structural economic features to growth instability, the ultimate objective is to identify which structural factors amplify instability and which help countries maintain more resilient growth paths.

---

## Computation of the Commodity Dependence Index (CDI)

The Commodity Dependence Index (CDI) measures the share of a country's export revenues that comes from primary commodities such as oil, minerals, metals, and agricultural products.

The index will be computed as:

$$CDI_{i,t} = \frac{\text{Commodity Exports}_{i,t}}{\text{Total Merchandise Exports}_{i,t}} \times 100$$

where $i$ denotes the country and $t$ the year.

A **3-year moving average** will be applied to the CDI to smooth out extreme year-to-year price fluctuations caused by short-term commodity shocks or data inconsistencies.

---

## Feature Engineering and Target Construction

### Target Variable (Y)

The target variable is **GDP growth volatility**, serving as a quantitative measure of macroeconomic instability.

It is computed as the five-year rolling standard deviation of annual real GDP growth, then log-transformed to stabilize variance:

$$Vol_{GDP_{i,t}} = \ln(\text{std}(GDP\_Growth_{i,t-4:t}))$$

A 5-year window captures medium-term fluctuations such as commodity shocks or crises while retaining enough observations for predictive modeling.

### Predictor Variables (X)

- **CDI** (smoothed 3-year average)
- **Inflation rate**
- **Trade openness** ((exports + imports)/GDP)
- **Governance quality** (World Governance Indicators)
- **Gross capital formation** (% of GDP, proxy for investment)

All predictors are **lagged by one year (t–1)** to avoid simultaneity between explanatory variables and the outcome.

---

## Modeling Approach

Two supervised regression models will be implemented to capture both linear and non-linear relationships between structural variables and economic instability:

### 1. Baseline Model — Ridge Linear Regression

- Provides a transparent benchmark model with interpretable coefficients
- Incorporates regularization (L2 penalty) to handle potential multicollinearity among predictors
- Allows clear economic interpretation: e.g., a positive CDI coefficient indicates that greater commodity dependence increases GDP volatility

### 2. Machine Learning Model — Gradient Boosting Regressor (GBR)

- Captures complex non-linear interactions, such as how the impact of CDI might depend on inflation or governance quality
- Typically yields higher predictive accuracy and can model threshold effects (e.g., volatility spikes once CDI > 70%)
- Produces feature importance metrics that help identify the most influential predictors of instability

### Model Evaluation

Model evaluation will rely on:
- **R²** (coefficient of determination)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

Feature importance (Permutation or SHAP values) will be used to interpret variable relevance across models.

---

## Data Sources

| Source | Description |
|--------|-------------|
| World Bank (WDI) | GDP growth, inflation, trade, investment |
| UNCTADstat | Commodity trade data |
| World Governance Indicators | Governance quality metrics |
| IMF | Additional macroeconomic indicators |

---

## Expected Challenges

- **Missing data:** Will use interpolation or country-year filtering
- **Heterogeneity:** Will apply normalization and lagging
- **Data quality:** Will validate against multiple sources

---

## Success Criteria

1. ✅ End-to-end pipeline runs reproducibly from raw data to results
2. ✅ Models achieve meaningful out-of-sample accuracy (e.g., R² ≥ 0.6 on test)
3. ✅ CDI emerges as an important predictor of volatility (positive and robust)

---

## Stretch Goal (if time permits)

### Forecast Next-Year GDP Growth

As an extension, this project will explore the potential to forecast next-year GDP growth based on current-year macroeconomic and structural indicators.

The model will aim to predict GDP growth in year $t+1$ using information available at year $t$, following the formulation:

$$GDP\_Growth_{i,t+1} = f(CDI_{i,t}, Inflation_{i,t}, Governance_{i,t}, Openness_{i,t}, Investment_{i,t})$$

where $i$ denotes the country.

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1-2 | Data collection, cleaning, CDI computation |
| 3-4 | Feature engineering, exploratory analysis |
| 5 | Model implementation (Ridge + GBR) |
| 6 | Model evaluation, interpretation |
| 7 | Documentation, report writing |

---

*Submitted: November 2025*