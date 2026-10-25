# Retail Predictive Analytics: Sales Forecasting and Classification

A predictive analytics project built in Python with scikit-learn, XGBoost and statsmodels. It forecasts retail sales, predicts loss-making orders and estimates repeat purchases, using leakage-safe pipelines, time-aware validation and a bias review. It builds on the findings of the earlier [EDA project](https://github.com/<your-username>/eda-retail-sales).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-modelling-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-green)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Project Overview](#project-overview)
- [Predictive Tasks](#predictive-tasks)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Modelling Workflow](#modelling-workflow)
- [Data Splitting](#data-splitting)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations and Bias Mitigation](#limitations-and-bias-mitigation)
- [Timeline](#timeline)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

Predictive analytics uses historical data to estimate what will happen next. This project turns the insights from the EDA stage (seasonal sales peaks, losses linked to heavy discounts, and a small group of customers driving most revenue) into models that support planning and decisions.

**Business goals:** plan inventory and staffing, protect profit margin, and improve customer retention.

## Predictive Tasks

| Task | Goal | Type | Target |
|---|---|---|---|
| T1 | Forecast monthly sales for the next 3 to 6 months | Time-series regression | Total sales |
| T2 | Identify loss-making orders | Binary classification | Profit < 0 |
| T3 | Predict repeat purchase within 90 days | Binary classification | Repeat purchase (yes/no) |
| T4 (stretch) | Predict order profit amount | Regression | Profit |

T1 and T2 are the priority; T3 and T4 are completed if time permits.

## Models Used

| Family | Algorithms | Role |
|---|---|---|
| Baselines | Seasonal naive, moving average, majority class | Minimum performance to beat |
| Regression | Linear, Ridge, Lasso, Logistic | Interpretable benchmarks |
| Decision trees | DecisionTree classifier and regressor | Explainable business rules |
| Ensembles | Random Forest, HistGradientBoosting, XGBoost, LightGBM | Expected top performers |
| Time series | Holt-Winters, SARIMA | Trend and seasonality with limited history |

**Selection criteria:** data size, target type, relationship shape, interpretability needs, class imbalance, training cost and robustness to outliers.

## Project Structure

```
predictive-retail-analytics/
|-- data/
|   `-- processed/            # cleaned data from the EDA stage
|-- notebooks/
|   |-- 01_features_targets.ipynb
|   |-- 02_baselines.ipynb
|   |-- 03_model_comparison.ipynb
|   |-- 04_tuning_diagnostics.ipynb
|   `-- 05_final_evaluation.ipynb
|-- src/
|   |-- features.py           # feature engineering
|   |-- pipelines.py          # preprocessing and model pipelines
|   |-- evaluate.py           # metrics, plots, error analysis
|   `-- forecast.py           # time-series helpers
|-- models/                   # saved pipelines (joblib)
|-- reports/
|   `-- figures/
|-- requirements.txt
|-- README.md
`-- LICENSE
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/predictive-retail-analytics.git
cd predictive-retail-analytics

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
numpy
scipy
scikit-learn
xgboost
lightgbm
statsmodels
matplotlib
seaborn
shap
joblib
jupyter
imbalanced-learn   # optional
```

## How to Run

1. Place the cleaned dataset in `data/processed/`.
2. Start Jupyter with `jupyter notebook`.
3. Run notebooks `01` to `05` in order.
4. Saved models appear in `models/` and figures in `reports/figures/`.

## Modelling Workflow

1. Define tasks, targets, metrics and success thresholds.
2. Build targets and engineer features (calendar, lag and rolling, order, product, RFM, geography).
3. Split the data (chronological for forecasting, stratified for classification).
4. Train baselines and record their scores.
5. Build preprocessing with `Pipeline` and `ColumnTransformer`.
6. Train interpretable models, then ensembles.
7. Cross-validate all candidates on the same folds.
8. Tune the top models with randomized search.
9. Diagnose errors by segment, calibrate and choose the decision threshold.
10. Evaluate once on the held-out test set.
11. Explain with feature importance and SHAP, then save the pipeline.

**Leakage safeguards:** features must be available at prediction time, scalers and encoders are fitted inside cross-validation folds, target-derived features use training data only, and time-based data is never shuffled.

## Data Splitting

| Task | Method |
|---|---|
| T1 forecasting | Chronological split (about 75% train, 10% validation, 15% test) |
| T2, T3 classification | Stratified 70/15/15 split; group by customer where customers repeat |
| T4 regression | Random split, or chronological if drift is visible |

The test set is used once, at the end, with `random_state=42` throughout.

## Evaluation Metrics

| Task | Primary metric | Supporting metrics |
|---|---|---|
| T1 forecasting | MAPE, RMSE | MAE, bias, prediction-interval coverage |
| T2 loss classification | PR-AUC, recall of loss class | ROC-AUC, precision, F1, calibration |
| T3 repeat purchase | ROC-AUC | PR-AUC, F1, lift at top 20% |
| T4 profit regression | MAE, RMSE | R-squared, residual plots |

**Validation:** stratified (or group) k-fold for classification, and walk-forward `TimeSeriesSplit` for forecasting.

**Success targets:** T1 beats the seasonal-naive baseline by at least 15%; T2 and T3 reach ROC-AUC of at least 0.75 and beat the majority-class baseline.

## Limitations and Bias Mitigation

| Risk | Mitigation |
|---|---|
| Short sales history | Simple seasonal models, report intervals, avoid long horizons |
| Data leakage | Pipelines, time-aware splits, review of suspiciously strong features |
| Class imbalance | Class weights, PR-AUC, threshold tuning |
| Concept drift | Monitor error monthly, retrain on recent data |
| Historical and sampling bias | Audit performance by region, category and segment |
| Correlation is not causation | Describe results as associations; test policy changes with experiments |
| Overfitting from tuning | Cross-validation and a single final test evaluation |

## Timeline

| Phase | Hours |
|---|---|
| Scoping and setup | 2 |
| Data and features | 5 |
| Splitting and baselines | 2 |
| Interpretable models | 4 |
| Ensemble models | 5 |
| Cross-validation and tuning | 5 |
| Diagnostics and bias review | 3 |
| Final evaluation | 2 |
| Documentation | 2 |
| Review and buffer | 2 |
| **Total** | **32** |

If the schedule slips by more than 3 hours, T3 and T4 and optional models are dropped.

## Future Work

- Add external features such as holidays, promotions and economic indicators
- Try Prophet or deep learning for forecasting
- Run controlled discount experiments to test causal effects
- Monitor forecast error and retrain on a schedule
- Deploy as an API or dashboard (FastAPI or Streamlit)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Your Name**
GitHub: [@your-username](https://github.com/your-username)
