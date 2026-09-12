# Employee Attrition Analysis

A machine learning project that predicts which employees are likely to leave a company, using HR data. Also includes an example cost scenario.

## Dataset

IBM HR Analytics Employee Attrition dataset:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

File used: `WA_Fn-UseC_-HR-Employee-Attrition.csv`. 1,470 employees, 35 features, about 16% left.

## Results

Trained a Random Forest with grid search and 5-fold cross validation (scoring on ROC-AUC), then picked a threshold that maximizes F1.

Best CV AUC: 0.804. Threshold chosen: 0.41.

| Metric | Value |
|---|---|
| Accuracy | 82.99% |
| Precision | 0.472 |
| Recall | 0.532 |
| F1 | 0.500 |
| AUC | 0.779 |

Classification report:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Stayed | 0.91 | 0.89 | 0.90 |
| Left | 0.47 | 0.53 | 0.50 |

Confusion matrix:

```
[[219  28]
 [ 22  25]]
```

Top features: MonthlyIncome, Age, TotalWorkingYears, DailyRate, YearsAtCompany.

Note: recall for "Left" is 0.53, so the model catches about half of leavers. Precision is 0.47, so when it says someone will leave it's right about half the time.

## Turnover cost analysis (example only)

This part is just an example. The numbers are based on assumptions, not real data.

Assumptions: replacement cost = 1 year salary, program cost = 15% of total, program keeps 60% of at-risk employees.

| Item | Value |
|---|---|
| High-risk employees | 247 |
| Total replacement cost | EUR 13,202,340 |
| Program cost (15%) | EUR 1,980,351 |
| Net savings (60% retained) | EUR 5,941,053 |

These are not real results. The model only gives probabilities, and it's wrong more than half the time when it says someone will leave. The savings number comes from plugging assumptions into a formula.

## Files

| File | What it is |
|---|---|
| `attrition_analysis.py` | main script |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | dataset |
| `01_attrition_distribution.png` | stayed vs left |
| `02_attrition_by_overtime.png` | attrition by overtime |
| `03_attrition_by_satisfaction.png` | attrition by satisfaction |
| `04_income_by_attrition.png` | income vs attrition |
| `05_age_by_attrition.png` | age vs attrition |
| `06_feature_importance.png` | top 10 features |
| `07_roc_curve.png` | ROC curve |
| `turnover_cost_results.txt` | cost scenario summary |

## Requirements

Python 3.12+:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to run

```
python attrition_analysis.py
```

The script cleans the data, saves 5 EDA plots, tunes a Random Forest, picks the best threshold by F1, evaluates the model, saves feature importance and ROC plots, then runs the cost scenario.
