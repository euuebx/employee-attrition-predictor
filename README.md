# Employee Attrition Predictor

A Python machine learning project that uses HR data to predict employee attrition and identify factors associated with employees leaving.

## Overview

The project uses the IBM HR Analytics dataset containing **1,470 employees**.

It includes:

* Data cleaning and preprocessing
* Exploratory data analysis
* Random Forest classification
* Hyperparameter tuning using GridSearchCV
* Classification threshold tuning
* Model evaluation
* Feature importance analysis
* Basic turnover cost estimation

## Results

The Random Forest model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 82.99% |
| Precision |  0.472 |
| Recall    |  0.532 |
| F1 Score  |  0.500 |
| ROC AUC   |  0.779 |

The best classification threshold was **0.41**.

### Top Features

The most important features were:

1. Monthly Income
2. Age
3. Total Working Years
4. Daily Rate
5. Years at Company
6. OverTime
7. Monthly Rate
8. Hourly Rate
9. Distance From Home
10. Number of Companies Worked

## Turnover Cost Analysis

The model identified **247 employees as high risk**.

Using the project's cost assumptions:

* Potential replacement cost: **€13.20M**
* Estimated retention programme cost: **€1.98M**
* Estimated net savings: **€5.94M**

These are estimates based on simplified assumptions rather than actual company costs.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then run:

```bash
python attrition_analysis.py
```

The script generates the analysis results, charts, ROC curve, feature importance plot, and turnover cost report.
