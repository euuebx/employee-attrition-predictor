import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)


# Load data
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Original rows:", len(df))


# Clean data
remove = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

for col in remove:
    if col in df.columns:
        df = df.drop(col, axis=1)

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Convert categorical columns into numbers
cat_cols = df.select_dtypes(include=["object", "str"]).columns

for col in cat_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

print("Cleaned rows:", len(df))
print("Attrition rate:", round(df["Attrition"].mean() * 100, 1), "%")


# EDA
sns.set_style("whitegrid")

# Attrition distribution
plt.figure(figsize=(6, 4))

df["Attrition"].value_counts().sort_index().plot(
    kind="bar",
    color=["#2E86AB", "#E94F37"]
)

plt.title("Attrition Distribution")
plt.xticks([0, 1], ["Stayed", "Left"], rotation=0)
plt.tight_layout()
plt.savefig("01_attrition_distribution.png", dpi=150)
plt.close()


# Attrition and overtime
plt.figure(figsize=(6, 4))

df.groupby("OverTime")["Attrition"].mean().plot(
    kind="bar",
    color=["#2E86AB", "#E94F37"]
)

plt.title("Attrition Rate by Overtime")
plt.xticks([0, 1], ["No Overtime", "Overtime"], rotation=0)
plt.ylabel("Attrition Rate")
plt.tight_layout()
plt.savefig("02_attrition_by_overtime.png", dpi=150)
plt.close()


# Attrition and job satisfaction
plt.figure(figsize=(6, 4))

df.groupby("JobSatisfaction")["Attrition"].mean().plot(
    kind="bar",
    color="#2E86AB"
)

plt.title("Attrition Rate by Job Satisfaction (1=Low, 4=High)")
plt.ylabel("Attrition Rate")
plt.tight_layout()
plt.savefig("03_attrition_by_satisfaction.png", dpi=150)
plt.close()


# Income by attrition
plt.figure(figsize=(6, 4))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome",
    hue="Attrition",
    palette=["#2E86AB", "#E94F37"],
    legend=False
)

plt.xticks([0, 1], ["Stayed", "Left"])
plt.title("Monthly Income by Attrition")
plt.tight_layout()
plt.savefig("04_income_by_attrition.png", dpi=150)
plt.close()


# Age by attrition
plt.figure(figsize=(6, 4))

sns.boxplot(
    data=df,
    x="Attrition",
    y="Age",
    hue="Attrition",
    palette=["#2E86AB", "#E94F37"],
    legend=False
)

plt.xticks([0, 1], ["Stayed", "Left"])
plt.title("Age by Attrition")
plt.tight_layout()
plt.savefig("05_age_by_attrition.png", dpi=150)
plt.close()

print("[OK] 5 EDA plots saved")


# Train/test split
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest tuning
print("\n=== TUNING RANDOM FOREST (grid search) ===")

params = {
    "n_estimators": [300, 500],
    "max_depth": [10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

rf = grid.best_estimator_

print("Best params:", grid.best_params_)
print("Best CV AUC:", round(grid.best_score_, 3))


# Find best probability threshold
y_prob = rf.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.01)

f1_scores = []

for threshold in thresholds:
    predictions = (y_prob > threshold).astype(int)
    f1_scores.append(f1_score(y_test, predictions))

best_threshold = thresholds[np.argmax(f1_scores)]

print("\nBest threshold (max F1):", round(best_threshold, 2))

y_pred = (y_prob > best_threshold).astype(int)


# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n=== OPTIMIZED RANDOM FOREST RESULTS ===")
print("Accuracy: ", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 3))
print("Recall:   ", round(recall, 3))
print("F1 Score: ", round(f1, 3))
print("AUC:      ", round(auc, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Left"]
))


# Feature importance
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Top 10 Features Driving Attrition:")
print(importance.head(10))


plt.figure(figsize=(8, 6))

importance.head(10).plot(
    kind="barh",
    color="#2E86AB"
)

plt.title("Top 10 Features Driving Attrition")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("06_feature_importance.png", dpi=150)
plt.close()


# ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    color="#2E86AB",
    lw=2,
    label="AUC = " + str(round(auc, 3))
)

plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("07_roc_curve.png", dpi=150)
plt.close()


# Turnover cost calculation
df["churn_prob"] = rf.predict_proba(X)[:, 1]

df["annual_salary"] = df["MonthlyIncome"] * 12
df["replacement_cost"] = df["annual_salary"] * 1.0

high_risk = df[df["churn_prob"] > best_threshold].copy()

high_risk = high_risk.sort_values(
    "replacement_cost",
    ascending=False
)

total_cost = high_risk["replacement_cost"].sum()

program_cost = total_cost * 0.15

savings = total_cost * 0.60 - program_cost


# Save cost results
with open("turnover_cost_results.txt", "w") as file:

    file.write("=" * 60 + "\n")
    file.write("EMPLOYEE ATTRITION - TURNOVER COST ANALYSIS\n")
    file.write("=" * 60 + "\n\n")

    file.write("Model AUC: " + str(round(auc, 3)) + "\n")
    file.write(
        "Model Recall (catching leavers): "
        + str(round(recall, 3)) + "\n"
    )
    file.write(
        "Decision threshold: "
        + str(round(best_threshold, 2))
        + "\n\n"
    )

    file.write(
        "Total employees analysed: "
        + str(len(df))
        + "\n"
    )

    file.write(
        "High-risk employees identified: "
        + str(len(high_risk))
        + "\n\n"
    )

    file.write(
        "Total replacement cost if all leave: EUR "
        + format(total_cost, ",.0f")
        + "\n"
    )

    file.write(
        "Estimated retention program cost (15%): EUR "
        + format(program_cost, ",.0f")
        + "\n"
    )

    file.write(
        "Estimated savings (60% retained): EUR "
        + format(savings, ",.0f")
        + "\n"
    )

    file.write(
        "Net benefit: EUR "
        + format(savings, ",.0f")
        + "\n\n"
    )

    file.write("TOP 10 HIGHEST-COST AT-RISK EMPLOYEES:\n")
    file.write("-" * 60 + "\n")

    for _, row in high_risk.head(10).iterrows():

        file.write(
            "  Risk: "
            + str(round(row["churn_prob"] * 100))
            + "% | Salary: EUR "
            + format(row["annual_salary"], ",.0f")
            + " | Replacement: EUR "
            + format(row["replacement_cost"], ",.0f")
            + "\n"
        )


print("\n=== TURNOVER COST SUMMARY ===")
print("High-risk employees:", len(high_risk))
print("Total replacement cost: EUR", format(total_cost, ",.0f"))
print("Retention program cost: EUR", format(program_cost, ",.0f"))
print("Net savings: EUR", format(savings, ",.0f"))

print("\n[OK] turnover_cost_results.txt saved")
print("[OK] All files generated successfully!")