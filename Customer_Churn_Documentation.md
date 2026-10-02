# Customer Churn Analysis and Prediction System — Documentation

**Dataset:** `customer_churn_dataset.csv`
**Model:** Random Forest Classifier inside a scikit-learn Pipeline
**Task:** Binary classification — predict whether a customer will churn (`Yes`) or stay (`No`)

---

## 1. Overview

This notebook walks through a full machine learning workflow for customer churn: loading the data, checking quality, exploring it visually, preprocessing, training a model, and evaluating the results.

| Stage | Purpose |
|---|---|
| Data loading and inspection | Understand shape, types, missing values, duplicates |
| Exploratory data analysis (EDA) | Summary statistics, churn balance, distributions |
| Preprocessing | Drop ID, clean types, scale numbers, encode categories, split data |
| Model training | Random Forest wrapped in a single Pipeline |
| Evaluation | Classification report, ROC-AUC, confusion matrix |
| Extra analysis | Correlation heatmap of numeric features |

---

## 2. Requirements

**Python libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (plus the standard `os` and `json` modules, imported but not used later).

**Input file:** `customer_churn_dataset.csv` must be in the working directory (in Colab, upload it to the session first).

---

## 3. Dataset Description

**Shape:** 20,000 rows × 11 columns

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Unique customer identifier (dropped before modeling) |
| `tenure` | int | Months the customer has been with the company (1–72) |
| `monthly_charges` | float | Monthly bill (20.00–120.00) |
| `total_charges` | float | Total amount billed over the customer's lifetime |
| `contract` | categorical | Contract type (e.g., Month-to-month, One year) |
| `payment_method` | categorical | Payment method (e.g., Credit, Debit, Cash, UPI) |
| `internet_service` | categorical | Internet service type (e.g., DSL, Fiber) |
| `tech_support` | categorical | Whether the customer has tech support (Yes/No) |
| `online_security` | categorical | Whether the customer has online security (Yes/No) |
| `support_calls` | int | Number of support calls made (0–8) |
| `churn` | categorical | **Target** — Yes / No |

### Data quality findings

- **Missing values:** only `internet_service` has missing values — **2,013 of 20,000 rows (~10%)**. All other columns are complete.
- **Duplicates:** 0 duplicate rows.
- **Target balance:** No = 65.79%, Yes = 34.22% (moderately imbalanced).

### Numeric summary

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| tenure | 36.47 | 20.77 | 1 | 36 | 72 |
| monthly_charges | 70.01 | 28.89 | 20.00 | 70.09 | 120.00 |
| total_charges | 2,543.98 | 1,882.95 | 20.23 | 2,096.50 | 8,629.92 |
| support_calls | 1.51 | 1.24 | 0 | 1 | 8 |

---

## 4. Step-by-Step Walkthrough

### Step 1 — Import libraries
Loads pandas/numpy for data handling, matplotlib/seaborn for plots, and scikit-learn for splitting, preprocessing, modeling, and metrics.

### Step 2 — Load the data
`pd.read_csv('customer_churn_dataset.csv')` reads the file into a DataFrame `df`; `df.head()` shows the first five rows.

### Steps 3–7 — Inspection
Checks shape and column names, data types (`df.info()`), missing values (`df.isnull().sum()`), and duplicates (`df.duplicated().sum()`).

### Step 8 — Exploratory analysis
Prints `df.describe()` for numeric columns and the churn class proportions.

### Step 9 — Visualizations
Two side-by-side plots:
1. **Box plot** of `tenure` by churn status — churned customers tend to have shorter tenure (lower median).
2. **KDE density plot** of `monthly_charges` by churn status — customers who stay are concentrated at lower monthly charges, while churners are concentrated at higher charges.

### Step 10 — Data preprocessing
1. Reload the CSV to guarantee a clean state.
2. Drop `customer_id` (an identifier carries no predictive signal).
3. Convert `total_charges` to numeric (`errors='coerce'`) in case of stray strings.
4. **`df.dropna()`** — removes every row with a missing value. Because only `internet_service` has missing values, this removes the 2,013 incomplete rows, leaving **17,987 rows**.
5. Split features `X` and target `y` (`Yes` → 1, `No` → 0).
6. Auto-detect numeric columns (`int64`, `float64`) and categorical columns (`object`).
7. Build a `ColumnTransformer`:
   - Numeric → `StandardScaler`
   - Categorical → `OneHotEncoder(handle_unknown='ignore')`
8. Train/test split: **80% / 20%**, `random_state=42`, `stratify=y` (keeps the churn ratio equal in both sets).

| Set | Shape |
|---|---|
| Training | (14,389, 9) |
| Testing | (3,598, 9) |

### Step 11 — Model training
A `Pipeline` chains the preprocessor and a `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`. Wrapping both in one pipeline means preprocessing is learned from training data only and applied consistently at prediction time.

### Step 12 — Evaluation
Predictions and churn probabilities are generated on the test set and scored with `classification_report` and `roc_auc_score`.

### Step 13 — Confusion matrix
Heatmap of true vs. predicted labels.

### Final cell — Correlation heatmap
Pearson correlation matrix of the numeric columns, shown as an annotated heatmap.

---

## 5. Results

### Classification report (test set, 3,598 customers)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 — Retained | 0.84 | 0.93 | 0.88 | 2,371 |
| 1 — Churned | 0.83 | 0.66 | 0.74 | 1,227 |
| **Accuracy** | | | **0.84** | 3,598 |
| Macro avg | 0.84 | 0.80 | 0.81 | 3,598 |
| Weighted avg | 0.84 | 0.84 | 0.83 | 3,598 |

**ROC-AUC:** 0.7947

### Confusion matrix

| | Predicted Retained | Predicted Churned |
|---|---|---|
| **Actual Retained** | 2,204 (TN) | 167 (FP) |
| **Actual Churned** | 416 (FN) | 811 (TP) |

### Interpretation

- The model is strong at identifying customers who stay (recall 0.93).
- It is weaker at catching churners: **recall for churn is 0.66**, meaning about 34% of customers who actually churn (416 of 1,227) are missed.
- Precision for churn is good (0.83) — when the model flags a customer as likely to churn, it is right most of the time.
- For a retention campaign, missing churners is usually the costlier error, so improving churn recall should be a priority.

### Correlation findings

| Pair | Correlation |
|---|---|
| tenure ↔ total_charges | 0.77 |
| monthly_charges ↔ total_charges | 0.55 |
| All other pairs | ≈ 0 |

`total_charges` is largely a product of tenure and monthly charges, so it is partly redundant with those two features. (This matters little for Random Forest but is worth knowing for interpretation and for linear models.)

---

## 6. Issues and Observations

1. **Rows with missing `internet_service` are dropped, not imputed.** This discards ~10% of the data. If the blanks mean "no internet service" they are informative, and filling them with a category such as `"None"` (or using `SimpleImputer`) would keep those customers and may improve the model.
2. **Imputation/cleaning is done outside the pipeline.** `dropna()` runs before the split. Moving imputation into the Pipeline is cleaner and avoids any leakage.
3. **`customer_id` appears in the correlation heatmap.** It was dropped in the preprocessing cell, so its presence suggests the cell order or kernel state differed when the heatmap ran. It has no meaning and should be excluded.
4. **Duplicate imports** across cells and unused imports (`os`, `json`, `roc_curve`) can be tidied.
5. **Seaborn `FutureWarning`:** `palette` is passed without `hue` in the box plot. Fix with `sns.boxplot(..., hue='churn', legend=False)`.
6. **Placeholder cells:** several cells contain only "Cell cleared" comments (upload, duplicate loading, Streamlit app, background server, IP display). They can be deleted.

---

## 7. Suggested Improvements

- Handle class imbalance: `class_weight='balanced'` or adjust the decision threshold to raise churn recall.
- Hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`) for `n_estimators`, `max_depth`, `min_samples_leaf`.
- Add cross-validation for a more reliable performance estimate.
- Report feature importances to explain which factors drive churn.
- Compare against other models (Logistic Regression, Gradient Boosting, XGBoost/LightGBM).
- Plot the ROC curve and precision–recall curve (the notebook already imports `roc_curve`).
- Save the trained pipeline with `joblib.dump` for reuse or deployment.

---

## 8. How to Run

1. Open the notebook in Google Colab (or Jupyter).
2. Upload `customer_churn_dataset.csv` to the working directory.
3. Run cells in order from top to bottom (**Runtime → Run all**).
4. Review the printed metrics and plots.

### Example: predicting for a new customer

```python
import pandas as pd

new_customer = pd.DataFrame([{
    "tenure": 12,
    "monthly_charges": 85.0,
    "total_charges": 1020.0,
    "contract": "Month-to-month",
    "payment_method": "Credit",
    "internet_service": "Fiber",
    "tech_support": "No",
    "online_security": "No",
    "support_calls": 3,
}])

churn_probability = model.predict_proba(new_customer)[:, 1][0]
print(f"Churn probability: {churn_probability:.2%}")
```

*(Column names and category values must match those in the training data.)*
