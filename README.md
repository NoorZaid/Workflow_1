# Predicting College Applications with Linear Regression

**A modeling workflow on U.S. News & World Report college data: clean, explore, engineer features, and predict how many applications a college receives.**

Project for **General Assembly** by **Noor Zaid**

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Results](#results)
- [Key Drivers of Applications](#key-drivers-of-applications)
- [Model Diagnostics](#model-diagnostics)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Author](#author)

---

## Problem Statement

Colleges receive very different numbers of applications, from under 100 at small schools to over 48,000 at large public universities. Admissions offices plan staffing, budgets, marketing, and enrollment targets around this number, but it is hard to forecast.

**Problem:** admissions and planning teams lack a data-driven way to estimate how many applications an institution should expect, or to know which institutional characteristics (size, cost, selectivity, faculty quality, public vs. private) drive application volume.

**Goal:** build a regression model that predicts the number of applications (`Apps`) a college receives from its known characteristics, using data on 777 U.S. colleges.

## Dataset

- **Name:** College (U.S. News and World Report's College Data), from the 1995 issue.
- **Source:** StatLib library at Carnegie Mellon University; also distributed with *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani, 2013), https://www.statlearning.com
- **Size:** 777 colleges, 18 variables (plus the college name).

| Variable | Description |
|---|---|
| `Private` | Private or public university (Yes/No) |
| `Apps` | Number of applications received (**target**) |
| `Accept` | Number of applications accepted |
| `Enroll` | Number of new students enrolled |
| `Top10perc` / `Top25perc` | % of new students from the top 10% / 25% of their high school class |
| `F.Undergrad` / `P.Undergrad` | Number of full-time / part-time undergraduates |
| `Outstate` | Out-of-state tuition |
| `Room.Board` | Room and board costs |
| `Books` | Estimated book costs |
| `Personal` | Estimated personal spending |
| `PhD` / `Terminal` | % of faculty with a Ph.D. / terminal degree |
| `S.F.Ratio` | Student/faculty ratio |
| `perc.alumni` | % of alumni who donate |
| `Expend` | Instructional expenditure per student |
| `Grad.Rate` | Graduation rate |

## Workflow

**1. Data cleaning**
- `PhD` hid missing values as the string `'?'`, so it loaded as text. Converted to numeric, giving 29 missing values.
- Found impossible values: `PhD` of 103 (Texas A&M Galveston) and `Grad.Rate` of 118 (Cazenovia College). Both were set to missing, since the true values can't be known, and imputed later.
- Kept the very large `Apps` values (e.g. Rutgers at 48,094). They are real and consistent with `Accept`, `Enroll`, and `F.Undergrad`, so the skew is handled by scaling rather than by deleting rows.

**2. Feature engineering**
- Binarized `Private` (Yes = 1, No = 0).
- Created `Accept.Rate` = `Accept` / `Apps`.
- Binned `Outstate` into an ordinal `Outstate.Tier`: Low (<8k), Medium (8k-12k), High (12k-16k), Very High (16k+).

**3. Exploratory data analysis**
- Histograms and skewness checks to choose scaling and transformations.
- Correlation heatmap and a search for highly correlated pairs (|r| >= 0.8).
- Pairplots of the strongest predictors against `Apps`.

**4. Feature selection and leakage check**

Dropped from the model:
- `University` (identifier).
- `Accept`, `Enroll`, `Accept.Rate`: these are outcomes of the application count, not inputs known beforehand (`Accept.Rate` is built from the target itself).
- `Top25perc` (r = 0.89 with `Top10perc`) and `Terminal` (r = 0.85 with `PhD`): kept one of each pair.
- `Outstate`: replaced by the ordinal `Outstate.Tier`.

**5. Model preparation**
- 80/20 train/test split (`random_state=42`).
- Median imputation, ordinal encoding of `Outstate.Tier`, and `RobustScaler`, all **fit on the training set only** to avoid data leakage.

**6. Modeling and evaluation**
- Linear Regression, scored with MAE on train and test sets, plus 5-fold cross-validation on the training set.
- Residual checks for the LINE assumptions (Linearity, Independence, Normality, Equal variance).

## Results

| Metric | Value |
|---|---|
| Train MAE | ~1,041 |
| 5-fold CV MAE (mean) | ~1,079 (std ~137) |
| Test MAE | ~1,122 |
| Baseline (always predict the training mean), test MAE | ~2,580 |

- Train, CV, and test errors are close, so the model is **not overfitting**.
- The model cuts the error to well under half of the baseline, so it captures real signal.

## Key Drivers of Applications

Coefficients from the scaled model:

| Feature | Coefficient | Reading |
|---|---|---|
| `F.Undergrad` | +1954 | Strongest driver: larger schools receive far more applications |
| `Grad.Rate` | +726 | Moderate positive effect |
| `Room.Board` | +657 | Moderate positive effect |
| `Top10perc` | +425 | Moderate positive effect |
| `Expend` | +370 | Moderate positive effect |
| `Private` | -534 | Private schools get about 534 fewer applications than similar public schools |
| `perc.alumni` | -474 | Likely a proxy for smaller, older schools |
| `P.Undergrad`, `PhD`, `Personal`, `Outstate.Tier` | -178 to -80 | Small negative effects |
| `S.F.Ratio`, `Books` | near 0 | Little effect |

Coefficients show association, not causation.

## Model Diagnostics

- **Normality:** residuals are heavy-tailed.
- **Equal variance:** the residual spread widens as predictions grow (heteroscedasticity).
- The model predicts **negative applications** for a few schools, which is impossible.

These are signs that a plain linear model on the raw target is not ideal for this data.

## Limitations and Next Steps

- **Dated data:** the data is from 1995, so patterns may not reflect today's admissions landscape.
- **Small sample:** 777 colleges, so results can vary by split. Cross-validation helps, but the CV step fit the imputer and scaler on all of the training data first, which leaks a very small amount of information into each fold.
- **Model 2 (planned):** apply log or power transformations (e.g. Yeo-Johnson) to the skewed target and features, then compare with Model 1. The notebook ends with this as a recommendation.
- **Pipelines:** wrap imputing, encoding, and scaling in a scikit-learn pipeline to remove the CV leakage.

## Repository Structure

```
.
├── model_workflow_NoorZaid.ipynb   # Full analysis and modeling notebook
├── College.csv                     # Dataset
├── worlflow.pdf                    # Data dictionary (ISLR College documentation)
└── README.md
```

## Getting Started

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

**2. Install dependencies**
```bash
pip install pandas numpy seaborn matplotlib scikit-learn scipy jupyter
```

**3. Run the notebook**
```bash
jupyter notebook model_workflow_NoorZaid.ipynb
```

> **Note:** the notebook loads the data from `/content/College.csv` (Google Colab). Change that path to `College.csv` if you run it locally.

**Tools:** Python, pandas, NumPy, seaborn, Matplotlib, scikit-learn, SciPy.

## Author

**Noor Zaid** - [GitHub](https://github.com/NoorZaid)
