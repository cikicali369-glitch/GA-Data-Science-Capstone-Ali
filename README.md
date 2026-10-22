# Early Prediction of Long-Running Administrative Cases

A Data Science proof of concept for SPAR using **BPI Challenge 2015 Municipality 1**.

The main goal is to predict whether an active administrative case is at risk of becoming long-running.

BPI data is external public process-mining data. It is not SPAR production data.

**GA Capstone Part 2 (Dataset + Data Collection) submission:** [`notebooks/Capstone_Part2_Dataset_EDA.ipynb`](notebooks/Capstone_Part2_Dataset_EDA.ipynb), with the data dictionary also saved as [`data/data_dictionary.csv`](data/data_dictionary.csv).

## Project structure

- `data/raw/`: local original dataset files (excluded from Git).
- `data/processed/`: generated datasets.
- `notebooks/`: data loading, EDA, feature engineering, modeling, and evaluation stages.
- `src/`: reusable Python modules.
- `outputs/figures/` and `outputs/tables/`: generated results.

## Environment

On Windows PowerShell, activate the existing environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

To recreate the environment with Python installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset setup

1. Download BPI Challenge 2015 Municipality 1 from the official BPI Challenge 2015 / 4TU Research Data source: DOI [10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17](https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17) (B. van Dongen, Eindhoven University of Technology, 2015; 4TU General Terms of Use).
2. Place `BPIC15_1.xes` in `data/raw/`.
3. The raw dataset is intentionally excluded from Git (it is a large third-party file, available from the DOI above) and is not SPAR production data.
4. Run the notebooks in order with the project `.venv`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_loading` | Read-only XES intake and structural validation |
| `02_eda` | Exploratory analysis of durations, activities, variants, resources and fields |
| `02b_target_definition` | Outcome cohort, duration definition, long-running rule, prediction point |
| `03_feature_engineering` | Leakage-safe Day-14 snapshot features → `data/processed/case_features_14d.csv` |
| `04_modeling` | Chronological train/validation/test evaluation of logistic regression and random forest |
| `05_evaluation` | Reproduces the locked result; interpretation, lessons and SPAR implications |
| `06_regression` | Secondary extension: predicts total duration in days from the same Day-14 inputs and split |

Run the tests with:

```powershell
.\.venv\Scripts\python.exe tests\test_features.py
.\.venv\Scripts\python.exe tests\test_models.py
.\.venv\Scripts\python.exe tests\test_regression.py
```

## Abstract

**Early Prediction of Long-Running Administrative Cases: A Proof of Concept for SPAR**

Administrative cases that take much longer than usual are costly, and an early warning could help staff decide where to look first. This capstone tests whether information available 14 days after a case starts can predict whether it will become long-running, defined as exceeding the 75th percentile of training-period durations (104.6 days). Using the public BPI Challenge 2015 event log of one Dutch municipality (1,199 permit cases, 52,217 events), we reconstructed cases, built leakage-safe point-in-time features from the first 14 days, and trained a logistic-regression baseline and a random forest. Evaluation was strictly chronological: the threshold, activity vocabulary and preprocessing were learned on 2010–2012 cases, model choices were made on 2013, and 2014 was scored once. A weak signal appeared in 2013, but on the 2014 test period neither model ranked cases better than chance (random forest ROC-AUC 0.49; PR-AUC 0.14 at a prevalence of 0.14). The pipeline worked; the early information did not generalise. For SPAR, the lesson is that predictive analytics depends on capturing rich, well-defined, time-stamped process context from the start, and on evaluating it honestly before relying on any risk score.

**Regression extension (secondary).** `06_regression` asks how many days a case will last, using the same 917 cases, Day-14 features and chronological split. No model beat the naive forecast "training median, 64 days" on the 2014 test period (test MAE: baseline 31 days; log-target linear regression 62; random forest 70; linear regression 119). The diagnostics show the early process changed after 2012. The classification result above is unchanged.

"Long-running" is a relative duration label, not a legal delay or a performance judgement. Results are associations, not causes, and do not transfer to SPAR without local data.
