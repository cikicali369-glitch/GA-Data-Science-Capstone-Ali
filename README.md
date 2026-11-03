# GA Data Science Capstone
## Early Warning for Long Administrative Cases

**Project question:** Can early information from an administrative process help us identify which cases are likely to
take longer than usual?

This project uses a public event log from **one Dutch municipality**. It is **not SPAR data**, so the results show a method
that could be tested on SPAR's own historical data, not a model that has been validated for SPAR.

For a short walk through the whole project, start with
[`notebooks/13_project_summary.ipynb`](notebooks/13_project_summary.ipynb).

## Dataset

**BPI Challenge 2015 — Municipality 1**: building-permit applications from one Dutch municipality, recorded as an event
log (events from October 2010 to July 2015).

- Creator: Boudewijn van Dongen (Eindhoven University of Technology)
- Published: 2015, on 4TU.ResearchData
- DOI: [10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17](https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17)
- Terms of use: 4TU General Terms of Use ([doi.org/10.4121/resource:terms_of_use](https://doi.org/10.4121/resource:terms_of_use)), as listed in the dataset's DOI record
- Format: XES event log (`BPIC15_1.xes`)
- Size: 52,217 events, 1,199 cases, 29 columns

This is a public research dataset, not confidential government data. The analysed event log uses resource identifiers
rather than employee names.

**The dataset is stored in this repository using Git LFS.** Git LFS is required when cloning to retrieve
`data/raw/BPIC15_1.xes`.

## What Is Being Predicted

- **Completed cases:** cases with a recorded `case:endDate` (956 cases).
- **Recorded activity span:** the time from the **first** to the **last** recorded event of a case, in days
  (`duration_days`). This is what the event log shows. It is **not** a verified administrative completion time: it can
  include later steps such as publication, objection or archiving.
- **Long case:** a recorded activity span of **more than 63 days**. The 63 days come from the exploratory analysis in
  notebook 01 (the median span of the completed cases). It is **not** an external legal or administrative deadline.

## Two Kinds of Experiments

1. **Whole-case regression exercises (notebooks 02–05).** These models predict the span in days from features that
   summarise the **whole finished case** (for example the total number of events). They were useful for learning and
   comparing regression methods, but they are **not early-warning models**, because that information only exists after
   the case has ended. They use their own random 80/20 split of the 956 completed cases (764 training / 192 test).
2. **Early-warning experiments (notebooks 06–12).** These use only information available at a fixed early moment: the
   first 5 events, day 14 or day 30. Only cases whose recorded span extends beyond that moment are included.

**Splits used in notebooks 07–12:**

- **Original random split:** created in notebook 07 (733 training / 184 test cases, stratified by the long-case label).
  Notebooks 08–10 and 12 reuse exactly these groups; for day 30 they are filtered (677 training / 161 test), never redrawn.
- **Temporal split (notebooks 11 and 12):** a separate check that trains on **older** cases and tests on **newer** cases
  (617 training / 168 test, boundary 2013-10-25).

## Notebooks

Notebooks are listed in the recommended reading order.

| Notebook | Purpose | Main finding |
|---|---|---|
| [01_EDA](notebooks/01_EDA.ipynb) | Load the XES log, check data quality, explore the data | 52,217 events and 1,199 cases; key columns complete; no duplicate rows. Completed cases have a median span of 63 days (mean about 98), with a long tail |
| [02_linear_regression_review](notebooks/02_linear_regression_review.ipynb) | Whole-case regression: Linear Regression | MAE 59.01 days vs 71.98 for predicting the mean; R² 0.26; a few impossible negative predictions |
| [03_decision_tree_review](notebooks/03_decision_tree_review.ipynb) | Whole-case regression: Decision Tree | MAE 58.73, R² 0.27: almost the same as Linear Regression, with some overfitting |
| [04_random_forest_review](notebooks/04_random_forest_review.ipynb) | Whole-case regression: Random Forest | MAE 52.00, R² 0.44, less overfitting than one tree |
| [05_gradient_boosting_review](notebooks/05_gradient_boosting_review.ipynb) | Whole-case regression: Gradient Boosting | MAE 51.19, R² 0.49: best of these exercises, but only a small gain over Random Forest |
| [06_early_warning_features_review](notebooks/06_early_warning_features_review.ipynb) | Classification from the first 5 events | Weak: accuracy 0.55 vs 0.51 for the majority baseline; recall 0.28 |
| [07_14_day_early_warning_review](notebooks/07_14_day_early_warning_review.ipynb) | Classification from 7 day-14 count features | No signal: accuracy 0.49 vs 0.52 baseline; the baseline's F1 (0.68) is also higher |
| [08_14_day_activity_features_review](notebooks/08_14_day_activity_features_review.ipynb) | Day-14 counts plus "which activities were recorded" | Small, exploratory gain: ROC-AUC 0.53 → 0.59 |
| [09_day14_regression_review](notebooks/09_day14_regression_review.ipynb) | Predict the span in days from day-14 features | No model beats the median baseline: MAE 70.75 (baseline) vs 76.90–81.27 |
| [10_day30_early_warning_review](notebooks/10_day30_early_warning_review.ipynb) | Day-30 classification with 5-fold cross-validation | Modest signal: Logistic Regression ROC-AUC 0.702 (cross-validation) and 0.659 (test) |
| [11_day30_temporal_validation_review](notebooks/11_day30_temporal_validation_review.ipynb) | Train on older cases, test on newer cases | Similar ranking (ROC-AUC 0.678), but recall drops to 0.55 |
| [12_day30_model_comparison_review](notebooks/12_day30_model_comparison_review.ipynb) | Day-30 comparison with Random Forest and Gradient Boosting | Random Forest is best on the random test only; no consistent gain over Logistic Regression |
| [13_project_summary](notebooks/13_project_summary.ipynb) | Short summary of the whole project (Markdown only) | — |

Other files that may exist in a local copy (for example review copies or earlier research notebooks) are not part of
this reading sequence.

## Main Result: Day 30

ROC-AUC measures how well a model **ranks** long cases above not-long cases (0.50 = no better than chance).

| ROC-AUC | Cross-validation (training cases) | Random test | Temporal test |
|---|---|---|---|
| Majority baseline | 0.500 | 0.500 | 0.500 |
| Logistic Regression | 0.702 | 0.659 | 0.678 |
| Random Forest | 0.701 | 0.745 | 0.674 |
| Gradient Boosting | 0.684 | 0.723 | 0.624 |

**Why Logistic Regression is the main simple candidate:** Random Forest had the strongest result on the random test
(0.745), but it was practically equal to Logistic Regression in cross-validation (0.701 vs 0.702) and on the temporal test
(0.674 vs 0.678). One better test result is not enough to prefer a more complex model. Random Forest's warning behaviour
also changed more between the tests: it flagged 118 of 161 random-test cases (recall 0.863) but only 51 of 168
temporal-test cases (recall 0.43).

**Baseline and the precision/recall trade-off:** the majority baseline predicts "long" for **every** case. This gives
perfect recall (it finds every long case) but no ability to tell cases apart. On the random test, Logistic Regression
flagged 98 of 161 cases; 69.4% of its warnings were correct, compared with 59.0% if cases were flagged at random, and it
found 71.6% of the long cases. On the temporal test it flagged 71 of 168 cases: 77.5% of its warnings were correct, but it
found only 55% of the long cases. A similar ROC-AUC on newer cases does not prove that the model generalises or that it
would be useful in practice.

## Limitations

- **Endpoint uncertainty.** For many completed cases the last recorded event comes after the recorded `case:endDate`,
  and for some it comes before it. The meaning of the end date is not documented in the sources available to this
  project, so the recorded activity span and the administrative end date can tell different stories.
- **Completed cases only.** Cases without a recorded end date are left out, so the results describe historically
  completed cases.
- **Timestamp assumption.** Event timestamps are assumed to show when information was available. The leakage checks
  confirm that the features only use events up to the cutoff, but they cannot prove when information was actually entered
  into the source system.
- **Reused test cases.** The same test cases were used across several notebooks, so the later results are exploratory,
  not an untouched final evaluation.
- **One municipality.** The data comes from one Dutch municipality and cannot establish performance for SPAR or
  Montenegro.

## Not Modelled: Time-Series Forecasting

Forecasting the number of new cases per month was **assessed for feasibility only**. Monthly arrival counts were checked,
but the coverage of the first and last months is uncertain, and no forecasting model was built.

## Repository Contents

| File | Description |
|---|---|
| `README.md` | This file |
| `requirements.txt` | Python packages used |
| `data/data_dictionary.csv` | Data dictionary for all 29 columns |
| `data/raw/README.md` | Where to get the raw dataset and where to put it |
| `notebooks/01_EDA.ipynb` … `notebooks/13_project_summary.ipynb` | The notebooks listed above |

## Running the Notebooks

1. Create an environment and install the packages:

   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. With Git LFS installed, run `git lfs pull` from the repository root to retrieve `data/raw/BPIC15_1.xes`.
3. Open the notebooks in Jupyter or VS Code with this environment and run all cells, in order from 01 to 12.
   Each notebook loads the raw data itself, using the path `../data/raw/BPIC15_1.xes`, so run it from the `notebooks`
   folder (the default when you open a notebook in Jupyter or VS Code).

## Data Source

van Dongen, B. (2015). *BPI Challenge 2015 Municipality 1*. Eindhoven University of Technology. 4TU.ResearchData. https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17
