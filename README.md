# GA Data Science Capstone

## Early Warning for Long Administrative Cases

**Author:** Ali Čikić · General Assembly – Data Science Bootcamp

**Project question:** Can early information from an administrative process help identify cases likely to have a longer
**recorded activity span**?

**Short answer:** At day 30, a simple Logistic Regression shows a **modest ranking signal** (ROC-AUC about 0.66–0.70),
but its warnings are unstable between test sets and the evidence is **not sufficient for operational use**. The tested
day-14 approaches showed only weak signal.

## 1. Why I Chose This Problem

In an administrative process, such as handling a building-permit application, some cases are finished in a few weeks
and others take many months. If the long cases could be recognised **early**, while they are still running, someone could
look at them sooner. That practical idea is the starting point of my project: what is already known about a case after
a few events, or after 14 or 30 days, and does it say anything about how long the case will run?

**Connection to SPAR.** The motivation comes from SPAR, where the same kind of question could later be asked about SPAR's
own historical cases. SPAR's data was **not** used in this project. I used a public event log from **one Dutch
municipality** instead, so this project demonstrates a **method** that might later be evaluated on SPAR's historical data.
The model is **not** validated for SPAR or Montenegro, and its numbers cannot be transferred to them.

## 2. The Data

**BPI Challenge 2015 — Municipality 1** (van Dongen, 2015, 4TU.ResearchData,
[DOI 10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17](https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17)):
building-permit applications from one Dutch municipality, recorded as an **event log**.

- **One event** (one row) is one recorded step in a case, for example "register submission date request", with a case
  ID, an activity code, a timestamp and a resource ID (a number, not a person's name).
- **One case** is one permit application. It consists of many events (typically about 44, between 2 and 101).
- **Scope:** 52,217 events, 1,199 cases, 29 columns, events from October 2010 to July 2015 (2010 and 2015 are partial years).
- **Quality checks** ([01_EDA](notebooks/01_EDA.ipynb)): the four key columns (case, activity, timestamp, resource) are
  complete and there are no duplicate rows. 14 of the 29 columns have missing values, mostly optional fields. I removed no
  rows or columns and filled in no values. Every column is described in
  [`data/data_dictionary.csv`](data/data_dictionary.csv); fields whose meaning is not documented are marked as such.
- **Exclusion:** I only use the **956 cases with a recorded end date** (`case:endDate`), the *historically completed
  cases*. The other 243 cases may have continued after the data was collected. A recorded end date does not guarantee
  that every step of the case is in the log.

This is a public research dataset, not SPAR data and not confidential government data.

## 3. What I Predict

**Recorded activity span** = days from the **first** to the **last** recorded event of a case (`duration_days` in the code).

This is what the event log shows. It is **not** a verified administrative completion time, a legal deadline, a measure of
lateness or a measure of anyone's performance. The last recorded event can be a later step such as publication or
archiving, and it often disagrees with the recorded end date: for 713 of the 956 completed cases the last event comes
**after** `case:endDate`, for 170 it comes **before**, and only 73 match. The meaning of `case:endDate` is not documented
in the sources I used, so I only use it to select the completed cases.

**Long case** = a recorded activity span of **more than 63 days** (472 of 956 completed cases).

- **How I chose 63 days:** it is the **median** span of all 956 completed cases in the EDA, so about half of the cases are long.
- **Limitations of this choice:** it is an **exploratory, data-derived** threshold, not an external deadline. I chose it
  while looking at **all** completed cases, including the cases that later became test cases, and not from training cases
  only. So none of the later evaluations is completely independent of this exploration.

## 4. Metrics and Baselines in Simple Words

- **Baseline:** the simplest possible "model". For regression: predict one constant number of days for every case (the
  mean or the median of the training cases). For classification: the **majority baseline** predicts the more common class
  for every case.
- **MAE:** the average number of days a prediction is wrong. On the training cases, the constant with the smallest MAE is
  the median, so the **median baseline** is the fairer constant to beat on MAE.
- **R²:** the share of the variation explained (0 = no better than predicting the mean).
- **Precision:** of the cases a model flags as long, the share that really are long ("how often a warning is correct").
- **Recall:** of the really long cases, the share the model flags ("how many long cases are found").
- **ROC-AUC:** how well the model **ranks** long cases above not-long cases, using its predicted probabilities.
  0.50 = no better than chance, 1.00 = perfect.

**Why high recall can be misleading.** At day 30 most eligible cases are long, so the majority baseline flags **every**
case as long. It finds all long cases (recall 1.00) and gets a high F1, but it cannot tell any two cases apart
(ROC-AUC 0.50). A model that flags nearly every case looks good on recall without being useful. That is why I compare
every model with its own baseline and use ROC-AUC as the main measure, together with how many cases are flagged.

## 5. The Investigation, Step by Step

| Step | Notebook | Cases (train / test) | Main verified result |
|---|---|---|---|
| Whole-case regression | 02–05 | 764 / 192 | MAE: median baseline 64.42 → Linear Regression 59.01 → Gradient Boosting 51.19 days |
| First five events | 06 | 763 / 191 | Accuracy 0.55 vs 0.51 (baseline); recall 0.28 |
| Day-14 counts | 07 | 733 / 184 | Accuracy 0.49 vs 0.52 (baseline); ROC-AUC 0.53 |
| Day-14 counts + activities | 08 | 733 / 184 | ROC-AUC 0.59; accuracy 0.58 |
| Day-14 regression | 09 | 733 / 184 | MAE: median baseline 70.75; models 76.90–81.27 |
| Day-30 classification | 10 | 677 / 161 | Logistic Regression ROC-AUC 0.702 (CV), 0.659 (test) |
| Temporal check | 11 | 617 / 168 | Logistic Regression ROC-AUC 0.678; recall 0.55 |
| Ensemble comparison | 12 | as 10 and 11 | No consistent gain over Logistic Regression |

**Step 1 — Whole-case regression (notebooks 02–05, supporting exercises).** I started with the regression models from
the course: Linear Regression, Decision Tree, Random Forest and Gradient Boosting, predicting the span in days from three
case-level features (number of events, unique activities, unique resources). On one random 80/20 split, Linear
Regression (MAE 59.01, R² 0.26) beat the median baseline (64.42) and the mean baseline (71.98). One Decision Tree was
almost the same (58.73) and fitted the training cases much better than the test cases (R² 0.52 vs 0.27). Random Forest
(52.00, R² 0.44) and Gradient Boosting (51.19, R² 0.49) were better on this split. **But these features describe the
finished case**: the total number of events is only known when the case has ended. So this is a regression exercise, not
an early-warning model, and I moved on to features that exist while a case is still running.

**Step 2 — First five events (notebook 06, exploratory).** I used only the first five events of each case. Logistic
Regression reached accuracy 0.55 against 0.51 for the baseline, and found only 28% of the long cases. Long cases had a
higher *mean* time to reach event 5 (15.67 vs 6.37 days) but almost the same *median* (5.38 vs 4.49), so the difference
comes from a minority of slow starts. There was also a problem: **21 cases already needed more than 63 days to reach their
fifth event**, so they were long before the prediction was made, and a prediction for them is not an advance warning.
Because the prediction moment differs so much between cases, I moved to a **fixed point in time**.

**Step 3 — Day-14 count features (notebook 07).** I used seven counts of what happened in the first 14 days (events,
unique activities, resources, recent inactivity and similar). Logistic Regression did **not** beat the baseline: accuracy
0.49 vs 0.52. Its recall (0.63) looked better than in step 2 only because it flagged many cases (119 of 184), and the
baseline that flags everything had even higher recall and F1 (1.00 and 0.68). Counting events was not enough, so I next
asked *which* activities had happened.

**Step 4 — Day-14 activity indicators (notebook 08).** I added 0/1 indicators for the 15 most common activity codes
recorded by day 14, selected on the training cases only. ROC-AUC rose from 0.53 to 0.59 and accuracy to 0.58: a small,
exploratory gain on a test set that I had already looked at in notebook 07. The signal was still weak.

**Step 5 — Day-14 regression (notebook 09, supporting).** I also tried to predict the span in days from the same day-14
features. No model beat the median baseline (MAE 70.75 vs 76.90–81.27 days). The tested day-14 approaches therefore had
**weak predictive signal**, and I tried a later snapshot.

**Step 6 — Day-30 classification with cross-validation (notebook 10).** Using the same kinds of features at day 30,
Logistic Regression reached a mean ROC-AUC of **0.702** in 5-fold cross-validation on the training cases (between 0.667
and 0.755 in the five folds) and **0.659** on the test cases. This is clearer than at day 14, but it is **not a perfectly
controlled comparison**: at day 30 the eligible cases change (838 instead of 917), the share of long cases rises (55.7% of
training and 59.0% of test cases), and the features cover 30 days instead of 14.

**Step 7 — Temporal check (notebook 11).** In practice a model is trained on older cases and used on newer ones, so I
trained on cases before 2013-10-25 and tested on the 168 newest cases. Logistic Regression ranked the newer cases about as
well (ROC-AUC **0.678**), but it flagged fewer cases and found only 55% of the long ones.

**Step 8 — Ensemble comparison (notebook 12).** Finally I compared Random Forest and Gradient Boosting with Logistic
Regression on exactly the same features and splits. The ensembles were better on the random test only, not in
cross-validation or on the temporal test.

## 6. Final Results (Day 30)

| ROC-AUC | Cross-validation mean | Random test (161 cases) | Temporal test (168 cases) |
|---|---|---|---|
| Majority baseline | 0.500 | 0.500 | 0.500 |
| Logistic Regression | 0.702 | 0.659 | 0.678 |
| Random Forest | 0.701 | 0.745 | 0.674 |
| Gradient Boosting | 0.684 | 0.723 | 0.624 |

| Logistic Regression | Cases flagged | Precision | Recall | Share of long cases in the test group |
|---|---|---|---|---|
| Random test | 98 of 161 | 0.694 | 0.716 | 59.0% |
| Temporal test | 71 of 168 | 0.775 | 0.55 | 59.5% |

**Why I keep Logistic Regression as the main simple candidate.** Random Forest had the best random-test result (0.745),
but it was practically equal to Logistic Regression in cross-validation (0.701 vs 0.702) and on the temporal test (0.674
vs 0.678). One favourable split is not enough to show that a model is better. Random Forest's warnings also changed more
between the tests: it flagged 118 of 161 random-test cases (recall 0.863) but only 51 of 168 temporal-test cases (recall
0.43). When results are similar, I prefer the simpler model that I can explain.

## 7. How Careful the Evaluation Is

- **Feature timing:** the day-14 and day-30 features use only events timestamped on or before the cutoff. A leakage check
  replaces all later activities and resources with fake values and confirms that the features do not change. This assumes
  that the timestamps show when the information was available.
- **Retrospective cohort:** a case is included at day N if its last recorded event is after first event + N days (day 14:
  917 cases, day 30: 838). This uses the last event, which is only known afterwards, and it does not prove that the case
  was administratively open at day N: 19 (day 14) and 39 (day 30) of these cases already have a recorded end date by then.
  The leakage check tests the feature code only; it does not remove this cohort-selection bias, and a real system would
  need to know which cases are really open.
- **Training data only:** activity indicators are selected on training cases only, and again inside every
  cross-validation training fold; the scaler is fitted on training data only (inside a pipeline).
- **Splits:** the random split from notebook 07 (733 / 184) is reused in notebooks 08–10 and 12; at day 30 it is filtered
  to 677 / 161, never redrawn. In the temporal split, training cases must have their last event and end date before the
  boundary (53 older cases were left out, 44 of them long), and the training and test groups do not overlap.
- **Reused test cases:** I looked at the same test cases in several notebooks, so none of the results is an untouched
  final test.
- **Temporal test is not independent:** 138 of the 168 temporal-test cases were **training** cases in the earlier random
  split (notebooks 07–10), and the features and settings were chosen while looking at those notebooks. The temporal
  evaluation is therefore a **chronological robustness check**, not an independent final evaluation.
- **Redundant features:** two of the seven count features are exact combinations of others (repeated activities = events
  − unique activities; events in the second interval = all events − events in the first interval). I kept them for
  continuity and do not interpret their coefficients separately.

## 8. What the Findings Support and What They Do Not

**They support:**
- At day 30, the tested features contain a modest ranking signal for long recorded activity spans in this municipality's
  historical cases, seen in cross-validation, on the random test and on newer cases.
- Logistic Regression is a reasonable simple candidate; the tested ensembles did not show a consistent improvement.

**They do not establish:**
- that the model would be useful in practice: at the default threshold it misses many long cases (45 of 100 on the
  temporal test), and its warnings changed between test sets;
- that early information is generally uninformative: the weak day-14 results apply to the **tested** features and models only;
- why cases take long: coefficients and feature importances are not causes;
- anything about legal delays, lateness or staff performance;
- any performance for SPAR, Montenegro or another municipality.

**What a future SPAR pilot would need:**
- SPAR's own historical cases with reliable timestamps for when each step was recorded;
- a reliable way to know which cases are really open at the prediction moment, instead of a retrospective rule;
- a target agreed with the people who handle the cases, for example a documented deadline, with the threshold chosen on
  training data only;
- training on older cases and a final test on newer cases that is kept untouched until the very end;
- a check of how many warnings staff can actually handle, and which threshold gives a useful balance of precision and recall.

## 9. Notebooks

**Main reading order**

1. [13_project_summary](notebooks/13_project_summary.ipynb) — short overview (Markdown only)
2. [01_EDA](notebooks/01_EDA.ipynb) — data, quality checks, target and threshold
3. [06_early_warning_features_review](notebooks/06_early_warning_features_review.ipynb) — first five events
4. [07_14_day_early_warning_review](notebooks/07_14_day_early_warning_review.ipynb) — day-14 counts (creates the random split)
5. [08_14_day_activity_features_review](notebooks/08_14_day_activity_features_review.ipynb) — day-14 activity indicators
6. [10_day30_early_warning_review](notebooks/10_day30_early_warning_review.ipynb) — day 30 with cross-validation
7. [11_day30_temporal_validation_review](notebooks/11_day30_temporal_validation_review.ipynb) — older vs newer cases
8. [12_day30_model_comparison_review](notebooks/12_day30_model_comparison_review.ipynb) — ensembles and final charts

**Supporting experiments**

- [02_linear_regression_review](notebooks/02_linear_regression_review.ipynb),
  [03_decision_tree_review](notebooks/03_decision_tree_review.ipynb),
  [04_random_forest_review](notebooks/04_random_forest_review.ipynb),
  [05_gradient_boosting_review](notebooks/05_gradient_boosting_review.ipynb) — whole-case regression exercises
- [09_day14_regression_review](notebooks/09_day14_regression_review.ipynb) — day-14 regression

Every notebook loads the raw data itself and can be run on its own.

## 10. Setup and Running the Notebooks

**Tested with Python 3.14.7 on Windows 11.** Other Python versions and operating systems were not tested.

**1. Get the project and the data.** The raw file `data/raw/BPIC15_1.xes` (about 41 MB) is stored in this repository
with **Git LFS** (configured in `.gitattributes`).

- Install Git LFS (<https://git-lfs.com>) and run `git lfs install` once, then clone the repository. Git LFS downloads
  the data file automatically; in a clone made without Git LFS, run `git lfs pull`.
- Without Git LFS the file is only a small text pointer. In that case, download `BPIC15_1.xes` from the DOI link above
  and place it at `data/raw/BPIC15_1.xes`. [`data/raw/README.md`](data/raw/README.md) explains this and how to check the
  file's SHA-256 hash.

**2. Create an environment and install the required packages** (from the project folder):

```
python -m venv .venv
.venv\Scripts\activate            (Windows)
source .venv/bin/activate         (macOS / Linux)
python -m pip install -r requirements.txt
```

`requirements.txt` contains the packages the notebooks use (pandas, matplotlib, PM4Py, scikit-learn) and `ipykernel`,
which lets an editor run the notebooks with this environment. It does **not** include a notebook editor.

**3. Open the notebooks with one of these:**

- **VS Code** with its Jupyter extension: open the project folder, open a notebook, select the `.venv` environment as the
  kernel and choose *Run All*. Nothing else needs to be installed into the environment.
- **JupyterLab** (optional): install it into the same environment with `python -m pip install jupyterlab==4.6.5`, start
  `jupyter lab` from the project folder and use *Run → Run All Cells*.

The data path is found from the project folder, so a notebook works whether Jupyter starts in the project folder or in
the `notebooks` folder. Each notebook ran in under 20 seconds on my computer, mostly to read the XES file. When PM4Py is
imported it prints a short welcome and licence message; this is normal.

## 11. Repository Contents

| Path | Description |
|---|---|
| `README.md` | This file |
| `requirements.txt` | Required Python packages |
| `data/raw/BPIC15_1.xes` | Raw event log (Git LFS, unchanged from the source) |
| `data/raw/README.md` | Source, terms of use, download and hash check for the raw data |
| `data/data_dictionary.csv` | Description of all 29 columns |
| `notebooks/` | Notebooks 01–13 listed above |

## 12. Data Source and Terms

van Dongen, B. (2015). *BPI Challenge 2015 Municipality 1*. Eindhoven University of Technology. 4TU.ResearchData.
https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17

Terms of use: 4TU General Terms of Use ([doi.org/10.4121/resource:terms_of_use](https://doi.org/10.4121/resource:terms_of_use)),
as listed in the dataset's DOI record.
