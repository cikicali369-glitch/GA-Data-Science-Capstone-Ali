# Research history

This document records how the project developed, stage by stage, so that the root README and the two final notebooks
can stay focused. Every number below comes from the saved outputs of the original notebooks. The original notebooks are
preserved in commit `f29aa8a` (archive tag `archive/research-notebooks-01-21`); each link opens the exact version that produced the results.

Regression results (in **days**, MAE / RMSE / R²) and classification results (**ROC-AUC**, precision, recall) measure
different things and are never compared with each other.

[01]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/01_EDA.ipynb
[02]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/02_linear_regression_review.ipynb
[03]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/03_decision_tree_review.ipynb
[04]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/04_random_forest_review.ipynb
[05]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/05_gradient_boosting_review.ipynb
[06]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/06_early_warning_features_review.ipynb
[07]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/07_14_day_early_warning_review.ipynb
[08]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/08_14_day_activity_features_review.ipynb
[09]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/09_day14_regression_review.ipynb
[10]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/10_day30_early_warning_review.ipynb
[11]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/11_day30_temporal_validation_review.ipynb
[12]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/12_day30_model_comparison_review.ipynb
[13]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/13_project_summary.ipynb
[14]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/14_day30_activity_groups_review.ipynb
[15]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/15_day30_activity_pca_review.ipynb
[16]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/16_day30_nlp_review.ipynb
[17]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/17_day30_hyperparameter_tuning_review.ipynb
[18]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/18_voting_regressor_review.ipynb
[19]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/19_model_persistence_review.ipynb
[20]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/20_regression_metrics_review.ipynb
[21]: https://github.com/cikicali369-glitch/GA-Data-Science-Capstone-Ali/blob/f29aa8a3da65f47a661828be5be17589692e838a/notebooks/21_readme_figures.ipynb

## Overview

| Stage | Original notebooks | Question | Main result |
|---|---|---|---|
| 1. EDA | [01][01] | What is in the data, and what is a "long" case? | 52,217 events, 1,199 cases; median recorded activity span 63 days |
| 2. Whole-case regression | [02][02], [03][03], [04][04], [05][05] | Can the span in days be predicted? | MAE 64.42 (median baseline) → 51.19 days (Gradient Boosting), but with information only known after a case ends |
| 3. First five events | [06][06] | Do the first five events help? | Weak: accuracy 0.55 vs 0.51 |
| 4. Day 14 | [07][07], [08][08], [09][09] | Does information at day 14 help? | Counts: no signal; activities: ROC-AUC 0.53 → 0.59; regression: no model beat the median baseline |
| 5. Day 30 | [10][10], [11][11], [12][12], [13][13] | Does information at day 30 help, also on newer cases? | Modest signal: Logistic Regression ROC-AUC 0.702 (CV), 0.659 (random test), 0.678 (temporal test) |
| 6. Follow-up recommendations | [14][14]–[21][21] | Do the reviewer's suggestions improve the work? | Small gains for activity groups and tuning; no gain from TF-IDF or VotingRegressor; better workflow |
| 7. Final selection | final `notebooks/02_modelling.ipynb` | Which model to present? | Logistic Regression C, chosen with the one-standard-error rule |

## Stage 1 — Exploratory data analysis ([01][01])

- **Question:** what does the event log contain, how good is the data, and what should the target be?
- **Method:** load the XES file with PM4Py, check missing values, duplicates and timestamps, and describe the cases.
- **Results:** 52,217 events in 1,199 cases with 29 columns (October 2010 – July 2015). The four key columns are complete
  and there are no fully duplicated rows. 956 cases have a recorded end date; their recorded activity span (first to last
  recorded event) has a median of 63 days, a mean of about 98 days and a maximum of 1,486 days. The recorded end date
  often differs from the last recorded event (after it for 713 cases, before it for 170, equal for 73). A supplement found
  that 45.9% of events are recorded exactly at local midnight, 30,433 events share a timestamp within their case, and 89
  of the 243 cases without an end date started in 2014.
- **What worked:** the data is complete enough for the project question.
- **What did not:** several columns are poorly documented, and timestamps are often only dates.
- **Next:** a case is called **long** if its recorded activity span is more than 63 days (472 of 956 cases). This
  exploratory threshold, chosen with all completed cases, is used in every later stage. The final version of this work is
  `notebooks/01_eda.ipynb`.

## Stage 2 — Whole-case regression ([02][02], [03][03], [04][04], [05][05], later [18][18] and [20][20])

- **Question:** can the recorded activity span **in days** be predicted from three case-level counts (events, different
  activities, different resources)?
- **Method:** one random 80/20 split of the 956 cases (764 / 192), four regression models, with mean and median baselines.
- **Results** (192 test cases; also in [`results/regression_model_comparison.csv`](results/regression_model_comparison.csv)):

| Model | MAE (days) | RMSE (days) | R² |
|---|---|---|---|
| Median baseline | 64.42 | 153.00 | −0.08 |
| Linear Regression ([02][02]) | 59.01 | 126.02 | 0.26 |
| Decision Tree ([03][03]) | 58.73 | 125.50 | 0.27 |
| Random Forest ([04][04]) | 52.00 | 109.60 | 0.44 |
| Gradient Boosting ([05][05]) | 51.19 | 104.63 | 0.49 |
| VotingRegressor ([18][18]) | 51.16 | 110.87 | 0.43 |

  (The mean baseline had an MAE of 71.98 days. The Decision Tree is not in the CSV, which lists the models compared in
  [20][20].)
- **What worked:** the ensemble models reduced the error compared with the median baseline.
- **What did not:** the counts describe the **whole finished case**, so these models cannot give an early warning. Linear
  Regression predicted 4 impossible negative spans, the single tree overfitted (training R² 0.52 vs test 0.27), and the
  longest cases were strongly under-predicted by every model.
- **Next:** use only information available early in a case.

## Stage 3 — First five events ([06][06])

- **Question:** do the first five recorded events say which cases will be long?
- **Method:** five simple features from the first five events of the 954 completed cases with at least five events;
  Logistic Regression on a stratified random split (763 / 191).
- **Results:** accuracy 0.55 vs 0.51 for the majority baseline; precision 0.60; recall 0.28.
- **What worked:** a slow start is weakly associated with a long span.
- **What did not:** the signal is weak, and 21 cases already needed more than 63 days to reach their fifth event, so for
  them a "prediction" is not a warning. The fifth event also comes at very different times in different cases.
- **Next:** predict at a fixed point in time instead of after a fixed number of events.

## Stage 4 — Day 14 ([07][07], [08][08], [09][09])

- **Question:** does information from the first 14 days help?
- **Method:** the 917 completed cases whose recorded span goes beyond day 14 (a retrospective rule); a stratified random
  split of 733 / 184 cases, which is reused, filtered, in all later random-split experiments.
- **Results:**
  - Seven count features ([07][07]): accuracy 0.49 vs 0.52 for the baseline; the model flagged 119 of 184 cases, and the
    baseline that flags every case had higher recall and F1 (1.00 and 0.68).
  - Counts plus 15 activity indicators ([08][08]): ROC-AUC 0.53 → 0.59, accuracy 0.58, precision 0.60, recall 0.54.
  - Predicting the span in days ([09][09]): no model beat the median baseline (MAE 70.75 days vs 76.90–81.27).
- **What worked:** knowing *which* activities happened added a little signal.
- **What did not:** overall the day-14 signal was weak.
- **Next:** try a later prediction point, day 30.

## Stage 5 — Day 30 and the temporal check ([10][10], [11][11], [12][12], [13][13])

- **Question:** does information from the first 30 days help, and does it also work on newer cases?
- **Method:** the 838 completed cases whose recorded span goes beyond day 30 (a retrospective rule); the day-14 random
  split filtered to day 30 (677 / 161); five-fold cross-validation on the training cases, with all feature choices made
  inside the folds; a temporal split training on older cases and testing on the newest 168 (617 / 168, boundary
  2013-10-25).
- **Results (ROC-AUC):** Logistic Regression 0.702 ± 0.033 in cross-validation, 0.659 on the random test ([10][10]) and
  0.678 on the temporal test ([11][11]); a Decision Tree reached 0.629 in cross-validation and 0.636 on the random test. In
  [12][12], Random Forest (0.701 / 0.745 / 0.674) and Gradient Boosting (0.684 / 0.723 / 0.624) were better than Logistic
  Regression on the random test only.
- **What worked:** a modest, repeatable ranking signal, also on newer cases.
- **What did not:** the warnings were unstable (Logistic Regression recall 0.716 on the random test, 0.55 on the temporal
  test). 138 of the 168 temporal test cases had been training cases in the random split, so the temporal test is a
  robustness check, not an independent test.
- **Next:** [13][13] summarised the project at this point; then the reviewer's recommendations were tested.

## Stage 6 — Follow-up recommendations by Hank Butler ([14][14]–[21][21])

| # | Recommendation | Original implementation | Verified outcome | Where it is now |
|---|---|---|---|---|
| 1 | Group activity codes into categories | [14][14]: groups by code prefix (first digit for `01_HOOFD`), selected on training cases | B (groups instead of codes) 0.694 / 0.680 / 0.709; **C (codes + groups)** 0.710 / 0.708 / 0.705 — a small CV gain | Final model C, `02_modelling.ipynb` §§4, 7 |
| 2 | One-hot encoding + PCA | [15][15]: all activity codes as 0/1 columns, PCA fitted inside the folds; 5, 10 or 20 components | 10 components chosen by CV: 0.709 / 0.721 / 0.729; no clear CV gain (5: 0.675, 20: 0.697) | `02_modelling.ipynb` §§4, 7 |
| 3 | NLP: text features | [16][16]: TF-IDF (30 words) of the English activity descriptions, fitted inside the folds | 0.702 / 0.725 / 0.657 — no CV gain, worse on newer cases (a negative result) | `02_modelling.ipynb` §§4, 7 |
| 4 | Tune the ensemble models | [17][17]: 20 new settings fixed in advance (12 Random Forest, 8 Gradient Boosting), chosen by CV | Tuned Random Forest 0.720 / 0.752 / 0.709 (200 trees, depth 5, leaf 1); tuned Gradient Boosting 0.710 / 0.726 / 0.636 | `02_modelling.ipynb` §7; all 22 settings (20 new + 2 original) in [`results/day30_tuning_results.csv`](results/day30_tuning_results.csv) |
| 5 | VotingRegressor | [18][18]: average of Linear Regression, Random Forest and Gradient Boosting | MAE 51.16 vs 51.19 days for Gradient Boosting (no meaningful gain); worse RMSE (110.87 vs 104.63) — a negative result | Stage 2 above; [`results/regression_model_comparison.csv`](results/regression_model_comparison.csv) |
| 6 | Save fitted models | [19][19]: pickle a trained model, reload it, compare predictions | All predictions identical after reloading | Final model saved locally by `02_modelling.ipynb` §12 (not published) |
| 7 | Reusable metrics function + CSV | [20][20]: one function for MAE / RMSE / R², results saved to CSV | Same results as before | Regression CSV in `docs/results/`; one `classification_metrics` function and CSVs in `02_modelling.ipynb` |
| 8 | Use scikit-learn's `root_mean_squared_error` | [18][18]–[20][20], and later in [02][02]–[05][05] and [09][09] | Identical RMSE values to the manual calculation | Historical regression notebooks (the final notebooks are classification only) |
| 9 | Export figures and show them in the README | [21][21] and the README charts | Charts reproduced from the notebooks | `docs/figures/`, root README |

ROC-AUC values are given as cross-validation / random test / temporal test.

## Stage 7 — Final comparison and selection (`notebooks/02_modelling.ipynb`)

The final notebook re-runs all day-30 candidates from the processed data and reproduces every result above exactly.

- **Highest mean CV ROC-AUC:** tuned Random Forest, 0.720 (fold standard deviation 0.035).
- **One-standard-error rule:** standard error = 0.035 / √5 ≈ 0.016, so the threshold is **0.705** (calculated from the
  unrounded values). Within it: Logistic Regression C (0.710), Logistic Regression D with PCA (0.709) and tuned Gradient
  Boosting (0.710).
- **Selected: Logistic Regression C** — the simplest of these models with features that can each be explained, and the
  smallest variation across folds (0.017). PCA had the highest temporal-test ROC-AUC (0.729), but the test results were
  not used for the choice.
- **Caveats:** the rule was formalised **after** the exploratory experiments; it does not prove that the models are
  statistically equal or that one is better.
- **Errors (random test, Logistic Regression C):** 73 of 95 long cases found, 27 false alarms; on the temporal test 50 of
  100 found. False positives and false negatives looked alike at day 30 in the observed test cases.

## Limitations that apply to every stage

- The target is the **recorded activity span**, not a verified administrative completion time or a legal delay.
- The 63-day threshold is exploratory and was chosen with all completed cases, including later test cases.
- Day-14 and day-30 eligibility use each case's last recorded event, which is only known afterwards (retrospective).
- Test sets were reused across the research; the temporal test overlaps with earlier random training cases.
- Timestamps are often only dates; many configurations were compared on the same folds.
- One historical Dutch municipality; not SPAR data; no model has been validated for operational use.
