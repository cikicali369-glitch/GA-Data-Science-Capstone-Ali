# Figures and supporting documents

## Figures (`figures/`)

Every figure is saved by notebook code; re-running the notebook re-creates it.

| Figure | What it shows | Made by |
|---|---|---|
| [`eda_missing_values.png`](figures/eda_missing_values.png) | Share of missing values in each of the 14 columns that have gaps | `notebooks/01_eda.ipynb` §7 |
| [`eda_events_per_case.png`](figures/eda_events_per_case.png) | How many events each of the 1,199 cases has (2 to 101) | `notebooks/01_eda.ipynb` §8 |
| [`eda_recorded_span_full_range.png`](figures/eda_recorded_span_full_range.png) | Recorded activity span of the 956 cases with an end date, full range | `notebooks/01_eda.ipynb` §8 |
| [`eda_recorded_span_log_scale.png`](figures/eda_recorded_span_log_scale.png) | The same spans on a logarithmic axis, with the 63-day median | `notebooks/01_eda.ipynb` §8 |
| [`eda_top_activities.png`](figures/eda_top_activities.png) | The ten most common activities | `notebooks/01_eda.ipynb` §8 |
| [`eda_correlation_heatmap.png`](figures/eda_correlation_heatmap.png) | Correlations between case size and recorded activity span | `notebooks/01_eda.ipynb` §9 |
| [`model_comparison_roc_auc.png`](figures/model_comparison_roc_auc.png) | **Headline figure:** ROC-AUC of all day-30 models in cross-validation, on the random test and on the temporal test | `notebooks/02_modelling.ipynb` §8 |
| [`model_final_confusion_matrices.png`](figures/model_final_confusion_matrices.png) | Correct and wrong predictions of the final model on both tests | `notebooks/02_modelling.ipynb` §9 |
| [`model_final_coefficients.png`](figures/model_final_coefficients.png) | The 15 largest coefficients of the final model | `notebooks/02_modelling.ipynb` §10 |

## Result tables (`results/`)

| File | What it contains | Made by |
|---|---|---|
| [`day30_model_comparison.csv`](results/day30_model_comparison.csv) | All day-30 models: cross-validation ROC-AUC (mean and standard deviation) and ROC-AUC, precision, recall and cases flagged on both tests | `notebooks/02_modelling.ipynb` §8 |
| [`day30_tuning_results.csv`](results/day30_tuning_results.csv) | All 22 ensemble settings (2 original + 20 tuning settings) with their cross-validation ROC-AUC | `notebooks/02_modelling.ipynb` §7 |
| [`final_model_coefficients.csv`](results/final_model_coefficients.csv) | Every coefficient of the final Logistic Regression C | `notebooks/02_modelling.ipynb` §10 |
| [`regression_model_comparison.csv`](results/regression_model_comparison.csv) | Historical whole-case regression results (MAE, RMSE, R²), including the VotingRegressor | original notebook 20 (see `research_history.md`) |

## Documents

- [`research_history.md`](research_history.md): how the research developed, with the results of every earlier experiment
  and links to the original notebooks.
