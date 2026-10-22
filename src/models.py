"""Time-aware modelling helpers: chronological split, train-only label, pipelines and evaluation.

Also holds the secondary Day-14 duration-regression helpers (bottom of the file).

Every fitted quantity (label threshold, preprocessing, model, probability threshold) is learned on the training
split or chosen on the validation split. The test split is only scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TRAIN, VALIDATION, TEST = "train", "validation", "test"
LABEL_PERCENTILE = 0.75
DEFAULT_PROBABILITY_THRESHOLD = 0.50
# A different probability threshold is adopted only if it beats 0.50 on validation F1 by at least this margin.
THRESHOLD_MIN_F1_GAIN = 0.05
THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)
# One-hot categories seen in fewer training cases than this are merged into an "infrequent" column.
MIN_CATEGORY_FREQUENCY = 20


def chronological_split(case_start: pd.Series, validation_start: pd.Timestamp, test_start: pd.Timestamp) -> pd.Series:
    """train: start < validation_start; validation: validation_start <= start < test_start; test: start >= test_start."""
    if not validation_start < test_start:
        raise ValueError("validation_start must be before test_start")
    split = np.where(case_start < validation_start, TRAIN, np.where(case_start < test_start, VALIDATION, TEST))
    return pd.Series(split, index=case_start.index, name="split")


def follow_up_flag(case_start: pd.Series, extract_end: pd.Timestamp, min_days: int = 365) -> pd.Series:
    """True if the case could have been observed for at least ``min_days`` before the extract ended."""
    return ((extract_end - case_start).dt.total_seconds() / 86400 >= min_days).rename(f"follow_up_{min_days}d")


def train_threshold(duration: pd.Series, split: pd.Series, percentile: float = LABEL_PERCENTILE) -> float:
    """Label threshold from TRAINING cases only."""
    return float(duration[split == TRAIN].quantile(percentile))


def long_running_label(duration: pd.Series, threshold_days: float) -> pd.Series:
    return (duration > threshold_days).astype(int).rename("long_running")


def constant_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if frame[column].nunique(dropna=False) <= 1]


def build_preprocessor(numeric: list[str], categorical: list[str], binary: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("numeric", numeric_step, numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_CATEGORY_FREQUENCY), categorical),
        ("binary", "passthrough", binary),
    ], verbose_feature_names_out=True)


def logistic_pipeline(numeric, categorical, binary, class_weight=None) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical, binary, scale_numeric=True)),
        ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight)),
    ])


def random_forest_pipeline(numeric, categorical, binary) -> Pipeline:
    # Small fixed configuration: enough trees for stable estimates; leaf size limits overfitting on ~600 training cases.
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical, binary, scale_numeric=False)),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                         class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def evaluate(y_true, probability, threshold: float = DEFAULT_PROBABILITY_THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    predicted = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "cases": len(y_true), "prevalence": y_true.mean(),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability) if len(set(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, probability) if y_true.sum() > 0 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_probability_threshold(y_validation, probability, grid=THRESHOLD_GRID,
                                 min_gain: float = THRESHOLD_MIN_F1_GAIN) -> tuple[float, pd.DataFrame]:
    """Keep 0.50 unless another grid threshold improves VALIDATION F1 by at least ``min_gain``."""
    table = pd.DataFrame([{"threshold": t, **evaluate(y_validation, probability, t)} for t in grid])
    f1_default = evaluate(y_validation, probability, DEFAULT_PROBABILITY_THRESHOLD)["f1"]
    best = table.loc[table["f1"].idxmax()]
    chosen = float(best["threshold"]) if best["f1"] - f1_default >= min_gain else DEFAULT_PROBABILITY_THRESHOLD
    return chosen, table


def bootstrap_interval(y_true, probability, metric, n_resamples: int = 1000, seed: int = RANDOM_STATE,
                       threshold: float | None = None) -> tuple[float, float]:
    """95% percentile bootstrap interval of a metric (resampling cases). Resamples with a single class are skipped."""
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[index])) < 2:
            continue
        values.append(evaluate(y_true[index], probability[index], threshold or DEFAULT_PROBABILITY_THRESHOLD)[metric])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# Choices frozen in notebooks/04_modeling.ipynb before the test period was scored. Do not change them after the fact.
LOCKED_VALIDATION_START = "2013-01-01"
LOCKED_TEST_START = "2014-01-01"
LOCKED_LOGISTIC_CLASS_WEIGHT = "balanced"
LOCKED_FINAL_MODEL = "random forest"
LOCKED_PROBABILITY_THRESHOLD = 0.25


def reproduce_locked_experiment(table: pd.DataFrame, snap: pd.DataFrame) -> dict:
    """Re-run the frozen Day-14 experiment exactly as locked in 04_modeling (no choice is re-made).

    ``table``: the 14-day feature CSV indexed by case_id. ``snap``: raw 14-day snapshot events (src.features.snapshot_events).
    Returns the split, labels, train-only threshold and vocabulary, fitted models and predicted probabilities.
    """
    from . import features as F

    timezone = F.LOCAL_TIMEZONE
    split = chronological_split(table["case_start"], pd.Timestamp(LOCKED_VALIDATION_START, tz=timezone),
                               pd.Timestamp(LOCKED_TEST_START, tz=timezone))
    threshold_days = train_threshold(table["event_log_days"], split)
    labels = long_running_label(table["event_log_days"], threshold_days)

    activity_table, codes = F.train_vocabulary_activity_features(snap, table.index[split == TRAIN])
    model_table = table[F.CORE_NUMERIC_FEATURES].join(activity_table, how="left")
    binary = [F.ACTIVITY_FLAG_PREFIX + c for c in codes]
    categorical = list(F.CATEGORICAL_FEATURES)
    numeric = list(F.CORE_NUMERIC_FEATURES)
    constant = constant_columns(model_table.loc[split == TRAIN, numeric + binary + categorical])
    numeric = [c for c in numeric if c not in constant]
    binary = [c for c in binary if c not in constant]
    features = numeric + categorical + binary

    train_mask = split == TRAIN
    models = {
        "logistic (balanced)": logistic_pipeline(numeric, categorical, binary, class_weight=LOCKED_LOGISTIC_CLASS_WEIGHT),
        "random forest": random_forest_pipeline(numeric, categorical, binary),
    }
    for model in models.values():
        model.fit(model_table.loc[train_mask, features], labels[train_mask])
    probability = {
        (name, part): model.predict_proba(model_table.loc[split == part, features])[:, 1]
        for name, model in models.items() for part in (VALIDATION, TEST)
    }
    return {
        "split": split, "labels": labels, "train_p75_days": threshold_days, "activity_codes": codes,
        "dropped_constant": constant, "features": features, "model_table": model_table,
        "models": models, "probability": probability,
    }


def logistic_coefficients(pipeline: Pipeline) -> pd.Series:
    """Coefficients on the preprocessed scale (numeric: per standard deviation; one-hot/flags: presence)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(pipeline.named_steps["model"].coef_[0], index=names, name="coefficient")


def forest_importances(pipeline: Pipeline) -> pd.Series:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(pipeline.named_steps["model"].feature_importances_, index=names, name="importance")


# ---------------------------------------------------------------------------------------------------------------------
# Secondary regression extension (notebooks/06_regression.ipynb): predict total event-log duration in days from the
# same Day-14 inputs, split and train-only vocabulary as the locked classification experiment. Nothing above changes.
# ---------------------------------------------------------------------------------------------------------------------
REGRESSION_TARGET = "event_log_days"
# Columns that describe the outcome or the case's future. None of them may ever be a predictor.
OUTCOME_COLUMNS = ["case_end", "event_log_days", "remaining_days_after_snapshot", "long_running",
                   "requestComplete", "caseStatus", "last_phase"]
# Exact algebraic identities among the Day-14 features (checked on the data in 06_regression):
#   repeated_event_count_14d = event_count_14d - unique_activity_count_14d
#   events_days_8_to_14      = event_count_14d - events_first_7_days
# Ordinary least squares cannot separate perfectly dependent columns, so the linear models drop the derived member of
# each pair. The prediction information is unchanged; only the coefficients become identifiable.
LINEAR_REDUNDANT_FEATURES = ["repeated_event_count_14d", "events_days_8_to_14"]


def regression_feature_groups(features: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a locked feature list into (numeric, categorical, binary) columns. Refuses any outcome column."""
    from . import features as F

    leaked = [c for c in features if c in OUTCOME_COLUMNS or c in F.METADATA_COLUMNS]
    if leaked:
        raise ValueError(f"outcome/metadata columns among predictors: {leaked}")
    categorical = [c for c in features if c in F.CATEGORICAL_FEATURES]
    binary = [c for c in features if c.startswith(F.ACTIVITY_FLAG_PREFIX)]
    numeric = [c for c in features if c not in categorical and c not in binary]
    return numeric, categorical, binary


def linear_preprocessor(numeric, categorical, binary) -> ColumnTransformer:
    # drop="first": the first category of each column becomes the reference level, so one-hot columns are not
    # collinear with the intercept. Unseen categories (handle_unknown="ignore") are encoded like the reference level.
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_CATEGORY_FREQUENCY, drop="first"),
         categorical),
        ("binary", "passthrough", binary),
    ], verbose_feature_names_out=True)


def _linear_columns(numeric):
    return [c for c in numeric if c not in LINEAR_REDUNDANT_FEATURES]


def linear_regression_pipeline(numeric, categorical, binary) -> Pipeline:
    """Ordinary least squares on duration in days."""
    return Pipeline([
        ("preprocess", linear_preprocessor(_linear_columns(numeric), categorical, binary)),
        ("model", LinearRegression()),
    ])


def log_linear_regression_pipeline(numeric, categorical, binary) -> TransformedTargetRegressor:
    """Least squares on log1p(days); predictions are transformed back with expm1, so they are in days again."""
    return TransformedTargetRegressor(regressor=linear_regression_pipeline(numeric, categorical, binary),
                                      func=np.log1p, inverse_func=np.expm1, check_inverse=True)


def random_forest_regressor_pipeline(numeric, categorical, binary) -> Pipeline:
    # Same modest fixed configuration as the classification forest; no hyperparameter search.
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical, binary, scale_numeric=False)),
        ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                        n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and R² in days, plus the mean residual (actual - predicted; > 0 means durations were underpredicted)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "cases": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "mean_residual": float(np.mean(y_true - y_pred)),
    }


def constant_baseline(y_train, statistic: str = "median") -> float:
    """Naive historical prediction learned from TRAINING outcomes only: their median (or mean)."""
    if statistic not in ("median", "mean"):
        raise ValueError("statistic must be 'median' or 'mean'")
    return float(getattr(pd.Series(y_train, dtype=float), statistic)())


def bootstrap_mae_gain(y_true, y_model, y_baseline, n_resamples: int = 1000, seed: int = RANDOM_STATE):
    """95% percentile bootstrap interval of MAE(baseline) - MAE(model), resampling cases (> 0 = model better)."""
    y_true, y_model, y_baseline = (np.asarray(v, dtype=float) for v in (y_true, y_model, y_baseline))
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y_true), len(y_true))
        gains.append(np.mean(np.abs(y_true[index] - y_baseline[index])) - np.mean(np.abs(y_true[index] - y_model[index])))
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def linear_regression_coefficients(model) -> pd.Series:
    """Coefficients mapped to preprocessed feature names (numeric: per standard deviation; one-hot/flags: presence).

    Accepts a fitted linear_regression_pipeline or log_linear_regression_pipeline (log-scale coefficients).
    """
    pipeline = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(pipeline.named_steps["model"].coef_, index=names, name="coefficient")
