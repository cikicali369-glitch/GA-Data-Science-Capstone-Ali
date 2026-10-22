"""Tests for the Day-14 duration-regression helpers in src/models.py.

Run with:  .venv\\Scripts\\python.exe tests\\test_regression.py   (plain asserts; also pytest-compatible)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import features as F  # noqa: E402
from src import models as M  # noqa: E402

DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2011-01-01", tz="UTC")
NUMERIC = ["event_count_14d", "unique_activity_count_14d", "repeated_event_count_14d"]
CATEGORICAL = ["first_activity"]
BINARY = [F.ACTIVITY_FLAG_PREFIX + "A"]


def toy_frame(n, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    events = rng.integers(1, 20, n).astype(float) + shift
    unique = np.minimum(events, rng.integers(1, 10, n))
    return pd.DataFrame({
        "event_count_14d": events, "unique_activity_count_14d": unique, "repeated_event_count_14d": events - unique,
        "first_activity": rng.choice(["x", "y"], n), BINARY[0]: rng.integers(0, 2, n),
    })


def test_feature_groups_refuse_outcome_columns():
    numeric, categorical, binary = M.regression_feature_groups(NUMERIC + CATEGORICAL + BINARY)
    assert (numeric, categorical, binary) == (NUMERIC, CATEGORICAL, BINARY)
    for leaked in ["event_log_days", "case_end", "requestComplete", "remaining_days_after_snapshot", "case_start"]:
        try:
            M.regression_feature_groups(NUMERIC + [leaked])
        except ValueError:
            continue
        raise AssertionError(f"{leaked} was accepted as a predictor")


def test_preprocessing_is_fitted_on_training_rows_only():
    train, later = toy_frame(50, seed=1), toy_frame(30, seed=2, shift=100.0)  # later period on a very different scale
    pipe = M.linear_regression_pipeline(NUMERIC, CATEGORICAL, BINARY).fit(train, np.arange(50.0))
    scaler = pipe.named_steps["preprocess"].named_transformers_["numeric"]
    assert np.allclose(scaler.mean_, train[["event_count_14d", "unique_activity_count_14d"]].mean())
    pipe.predict(later)  # scoring the later period must not refit anything
    assert np.allclose(scaler.mean_, train[["event_count_14d", "unique_activity_count_14d"]].mean())


def test_linear_models_drop_exactly_redundant_columns_and_handle_unseen_categories():
    train = toy_frame(40)
    pipe = M.linear_regression_pipeline(NUMERIC, CATEGORICAL, BINARY).fit(train, np.arange(40.0))
    names = M.linear_regression_coefficients(pipe).index
    assert not any("repeated_event_count_14d" in name for name in names)
    assert len(names) == 2 + 1 + 1  # two scaled numerics, one non-reference category, one flag
    unseen = train.head(1).assign(first_activity="never_seen")
    assert np.isfinite(pipe.predict(unseen)).all()


def test_log_target_round_trip_returns_days():
    days = np.array([0.0, 14.5, 104.6, 1500.0])
    assert np.allclose(np.expm1(np.log1p(days)), days)
    train = toy_frame(60, seed=3)
    y = 20 + 5 * train["event_count_14d"].to_numpy()
    model = M.log_linear_regression_pipeline(NUMERIC, CATEGORICAL, BINARY).fit(train, y)
    inner = model.regressor_.predict(train)  # fitted on the log scale
    assert np.allclose(model.predict(train), np.expm1(inner))
    assert model.predict(train).min() > 0  # back in days, never negative
    assert len(M.linear_regression_coefficients(model)) == 4


def test_regression_metrics_known_values():
    metrics = M.regression_metrics([10, 20, 30, 40], [12, 18, 33, 37])
    assert metrics["mae"] == 2.5
    assert np.isclose(metrics["rmse"], np.sqrt((4 + 4 + 9 + 9) / 4))
    assert np.isclose(metrics["r2"], 1 - 26 / 500)
    assert metrics["mean_residual"] == 0.0
    perfect = M.regression_metrics([1, 2, 3], [1, 2, 3])
    assert perfect["mae"] == 0 and perfect["r2"] == 1


def test_constant_baseline_uses_training_outcomes_only():
    duration = pd.Series([10.0, 20, 30, 40, 1000, 2000])
    split = pd.Series(["train"] * 4 + ["test"] * 2)
    assert M.constant_baseline(duration[split == "train"]) == 25.0
    assert M.constant_baseline(duration[split == "train"], "mean") == 25.0
    # a constant prediction scores R² <= 0 on any period: it explains none of the variation
    assert M.regression_metrics(duration, np.full(6, 25.0))["r2"] < 0


def test_bootstrap_mae_gain_sign():
    y = np.arange(100.0)
    low, high = M.bootstrap_mae_gain(y, y + 1, np.full(100, y.mean()))
    assert 0 < low <= high  # a near-perfect model clearly beats the constant
    low, high = M.bootstrap_mae_gain(y, y, y)
    assert low == high == 0


def test_split_is_deterministic():
    starts = pd.Series([T0 + d * DAY for d in (0, 700, 400, 900, 729)], index=list("abcde"))
    validation, test = T0 + 365 * DAY, T0 + 730 * DAY
    first = M.chronological_split(starts, validation, test)
    assert first.equals(M.chronological_split(starts, validation, test))
    assert first.equals(M.chronological_split(starts.sample(frac=1, random_state=0), validation, test).reindex(starts.index))
    assert first.to_dict() == {"a": "train", "b": "validation", "c": "validation", "d": "test", "e": "validation"}


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"{len(tests)} tests passed")
