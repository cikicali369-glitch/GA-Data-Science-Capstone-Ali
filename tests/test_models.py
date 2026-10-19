"""Tests for src/models.py and the train-only activity vocabulary helper.

Run with:  .venv\\Scripts\\python.exe tests\\test_models.py   (plain asserts; also pytest-compatible)
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


def test_chronological_split_boundaries():
    starts = pd.Series([T0, T0 + 400 * DAY, T0 + 730 * DAY, T0 + 800 * DAY], index=list("abcd"))
    split = M.chronological_split(starts, T0 + 365 * DAY, T0 + 730 * DAY)
    assert split.to_dict() == {"a": "train", "b": "validation", "c": "test", "d": "test"}


def test_threshold_uses_training_cases_only():
    duration = pd.Series([10.0, 20, 30, 40, 1000, 2000])
    split = pd.Series(["train"] * 4 + ["test"] * 2)
    threshold = M.train_threshold(duration, split)
    assert threshold == duration[:4].quantile(0.75)  # test durations do not move it
    assert M.long_running_label(duration, threshold).tolist() == [0, 0, 0, 1, 1, 1]


def test_follow_up_flag():
    end = T0 + 1000 * DAY
    flag = M.follow_up_flag(pd.Series([T0, end - 364 * DAY, end - 365 * DAY]), end)
    assert flag.tolist() == [True, False, True]


def test_preprocessing_handles_unseen_categories():
    train = pd.DataFrame({"x": [0.0, 1, 2, 3], "c": ["a"] * 4, "b": [0, 1, 0, 1]})
    pipe = M.logistic_pipeline(["x"], ["c"], ["b"]).fit(train, [0, 0, 1, 1])
    unseen = pd.DataFrame({"x": [1.5], "c": ["never_seen"], "b": [1]})
    assert 0 <= pipe.predict_proba(unseen)[0, 1] <= 1


def test_evaluate_and_threshold_rule():
    y = np.array([0, 0, 1, 1])
    metrics = M.evaluate(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert metrics["accuracy"] == 1 and metrics["tp"] == 2 and metrics["roc_auc"] == 1
    # 0.50 already perfect -> no alternative can gain >= 0.05, so 0.50 is kept
    chosen, _ = M.choose_probability_threshold(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert chosen == 0.5
    # positives scored at 0.3 -> 0.50 finds none; a lower threshold clearly helps and is adopted
    chosen, _ = M.choose_probability_threshold(y, np.array([0.05, 0.1, 0.3, 0.35]))
    assert chosen < 0.5


def test_train_vocabulary_ignores_validation_and_test_codes():
    snap = pd.DataFrame({
        F.CASE: ["t1", "t1", "t2", "t3", "v1", "v1"],
        F.ACTIVITY: ["A", "B", "A", "A", "C", "B"],
    })
    features, codes = F.train_vocabulary_activity_features(snap, ["t1", "t2", "t3"], min_share=0.1, max_share=0.9)
    assert codes == ["B"]  # A is in 100% of training cases, C never occurs in training
    assert F.ACTIVITY_FLAG_PREFIX + "C" not in features.columns
    assert features.loc["v1", F.ACTIVITY_FLAG_PREFIX + "B"] == 1
    assert features.loc["v1", "first_activity"] == "C"  # ungrouped raw code


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"{len(tests)} tests passed")
