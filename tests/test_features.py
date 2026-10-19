"""Tests for src/features.py on small synthetic event logs.

Run with:  .venv\\Scripts\\python.exe tests\\test_features.py   (plain asserts; also pytest-compatible)
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import features as F  # noqa: E402

START = pd.Timestamp("2014-03-02 23:00", tz="UTC")  # local midnight, like the real log


def make_log(rows):
    """rows: (case, day_offset, activity, resource, has_end_date)."""
    return pd.DataFrame({
        F.CASE: [r[0] for r in rows],
        F.TIMESTAMP: [START + pd.Timedelta(days=r[1]) for r in rows],
        F.ACTIVITY: [r[2] for r in rows],
        F.RESOURCE: [r[3] for r in rows],
        F.END_DATE: [START + pd.Timedelta(days=100) if r[4] else pd.NaT for r in rows],
    })


BASE = [
    # case c1: repeats, resource switches, same-day events, events after day 14
    ("c1", 0, "A", "r1", True), ("c1", 0, "A", "r1", True), ("c1", 2, "A", "r2", True),
    ("c1", 9, "B", "r1", True), ("c1", 20, "C", "r3", True),
    # case c2: single event before the snapshot
    ("c2", 0, "A", "r1", True), ("c2", 30, "B", "r1", True),
    # case c3: ends exactly at the snapshot -> not prediction-eligible
    ("c3", 0, "A", "r1", True), ("c3", 14, "B", "r1", True),
    # case c4: no end date -> not outcome-eligible
    ("c4", 0, "A", "r1", False), ("c4", 40, "B", "r1", False),
]


def build(rows=BASE, codes=None):
    table, _ = F.build_feature_table(make_log(rows), activity_codes=codes or ["A", "B"])
    return table.set_index("case_id")


def test_eligibility():
    timing = F.case_timing(make_log(BASE))
    assert timing["outcome_eligible"].to_dict() == {"c1": True, "c2": True, "c3": True, "c4": False}
    assert timing["prediction_eligible"].to_dict() == {"c1": True, "c2": True, "c3": False, "c4": False}
    assert sorted(build().index) == ["c1", "c2"]


def test_counts_repeats_and_switches():
    c1 = build().loc["c1"]
    assert c1["event_count_14d"] == 4  # the day-20 event is excluded
    assert c1["unique_activity_count_14d"] == 2
    assert c1["repeated_event_count_14d"] == 2
    assert c1["repeated_activity_count_14d"] == 1
    assert c1["activities_repeated_more_than_once_14d"] == 1  # A occurs 3 times
    assert c1["resource_switch_count_14d"] == 2  # r1 r1 r2 r1
    assert c1["unique_resource_count_14d"] == 2
    assert c1["active_days_count_14d"] == 3
    assert c1["events_first_3_days"] == 3 and c1["events_first_7_days"] == 3 and c1["events_days_8_to_14"] == 1
    assert c1["max_inter_event_days"] == 7 and c1["days_since_last_event"] == 5
    assert c1["latest_activity_at_snapshot"] == "OTHER"  # rare categories are grouped (min 20 cases)


def test_single_event_case():
    c2 = build().loc["c2"]
    assert c2["event_count_14d"] == 1
    assert c2["mean_inter_event_days"] == 0 and c2["max_inter_event_days"] == 0
    assert c2["events_per_active_day"] == 1 and c2["resource_switch_count_14d"] == 0
    assert c2["days_since_last_event"] == 14
    assert c2[F.ACTIVITY_FLAG_PREFIX + "A"] == 1 and c2[F.ACTIVITY_FLAG_PREFIX + "B"] == 0  # B happens on day 30


def test_future_events_do_not_change_predictors():
    altered = [r if r[1] <= 14 else (r[0], r[1], "ZZZ", "r9", r[4]) for r in BASE]
    original, changed = build(), build(altered)
    predictors = F.predictor_columns(original.reset_index())
    pd.testing.assert_frame_equal(original[predictors], changed[predictors])


def test_workload_uses_only_past_events():
    rows = BASE + [("c5", -10, "A", "r1", True), ("c5", 13, "A", "r1", True), ("c6", 15, "A", "r1", True)]
    table = build(rows)
    # c1 snapshot = day 14: c2, c3, c4 (day 0) and c5 (day -10, 13) are inside (day -16, day 14]; c6 (day 15) is in the future
    assert table.loc["c1", "workload_cases_active_prev_30d"] == 4


def test_activity_selection_is_frequency_only():
    snap = pd.DataFrame({F.CASE: ["a", "b", "c", "a"], F.ACTIVITY: ["X", "X", "X", "Y"]})
    assert F.select_activity_codes(snap, min_share=0.1, max_share=0.9) == ["Y"]  # X is in 100% of cases


def test_csv_roundtrip():
    table = build().reset_index()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "features.csv"
        F.save_features_csv(table, path)
        loaded = F.load_features_csv(path)
    assert loaded["case_start"].equals(table["case_start"].astype(loaded["case_start"].dtype))
    assert str(loaded["first_activity"].dtype) == "category"
    assert loaded.shape == table.shape


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"{len(tests)} tests passed")
