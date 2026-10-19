"""Point-in-time (snapshot) feature engineering for early prediction of long-running cases.

Every predictor is computed only from events whose timestamp is at or before the case's snapshot time
(first event + ``snapshot_days``). Outcome information (final event time, final duration) is kept in separate
metadata columns and is never used to build predictors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CASE = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
RESOURCE = "org:resource"
END_DATE = "case:endDate"
EVENT_ORDER = "event_order"

SNAPSHOT_DAYS = 14
LOCAL_TIMEZONE = "Europe/Amsterdam"
SECONDS_PER_DAY = 24 * 60 * 60

# Activity-flag selection: codes observed by the snapshot in at least / at most this share of cases.
# Frequency-only rule (no outcome information): drops near-universal codes (no variation) and rare codes (unstable).
ACTIVITY_MIN_CASE_SHARE = 0.10
ACTIVITY_MAX_CASE_SHARE = 0.90
# Categorical values seen in fewer cases than this are grouped into "OTHER".
MIN_CATEGORY_CASES = 20
OTHER_CATEGORY = "OTHER"
# Look-back window for the workload-context feature.
WORKLOAD_WINDOW_DAYS = 30

METADATA_COLUMNS = ["case_id", "case_start", "snapshot_time", "case_end", "event_log_days", "elapsed_days"]
TIMESTAMP_METADATA = ["case_start", "snapshot_time", "case_end"]
CATEGORICAL_FEATURES = ["first_activity", "latest_activity_at_snapshot"]
CORE_NUMERIC_FEATURES = [
    "event_count_14d", "unique_activity_count_14d", "unique_resource_count_14d",
    "repeated_event_count_14d", "repeated_activity_count_14d", "activities_repeated_more_than_once_14d",
    "days_since_last_event", "mean_inter_event_days", "median_inter_event_days", "max_inter_event_days",
    "events_first_3_days", "events_first_7_days", "events_days_8_to_14",
    "active_days_count_14d", "events_per_active_day",
    "resource_switch_count_14d",
    "workload_cases_active_prev_30d",
    "start_month", "start_weekday",
]
ACTIVITY_FLAG_PREFIX = "has_act_"


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / SECONDS_PER_DAY


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the columns needed for snapshot features and sort deterministically.

    Order = case, timestamp, then original source order (``event_order``) for same-timestamp ties.
    Timestamps in this log are mostly date-level, so same-day order follows the source file.
    """
    events = df[[CASE, ACTIVITY, TIMESTAMP, RESOURCE]].copy()
    events[EVENT_ORDER] = np.arange(len(df)) if EVENT_ORDER not in df.columns else df[EVENT_ORDER].to_numpy()
    return events.sort_values([CASE, TIMESTAMP, EVENT_ORDER], kind="mergesort").reset_index(drop=True)


def case_timing(df: pd.DataFrame, snapshot_days: int = SNAPSHOT_DAYS) -> pd.DataFrame:
    """One row per case with timing, outcome duration and eligibility flags.

    outcome_eligible: case:endDate present (closure recorded).
    prediction_eligible: outcome_eligible and logged activity strictly after the snapshot time.
    """
    grouped = df.groupby(CASE, sort=True)
    timing = pd.DataFrame({
        "case_start": grouped[TIMESTAMP].min(),
        "case_end": grouped[TIMESTAMP].max(),
        "has_end_date": grouped[END_DATE].first().notna(),
    })
    timing["event_log_days"] = _days(timing["case_end"] - timing["case_start"])
    timing["snapshot_time"] = timing["case_start"] + pd.Timedelta(days=snapshot_days)
    timing["elapsed_days"] = float(snapshot_days)
    timing["outcome_eligible"] = timing["has_end_date"]
    timing["prediction_eligible"] = timing["outcome_eligible"] & (timing["case_end"] > timing["snapshot_time"])
    timing.index.name = "case_id"
    return timing


def snapshot_events(events: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    """Events of prediction-eligible cases with timestamp <= that case's snapshot time."""
    eligible = timing.index[timing["prediction_eligible"]]
    snap = events[events[CASE].isin(eligible)].copy()
    snap["case_start"] = snap[CASE].map(timing["case_start"])
    snap["snapshot_time"] = snap[CASE].map(timing["snapshot_time"])
    snap = snap[snap[TIMESTAMP] <= snap["snapshot_time"]]
    snap["elapsed_days_at_event"] = _days(snap[TIMESTAMP] - snap["case_start"])
    return snap.reset_index(drop=True)


def core_features(snap: pd.DataFrame) -> pd.DataFrame:
    """Counts, repetition, timing, activity-rhythm and resource-handoff features from snapshot events."""
    grouped = snap.groupby(CASE, sort=True)
    features = pd.DataFrame({
        "event_count_14d": grouped.size(),
        "unique_activity_count_14d": grouped[ACTIVITY].nunique(),
        "unique_resource_count_14d": grouped[RESOURCE].nunique(),
    })

    occurrences = snap.groupby([CASE, ACTIVITY], sort=True).size()
    features["repeated_event_count_14d"] = features["event_count_14d"] - features["unique_activity_count_14d"]
    features["repeated_activity_count_14d"] = (occurrences > 1).groupby(level=CASE).sum()
    features["activities_repeated_more_than_once_14d"] = (occurrences > 2).groupby(level=CASE).sum()

    last_time = grouped[TIMESTAMP].max()
    snapshot_time = grouped["snapshot_time"].first()
    features["days_since_last_event"] = _days(snapshot_time - last_time)

    # Gaps between consecutive events; a case with a single event has no gaps -> 0.0 (no observed waiting between events).
    gaps = _days(grouped[TIMESTAMP].diff()).dropna()
    gap_stats = gaps.groupby(snap.loc[gaps.index, CASE]).agg(["mean", "median", "max"])
    for stat in ["mean", "median", "max"]:
        features[f"{stat}_inter_event_days"] = gap_stats[stat].reindex(features.index).fillna(0.0)

    # Day index 0 = calendar day of the first event (timestamps are mostly date-level).
    day_index = np.floor(snap["elapsed_days_at_event"])
    case_ids = snap[CASE]
    features["events_first_3_days"] = (day_index < 3).groupby(case_ids).sum()
    features["events_first_7_days"] = (day_index < 7).groupby(case_ids).sum()
    features["events_days_8_to_14"] = (day_index >= 7).groupby(case_ids).sum()

    local_date = snap[TIMESTAMP].dt.tz_convert(LOCAL_TIMEZONE).dt.date
    features["active_days_count_14d"] = local_date.groupby(case_ids).nunique()
    features["events_per_active_day"] = features["event_count_14d"] / features["active_days_count_14d"]

    # Resource switch = consecutive events whose (non-null) resources differ. Describes hand-overs, not performance.
    previous_resource = grouped[RESOURCE].shift()
    switched = snap[RESOURCE].notna() & previous_resource.notna() & snap[RESOURCE].ne(previous_resource)
    features["resource_switch_count_14d"] = switched.groupby(case_ids).sum()

    start_local = grouped["case_start"].first().dt.tz_convert(LOCAL_TIMEZONE)
    features["start_month"] = start_local.dt.month
    features["start_weekday"] = start_local.dt.weekday
    features.index.name = "case_id"
    return features


def workload_feature(events: pd.DataFrame, timing: pd.DataFrame, window_days: int = WORKLOAD_WINDOW_DAYS) -> pd.Series:
    """Number of OTHER cases with at least one logged event in (snapshot - window, snapshot].

    Uses only events at or before the snapshot time, so it is known at prediction time.
    Underestimates workload during the first weeks of the log (log starts October 2010).
    """
    def naive_utc(values):
        return values.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()

    eligible = timing[timing["prediction_eligible"]]
    times = naive_utc(events[TIMESTAMP])
    cases = events[CASE].to_numpy()
    snapshots = naive_utc(eligible["snapshot_time"])
    window = np.timedelta64(window_days, "D")
    counts = {}
    for case_id, snapshot in zip(eligible.index, snapshots):
        in_window = (times > snapshot - window) & (times <= snapshot) & (cases != case_id)
        counts[case_id] = len(np.unique(cases[in_window]))
    return pd.Series(counts, name="workload_cases_active_prev_30d", dtype="int64").rename_axis("case_id")


def select_activity_codes(snap: pd.DataFrame, min_share: float = ACTIVITY_MIN_CASE_SHARE,
                          max_share: float = ACTIVITY_MAX_CASE_SHARE) -> list[str]:
    """Activity codes seen by the snapshot in a share of cases within [min_share, max_share].

    Frequency-only and outcome-agnostic. The modelling step can re-run it on training cases only.
    """
    share = snap.groupby(ACTIVITY)[CASE].nunique() / snap[CASE].nunique()
    selected = share[(share >= min_share) & (share <= max_share)]
    return sorted(selected.index)


def activity_features(snap: pd.DataFrame, codes: list[str], min_category_cases: int = MIN_CATEGORY_CASES) -> pd.DataFrame:
    """First/latest activity code at the snapshot (rare values -> OTHER) and 0/1 flags for the selected codes."""
    grouped = snap.groupby(CASE, sort=True)[ACTIVITY]
    features = pd.DataFrame({
        "first_activity": grouped.first(),
        "latest_activity_at_snapshot": grouped.last(),
    })
    for column in CATEGORICAL_FEATURES:
        counts = features[column].value_counts()
        rare = counts.index[counts < min_category_cases]
        features[column] = features[column].where(~features[column].isin(rare), OTHER_CATEGORY)

    present = pd.crosstab(snap[CASE], snap[ACTIVITY]).reindex(columns=codes, fill_value=0) > 0
    flags = present.astype("int8").add_prefix(ACTIVITY_FLAG_PREFIX)
    features = features.join(flags)
    features.index.name = "case_id"
    return features


def train_vocabulary_activity_features(snap: pd.DataFrame, train_case_ids, min_share: float = ACTIVITY_MIN_CASE_SHARE,
                                       max_share: float = ACTIVITY_MAX_CASE_SHARE) -> tuple[pd.DataFrame, list[str]]:
    """Activity features with the code vocabulary selected on TRAINING cases only, then applied to every case.

    Categorical first/latest activity codes are returned ungrouped (min_category_cases=0) so that rare-value grouping
    can be fitted on the training data by the model's encoder instead of on the full dataset.
    """
    train_snap = snap[snap[CASE].isin(set(train_case_ids))]
    codes = select_activity_codes(train_snap, min_share, max_share)
    return activity_features(snap, codes, min_category_cases=0), codes


def build_feature_table(df: pd.DataFrame, snapshot_days: int = SNAPSHOT_DAYS,
                        activity_codes: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Full pipeline: one row per prediction-eligible case, metadata columns first, then predictors."""
    events = prepare_events(df)
    timing = case_timing(df, snapshot_days)
    snap = snapshot_events(events, timing)
    if activity_codes is None:
        activity_codes = select_activity_codes(snap)

    table = (
        timing.loc[timing["prediction_eligible"], METADATA_COLUMNS[1:]]
        .join(core_features(snap))
        .join(workload_feature(events, timing))
        .join(activity_features(snap, activity_codes))
    )
    table = table.reset_index()
    predictors = predictor_columns(table)
    return table[METADATA_COLUMNS + predictors], activity_codes


def predictor_columns(table: pd.DataFrame) -> list[str]:
    flags = sorted(c for c in table.columns if c.startswith(ACTIVITY_FLAG_PREFIX))
    return CORE_NUMERIC_FEATURES + flags + CATEGORICAL_FEATURES


def save_features_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")


def load_features_csv(path: Path) -> pd.DataFrame:
    """Read the CSV back with the original dtypes (UTC timestamps, string case_id, categorical columns)."""
    table = pd.read_csv(path, dtype={"case_id": str})
    for column in TIMESTAMP_METADATA:
        table[column] = pd.to_datetime(table[column], utc=True, format="ISO8601")
    for column in CATEGORICAL_FEATURES:
        table[column] = table[column].astype("category")
    return table
