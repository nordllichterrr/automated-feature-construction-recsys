import pandas as pd

from src.feature_construction.features import (
    FeatureConfig,
    build_historical_features,
)


def test_future_events_do_not_affect_past_features():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [10, 10, 20],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-10 10:00:00",
            ],
            "label": [1, 0, 1],
        }
    )

    result = build_historical_features(
        events,
        FeatureConfig(windows_days=(1, 7)),
    )

    assert result.loc[0, "user_total_interactions"] == 0
    assert result.loc[1, "user_total_interactions"] == 1
    assert result.loc[1, "user_item_total_interactions"] == 1
    assert result.loc[2, "user_total_interactions"] == 2


def test_simultaneous_events_do_not_leak_into_each_other():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [10, 20, 10],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
            ],
            "label": [1, 0, 1],
        }
    )

    result = build_historical_features(
        events,
        FeatureConfig(windows_days=(1, 7)),
    )

    assert result.loc[0, "user_total_interactions"] == 0
    assert result.loc[1, "user_total_interactions"] == 0
    assert result.loc[2, "user_total_interactions"] == 2
    assert result.loc[2, "user_item_total_interactions"] == 1


def test_window_boundary_is_inclusive():
    events = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 10],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
            ],
            "label": [1, 0],
        }
    )

    result = build_historical_features(
        events,
        FeatureConfig(windows_days=(1,)),
    )

    assert result.loc[1, "user_interactions_1d"] == 1
    assert result.loc[1, "item_interactions_1d"] == 1
    assert result.loc[1, "user_item_interactions_1d"] == 1


def test_window_excludes_events_older_than_lower_boundary():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [10, 10, 10],
            "timestamp": [
                "2026-01-01 09:59:59.999999",
                "2026-01-01 10:00:00.000000",
                "2026-01-02 10:00:00.000000",
            ],
            "label": [1, 0, 1],
        }
    )

    result = build_historical_features(
        events,
        FeatureConfig(windows_days=(1,)),
    )

    assert result.loc[2, "user_interactions_1d"] == 1
    assert result.loc[2, "item_interactions_1d"] == 1
    assert result.loc[2, "user_item_interactions_1d"] == 1


def test_simultaneous_events_are_excluded_from_rolling_windows():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [10, 20, 10],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
            ],
            "label": [1, 0, 1],
        }
    )

    result = build_historical_features(
        events,
        FeatureConfig(windows_days=(1,)),
    )

    assert result.loc[0, "user_interactions_1d"] == 0
    assert result.loc[1, "user_interactions_1d"] == 0
    assert result.loc[2, "user_interactions_1d"] == 2
    assert result.loc[2, "item_interactions_1d"] == 1
    assert result.loc[2, "user_item_interactions_1d"] == 1
