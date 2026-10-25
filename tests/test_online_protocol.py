import pandas as pd

from src.feature_construction.features import (
    FeatureConfig,
    build_historical_features,
)


def test_unsorted_input_preserves_original_event_order():
    events = pd.DataFrame(
        {
            "event_id": ["later", "first", "middle"],
            "user_id": [1, 1, 1],
            "item_id": [20, 10, 30],
            "timestamp": [
                "2026-01-03 10:00:00",
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
            ],
            "rating": [5, 4, 2],
            "label": [1, 1, 0],
        }
    )

    result = build_historical_features(
        events,
        FeatureConfig(windows_days=(1, 7)),
    )

    assert result["event_id"].tolist() == ["later", "first", "middle"]

    # Событие first — первое по времени, поэтому история пользователя пуста.
    assert result.loc[1, "user_total_interactions"] == 0

    # Событие middle учитывает только событие first.
    assert result.loc[2, "user_total_interactions"] == 1

    # Событие later учитывает оба более ранних события.
    assert result.loc[0, "user_total_interactions"] == 2


def test_repeated_user_item_interactions_use_only_past_events():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [10, 10, 10],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-03 10:00:00",
            ],
            "rating": [5, 1, 4],
            "label": [1, 0, 1],
        }
    )

    result = build_historical_features(
        events,
        FeatureConfig(windows_days=(1, 7)),
    )

    assert result["user_item_total_interactions"].tolist() == [0, 1, 2]

    assert pd.isna(result.loc[0, "user_item_mean_rating"])
    assert result.loc[1, "user_item_mean_rating"] == 5.0
    assert result.loc[2, "user_item_mean_rating"] == 3.0

    assert result["user_item_interactions_1d"].tolist() == [0, 1, 1]
    assert result["user_item_interactions_7d"].tolist() == [0, 1, 2]


def test_permuting_simultaneous_events_does_not_change_their_features():
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
            ],
            "rating": [5, 1, 4],
            "label": [1, 0, 1],
        }
    )

    permuted = events.iloc[[1, 0, 2]].reset_index(drop=True)

    config = FeatureConfig(windows_days=(1, 7))
    original_features = build_historical_features(events, config)
    permuted_features = build_historical_features(permuted, config)

    feature_columns = [
        "user_total_interactions",
        "item_total_interactions",
        "user_item_total_interactions",
        "user_mean_rating",
        "item_mean_rating",
        "user_positive_rate",
        "item_positive_rate",
        "user_item_mean_rating",
        "user_interactions_1d",
        "item_interactions_1d",
        "user_item_interactions_1d",
        "user_interactions_7d",
        "item_interactions_7d",
        "user_item_interactions_7d",
    ]

    original_by_event = (
        original_features.set_index("event_id")[feature_columns].sort_index()
    )
    permuted_by_event = (
        permuted_features.set_index("event_id")[feature_columns].sort_index()
    )

    pd.testing.assert_frame_equal(
        original_by_event,
        permuted_by_event,
    )
