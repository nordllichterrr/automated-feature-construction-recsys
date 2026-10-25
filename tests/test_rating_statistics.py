import pandas as pd

from src.feature_construction.features import (
    FeatureConfig,
    build_historical_features,
)


def make_events(ratings):
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-03 10:00:00",
            ],
            "rating": ratings,
            "label": [int(rating >= 4) for rating in ratings],
        }
    )


def test_future_ratings_do_not_change_past_statistics():
    first = make_events([5, 1, 2])
    second = make_events([5, 5, 4])

    config = FeatureConfig(windows_days=(1, 7))
    first_features = build_historical_features(first, config)
    second_features = build_historical_features(second, config)

    columns = [
        "user_mean_rating",
        "item_mean_rating",
        "user_positive_rate",
        "item_positive_rate",
        "user_item_mean_rating",
    ]

    pd.testing.assert_series_equal(
        first_features.loc[1, columns],
        second_features.loc[1, columns],
    )

    pd.testing.assert_series_equal(
        first_features.loc[0, columns],
        second_features.loc[0, columns],
    )


def test_simultaneous_events_do_not_leak_ratings():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
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

    features = build_historical_features(
        events,
        FeatureConfig(windows_days=(1, 7)),
    )

    assert pd.isna(features.loc[0, "user_mean_rating"])
    assert pd.isna(features.loc[1, "user_mean_rating"])
    assert features.loc[2, "user_mean_rating"] == 3.0
    assert features.loc[2, "user_positive_rate"] == 0.5


def test_first_interaction_has_no_rating_history():
    events = make_events([5, 1, 2])
    features = build_historical_features(
        events,
        FeatureConfig(windows_days=(1,)),
    )

    assert pd.isna(features.loc[0, "user_mean_rating"])
    assert pd.isna(features.loc[0, "item_mean_rating"])
    assert pd.isna(features.loc[0, "user_positive_rate"])
    assert pd.isna(features.loc[0, "item_positive_rate"])
