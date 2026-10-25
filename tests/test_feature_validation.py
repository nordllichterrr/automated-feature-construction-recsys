import pandas as pd
import pytest

from src.feature_construction.features import build_historical_features


@pytest.fixture
def valid_events():
    return pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "timestamp": ["2026-01-01 10:00:00", "2026-01-02 10:00:00"],
        "label": [1, 0],
        "rating": [5, 3],
    })


@pytest.mark.parametrize("invalid_label", [2, -1, 0.5, "positive"])
def test_invalid_labels_are_rejected(valid_events, invalid_label):
    valid_events["label"] = valid_events["label"].astype(object)
    valid_events.loc[0, "label"] = invalid_label

    with pytest.raises(ValueError, match="label"):
        build_historical_features(valid_events)


@pytest.mark.parametrize(
    "invalid_rating",
    [float("inf"), float("-inf"), float("nan")],
)
def test_non_finite_ratings_are_rejected(valid_events, invalid_rating):
    valid_events["rating"] = valid_events["rating"].astype(float)
    valid_events.loc[0, "rating"] = invalid_rating

    with pytest.raises(ValueError, match="rating"):
        build_historical_features(valid_events)


def test_non_numeric_ratings_are_rejected(valid_events):
    valid_events["rating"] = valid_events["rating"].astype(object)
    valid_events.loc[0, "rating"] = "unknown"

    with pytest.raises((ValueError, TypeError)):
        build_historical_features(valid_events)


def test_missing_labels_are_rejected(valid_events):
    valid_events.loc[0, "label"] = None

    with pytest.raises(ValueError, match="label"):
        build_historical_features(valid_events)


def test_missing_ratings_are_rejected(valid_events):
    valid_events["rating"] = valid_events["rating"].astype(float)
    valid_events.loc[0, "rating"] = float("nan")

    with pytest.raises(ValueError, match="rating"):
        build_historical_features(valid_events)
