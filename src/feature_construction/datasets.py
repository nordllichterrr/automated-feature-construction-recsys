import pandas as pd


BASE_FEATURES = [
    "user_id",
    "item_id",
    "hour",
    "day_of_week",
    "month",
]

HISTORY_PREFIXES = (
    "user_total_interactions",
    "item_total_interactions",
    "user_item_total_interactions",
    "user_interactions_",
    "item_interactions_",
    "user_item_interactions_",
)

RATING_STATISTICS = [
    "user_mean_rating",
    "item_mean_rating",
    "user_positive_rate",
    "item_positive_rate",
]


def prepare_model_data(
    features,
    include_history=False,
    include_rating_statistics=False,
):
    data = features.copy()

    data["user_id"] = data["user_id"].astype(str)
    data["item_id"] = data["item_id"].astype(str)

    timestamp = pd.to_datetime(data["timestamp"], utc=True)
    data["hour"] = timestamp.dt.hour.astype("int8")
    data["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    data["month"] = timestamp.dt.month.astype("int8")

    feature_columns = BASE_FEATURES.copy()

    if include_history:
        history_columns = [
            column
            for column in data.columns
            if column.startswith(HISTORY_PREFIXES)
        ]
        feature_columns.extend(history_columns)

    if include_rating_statistics:
        available_statistics = [
            column
            for column in RATING_STATISTICS
            if column in data.columns
        ]
        feature_columns.extend(available_statistics)

    X = data[feature_columns].copy()
    categorical_features = ["user_id", "item_id"]
    y = data["label"].astype("int8")

    return X, y, categorical_features
