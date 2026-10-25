import pandas as pd


def prepare_model_data(features, include_history=False):
    data = features.copy()

    data["user_id"] = data["user_id"].astype(str)
    data["item_id"] = data["item_id"].astype(str)

    timestamp = pd.to_datetime(data["timestamp"], utc=True)

    data["hour"] = timestamp.dt.hour.astype("int8")
    data["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    data["month"] = timestamp.dt.month.astype("int8")

    feature_columns = [
        "user_id",
        "item_id",
        "hour",
        "day_of_week",
        "month",
    ]

    if include_history:
        history_columns = [
            column
            for column in data.columns
            if column.startswith(
                (
                    "user_total_interactions",
                    "item_total_interactions",
                    "user_item_total_interactions",
                    "user_interactions_",
                    "item_interactions_",
                    "user_item_interactions_",
                )
            )
        ]
        feature_columns.extend(history_columns)

    X = data[feature_columns].copy()
    categorical_features = ["user_id", "item_id"]
    y = data["label"].astype("int8")

    return X, y, categorical_features
