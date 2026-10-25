from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"user_id", "item_id", "timestamp", "label"}


@dataclass(frozen=True)
class FeatureConfig:
    windows_days: Tuple[int, ...] = (1, 7, 30)
    include_lifetime_counts: bool = True

    def __post_init__(self):
        if not self.windows_days:
            raise ValueError("windows_days не должен быть пустым")
        if any(not isinstance(w, int) or w <= 0 for w in self.windows_days):
            raise ValueError("Все временные окна должны быть положительными целыми числами")
        if len(set(self.windows_days)) != len(self.windows_days):
            raise ValueError("Временные окна не должны повторяться")


def _validate_events(events):
    missing = REQUIRED_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError("Отсутствуют обязательные столбцы: {}".format(sorted(missing)))
    if events.empty:
        raise ValueError("Таблица событий не должна быть пустой")

    df = events.copy()

    if df[["user_id", "item_id", "timestamp"]].isna().any().any():
        raise ValueError("user_id, item_id и timestamp не должны содержать пропуски")
    if df["label"].isna().any():
        raise ValueError("label не должен содержать пропуски")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df["_original_position"] = np.arange(len(df))

    return df


def build_historical_features(events, config=FeatureConfig()):
    original_columns = list(events.columns)
    df = _validate_events(events)
    df = df.sort_values(
        ["timestamp", "_original_position"], kind="mergesort"
    ).reset_index(drop=True)

    n = len(df)
    users = df["user_id"].to_numpy()
    items = df["item_id"].to_numpy()
    times = df["timestamp"].astype("int64").to_numpy()

    batch_starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    batch_ends = np.r_[batch_starts[1:], n]

    user_counts = {}
    item_counts = {}
    pair_counts = {}

    user_lifetime = np.zeros(n, dtype=np.int64)
    item_lifetime = np.zeros(n, dtype=np.int64)
    pair_lifetime = np.zeros(n, dtype=np.int64)

    for start, end in zip(batch_starts, batch_ends):
        for i in range(start, end):
            user = users[i]
            item = items[i]
            pair = (user, item)

            user_lifetime[i] = user_counts.get(user, 0)
            item_lifetime[i] = item_counts.get(item, 0)
            pair_lifetime[i] = pair_counts.get(pair, 0)

        # События с одинаковым временем добавляем в историю только после расчёта признаков всей группы.
        for i in range(start, end):
            user = users[i]
            item = items[i]
            pair = (user, item)

            user_counts[user] = user_counts.get(user, 0) + 1
            item_counts[item] = item_counts.get(item, 0) + 1
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    output = pd.DataFrame(index=np.arange(n))

    for column in original_columns:
        output[column] = df[column].to_numpy()

    if config.include_lifetime_counts:
        output["user_total_interactions"] = user_lifetime
        output["item_total_interactions"] = item_lifetime
        output["user_item_total_interactions"] = pair_lifetime

    user_positions = {}
    item_positions = {}
    pair_positions = {}

    for i, (user, item) in enumerate(zip(users, items)):
        user_positions.setdefault(user, []).append(i)
        item_positions.setdefault(item, []).append(i)
        pair_positions.setdefault((user, item), []).append(i)

    user_times = {
        key: times[np.asarray(positions, dtype=np.int64)]
        for key, positions in user_positions.items()
    }
    item_times = {
        key: times[np.asarray(positions, dtype=np.int64)]
        for key, positions in item_positions.items()
    }
    pair_times = {
        key: times[np.asarray(positions, dtype=np.int64)]
        for key, positions in pair_positions.items()
    }

    for window in config.windows_days:
        delta = int(pd.Timedelta(days=window).value)
        user_feature = np.zeros(n, dtype=np.int64)
        item_feature = np.zeros(n, dtype=np.int64)
        pair_feature = np.zeros(n, dtype=np.int64)

        for i, (user, item, timestamp) in enumerate(zip(users, items, times)):
            lower_bound = timestamp - delta

            history_times = user_times[user]
            left = np.searchsorted(history_times, lower_bound, side="left")
            right = np.searchsorted(history_times, timestamp, side="left")
            user_feature[i] = right - left

            history_times = item_times[item]
            left = np.searchsorted(history_times, lower_bound, side="left")
            right = np.searchsorted(history_times, timestamp, side="left")
            item_feature[i] = right - left

            history_times = pair_times[(user, item)]
            left = np.searchsorted(history_times, lower_bound, side="left")
            right = np.searchsorted(history_times, timestamp, side="left")
            pair_feature[i] = right - left

        output["user_interactions_{}d".format(window)] = user_feature
        output["item_interactions_{}d".format(window)] = item_feature
        output["user_item_interactions_{}d".format(window)] = pair_feature

    output["_original_position"] = df["_original_position"].to_numpy()
    output = output.sort_values("_original_position", kind="mergesort")
    output = output.drop(columns=["_original_position"])
    output.index = events.index

    return output
