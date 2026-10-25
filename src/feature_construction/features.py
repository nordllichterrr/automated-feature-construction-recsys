from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"user_id", "item_id", "timestamp", "label"}


@dataclass(frozen=True)
class FeatureConfig:
    windows_days: Tuple[int, ...] = (1, 7, 30)
    include_lifetime_counts: bool = True
    include_rating_statistics: bool = True

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
        raise ValueError(
            "Отсутствуют обязательные столбцы: {}".format(sorted(missing))
        )
    if events.empty:
        raise ValueError("Таблица событий не должна быть пустой")

    df = events.copy()

    if df[["user_id", "item_id", "timestamp"]].isna().any().any():
        raise ValueError("user_id, item_id и timestamp не должны содержать пропуски")
    if df["label"].isna().any():
        raise ValueError("label не должен содержать пропуски")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("label должен содержать только значения 0 и 1")

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="raise")

        if df["rating"].isna().any():
            raise ValueError("rating не должен содержать пропуски")

        if not np.isfinite(df["rating"].to_numpy(dtype=np.float64)).all():
            raise ValueError("rating должен содержать только конечные числа")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="raise"
    )
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
    has_rating = "rating" in df.columns
    include_rating_statistics = config.include_rating_statistics and has_rating
    ratings = (
        df["rating"].to_numpy(dtype=np.float64)
        if has_rating else np.zeros(n, dtype=np.float64)
    )
    labels = df["label"].to_numpy(dtype=np.float64)

    batch_starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    batch_ends = np.r_[batch_starts[1:], n]

    user_counts = {}
    item_counts = {}
    pair_counts = {}

    user_rating_sums = {}
    item_rating_sums = {}
    pair_rating_sums = {}

    user_positive_sums = {}
    item_positive_sums = {}

    user_lifetime = np.zeros(n, dtype=np.int64)
    item_lifetime = np.zeros(n, dtype=np.int64)
    pair_lifetime = np.zeros(n, dtype=np.int64)

    user_mean_rating = np.full(n, np.nan)
    item_mean_rating = np.full(n, np.nan)
    pair_mean_rating = np.full(n, np.nan)
    user_positive_rate = np.full(n, np.nan)
    item_positive_rate = np.full(n, np.nan)

    for start, end in zip(batch_starts, batch_ends):
        for i in range(start, end):
            user = users[i]
            item = items[i]
            pair = (user, item)

            user_count = user_counts.get(user, 0)
            item_count = item_counts.get(item, 0)
            pair_count = pair_counts.get(pair, 0)

            user_lifetime[i] = user_count
            item_lifetime[i] = item_count
            pair_lifetime[i] = pair_count

            if include_rating_statistics:
                if user_count:
                    user_mean_rating[i] = (
                        user_rating_sums[user] / user_count
                    )
                    user_positive_rate[i] = (
                        user_positive_sums[user] / user_count
                    )

                if item_count:
                    item_mean_rating[i] = (
                        item_rating_sums[item] / item_count
                    )
                    item_positive_rate[i] = (
                        item_positive_sums[item] / item_count
                    )

                if pair_count:
                    pair_mean_rating[i] = (
                        pair_rating_sums[pair] / pair_count
                    )

        # Одинаковые временные метки обрабатываются одной группой, чтобы события не раскрывали друг другу оценки.
        for i in range(start, end):
            user = users[i]
            item = items[i]
            pair = (user, item)
            user_counts[user] = user_counts.get(user, 0) + 1
            item_counts[item] = item_counts.get(item, 0) + 1
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if include_rating_statistics:
                rating = ratings[i]
                positive = labels[i]
                user_rating_sums[user] = user_rating_sums.get(user, 0.0) + rating
                item_rating_sums[item] = item_rating_sums.get(item, 0.0) + rating
                pair_rating_sums[pair] = pair_rating_sums.get(pair, 0.0) + rating
                user_positive_sums[user] = (
                    user_positive_sums.get(user, 0.0) + positive
                )
                item_positive_sums[item] = (
                    item_positive_sums.get(item, 0.0) + positive
                )

    output = pd.DataFrame(index=np.arange(n))

    for column in original_columns:
        output[column] = df[column].to_numpy()

    if config.include_lifetime_counts:
        output["user_total_interactions"] = user_lifetime
        output["item_total_interactions"] = item_lifetime
        output["user_item_total_interactions"] = pair_lifetime

    if include_rating_statistics:
        output["user_mean_rating"] = user_mean_rating
        output["item_mean_rating"] = item_mean_rating
        output["user_positive_rate"] = user_positive_rate
        output["item_positive_rate"] = item_positive_rate
        output["user_item_mean_rating"] = pair_mean_rating

    user_positions = {}
    item_positions = {}

    for i, (user, item) in enumerate(zip(users, items)):
        user_positions.setdefault(user, []).append(i)
        item_positions.setdefault(item, []).append(i)

    user_groups = [
        np.asarray(positions, dtype=np.int64)
        for positions in user_positions.values()
    ]
    item_groups = [
        np.asarray(positions, dtype=np.int64)
        for positions in item_positions.values()
    ]

    pair_codes, _ = pd.factorize(
        pd.MultiIndex.from_arrays([users, items]),
        sort=False,
    )
    pair_counts_for_windows = np.bincount(pair_codes)
    repeated_pair_positions = np.flatnonzero(
        pair_counts_for_windows[pair_codes] > 1
    )

    if repeated_pair_positions.size:
        repeated_codes = pair_codes[repeated_pair_positions]
        repeated_order = np.argsort(repeated_codes, kind="mergesort")
        repeated_positions = repeated_pair_positions[repeated_order]
        sorted_repeated_codes = repeated_codes[repeated_order]

        group_starts = np.r_[
            0,
            np.flatnonzero(
                sorted_repeated_codes[1:] != sorted_repeated_codes[:-1]
            ) + 1,
        ]
        group_ends = np.r_[group_starts[1:], len(repeated_positions)]

        repeated_pair_groups = [
            repeated_positions[group_start:group_end]
            for group_start, group_end in zip(group_starts, group_ends)
        ]
    else:
        repeated_pair_groups = []

    def fill_window_counts(groups, delta, result):
        for positions in groups:
            history_times = times[positions]
            left = np.searchsorted(
                history_times, history_times - delta, side="left"
            )
            right = np.searchsorted(
                history_times, history_times, side="left"
            )
            result[positions] = right - left

    for window in config.windows_days:
        delta = int(pd.Timedelta(days=window).value)
        user_feature = np.zeros(n, dtype=np.int64)
        item_feature = np.zeros(n, dtype=np.int64)
        pair_feature = np.zeros(n, dtype=np.int64)

        fill_window_counts(user_groups, delta, user_feature)
        fill_window_counts(item_groups, delta, item_feature)
        fill_window_counts(repeated_pair_groups, delta, pair_feature)

        output["user_interactions_{}d".format(window)] = user_feature
        output["item_interactions_{}d".format(window)] = item_feature
        output["user_item_interactions_{}d".format(window)] = pair_feature

    output["_original_position"] = df["_original_position"].to_numpy()
    output = output.sort_values("_original_position", kind="mergesort")
    output = output.drop(columns=["_original_position"])
    output.index = events.index

    return output
