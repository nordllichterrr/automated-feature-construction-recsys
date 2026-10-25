import numpy as np


def temporal_split(events, train_ratio=0.70, validation_ratio=0.20):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio должен быть между 0 и 1")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio должен быть между 0 и 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Для test должна оставаться ненулевая доля данных")

    ordered = events.sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)

    timestamps = ordered["timestamp"]
    n = len(ordered)

    train_target = int(n * train_ratio)
    validation_target = int(n * (train_ratio + validation_ratio))

    boundaries = np.flatnonzero(
        timestamps.iloc[1:].to_numpy() != timestamps.iloc[:-1].to_numpy()
    ) + 1

    if len(boundaries) < 2:
        raise ValueError("Недостаточно уникальных временных групп")

    train_end = int(boundaries[np.argmin(np.abs(boundaries - train_target))])
    validation_candidates = boundaries[boundaries > train_end]

    if len(validation_candidates) == 0:
        raise ValueError("Не удалось определить границу validation/test")

    validation_end = int(
        validation_candidates[
            np.argmin(np.abs(validation_candidates - validation_target))
        ]
    )

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Одно из временных подмножеств оказалось пустым")

    return train, validation, test
