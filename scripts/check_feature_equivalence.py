import numpy as np
import pandas as pd

from src.feature_construction.features import (
    FeatureConfig,
    build_historical_features,
)


events = pd.DataFrame({
    "user_id": [1, 1, 2, 1, 2, 1],
    "item_id": [10, 20, 10, 10, 20, 30],
    "timestamp": [
        "2026-01-01 10:00:00",
        "2026-01-01 10:00:00",
        "2026-01-01 10:00:00",
        "2026-01-02 10:00:00",
        "2026-01-03 10:00:00",
        "2026-01-10 10:00:00",
    ],
    "rating": [5, 3, 4, 2, 5, 4],
    "label": [1, 0, 1, 0, 1, 1],
})

reference = pd.read_pickle("/tmp/features_reference.pkl")
actual = build_historical_features(
    events,
    FeatureConfig(windows_days=(1, 7, 30)),
)

pd.testing.assert_frame_equal(
    actual,
    reference,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12,
)

print("Проверка пройдена: признаки совпадают с эталоном.")
