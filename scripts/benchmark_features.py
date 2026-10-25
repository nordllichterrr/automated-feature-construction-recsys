import time
import tracemalloc

from src.feature_construction.data import load_movielens_1m
from src.feature_construction.features import (
    FeatureConfig,
    build_historical_features,
)


def dataframe_size_mb(df):
    return df.memory_usage(index=True, deep=True).sum() / (1024 ** 2)


def main():
    events = load_movielens_1m()

    print("Количество событий:", len(events))
    print(
        "Размер входного DataFrame: {:.2f} MiB".format(
            dataframe_size_mb(events)
        )
    )

    config = FeatureConfig(windows_days=(1, 7, 30))

    tracemalloc.start()
    started = time.perf_counter()

    features = build_historical_features(events, config)

    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("Количество строк признаков:", len(features))
    print("Количество столбцов:", len(features.columns))
    print(
        "Размер выходного DataFrame: {:.2f} MiB".format(
            dataframe_size_mb(features)
        )
    )
    print("Время построения: {:.2f} сек.".format(elapsed))
    print(
        "Пиковая отслеживаемая память Python: {:.2f} MiB".format(
            peak_bytes / (1024 ** 2)
        )
    )


if __name__ == "__main__":
    main()
