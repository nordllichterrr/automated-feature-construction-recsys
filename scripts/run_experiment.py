from pathlib import Path
import time

import pandas as pd

from src.feature_construction.data import load_movielens_1m
from src.feature_construction.datasets import prepare_model_data
from src.feature_construction.evaluation import evaluate_binary_classifier
from src.feature_construction.features import FeatureConfig, build_historical_features
from src.feature_construction.models import create_model
from src.feature_construction.split import temporal_split


RESULTS_DIR = Path("results")

EXPERIMENTS = [
    {
        "name": "baseline",
        "include_history": False,
        "include_rating_statistics": False,
    },
    {
        "name": "history_counts",
        "include_history": True,
        "include_rating_statistics": False,
    },
    {
        "name": "rating_statistics",
        "include_history": False,
        "include_rating_statistics": True,
    },
    {
        "name": "combined",
        "include_history": True,
        "include_rating_statistics": True,
    },
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    events = load_movielens_1m()
    train, validation, test = temporal_split(events)

    train_end = train["timestamp"].max()
    validation_end = validation["timestamp"].max()

    print("Размеры выборок:", len(train), len(validation), len(test))
    print("Строим исторические признаки...")

    started = time.time()
    features = build_historical_features(
        events,
        FeatureConfig(windows_days=(1, 7, 30)),
    )
    feature_build_seconds = time.time() - started

    print("Время построения: {:.1f} сек.".format(feature_build_seconds))

    timestamps = pd.to_datetime(features["timestamp"], utc=True)

    masks = [
        timestamps <= train_end,
        (timestamps > train_end) & (timestamps <= validation_end),
        timestamps > validation_end,
    ]

    partitions = [features.loc[mask].copy() for mask in masks]

    expected_lengths = [len(train), len(validation), len(test)]
    actual_lengths = [len(partition) for partition in partitions]

    if actual_lengths != expected_lengths:
        raise RuntimeError(
            "Размеры выборок не совпали после построения признаков: "
            f"{actual_lengths} != {expected_lengths}"
        )

    results = []

    for experiment in EXPERIMENTS:
        name = experiment["name"]
        include_history = experiment["include_history"]
        include_rating_statistics = experiment["include_rating_statistics"]

        print("\nЭксперимент:", name)

        X_train, y_train, categorical = prepare_model_data(
            partitions[0],
            include_history=include_history,
            include_rating_statistics=include_rating_statistics,
        )
        X_val, y_val, _ = prepare_model_data(
            partitions[1],
            include_history=include_history,
            include_rating_statistics=include_rating_statistics,
        )
        X_test, y_test, _ = prepare_model_data(
            partitions[2],
            include_history=include_history,
            include_rating_statistics=include_rating_statistics,
        )

        model = create_model()
        started = time.time()

        model.fit(
            X_train,
            y_train,
            cat_features=categorical,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100,
        )

        fit_seconds = time.time() - started

        val_probabilities = model.predict_proba(X_val)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]

        val_metrics = evaluate_binary_classifier(y_val, val_probabilities)
        test_metrics = evaluate_binary_classifier(y_test, test_probabilities)

        result = {
            "experiment": name,
            "include_history": include_history,
            "include_rating_statistics": include_rating_statistics,
            "feature_count": X_train.shape[1],
            "best_iteration": model.get_best_iteration(),
            "fit_seconds": round(fit_seconds, 2),
            "feature_build_seconds": round(feature_build_seconds, 2),
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_log_loss": val_metrics["log_loss"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_log_loss": test_metrics["log_loss"],
        }

        results.append(result)
        print(result)

    results_df = pd.DataFrame(results)
    results_path = RESULTS_DIR / "metrics.csv"
    results_df.to_csv(results_path, index=False)

    print("\nИтоговая таблица:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nРезультаты сохранены в", results_path)


if __name__ == "__main__":
    main()
