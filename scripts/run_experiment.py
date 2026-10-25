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
    print("Время построения: {:.1f} сек.".format(time.time() - started))

    timestamps = pd.to_datetime(features["timestamp"], utc=True)
    train_mask = timestamps <= train_end
    validation_mask = (timestamps > train_end) & (timestamps <= validation_end)
    test_mask = timestamps > validation_end

    partitions = [
        features.loc[train_mask].copy(),
        features.loc[validation_mask].copy(),
        features.loc[test_mask].copy(),
    ]

    if [len(part) for part in partitions] != [
        len(train), len(validation), len(test)
    ]:
        raise RuntimeError("Размеры выборок не совпали после построения признаков")

    results = []

    for name, include_history in [
        ("baseline", False),
        ("manual_history", True),
    ]:
        print("\nЭксперимент:", name)

        X_train, y_train, categorical = prepare_model_data(
            partitions[0], include_history
        )
        X_val, y_val, _ = prepare_model_data(
            partitions[1], include_history
        )
        X_test, y_test, _ = prepare_model_data(
            partitions[2], include_history
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

        val_probabilities = model.predict_proba(X_val)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]

        val_metrics = evaluate_binary_classifier(y_val, val_probabilities)
        test_metrics = evaluate_binary_classifier(y_test, test_probabilities)

        result = {
            "experiment": name,
            "best_iteration": model.get_best_iteration(),
            "fit_seconds": round(time.time() - started, 2),
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_log_loss": val_metrics["log_loss"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_log_loss": test_metrics["log_loss"],
        }
        results.append(result)
        print(result)

    results_path = RESULTS_DIR / "metrics.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)

    print("\nИтоговая таблица:")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nРезультаты сохранены в", results_path)


if __name__ == "__main__":
    main()
