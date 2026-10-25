from catboost import CatBoostClassifier


def create_model(iterations=500, depth=8, learning_rate=0.05):
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )
