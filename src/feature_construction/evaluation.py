import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def evaluate_binary_classifier(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Ожидаются одномерные массивы")

    if len(y_true) != len(probabilities):
        raise ValueError("Размеры y_true и probabilities не совпадают")

    if len(y_true) == 0:
        raise ValueError("Нельзя оценивать пустую выборку")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true должен содержать только значения 0 и 1")

    if len(np.unique(y_true)) < 2:
        raise ValueError("Для ROC-AUC необходимы оба класса")

    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
    }
