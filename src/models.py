from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

RANDOM_SEED = 42

CLASSES = [0, 1, 2]
CLASS_LABELS = ["A", "D", "H"]


def evaluate_model(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    split_name: str = "val",
) -> dict:
    y_pred = model.predict(X)
    wf1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y, y_pred)
    per_class = f1_score(y, y_pred, average=None, labels=CLASSES, zero_division=0)

    print(f"\n{'='*50}")
    print(f"Результаты на {split_name}")
    print(f"  Weighted F1 : {wf1:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 по классам: A={per_class[0]:.3f}  D={per_class[1]:.3f}  H={per_class[2]:.3f}")
    print("=" * 50)
    print(classification_report(y, y_pred, target_names=CLASS_LABELS, zero_division=0))

    return {
        "split": split_name,
        "weighted_f1": round(wf1, 4),
        "accuracy": round(acc, 4),
        "f1_A": round(float(per_class[0]), 4),
        "f1_D": round(float(per_class[1]), 4),
        "f1_H": round(float(per_class[2]), 4),
    }


def plot_confusion_matrix(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    title: str = "Матрица ошибок",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=CLASSES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(title)
    return ax


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Важность признаков",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    idx = np.argsort(importances)[-top_n:]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="#3498db",
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("Важность")
    return ax


def results_table(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records).round(4)
