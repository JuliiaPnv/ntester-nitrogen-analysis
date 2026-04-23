from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_classifiers(random_state: int) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            random_state=random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
        ),
    }


def make_classification_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def train_classification_models(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target_col: str,
    random_state: int,
    *,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Бинарная классификация: порог — медиана непрерывной цели ``target_col``,
    класс ``(y > threshold).astype(int)``. Результаты по наборам признаков и моделям.
    """
    y_cont = df[target_col]
    valid = y_cont.notna()
    if int(valid.sum()) < 2:
        raise ValueError(f"Недостаточно наблюдений с непустым {target_col} для классификации.")

    threshold = float(y_cont[valid].median())
    y_class_full = pd.Series(np.nan, index=df.index, dtype=float)
    y_class_full.loc[valid] = (y_cont[valid] > threshold).astype(int)

    results: list[dict[str, object]] = []
    models = _build_classifiers(random_state=random_state)

    for feature_set_name, features in feature_sets.items():
        sub = df.loc[valid, features].copy()
        y = y_class_full.loc[valid].astype(int)

        n_samples = len(sub)
        if n_samples < 2:
            raise ValueError("Недостаточно строк после отбора по целевой переменной.")

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            sub,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        for model_name, clf in models.items():
            pipe = make_classification_pipeline(clf)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            m = _binary_metrics(y_test.to_numpy(), np.asarray(preds))
            results.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "threshold_median": threshold,
                    **m,
                }
            )

    return pd.DataFrame(results).sort_values(
        by=["accuracy", "f1_score"],
        ascending=[False, False],
    )


def print_classification_best_models(results_df: pd.DataFrame, scenario_label: str) -> None:
    """Консоль: лучшая модель по accuracy и по F1."""
    print(f"\n{'=' * 60}")
    print(scenario_label)
    print(f"{'=' * 60}")

    if results_df.empty:
        print("Нет результатов классификации.")
        return

    print("\nРезультаты моделей (сортировка по accuracy, затем F1):")
    print(results_df)

    best_acc = results_df.loc[results_df["accuracy"].idxmax()]
    best_f1 = results_df.loc[results_df["f1_score"].idxmax()]
    print(
        f"\nЛучшая модель по accuracy: feature_set={best_acc['feature_set']}, "
        f"model={best_acc['model']}, accuracy={float(best_acc['accuracy']):.4f}"
    )
    print(
        f"Лучшая модель по F1: feature_set={best_f1['feature_set']}, "
        f"model={best_f1['model']}, f1_score={float(best_f1['f1_score']):.4f}"
    )
