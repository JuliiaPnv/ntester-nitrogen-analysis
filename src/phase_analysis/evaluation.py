from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from .models import build_models, make_pipeline
from .prediction_plots import plot_predicted_vs_actual


def train_models(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_sets: dict[str, list[str]],
    random_state: int = 42,
    test_size: float = 0.2,
    predictions_plots_dir: str | Path,
    target_display_name: str,
) -> pd.DataFrame:
    """Обучение моделей по наборам признаков; основная метрика — R² CV (R2_cv_mean)."""
    results: list[dict[str, object]] = []

    models = build_models(random_state=random_state)

    for feature_set_name, features in feature_sets.items():
        X = df[features]
        y = df[target_col]

        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("Для регрессии нужно минимум 2 строки с данными.")
        n_splits = max(2, min(5, n_samples - 1))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        for model_name, model in models.items():
            pipe = make_pipeline(model_name, model)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            plot_predicted_vs_actual(
                y_true=y_test.to_numpy(),
                y_pred=np.asarray(preds),
                model_name=model_name,
                feature_set_name=feature_set_name,
                out_dir=Path(predictions_plots_dir),
                target_display_name=target_display_name,
            )

            r2 = float(r2_score(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            med_ae = float(median_absolute_error(y_test, preds))

            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

            results.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "R2": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "MedianAE": med_ae,
                    "R2_cv_mean": float(np.mean(cv_scores)),
                    "R2_cv_std": float(np.std(cv_scores)),
                }
            )

    results_df = pd.DataFrame(results).sort_values(by=["R2_cv_mean", "RMSE"], ascending=[False, True])
    return results_df


def save_model_results(results_df: pd.DataFrame, out_path: str | Path) -> Path:
    from .excel_utils import save_excel_wait

    return save_excel_wait(results_df, out_path)
