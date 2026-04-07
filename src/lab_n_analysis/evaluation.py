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

from .constants import FEATURE_SETS, TARGET_COL
from .excel_utils import save_excel_wait
from .models import build_models, make_pipeline
from .prediction_plots import plot_predicted_vs_actual


def train_models(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
    plots_dir: str | Path = "plots",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Обучение всех моделей по наборам признаков; важности — только RF на combined."""
    results: list[dict[str, object]] = []

    models = build_models(random_state=random_state)
    rf_importance_df: pd.DataFrame | None = None
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for feature_set_name, features in FEATURE_SETS.items():
        X = df[features]
        y = df[TARGET_COL]

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
                out_dir=Path(plots_dir) / "predictions",
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

            if (
                model_name == "RandomForestRegressor"
                and feature_set_name == "combined"
                and rf_importance_df is None
            ):
                rf_model = pipe.named_steps["model"]
                importances = getattr(rf_model, "feature_importances_", None)
                if importances is not None:
                    rf_importance_df = pd.DataFrame(
                        {"feature": features, "importance": importances}
                    ).sort_values("importance", ascending=False)

    # Сортировка: главный критерий — R² по CV; RMSE только с одного test-split (вторично).
    results_df = pd.DataFrame(results).sort_values(by=["R2_cv_mean", "RMSE"], ascending=[False, True])

    if rf_importance_df is None:
        rf_importance_df = pd.DataFrame({"feature": [], "importance": []})

    return results_df, rf_importance_df


def save_results(
    results_df: pd.DataFrame,
    rf_importance_df: pd.DataFrame,
    out_results_path: str | Path = "results/model_results_labN.xlsx",
    out_importance_path: str | Path = "results/feature_importance.xlsx",
) -> tuple[Path, Path]:
    results_path = save_excel_wait(results_df, out_results_path)
    importance_path = save_excel_wait(rf_importance_df, out_importance_path)
    return results_path, importance_path

