from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import make_pipeline
from .prediction_plots import plot_predicted_vs_actual

# На Windows n_jobs=-1 у GridSearchCV иногда конфликтует с GUI-бэкендом matplotlib в дочерних процессах.
_GRID_N_JOBS = 1 if sys.platform == "win32" else -1


def _params_for_json(params: dict[str, Any]) -> dict[str, Any]:
    """GridSearch возвращает numpy-типы — приводим к обычным для json.dumps."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _yield_cv(n_samples: int, random_state: int) -> KFold:
    """Та же логика n_splits, что в ``evaluation.train_models``."""
    n_splits = max(2, min(5, n_samples - 1))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _metrics_row(
    *,
    feature_set: str,
    model: str,
    tuning: str,
    y_test: pd.Series,
    preds: np.ndarray,
    cv_scores: np.ndarray,
    best_params: dict[str, Any] | None,
) -> dict[str, object]:
    r2 = float(r2_score(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    med_ae = float(median_absolute_error(y_test, preds))
    row: dict[str, object] = {
        "feature_set": feature_set,
        "model": model,
        "tuning": tuning,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MedianAE": med_ae,
        "R2_cv_mean": float(np.mean(cv_scores)),
        "R2_cv_std": float(np.std(cv_scores)),
        "best_params": json.dumps(_params_for_json(best_params), ensure_ascii=False) if best_params else "",
    }
    return row


def _plot_pair(
    y_test: pd.Series,
    preds: np.ndarray,
    model: str,
    tuning: str,
    feature_set_name: str,
    out_dir: Path,
    target_display_name: str,
) -> None:
    label = f"{model}_{tuning}"
    plot_predicted_vs_actual(
        y_true=y_test.to_numpy(),
        y_pred=np.asarray(preds),
        model_name=label,
        feature_set_name=feature_set_name,
        out_dir=out_dir,
        target_display_name=target_display_name,
    )


def train_yield_regression_baseline_and_tuned(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_sets: dict[str, list[str]],
    random_state: int = 42,
    test_size: float = 0.2,
    predictions_plots_dir: str | Path,
    target_display_name: str,
) -> pd.DataFrame:
    """
    Только yield: Ridge, ElasticNet, RandomForestRegressor.
    Для каждого набора признаков — тот же ``train_test_split`` и тот же KFold (по числу объектов),
    что в ``train_models``: сначала baseline (как в ``models.build_models``), затем GridSearchCV.
    """
    results: list[dict[str, object]] = []
    plots_root = Path(predictions_plots_dir)

    for feature_set_name, features in feature_sets.items():
        X = df[features]
        y = df[target_col]
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("Для регрессии нужно минимум 2 строки с данными.")

        cv = _yield_cv(n_samples, random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        # --- Baseline (дефолты как в models.build_models) ---
        baseline_specs: list[tuple[str, str, object]] = [
            ("Ridge", "Ridge", Ridge(alpha=1.0)),
            (
                "ElasticNet",
                "ElasticNet",
                ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=5000),
            ),
            (
                "RandomForestRegressor",
                "RandomForestRegressor",
                RandomForestRegressor(n_estimators=300, random_state=random_state),
            ),
        ]

        for model_name, pipe_key, estimator in baseline_specs:
            pipe = make_pipeline(pipe_key, estimator)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            results.append(
                _metrics_row(
                    feature_set=feature_set_name,
                    model=model_name,
                    tuning="baseline",
                    y_test=y_test,
                    preds=preds,
                    cv_scores=cv_scores,
                    best_params=None,
                )
            )
            _plot_pair(y_test, preds, model_name, "baseline", feature_set_name, plots_root, target_display_name)

        # --- GridSearchCV (тот же cv-объект; обучение на train, как в типичном тюнинге) ---
        ridge_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        )
        grid_ridge = GridSearchCV(
            ridge_pipe,
            param_grid={"model__alpha": [0.01, 0.1, 1, 10, 100]},
            cv=cv,
            scoring="r2",
            n_jobs=_GRID_N_JOBS,
            refit=True,
        )
        grid_ridge.fit(X_train, y_train)
        pr = grid_ridge.predict(X_test)
        cv_tuned_ridge = cross_val_score(grid_ridge.best_estimator_, X, y, cv=cv, scoring="r2")
        results.append(
            _metrics_row(
                feature_set=feature_set_name,
                model="Ridge",
                tuning="tuned",
                y_test=y_test,
                preds=pr,
                cv_scores=cv_tuned_ridge,
                best_params=dict(grid_ridge.best_params_),
            )
        )
        _plot_pair(y_test, pr, "Ridge", "tuned", feature_set_name, plots_root, target_display_name)

        en_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(random_state=random_state, max_iter=5000),
                ),
            ]
        )
        grid_en = GridSearchCV(
            en_pipe,
            param_grid={
                "model__alpha": [0.01, 0.1, 1, 10],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
            cv=cv,
            scoring="r2",
            n_jobs=_GRID_N_JOBS,
            refit=True,
        )
        grid_en.fit(X_train, y_train)
        pe = grid_en.predict(X_test)
        cv_tuned_en = cross_val_score(grid_en.best_estimator_, X, y, cv=cv, scoring="r2")
        results.append(
            _metrics_row(
                feature_set=feature_set_name,
                model="ElasticNet",
                tuning="tuned",
                y_test=y_test,
                preds=pe,
                cv_scores=cv_tuned_en,
                best_params=dict(grid_en.best_params_),
            )
        )
        _plot_pair(y_test, pe, "ElasticNet", "tuned", feature_set_name, plots_root, target_display_name)

        rf_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(random_state=random_state),
                ),
            ]
        )
        grid_rf = GridSearchCV(
            rf_pipe,
            param_grid={
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 3, 5],
                "model__min_samples_leaf": [1, 3, 5],
            },
            cv=cv,
            scoring="r2",
            n_jobs=_GRID_N_JOBS,
            refit=True,
        )
        grid_rf.fit(X_train, y_train)
        p_rf = grid_rf.predict(X_test)
        cv_tuned_rf = cross_val_score(grid_rf.best_estimator_, X, y, cv=cv, scoring="r2")
        results.append(
            _metrics_row(
                feature_set=feature_set_name,
                model="RandomForestRegressor",
                tuning="tuned",
                y_test=y_test,
                preds=p_rf,
                cv_scores=cv_tuned_rf,
                best_params=dict(grid_rf.best_params_),
            )
        )
        _plot_pair(y_test, p_rf, "RandomForestRegressor", "tuned", feature_set_name, plots_root, target_display_name)

    results_df = pd.DataFrame(results)
    # Сортировка как раньше по главной метрике; tuned/baseline рядом по model
    results_df = results_df.sort_values(
        by=["R2_cv_mean", "RMSE"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return results_df


def print_yield_tuning_comparison(results_df: pd.DataFrame) -> None:
    """Сводка baseline vs tuned по R2_cv_mean и R2 (test)."""
    print(f"\n{'=' * 60}")
    print("Yield: сравнение baseline vs tuned (Ridge, ElasticNet, RandomForest)")
    print(f"{'=' * 60}")

    if results_df.empty:
        print("Нет результатов.")
        return

    for model in ["Ridge", "ElasticNet", "RandomForestRegressor"]:
        sub = results_df[results_df["model"] == model]
        if sub.empty:
            continue
        print(f"\n--- {model} ---")
        for fs in sub["feature_set"].unique():
            pair = sub[sub["feature_set"] == fs]
            b = pair[pair["tuning"] == "baseline"]
            t = pair[pair["tuning"] == "tuned"]
            if b.empty or t.empty:
                continue
            b = b.iloc[0]
            t = t.iloc[0]
            d_cv = float(t["R2_cv_mean"]) - float(b["R2_cv_mean"])
            d_r2 = float(t["R2"]) - float(b["R2"])
            print(
                f"  {fs}: R2_cv_mean {float(b['R2_cv_mean']):.4f} → {float(t['R2_cv_mean']):.4f} "
                f"(Δ {d_cv:+.4f}); R2_test {float(b['R2']):.4f} → {float(t['R2']):.4f} (Δ {d_r2:+.4f})"
            )
            if t["best_params"]:
                print(f"    best_params: {t['best_params']}")

    print(
        "\nОриентир: рост R2_cv_mean на 0.02–0.05 после тюнинга — заметное улучшение; "
        "≈0 — сигнал, что узкое место скорее в данных/признаках; отрицательный Δ — риск переобучения."
    )
