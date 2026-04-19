from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import correlation_analysis, plot_scatter_features_vs_target
from .constants import (
    N1_CORRELATION_SPECS,
    N1_FEATURE_SETS,
    N2_CORRELATION_SPECS,
    N2_FEATURE_SETS,
    PREDICTIONS_SUBDIR_N1,
    PREDICTIONS_SUBDIR_N2,
    PREDICTIONS_SUBDIR_YIELD,
    RESULTS_PHASE_ROOT,
    SCATTER_SUBDIR_N1,
    SCATTER_SUBDIR_N2,
    SCATTER_SUBDIR_YIELD,
    TARGET_N1,
    TARGET_N2,
    TARGET_YIELD,
    YIELD_CORRELATION_SPECS,
    YIELD_FEATURE_SETS,
    YIELD_PHASE_FEATURE_SET_KEYS,
    pearson_column_name,
)
from .evaluation import save_model_results, train_models
from .io_utils import load_yield_table
from .preprocessing import print_dataset_overview, validate_columns


def _union_features(feature_sets: dict[str, list[str]]) -> list[str]:
    s: set[str] = set()
    for feats in feature_sets.values():
        s.update(feats)
    return sorted(s)


def _print_scenario_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def _print_target_summary(
    label: str,
    *,
    corr_primary: pd.DataFrame,
    pearson_col: str,
    results_df: pd.DataFrame,
    corr_context: str,
) -> None:
    """Сводка по одному target: корреляция, лучший R² на тесте, лучший R² CV."""
    _print_scenario_header(label)

    print(f"\nКорреляции ({corr_context}):")
    print(corr_primary)

    if not corr_primary.empty and pearson_col in corr_primary.columns:
        top = corr_primary.iloc[0]
        print(
            f"\nНаибольшая |r| по Пирсону: {top['feature']} "
            f"(r={float(top[pearson_col]):.4f})"
        )

    print("\nРезультаты моделей (сортировка по R2_cv_mean, затем RMSE):")
    print(results_df)

    if not results_df.empty:
        best_test = results_df.loc[results_df["R2"].idxmax()]
        best_cv = results_df.iloc[0]
        print(
            f"\nЛучшая модель по R² на тестовой выборке: "
            f"feature_set={best_test['feature_set']}, model={best_test['model']}, "
            f"R2={float(best_test['R2']):.4f}"
        )
        print(
            "Наиболее надёжная модель по кросс-валидации (основной ориентир): "
            f"feature_set={best_cv['feature_set']}, model={best_cv['model']}, "
            f"R2_cv_mean={float(best_cv['R2_cv_mean']):.4f} ± {float(best_cv['R2_cv_std']):.4f}"
        )


def _best_cv_per_feature_set(results_df: pd.DataFrame, feature_set_key: str) -> tuple[float, str] | None:
    sub = results_df[results_df["feature_set"] == feature_set_key]
    if sub.empty:
        return None
    best = sub.loc[sub["R2_cv_mean"].idxmax()]
    return float(best["R2_cv_mean"]), str(best["model"])


def _print_yield_phase_comparison(results_df: pd.DataFrame) -> None:
    """Сравнение фаз по урожайности и all_phases vs лучшая фаза."""
    print(f"\n{'=' * 60}")
    print("Сравнение сценариев по урожайности (yield): фазы и объединение")
    print(f"{'=' * 60}")

    phase_best: list[tuple[str, float, str]] = []
    for key in YIELD_PHASE_FEATURE_SET_KEYS:
        r = _best_cv_per_feature_set(results_df, key)
        if r is None:
            continue
        cv_mean, model = r
        phase_best.append((key, cv_mean, model))

    if not phase_best:
        print("Нет данных по фазам (пустые результаты).")
        return

    phase_best.sort(key=lambda x: x[1], reverse=True)
    best_key, best_cv, best_model = phase_best[0]
    print("\nЛучший R2_cv_mean среди отдельных фаз (только индексы этой фазы):")
    for k, cv, m in phase_best:
        print(f"  {k}: R2_cv_mean={cv:.4f} (лучшая модель: {m})")
    print(f"\n→ Лучшая фаза по CV: {best_key} (R2_cv_mean={best_cv:.4f})")

    all_ph = _best_cv_per_feature_set(results_df, "all_phases_indices")
    comb = _best_cv_per_feature_set(results_df, "combined")

    if all_ph:
        print(
            f"\nВсе фазы сразу (all_phases_indices): лучший R2_cv_mean={all_ph[0]:.4f} ({all_ph[1]})"
        )
        if best_cv > all_ph[0]:
            print(
                f"Отдельная фаза {best_key} даёт более высокий R2_cv_mean, чем объединение всех индексов."
            )
        elif all_ph[0] > best_cv:
            print(
                "Объединение индексов всех фаз (all_phases_indices) улучшает R2_cv_mean "
                f"относительно лучшей одиночной фазы ({best_key})."
            )
        else:
            print("Качество all_phases_indices близко к лучшей одиночной фазе.")

    if comb:
        print(
            f"\nКомбинированный набор (N_test + все индексы): лучший R2_cv_mean={comb[0]:.4f} ({comb[1]})"
        )


def run_all(
    input_path: str | Path,
    plots_dir: str | Path,
    random_state: int = 42,
) -> None:
    df = load_yield_table(input_path)
    validate_columns(df)

    print_dataset_overview("Датасет yield_analys (wide, одна строка — одна точка)", df)

    results_root = Path(RESULTS_PHASE_ROOT)
    plots_root = Path(plots_dir)
    phase_plots = plots_root  # plots_dir уже корень для phase, например plots/phase

    # --- N_1 ---
    n1_dir = results_root / "N1"
    pear_n1 = pearson_column_name(TARGET_N1)
    corr_n1_primary: pd.DataFrame | None = None

    for spec in N1_CORRELATION_SPECS:
        out_p = n1_dir / str(spec["file"])
        cdf = correlation_analysis(
            df,
            feature_cols=list(spec["features"]),
            target_col=TARGET_N1,
            pearson_col_name=pear_n1,
            out_path=out_p,
        )
        if spec["file"] == "correlations_N1_phase1.xlsx":
            corr_n1_primary = cdf

    plot_scatter_features_vs_target(
        df,
        feature_cols=_union_features(N1_FEATURE_SETS),
        target_col=TARGET_N1,
        plots_dir=phase_plots,
        scatter_subdir=SCATTER_SUBDIR_N1,
    )

    results_n1 = train_models(
        df,
        target_col=TARGET_N1,
        feature_sets=N1_FEATURE_SETS,
        random_state=random_state,
        predictions_plots_dir=phase_plots / PREDICTIONS_SUBDIR_N1,
        target_display_name=TARGET_N1,
    )
    save_model_results(results_n1, n1_dir / "model_results_N1.xlsx")

    if corr_n1_primary is None:
        corr_n1_primary = pd.DataFrame()

    _print_target_summary(
        "Сценарий N_1 (азот в растениях): предсказание только по критериям фазы 1",
        corr_primary=corr_n1_primary,
        pearson_col=pear_n1,
        results_df=results_n1,
        corr_context="все предикторы фазы 1; модели — N_test_only, NDVI_only, …, combined",
    )

    # --- N_2 ---
    n2_dir = results_root / "N2"
    pear_n2 = pearson_column_name(TARGET_N2)
    corr_n2_primary: pd.DataFrame | None = None

    for spec in N2_CORRELATION_SPECS:
        out_p = n2_dir / str(spec["file"])
        cdf = correlation_analysis(
            df,
            feature_cols=list(spec["features"]),
            target_col=TARGET_N2,
            pearson_col_name=pear_n2,
            out_path=out_p,
        )
        if spec["file"] == "correlations_N2_phase2.xlsx":
            corr_n2_primary = cdf

    plot_scatter_features_vs_target(
        df,
        feature_cols=_union_features(N2_FEATURE_SETS),
        target_col=TARGET_N2,
        plots_dir=phase_plots,
        scatter_subdir=SCATTER_SUBDIR_N2,
    )

    results_n2 = train_models(
        df,
        target_col=TARGET_N2,
        feature_sets=N2_FEATURE_SETS,
        random_state=random_state,
        predictions_plots_dir=phase_plots / PREDICTIONS_SUBDIR_N2,
        target_display_name=TARGET_N2,
    )
    save_model_results(results_n2, n2_dir / "model_results_N2.xlsx")

    if corr_n2_primary is None:
        corr_n2_primary = pd.DataFrame()

    _print_target_summary(
        "Сценарий N_2 (азот в растениях): предсказание только по критериям фазы 2",
        corr_primary=corr_n2_primary,
        pearson_col=pear_n2,
        results_df=results_n2,
        corr_context="все предикторы фазы 2; модели — N_test_only, NDVI_only, …, combined",
    )

    # --- yield ---
    y_dir = results_root / "yield"
    pear_y = pearson_column_name(TARGET_YIELD)
    corr_yield_primary: pd.DataFrame | None = None

    for spec in YIELD_CORRELATION_SPECS:
        out_p = y_dir / str(spec["file"])
        cdf = correlation_analysis(
            df,
            feature_cols=list(spec["features"]),
            target_col=TARGET_YIELD,
            pearson_col_name=pear_y,
            out_path=out_p,
        )
        if spec["file"] == "correlations_yield_all_phases.xlsx":
            corr_yield_primary = cdf

    plot_scatter_features_vs_target(
        df,
        feature_cols=_union_features(YIELD_FEATURE_SETS),
        target_col=TARGET_YIELD,
        plots_dir=phase_plots,
        scatter_subdir=SCATTER_SUBDIR_YIELD,
    )

    results_y = train_models(
        df,
        target_col=TARGET_YIELD,
        feature_sets=YIELD_FEATURE_SETS,
        random_state=random_state,
        predictions_plots_dir=phase_plots / PREDICTIONS_SUBDIR_YIELD,
        target_display_name=TARGET_YIELD,
    )
    save_model_results(results_y, y_dir / "model_results_yield.xlsx")

    if corr_yield_primary is None:
        corr_yield_primary = pd.DataFrame()

    _print_target_summary(
        "Сценарий yield (урожайность)",
        corr_primary=corr_yield_primary,
        pearson_col=pear_y,
        results_df=results_y,
        corr_context="все спектральные индексы по фазам (all_phases) — обзор корреляций с yield",
    )
    _print_yield_phase_comparison(results_y)

    # --- Сохранённые файлы ---
    print(f"\n{'=' * 60}")
    print("Сохранённые результаты (отдельно от labN / Ntest)")
    print(f"{'=' * 60}")
    print(f"\n{results_root.resolve()}/N1/ — корреляции, model_results_N1.xlsx")
    print(f"{results_root.resolve()}/N2/ — корреляции, model_results_N2.xlsx")
    print(f"{results_root.resolve()}/yield/ — корреляции, model_results_yield.xlsx")
    print(f"\nГрафики: {phase_plots.resolve()}/ — {SCATTER_SUBDIR_N1}, {SCATTER_SUBDIR_N2}, {SCATTER_SUBDIR_YIELD}, "
          f"{PREDICTIONS_SUBDIR_N1}, {PREDICTIONS_SUBDIR_N2}, {PREDICTIONS_SUBDIR_YIELD}")
