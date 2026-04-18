from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import correlation_analysis, plot_scatter_features_vs_target
from .constants import (
    CORR_FEATURES,
    FEATURE_SETS,
    LAB_N_RESULTS_SUBDIR,
    N_TEST_DATASET_OVERVIEW_COLS,
    PEARSON_COL_LAB_N,
    PEARSON_COL_N_TEST,
    PREDICTIONS_SUBDIR_LAB_N,
    PREDICTIONS_SUBDIR_N_TEST,
    RESULTS_DIR,
    RF_IMPORTANCE_FEATURE_SET_LAB_N,
    RF_IMPORTANCE_FEATURE_SET_N_TEST,
    SCATTER_SUBDIR_LAB_N,
    SCATTER_SUBDIR_N_TEST,
    TARGET_COL,
)
from .evaluation import save_results, train_models
from .io_utils import load_data
from .n_tester_analysis import run_n_test_analysis
from .preprocessing import clean_data, clean_data_n_test, print_dataset_overview


def _print_modeling_block(
    title: str,
    *,
    corr_df: pd.DataFrame,
    pearson_col: str,
    results_df: pd.DataFrame,
    corr_target_label: str,
    low_quality_threshold: float = 0.3,
    indirect_features_note: bool = True,
) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    print("\nКорреляции:")
    print(corr_df)

    print("\nРезультаты моделей:")
    print(results_df)

    best_corr_feature = corr_df.iloc[0]["feature"] if not corr_df.empty else None
    best_row_by_cv = results_df.iloc[0] if not results_df.empty else None
    best_row_by_test = (
        results_df.sort_values(by=["R2", "RMSE"], ascending=[False, True]).iloc[0]
        if not results_df.empty
        else None
    )

    if best_corr_feature is not None:
        best_corr_val = float(corr_df.iloc[0][pearson_col])
        print(f"\nМаксимальная корреляция с {corr_target_label}: {best_corr_feature} (r={best_corr_val:.3f})")

    if best_row_by_test is not None:
        print(
            "\nЛучшая модель по R2 на тестовой выборке (дополнительная оценка):"
            f" feature_set={best_row_by_test['feature_set']}, model={best_row_by_test['model']}, "
            f"R2={float(best_row_by_test['R2']):.3f}"
        )

    if best_row_by_cv is not None:
        print(
            "\nНаиболее надёжная модель по результатам кросс-валидации (основной вывод):"
            f" feature_set={best_row_by_cv['feature_set']}, model={best_row_by_cv['model']}, "
            f"R2_cv_mean={float(best_row_by_cv['R2_cv_mean']):.3f} ± {float(best_row_by_cv['R2_cv_std']):.3f}"
        )

    if indirect_features_note and (not results_df.empty) and (results_df["R2_cv_mean"] < low_quality_threshold).all():
        print(
            "\nМодели показывают низкое качество прогнозирования. Это может быть связано с тем, что используемые "
            "признаки являются косвенными характеристиками и не отражают напрямую целевую величину."
        )

    print(
        "\nАнализ графиков predicted vs actual показывает степень совпадения предсказанных и фактических значений.\n"
        "Значительный разброс точек свидетельствует о низкой точности моделей."
    )


def run_all(input_path: str | Path, plots_dir: str | Path, random_state: int = 42) -> None:
    results_root = Path(RESULTS_DIR)
    lab_dir = results_root / LAB_N_RESULTS_SUBDIR
    lab_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(input_path)
    df_lab = clean_data(df_raw)
    df_ntest = clean_data_n_test(df_raw)

    print_dataset_overview("Датасет для анализа lab_N", df_raw, df_lab)
    print_dataset_overview(
        "Датасет для анализа N-test",
        df_raw,
        df_ntest,
        overview_columns=N_TEST_DATASET_OVERVIEW_COLS,
    )

    # --- lab_N (без изменений логики, другие только пути результатов и графиков) ---
    corr_path_lab = lab_dir / "correlations_labN.xlsx"
    corr_df_lab = correlation_analysis(
        df_lab,
        feature_cols=CORR_FEATURES,
        target_col=TARGET_COL,
        pearson_col_name=PEARSON_COL_LAB_N,
        out_path=corr_path_lab,
    )

    plot_scatter_features_vs_target(
        df_lab,
        feature_cols=CORR_FEATURES,
        target_col=TARGET_COL,
        plots_dir=plots_dir,
        scatter_subdir=SCATTER_SUBDIR_LAB_N,
    )

    results_df_lab, rf_importance_lab = train_models(
        df_lab,
        target_col=TARGET_COL,
        feature_sets=FEATURE_SETS,
        random_state=random_state,
        predictions_plots_dir=Path(plots_dir) / PREDICTIONS_SUBDIR_LAB_N,
        target_display_name="lab_N",
        rf_importance_feature_set=RF_IMPORTANCE_FEATURE_SET_LAB_N,
    )

    results_path_lab, importance_path_lab = save_results(
        results_df=results_df_lab,
        rf_importance_df=rf_importance_lab,
        out_results_path=lab_dir / "model_results_labN.xlsx",
        out_importance_path=lab_dir / "feature_importance_labN.xlsx",
    )

    # --- N-test ---
    corr_df_nt, results_df_nt, corr_path_nt, results_path_nt, importance_path_nt = run_n_test_analysis(
        df_ntest,
        plots_dir=plots_dir,
        random_state=random_state,
        results_root=results_root,
    )

    # Итоговый вывод: сначала lab_N, затем N-test
    _print_modeling_block(
        "ИТОГИ: лабораторный азот (lab_N)",
        corr_df=corr_df_lab,
        pearson_col=PEARSON_COL_LAB_N,
        results_df=results_df_lab,
        corr_target_label="lab_N",
    )
    _print_modeling_block(
        "ИТОГИ: N-тестер (N-test), предикторы — NDVI, GNDVI, NDRE, RECI",
        corr_df=corr_df_nt,
        pearson_col=PEARSON_COL_N_TEST,
        results_df=results_df_nt,
        corr_target_label="N-test",
    )

    print(f"\n{'=' * 60}")
    print("Сохранённые файлы")
    print(f"{'=' * 60}")
    print("\nresults/labN/:")
    print(f"- {corr_path_lab.resolve()}")
    print(f"- {results_path_lab.resolve()}")
    print(
        f"- {importance_path_lab.resolve()} "
        f"(важности признаков для RandomForestRegressor, набор «{RF_IMPORTANCE_FEATURE_SET_LAB_N}»)"
    )
    print("\nresults/Ntest/:")
    print(f"- {corr_path_nt.resolve()}")
    print(f"- {results_path_nt.resolve()}")
    print(
        f"- {importance_path_nt.resolve()} "
        f"(важности признаков для RandomForestRegressor, набор «{RF_IMPORTANCE_FEATURE_SET_N_TEST}»)"
    )
    print("\nГрафики:")
    print(f"- scatter: {Path(plots_dir).resolve()}/{SCATTER_SUBDIR_LAB_N}/ и .../{SCATTER_SUBDIR_N_TEST}/")
    print(
        f"- predicted vs actual: {Path(plots_dir).resolve()}/{PREDICTIONS_SUBDIR_LAB_N}/ "
        f"и .../{PREDICTIONS_SUBDIR_N_TEST}/"
    )
