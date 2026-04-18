from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import correlation_analysis, plot_scatter_features_vs_target
from .constants import (
    FEATURE_SETS_N_TEST,
    N_TEST_COL,
    N_TEST_RESULTS_SUBDIR,
    PEARSON_COL_N_TEST,
    PREDICTIONS_SUBDIR_N_TEST,
    RF_IMPORTANCE_FEATURE_SET_N_TEST,
    SCATTER_SUBDIR_N_TEST,
    VEGETATION_INDICES,
)
from .evaluation import save_results, train_models


def run_n_test_analysis(
    df_ntest: pd.DataFrame,
    *,
    plots_dir: str | Path,
    random_state: int,
    results_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    """
    Корреляции индексов с N-test, scatter, те же модели по NDVI..RECI → N-test.
    Возвращает corr_df, results_df и пути к сохранённым Excel.
    """
    out_dir = results_root / N_TEST_RESULTS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_path = out_dir / "correlations_N_test.xlsx"
    corr_df = correlation_analysis(
        df_ntest,
        feature_cols=VEGETATION_INDICES,
        target_col=N_TEST_COL,
        pearson_col_name=PEARSON_COL_N_TEST,
        out_path=corr_path,
    )

    plot_scatter_features_vs_target(
        df_ntest,
        feature_cols=VEGETATION_INDICES,
        target_col=N_TEST_COL,
        plots_dir=plots_dir,
        scatter_subdir=SCATTER_SUBDIR_N_TEST,
    )

    results_df, rf_importance_df = train_models(
        df_ntest,
        target_col=N_TEST_COL,
        feature_sets=FEATURE_SETS_N_TEST,
        random_state=random_state,
        predictions_plots_dir=Path(plots_dir) / PREDICTIONS_SUBDIR_N_TEST,
        target_display_name="N-test",
        rf_importance_feature_set=RF_IMPORTANCE_FEATURE_SET_N_TEST,
    )

    results_path, importance_path = save_results(
        results_df=results_df,
        rf_importance_df=rf_importance_df,
        out_results_path=out_dir / "model_results_N_test.xlsx",
        out_importance_path=out_dir / "feature_importance_N_test.xlsx",
    )

    return corr_df, results_df, corr_path, results_path, importance_path
