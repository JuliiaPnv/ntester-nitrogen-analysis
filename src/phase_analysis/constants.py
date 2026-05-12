from __future__ import annotations

RESULTS_PHASE_ROOT: str = "results/phase"
PLOTS_PHASE_ROOT: str = "plots/phase"

DEFAULT_INPUT: str = "yield_analys.xlsx"

MIN_RECOMMENDED_SAMPLES: int = 15


def pearson_column_name(target_col: str) -> str:
    return f"pearson_corr_with_{target_col}"


def scatter_subdir_for_target(target_col: str) -> str:
    """Имя подпапки для scatter: scatter_<target>."""
    return f"scatter_{target_col}"


def predictions_subdir_for_target(target_col: str) -> str:
    """Имя подпапки для графиков предсказаний: predictions_<target>."""
    return f"predictions_{target_col}"
