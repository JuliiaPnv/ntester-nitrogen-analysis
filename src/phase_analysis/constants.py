from __future__ import annotations

# Корни вывода по умолчанию (можно переопределить через CLI --results-dir / --plots-dir)
RESULTS_PHASE_ROOT: str = "results/phase"
PLOTS_PHASE_ROOT: str = "plots/phase"

DEFAULT_INPUT: str = "yield_analys.xlsx"

# Минимум строк после удаления пропусков, ниже которого выводится предупреждение
MIN_RECOMMENDED_SAMPLES: int = 15


def pearson_column_name(target_col: str) -> str:
    return f"pearson_corr_with_{target_col}"


def scatter_subdir_for_target(target_col: str) -> str:
    """Подпапка scatter-графиков: ``scatter_<target>``."""
    return f"scatter_{target_col}"


def predictions_subdir_for_target(target_col: str) -> str:
    """Подпапка графиков предсказаний: ``predictions_<target>``."""
    return f"predictions_{target_col}"
