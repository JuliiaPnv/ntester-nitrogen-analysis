from __future__ import annotations

import pandas as pd

from .constants import all_expected_columns


class PhaseDataError(ValueError):
    pass


def validate_columns(df: pd.DataFrame) -> None:
    """Проверяет наличие нужных столбцов; таблица остаётся wide, одна строка — одна точка."""
    expected = all_expected_columns()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise PhaseDataError(
            "В таблице отсутствуют ожидаемые столбцы (проверьте yield_analys.xlsx): "
            + ", ".join(missing)
        )


def print_dataset_overview(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"Строк (точек): {len(df)}, столбцов: {len(df.columns)}")
