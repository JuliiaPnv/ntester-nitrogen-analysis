from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import NUMERIC_COLS, TARGET_COL


class MissingColumnsError(ValueError):
    pass


class EmptyDatasetError(ValueError):
    pass


def _ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnsError(f"Отсутствуют необходимые столбцы: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка: lab_N без «-», числа в float, строки без lab_N удаляются."""
    _ensure_required_columns(df, ["id", *NUMERIC_COLS])

    cleaned = df.copy()
    cleaned[TARGET_COL] = cleaned[TARGET_COL].replace("-", np.nan)

    for col in NUMERIC_COLS:
        cleaned[col] = cleaned[col].astype(str).str.replace(",", ".", regex=False)
        cleaned[col] = cleaned[col].replace(["nan", "NaN", "None", ""], np.nan)
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COL]).copy()

    if cleaned.empty:
        raise EmptyDatasetError("После очистки датасет пустой (нет валидных значений lab_N).")

    return cleaned


def print_basic_info(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    print("Названия столбцов:")
    print(df_raw.columns.tolist())
    print("\nПервые 5 строк:")
    print(df_raw.head())
    print("\nПервые 5 строк после очистки:")
    print(df_clean.head())

    print("\nРазмер очищенного датасета:", df_clean.shape)
    print("\nПропуски по столбцам:")
    print(df_clean[NUMERIC_COLS].isna().sum())
    print("\nОписательная статистика:")
    print(df_clean[NUMERIC_COLS].describe())

