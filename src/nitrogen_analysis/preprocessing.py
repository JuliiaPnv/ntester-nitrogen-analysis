from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import N_TEST_COL, NUMERIC_COLS, TARGET_COL


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


def clean_data_n_test(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка для сценария N-test, как для lab_N"""
    _ensure_required_columns(df, ["id", *NUMERIC_COLS])

    cleaned = df.copy()
    cleaned[N_TEST_COL] = cleaned[N_TEST_COL].replace("-", np.nan)

    for col in NUMERIC_COLS:
        cleaned[col] = cleaned[col].astype(str).str.replace(",", ".", regex=False)
        cleaned[col] = cleaned[col].replace(["nan", "NaN", "None", ""], np.nan)
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=[N_TEST_COL]).copy()

    if cleaned.empty:
        raise EmptyDatasetError("После очистки датасет пустой (нет валидных значений N-test).")

    return cleaned


def print_dataset_overview(
    section_title: str,
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    *,
    overview_columns: list[str] | None = None,
) -> None:
    """Сводка по сырому и очищенному датасету. Если задан overview_columns — только эти столбцы (для N-test)."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(section_title)
    print(sep)
    if overview_columns is None:
        names = df_raw.columns.tolist()
        raw_head = df_raw
        clean_head = df_clean
        na_cols = NUMERIC_COLS
        desc_cols = [c for c in NUMERIC_COLS if c in df_clean.columns]
    else:
        names = [c for c in overview_columns if c in df_raw.columns and c in df_clean.columns]
        raw_head = df_raw[names]
        clean_head = df_clean[names]
        na_cols = names
        desc_cols = [c for c in names if pd.api.types.is_numeric_dtype(df_clean[c])]

    print("\nНазвания столбцов:")
    print(names)
    print("\nПервые 5 строк:")
    print(raw_head.head())
    print("\nПервые 5 строк после очистки:")
    print(clean_head.head())

    print("\nРазмер очищенного датасета:", df_clean.shape)
    print("\nПропуски по столбцам:")
    print(df_clean[na_cols].isna().sum())
    print("\nОписательная статистика:")
    if desc_cols:
        print(df_clean[desc_cols].describe())
    else:
        print("(нет числовых столбцов для describe)")

