from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import MIN_RECOMMENDED_SAMPLES


class PhaseDataError(ValueError):
    pass


def _missing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def validate_input_path(path: str | Path) -> Path:
    """Проверка, что файл по пути существует."""
    p = Path(path)
    if not p.exists():
        raise PhaseDataError(f"Файл не найден: {p.resolve()}")
    return p


def prepare_analysis_frame(
    df: pd.DataFrame,
    *,
    target_col: str,
    device_features: list[str],
    index_features: list[str],
) -> pd.DataFrame:
    """Оставляет нужные столбцы, приводит к числу, удаляет строки с пропусками."""
    use_cols = [target_col, *device_features, *index_features]
    missing = _missing_columns(df, use_cols)
    if missing:
        raise PhaseDataError("В таблице отсутствуют столбцы: " + ", ".join(missing))

    out = df.loc[:, use_cols].copy()
    for col in use_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna()
    dropped = before - len(out)
    if dropped:
        print(f"\nУдалено строк с пропусками или нечисловыми значениями: {dropped} (из {before}).")

    if len(out) < MIN_RECOMMENDED_SAMPLES:
        print(
            "\nВнимание: после удаления пропусков осталось слишком мало наблюдений "
            "для устойчивого обучения моделей."
        )

    return out


def print_dataset_overview(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"Строк (точек): {len(df)}, столбцов: {len(df.columns)}")
