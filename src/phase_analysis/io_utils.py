from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataFileNotFoundError(FileNotFoundError):
    pass


def load_yield_table(path: str | Path) -> pd.DataFrame:
    """Читает yield_analys.xlsx как есть (wide), подчищает имена столбцов."""
    p = Path(path)
    if not p.exists():
        raise DataFileNotFoundError(f"Файл не найден: {p.resolve()}")
    try:
        df = pd.read_excel(p)
    except ImportError as e:
        msg = str(e).lower()
        if "openpyxl" in msg:
            raise RuntimeError(
                "Не установлен пакет 'openpyxl', необходимый для чтения .xlsx. "
                "Установите: pip install openpyxl"
            ) from e
        raise
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Не удалось прочитать Excel-файл: {p.resolve()}") from e

    df.columns = [str(c).strip() for c in df.columns]
    return df
