from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataFileNotFoundError(FileNotFoundError):
    pass


def load_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DataFileNotFoundError(f"Файл не найден: {p.resolve()}")
    try:
        return pd.read_excel(p)
    except ImportError as e:
        msg = str(e).lower()
        if "openpyxl" in msg:
            raise RuntimeError(
                "Не установлен пакет 'openpyxl', необходимый для чтения .xlsx. "
                "Установите зависимости командой: pip install -r requirements.txt"
            ) from e
        raise
    except Exception as e:  # noqa: BLE001 — прочие сбои чтения Excel
        raise RuntimeError(f"Не удалось прочитать Excel-файл: {p.resolve()}") from e

