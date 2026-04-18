from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def save_excel_wait(df: pd.DataFrame, path: str | Path) -> Path:
    """Сохранение в Excel; при занятом файле (например, открыт в Excel) — ждём и повторяем."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    warned = False
    while True:
        try:
            df.to_excel(p, index=False)
            return p
        except PermissionError:
            if not warned:
                print(
                    f"\nНе удаётся сохранить файл, потому что он занят (скорее всего открыт в Excel): {p.name}\n"
                    "Закройте файл и сохранение продолжится автоматически..."
                )
                warned = True
            time.sleep(2)

