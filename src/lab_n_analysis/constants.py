from __future__ import annotations

# Excel-результаты сохраняются сюда (создаётся автоматически при записи).
RESULTS_DIR: str = "results"

NUMERIC_COLS: list[str] = ["T", "N-test", "NDVI", "GNDVI", "NDRE", "RECI", "lab_N"]

TARGET_COL: str = "lab_N"

CORR_FEATURES: list[str] = ["T", "N-test", "NDVI", "GNDVI", "NDRE", "RECI"]

FEATURE_SETS: dict[str, list[str]] = {
    "N_test_only": ["N-test"],
    "multispectral_only": ["NDVI", "GNDVI", "NDRE", "RECI"],
    "combined": ["N-test", "NDVI", "GNDVI", "NDRE", "RECI", "T"],
}

