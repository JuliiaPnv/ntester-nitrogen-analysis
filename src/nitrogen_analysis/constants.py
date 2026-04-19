from __future__ import annotations

# Excel-результаты: подпапки внутри results/
RESULTS_DIR: str = "results"
LAB_N_RESULTS_SUBDIR: str = "labN"
N_TEST_RESULTS_SUBDIR: str = "Ntest"

NUMERIC_COLS: list[str] = ["T", "N-test", "NDVI", "GNDVI", "NDRE", "RECI", "lab_N"]

TARGET_COL: str = "lab_N"
N_TEST_COL: str = "N-test"

CORR_FEATURES: list[str] = ["T", "N-test", "NDVI", "GNDVI", "NDRE", "RECI"]

VEGETATION_INDICES: list[str] = ["NDVI", "GNDVI", "NDRE", "RECI"]

# Колонки для консольной сводки датасета N-test (без T и lab_N)
N_TEST_DATASET_OVERVIEW_COLS: list[str] = ["id", N_TEST_COL, *VEGETATION_INDICES]

PEARSON_COL_LAB_N: str = "pearson_corr_with_lab_N"
PEARSON_COL_N_TEST: str = "pearson_corr_with_N_test"

FEATURE_SETS: dict[str, list[str]] = {
    "N_test_only": ["N-test"],
    "multispectral_only": ["NDVI", "GNDVI", "NDRE", "RECI"],
    "combined": ["N-test", "NDVI", "GNDVI", "NDRE", "RECI", "T"],
    "NDVI_only": ["NDVI"],
    "GNDVI_only": ["GNDVI"],
    "NDRE_only": ["NDRE"],
    "RECI_only": ["RECI"],
}

# N-test как target: по одному индексу и все четыре вместе
FEATURE_SETS_N_TEST: dict[str, list[str]] = {
    "NDVI_only": ["NDVI"],
    "GNDVI_only": ["GNDVI"],
    "NDRE_only": ["NDRE"],
    "RECI_only": ["RECI"],
    "vegetation_indices": ["NDVI", "GNDVI", "NDRE", "RECI"],
}

RF_IMPORTANCE_FEATURE_SET_LAB_N: str = "combined"
RF_IMPORTANCE_FEATURE_SET_N_TEST: str = "vegetation_indices"

SCATTER_SUBDIR_LAB_N: str = "scatter_labN"
SCATTER_SUBDIR_N_TEST: str = "scatter_Ntest"
PREDICTIONS_SUBDIR_LAB_N: str = "predictions_labN"
PREDICTIONS_SUBDIR_N_TEST: str = "predictions_Ntest"
