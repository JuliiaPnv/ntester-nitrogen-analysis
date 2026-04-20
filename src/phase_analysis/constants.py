from __future__ import annotations

# Корни вывода (не пересекаются с results/labN, results/Ntest, plots/scatter_labN, …)
RESULTS_PHASE_ROOT: str = "results/phase"
PLOTS_PHASE_ROOT: str = "plots/phase"

DEFAULT_INPUT: str = "yield_analys.xlsx"

TARGET_N1: str = "N_1"
TARGET_N2: str = "N_2"
TARGET_YIELD: str = "yield"


def pearson_column_name(target_col: str) -> str:
    return f"pearson_corr_with_{target_col}"


def phase_index_features(phase: int) -> list[str]:
    return [f"NDVI_{phase}", f"GNDVI_{phase}", f"NDRE_{phase}", f"CI_{phase}"]


def n_test_col(phase: int) -> str:
    return f"N_test_{phase}"


# --- Все ожидаемые столбцы ---
def all_expected_columns() -> list[str]:
    cols = [TARGET_N1, TARGET_N2, TARGET_YIELD]
    for p in range(1, 5):
        cols.append(n_test_col(p))
        cols.extend(phase_index_features(p))
    return cols


# --- Сценарий N_1 (только фаза 1): как lab_N — по одному предиктору + combined ---
def _n1_feature_sets() -> dict[str, list[str]]:
    nt = n_test_col(1)
    ndvi, gndvi, ndre, ci = phase_index_features(1)
    return {
        "N_test_only": [nt],
        "NDVI_only": [ndvi],
        "GNDVI_only": [gndvi],
        "NDRE_only": [ndre],
        "CI_only": [ci],
        "combined": [nt, ndvi, gndvi, ndre, ci],
    }


N1_FEATURE_SETS: dict[str, list[str]] = _n1_feature_sets()

FEATURES_N1_PHASE1: list[str] = N1_FEATURE_SETS["combined"]

N1_CORRELATION_SPECS: list[dict[str, str | list[str]]] = [
    {
        "file": "correlations_N1_phase1.xlsx",
        "features": FEATURES_N1_PHASE1,
    },
]


# --- Сценарий N_2 (только фаза 2): те же имена наборов, столбцы фазы 2 ---
def _n2_feature_sets() -> dict[str, list[str]]:
    nt = n_test_col(2)
    ndvi, gndvi, ndre, ci = phase_index_features(2)
    return {
        "N_test_only": [nt],
        "NDVI_only": [ndvi],
        "GNDVI_only": [gndvi],
        "NDRE_only": [ndre],
        "CI_only": [ci],
        "combined": [nt, ndvi, gndvi, ndre, ci],
    }


N2_FEATURE_SETS: dict[str, list[str]] = _n2_feature_sets()

FEATURES_N2_PHASE2: list[str] = N2_FEATURE_SETS["combined"]

N2_CORRELATION_SPECS: list[dict[str, str | list[str]]] = [
    {
        "file": "correlations_N2_phase2.xlsx",
        "features": FEATURES_N2_PHASE2,
    },
]

# --- Сценарий yield: наборы признаков для моделей ---
YIELD_FEATURE_SETS: dict[str, list[str]] = {
    "N_test_only": [n_test_col(1)],
    "phase1_indices": phase_index_features(1),
    "phase2_indices": phase_index_features(2),
    "phase3_indices": phase_index_features(3),
    "phase4_indices": phase_index_features(4),
    "all_phases_indices": [
        *phase_index_features(1),
        *phase_index_features(2),
        *phase_index_features(3),
        *phase_index_features(4),
    ],
    "combined": [n_test_col(1)]
    + [
        *phase_index_features(1),
        *phase_index_features(2),
        *phase_index_features(3),
        *phase_index_features(4),
    ],
}

# Корреляции yield: отдельный файл на поднабор
YIELD_CORRELATION_SPECS: list[dict[str, str | list[str]]] = [
    {"file": "correlations_yield_ntest.xlsx", "features": [n_test_col(1)]},
    {"file": "correlations_yield_phase1.xlsx", "features": phase_index_features(1)},
    {"file": "correlations_yield_phase2.xlsx", "features": phase_index_features(2)},
    {"file": "correlations_yield_phase3.xlsx", "features": phase_index_features(3)},
    {"file": "correlations_yield_phase4.xlsx", "features": phase_index_features(4)},
    {
        "file": "correlations_yield_all_phases.xlsx",
        "features": YIELD_FEATURE_SETS["all_phases_indices"],
    },
    {"file": "correlations_yield_combined.xlsx", "features": YIELD_FEATURE_SETS["combined"]},
]

# Подпапки графиков
SCATTER_SUBDIR_N1: str = "scatter_N1"
SCATTER_SUBDIR_N2: str = "scatter_N2"
SCATTER_SUBDIR_YIELD: str = "scatter_yield"
PREDICTIONS_SUBDIR_N1: str = "predictions_N1"
PREDICTIONS_SUBDIR_N2: str = "predictions_N2"
PREDICTIONS_SUBDIR_YIELD: str = "predictions_yield"

# Имена для сводки: фазы yield для сравнения качества
YIELD_PHASE_FEATURE_SET_KEYS: tuple[str, ...] = (
    "phase1_indices",
    "phase2_indices",
    "phase3_indices",
    "phase4_indices",
)
