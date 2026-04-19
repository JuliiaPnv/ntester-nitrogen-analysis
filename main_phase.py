from __future__ import annotations

import argparse
import sys
import warnings

from src.phase_analysis.constants import DEFAULT_INPUT, PLOTS_PHASE_ROOT, RESULTS_PHASE_ROOT
from src.phase_analysis.pipeline import run_all


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Анализ по фазам: азот в растениях (N_1, N_2) и урожайность (yield) по wide-таблице yield_analys.xlsx. "
            "Результаты: "
            f"{RESULTS_PHASE_ROOT}/N1, {RESULTS_PHASE_ROOT}/N2, {RESULTS_PHASE_ROOT}/yield; "
            f"графики: {PLOTS_PHASE_ROOT}/…"
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Путь к Excel (по умолчанию: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--plots-dir",
        default=PLOTS_PHASE_ROOT,
        help=f"Корневая папка для графиков phase (по умолчанию: {PLOTS_PHASE_ROOT})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed для train_test_split, KFold и моделей (по умолчанию: 42)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:  # noqa: BLE001
        pass

    args = parse_args(argv)
    run_all(
        input_path=args.input,
        plots_dir=args.plots_dir,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
