from __future__ import annotations

import argparse
import sys
import warnings

from src.phase_analysis.constants import DEFAULT_INPUT, PLOTS_PHASE_ROOT, RESULTS_PHASE_ROOT
from src.phase_analysis.pipeline import run_analysis
from src.phase_analysis.preprocessing import PhaseDataError


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Анализ связи показаний прибора и вегетационных индексов с целью "
            "(макроэлемент с фазой или урожайность yield). "
            "Результаты в --results-dir, графики в --plots-dir."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Путь к Excel (по умолчанию: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Целевая переменная: макроэлемент с фазой (N_1, P_2, …) или урожайность: yield",
    )
    parser.add_argument(
        "--device-features",
        nargs="+",
        required=True,
        metavar="COL",
        help="Один или несколько столбцов показаний прибора (например N_test_1 или N_test_1 … N_test_4)",
    )
    parser.add_argument(
        "--index-features",
        nargs="*",
        default=[],
        metavar="COL",
        help="Список вегетационных индексов (столбцы Excel). Можно оставить пустым: --index-features",
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification", "both"],
        default="both",
        help="Режим: регрессия, только классификация или оба (по умолчанию: both)",
    )
    parser.add_argument(
        "--plots-dir",
        default=PLOTS_PHASE_ROOT,
        help=f"Корневая папка для графиков (по умолчанию: {PLOTS_PHASE_ROOT})",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_PHASE_ROOT,
        help=f"Корневая папка для Excel-результатов (по умолчанию: {RESULTS_PHASE_ROOT})",
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
    try:
        run_analysis(
            input_path=args.input,
            target=args.target,
            device_features=args.device_features,
            index_features=args.index_features,
            task=args.task,
            plots_dir=args.plots_dir,
            results_dir=args.results_dir,
            random_state=args.random_state,
        )
    except PhaseDataError as e:
        print(f"\nОшибка данных: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"\nОшибка параметров: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
