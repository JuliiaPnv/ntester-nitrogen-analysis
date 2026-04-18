from __future__ import annotations

import argparse
import sys
import warnings

from src.nitrogen_analysis.pipeline import run_all


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Два сценария: (1) связь признаков с lab_N и регрессионные модели для lab_N; "
            "(2) корреляции NDVI/GNDVI/NDRE/RECI с N-test и модели предсказания N-test по этим индексам. "
            "Результаты в results/labN и results/Ntest; графики — scatter_labN / scatter_Ntest и predictions_*."
        )
    )
    parser.add_argument(
        "--input",
        default="points_all_data.xlsx",
        help="Путь к входному Excel-файлу (по умолчанию: points_all_data.xlsx)",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Папка для PNG-графиков (по умолчанию: plots)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed для воспроизводимости (по умолчанию: 42)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    # Скрываем ConvergenceWarning у MLP только для читаемого вывода; на выводы по качеству опираемся на CV.
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