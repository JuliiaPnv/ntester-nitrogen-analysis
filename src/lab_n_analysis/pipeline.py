from __future__ import annotations

from pathlib import Path

from .analysis import correlation_analysis, plot_data
from .constants import RESULTS_DIR
from .io_utils import load_data
from .evaluation import save_results, train_models
from .preprocessing import clean_data, print_basic_info


def run_all(input_path: str | Path, plots_dir: str | Path, random_state: int = 42) -> None:
    results_dir = Path(RESULTS_DIR)
    df_raw = load_data(input_path)
    df_clean = clean_data(df_raw)

    print_basic_info(df_raw=df_raw, df_clean=df_clean)

    corr_path = results_dir / "correlations_labN.xlsx"
    corr_df = correlation_analysis(df_clean, out_path=corr_path)
    print("\nКорреляции с lab_N:")
    print(corr_df)

    plot_data(df_clean, plots_dir=plots_dir)

    results_df, rf_importance_df = train_models(df_clean, random_state=random_state, plots_dir=plots_dir)
    print("\nРезультаты моделей:")
    print(results_df)

    results_path, importance_path = save_results(
        results_df=results_df,
        rf_importance_df=rf_importance_df,
        out_results_path=results_dir / "model_results_labN.xlsx",
        out_importance_path=results_dir / "feature_importance.xlsx",
    )

    best_corr_feature = corr_df.iloc[0]["feature"] if not corr_df.empty else None
    best_row_by_cv = results_df.iloc[0] if not results_df.empty else None
    best_row_by_test = (
        results_df.sort_values(by=["R2", "RMSE"], ascending=[False, True]).iloc[0]
        if not results_df.empty
        else None
    )

    if best_corr_feature is not None:
        best_corr_val = float(corr_df.iloc[0]["pearson_corr_with_lab_N"])
        print(f"\nМаксимальная корреляция с lab_N: {best_corr_feature} (r={best_corr_val:.3f})")

    if best_row_by_test is not None:
        print(
            "\nЛучшая модель по R2 на тестовой выборке (дополнительная оценка):"
            f" feature_set={best_row_by_test['feature_set']}, model={best_row_by_test['model']}, R2={float(best_row_by_test['R2']):.3f}"
        )

    if best_row_by_cv is not None:
        print(
            "\nНаиболее надёжная модель по результатам кросс-валидации (основной вывод):"
            f" feature_set={best_row_by_cv['feature_set']}, model={best_row_by_cv['model']}, "
            f"R2_cv_mean={float(best_row_by_cv['R2_cv_mean']):.3f} ± {float(best_row_by_cv['R2_cv_std']):.3f}"
        )

    if not results_df.empty and (results_df["R2_cv_mean"] < 0.3).all():
        print(
            "\nМодели показывают низкое качество прогнозирования. Это связано с тем, что используемые признаки "
            "являются косвенными характеристиками и не отражают напрямую содержание азота."
        )

    print(
        "\nАнализ графиков показывает степень совпадения предсказанных и фактических значений.\n"
        "Значительный разброс точек свидетельствует о низкой точности моделей."
    )

    print("\nСохранены файлы (папка results/):")
    print(f"- {corr_path.resolve()}")
    print(f"- {results_path.resolve()}")
    print(
        f"- {importance_path.resolve()} (важности признаков для RandomForestRegressor на наборе combined)"
    )
    print(f"- plots/ (PNG-графики) -> {Path(plots_dir).resolve()}")

