from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import correlation_analysis, plot_scatter_features_vs_target
from .classification import print_classification_best_models, train_classification_models
from .constants import pearson_column_name, predictions_subdir_for_target, scatter_subdir_for_target
from .evaluation import save_model_results, train_models
from .feature_sets import (
    build_feature_sets,
    index_prefix_before_phase,
    validate_target_name,
    yield_phase_index_feature_keys,
)
from .io_utils import load_yield_table
from .preprocessing import PhaseDataError, prepare_analysis_frame, print_dataset_overview, validate_input_path


def _dedupe_preserve(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _union_device_index(device_features: list[str], index_features: list[str]) -> list[str]:
    """Признаки для корреляций и scatter: device, затем индексы, без дубликатов."""
    return _dedupe_preserve([*device_features, *index_features])


def _print_scenario_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def _print_formed_feature_sets(feature_sets: dict[str, list[str]]) -> None:
    print("\nСформированные наборы признаков:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {', '.join(cols)}")


def _print_correlations_block(target: str, corr_df: pd.DataFrame, pearson_col: str) -> None:
    print(f"\nКорреляции с {target}:")
    print(corr_df)
    if not corr_df.empty and pearson_col in corr_df.columns:
        top = corr_df.iloc[0]
        print(
            f"\nНаибольшая |r| по Пирсону: {top['feature']} "
            f"(r={float(top[pearson_col]):.4f})"
        )


def _print_regression_summary(target: str, results_df: pd.DataFrame) -> None:
    print("\nРезультаты моделей (сортировка по R2_cv_mean, затем RMSE):")
    print(results_df)
    if results_df.empty:
        return
    best_test = results_df.loc[results_df["R2"].idxmax()]
    best_cv = results_df.iloc[0]
    print(
        f"\nЛучшая модель по R2 на тестовой выборке: "
        f"feature_set={best_test['feature_set']}, model={best_test['model']}, "
        f"R2={float(best_test['R2']):.4f}"
    )
    print(
        "Наиболее надёжная модель по кросс-валидации (основной ориентир): "
        f"feature_set={best_cv['feature_set']}, model={best_cv['model']}, "
        f"R2_cv_mean={float(best_cv['R2_cv_mean']):.4f} ± {float(best_cv['R2_cv_std']):.4f}"
    )


def _best_cv_per_feature_set(results_df: pd.DataFrame, feature_set_key: str) -> tuple[float, str] | None:
    sub = results_df[results_df["feature_set"] == feature_set_key]
    if sub.empty:
        return None
    best = sub.loc[sub["R2_cv_mean"].idxmax()]
    return float(best["R2_cv_mean"]), str(best["model"])


def _print_yield_phase_comparison(results_df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> None:
    """Сравнение по фазам для yield: phaseN_indices, all_phases_indices, combined."""
    print(f"\n{'=' * 60}")
    print("Сравнение сценариев по урожайности (yield): фазы и объединение")
    print(f"{'=' * 60}")

    phase_keys = yield_phase_index_feature_keys(feature_sets)
    phase_best: list[tuple[str, float, str]] = []
    for key in phase_keys:
        r = _best_cv_per_feature_set(results_df, key)
        if r is None:
            continue
        cv_mean, model = r
        phase_best.append((key, cv_mean, model))

    if not phase_best:
        print("\nНет данных по фазам (пустые результаты или отсутствуют наборы phase*_indices).")
        return

    phase_best.sort(key=lambda x: x[1], reverse=True)
    print("\nЛучший R2_cv_mean среди отдельных фаз:")
    for k, cv, m in phase_best:
        print(f"  {k}: R2_cv_mean={cv:.4f} (лучшая модель: {m})")

    best_key, best_cv, best_model = phase_best[0]
    print(f"\n→ Лучшая фаза по CV: {best_key} (R2_cv_mean={best_cv:.4f}, модель: {best_model})")

    all_ph = _best_cv_per_feature_set(results_df, "all_phases_indices")
    comb = _best_cv_per_feature_set(results_df, "combined")

    if all_ph:
        print(f"\nВсе фазы сразу (all_phases_indices): лучший R2_cv_mean={all_ph[0]:.4f} ({all_ph[1]})")
        if best_cv > all_ph[0]:
            print(
                f"Отдельная фаза {best_key} даёт более высокий R2_cv_mean, чем объединение всех индексов."
            )
        elif all_ph[0] > best_cv:
            print(
                "Объединение индексов всех фаз (all_phases_indices) улучшает R2_cv_mean "
                f"относительно лучшей одиночной фазы ({best_key})."
            )
        else:
            print("Качество all_phases_indices близко к лучшей одиночной фазе.")

    if comb:
        print(f"\nКомбинированный набор (combined): лучший R2_cv_mean={comb[0]:.4f} ({comb[1]})")


def run_analysis(
    *,
    input_path: str | Path,
    target: str,
    device_features: list[str],
    index_features: list[str],
    task: str,
    plots_dir: str | Path,
    results_dir: str | Path,
    random_state: int = 42,
) -> None:
    """Корреляции, scatter, регрессия и/или классификация по значению task."""
    validate_input_path(input_path)
    validate_target_name(target)

    dev = _dedupe_preserve(list(device_features))
    idx = _dedupe_preserve(list(index_features))
    overlap = set(dev) & set(idx)
    if overlap:
        raise PhaseDataError(
            "Один и тот же столбец указан и в --device-features, и в --index-features: "
            + ", ".join(sorted(overlap))
        )
    if target in dev or target in idx:
        raise PhaseDataError(
            f"Целевая переменная '{target}' не должна входить в список признаков (device/index)."
        )

    for c in idx:
        if index_prefix_before_phase(c) == "N_test":
            raise PhaseDataError(
                f"Столбец '{c}' относится к показаниям прибора: укажите его в --device-features, "
                "а не в --index-features (иначе конфликт имени набора N_test_only)."
            )

    df_raw = load_yield_table(input_path)
    df = prepare_analysis_frame(
        df_raw,
        target_col=target,
        device_features=dev,
        index_features=idx,
    )

    feature_sets = build_feature_sets(target=target, device_features=dev, index_features=idx)
    corr_cols = _union_device_index(dev, idx)

    results_root = Path(results_dir)
    plots_root = Path(plots_dir)
    target_dir = results_root / target
    pear_col = pearson_column_name(target)

    _print_scenario_header(f"Сценарий {target}")
    print(f"\nЦелевая переменная: {target}")
    print(f"Признаки прибора: {', '.join(dev) if dev else '(не заданы)'}")
    print(f"Индексы: {', '.join(idx) if idx else '(не заданы)'}")
    _print_formed_feature_sets(feature_sets)

    print_dataset_overview(
        f"Датасет после подготовки (числовые столбцы, без пропусков по target и признакам)",
        df,
    )

    run_regression = task in {"regression", "both"}
    run_classification = task in {"classification", "both"}

    if run_regression:
        corr_path = target_dir / f"correlations_{target}.xlsx"
        corr_df = correlation_analysis(
            df,
            feature_cols=corr_cols,
            target_col=target,
            pearson_col_name=pear_col,
            out_path=corr_path,
        )
        _print_correlations_block(target, corr_df, pear_col)

        plot_scatter_features_vs_target(
            df,
            feature_cols=corr_cols,
            target_col=target,
            plots_dir=plots_root,
            scatter_subdir=scatter_subdir_for_target(target),
        )

        results_reg = train_models(
            df,
            target_col=target,
            feature_sets=feature_sets,
            random_state=random_state,
            predictions_plots_dir=plots_root / predictions_subdir_for_target(target),
            target_display_name=target,
        )
        save_model_results(results_reg, target_dir / f"model_results_{target}.xlsx")
        _print_regression_summary(target, results_reg)

        if target == "yield":
            _print_yield_phase_comparison(results_reg, feature_sets)

    if run_classification:
        class_dir = results_root / f"{target}_class"
        results_clf = train_classification_models(
            df,
            feature_sets=feature_sets,
            target_col=target,
            random_state=random_state,
        )
        out_class = class_dir / f"model_results_{target}_class.xlsx"
        save_model_results(results_clf, out_class)
        print_classification_best_models(
            results_clf,
            f"Классификация {target}_class (низкий/высокий уровень по медиане {target})",
        )

    print(f"\n{'=' * 60}")
    print("Сохранённые результаты")
    print(f"{'=' * 60}")
    if run_regression:
        print(f"  {target_dir.resolve()}/ — correlations_{target}.xlsx, model_results_{target}.xlsx")
    if run_classification:
        print(
            f"  {(results_root / f'{target}_class').resolve()}/ — "
            f"model_results_{target}_class.xlsx"
        )
    if run_regression:
        print(
            f"\nГрафики: {plots_root.resolve()}/ — "
            f"{scatter_subdir_for_target(target)}/, {predictions_subdir_for_target(target)}/"
        )
    elif run_classification:
        print(
            f"\nГрафики: для режима --task classification scatter и predicted-vs-actual "
            f"не строятся (см. только Excel в results)."
        )


def run_all(
    input_path: str | Path,
    plots_dir: str | Path,
    random_state: int = 42,
) -> None:
    raise RuntimeError(
        "run_all() устарел: используйте CLI main_phase.py с параметрами "
        "--target, --device-features, --index-features, --task, --results-dir."
    )
