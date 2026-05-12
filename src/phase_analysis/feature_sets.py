from __future__ import annotations

import re
from collections import defaultdict

# Суффикс фазы в имени столбца: ..._1, ..._12 и т.д. (последняя группа _<число> в конце)
_PHASE_SUFFIX_RE = re.compile(r"_(\d+)$")


def parse_phase_suffix(column: str) -> int | None:
    """Извлекает номер фазы из суффикса ``_<n>`` в конце имени столбца."""
    m = _PHASE_SUFFIX_RE.search(column)
    if not m:
        return None
    return int(m.group(1))


def index_prefix_before_phase(column: str) -> str | None:
    """Префикс до суффикса фазы: ``NDVI_3`` → ``NDVI``; без суффикса → ``None``."""
    m = _PHASE_SUFFIX_RE.search(column)
    if not m:
        return None
    prefix = column[: m.start()]
    return prefix if prefix else None


def validate_target_name(target: str) -> None:
    """
    Правила из ТЗ:
    - урожайность: ровно ``yield``;
    - макроэлемент / показатель по фазе: имя с суффиксом ``_<номер>`` (например ``N_1``, ``P_2``).
    """
    if target == "yield":
        return
    if target.lower().startswith("yield") and target != "yield":
        raise ValueError(
            f"Некорректное имя цели '{target}': для урожайности укажите ровно --target yield "
            "(без фазового суффикса, например не yield_1)."
        )
    if parse_phase_suffix(target) is None:
        raise ValueError(
            f"Некорректное имя цели '{target}': для макроэлемента ожидается суффикс фазы "
            "в формате …_<номер> (например N_1, P_1, K_2)."
        )


def _order_map(columns: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(columns)}


def _sort_by_phase_then_column_order(cols: list[str], order: dict[str, int]) -> list[str]:
    return sorted(cols, key=lambda c: (parse_phase_suffix(c) if parse_phase_suffix(c) is not None else 10**9, order.get(c, 0)))


def _drop_empty_sets(fs: dict[str, list[str]]) -> dict[str, list[str]]:
    return {k: v for k, v in fs.items() if v}


def build_macro_element_feature_sets(
    device_features: list[str],
    index_features: list[str],
) -> dict[str, list[str]]:
    """
    Наборы для макроэлемента (одна фаза в имени target, признаки задаёт пользователь):
    N_test_only, <префикс>_only для каждого типа индекса, indexes_only, combined.
    Порядок наборов *_only совпадает с порядком первого появления типа в ``index_features``.
    """
    fs: dict[str, list[str]] = {}
    fs["N_test_only"] = list(device_features)

    # ключ f"{base}_only" → список столбцов; key_order — порядок вставки как в ТЗ
    groups: dict[str, list[str]] = {}
    key_order: list[str] = []
    for col in index_features:
        pref = index_prefix_before_phase(col)
        base = pref if pref is not None else col
        key = f"{base}_only"
        if key not in groups:
            groups[key] = []
            key_order.append(key)
        groups[key].append(col)

    order = _order_map(index_features)
    for key in key_order:
        fs[key] = _sort_by_phase_then_column_order(groups[key], order)

    if index_features:
        fs["indexes_only"] = list(index_features)

    fs["combined"] = list(device_features) + list(index_features)
    return _drop_empty_sets(fs)


def build_yield_feature_sets(
    device_features: list[str],
    index_features: list[str],
) -> dict[str, list[str]]:
    """
    Наборы для ``yield``: N_test_only, N_test_phase*, *_only по типам индексов,
    для каждого типа индекса с фазой — отдельные наборы ``{префикс}_phase{n}`` (например
    ``NDVI_phase1`` = ``[NDVI_1]``, …), затем ``phase*_indices``, ``all_phases_indices``,
    ``phase*_combined``, ``combined``.
    Столбцы без суффикса фазы не попадают в фазовые поднаборы, но остаются в ``combined`` и
    в конце ``all_phases_indices`` (после фазовых индексов).
    """
    fs: dict[str, list[str]] = {}
    fs["N_test_only"] = list(device_features)

    dev_by_phase: dict[int, list[str]] = defaultdict(list)
    dev_order = _order_map(device_features)
    for c in device_features:
        ph = parse_phase_suffix(c)
        if ph is not None:
            dev_by_phase[ph].append(c)
    for ph in sorted(dev_by_phase):
        fs[f"N_test_phase{ph}"] = _sort_by_phase_then_column_order(dev_by_phase[ph], dev_order)

    idx_order = _order_map(index_features)
    unphased_cols = [c for c in index_features if parse_phase_suffix(c) is None]

    # *_only по типам индексов; порядок наборов — как первое появление типа в index_features
    prefix_groups: dict[str, list[str]] = {}
    prefix_key_order: list[str] = []
    for c in index_features:
        pref = index_prefix_before_phase(c)
        if pref is None:
            key = f"{c}_only"
        else:
            key = f"{pref}_only"
        if key not in prefix_groups:
            prefix_groups[key] = []
            prefix_key_order.append(key)
        prefix_groups[key].append(c)
    for key in prefix_key_order:
        cols_in_group = prefix_groups[key]
        fs[key] = _sort_by_phase_then_column_order(cols_in_group, idx_order)
        # Фазовые наборы по одному индексу на фазу: NDVI_phase1 = [NDVI_1], …
        if not key.endswith("_only"):
            continue
        base = key[: -len("_only")]
        if not any(parse_phase_suffix(c) is not None for c in cols_in_group):
            continue
        by_ph: dict[int, list[str]] = defaultdict(list)
        for c in cols_in_group:
            ph = parse_phase_suffix(c)
            pref = index_prefix_before_phase(c)
            if ph is not None and pref == base:
                by_ph[ph].append(c)
        for ph in sorted(by_ph):
            fs[f"{base}_phase{ph}"] = sorted(by_ph[ph], key=lambda c: idx_order.get(c, 0))

    phases: set[int] = set(dev_by_phase)
    for c in index_features:
        p = parse_phase_suffix(c)
        if p is not None:
            phases.add(p)

    for ph in sorted(phases):
        cols_ph = [c for c in index_features if parse_phase_suffix(c) == ph]
        if cols_ph:
            fs[f"phase{ph}_indices"] = sorted(cols_ph, key=lambda c: idx_order.get(c, 0))

    all_idx: list[str] = []
    for ph in sorted(phases):
        all_idx.extend([c for c in index_features if parse_phase_suffix(c) == ph])
    all_idx.extend(unphased_cols)
    fs["all_phases_indices"] = all_idx

    for ph in sorted(phases):
        dcols = dev_by_phase.get(ph, [])
        icols = [c for c in index_features if parse_phase_suffix(c) == ph]
        if dcols or icols:
            d_sorted = sorted(dcols, key=lambda c: dev_order.get(c, 0))
            i_sorted = sorted(icols, key=lambda c: idx_order.get(c, 0))
            fs[f"phase{ph}_combined"] = d_sorted + i_sorted

    fs["combined"] = list(device_features) + list(index_features)
    return _drop_empty_sets(fs)


def build_feature_sets(
    *,
    target: str,
    device_features: list[str],
    index_features: list[str],
) -> dict[str, list[str]]:
    """Строит словарь feature_set → список столбцов в зависимости от типа цели."""
    if target == "yield":
        return build_yield_feature_sets(device_features, index_features)
    return build_macro_element_feature_sets(device_features, index_features)


def yield_phase_index_feature_keys(feature_sets: dict[str, list[str]]) -> list[str]:
    """Ключи ``phaseN_indices`` для сравнения фаз по CV (порядок: по номеру фазы)."""
    keys: list[tuple[int, str]] = []
    for k in feature_sets:
        if k.startswith("phase") and k.endswith("_indices"):
            mid = k[len("phase") : -len("_indices")]
            if mid.isdigit():
                keys.append((int(mid), k))
    return [k for _, k in sorted(keys)]
