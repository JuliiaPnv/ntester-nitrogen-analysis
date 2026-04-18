from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .excel_utils import save_excel_wait


def correlation_analysis(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    pearson_col_name: str,
    out_path: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in feature_cols:
        rows.append(
            {
                "feature": col,
                pearson_col_name: float(df[col].corr(df[target_col])),
            }
        )
    corr_df = pd.DataFrame(rows)
    corr_df["abs_corr"] = corr_df[pearson_col_name].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])
    save_excel_wait(corr_df, out_path)
    return corr_df


def plot_scatter_features_vs_target(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    plots_dir: str | Path,
    scatter_subdir: str,
) -> None:
    out_dir = Path(plots_dir) / scatter_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in feature_cols:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[col], df[target_col])
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        out_path = out_dir / f"scatter_{col}_vs_{target_col}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
