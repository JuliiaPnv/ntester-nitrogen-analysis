from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .constants import CORR_FEATURES, TARGET_COL
from .excel_utils import save_excel_wait


def correlation_analysis(df: pd.DataFrame, out_path: str | Path = "results/correlations_labN.xlsx") -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for col in CORR_FEATURES:
        rows.append(
            {
                "feature": col,
                "pearson_corr_with_lab_N": float(df[col].corr(df[TARGET_COL])),
            }
        )
    corr_df = pd.DataFrame(rows)
    corr_df["abs_corr"] = corr_df["pearson_corr_with_lab_N"].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])
    save_excel_wait(corr_df, out_path)
    return corr_df


def plot_data(df: pd.DataFrame, plots_dir: str | Path = "plots") -> None:
    out_dir = Path(plots_dir) / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in CORR_FEATURES:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[col], df[TARGET_COL])
        plt.xlabel(col)
        plt.ylabel(TARGET_COL)
        plt.title(f"{col} vs {TARGET_COL}")
        plt.tight_layout()
        out_path = out_dir / f"scatter_{col}_vs_{TARGET_COL}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

