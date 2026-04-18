from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _safe_name(name: str) -> str:
    """Имя папки/файла без символов, запрещённых в Windows."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name).strip("._")


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    feature_set_name: str,
    out_dir: str | Path,
    *,
    target_display_name: str = "lab_N",
) -> Path:
    out_path_dir = Path(out_dir) / _safe_name(model_name)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[finite_mask]
    y_pred_f = y_pred[finite_mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_f, y_pred_f)

    if y_true_f.size and y_pred_f.size:
        vmin = float(np.min([y_true_f.min(), y_pred_f.min()]))
        vmax = float(np.max([y_true_f.max(), y_pred_f.max()]))
        plt.plot([vmin, vmax], [vmin, vmax])

    plt.xlabel(f"Actual {target_display_name}")
    plt.ylabel(f"Predicted {target_display_name}")
    plt.title(f"Predicted vs Actual: {model_name} ({feature_set_name})")
    plt.tight_layout()

    filename = f"pred_vs_actual_{_safe_name(model_name)}_{_safe_name(feature_set_name)}.png"
    out_path = out_path_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path

