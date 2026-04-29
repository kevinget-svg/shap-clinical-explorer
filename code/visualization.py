"""
Publication-ready SHAP visualizations for clinical trial analysis.

All figures conform to the style specifications defined in CLAUDE.md Section 5.2.
Output: PNG (300 DPI) + SVG (vector) for every figure.

Filename convention:
    {design}_{endpoint}_{plot_type}.{fmt}
    e.g. RCT_BINARY_beeswarm.png, SIG_CONTINUOUS_summary_bar.svg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from shared.config import (
    CLINICAL_COLORS,
    SHAP_COLORMAP,
    FIGURE_DPI,
    FIGURE_FORMATS,
    setup_matplotlib_style,
    TrialDesign,
    EndpointType,
)

logger = logging.getLogger(__name__)

# Ensure rcParams are applied when this module is loaded
setup_matplotlib_style()


# ---------------------------------------------------------------------------
# Design/Endpoint label mapping for filenames
# ---------------------------------------------------------------------------
_DESIGN_LABEL = {
    TrialDesign.SINGLE_ARM: "SIG",
    TrialDesign.RCT_TWO_ARM: "RCT",
    TrialDesign.PARALLEL_MULTI_COHORT: "MULTI",
}

_ENDPOINT_LABEL = {
    EndpointType.CONTINUOUS: "CONT",
    EndpointType.BINARY: "BINARY",
    EndpointType.SURVIVAL: "SURV",
    EndpointType.COUNT: "COUNT",
}


def make_prefix(design: TrialDesign, endpoint: EndpointType) -> str:
    """Build filename prefix from design and endpoint type.

    >>> make_prefix(TrialDesign.RCT_TWO_ARM, EndpointType.BINARY)
    'RCT_BINARY'
    """
    d = _DESIGN_LABEL.get(design, design.value.upper())
    e = _ENDPOINT_LABEL.get(endpoint, endpoint.value.upper())
    return f"{d}_{e}"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _save(
    fig: plt.Figure,
    prefix: str,
    plot_name: str,
    output_dir: str | Path,
) -> None:
    """Save figure as {prefix}_{plot_name}.{fmt} in output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{plot_name}"
    for fmt in FIGURE_FORMATS:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.05)
        logger.info(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Beeswarm (Summary Plot)
# ---------------------------------------------------------------------------
def plot_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    prefix: str,
) -> plt.Figure:
    """SHAP Beeswarm plot showing feature impact distribution."""
    n_features = shap_values.shape[1]
    importance_order = np.argsort(np.abs(shap_values).mean(axis=0))

    fig, ax = plt.subplots(figsize=(8, max(3, n_features * 0.35)))

    shap_ordered = shap_values[:, importance_order]
    X_ordered = X[:, importance_order]
    names_ordered = [feature_names[i] for i in importance_order]

    colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        "shap_diverging", SHAP_COLORMAP
    )

    for i in range(n_features):
        y_pos = np.full(shap_ordered.shape[0], i)
        vals = shap_ordered[:, i]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        sc = ax.scatter(
            vals, y_pos + jitter,
            c=X_ordered[:, i], cmap=colors,
            s=8, alpha=0.6, edgecolors="none",
        )

    ax.axvline(x=0, color=CLINICAL_COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(names_ordered)
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP Beeswarm Plot", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.12, aspect=40)
    cbar.set_label("Feature value")

    fig.tight_layout()
    _save(fig, prefix, "beeswarm", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Summary Bar Plot
# ---------------------------------------------------------------------------
def plot_summary_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    prefix: str,
) -> plt.Figure:
    """Horizontal bar chart of mean(|SHAP|) per feature, ranked descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names) * 0.35)))
    bars = ax.barh(sorted_names, sorted_vals, height=0.6,
                   color=CLINICAL_COLORS["primary"])

    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

    ax.set_xlabel("Mean(|SHAP value|)")
    ax.set_title("SHAP Feature Importance", fontsize=12, fontweight="bold")
    ax.set_xlim(0, sorted_vals.max() * 1.2)

    fig.tight_layout()
    _save(fig, prefix, "summary_bar", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Dependence Plot
# ---------------------------------------------------------------------------
def plot_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    prefix: str,
    target_feature: str = "ARM",
    interaction_feature: Optional[str] = None,
) -> plt.Figure:
    """Dependence plot for a single feature, colored by an interaction feature."""
    if target_feature not in feature_names:
        raise ValueError(f"Feature '{target_feature}' not found in feature_names")

    idx = feature_names.index(target_feature)
    feat_vals = X[:, idx]

    if interaction_feature is None:
        mean_abs = np.abs(shap_values).mean(axis=0)
        other_indices = [i for i in range(len(feature_names)) if i != idx]
        interaction_idx = other_indices[np.argmax(mean_abs[other_indices])]
        interaction_feature = feature_names[interaction_idx]
    else:
        interaction_idx = feature_names.index(interaction_feature)

    colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        "shap_diverging", SHAP_COLORMAP
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(
        feat_vals, shap_values[:, idx],
        c=X[:, interaction_idx], cmap=colors,
        s=6, alpha=0.5, edgecolors="none",
    )
    ax.axhline(y=0, color=CLINICAL_COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xlabel(target_feature)
    ax.set_ylabel(f"SHAP value for {target_feature}")
    ax.set_title(f"SHAP Dependence: {target_feature}", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(interaction_feature)

    fig.tight_layout()
    _save(fig, prefix, f"dependence_{target_feature}", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Waterfall Plot (single-sample)
# ---------------------------------------------------------------------------
def plot_waterfall(
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_idx: int,
    output_dir: str | Path,
    prefix: str,
    base_value: float = 0.0,
    max_display: int = 10,
) -> plt.Figure:
    """Waterfall plot for a single test sample."""
    sample_shap = shap_values[sample_idx]
    order = np.argsort(np.abs(sample_shap))
    top_idx = order[-max_display:]

    top_names = [feature_names[i] for i in top_idx]
    top_vals = sample_shap[top_idx]

    fig, ax = plt.subplots(figsize=(8, max(3, max_display * 0.3)))

    labels = ["E[f(x)]"] + top_names
    colors_bar = [
        CLINICAL_COLORS["positive"] if v >= 0 else CLINICAL_COLORS["negative"]
        for v in top_vals
    ]

    left = base_value
    for i, (val, color) in enumerate(zip(top_vals, colors_bar)):
        ax.barh(i + 1, val, left=left, color=color, height=0.6, alpha=0.8)
        if val >= 0:
            ax.text(left + val + 0.02, i + 1, f"{val:+.3f}", va="center", fontsize=7)
        else:
            ax.text(left + val - 0.02, i + 1, f"{val:+.3f}", va="center",
                    ha="right", fontsize=7)
        left += val

    ax.barh(0, base_value, color=CLINICAL_COLORS["neutral"], height=0.6, alpha=0.5)
    ax.text(base_value + 0.02, 0, f"{base_value:.3f}", va="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(x=base_value, color=CLINICAL_COLORS["neutral"],
               linestyle="--", linewidth=0.8)
    ax.set_xlabel("Model output")
    ax.set_title(f"SHAP Waterfall (sample #{sample_idx})",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save(fig, prefix, f"waterfall_sample{sample_idx}", output_dir)
    return fig


# ---------------------------------------------------------------------------
# RCT Comparison Plot
# ---------------------------------------------------------------------------
def plot_rct_comparison(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    treatment_col_idx: int,
    output_dir: str | Path,
    prefix: str,
) -> plt.Figure:
    """RCT-specific: compare mean(|SHAP|) between treatment and control arms."""
    arm_vals = X[:, treatment_col_idx]
    mask_control = arm_vals < 0.5
    mask_treatment = arm_vals >= 0.5

    control_imp = np.abs(shap_values[mask_control]).mean(axis=0)
    treat_imp = np.abs(shap_values[mask_treatment]).mean(axis=0)

    overall = (control_imp + treat_imp) / 2
    order = np.argsort(overall)
    names_ordered = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(3, len(feature_names) * 0.35)))
    y = np.arange(len(feature_names))
    height = 0.35

    ax.barh(y - height / 2, control_imp[order], height,
            color=CLINICAL_COLORS["control"], label="Control", alpha=0.85)
    ax.barh(y + height / 2, treat_imp[order], height,
            color=CLINICAL_COLORS["treatment"], label="Treatment", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(names_ordered)
    ax.set_xlabel("Mean(|SHAP value|)")
    ax.set_title("SHAP Feature Importance by Treatment Arm",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, prefix, "rct_comparison", output_dir)
    return fig


# ---------------------------------------------------------------------------
# ROC Curve (binary endpoint)
# ---------------------------------------------------------------------------
def plot_roc_curve(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> plt.Figure:
    """ROC curve for binary classification models.

    Parameters
    ----------
    model : trained classifier (must have predict_proba)
    X_test : np.ndarray
    y_test : np.ndarray
    output_dir : str | Path
    prefix : str
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=CLINICAL_COLORS["primary"], linewidth=1.2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=CLINICAL_COLORS["neutral"],
            linestyle="--", linewidth=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    _save(fig, prefix, "roc_curve", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Multi-panel summary (Beeswarm + Bar)
# ---------------------------------------------------------------------------
def plot_summary_panel(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    prefix: str,
) -> plt.Figure:
    """Combined panel: Beeswarm (left) + Summary Bar (right).

    Mimics the layout in the AKI prediction reference paper (Fig.3).
    """
    n_features = shap_values.shape[1]
    importance_order = np.argsort(np.abs(shap_values).mean(axis=0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n_features * 0.35)))

    shap_ordered = shap_values[:, importance_order]
    X_ordered = X[:, importance_order]
    names_ordered = [feature_names[i] for i in importance_order]

    colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        "shap_diverging", SHAP_COLORMAP
    )

    for i in range(n_features):
        y_pos = np.full(shap_ordered.shape[0], i)
        vals = shap_ordered[:, i]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        sc = ax1.scatter(
            vals, y_pos + jitter,
            c=X_ordered[:, i], cmap=colors,
            s=8, alpha=0.6, edgecolors="none",
        )
    ax1.axvline(x=0, color=CLINICAL_COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax1.set_yticks(range(n_features))
    ax1.set_yticklabels(names_ordered)
    ax1.set_xlabel("SHAP value")
    ax1.set_title("A. SHAP Beeswarm", fontsize=11, fontweight="bold", loc="left")

    mean_abs = np.abs(shap_ordered).mean(axis=0)
    bars = ax2.barh(names_ordered, mean_abs, height=0.6,
                    color=CLINICAL_COLORS["primary"])
    for bar, val in zip(bars, mean_abs):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7)
    ax2.set_xlabel("Mean(|SHAP value|)")
    ax2.set_title("B. Feature Importance", fontsize=11, fontweight="bold", loc="left")
    ax2.set_xlim(0, mean_abs.max() * 1.2)

    cbar_ax = fig.add_axes([0.15, -0.02, 0.35, 0.02])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Feature value")

    fig.tight_layout()
    _save(fig, prefix, "summary_panel", output_dir)
    return fig
