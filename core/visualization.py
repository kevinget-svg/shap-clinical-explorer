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
    SEED,
    BEESWARM_POINT_SIZE,
    BEESWARM_ALPHA,
    BEESWARM_JITTER,
    DEPENDENCE_POINT_SIZE,
    DEPENDENCE_ALPHA,
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

    fig, ax = plt.subplots(figsize=(8, 4.5))

    shap_ordered = shap_values[:, importance_order]
    X_ordered = X[:, importance_order]
    names_ordered = [feature_names[i] for i in importance_order]

    colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        "shap_diverging", SHAP_COLORMAP
    )

    for i in range(n_features):
        y_pos = np.full(shap_ordered.shape[0], i)
        vals = shap_ordered[:, i]
        jitter = np.random.default_rng(SEED).uniform(
            -BEESWARM_JITTER, BEESWARM_JITTER, len(vals))
        sc = ax.scatter(
            vals, y_pos + jitter,
            c=X_ordered[:, i], cmap=colors,
            s=BEESWARM_POINT_SIZE, alpha=BEESWARM_ALPHA, edgecolors="none",
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(
        feat_vals, shap_values[:, idx],
        c=X[:, interaction_idx], cmap=colors,
        s=DEPENDENCE_POINT_SIZE, alpha=DEPENDENCE_ALPHA, edgecolors="none",
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

    fig, ax = plt.subplots(figsize=(8, 4.5))

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
    # Align rows (SHAP may be subsampled vs X, e.g. survshap)
    n = min(shap_values.shape[0], X.shape[0])
    shap_aligned = shap_values[:n]
    X_aligned = X[:n]

    arm_vals = X_aligned[:, treatment_col_idx]
    mask_control = arm_vals < 0.5
    mask_treatment = arm_vals >= 0.5

    control_imp = np.abs(shap_aligned[mask_control]).mean(axis=0)
    treat_imp = np.abs(shap_aligned[mask_treatment]).mean(axis=0)

    overall = (control_imp + treat_imp) / 2
    order = np.argsort(overall)
    names_ordered = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
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
# Count endpoint: Observed vs Predicted by Arm
# ---------------------------------------------------------------------------
def plot_count_obs_vs_pred(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    prefix: str,
) -> plt.Figure:
    """Grouped bar chart: observed vs predicted mean count by treatment arm.

    Parameters
    ----------
    model : trained count model (must have .predict)
    X_test : np.ndarray
    y_test : np.ndarray
        Actual observed event counts.
    feature_names : list[str]
    output_dir : str | Path
    prefix : str
    """
    y_pred = np.clip(model.predict(X_test), 0, None)
    y_test = np.asarray(y_test, dtype=float)

    # Find treatment arm column
    arm_idx = feature_names.index("ARM") if "ARM" in feature_names else 0
    arm_vals = X_test[:, arm_idx]
    mask_ctrl = arm_vals < 0.5
    mask_trt = arm_vals >= 0.5

    groups = ["Control", "Treatment"]
    obs_means = [y_test[mask_ctrl].mean(), y_test[mask_trt].mean()]
    pred_means = [y_pred[mask_ctrl].mean(), y_pred[mask_trt].mean()]

    # SEM for error bars
    from scipy import stats as sp_stats
    obs_sem = [
        sp_stats.sem(y_test[mask_ctrl]) if mask_ctrl.sum() > 1 else 0,
        sp_stats.sem(y_test[mask_trt]) if mask_trt.sum() > 1 else 0,
    ]
    pred_sem = [
        sp_stats.sem(y_pred[mask_ctrl]) if mask_ctrl.sum() > 1 else 0,
        sp_stats.sem(y_pred[mask_trt]) if mask_trt.sum() > 1 else 0,
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(x - width / 2, obs_means, width,
                   color=CLINICAL_COLORS["primary"], label="Observed",
                   yerr=obs_sem, capsize=4, alpha=0.85)
    bars2 = ax.bar(x + width / 2, pred_means, width,
                   color=CLINICAL_COLORS["secondary"], label="Predicted",
                   yerr=pred_sem, capsize=4, alpha=0.85)

    # Annotate bar values
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean Event Count")
    ax.set_title("Observed vs Predicted Count by Treatment Arm",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")

    # Annotate rate ratio
    rr = obs_means[1] / (obs_means[0] + 1e-10)
    ax.text(0.98, 0.05,
            f"Observed Rate Ratio\n(Trt / Ctrl) = {rr:.3f}",
            transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.5})

    fig.tight_layout()
    _save(fig, prefix, "count_obs_vs_pred", output_dir)
    return fig


def plot_count_panel(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model: object,
    output_dir: str | Path,
    prefix: str,
) -> plt.Figure:
    """Count-specific panel: calibration (left) + obs/pred bar (right).

    Left: Observed vs Predicted scatter colored by treatment arm,
          with y=x reference and LOESS trend.
    Right: Grouped bar chart of mean observed/predicted count by arm.

    16:9 aspect ratio.
    """
    y_pred = np.clip(model.predict(X_test), 0, None)
    y_test = np.asarray(y_test, dtype=float)

    arm_idx = feature_names.index("ARM") if "ARM" in feature_names else 0
    arm_vals = X_test[:, arm_idx]
    mask_ctrl = arm_vals < 0.5
    mask_trt = arm_vals >= 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # --- Left: Calibration scatter ---
    ax1.scatter(y_pred[mask_ctrl], y_test[mask_ctrl],
                c=CLINICAL_COLORS["control"], label="Control",
                s=30, alpha=0.6, edgecolors="none")
    ax1.scatter(y_pred[mask_trt], y_test[mask_trt],
                c=CLINICAL_COLORS["treatment"], label="Treatment",
                s=30, alpha=0.6, edgecolors="none")

    # y=x reference
    lims = [0, max(y_test.max(), y_pred.max()) * 1.1]
    ax1.plot(lims, lims, color=CLINICAL_COLORS["neutral"],
             linestyle="--", linewidth=0.8, label="Perfect calibration")

    # LOESS trend (combined)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = np.argsort(y_pred)
        smoothed = lowess(y_test[sorted_idx], y_pred[sorted_idx], frac=0.5)
        ax1.plot(smoothed[:, 0], smoothed[:, 1],
                 color="black", linewidth=1.2, label="LOESS trend")
    except Exception:
        pass

    ax1.set_xlabel("Predicted Count")
    ax1.set_ylabel("Observed Count")
    ax1.set_title("A. Calibration: Observed vs Predicted",
                  fontsize=11, fontweight="bold", loc="left")
    ax1.legend(fontsize=7, loc="upper left")

    # --- Right: Observed vs Predicted bar by arm ---
    groups = ["Control", "Treatment"]
    obs_means = [y_test[mask_ctrl].mean(), y_test[mask_trt].mean()]
    pred_means = [y_pred[mask_ctrl].mean(), y_pred[mask_trt].mean()]

    from scipy import stats as sp_stats
    obs_sem = [
        sp_stats.sem(y_test[mask_ctrl]) if mask_ctrl.sum() > 1 else 0,
        sp_stats.sem(y_test[mask_trt]) if mask_trt.sum() > 1 else 0,
    ]
    pred_sem = [
        sp_stats.sem(y_pred[mask_ctrl]) if mask_ctrl.sum() > 1 else 0,
        sp_stats.sem(y_pred[mask_trt]) if mask_trt.sum() > 1 else 0,
    ]

    x = np.arange(len(groups))
    width = 0.35
    ax2.bar(x - width / 2, obs_means, width,
            color=CLINICAL_COLORS["primary"], label="Observed",
            yerr=obs_sem, capsize=4, alpha=0.85)
    ax2.bar(x + width / 2, pred_means, width,
            color=CLINICAL_COLORS["secondary"], label="Predicted",
            yerr=pred_sem, capsize=4, alpha=0.85)

    # Annotate values
    for i, (om, pm) in enumerate(zip(obs_means, pred_means)):
        ax2.text(i - width / 2, om + 0.05, f"{om:.2f}",
                 ha="center", va="bottom", fontsize=8)
        ax2.text(i + width / 2, pm + 0.05, f"{pm:.2f}",
                 ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel("Mean Event Count")
    ax2.set_title("B. Observed vs Predicted Count by Arm",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.legend(fontsize=7, loc="upper right")

    rr_obs = obs_means[1] / (obs_means[0] + 1e-10)
    rr_pred = pred_means[1] / (pred_means[0] + 1e-10)
    ax2.text(0.98, 0.05,
             f"Obs RR = {rr_obs:.3f}\nPred RR = {rr_pred:.3f}",
             transform=ax2.transAxes, fontsize=8,
             ha="right", va="bottom",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.5})

    fig.tight_layout()
    _save(fig, prefix, "count_panel", output_dir)
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
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
    # Align X rows with shap_values (may differ when survshap subsamples)
    n_samples = shap_values.shape[0]
    if X.shape[0] != n_samples:
        X = X[:n_samples]

    importance_order = np.argsort(np.abs(shap_values).mean(axis=0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    shap_ordered = shap_values[:, importance_order]
    X_ordered = X[:, importance_order]
    names_ordered = [feature_names[i] for i in importance_order]

    colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        "shap_diverging", SHAP_COLORMAP
    )

    for i in range(n_features):
        y_pos = np.full(shap_ordered.shape[0], i)
        vals = shap_ordered[:, i]
        jitter = np.random.default_rng(SEED).uniform(
            -BEESWARM_JITTER, BEESWARM_JITTER, len(vals))
        sc = ax1.scatter(
            vals, y_pos + jitter,
            c=X_ordered[:, i], cmap=colors,
            s=BEESWARM_POINT_SIZE, alpha=BEESWARM_ALPHA, edgecolors="none",
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


# ---------------------------------------------------------------------------
# SurvSHAP(t) Time-Dependent Plot (survival endpoint)
# ---------------------------------------------------------------------------
def plot_survshap_time(
    survshap_values: np.ndarray,
    feature_names: list[str],
    times: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    max_features: int = 8,
) -> plt.Figure:
    """Plot SurvSHAP(t) values over time for top features.

    Parameters
    ----------
    survshap_values : np.ndarray (n_features, n_times)
        Time-dependent SHAP values aggregated across samples.
    feature_names : list[str]
    times : np.ndarray (n_times,)
        Time grid points.
    output_dir : str | Path
    prefix : str
    max_features : int
        Number of top features to display (by mean absolute SHAP).
    """
    # Select top features by mean absolute SHAP across time
    mean_abs = np.abs(survshap_values).mean(axis=1)
    top_idx = np.argsort(mean_abs)[-max_features:]

    colors_cycle = [
        CLINICAL_COLORS["primary"], CLINICAL_COLORS["secondary"],
        CLINICAL_COLORS["positive"], CLINICAL_COLORS["negative"],
        CLINICAL_COLORS["treatment"], CLINICAL_COLORS["control"],
        CLINICAL_COLORS["neutral"], "#e377c2",
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, feat_idx in enumerate(top_idx):
        color = colors_cycle[j % len(colors_cycle)]
        ax.plot(times, survshap_values[feat_idx, :],
                linewidth=1.2, color=color,
                label=feature_names[feat_idx])

    ax.axhline(y=0, color=CLINICAL_COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("SurvSHAP(t)")
    ax.set_title("SurvSHAP(t) Time-Dependent Feature Effects",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    _save(fig, prefix, "survshap_time", output_dir)
    return fig


def plot_survshap_panel(
    survshap_values_3d: np.ndarray,
    feature_names: list[str],
    times: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    max_features: int = 8,
) -> plt.Figure:
    """Combined SurvSHAP(t) panel: time curves (left) + box plot (right).

    Left: Mean SurvSHAP(t) over time for top features (line plot).
    Right: Box plot of per-sample time-aggregated SHAP values,
           showing the distribution of each feature's contribution
           across subjects.

    Parameters
    ----------
    survshap_values_3d : np.ndarray (n_samples, n_features, n_times)
    feature_names : list[str]
    times : np.ndarray (n_times,)
    output_dir : str | Path
    prefix : str
    max_features : int
    """
    # Aggregate across samples for the time-plot (mean per feature per time)
    sv_mean = survshap_values_3d.mean(axis=0)  # (n_features, n_times)

    # Aggregate across time for the box plot (integral / mean per sample per feature)
    # Use mean over time as the per-sample aggregate
    sv_sample_agg = survshap_values_3d.mean(axis=2)  # (n_samples, n_features)

    # Select top features by overall mean absolute SHAP
    mean_abs = np.abs(sv_mean).mean(axis=1)
    top_idx = np.argsort(mean_abs)[-max_features:]

    colors_cycle = [
        CLINICAL_COLORS["primary"], CLINICAL_COLORS["secondary"],
        CLINICAL_COLORS["positive"], CLINICAL_COLORS["negative"],
        CLINICAL_COLORS["treatment"], CLINICAL_COLORS["control"],
        CLINICAL_COLORS["neutral"], "#e377c2",
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # Build consistent feature→color map (most important = colors_cycle[0])
    feature_color = {}
    for rank, feat_idx in enumerate(top_idx[::-1]):
        feature_color[feat_idx] = colors_cycle[rank % len(colors_cycle)]

    # --- Left: SurvSHAP(t) time curves ---
    for j, feat_idx in enumerate(top_idx):
        color = feature_color[feat_idx]
        ax1.plot(times, sv_mean[feat_idx, :],
                 linewidth=1.2, color=color,
                 label=feature_names[feat_idx])

    ax1.axhline(y=0, color=CLINICAL_COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mean SurvSHAP(t)")
    ax1.set_title("A. SurvSHAP(t) Time-Dependent Effects",
                  fontsize=11, fontweight="bold", loc="left")
    ax1.legend(fontsize=7, loc="upper right")

    # --- Right: Box plot (vertical, most important leftmost) ---
    box_data = []
    box_labels = []
    box_colors = []
    for feat_idx in top_idx[::-1]:  # most important → least important (left→right)
        box_data.append(sv_sample_agg[:, feat_idx])
        box_labels.append(feature_names[feat_idx])
        box_colors.append(feature_color[feat_idx])

    bp = ax2.boxplot(
        box_data, vert=True, patch_artist=True, widths=0.6,
        medianprops={"color": "black", "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 3,
                    "markerfacecolor": "gray", "alpha": 0.4},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
        boxprops={"linewidth": 0.6},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax2.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
    ax2.axhline(y=0, color=CLINICAL_COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Time-Aggregated SHAP Value")
    ax2.set_title("B. Per-Subject SHAP Distribution",
                  fontsize=11, fontweight="bold", loc="left")

    fig.tight_layout()
    _save(fig, prefix, "survshap_panel", output_dir)
    return fig


def plot_survshap_aggregated(
    survshap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    prefix: str,
    sample_idx: int = 0,
    times: Optional[np.ndarray] = None,
) -> plt.Figure:
    """SurvSHAP(t) aggregated decomposition for a single subject.

    Shows how each feature contributes to the survival function over time
    for a given individual — stacked area style.

    Parameters
    ----------
    survshap_values : np.ndarray (n_samples, n_features, n_times)
    X : np.ndarray
    feature_names : list[str]
    output_dir : str | Path
    prefix : str
    sample_idx : int
    times : np.ndarray (n_times,), optional
        Actual time grid. Defaults to linspace(0, 1, n_times).
    """
    sample_shap = survshap_values[sample_idx]  # (n_features, n_times)
    if times is None:
        times = np.linspace(0, 1, sample_shap.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Sort features by mean abs contribution
    order = np.argsort(np.abs(sample_shap).mean(axis=1))[::-1]
    top_n = min(8, len(order))
    top_order = order[:top_n]

    colors_cycle = [
        CLINICAL_COLORS["primary"], CLINICAL_COLORS["negative"],
        CLINICAL_COLORS["positive"], CLINICAL_COLORS["secondary"],
        CLINICAL_COLORS["treatment"], CLINICAL_COLORS["control"],
        CLINICAL_COLORS["neutral"], "#e377c2",
    ]

    cumsum = np.zeros(sample_shap.shape[1])
    for j, feat_idx in enumerate(top_order):
        color = colors_cycle[j % len(colors_cycle)]
        vals = sample_shap[feat_idx, :]
        ax.fill_between(times, cumsum, cumsum + vals,
                        color=color, alpha=0.25)
        ax.plot(times, cumsum + vals, color=color, linewidth=1.0,
                label=feature_names[feat_idx])
        cumsum += vals

    ax.axhline(y=0, color=CLINICAL_COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("SurvSHAP(t)")
    ax.set_title(f"SurvSHAP(t) Decomposition (sample #{sample_idx})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    _save(fig, prefix, f"survshap_decomp_sample{sample_idx}", output_dir)
    return fig
