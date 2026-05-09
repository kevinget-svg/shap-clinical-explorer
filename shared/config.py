"""Shared configuration for the SHAP Clinical Analysis project."""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CORE_DIR: Path = PROJECT_ROOT / "core"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
TRAIN_SIZE: float = 0.80
TEST_SIZE: float = 0.20
SMALL_SAMPLE_THRESHOLD: int = 200  # N below this triggers CV + auto-tuning
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Clinical trial endpoint types
# ---------------------------------------------------------------------------
from enum import StrEnum


class EndpointType(StrEnum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    SURVIVAL = "survival"
    COUNT = "count"


class TrialDesign(StrEnum):
    SINGLE_ARM = "single_arm"
    RCT_TWO_ARM = "rct_2_arm"
    PARALLEL_MULTI_COHORT = "parallel_multi_cohort"


# ---------------------------------------------------------------------------
# SHAP computation parameters
# ---------------------------------------------------------------------------
# Background sample size for TreeExplainer / KernelExplainer (non-survival)
SHAP_BG_SAMPLES: int = 100
# Background sample size for SurvSHAP(t)
SURVSHAP_BG_SAMPLES: int = 80
# Max test-set rows explained by SurvSHAP(t) (limits runtime)
SURVSHAP_MAX_SAMPLES: int = 50

# ---------------------------------------------------------------------------
# Publication-ready visualization settings
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 300
FIGURE_FORMATS: list[str] = ["png", "svg"]

# Beeswarm plot
BEESWARM_POINT_SIZE: int = 8
BEESWARM_POINT_SIZE_LARGE_N: int = 3
BEESWARM_LARGE_N_THRESHOLD: int = 1000
BEESWARM_ALPHA: float = 0.6
BEESWARM_JITTER: float = 0.15

# Dependence plot
DEPENDENCE_POINT_SIZE: int = 6
DEPENDENCE_ALPHA: float = 0.5

# Clinical color palette (color-blind friendly, journal-safe)
CLINICAL_COLORS: dict[str, str] = {
    "primary": "#1f77b4",    # blue
    "secondary": "#ff7f0e",  # orange
    "positive": "#2ca02c",   # green
    "negative": "#d62728",   # red
    "neutral": "#7f7f7f",    # gray
    "treatment": "#9467bd",  # purple
    "control": "#17becf",    # cyan
}

# SHAP Beeswarm 图使用的发散色 (低特征值→中位→高特征值)
SHAP_COLORMAP: list[str] = [
    CLINICAL_COLORS["negative"],   # 低特征值 → 红色
    CLINICAL_COLORS["neutral"],    # 中位 → 灰色
    CLINICAL_COLORS["primary"],    # 高特征值 → 蓝色
]


def setup_matplotlib_style() -> None:
    """Apply publication-ready matplotlib rcParams globally.

    Call once at the entry point of visualization.py before any plotting.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        # Output
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Fonts (Arial → SimSun → DejaVu Sans fallback chain)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "SimSun", "DejaVu Sans"],
        "font.size": 9,
        # Tick marks
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Axes
        "axes.linewidth": 0.5,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.loc": "best",
        # Grid: off by default
        "axes.grid": False,
    })

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_CONFIG: dict = {
    "level": logging.INFO,
    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}


def setup_logging() -> None:
    """Apply project-wide logging configuration."""
    logging.basicConfig(**LOG_CONFIG)
