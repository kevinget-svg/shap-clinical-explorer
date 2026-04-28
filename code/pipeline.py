"""
SHAP Clinical Analysis Pipeline

Full pipeline from raw clinical data to SHAP-based visualization outputs.

Usage:
    python -m code.pipeline --input data/trial.sas7bdat --output output/ --target AVAL
    python -m code.pipeline --input data/trial.xlsx --output output/ --target outcome --design rct_2_arm

Architecture:
    Raw Data  →  [1] load  →  [2] preprocess  →  [3] model  →  [4] shap  →  [5] visualize  →  output/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from shared.config import (
    SEED,
    DATA_DIR,
    OUTPUT_DIR,
    EndpointType,
    TrialDesign,
    setup_logging,
    setup_matplotlib_style,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHAP Clinical Trial Exploratory Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m code.pipeline -i data/study.sas7bdat -t AVAL -e continuous
  python -m code.pipeline -i data/study.xlsx -t outcome -e binary -d rct_2_arm
  python -m code.pipeline -i data/study.sas7bdat -t AVAL -e survival -c CNSR -d single_arm
        """,
    )
    # Required
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Path to input data file (.sas7bdat / .RData / .xlsx / .csv)",
    )
    parser.add_argument(
        "-t", "--target", type=str, required=True,
        help="Target / endpoint column name (e.g. AVAL, outcome)",
    )
    # Endpoint type
    parser.add_argument(
        "-e", "--endpoint", type=str, required=True,
        choices=[e.value for e in EndpointType],
        help="Endpoint type",
    )
    # Optional
    parser.add_argument(
        "-c", "--censor", type=str, default=None,
        help="Censor column name (required for survival endpoints; 0=event, 1=censored)",
    )
    parser.add_argument(
        "-d", "--design", type=str, default="single_arm",
        choices=[d.value for d in TrialDesign],
        help="Trial design type (default: single_arm)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default="auto",
        help="Model override: 'rf', 'xgb', 'cox', 'glm', or 'auto' (auto-select based on endpoint)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip visualization; only output SHAP values CSV",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline steps (placeholders — implemented incrementally)
# ---------------------------------------------------------------------------
def step_load(input_path: Path) -> "pd.DataFrame":
    """[Step 1] Load and validate raw clinical data."""
    logger.info(f"Loading data: {input_path}")
    # TODO: delegate to code.data_loader
    raise NotImplementedError("data_loader module not yet implemented")


def step_preprocess(df, target_col: str, censor_col: Optional[str], endpoint: EndpointType) -> tuple:
    """[Step 2] ADaM-standard cleaning and feature engineering."""
    logger.info(f"Preprocessing: target={target_col}, endpoint={endpoint.value}")
    # TODO: delegate to code.preprocessing
    raise NotImplementedError("preprocessing module not yet implemented")


def step_model(
    X_train, X_test, y_train, y_test,
    endpoint: EndpointType, model_choice: str
):
    """[Step 3] Train/test split → model fitting → (optional) hyperparameter tuning."""
    logger.info(f"Modeling: endpoint={endpoint.value}, model={model_choice}")
    # TODO: delegate to code.modeling
    raise NotImplementedError("modeling module not yet implemented")


def step_shap(model, X_train, X_test, endpoint: EndpointType) -> "pd.DataFrame":
    """[Step 4] Compute SHAP values and feature importance ranking."""
    logger.info("Computing SHAP values")
    # TODO: delegate to code.shap_analysis
    raise NotImplementedError("shap_analysis module not yet implemented")


def step_visualize(shap_df, output_dir: Path, endpoint: EndpointType, design: TrialDesign) -> None:
    """[Step 5] Generate publication-ready figures."""
    logger.info(f"Generating visualizations → {output_dir}")
    # TODO: delegate to code.visualization
    raise NotImplementedError("visualization module not yet implemented")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    endpoint = EndpointType(args.endpoint)
    design = TrialDesign(args.design)

    # Validate: survival requires censor column
    if endpoint == EndpointType.SURVIVAL and not args.censor:
        logger.error("--censor is required for survival endpoints")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"SHAP Pipeline: {args.input.name}")
    logger.info(f"  Endpoint: {endpoint.value}  |  Design: {design.value}  |  Seed: {SEED}")
    logger.info(f"  Output:   {args.output}")
    logger.info("=" * 60)

    try:
        # [1] Load
        df = step_load(args.input)

        # [2] Preprocess
        X_train, X_test, y_train, y_test, feature_names = step_preprocess(
            df, args.target, args.censor, endpoint
        )

        # [3] Model
        model = step_model(X_train, X_test, y_train, y_test, endpoint, args.model)

        # [4] SHAP
        shap_df = step_shap(model, X_train, X_test, endpoint)

        # [5] Visualize
        if not args.no_plot:
            setup_matplotlib_style()
            step_visualize(shap_df, args.output, endpoint, design)

        logger.info("Pipeline completed successfully.")

    except NotImplementedError as e:
        logger.warning(f"Pipeline step not yet implemented: {e}")
        logger.info("Run 'git status' to check development progress.")
        sys.exit(0)
    except Exception:
        logger.exception("Pipeline failed with unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
