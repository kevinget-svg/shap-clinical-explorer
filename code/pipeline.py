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

import numpy as np
import pandas as pd

from shared.config import (
    SEED,
    OUTPUT_DIR,
    EndpointType,
    TrialDesign,
    setup_logging,
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
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Path to input data file (.sas7bdat / .RData / .xlsx / .csv)",
    )
    parser.add_argument(
        "-t", "--target", type=str, required=True,
        help="Target / endpoint column name (e.g. AVAL, outcome)",
    )
    parser.add_argument(
        "-e", "--endpoint", type=str, required=True,
        choices=[e.value for e in EndpointType],
        help="Endpoint type",
    )
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
        help="Model override: 'rf', 'xgb', 'glm', or 'auto' (auto-select based on endpoint)",
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
# Pipeline steps
# ---------------------------------------------------------------------------
def step_load(input_path: Path) -> pd.DataFrame:
    """[Step 1] Load and validate raw clinical data."""
    from code.data_loader import load_clinical_data, get_data_summary
    logger.info(f"Loading data: {input_path}")
    df = load_clinical_data(input_path)
    logger.info(get_data_summary(df))
    return df


def step_preprocess(
    df: pd.DataFrame,
    target_col: str,
    design: TrialDesign,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """[Step 2] ADaM-standard cleaning, feature engineering, train/test split.

    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    from code.preprocessing import Preprocessor, split_train_test
    logger.info(f"Preprocessing: target={target_col}, design={design.value}")

    preprocessor = Preprocessor()
    preprocessor.fit(df, target_col, design)

    # Separate target from features
    feature_df = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    # Transform features (impute + encode)
    X = preprocessor.transform(feature_df)
    feature_names = preprocessor.feature_names
    logger.info(f"Features after encoding: {len(feature_names)} — {feature_names}")

    # Train/test split (stratified by ARM if RCT)
    treatment_col = "ARM" if design == TrialDesign.RCT_TWO_ARM and "ARM" in X.columns else None
    X_train, X_test, y_train, y_test = split_train_test(X, y, treatment_col)

    return X_train, X_test, y_train, y_test, feature_names


def step_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    endpoint: EndpointType,
    model_choice: str,
) -> tuple[object, dict[str, float]]:
    """[Step 3] Model training + evaluation.

    Returns: (trained_model, metrics_dict)
    """
    from code.modeling import ModelTrainer
    logger.info(f"Modeling: endpoint={endpoint.value}, model={model_choice}")

    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train, model_type=model_choice, endpoint=endpoint)
    metrics = trainer.evaluate(X_test, y_test)

    logger.info(f"Test metrics: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    return model, metrics


def step_shap(
    model: object,
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """[Step 4] Compute SHAP values + feature importance.

    Returns: (shap_values, shap_df, importance_df)
    """
    from code.shap_analysis import SHAPAnalyzer
    logger.info("Computing SHAP values")

    analyzer = SHAPAnalyzer()
    shap_values = analyzer.compute(model, X_train, X_test, model_type=model_type)
    importance = analyzer.get_feature_importance(feature_names)
    shap_df = analyzer.get_shap_dataframe(feature_names)

    logger.info(f"Top 5 features:\n{importance.head(5).to_string(index=False)}")
    return shap_values, shap_df, importance


def step_visualize(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    importance_df: pd.DataFrame,
    output_dir: Path,
    design: TrialDesign,
    model_type: str,
) -> None:
    """[Step 5] Generate publication-ready SHAP figures."""
    from code.visualization import (
        plot_beeswarm, plot_summary_bar, plot_dependence,
        plot_waterfall, plot_rct_comparison, plot_summary_panel,
    )
    logger.info(f"Generating visualizations → {output_dir}")

    # Always: Beeswarm + Summary Bar + Summary Panel
    plot_beeswarm(shap_values, X_test, feature_names, output_dir)
    plot_summary_bar(shap_values, feature_names, output_dir)
    plot_summary_panel(shap_values, X_test, feature_names, output_dir)

    # Top feature dependence plot
    top_features = importance_df["feature"].head(3).tolist()
    for feat in top_features:
        plot_dependence(shap_values, X_test, feature_names, output_dir,
                        target_feature=feat)

    # Waterfall for a representative sample (first test sample)
    plot_waterfall(shap_values, feature_names, sample_idx=0,
                   output_dir=output_dir, max_display=10)

    # RCT-specific: treatment arm comparison
    if design == TrialDesign.RCT_TWO_ARM:
        treatment_idx = feature_names.index("ARM") if "ARM" in feature_names else 0
        plot_rct_comparison(shap_values, X_test, feature_names,
                            treatment_col_idx=treatment_idx, output_dir=output_dir)


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
            df, args.target, design
        )

        # [3] Model
        model, metrics = step_model(
            X_train, X_test, y_train, y_test, endpoint, args.model
        )

        # [4] SHAP
        shap_values, shap_df, importance_df = step_shap(
            model, X_train, X_test, args.model, feature_names
        )

        # Save SHAP values CSV
        shap_df.to_csv(args.output / "shap_values.csv", index=False)
        importance_df.to_csv(args.output / "feature_importance.csv", index=False)

        # [5] Visualize
        if not args.no_plot:
            step_visualize(
                shap_values, X_test, feature_names, importance_df,
                args.output, design, args.model,
            )

        logger.info("Pipeline completed successfully.")
        logger.info(f"  Output files in: {args.output}")

    except Exception:
        logger.exception("Pipeline failed with unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
