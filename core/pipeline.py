"""
SHAP Clinical Analysis Pipeline

Full pipeline from raw clinical data to SHAP-based visualization outputs.

Usage:
    python -m core.pipeline --input data/trial.sas7bdat --output output/ --target AVAL
    python -m core.pipeline --input data/trial.xlsx --output output/ --target outcome --design rct_2_arm

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
  python -m core.pipeline -i data/study.sas7bdat -t AVAL -e continuous
  python -m core.pipeline -i data/study.xlsx -t outcome -e binary -d rct_2_arm
  python -m core.pipeline -i data/study.sas7bdat -t AVAL -e survival -c CNSR -d single_arm
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
    from core.data_loader import load_clinical_data, get_data_summary
    logger.info(f"Loading data: {input_path}")
    df = load_clinical_data(input_path)
    logger.info(get_data_summary(df))
    return df


def step_preprocess(
    df: pd.DataFrame,
    target_col: str,
    design: TrialDesign,
    censor_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """[Step 2] ADaM-standard cleaning, feature engineering, train/test split.

    Returns: X_train, X_test, y_train, y_test, feature_names, X_train_df, X_test_df
    The DataFrame versions are used by survshap (survival endpoint).
    """
    from core.preprocessing import Preprocessor, split_train_test
    logger.info(f"Preprocessing: target={target_col}, design={design.value}")

    preprocessor = Preprocessor()
    preprocessor.fit(df, target_col, design)

    feature_df = df.drop(columns=[target_col], errors="ignore")

    # For survival, also drop the censor column from features
    if censor_col and censor_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[censor_col])

    X = preprocessor.transform(feature_df)
    feature_names = preprocessor.feature_names
    logger.info(f"Features after encoding: {len(feature_names)} — {feature_names}")

    # Build target (structured array for survival, Series otherwise)
    if censor_col:
        from sksurv.util import Surv
        cnsr = df[censor_col].values.astype(bool)
        time = df[target_col].values.astype(float)
        y = Surv.from_arrays(event=~cnsr, time=time)
        logger.info(f"Survival y: {len(y)} samples, "
                    f"events={(~cnsr).sum()}, censored={cnsr.sum()}")
    else:
        y = df[target_col]

    # Train/test split (stratified by ARM if RCT)
    treatment_col = None
    if design == TrialDesign.RCT_TWO_ARM and "ARM" in X.columns:
        treatment_col = "ARM"

    X_train, X_test, y_train, y_test = split_train_test(X, y, treatment_col)

    # Keep DataFrame copies for survshap
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    return X_train, X_test, y_train, y_test, feature_names, X_train_df, X_test_df


def step_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    endpoint: EndpointType,
    model_choice: str,
) -> tuple[object, dict[str, float], str]:
    """[Step 3] Model training + evaluation.

    Returns: (trained_model, metrics_dict, actual_model_type)
    """
    from core.modeling import ModelTrainer
    logger.info(f"Modeling: endpoint={endpoint.value}, model={model_choice}")

    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train, model_type=model_choice, endpoint=endpoint)
    metrics = trainer.evaluate(X_test, y_test)
    actual_type = trainer.model_type_

    if endpoint == EndpointType.SURVIVAL:
        logger.info(f"Test C-index: {metrics['c_index']:.4f}")
    elif endpoint == EndpointType.BINARY:
        logger.info(f"Test metrics: ROC AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    elif endpoint == EndpointType.COUNT:
        logger.info(f"Test metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, D²={metrics['d2_pseudo_r2']:.4f}")
    else:
        logger.info(f"Test metrics: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    return model, metrics, actual_type


def step_shap(
    model: object,
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str,
    feature_names: list[str],
    endpoint: EndpointType = EndpointType.CONTINUOUS,
    X_train_df: Optional[pd.DataFrame] = None,
    X_test_df: Optional[pd.DataFrame] = None,
    y_train: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, object]:
    """[Step 4] Compute SHAP values + feature importance.

    Returns: (shap_values, shap_df, importance_df, analyzer)
    The analyzer object carries survshap 3D data for survival plots.
    """
    from core.shap_analysis import SHAPAnalyzer
    logger.info("Computing SHAP values")

    analyzer = SHAPAnalyzer()

    if endpoint == EndpointType.SURVIVAL:
        if X_train_df is None or X_test_df is None or y_train is None:
            raise ValueError(
                "X_train_df, X_test_df, and y_train are required for survival SHAP"
            )
        shap_values = analyzer.compute_survival(
            model, X_train_df, X_test_df, y_train, feature_names,
            model_type=model_type,
        )
    else:
        shap_values = analyzer.compute(model, X_train, X_test, model_type=model_type)

    importance = analyzer.get_feature_importance(feature_names)
    shap_df = analyzer.get_shap_dataframe(feature_names)

    logger.info(f"Top 5 features:\n{importance.head(5).to_string(index=False)}")
    return shap_values, shap_df, importance, analyzer


def step_visualize(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    importance_df: pd.DataFrame,
    output_dir: Path,
    design: TrialDesign,
    endpoint: EndpointType,
    model: object,
    analyzer: object = None,
) -> None:
    """[Step 5] Generate publication-ready SHAP figures.

    Filename convention: {design_label}_{endpoint_label}_{plot_type}.{fmt}
    e.g. RCT_BINARY_beeswarm.png, SIG_CONTINUOUS_summary_bar.svg
    """
    from core.visualization import (
        plot_beeswarm, plot_summary_bar, plot_dependence,
        plot_waterfall, plot_rct_comparison, plot_summary_panel,
        plot_roc_curve, plot_survshap_time, plot_survshap_aggregated,
        plot_survshap_panel, plot_count_obs_vs_pred, plot_count_panel,
        make_prefix,
    )

    prefix = make_prefix(design, endpoint)
    logger.info(f"Generating visualizations → {output_dir}  (prefix: {prefix})")

    # Survival: Beeswarm + Summary Bar panel, plus SurvSHAP(t) time plots
    if endpoint == EndpointType.SURVIVAL:
        plot_summary_panel(shap_values, X_test, feature_names, output_dir, prefix)

        # RCT-specific: treatment arm comparison (align X with subsampled SHAP)
        if design == TrialDesign.RCT_TWO_ARM:
            treatment_idx = feature_names.index("ARM") if "ARM" in feature_names else 0
            X_aligned = X_test[:shap_values.shape[0]]
            plot_rct_comparison(shap_values, X_aligned, feature_names,
                                treatment_col_idx=treatment_idx,
                                output_dir=output_dir, prefix=prefix)

        if analyzer is not None and hasattr(analyzer, "survshap_values_") and analyzer.survshap_values_ is not None:
            sv_3d = analyzer.survshap_values_
            times = analyzer.survshap_times_
            n_features = len(feature_names)

            sv_agg = sv_3d.mean(axis=0)
            if sv_agg.ndim == 2 and sv_agg.shape[0] == n_features:
                # Combined panel: time curves + box plot
                plot_survshap_panel(sv_3d, feature_names, times,
                                    output_dir, prefix)
                # Supplementary: single-subject decomposition
                plot_survshap_aggregated(sv_3d, X_test, feature_names,
                                         output_dir, prefix, sample_idx=0,
                                         times=times)
                plot_survshap_aggregated(sv_3d, X_test, feature_names,
                                         output_dir, prefix, sample_idx=1,
                                         times=times)
        return

    # Count: standard SHAP plots + count-specific diagnostics
    if endpoint == EndpointType.COUNT:
        plot_beeswarm(shap_values, X_test, feature_names, output_dir, prefix)
        plot_summary_bar(shap_values, feature_names, output_dir, prefix)
        plot_summary_panel(shap_values, X_test, feature_names, output_dir, prefix)

        top_features = importance_df["feature"].head(3).tolist()
        for feat in top_features:
            plot_dependence(shap_values, X_test, feature_names, output_dir,
                            prefix, target_feature=feat)

        plot_waterfall(shap_values, feature_names, sample_idx=0,
                       output_dir=output_dir, prefix=prefix, max_display=10)

        if design == TrialDesign.RCT_TWO_ARM:
            treatment_idx = feature_names.index("ARM") if "ARM" in feature_names else 0
            plot_rct_comparison(shap_values, X_test, feature_names,
                                treatment_col_idx=treatment_idx,
                                output_dir=output_dir, prefix=prefix)

        # Count-specific: observed vs predicted by arm
        plot_count_obs_vs_pred(model, X_test, y_test, feature_names,
                               output_dir=output_dir, prefix=prefix)
        plot_count_panel(shap_values, X_test, y_test, feature_names, model,
                         output_dir=output_dir, prefix=prefix)
        return

    # Always: Beeswarm + Summary Bar + Summary Panel
    plot_beeswarm(shap_values, X_test, feature_names, output_dir, prefix)
    plot_summary_bar(shap_values, feature_names, output_dir, prefix)
    plot_summary_panel(shap_values, X_test, feature_names, output_dir, prefix)

    # Top 3 feature dependence plots
    top_features = importance_df["feature"].head(3).tolist()
    for feat in top_features:
        plot_dependence(shap_values, X_test, feature_names, output_dir,
                        prefix, target_feature=feat)

    # Waterfall for a representative sample (first test sample)
    plot_waterfall(shap_values, feature_names, sample_idx=0,
                   output_dir=output_dir, prefix=prefix, max_display=10)

    # RCT-specific: treatment arm comparison
    if design == TrialDesign.RCT_TWO_ARM:
        treatment_idx = feature_names.index("ARM") if "ARM" in feature_names else 0
        plot_rct_comparison(shap_values, X_test, feature_names,
                            treatment_col_idx=treatment_idx,
                            output_dir=output_dir, prefix=prefix)

    # Binary-specific: ROC curve
    if endpoint == EndpointType.BINARY:
        plot_roc_curve(model, X_test, y_test, output_dir=output_dir, prefix=prefix)


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
        (X_train, X_test, y_train, y_test, feature_names,
         X_train_df, X_test_df) = step_preprocess(
            df, args.target, design, censor_col=args.censor
        )

        # [3] Model
        model, metrics, actual_model_type = step_model(
            X_train, X_test, y_train, y_test, endpoint, args.model
        )

        # [4] SHAP
        shap_values, shap_df, importance_df, analyzer = step_shap(
            model, X_train, X_test, actual_model_type, feature_names,
            endpoint=endpoint, X_train_df=X_train_df, X_test_df=X_test_df,
            y_train=y_train,
        )

        # Save SHAP values CSV
        shap_df.to_csv(args.output / "shap_values.csv", index=False)
        importance_df.to_csv(args.output / "feature_importance.csv", index=False)

        # [5] Visualize
        if not args.no_plot:
            step_visualize(
                shap_values, X_test, y_test, feature_names, importance_df,
                args.output, design, endpoint, model, analyzer=analyzer,
            )

        logger.info("Pipeline completed successfully.")
        logger.info(f"  Output files in: {args.output}")

    except Exception:
        logger.exception("Pipeline failed with unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
