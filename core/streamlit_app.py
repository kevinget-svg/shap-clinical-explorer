"""
Streamlit interactive dashboard for SHAP-based clinical trial analysis.

Usage:
    streamlit run core/streamlit_app.py
    python -m streamlit run core/streamlit_app.py
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path for internal imports
_sys_project_root = Path(__file__).resolve().parent.parent
if str(_sys_project_root) not in sys.path:
    sys.path.insert(0, str(_sys_project_root))

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Deployment diagnostics: catch import errors early so the real error message
# is visible in the Streamlit UI instead of being redacted by the platform.
# ---------------------------------------------------------------------------
try:
    import shap
    from shared.config import (
        EndpointType,
        TrialDesign,
        SEED,
        OUTPUT_DIR,
        SHAP_BG_SAMPLES,
        SURVSHAP_MAX_SAMPLES,
    )
    from core.synthetic_data import (
        get_demo_data,
        get_demo_data_binary,
        get_demo_data_survival,
        get_demo_data_count,
    )
    from core.preprocessing import Preprocessor, split_train_test
    from core.modeling import ModelTrainer
    from core.shap_analysis import SHAPAnalyzer
    from core.visualization import (
        plot_beeswarm,
        plot_summary_bar,
        plot_dependence,
        plot_waterfall,
        plot_rct_comparison,
        plot_summary_panel,
        plot_roc_curve,
        plot_survshap_panel,
        plot_survshap_aggregated,
        plot_count_obs_vs_pred,
        plot_count_panel,
        make_prefix,
        _save,
    )
except Exception:
    import traceback
    st.error(f"Import failed:\n\n```\n{traceback.format_exc()}\n```")
    st.stop()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SHAP Clinical Explorer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.getLogger().setLevel(logging.WARNING)  # suppress noisy logs

# ---------------------------------------------------------------------------
# Model type options per endpoint
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    EndpointType.CONTINUOUS: ["auto", "rf", "xgb", "glm"],
    EndpointType.BINARY: ["auto", "rf", "xgb", "glm"],
    EndpointType.SURVIVAL: ["auto", "cox", "rsf"],
    EndpointType.COUNT: ["auto", "rf", "xgb", "glm"],
}

DEMO_GENERATORS = {
    EndpointType.CONTINUOUS: lambda n: get_demo_data(n_subjects=n),
    EndpointType.BINARY: lambda n: get_demo_data_binary(n_subjects=n),
    EndpointType.SURVIVAL: lambda n: get_demo_data_survival(n_subjects=n),
    EndpointType.COUNT: lambda n: get_demo_data_count(n_subjects=n),
}

# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(
    endpoint_value: str,
    design_value: str,
    model_choice: str,
    n_samples: int,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
) -> dict:
    """Run steps 1-4 of the pipeline. Returns results dict for visualization."""
    endpoint = EndpointType(endpoint_value)
    design = TrialDesign(design_value)

    # --- Load ---
    if file_bytes is not None and file_name is not None:
        suffix = Path(file_name).suffix.lower()
        tmp_path = Path(tempfile.gettempdir()) / file_name
        tmp_path.write_bytes(file_bytes)
        from core.data_loader import load_clinical_data
        df = load_clinical_data(tmp_path)
        tmp_path.unlink(missing_ok=True)
    else:
        gen = DEMO_GENERATORS[endpoint]
        df = gen(n_samples)

    # --- Preprocess ---
    target_col = "AVAL"
    censor_col = "CNSR" if endpoint == EndpointType.SURVIVAL else None

    preprocessor = Preprocessor()
    preprocessor.fit(df, target_col, design)

    feature_df = df.drop(columns=[target_col], errors="ignore")
    if censor_col and censor_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[censor_col])

    X = preprocessor.transform(feature_df)
    feature_names = preprocessor.feature_names

    # Build y
    if censor_col:
        from sksurv.util import Surv
        cnsr = df[censor_col].values.astype(bool)
        time = df[target_col].values.astype(float)
        y = Surv.from_arrays(event=~cnsr, time=time)
    else:
        y = df[target_col].values if endpoint == EndpointType.COUNT else df[target_col]

    treatment_col = "ARM" if design == TrialDesign.RCT_TWO_ARM and "ARM" in X.columns else None
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col)

    X_train_df = pd.DataFrame(Xtr, columns=feature_names)
    X_test_df = pd.DataFrame(Xte, columns=feature_names)

    # --- Model ---
    trainer = ModelTrainer()
    model = trainer.train(Xtr, ytr, model_type=model_choice, endpoint=endpoint)
    metrics = trainer.evaluate(Xte, yte)
    actual_type = trainer.model_type_

    # --- SHAP ---
    analyzer = SHAPAnalyzer()

    if endpoint == EndpointType.SURVIVAL:
        shap_vals = analyzer.compute_survival(
            model, X_train_df, X_test_df, ytr, feature_names,
            model_type=actual_type, max_samples=SURVSHAP_MAX_SAMPLES,
        )
        survshap_3d = analyzer.survshap_values_
        survshap_times = analyzer.survshap_times_
        # Align X_test / y_test with subsampled SHAP values (compute_survival
        # subsamples test rows to max_samples for performance)
        n_sv = shap_vals.shape[0]
        Xte = Xte[:n_sv]
        yte = yte[:n_sv]
    else:
        shap_vals = analyzer.compute(model, Xtr, Xte, model_type=actual_type, max_samples=SHAP_BG_SAMPLES)
        survshap_3d = None
        survshap_times = None

    importance = analyzer.get_feature_importance(feature_names)
    shap_df = analyzer.get_shap_dataframe(feature_names)

    return {
        "shap_values": shap_vals,
        "shap_df": shap_df,
        "importance": importance,
        "feature_names": feature_names,
        "X_test": Xte,
        "X_train": Xtr,
        "y_test": yte,
        "y_train": ytr,
        "model": model,
        "metrics": metrics,
        "model_type": actual_type,
        "endpoint": endpoint,
        "design": design,
        "analyzer": analyzer,
        "survshap_3d": survshap_3d,
        "survshap_times": survshap_times,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 SHAP Clinical Explorer")

data_source = st.sidebar.radio(
    "Data Source",
    ["Demo (Synthetic)", "Upload File"],
)

if data_source == "Upload File":
    uploaded = st.sidebar.file_uploader(
        "Clinical data file",
        type=["csv", "xlsx", "sas7bdat", "RData"],
    )
    n_samples_slider = None
else:
    uploaded = None
    endpoint_sel = st.sidebar.selectbox(
        "Endpoint Type",
        [e.value for e in EndpointType],
    )
    design_sel = st.sidebar.selectbox(
        "Trial Design",
        [TrialDesign.RCT_TWO_ARM.value, TrialDesign.SINGLE_ARM.value],
    )
    model_sel = st.sidebar.selectbox(
        "Model",
        MODEL_OPTIONS[EndpointType(endpoint_sel)],
    )
    n_samples_slider = st.sidebar.slider(
        "Sample Size (Demo)", 100, 800, 300, step=50,
    )

run_clicked = st.sidebar.button("▶ Run Analysis", type="primary", use_container_width=True)

# Persist analysis state across widget interactions (st.button only returns True
# on the exact rerun when it was clicked; subsequent widget changes would lose it)
if run_clicked:
    st.session_state.analysis_triggered = True

if not st.session_state.get("analysis_triggered", False):
    st.markdown("## Welcome to SHAP Clinical Explorer")
    st.markdown("""
    This interactive dashboard leverages **SHAP (SHapley Additive exPlanations)**
    values to explain machine learning model predictions on clinical trial data.

    ### Getting Started
    1. Select **Demo (Synthetic)** data or upload your own clinical dataset
    2. Choose endpoint type, trial design, and model
    3. Click **Run Analysis** to compute SHAP values and generate visualizations

    ### Supported Endpoints
    | Type | Models | Output |
    |------|--------|--------|
    | Continuous | RF, XGBoost, GLM | Beeswarm, Dependence, Waterfall |
    | Binary | RF, XGBoost, Logistic | + ROC Curve |
    | Survival | Cox PH, RSF | + SurvSHAP(t) |
    | Count | RF, XGBoost, Poisson | + Observed vs Predicted |
    """)
    st.stop()

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
with st.spinner("Running analysis pipeline..."):
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name
        endpoint_sel = "continuous"
        design_sel = "rct_2_arm"
        model_sel = "auto"
    elif data_source == "Upload File":
        st.error("Please upload a file first, or switch to Demo data.")
        st.stop()
    else:
        file_bytes = None
        file_name = None

    results = run_analysis(
        endpoint_value=endpoint_sel,
        design_value=design_sel,
        model_choice=model_sel,
        n_samples=n_samples_slider or 300,
        file_bytes=file_bytes,
        file_name=file_name,
    )

# Unpack results
shap_vals = results["shap_values"]
shap_df = results["shap_df"]
importance = results["importance"]
feature_names = results["feature_names"]
X_test = results["X_test"]
X_train = results["X_train"]
y_test = results["y_test"]
model = results["model"]
metrics = results["metrics"]
model_type = results["model_type"]
endpoint = results["endpoint"]
design = results["design"]
analyzer = results["analyzer"]
survshap_3d = results["survshap_3d"]
survshap_times = results["survshap_times"]

n_features = len(feature_names)
prefix = make_prefix(design, endpoint)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"SHAP Analysis: {endpoint.value.title()} Endpoint")
st.caption(f"Design: {design.value}  |  Model: {model_type}  |  Features: {n_features}")

# --- Metrics row ---
cols = st.columns(len(metrics))
for i, (k, v) in enumerate(metrics.items()):
    cols[i].metric(label=k.replace("_", " ").title(), value=f"{v:.4f}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Feature Dependence", "Individual Explain",
    "RCT Comparison", "Data Export",
])

# =========================================================================
# Tab 1: Overview
# =========================================================================
with tab1:
    st.subheader("SHAP Beeswarm & Feature Importance")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        fig1 = plot_beeswarm(shap_vals, X_test, feature_names, out, prefix)
        st.pyplot(fig1)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        fig2 = plot_summary_bar(shap_vals, feature_names, out, prefix)
        st.pyplot(fig2)

    # --- endpoint-specific panel ---
    if endpoint == EndpointType.SURVIVAL and survshap_3d is not None:
        st.subheader("SurvSHAP(t) Time-Dependent Panel")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            fig_s = plot_survshap_panel(survshap_3d, feature_names, survshap_times, out, prefix)
            st.pyplot(fig_s)

    elif endpoint == EndpointType.BINARY:
        st.subheader("ROC Curve")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            fig_r = plot_roc_curve(model, X_test, y_test, out, prefix)
            st.pyplot(fig_r)

    elif endpoint == EndpointType.COUNT:
        st.subheader("Count: Observed vs Predicted")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            fig_c = plot_count_obs_vs_pred(model, X_test, y_test, feature_names, out, prefix)
            st.pyplot(fig_c)

# =========================================================================
# Tab 2: Feature Dependence
# =========================================================================
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        target_feat = st.selectbox("Target Feature", feature_names)
    with col2:
        interaction_feat = st.selectbox(
            "Interaction Feature (color)",
            ["auto"] + feature_names,
        )

    interaction_feat_arg = None if interaction_feat == "auto" else interaction_feat
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        try:
            fig_d = plot_dependence(
                shap_vals, X_test, feature_names, out, prefix,
                target_feature=target_feat,
                interaction_feature=interaction_feat_arg,
            )
            st.pyplot(fig_d)
        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================================
# Tab 3: Individual Explain
# =========================================================================
with tab3:
    n_test = shap_vals.shape[0]
    sample_idx = st.slider("Select Subject (test set index)", 0, n_test - 1, 0)

    col_force, col_waterfall = st.columns(2)

    with col_force:
        st.subheader(f"Force Plot — Subject #{sample_idx}")
        try:
            if hasattr(analyzer, "explainer_") and analyzer.explainer_ is not None:
                expl = analyzer.explainer_
                expected_value = expl.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = float(expected_value[0]) if endpoint == EndpointType.BINARY else float(expected_value)
            else:
                expected_value = float(np.mean(model.predict(X_train))) if hasattr(model, "predict") else 0.0

            force_plot = shap.plots.force(
                expected_value,
                shap_vals[sample_idx],
                feature_names=feature_names,
                matplotlib=False,
            )
            # shap.save_html includes the full JS bundle (force_plot.html()
            # only returns the plot fragment without JS libraries)
            tmp_html = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            try:
                shap.save_html(tmp_html.name, force_plot)
                with open(tmp_html.name) as f:
                    force_html = f.read()
            finally:
                Path(tmp_html.name).unlink(missing_ok=True)
            st.components.v1.html(force_html, height=200, scrolling=True)
        except Exception as e:
            st.warning(f"Force plot unavailable: {e}")

    with col_waterfall:
        st.subheader(f"Waterfall — Subject #{sample_idx}")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            fig_w = plot_waterfall(
                shap_vals, feature_names, sample_idx=sample_idx,
                output_dir=out, prefix=prefix, max_display=10,
            )
            st.pyplot(fig_w)

    # SurvSHAP(t) per-subject decomposition
    if endpoint == EndpointType.SURVIVAL and survshap_3d is not None:
        st.subheader(f"SurvSHAP(t) Decomposition — Subject #{sample_idx}")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            fig_sd = plot_survshap_aggregated(
                survshap_3d, X_test, feature_names,
                out, prefix, sample_idx=sample_idx,
                times=survshap_times,
            )
            st.pyplot(fig_sd)

# =========================================================================
# Tab 4: RCT Comparison
# =========================================================================
with tab4:
    if design == TrialDesign.RCT_TWO_ARM and "ARM" in feature_names:
        st.subheader("SHAP Feature Importance by Treatment Arm")
        treatment_idx = feature_names.index("ARM")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            fig_rc = plot_rct_comparison(
                shap_vals, X_test, feature_names,
                treatment_col_idx=treatment_idx,
                output_dir=out, prefix=prefix,
            )
            st.pyplot(fig_rc)
    else:
        st.info("RCT Comparison is only available for RCT 2-Arm designs with an ARM column.")

    # Count panel for count endpoints
    if endpoint == EndpointType.COUNT and design == TrialDesign.RCT_TWO_ARM:
        st.subheader("Count Calibration Panel")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            fig_cp = plot_count_panel(shap_vals, X_test, y_test, feature_names, model, out, prefix)
            st.pyplot(fig_cp)

# =========================================================================
# Tab 5: Data Export
# =========================================================================
with tab5:
    st.subheader("Download Analysis Results")

    col_a, col_b = st.columns(2)

    with col_a:
        # SHAP values CSV
        csv_shap = shap_df.to_csv(index=False)
        st.download_button(
            f"Download SHAP Values ({shap_df.shape[0]}×{shap_df.shape[1]})",
            data=csv_shap,
            file_name=f"{prefix}_shap_values.csv",
            mime="text/csv",
        )

        # Feature importance CSV
        csv_imp = importance.to_csv(index=False)
        st.download_button(
            "Download Feature Importance",
            data=csv_imp,
            file_name=f"{prefix}_feature_importance.csv",
            mime="text/csv",
        )

    with col_b:
        # Generate all plots as a zip of SVGs
        st.markdown("#### All Figures (SVG zip)")
        if st.button("Generate Figure Bundle"):
            with st.spinner("Generating all figures..."):
                zip_buf = io.BytesIO()
                with tempfile.TemporaryDirectory() as tmpdir:
                    out = Path(tmpdir)
                    # Standard plots
                    plot_beeswarm(shap_vals, X_test, feature_names, out, prefix)
                    plot_summary_bar(shap_vals, feature_names, out, prefix)
                    top_f = importance["feature"].head(3).tolist()
                    for feat in top_f:
                        plot_dependence(shap_vals, X_test, feature_names, out, prefix, target_feature=feat)
                    plot_waterfall(shap_vals, feature_names, 0, out, prefix)
                    if design == TrialDesign.RCT_TWO_ARM and "ARM" in feature_names:
                        plot_rct_comparison(shap_vals, X_test, feature_names,
                                            feature_names.index("ARM"), out, prefix)

                    if endpoint == EndpointType.BINARY:
                        plot_roc_curve(model, X_test, y_test, out, prefix)
                    if endpoint == EndpointType.SURVIVAL and survshap_3d is not None:
                        plot_survshap_panel(survshap_3d, feature_names, survshap_times, out, prefix)
                    if endpoint == EndpointType.COUNT:
                        plot_count_obs_vs_pred(model, X_test, y_test, feature_names, out, prefix)

                    # Zip all SVGs
                    svg_files = sorted(out.glob("*.svg"))
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for sf in svg_files:
                            zf.write(sf, sf.name)

                zip_buf.seek(0)
                st.download_button(
                    f"Download {len(svg_files)} SVGs (zip)",
                    data=zip_buf,
                    file_name=f"{prefix}_figures.zip",
                    mime="application/zip",
                )
                st.success(f"Generated {len(svg_files)} figures")

    # Data preview
    st.markdown("---")
    st.markdown("### SHAP Value Preview")
    st.dataframe(shap_df.head(20), use_container_width=True)

    st.markdown("### Feature Importance")
    st.dataframe(importance, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"SHAP Clinical Explorer | Seed: {SEED} | "
    f"Features: {n_features} | Test set: {X_test.shape[0]} subjects"
)
