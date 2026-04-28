"""Clinical domain constants for the SHAP analysis project.

These constants define ADaM dataset structures, expected column names,
and value mappings used across the project.
"""

# ---------------------------------------------------------------------------
# ADaM Dataset Types
# ---------------------------------------------------------------------------
ADSL = "adsl"    # Subject-Level Analysis Dataset
ADTTE = "adtte"  # Time-to-Event Analysis Dataset
ADLB = "adlb"    # Laboratory Results Analysis Dataset

# ---------------------------------------------------------------------------
# ADSL (Subject-Level) -- required columns
# ---------------------------------------------------------------------------
ADSL_REQUIRED_COLS: list[str] = [
    "USUBJID",     # Unique subject identifier
    "ARM",          # Treatment arm
    "AGE",          # Age
    "SEX",          # Sex
    "RACE",         # Race
    "SAFFL",        # Safety population flag
]

# ---------------------------------------------------------------------------
# ADTTE (Time-to-Event) -- required columns
# ---------------------------------------------------------------------------
ADTTE_REQUIRED_COLS: list[str] = [
    "USUBJID",
    "PARAMCD",      # Parameter code (e.g., OS, PFS)
    "AVAL",          # Analysis value (time)
    "CNSR",          # Censor indicator (0=event, 1=censored)
]

# ---------------------------------------------------------------------------
# ADLB (Lab) -- required columns
# ---------------------------------------------------------------------------
ADLB_REQUIRED_COLS: list[str] = [
    "USUBJID",
    "PARAMCD",
    "AVAL",          # Analysis value
    "AVISIT",        # Analysis visit
    "ADY",           # Analysis relative day
]

# ---------------------------------------------------------------------------
# Endpoint-to-Model mapping
# ---------------------------------------------------------------------------
from shared.config import EndpointType

ENDPOINT_MODEL_MAP: dict[EndpointType, list[str]] = {
    EndpointType.CONTINUOUS: ["linear_regression", "random_forest", "xgboost"],
    EndpointType.BINARY: ["logistic_regression", "xgboost_classifier", "random_forest_classifier"],
    EndpointType.SURVIVAL: ["cox_ph", "random_survival_forest"],
    EndpointType.COUNT: ["poisson_glm", "negative_binomial"],
}
