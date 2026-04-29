"""
ADaM-standard preprocessing and feature engineering for clinical trial data.

Handles missing value imputation, categorical encoding, and
train/test splitting with treatment-arm stratification for RCT designs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared.config import SEED, TRAIN_SIZE, TEST_SIZE, TrialDesign

logger = logging.getLogger(__name__)


class Preprocessor:
    """Fit-transform pipeline for clinical data.

    - Numeric columns: median imputation
    - Categorical columns: mode imputation + one-hot encoding
    - Keeps track of fitted values for reproducibility.
    """

    def __init__(self):
        self._numeric_medians: dict[str, float] = {}
        self._categorical_modes: dict[str, object] = {}
        self._categorical_cols: list[str] = []
        self._numeric_cols: list[str] = []
        self._feature_names_in: list[str] = []
        self._feature_names_out: list[str] = []
        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        """Feature names after transform (one-hot expanded)."""
        return self._feature_names_out

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        design: TrialDesign = TrialDesign.RCT_TWO_ARM,
    ) -> Preprocessor:
        """Learn imputation values and feature categories.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
            Name of the outcome column (excluded from features).
        design : TrialDesign
            Trial design — influences which columns are treated as
            stratifying variables.

        Returns
        -------
        self
        """
        # Exclude target and subject ID
        exclude = {target_col}
        if "USUBJID" in df.columns:
            exclude.add("USUBJID")

        feature_df = df.drop(columns=list(exclude), errors="ignore")
        self._feature_names_in = list(feature_df.columns)

        for col in feature_df.columns:
            if pd.api.types.is_numeric_dtype(feature_df[col]):
                self._numeric_cols.append(col)
                self._numeric_medians[col] = feature_df[col].median()
            else:
                self._categorical_cols.append(col)
                mode_val = feature_df[col].mode()
                self._categorical_modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else "MISSING"

        self._fitted = True
        logger.info(
            f"Preprocessor fitted: {len(self._numeric_cols)} numeric, "
            f"{len(self._categorical_cols)} categorical features"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation and one-hot encoding.

        Parameters
        ----------
        df : pd.DataFrame
            Raw feature dataframe (may include target/USUBJID — they are dropped).

        Returns
        -------
        pd.DataFrame
            Fully numeric feature matrix.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call .fit() first.")

        df = df.copy()
        logger.debug(f"Transform input shape: {df.shape}")

        # Impute numeric
        for col, med in self._numeric_medians.items():
            if col in df.columns:
                df.loc[:, col] = df[col].fillna(med)

        # Impute categorical
        for col, mode_val in self._categorical_modes.items():
            if col in df.columns:
                df.loc[:, col] = df[col].fillna(mode_val)

        # One-hot encode categoricals (only those present)
        cat_cols_present = [c for c in self._categorical_cols if c in df.columns]
        if cat_cols_present:
            df = pd.get_dummies(df, columns=cat_cols_present, drop_first=False)

        # Drop any remaining non-numeric (e.g., USUBJID)
        non_num = df.select_dtypes(exclude="number").columns
        if len(non_num) > 0:
            df = df.drop(columns=non_num)

        self._feature_names_out = list(df.columns)
        logger.debug(f"Transform output shape: {df.shape}")
        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        design: TrialDesign = TrialDesign.RCT_TWO_ARM,
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df, target_col, design)
        return self.transform(df)


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    treatment_col: Optional[str] = None,
    train_size: float = TRAIN_SIZE,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test, stratified by treatment arm if available.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values.
    treatment_col : str, optional
        Column name for the treatment arm indicator. If present and,
        ``TrialDesign`` is RCT, splitting is stratified by this column.
    train_size : float
        Proportion for training set.
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    stratify = None
    if treatment_col and treatment_col in X.columns:
        stratify = X[treatment_col]
        logger.info(f"Stratified split by '{treatment_col}'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )
    logger.info(
        f"Split: train={X_train.shape[0]}, test={X_test.shape[0]}  "
        f"(ratio={train_size:.0%}/{1 - train_size:.0%})"
    )
    return (
        X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train,
        X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test,
        y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train,
        y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test,
    )
