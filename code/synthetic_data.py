"""
Synthetic clinical trial data generator for development and testing.

Generates realistic RCT datasets with known treatment effects
for Continuous endpoint × 2-arm design.

Usage:
    python -m code.synthetic_data                    # Generate and save to data/
    from code.synthetic_data import get_demo_data    # In-memory for tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from shared.config import SEED, DATA_DIR

rng = np.random.default_rng(SEED)


def get_demo_data(
    n_subjects: int = 300,
    treatment_effect: float = 2.5,
    noise_scale: float = 3.0,
) -> pd.DataFrame:
    """Generate a synthetic RCT dataset with continuous endpoint.

    Parameters
    ----------
    n_subjects : int
        Total number of subjects.
    treatment_effect : float
        True additive treatment effect on the continuous outcome.
    noise_scale : float
        Standard deviation of random noise added to outcome.

    Returns
    -------
    pd.DataFrame
        Columns: USUBJID, ARM, AGE, SEX, RACE, BMI, BL_SCORE,
                 DURATION, PRIOR_MED, AVAL
    """
    n = n_subjects
    half = n // 2

    # --- Subject IDs ---
    usubjid = [f"S{str(i).zfill(4)}" for i in range(1, n + 1)]

    # --- Treatment arm (0=Control, 1=Treatment) ---
    arm = np.array([0] * half + [1] * (n - half))
    rng.shuffle(arm)

    # --- Demographics ---
    age = rng.normal(55, 12, n).clip(18, 85).round(1)
    sex = rng.choice(["M", "F"], n)
    race = rng.choice(["White", "Black", "Asian", "Other"], n, p=[0.6, 0.15, 0.2, 0.05])
    bmi = rng.normal(26, 5, n).clip(16, 45).round(1)

    # --- Clinical covariates ---
    bl_score = rng.normal(10, 4, n).clip(2, 20).round(2)  # baseline severity
    duration = rng.exponential(5, n).clip(1, 30).round(1)   # years since diagnosis
    prior_med = rng.choice([0, 1], n, p=[0.4, 0.6])         # prior medication use

    # --- Outcome (continuous) ---
    # Base = baseline severity effect + age effect + some interactions
    base = (
        5.0
        + 0.8 * bl_score
        + 0.03 * age
        - 0.15 * bmi
        + 0.5 * prior_med
        - 0.2 * duration
    )
    # Treatment effect + treatment × baseline interaction
    treatment_component = arm * (treatment_effect + 0.3 * (bl_score - 10))
    noise = rng.normal(0, noise_scale, n)
    aval = base + treatment_component + noise

    # --- Assemble ---
    df = pd.DataFrame({
        "USUBJID": usubjid,
        "ARM": arm,
        "AGE": age,
        "SEX": sex,
        "RACE": race,
        "BMI": bmi,
        "BL_SCORE": bl_score,
        "DURATION": duration,
        "PRIOR_MED": prior_med,
        "AVAL": aval.round(3),
    })
    return df


def get_demo_data_binary(
    n_subjects: int = 500,
    treatment_odds_ratio: float = 0.45,
    control_event_rate: float = 0.35,
) -> pd.DataFrame:
    """Generate a synthetic RCT dataset with binary endpoint.

    Uses a logistic model to generate the probability of event,
    then samples the binary outcome from that probability.

    Parameters
    ----------
    n_subjects : int
        Total number of subjects.
    treatment_odds_ratio : float
        Odds ratio of treatment effect (ARM=1 vs ARM=0). Values < 1
        indicate treatment reduces event risk.
    control_event_rate : float
        Baseline event rate in the control arm.

    Returns
    -------
    pd.DataFrame
        Columns: USUBJID, ARM, AGE, SEX, RACE, BMI, BL_SCORE,
                 DURATION, PRIOR_MED, AVAL (binary 0/1)
    """
    n = n_subjects
    half = n // 2

    usubjid = [f"S{str(i).zfill(4)}" for i in range(1, n + 1)]
    arm = np.array([0] * half + [1] * (n - half))
    rng.shuffle(arm)

    age = rng.normal(55, 12, n).clip(18, 85).round(1)
    sex = rng.choice(["M", "F"], n)
    race = rng.choice(["White", "Black", "Asian", "Other"], n, p=[0.6, 0.15, 0.2, 0.05])
    bmi = rng.normal(26, 5, n).clip(16, 45).round(1)
    bl_score = rng.normal(10, 4, n).clip(2, 20).round(2)
    duration = rng.exponential(5, n).clip(1, 30).round(1)
    prior_med = rng.choice([0, 1], n, p=[0.4, 0.6])

    # --- Logistic model for binary outcome probability ---
    # Log-odds: intercept chosen to achieve ~control_event_rate at mean covariates
    logit_control = np.log(control_event_rate / (1 - control_event_rate))
    log_odds = (
        logit_control
        + np.log(treatment_odds_ratio) * arm
        + 0.04 * (age - 55)
        + 0.12 * (bl_score - 10)
        - 0.06 * (bmi - 26)
        + 0.3 * (prior_med - 0.6)
        - 0.05 * (duration - 5)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    aval = rng.binomial(1, prob).astype(int)

    df = pd.DataFrame({
        "USUBJID": usubjid,
        "ARM": arm,
        "AGE": age,
        "SEX": sex,
        "RACE": race,
        "BMI": bmi,
        "BL_SCORE": bl_score,
        "DURATION": duration,
        "PRIOR_MED": prior_med,
        "AVAL": aval,
    })
    return df


def get_demo_data_survival(
    n_subjects: int = 400,
    treatment_hazard_ratio: float = 0.55,
    censoring_rate: float = 0.30,
    max_time: float = 365,
) -> pd.DataFrame:
    """Generate a synthetic RCT dataset with survival (time-to-event) endpoint.

    Uses a proportional hazards model with Weibull baseline hazard to
    generate event times, then applies independent administrative censoring.

    Parameters
    ----------
    n_subjects : int
        Total number of subjects.
    treatment_hazard_ratio : float
        Hazard ratio for treatment (ARM=1 vs ARM=0). HR < 1 means
        treatment reduces hazard (prolongs survival).
    censoring_rate : float
        Target proportion of censored observations.
    max_time : float
        Maximum follow-up time (administrative censoring boundary).

    Returns
    -------
    pd.DataFrame
        Columns: USUBJID, ARM, AGE, SEX, RACE, BMI, BL_SCORE,
                 DURATION, PRIOR_MED, AVAL (time), CNSR (censoring)
    """
    n = n_subjects
    half = n // 2

    usubjid = [f"S{str(i).zfill(4)}" for i in range(1, n + 1)]
    arm = np.array([0] * half + [1] * (n - half))
    rng.shuffle(arm)

    age = rng.normal(55, 12, n).clip(18, 85).round(1)
    sex = rng.choice(["M", "F"], n)
    race = rng.choice(["White", "Black", "Asian", "Other"], n, p=[0.6, 0.15, 0.2, 0.05])
    bmi = rng.normal(26, 5, n).clip(16, 45).round(1)
    bl_score = rng.normal(10, 4, n).clip(2, 20).round(2)
    duration = rng.exponential(5, n).clip(1, 30).round(1)
    prior_med = rng.choice([0, 1], n, p=[0.4, 0.6])

    # --- Proportional hazards model (Weibull baseline) ---
    # Linear predictor: log relative hazard
    lp = (
        np.log(treatment_hazard_ratio) * arm
        + 0.02 * (age - 55)
        + 0.10 * (bl_score - 10)
        - 0.05 * (bmi - 26)
        + 0.25 * (prior_med - 0.6)
        - 0.08 * (duration - 5)
    )
    # Weibull: shape=k, scale=λ. Baseline: k=1.2, median ~ 200 days
    weibull_k = 1.2
    weibull_lambda = 250.0
    # Generate event times via inverse transform sampling with PH
    u = rng.uniform(0.001, 0.999, n)
    event_time = (-np.log(u) / (weibull_lambda ** (-weibull_k) * np.exp(lp))) ** (1 / weibull_k)
    event_time = np.clip(event_time, 1, max_time * 2)

    # --- Censoring ---
    # Administrative censoring at max_time
    admin_censor = event_time > max_time
    event_time[admin_censor] = max_time

    # Random independent censoring to reach target rate
    n_events_total = (~admin_censor).sum()
    target_events = int(n * (1 - censoring_rate))
    if n_events_total > target_events:
        extra_censor_idx = rng.choice(
            np.where(~admin_censor)[0],
            size=n_events_total - target_events,
            replace=False,
        )
        censor_time = rng.uniform(1, event_time[extra_censor_idx])
        event_time[extra_censor_idx] = censor_time

    cnsr = np.where(event_time >= max_time, 1, 0)
    # Any subject with event_time clamped is censored
    cnsr = cnsr | (event_time >= max_time)

    df = pd.DataFrame({
        "USUBJID": usubjid,
        "ARM": arm,
        "AGE": age,
        "SEX": sex,
        "RACE": race,
        "BMI": bmi,
        "BL_SCORE": bl_score,
        "DURATION": duration,
        "PRIOR_MED": prior_med,
        "AVAL": event_time.round(1),
        "CNSR": cnsr.astype(int),
    })
    return df


def get_demo_data_count(
    n_subjects: int = 400,
    treatment_rate_ratio: float = 0.60,
    baseline_rate: float = 2.5,
    overdispersion: float = 0.5,
) -> pd.DataFrame:
    """Generate a synthetic RCT dataset with count (Poisson/NB) endpoint.

    Simulates adverse event counts (e.g., hospitalizations, exacerbations)
    with a Negative Binomial model (Poisson with overdispersion).

    Parameters
    ----------
    n_subjects : int
        Total number of subjects.
    treatment_rate_ratio : float
        Rate ratio for treatment (ARM=1 vs ARM=0). RR < 1 means
        treatment reduces event rate.
    baseline_rate : float
        Expected event count in the control arm at mean covariates.
    overdispersion : float
        Overdispersion parameter (>0). 0 recovers Poisson. Larger → more
        variance beyond the mean.

    Returns
    -------
    pd.DataFrame
        Columns: USUBJID, ARM, AGE, SEX, RACE, BMI, BL_SCORE,
                 DURATION, PRIOR_MED, AVAL (count integer >=0)
    """
    n = n_subjects
    half = n // 2

    usubjid = [f"S{str(i).zfill(4)}" for i in range(1, n + 1)]
    arm = np.array([0] * half + [1] * (n - half))
    rng.shuffle(arm)

    age = rng.normal(55, 12, n).clip(18, 85).round(1)
    sex = rng.choice(["M", "F"], n)
    race = rng.choice(["White", "Black", "Asian", "Other"], n, p=[0.6, 0.15, 0.2, 0.05])
    bmi = rng.normal(26, 5, n).clip(16, 45).round(1)
    bl_score = rng.normal(10, 4, n).clip(2, 20).round(2)
    duration = rng.exponential(5, n).clip(1, 30).round(1)
    prior_med = rng.choice([0, 1], n, p=[0.4, 0.6])

    # --- Log-linear model for count outcome ---
    # log(λ) = intercept + log(RR) * arm + covariate effects
    log_baseline = np.log(baseline_rate)
    log_lambda = (
        log_baseline
        + np.log(treatment_rate_ratio) * arm
        + 0.015 * (age - 55)
        + 0.06 * (bl_score - 10)
        + 0.02 * (bmi - 26)
        + 0.15 * (prior_med - 0.6)
        - 0.03 * (duration - 5)
    )
    lam = np.exp(log_lambda)

    # Negative Binomial: Gamma-Poisson mixture
    # NB(n, p): n = 1/α, p = 1/(1 + α·μ)  →  mean = μ, var = μ + α·μ²
    if overdispersion > 0:
        alpha = overdispersion
        size = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * lam)
        aval = rng.negative_binomial(size, p, n).clip(0, 30)
    else:
        aval = rng.poisson(lam, n).clip(0, 30)

    df = pd.DataFrame({
        "USUBJID": usubjid,
        "ARM": arm,
        "AGE": age,
        "SEX": sex,
        "RACE": race,
        "BMI": bmi,
        "BL_SCORE": bl_score,
        "DURATION": duration,
        "PRIOR_MED": prior_med,
        "AVAL": aval,
    })
    return df


def main(output_dir: Path | None = None) -> Path:
    """Generate and save all synthetic RCT datasets."""
    output_dir = Path(output_dir) if output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Continuous
    df_cont = get_demo_data()
    path_cont = output_dir / "synthetic_RCT_continuous.csv"
    df_cont.to_csv(path_cont, index=False)
    print(f"Generated {len(df_cont)} subjects → {path_cont}")
    print(f"  ARM distribution:\n{df_cont['ARM'].value_counts().to_string()}")
    print(f"  AVAL summary:\n{df_cont['AVAL'].describe().to_string()}")

    # Binary
    df_bin = get_demo_data_binary()
    path_bin = output_dir / "synthetic_RCT_binary.csv"
    df_bin.to_csv(path_bin, index=False)
    print(f"\nGenerated {len(df_bin)} subjects → {path_bin}")
    print(f"  ARM distribution:\n{df_bin['ARM'].value_counts().to_string()}")
    print(f"  Event rate: {df_bin['AVAL'].mean():.3f}")
    print(f"  Event rate by arm:\n{df_bin.groupby('ARM')['AVAL'].mean().to_string()}")

    # Survival
    df_surv = get_demo_data_survival()
    path_surv = output_dir / "synthetic_RCT_survival.csv"
    df_surv.to_csv(path_surv, index=False)
    print(f"\nGenerated {len(df_surv)} subjects → {path_surv}")
    print(f"  ARM distribution:\n{df_surv['ARM'].value_counts().to_string()}")
    print(f"  Event rate: {(1 - df_surv['CNSR'].mean()):.3f}")
    print(f"  Event rate by arm:\n{(1 - df_surv.groupby('ARM')['CNSR'].mean()).to_string()}")

    # Count
    df_cnt = get_demo_data_count()
    path_cnt = output_dir / "synthetic_RCT_count.csv"
    df_cnt.to_csv(path_cnt, index=False)
    print(f"\nGenerated {len(df_cnt)} subjects → {path_cnt}")
    print(f"  ARM distribution:\n{df_cnt['ARM'].value_counts().to_string()}")
    print(f"  Event rate: {df_cnt['AVAL'].mean():.3f}")
    print(f"  Event rate by arm:\n{df_cnt.groupby('ARM')['AVAL'].mean().to_string()}")
    print(f"  AVAL summary:\n{df_cnt['AVAL'].describe().to_string()}")

    return path_cont


if __name__ == "__main__":
    main()
