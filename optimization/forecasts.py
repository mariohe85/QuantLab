"""Pure expected-return estimators for portfolio optimization.

All return inputs are decimal returns and expected stock returns are annualized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpectedReturnShrinkage:
    expected_returns: pd.Series
    raw_expected_returns: pd.Series
    shrinkage_weights: pd.Series
    sampling_variance: pd.Series
    observations: pd.Series
    prior_mean: float
    prior_variance: float


@dataclass(frozen=True)
class BetaShrinkage:
    betas: pd.DataFrame
    raw_betas: pd.DataFrame
    shrinkage_weights: pd.DataFrame
    prior_variance: pd.Series
    informative_counts: pd.Series


@dataclass(frozen=True)
class FactorPremiumShrinkage:
    premia: pd.Series
    prior_premia: pd.Series
    sample_premia: pd.Series
    shrinkage_weights: pd.Series
    standard_errors: pd.Series
    observations: pd.Series
    prior_dispersion: float


@dataclass(frozen=True)
class FactorImpliedExpectedReturns:
    expected_returns: pd.Series
    factor_premia: pd.Series
    effective_betas: pd.DataFrame
    annualized_contributions: pd.DataFrame
    risk_free_rate: float
    dispersion: float


@dataclass(frozen=True)
class EqualSharpeExpectedReturns:
    expected_returns: pd.Series
    volatilities: pd.Series
    common_sharpe: float
    risk_free_rate: float
    dispersion: float


def _numeric_frame(
    frame: pd.DataFrame, name: str, *, allow_nan: bool = False
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")
    if frame.columns.has_duplicates or frame.index.has_duplicates:
        raise ValueError(f"{name} must have unique index and columns")
    try:
        result = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc
    values = result.to_numpy()
    if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
        qualifier = "finite or missing" if allow_nan else "finite"
        raise ValueError(f"{name} values must be {qualifier}")
    return result


def _numeric_series(values: pd.Series | Mapping[str, float], name: str) -> pd.Series:
    try:
        result = pd.Series(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc
    if (
        result.empty
        or result.index.has_duplicates
        or not np.isfinite(result.to_numpy()).all()
    ):
        raise ValueError(f"{name} must be non-empty, uniquely labelled, and finite")
    return result


def _validate_annualization(annualization: int) -> None:
    if not isinstance(annualization, int) or annualization <= 0:
        raise ValueError("annualization must be a positive integer")


def shrink_annualized_expected_returns(
    daily_returns: pd.DataFrame,
    *,
    annualization: int = 252,
    min_observations: int = 60,
    prior_mean: float | None = None,
    prior_variance: float | None = None,
) -> ExpectedReturnShrinkage:
    """Shrink annualized sample means toward a cross-sectional normal prior.

    The empirical prior variance is the positive part of cross-sectional
    variance in raw estimates less average estimation variance.  Posterior
    weights are ``tau² / (tau² + sampling_variance)``.
    """

    _validate_annualization(annualization)
    if not isinstance(min_observations, int) or min_observations < 2:
        raise ValueError("min_observations must be an integer of at least two")
    clean = _numeric_frame(daily_returns, "daily_returns", allow_nan=True)
    observations = clean.notna().sum()
    if (observations < min_observations).any():
        names = observations.index[observations < min_observations].tolist()
        raise ValueError(f"Insufficient observations for: {names}")

    raw = clean.mean() * annualization
    sampling_variance = clean.var(ddof=1) * annualization**2 / observations
    if (
        not np.isfinite(raw.to_numpy()).all()
        or not np.isfinite(sampling_variance.to_numpy()).all()
    ):
        raise ValueError("Expected-return moments could not be estimated")

    if prior_mean is None:
        if len(raw) < 2:
            raise ValueError("At least two stocks are required to estimate prior_mean")
        precision = 1.0 / sampling_variance.clip(lower=np.finfo(float).eps)
        estimated_prior_mean = float((raw * precision).sum() / precision.sum())
    else:
        estimated_prior_mean = float(prior_mean)
        if not np.isfinite(estimated_prior_mean):
            raise ValueError("prior_mean must be finite")

    if prior_variance is None:
        if len(raw) < 2:
            raise ValueError(
                "At least two stocks are required to estimate prior_variance"
            )
        estimated_prior_variance = max(
            float(raw.var(ddof=1) - sampling_variance.mean()), 0.0
        )
    else:
        estimated_prior_variance = float(prior_variance)
        if not np.isfinite(estimated_prior_variance) or estimated_prior_variance < 0:
            raise ValueError("prior_variance must be finite and non-negative")

    denominator = estimated_prior_variance + sampling_variance
    weights = pd.Series(
        np.divide(
            estimated_prior_variance,
            denominator,
            out=np.zeros(len(denominator), dtype=float),
            where=denominator.to_numpy() > 0,
        ),
        index=raw.index,
    )
    posterior = estimated_prior_mean + weights * (raw - estimated_prior_mean)
    return ExpectedReturnShrinkage(
        expected_returns=posterior,
        raw_expected_returns=raw,
        shrinkage_weights=weights,
        sampling_variance=sampling_variance,
        observations=observations,
        prior_mean=estimated_prior_mean,
        prior_variance=estimated_prior_variance,
    )


def shrink_betas_to_zero(
    betas: pd.DataFrame,
    standard_errors: pd.DataFrame,
    *,
    prior_variance: pd.Series | Mapping[str, float] | None = None,
    max_informative_error: float = 1.0,
) -> BetaShrinkage:
    """Shrink each factor's stock betas toward zero using matched standard errors.

    Stock models only estimate the factors they select, so an unselected factor
    arrives as a zero beta carrying a placeholder standard error. Those entries
    say nothing about how large real betas are, and because the prior variance is
    a cross-sectional moment a single placeholder would drag it below zero and
    shrink every stock's beta to zero. Entries whose standard error reaches
    ``max_informative_error`` are therefore excluded from the prior variance;
    they are still shrunk individually, which leaves them at zero as intended.
    """

    raw = _numeric_frame(betas, "betas")
    errors = _numeric_frame(standard_errors, "standard_errors")
    if set(raw.index) != set(errors.index) or set(raw.columns) != set(errors.columns):
        raise ValueError("betas and standard_errors must have identical labelled axes")
    errors = errors.reindex(index=raw.index, columns=raw.columns)
    if (errors < 0).any().any():
        raise ValueError("standard_errors must be non-negative")
    if not np.isfinite(max_informative_error) or max_informative_error <= 0:
        raise ValueError("max_informative_error must be a positive finite number")

    informative = errors < max_informative_error
    informative_counts = informative.sum(axis=0).astype(int)
    if prior_variance is None:
        moments = (raw.pow(2) - errors.pow(2)).where(informative)
        tau2 = moments.mean(axis=0).fillna(0.0).clip(lower=0)
    else:
        supplied = _numeric_series(prior_variance, "prior_variance")
        if set(supplied.index) != set(raw.columns):
            raise ValueError("prior_variance labels must exactly match beta columns")
        if (supplied < 0).any():
            raise ValueError("prior_variance must be non-negative")
        tau2 = supplied.reindex(raw.columns)

    numerator = np.broadcast_to(tau2.to_numpy(), raw.shape)
    denominator = numerator + errors.to_numpy() ** 2
    weights = pd.DataFrame(
        np.divide(
            numerator,
            denominator,
            out=np.zeros(raw.shape, dtype=float),
            where=denominator > 0,
        ),
        index=raw.index,
        columns=raw.columns,
    )
    return BetaShrinkage(
        betas=raw * weights,
        raw_betas=raw,
        shrinkage_weights=weights,
        prior_variance=tau2,
        informative_counts=informative_counts,
    )


def shrink_factor_premia(
    factor_returns: pd.DataFrame,
    prior_premia: pd.Series | Mapping[str, float],
    *,
    annualization: int = 252,
    prior_dispersion: float = 0.03,
    min_observations: int = 252,
) -> FactorPremiumShrinkage:
    """Blend trailing factor means toward supplied premium assumptions.

    Factor means are estimated far too imprecisely to use raw: over a typical
    lookback the annualized standard error is comparable to the premium itself.
    The weight on the sample mean is ``tau² / (tau² + standard_error²)``, so a
    factor only earns its way toward the data when the window measures its mean
    precisely.  ``prior_dispersion`` is the standard deviation of the prior and
    therefore states how far the assumption may plausibly be wrong; zero pins
    the result to the assumption.
    """

    _validate_annualization(annualization)
    if not isinstance(min_observations, int) or min_observations < 2:
        raise ValueError("min_observations must be an integer of at least two")
    if not np.isfinite(prior_dispersion) or prior_dispersion < 0:
        raise ValueError("prior_dispersion must be finite and non-negative")

    clean = _numeric_frame(factor_returns, "factor_returns", allow_nan=True)
    prior = _numeric_series(prior_premia, "prior_premia")
    unknown = sorted(set(prior.index) - set(clean.columns))
    if unknown:
        raise ValueError(f"Unknown factor premia: {unknown}")

    window = clean.reindex(columns=prior.index)
    observations = window.notna().sum()
    if (observations < min_observations).any():
        names = observations.index[observations < min_observations].tolist()
        raise ValueError(f"Insufficient factor history for: {names}")

    sample = window.mean() * annualization
    standard_errors = (
        window.std(ddof=1) * annualization / np.sqrt(observations.astype(float))
    )
    if (
        not np.isfinite(sample.to_numpy()).all()
        or not np.isfinite(standard_errors.to_numpy()).all()
    ):
        raise ValueError("Factor premium moments could not be estimated")

    tau2 = float(prior_dispersion) ** 2
    denominator = tau2 + standard_errors.pow(2)
    weights = pd.Series(
        np.divide(
            tau2,
            denominator.to_numpy(),
            out=np.zeros(len(denominator), dtype=float),
            where=denominator.to_numpy() > 0,
        ),
        index=prior.index,
    )
    return FactorPremiumShrinkage(
        premia=prior + weights * (sample - prior),
        prior_premia=prior,
        sample_premia=sample,
        shrinkage_weights=weights,
        standard_errors=standard_errors,
        observations=observations,
        prior_dispersion=float(prior_dispersion),
    )


def factor_implied_expected_returns(
    betas: pd.DataFrame,
    factor_premia: pd.Series | Mapping[str, float],
    *,
    risk_free_rate: float = 0.0,
    standard_errors: pd.DataFrame | None = None,
    shrink_betas: bool = False,
) -> FactorImpliedExpectedReturns:
    """Price stock factor exposures using annual excess-return assumptions."""

    raw_betas = _numeric_frame(betas, "betas")
    premia = _numeric_series(factor_premia, "factor_premia")
    unknown = sorted(set(premia.index) - set(raw_betas.columns))
    if unknown:
        raise ValueError(f"Unknown factor premia: {unknown}")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if not premia.ne(0).any():
        raise ValueError("At least one factor premium must be non-zero")

    effective_betas = raw_betas
    if shrink_betas:
        if standard_errors is None:
            raise ValueError(
                "standard_errors are required when shrink_betas is enabled"
            )
        effective_betas = shrink_betas_to_zero(raw_betas, standard_errors).betas

    priced_betas = effective_betas.reindex(columns=premia.index)
    contributions = priced_betas.mul(premia, axis=1)
    expected = contributions.sum(axis=1) + float(risk_free_rate)
    return FactorImpliedExpectedReturns(
        expected_returns=expected,
        factor_premia=premia,
        effective_betas=effective_betas,
        annualized_contributions=contributions,
        risk_free_rate=float(risk_free_rate),
        dispersion=float(expected.std(ddof=0)),
    )


def equal_sharpe_expected_returns(
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    common_sharpe: float = 0.5,
) -> EqualSharpeExpectedReturns:
    """Assume every name has the same Sharpe ratio: excess return = k × volatility.

    Volatility is the square root of the covariance diagonal, so the forecast is
    consistent with the risk model used in the solve.  For a maximum-Sharpe
    objective the scalar ``k`` cancels and the unconstrained tangency solution
    is the Maximum Diversification Portfolio ``w ∝ Σ⁻¹σ``.
    """

    cov = _numeric_frame(covariance, "covariance")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if not np.isfinite(common_sharpe) or common_sharpe <= 0:
        raise ValueError("common_sharpe must be a positive finite number")
    diagonal = np.diag(cov.to_numpy())
    if (diagonal <= 0).any() or not np.isfinite(diagonal).all():
        raise ValueError("covariance diagonal must be positive and finite")
    volatilities = pd.Series(np.sqrt(diagonal), index=cov.index)
    expected = float(risk_free_rate) + float(common_sharpe) * volatilities
    return EqualSharpeExpectedReturns(
        expected_returns=expected,
        volatilities=volatilities,
        common_sharpe=float(common_sharpe),
        risk_free_rate=float(risk_free_rate),
        dispersion=float(expected.std(ddof=0)),
    )
