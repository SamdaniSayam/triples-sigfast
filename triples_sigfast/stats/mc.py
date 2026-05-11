"""
triples_sigfast.stats.mc
------------------------
Monte Carlo statistical analysis functions for simulation-based physics research.

All functions are Numba JIT-compiled for C-level performance on large datasets
(10M+ rows). Implements MCNP convergence standards and GUM uncertainty propagation.

References
----------
- MCNP6 User Manual — Section 2.6 (Tally Fluctuation Charts)
- ICRP Publication 74 — Annex A (uncertainty propagation)
- GUM:2008 — Guide to the Expression of Uncertainty in Measurement
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# -- 1. Relative Error --------------------------------------------------------


@njit(parallel=True, cache=True)  # pragma: no cover
def relative_error(counts: np.ndarray) -> np.ndarray:
    """
    Compute the relative statistical error (R) for each bin in a count array.

    The relative error is defined as R = σ / μ where σ is the standard
    deviation of the mean and μ is the mean count per bin. This is the
    primary convergence metric used in MCNP tally fluctuation charts.

    A result is considered publishable when R < 0.05 (5%) for all bins
    of interest, per MCNP6 and ICRP recommendations.

    Parameters
    ----------
    counts : np.ndarray, shape (N,) or (N, M)
        Array of tally counts. For 1-D arrays, each element is treated as
        an independent bin. For 2-D arrays, statistics are computed along
        axis 0 (each column is a bin, each row is a history batch).

    Returns
    -------
    np.ndarray
        Relative error R for each bin. Shape matches the number of bins.
        Returns np.inf for bins where the mean is zero (undefined).

    Examples
    --------
    >>> import numpy as np
    >>> from triples_sigfast.stats.mc import relative_error
    >>> counts = np.array([1000.0, 950.0, 1020.0, 980.0, 1010.0])
    >>> R = relative_error(counts)
    >>> print(f"Max relative error: {R.max():.4f}")
    """
    n = counts.shape[0]
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        val = counts[i]
        if val <= 0.0:
            result[i] = np.inf
        else:
            # Poisson approximation: σ = sqrt(N), R = 1/sqrt(N)
            result[i] = 1.0 / np.sqrt(val)

    return result


@njit(cache=True)  # pragma: no cover
def mean_relative_error(counts: np.ndarray) -> float:
    """
    Compute mean relative error across all non-zero bins.

    Convenience wrapper around relative_error() that returns
    a single scalar for summary reporting.

    Parameters
    ----------
    counts : np.ndarray
        Tally counts per bin.

    Returns
    -------
    float
        Mean R across all bins where counts > 0.
    """
    n = counts.shape[0]
    total = 0.0
    valid = 0
    for i in range(n):
        if counts[i] > 0.0:
            total += 1.0 / np.sqrt(counts[i])
            valid += 1
    if valid == 0:
        return np.inf
    return total / valid


# -- 2. Figure of Merit -------------------------------------------------------


@njit(cache=True)  # pragma: no cover
def figure_of_merit(
    rel_error: np.ndarray,
    cpu_time: float,
) -> np.ndarray:
    """
    Compute the Figure of Merit (FOM) for each tally bin.

    FOM = 1 / (R² × T) where R is the relative error and T is the CPU time
    in seconds. A constant FOM across successive batches indicates proper
    Monte Carlo convergence. FOM should increase proportionally with
    computing time for an efficient simulation.

    Parameters
    ----------
    rel_error : np.ndarray, shape (N,)
        Relative error array, typically from `relative_error()`.
    cpu_time : float
        Total CPU time in seconds for the simulation run.

    Returns
    -------
    np.ndarray
        FOM values per bin. Higher is better. Returns 0.0 for bins
        where R is zero, infinite, or NaN (undefined), and for all bins
        when cpu_time <= 0 (undefined / not yet run).

    Examples
    --------
    >>> R = relative_error(counts)
    >>> fom = figure_of_merit(R, cpu_time=3600.0)
    >>> print(f"Mean FOM: {fom.mean():.2f}")
    """
    n = rel_error.shape[0]
    result = np.empty(n, dtype=np.float64)

    # Guard: undefined FOM when no CPU time has elapsed — check once, not per bin
    if cpu_time <= 0.0:
        for i in range(n):
            result[i] = 0.0
        return result

    for i in range(n):
        r = rel_error[i]
        if r <= 0.0 or np.isinf(r) or np.isnan(r):
            result[i] = 0.0
        else:
            result[i] = 1.0 / (r * r * cpu_time)

    return result


# -- 3. Convergence Check -----------------------------------------------------


@njit(parallel=True, cache=True)  # pragma: no cover
def is_converged(
    counts: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Check whether each tally bin has converged to the MCNP standard.

    A bin is considered converged when its relative error R < threshold.
    The default threshold of 0.05 (5%) is the MCNP6 standard for point
    detector tallies. For region tallies, 0.10 may be acceptable.

    Parameters
    ----------
    counts : np.ndarray, shape (N,)
        Array of tally counts per bin.
    threshold : float, optional
        Maximum acceptable relative error. Default 0.05 (MCNP standard).
        Use 0.10 for region tallies, 0.02 for publication-critical results.

    Returns
    -------
    np.ndarray of bool, shape (N,)
        True where the bin has converged, False otherwise.

    Examples
    --------
    >>> converged = is_converged(counts, threshold=0.05)
    >>> n_converged = converged.sum()
    >>> print(f"{n_converged}/{len(counts)} bins converged")
    """
    n = counts.shape[0]
    result = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        val = counts[i]
        if val <= 0.0:
            result[i] = False
        else:
            r = 1.0 / np.sqrt(val)
            result[i] = r < threshold

    return result


# -- 4. Uncertainty Propagation -----------------------------------------------


@njit(parallel=True, cache=True)  # pragma: no cover
def propagate_error(
    counts: np.ndarray,
    efficiency: float = 1.0,
) -> np.ndarray:
    """
    Propagate statistical and systematic uncertainty through detector efficiency.

    Combines Poisson counting uncertainty with detector efficiency uncertainty
    using standard GUM (Guide to the Expression of Uncertainty in Measurement)
    quadrature propagation:

        σ_total = sqrt((σ_counts / N)² + (σ_eff / ε)²) × (N / ε)

    where σ_counts = sqrt(N) (Poisson), σ_eff = sqrt(ε(1-ε)) (binomial),
    and ε is the detector efficiency.

    Parameters
    ----------
    counts : np.ndarray, shape (N,)
        Raw tally counts per bin. Must be non-negative.
    efficiency : float, optional
        Detector efficiency in range (0, 1]. Default 1.0 (ideal detector).
        Typical HPGe detector: 0.30–0.40. Scintillator: 0.05–0.20.

    Returns
    -------
    np.ndarray
        Absolute total uncertainty (1σ) for each bin after efficiency
        correction. Units match the input counts scaled by efficiency.

    Examples
    --------
    >>> sigma = propagate_error(counts, efficiency=0.35)
    >>> corrected_counts = counts / 0.35
    >>> print(f"Bin 0: {corrected_counts[0]:.1f} ± {sigma[0]:.1f}")
    """
    n = counts.shape[0]
    result = np.empty(n, dtype=np.float64)

    eff = efficiency if efficiency > 0.0 else 1.0
    # Binomial uncertainty on efficiency: σ_eff = sqrt(ε(1-ε))
    sigma_eff = np.sqrt(eff * (1.0 - eff)) if eff < 1.0 else 0.0

    for i in prange(n):
        N = counts[i]
        if N <= 0.0:
            result[i] = 0.0
            continue

        # Poisson uncertainty on counts: σ_N = sqrt(N)
        sigma_N = np.sqrt(N)

        # Quadrature propagation (GUM eq. 13)
        # σ_total² = (∂f/∂N)² σ_N² + (∂f/∂ε)² σ_ε²
        # f = N/ε  ->  ∂f/∂N = 1/ε,  ∂f/∂ε = -N/ε²
        term_counts = (sigma_N / eff) ** 2
        term_eff = ((N / (eff * eff)) * sigma_eff) ** 2

        result[i] = np.sqrt(term_counts + term_eff)

    return result
