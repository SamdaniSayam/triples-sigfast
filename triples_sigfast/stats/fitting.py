"""
triples_sigfast.stats.fitting
-----------------------------
Numba-compiled probability density functions (PDFs) and NLL factory
for high-performance maximum likelihood estimation.
"""

import math

import numpy as np
from numba import njit

__all__ = [
    "crystal_ball_pdf",
    "pseudo_voigt_pdf",
    "voigtian_pdf",
    "build_nll_cost_function",
]


@njit(fastmath=True, cache=True)
def crystal_ball_pdf(x: float, params: np.ndarray) -> float:
    """
    Crystal Ball PDF (unnormalized).
    params: [alpha, n, mu, sigma]
    """
    alpha = params[0]
    n = params[1]
    mu = params[2]
    sigma = params[3]

    if sigma <= 0 or n <= 0:
        return 0.0

    abs_alpha = abs(alpha)

    z = (x - mu) / sigma

    if alpha > 0:
        condition = z > -alpha
    else:
        condition = z < alpha

    if condition:
        return math.exp(-0.5 * z * z)
    else:
        A = ((n / abs_alpha) ** n) * math.exp(-0.5 * abs_alpha * abs_alpha)
        B = (n / abs_alpha) - abs_alpha

        # Avoid division by zero or negative base in power
        if alpha > 0:
            base = B - z
        else:
            base = B + z

        if base <= 0:
            return 0.0

        return A * (base**-n)


@njit(fastmath=True, cache=True)
def pseudo_voigt_pdf(x: float, params: np.ndarray) -> float:
    """
    Pseudo-Voigt PDF (approximation of Voigtian).
    params: [mu, sigma, gamma, eta]
    where sigma is Gaussian width, gamma is Lorentzian width,
    and eta is the mixing parameter (0 to 1).
    """
    mu = params[0]
    sigma = params[1]
    gamma = params[2]
    eta = params[3]

    if sigma <= 0 or gamma <= 0 or eta < 0 or eta > 1:
        return 0.0

    # Gaussian part
    z = (x - mu) / sigma
    gaussian = math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))

    # Lorentzian part
    lorentzian = gamma / (math.pi * ((x - mu) ** 2 + gamma**2))

    return eta * lorentzian + (1.0 - eta) * gaussian


@njit(fastmath=True, cache=True)
def voigtian_pdf(x: float, params: np.ndarray) -> float:
    """
    Wrapper for Voigtian PDF using the pseudo-voigt approximation.
    params: [mu, sigma, gamma, eta]
    """
    return pseudo_voigt_pdf(x, params)


def build_nll_cost_function(pdf_func, data: np.ndarray):
    """
    Factory that returns a JIT-compiled Negative Log-Likelihood (NLL)
    cost function for scipy.optimize.

    Parameters
    ----------
    pdf_func : numba-jitted function
        The PDF function taking (x: float, params: np.ndarray) -> float.
    data : np.ndarray
        The 1D array of observed data points.

    Returns
    -------
    callable
        A numba-jitted function `nll(params)` that evaluates the negative
        log-likelihood over all data points.
    """
    # We must ensure data is contiguous and float64 for fast numba access
    _data = np.ascontiguousarray(data, dtype=np.float64)

    @njit(fastmath=True)
    def nll_cost(params: np.ndarray) -> float:
        nll = 0.0
        for i in range(len(_data)):
            p = pdf_func(_data[i], params)
            if p > 0:
                nll -= math.log(p)
            else:
                nll += 1e10  # Heavy penalty for invalid parameters
        return nll

    return nll_cost
