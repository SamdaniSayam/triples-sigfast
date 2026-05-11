"""
triples_sigfast.core.signal
----------------------------
JIT-compiled signal processing and attenuation kernels.

This module provides the computational core of triples-sigfast.  Every
function that operates on large numerical arrays is accelerated either by
Numba JIT compilation (@njit) or by fully vectorised NumPy operations, giving
throughput of 100 M+ rows per second on typical hardware.

Functions
---------
rolling_average          -- O(N) prefix-sum moving average
ema                      -- Exponential Moving Average (Numba JIT)
ema_crossover_strategy   -- Dual-EMA trading signal generator (Numba JIT)
detect_anomalies         -- Z-score outlier detection (vectorised NumPy)
savitzky_golay           -- Savitzky-Golay spectrum smoother (Numba JIT)
find_peaks               -- Gaussian peak finder for nuclear spectra (Numba JIT)
flux_to_dose             -- ICRP-74 particle flux to dose rate conversion
attenuation              -- Beer-Lambert gamma transmission (single thickness)
attenuation_series       -- Beer-Lambert gamma transmission (thickness array)

Implementation notes
--------------------
- All Numba kernels are compiled with fastmath=True, cache=True, and
  nogil=True.  fastmath allows the compiler to reorder floating-point
  operations for speed; cache=True writes the compiled object to disk so
  subsequent calls avoid JIT warmup.
- Data is always normalised to C-contiguous float64 arrays before being
  passed to JIT functions, which prevents silent type-mismatch errors.
- Heavy dependencies (reportlab, nuclear sub-package) are imported lazily
  inside function bodies, not at module import time.
"""

import numpy as np
import pandas as pd
from numba import njit


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _ensure_float64_numpy(data) -> np.ndarray:
    """Convert the input to a C-contiguous 64-bit float NumPy array.

    Accepted input types: numpy.ndarray, pandas.Series, pandas.DataFrame,
    and Python lists.  DataFrames are flattened to 1-D.

    This normalisation step is applied before every Numba kernel call to
    ensure correct memory layout and numeric type, both of which are
    required for the @njit compiled functions to operate correctly.

    Parameters
    ----------
    data : array-like
        The input data in any of the accepted types.

    Returns
    -------
    np.ndarray
        A C-contiguous array with dtype=float64.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        arr = data.to_numpy().flatten()
    elif isinstance(data, list):
        arr = np.array(data)
    else:
        arr = data
    return np.ascontiguousarray(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Rolling Average
# ---------------------------------------------------------------------------

def rolling_average(data, window_size: int):
    """Compute a moving average using the O(N) prefix-sum (cumsum) trick.

    The standard sliding-window sum has complexity O(N x W) because it
    re-sums W elements for each of the N output positions.  Using the
    cumulative sum identity:

        sum(data[i : i+W]) = cumsum[i+W] - cumsum[i]

    all N sums can be evaluated in O(N) time with no inner loop, giving a
    speedup of approximately W times (e.g. 50x for window=50).

    Parameters
    ----------
    data : array-like
        The input time-series or spectral data (1-D).
    window_size : int
        Number of samples in the moving window.  Must be >= 1.

    Returns
    -------
    np.ndarray or pd.Series
        Smoothed array of length len(data) - window_size + 1.
        Returns pd.Series (with aligned index) when input is pd.Series.
    """
    if window_size <= 0:
        raise ValueError("window_size must be >= 1.")
    if len(data) < window_size:
        raise ValueError("Data length must be >= window_size.")

    is_series  = isinstance(data, pd.Series)
    clean_data = _ensure_float64_numpy(data)

    # Compute prefix sums once; derive all window sums via subtraction.
    cumsum   = np.cumsum(clean_data)
    result   = np.empty(len(clean_data) - window_size + 1, dtype=np.float64)
    result[0] = cumsum[window_size - 1] / window_size
    result[1:] = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    if is_series:
        return pd.Series(result, index=data.index[window_size - 1:])
    return result


# ---------------------------------------------------------------------------
# 2. Exponential Moving Average (EMA)
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True, nogil=True)
def _numba_ema(data: np.ndarray, alpha: float):  # pragma: no cover
    """Numba JIT kernel for the Exponential Moving Average recurrence.

    Implements the recurrence:
        result[0] = data[0]
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]

    The GIL is released (nogil=True) so this kernel can be used inside
    parallel Numba regions.

    Parameters
    ----------
    data : np.ndarray
        C-contiguous float64 input array.
    alpha : float
        Smoothing factor in (0, 1].  Derived from span as 2/(span+1).
    """
    n = len(data)
    result    = np.empty(n, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
    return result


def ema(data, span: int):
    """Calculate the Exponential Moving Average (EMA) at JIT-compiled speed.

    The smoothing factor is derived from the span parameter using the
    standard financial analysis convention: alpha = 2 / (span + 1).

    Parameters
    ----------
    data : array-like
        The input time-series data (1-D).
    span : int
        The EMA span.  Larger values produce more smoothing.

    Returns
    -------
    np.ndarray or pd.Series
        EMA array of the same length as `data`.
    """
    if span <= 0:
        raise ValueError("span must be >= 1.")
    clean_data = _ensure_float64_numpy(data)
    alpha  = 2.0 / (span + 1.0)
    result = _numba_ema(clean_data, alpha)

    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index)
    return result


# ---------------------------------------------------------------------------
# 3. Z-score Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(data, threshold: float = 3.0):
    """Identify statistical outliers using the Z-score method.

    A data point is flagged as an anomaly when its distance from the sample
    mean exceeds `threshold` standard deviations.  This is the standard
    3-sigma rule used in nuclear spectroscopy for identifying artefacts.

    The implementation is fully vectorised (single NumPy pass), which is
    equivalent in speed to the former Numba version for real-world spectrum
    sizes while requiring no JIT warmup.

    Parameters
    ----------
    data : array-like
        The input data (1-D).
    threshold : float
        Z-score threshold.  Data points with |Z| > threshold are flagged.
        Default 3.0 (three-sigma rule).

    Returns
    -------
    np.ndarray or pd.Series
        Boolean array; True at positions that are outliers.
    """
    is_series = isinstance(data, pd.Series)
    arr       = _ensure_float64_numpy(data)

    std_val = arr.std()
    if std_val == 0.0:
        # Constant array: no outliers by definition.
        result = np.zeros(len(arr), dtype=np.bool_)
    else:
        result = np.abs(arr - arr.mean()) / std_val > threshold

    if is_series:
        return pd.Series(result, index=data.index)
    return result


# ---------------------------------------------------------------------------
# 4. EMA Crossover Strategy
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True, nogil=True)
def _numba_crossover(fast_ema: np.ndarray, slow_ema: np.ndarray):  # pragma: no cover
    """Numba JIT kernel for detecting EMA crossover events.

    Scans both EMA arrays in a single pass and emits:
        +1 (BUY)  when fast crosses above slow
        -1 (SELL) when fast crosses below slow
         0        at all other positions (HOLD)

    Parameters
    ----------
    fast_ema : np.ndarray
        Short-period EMA values.
    slow_ema : np.ndarray
        Long-period EMA values.  Must be the same length as fast_ema.
    """
    n       = len(fast_ema)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if fast_ema[i] > slow_ema[i] and fast_ema[i - 1] <= slow_ema[i - 1]:
            signals[i] = 1   # Bullish crossover: fast rises above slow.
        elif fast_ema[i] < slow_ema[i] and fast_ema[i - 1] >= slow_ema[i - 1]:
            signals[i] = -1  # Bearish crossover: fast drops below slow.
    return signals


def ema_crossover_strategy(data, fast_span: int = 9, slow_span: int = 21):
    """Generate trading signals from a dual-EMA crossover strategy.

    Computes both EMAs and scans for crossover events in a single Numba-
    accelerated pass.  The signal convention follows standard quantitative
    finance practice.

    Parameters
    ----------
    data : array-like
        Asset price or indicator series (1-D).
    fast_span : int
        Span of the shorter-period EMA.  Default 9.
    slow_span : int
        Span of the longer-period EMA.  Default 21.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (fast_ema, slow_ema, signals)
        signals: int8 array of 1 (BUY), -1 (SELL), or 0 (HOLD).
    """
    clean_data = _ensure_float64_numpy(data)
    fast_ema   = _numba_ema(clean_data, 2.0 / (fast_span + 1.0))
    slow_ema   = _numba_ema(clean_data, 2.0 / (slow_span + 1.0))
    signals    = _numba_crossover(fast_ema, slow_ema)
    return fast_ema, slow_ema, signals


# ---------------------------------------------------------------------------
# 5. Savitzky-Golay Filter
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True, nogil=True)
def _numba_savitzky_golay(
    data: np.ndarray, coeffs: np.ndarray, half_win: int
):  # pragma: no cover
    """Numba JIT kernel for Savitzky-Golay convolution.

    Applies the pre-computed polynomial convolution coefficients to the
    interior of the array.  Edge samples (within half_win of each end) are
    passed through unchanged to avoid boundary artefacts.

    Parameters
    ----------
    data : np.ndarray
        C-contiguous float64 input array.
    coeffs : np.ndarray
        Convolution coefficients of length (2 * half_win + 1), computed by
        _compute_sg_coeffs().
    half_win : int
        Half the filter window width (window // 2).
    """
    n      = len(data)
    result = np.empty(n, dtype=np.float64)

    # Copy edge samples unchanged (the filter cannot be applied here without
    # padding the data, which would alter the edge peak shapes).
    for i in range(half_win):
        result[i]           = data[i]
        result[n - 1 - i]   = data[n - 1 - i]

    # Apply the polynomial convolution to the interior samples.
    for i in range(half_win, n - half_win):
        val = 0.0
        for j in range(len(coeffs)):
            val += coeffs[j] * data[i - half_win + j]
        result[i] = val

    return result


def _compute_sg_coeffs(window: int, polyorder: int) -> np.ndarray:
    """Compute Savitzky-Golay convolution coefficients via least squares.

    Constructs the Vandermonde design matrix A for the polynomial fit
    using a single NumPy outer-product broadcast, then solves the normal
    equations with the Moore-Penrose pseudoinverse.  Only the first row of
    pinv(A) is needed (the smoothing coefficients for the centre point).

    Parameters
    ----------
    window : int
        Number of samples in the smoothing window (must be odd and >= 3).
    polyorder : int
        Order of the fitting polynomial.  Must be < window.

    Returns
    -------
    np.ndarray
        C-contiguous float64 coefficient array of length `window`.
    """
    half_win = window // 2
    # x-coordinates of the window samples, centred at zero.
    x = np.arange(-half_win, half_win + 1, dtype=np.float64)

    # Vandermonde matrix: A[i, j] = x[i]^j, built without a Python loop
    # via NumPy broadcasting (outer product of x and power exponents).
    A = x[:, np.newaxis] ** np.arange(polyorder + 1, dtype=np.float64)

    # The smoothing coefficients are the first row of the pseudoinverse of A.
    coeffs = np.linalg.pinv(A)[0]
    return np.ascontiguousarray(coeffs, dtype=np.float64)


def savitzky_golay(data, window: int = 11, polyorder: int = 3):
    """Smooth a signal using the Savitzky-Golay filter.

    Unlike a simple moving average, the Savitzky-Golay filter fits a
    polynomial through each window of samples, which preserves the height
    and position of peaks.  This is essential for nuclear energy spectra,
    where the position and amplitude of gamma-ray lines and neutron capture
    peaks carry physical meaning.

    The convolution coefficients are computed once via _compute_sg_coeffs()
    and then applied across the entire array by the Numba JIT kernel.

    Parameters
    ----------
    data : array-like
        Input spectral or time-series data (1-D).
    window : int
        Number of samples in the smoothing window.  Must be odd and >= 3.
        Default 11.
    polyorder : int
        Polynomial order for the local fit.  Must be < window.  Default 3.

    Returns
    -------
    np.ndarray or pd.Series
        Smoothed array of the same length as `data`.

    Examples
    --------
    >>> smoothed = savitzky_golay(neutron_counts, window=11, polyorder=3)
    """
    if window % 2 == 0:
        raise ValueError("window must be odd (e.g., 11, 13, 15).")
    if polyorder >= window:
        raise ValueError("polyorder must be less than window.")
    if len(data) < window:
        raise ValueError("Data length must be >= window.")

    clean_data = _ensure_float64_numpy(data)
    coeffs     = _compute_sg_coeffs(window, polyorder)
    half_win   = window // 2
    result     = _numba_savitzky_golay(clean_data, coeffs, half_win)

    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index)
    return result


# ---------------------------------------------------------------------------
# 6. Peak Finder
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _numba_find_peaks(
    data: np.ndarray, min_height: float, min_distance: int
):  # pragma: no cover
    """Numba JIT kernel for peak detection in a 1-D array.

    A sample at index i is a peak if:
    1. Its value exceeds min_height.
    2. It is strictly greater than both its immediate neighbours (local max).
    3. Its index is at least min_distance samples away from the most recently
       accepted peak (separation constraint).

    Parameters
    ----------
    data : np.ndarray
        C-contiguous float64 spectral data.
    min_height : float
        Minimum value a sample must have to be considered a peak.
    min_distance : int
        Minimum number of bins between two accepted peaks.
    """
    n           = len(data)
    peak_indices = np.empty(n, dtype=np.int64)
    peak_count   = 0

    for i in range(1, n - 1):
        # Check height threshold and local maximum condition.
        if data[i] > min_height and data[i] > data[i - 1] and data[i] > data[i + 1]:
            # Check minimum separation from the last accepted peak.
            if peak_count == 0 or (i - peak_indices[peak_count - 1]) >= min_distance:
                peak_indices[peak_count] = i
                peak_count += 1

    return peak_indices[:peak_count]


def find_peaks(data, min_height: float = 0.0, min_distance: int = 1):
    """Detect spectral peaks in a 1-D array using Numba JIT acceleration.

    Designed for nuclear energy spectra where peaks correspond to physical
    gamma-ray lines, neutron capture resonances, or fission product peaks
    in Geant4 / ROOT output.

    Parameters
    ----------
    data : array-like
        Input spectral data (counts per energy bin).
    min_height : float
        Minimum peak height threshold.  Setting this to 5% of the spectrum
        maximum (the default in the CLI) avoids spurious noise peaks.
        Default 0.0 (all local maxima).
    min_distance : int
        Minimum number of bins between two accepted peaks.  Increase for
        spectra with many adjacent noise spikes.  Default 1.

    Returns
    -------
    np.ndarray
        Integer array of bin indices at which peaks were detected.

    Examples
    --------
    >>> peaks = find_peaks(gamma_spectrum, min_height=50, min_distance=10)
    >>> print(f"Found {len(peaks)} gamma lines at bins: {peaks}")
    """
    if min_distance < 1:
        raise ValueError("min_distance must be >= 1.")
    clean_data = _ensure_float64_numpy(data)
    return _numba_find_peaks(clean_data, min_height, min_distance)


# ---------------------------------------------------------------------------
# 7. Flux-to-Dose Conversion (ICRP Publication 74)
# ---------------------------------------------------------------------------
# This function delegates entirely to triples_sigfast.nuclear.dose, which
# holds the single authoritative copy of the ICRP 74 fluence-to-dose
# coefficient tables.  The delegation avoids the duplicate-table inconsistency
# that existed in earlier versions (where two separate copies of the tables
# could drift out of sync).

def flux_to_dose(flux, energy_mev: float, particle: str = "neutron"):
    """Convert particle flux to ambient dose equivalent rate H*(10).

    Applies ICRP Publication 74 (1996) fluence-to-dose conversion
    coefficients using log-log interpolation between the tabulated energy
    grid points.  This is the international standard for radiation
    protection dose assessment.

    Parameters
    ----------
    flux : float or np.ndarray
        Particle flux in particles / cm2 / s.
    energy_mev : float
        Particle kinetic energy in MeV.
    particle : str
        Either 'neutron' or 'gamma'.  Default 'neutron'.

    Returns
    -------
    float or np.ndarray
        Dose equivalent rate in microSievert per hour (uSv/hr).

    Examples
    --------
    >>> # 252Cf source, average neutron energy 2.35 MeV
    >>> dose = flux_to_dose(flux=1e6, energy_mev=2.35, particle="neutron")
    >>> print(f"Dose rate: {dose:.4f} uSv/hr")

    References
    ----------
    ICRP Publication 74, Annals of the ICRP 26(3/4), 1996.
    """
    # Lazy import: keeps core/signal.py free of a hard dependency on the
    # nuclear sub-package at module import time.  The nuclear module is
    # only loaded when flux_to_dose() is actually called.
    from triples_sigfast.nuclear.dose import (
        _NEUTRON_H_PHI,
        _PHOTON_H_PHI,
        _PSV_S_TO_USV_HR,
        _interpolate_h_phi,
    )

    particle = particle.lower()
    if particle not in ("neutron", "gamma"):
        raise ValueError("particle must be 'neutron' or 'gamma'.")
    if energy_mev <= 0:
        raise ValueError("energy_mev must be > 0.")

    # Select the correct ICRP 74 coefficient table.
    table = _NEUTRON_H_PHI if particle == "neutron" else _PHOTON_H_PHI

    # Look up the fluence-to-dose coefficient h*(10) for this energy (pSv cm2).
    conversion_pSv_cm2 = _interpolate_h_phi(energy_mev, table)

    # Multiply: flux (particles/cm2/s) x h*(10) (pSv cm2) = pSv/s
    # Then convert pSv/s to uSv/hr using the _PSV_S_TO_USV_HR factor.
    return flux * conversion_pSv_cm2 * _PSV_S_TO_USV_HR


# ---------------------------------------------------------------------------
# 8. Gamma Attenuation (Beer-Lambert Law)
# ---------------------------------------------------------------------------
# Mass attenuation coefficients (mu/rho) in cm2/g at approximately 1 MeV.
# Source: NIST XCOM Photon Cross Sections Database.
# Density values are standard reference values in g/cm3.
#
# These values are used for the Beer-Lambert simple-exponential attenuation
# calculation in attenuation() and attenuation_series().  For a more accurate
# calculation that accounts for scattered photons (buildup), use
# triples_sigfast.nuclear.shielding.attenuation_with_buildup() instead.

_ATTENUATION_MATERIALS = {
    "lead":         {"density": 11.35, "mu_rho": 0.0708},
    "polyethylene": {"density":  0.95, "mu_rho": 0.0636},
    "concrete":     {"density":  2.30, "mu_rho": 0.0664},
    "water":        {"density":  1.00, "mu_rho": 0.0706},
    "iron":         {"density":  7.87, "mu_rho": 0.0599},
    "bismuth":      {"density":  9.79, "mu_rho": 0.0705},
    "tungsten":     {"density": 19.30, "mu_rho": 0.0880},
    "borated_poly": {"density":  1.06, "mu_rho": 0.0640},
    "polysulfone":  {"density":  1.24, "mu_rho": 0.0638},
}


def attenuation(
    thickness_cm: float,
    material: str = "lead",
    mu_rho: float | None = None,
    density: float | None = None,
):
    """Calculate gamma ray transmission through a single shielding layer.

    Applies Beer-Lambert law:  T = exp(-mu * x)

    where mu (cm-1) is the linear attenuation coefficient derived from the
    mass attenuation coefficient:  mu = (mu/rho) x rho

    This is a simple exponential calculation that does not account for
    scattered photon buildup.  For a more physically accurate result,
    use triples_sigfast.nuclear.shielding.attenuation_with_buildup().

    Built-in materials (from _ATTENUATION_MATERIALS, NIST XCOM at ~1 MeV):
        lead, polyethylene, concrete, water, iron,
        bismuth, tungsten, borated_poly, polysulfone

    Parameters
    ----------
    thickness_cm : float
        Shield thickness in centimetres.  Must be >= 0.
    material : str
        Material name from the built-in database.  Default 'lead'.
    mu_rho : float, optional
        Custom mass attenuation coefficient in cm2/g.  Overrides `material`.
    density : float, optional
        Custom material density in g/cm3.  Required when mu_rho is provided.

    Returns
    -------
    float
        Transmission fraction in [0, 1].
        0.0 = fully attenuated; 1.0 = no shielding.

    Examples
    --------
    >>> T = attenuation(thickness_cm=10, material="lead")
    >>> print(f"Transmission: {T * 100:.4f}%")

    >>> # Composite shield: 5 cm lead followed by 10 cm polyethylene
    >>> T_composite = attenuation(5, "lead") * attenuation(10, "polyethylene")

    References
    ----------
    NIST XCOM Photon Cross Sections Database.
    """
    if thickness_cm < 0:
        raise ValueError("thickness_cm must be >= 0.")

    if mu_rho is not None:
        # Custom material: derive the linear coefficient from user-supplied values.
        if density is None:
            raise ValueError("density must be provided when using a custom mu_rho.")
        mu_linear = mu_rho * density
    else:
        # Built-in material lookup.
        mat = material.lower()
        if mat not in _ATTENUATION_MATERIALS:
            available = list(_ATTENUATION_MATERIALS.keys())
            raise ValueError(f"Unknown material '{material}'. Available: {available}")
        props     = _ATTENUATION_MATERIALS[mat]
        mu_linear = props["mu_rho"] * props["density"]

    return float(np.exp(-mu_linear * thickness_cm))


def attenuation_series(thickness_range, material: str = "lead"):
    """Calculate gamma transmission across an array of shield thicknesses.

    Vectorised equivalent of calling attenuation() for each element of
    `thickness_range`.  Useful for generating shielding effectiveness curves
    as a function of shield thickness -- a standard figure in nuclear
    shielding research.

    Parameters
    ----------
    thickness_range : array-like
        Thickness values in centimetres.
    material : str
        Material name from the built-in database.  Default 'lead'.

    Returns
    -------
    np.ndarray
        Transmission fractions, one per element of `thickness_range`.

    Examples
    --------
    >>> import numpy as np
    >>> thicknesses = np.linspace(0, 30, 100)
    >>> T = attenuation_series(thicknesses, material="lead")
    """
    thicknesses = _ensure_float64_numpy(thickness_range)

    # Validate the material name and extract coefficients once, outside the
    # vectorised expression, to catch errors before the computation runs.
    mat = material.lower()
    if mat not in _ATTENUATION_MATERIALS:
        available = list(_ATTENUATION_MATERIALS.keys())
        raise ValueError(f"Unknown material '{material}'. Available: {available}")

    props     = _ATTENUATION_MATERIALS[mat]
    mu_linear = props["mu_rho"] * props["density"]

    # Single vectorised exp over all thicknesses -- avoids a Python loop.
    return np.exp(-mu_linear * thicknesses)
