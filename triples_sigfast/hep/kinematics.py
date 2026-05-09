"""
triples_sigfast.hep.kinematics
───────────────────────────────
JIT-compiled Lorentz 4-vector mathematics for high-energy physics analysis.

All functions accept NumPy float64 arrays and operate on millions of
particles simultaneously using Numba parallel JIT compilation.

Standard HEP 4-vector convention: (E, px, py, pz) in GeV.

Functions
---------
calculate_invariant_mass  — M = sqrt((E1+E2)² - |p1+p2|²)
calculate_pseudorapidity  — η = 0.5 * ln((p+pz)/(p-pz))
delta_r_matching          — ΔR = sqrt(Δη² + Δφ²)
transverse_momentum       — pT = sqrt(px² + py²)
azimuthal_angle           — φ = arctan2(py, px)
rapidity                  — y = 0.5 * ln((E+pz)/(E-pz))

References
----------
Peskin & Schroeder, "An Introduction to Quantum Field Theory", Chapter 3.
PDG Review of Particle Physics: https://pdg.lbl.gov
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# ============================================================
#  INTERNAL JIT KERNELS
# ============================================================


@njit(fastmath=True, cache=True, parallel=True)
def _kernel_invariant_mass(
    E1: np.ndarray,
    px1: np.ndarray,
    py1: np.ndarray,
    pz1: np.ndarray,
    E2: np.ndarray,
    px2: np.ndarray,
    py2: np.ndarray,
    pz2: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    """Parallel JIT kernel for invariant mass computation."""
    n = len(E1)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        dE = E1[i] + E2[i]
        dpx = px1[i] + px2[i]
        dpy = py1[i] + py2[i]
        dpz = pz1[i] + pz2[i]
        m2 = dE * dE - dpx * dpx - dpy * dpy - dpz * dpz
        result[i] = math.sqrt(m2) if m2 >= 0.0 else 0.0
    return result


@njit(fastmath=True, cache=True, parallel=True)
def _kernel_pseudorapidity(
    pz: np.ndarray, p_tot: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Parallel JIT kernel for pseudorapidity using the numerically stable log form."""
    n = len(pz)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        p = p_tot[i]
        pzi = pz[i]
        # Numerically stable: eta = 0.5 * ln((p + pz)/(p - pz))
        # Clamp to avoid divide-by-zero at pz = ±p (beam direction)
        denom = p - pzi
        if denom <= 0.0:
            result[i] = 1e10  # forward beam direction → η → +∞
        elif p + pzi <= 0.0:
            result[i] = -1e10  # backward beam direction → η → -∞
        else:
            result[i] = 0.5 * math.log((p + pzi) / denom)
    return result


@njit(fastmath=True, cache=True, parallel=True)
def _kernel_delta_r(
    eta1: np.ndarray,
    phi1: np.ndarray,
    eta2: np.ndarray,
    phi2: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    """Parallel JIT kernel for ΔR distance computation."""
    n = len(eta1)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        d_eta = eta1[i] - eta2[i]
        # Wrap Δφ to [-π, π] for correct minimum distance
        d_phi = phi1[i] - phi2[i]
        while d_phi > math.pi:
            d_phi -= 2.0 * math.pi
        while d_phi < -math.pi:
            d_phi += 2.0 * math.pi
        result[i] = math.sqrt(d_eta * d_eta + d_phi * d_phi)
    return result


@njit(fastmath=True, cache=True, parallel=True)
def _kernel_pt(px: np.ndarray, py: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Parallel JIT kernel for transverse momentum."""
    n = len(px)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = math.sqrt(px[i] * px[i] + py[i] * py[i])
    return result


@njit(fastmath=True, cache=True, parallel=True)
def _kernel_rapidity(E: np.ndarray, pz: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Parallel JIT kernel for rapidity."""
    n = len(E)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        num = E[i] + pz[i]
        den = E[i] - pz[i]
        if den <= 0.0 or num <= 0.0:
            result[i] = 1e10 if pz[i] > 0.0 else -1e10
        else:
            result[i] = 0.5 * math.log(num / den)
    return result


# ============================================================
#  PUBLIC API
# ============================================================


def _ensure_f64(*arrays) -> list[np.ndarray]:
    """Force all inputs to C-contiguous float64 arrays."""
    return [np.ascontiguousarray(a, dtype=np.float64) for a in arrays]


def calculate_invariant_mass(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute the Lorentz-invariant mass of every particle pair at C-speed.

    Formula:  M = sqrt((E1+E2)² - (px1+px2)² - (py1+py2)² - (pz1+pz2)²)

    This is the primary observable for resonance searches (Higgs boson,
    Z boson, exotic particles). Computes on millions of pairs in parallel.

    Parameters
    ----------
    p1 : np.ndarray, shape (N, 4)
        4-vectors of the first particle in each pair. Columns: [E, px, py, pz].
    p2 : np.ndarray, shape (N, 4)
        4-vectors of the second particle in each pair. Same convention.

    Returns
    -------
    np.ndarray, shape (N,)
        Invariant mass in the same units as the input (typically GeV).
        Returns 0.0 for pairs where M² < 0 (numerical rounding artefact).

    Examples
    --------
    >>> # Z boson: two muons each ~45 GeV back-to-back
    >>> mu1 = np.array([[45.0, 0.0,  44.9, 1.0]])   # [E, px, py, pz]
    >>> mu2 = np.array([[45.0, 0.0, -44.9, -1.0]])
    >>> M = calculate_invariant_mass(mu1, mu2)
    >>> print(f"M = {M[0]:.2f} GeV")   # → ~90 GeV (Z mass)

    References
    ----------
    PDG Review: https://pdg.lbl.gov/2023/reviews/rpp2023-rev-kinematics.pdf
    """
    p1 = np.atleast_2d(np.asarray(p1, dtype=np.float64))
    p2 = np.atleast_2d(np.asarray(p2, dtype=np.float64))
    if p1.shape[1] != 4 or p2.shape[1] != 4:
        raise ValueError(
            "p1 and p2 must have shape (N, 4) with columns [E, px, py, pz]."
        )
    if p1.shape[0] != p2.shape[0]:
        raise ValueError(
            f"p1 and p2 must have the same number of rows. Got {p1.shape[0]} vs {p2.shape[0]}."
        )

    E1, px1, py1, pz1 = _ensure_f64(p1[:, 0], p1[:, 1], p1[:, 2], p1[:, 3])
    E2, px2, py2, pz2 = _ensure_f64(p2[:, 0], p2[:, 1], p2[:, 2], p2[:, 3])
    return _kernel_invariant_mass(E1, px1, py1, pz1, E2, px2, py2, pz2)


def calculate_pseudorapidity(pz: np.ndarray, p_tot: np.ndarray) -> np.ndarray:
    """
    Compute pseudorapidity η for every particle at C-speed.

    Formula:  η = 0.5 · ln((|p|+pz) / (|p|-pz)) = -ln(tan(θ/2))

    Pseudorapidity is the dominant coordinate in collider detectors.
    It is approximately equal to rapidity y for massless particles.
    Detector acceptance windows are defined in η (e.g. |η| < 2.5 for
    the ATLAS inner tracker).

    Parameters
    ----------
    pz : np.ndarray, shape (N,)
        z-component of 3-momentum (beam direction) in GeV.
    p_tot : np.ndarray, shape (N,)
        Total 3-momentum magnitude |p| = sqrt(px²+py²+pz²) in GeV.

    Returns
    -------
    np.ndarray, shape (N,)
        Pseudorapidity η. Clamped to ±1e10 for beam-direction particles.

    Examples
    --------
    >>> px = np.array([1.0, 0.0, 0.0])
    >>> py = np.array([0.0, 1.0, 0.0])
    >>> pz = np.array([0.0, 0.0, 1000.0])   # last particle: beam direction
    >>> p = np.sqrt(px**2 + py**2 + pz**2)
    >>> eta = calculate_pseudorapidity(pz, p)
    """
    pz_arr, p_arr = _ensure_f64(pz, p_tot)
    if len(pz_arr) != len(p_arr):
        raise ValueError("pz and p_tot must have the same length.")
    return _kernel_pseudorapidity(pz_arr, p_arr)


def delta_r_matching(
    eta1: np.ndarray,
    phi1: np.ndarray,
    eta2: np.ndarray,
    phi2: np.ndarray,
) -> np.ndarray:
    """
    Compute angular distance ΔR between particle pairs at C-speed.

    Formula:  ΔR = sqrt(Δη² + Δφ²)

    ΔR is the standard HEP metric for deciding if two reconstructed
    objects (jets, leptons, photons) originated from the same physical
    particle. Typical matching criteria: ΔR < 0.4 (tight) or ΔR < 0.1.

    Parameters
    ----------
    eta1, phi1 : np.ndarray, shape (N,)
        η and φ of the first particle in each pair.
    eta2, phi2 : np.ndarray, shape (N,)
        η and φ of the second particle in each pair.

    Returns
    -------
    np.ndarray, shape (N,)
        ΔR values. Δφ is correctly wrapped to [-π, π].

    Examples
    --------
    >>> # Two particles separated by ΔR = 0.5 in η only
    >>> eta1 = np.array([0.0]); phi1 = np.array([0.0])
    >>> eta2 = np.array([0.5]); phi2 = np.array([0.0])
    >>> dR = delta_r_matching(eta1, phi1, eta2, phi2)
    >>> assert abs(dR[0] - 0.5) < 1e-10
    """
    eta1_a, phi1_a, eta2_a, phi2_a = _ensure_f64(eta1, phi1, eta2, phi2)
    n = len(eta1_a)
    if not (len(phi1_a) == len(eta2_a) == len(phi2_a) == n):
        raise ValueError("All four arrays must have the same length.")
    return _kernel_delta_r(eta1_a, phi1_a, eta2_a, phi2_a)


def transverse_momentum(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Compute transverse momentum pT = sqrt(px² + py²) at C-speed.

    pT is the most fundamental observable at hadron colliders. It is
    Lorentz-invariant under boosts along the beam axis (z direction).

    Parameters
    ----------
    px, py : np.ndarray, shape (N,)
        Transverse momentum components in GeV.

    Returns
    -------
    np.ndarray, shape (N,)
        pT values in GeV.
    """
    px_a, py_a = _ensure_f64(px, py)
    if len(px_a) != len(py_a):
        raise ValueError("px and py must have the same length.")
    return _kernel_pt(px_a, py_a)


def azimuthal_angle(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Compute azimuthal angle φ = arctan2(py, px) in radians.

    Parameters
    ----------
    px, py : np.ndarray, shape (N,)

    Returns
    -------
    np.ndarray, shape (N,)
        φ in [-π, π].
    """
    px_a, py_a = _ensure_f64(px, py)
    if len(px_a) != len(py_a):
        raise ValueError("px and py must have the same length.")
    return np.arctan2(py_a, px_a)


def rapidity(E: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """
    Compute rapidity y = 0.5 · ln((E+pz)/(E-pz)).

    Unlike pseudorapidity, rapidity is exactly Lorentz-invariant under
    longitudinal boosts. For massless particles, y ≡ η.

    Parameters
    ----------
    E, pz : np.ndarray, shape (N,)
        Energy and z-momentum in GeV.

    Returns
    -------
    np.ndarray, shape (N,)
        Rapidity values.
    """
    E_a, pz_a = _ensure_f64(E, pz)
    if len(E_a) != len(pz_a):
        raise ValueError("E and pz must have the same length.")
    return _kernel_rapidity(E_a, pz_a)
