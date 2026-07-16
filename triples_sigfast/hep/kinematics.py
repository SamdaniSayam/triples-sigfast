"""
triples_sigfast.hep.kinematics
-------------------------------
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

__all__ = [
    "calculate_invariant_mass",
    "calculate_pseudorapidity",
    "delta_r_matching",
    "transverse_momentum",
    "azimuthal_angle",
    "rapidity",
    "decay_two_body",
    "LorentzVector",
]

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
            result[i] = 1e10  # forward beam direction -> η -> +∞
        elif p + pzi <= 0.0:
            result[i] = -1e10  # backward beam direction -> η -> -∞
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


@njit(fastmath=True, cache=True, parallel=True)
def _kernel_decay_two_body(
    E: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(E)
    p1_out = np.empty((n, 4), dtype=np.float64)
    p2_out = np.empty((n, 4), dtype=np.float64)
    for i in prange(n):
        e_p = E[i]
        px_p = px[i]
        py_p = py[i]
        pz_p = pz[i]
        mass1 = m1[i]
        mass2 = m2[i]

        m2_parent = e_p * e_p - px_p * px_p - py_p * py_p - pz_p * pz_p
        m_parent = math.sqrt(m2_parent) if m2_parent > 0 else 0.0

        if m_parent < mass1 + mass2:
            # Kinematically forbidden decay
            p1_out[i, :] = 0.0
            p2_out[i, :] = 0.0
            continue

        # Rest frame energies and momentum
        e1_cm = (m_parent * m_parent + mass1 * mass1 - mass2 * mass2) / (2.0 * m_parent)
        e2_cm = m_parent - e1_cm
        p_cm2 = e1_cm * e1_cm - mass1 * mass1
        p_cm = math.sqrt(p_cm2) if p_cm2 > 0 else 0.0

        # Isotropic decay
        cos_theta = np.random.uniform(-1.0, 1.0)
        phi = np.random.uniform(0.0, 2.0 * math.pi)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        px_cm = p_cm * sin_theta * math.cos(phi)
        py_cm = p_cm * sin_theta * math.sin(phi)
        pz_cm = p_cm * cos_theta

        # Boost back to lab frame
        # v = p / E
        bx = px_p / e_p
        by = py_p / e_p
        bz = pz_p / e_p
        b2 = bx * bx + by * by + bz * bz
        gamma = 1.0 / math.sqrt(1.0 - b2) if b2 < 1.0 else 1e10

        # Boost matrix application for p1
        bp_1 = bx * px_cm + by * py_cm + bz * pz_cm
        p1_out[i, 0] = gamma * (e1_cm + bp_1)
        factor1 = (gamma - 1.0) * bp_1 / b2 if b2 > 0 else 0.0
        p1_out[i, 1] = px_cm + bx * (gamma * e1_cm + factor1)
        p1_out[i, 2] = py_cm + by * (gamma * e1_cm + factor1)
        p1_out[i, 3] = pz_cm + bz * (gamma * e1_cm + factor1)

        # Boost matrix application for p2
        px2_cm = -px_cm
        py2_cm = -py_cm
        pz2_cm = -pz_cm
        bp_2 = bx * px2_cm + by * py2_cm + bz * pz2_cm
        p2_out[i, 0] = gamma * (e2_cm + bp_2)
        factor2 = (gamma - 1.0) * bp_2 / b2 if b2 > 0 else 0.0
        p2_out[i, 1] = px2_cm + bx * (gamma * e2_cm + factor2)
        p2_out[i, 2] = py2_cm + by * (gamma * e2_cm + factor2)
        p2_out[i, 3] = pz2_cm + bz * (gamma * e2_cm + factor2)

    return p1_out, p2_out


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
    >>> print(f"M = {M[0]:.2f} GeV")   # -> ~90 GeV (Z mass)

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

    Formula:  η = 0.5 . ln((|p|+pz) / (|p|-pz)) = -ln(tan(θ/2))

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
    Compute rapidity y = 0.5 . ln((E+pz)/(E-pz)).

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


def decay_two_body(
    parent_p4: np.ndarray, m1: np.ndarray | float, m2: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decay parent particles into two child particles isotropically in the rest frame.

    Parameters
    ----------
    parent_p4 : np.ndarray, shape (N, 4)
        4-vectors of the parent particles. Columns: [E, px, py, pz].
    m1 : np.ndarray or float
        Mass(es) of the first child particle(s).
    m2 : np.ndarray or float
        Mass(es) of the second child particle(s).

    Returns
    -------
    tuple of np.ndarray
        (p1, p2) each of shape (N, 4) containing the 4-vectors of the child particles.
    """
    parent_p4 = np.atleast_2d(np.asarray(parent_p4, dtype=np.float64))
    if parent_p4.shape[1] != 4:
        raise ValueError(
            "parent_p4 must have shape (N, 4) with columns [E, px, py, pz]."
        )

    N = parent_p4.shape[0]

    m1_arr = np.broadcast_to(np.asarray(m1, dtype=np.float64), (N,))
    m2_arr = np.broadcast_to(np.asarray(m2, dtype=np.float64), (N,))

    E, px, py, pz = _ensure_f64(
        parent_p4[:, 0], parent_p4[:, 1], parent_p4[:, 2], parent_p4[:, 3]
    )
    m1_arr, m2_arr = _ensure_f64(m1_arr, m2_arr)

    return _kernel_decay_two_body(E, px, py, pz, m1_arr, m2_arr)


# ============================================================
#  LORENTZ VECTOR — OOP CONVENIENCE WRAPPER
# ============================================================


class LorentzVector:
    """Convenience class for single-particle Lorentz 4-vector algebra.

    Wraps a single (E, px, py, pz) 4-vector and exposes computed
    kinematic quantities as properties.  Arithmetic operators (+, -)
    combine two 4-vectors by component-wise addition/subtraction.

    This class is designed for interactive exploration and single-event
    analyses.  For batch processing of millions of particles, use the
    vectorised module-level functions (``calculate_invariant_mass``,
    ``transverse_momentum``, etc.) which operate on NumPy arrays
    with JIT-compiled kernels.

    Parameters
    ----------
    E : float
        Energy in GeV.
    px : float
        x-momentum component in GeV.
    py : float
        y-momentum component in GeV.
    pz : float
        z-momentum component in GeV.

    Examples
    --------
    >>> # Z boson reconstruction from two muons
    >>> mu1 = LorentzVector(E=45.1, px=0.0, py=44.9, pz=1.0)
    >>> mu2 = LorentzVector(E=45.1, px=0.0, py=-44.9, pz=-1.0)
    >>> Z = mu1 + mu2
    >>> print(f"M_Z = {Z.mass:.2f} GeV")
    >>> print(f"Z pT = {Z.pt:.3f} GeV")

    >>> # Single muon kinematics
    >>> mu = LorentzVector(E=50.0, px=30.0, py=20.0, pz=35.0)
    >>> print(f"eta={mu.eta:.3f}, phi={mu.phi:.3f} rad, pT={mu.pt:.2f} GeV")
    """

    __slots__ = ("E", "px", "py", "pz")

    def __init__(self, E: float, px: float, py: float, pz: float) -> None:
        self.E = float(E)
        self.px = float(px)
        self.py = float(py)
        self.pz = float(pz)

    # ------------------------------------------------------------------
    # Kinematic properties
    # ------------------------------------------------------------------

    @property
    def pt(self) -> float:
        """Transverse momentum pT = sqrt(px² + py²) in GeV."""
        return math.sqrt(self.px**2 + self.py**2)

    @property
    def p(self) -> float:
        """Total 3-momentum magnitude |p| = sqrt(px²+py²+pz²) in GeV."""
        return math.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def mass(self) -> float:
        """Invariant mass M = sqrt(E² - |p|²) in GeV.  Returns 0 if M² < 0."""
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return math.sqrt(m2) if m2 >= 0.0 else 0.0

    @property
    def eta(self) -> float:
        """Pseudorapidity η = 0.5 · ln((|p|+pz)/(|p|-pz)).
        Clamped to ±1e10 for beam-direction particles."""
        p = self.p
        if p <= 0.0:
            return 0.0
        denom = p - self.pz
        if denom <= 0.0:
            return 1e10
        if p + self.pz <= 0.0:
            return -1e10
        return 0.5 * math.log((p + self.pz) / denom)

    @property
    def phi(self) -> float:
        """Azimuthal angle φ = arctan2(py, px) in radians ∈ [-π, π]."""
        return math.atan2(self.py, self.px)

    @property
    def rapidity(self) -> float:
        """Rapidity y = 0.5 · ln((E+pz)/(E-pz)).
        Clamped to ±1e10 for beam-direction particles."""
        num = self.E + self.pz
        den = self.E - self.pz
        if den <= 0.0 or num <= 0.0:
            return 1e10 if self.pz > 0.0 else -1e10
        return 0.5 * math.log(num / den)

    @property
    def beta(self) -> float:
        """Velocity β = |p| / E (dimensionless, in units of c)."""
        if self.E <= 0.0:
            return 0.0
        return min(self.p / self.E, 1.0)

    @property
    def gamma(self) -> float:
        """Lorentz factor γ = E / M.  Returns inf for massless particles."""
        m = self.mass
        if m <= 0.0:
            return float("inf")
        return self.E / m

    # ------------------------------------------------------------------
    # Angular separation from another LorentzVector
    # ------------------------------------------------------------------

    def delta_r(self, other: LorentzVector) -> float:
        """Compute ΔR = sqrt(Δη² + Δφ²) between this and another 4-vector.

        Parameters
        ----------
        other : LorentzVector
            The second particle.

        Returns
        -------
        float
            ΔR distance (always >= 0).
        """
        d_eta = self.eta - other.eta
        d_phi = self.phi - other.phi
        # Wrap Δφ to [-π, π]
        while d_phi > math.pi:
            d_phi -= 2.0 * math.pi
        while d_phi < -math.pi:
            d_phi += 2.0 * math.pi
        return math.sqrt(d_eta**2 + d_phi**2)

    def invariant_mass_with(self, other: LorentzVector) -> float:
        """Compute invariant mass of this + other 4-vector system.

        Parameters
        ----------
        other : LorentzVector
            The second particle.

        Returns
        -------
        float
            Invariant mass M in GeV.  Returns 0 if M² < 0 (numerical rounding).
        """
        return (self + other).mass

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: LorentzVector) -> LorentzVector:
        """Add two 4-vectors component-wise."""
        return LorentzVector(
            E=self.E + other.E,
            px=self.px + other.px,
            py=self.py + other.py,
            pz=self.pz + other.pz,
        )

    def __sub__(self, other: LorentzVector) -> LorentzVector:
        """Subtract two 4-vectors component-wise."""
        return LorentzVector(
            E=self.E - other.E,
            px=self.px - other.px,
            py=self.py - other.py,
            pz=self.pz - other.pz,
        )

    def __neg__(self) -> LorentzVector:
        """Unary negation (all components negated)."""
        return LorentzVector(E=-self.E, px=-self.px, py=-self.py, pz=-self.pz)

    def __eq__(self, other: object) -> bool:
        """Component-wise equality."""
        if not isinstance(other, LorentzVector):
            return NotImplemented
        return (
            self.E == other.E
            and self.px == other.px
            and self.py == other.py
            and self.pz == other.pz
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return the 4-vector as a NumPy array [E, px, py, pz] (shape (4,))."""
        return np.array([self.E, self.px, self.py, self.pz], dtype=np.float64)

    @classmethod
    def from_pt_eta_phi_mass(
        cls,
        pt: float,
        eta: float,
        phi: float,
        mass: float = 0.0,
    ) -> LorentzVector:
        """Construct a LorentzVector from the (pT, η, φ, m) representation.

        Parameters
        ----------
        pt : float
            Transverse momentum in GeV.
        eta : float
            Pseudorapidity η.
        phi : float
            Azimuthal angle φ in radians.
        mass : float
            Particle mass in GeV.  Default 0 (massless).

        Returns
        -------
        LorentzVector
        """
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        p = pt * math.cosh(eta)
        E = math.sqrt(p**2 + mass**2)
        return cls(E=E, px=px, py=py, pz=pz)

    def __repr__(self) -> str:
        return (
            f"LorentzVector(E={self.E:.4g}, px={self.px:.4g}, "
            f"py={self.py:.4g}, pz={self.pz:.4g}, "
            f"mass={self.mass:.4g} GeV, pt={self.pt:.4g} GeV)"
        )
