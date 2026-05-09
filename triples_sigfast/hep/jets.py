"""
triples_sigfast.hep.jets
─────────────────────────
JIT-compiled anti-kT jet clustering algorithm.

Implements the anti-kT algorithm (Cacciari, Salam, Soyez 2008), the CERN
standard for jet reconstruction at the LHC. Provides a Python-native
alternative to the C++ FastJet library.

Algorithm summary
-----------------
For each pair of particles (i, j), compute:

    d_ij = min(pT_i^{-2}, pT_j^{-2}) · ΔR_ij² / R²
    d_iB = pT_i^{-2}

Find the global minimum:
  - If min is d_ij → merge particles i and j (add 4-vectors).
  - If min is d_iB → promote particle i to a final jet, remove from list.

Repeat until no particles remain.

Complexity
----------
This implementation is O(N²) per clustering step — correct for Numba and
competitive with FastJet for typical LHC event multiplicities (N < 500
particles). For heavy-ion collisions (N > 5000), FastJet's O(N ln N)
nearest-neighbour tree is faster; document this clearly for users.

References
----------
Cacciari, Salam, Soyez, JHEP 04 (2008) 063.
  https://arxiv.org/abs/0802.1189
FastJet manual: https://fastjet.fr/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

# ============================================================
#  JET DATACLASS
# ============================================================


@dataclass
class Jet:
    """
    A reconstructed jet from anti-kT clustering.

    Attributes
    ----------
    px, py, pz, E : float
        4-vector components in GeV.
    constituents : np.ndarray of int
        Indices into the original input particle arrays.
    pt : float
        Transverse momentum sqrt(px²+py²) in GeV.
    eta : float
        Pseudorapidity η.
    phi : float
        Azimuthal angle φ in radians [-π, π].
    mass : float
        Jet invariant mass in GeV.
    """

    px: float
    py: float
    pz: float
    E: float
    constituents: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    @property
    def pt(self) -> float:
        return math.sqrt(self.px**2 + self.py**2)

    @property
    def eta(self) -> float:
        p = math.sqrt(self.px**2 + self.py**2 + self.pz**2)
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
        return math.atan2(self.py, self.px)

    @property
    def mass(self) -> float:
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return math.sqrt(m2) if m2 >= 0.0 else 0.0

    def __repr__(self) -> str:
        return (
            f"Jet(pt={self.pt:.2f} GeV, eta={self.eta:.3f}, "
            f"phi={self.phi:.3f}, mass={self.mass:.2f} GeV, "
            f"n_constituents={len(self.constituents)})"
        )


# ============================================================
#  NUMBA JIT KERNELS
# ============================================================


@njit(fastmath=True, cache=True)
def _compute_pt2(px: np.ndarray, py: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Compute pT² for all active particles."""
    n = len(px)
    pt2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        pt2[i] = px[i] * px[i] + py[i] * py[i]
    return pt2


@njit(fastmath=True, cache=True)
def _delta_r2(
    eta_i: float, phi_i: float, eta_j: float, phi_j: float
) -> float:  # pragma: no cover
    """ΔR² between two particles with correct Δφ wrap."""
    d_eta = eta_i - eta_j
    d_phi = phi_i - phi_j
    # Wrap Δφ to [-π, π]
    while d_phi > math.pi:
        d_phi -= 2.0 * math.pi
    while d_phi < -math.pi:
        d_phi += 2.0 * math.pi
    return d_eta * d_eta + d_phi * d_phi


@njit(fastmath=True, cache=True)
def _pseudorapidity(px: float, py: float, pz: float) -> float:  # pragma: no cover
    """Single-particle pseudorapidity."""
    p = math.sqrt(px * px + py * py + pz * pz)
    if p <= 0.0:
        return 0.0
    denom = p - pz
    if denom <= 0.0:
        return 1e10
    if p + pz <= 0.0:
        return -1e10
    return 0.5 * math.log((p + pz) / denom)


@njit(fastmath=True, cache=True)
def _find_minimum(
    pt2: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    active: np.ndarray,
    R2: float,
) -> tuple[float, int, int]:  # pragma: no cover
    """
    Find the global minimum distance in the anti-kT metric.

    Returns (min_dist, i_min, j_min).
    If j_min == -1, particle i_min should be promoted to a jet (d_iB is min).
    """
    min_dist = 1e300
    i_min = -1
    j_min = -1

    active_list = []
    for i in range(len(active)):
        if active[i]:
            active_list.append(i)

    n_active = len(active_list)

    for ai in range(n_active):
        i = active_list[ai]
        pt2_i = pt2[i]
        inv_pt2_i = 1.0 / pt2_i if pt2_i > 0.0 else 1e300

        # d_iB = pT_i^{-2} (anti-kT uses negative power → hard particles cluster last)
        d_iB = inv_pt2_i
        if d_iB < min_dist:
            min_dist = d_iB
            i_min = i
            j_min = -1

        # d_ij = min(pT_i^{-2}, pT_j^{-2}) * ΔR_ij² / R²
        for aj in range(ai + 1, n_active):
            j = active_list[aj]
            pt2_j = pt2[j]
            inv_pt2_j = 1.0 / pt2_j if pt2_j > 0.0 else 1e300
            kt_factor = min(inv_pt2_i, inv_pt2_j)
            dr2 = _delta_r2(eta[i], phi[i], eta[j], phi[j])
            d_ij = kt_factor * dr2 / R2
            if d_ij < min_dist:
                min_dist = d_ij
                i_min = i
                j_min = j

    return min_dist, i_min, j_min


# ============================================================
#  PUBLIC CLUSTERING FUNCTION
# ============================================================


def cluster_jets(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    E: np.ndarray,
    R: float = 0.4,
    min_pt: float = 0.0,
) -> list[Jet]:
    """
    Cluster particles into jets using the anti-kT algorithm.

    This is the CERN standard jet algorithm used in all LHC analyses.
    Anti-kT produces geometrically regular (conical) jets that are
    insensitive to soft radiation, making them ideal for precision
    measurements and machine-learning-based analyses.

    Parameters
    ----------
    px, py, pz, E : np.ndarray, shape (N,)
        4-vector components of input particles in GeV.
    R : float
        Jet radius parameter. Typical values:
        0.4 → ATLAS/CMS small-R jets (boosted topology studies)
        0.8 → CMS AK8 jets (fat jets for W/Z/H tagging)
        1.0 → ATLAS large-R jets
    min_pt : float
        Minimum jet pT threshold in GeV. Jets below this are discarded.
        Default 0.0 (keep all jets).

    Returns
    -------
    list[Jet]
        Reconstructed jets sorted by pT (highest first). Each Jet object
        contains the 4-vector, pT, η, φ, mass, and constituent indices.

    Notes
    -----
    **Complexity:** O(N²) per clustering step. Competitive with FastJet
    for typical LHC event multiplicities (N < 500 particles per event).
    For heavy-ion events (N > 5000), FastJet's O(N ln N) KD-tree is faster.

    Examples
    --------
    >>> # Cluster final-state particles from a PYTHIA LHE file
    >>> reader = LHEReader("pythia.lhe")
    >>> p = reader.get_particles(status=1)
    >>> jets = cluster_jets(p["px"], p["py"], p["pz"], p["E"], R=0.4, min_pt=20.0)
    >>> print(f"Found {len(jets)} jets with pT > 20 GeV")
    >>> leading_jet = jets[0]
    >>> print(leading_jet)

    References
    ----------
    Cacciari, Salam, Soyez, JHEP 04 (2008) 063. arXiv:0802.1189
    """
    if R <= 0.0:
        raise ValueError("Jet radius R must be > 0.")
    if len(px) == 0:
        return []

    # Force float64 contiguous arrays
    px_w = np.ascontiguousarray(px, dtype=np.float64).copy()
    py_w = np.ascontiguousarray(py, dtype=np.float64).copy()
    pz_w = np.ascontiguousarray(pz, dtype=np.float64).copy()
    E_w = np.ascontiguousarray(E, dtype=np.float64).copy()
    n = len(px_w)

    # Active flag: True = particle still in pool
    active = np.ones(n, dtype=np.bool_)

    # Constituent tracking: list of constituent index sets per pseudo-particle
    constituents: list[list[int]] = [[i] for i in range(n)]

    R2 = R * R
    final_jets: list[Jet] = []

    # Precompute η and φ for all particles
    eta = np.array([_pseudorapidity(px_w[i], py_w[i], pz_w[i]) for i in range(n)])
    phi = np.arctan2(py_w, px_w)
    pt2 = _compute_pt2(px_w, py_w)

    while active.any():
        # Find minimum distance pair (JIT-compiled inner loop)
        _min_dist, i_min, j_min = _find_minimum(pt2, eta, phi, active, R2)

        if j_min == -1:
            # Particle i_min becomes a final jet
            jet = Jet(
                px=float(px_w[i_min]),
                py=float(py_w[i_min]),
                pz=float(pz_w[i_min]),
                E=float(E_w[i_min]),
                constituents=np.array(constituents[i_min], dtype=np.int32),
            )
            if jet.pt >= min_pt:
                final_jets.append(jet)
            active[i_min] = False
        else:
            # Merge particles i_min and j_min
            px_w[i_min] += px_w[j_min]
            py_w[i_min] += py_w[j_min]
            pz_w[i_min] += pz_w[j_min]
            E_w[i_min] += E_w[j_min]
            # Update cached kinematics for merged pseudo-particle
            pt2[i_min] = px_w[i_min] ** 2 + py_w[i_min] ** 2
            eta[i_min] = _pseudorapidity(px_w[i_min], py_w[i_min], pz_w[i_min])
            phi[i_min] = math.atan2(py_w[i_min], px_w[i_min])
            # Merge constituent lists
            constituents[i_min] = constituents[i_min] + constituents[j_min]
            active[j_min] = False

    # Sort jets by pT descending
    final_jets.sort(key=lambda j: j.pt, reverse=True)
    return final_jets
