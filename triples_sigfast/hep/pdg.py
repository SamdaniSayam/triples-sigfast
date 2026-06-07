"""
triples_sigfast.hep.pdg
-----------------------
AOT ingestion of PDG particle data for fast cache-safe, GIL-free lookups.
Uses the `particle` package to build flat NumPy arrays of IDs, masses, and widths.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from particle import Particle


def _build_pdg_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particles = Particle.findall()
    unique_particles = {}
    for p in particles:
        pid = int(p.pdgid)
        if pid not in unique_particles:
            unique_particles[pid] = p

    sorted_ids = sorted(unique_particles.keys())
    pdg_ids = np.array(sorted_ids, dtype=np.int64)

    # Particle properties are in MeV, convert to GeV.
    masses = np.array(
        [
            unique_particles[pid].mass / 1000.0
            if unique_particles[pid].mass is not None
            else 0.0
            for pid in sorted_ids
        ],
        dtype=np.float64,
    )

    widths = np.array(
        [
            unique_particles[pid].width / 1000.0
            if unique_particles[pid].width is not None
            else 0.0
            for pid in sorted_ids
        ],
        dtype=np.float64,
    )

    return pdg_ids, masses, widths


# Build arrays at import time
_pdg_ids, _masses, _widths = _build_pdg_arrays()


@njit(cache=True)
def get_mass(pdgid: int) -> float:
    """Get the mass of a particle in GeV by its PDG ID."""
    idx = np.searchsorted(_pdg_ids, pdgid)
    if idx < len(_pdg_ids) and _pdg_ids[idx] == pdgid:
        return _masses[idx]
    return 0.0


@njit(cache=True)
def get_width(pdgid: int) -> float:
    """Get the decay width of a particle in GeV by its PDG ID."""
    idx = np.searchsorted(_pdg_ids, pdgid)
    if idx < len(_pdg_ids) and _pdg_ids[idx] == pdgid:
        return _widths[idx]
    return 0.0


@njit(cache=True, parallel=True)
def get_mass_array(pdgids: np.ndarray) -> np.ndarray:
    """Get the masses of particles in GeV by their PDG IDs in parallel."""
    n = len(pdgids)
    out = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        pid = pdgids[i]
        idx = np.searchsorted(_pdg_ids, pid)
        if idx < len(_pdg_ids) and _pdg_ids[idx] == pid:
            out[i] = _masses[idx]
    return out


@njit(cache=True, parallel=True)
def get_width_array(pdgids: np.ndarray) -> np.ndarray:
    """Get the decay widths of particles in GeV by their PDG IDs in parallel."""
    n = len(pdgids)
    out = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        pid = pdgids[i]
        idx = np.searchsorted(_pdg_ids, pid)
        if idx < len(_pdg_ids) and _pdg_ids[idx] == pid:
            out[i] = _widths[idx]
    return out
