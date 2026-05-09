"""
triples_sigfast.hep
────────────────────
High-Energy Physics (HEP) analysis toolkit for triples-sigfast.

Modules
-------
kinematics  — Lorentz 4-vector mathematics (invariant mass, η, ΔR, pT, φ, y)
jets        — Anti-kT jet clustering (CERN standard algorithm)
"""

from .jets import Jet, cluster_jets
from .kinematics import (
    azimuthal_angle,
    calculate_invariant_mass,
    calculate_pseudorapidity,
    delta_r_matching,
    rapidity,
    transverse_momentum,
)

__all__ = [
    "calculate_invariant_mass",
    "calculate_pseudorapidity",
    "delta_r_matching",
    "transverse_momentum",
    "azimuthal_angle",
    "rapidity",
    "Jet",
    "cluster_jets",
]
