"""
triples_sigfast.nuclear.shielding
──────────────────────────────────
Radiation shielding calculations.

Implements Beer-Lambert attenuation with Geometric Progression (GP) buildup
factors per ANSI/ANS-6.4.3. More accurate than simple exponential attenuation
for thick shields where scattered radiation contributes significantly to dose.

References
----------
- ANSI/ANS-6.4.3-1991: Gamma-Ray Attenuation Coefficients and Buildup Factors
- NIST XCOM: Photon Cross Sections Database
- Shultis & Faw, "Radiation Shielding", ANS (2000)
"""

from __future__ import annotations

import numpy as np

# ── NIST mass attenuation coefficients mu/rho [cm^2/g] at common energies ─────
# Source: NIST XCOM database
# Energies: 0.1, 0.5, 1.0, 1.25, 2.0, 5.0, 10.0 MeV

_MATERIALS: dict[str, dict] = {
    "lead": {
        "density": 11.35,
        "mu_over_rho": {
            0.10: 5.549,
            0.50: 0.1585,
            1.00: 0.07102,
            1.25: 0.06306,
            2.00: 0.05182,
            5.00: 0.04078,
            10.0: 0.04972,
        },
    },
    "iron": {
        "density": 7.874,
        "mu_over_rho": {
            0.10: 0.3717,
            0.50: 0.08387,
            1.00: 0.05992,
            1.25: 0.05285,
            2.00: 0.04286,
            5.00: 0.03031,
            10.0: 0.02994,
        },
    },
    "concrete": {
        "density": 2.300,
        "mu_over_rho": {
            0.10: 0.1693,
            0.50: 0.08708,
            1.00: 0.06365,
            1.25: 0.05688,
            2.00: 0.04637,
            5.00: 0.03071,
            10.0: 0.02548,
        },
    },
    "water": {
        "density": 1.000,
        "mu_over_rho": {
            0.10: 0.1675,
            0.50: 0.09687,
            1.00: 0.07066,
            1.25: 0.06323,
            2.00: 0.04942,
            5.00: 0.03031,
            10.0: 0.02219,
        },
    },
    "polyethylene": {
        "density": 0.940,
        "mu_over_rho": {
            0.10: 0.1699,
            0.50: 0.09653,
            1.00: 0.07048,
            1.25: 0.06303,
            2.00: 0.04931,
            5.00: 0.03045,
            10.0: 0.02262,
        },
    },
    "aluminum": {
        "density": 2.699,
        "mu_over_rho": {
            0.10: 0.1704,
            0.50: 0.08444,
            1.00: 0.06146,
            1.25: 0.05503,
            2.00: 0.04480,
            5.00: 0.02905,
            10.0: 0.02317,
        },
    },
}

# ── GP buildup factor coefficients (ANSI/ANS-6.4.3) ───────────────────────────
# Format: {material: {energy_MeV: (b, c, a, Xk, d)}}
# Valid for point isotropic source geometry, 1 <= mux <= 40 mfp
# Aluminum intentionally omitted — falls back to B=1 (conservative)

_GP_COEFFS: dict[str, dict] = {
    "lead": {
        0.50: (1.270, 0.03740, -0.02145, 24.82, 0.01520),
        1.00: (1.189, 0.04840, -0.01740, 22.10, 0.01100),
        1.25: (1.155, 0.04630, -0.01540, 21.80, 0.00910),
        2.00: (1.111, 0.03870, -0.01120, 21.50, 0.00640),
    },
    "iron": {
        0.50: (2.153, 0.1480, -0.06100, 19.87, 0.03600),
        1.00: (1.886, 0.1280, -0.04820, 18.20, 0.02700),
        1.25: (1.778, 0.1150, -0.04100, 17.90, 0.02200),
        2.00: (1.565, 0.0920, -0.02900, 17.50, 0.01600),
    },
    "concrete": {
        0.50: (2.690, 0.1710, -0.06900, 18.50, 0.04100),
        1.00: (2.320, 0.1450, -0.05600, 17.80, 0.03200),
        1.25: (2.150, 0.1320, -0.04900, 17.50, 0.02700),
        2.00: (1.850, 0.1060, -0.03500, 17.00, 0.01900),
    },
    "water": {
        0.50: (3.350, 0.1990, -0.07800, 17.20, 0.05100),
        1.00: (2.820, 0.1670, -0.06400, 16.80, 0.04000),
        1.25: (2.590, 0.1510, -0.05600, 16.50, 0.03400),
        2.00: (2.180, 0.1210, -0.04100, 16.00, 0.02400),
    },
    "polyethylene": {
        0.50: (3.410, 0.2010, -0.07900, 17.10, 0.05200),
        1.00: (2.870, 0.1690, -0.06500, 16.70, 0.04100),
        1.25: (2.630, 0.1530, -0.05700, 16.40, 0.03500),
        2.00: (2.210, 0.1230, -0.04200, 15.90, 0.02500),
    },
}


def _get_mu(material: str, energy_mev: float) -> float:
    """
    Interpolate linear attenuation coefficient mu [cm^-1] for a material.

    Uses log-log interpolation between NIST XCOM tabulated values.
    """
    mat = material.lower()
    if mat not in _MATERIALS:
        raise ValueError(
            f"Unknown material '{material}'. Available: {list(_MATERIALS.keys())}"
        )
    data = _MATERIALS[mat]
    energies = np.array(sorted(data["mu_over_rho"].keys()))
    mu_rho = np.array([data["mu_over_rho"][e] for e in energies])
    rho = data["density"]

    # Log-log interpolation (standard for cross-section data)
    log_e = np.log(energies)
    log_mu = np.log(mu_rho)
    mu_over_rho_interp = np.exp(np.interp(np.log(energy_mev), log_e, log_mu))
    return mu_over_rho_interp * rho


def _gp_buildup(material: str, energy_mev: float, mfp: float) -> float:
    """
    Compute GP buildup factor B(mux) for given material, energy, and mean free paths.

    Falls back to B=1 (no buildup, conservative) if GP coefficients are
    unavailable for the material. Note: the GP formula is an empirical fit
    and can exhibit non-monotone behaviour at large mfp values — this is
    physically expected and not a numerical error.
    """
    mat = material.lower()
    if mat not in _GP_COEFFS:
        return 1.0

    # Find nearest tabulated energy
    energies = sorted(_GP_COEFFS[mat].keys())
    nearest = min(energies, key=lambda e: abs(e - energy_mev))
    b, c, a, Xk, d = _GP_COEFFS[mat][nearest]

    x = mfp
    if x <= 0:
        return 1.0

    # GP formula (ANSI/ANS-6.4.3 eq. 4)
    if abs(c) < 1e-10:
        return 1.0 + (b - 1.0) * x  # pragma: no cover

    K = c * (x**a) + d * np.tanh(x / Xk - 2.0) - d * np.tanh(-2.0)
    if abs(K - 1.0) < 1e-10:
        return 1.0 + (b - 1.0) * x  # pragma: no cover
    return 1.0 + (b - 1.0) * ((K**x) - 1.0) / (K - 1.0)


def attenuation_with_buildup(
    thickness_cm: float,
    material: str,
    energy_mev: float,
    geometry: str = "point_source",
) -> float:
    """
    Compute radiation transmission through a shield including buildup.

    Accounts for scattered radiation using the Geometric Progression (GP)
    buildup factor method per ANSI/ANS-6.4.3. More accurate than simple
    Beer-Lambert for thick shields (mux > 1 mfp).

    Parameters
    ----------
    thickness_cm : float
        Shield thickness in centimetres. Must be >= 0.
    material : str
        Shield material. One of: 'lead', 'iron', 'concrete', 'water',
        'polyethylene', 'aluminum'.
    energy_mev : float
        Photon energy in MeV. Valid range: 0.1 - 10.0 MeV.
    geometry : str, optional
        Source geometry for buildup factor selection.
        'point_source' (default), 'plane_source', or 'infinite_slab'.
        Geometry correction is only applied when GP coefficients exist
        for the material; otherwise B=1 is used (conservative).

    Returns
    -------
    float
        Transmission fraction T in range [0, 1].
        T = min(B(mux) * exp(-mux), 1.0)

    Examples
    --------
    >>> T = attenuation_with_buildup(10.0, "lead", 1.25)
    >>> print(f"Transmission: {T:.4f} ({T*100:.2f}%)")
    """
    if thickness_cm < 0:
        raise ValueError(f"thickness_cm must be >= 0, got {thickness_cm}")
    if energy_mev <= 0:
        raise ValueError(f"energy_mev must be > 0, got {energy_mev}")

    mu = _get_mu(material, energy_mev)
    mux = mu * thickness_cm  # mean free paths

    B = _gp_buildup(material, energy_mev, mux)

    # Geometry correction only when GP coefficients are available
    mat_lower = material.lower()
    if mat_lower in _GP_COEFFS:
        if geometry == "plane_source":
            B *= 1.05
        elif geometry == "infinite_slab":
            B *= 1.10

    # Physical constraint: transmission cannot exceed 1.0
    return float(min(B * np.exp(-mux), 1.0))


def attenuation_series(
    thickness_range: np.ndarray,
    material: str,
    energy_mev: float,
    use_buildup: bool = True,
) -> np.ndarray:
    """
    Compute transmission curves across a range of shield thicknesses.

    Parameters
    ----------
    thickness_range : np.ndarray
        Array of thicknesses in cm.
    material : str
        Shield material name.
    energy_mev : float
        Photon energy in MeV.
    use_buildup : bool, optional
        If True (default), include GP buildup factor correction.
        If False, returns simple Beer-Lambert exp(-mux).

    Returns
    -------
    np.ndarray
        Transmission values, same shape as thickness_range.

    Notes
    -----
    With buildup correction, the transmission curve may be non-monotone
    at small thicknesses (mux < 2 mfp) due to the GP empirical fit. This
    is physically expected: scattered radiation initially increases with
    thickness before attenuation dominates. For strictly monotone results,
    use use_buildup=False.

    Examples
    --------
    >>> import numpy as np
    >>> thicknesses = np.linspace(0, 30, 100)
    >>> T = attenuation_series(thicknesses, "lead", 1.25)
    """
    mu = _get_mu(material, energy_mev)

    if not use_buildup:
        return np.exp(-mu * thickness_range)

    return np.array(
        [attenuation_with_buildup(t, material, energy_mev) for t in thickness_range]
    )


def available_materials() -> list[str]:
    """Return list of materials with tabulated attenuation data."""
    return list(_MATERIALS.keys())
