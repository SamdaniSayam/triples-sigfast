"""
triples_sigfast.nuclear.dose
─────────────────────────────
Point source dose rate calculations.

Implements gamma-ray and neutron dose rate calculations for unshielded
and shielded point sources using ICRP 74 flux-to-dose conversion factors
and Beer-Lambert / GP buildup attenuation.

References
----------
- ICRP Publication 74: Conversion Coefficients for Radiological Protection
- NCRP Report 49: Structural Shielding Design and Evaluation
- Shultis & Faw, "Radiation Shielding", ANS (2000), Ch. 9
"""

from __future__ import annotations

import numpy as np

from triples_sigfast.nuclear.shielding import attenuation_with_buildup

# ── ICRP 74 flux-to-dose coefficients h_phi [pSv·cm²] ───────────────────────
# Converts photon/neutron fluence rate [cm⁻²·s⁻¹] to ambient dose H*(10)
# Photon energies: MeV; values: pSv·cm²
# Source: ICRP Publication 74, Table A.1

_PHOTON_H_PHI: dict[float, float] = {
    0.010: 0.0685,
    0.015: 0.1560,
    0.020: 0.2560,
    0.030: 0.4560,
    0.050: 0.8570,
    0.060: 1.0600,
    0.080: 1.4100,
    0.100: 1.7300,
    0.150: 2.5200,
    0.200: 2.9700,
    0.300: 3.5100,
    0.400: 3.7300,
    0.500: 3.7700,
    0.600: 3.7400,
    0.800: 3.6300,
    1.000: 3.4400,
    1.250: 3.2200,
    1.500: 3.0100,
    2.000: 2.7200,
    3.000: 2.3000,
    4.000: 2.0800,
    5.000: 1.9600,
    6.000: 1.9000,
    8.000: 1.9200,
    10.00: 2.0000,
}

# Neutron h_phi [pSv·cm²] — ICRP 74 Table A.41
_NEUTRON_H_PHI: dict[float, float] = {
    1.0e-9: 2.69e-3,
    1.0e-8: 4.36e-3,
    2.5e-8: 5.47e-3,
    1.0e-7: 6.69e-3,
    2.0e-7: 7.08e-3,
    5.0e-7: 7.38e-3,
    1.0e-6: 7.72e-3,
    1.0e-5: 9.50e-3,
    1.0e-4: 1.18e-2,
    1.0e-3: 1.78e-2,
    1.0e-2: 3.45e-2,
    0.100: 1.23e-1,
    0.500: 3.58e-1,
    1.000: 4.58e-1,
    2.000: 5.55e-1,
    5.000: 6.48e-1,
    10.00: 7.07e-1,
    20.00: 8.24e-1,
}

# Unit conversion: pSv/s → µSv/hr
_PSV_S_TO_USV_HR = 3.6e-3

# ── Pre-computed log-log interpolation arrays (computed once at module load) ─────────
# Sorting and np.log() are constant-time operations on fixed-size tables.
# Pre-computing here eliminates redundant work on every flux_to_dose /
# point_source / dose_rate_vs_distance call.
_PHOTON_E_SORTED = np.array(sorted(_PHOTON_H_PHI.keys()), dtype=np.float64)
_PHOTON_H_SORTED = np.array(
    [_PHOTON_H_PHI[e] for e in _PHOTON_E_SORTED], dtype=np.float64
)
_LOG_PHOTON_E = np.log(_PHOTON_E_SORTED)
_LOG_PHOTON_H = np.log(_PHOTON_H_SORTED)

_NEUTRON_E_SORTED = np.array(sorted(_NEUTRON_H_PHI.keys()), dtype=np.float64)
_NEUTRON_H_SORTED = np.array(
    [_NEUTRON_H_PHI[e] for e in _NEUTRON_E_SORTED], dtype=np.float64
)
_LOG_NEUTRON_E = np.log(_NEUTRON_E_SORTED)
_LOG_NEUTRON_H = np.log(_NEUTRON_H_SORTED)


def _interpolate_h_phi(energy_mev: float, table: dict) -> float:
    """Log-log interpolate h_phi from ICRP 74 table.

    Uses pre-computed sorted arrays and log-tables to avoid repeated
    allocation and np.log() calls on every invocation.
    """
    if table is _PHOTON_H_PHI:
        log_e_arr, log_h_arr = _LOG_PHOTON_E, _LOG_PHOTON_H
    else:
        log_e_arr, log_h_arr = _LOG_NEUTRON_E, _LOG_NEUTRON_H
    return float(np.exp(np.interp(np.log(energy_mev), log_e_arr, log_h_arr)))


def point_source(
    activity_bq: float,
    energy_mev: float,
    distance_cm: float,
    particle: str = "gamma",
    photons_per_decay: float = 1.0,
) -> float:
    """
    Compute ambient dose rate H*(10) from an unshielded point source.

    Uses the point-source approximation with ICRP 74 flux-to-dose
    conversion coefficients:

        Φ = A × n / (4π × r²)
        H = Φ × h_phi

    Parameters
    ----------
    activity_bq : float
        Source activity in Becquerels.
    energy_mev : float
        Photon or neutron energy in MeV.
    distance_cm : float
        Distance from source to dose point in centimetres. Must be > 0.
    particle : str, optional
        Radiation type: 'gamma' or 'neutron'. Default 'gamma'.
    photons_per_decay : float, optional
        Number of photons (or neutrons) emitted per decay. Default 1.0.
        For Co-60: use 2.0 (two gammas per decay).

    Returns
    -------
    float
        Ambient dose rate H*(10) in µSv/hr.

    Raises
    ------
    ValueError
        If particle is not 'gamma' or 'neutron', or if distance_cm <= 0.

    Examples
    --------
    >>> # 1 GBq Co-60 source at 1 metre
    >>> rate = point_source(
    ...     activity_bq=1e9,
    ...     energy_mev=1.25,
    ...     distance_cm=100,
    ...     particle="gamma",
    ...     photons_per_decay=2.0,
    ... )
    >>> print(f"Dose rate: {rate:.2f} µSv/hr")
    """
    if distance_cm <= 0:
        raise ValueError(f"distance_cm must be > 0, got {distance_cm}")
    if particle not in ("gamma", "neutron"):
        raise ValueError(f"particle must be 'gamma' or 'neutron', got '{particle}'")
    if energy_mev <= 0:
        raise ValueError(f"energy_mev must be > 0, got {energy_mev}")

    table = _PHOTON_H_PHI if particle == "gamma" else _NEUTRON_H_PHI
    h_phi = _interpolate_h_phi(energy_mev, table)

    # Fluence rate at distance r [cm⁻²·s⁻¹]
    fluence_rate = (activity_bq * photons_per_decay) / (4.0 * np.pi * distance_cm**2)

    # Dose rate [pSv/s] → [µSv/hr]
    dose_pSv_s = fluence_rate * h_phi
    return float(dose_pSv_s * _PSV_S_TO_USV_HR)


def point_source_shielded(
    activity_bq: float,
    energy_mev: float,
    distance_cm: float,
    shield_material: str,
    shield_thickness_cm: float,
    particle: str = "gamma",
    photons_per_decay: float = 1.0,
) -> float:
    """
    Compute dose rate from a point source with a single-layer shield.

    Combines the point-source dose model with GP buildup-corrected
    attenuation through the shield:

        H_shielded = H_unshielded × T(shield)

    where T = B(μx) × exp(−μx) from the GP buildup model.

    Parameters
    ----------
    activity_bq : float
        Source activity in Becquerels.
    energy_mev : float
        Photon energy in MeV.
    distance_cm : float
        Total distance from source to dose point in centimetres.
        The shield is assumed to fill part of this distance.
    shield_material : str
        Shield material: 'lead', 'iron', 'concrete', 'water',
        'polyethylene', 'aluminum'.
    shield_thickness_cm : float
        Shield thickness in centimetres.
    particle : str, optional
        Radiation type: 'gamma' or 'neutron'. Default 'gamma'.
        Note: buildup factors are only available for gamma; neutron
        shielding uses simple Beer-Lambert.
    photons_per_decay : float, optional
        Photons emitted per decay. Default 1.0.

    Returns
    -------
    float
        Shielded ambient dose rate H*(10) in µSv/hr.

    Examples
    --------
    >>> rate = point_source_shielded(
    ...     activity_bq=1e9,
    ...     energy_mev=1.25,
    ...     distance_cm=100,
    ...     shield_material="lead",
    ...     shield_thickness_cm=5.0,
    ... )
    >>> print(f"Shielded dose rate: {rate:.4f} µSv/hr")
    """
    unshielded = point_source(
        activity_bq=activity_bq,
        energy_mev=energy_mev,
        distance_cm=distance_cm,
        particle=particle,
        photons_per_decay=photons_per_decay,
    )
    transmission = attenuation_with_buildup(
        thickness_cm=shield_thickness_cm,
        material=shield_material,
        energy_mev=energy_mev,
    )
    return float(unshielded * transmission)


def dose_rate_vs_distance(
    activity_bq: float,
    energy_mev: float,
    distances_cm: np.ndarray,
    particle: str = "gamma",
    photons_per_decay: float = 1.0,
) -> np.ndarray:
    """
    Compute dose rate profile as a function of distance from a point source.

    Parameters
    ----------
    activity_bq : float
        Source activity in Becquerels.
    energy_mev : float
        Energy in MeV.
    distances_cm : np.ndarray
        Array of distances in centimetres. Must all be > 0.
    particle : str, optional
        'gamma' or 'neutron'.
    photons_per_decay : float, optional
        Photons per decay.

    Returns
    -------
    np.ndarray
        Dose rates in µSv/hr, same shape as distances_cm.

    Examples
    --------
    >>> import numpy as np
    >>> distances = np.linspace(10, 500, 200)
    >>> rates = dose_rate_vs_distance(1e9, 1.25, distances)
    """
    if particle not in ("gamma", "neutron"):
        raise ValueError(f"particle must be 'gamma' or 'neutron', got '{particle}'")

    table = _PHOTON_H_PHI if particle == "gamma" else _NEUTRON_H_PHI
    # h_phi depends only on energy — compute once, not once per distance
    h_phi = _interpolate_h_phi(energy_mev, table)

    distances_cm = np.asarray(distances_cm, dtype=np.float64)
    # Vectorized 1/r² law applied to the whole array in a single NumPy operation
    fluence_rates = (activity_bq * photons_per_decay) / (4.0 * np.pi * distances_cm**2)
    return fluence_rates * h_phi * _PSV_S_TO_USV_HR


def inverse_square_distance(
    activity_bq: float,
    energy_mev: float,
    target_dose_usvhr: float,
    particle: str = "gamma",
    photons_per_decay: float = 1.0,
) -> float:
    """
    Compute the minimum safe distance for a given dose rate limit.

    Solves H = Φ × h_phi for distance r given a target dose rate.

    Parameters
    ----------
    activity_bq : float
        Source activity in Becquerels.
    energy_mev : float
        Energy in MeV.
    target_dose_usvhr : float
        Maximum acceptable dose rate in µSv/hr.
    particle : str, optional
        'gamma' or 'neutron'.
    photons_per_decay : float, optional
        Photons per decay.

    Returns
    -------
    float
        Minimum safe distance in centimetres.

    Examples
    --------
    >>> # Minimum distance for < 1 µSv/hr from 1 GBq Co-60
    >>> d = inverse_square_distance(1e9, 1.25, 1.0, photons_per_decay=2.0)
    >>> print(f"Safe distance: {d/100:.1f} m")
    """
    table = _PHOTON_H_PHI if particle == "gamma" else _NEUTRON_H_PHI
    h_phi = _interpolate_h_phi(energy_mev, table)

    # H [µSv/hr] = A × n × h_phi × 3.6e-3 / (4π × r²)
    # Solve for r:
    numerator = activity_bq * photons_per_decay * h_phi * _PSV_S_TO_USV_HR
    r_squared = numerator / (4.0 * np.pi * target_dose_usvhr)
    return float(np.sqrt(r_squared))
