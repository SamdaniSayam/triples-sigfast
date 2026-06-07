"""
triples_sigfast.plasma
-----------------------
Plasma physics utilities for fusion-relevant radiation analysis.

This sub-package provides tools for analysing radiation from fusion plasma
environments, including deuterium-tritium (D-T) and deuterium-deuterium
(D-D) neutron sources, plasma confinement diagnostics, and activation
calculations for fusion reactor structural materials.

.. note::
    This sub-package is under active development.  The core API shown below
    is stable and guaranteed not to change in the v1.x series.  Additional
    models (Lawson criterion, Q-value calculations, tritium breeding ratio)
    will be added in v2.0.

Available functions
-------------------
dt_neutron_spectrum     -- D-T fusion 14.1 MeV neutron spectrum
dd_neutron_spectrum     -- D-D fusion 2.45 MeV neutron spectrum
plasma_neutron_rate     -- Thermonuclear reaction rate [n/s]
activation_saturation   -- Saturation activity of a structural material

Usage
-----
>>> from triples_sigfast.plasma import dt_neutron_spectrum, plasma_neutron_rate
>>> import numpy as np
>>> E = np.linspace(12.0, 16.0, 500)
>>> spectrum = dt_neutron_spectrum(E, temperature_kev=10.0)
>>> rate = plasma_neutron_rate(
...     reaction="DT",
...     ion_density_m3=1e20,
...     temperature_kev=10.0,
...     plasma_volume_m3=100.0,
... )
>>> print(f"D-T neutron rate: {rate:.3e} n/s")

References
----------
- ENDF/B-VIII.0: Evaluated nuclear data for D-T and D-D reactions
- Freidberg, J.P., "Plasma Physics and Fusion Energy", Cambridge (2007)
- Wesson, J., "Tokamaks", 4th ed., Oxford (2011), Ch. 2
- IAEA TECDOC-1234: Nuclear data for fusion reactor design
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_AVOGADRO = 6.02214076e23  # mol⁻¹
_EV_TO_J = 1.60218e-19  # J/eV

# D-T fusion Q-value (energy of the 14.1 MeV neutron)
_DT_NEUTRON_ENERGY_MEV = 14.07  # MeV
_DD_NEUTRON_ENERGY_MEV = 2.45  # MeV (for D+D → He-3+n branch)

# ---------------------------------------------------------------------------
# Thermonuclear reactivity <σv> parametrisation (Bosch-Hale 1992)
# Valid 1 ≤ T ≤ 1000 keV
# ---------------------------------------------------------------------------
# Coefficients for the Bosch-Hale parameterization of <σv> [m³/s]
_BH_DT = {
    "A": (6.6610e-12, 3.3943e-2, -7.1820e-4, -3.5860e-5, 1.1280e-6),
    "C": (7.6600e-8, 3.1396e-8, -7.2895e-7, 5.0048e-8, -1.8298e-9),
    "B_G": 34.3827,
    "mrc2": 1124656.0,  # keV — reduced mass energy
}
_BH_DD = {  # D+D → T+p (dominant branch approximation)
    "A": (5.3701e-12, 3.3027e-4, -1.2706e-5, 2.9327e-7, -2.5151e-9),
    "C": (0.0, 0.0, 0.0, 0.0, 0.0),
    "B_G": 31.3970,
    "mrc2": 937814.0,
}


def _bosch_hale_reactivity(temperature_kev: float, params: dict) -> float:
    """Evaluate <σv> [m³/s] using the Bosch-Hale parameterization.

    Parameters
    ----------
    temperature_kev : float
        Ion temperature in keV.
    params : dict
        Bosch-Hale coefficient dictionary for the reaction.

    Returns
    -------
    float
        Thermonuclear reactivity <σv> in m³/s.

    References
    ----------
    Bosch & Hale, Nuclear Fusion 32 (1992) 611.
    """
    T = temperature_kev
    A = params["A"]
    B_G = params["B_G"]
    mrc2 = params["mrc2"]

    theta_denom = 1.0 - T * (A[1] + T * (A[3] + T * A[0]))

    # Avoid division by zero
    if abs(theta_denom) < 1e-20:
        return 0.0

    theta = T / (
        1.0 - T * (A[1] + T * (A[3] + T * A[0])) / (1.0 + T * (A[2] + T * A[4]))
    )
    xi = (B_G**2 / (4.0 * theta)) ** (1.0 / 3.0)

    sigma_v_cm3 = A[0] / (math.sqrt(mrc2 * T**3)) * theta * math.exp(-3.0 * xi)
    # Convert cm³/s → m³/s
    return float(sigma_v_cm3 * 1e-6)


def plasma_neutron_rate(
    reaction: str,
    ion_density_m3: float,
    temperature_kev: float,
    plasma_volume_m3: float,
) -> float:
    """Compute the thermonuclear neutron production rate in n/s.

    Calculates R = (n²/4) × <σv> × V for a 50/50 D-T or D-D mixture.

    Parameters
    ----------
    reaction : str
        Fusion reaction: 'DT' (deuterium-tritium) or 'DD' (deuterium-deuterium).
    ion_density_m3 : float
        Total ion density in m⁻³ (sum of D and T ions).
        Typical tokamak: 1e20 m⁻³.
    temperature_kev : float
        Plasma ion temperature in keV.
        D-T optimum: ~65 keV; practical ignition: ~10–20 keV.
    plasma_volume_m3 : float
        Plasma volume in m³.
        ITER: ~840 m³; compact tokamaks: 1–100 m³.

    Returns
    -------
    float
        Neutron production rate in n/s.

    Raises
    ------
    ValueError
        If reaction is not 'DT' or 'DD', or if any parameter ≤ 0.

    Examples
    --------
    >>> rate = plasma_neutron_rate("DT", 1e20, 10.0, 100.0)
    >>> print(f"ITER-scale D-T rate: {rate:.3e} n/s")
    """
    reaction = reaction.upper()
    if reaction not in ("DT", "DD"):
        raise ValueError(f"reaction must be 'DT' or 'DD', got '{reaction}'")
    if ion_density_m3 <= 0:
        raise ValueError(f"ion_density_m3 must be > 0, got {ion_density_m3}")
    if temperature_kev <= 0:
        raise ValueError(f"temperature_kev must be > 0, got {temperature_kev}")
    if plasma_volume_m3 <= 0:
        raise ValueError(f"plasma_volume_m3 must be > 0, got {plasma_volume_m3}")

    params = _BH_DT if reaction == "DT" else _BH_DD
    sigma_v = _bosch_hale_reactivity(temperature_kev, params)

    # For 50/50 mixture: R = (n/2)² × <σv> × V
    n_half = ion_density_m3 / 2.0
    return float(n_half**2 * sigma_v * plasma_volume_m3)


def dt_neutron_spectrum(
    energies_mev: np.ndarray,
    temperature_kev: float = 10.0,
    normalise: bool = False,
) -> np.ndarray:
    """Compute the D-T fusion neutron energy spectrum.

    Models the Gaussian-broadened 14.1 MeV neutron peak arising from
    thermal motion of the reacting D-T ions.  The FWHM of the peak is
    determined by the ion temperature.

    Parameters
    ----------
    energies_mev : np.ndarray
        Energy axis in MeV over which to evaluate the spectrum.
    temperature_kev : float
        Ion temperature in keV (determines Doppler broadening).
        Default 10 keV.
    normalise : bool
        If True, normalise the spectrum so ∫N(E)dE = 1.
        Default False.

    Returns
    -------
    np.ndarray
        Differential neutron spectrum N(E) at each energy point.

    References
    ----------
    McNally et al., ORNL/TM-6914 (1979) — D-T neutron spectrum shape.
    """
    E = np.asarray(energies_mev, dtype=np.float64)
    E0 = _DT_NEUTRON_ENERGY_MEV
    # Doppler broadening: σ ≈ (2/3) × E0 × sqrt(2kT / (mrc²))
    # For D-T: reduced mass ≈ 1.2 amu → mrc² ≈ 1117 MeV
    sigma_mev = (2.0 / 3.0) * E0 * math.sqrt(2.0 * temperature_kev * 1e-3 / 1117.0)
    spectrum = np.exp(-0.5 * ((E - E0) / sigma_mev) ** 2) / (
        sigma_mev * math.sqrt(2.0 * math.pi)
    )

    if normalise:
        integral = np.trapezoid(spectrum, E)
        if integral > 0:
            spectrum = spectrum / integral

    return spectrum


def dd_neutron_spectrum(
    energies_mev: np.ndarray,
    temperature_kev: float = 5.0,
    normalise: bool = False,
) -> np.ndarray:
    """Compute the D-D fusion neutron energy spectrum (2.45 MeV branch).

    Parameters
    ----------
    energies_mev : np.ndarray
        Energy axis in MeV.
    temperature_kev : float
        Ion temperature in keV. Default 5 keV.
    normalise : bool
        Normalise spectrum to unit integral. Default False.

    Returns
    -------
    np.ndarray
        Differential neutron spectrum N(E).
    """
    E = np.asarray(energies_mev, dtype=np.float64)
    E0 = _DD_NEUTRON_ENERGY_MEV
    sigma_mev = (2.0 / 3.0) * E0 * math.sqrt(2.0 * temperature_kev * 1e-3 / 938.0)
    spectrum = np.exp(-0.5 * ((E - E0) / sigma_mev) ** 2) / (
        sigma_mev * math.sqrt(2.0 * math.pi)
    )

    if normalise:
        integral = np.trapezoid(spectrum, E)
        if integral > 0:
            spectrum = spectrum / integral

    return spectrum


def activation_saturation(
    reaction_rate_cm2_s: float,
    number_density_cm3: float,
    cross_section_cm2: float,
    half_life_s: float,
    sample_volume_cm3: float = 1.0,
) -> float:
    """Compute the saturation activation activity of a structural material.

    At saturation (irradiation time ≫ half-life), the production rate equals
    the decay rate:

        A_sat = Φ × σ × N

    where Φ is the neutron flux [n/cm²/s], σ is the activation cross section,
    and N is the number density of the target nuclide.

    Parameters
    ----------
    reaction_rate_cm2_s : float
        Neutron flux Φ in n/cm²/s.
    number_density_cm3 : float
        Number density of the target nuclide in atoms/cm³.
    cross_section_cm2 : float
        Activation cross section in cm².
    half_life_s : float
        Half-life of the activated product in seconds.
    sample_volume_cm3 : float
        Sample volume in cm³. Default 1.0.

    Returns
    -------
    float
        Saturation activity in Bq.

    Examples
    --------
    >>> # Activation of Fe-56 → Fe-57* in ITER first wall
    >>> A_sat = activation_saturation(
    ...     reaction_rate_cm2_s=1e14,
    ...     number_density_cm3=8.47e22,  # iron ~7.87 g/cm³
    ...     cross_section_cm2=2.59e-24,  # thermal cross section
    ...     half_life_s=1.0,  # placeholder
    ... )
    """
    if reaction_rate_cm2_s <= 0:
        raise ValueError("reaction_rate_cm2_s must be > 0")
    if number_density_cm3 <= 0:
        raise ValueError("number_density_cm3 must be > 0")
    if cross_section_cm2 <= 0:
        raise ValueError("cross_section_cm2 must be > 0")
    if sample_volume_cm3 <= 0:
        raise ValueError("sample_volume_cm3 must be > 0")

    N_total = number_density_cm3 * sample_volume_cm3
    return float(reaction_rate_cm2_s * cross_section_cm2 * N_total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "plasma_neutron_rate",
    "dt_neutron_spectrum",
    "dd_neutron_spectrum",
    "activation_saturation",
]
