"""
triples_sigfast.nuclear.sources
--------------------------------
Neutron and photon source spectrum generators.

Implements the Watt fission spectrum for spontaneous and induced fission
sources, and the Maxwell-Boltzmann thermal neutron spectrum. All functions
return normalised spectral flux distributions.

References
----------
- IAEA-TECDOC-1234: Neutron Generators for Analytical Purposes
- Knoll, "Radiation Detection and Measurement", 4th ed., Ch. 2
- Mughabghab, "Atlas of Neutron Resonances", 6th ed.
"""

from __future__ import annotations

import numpy as np

# -- Watt spectrum parameters (a, b) for common fission sources ---------------
# N(E) = C × exp(-E/a) × sinh(sqrt(b×E))
# Source: ENDF/B-VIII.0, JEFF-3.3

_WATT_PARAMS: dict[str, tuple[float, float]] = {
    "Cf-252": (1.1800, 1.0340),  # Spontaneous fission
    "U-235": (0.9880, 2.2490),  # Thermal neutron induced
    "U-238": (0.8720, 2.9800),  # Fast neutron induced
    "Pu-239": (0.9660, 2.8420),  # Thermal neutron induced
    "Pu-241": (0.9600, 2.8830),
    "Th-232": (1.0000, 2.6460),
}

# -- Common moderator temperatures as kT [MeV] (for reference / future use) ---
# Room temperature (293 K): kT = 0.02526 eV = 2.526e-5 MeV
# Hot moderator  (600 K):   kT = 0.05170 eV = 5.170e-5 MeV
# Pass these values directly to maxwell_spectrum(temperature_mev=...)


def watt_spectrum(
    energies: np.ndarray,
    source: str = "Cf-252",
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute the Watt fission neutron energy spectrum N(E).

    The Watt spectrum describes the energy distribution of neutrons emitted
    from spontaneous or neutron-induced fission:

        N(E) = C × exp(−E/a) × sinh(√(b×E))

    where a and b are empirically fitted parameters from ENDF/B-VIII.0.

    Parameters
    ----------
    energies : np.ndarray
        Neutron energies in MeV. Typically np.linspace(0.01, 15, 1000).
        Zero and negative values are set to zero flux.
    source : str, optional
        Fission source isotope. One of: 'Cf-252', 'U-235', 'U-238',
        'Pu-239', 'Pu-241', 'Th-232'. Default 'Cf-252'.
    normalise : bool, optional
        If True (default), normalise so that the integral over all energies
        equals 1.0 (probability density). If False, returns raw N(E).

    Returns
    -------
    np.ndarray
        Spectral flux N(E), same shape as energies.

    Raises
    ------
    ValueError
        If source is not in the database.

    Examples
    --------
    >>> import numpy as np
    >>> energies = np.linspace(0.01, 15, 1000)
    >>> flux = watt_spectrum(energies, source="Cf-252")
    >>> peak_energy = energies[flux.argmax()]
    >>> print(f"Peak neutron energy: {peak_energy:.2f} MeV")
    """
    source = source.strip()
    if source not in _WATT_PARAMS:
        raise ValueError(
            f"Unknown source '{source}'. Available: {list(_WATT_PARAMS.keys())}"
        )

    a, b = _WATT_PARAMS[source]
    E = np.asarray(energies, dtype=np.float64)
    # 1. Sanitize inputs to prevent negative values from hitting the sqrt function
    safe_E = np.maximum(E, 0.0)

    # 2. Calculate using the safe array
    calculation = np.exp(-safe_E / a) * np.sinh(np.sqrt(b * safe_E))

    # 3. Apply the mask for exactly zero or negative inputs
    flux = np.where(E > 0, calculation, 0.0)

    if normalise and flux.sum() > 0:
        # np.trapezoid is the correct trapezoidal rule; np.gradient had boundary errors
        total = np.trapezoid(flux, E)
        if total > 0:
            flux = flux / total

    return flux


def maxwell_spectrum(
    energies: np.ndarray,
    temperature_mev: float = 0.0253e-3,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute the Maxwell-Boltzmann thermal neutron energy spectrum.

    Describes the energy distribution of neutrons in thermal equilibrium
    with a moderator at temperature T:

        N(E) = 2π × (1/(πkT))^(3/2) × sqrt(E) × exp(−E/kT)

    Parameters
    ----------
    energies : np.ndarray
        Neutron energies in MeV.
    temperature_mev : float, optional
        Moderator temperature as kT in MeV.
        Room temperature (293 K): kT = 0.0253 eV = 2.53e-5 MeV (default).
        Hot moderator (600 K):    kT = 5.17e-5 MeV.
    normalise : bool, optional
        If True (default), normalise to unit integral.

    Returns
    -------
    np.ndarray
        Maxwell-Boltzmann spectral flux, same shape as energies.

    Examples
    --------
    >>> energies = np.linspace(1e-6, 0.01, 1000)  # eV to 10 keV
    >>> flux = maxwell_spectrum(energies, temperature_mev=0.0000253)
    """
    kT = temperature_mev
    E = np.asarray(energies, dtype=np.float64)
    flux = np.where(E > 0, np.sqrt(E) * np.exp(-E / kT), 0.0)

    if normalise and flux.sum() > 0:
        # np.trapezoid is the correct trapezoidal rule; np.gradient had boundary errors
        total = np.trapezoid(flux, E)
        if total > 0:
            flux = flux / total

    return flux


def available_sources() -> list[str]:
    """Return list of fission sources with tabulated Watt parameters."""
    return list(_WATT_PARAMS.keys())


def watt_mean_energy(source: str = "Cf-252") -> float:
    """
    Return the mean neutron energy for a Watt spectrum source in MeV.

    Computed analytically as <E> = (3a/2) + (ab/4).

    Parameters
    ----------
    source : str
        Fission source isotope.

    Returns
    -------
    float
        Mean neutron energy in MeV.
    """
    if source not in _WATT_PARAMS:
        raise ValueError(
            f"Unknown source '{source}'. Available: {list(_WATT_PARAMS.keys())}"
        )
    a, b = _WATT_PARAMS[source]
    return 1.5 * a + 0.25 * a * b
