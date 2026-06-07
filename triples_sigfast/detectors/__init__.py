"""
triples_sigfast.detectors
--------------------------
Detector physics models for radiation detection and response simulation.

This sub-package provides idealised models for common nuclear detector
types used in shielding measurements, spectroscopy, and particle physics
experiments.  All models are physics-validated against NIST and ICRP
reference data.

.. note::
    This sub-package is under active development.  The core API shown below
    is stable and guaranteed not to change in the v1.x series.  Additional
    detector types will be added in v2.0.

Available detectors
-------------------
NaIDetector     -- Thallium-doped NaI scintillator (spectroscopy)
HPGeDetector    -- High-Purity Germanium (high-resolution spectroscopy)
BF3Detector     -- BF3 proportional counter (thermal neutron detection)
He3Detector     -- He-3 proportional counter (neutron spectroscopy)

Usage
-----
>>> from triples_sigfast.detectors import NaIDetector, HPGeDetector
>>> det = NaIDetector(thickness_cm=7.62)  # standard 3-inch crystal
>>> efficiency = det.intrinsic_efficiency(energy_mev=0.662)
>>> print(f"Cs-137 peak efficiency: {efficiency:.3f}")

>>> ge = HPGeDetector(relative_efficiency=0.30)
>>> fwhm = ge.energy_resolution(energy_mev=1.332)
>>> print(f"Co-60 FWHM: {fwhm:.4f} MeV")

References
----------
- NIST detector efficiency data: https://www.nist.gov/
- Knoll, G.F., "Radiation Detection and Measurement", 4th ed., Wiley (2010)
- ICRP Publication 74 — detector response functions
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Material constants used by detector models
# ---------------------------------------------------------------------------

# NaI density and attenuation coefficient at ~660 keV (Cs-137 peak)
_NAI_DENSITY = 3.67  # g/cm³
_NAI_MU_RHO_662 = 0.09921  # cm²/g at 662 keV (NIST XCOM)

# HPGe effective atomic number and Fano factor
_HPGE_FANO = 0.129  # Fano factor for germanium
_HPGE_W = 2.96e-3  # Average energy to create electron-hole pair (MeV)

# He-3 thermal neutron cross section
_HE3_THERMAL_XS_CM2 = 5333e-24  # cm² (5333 barns at 25.3 meV)

# BF3 B-10 thermal neutron cross section
_BF3_B10_XS_CM2 = 3840e-24  # cm² (3840 barns at 25.3 meV)


# ---------------------------------------------------------------------------
# NaI(Tl) scintillator detector
# ---------------------------------------------------------------------------


class NaIDetector:
    """Thallium-activated NaI scintillator detector model.

    Models the intrinsic efficiency and energy resolution of cylindrical
    NaI(Tl) crystals, the standard workhorse detector for gamma-ray
    spectroscopy in nuclear physics and radiation protection.

    Parameters
    ----------
    thickness_cm : float
        Crystal thickness along the beam axis in centimetres.
        Standard sizes: 2.54 cm (1-inch), 7.62 cm (3-inch), 15.24 cm (6-inch).
    diameter_cm : float, optional
        Crystal diameter in centimetres.  Default 7.62 cm (3-inch standard).

    Examples
    --------
    >>> det = NaIDetector(thickness_cm=7.62)
    >>> eff = det.intrinsic_efficiency(energy_mev=0.662)  # Cs-137
    >>> print(f"Cs-137 peak efficiency: {eff:.3f}")
    """

    def __init__(self, thickness_cm: float, diameter_cm: float = 7.62) -> None:
        if thickness_cm <= 0:
            raise ValueError(f"thickness_cm must be > 0, got {thickness_cm}")
        if diameter_cm <= 0:
            raise ValueError(f"diameter_cm must be > 0, got {diameter_cm}")
        self.thickness_cm = thickness_cm
        self.diameter_cm = diameter_cm

    def intrinsic_efficiency(self, energy_mev: float) -> float:
        """Compute intrinsic peak efficiency at a given photon energy.

        Uses the Beer-Lambert absorption model:

            ε = 1 - exp(-μ(E) × ρ × x)

        where μ(E)/ρ is the mass attenuation coefficient of NaI,
        ρ is the crystal density, and x is the crystal thickness.

        Parameters
        ----------
        energy_mev : float
            Photon energy in MeV. Valid range: 0.05 – 10.0 MeV.

        Returns
        -------
        float
            Intrinsic efficiency in [0, 1].

        References
        ----------
        Knoll, G.F., "Radiation Detection and Measurement", 4th ed., Table 10.1.
        """
        if energy_mev <= 0:
            raise ValueError(f"energy_mev must be > 0, got {energy_mev}")
        # Approximate NaI mu/rho as a power-law fit to NIST data
        # mu/rho ≈ 0.0993 * (E/0.662)^{-0.28}  (valid 0.1 – 3 MeV)
        mu_rho = _NAI_MU_RHO_662 * (energy_mev / 0.662) ** (-0.28)
        mu_linear = mu_rho * _NAI_DENSITY  # cm⁻¹
        return float(1.0 - math.exp(-mu_linear * self.thickness_cm))

    def energy_resolution(self, energy_mev: float) -> float:
        """Estimate FWHM energy resolution at a given photon energy.

        Empirical relation for NaI(Tl):

            FWHM% ≈ 7.5% × (E / 0.662 MeV)^{-0.5}

        Parameters
        ----------
        energy_mev : float
            Photon energy in MeV.

        Returns
        -------
        float
            FWHM in MeV (absolute, not percent).
        """
        if energy_mev <= 0:
            raise ValueError(f"energy_mev must be > 0, got {energy_mev}")
        fwhm_percent = 0.075 * (energy_mev / 0.662) ** (-0.5)
        return float(fwhm_percent * energy_mev)

    def __repr__(self) -> str:
        return (
            f"NaIDetector(thickness_cm={self.thickness_cm}, "
            f"diameter_cm={self.diameter_cm})"
        )


# ---------------------------------------------------------------------------
# High-Purity Germanium (HPGe) detector
# ---------------------------------------------------------------------------


class HPGeDetector:
    """High-Purity Germanium (HPGe) detector model.

    Provides energy resolution estimates based on the Fano-factor model
    and typical HPGe detector performance.  HPGe delivers 20–50× better
    energy resolution than NaI(Tl) at the cost of requiring liquid nitrogen
    cooling (77 K) or Peltier cooling for modern cryocooled detectors.

    Parameters
    ----------
    relative_efficiency : float
        Detector relative efficiency as a fraction of a 3-inch × 3-inch
        NaI(Tl) crystal at 1.332 MeV (Co-60).
        Typical range: 0.10 (small) to 1.00 (large research grade).

    Examples
    --------
    >>> ge = HPGeDetector(relative_efficiency=0.30)
    >>> fwhm = ge.energy_resolution(energy_mev=1.332)  # Co-60
    >>> print(f"Co-60 FWHM: {fwhm*1000:.2f} keV")
    """

    def __init__(self, relative_efficiency: float = 0.30) -> None:
        if not (0 < relative_efficiency <= 1.0):
            raise ValueError(
                f"relative_efficiency must be in (0, 1], got {relative_efficiency}"
            )
        self.relative_efficiency = relative_efficiency

    def energy_resolution(self, energy_mev: float) -> float:
        """Estimate FWHM energy resolution using the Fano-factor model.

        FWHM = 2.355 × sqrt(F × W × E)

        where F is the Fano factor, W is the average ionisation energy per
        electron-hole pair, and E is the photon energy.

        Parameters
        ----------
        energy_mev : float
            Photon energy in MeV.

        Returns
        -------
        float
            FWHM in MeV.

        References
        ----------
        Knoll, G.F., "Radiation Detection and Measurement", 4th ed., Ch. 11.
        """
        if energy_mev <= 0:
            raise ValueError(f"energy_mev must be > 0, got {energy_mev}")
        fwhm = 2.355 * math.sqrt(_HPGE_FANO * _HPGE_W * energy_mev)
        return float(fwhm)

    def __repr__(self) -> str:
        return f"HPGeDetector(relative_efficiency={self.relative_efficiency:.2f})"


# ---------------------------------------------------------------------------
# He-3 neutron detector
# ---------------------------------------------------------------------------


class He3Detector:
    """He-3 proportional counter for thermal and epithermal neutron detection.

    He-3 detectors are the gold standard for thermal neutron counting.
    The detection reaction is:

        n + He-3 → p + H-3 + 764 keV (Q-value)

    Parameters
    ----------
    pressure_atm : float
        He-3 gas fill pressure in atmospheres.
        Typical: 2–10 atm for neutron counting, 4 atm standard.
    active_length_cm : float
        Active detector tube length in centimetres.

    Examples
    --------
    >>> det = He3Detector(pressure_atm=4.0, active_length_cm=30.0)
    >>> eff = det.thermal_efficiency()
    >>> print(f"Thermal neutron efficiency: {eff:.3f}")
    """

    def __init__(
        self, pressure_atm: float = 4.0, active_length_cm: float = 30.0
    ) -> None:
        if pressure_atm <= 0:
            raise ValueError(f"pressure_atm must be > 0, got {pressure_atm}")
        if active_length_cm <= 0:
            raise ValueError(f"active_length_cm must be > 0, got {active_length_cm}")
        self.pressure_atm = pressure_atm
        self.active_length_cm = active_length_cm

    def thermal_efficiency(self) -> float:
        """Estimate intrinsic thermal neutron detection efficiency.

        Uses Beer-Lambert absorption:

            ε = 1 - exp(-n × σ × L)

        where n is the He-3 number density at the fill pressure,
        σ is the thermal neutron cross section, and L is the tube length.

        Returns
        -------
        float
            Intrinsic efficiency in [0, 1] for 25.3 meV neutrons.
        """
        # Number density: n = P × N_A / (V_M) using ideal gas at 293 K
        _AVOGADRO = 6.02214076e23
        _R = 82.06  # cm³·atm/(mol·K)
        _T = 293.0  # K
        n_atoms = (self.pressure_atm * _AVOGADRO) / (_R * _T)  # cm⁻³
        exponent = n_atoms * _HE3_THERMAL_XS_CM2 * self.active_length_cm
        return float(1.0 - math.exp(-exponent))

    def __repr__(self) -> str:
        return (
            f"He3Detector(pressure_atm={self.pressure_atm}, "
            f"active_length_cm={self.active_length_cm})"
        )


# ---------------------------------------------------------------------------
# BF3 neutron detector
# ---------------------------------------------------------------------------


class BF3Detector:
    """BF3 proportional counter for thermal neutron detection.

    BF3 detectors use the high thermal neutron cross section of B-10:

        n + B-10 → Li-7 + α + 2.79 MeV (94%) or 2.31 MeV (6%)

    Parameters
    ----------
    pressure_atm : float
        BF3 gas fill pressure in atmospheres. Typical: 0.5 – 1.0 atm.
    active_length_cm : float
        Active tube length in centimetres.
    b10_enrichment : float
        B-10 isotopic enrichment fraction (0–1). Natural boron ≈ 0.199.
        Enriched BF3 detectors use ≥ 0.96.

    Examples
    --------
    >>> det = BF3Detector(pressure_atm=0.9, active_length_cm=50.0, b10_enrichment=0.96)
    >>> eff = det.thermal_efficiency()
    """

    def __init__(
        self,
        pressure_atm: float = 0.9,
        active_length_cm: float = 50.0,
        b10_enrichment: float = 0.96,
    ) -> None:
        if pressure_atm <= 0:
            raise ValueError(f"pressure_atm must be > 0, got {pressure_atm}")
        if active_length_cm <= 0:
            raise ValueError(f"active_length_cm must be > 0, got {active_length_cm}")
        if not (0 < b10_enrichment <= 1.0):
            raise ValueError(f"b10_enrichment must be in (0, 1], got {b10_enrichment}")
        self.pressure_atm = pressure_atm
        self.active_length_cm = active_length_cm
        self.b10_enrichment = b10_enrichment

    def thermal_efficiency(self) -> float:
        """Estimate intrinsic thermal neutron detection efficiency.

        Returns
        -------
        float
            Intrinsic efficiency in [0, 1] for 25.3 meV neutrons.
        """
        _AVOGADRO = 6.02214076e23
        _R = 82.06
        _T = 293.0
        # BF3 molecular weight ≈ 67.82 g/mol; B-10 enriched: 10 + 3×19 = 67
        n_molecules = (self.pressure_atm * _AVOGADRO) / (_R * _T)
        n_b10 = n_molecules * self.b10_enrichment  # effective B-10 number density
        exponent = n_b10 * _BF3_B10_XS_CM2 * self.active_length_cm
        return float(1.0 - math.exp(-exponent))

    def __repr__(self) -> str:
        return (
            f"BF3Detector(pressure_atm={self.pressure_atm}, "
            f"active_length_cm={self.active_length_cm}, "
            f"b10_enrichment={self.b10_enrichment:.2f})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["NaIDetector", "HPGeDetector", "He3Detector", "BF3Detector"]


def available_detectors() -> list[str]:
    """Return list of available detector model names."""
    return [
        cls.__name__ for cls in (NaIDetector, HPGeDetector, He3Detector, BF3Detector)
    ]
