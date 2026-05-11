"""
triples_sigfast.nuclear.isotope
--------------------------------
Isotope database with decay, activity, and neutron cross-section data.

Provides a unified interface to nuclear data for isotopes commonly used
in radiation shielding and detector physics research.

Data sources
------------
- NUBASE2020: Nuclear structure and decay data
- ENDF/B-VIII.0: Evaluated nuclear data file
- IAEA Nuclear Data Services: https://www-nds.iaea.org
"""

from __future__ import annotations

import numpy as np

# -- Nuclear constants ---------------------------------------------------------
_AVOGADRO = 6.02214076e23  # mol⁻¹
_LN2 = np.log(2)
_SEC_PER_YEAR = 3.15576e7  # seconds per year

# -- Isotope database ----------------------------------------------------------
# Each entry: {
#   "Z": atomic number,
#   "A": mass number,
#   "mass_amu": atomic mass in u,
#   "half_life_s": half-life in seconds (np.inf for stable),
#   "decay_mode": primary decay mode string,
#   "thermal_xs_b": thermal neutron cross section in barns (2200 m/s),
#   "resonance_integral_b": resonance integral in barns,
#   "neutron_yield": spontaneous fission neutron yield n/s/g (0 if not SF),
#   "gamma_energies_mev": list of primary gamma energies in MeV,
# }

_ISOTOPE_DB: dict[str, dict] = {
    "Cf-252": {
        "Z": 98,
        "A": 252,
        "mass_amu": 252.0816,
        "half_life_s": 2.645 * _SEC_PER_YEAR,
        "decay_mode": "SF+alpha",
        "thermal_xs_b": 20.4,
        "resonance_integral_b": 43.0,
        "neutron_yield": 2.314e12,
        "gamma_energies_mev": [0.100],
    },
    "Co-60": {
        "Z": 27,
        "A": 60,
        "mass_amu": 59.9338,
        "half_life_s": 5.2713 * _SEC_PER_YEAR,
        "decay_mode": "beta-",
        "thermal_xs_b": 2.0,
        "resonance_integral_b": 2.0,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [1.173, 1.332],
    },
    "Cs-137": {
        "Z": 55,
        "A": 137,
        "mass_amu": 136.9071,
        "half_life_s": 30.17 * _SEC_PER_YEAR,
        "decay_mode": "beta-",
        "thermal_xs_b": 0.11,
        "resonance_integral_b": 0.36,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [0.662],
    },
    "B-10": {
        "Z": 5,
        "A": 10,
        "mass_amu": 10.0129,
        "half_life_s": np.inf,
        "decay_mode": "stable",
        "thermal_xs_b": 3840.0,
        "resonance_integral_b": 1722.0,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [0.478],
    },
    "Gd-157": {
        "Z": 64,
        "A": 157,
        "mass_amu": 156.9240,
        "half_life_s": np.inf,
        "decay_mode": "stable",
        "thermal_xs_b": 259000.0,
        "resonance_integral_b": 800.0,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [0.182],
    },
    "Am-241": {
        "Z": 95,
        "A": 241,
        "mass_amu": 241.0568,
        "half_life_s": 432.2 * _SEC_PER_YEAR,
        "decay_mode": "alpha",
        "thermal_xs_b": 3.2,
        "resonance_integral_b": 14.0,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [0.0595],
    },
    "Na-22": {
        "Z": 11,
        "A": 22,
        "mass_amu": 21.9944,
        "half_life_s": 2.6019 * _SEC_PER_YEAR,
        "decay_mode": "beta+",
        "thermal_xs_b": 0.34,
        "resonance_integral_b": 0.23,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [0.511, 1.275],
    },
    "U-238": {
        "Z": 92,
        "A": 238,
        "mass_amu": 238.0508,
        "half_life_s": 4.468e9 * _SEC_PER_YEAR,
        "decay_mode": "alpha+SF",
        "thermal_xs_b": 2.68,
        "resonance_integral_b": 277.0,
        "neutron_yield": 13.6,
        "gamma_energies_mev": [0.050],
    },
    "Pu-239": {
        "Z": 94,
        "A": 239,
        "mass_amu": 239.0522,
        "half_life_s": 24110.0 * _SEC_PER_YEAR,
        "decay_mode": "alpha",
        "thermal_xs_b": 1011.0,
        "resonance_integral_b": 300.0,
        "neutron_yield": 21.8,
        "gamma_energies_mev": [0.129, 0.375, 0.414],
    },
    "Fe-56": {
        "Z": 26,
        "A": 56,
        "mass_amu": 55.9349,
        "half_life_s": np.inf,
        "decay_mode": "stable",
        "thermal_xs_b": 2.59,
        "resonance_integral_b": 1.40,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [],
    },
    "H-1": {
        "Z": 1,
        "A": 1,
        "mass_amu": 1.00794,
        "half_life_s": np.inf,
        "decay_mode": "stable",
        "thermal_xs_b": 0.3326,
        "resonance_integral_b": 0.149,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [],
    },
    "Pb-208": {
        "Z": 82,
        "A": 208,
        "mass_amu": 207.9767,
        "half_life_s": np.inf,
        "decay_mode": "stable",
        "thermal_xs_b": 0.00048,
        "resonance_integral_b": 0.00030,
        "neutron_yield": 0.0,
        "gamma_energies_mev": [],
    },
}


class Isotope:
    """
    Nuclear isotope with decay, activity, and cross-section data.

    Parameters
    ----------
    name : str
        Isotope name in format 'Symbol-A', e.g. 'Cf-252', 'Co-60', 'B-10'.
        Case-insensitive for the symbol, e.g. 'cf-252' also works.

    Examples
    --------
    >>> cf = Isotope("Cf-252")
    >>> print(f"Half-life: {cf.half_life:.3f} years")
    >>> print(f"Activity of 1 mg: {cf.activity(mass_g=0.001):.3e} Bq")
    >>> print(f"Neutron yield: {cf.neutron_yield:.3e} n/s/g")

    >>> b10 = Isotope("B-10")
    >>> print(f"Thermal cross section: {b10.thermal_cross_section:.0f} barns")
    """

    def __init__(self, name: str) -> None:
        self.name = self._resolve_name(name)
        self._data = _ISOTOPE_DB[self.name]

    @staticmethod
    def _resolve_name(name: str) -> str:
        """Normalise name: 'cf-252' -> 'Cf-252'."""
        name = name.strip()
        if name in _ISOTOPE_DB:
            return name
        # Try case-insensitive match
        for key in _ISOTOPE_DB:
            if key.lower() == name.lower():
                return key
        raise ValueError(
            f"Isotope '{name}' not found in database. "
            f"Available: {list(_ISOTOPE_DB.keys())}"
        )

    # -- Properties --------------------------------------------------------

    @property
    def Z(self) -> int:
        """Atomic number."""
        return self._data["Z"]

    @property
    def A(self) -> int:
        """Mass number."""
        return self._data["A"]

    @property
    def mass_amu(self) -> float:
        """Atomic mass in unified atomic mass units (u)."""
        return self._data["mass_amu"]

    @property
    def half_life(self) -> float:
        """Half-life in years. Returns np.inf for stable isotopes."""
        t_s = self._data["half_life_s"]
        if np.isinf(t_s):
            return np.inf
        return t_s / _SEC_PER_YEAR

    @property
    def half_life_seconds(self) -> float:
        """Half-life in seconds."""
        return self._data["half_life_s"]

    @property
    def decay_constant(self) -> float:
        """Decay constant λ in s⁻¹. Returns 0 for stable isotopes."""
        t = self._data["half_life_s"]
        if np.isinf(t):
            return 0.0
        return _LN2 / t

    @property
    def decay_mode(self) -> str:
        """Primary decay mode (e.g. 'alpha', 'beta-', 'SF+alpha')."""
        return self._data["decay_mode"]

    @property
    def thermal_cross_section(self) -> float:
        """Thermal neutron absorption cross section in barns (at 2200 m/s)."""
        return self._data["thermal_xs_b"]

    @property
    def resonance_integral(self) -> float:
        """Resonance integral in barns."""
        return self._data["resonance_integral_b"]

    @property
    def neutron_yield(self) -> float:
        """Spontaneous fission neutron yield in n/s/g. 0 for non-SF isotopes."""
        return self._data["neutron_yield"]

    @property
    def gamma_energies(self) -> list[float]:
        """Primary gamma emission energies in MeV."""
        return self._data["gamma_energies_mev"]

    # -- Methods -----------------------------------------------------------

    def activity(self, mass_g: float) -> float:
        """
        Compute activity in Becquerels for a given mass.

        A = λ × N = (ln2 / t½) × (m × Nₐ / M)

        Parameters
        ----------
        mass_g : float
            Mass of the isotope in grams.

        Returns
        -------
        float
            Activity in Bq. Returns 0.0 for stable isotopes.

        Examples
        --------
        >>> cf = Isotope("Cf-252")
        >>> print(f"1 µg Cf-252: {cf.activity(1e-6):.3e} Bq")
        """
        if self.decay_constant == 0.0:
            return 0.0
        N = (mass_g / self.mass_amu) * _AVOGADRO
        return self.decay_constant * N

    def atoms_per_gram(self) -> float:
        """Number of atoms per gram of this isotope."""
        return _AVOGADRO / self.mass_amu

    def neutron_source_rate(self, mass_g: float) -> float:
        """
        Compute spontaneous fission neutron emission rate in n/s.

        Parameters
        ----------
        mass_g : float
            Mass in grams.

        Returns
        -------
        float
            Neutron emission rate in n/s. Returns 0.0 for non-SF isotopes.
        """
        return self.neutron_yield * mass_g

    def is_stable(self) -> bool:
        """Return True if the isotope is stable."""
        return np.isinf(self._data["half_life_s"])

    def __repr__(self) -> str:
        hl = f"{self.half_life:.4g} yr" if not self.is_stable() else "stable"
        return f"Isotope('{self.name}', Z={self.Z}, A={self.A}, t½={hl})"


def available_isotopes() -> list[str]:
    """Return list of isotopes in the database."""
    return list(_ISOTOPE_DB.keys())
