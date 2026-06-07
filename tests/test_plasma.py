"""
tests/test_plasma.py
─────────────────────
Test suite for triples_sigfast.plasma.

Validates D-T/D-D neutron spectra, thermonuclear reaction rates,
and activation saturation against known physical benchmarks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from triples_sigfast.plasma import (
    activation_saturation,
    dd_neutron_spectrum,
    dt_neutron_spectrum,
    plasma_neutron_rate,
)

# ─────────────────────────────────────────────────────────────────────────────
# D-T neutron spectrum
# ─────────────────────────────────────────────────────────────────────────────


class TestDTNeutronSpectrum:
    def test_peak_at_14_1_mev(self):
        """D-T spectrum peaks near 14.1 MeV."""
        E = np.linspace(10.0, 18.0, 1000)
        spectrum = dt_neutron_spectrum(E, temperature_kev=10.0)
        peak_E = E[np.argmax(spectrum)]
        assert abs(peak_E - 14.07) < 0.2

    def test_spectrum_positive(self):
        """All spectral values are non-negative."""
        E = np.linspace(10.0, 18.0, 500)
        spectrum = dt_neutron_spectrum(E, temperature_kev=10.0)
        assert np.all(spectrum >= 0.0)

    def test_normalised_integrates_to_unity(self):
        """Normalised spectrum integrates to approximately 1."""
        E = np.linspace(8.0, 20.0, 5000)
        spectrum = dt_neutron_spectrum(E, temperature_kev=10.0, normalise=True)
        integral = np.trapezoid(spectrum, E)
        assert abs(integral - 1.0) < 0.01

    def test_wider_at_higher_temperature(self):
        """Higher ion temperature → broader peak."""
        E = np.linspace(10.0, 18.0, 1000)
        s_cold = dt_neutron_spectrum(E, temperature_kev=1.0)
        s_hot = dt_neutron_spectrum(E, temperature_kev=100.0)
        # Hot plasma has lower peak (wider distribution)
        assert s_cold.max() > s_hot.max()

    def test_output_shape_matches_input(self):
        E = np.linspace(12.0, 16.0, 200)
        spectrum = dt_neutron_spectrum(E, temperature_kev=10.0)
        assert spectrum.shape == E.shape


# ─────────────────────────────────────────────────────────────────────────────
# D-D neutron spectrum
# ─────────────────────────────────────────────────────────────────────────────


class TestDDNeutronSpectrum:
    def test_peak_at_2_45_mev(self):
        """D-D spectrum peaks near 2.45 MeV."""
        E = np.linspace(1.0, 4.0, 1000)
        spectrum = dd_neutron_spectrum(E, temperature_kev=5.0)
        peak_E = E[np.argmax(spectrum)]
        assert abs(peak_E - 2.45) < 0.1

    def test_spectrum_positive(self):
        E = np.linspace(1.0, 4.0, 500)
        spectrum = dd_neutron_spectrum(E, temperature_kev=5.0)
        assert np.all(spectrum >= 0.0)

    def test_normalised_integrates_to_unity(self):
        E = np.linspace(0.5, 5.0, 5000)
        spectrum = dd_neutron_spectrum(E, temperature_kev=5.0, normalise=True)
        integral = np.trapezoid(spectrum, E)
        assert abs(integral - 1.0) < 0.01

    def test_output_shape(self):
        E = np.linspace(1.0, 4.0, 300)
        assert dd_neutron_spectrum(E).shape == E.shape


# ─────────────────────────────────────────────────────────────────────────────
# plasma_neutron_rate
# ─────────────────────────────────────────────────────────────────────────────


class TestPlasmaNeutronRate:
    def test_dt_rate_positive(self):
        rate = plasma_neutron_rate("DT", 1e20, 10.0, 100.0)
        assert rate > 0.0

    def test_dd_rate_positive(self):
        rate = plasma_neutron_rate("DD", 1e20, 10.0, 100.0)
        assert rate > 0.0

    def test_dt_much_greater_than_dd(self):
        """At 10 keV, D-T reactivity >> D-D reactivity."""
        dt = plasma_neutron_rate("DT", 1e20, 10.0, 100.0)
        dd = plasma_neutron_rate("DD", 1e20, 10.0, 100.0)
        assert dt > dd

    def test_scales_quadratically_with_density(self):
        """Doubling density → 4x rate (R ∝ n²)."""
        r1 = plasma_neutron_rate("DT", 1e20, 10.0, 100.0)
        r2 = plasma_neutron_rate("DT", 2e20, 10.0, 100.0)
        ratio = r2 / r1
        assert abs(ratio - 4.0) < 0.5

    def test_scales_linearly_with_volume(self):
        """Doubling volume → 2x rate."""
        r1 = plasma_neutron_rate("DT", 1e20, 10.0, 100.0)
        r2 = plasma_neutron_rate("DT", 1e20, 10.0, 200.0)
        assert abs(r2 / r1 - 2.0) < 0.01

    def test_invalid_reaction_raises(self):
        with pytest.raises(ValueError, match="DT.*DD"):
            plasma_neutron_rate("HE", 1e20, 10.0, 100.0)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError, match="ion_density"):
            plasma_neutron_rate("DT", 0.0, 10.0, 100.0)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            plasma_neutron_rate("DT", 1e20, 0.0, 100.0)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError, match="plasma_volume"):
            plasma_neutron_rate("DT", 1e20, 10.0, 0.0)

    def test_case_insensitive_reaction(self):
        """'dt' should work the same as 'DT'."""
        r_upper = plasma_neutron_rate("DT", 1e20, 10.0, 100.0)
        r_lower = plasma_neutron_rate("dt", 1e20, 10.0, 100.0)
        assert abs(r_upper - r_lower) < 1.0  # Same result


# ─────────────────────────────────────────────────────────────────────────────
# activation_saturation
# ─────────────────────────────────────────────────────────────────────────────


class TestActivationSaturation:
    def test_returns_positive_activity(self):
        A = activation_saturation(
            reaction_rate_cm2_s=1e14,
            number_density_cm3=8.47e22,
            cross_section_cm2=2.59e-24,
            half_life_s=1.0,
        )
        assert A > 0.0

    def test_scales_linearly_with_flux(self):
        A1 = activation_saturation(1e13, 1e22, 1e-24, 1.0)
        A2 = activation_saturation(2e13, 1e22, 1e-24, 1.0)
        assert abs(A2 / A1 - 2.0) < 1e-10

    def test_scales_linearly_with_number_density(self):
        A1 = activation_saturation(1e13, 1e22, 1e-24, 1.0)
        A2 = activation_saturation(1e13, 2e22, 1e-24, 1.0)
        assert abs(A2 / A1 - 2.0) < 1e-10

    def test_scales_linearly_with_cross_section(self):
        A1 = activation_saturation(1e13, 1e22, 1e-24, 1.0)
        A2 = activation_saturation(1e13, 1e22, 2e-24, 1.0)
        assert abs(A2 / A1 - 2.0) < 1e-10

    def test_scales_linearly_with_volume(self):
        A1 = activation_saturation(1e13, 1e22, 1e-24, 1.0, sample_volume_cm3=1.0)
        A2 = activation_saturation(1e13, 1e22, 1e-24, 1.0, sample_volume_cm3=5.0)
        assert abs(A2 / A1 - 5.0) < 1e-10

    def test_zero_flux_raises(self):
        with pytest.raises(ValueError, match="reaction_rate"):
            activation_saturation(0.0, 1e22, 1e-24, 1.0)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError, match="number_density"):
            activation_saturation(1e13, 0.0, 1e-24, 1.0)

    def test_zero_cross_section_raises(self):
        with pytest.raises(ValueError, match="cross_section"):
            activation_saturation(1e13, 1e22, 0.0, 1.0)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError, match="sample_volume"):
            activation_saturation(1e13, 1e22, 1e-24, 1.0, sample_volume_cm3=0.0)
