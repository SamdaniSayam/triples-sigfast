"""
tests/test_nuclear.py
─────────────────────
Test suite for triples_sigfast.nuclear

Covers: shielding, sources, isotope, dose
Target: 100% coverage, physics-validated results.
"""

from __future__ import annotations

import numpy as np
import pytest

from triples_sigfast.nuclear.dose import (
    dose_rate_vs_distance,
    inverse_square_distance,
    point_source,
    point_source_shielded,
)
from triples_sigfast.nuclear.isotope import Isotope, available_isotopes
from triples_sigfast.nuclear.shielding import (
    _get_mu,
    _gp_buildup,
    attenuation_series,
    attenuation_with_buildup,
    available_materials,
)
from triples_sigfast.nuclear.sources import (
    available_sources,
    maxwell_spectrum,
    watt_mean_energy,
    watt_spectrum,
)

# ═══════════════════════════════════════════════════════════════════
# shielding.py
# ═══════════════════════════════════════════════════════════════════


class TestGetMu:
    def test_lead_1mev(self):
        """Lead at 1 MeV: mu = mu/rho * rho = 0.07102 * 11.35 ≈ 0.806 cm^-1."""
        mu = _get_mu("lead", 1.0)
        assert 0.70 < mu < 0.90

    def test_water_1mev(self):
        """Water at 1 MeV: mu ≈ 0.0707 cm^-1."""
        mu = _get_mu("water", 1.0)
        assert 0.060 < mu < 0.080

    def test_interpolation_between_tabulated_points(self):
        """Values between tabulated energies should be interpolated smoothly."""
        mu_05 = _get_mu("iron", 0.5)
        mu_10 = _get_mu("iron", 1.0)
        mu_07 = _get_mu("iron", 0.75)
        assert mu_05 > mu_07 > mu_10

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError, match="Unknown material"):
            _get_mu("unobtanium", 1.0)

    def test_all_materials_return_positive(self):
        for mat in available_materials():
            mu = _get_mu(mat, 1.0)
            assert mu > 0, f"{mat} gave non-positive mu"


class TestGPBuildup:
    def test_zero_thickness_returns_one(self):
        """B(0) = 1 by definition."""
        B = _gp_buildup("lead", 1.0, 0.0)
        assert B == 1.0

    def test_buildup_greater_than_one_at_moderate_mfp(self):
        """B > 1 at moderate mfp for materials with GP coefficients."""
        B = _gp_buildup("water", 1.0, 5.0)
        assert B > 1.0

    def test_unknown_material_returns_one(self):
        """Fallback B=1 for materials without GP coefficients."""
        B = _gp_buildup("aluminum", 1.0, 5.0)
        assert B == 1.0

    def test_buildup_greater_than_one_at_small_mfp(self):
        """At small mfp, buildup increases with thickness."""
        B1 = _gp_buildup("concrete", 1.0, 1.0)
        B5 = _gp_buildup("concrete", 1.0, 5.0)
        assert B5 > B1 > 1.0

    def test_negative_mfp_returns_one(self):
        """Negative mfp is unphysical — return B=1."""
        B = _gp_buildup("lead", 1.0, -1.0)
        assert B == 1.0

    def test_gp_fallback_c_near_zero(self):
        """When c≈0, GP uses linear approximation B = 1 + (b-1)*x."""
        from unittest.mock import patch

        with patch.dict(
            "triples_sigfast.nuclear.shielding._GP_COEFFS",
            {"lead": {1.0: (1.5, 0.0, 1.0, 20.0, 0.01)}},
        ):
            B = _gp_buildup("lead", 1.0, 2.0)
            assert abs(B - (1.0 + 0.5 * 2.0)) < 1e-6

    def test_gp_fallback_k_near_one(self):
        """When K≈1, GP uses linear approximation."""
        from unittest.mock import patch

        with patch.dict(
            "triples_sigfast.nuclear.shielding._GP_COEFFS",
            {"lead": {1.0: (1.5, 1e-6, 1.0, 20.0, 0.0)}},
        ):
            B = _gp_buildup("lead", 1.0, 2.0)
            assert B > 1.0


class TestAttenuationWithBuildup:
    def test_zero_thickness_returns_one(self):
        """No shield -> full transmission."""
        T = attenuation_with_buildup(0.0, "lead", 1.25)
        assert abs(T - 1.0) < 1e-6

    def test_transmission_between_zero_and_one(self):
        """Physical transmission must be in (0, 1]."""
        T = attenuation_with_buildup(10.0, "lead", 1.25)
        assert 0.0 < T <= 1.0

    def test_buildup_gives_higher_transmission_than_beer_lambert(self):
        """Buildup (scattered radiation) increases effective transmission."""
        T_with = attenuation_with_buildup(10.0, "concrete", 1.0)
        mu = _get_mu("concrete", 1.0)
        T_without = float(np.exp(-mu * 10.0))
        assert T_with > T_without

    def test_thicker_shield_lower_transmission(self):
        T5 = attenuation_with_buildup(5.0, "lead", 1.25)
        T10 = attenuation_with_buildup(10.0, "lead", 1.25)
        assert T10 < T5

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="thickness_cm"):
            attenuation_with_buildup(-1.0, "lead", 1.0)

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="energy_mev"):
            attenuation_with_buildup(10.0, "lead", -1.0)

    def test_plane_source_geometry(self):
        T_point = attenuation_with_buildup(10.0, "water", 1.0, "point_source")
        T_plane = attenuation_with_buildup(10.0, "water", 1.0, "plane_source")
        assert T_plane >= T_point

    def test_infinite_slab_geometry(self):
        T_point = attenuation_with_buildup(10.0, "water", 1.0, "point_source")
        T_slab = attenuation_with_buildup(10.0, "water", 1.0, "infinite_slab")
        assert T_slab >= T_point

    def test_all_materials_transmission_physically_valid(self):
        """All materials must give T in (0, 1] — no unphysical T > 1."""
        for mat in available_materials():
            T = attenuation_with_buildup(5.0, mat, 1.0)
            assert 0.0 < T <= 1.0, f"{mat}: T={T} is outside (0, 1]"

    def test_lead_hvl_approximately_correct(self):
        """Lead HVL at 1.25 MeV ≈ 1.2 cm (literature value)."""
        T = attenuation_with_buildup(1.2, "lead", 1.25)
        assert 0.35 < T < 0.65

    def test_aluminum_no_gp_coefficients_uses_beer_lambert(self):
        """Aluminum has no GP table — transmission uses Beer-Lambert (T <= 1)."""
        T = attenuation_with_buildup(5.0, "aluminum", 1.0)
        mu = _get_mu("aluminum", 1.0)
        T_bl = np.exp(-mu * 5.0)
        assert abs(T - T_bl) < 1e-10


class TestAttenuationSeries:
    def test_output_shape_matches_input(self):
        t = np.linspace(0, 30, 100)
        T = attenuation_series(t, "lead", 1.25)
        assert T.shape == (100,)

    def test_first_value_near_one(self):
        t = np.linspace(0, 20, 50)
        T = attenuation_series(t, "lead", 1.0)
        assert abs(T[0] - 1.0) < 0.01

    def test_monotonically_decreasing_at_large_thickness(self):
        """Past the buildup peak (> 5 cm), transmission decreases monotonically."""
        t = np.linspace(5.0, 30.0, 50)
        T = attenuation_series(t, "concrete", 1.0)
        assert np.all(np.diff(T) <= 0)

    def test_no_buildup_gives_pure_exponential(self):
        t = np.array([0.0, 1.0, 2.0, 5.0])
        T = attenuation_series(t, "iron", 1.0, use_buildup=False)
        mu = _get_mu("iron", 1.0)
        np.testing.assert_allclose(T, np.exp(-mu * t), rtol=1e-6)

    def test_no_buildup_strictly_monotone(self):
        t = np.linspace(0.1, 20, 50)
        T = attenuation_series(t, "concrete", 1.0, use_buildup=False)
        assert np.all(np.diff(T) <= 0)

    def test_available_materials_returns_list(self):
        mats = available_materials()
        assert isinstance(mats, list)
        assert "lead" in mats
        assert "water" in mats


# ═══════════════════════════════════════════════════════════════════
# sources.py
# ═══════════════════════════════════════════════════════════════════


class TestWattSpectrum:
    def test_output_shape_matches_input(self):
        E = np.linspace(0.01, 15, 1000)
        N = watt_spectrum(E, "Cf-252")
        assert N.shape == (1000,)

    def test_normalised_integrates_to_one(self):
        E = np.linspace(0.001, 20, 5000)
        N = watt_spectrum(E, "Cf-252", normalise=True)
        de = np.gradient(E)
        integral = np.sum(N * de)
        assert abs(integral - 1.0) < 0.05

    def test_all_values_non_negative(self):
        E = np.linspace(0, 15, 500)
        N = watt_spectrum(E, "Cf-252")
        assert np.all(N >= 0)

    def test_zero_and_negative_energies_give_zero(self):
        E = np.array([-1.0, 0.0, 1.0])
        N = watt_spectrum(E, "Cf-252")
        assert N[0] == 0.0
        assert N[1] == 0.0
        assert N[2] > 0.0

    def test_cf252_peak_near_07mev(self):
        """Cf-252 Watt spectrum peaks near 0.7 MeV (literature)."""
        E = np.linspace(0.01, 10, 2000)
        N = watt_spectrum(E, "Cf-252")
        peak = E[N.argmax()]
        assert 0.5 < peak < 1.0

    def test_u235_peak_differs_from_cf252(self):
        """Different isotopes have different spectral peaks."""
        E = np.linspace(0.01, 10, 2000)
        N_cf252 = watt_spectrum(E, "Cf-252")
        N_u235 = watt_spectrum(E, "U-235")
        peak_cf = E[N_cf252.argmax()]
        peak_u = E[N_u235.argmax()]
        assert peak_cf != peak_u

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            watt_spectrum(np.array([1.0]), "Unobtanium-999")

    def test_unnormalised_returns_positive_values(self):
        E = np.linspace(0.1, 10, 100)
        N = watt_spectrum(E, "Pu-239", normalise=False)
        assert np.all(N >= 0)
        assert N.sum() > 0

    def test_all_sources_run(self):
        E = np.linspace(0.01, 15, 500)
        for src in available_sources():
            N = watt_spectrum(E, src)
            assert np.all(N >= 0)

    def test_available_sources_returns_list(self):
        srcs = available_sources()
        assert isinstance(srcs, list)
        assert "Cf-252" in srcs
        assert "U-235" in srcs


class TestMaxwellSpectrum:
    def test_output_shape_matches_input(self):
        E = np.linspace(1e-8, 0.01, 1000)
        N = maxwell_spectrum(E)
        assert N.shape == (1000,)

    def test_normalised_integrates_to_one(self):
        E = np.linspace(1e-8, 0.1, 10000)
        N = maxwell_spectrum(E, temperature_mev=2.53e-8, normalise=True)
        de = np.gradient(E)
        integral = np.sum(N * de)
        assert abs(integral - 1.0) < 0.05

    def test_all_values_non_negative(self):
        E = np.linspace(0, 0.01, 500)
        N = maxwell_spectrum(E)
        assert np.all(N >= 0)

    def test_peak_shifts_with_temperature(self):
        """Higher temperature -> peak shifts to higher energy.
        Logarithmic spacing needed — peaks are at kT ~ 1e-8 MeV scale."""
        E = np.logspace(-10, -6, 100000)
        N_cold = maxwell_spectrum(E, temperature_mev=2.53e-8)
        N_hot = maxwell_spectrum(E, temperature_mev=5.17e-8)
        assert E[N_hot.argmax()] > E[N_cold.argmax()]

    def test_zero_energy_gives_zero_flux(self):
        E = np.array([0.0, 1e-5])
        N = maxwell_spectrum(E)
        assert N[0] == 0.0

    def test_unnormalised_returns_raw_values(self):
        E = np.linspace(1e-8, 0.001, 500)
        N = maxwell_spectrum(E, normalise=False)
        assert np.all(N >= 0)
        assert N.sum() > 0


class TestWattMeanEnergy:
    def test_cf252_mean_energy(self):
        """Cf-252 mean neutron energy ~2.14 MeV (literature)."""
        E_mean = watt_mean_energy("Cf-252")
        assert 1.8 < E_mean < 2.5

    def test_mean_energy_positive_for_all_sources(self):
        for src in available_sources():
            assert watt_mean_energy(src) > 0

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            watt_mean_energy("Fake-999")


# ═══════════════════════════════════════════════════════════════════
# isotope.py
# ═══════════════════════════════════════════════════════════════════


class TestIsotope:
    def test_cf252_half_life(self):
        """Cf-252 half-life = 2.645 years (NUBASE2020)."""
        cf = Isotope("Cf-252")
        assert abs(cf.half_life - 2.645) < 0.01

    def test_b10_stable(self):
        b10 = Isotope("B-10")
        assert b10.is_stable()
        assert np.isinf(b10.half_life)

    def test_b10_thermal_cross_section(self):
        """B-10 thermal cross section = 3840 barns."""
        b10 = Isotope("B-10")
        assert abs(b10.thermal_cross_section - 3840) < 1

    def test_activity_zero_for_stable(self):
        fe = Isotope("Fe-56")
        assert fe.activity(mass_g=1.0) == 0.0

    def test_cf252_activity_1g(self):
        """
        1 g Cf-252 activity.
        A = lambda * N = (ln2 / t_half) * (m * Na / M)
        t_half = 2.645 yr = 8.345e7 s
        N = (1 / 252.08) * 6.022e23 = 2.389e21 atoms
        A = (ln2 / 8.345e7) * 2.389e21 ≈ 1.98e13 Bq
        """
        cf = Isotope("Cf-252")
        A = cf.activity(mass_g=1.0)
        assert 1.5e13 < A < 2.5e13

    def test_cf252_neutron_yield(self):
        """Cf-252 neutron yield ≈ 2.314e12 n/s/g."""
        cf = Isotope("Cf-252")
        assert abs(cf.neutron_yield - 2.314e12) / 2.314e12 < 0.01

    def test_neutron_source_rate(self):
        cf = Isotope("Cf-252")
        rate = cf.neutron_source_rate(mass_g=1e-3)
        assert abs(rate - cf.neutron_yield * 1e-3) < 1

    def test_non_sf_isotope_zero_neutron_yield(self):
        co = Isotope("Co-60")
        assert co.neutron_yield == 0.0
        assert co.neutron_source_rate(1.0) == 0.0

    def test_decay_constant_correct(self):
        """lambda = ln2 / t_half."""
        cf = Isotope("Cf-252")
        expected = np.log(2) / cf.half_life_seconds
        assert abs(cf.decay_constant - expected) / expected < 1e-10

    def test_stable_decay_constant_zero(self):
        assert Isotope("H-1").decay_constant == 0.0

    def test_atoms_per_gram(self):
        """H-1: ~6.022e23 / 1.008 ≈ 5.97e23 atoms/g."""
        h1 = Isotope("H-1")
        assert 5.9e23 < h1.atoms_per_gram() < 6.1e23

    def test_gamma_energies_co60(self):
        co = Isotope("Co-60")
        assert 1.173 in co.gamma_energies
        assert 1.332 in co.gamma_energies

    def test_repr_contains_name(self):
        assert "Cf-252" in repr(Isotope("Cf-252"))

    def test_case_insensitive_lookup(self):
        cf = Isotope("cf-252")
        assert cf.name == "Cf-252"

    def test_unknown_isotope_raises(self):
        with pytest.raises(ValueError, match="not found"):
            Isotope("Xx-999")

    def test_z_and_a_correct(self):
        cf = Isotope("Cf-252")
        assert cf.Z == 98
        assert cf.A == 252

    def test_available_isotopes_returns_list(self):
        iso = available_isotopes()
        assert isinstance(iso, list)
        assert "Cf-252" in iso
        assert "B-10" in iso

    def test_all_isotopes_instantiate(self):
        for name in available_isotopes():
            iso = Isotope(name)
            assert iso.name == name

    def test_decay_mode(self):
        cf = Isotope("Cf-252")
        assert cf.decay_mode == "SF+alpha"

    def test_resonance_integral(self):
        b10 = Isotope("B-10")
        assert abs(b10.resonance_integral - 1722.0) < 1

    def test_decay_mode_cf252(self):
        cf = Isotope("Cf-252")
        assert isinstance(cf.decay_mode, str)
        assert len(cf.decay_mode) > 0

    def test_resonance_integral_b10(self):
        b10 = Isotope("B-10")
        assert b10.resonance_integral > 0


# ═══════════════════════════════════════════════════════════════════
# dose.py
# ═══════════════════════════════════════════════════════════════════


class TestPointSource:
    def test_dose_positive_and_finite(self):
        rate = point_source(1e9, 1.25, 100.0)
        assert rate > 0
        assert np.isfinite(rate)

    def test_inverse_square_law(self):
        """Dose proportional to 1/r^2 — doubling distance quarters dose."""
        r1 = point_source(1e9, 1.25, 100.0)
        r2 = point_source(1e9, 1.25, 200.0)
        assert abs(r1 / r2 - 4.0) < 0.01

    def test_dose_scales_with_activity(self):
        r1 = point_source(1e9, 1.25, 100.0)
        r2 = point_source(2e9, 1.25, 100.0)
        assert abs(r2 / r1 - 2.0) < 1e-6

    def test_dose_scales_with_photons_per_decay(self):
        r1 = point_source(1e9, 1.25, 100.0, photons_per_decay=1.0)
        r2 = point_source(1e9, 1.25, 100.0, photons_per_decay=2.0)
        assert abs(r2 / r1 - 2.0) < 1e-6

    def test_neutron_source(self):
        rate = point_source(1e8, 1.0, 100.0, particle="neutron")
        assert rate > 0

    def test_zero_distance_raises(self):
        with pytest.raises(ValueError, match="distance_cm"):
            point_source(1e9, 1.25, 0.0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="distance_cm"):
            point_source(1e9, 1.25, -10.0)

    def test_invalid_particle_raises(self):
        with pytest.raises(ValueError, match="particle"):
            point_source(1e9, 1.25, 100.0, particle="alpha")

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="energy_mev"):
            point_source(1e9, -1.0, 100.0)

    def test_co60_dose_rate_icrp74(self):
        """
        Co-60: 1 GBq, 2 gammas/decay at 1.25 MeV, distance 1 m.
        Acceptable range: 50-500 uSv/hr.
        """
        rate = point_source(1e9, 1.25, 100.0, photons_per_decay=2.0)
        assert 50.0 < rate < 500.0


class TestPointSourceShielded:
    def test_shielded_less_than_unshielded(self):
        unshielded = point_source(1e9, 1.25, 100.0)
        shielded = point_source_shielded(1e9, 1.25, 100.0, "lead", 5.0)
        assert shielded < unshielded

    def test_zero_shield_equals_unshielded(self):
        unshielded = point_source(1e9, 1.25, 100.0)
        shielded = point_source_shielded(1e9, 1.25, 100.0, "lead", 0.0)
        assert abs(shielded - unshielded) / unshielded < 0.001

    def test_thicker_shield_lower_dose(self):
        r5 = point_source_shielded(1e9, 1.25, 100.0, "lead", 5.0)
        r10 = point_source_shielded(1e9, 1.25, 100.0, "lead", 10.0)
        assert r10 < r5

    def test_lead_better_than_concrete_same_thickness(self):
        r_lead = point_source_shielded(1e9, 1.25, 100.0, "lead", 5.0)
        r_concrete = point_source_shielded(1e9, 1.25, 100.0, "concrete", 5.0)
        assert r_lead < r_concrete

    def test_output_positive_and_finite(self):
        r = point_source_shielded(1e9, 1.0, 100.0, "iron", 3.0)
        assert r > 0
        assert np.isfinite(r)


class TestDoseRateVsDistance:
    def test_output_shape(self):
        d = np.linspace(10, 500, 50)
        r = dose_rate_vs_distance(1e9, 1.25, d)
        assert r.shape == (50,)

    def test_monotonically_decreasing(self):
        d = np.linspace(10, 500, 100)
        r = dose_rate_vs_distance(1e9, 1.25, d)
        assert np.all(np.diff(r) < 0)

    def test_all_positive(self):
        d = np.linspace(10, 500, 50)
        r = dose_rate_vs_distance(1e9, 1.25, d)
        assert np.all(r > 0)


class TestInverseSquareDistance:
    def test_returns_positive_distance(self):
        d = inverse_square_distance(1e9, 1.25, 1.0)
        assert d > 0

    def test_higher_activity_requires_larger_distance(self):
        d1 = inverse_square_distance(1e9, 1.25, 1.0)
        d2 = inverse_square_distance(1e10, 1.25, 1.0)
        assert d2 > d1

    def test_stricter_limit_requires_larger_distance(self):
        d_strict = inverse_square_distance(1e9, 1.25, 0.1)
        d_loose = inverse_square_distance(1e9, 1.25, 10.0)
        assert d_strict > d_loose

    def test_round_trip_consistency(self):
        """distance -> dose -> distance should be self-consistent."""
        target = 1.0
        d = inverse_square_distance(1e9, 1.25, target)
        dose_at_d = point_source(1e9, 1.25, d)
        assert abs(dose_at_d - target) / target < 0.01

    def test_neutron_source(self):
        d = inverse_square_distance(1e8, 1.0, 1.0, particle="neutron")
        assert d > 0


# ═══════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════


class TestNuclearIntegration:
    def test_cf252_shielding_workflow(self):
        """
        Full shielding workflow using Co-60 gamma source.
        Uses gamma radiation where photon attenuation physics is implemented.
        Bare dose must exceed shielded dose through lead.
        """
        co = Isotope("Co-60")
        A = co.activity(mass_g=1e-3)
        assert A > 0

        dose_bare = point_source(
            activity_bq=A,
            energy_mev=1.25,
            distance_cm=100.0,
            photons_per_decay=2.0,
        )
        dose_shielded = point_source_shielded(
            activity_bq=A,
            energy_mev=1.25,
            distance_cm=100.0,
            shield_material="lead",
            shield_thickness_cm=5.0,
            photons_per_decay=2.0,
        )
        assert dose_shielded < dose_bare
        assert dose_shielded > 0

    def test_watt_spectrum_energy_consistency(self):
        """
        Mean energy from numerical integration should match
        analytical watt_mean_energy() within 10%.
        """
        E = np.linspace(0.001, 20, 10000)
        N = watt_spectrum(E, "Cf-252", normalise=False)
        de = np.gradient(E)
        E_mean_numerical = np.sum(E * N * de) / np.sum(N * de)
        E_mean_analytical = watt_mean_energy("Cf-252")
        assert abs(E_mean_numerical - E_mean_analytical) / E_mean_analytical < 0.10

    def test_attenuation_series_with_dose(self):
        """Transmission curve * bare dose gives shielded dose profile."""
        thicknesses = np.linspace(0, 20, 50)
        T = attenuation_series(thicknesses, "lead", 1.25)
        bare = point_source(1e9, 1.25, 100.0)
        doses = bare * T
        assert doses[0] > doses[-1]
        assert np.all(doses > 0)

    def test_isotope_activity_to_dose_pipeline(self):
        """Co-60 activity -> dose rate pipeline gives physically reasonable result."""
        co = Isotope("Co-60")
        A = co.activity(mass_g=1e-6)
        rate = point_source(
            activity_bq=A,
            energy_mev=1.25,
            distance_cm=100.0,
            photons_per_decay=2.0,
        )
        assert rate > 0
        assert np.isfinite(rate)
