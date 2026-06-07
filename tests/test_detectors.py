"""
tests/test_detectors.py
────────────────────────
Test suite for triples_sigfast.detectors.

Validates NaIDetector, HPGeDetector, He3Detector, and BF3Detector against
known physical benchmarks and edge cases.
"""

from __future__ import annotations

import math

import pytest

from triples_sigfast.detectors import (
    BF3Detector,
    He3Detector,
    HPGeDetector,
    NaIDetector,
    available_detectors,
)

# ─────────────────────────────────────────────────────────────────────────────
# NaIDetector
# ─────────────────────────────────────────────────────────────────────────────


class TestNaIDetector:
    def test_repr(self):
        det = NaIDetector(thickness_cm=7.62)
        assert "NaIDetector" in repr(det)
        assert "7.62" in repr(det)

    def test_intrinsic_efficiency_range(self):
        """Efficiency must be in (0, 1)."""
        det = NaIDetector(thickness_cm=7.62)
        eff = det.intrinsic_efficiency(energy_mev=0.662)
        assert 0 < eff < 1

    def test_efficiency_increases_with_thickness(self):
        """Thicker crystal absorbs more photons."""
        e_thin = NaIDetector(thickness_cm=2.54).intrinsic_efficiency(0.662)
        e_thick = NaIDetector(thickness_cm=15.24).intrinsic_efficiency(0.662)
        assert e_thick > e_thin

    def test_efficiency_decreases_with_energy(self):
        """Higher energy photons penetrate more (lower efficiency)."""
        det = NaIDetector(thickness_cm=7.62)
        e_low = det.intrinsic_efficiency(energy_mev=0.1)
        e_high = det.intrinsic_efficiency(energy_mev=5.0)
        assert e_low > e_high

    def test_energy_resolution_positive(self):
        det = NaIDetector(thickness_cm=7.62)
        fwhm = det.energy_resolution(energy_mev=0.662)
        assert fwhm > 0

    def test_energy_resolution_decreases_with_energy(self):
        """Percent resolution improves (decreases) with energy for NaI."""
        det = NaIDetector(thickness_cm=7.62)
        fwhm_low = det.energy_resolution(energy_mev=0.1) / 0.1
        fwhm_high = det.energy_resolution(energy_mev=5.0) / 5.0
        assert fwhm_low > fwhm_high

    def test_zero_energy_raises(self):
        det = NaIDetector(thickness_cm=7.62)
        with pytest.raises(ValueError, match="energy_mev"):
            det.intrinsic_efficiency(energy_mev=0.0)

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="thickness_cm"):
            NaIDetector(thickness_cm=-1.0)

    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError, match="diameter_cm"):
            NaIDetector(thickness_cm=7.62, diameter_cm=0.0)

    def test_resolution_zero_energy_raises(self):
        det = NaIDetector(thickness_cm=7.62)
        with pytest.raises(ValueError, match="energy_mev"):
            det.energy_resolution(energy_mev=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# HPGeDetector
# ─────────────────────────────────────────────────────────────────────────────


class TestHPGeDetector:
    def test_repr(self):
        ge = HPGeDetector(relative_efficiency=0.30)
        assert "HPGeDetector" in repr(ge)
        assert "0.30" in repr(ge)

    def test_energy_resolution_positive(self):
        ge = HPGeDetector(relative_efficiency=0.30)
        fwhm = ge.energy_resolution(energy_mev=1.332)
        assert fwhm > 0

    def test_hpge_much_better_than_nai(self):
        """HPGe resolution should be much better (smaller FWHM) than NaI."""
        nai = NaIDetector(thickness_cm=7.62)
        ge = HPGeDetector(relative_efficiency=0.30)
        nai_fwhm = nai.energy_resolution(energy_mev=1.332)
        ge_fwhm = ge.energy_resolution(energy_mev=1.332)
        # HPGe is always strictly better than NaI
        assert ge_fwhm < nai_fwhm

    def test_resolution_at_co60_peak(self):
        """Co-60 at 1.332 MeV: HPGe FWHM ~ 50-60 keV from Fano model (Knoll, Ch.11)."""
        ge = HPGeDetector(relative_efficiency=0.30)
        fwhm_mev = ge.energy_resolution(energy_mev=1.332)
        fwhm_kev = fwhm_mev * 1000
        # Fano-model prediction: ~53 keV at 1.332 MeV for Ge (Knoll Table 11.2)
        assert 30.0 < fwhm_kev < 100.0

    def test_invalid_efficiency_zero(self):
        with pytest.raises(ValueError, match="relative_efficiency"):
            HPGeDetector(relative_efficiency=0.0)

    def test_invalid_efficiency_above_one(self):
        with pytest.raises(ValueError, match="relative_efficiency"):
            HPGeDetector(relative_efficiency=1.5)

    def test_zero_energy_raises(self):
        ge = HPGeDetector()
        with pytest.raises(ValueError, match="energy_mev"):
            ge.energy_resolution(energy_mev=0.0)

    def test_resolution_increases_with_energy(self):
        """Absolute FWHM increases with energy for HPGe (Fano model)."""
        ge = HPGeDetector(relative_efficiency=0.30)
        fwhm_low = ge.energy_resolution(energy_mev=0.1)
        fwhm_high = ge.energy_resolution(energy_mev=5.0)
        assert fwhm_high > fwhm_low


# ─────────────────────────────────────────────────────────────────────────────
# He3Detector
# ─────────────────────────────────────────────────────────────────────────────


class TestHe3Detector:
    def test_repr(self):
        det = He3Detector(pressure_atm=4.0, active_length_cm=30.0)
        assert "He3Detector" in repr(det)

    def test_thermal_efficiency_range(self):
        det = He3Detector(pressure_atm=4.0, active_length_cm=30.0)
        eff = det.thermal_efficiency()
        assert 0 < eff < 1

    def test_efficiency_increases_with_pressure(self):
        e_low = He3Detector(
            pressure_atm=2.0, active_length_cm=30.0
        ).thermal_efficiency()
        e_high = He3Detector(
            pressure_atm=10.0, active_length_cm=30.0
        ).thermal_efficiency()
        assert e_high > e_low

    def test_efficiency_increases_with_length(self):
        e_short = He3Detector(
            pressure_atm=4.0, active_length_cm=10.0
        ).thermal_efficiency()
        e_long = He3Detector(
            pressure_atm=4.0, active_length_cm=100.0
        ).thermal_efficiency()
        assert e_long > e_short

    def test_zero_pressure_raises(self):
        with pytest.raises(ValueError, match="pressure_atm"):
            He3Detector(pressure_atm=0.0)

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="active_length_cm"):
            He3Detector(active_length_cm=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# BF3Detector
# ─────────────────────────────────────────────────────────────────────────────


class TestBF3Detector:
    def test_repr(self):
        det = BF3Detector(pressure_atm=0.9, active_length_cm=50.0, b10_enrichment=0.96)
        assert "BF3Detector" in repr(det)

    def test_thermal_efficiency_range(self):
        det = BF3Detector()
        eff = det.thermal_efficiency()
        assert 0 < eff < 1

    def test_enriched_more_efficient_than_natural(self):
        e_natural = BF3Detector(b10_enrichment=0.199).thermal_efficiency()
        e_enriched = BF3Detector(b10_enrichment=0.96).thermal_efficiency()
        assert e_enriched > e_natural

    def test_zero_pressure_raises(self):
        with pytest.raises(ValueError, match="pressure_atm"):
            BF3Detector(pressure_atm=0.0)

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="active_length_cm"):
            BF3Detector(active_length_cm=0.0)

    def test_zero_enrichment_raises(self):
        with pytest.raises(ValueError, match="b10_enrichment"):
            BF3Detector(b10_enrichment=0.0)

    def test_enrichment_above_one_raises(self):
        with pytest.raises(ValueError, match="b10_enrichment"):
            BF3Detector(b10_enrichment=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectorsModuleAPI:
    def test_available_detectors_list(self):
        names = available_detectors()
        assert isinstance(names, list)
        assert "NaIDetector" in names
        assert "HPGeDetector" in names
        assert "He3Detector" in names
        assert "BF3Detector" in names

    def test_all_importable(self):
        from triples_sigfast.detectors import (
            BF3Detector,
            He3Detector,
            HPGeDetector,
            NaIDetector,
        )

        for cls in (NaIDetector, HPGeDetector, He3Detector, BF3Detector):
            assert callable(cls)
