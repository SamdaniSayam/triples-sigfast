"""
tests/test_isotope_id.py
────────────────────────
Test suite for triples_sigfast.nuclear.isotope_id

Covers: peak matching, tolerance edge cases, 0.511 ambiguity, summary output.
"""

from __future__ import annotations

import numpy as np
import pytest

from triples_sigfast.nuclear.isotope_id import (
    _match_annihilation,
    _match_peak,
    identify_isotopes,
    isotope_summary,
)

# ═══════════════════════════════════════════════════════════════════
# _match_peak  (JIT core)
# ═══════════════════════════════════════════════════════════════════


class TestMatchPeakJIT:
    """Low-level tests for the Numba decision tree."""

    def test_cs137_exact(self):
        """Cs-137 at exactly 0.662 MeV."""
        code = _match_peak(0.662, 0.002)
        assert code >= 0
        assert code // 100 == 0  # isotope code 0 = Cs-137

    def test_co60_line1(self):
        """Co-60 first line at 1.173 MeV."""
        code = _match_peak(1.173, 0.002)
        assert code >= 0
        assert code // 100 == 1  # Co-60

    def test_co60_line2(self):
        """Co-60 second line at 1.332 MeV."""
        code = _match_peak(1.332, 0.002)
        assert code >= 0
        assert code // 100 == 1  # Co-60

    def test_na22_511(self):
        """Na-22 at 0.511 MeV (primary tree returns Na-22)."""
        code = _match_peak(0.511, 0.002)
        assert code >= 0
        assert code // 100 == 2  # Na-22

    def test_na22_1275(self):
        """Na-22 at 1.275 MeV."""
        code = _match_peak(1.275, 0.002)
        assert code >= 0
        assert code // 100 == 2  # Na-22

    def test_am241(self):
        """Am-241 at 0.0595 MeV."""
        code = _match_peak(0.0595, 0.002)
        assert code >= 0
        assert code // 100 == 3  # Am-241

    def test_k40(self):
        """K-40 at 1.461 MeV."""
        code = _match_peak(1.461, 0.002)
        assert code >= 0
        assert code // 100 == 6  # K-40

    def test_tl208(self):
        """Tl-208 at 2.614 MeV."""
        code = _match_peak(2.614, 0.002)
        assert code >= 0
        assert code // 100 == 12  # Tl-208

    def test_no_match_random_energy(self):
        """0.999 MeV should not match any known line."""
        code = _match_peak(0.999, 0.002)
        assert code == -1

    def test_no_match_far_from_any_line(self):
        """3.5 MeV — far above all reference energies."""
        code = _match_peak(3.5, 0.002)
        assert code == -1

    def test_annihilation_helper(self):
        """_match_annihilation returns Annihilation code for 0.511."""
        code = _match_annihilation(0.511, 0.002)
        assert code >= 0
        assert code // 100 == 11  # Annihilation

    def test_annihilation_no_match(self):
        code = _match_annihilation(0.662, 0.002)
        assert code == -1


# ═══════════════════════════════════════════════════════════════════
# identify_isotopes  (public API)
# ═══════════════════════════════════════════════════════════════════


class TestIdentifyIsotopes:
    """Integration tests for the public identification function."""

    def test_cs137(self):
        peaks = np.array([0.662])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        names = [r["isotope"] for r in results]
        assert "Cs-137" in names

    def test_co60_both_lines(self):
        peaks = np.array([1.173, 1.332])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        co_matches = [r for r in results if r["isotope"] == "Co-60"]
        assert len(co_matches) == 2
        energies = {r["energy_mev"] for r in co_matches}
        assert 1.173 in energies
        assert 1.332 in energies

    def test_na22_both_lines(self):
        peaks = np.array([0.511, 1.275])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        na_matches = [r for r in results if r["isotope"] == "Na-22"]
        assert len(na_matches) == 2

    def test_na22_511_also_reports_annihilation(self):
        """0.511 MeV alone → Annihilation only (no Na-22 without 1.275)."""
        peaks = np.array([0.511])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        names = [r["isotope"] for r in results]
        assert "Na-22" not in names
        assert "Annihilation" in names

    def test_na22_511_with_1275_reports_na22(self):
        """0.511 + 1.275 → both Na-22 and Annihilation."""
        peaks = np.array([0.511, 1.275])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        names = [r["isotope"] for r in results]
        assert "Na-22" in names
        assert "Annihilation" in names

    def test_no_match_random(self):
        peaks = np.array([0.999])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        assert len(results) == 0

    def test_empty_input(self):
        peaks = np.array([], dtype=np.float64)
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        assert results == []

    def test_confidence_high_exact(self):
        """Exact match → high confidence (Δ = 0 keV)."""
        peaks = np.array([0.662])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        cs_match = [r for r in results if r["isotope"] == "Cs-137"][0]
        assert cs_match["confidence"] == "high"

    def test_confidence_medium_near_edge(self):
        """Match at 1.5 keV offset → medium confidence."""
        peaks = np.array([0.662 + 0.0015])  # +1.5 keV
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        cs_match = [r for r in results if r["isotope"] == "Cs-137"][0]
        assert cs_match["confidence"] == "medium"

    def test_confidence_high_within_1kev(self):
        """Match at exactly 1.0 keV offset → high confidence (≤ 1 keV)."""
        peaks = np.array([0.662 + 0.001])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        cs_match = [r for r in results if r["isotope"] == "Cs-137"][0]
        assert cs_match["confidence"] == "high"

    def test_tolerance_just_outside_no_match(self):
        """Peak just outside tolerance window → no match."""
        peaks = np.array([0.662 + 0.0025])  # +2.5 keV, beyond 2 keV window
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        cs_matches = [r for r in results if r["isotope"] == "Cs-137"]
        assert len(cs_matches) == 0

    def test_tolerance_just_inside_matches(self):
        """Peak just inside tolerance window → match."""
        peaks = np.array([0.662 + 0.0019])  # +1.9 keV, within 2 keV
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        cs_matches = [r for r in results if r["isotope"] == "Cs-137"]
        assert len(cs_matches) == 1

    def test_wider_tolerance(self):
        """Wider tolerance should catch more peaks."""
        peaks = np.array([0.662 + 0.004])  # +4 keV
        # Default 2 keV — should miss
        results_tight = identify_isotopes(peaks, tolerance_kev=2.0)
        assert len([r for r in results_tight if r["isotope"] == "Cs-137"]) == 0
        # 5 keV — should catch
        results_wide = identify_isotopes(peaks, tolerance_kev=5.0)
        assert len([r for r in results_wide if r["isotope"] == "Cs-137"]) == 1

    def test_multiple_isotopes_mixed(self):
        """Multiple peaks identifying different isotopes."""
        peaks = np.array([0.662, 1.173, 1.332, 0.0595])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        names = {r["isotope"] for r in results}
        assert "Cs-137" in names
        assert "Co-60" in names
        assert "Am-241" in names

    def test_matched_peak_value(self):
        """The matched_peak_mev should be the actual input value."""
        peaks = np.array([0.663])  # slightly off
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        cs_match = [r for r in results if r["isotope"] == "Cs-137"]
        assert len(cs_match) == 1
        assert cs_match[0]["matched_peak_mev"] == pytest.approx(0.663)

    def test_ba133_lines(self):
        """Ba-133 has five gamma lines — check a selection."""
        peaks = np.array([0.081, 0.356])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        ba_matches = [r for r in results if r["isotope"] == "Ba-133"]
        assert len(ba_matches) == 2

    def test_eu152_lines(self):
        """Eu-152 has seven gamma lines — check a selection."""
        peaks = np.array([0.122, 0.344, 1.408])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        eu_matches = [r for r in results if r["isotope"] == "Eu-152"]
        assert len(eu_matches) == 3

    def test_bi214_lines(self):
        """Bi-214 lines at 0.609, 1.120, 1.764."""
        peaks = np.array([0.609, 1.120, 1.764])
        results = identify_isotopes(peaks, tolerance_kev=2.0)
        bi_matches = [r for r in results if r["isotope"] == "Bi-214"]
        assert len(bi_matches) == 3


# ═══════════════════════════════════════════════════════════════════
# isotope_summary  (CLI-friendly output)
# ═══════════════════════════════════════════════════════════════════


class TestIsotopeSummary:
    def test_returns_string(self):
        peaks = np.array([0.662, 1.173])
        out = isotope_summary(peaks)
        assert isinstance(out, str)
        assert "Cs-137" in out
        assert "Co-60" in out

    def test_no_matches_message(self):
        peaks = np.array([0.999])
        out = isotope_summary(peaks)
        assert "No isotopes identified" in out

    def test_empty_array(self):
        peaks = np.array([], dtype=np.float64)
        out = isotope_summary(peaks)
        assert "No isotopes identified" in out

    def test_total_count_shown(self):
        peaks = np.array([0.662])
        out = isotope_summary(peaks)
        assert "Total matches:" in out
