"""
tests/test_hep_jets.py
───────────────────────
Test suite for triples_sigfast.hep.jets — anti-kT jet clustering.

Validates the Jet dataclass and cluster_jets() function against
known physical results using synthetic particle configurations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from triples_sigfast.hep.jets import Jet, cluster_jets

# ── Jet dataclass ─────────────────────────────────────────────────────────────


class TestJetDataclass:
    def test_pt_calculation(self):
        j = Jet(px=3.0, py=4.0, pz=0.0, E=10.0)
        np.testing.assert_allclose(j.pt, 5.0, rtol=1e-10)

    def test_eta_transverse_particle(self):
        """Particle with pz = 0 → η = 0."""
        j = Jet(px=10.0, py=0.0, pz=0.0, E=10.0)
        np.testing.assert_allclose(j.eta, 0.0, atol=1e-10)

    def test_eta_forward_particle(self):
        """Forward particle → large positive η."""
        j = Jet(px=0.1, py=0.1, pz=100.0, E=100.0)
        assert j.eta > 4.0

    def test_phi_positive_x(self):
        j = Jet(px=1.0, py=0.0, pz=0.0, E=1.0)
        np.testing.assert_allclose(j.phi, 0.0, atol=1e-10)

    def test_phi_positive_y(self):
        j = Jet(px=0.0, py=1.0, pz=0.0, E=1.0)
        np.testing.assert_allclose(j.phi, math.pi / 2, rtol=1e-10)

    def test_mass_massless(self):
        """Massless particle: E = |p|."""
        E = 50.0
        j = Jet(px=E, py=0.0, pz=0.0, E=E)
        np.testing.assert_allclose(j.mass, 0.0, atol=1e-6)

    def test_mass_known(self):
        """Particle at rest: M = E."""
        j = Jet(px=0.0, py=0.0, pz=0.0, E=91.2)
        np.testing.assert_allclose(j.mass, 91.2, rtol=1e-6)

    def test_repr(self):
        j = Jet(px=30.0, py=10.0, pz=5.0, E=35.0)
        r = repr(j)
        assert "Jet" in r
        assert "pt=" in r
        assert "eta=" in r

    def test_constituents_default_empty(self):
        j = Jet(px=1.0, py=0.0, pz=0.0, E=1.0)
        assert len(j.constituents) == 0

    def test_constituents_set(self):
        j = Jet(px=1.0, py=0.0, pz=0.0, E=1.0, constituents=np.array([0, 3, 7]))
        assert list(j.constituents) == [0, 3, 7]


# ── cluster_jets ──────────────────────────────────────────────────────────────


class TestClusterJets:
    def _two_isolated_particles(self, sep=2.0):
        """Two well-separated high-pT particles that should form two jets."""
        # Particle 1: η=0, φ=0, pT=50 GeV
        px1, py1, pz1 = 50.0, 0.0, 0.0
        E1 = math.sqrt(px1**2 + py1**2 + pz1**2)
        # Particle 2: η=sep, φ=π, pT=40 GeV — separated by ΔR ≈ sep (in η)
        # Put φ = π so it's on the opposite side
        px2, py2, pz2 = -40.0, 0.0, 0.0
        E2 = math.sqrt(px2**2 + py2**2 + pz2**2)
        px = np.array([px1, px2])
        py = np.array([py1, py2])
        pz = np.array([pz1, pz2])
        E = np.array([E1, E2])
        return px, py, pz, E

    def test_two_isolated_particles_give_two_jets(self):
        """Two well-separated particles should each become their own jet."""
        px, py, pz, E = self._two_isolated_particles(sep=2.0)
        jets = cluster_jets(px, py, pz, E, R=0.4)
        assert len(jets) == 2

    def test_jets_sorted_by_pt(self):
        """Leading jet should have highest pT."""
        px, py, pz, E = self._two_isolated_particles()
        jets = cluster_jets(px, py, pz, E, R=0.4)
        assert len(jets) >= 2
        assert jets[0].pt >= jets[1].pt

    def test_jet_pt_values(self):
        """Each jet's pT should match one of the input particles."""
        px, py, pz, E = self._two_isolated_particles()
        jets = cluster_jets(px, py, pz, E, R=0.4)
        pts = sorted([j.pt for j in jets], reverse=True)
        np.testing.assert_allclose(pts[0], 50.0, rtol=1e-6)
        np.testing.assert_allclose(pts[1], 40.0, rtol=1e-6)

    def test_jet_returns_list(self):
        px, py, pz, E = self._two_isolated_particles()
        jets = cluster_jets(px, py, pz, E, R=0.4)
        assert isinstance(jets, list)
        assert all(isinstance(j, Jet) for j in jets)

    def test_constituents_tracked(self):
        """Each jet should track which input particles it contains."""
        px, py, pz, E = self._two_isolated_particles()
        jets = cluster_jets(px, py, pz, E, R=0.4)
        all_constituents = set()
        for j in jets:
            all_constituents.update(j.constituents.tolist())
        # Both input particles must be accounted for
        assert all_constituents == {0, 1}

    def test_collinear_particles_merge_into_one_jet(self):
        """Two collinear (same direction) particles → merge into 1 jet."""
        # Both particles at η≈0, φ≈0, tiny angular separation << R=0.4
        E1 = 50.0
        E2 = 30.0
        # Tiny 0.01 rad separation
        px = np.array([E1 * math.cos(0.00), E2 * math.cos(0.01)])
        py = np.array([E1 * math.sin(0.00), E2 * math.sin(0.01)])
        pz = np.array([0.0, 0.0])
        E = np.array([E1, E2])
        jets = cluster_jets(px, py, pz, E, R=0.4)
        assert len(jets) == 1
        # Combined jet should have both constituents
        assert len(jets[0].constituents) == 2
        # Energy should be conserved
        np.testing.assert_allclose(jets[0].E, E1 + E2, rtol=1e-10)

    def test_min_pt_filter(self):
        """Jets below min_pt threshold must be discarded."""
        px, py, pz, E = self._two_isolated_particles()
        # Both jets have pT 40 and 50 GeV; cut at 45 GeV removes lower one
        jets = cluster_jets(px, py, pz, E, R=0.4, min_pt=45.0)
        assert len(jets) == 1
        assert jets[0].pt >= 45.0

    def test_empty_input(self):
        """Empty input → empty jet list."""
        jets = cluster_jets(np.array([]), np.array([]), np.array([]), np.array([]))
        assert jets == []

    def test_single_particle_becomes_jet(self):
        """One particle → one jet."""
        jets = cluster_jets(
            np.array([50.0]), np.array([0.0]), np.array([0.0]), np.array([50.0]), R=0.4
        )
        assert len(jets) == 1
        np.testing.assert_allclose(jets[0].pt, 50.0, rtol=1e-6)

    def test_invalid_R_raises(self):
        with pytest.raises(ValueError, match="radius R must be > 0"):
            cluster_jets(
                np.array([50.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([50.0]),
                R=0.0,
            )

    def test_jet_energy_momentum_conservation(self):
        """Total 4-momentum of jets must equal total input 4-momentum."""
        N = 20
        rng = np.random.default_rng(10)
        px = rng.uniform(-50, 50, N)
        py = rng.uniform(-50, 50, N)
        pz = rng.uniform(-30, 30, N)
        E = np.sqrt(px**2 + py**2 + pz**2) + rng.uniform(0.5, 2.0, N)
        jets = cluster_jets(px, py, pz, E, R=0.4, min_pt=0.0)
        # Sum of jet 4-vectors must equal input sum
        total_px_in = px.sum()
        total_py_in = py.sum()
        total_E_in = E.sum()
        total_px_out = sum(j.px for j in jets)
        total_py_out = sum(j.py for j in jets)
        total_E_out = sum(j.E for j in jets)
        np.testing.assert_allclose(total_px_out, total_px_in, rtol=1e-10)
        np.testing.assert_allclose(total_py_out, total_py_in, rtol=1e-10)
        np.testing.assert_allclose(total_E_out, total_E_in, rtol=1e-10)

    def test_large_event_runs(self):
        """100-particle event runs without error."""
        N = 100
        rng = np.random.default_rng(11)
        px = rng.uniform(-50, 50, N)
        py = rng.uniform(-50, 50, N)
        pz = rng.uniform(-100, 100, N)
        E = np.sqrt(px**2 + py**2 + pz**2) + 1.0
        jets = cluster_jets(px, py, pz, E, R=0.4, min_pt=5.0)
        assert isinstance(jets, list)
        assert all(j.pt >= 5.0 for j in jets)

    def test_jet_mass_non_negative(self):
        """Reconstructed jet mass must never be negative."""
        N = 50
        rng = np.random.default_rng(12)
        px = rng.uniform(-50, 50, N)
        py = rng.uniform(-50, 50, N)
        pz = rng.uniform(-50, 50, N)
        E = np.sqrt(px**2 + py**2 + pz**2) + 0.5
        jets = cluster_jets(px, py, pz, E, R=0.4)
        for j in jets:
            assert j.mass >= 0.0

    def test_r_parameter_effect(self):
        """Larger R should produce fewer, larger jets from the same event."""
        N = 30
        rng = np.random.default_rng(13)
        px = rng.uniform(-30, 30, N)
        py = rng.uniform(-30, 30, N)
        pz = rng.uniform(-30, 30, N)
        E = np.sqrt(px**2 + py**2 + pz**2) + 1.0
        jets_small_R = cluster_jets(px, py, pz, E, R=0.4)
        jets_large_R = cluster_jets(px, py, pz, E, R=1.0)
        assert len(jets_large_R) <= len(jets_small_R)
