"""
tests/test_hep_kinematics.py
─────────────────────────────
Test suite for triples_sigfast.hep.kinematics.

Validates all JIT-compiled functions against known physics results:
- Z boson mass (90 GeV) from two back-to-back muons
- η asymptotic behaviour for beam-direction particles
- ΔR geometry
- pT, φ, rapidity arithmetic
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from triples_sigfast.hep import kinematics

# ── calculate_invariant_mass ──────────────────────────────────────────────────


class TestInvariantMass:
    def _make_pair(self, E1, px1, py1, pz1, E2, px2, py2, pz2):
        p1 = np.array([[E1, px1, py1, pz1]], dtype=np.float64)
        p2 = np.array([[E2, px2, py2, pz2]], dtype=np.float64)
        return p1, p2

    def test_z_boson_mass(self):
        """Two 45 GeV muons back-to-back → M ≈ 90 GeV (Z boson)."""
        p1, p2 = self._make_pair(45.0, 0.0, 44.9, 1.0, 45.0, 0.0, -44.9, -1.0)
        M = kinematics.calculate_invariant_mass(p1, p2)
        assert abs(M[0] - 90.0) < 0.5

    def test_massless_back_to_back(self):
        """Two photons, each 45 GeV, perfectly back-to-back → M = 90 GeV."""
        E = 45.0
        p1 = np.array([[E, E, 0.0, 0.0]])
        p2 = np.array([[E, -E, 0.0, 0.0]])
        M = kinematics.calculate_invariant_mass(p1, p2)
        np.testing.assert_allclose(M[0], 90.0, rtol=1e-6)

    def test_single_particle_zero_mass(self):
        """Massless particle with itself → M = 0."""
        p1 = np.array([[10.0, 10.0, 0.0, 0.0]])
        M = kinematics.calculate_invariant_mass(p1, p1)
        np.testing.assert_allclose(M[0], 0.0, atol=1e-10)

    def test_vectorized_n_pairs(self):
        """Batch of N pairs — output shape must be (N,)."""
        N = 500
        rng = np.random.default_rng(0)
        p1 = rng.uniform(10, 100, (N, 4))
        p2 = rng.uniform(10, 100, (N, 4))
        # Make energies large enough to avoid negative M²
        p1[:, 0] = np.linalg.norm(p1[:, 1:], axis=1) + 5.0
        p2[:, 0] = np.linalg.norm(p2[:, 1:], axis=1) + 5.0
        M = kinematics.calculate_invariant_mass(p1, p2)
        assert M.shape == (N,)
        assert np.all(M >= 0.0)

    def test_dtype_float64(self):
        """Output is always float64."""
        p1 = np.array([[45.0, 0.0, 44.9, 1.0]])
        p2 = np.array([[45.0, 0.0, -44.9, -1.0]])
        M = kinematics.calculate_invariant_mass(p1, p2)
        assert M.dtype == np.float64

    def test_shape_mismatch_raises(self):
        p1 = np.array([[45.0, 0.0, 44.9, 1.0]])
        p2 = np.array([[45.0, 0.0, -44.9, -1.0], [30.0, 15.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="same number of rows"):
            kinematics.calculate_invariant_mass(p1, p2)

    def test_bad_shape_raises(self):
        p1 = np.array([[45.0, 0.0, 44.9]])  # only 3 columns
        p2 = np.array([[45.0, 0.0, -44.9]])
        with pytest.raises(ValueError, match="shape"):
            kinematics.calculate_invariant_mass(p1, p2)


# ── calculate_pseudorapidity ──────────────────────────────────────────────────


class TestPseudorapidity:
    def test_transverse_particle_eta_zero(self):
        """Particle with pz=0 (90° to beam) → η = 0."""
        pz = np.array([0.0])
        p = np.array([1.0])
        eta = kinematics.calculate_pseudorapidity(pz, p)
        np.testing.assert_allclose(eta[0], 0.0, atol=1e-10)

    def test_forward_particle_large_eta(self):
        """Particle nearly along beam → |η| >> 1."""
        pz = np.array([9999.9])
        p = np.array([10000.0])
        eta = kinematics.calculate_pseudorapidity(pz, p)
        assert eta[0] > 5.0

    def test_backward_particle_negative_eta(self):
        """Particle nearly anti-parallel to beam → η << -1."""
        pz = np.array([-9999.9])
        p = np.array([10000.0])
        eta = kinematics.calculate_pseudorapidity(pz, p)
        assert eta[0] < -5.0

    def test_45_degree_known_value(self):
        """θ = 45° → η = -ln(tan(π/8)) ≈ 0.8814."""
        p = math.sqrt(2.0)
        pz = np.array([1.0])
        p_arr = np.array([p])
        eta = kinematics.calculate_pseudorapidity(pz, p_arr)
        expected = -math.log(math.tan(math.pi / 8.0))
        np.testing.assert_allclose(eta[0], expected, rtol=1e-6)

    def test_array_vectorized(self):
        """Batch of 1000 particles — output shape must be (1000,)."""
        N = 1000
        rng = np.random.default_rng(1)
        px = rng.uniform(-50, 50, N)
        py = rng.uniform(-50, 50, N)
        pz = rng.uniform(-50, 50, N)
        p = np.sqrt(px**2 + py**2 + pz**2)
        eta = kinematics.calculate_pseudorapidity(pz, p)
        assert eta.shape == (N,)
        assert np.all(np.isfinite(eta) | (np.abs(eta) >= 1e9))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            kinematics.calculate_pseudorapidity(np.array([1.0, 2.0]), np.array([1.0]))


# ── delta_r_matching ──────────────────────────────────────────────────────────


class TestDeltaR:
    def test_identical_particles_zero_dr(self):
        """Same particle → ΔR = 0."""
        eta = np.array([1.5])
        phi = np.array([0.3])
        dR = kinematics.delta_r_matching(eta, phi, eta, phi)
        np.testing.assert_allclose(dR[0], 0.0, atol=1e-12)

    def test_eta_separation_only(self):
        """ΔR = Δη when Δφ = 0."""
        eta1 = np.array([0.0])
        phi1 = np.array([0.0])
        eta2 = np.array([0.5])
        phi2 = np.array([0.0])
        dR = kinematics.delta_r_matching(eta1, phi1, eta2, phi2)
        np.testing.assert_allclose(dR[0], 0.5, rtol=1e-10)

    def test_phi_separation_only(self):
        """ΔR = Δφ when Δη = 0."""
        eta1 = np.array([0.0])
        phi1 = np.array([0.0])
        eta2 = np.array([0.0])
        phi2 = np.array([0.3])
        dR = kinematics.delta_r_matching(eta1, phi1, eta2, phi2)
        np.testing.assert_allclose(dR[0], 0.3, rtol=1e-10)

    def test_phi_wrapping_across_pi(self):
        """ΔR should use minimum Δφ — wrap [-π, π] correctly."""
        eta1 = np.array([0.0])
        phi1 = np.array([math.pi - 0.1])
        eta2 = np.array([0.0])
        phi2 = np.array([-math.pi + 0.1])
        dR = kinematics.delta_r_matching(eta1, phi1, eta2, phi2)
        # Δφ = 0.2 via wrapping, not 2π - 0.2
        np.testing.assert_allclose(dR[0], 0.2, atol=1e-10)

    def test_typical_jet_cone(self):
        """Two particles separated by ΔR = sqrt(0.3²+0.3²) ≈ 0.424."""
        eta1 = np.array([0.0])
        phi1 = np.array([0.0])
        eta2 = np.array([0.3])
        phi2 = np.array([0.3])
        dR = kinematics.delta_r_matching(eta1, phi1, eta2, phi2)
        expected = math.sqrt(0.3**2 + 0.3**2)
        np.testing.assert_allclose(dR[0], expected, rtol=1e-10)

    def test_vectorized_n_pairs(self):
        N = 2000
        rng = np.random.default_rng(2)
        eta1 = rng.uniform(-5, 5, N)
        phi1 = rng.uniform(-math.pi, math.pi, N)
        eta2 = rng.uniform(-5, 5, N)
        phi2 = rng.uniform(-math.pi, math.pi, N)
        dR = kinematics.delta_r_matching(eta1, phi1, eta2, phi2)
        assert dR.shape == (N,)
        assert np.all(dR >= 0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            kinematics.delta_r_matching(
                np.array([0.0, 1.0]),
                np.array([0.0]),
                np.array([0.5, 0.5]),
                np.array([0.0]),
            )


# ── transverse_momentum ───────────────────────────────────────────────────────


class TestTransverseMomentum:
    def test_known_value(self):
        px = np.array([3.0])
        py = np.array([4.0])
        pt = kinematics.transverse_momentum(px, py)
        np.testing.assert_allclose(pt[0], 5.0, rtol=1e-10)

    def test_zero_momentum(self):
        pt = kinematics.transverse_momentum(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(pt[0], 0.0, atol=1e-15)

    def test_vectorized(self):
        N = 10000
        rng = np.random.default_rng(3)
        px = rng.uniform(-100, 100, N)
        py = rng.uniform(-100, 100, N)
        pt = kinematics.transverse_momentum(px, py)
        expected = np.sqrt(px**2 + py**2)
        np.testing.assert_allclose(pt, expected, rtol=1e-12)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            kinematics.transverse_momentum(np.array([1.0, 2.0]), np.array([1.0]))


# ── azimuthal_angle ───────────────────────────────────────────────────────────


class TestAzimuthalAngle:
    def test_positive_x_axis(self):
        phi = kinematics.azimuthal_angle(np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(phi[0], 0.0, atol=1e-15)

    def test_positive_y_axis(self):
        phi = kinematics.azimuthal_angle(np.array([0.0]), np.array([1.0]))
        np.testing.assert_allclose(phi[0], math.pi / 2, rtol=1e-10)

    def test_range(self):
        N = 1000
        rng = np.random.default_rng(4)
        px = rng.uniform(-100, 100, N)
        py = rng.uniform(-100, 100, N)
        phi = kinematics.azimuthal_angle(px, py)
        assert np.all(phi >= -math.pi) and np.all(phi <= math.pi)


# ── rapidity ──────────────────────────────────────────────────────────────────


class TestRapidity:
    def test_transverse_particle_y_zero(self):
        """pz = 0 → y = 0."""
        y = kinematics.rapidity(np.array([10.0]), np.array([0.0]))
        np.testing.assert_allclose(y[0], 0.0, atol=1e-12)

    def test_massless_equals_pseudorapidity(self):
        """For massless particles, y ≈ η."""
        E = 50.0
        pz = 30.0
        pt = math.sqrt(E**2 - pz**2)
        p_tot = math.sqrt(pt**2 + pz**2)
        y = kinematics.rapidity(np.array([E]), np.array([pz]))
        eta = kinematics.calculate_pseudorapidity(np.array([pz]), np.array([p_tot]))
        np.testing.assert_allclose(y[0], eta[0], rtol=1e-3)

    def test_vectorized(self):
        N = 500
        rng = np.random.default_rng(5)
        pz = rng.uniform(-50, 50, N)
        E = np.abs(pz) + rng.uniform(1, 10, N)
        y = kinematics.rapidity(E, pz)
        assert y.shape == (N,)
