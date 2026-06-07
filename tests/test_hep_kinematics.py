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


# ── LorentzVector ────────────────────────────────────────────────────────────


class TestLorentzVector:
    """Tests for the LorentzVector convenience OOP wrapper."""

    def _muon(self):
        """45 GeV muon with moderate pT."""
        return kinematics.LorentzVector(E=45.0, px=0.0, py=44.9, pz=1.0)

    def _antimuon(self):
        return kinematics.LorentzVector(E=45.0, px=0.0, py=-44.9, pz=-1.0)

    # -- Construction and repr --------------------------------------------------

    def test_construction(self):
        v = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=0.0)
        assert v.E == 10.0
        assert v.px == 3.0
        assert v.py == 4.0
        assert v.pz == 0.0

    def test_repr(self):
        v = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=0.0)
        s = repr(v)
        assert "LorentzVector" in s
        assert "10" in s

    # -- Kinematic properties --------------------------------------------------

    def test_pt_3_4_5(self):
        """px=3, py=4 → pT = 5 (3-4-5 right triangle)."""
        v = kinematics.LorentzVector(E=20.0, px=3.0, py=4.0, pz=0.0)
        assert abs(v.pt - 5.0) < 1e-12

    def test_p_total_momentum(self):
        v = kinematics.LorentzVector(E=20.0, px=1.0, py=2.0, pz=2.0)
        expected = math.sqrt(1**2 + 2**2 + 2**2)
        assert abs(v.p - expected) < 1e-12

    def test_mass_massless_particle(self):
        """Photon: E = |p| → mass = 0."""
        v = kinematics.LorentzVector(E=10.0, px=10.0, py=0.0, pz=0.0)
        assert abs(v.mass) < 1e-10

    def test_mass_pion(self):
        """Approximate pion mass ~0.135 GeV."""
        E = 1.0
        p = math.sqrt(E**2 - 0.135**2)
        v = kinematics.LorentzVector(E=E, px=p, py=0.0, pz=0.0)
        assert abs(v.mass - 0.135) < 0.001

    def test_mass_negative_m2_returns_zero(self):
        """If M² < 0 due to rounding, return 0 not NaN."""
        v = kinematics.LorentzVector(E=1.0, px=1.0, py=0.001, pz=0.0)
        assert v.mass >= 0.0

    def test_eta_transverse_particle(self):
        """pz = 0 → η = 0."""
        v = kinematics.LorentzVector(E=10.0, px=10.0, py=0.0, pz=0.0)
        assert abs(v.eta) < 1e-10

    def test_eta_forward_particle(self):
        """Near beam direction → |η| >> 1."""
        v = kinematics.LorentzVector(E=100.0, px=0.1, py=0.0, pz=99.9)
        assert v.eta > 3.0

    def test_eta_backward_particle(self):
        v = kinematics.LorentzVector(E=100.0, px=0.1, py=0.0, pz=-99.9)
        assert v.eta < -3.0

    def test_phi_positive_x_axis(self):
        v = kinematics.LorentzVector(E=10.0, px=1.0, py=0.0, pz=0.0)
        assert abs(v.phi) < 1e-12

    def test_phi_positive_y_axis(self):
        v = kinematics.LorentzVector(E=10.0, px=0.0, py=1.0, pz=0.0)
        assert abs(v.phi - math.pi / 2) < 1e-10

    def test_rapidity_transverse(self):
        """pz = 0, E > 0 → y = 0."""
        v = kinematics.LorentzVector(E=10.0, px=0.0, py=10.0, pz=0.0)
        assert abs(v.rapidity) < 1e-10

    def test_beta_less_than_one(self):
        v = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=0.0)
        assert 0 < v.beta < 1

    def test_gamma_massless_is_inf(self):
        v = kinematics.LorentzVector(E=10.0, px=10.0, py=0.0, pz=0.0)
        assert v.gamma == float("inf")

    def test_gamma_massive(self):
        """For a 5 GeV proton: γ = E/m ≈ 5/0.938."""
        m = 0.938
        E = 5.0
        p = math.sqrt(E**2 - m**2)
        v = kinematics.LorentzVector(E=E, px=p, py=0.0, pz=0.0)
        expected = E / m
        assert abs(v.gamma - expected) < 0.01

    # -- Eta edge cases --------------------------------------------------------

    def test_eta_zero_momentum_returns_zero(self):
        """p = 0 (all zeros except E) → eta returns 0 without crashing."""
        v = kinematics.LorentzVector(E=0.0, px=0.0, py=0.0, pz=0.0)
        eta = v.eta
        assert isinstance(eta, float)

    # -- Z boson reconstruction -----------------------------------------------

    def test_z_boson_mass(self):
        """Two back-to-back muons → M ≈ 90 GeV (Z boson)."""
        Z = self._muon() + self._antimuon()
        assert abs(Z.mass - 90.0) < 1.0

    def test_z_boson_pt_near_zero(self):
        """Back-to-back muons have pT ≈ 0 (exact only if perfectly back-to-back)."""
        Z = self._muon() + self._antimuon()
        assert Z.pt < 0.1

    # -- Arithmetic operators -------------------------------------------------

    def test_add_components(self):
        v1 = kinematics.LorentzVector(E=1.0, px=2.0, py=3.0, pz=4.0)
        v2 = kinematics.LorentzVector(E=5.0, px=6.0, py=7.0, pz=8.0)
        v3 = v1 + v2
        assert v3.E == 6.0
        assert v3.px == 8.0
        assert v3.py == 10.0
        assert v3.pz == 12.0

    def test_sub_components(self):
        v1 = kinematics.LorentzVector(E=5.0, px=6.0, py=7.0, pz=8.0)
        v2 = kinematics.LorentzVector(E=1.0, px=2.0, py=3.0, pz=4.0)
        v3 = v1 - v2
        assert v3.E == 4.0
        assert v3.px == 4.0

    def test_negate(self):
        v = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=5.0)
        neg = -v
        assert neg.E == -10.0
        assert neg.px == -3.0

    def test_equality(self):
        v1 = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=5.0)
        v2 = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=5.0)
        assert v1 == v2

    def test_inequality(self):
        v1 = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=5.0)
        v2 = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=6.0)
        assert v1 != v2

    def test_equality_wrong_type(self):
        v = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=5.0)
        assert v.__eq__("not_a_vector") is NotImplemented

    # -- to_numpy -------------------------------------------------------------

    def test_to_numpy_shape(self):
        import numpy as np

        v = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=5.0)
        arr = v.to_numpy()
        assert arr.shape == (4,)
        assert arr.dtype == np.float64

    def test_to_numpy_values(self):
        import numpy as np

        v = kinematics.LorentzVector(E=10.0, px=3.0, py=4.0, pz=5.0)
        arr = v.to_numpy()
        np.testing.assert_array_equal(arr, [10.0, 3.0, 4.0, 5.0])

    # -- from_pt_eta_phi_mass -------------------------------------------------

    def test_from_pt_eta_phi_mass_roundtrip(self):
        """Construct from (pT, η, φ, m) and verify kinematic properties."""
        pt, eta, phi, mass = 30.0, 1.5, 0.8, 0.0
        v = kinematics.LorentzVector.from_pt_eta_phi_mass(pt, eta, phi, mass)
        assert abs(v.pt - pt) < 1e-8
        assert abs(v.eta - eta) < 1e-6
        assert abs(v.phi - phi) < 1e-8

    def test_from_pt_eta_phi_mass_with_mass(self):
        """Massive particle constructed from (pT, η, φ, m) has correct mass."""
        v = kinematics.LorentzVector.from_pt_eta_phi_mass(30.0, 0.0, 0.0, 0.938)
        assert abs(v.mass - 0.938) < 1e-6

    # -- delta_r --------------------------------------------------------------

    def test_delta_r_same_particle_zero(self):
        """ΔR between a particle and itself = 0."""
        v = kinematics.LorentzVector(E=50.0, px=30.0, py=20.0, pz=10.0)
        assert abs(v.delta_r(v)) < 1e-12

    def test_delta_r_eta_separation(self):
        """Two particles separated only in η: ΔR = |Δη|."""
        v1 = kinematics.LorentzVector.from_pt_eta_phi_mass(30.0, 0.0, 0.0)
        v2 = kinematics.LorentzVector.from_pt_eta_phi_mass(30.0, 0.5, 0.0)
        assert abs(v1.delta_r(v2) - 0.5) < 1e-6

    def test_delta_r_phi_wrapping(self):
        """ΔR uses minimum Δφ (wrapped to [-π, π])."""
        v1 = kinematics.LorentzVector.from_pt_eta_phi_mass(30.0, 0.0, math.pi - 0.1)
        v2 = kinematics.LorentzVector.from_pt_eta_phi_mass(30.0, 0.0, -math.pi + 0.1)
        # Minimum Δφ = 0.2 (not 2π - 0.2)
        assert abs(v1.delta_r(v2) - 0.2) < 1e-6

    # -- invariant_mass_with ---------------------------------------------------

    def test_invariant_mass_with_z_boson(self):
        """Z boson reconstruction via invariant_mass_with."""
        mu = self._muon()
        antimu = self._antimuon()
        M = mu.invariant_mass_with(antimu)
        assert abs(M - 90.0) < 1.0


# ── decay_two_body ────────────────────────────────────────────────────────────


class TestDecayTwoBody:
    def test_decay_kinematics(self):
        """Test energy and momentum conservation in rest frame."""
        parent = np.array([[100.0, 0.0, 0.0, 0.0]])
        p1, p2 = kinematics.decay_two_body(parent, 10.0, 20.0)

        # Energy conservation
        assert abs((p1[0, 0] + p2[0, 0]) - 100.0) < 1e-10
        # Momentum conservation
        assert abs(p1[0, 1] + p2[0, 1]) < 1e-10
        assert abs(p1[0, 2] + p2[0, 2]) < 1e-10
        assert abs(p1[0, 3] + p2[0, 3]) < 1e-10

        # Masses correct
        m1_obs = math.sqrt(
            abs(p1[0, 0] ** 2 - p1[0, 1] ** 2 - p1[0, 2] ** 2 - p1[0, 3] ** 2)
        )
        m2_obs = math.sqrt(
            abs(p2[0, 0] ** 2 - p2[0, 1] ** 2 - p2[0, 2] ** 2 - p2[0, 3] ** 2)
        )
        assert abs(m1_obs - 10.0) < 1e-5
        assert abs(m2_obs - 20.0) < 1e-5

    def test_forbidden_decay(self):
        """Test parent mass less than sum of child masses."""
        parent = np.array([[10.0, 0.0, 0.0, 0.0]])
        p1, p2 = kinematics.decay_two_body(parent, 6.0, 5.0)
        assert np.all(p1 == 0.0)
        assert np.all(p2 == 0.0)

    def test_boosted_decay(self):
        """Test conservation in boosted frame."""
        # Parent with mass 100 GeV, E = 200, px = pz = 0, py = sqrt(30000)
        E = 200.0
        py = math.sqrt(E**2 - 100.0**2)
        parent = np.array([[E, 0.0, py, 0.0]])
        p1, p2 = kinematics.decay_two_body(parent, 30.0, 40.0)

        assert abs((p1[0, 0] + p2[0, 0]) - E) < 1e-9
        assert abs(p1[0, 1] + p2[0, 1]) < 1e-9
        assert abs(p1[0, 2] + p2[0, 2] - py) < 1e-9
        assert abs(p1[0, 3] + p2[0, 3]) < 1e-9

    def test_vectorized(self):
        N = 1000
        rng = np.random.default_rng(42)
        pz = rng.uniform(-100, 100, N)
        E = np.abs(pz) + rng.uniform(50, 100, N)
        parent = np.zeros((N, 4))
        parent[:, 0] = E
        parent[:, 3] = pz

        p1, p2 = kinematics.decay_two_body(parent, 10.0, 20.0)

        # Check conservation
        assert np.allclose(p1[:, 0] + p2[:, 0], parent[:, 0])
        assert np.allclose(p1[:, 3] + p2[:, 3], parent[:, 3])
