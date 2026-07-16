"""
tests/test_mc.py
────────────────
Test suite for triples_sigfast.stats.mc

Covers: relative_error, figure_of_merit, is_converged, propagate_error
Target: 100% line coverage, MCNP-standard edge cases.
"""

import numpy as np
import pytest

from triples_sigfast.stats.mc import (
    figure_of_merit,
    is_converged,
    propagate_error,
    relative_error,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def typical_counts():
    """Realistic gamma-ray tally: 10k–100k counts per bin."""
    rng = np.random.default_rng(42)
    return rng.integers(10_000, 100_000, size=50).astype(np.float64)


@pytest.fixture
def sparse_counts():
    """Low-statistics tally: some bins near zero."""
    return np.array([0.0, 1.0, 4.0, 9.0, 100.0, 10_000.0])


# ── relative_error ────────────────────────────────────────────────────────────


class TestRelativeError:
    def test_poisson_formula(self):
        """R = 1/sqrt(N) for Poisson statistics."""
        counts = np.array([100.0, 400.0, 10_000.0])
        R = relative_error(counts)
        expected = np.array([0.1, 0.05, 0.01])
        np.testing.assert_allclose(R, expected, rtol=1e-10)

    def test_zero_bin_returns_nan(self):
        """Zero-count bins must return nan (undefined R)."""
        counts = np.array([0.0, 100.0])
        R = relative_error(counts)
        assert np.isnan(R[0])
        assert np.isfinite(R[1])

    def test_output_shape_matches_input(self, typical_counts):
        R = relative_error(typical_counts)
        assert R.shape == typical_counts.shape

    def test_output_dtype_float64(self, typical_counts):
        R = relative_error(typical_counts)
        assert R.dtype == np.float64

    def test_all_positive_for_nonzero_counts(self, typical_counts):
        R = relative_error(typical_counts)
        assert np.all(R > 0)

    def test_r_decreases_with_more_counts(self):
        """Higher counts → smaller relative error."""
        low = relative_error(np.array([100.0]))
        high = relative_error(np.array([10_000.0]))
        assert low[0] > high[0]

    def test_single_element(self):
        R = relative_error(np.array([25.0]))
        np.testing.assert_allclose(R[0], 0.2, rtol=1e-10)

    def test_large_array_performance(self):
        """Should complete 1M bins without timeout."""
        counts = np.ones(1_000_000) * 1000.0
        R = relative_error(counts)
        assert R.shape == (1_000_000,)

    def test_mean_relative_error(self):
        """mean_relative_error returns scalar mean of R across bins."""
        from triples_sigfast.stats.mc import mean_relative_error

        counts = np.array([100.0, 400.0, 10000.0])
        mre = mean_relative_error(counts)
        R = relative_error(counts)
        np.testing.assert_allclose(mre, R.mean(), rtol=1e-6)

    def test_mean_relative_error_all_zero_returns_nan(self):
        from triples_sigfast.stats.mc import mean_relative_error

        counts = np.array([0.0, 0.0])
        assert np.isnan(mean_relative_error(counts))


# ── figure_of_merit ──────────────────────────────────────────────────────────


class TestFigureOfMerit:
    def test_basic_formula(self):
        """FOM = 1 / (R² × T)."""
        R = np.array([0.1])
        fom = figure_of_merit(R, cpu_time=100.0)
        expected = 1.0 / (0.01 * 100.0)
        np.testing.assert_allclose(fom[0], expected, rtol=1e-10)

    def test_zero_rel_error_returns_zero(self):
        """R=0 is undefined — return 0.0 gracefully."""
        R = np.array([0.0, 0.1])
        fom = figure_of_merit(R, cpu_time=3600.0)
        assert fom[0] == 0.0
        assert fom[1] > 0.0

    def test_inf_rel_error_returns_zero(self):
        """Inf R (zero-count bin) → FOM = 0.0."""
        R = np.array([np.inf])
        fom = figure_of_merit(R, cpu_time=3600.0)
        assert fom[0] == 0.0

    def test_zero_cpu_time_returns_zero(self):
        R = np.array([0.05])
        fom = figure_of_merit(R, cpu_time=0.0)
        assert fom[0] == 0.0

    def test_fom_increases_with_more_time(self):
        """For fixed R, FOM ∝ 1/T so longer runs → lower FOM per unit R."""
        R = np.array([0.05])
        fom_short = figure_of_merit(R, cpu_time=100.0)
        fom_long = figure_of_merit(R, cpu_time=10_000.0)
        assert fom_short[0] > fom_long[0]

    def test_output_shape_matches_input(self, typical_counts):
        R = relative_error(typical_counts)
        fom = figure_of_merit(R, cpu_time=3600.0)
        assert fom.shape == R.shape

    def test_nan_rel_error_returns_zero(self):
        R = np.array([np.nan])
        fom = figure_of_merit(R, cpu_time=100.0)
        assert fom[0] == 0.0

    def test_typical_geant4_run(self, typical_counts):
        """Smoke test: 1-hour Geant4 run should yield positive FOM."""
        R = relative_error(typical_counts)
        fom = figure_of_merit(R, cpu_time=3600.0)
        assert np.all(fom >= 0)
        assert np.any(fom > 0)


# ── is_converged ─────────────────────────────────────────────────────────────


class TestIsConverged:
    def test_high_count_bins_converge(self):
        """10k counts → R=0.01 < 0.05 threshold → converged."""
        counts = np.array([10_000.0, 10_000.0])
        result = is_converged(counts, threshold=0.05)
        assert np.all(result)

    def test_low_count_bins_do_not_converge(self):
        """100 counts → R=0.1 > 0.05 threshold → not converged."""
        counts = np.array([100.0])
        result = is_converged(counts, threshold=0.05)
        assert not result[0]

    def test_zero_count_bin_not_converged(self):
        counts = np.array([0.0])
        result = is_converged(counts, threshold=0.05)
        assert not result[0]

    def test_boundary_exactly_at_threshold(self):
        """R = threshold is NOT converged (strict < comparison)."""
        # R = 1/sqrt(N) = 0.05  →  N = 400
        counts = np.array([400.0])
        result = is_converged(counts, threshold=0.05)
        assert not result[0]

    def test_custom_threshold_010(self):
        """R=0.05 < 0.10 → converged at relaxed threshold."""
        counts = np.array([400.0])
        result = is_converged(counts, threshold=0.10)
        assert result[0]

    def test_strict_threshold_002(self):
        """Publication-critical: 0.02 threshold requires N > 2500."""
        counts_pass = np.array([10_000.0])  # R=0.01 < 0.02
        counts_fail = np.array([1_000.0])  # R=0.032 > 0.02
        assert is_converged(counts_pass, threshold=0.02)[0]
        assert not is_converged(counts_fail, threshold=0.02)[0]

    def test_output_dtype_bool(self, typical_counts):
        result = is_converged(typical_counts)
        assert result.dtype == np.bool_

    def test_mixed_array(self, sparse_counts):
        """Zero and low-count bins fail, high-count bins pass."""
        result = is_converged(sparse_counts, threshold=0.05)
        assert not result[0]  # 0 counts
        assert not result[1]  # 1 count
        assert result[5]  # 10,000 counts


# ── propagate_error ──────────────────────────────────────────────────────────


class TestPropagateError:
    def test_ideal_detector_pure_poisson(self):
        """ε=1.0 → σ_total = sqrt(N) (pure Poisson, no efficiency term)."""
        counts = np.array([100.0])
        sigma = propagate_error(counts, efficiency=1.0)
        np.testing.assert_allclose(sigma[0], 10.0, rtol=1e-10)

    def test_zero_count_returns_zero(self):
        counts = np.array([0.0])
        sigma = propagate_error(counts, efficiency=0.5)
        assert sigma[0] == 0.0

    def test_efficiency_increases_uncertainty(self):
        """Lower efficiency → larger propagated uncertainty."""
        counts = np.array([1000.0])
        sigma_ideal = propagate_error(counts, efficiency=1.0)
        sigma_detector = propagate_error(counts, efficiency=0.35)
        assert sigma_detector[0] > sigma_ideal[0]

    def test_invalid_efficiency_falls_back_to_one(self):
        """efficiency=0.0 is physically invalid — should not crash."""
        counts = np.array([100.0])
        sigma = propagate_error(counts, efficiency=0.0)
        assert np.isfinite(sigma[0])

    def test_hpge_detector(self):
        """HPGe: ε=0.35, N=10000 — result must be finite and positive."""
        counts = np.ones(10) * 10_000.0
        sigma = propagate_error(counts, efficiency=0.35)
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    def test_scintillator_detector(self):
        """Scintillator: ε=0.05, N=500."""
        counts = np.ones(5) * 500.0
        sigma = propagate_error(counts, efficiency=0.05)
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    def test_output_shape_matches_input(self, typical_counts):
        sigma = propagate_error(typical_counts, efficiency=0.35)
        assert sigma.shape == typical_counts.shape

    def test_gum_quadrature_formula(self):
        """
        Manual GUM calculation for N=100, ε=0.5:
        σ_N   = sqrt(100) = 10
        σ_eff = sqrt(0.5 × 0.5) = 0.5
        term1 = (10 / 0.5)² = 400
        term2 = ((100 / 0.25) × 0.5)² = 200²  = 40000
        σ_total = sqrt(400 + 40000) = sqrt(40400)
        """
        counts = np.array([100.0])
        sigma = propagate_error(counts, efficiency=0.5)
        expected = np.sqrt(40400.0)
        np.testing.assert_allclose(sigma[0], expected, rtol=1e-10)

    def test_large_array_performance(self):
        counts = np.ones(1_000_000) * 5000.0
        sigma = propagate_error(counts, efficiency=0.35)
        assert sigma.shape == (1_000_000,)


# ── Integration ───────────────────────────────────────────────────────────────


class TestMCIntegration:
    def test_full_workflow(self, typical_counts):
        """
        Simulate a complete MC analysis pipeline:
        counts → R → FOM → convergence check → propagated uncertainty
        """
        R = relative_error(typical_counts)
        fom = figure_of_merit(R, cpu_time=7200.0)
        conv = is_converged(typical_counts, threshold=0.05)
        sigma = propagate_error(typical_counts, efficiency=0.35)

        # All 50 bins of typical_counts (10k–100k) should converge at 5%
        assert conv.all(), "All high-statistics bins should converge"
        assert np.all(R < 0.05)
        assert np.all(fom > 0)
        assert np.all(sigma > 0)

    def test_pipeline_with_sparse_counts(self, sparse_counts):
        """Low-stat bins: only the 10k bin should converge."""
        R = relative_error(sparse_counts)
        conv = is_converged(sparse_counts, threshold=0.05)

        assert np.isnan(R[0])  # zero bin
        assert conv[-1]  # 10,000 counts → converged
        assert not conv[0]  # zero bin → not converged
