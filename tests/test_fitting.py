import numpy as np
import pytest

from triples_sigfast.stats.fitting import (
    build_nll_cost_function,
    crystal_ball_pdf,
    pseudo_voigt_pdf,
    voigtian_pdf,
)


def test_crystal_ball_pdf():
    # alpha, n, mu, sigma
    params = np.array([1.0, 1.0, 0.0, 1.0])

    # Gaussian part
    val = crystal_ball_pdf(0.0, params)
    assert np.isclose(val, 1.0)

    # Tail part
    val_tail = crystal_ball_pdf(-2.0, params)
    assert val_tail > 0.0


def test_pseudo_voigt_pdf():
    # mu, sigma, gamma, eta
    params = np.array([0.0, 1.0, 1.0, 0.5])
    val = pseudo_voigt_pdf(0.0, params)
    assert val > 0.0


def test_voigtian_pdf():
    params = np.array([0.0, 1.0, 1.0, 0.5])
    assert voigtian_pdf(0.0, params) == pseudo_voigt_pdf(0.0, params)


def test_build_nll_cost_function():
    data = np.array([0.0, 0.1, -0.1, 0.2, -0.2])
    nll_func = build_nll_cost_function(crystal_ball_pdf, data)

    params = np.array([1.0, 1.0, 0.0, 1.0])
    cost = nll_func(params)
    assert cost > 0

    # Bad params (sigma <= 0) should yield heavy penalty
    bad_params = np.array([1.0, 1.0, 0.0, -1.0])
    bad_cost = nll_func(bad_params)
    assert bad_cost >= 1e10
