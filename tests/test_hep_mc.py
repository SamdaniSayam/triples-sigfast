"""
tests/test_hep_mc.py
--------------------
Tests for RAMBO Monte Carlo phase space generator.
"""

import math

import numpy as np
import pytest

from triples_sigfast.hep.mc import rambo


def test_rambo_massless_two_body():
    n_events = 100
    e_cm = 100.0
    masses = [0.0, 0.0]

    p1, p2 = rambo(n_events, e_cm, masses)

    assert p1.shape == (100, 4)
    assert p2.shape == (100, 4)

    for i in range(n_events):
        # Energy sum
        assert abs((p1[i, 0] + p2[i, 0]) - e_cm) < 1e-9

        # Momentum sum (should be 0 in CM frame)
        assert abs(p1[i, 1] + p2[i, 1]) < 1e-9
        assert abs(p1[i, 2] + p2[i, 2]) < 1e-9
        assert abs(p1[i, 3] + p2[i, 3]) < 1e-9

        # Massless check
        m1_sq = p1[i, 0] ** 2 - p1[i, 1] ** 2 - p1[i, 2] ** 2 - p1[i, 3] ** 2
        m2_sq = p2[i, 0] ** 2 - p2[i, 1] ** 2 - p2[i, 2] ** 2 - p2[i, 3] ** 2
        assert abs(m1_sq) < 1e-6
        assert abs(m2_sq) < 1e-6


def test_rambo_massive_three_body():
    n_events = 50
    e_cm = 250.0
    masses = [10.0, 20.0, 30.0]

    p1, p2, p3 = rambo(n_events, e_cm, masses)

    for i in range(n_events):
        # Energy sum
        assert abs((p1[i, 0] + p2[i, 0] + p3[i, 0]) - e_cm) < 1e-9

        # Momentum sum
        assert abs(p1[i, 1] + p2[i, 1] + p3[i, 1]) < 1e-9
        assert abs(p1[i, 2] + p2[i, 2] + p3[i, 2]) < 1e-9
        assert abs(p1[i, 3] + p2[i, 3] + p3[i, 3]) < 1e-9

        # Mass check
        m1_sq = p1[i, 0] ** 2 - p1[i, 1] ** 2 - p1[i, 2] ** 2 - p1[i, 3] ** 2
        assert abs(math.sqrt(abs(m1_sq)) - 10.0) < 1e-6
        m2_sq = p2[i, 0] ** 2 - p2[i, 1] ** 2 - p2[i, 2] ** 2 - p2[i, 3] ** 2
        assert abs(math.sqrt(abs(m2_sq)) - 20.0) < 1e-6
        m3_sq = p3[i, 0] ** 2 - p3[i, 1] ** 2 - p3[i, 2] ** 2 - p3[i, 3] ** 2
        assert abs(math.sqrt(abs(m3_sq)) - 30.0) < 1e-6


def test_rambo_forbidden_decay():
    """Test where sum of masses is greater than center of mass energy."""
    n_events = 10
    e_cm = 50.0
    masses = [30.0, 30.0]  # Sum = 60 > 50

    p1, p2 = rambo(n_events, e_cm, masses)

    assert np.all(p1 == 0.0)
    assert np.all(p2 == 0.0)


def test_rambo_vectorized_large():
    n_events = 10000
    e_cm = 1000.0
    masses = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)

    ps = rambo(n_events, e_cm, masses)
    assert len(ps) == 4
    for p in ps:
        assert p.shape == (n_events, 4)
