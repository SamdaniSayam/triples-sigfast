# ============================================================
#  triples-sigfast — Pytest Test Suite
# ============================================================

# 1. FIX: Add the missing imports
import numpy as np
import pytest

# 2. FIX: Tell Python how to find your library from this 'tests' subfolder
# This goes up one level ('..') and finds 'triples_sigfast'
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from triples_sigfast.core import (
    rolling_average,
    ema,
    detect_anomalies,
    ema_crossover_strategy,
)


# --- Fixtures ---

@pytest.fixture
def sample_data():
    """Standard 1000-point time series for all tests."""
    np.random.seed(42)
    return np.random.randn(1000).cumsum()


@pytest.fixture
def flat_data():
    """Flat line — edge case for std=0."""
    return np.ones(100)


# --- Test Suites ---

class TestRollingAverage:
    def test_output_length(self, sample_data):
        result = rolling_average(sample_data, 10)
        # 3. FIX: Rolling average output is shorter than input
        assert len(result) == len(sample_data) - 10 + 1

    def test_known_values(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_average(data, 3)
        assert len(result) == 3
        assert np.isclose(result[0], 2.0)
        assert np.isclose(result[1], 3.0)
        assert np.isclose(result[2], 4.0)

    def test_invalid_window_zero(self, sample_data):
        with pytest.raises(ValueError):
            rolling_average(sample_data, 0)

    def test_invalid_window_too_large(self, sample_data):
        with pytest.raises(ValueError):
            rolling_average(sample_data, len(sample_data) + 1)

    def test_accepts_list_input(self):
        result = rolling_average([1.0, 2.0, 3.0, 4.0, 5.0], 2)
        assert len(result) == 4

class TestEma:
    def test_output_length(self, sample_data):
        result = ema(sample_data, 10)
        assert len(result) == len(sample_data)

    def test_invalid_span_zero(self, sample_data):
        with pytest.raises(ValueError):
            ema(sample_data, 0)

class TestDetectAnomalies:
    def test_output_length(self, sample_data):
        result = detect_anomalies(sample_data)
        assert len(result) == len(sample_data)

    def test_output_is_boolean(self, sample_data):
        result = detect_anomalies(sample_data)
        assert result.dtype == np.bool_

    def test_flat_data_no_anomalies(self, flat_data):
        result = detect_anomalies(flat_data)
        assert not result.any()

    def test_spike_detected(self):
        data = np.ones(100)
        data[50] = 1000.0
        result = detect_anomalies(data)
        assert result[50]

class TestEmaCrossoverStrategy:
    def test_output_structure(self, sample_data):
        fast_ema, slow_ema, signals = ema_crossover_strategy(sample_data, 5, 20)
        assert len(fast_ema) == len(sample_data)
        assert len(slow_ema) == len(sample_data)
        assert len(signals) == len(sample_data)

    def test_signals_only_valid_values(self, sample_data):
        _, _, signals = ema_crossover_strategy(sample_data, 5, 20)
        unique = set(np.unique(signals))
        assert unique.issubset({-1, 0, 1})