# ============================================================
#  triples-sigfast — Full Pytest Test Suite (90%+ coverage)
# ============================================================

import numpy as np
import pandas as pd
import pytest

from triples_sigfast.core import (
    detect_anomalies,
    ema,
    ema_crossover_strategy,
    rolling_average,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def sample_data():
    """Standard 1000-point time series."""
    np.random.seed(42)
    return np.random.randn(1000).cumsum()


@pytest.fixture
def flat_data():
    """Flat line — edge case for std=0."""
    return np.ones(100)


@pytest.fixture
def pandas_series():
    """Pandas Series input."""
    np.random.seed(42)
    return pd.Series(np.random.randn(500).cumsum())


@pytest.fixture
def list_data():
    """Plain Python list input."""
    return [float(i) for i in range(1, 101)]


# ── rolling_average ──────────────────────────────────────────


class TestRollingAverage:
    def test_output_length(self, sample_data):
        window = 10
        result = rolling_average(sample_data, window)
        assert len(result) == len(sample_data) - window + 1

    def test_known_values(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_average(data, 3)
        assert np.isclose(result[0], 2.0)
        assert np.isclose(result[1], 3.0)
        assert np.isclose(result[2], 4.0)

    def test_window_size_one(self, sample_data):
        result = rolling_average(sample_data, 1)
        assert len(result) == len(sample_data)
        assert np.allclose(result, sample_data)

    def test_window_equals_data_length(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_average(data, 5)
        assert len(result) == 1
        assert np.isclose(result[0], 3.0)

    def test_invalid_window_zero(self, sample_data):
        with pytest.raises(ValueError):
            rolling_average(sample_data, 0)

    def test_invalid_window_negative(self, sample_data):
        with pytest.raises(ValueError):
            rolling_average(sample_data, -1)

    def test_invalid_window_too_large(self, sample_data):
        with pytest.raises(ValueError):
            rolling_average(sample_data, len(sample_data) + 1)

    def test_accepts_list_input(self, list_data):
        result = rolling_average(list_data, 2)
        assert len(result) == len(list_data) - 1

    def test_accepts_pandas_series(self, pandas_series):
        result = rolling_average(pandas_series, 10)
        assert len(result) == len(pandas_series) - 9

    def test_output_is_numpy_array(self, sample_data):
        result = rolling_average(sample_data, 10)
        assert isinstance(result, np.ndarray)

    def test_values_are_float64(self, sample_data):
        result = rolling_average(sample_data, 10)
        assert result.dtype == np.float64

    def test_monotonic_input(self):
        data = np.arange(1.0, 101.0)
        result = rolling_average(data, 10)
        # Rolling average of arithmetic sequence should also be arithmetic
        assert np.isclose(result[0], 5.5)
        assert np.isclose(result[-1], 95.5)


# ── ema ──────────────────────────────────────────────────────


class TestEma:
    def test_output_length(self, sample_data):
        result = ema(sample_data, 10)
        assert len(result) == len(sample_data)

    def test_invalid_span_zero(self, sample_data):
        with pytest.raises(ValueError):
            ema(sample_data, 0)

    def test_invalid_span_negative(self, sample_data):
        with pytest.raises(ValueError):
            ema(sample_data, -5)

    def test_accepts_list_input(self, list_data):
        result = ema(list_data, 3)
        assert len(result) == len(list_data)

    def test_accepts_pandas_series(self, pandas_series):
        result = ema(pandas_series, 10)
        assert len(result) == len(pandas_series)

    def test_output_is_numpy_array(self, sample_data):
        result = ema(sample_data, 10)
        assert isinstance(result, np.ndarray)

    def test_values_are_float64(self, sample_data):
        result = ema(sample_data, 10)
        assert result.dtype == np.float64

    def test_span_one_equals_input(self):
        # EMA with span=1 → alpha=1 → output equals input
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(data, 1)
        assert np.allclose(result, data)

    def test_large_span_smooths_data(self, sample_data):
        # Larger span = more smoothing = less variance
        result_small = ema(sample_data, 5)
        result_large = ema(sample_data, 50)
        assert result_large.std() < result_small.std()

    def test_single_element(self):
        data = np.array([42.0])
        result = ema(data, 1)
        assert np.isclose(result[0], 42.0)


# ── detect_anomalies ─────────────────────────────────────────


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

    def test_spike_is_detected(self):
        data = np.ones(200)
        data[100] = 1000.0
        result = detect_anomalies(data)
        assert result[100]

    def test_normal_data_mostly_clean(self):
        np.random.seed(0)
        data = np.random.randn(10000)
        result = detect_anomalies(data, threshold=3.0)
        # At threshold=3, expect < 1% flagged on normal distribution
        assert result.mean() < 0.01

    def test_strict_threshold_flags_more(self, sample_data):
        loose = detect_anomalies(sample_data, threshold=3.0)
        strict = detect_anomalies(sample_data, threshold=1.0)
        assert strict.sum() >= loose.sum()

    def test_accepts_list_input(self, list_data):
        result = detect_anomalies(list_data)
        assert len(result) == len(list_data)

    def test_accepts_pandas_series(self, pandas_series):
        result = detect_anomalies(pandas_series)
        assert len(result) == len(pandas_series)

    def test_multiple_spikes_detected(self):
        data = np.ones(500)
        data[100] = 999.0
        data[300] = -999.0
        result = detect_anomalies(data)
        assert result[100]
        assert result[300]


# ── ema_crossover_strategy ───────────────────────────────────


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

    def test_fast_ema_more_reactive(self, sample_data):
        fast, slow, _ = ema_crossover_strategy(sample_data, 5, 50)
        assert fast.std() > slow.std()

    def test_accepts_list_input(self, list_data):
        fast, slow, signals = ema_crossover_strategy(list_data, 3, 10)
        assert len(signals) == len(list_data)

    def test_accepts_pandas_series(self, pandas_series):
        fast, slow, signals = ema_crossover_strategy(pandas_series, 5, 20)
        assert len(signals) == len(pandas_series)

    def test_buy_and_sell_signals_exist(self, sample_data):
        _, _, signals = ema_crossover_strategy(sample_data, 5, 20)
        # On 1000 points there must be at least some signals
        assert (signals == 1).any() or (signals == -1).any()

    def test_output_arrays_are_numpy(self, sample_data):
        fast, slow, signals = ema_crossover_strategy(sample_data, 5, 20)
        assert isinstance(fast, np.ndarray)
        assert isinstance(slow, np.ndarray)
        assert isinstance(signals, np.ndarray)


# ── __init__.py public API ───────────────────────────────────


class TestPublicAPI:
    def test_rolling_average_importable(self):
        from triples_sigfast import rolling_average

        assert callable(rolling_average)

    def test_ema_importable(self):
        from triples_sigfast import ema

        assert callable(ema)

    def test_detect_anomalies_importable(self):
        from triples_sigfast import detect_anomalies

        assert callable(detect_anomalies)

    def test_ema_crossover_strategy_importable(self):
        from triples_sigfast import ema_crossover_strategy

        assert callable(ema_crossover_strategy)

    def test_version_exists(self):
        import triples_sigfast

        assert hasattr(triples_sigfast, "__version__")
        assert isinstance(triples_sigfast.__version__, str)
