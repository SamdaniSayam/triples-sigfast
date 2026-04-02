import numpy as np
import pandas as pd
from numba import njit, prange


# --- 1. MEMORY MANAGEMENT (Internal) ---
def _ensure_float64_numpy(data):
    """
    Forces input data into a C-contiguous, 64-bit float NumPy array
    for maximum CPU cache efficiency and memory alignment.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        arr = data.to_numpy().flatten()
    elif isinstance(data, list):
        arr = np.array(data)
    else:
        arr = data
    return np.ascontiguousarray(arr, dtype=np.float64)


# --- 2. ROLLING AVERAGE (HPC Optimized) ---
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _numba_rolling_avg(data: np.ndarray, window_size: int):  # pragma: no cover
    n = len(data)
    result = np.empty(n - window_size + 1, dtype=np.float64)
    for i in prange(n - window_size + 1):
        window_sum = 0.0
        for j in range(window_size):
            window_sum += data[i + j]
        result[i] = window_sum / window_size
    return result


def rolling_average(data, window_size: int):
    """
    Computes a moving average on a 1D array at C-speed using Numba JIT.

    This function bypasses the Python GIL via multithreading and is optimized
    for cache-locality on large datasets (100M+ rows).

    Args:
        data (np.ndarray | pd.Series | list): The input time-series data.
        window_size (int): The number of samples for the moving window.

    Returns:
        np.ndarray or pd.Series: The smoothed data array.
    """
    if window_size <= 0:
        raise ValueError("Window size must be > 0.")
    if len(data) < window_size:
        raise ValueError("Data length must be >= window size.")

    clean_data = _ensure_float64_numpy(data)
    result = _numba_rolling_avg(clean_data, window_size)

    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index[window_size - 1 :])
    return result


# --- 3. EXPONENTIAL MOVING AVERAGE (HPC Optimized) ---
@njit(fastmath=True, cache=True, nogil=True)
def _numba_ema(data: np.ndarray, alpha: float):  # pragma: no cover
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
    return result


def ema(data, span: int):
    """
    Calculates the Exponential Moving Average (EMA) at C-speed.

    Args:
        data (np.ndarray | pd.Series | list): The input time-series data.
        span (int): The span of the EMA, common in financial analysis.

    Returns:
        np.ndarray or pd.Series: The EMA data array.
    """
    if span <= 0:
        raise ValueError("Span must be > 0")
    clean_data = _ensure_float64_numpy(data)
    alpha = 2.0 / (span + 1.0)
    result = _numba_ema(clean_data, alpha)

    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index)
    return result


# --- 4. Z-SCORE ANOMALY DETECTION (HPC Optimized) ---
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _numba_zscore_anomalies(data: np.ndarray, threshold: float):  # pragma: no cover
    n = len(data)
    mean_val = np.mean(data)
    std_val = np.std(data)

    is_anomaly = np.zeros(n, dtype=np.bool_)
    if std_val == 0:
        return is_anomaly

    for i in prange(n):
        z_score = abs(data[i] - mean_val) / std_val
        if z_score > threshold:
            is_anomaly[i] = True

    return is_anomaly


def detect_anomalies(data, threshold: float = 3.0):
    """
    Identifies anomalies in a dataset using the Z-score method at C-speed.

    Args:
        data (np.ndarray | pd.Series | list): The input time-series data.
        threshold (float): The Z-score threshold to classify a point as anomalous. Default is 3.0.

    Returns:
        np.ndarray or pd.Series: A boolean array where True indicates an anomaly.
    """
    clean_data = _ensure_float64_numpy(data)
    result = _numba_zscore_anomalies(clean_data, threshold)
    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index)
    return result


# --- 5. QUANT TRADING: EMA CROSSOVER (HPC Optimized) ---
@njit(fastmath=True, cache=True, nogil=True)
def _numba_crossover(fast_ema: np.ndarray, slow_ema: np.ndarray):  # pragma: no cover
    n = len(fast_ema)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if fast_ema[i] > slow_ema[i] and fast_ema[i - 1] <= slow_ema[i - 1]:
            signals[i] = 1  # BUY
        elif fast_ema[i] < slow_ema[i] and fast_ema[i - 1] >= slow_ema[i - 1]:
            signals[i] = -1  # SELL
    return signals


def ema_crossover_strategy(data, fast_span: int = 9, slow_span: int = 21):
    """
    Generates trading signals based on the EMA crossover strategy at C-speed.

    Args:
        data (np.ndarray | pd.Series | list): Input asset price data.
        fast_span (int): The span for the shorter-term EMA.
        slow_span (int): The span for the longer-term EMA.

    Returns:
        tuple: (fast_ema, slow_ema, signals) where signals is an array of 1 (Buy), -1 (Sell), or 0 (Hold).
    """
    clean_data = _ensure_float64_numpy(data)
    fast_ema = _numba_ema(clean_data, 2.0 / (fast_span + 1.0))
    slow_ema = _numba_ema(clean_data, 2.0 / (slow_span + 1.0))
    signals = _numba_crossover(fast_ema, slow_ema)
    return fast_ema, slow_ema, signals
