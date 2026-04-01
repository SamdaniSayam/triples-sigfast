# ⚡ triples-sigfast

![CI](https://github.com/SamdaniSayam/triples-sigfast/actions/workflows/main.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/triples-sigfast)
![Python](https://img.shields.io/pypi/pyversions/triples-sigfast)
![License](https://img.shields.io/pypi/l/triples-sigfast)
![Downloads](https://img.shields.io/pypi/dm/triples-sigfast)

**An enterprise-grade, JIT-compiled time-series engine stress-tested on 100M+ row datasets.**

`triples-sigfast` uses [Numba](https://numba.pydata.org/) to compile Python functions to machine code at runtime, delivering C-level performance for time-series analysis — with zero C code required.

---

## Features

- **Rolling Average** — JIT-compiled sliding window mean
- **EMA** — Exponential Moving Average at C-speed
- **Anomaly Detection** — Z-score based outlier detection
- **EMA Crossover Strategy** — Trading signal generation (Buy/Sell/Hold)
- **Pandas + NumPy compatible** — Works with arrays, lists, and Series
- **Cross-platform** — Tested on Ubuntu, macOS, and Windows

---

## Installation
```bash
pip install triples-sigfast
```

---

## Benchmark

Tested on real datasets using JIT-compiled engine:

| Dataset Size | RAM Size | Execution Time | Peak RAM |
|---|---|---|---|
| 1,000,000 rows | 8 MB | 0.339s | 192 MB |
| 10,000,000 rows | 80 MB | 0.281s | 404 MB |
| 50,000,000 rows | 400 MB | 0.747s | 940 MB |
| 100,000,000 rows | 800 MB | **1.225s** | 1,596 MB |

> Survived the **100M row Crucible Test** — processing 800MB of data in just over 1 second.

## Usage

### Rolling Average
```python
import numpy as np
from triples_sigfast import rolling_average

data = np.random.randn(1_000_000)
result = rolling_average(data, window_size=50)
print(result[:5])
```

### Exponential Moving Average (EMA)
```python
from triples_sigfast import ema

prices = np.random.randn(1_000_000).cumsum() + 100
smoothed = ema(prices, span=20)
print(smoothed[:5])
```

### Anomaly Detection
```python
from triples_sigfast import detect_anomalies

data = np.random.randn(1_000_000)
data[500_000] = 999.0  # inject a spike

anomalies = detect_anomalies(data, threshold=3.0)
print(f"Anomalies found: {anomalies.sum()}")
```

### EMA Crossover Strategy (Trading Signals)
```python
from triples_sigfast import ema_crossover_strategy

prices = np.random.randn(1_000_000).cumsum() + 100
fast_ema, slow_ema, signals = ema_crossover_strategy(prices, fast_span=12, slow_span=26)

# signals: 1 = Buy, -1 = Sell, 0 = Hold
print(f"Buy signals:  {(signals == 1).sum()}")
print(f"Sell signals: {(signals == -1).sum()}")
```

---

## Running Tests
```bash
pip install triples-sigfast[dev]
pytest --cov=triples_sigfast
```

---

## Requirements

- Python >= 3.10
- NumPy >= 1.20.0
- Numba >= 0.55.0
- Pandas >= 1.3.0

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**TripleS Studio**
- PyPI: [triples-sigfast](https://pypi.org/project/triples-sigfast/)
- GitHub: [SamdaniSayam/triples-sigfast](https://github.com/SamdaniSayam/triples-sigfast)