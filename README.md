#  SigFast

![PyPI](https://img.shields.io/badge/PyPI-v0.3.1-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A high-performance time-series processing library built for Data Scientists and Physicists. Uses **Numba JIT** and **C-level multithreading** to bypass the Python GIL.

### Why SigFast?
Pandas is great, but it runs on a single thread. When analyzing millions of data points (IoT sensors, high-frequency trading, astrophysics), Pandas becomes a bottleneck. SigFast distributes the math across all your CPU cores.

**Benchmark (10 Million Data Points - Rolling Window):**
*    Pandas `.rolling().mean()`: **~1.20 seconds**
*    SigFast Engine: **~0.03 seconds (40x Faster)**

### Installation
```bash
pip install sigfast
