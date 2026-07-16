"""
triples_sigfast
---------------
GIL-free physics simulation analysis engine.

This module re-exports the most commonly used public symbols so that callers
can write:

    from triples_sigfast import savitzky_golay, find_peaks, flux_to_dose

rather than importing from the full submodule path.

Sub-packages
------------
core      -- JIT-compiled signal processing kernels (Numba/NumPy)
stats     -- Monte Carlo convergence and statistics utilities
nuclear   -- Nuclear physics standards (ICRP-74, ANSI/ANS-6.4.3, NIST XCOM)
io        -- Simulation file readers (Geant4, FLUKA, MCNP, SERPENT, raw data)
viz       -- Publication-quality spectrum and shielding plots
cli       -- Command-line interface (sigfast analyze, compare, dose, shield ...)
hep       -- High-energy physics sub-package (LHE/HepMC3 I/O, jet clustering)
detectors -- Detector physics models (NaI, HPGe, He-3, BF3 response)
plasma    -- Plasma physics models (fusion neutronics and material activation)
"""

# ---------------------------------------------------------------------------
# Package version string.
# Uses importlib.metadata to stay in sync with pyproject.toml automatically.
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("triples-sigfast")
except Exception:
    __version__ = "2.1.0"

# ---------------------------------------------------------------------------
# Core signal processing.
# Symbols are re-exported explicitly so that static analysers and IDEs can
# resolve them without inspecting submodule internals.
# ---------------------------------------------------------------------------
from .core.signal import attenuation as attenuation
from .core.signal import attenuation_series as attenuation_series
from .core.signal import detect_anomalies as detect_anomalies
from .core.signal import ema as ema
from .core.signal import ema_crossover_strategy as ema_crossover_strategy
from .core.signal import find_peaks as find_peaks
from .core.signal import flux_to_dose as flux_to_dose
from .core.signal import rolling_average as rolling_average
from .core.signal import savitzky_golay as savitzky_golay

# ---------------------------------------------------------------------------
# Lazy imports for heavy sub-packages.
# The hep sub-package and AutoReport are loaded on first access to avoid
# triggering Numba JIT compilation and reportlab loading at import time.
# ---------------------------------------------------------------------------
__all__ = [
    # Core signal processing
    "rolling_average",
    "ema",
    "ema_crossover_strategy",
    "detect_anomalies",
    "savitzky_golay",
    "find_peaks",
    "flux_to_dose",
    "attenuation",
    "attenuation_series",
    # Lazy-loaded
    "hep",
    "AutoReport",
    # Version
    "__version__",
]


def __getattr__(name: str):
    """Lazy-load heavy sub-packages on first access."""
    if name == "hep":
        from . import hep as _hep

        globals()["hep"] = _hep
        return _hep
    if name == "AutoReport":
        from .cli.report import AutoReport as _AutoReport

        globals()["AutoReport"] = _AutoReport
        return _AutoReport
    raise AttributeError(f"module 'triples_sigfast' has no attribute {name!r}")
