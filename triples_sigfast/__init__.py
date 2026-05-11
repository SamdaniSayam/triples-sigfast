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
detectors -- Detector physics models (planned v2.0)
plasma    -- Plasma physics models (planned v2.0)
"""

# ---------------------------------------------------------------------------
# Package version string.
# Must be kept in sync with pyproject.toml and setup.py.
# ---------------------------------------------------------------------------
__version__ = "1.8.1"

# ---------------------------------------------------------------------------
# Core signal processing.
# Symbols are re-exported explicitly so that static analysers and IDEs can
# resolve them without inspecting submodule internals.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# High-energy physics sub-package.
# Imported as a namespace (not unpacked) to defer Numba JIT compilation until
# a caller actually accesses hep.* symbols.  This avoids JIT warmup cost for
# users who only need the nuclear physics side of the library.
# ---------------------------------------------------------------------------
from . import hep as hep

# ---------------------------------------------------------------------------
# PDF report generator -- available at the top level for convenience.
# ---------------------------------------------------------------------------
from .cli.report import AutoReport as AutoReport
from .core.signal import attenuation as attenuation
from .core.signal import attenuation_series as attenuation_series
from .core.signal import detect_anomalies as detect_anomalies
from .core.signal import ema as ema
from .core.signal import ema_crossover_strategy as ema_crossover_strategy
from .core.signal import find_peaks as find_peaks
from .core.signal import flux_to_dose as flux_to_dose
from .core.signal import rolling_average as rolling_average
from .core.signal import savitzky_golay as savitzky_golay
