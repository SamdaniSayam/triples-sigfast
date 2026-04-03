# ── Core signal processing ───────────────────────────────────
from .core.signal import attenuation as attenuation
from .core.signal import attenuation_series as attenuation_series
from .core.signal import detect_anomalies as detect_anomalies
from .core.signal import ema as ema
from .core.signal import ema_crossover_strategy as ema_crossover_strategy
from .core.signal import find_peaks as find_peaks
from .core.signal import flux_to_dose as flux_to_dose
from .core.signal import rolling_average as rolling_average
from .core.signal import savitzky_golay as savitzky_golay

__version__ = "1.4.0"
