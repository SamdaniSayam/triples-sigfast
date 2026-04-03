# ⚡ triples-sigfast

[![CI](https://github.com/SamdaniSayam/triples-sigfast/actions/workflows/ci.yml/badge.svg)](https://github.com/SamdaniSayam/triples-sigfast/actions)
[![PyPI](https://img.shields.io/pypi/v/triples-sigfast)](https://pypi.org/project/triples-sigfast/)
[![Python](https://img.shields.io/pypi/pyversions/triples-sigfast)](https://pypi.org/project/triples-sigfast/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/triples-sigfast)](https://pypi.org/project/triples-sigfast/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/SamdaniSayam/triples-sigfast)

**A high-performance, Numba JIT-compiled data analysis engine for simulation-based physics research — built for scientists, by a scientist.**

triples-sigfast bridges the gap between simulation codes (Geant4, FLUKA, MCNP, SERPENT) and publication-ready results — with C-level performance via Numba JIT compilation, zero C code required, and an API accessible to complete beginners.

---

## Why triples-sigfast?

There is currently no Python library that simultaneously offers:

- **High-performance JIT-compiled signal processing** for large Monte Carlo datasets (100M+ rows)
- **Built-in nuclear physics domain knowledge** — ICRP 74 standards, NIST XCOM databases, ANSI/ANS-6.4.3 buildup factors
- **Native readers for all major simulation code output formats** — Geant4 ROOT, FLUKA, MCNP, SERPENT
- **Monte Carlo statistical analysis** — relative error, figure of merit, convergence checking, uncertainty propagation
- **A beginner-friendly API** designed for researchers new to programming

triples-sigfast fills this gap.

---

## Installation

```bash
pip install triples-sigfast
```

**Requirements:** Python >= 3.10, NumPy, Numba, Pandas, uproot, awkward, matplotlib

---

## Performance

Tested on real datasets using the JIT-compiled engine:

| Dataset Size | RAM | Execution Time | Peak RAM |
|---|---|---|---|
| 1,000,000 rows | 8 MB | 0.339s | 192 MB |
| 10,000,000 rows | 80 MB | 0.281s | 404 MB |
| 50,000,000 rows | 400 MB | 0.747s | 940 MB |
| 100,000,000 rows | 800 MB | 1.225s | 1,596 MB |

Survived the 100M row Crucible Test — processing 800 MB of data in just over 1 second.

---

## Features

### Signal Processing (`triples_sigfast.core`)

```python
import numpy as np
from triples_sigfast import rolling_average, ema, detect_anomalies, ema_crossover_strategy
from triples_sigfast import savitzky_golay, find_peaks, flux_to_dose, attenuation

# JIT-compiled rolling average — 100M+ rows
data = np.random.randn(1_000_000)
result = rolling_average(data, window_size=50)

# Exponential moving average at C-speed
prices = np.random.randn(1_000_000).cumsum() + 100
smoothed = ema(prices, span=20)

# Z-score anomaly detection
data[500_000] = 999.0
anomalies = detect_anomalies(data, threshold=3.0)

# Savitzky-Golay spectrum smoother
counts = np.random.poisson(lam=500, size=1000).astype(float)
smooth = savitzky_golay(counts, window=11, polyorder=3)

# Gamma ray peak detection
peaks = find_peaks(smooth, min_height=50, min_distance=10)

# ICRP 74 flux-to-dose conversion
dose = flux_to_dose(flux=1e6, energy_mev=2.35, particle="neutron")

# Beer-Lambert shielding (9 materials, NIST data)
T = attenuation(thickness_cm=10, material="lead")
```

---

### Monte Carlo Statistics (`triples_sigfast.stats.mc`)

```python
from triples_sigfast.stats.mc import (
    relative_error,
    figure_of_merit,
    is_converged,
    propagate_error,
)
import numpy as np

counts = np.array([10_000.0, 5_000.0, 1_000.0, 250.0])

# Relative error R = 1/sqrt(N) per bin (MCNP standard)
R = relative_error(counts)

# Figure of Merit: FOM = 1 / (R^2 * T)
fom = figure_of_merit(R, cpu_time=3600.0)

# Convergence check (MCNP standard: R < 0.05)
converged = is_converged(counts, threshold=0.05)
print(f"{converged.sum()}/{len(counts)} bins converged")

# GUM uncertainty propagation through detector efficiency
sigma = propagate_error(counts, efficiency=0.35)  # HPGe detector
```

---

### Simulation File Readers (`triples_sigfast.io`)

One unified API regardless of simulation code:

```python
from triples_sigfast.io import SimReader

# Geant4 ROOT output
reader = SimReader("simulation.root")

# FLUKA output
reader = SimReader("output.flair")

# MCNP MCTAL file
reader = SimReader("output.mctal")

# SERPENT detector file
reader = SimReader("output.det")

# Same API for all formats
spectrum, energies = reader.get_spectrum()
tally = reader.get_tally("neutron_flux")
reader.summary()
```

#### Geant4 ROOT Reader

```python
from triples_sigfast.io import RootReader

reader = RootReader("simulation.root")
reader.summary()                              # auto-lists all histograms
counts, energies = reader.get_spectrum("neutron")
reader.export_csv("results.csv")
reader.export_hdf5("results.h5")
```

#### FLUKA Reader

```python
from triples_sigfast.io import FlukaReader

reader = FlukaReader("output.flair")
reader.summary()
usrbin  = reader.get_usrbin("neutron_fluence")
usrbdx  = reader.get_usrbdx("boundary_crossing")
values, bins = reader.get_spectrum("gamma_dose")
```

#### MCNP Reader

```python
from triples_sigfast.io import MCNPReader

reader = MCNPReader("mctal_file")
reader.summary()
tally  = reader.get_tally(4)           # by tally number
tally  = reader.get_tally("tally_4")   # by key string
fom    = reader.get_fom()              # figure of merit from TFC
values, bins = reader.get_spectrum()
```

#### SERPENT Reader

```python
from triples_sigfast.io import SerpentReader

reader = SerpentReader("serpent_det.m")
reader.summary()
flux, energies = reader.get_detector("neutron_flux")
keff    = reader.get_keff()
burnup  = reader.get_burnup()
print(f"k-eff = {keff['ana_keff']:.5f} ± {keff['ana_err']:.5f}")
```

---

### Nuclear Physics (`triples_sigfast.nuclear`)

#### Radiation Shielding

```python
from triples_sigfast.nuclear.shielding import (
    attenuation_with_buildup,
    attenuation_series,
    available_materials,
)
import numpy as np

# GP buildup-corrected attenuation (ANSI/ANS-6.4.3)
T = attenuation_with_buildup(
    thickness_cm=10.0,
    material="lead",
    energy_mev=1.25,
    geometry="point_source",
)
print(f"Transmission: {T:.4f} ({T*100:.2f}%)")

# Transmission curve across thickness range
thicknesses = np.linspace(0, 30, 100)
T_curve = attenuation_series(thicknesses, "concrete", 1.25)

# Available materials: lead, iron, concrete, water, polyethylene, aluminum
print(available_materials())
```

#### Neutron Source Spectra

```python
from triples_sigfast.nuclear.sources import watt_spectrum, maxwell_spectrum

energies = np.linspace(0.01, 15, 1000)  # MeV

# Watt fission spectrum (Cf-252, U-235, Pu-239, ...)
flux_cf252 = watt_spectrum(energies, source="Cf-252")
flux_u235  = watt_spectrum(energies, source="U-235")

# Maxwell-Boltzmann thermal spectrum
flux_thermal = maxwell_spectrum(energies, temperature_mev=0.0000253)
```

#### Isotope Database

```python
from triples_sigfast.nuclear.isotope import Isotope

cf252 = Isotope("Cf-252")
print(f"Half-life:      {cf252.half_life:.3f} years")
print(f"Activity (1g):  {cf252.activity(mass_g=1.0):.3e} Bq")
print(f"Neutron yield:  {cf252.neutron_yield:.3e} n/s/g")
print(f"Decay mode:     {cf252.decay_mode}")

b10 = Isotope("B-10")
print(f"Thermal xs:     {b10.thermal_cross_section:.0f} barns")
print(f"Res. integral:  {b10.resonance_integral:.0f} barns")
```

#### Dose Calculations

```python
from triples_sigfast.nuclear.dose import (
    point_source,
    point_source_shielded,
    dose_rate_vs_distance,
    inverse_square_distance,
)

# Unshielded point source dose rate (ICRP 74)
rate = point_source(
    activity_bq=1e9,
    energy_mev=1.25,
    distance_cm=100,
    particle="gamma",
    photons_per_decay=2.0,   # Co-60: 2 gammas per decay
)
print(f"Dose rate: {rate:.2f} uSv/hr")

# Shielded dose rate (GP buildup-corrected)
rate_shielded = point_source_shielded(
    activity_bq=1e9,
    energy_mev=1.25,
    distance_cm=100,
    shield_material="lead",
    shield_thickness_cm=5.0,
)

# Dose profile vs distance
distances = np.linspace(10, 500, 200)
rates = dose_rate_vs_distance(1e9, 1.25, distances)

# Minimum safe distance for dose limit
d_safe = inverse_square_distance(1e9, 1.25, target_dose_usvhr=1.0)
print(f"Safe distance: {d_safe/100:.1f} m")
```

---

## Complete Workflow Example

A full nuclear shielding analysis pipeline from simulation output to results:

```python
import numpy as np
from triples_sigfast.io import RootReader
from triples_sigfast.stats.mc import relative_error, is_converged, propagate_error
from triples_sigfast.nuclear.isotope import Isotope
from triples_sigfast.nuclear.shielding import attenuation_with_buildup
from triples_sigfast.nuclear.dose import point_source_shielded
from triples_sigfast import savitzky_golay, find_peaks

# 1. Load Geant4 simulation output
reader = RootReader("shielding_simulation.root")
reader.summary()
counts, energies = reader.get_spectrum("neutron_spectrum")

# 2. Check Monte Carlo convergence
R = relative_error(counts)
converged = is_converged(counts, threshold=0.05)
print(f"Converged bins: {converged.sum()}/{len(counts)}")

# 3. Propagate uncertainty through HPGe detector efficiency
sigma = propagate_error(counts, efficiency=0.35)

# 4. Smooth spectrum and find peaks
smoothed = savitzky_golay(counts, window=11, polyorder=3)
peaks = find_peaks(smoothed, min_height=50, min_distance=10)

# 5. Compute shielded dose rate for Cf-252 source
cf = Isotope("Cf-252")
n_rate = cf.neutron_source_rate(mass_g=1e-3)

dose = point_source_shielded(
    activity_bq=n_rate,
    energy_mev=2.0,
    distance_cm=100,
    shield_material="polyethylene",
    shield_thickness_cm=10.0,
    particle="neutron",
)
print(f"Shielded dose rate: {dose:.4f} uSv/hr")
```

---

## Running Tests

```bash
pip install triples-sigfast[dev]
pytest --cov=triples_sigfast
```

**Test statistics (v1.4.0):** 323 tests, 100% coverage, tested on Ubuntu / macOS / Windows across Python 3.10, 3.11, 3.12.

---

## Architecture

```
triples_sigfast/
├── core/           # JIT-compiled signal processing (Numba)
├── stats/          # Monte Carlo statistics
├── io/             # Simulation file readers
│   ├── root_reader.py   # Geant4 ROOT (uproot)
│   ├── fluka.py         # FLUKA USRBIN/USRBDX/USRTRACK
│   ├── mcnp.py          # MCNP6 MCTAL format
│   ├── serpent.py       # SERPENT2 detector output
│   └── sim_reader.py    # Universal auto-detecting reader
├── nuclear/        # Nuclear physics
│   ├── shielding.py     # GP buildup factors (ANSI/ANS-6.4.3)
│   ├── sources.py       # Watt / Maxwell fission spectra
│   ├── isotope.py       # Isotope database (NUBASE2020)
│   └── dose.py          # ICRP 74 dose calculations
├── viz/            # Publication-quality plots (coming v1.5.0)
├── cli/            # Command-line interface (coming v1.6.0)
├── detectors/      # Detector physics (planned)
└── plasma/         # Plasma physics (planned)
```

---

## Roadmap

| Version | Status | Milestone |
|---|---|---|
| v1.1.1 | ✅ Released | Core signal processing, 86 tests |
| v1.2.0 | ✅ Released | Monte Carlo statistics, ROOT reader, SimReader |
| v1.3.0 | ✅ Released | Nuclear physics — buildup factors, Watt spectrum, isotope database, ICRP 74 dose |
| v1.4.0 | ✅ Released | Native FLUKA, MCNP, SERPENT readers — 323 tests, 100% coverage |
| v1.5.0 | 🔄 In progress | Visualization engine — publication-quality plots, LaTeX/PDF/SVG export |
| v1.6.0 | ⬜ Planned | CLI, auto-report generator, guided beginner mode |
| v2.0.0 | ⬜ Planned | Community launch, academic paper submission |

---

## Origin

> "I needed to analyze Geant4 ROOT files for my nuclear shielding research and nothing made it easy. So I built triples-sigfast."

The primary audience is physics researchers who are beginners in programming — graduate students, junior researchers, and academics who run simulations but struggle with data analysis pipelines. The library must be powerful enough for experts and simple enough for beginners.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**TripleS Studio**

- PyPI: [triples-sigfast](https://pypi.org/project/triples-sigfast/)
- GitHub: [SamdaniSayam/triples-sigfast](https://github.com/SamdaniSayam/triples-sigfast)