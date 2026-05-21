# ⚡ triples-sigfast

[![CI](https://github.com/SamdaniSayam/triples-sigfast/actions/workflows/ci.yml/badge.svg)](https://github.com/SamdaniSayam/triples-sigfast/actions)
[![PyPI](https://img.shields.io/pypi/v/triples-sigfast)](https://pypi.org/project/triples-sigfast/)
[![Python](https://img.shields.io/pypi/pyversions/triples-sigfast)](https://pypi.org/project/triples-sigfast/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/triples-sigfast)](https://pypi.org/project/triples-sigfast/)
[![Coverage](https://img.shields.io/badge/coverage-96.78%25-brightgreen)](https://github.com/SamdaniSayam/triples-sigfast)

**An enterprise-grade, Numba JIT-compiled data analysis engine for time-series, nuclear physics, and high-energy physics research — built for scientists, by scientists.**

`triples-sigfast` bridges the gap between raw Monte Carlo simulation outputs (Geant4, FLUKA, MCNP, SERPENT) and publication-ready scientific results. By utilizing Numba's Just-In-Time (JIT) compiler, the library achieves native C-level execution speeds with GIL-free parallel execution across all CPU cores, without requiring researchers to write a single line of C or C++.

---

## Key Features

- **⚡ GIL-Free Parallel Execution**: Native performance scaling on massive datasets (100M+ particle tracks) via parallelized JIT compilation.
- **☢️ Standard Nuclear Physics Models**: Out-of-the-box support for ICRP-74 dose conversion, ANSI/ANS-6.4.3 shielding buildup factors, NIST XCOM mass attenuation data, and NUBASE2020 isotope databases.
- **🛰️ Unified Simulation Readers**: A single, clean API (`SimReader`) that auto-detects and ingests simulation data from Geant4 (ROOT), FLUKA, MCNP, and SERPENT output files.
- **🌌 High-Energy Physics & Kinematics**: JIT-accelerated kinematics (invariant mass, rapidity, pseudorapidity, azimuthal separations) paired with an ergonomic, OOP-style `LorentzVector` class for single-particle and event-level calculations.
- **🔬 Detector & Plasma Physics**: Native models for scintillator (NaI) and semiconductor (HPGe, He-3, BF3) detectors alongside fusion plasma neutronics (Bosch-Hale D-T/D-D thermonuclear reactivities and Doppler-broadened source spectra).
- **📊 Publication-Ready Visualization**: Automated report generators and styling engines optimized for journal submissions (LaTeX-formatted labels, Physical Review/Nature style sheets, vector exports).

---

## Installation

```bash
pip install triples-sigfast
```

*Requirements: Python >= 3.10, NumPy, Numba, Pandas, uproot, awkward, matplotlib, click, rich, reportlab, plotext*

For development or test execution:
```bash
pip install triples-sigfast[dev]
```

---

## Performance Benchmark

The Numba JIT engine is stress-tested on multi-million row datasets to verify linear scaling and GIL-free throughput:

| Dataset Size | RAM | Execution Time (s) | Peak RAM |
|---|---|---|---|
| 1,000,000 rows | 8 MB | 0.339 s | 192 MB |
| 10,000,000 rows | 80 MB | 0.281 s | 404 MB |
| 50,000,000 rows | 400 MB | 0.747 s | 940 MB |
| 100,000,000 rows | 800 MB | 1.225 s | 1,596 MB |

*Tested on 8-core Intel Core i7 @ 2.3 GHz. Survival of the 100M row Crucible Test demonstrates sub-1.3 second processing time for 800 MB of simulation track data.*

---

## Quick-Start Guides & API References

### 1. Signal Processing (`triples_sigfast.core`)
```python
import numpy as np
from triples_sigfast import (
    rolling_average,
    ema,
    detect_anomalies,
    savitzky_golay,
    find_peaks,
    flux_to_dose,
    attenuation
)

# JIT-compiled rolling average over 1M points
data = np.random.randn(1_000_000)
result = rolling_average(data, window_size=50)

# Exponential moving average (EMA)
smoothed = ema(data, span=20)

# Z-score anomaly detection (outliers)
anomalies = detect_anomalies(data, threshold=3.0)

# Peak-preserving Savitzky-Golay spectral smoothing
counts = np.random.poisson(lam=500, size=1000).astype(float)
smooth = savitzky_golay(counts, window=11, polyorder=3)

# Gamma/Neutron peak detection (finds bin indices)
peaks = find_peaks(smooth, min_height=500, min_distance=10)

# ICRP-74 flux-to-dose equivalent conversion
dose_rate = flux_to_dose(flux=1e6, energy_mev=2.35, particle="neutron")

# Beer-Lambert shielding (utilizes canonical material database)
transmission = attenuation(thickness_cm=10.0, material="lead")
```

---

### 2. High-Energy Physics & Kinematics (`triples_sigfast.hep`)
Exposes both vectorized JIT-compiled mathematical kernels for batch operations, and a convenience `LorentzVector` class:

```python
import math
from triples_sigfast.hep.kinematics import LorentzVector

# Construct vectors from 4-momentum (E, px, py, pz) in GeV
mu1 = LorentzVector(E=45.1, px=0.0, py=44.9, pz=1.0)
mu2 = LorentzVector(E=45.1, px=0.0, py=-44.9, pz=-1.0)

# Reconstruct parent particle (e.g. Z boson) via operator overloading
Z = mu1 + mu2
print(f"Z Boson Mass: {Z.mass:.2f} GeV")      # ~90.20 GeV
print(f"Z Boson pT:   {Z.pt:.3f} GeV")        # ~0.0 GeV

# Inspect coordinate system variables
mu = LorentzVector.from_pt_eta_phi_mass(pt=30.0, eta=1.5, phi=0.8, mass=0.105)
print(f"Velocity beta = {mu.beta:.3f}c, Lorentz gamma = {mu.gamma:.2f}")
print(f"Rapidity: {mu.rapidity:.3f}, Pseudorapidity: {mu.eta:.3f}")

# Compute angular separations between particles
delta_r = mu1.delta_r(mu2)
```

---

### 3. Detector Physics Response (`triples_sigfast.detectors`)
Provides physics-validated response functions for standard experimental radiation detectors:

```python
from triples_sigfast.detectors import NaIDetector, HPGeDetector, He3Detector

# NaI(Tl) Scintillator: computes intrinsic peak efficiency and resolution
nai = NaIDetector(thickness_cm=7.62, diameter_cm=7.62)  # Standard 3x3 inch crystal
eff_662 = nai.intrinsic_efficiency(energy_mev=0.662)     # Cs-137 peak
res_662 = nai.energy_resolution(energy_mev=0.662)        # FWHM resolution in MeV
print(f"NaI Peak Efficiency: {eff_662*100:.2f}%, FWHM: {res_662*1000:.1f} keV")

# High-Purity Germanium (HPGe): Fano-factor-based resolution model
hpge = HPGeDetector(relative_efficiency=0.30)
res_co60 = hpge.energy_resolution(energy_mev=1.332)      # Co-60 peak
print(f"HPGe FWHM at 1.33 MeV: {res_co60*1000:.2f} keV")   # ~53 keV (Fano limit)

# Gas Proportional Counters: thermal neutron absorption efficiency
he3 = He3Detector(pressure_atm=4.0, active_length_cm=30.0)
eff_neutron = he3.thermal_efficiency()
print(f"He-3 Thermal Neutron Efficiency: {eff_neutron*100:.1f}%")
```

---

### 4. Plasma Physics & Fusion Neutronics (`triples_sigfast.plasma`)
Analyzes radiation and thermonuclear fusion reactivities from hot plasma cores:

```python
import numpy as np
from triples_sigfast.plasma import (
    plasma_neutron_rate,
    dt_neutron_spectrum,
    activation_saturation
)

# Calculate thermonuclear neutron production rate (Tokamak D-T/D-D Bosch-Hale)
rate = plasma_neutron_rate(
    reaction="DT",
    ion_density_m3=1e20,
    temperature_kev=10.0,
    plasma_volume_m3=840.0  # ITER scale
)
print(f"D-T Fusion Production Rate: {rate:.3e} neutrons/sec")

# Generate Doppler-broadened fusion neutron spectrum (Gaussian expansion)
energies = np.linspace(12.0, 16.0, 500)
spec = dt_neutron_spectrum(energies, temperature_kev=10.0, normalise=True)

# Calculate structural material activation saturation under neutron flux
sat_activity = activation_saturation(
    reaction_rate_cm2_s=1e14,       # Neutron flux (n/cm^2/s)
    number_density_cm3=8.47e22,     # Iron atomic density (g/cm^3)
    cross_section_cm2=2.59e-24,     # Target activation cross section
    half_life_s=9.4e5,              # Product half-life
    sample_volume_cm3=100.0         # Component volume
)
```

---

### 5. Radiation Shielding & Isotope Database (`triples_sigfast.nuclear`)

```python
from triples_sigfast.nuclear.shielding import attenuation_with_buildup
from triples_sigfast.nuclear.isotope import Isotope
from triples_sigfast.nuclear.dose import point_source_shielded

# Geometric Progression (GP) buildup factor correction (ANSI/ANS-6.4.3)
# Solves the systematic dose underestimation of Beer-Lambert in thick shields
transmission = attenuation_with_buildup(
    thickness_cm=10.0,
    material="lead",
    energy_mev=1.25,
    geometry="point_source"
)

# Comprehensive Isotope Database (NUBASE2020)
isotope = Isotope("Cf-252")
print(f"Half-life:     {isotope.half_life:.2f} years")
print(f"Neutron Yield: {isotope.neutron_yield:.3e} n/s/g")

# Shielded point source biological dose calculation
dose = point_source_shielded(
    activity_bq=1e9,
    energy_mev=1.25,
    distance_cm=100.0,
    shield_material="iron",
    shield_thickness_cm=5.0
)
```

---

### 6. Universal Simulation Readers (`triples_sigfast.io`)

Read and parse data with the universal `SimReader` class, which handles format parsing dynamically:

```python
from triples_sigfast.io import SimReader

# Auto-detects formatting (ROOT, Flair/FLUKA USRBIN, MCNP mctal, SERPENT det)
reader = SimReader("simulation_results.root")
counts, energies = reader.get_spectrum()
reader.summary()
```

- **`RootReader`**: Extracts histograms and tracks from Geant4 files, with CSV and HDF5 exporters.
- **`FlukaReader`**: Ingests multi-detector `USRBIN`, `USRBDX`, and `USRTRACK` fluence grids.
- **`MCNPReader`**: Parses `mctal` tally sheets and TFC (tally fluctuation charts) to compute Figure of Merit.
- **`SerpentReader`**: Ingests `.m` files, recovering spectral detectors, burnup variables, and $k_{eff}$.

---

## Complete Workflow Example

A unified pipeline from raw Geant4 ROOT output, checking Monte Carlo statistical convergence, identifying spectral photopeaks, and computing biological shielding effectiveness:

```python
import numpy as np
from triples_sigfast.io import RootReader
from triples_sigfast.stats.mc import relative_error, is_converged
from triples_sigfast.nuclear.isotope import Isotope
from triples_sigfast.nuclear.dose import point_source_shielded
from triples_sigfast import savitzky_golay, find_peaks

# 1. Ingest simulation data
reader = RootReader("shielding_simulation.root")
counts, energies = reader.get_spectrum("neutron_flux")

# 2. Verify statistical convergence (MCNP standard: R < 0.05)
r_err = relative_error(counts)
converged = is_converged(counts, threshold=0.05)
print(f"Statistical Convergence: {converged.sum()}/{len(counts)} bins passed")

# 3. Peak-preservation filtering and gamma/neutron spectroscopy
smoothed = savitzky_golay(counts, window=11, polyorder=3)
peaks = find_peaks(smoothed, min_height=50.0, min_distance=10)

# 4. Compute biological shielding safety levels for a Cf-252 source
cf = Isotope("Cf-252")
source_rate = cf.neutron_source_rate(mass_g=0.005)  # 5 milligram source

shielded_dose = point_source_shielded(
    activity_bq=source_rate,
    energy_mev=2.0,
    distance_cm=150.0,
    shield_material="concrete",
    shield_thickness_cm=20.0,
    particle="neutron"
)
print(f"Shielded biological dose rate: {shielded_dose:.3f} uSv/hr")
```

---

## Repository Architecture

```
triples_sigfast/
├── core/                # JIT-compiled core signal processing (Numba)
├── stats/               # Monte Carlo statistical calculations & convergence
├── io/                  # Simulation file readers (ROOT, FLUKA, MCNP, Serpent)
├── nuclear/             # Shielding, ICRP-74 dose tables, Maxwell/Watt spectra, NUBASE2020
├── hep/                 # High-Energy Physics kinematics & JIT jet clustering
│   ├── kinematics.py    # Vectorised kinematics & LorentzVector OOP class
│   └── jets.py          # JIT-compiled anti-kT jet clustering
├── detectors/           # Detector physics response (NaI, HPGe, He-3, BF3)
├── plasma/              # Fusion plasma neutronics & material activation saturation
├── viz/                 # Visualization engine & PhysicsPlot style overrides
└── cli/                 # Welcome page & CLI interactive commands
```

---

## Verification & Testing

The package includes a comprehensive testing suite containing **650 unit and integration tests** validating physics accuracy, numerical safety, CLI welcome pages, and lazy imports.

To execute tests and print the detailed coverage report:
```bash
pytest tests/ -v --cov=triples_sigfast --cov-report=term-missing
```

- **Test coverage**: **96.78%** across the entire codebase.
- **Linting standards**: Fully compliant with modern PEP-8 and PEP-484 standards, verified via Ruff (0 errors).
- **Cross-platform validation**: Matrix tested on Ubuntu, macOS, and Windows environments under Python 3.10, 3.11, 3.12, and 3.13.

---

## Roadmap

| Version | Status | Milestone Achievements |
|---|---|---|
| **v1.1.0** | ✅ Released | JIT-compiled signal processing, exponential smoothing, peak detection |
| **v1.2.0** | ✅ Released | Monte Carlo statistics, relative error, Geant4 ROOT readers |
| **v1.4.0** | ✅ Released | Native FLUKA, MCNP, and SERPENT reader backends |
| **v1.6.0** | ✅ Released | Interactive CLI welcome screen, `AutoReport` PDF generator |
| **v1.7.0** | ✅ Released | Native LHE/HepMC parsers and JIT-compiled anti-kT jet clustering |
| **v1.8.0** | ✅ Released | Plotext terminal plots, standardization of package interfaces |
| **v1.8.1** | ✅ Released | Initial stubs for detectors and plasma packages, JIT kinematics updates |
| **v1.8.2** | ✅ Released | Fully integrated physics-validated `detectors/` and `plasma/` packages, OOP `LorentzVector` class, unified material tables, 650+ tests |
| **v2.0.0** | ⬜ Planned | Community launch, JOSS paper submission |

---

## License

`triples-sigfast` is open-source software licensed under the [MIT License](LICENSE).

---

## Developer Contact

Developed by **TripleS Studio**.
- **PyPI**: [project page](https://pypi.org/project/triples-sigfast/)
- **GitHub**: [repository home](https://github.com/SamdaniSayam/triples-sigfast)