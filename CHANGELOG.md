# Changelog

All notable changes to triples-sigfast will be documented here.

## [2.0.0] - 2026-06-08

### Features

- **Massive Data Handling (ROOT-like)**: Introduced `SigPipeline` for pure Python out-of-core streaming of datasets exceeding memory capacity. Includes Pandas chunked reading and `uproot.iterate` for ROOT `TTree` chunking.
- **Monte Carlo Generation (PYTHIA-like)**: Added `hep.mc` containing a fully vectorized Numba RAMBO algorithm for N-body phase space generation, natively producing flat $N \times 4$ NumPy arrays.
- **Data-Oriented Decays**: Updated `hep.kinematics` with `decay_two_body`, adhering to strict Data-Oriented Design by processing massive arrays without object instantiation overhead.
- **Ahead-of-Time PDG Ingestion**: Added the `particle` library dependency. Extracted PDG data into Numba `typed.Dict`/arrays during setup for cache-safe, GIL-free $O(\log N)$ binary search lookups inside hot loops.
- **JIT-Compiled Fitting**: Expanded `stats.fitting` with Numba PDFs (Crystal Ball, Voigtian) and a `build_nll_cost_function` factory that compiles entire Negative Log-Likelihood cost functions to machine code for massive speedups in `scipy.optimize`.

### Refactoring

- **CLI Streaming**: Added `--stream` flag to CLI to natively support the new out-of-core analysis.
- **OOP JIT Deprecation**: Deprecated using `@jitclass` `LorentzVector` in hot loops in favor of strict flat array vectorization.

## [1.8.2] - 2026-05-21
### Documentation

- Auto-update CHANGELOG for v1.8.1

- Auto-update CHANGELOG for main


### style

- Apply ruff formatting to main.py


## [1.8.1] - 2026-05-11

### Documentation

- Auto-update CHANGELOG for v1.7.0

- Auto-update CHANGELOG for 1.8.0


### Features

- **Detectors Package (`triples_sigfast.detectors`)**: Fully implemented physics-validated response models for NaI(Tl), HPGe, He-3, and BF3 counters. Computes intrinsic photon/neutron efficiencies and Fano-factor-based resolution estimation.
- **Plasma Sub-package (`triples_sigfast.plasma`)**: Added Doppler-broadened fusion neutron spectrum simulation (D-T and D-D), thermonuclear reaction rates via Bosch-Hale parameterisation, and structural material activation saturation calculations.
- **LorentzVector class (`triples_sigfast.hep.kinematics`)**: Implemented ergonomic, OOP-style 4-vector algebra with properties (pt, p, mass, eta, phi, rapidity, beta, gamma), arithmetic operator overloading, $\Delta R$ angular separations, and component-wise NumPy exports.
- **Unified Material Tables**: Consolidated and centralized NIST XCOM and ANSI/ANS-6.4.3 material attenuation properties inside `nuclear/shielding.py` as a single authoritative source, dynamically building the `core/signal.py` lookup tables at import time. Added bismuth, tungsten, borated polyethylene, and polysulfone.

### Testing and Quality Assurance

- Expanded tests to **650 unit and integration tests** (+180 tests), raising total test coverage to **96.78%** (with `io/raw.py` coverage rising from 17% to 95%).
- Resolved welcome TTY-drawing branches coverage using mock terminal consoles.
- Standardized codebase quality with zero lint or formatting errors (verified with Ruff).


### style

- Apply ruff formatting


## [1.7.0] - 2026-05-09

### Bug Fixes

- Expose AutoReport from top-level __init__.py

- Add .coverage, plots/, flex_benchmark.py to .gitignore

- Pin macos-13 Intel architecture and bust pip cache


### Documentation

- Auto-update CHANGELOG for v1.6.0

- Add JOSS paper (paper.md and paper.bib)

- Add JOSS paper (paper.md and paper.bib)

- Add LICENSE file, move paper to root, add missing JOSS sections

- Add LICENSE file, move paper to root, add missing JOSS sections


### Features

- Release v1.7.0 - Native LHE/HepMC parsers and Anti-kT JIT clustering


## [1.6.0] - 2026-04-06

### Documentation

- Auto-update CHANGELOG for v1.5.3


### Features

- V1.6.0 - CLI, AutoReport, guided mode, 385 tests, 100% coverage


## [1.5.3] - 2026-04-05

### Bug Fixes

- Remove dist/ from git tracking and add to .gitignore

- Sync version to 1.5.3 in both pyproject.toml and setup.py


### Documentation

- Auto-update CHANGELOG for v1.5.2


## [1.5.2] - 2026-04-05

### Bug Fixes

- Clean dist artifacts before build + bump to 1.5.2


### Documentation

- Auto-update CHANGELOG for v1.5.1


## [1.5.1] - 2026-04-05

### Bug Fixes

- Changelog workflow detached HEAD - checkout main before push


### Documentation

- Auto-update CHANGELOG for main


### Releases

- Bump to 1.5.1 to re-trigger PyPI publish


## [1.5.0] - 2026-04-05

### Documentation

- Auto-update CHANGELOG for main

- Update test stats to 323 tests, 100% coverage

- Auto-update CHANGELOG for main


### Features

- V1.5.0 - PhysicsPlot visualization engine, 361 tests, 100% coverage


## [1.4.1] - 2026-04-03

### Bug Fixes

- Correct setuptools build backend in pyproject.toml

- Correct version to 1.4.0 in setup.py


### CI/CD

- Fix changelog workflow - replace Docker action with binary installer


### Documentation

- Rewrite README for v1.4.0 — all modules documented

- Auto-update CHANGELOG for main


### Releases

- Bump version to 1.4.0 for PyPI deployment

- V1.4.0 - nuclear physics, MC stats, multi-code readers, 323 tests, 100% coverage

- Bump to 1.4.1 to fix PyPI upload version mismatch


## [1.4.0] - 2026-04-03

### Features

- Add native FlukaReader, MCNPReader, SerpentReader (v1.4.0)


### Testing

- Achieve 100% coverage — decay_mode, resonance_integral, GP fallback paths


### style

- Ruff format and lint fixes


## [1.3.0] - 2026-04-03

### CI/CD

- Fix detached head in changelog workflow

- Fix detached HEAD error in changelog workflow

- Force git checkout main in changelog workflow to fix detached head


### Features

- Implemented watt spectrum, dose conversions, and isotope databases for v1.3.0


### Releases

- Bump version to 1.3.0 for PyPI deployment


### style

- Ruff format __init__.py

- Fix import ordering (ruff I001)

- Ruff format all files

- Removed trailing whitespace in sources.py


## [1.2.0] - 2026-04-02

### Features

- Add mc module — relative_error, figure_of_merit, is_converged, propagate_error

- Add RootReader and SimReader with Geant4/FLUKA/MCNP/SERPENT support


## [1.2.0-dev] - 2026-04-02

### Refactoring

- Restructure package into submodules (core, nuclear, io, stats, viz, detectors, plasma, cli)


## [1.1.1] - 2026-04-02

### Bug Fixes

- Clean up setup.py, fix encoding, bump python_requires to 3.10

- Add trailing newline to setup.py

- Include requirements.txt in package build via MANIFEST.in


### Documentation

- Auto-update CHANGELOG for main


### Releases

- Bump version to 1.1.1


## [1.1.0] - 2026-04-02

### Bug Fixes

- Remove invalid strtitle filter from cliff.toml template

- Correct type hints in attenuation and clean main.yml


### CI/CD

- Fix changelog YAML indentation

- Rewrite changelog workflow with manual trigger


### Documentation

- Auto-update CHANGELOG for main


### Features

- Add nuclear physics features - savitzky_golay, find_peaks, flux_to_dose, attenuation


## [1.0.3] - 2026-04-02

### CI/CD

- Fix changelog workflow by installing git-cliff as binary


### Releases

- Bump version to 1.0.3


## [1.0.2] - 2026-04-02

### Documentation

- Update benchmark table with real stress test results

- Add CONTRIBUTING.md, changelog workflow, and expanded test suite


### Testing

- Achieve 100% coverage with pragma no cover on Numba JIT kernels


## [1.0.1] - 2026-04-01

### Bug Fixes

- Renamed package folder sigfast → triples_sigfast to match import name

- Corrected import paths from sigfast.core to triples_sigfast.core

- Removed emoji from smoke test to fix Windows CP1252 encoding error

- Set PYTHONUTF8=1 globally to handle emoji output on Windows

- Resolve all remaining Ruff lint errors via ruff format


### CI/CD

- Implemented bulletproof CI/CD pipeline with matrix build and caching


### Releases

- Bump version to 1.0.1 with import fixes and code formatting


### Testing

- Migrated to pytest and added 13 comprehensive unit tests


### style

- Auto-fixed import ordering and formatting with Ruff

- Auto-formatted entire codebase with ruff


## [1.0.0] - 2026-03-16

### chore

- Update branding to triples-sigfast


## [0.1.0] - 2026-03-10

### Refactor

- Rebranded library to sigfast, updated setup.py, and added Pandas support



