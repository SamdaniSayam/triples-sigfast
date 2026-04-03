# Changelog

All notable changes to triples-sigfast will be documented here.

## [Unreleased]

### Documentation

- Auto-update CHANGELOG for main

- Update test stats to 323 tests, 100% coverage


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



