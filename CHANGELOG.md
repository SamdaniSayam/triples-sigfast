# Changelog

All notable changes to triples-sigfast will be documented here.

## [Unreleased]

### Bug Fixes

- Remove invalid strtitle filter from cliff.toml template


### CI/CD

- Fix changelog YAML indentation

- Rewrite changelog workflow with manual trigger


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



