# Changelog

All notable changes to this project will be documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Quarto-based theory and developer guide covering interval merge mathematics, API selection, aggregation behavior, and rendered HTML/PDF documentation outputs.
- Regression tests covering fast-path dispatch, aggregation dtype validation, and categorical `KeepLongest()` tie behavior.

### Changed
- `on_slk_intervals(..., legacy=False)` and `on_slk_intervals_auto()` now route through the same fastest-available fast path, preferring the Numba backend when available.
- Numeric-only aggregations now fail early with `InvalidAggregationError` when used on non-numeric source columns.
- `KeepLongest()` now uses stable first-seen tie-breaking across the legacy and categorical fallback paths.
- Improved fast-path performance on the 5,000 target / 15,000 data / 5-group benchmark: auto-dispatch dropped from about `0.97s` to `0.03s` when Numba is available, and the dense optimized path runs at about `0.59s` versus `18.87s` for the legacy path.

### Fixed
- Removed the broad silent fallback from the fast path so optimized-backend failures surface instead of being masked.

## [1.1.0] - 2025-12-22

### Added
- Numba-accelerated sparse merge implementation (`_numba_merge.py`) to avoid OOM on large datasets and improve performance.
- Tests covering the Numba path and a new benchmarking script (`benchmarks/compare_merges.py`) to compare legacy, optimized, and numba implementations.
- CI/tooling updates: switched from Black to Ruff, updated `nox` sessions, and clarified dev install instructions.

### Changed
- Bumped requires-python to >= 3.10 and adjusted docs.

## [1.0.0] - 2025-10-22

### Added
- Documented the supported public API surface in the README alongside a clearer
  semantic versioning policy and release hygiene expectations.
- Published benchmark tooling (`benchmarks/compare_merges.py`) capturing
  performance parity between legacy and optimized merge paths, plus optional
  runtime performance logging hooks and auto-dispatch helper.
- Expanded automated verification with new validation utilities, domain
  exceptions, focused regression tests, and property-based fuzz coverage.
- Introduced development tooling extras, reproducible requirements, a `nox`
  command suite, and CI workflows covering linting, typing, testing, coverage,
  and wheel smoke tests across platforms.

### Changed
- Reworked packaging metadata to classify optional dependencies as extras,
  tightened minimum supported versions, and refreshed installation guidance to
  emphasise virtual environments and contributor setup.

### Removed
- Historical TODO comment about untested pandas versions from the dependency
  list.

## [0.6.1] - 2023-07-04

> Historical release prior to adopting this changelog. Refer to the git history
> for specific changes.
