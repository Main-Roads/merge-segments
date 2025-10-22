# Changelog

All notable changes to this project will be documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Nothing yet.

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

## [0.6.1] - 2024-??-??

> Historical release prior to adopting this changelog. Refer to the git history
> for specific changes.
