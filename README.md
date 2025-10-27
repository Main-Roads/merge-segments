# `merge_segments`<!-- omit in toc -->

- [1. About This Fork](#1-about-this-fork)
- [2. Introduction](#2-introduction)
    - [2.1. Public API Surface](#21-public-api-surface)
- [3. Install, Upgrade, Uninstall](#3-install-upgrade-uninstall)
    - [3.1. Recommended Environment Setup](#31-recommended-environment-setup)
    - [3.2. Optional Extras](#32-optional-extras)
    - [3.3. Development Install](#33-development-install)
- [4. Module `merge`](#4-module-merge)
  - [4.1. Function `merge.on_slk_intervals()`](#41-function-mergeon_slk_intervals)
    - [4.1.1. Function `merge.on_slk_intervals_optimized()`](#411-function-mergeon_slk_intervals_optimized)
  - [4.2. Class `merge.Action`](#42-class-mergeaction)
  - [4.3. Class `merge.Aggregation`](#43-class-mergeaggregation)
    - [4.3.1. Notes about `KeepLongest()`](#431-notes-about-keeplongest)
    - [4.3.2. Notes about `LengthWeightedPercentile(...)`](#432-notes-about-lengthweightedpercentile)
  - [4.4. Practical Example of Merge](#44-practical-example-of-merge)
- [5. Notes](#5-notes)
  - [5.1. Correctness, Robustness, Test Coverage and Performance](#51-correctness-robustness-test-coverage-and-performance)
- [6. Release & Versioning](#6-release--versioning)
    - [6.1. Semantic Versioning Policy](#61-semantic-versioning-policy)
    - [6.2. Changelog](#62-changelog)
    - [6.3. Static Type Checking](#63-static-type-checking)
- [7. Performance & Controls](#7-performance--controls)
    - [7.1. Benchmarking](#71-benchmarking)
    - [7.2. Performance Logging](#72-performance-logging)
    - [7.3. Selecting the Optimized Path](#73-selecting-the-optimized-path)
    - [7.4. Validating Optimized Outputs](#74-validating-optimized-outputs)

## 1. About This Fork

This repository is a **maintained fork** of the original
[`merge_segments`](https://github.com/thehappycheese/merge-segments) package
created and maintained by [@thehappycheese](https://github.com/thehappycheese).
We are grateful for the original work, which provided a robust, well-tested
foundation for merging linear segment data.

**Maintainer:** Dagmawi Tadesse  
**Original Author:** Nicholas Archer

### Why This Fork Exists

The original `merge_segments` package on PyPI is no longer actively maintained.
This fork was created by Main Roads Western Australia to:

- **Continue maintenance and bug fixes** for production usage
- **Add performance optimizations** including a new vectorized merge implementation
- **Improve developer experience** with modern CI/CD, type checking, and tooling
- **Extend functionality** while maintaining backward compatibility

### Key Differences from the Original

| Feature | Original (PyPI) | This Fork (Main-Roads) |
|---------|----------------|------------------------|
| **Maintenance** | No longer maintained | Actively maintained |
| **Installation** | `pip install merge_segments` | `python -m pip install git+https://github.com/Main-Roads/merge-segments.git` |
| **Optimized Path** | Legacy implementation only | Legacy + vectorized optimized implementation |
| **CI/CD** | Basic | Comprehensive (linting, type-checking, testing, wheels) |
| **Type Hints** | Partial | Full mypy compliance with `py.typed` |
| **Documentation** | Original README | Extended with performance guides and migration notes |

### Choosing Between Versions

- **Use the PyPI version** (`pip install merge_segments`) if you need the last
  stable release and don't require the latest features or optimizations.
- **Use this fork** if you need continued maintenance, performance improvements,
  or want to contribute to ongoing development.

Both versions maintain the same core API for backward compatibility, though this
fork includes additional features (like the `legacy` parameter and optimized
implementations) not available in the PyPI version.

## 2. Introduction

The purpose is to combine two data tables which have a linear segment index ("from" and "to" columns); ie where each row in the input tables represents some linear portion of an entity; for example a road segment from 5km to 10km.

There is an ongoing effort to accelerate and parallelise the merge function
under a new repo called
[megamerge](https://github.com/thehappycheese/megamerge)

### 1.1. Public API Surface

The `merge_segments` package guarantees the following top-level symbols. They
are re-exported from the package root and covered by semantic versioning.

| Symbol | Kind | Description |
| ------ | ---- | ----------- |
| `Action` | class | Describes a column aggregation request (source column, aggregation strategy, optional rename). |
| `Aggregation` | class | Factory for supported aggregation strategies such as `KeepLongest`, `LengthWeightedAverage`, and `LengthWeightedPercentile`. |
| `on_slk_intervals` | function | Main merge function with `legacy` parameter to choose between implementations (defaults to legacy for backward compatibility). |
| `on_slk_intervals_legacy` | function | Legacy, fully tested merge routine that operates row-by-row. |
| `on_slk_intervals_optimized` | function | Vectorised merge routine that accelerates most numeric workloads while maintaining feature parity with the legacy path. |
| `on_slk_intervals_fallback` | function | Categorical-friendly implementation that the optimized function delegates to when required. |
| `__version__` | string | Package version determined from installed metadata. |

Import them directly from the package root:

```python
from merge_segments import Action, Aggregation, on_slk_intervals
```

Additional implementation details (validation helpers, exception types) remain
internal. Backwards compatibility promises apply to the table above and any
behaviour explicitly documented in this README.

## 3. Install, Upgrade, Uninstall

### 3.1. Recommended Environment Setup

Create an isolated environment before installing to keep dependencies tidy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install git+https://github.com/Main-Roads/merge-segments.git
```

To install a specific version or branch:

```powershell
# Install a specific version tag
python -m pip install git+https://github.com/Main-Roads/merge-segments.git@v1.0.0

# Install a specific branch
python -m pip install git+https://github.com/Main-Roads/merge-segments.git@branch-name

# Install a specific commit
python -m pip install git+https://github.com/Main-Roads/merge-segments.git@commit-hash
```

Upgrade, inspect, or remove the package from the same environment as needed:

```powershell
python -m pip install --upgrade --force-reinstall git+https://github.com/Main-Roads/merge-segments.git
python -m pip show merge_segments
python -m pip uninstall merge_segments
```

### 3.2. Optional Extras

The core package now keeps optional tooling out of the default install. Add
extras when you need progress bars or plotting helpers:

```powershell
python -m pip install "git+https://github.com/Main-Roads/merge-segments.git#egg=merge_segments[progress]"      # tqdm-backed progress bars
python -m pip install "git+https://github.com/Main-Roads/merge-segments.git#egg=merge_segments[plotting]"      # matplotlib-based charts
python -m pip install "git+https://github.com/Main-Roads/merge-segments.git#egg=merge_segments[progress,plotting]"  # All optional features
```

### 3.3. Development Install

Contributors can clone the repository and install the pinned toolchain with the
provided requirements file:

```powershell
git clone https://github.com/Main-Roads/merge-segments.git
cd merge-segments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

This installs the package in editable mode with optional extras, plus linters,
type checkers, and the test stack declared in `pyproject.toml`.

## 4. Module `merge`

### 3.1. Function `merge.on_slk_intervals()`

`merge.on_slk_intervals()` is the main entry point for merging interval data. It
provides a `legacy` parameter to choose between the legacy implementation (default)
and the optimized implementation. By default, it uses the legacy implementation
for backward compatibility and extensive real-world validation.

The following code demonstrates `merge.on_slk_intervals()` by merging the dummy
dataset `pavement_data` against the target `segmentation` dataframe.

```python
import merge_segments.merge as merge

segmentation = pd.DataFrame(
    columns=["road_no", "carriageway", "slk_from", "slk_to"],
    data=[
        ["H001", "L",  10,  50],
        ["H001", "L",  50, 100],
        ["H001", "L", 100, 150],
    ]
)

pavement_data = pd.DataFrame(
    columns=["road_no", "carriageway", "slk_from", "slk_to", "pavement_width", "pavement_type"],
    data=[
        ["H001", "L",  00,  10, 3.10,  "tA"],
        ["H001", "L",  10,  20, 4.00,  "tA"],
        ["H001", "L",  20,  40, 3.50,  "tA"],
        ["H001", "L",  40,  80, 3.80,  "tC"],
        ["H001", "L",  80, 130, 3.10,  "tC"],
        ["H001", "L", 130, 140, 3.00,  "tB"],
    ]
)

result = merge.on_slk_intervals(
    target=segmentation,
    data=pavement_data,
    join_left=["road_no", "carriageway"],
    column_actions=[
        merge.Action("pavement_width",  merge.Aggregation.LengthWeightedAverage()),
        merge.Action("pavement_type",   merge.Aggregation.KeepLongest())
    ],
    from_to=("slk_from", "slk_to")
)

assert result.compare(
    pd.DataFrame(
        columns=["road_no", "carriageway", "slk_from", "slk_to", "pavement_width", "pavement_type"],
        data=[
            ["H001", "L",  10,  50, 3.700, "tA"],
            ["H001", "L",  50, 100, 3.520, "tC"],
            ["H001", "L", 100, 150, 3.075, "tC"],
        ]
    )
).empty

```

To use the optimized implementation, set `legacy=False`:

```python
result = merge.on_slk_intervals(
    target=segmentation,
    data=pavement_data,
    join_left=["road_no", "carriageway"],
    column_actions=[
        merge.Action("pavement_width",  merge.Aggregation.LengthWeightedAverage()),
        merge.Action("pavement_type",   merge.Aggregation.KeepLongest())
    ],
    from_to=("slk_from", "slk_to"),
    legacy=False  # Use optimized implementation
)
```

For debugging or diagnostic purposes, enable verbose output with `verbose=True`:

```python
result = merge.on_slk_intervals(
    target=segmentation,
    data=pavement_data,
    join_left=["road_no", "carriageway"],
    column_actions=[
        merge.Action("pavement_width",  merge.Aggregation.LengthWeightedAverage()),
        merge.Action("pavement_type",   merge.Aggregation.KeepLongest())
    ],
    from_to=("slk_from", "slk_to"),
    legacy=False,
    verbose=True  # Show diagnostic messages and fallback notices
)
```

When `verbose=True`, the function will print diagnostic information such as which implementation path is being used and when fallback to categorical handling occurs. By default (`verbose=False`), only progress bars are shown and pandas warnings are suppressed for cleaner output.

### 3.1.1. Function `merge.on_slk_intervals_optimized()`

`merge.on_slk_intervals_optimized()` can also be called directly. It uses a
vectorised implementation that prioritises speed and has passed the automated
test suite, producing matching outputs for the covered scenarios. However, it
still requires additional testing on large, real-world datasets before being
treated as a production-ready drop-in replacement. The helper automatically
falls back to the legacy logic for categorical aggregations, so behaviour
remains consistent even when the fast path cannot be used.

A minimal invocation:

```python
result = merge.on_slk_intervals_optimized(
    target=segmentation,
    data=pavement_data,
    join_left=["road_no", "carriageway"],
    column_actions=[
        merge.Action("pavement_width",  merge.Aggregation.LengthWeightedAverage()),
        merge.Action("pavement_type",   merge.Aggregation.KeepLongest()),
    ],
    from_to=("slk_from", "slk_to"),
)
```

Use the optimized implementation (via `legacy=False` or by calling
`on_slk_intervals_optimized()` directly) when you need shorter runtimes and are
able to perform additional validation in your environment; otherwise prefer the
legacy implementation for its battle-tested reliability.

| Parameter      | Type                 | Note                                                                                                                                                                                                                                                                                                              |
| -------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| target         | `pandas.DataFrame`   | The result will have <ul><li>The same number of rows as the `target` data frame</li><li>The same sort-order as the `target` dataframe, and</li><li>each row of the result will match `slk_from` and `slk_to` of the `target` dataframe.</li></ul>                                                                 |
| data           | `pandas.DataFrame`   | Columns from this DataFrame will be aggregated to match the `target` slk segmentation                                                                                                                                                                                                                             |
| join_left      | `list[str]`          | Ordered list of column names to join with.<br>Typically `["road_no","cway"]`.<br>Note:<ul><li>These column names must match in both the `target` and `data` DataFrames</li></ul>                                                                                                                                  |
| column_actions | `list[merge.Action]` | A list of `merge.Action()` objects describing the aggregation to be used for each column of data that is to be added to the target. See examples below.                                                                                                                                                           |
| from_to        | `tuple[str, str]`    | The name of the start and end interval measures.<br>Typically `("slk_from", "slk_to")`.<br>Note:<ul><li>These column names must match in both the `target` and `data` DataFrames</li><li>These columns should be converted to integers for reliable results prior to calling merge (see example below.)</li></ul> |
| legacy         | `bool`               | **Optional.** If `True` (default), uses the legacy implementation. If `False`, uses the optimized vectorized implementation. Can be omitted for backward compatibility.                                                                                                                                          |
| verbose        | `bool`               | **Optional.** If `True`, prints diagnostic messages including implementation path and fallback notices. If `False` (default), only shows progress bars and suppresses pandas warnings for cleaner output.                                                                                                          |

### 3.2. Class `merge.Action`

The `merge.Action` class is used to specify how a new column will be added to
the `target`.

Normally this would only ever be used as part of a call to the
`on_slk_intervals` function as shown below:

```python
import merge_segments.merge as merge

result = merge.on_slk_intervals(
    ..., 
    column_actions = [
        merge.Action(column_name="column1", aggregation=merge.Aggregation.KeepLongest(), rename="column1_longest"),
        merge.Action("column1", merge.Aggregation.LengthWeightedAverage(), "column1_avg"),
        merge.Action("column2", merge.Aggregation.LengthWeightedPercentile(0.75)),
    ]
)

```

| Parameter   | Type                | Note                                                                                                                                                          |
| ----------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| column_name | `str`               | Name of column to aggregate in the `data` dataframe                                                                                                           |
| aggregation | `merge.Aggregation` | One of the available merge aggregations described in the section below.                                                                                       |
| rename      | `Optional[str]`     | New name for aggregated column in the result dataframe. Note that this allows you to output multiple aggregations from a single input column. Can be omitted. |

### 3.3. Class `merge.Aggregation`

The following merge aggregations are supported:

| Constructor                                                   | Purpose                                                                                                                                                                                                                          |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `merge.Aggregation.First()`                                   | Keep the first non-blank value.                                                                                                                                                                                                  |
| `merge.Aggregation.KeepLongest()`                             | Keep the longest non-blank value. ( [see notes below](#331-notes-about-aggregationkeeplongest) )                                                                                                                                                                                |
| `merge.Aggregation.LengthWeightedAverage()`                   | Compute the length weighted average of non-blank values                                                                                                                                                                          |
| `merge.Aggregation.Average()`                                 | The average non-blank overlapping value.                                                                                                                                         |
| `merge.Aggregation.LengthWeightedPercentile(percentile=0.75)` | Compute the length weighted percentile ( [see notes below](#332-notes-about-aggregationlengthweightedpercentilepercentile) ). Value should be between 0.0 and 1.0. 0.75 means 75th percentile.                                                                                       |
| `merge.Aggregation.SumProportionOfData()`                     | The sum of all overlapping `data` segments, where the value of each overlapping segment is multiplied by the length of the overlap divided by the length of the `data` segment. This is the same behaviour as the old VBA macro. |
| `merge.Aggregation.SumProportionOfTarget()`                   | The sum of all overlapping `data` segments, where the value of each overlapping segment is multiplied by the length of the overlap divided by the length of the `target` segment. This aggregation method is suitable when aggregating columns measured in `Units per Kilometre` or `% of length`. The aggregated value will have the same unit.                                                |
| `merge.Aggregation.Sum()`                                     | Compute the sum of all data overlapping the target segment.                                                                                                                                                                      |
| `merge.Aggregation.Min()`                                     | The minimum value in `data` which overlaps the segment in `target`.                                                                                                                                                              |
| `merge.Aggregation.Max()`                                     | The maximum value in `data` which overlaps the segment in `target`.                                                                                                                                                              |
| `merge.Aggregation.IndexOfMin()`                              | The row-index in the `data` with the minimum value. After merging the index can be used to fetch things like `"Surface Type"` of `"Oldest Surface"` (ie minimum `"Surface Year"`)                                                |
| `merge.Aggregation.IndexOfMax()`                              | The row-index in the `data` with the maximum value.                                                                                                                                                                              |

#### 3.3.1. Notes about `KeepLongest()`

`KeepLongest()` works by observing both the segment lengths and segment values
for data rows matching a particular target segment.

**Note 1:** If all segments are the same length but have different values, then
the first segment to appear in the data input table will be selected. This
'select first' behaviour is determined by the internal behaviour of pandas and
numpy and should not be relied upon to stay consistent in the future. Any random
segment may be chosen:

```text
Target Segment:       |==========================|
Data Segments:        |== 33 ==|== 55 ==|== 66 ==|== 77 ==|
KeepLongest:          |== 33 ====================|
```

**Note 2:** If the data to be merged has several short segments with the same
value, which together form the 'longest' value then this longest non-missing
value will be selected. For example in the situation below the data segment `55`
is the longest individual *segment*, but `99` is the longest *value*. The result
is therefore `99`.

```text
Target Segment:          |================================|
Data segment:      |=======55=======|==99==|==99==|==99==|==11==|
KeepLongest:             |=============99=================|
```

**Note 3:** Continuity of the data in `KeepLongest` is not considered. In the following
example the value 55 is the longest *continuous* overlapping value, but the
output 99 is selected because it is still the longest overlapping value
*when ignoring continuity*.

```text
Target Segment:          |==================================|
Data segment:          |==99==|======55======|==99==|==99==|==99==|==11==|
KeepLongest:             |=============99===================|
```


**Note 4:** Blank (`numpy.nan`) values are not considered when looking for the longest
value. In the following example the `KeepLongest` will keep the value 55, even
though the longest overlapping value is `numpy.nan`
```text
Target Segment:          |=======================================|
Data segment:       |=== nan ===|== 55 ==|== nan ==|== nan ==|== nan ==|
KeepLongest:             |========= 55 ==========================|
```

**Note 5:** No rounding is performed to facilitate the behaviour described in
Notes 2, 3 and 4. Data must be pre-processed if it is expected that issues
regarding floating point number equality (ie `1.0 == 0.99999999999999999`) will
cause misbehaviour for the `KeepLongest` aggregation. Internally the `pandas`
`Series.groupby()` function is used to choose the longest segment by grouping by
segment values. Actual behaviour will depend on how that function is implemented
by `pandas` internal code.

#### 3.3.2. Notes about `LengthWeightedPercentile(...)`

A the 'length weighted' version of percentile is a fairly uncommon operation
that only really makes sense when aggregating values for segments of varying
lengths;

The procedure is similar to a normal percentile calculation in that it involves
sorting the values to be merged in ascending order onto a vertical bar chart,
then sampling the `y` value of the chart at some fraction (percentage) of the
way along the `x` axis.

The 'length weighted' version provided by this package is very similar, except
that the 'width' of the bars in the bar chart are increased to match the (slk)
length of the segments they represent. Values are still sorted by ascending
order along the `x` axis, not by length of segment. The percentage is then
measured from the midpoint of the first bar to the midpoint of the last bar, and
linear interpolation is performed between the midpoint of each bar in between.

```text
      |                          _○_
      |                         |   |
      |                      ▴  |   |   <---- 75th percentile Value
Value |              _____○_____|   |
      |        __○__|           |   |
      |       |     |           |   |
      |  __○__|     |           |   |
      | |     |     |           |   |
           |<-----SLK Length----->|
           0%                ↑   100%
                             │
      75th percentile ───────┘
```

### 3.4. Practical Example of Merge

```python
import pandas as pd
import merge_segments.merge as merge

# =====================================================
# Use a data class to hold some standard column names
# =====================================================
class CN:
    road_number = "road_no"
    carriageway = "cway"
    segment_name = "seg_name"
    slk_from = "slk_from"
    slk_to = "slk_to"
    pavement_total_width = "PaveW"
    pavement_year_constructed = "PaveY"

# =====================================================
# load target segmentation
# =====================================================
segmentation = pd.read_csv("network_segmentation.csv")

# Rename columns to our standard names:
segmentation = segmentation.rename(columns={
    "RoadName":     CN.road_number,
    "Cway":         CN.carriageway,
    "Name":         CN.segment_name,
    "From":         CN.slk_from,
    "To":           CN.slk_to
})

# Drop rows where critical fields are blank
segmentation = segmentation.dropna(subset=[CN.road_number, CN.carriageway, CN.slk_from, CN.slk_to])

# Convert SLKs to meters and round to integer
segmentation[CN.slk_from] = (segmentation[CN.slk_from]*1000.0).round().astype("int")
segmentation[CN.slk_to]   = (segmentation[CN.slk_to]  *1000.0).round().astype("int")
# Note that .round() is required, otherwise .astype("int") 
# will always round toward zero (ie 1.99999 would become 1)

# =====================================================
# load data to be merged
# =====================================================
pavement_data = pd.read_csv("pavement_details.csv")

# Rename columns to our standard names:
pavement_data = pavement_data.rename(columns={
    "ROAD_NO":          CN.road_number,
    "CWY":              CN.carriageway,
    "START_SLK":        CN.slk_from,
    "END_SLK":          CN.slk_to,
    "TOTAL_WIDTH":      CN.pavement_total_width,
    "PAOR_PAVE_YEAR":   CN.pavement_year_constructed,
})

# Drop rows where critical fields are blank
pavement_data = pavement_data.dropna(subset=[CN.road_number, CN.carriageway, CN.slk_from, CN.slk_to])

# Convert SLKs to meters and round to integer
pavement_data[CN.slk_from] = (pavement_data[CN.slk_from]*1000.0).round().astype("int")
pavement_data[CN.slk_to]   = (pavement_data[CN.slk_to]  *1000.0).round().astype("int")

# =====================================================
# Execute the merge:
# =====================================================

segmentation_pavement = merge.on_slk_intervals(
    target=segmentation,
    data=pavement_data,
    join_left=[CN.road_number, CN.carriageway],
    column_actions=[
        merge.Action(CN.pavement_total_width,        merge.Aggregation.LengthWeightedAverage()),
        merge.Action(CN.pavement_year_constructed,   merge.Aggregation.KeepLongest())
    ],
    from_to=(CN.slk_from, CN.slk_to)
)

segmentation_pavement.to_csv("output.csv")
```

## 4. Notes

### 4.1. Correctness, Robustness, Test Coverage and Performance

This package aims to be as robust as its predecessor; an old VBA Excel Macro.
The old Macro is well trusted and has a proven track record.

The merge function performs several checks before proceeding:

- Some checking for correct parameter datatypes (ie target is a `DataFrame`, not a `Series`)
- `MultiIndex` `DataFrame`s are not permitted.
  - Originally this module was designed to work well with `MultiIndex`s but
    there are many unexpected situations where this causes cryptic warnings,
    misbehaviour, or even errors. I blame `pandas` for this, since in most cases
    these issues arise from un/poorly-documented pandas behaviour.
- Duplicates in the index
- Clashing column names

This list is not exhaustive, since there are many errors which can arise from
malformed input data. We can't catch them all!

However, if we assume input data is well formed, then we can test to make sure
we get correct outputs. Currently there is a limited suit of tests which run
using the `pytest` library.

- About 70% of the total functionality is tested
- The other 30% has been extensively hand checked to confirm outputs are as expected.

Finally, performance is relatively poor, in the future, performance
optimisations could be explored

- column-wise parallelism
- building a Rust python module (see https://github.com/thehappycheese/megamerge)

## 5. Release & Versioning

### 5.1. Semantic Versioning Policy

`merge_segments` follows [Semantic Versioning 2.0.0](https://semver.org/). The
documented public API in [Section&nbsp;1.1](#11-public-api-surface) remains stable
within a major release. Backwards-incompatible changes will only ship alongside
major version bumps after being highlighted in the changelog.

### 5.2. Changelog

User-facing changes are recorded in [`CHANGELOG.md`](./CHANGELOG.md). Each
release entry summarises new features, bug fixes, and deprecations so consumers
can assess impact before upgrading.

Latest release: **1.0.0** (2025-10-22).

### 5.3. Static Type Checking

The distribution includes inline type hints and a `py.typed` marker so type
checkers recognise the package as typed. A MyPy configuration (see
`[tool.mypy]` in `pyproject.toml`) targets the `merge_segments` package and
enables optional static analysis via:

```powershell
python -m pip install mypy
python -m mypy src/merge_segments
```

## 6. Performance & Controls

### 6.1. Benchmarking

Use `benchmarks/compare_merges.py` to compare the legacy and optimized paths on
synthetic data:

```powershell
python benchmarks/compare_merges.py --targets 5000 --data 15000 --groups 5 --repeats 5
```

The script times both implementations, asserts that their outputs match, and
prints a speedup factor. Adjust the row counts to resemble your production data
volume before extrapolating the results.

### 6.2. Performance Logging

Register a custom logger via `merge.configure_performance_logger` to collect
merge durations, row counts, and whether the optimized path delegated to the
fallback:

```python
from merge_segments import configure_performance_logger

def log_event(name: str, metrics: dict[str, float]) -> None:
    print(f"event={name} metrics={metrics}")

configure_performance_logger(log_event)
```

For ad-hoc tracing, set an environment variable before importing the package:

```powershell
# Current PowerShell session only
$env:MERGE_SEGMENTS_PERF_LOG = "stdout"
```

The module will emit timing messages such as
`[merge_segments][perf] on_slk_intervals_optimized: duration=12.345000, ...`.

### 6.3. Selecting the Optimized Path

Call `merge.on_slk_intervals_auto` to let the library choose between the
optimized and legacy helpers:

```python
from merge_segments import merge

result = merge.on_slk_intervals_auto(
    target=target,
    data=data,
    join_left=["road"],
    column_actions=actions,
    from_to=("slk_from", "slk_to"),
)
```

Set `prefer_optimized=False` to force the legacy path for a single call, or set
`MERGE_SEGMENTS_DEFAULT_MODE=legacy` (or `optimized`) to configure the default
mode for the current process.

### 6.4. Validating Optimized Outputs

When adopting the optimized path in production, validate it against the legacy
result set before rollout:

```python
from pandas.testing import assert_frame_equal
from merge_segments import merge

# Compare using the legacy parameter
legacy = merge.on_slk_intervals(..., legacy=True)
optimized = merge.on_slk_intervals(..., legacy=False)

# Or compare by calling the functions directly
legacy = merge.on_slk_intervals_legacy(...)
optimized = merge.on_slk_intervals_optimized(...)

assert_frame_equal(
    legacy.sort_index(axis=1),
    optimized.sort_index(axis=1),
    check_dtype=False,
)
```

For large datasets, run this comparison on representative samples or pair it
with the `benchmarks/compare_merges.py` script to generate timing and parity
reports.
