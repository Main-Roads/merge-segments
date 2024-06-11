# `merge_segments`<!-- omit in toc -->

[![Publish to PyPi](https://github.com/thehappycheese/merge-segments/actions/workflows/publish_to_pypi.yml/badge.svg?branch=main)](https://github.com/thehappycheese/merge-segments/actions/workflows/publish_to_pypi.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/merge_segments.svg)](https://pypi.org/project/wideprint)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/merge_segments.svg)](https://pypi.org/project/wideprint)

- [1. Introduction](#1-introduction)
- [2. Install, Upgrade, Uninstall](#2-install-upgrade-uninstall)
- [3. Module `merge`](#3-module-merge)
  - [3.1. Function `merge.on_slk_intervals()`](#31-function-mergeon_slk_intervals)
  - [3.2. Class `merge.Action`](#32-class-mergeaction)
  - [3.3. Class `merge.Aggregation`](#33-class-mergeaggregation)
    - [3.3.1. Notes about `KeepLongest()`](#331-notes-about-keeplongest)
    - [3.3.2. Notes about `LengthWeightedPercentile(...)`](#332-notes-about-lengthweightedpercentile)
  - [3.4. Practical Example of Merge](#34-practical-example-of-merge)
- [4. Notes](#4-notes)
  - [4.1. Correctness, Robustness, Test Coverage and Performance](#41-correctness-robustness-test-coverage-and-performance)

## 1. Introduction

The purpose is to combine two data tables which have a linear segment index ("from" and "to" columns); ie where each row in the input tables represents some linear portion of an entity; for example a road segment from 5km to 10km.

There is an ongoing effort to accelerate and parallelise the merge function
under a new repo called
[megamerge](https://github.com/thehappycheese/megamerge)

## 2. Install, Upgrade, Uninstall

To install:

```powershell
pip install merge_segments
```

To Upgrade:

```powershell
pip install --upgrade merge_segments
```

To show installed version:

```powershell
pip show merge_segments
```

To remove:

```powershell
pip uninstall merge_segments
```

## 3. Module `merge`

### 3.1. Function `merge.on_slk_intervals()`

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

| Parameter      | Type                 | Note                                                                                                                                                                                                                                                                                                              |
| -------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| target         | `pandas.DataFrame`   | The result will have <ul><li>The same number of rows as the `target` data frame</li><li>The same sort-order as the `target` dataframe, and</li><li>each row of the result will match `slk_from` and `slk_to` of the `target` dataframe.</li></ul>                                                                 |
| data           | `pandas.DataFrame`   | Columns from this DataFrame will be aggregated to match the `target` slk segmentation                                                                                                                                                                                                                             |
| join_left      | `list[str]`          | Ordered list of column names to join with.<br>Typically `["road_no","cway"]`.<br>Note:<ul><li>These column names must match in both the `target` and `data` DataFrames</li></ul>                                                                                                                                  |
| column_actions | `list[merge.Action]` | A list of `merge.Action()` objects describing the aggregation to be used for each column of data that is to be added to the target. See examples below.                                                                                                                                                           |
| from_to        | `tuple[str, str]`    | The name of the start and end interval measures.<br>Typically `("slk_from", "slk_to")`.<br>Note:<ul><li>These column names must match in both the `target` and `data` DataFrames</li><li>These columns should be converted to integers for reliable results prior to calling merge (see example below.)</li></ul> |

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
