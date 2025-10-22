"""Benchmark merge_segments legacy vs optimized implementations.

Usage:
    python benchmarks/compare_merges.py --targets 5000 --data 15000 --groups 5 --repeats 5

The script generates synthetic interval data, executes both merge paths, asserts
that their outputs match, and prints timing statistics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from merge_segments import merge as merge_module


@dataclass
class BenchmarkConfig:
    target_rows: int
    data_rows: int
    groups: int
    repeats: int
    seed: int


def _build_frames(config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[merge_module.Action]]:
    rng = np.random.default_rng(config.seed)
    segment_length = 100
    target_rows_per_group = config.target_rows // config.groups
    data_rows_per_group = config.data_rows // config.groups

    target_records = []
    data_records = []

    for group_idx in range(config.groups):
        road = f"R{group_idx:03d}"
        start = 0
        for _ in range(target_rows_per_group):
            end = start + segment_length
            target_records.append((road, start, end))
            start = end

        total_length = target_rows_per_group * segment_length
        starts = rng.integers(0, total_length - 5, size=data_rows_per_group)
        lengths = rng.integers(5, segment_length * 2, size=data_rows_per_group)
        ends = np.minimum(starts + lengths, total_length)
        # ensure non-zero length
        ends = np.where(ends == starts, ends + 1, ends)

        values = rng.random(size=data_rows_per_group) * 100
        load = rng.random(size=data_rows_per_group) * 10

        for s, e, val, load_val in zip(starts, ends, values, load):
            data_records.append((road, int(s), int(e), float(val), float(load_val)))

    target = pd.DataFrame(target_records, columns=["road", "slk_from", "slk_to"])
    data = pd.DataFrame(
        data_records,
        columns=["road", "slk_from", "slk_to", "value", "load"],
    )

    actions = [
        merge_module.Action(
            "value",
            rename="value_avg",
            aggregation=merge_module.Aggregation.LengthWeightedAverage(),
        ),
        merge_module.Action(
            "value",
            rename="value_pct90",
            aggregation=merge_module.Aggregation.LengthWeightedPercentile(0.90),
        ),
        merge_module.Action(
            "load",
            rename="load_target_sum",
            aggregation=merge_module.Aggregation.SumProportionOfTarget(),
        ),
    ]

    return target, data, actions


def _time_function(func, *args, repeats: int) -> tuple[float, pd.DataFrame]:
    best_elapsed = float("inf")
    best_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_result = result
    assert best_result is not None
    return best_elapsed, best_result


def run_benchmark(config: BenchmarkConfig) -> None:
    target, data, actions = _build_frames(config)
    join_left = ["road"]
    from_to = ("slk_from", "slk_to")

    legacy_time, legacy_result = _time_function(
        merge_module.on_slk_intervals,
        target,
        data,
        join_left,
        actions,
        from_to,
        repeats=config.repeats,
    )

    optimized_time, optimized_result = _time_function(
        merge_module.on_slk_intervals_optimized,
        target,
        data,
        join_left,
        actions,
        from_to,
        repeats=config.repeats,
    )

    assert_frame_equal(legacy_result.sort_index(axis=1), optimized_result.sort_index(axis=1))

    speedup = legacy_time / optimized_time if optimized_time else float("inf")

    print("Benchmark results (best of repeats):")
    print(f"  Legacy   : {legacy_time*1000:.2f} ms")
    print(f"  Optimized: {optimized_time*1000:.2f} ms")
    print(f"  Speedup  : {speedup:.2f}x")


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark merge_segments helpers")
    parser.add_argument("--targets", type=int, default=5000, help="Total target rows")
    parser.add_argument("--data", type=int, default=15000, help="Total data rows")
    parser.add_argument("--groups", type=int, default=5, help="Number of join groups")
    parser.add_argument("--repeats", type=int, default=5, help="Repeat runs and keep best timing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return BenchmarkConfig(
        target_rows=args.targets,
        data_rows=args.data,
        groups=args.groups,
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_benchmark(parse_args())
