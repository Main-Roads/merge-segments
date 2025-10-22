"""Public package interface for merge_segments.

Exposes the primary merge helpers and aggregation utilities so callers can
import them directly from :mod:`merge_segments` instead of navigating the
internal module layout.
"""

from importlib.metadata import PackageNotFoundError, version

from .merge import (  # noqa: F401 - re-exported for convenience
    Action,
    Aggregation,
    configure_performance_logger,
    on_slk_intervals,
    on_slk_intervals_auto,
    on_slk_intervals_fallback,
    on_slk_intervals_optimized,
)

try:
    __version__ = version("merge_segments")
except PackageNotFoundError:  # pragma: no cover - during local editing
    __version__ = "0.0.0"

__all__ = [
    "Action",
    "Aggregation",
    "configure_performance_logger",
    "on_slk_intervals",
    "on_slk_intervals_auto",
    "on_slk_intervals_fallback",
    "on_slk_intervals_optimized",
    "__version__",
]
