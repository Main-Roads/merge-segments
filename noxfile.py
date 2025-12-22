"""Automation sessions for merge_segments contributors."""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "type_check", "tests"]

LINT_TARGETS = ("src", "tests", "benchmarks")


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run format and lint checks."""
    # Install development extras to ensure consistent tool versions
    session.install(".[dev]")
    session.run("python", "-m", "ruff", "format", "--check", *LINT_TARGETS)
    session.run("python", "-m", "ruff", "check", *LINT_TARGETS)


@nox.session(reuse_venv=True)
def type_check(session: nox.Session) -> None:
    """Execute static type checking with mypy."""
    session.install(".[dev]")
    session.run("python", "-m", "mypy", "src/merge_segments")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"], reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run the pytest suite with coverage."""
    session.install("pytest>=8.2", "pytest-cov>=5.0")
    # Install development extras to ensure linters and test helpers are available
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "pytest",
        "--maxfail=1",
        "--cov=src/merge_segments",
        "--cov-report=term-missing",
        "--cov-report=xml",
    )
