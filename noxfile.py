"""Automation sessions for merge_segments contributors."""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "type_check", "tests"]

LINT_TARGETS = ("src", "tests", "benchmarks")


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run format and lint checks."""
    session.install("black>=24.8", "ruff>=0.6.9")
    session.install(".[progress,plotting]")
    session.run("black", "--check", *LINT_TARGETS)
    session.run("ruff", "check", *LINT_TARGETS)


@nox.session(reuse_venv=True)
def type_check(session: nox.Session) -> None:
    """Execute static type checkers."""
    session.install("mypy>=1.10", "pyright>=1.1.379")
    session.install(".[progress,plotting]")
    session.run("mypy", "src/merge_segments")
    session.run("pyright")


@nox.session(python=["3.8", "3.9", "3.10", "3.11"], reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run the pytest suite with coverage."""
    session.install("pytest>=8.2", "pytest-cov>=5.0")
    session.install(".[progress,plotting]")
    session.run(
        "pytest",
        "--maxfail=1",
        "--cov=src/merge_segments",
        "--cov-report=term-missing",
        "--cov-report=xml",
    )
