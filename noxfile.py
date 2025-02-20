# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""Nox sessions."""

from __future__ import annotations

import argparse

import nox

nox.needs_version = ">=2024.3.2"
# nox.options.default_venv_backend = "uv|virtualenv"

nox.options.sessions = ["lint", "tests"]


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install("-e.[test]")
    session.install("pytest")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving. First positional argument is the target directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--builder",
        dest="builder",
        default="html",
        help="Build target (default: html)",
    )
    parser.add_argument(
        "-v",
        "--versions",
        dest="versions",
        action="store_true",
        default=False,
        help="Build multiple versions (default: False)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_false",
        default=True,
        help="Build gallery (default: True)",
    )
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e.[doc]", "sphinx-autobuild")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )
    autobuild_args = (
        "--open-browser",
        "--port=8009",
        "--ignore",
        "*.html",
    )

    if not args.plot:
        shared_args += ("-D", "plot_gallery=0")

    if serve:
        session.run(
            "sphinx-autobuild",
            *autobuild_args,
            *shared_args,
        )
    else:
        cmd = "sphinx-multiversion" if args.versions else "sphinx-build"
        session.run(cmd, "--keep-going", *shared_args)


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )
