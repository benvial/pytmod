# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""Nox sessions."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import nox

nox.needs_version = ">=2024.3.2"
# nox.options.default_venv_backend = "uv|virtualenv"

nox.options.sessions = ["lint", "tests", "docs", "clean"]


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install("-e.[test]")
    session.install("pytest")
    session.run("pytest", *session.posargs)


TAG_REGEX = re.compile(r"^v\d+\.\d+\.\d+$")


@nox.session
def tag(session: nox.Session) -> None:
    """
    Tag versions, ensuring the tag follows the format vX.Y.Z (e.g., v1.2.3).
    Also adds a "latest" tag.
    """
    parser = argparse.ArgumentParser()
    args, posargs = parser.parse_known_args(session.posargs)

    if not posargs:
        session.error("No tag provided! Usage: nox -s tag -- v1.2.3")

    tag_name = posargs[0]

    if not TAG_REGEX.match(tag_name):
        session.error(
            f"Invalid tag format: {tag_name}. Expected format: vX.Y.Z (e.g., v1.2.3)"
        )

    session.log(f"Tagging version: {tag_name}")

    # Run Git commands
    session.run("git", "tag", "-a", tag_name, "-m", "Version {tag_name[1:]}")
    session.run("git", "push", "origin", tag_name)
    session.run("git", "push", "--tags")
    session.run("git", "tag", "-f", "latest", tag_name)
    session.run("git", "push", "origin", "latest", "--force")


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

    output = args.output or f"docs/_build/{args.builder}"

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        output,
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

    if args.versions:
        shared_args += ("-D", "versions=1")

    if serve:
        if args.versions:
            import webbrowser

            session.run("sphinx-multiversion", *shared_args)

            path = Path(output).resolve()  # Convert to absolute path
            webbrowser.open(f"file://{path}/index.html")

        else:
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


@nox.session
def clean(session):
    """Remove build artifacts and temporary files."""
    paths = [
        "build",
        "dist",
        ".venv",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        "builddir",
        "build",
        ".coverage",
        "coverage.xml",
        "coverage.json",
        "docs/_build",
        "docs/examples",
        "docs/autoapi",
        "docs/generated",
        "docs/sg_execution_times.rst",
    ]

    for path in paths:
        if Path.is_dir(Path(path)):
            shutil.rmtree(path, ignore_errors=True)
        elif Path.is_file(Path(path)):
            Path.unlink(Path(path))

    # Remove all __pycache__ directories
    for root, dirs, _ in os.walk("."):
        if "__pycache__" in dirs:
            shutil.rmtree(Path(root) / "__pycache__", ignore_errors=True)
    # Remove all *.egg-info files and directories
    for egg_info in Path().glob("*.egg-info"):
        if Path.is_dir(egg_info):
            shutil.rmtree(egg_info, ignore_errors=True)
        else:
            Path.unlink(egg_info)
    session.log("Cleanup complete!")
