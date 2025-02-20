# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""This module implements the pytmod API."""

from __future__ import annotations

from importlib import metadata

from ._version import __version__
from .material import Material
from .slab import Slab

_data = metadata.metadata("pytmod")
__author__ = _data.get("author")
__description__ = _data.get("summary")

__all__ = ["Material", "Slab", "__author__", "__description__", "__version__"]
