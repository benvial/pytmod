# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""This module implements the pytmod API."""

from __future__ import annotations

from .__about__ import __author__, __description__, __version__
from .material import Material
from .slab import Slab

__all__ = ["Material", "Slab", "__author__", "__description__", "__version__"]
