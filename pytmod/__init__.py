#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

"""This module implements the pytmod API."""

from .__about__ import __version__, __author__, __description__

from .material import Material
from .slab import Slab

__all__ = ["Material", "Slab"]
