# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod
from __future__ import annotations

from importlib import metadata

data = metadata.metadata("pytmod")
__version__ = metadata.version("pytmod")
__author__ = data.get("author")
__description__ = data.get("summary")
