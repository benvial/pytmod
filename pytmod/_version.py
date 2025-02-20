# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod
from __future__ import annotations

TYPE_CHECKING = False
if TYPE_CHECKING:
    VERSION_TUPLE = tuple[int | str, ...]
else:
    VERSION_TUPLE = object

version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE

__version__ = version = "0.0.2.dev6+g76db6b4.d20250220"
__version_tuple__ = version_tuple = (0, 0, 2, "dev6", "g76db6b4.d20250220")
