#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

import importlib.metadata as metadata

data = metadata.metadata("pytmod")
__version__ = metadata.version("pytmod")
__author__ = data.get("author")
__description__ = data.get("summary")
