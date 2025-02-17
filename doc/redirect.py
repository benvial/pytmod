#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from get_versions import get_latest_version_tag


content = f"""
<!DOCTYPE html>
<html>
  <head>
    <title>Redirecting to main branch</title>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=./main/index.html" />
    <link rel="canonical" href="https://benvial.github.io/pytmod/{get_latest_version_tag()}/index.html" />
  </head>
</html>
"""


with open("_build/html/redirect.html", "w") as f:
    f.write(content)
