# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from __future__ import annotations

from pathlib import Path

from get_versions import get_latest_version_tag

last_tag = get_latest_version_tag()

content = f"""
<!DOCTYPE html>
<html>
  <head>
    <title>Redirecting to latest version</title>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=./{last_tag}/index.html" />
    <link rel="canonical" href="https://benvial.github.io/pytmod/{last_tag}/index.html" />
  </head>
</html>
"""


with Path.open("_build/html/index.html", "w") as f:
    f.write(content)
