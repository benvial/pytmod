# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod
from __future__ import annotations

import re
import subprocess

from packaging.version import Version


def get_latest_version_tag():
    try:
        # Get all tags from git
        result = subprocess.run(
            ["git", "tag"], capture_output=True, text=True, check=True
        )
        tags = result.stdout.splitlines()

        # Filter tags matching vX.Y.Z format
        version_tags = [tag for tag in tags if re.fullmatch(r"v\d+\.\d+\.\d+", tag)]

        if not version_tags:
            return None

        # Sort tags using Version class from packaging
        return max(version_tags, key=lambda v: Version(v[1:]))
    except subprocess.CalledProcessError:
        return None


if __name__ == "__main__":
    pass
