#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


import subprocess
import re
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
        latest_tag = max(version_tags, key=lambda v: Version(v[1:]))
        return latest_tag
    except subprocess.CalledProcessError as e:
        print("Error executing git command:", e)
        return None


if __name__ == "__main__":
    print(get_latest_version_tag())
