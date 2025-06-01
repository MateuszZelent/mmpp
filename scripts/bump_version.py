#!/usr/bin/env python3
"""Version bumping utilities."""

import re
import sys


def bump_version(bump_type):
    """Bump version in pyproject.toml."""
    with open("pyproject.toml") as f:
        content = f.read()

    version_pattern = r'version = "(\d+)\.(\d+)\.(\d+)"'
    version_match = re.search(version_pattern, content)

    if not version_match:
        print("Version not found in pyproject.toml")
        return

    major, minor, patch = map(int, version_match.groups())

    if bump_type == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    elif bump_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif bump_type == "major":
        new_version = f"{major + 1}.0.0"
    else:
        print(f"Unknown bump type: {bump_type}")
        return

    new_content = re.sub(version_pattern, f'version = "{new_version}"', content)

    with open("pyproject.toml", "w") as f:
        f.write(new_content)

    print(f"Version bumped to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [patch|minor|major]")
        sys.exit(1)

    bump_version(sys.argv[1])
