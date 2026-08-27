#!/usr/bin/env python3
"""Version bumping utilities.

The runtime version in ``mmpp/__init__.py`` is the canonical version source.
"""

import re
import sys
from pathlib import Path


def bump_version(bump_type):
    """Bump the canonical version and the documentation display version."""
    init_path = Path("mmpp/__init__.py")
    with init_path.open(encoding="utf-8") as f:
        content = f.read()

    version_pattern = r'__version__ = "(\d+)\.(\d+)\.(\d+)"'
    version_match = re.search(version_pattern, content)

    if not version_match:
        raise SystemExit("Version not found in mmpp/__init__.py")

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

    new_content = re.sub(
        version_pattern, f'__version__ = "{new_version}"', content, count=1
    )

    with init_path.open("w", encoding="utf-8") as f:
        f.write(new_content)

    conf_path = Path("docs/conf.py")
    if conf_path.exists():
        conf_content = conf_path.read_text(encoding="utf-8")
        conf_content = re.sub(
            r'^(release|version) = "\d+\.\d+\.\d+"$',
            lambda match: f'{match.group(1)} = "{new_version}"',
            conf_content,
            flags=re.MULTILINE,
        )
        conf_path.write_text(conf_content, encoding="utf-8")

    print(f"Version bumped to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [patch|minor|major]")
        sys.exit(1)

    bump_version(sys.argv[1])
