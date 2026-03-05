# mmpp Development Guide

This guide describes the current local workflow for developing `mmpp`.

## Prerequisites
- Python 3.9+
- `pip`
- `git`
- Optional: `just` command runner (`cargo install just` or `pip install just-install`)

## Quick setup

```bash
git clone https://github.com/mateuszzelent/mmpp.git
cd mmpp
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[interactive,plotting,tui]"
```

## Daily workflow

If you use `just`:

```bash
just format
just lint
just test
```

Manual equivalents:

```bash
python -m ruff format mmpp/ tests/ scripts/
python -m ruff check mmpp/ tests/ scripts/
python -m mypy mmpp/
python -m pytest tests/ -q
```

## Documentation

Build docs locally:

```bash
just docs
# or
cd docs && sphinx-build -b html . _build --keep-going
```

## Useful `just` commands
- `just install-dev` - install editable package with dev dependencies
- `just format` - run ruff formatter
- `just lint` - run ruff + mypy
- `just test` - run pytest with coverage
- `just build` - build sdist + wheel
- `just check` - validate built package metadata
- `just prepare-release` - format + lint + test + build + check
- `just release-test` - publish to TestPyPI
- `just release` - publish to PyPI

## CI/CD expectations

The GitHub workflows enforce:
- style checks (`ruff format --check`, `ruff check`),
- type checks (`mypy`),
- test suite (`pytest`),
- strict release verification before publishing.

Do not rely on release workflows to auto-fix code formatting.

## Release process

1. Update version metadata and release notes.
2. Run locally:

```bash
just prepare-release
```

3. Create and push a tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

4. Release workflow publishes to PyPI (or TestPyPI via manual dispatch).

## Troubleshooting

- If imports fail, verify your virtualenv is active and dependencies are installed.
- If docs fail with MyST link errors, install `linkify-it-py`.
- If type checks fail after refactors, fix annotations instead of silencing checks.
