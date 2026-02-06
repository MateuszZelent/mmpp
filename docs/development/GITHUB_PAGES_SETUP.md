# GitHub Pages Deployment (`mateuszzelent.github.io/mmpp`)

This repository deploys documentation from the workflow:

- `.github/workflows/docs.yml`

It now works in two stages:

1. `build-docs` on every `push` and `pull_request` to `main/master`
2. `deploy-docs` only on `push` to `main/master`

The published URL is:

- `https://mateuszzelent.github.io/mmpp/`

## Required Repository Settings

1. Open `Settings -> Pages`
2. Set source to `GitHub Actions`

1. Open `Settings -> Actions -> General`
2. Ensure workflow permissions allow deployments:
   `Read and write permissions`

## What Is Deployed

- Sphinx output from `docs/_build`
- `.nojekyll` file is included to avoid Jekyll processing

## Local Build

```bash
pip install -e .[dev]
pip install linkify-it-py
sphinx-build -b html docs docs/_build
```

Open:

- `docs/_build/index.html`
