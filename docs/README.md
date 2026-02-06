# Documentation (Sphinx + MyST)

This directory contains the project documentation.

## Build Locally

```bash
pip install -e .[dev]
cd docs
sphinx-build -b html . _build
```

Open `docs/_build/index.html` in a browser.

## Main Structure

- `index.md`: main landing page
- `tutorials/`: task-oriented guides
- `api/`: API reference based on autodoc

### Tutorials

- `tutorials/getting_started.md`
- `tutorials/fft_spectrum_analysis.md`
- `tutorials/dispersion_analysis.md`
- `tutorials/batch_operations.md`
- `tutorials/examples.md`

### API

- `api/core.md`
- `api/batch_operations.md`
- `api/plotting.md`
- `api/simulation.md`
- `api/logging_config.md`
- `api/fft/*`

## Notes

- API pages rely on importable Python modules during Sphinx build.
- If autodoc fails, verify optional dependencies and your active environment.
