# MMPP Documentation

This directory contains the documentation for MMPP built with Sphinx.

## Quick Start

### Building Documentation Locally

1. **Install dependencies:**
   ```bash
   pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
   ```

2. **Build documentation:**
   ```bash
   cd docs
   sphinx-build -b html . _build
   ```

3. **View documentation:**
   Open `_build/index.html` in your browser

### Using the Build Script

We provide a convenience script for building documentation:

```bash
# Build documentation
./build_docs.sh

# Build and serve locally
./build_docs.sh --serve
```

## GitHub Pages Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch.

### Setup Steps:

1. **Enable GitHub Pages:**
   - Go to Repository → Settings → Pages
   - Source: "Deploy from a branch"
   - Branch: "gh-pages"

2. **Configure Workflow Permissions:**
   - Go to Repository → Settings → Actions → General
   - Workflow permissions: "Read and write permissions"
   - Allow GitHub Actions to create pull requests: ✅

3. **Push changes** - documentation will be automatically deployed

### Access Documentation:
Your documentation will be available at:
`https://<username>.github.io/<repository-name>/`

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.md               # Main documentation page
├── api/                   # API reference
│   ├── index.md
│   ├── core.md
│   ├── batch_operations.md
│   ├── plotting.md
│   ├── simulation.md
│   ├── logging_config.md
│   └── fft/
│       ├── index.md
│       └── core.md
├── tutorials/             # User tutorials
│   ├── index.md
│   ├── getting_started.md
│   ├── batch_operations.md
│   └── examples.md
├── _static/              # Static assets
└── _build/               # Generated documentation (git-ignored)
```

## Customization

### Theme Configuration
The documentation uses the Read the Docs theme. You can customize it in `conf.py`:

```python
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'style_nav_header_background': '#2980B9',
    # ... more options
}
```

### Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add it to the `toctree` in the relevant index file
3. Rebuild documentation

### API Documentation
API documentation is automatically generated from docstrings using Sphinx autodoc.

## Troubleshooting

### Common Issues

1. **Import errors during build:**
   - Ensure MMPP package is installed: `pip install -e .`
   - Check that all dependencies are available

2. **GitHub Pages deployment fails:**
   - Check the Actions tab for detailed error messages
   - Verify workflow permissions are set correctly
   - Ensure the gh-pages branch exists

3. **Missing pages:**
   - Check that new pages are added to the appropriate `toctree`
   - Verify file paths are correct

4. **Formatting issues:**
   - MyST Parser handles Markdown with Sphinx extensions
   - Use proper syntax for cross-references and directives

### Getting Help

- Check Sphinx documentation: https://www.sphinx-doc.org/
- MyST Parser docs: https://myst-parser.readthedocs.io/
- Read the Docs theme: https://sphinx-rtd-theme.readthedocs.io/
