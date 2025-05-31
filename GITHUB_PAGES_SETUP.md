# GitHub Pages Documentation Deployment

This repository provides two GitHub Actions workflows for automatically building and deploying Sphinx documentation to GitHub Pages.

## Workflow Options

### Option 1: Simple Deployment (docs.yml) ⭐ Recommended
Uses `peaceiris/actions-gh-pages` action for straightforward deployment.

**Pros:**
- Simple and reliable
- Works with all repository types
- No special GitHub Pages configuration needed

**Setup:**
1. Enable GitHub Pages: Settings → Pages → Source: "Deploy from a branch" → "gh-pages"
2. Enable Actions permissions: Settings → Actions → General → "Read and write permissions"

### Option 2: Native GitHub Pages (deploy-docs.yml) 
Uses GitHub's native Pages deployment action.

**Pros:**
- Official GitHub Pages deployment method
- Better integration with GitHub Pages environment

**Setup:**
1. Enable GitHub Pages: Settings → Pages → Source: "GitHub Actions"
2. The workflow will automatically deploy to the Pages environment

## Quick Setup Instructions

### 1. Choose Your Workflow
- For most users: Keep `docs.yml` (already configured)
- For advanced users: Use `deploy-docs.yml` and delete `docs.yml`

### 2. Enable GitHub Pages

**For docs.yml workflow:**
1. Go to your repository on GitHub
2. Click **Settings** tab
3. Scroll to **Pages** section
4. Source: **Deploy from a branch**
5. Branch: **gh-pages**
6. Click **Save**

**For deploy-docs.yml workflow:**
1. Go to your repository on GitHub  
2. Click **Settings** tab
3. Scroll to **Pages** section
4. Source: **GitHub Actions**
5. No additional configuration needed

### 3. Configure Repository Permissions

1. Go to **Settings** → **Actions** → **General**
2. Under **Workflow permissions**:
   - Select **Read and write permissions**
   - Check **Allow GitHub Actions to create and approve pull requests**
3. Click **Save**

### 4. Push Changes

The documentation will be automatically built and deployed when you push changes to the main branch.

## Local Development

Build documentation locally:

```bash
# Using the convenience script
./build_docs.sh [--serve]

# Or manually
cd docs
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
pip install -e ..
sphinx-build -b html . _build
```

## Accessing Documentation

Once deployed, your documentation will be available at:
`https://<username>.github.io/<repository-name>/`

## Troubleshooting

### Common Issues

1. **Build fails with "No module named 'mmpp'"**
   - The workflow installs the package with `pip install -e .`
   - Ensure `pyproject.toml` is properly configured

2. **Actions workflow fails**
   - Check Actions tab for detailed error messages
   - Verify all dependencies are listed correctly
   - Ensure the workflow has write permissions

3. **GitHub Pages not updating**
   - Check that the workflow completed successfully
   - Verify GitHub Pages is configured correctly
   - Sometimes it takes a few minutes for changes to appear

4. **Documentation builds locally but fails in CI**
   - Check for missing dependencies in the workflow
   - Ensure all imports work when package is installed fresh

### Workflow Comparison

| Feature | docs.yml | deploy-docs.yml |
|---------|----------|-----------------|
| Setup complexity | Simple | Medium |
| GitHub Pages config | Deploy from branch | GitHub Actions |
| Permissions needed | Read/write | Pages write + ID token |
| Deployment method | peaceiris action | Native GitHub Pages |
| Artifact handling | Direct publish | Upload then deploy |
| Environment | None | github-pages |

## Advanced Configuration

### Custom Domain
Add a `CNAME` file to `docs/_static/` with your domain name.

### Build Optimization
The workflows include:
- Dependency caching for faster builds
- Parallel job execution
- Error handling with `--keep-going`
- Proper permissions and security

### Multiple Branch Support
Both workflows support deployment from `main` or `master` branches.
