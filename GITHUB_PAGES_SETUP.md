# GitHub Pages Documentation Deployment

This repository uses GitHub Actions to automatically build and deploy Sphinx documentation to GitHub Pages.

## Setup Instructions

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Scroll down to **Pages** section
4. Under **Source**, select **Deploy from a branch**
5. Choose **gh-pages** branch
6. Click **Save**

### 2. Configure Repository Permissions

Make sure the GitHub Actions workflow has the necessary permissions:

1. Go to **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select **Read and write permissions**
3. Check **Allow GitHub Actions to create and approve pull requests**
4. Click **Save**

### 3. Push Changes

The documentation will be automatically built and deployed when you push changes to the main branch.

## Manual Build

To build documentation locally:

```bash
cd docs
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
sphinx-build -b html . _build
```

## Accessing Documentation

Once deployed, your documentation will be available at:
`https://<username>.github.io/<repository-name>/`

## Troubleshooting

If the deployment fails:
1. Check the Actions tab for error messages
2. Ensure all dependencies are listed in the workflow
3. Verify that the documentation builds successfully locally
4. Check that the gh-pages branch was created
