#!/bin/bash

# Script to build documentation locally
# Usage: ./build_docs.sh [--serve]

set -e

echo "🔧 Building MMPP documentation..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

# Install documentation dependencies
echo "📦 Installing documentation dependencies..."
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints linkify-it-py

# Install the package
echo "📦 Installing MMPP package..."
pip install -e .

# Build documentation
echo "🏗️  Building documentation..."
cd docs
rm -rf _build
sphinx-build -b html -W . _build

# Copy .nojekyll file
cp .nojekyll _build/.nojekyll 2>/dev/null || touch _build/.nojekyll

echo "✅ Documentation built successfully!"
echo "📁 Documentation available at: docs/_build/index.html"

# Serve documentation if requested
if [ "$1" = "--serve" ]; then
    echo "🌐 Starting local server..."
    echo "📖 Documentation available at: http://localhost:8000"
    python -m http.server 8000 -d _build
fi
