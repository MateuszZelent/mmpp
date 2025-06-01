# Release v0.5.2 - GitHub Actions Cleanup and Workflow Optimization

## 🧹 GitHub Actions Cleanup

### Removed Redundant Workflows
- **Removed `ci.yml`** - Functionality moved to `release.yml` for better consolidation
- **Removed `github-release.yml`** - Redundant with existing release workflow

### Active Workflows
- **`auto-format.yml`** - Automatic code formatting with black and isort
- **`release.yml`** - Comprehensive testing and PyPI publishing
- **`docs.yml`** - Documentation building and GitHub Pages deployment

## 🔧 Workflow Improvements

### Auto-formatting Integration
- Release workflow now automatically formats code before testing
- Eliminates CI failures due to formatting issues
- Consistent code style across the entire codebase

### Streamlined CI/CD Pipeline
- Reduced workflow complexity
- Faster build times with optimized job structure
- Better separation of concerns between workflows

## 📈 Enhanced Development Experience

### Automatic Code Quality
- Black formatting applied automatically on push
- Import sorting with isort
- Consistent code style maintenance

### Simplified Maintenance
- Fewer workflow files to maintain
- Clear responsibility separation
- Reduced redundancy in CI/CD pipeline

## 🚀 What's Next

This release focuses on improving the development experience and CI/CD reliability. The streamlined GitHub Actions setup provides:

- **Reliable builds** - No more formatting-related failures
- **Consistent code style** - Automatic formatting ensures uniformity
- **Simplified maintenance** - Fewer files to manage and update

---

**Full Changelog**: https://github.com/MateuszZelent/mmpp/compare/v0.5.1...v0.5.2
