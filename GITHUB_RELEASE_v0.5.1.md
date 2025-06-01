# 🐛 MMPP v0.5.1: Python Compatibility Hotfix

## Critical Bug Fix Release

This is a hotfix release that addresses Python version compatibility issues discovered in v0.5.0.

### 🔧 **What's Fixed**

#### 🐍 **Python Version Compatibility** 
- **Fixed minimum Python version**: Updated from Python 3.8 to **Python 3.9+**
- **Root cause**: Library uses modern Python features introduced in 3.9
- **Impact**: Ensures proper installation and functionality across supported platforms

### 📋 **Technical Changes**

- ✅ Updated `requires-python = ">=3.9"` in `pyproject.toml`
- ✅ Updated Python version classifiers (3.9, 3.10, 3.11, 3.12)
- ✅ Updated CI workflows to test against correct Python versions
- ✅ Updated documentation to reflect minimum version requirements

### 🚀 **Installation & Upgrade**

#### New Installation
```bash
# Requires Python 3.9 or higher
pip install mmpp==0.5.1
```

#### Upgrade from v0.5.0
```bash
pip install --upgrade mmpp
```

#### From Source
```bash
git clone https://github.com/MateuszZelent/mmpp
cd mmpp
pip install -e .
```

### ✨ **No Feature Changes**

This is a **drop-in replacement** for v0.5.0 with **zero API changes**:

- 🤖 All auto-selection features work exactly the same
- ⚡ All batch processing capabilities unchanged  
- 🔧 All examples and documentation remain valid
- 📊 All performance characteristics identical

### 🎯 **Compatibility Matrix**

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.8 | ❌ **Not Supported** | Missing required language features |
| 3.9 | ✅ **Supported** | Minimum required version |
| 3.10 | ✅ **Supported** | Fully tested |
| 3.11 | ✅ **Supported** | Fully tested |
| 3.12 | ✅ **Supported** | Fully tested |

### 🔗 **Quick Links**

- **📖 [Documentation](https://MateuszZelent.github.io/mmpp/)**
- **🚀 [Getting Started](https://MateuszZelent.github.io/mmpp/tutorials/getting_started/)**
- **🔬 [API Reference](https://MateuszZelent.github.io/mmpp/api/)**

### 🙏 **Thanks**

Thanks to users who reported the Python 3.8 compatibility issue! This helps us maintain high quality and broad compatibility.

---

**Full Changelog**: [v0.5.0...v0.5.1](https://github.com/MateuszZelent/mmpp/compare/v0.5.0...v0.5.1)
