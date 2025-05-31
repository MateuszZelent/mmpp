#!/usr/bin/env python3
"""
Test script to verify the package setup.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def test_package_setup():
    """Test the package setup."""
    print("🚀 Testing MMPP2 Package Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Run this script from the package root.")
        return False
    
    tests = [
        ("python -c \"import sys; print(f'Python version: {sys.version}')\"", "Python version check"),
        ("python -c \"import mmpp; print(f'Package version: {mmpp.__version__}')\"", "Package import test"),
        ("python -c \"import mmpp; print(f'Author: {mmpp.__author__}')\"", "Package metadata test"),
        ("python -c \"import mmpp; print(f'Features: {mmpp.__features__}')\"", "Feature availability test"),
    ]
    
    success_count = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{len(tests)} tests passed")
    
    if success_count == len(tests):
        print("🎉 All tests passed! Package setup looks good.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = test_package_setup()
    sys.exit(0 if success else 1)
