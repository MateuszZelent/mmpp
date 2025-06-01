#!/usr/bin/env python3
"""
Script to automatically fix common ruff linting issues.
"""

import subprocess
import re
import sys
from pathlib import Path

def run_ruff_check():
    """Run ruff check and return the output."""
    try:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", "mmpp/", "tests/", "scripts/"],
            capture_output=True,
            text=True,
            cwd="/home/kkingstoun/git/mmpp"
        )
        return result.stdout
    except Exception as e:
        print(f"Error running ruff: {e}")
        return ""

def fix_unused_imports():
    """Fix unused imports by running ruff with --fix."""
    try:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", "--fix", "--unsafe-fixes", 
             "--select", "F401", "mmpp/", "tests/", "scripts/"],
            capture_output=True,
            text=True,
            cwd="/home/kkingstoun/git/mmpp"
        )
        print("✅ Fixed unused imports")
        return result.returncode == 0
    except Exception as e:
        print(f"Error fixing imports: {e}")
        return False

def fix_bare_except():
    """Fix bare except clauses."""
    files_to_fix = [
        "tests/example_smart_legend.py",
        "tests/test_fmr_modes.py",
    ]
    
    for file_path in files_to_fix:
        full_path = Path("/home/kkingstoun/git/mmpp") / file_path
        if full_path.exists():
            try:
                content = full_path.read_text()
                # Replace bare except with except Exception
                content = re.sub(r'\bexcept\s*:', 'except Exception:', content)
                full_path.write_text(content)
                print(f"✅ Fixed bare except in {file_path}")
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")

def main():
    print("🔧 Fixing ruff linting issues...")
    
    # Fix unused imports
    print("\n1. Fixing unused imports...")
    fix_unused_imports()
    
    # Fix bare except clauses
    print("\n2. Fixing bare except clauses...")
    fix_bare_except()
    
    # Run final check
    print("\n3. Running final ruff check...")
    output = run_ruff_check()
    
    if output:
        # Count remaining errors
        lines = output.strip().split('\n')
        error_lines = [line for line in lines if 'Found' in line and 'error' in line]
        if error_lines:
            print(f"📊 {error_lines[-1]}")
        else:
            print("✅ No ruff errors found!")
    else:
        print("✅ No ruff errors found!")

if __name__ == "__main__":
    main()
