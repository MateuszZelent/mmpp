#!/usr/bin/env python3
"""
Quick test for recursive font scanning.
"""
import os
import sys
from pathlib import Path

# Add mmpp to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mmpp

    print("🔍 Testing Font Manager Recursive Scanning")
    print("=" * 50)

    # Show basic info
    print(f"Available fonts: {len(mmpp.fonts.available)}")
    print(f"Font paths: {len(mmpp.fonts.paths)}")

    # Show structure of first few paths
    print("\n📂 Font Directory Structure (limited):")
    for i, path in enumerate(mmpp.fonts.paths[:3]):
        print(f"\n{i+1}. {path}")
        if os.path.exists(path):
            font_count = 0
            try:
                # Quick scan - just top level and one level deep
                for root, dirs, files in os.walk(path):
                    depth = root.replace(path, "").count(os.sep)
                    if depth > 1:  # Limit depth for quick test
                        continue

                    font_files = [
                        f for f in files if f.lower().endswith((".ttf", ".otf"))
                    ]
                    if font_files:
                        indent = "  " * depth
                        folder = os.path.basename(root) if root != path else "[ROOT]"
                        print(f"{indent}📁 {folder}/ - {len(font_files)} fonts")
                        font_count += len(font_files)

                        # Show a few font examples
                        if depth == 0 and font_files:
                            for font in font_files[:2]:
                                print(f"{indent}  └─ {font}")

                print(f"   Total fonts found: {font_count}")
            except Exception as e:
                print(f"   Error scanning: {e}")
        else:
            print("   [Path does not exist]")

    # Test finding specific fonts
    print(f"\n🎯 Font Search Tests:")

    test_fonts = ["arial", "Arial", "dejavu", "liberation"]
    for font_name in test_fonts:
        result = mmpp.fonts.find_font(font_name)
        status = "✓ Found" if result else "✗ Not found"
        print(f"   {font_name}: {status}")
        if result:
            print(f"     Path: {result}")

    print(f"\n✅ Quick test completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
