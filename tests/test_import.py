#!/usr/bin/env python3

try:
    import mmpp as mp

    print("✅ MMPP imported successfully!")

    # Check if open function exists
    if hasattr(mp, "open"):
        print("✅ mp.open() function is available!")
    else:
        print("❌ mp.open() function not found!")

    print(f"📦 MMPP version: {mp.__version__}")
    print(f"👤 Author: {mp.__author__}")

    # Test open function exists and works (without actually calling it)
    print(f"🔧 open function: {mp.open}")
    print(f"📝 open docstring preview: {mp.open.__doc__[:100]}...")

    print("\n✅ Everything looks good! You can now use:")
    print("import mmpp as mp")
    print("db = mp.open('/path/to/your/zarr/files')")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
