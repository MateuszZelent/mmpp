#!/usr/bin/env python3
"""
Script to reload MMPP and test the fix.
"""

import sys
import importlib

def reload_mmpp():
    """Reload MMPP modules to get the latest changes."""
    print("🔄 Reloading MMPP modules...")
    
    # Remove existing modules from cache
    modules_to_reload = [mod for mod in sys.modules.keys() if mod.startswith('mmpp')]
    for mod in modules_to_reload:
        print(f"  Removing {mod} from cache...")
        del sys.modules[mod]
    
    # Import fresh
    import mmpp
    print("✅ MMPP reloaded with latest changes")
    
    # Test the fixed function
    print("\n🧪 Testing fixed install_ffmpeg...")
    ffmpeg_path = mmpp.install_ffmpeg(verbose=False)
    print(f"✅ FFmpeg path: {ffmpeg_path}")
    
    return True

if __name__ == "__main__":
    success = reload_mmpp()
    if success:
        print("\n🎉 MMPP reloaded successfully!")
        print("Now run your animation code again - the error should be fixed.")