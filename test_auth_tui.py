#!/usr/bin/env python3
"""
Test script for MMPP TUI Authentication Screen.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tui_auth():
    """Test the TUI authentication screen."""
    try:
        from mmpp.cli_new import MMPPApp
        
        print("🔧 Testing MMPP TUI Authentication...")
        
        # Create TUI app
        app = MMPPApp()
        
        print("✅ TUI app created successfully")
        print("🔄 Starting TUI (press 'a' to go to Auth screen)...")
        print("   Use keyboard shortcuts:")
        print("   - 'a' to go to Auth screen")
        print("   - 'escape' to go back")
        print("   - 'q' to quit")
        print("   - In Auth screen: 'l' for login focus, 's' for status")
        
        # Run the app
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install textual rich")
        return False
    except Exception as e:
        print(f"❌ Error running TUI: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 MMPP TUI Authentication Test")
    print("=" * 50)
    test_tui_auth()
