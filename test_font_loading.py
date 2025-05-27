#!/usr/bin/env python3
"""
Test script to verify font management functionality works correctly.
Tests the new mmpp.fonts interface.
"""
import os
import sys
from pathlib import Path

# Add mmpp to path  
sys.path.insert(0, str(Path(__file__).parent))

def test_font_manager():
    """Test the new font manager functionality."""
    print("🔤 Testing MMPP2 Font Management")
    print("=" * 40)
    
    try:
        import mmpp
        
        print("✓ Successfully imported mmpp")
        
        # Test basic font manager access
        print(f"\n📋 Font manager info:")
        print(f"   {mmpp.fonts}")
        
        # Test available fonts
        available_fonts = mmpp.fonts.available
        print(f"\n🔍 Available fonts: {len(available_fonts)} found")
        if available_fonts:
            print(f"   Sample fonts: {', '.join(available_fonts[:5])}")
            if len(available_fonts) > 5:
                print(f"   ... and {len(available_fonts) - 5} more")
        
        # Test font search paths
        print(f"\n📁 Font search paths:")
        for i, path in enumerate(mmpp.fonts.paths):
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"   {i+1}. {exists} {path}")
        
        # Test current default font
        print(f"\n🎨 Current default font: {mmpp.fonts.default_font}")
        
        # Test font search
        print(f"\n🔍 Testing font search:")
        arial_fonts = mmpp.fonts.find_font("arial")
        print(f"   Arial-like fonts: {arial_fonts}")
        
        dejavu_fonts = mmpp.fonts.find_font("dejavu")
        print(f"   DejaVu-like fonts: {dejavu_fonts}")
        
        # Test recursive directory scanning
        print(f"\n📂 Testing recursive directory scanning:")
        mmpp.fonts.show_font_structure(max_depth=2)
        
        # Test finding fonts by family name
        available = mmpp.fonts.available
        if available:
            test_family = available[0]
            print(f"\n🎯 Testing find by family name: '{test_family}'")
            found_path = mmpp.fonts.find_font(test_family)
            print(f"   Found at: {found_path}")
        
        # Test adding a custom path (create temp directory)
        temp_font_dir = "/tmp/test_fonts"
        print(f"\n📂 Testing add_path with: {temp_font_dir}")
        
        # Create test directory
        try:
            os.makedirs(temp_font_dir, exist_ok=True)
            result = mmpp.fonts.add_path(temp_font_dir)
            print(f"   Add path result: {result}")
            
            # Clean up
            try:
                os.rmdir(temp_font_dir)
            except:
                pass
                
        except Exception as e:
            print(f"   Could not test add_path: {e}")
        
        # Test setting default font
        print(f"\n🎨 Testing set_default_font:")
        if available_fonts:
            test_font = available_fonts[0]
            print(f"   Attempting to set font to: {test_font}")
            result = mmpp.fonts.set_default_font(test_font)
            print(f"   Set font result: {result}")
            print(f"   New default font: {mmpp.fonts.default_font}")
        else:
            print(f"   No fonts available to test with")
        
        # Test refresh functionality
        print(f"\n🔄 Testing font refresh:")
        refresh_count = mmpp.fonts.refresh()
        print(f"   Fonts processed during refresh: {refresh_count}")
        
        print(f"\n✅ Font manager test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during font testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_old_font_functions():
    """Test the old individual font functions for backward compatibility."""
    print(f"\n🔧 Testing backward compatibility:")
    
    try:
        from mmpp.plotting import setup_custom_fonts, load_paper_style
        
        print("✓ Old font functions still importable")
        
        # Test old functions
        font_success = setup_custom_fonts(verbose=True)
        print(f"   setup_custom_fonts(): {font_success}")
        
        style_success = load_paper_style(verbose=True)
        print(f"   load_paper_style(): {style_success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing old functions: {e}")
        return False

if __name__ == "__main__":
    print("🧪 MMPP2 Font Management Test Suite")
    print("=" * 50)
    
    success1 = test_font_manager()
    success2 = test_old_font_functions()
    
    if success1 and success2:
        print(f"\n🎉 All font tests passed!")
        print(f"\n💡 Usage examples:")
        print(f"   import mmpp")
        print(f"   print(mmpp.fonts)              # Show available fonts")
        print(f"   print(mmpp.fonts.paths)        # Show search paths")
        print(f"   mmpp.fonts.add_path('/my/fonts')  # Add font directory")
        print(f"   mmpp.fonts.set_default_font('Arial')  # Set default font")
        print(f"   fonts = mmpp.fonts.find_font('arial')  # Search fonts")
    else:
        print(f"\n💥 Some font tests failed.")
