#!/usr/bin/env python3
"""
MMPP TUI Demo - Demonstracja funkcjonalności nowoczesnego interfejsu

Ten skrypt pokazuje jak programatycznie używać MMPP TUI oraz 
jak zintegrować go z własnym kodem.
"""

import sys
import time
from pathlib import Path

def demo_tui_basic():
    """Podstawowa demonstracja TUI."""
    print("🚀 MMPP TUI Demo - Podstawowe użycie")
    print("=" * 50)
    
    try:
        from mmpp.cli_new import MMPPApp, __version__
        
        print(f"✅ MMPP TUI v{__version__} loaded successfully")
        print("📱 Starting interactive demo...")
        print()
        print("💡 Skróty klawiszowe:")
        print("   - Ctrl+Q: Wyjście")
        print("   - A: Uwierzytelnianie") 
        print("   - J: Zadania")
        print("   - S: Swap parametrów")
        print("   - I: Informacje systemowe")
        print()
        
        # Initialize and run the app
        app = MMPPApp()
        app.run()
        
    except ImportError as e:
        print(f"❌ Textual not available: {e}")
        print("💡 Install with: pip install mmpp[tui]")
        return False
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False
    
    return True


def demo_fallback_cli():
    """Demonstracja automatycznego fallback do traditional CLI."""
    print("🔄 MMPP Fallback Demo - Traditional CLI")
    print("=" * 50)
    
    try:
        from mmpp.cli.main import main as cli_main
        
        print("✅ Traditional CLI loaded successfully")
        print("📋 Available commands:")
        print("   - mmpp info")
        print("   - mmpp auth login")
        print("   - mmpp jobs list") 
        print("   - mmpp swap init")
        print()
        
        # Show info using traditional CLI
        print("📊 Library Information:")
        print("-" * 30)
        
        # Temporarily modify sys.argv to show info
        original_argv = sys.argv[:]
        sys.argv = ["mmpp", "info"]
        
        try:
            cli_main()
        except SystemExit:
            pass  # Expected for CLI
        
        # Restore original argv
        sys.argv = original_argv
        
    except Exception as e:
        print(f"❌ CLI Demo error: {e}")
        return False
    
    return True


def demo_smart_interface():
    """Demonstracja inteligentnego wyboru interfejsu."""
    print("🧠 MMPP Smart Interface Demo")
    print("=" * 50)
    
    try:
        from mmpp.cli import main as smart_main, TUI_AVAILABLE
        
        print(f"🔍 TUI Available: {'✅ Yes' if TUI_AVAILABLE else '❌ No'}")
        
        if TUI_AVAILABLE:
            print("📱 Smart interface will use TUI by default")
            print("💡 Use --classic flag to force traditional CLI")
        else:
            print("🖥️ Smart interface will use traditional CLI")
            print("💡 Install textual for modern TUI: pip install mmpp[tui]")
        
        print()
        print("🎯 Demo: Smart interface selection")
        
        # Show what would happen with different arguments
        test_cases = [
            ([], "Default (auto-select)"),
            (["--tui"], "Force TUI"),
            (["--classic"], "Force CLI"),
            (["auth", "status"], "Specific command"),
        ]
        
        for args, description in test_cases:
            print(f"   {description}: {args or 'no args'}")
        
        print()
        print("▶️ Running smart interface...")
        
        # Run the smart interface
        smart_main()
        
    except Exception as e:
        print(f"❌ Smart interface demo error: {e}")
        return False
    
    return True


def demo_styling():
    """Demonstracja motywu Dracula i stylowania."""
    print("🎨 MMPP Dracula Theme Demo")
    print("=" * 50)
    
    # Show color palette
    colors = {
        "🟣 Purple (Primary)": "#bd93f9",
        "🔵 Cyan (Secondary)": "#8be9fd", 
        "🟢 Green (Success)": "#50fa7b",
        "🟡 Yellow (Warning)": "#f1fa8c",
        "🔴 Red (Error)": "#ff5555",
        "🩷 Pink (Accent)": "#ff79c6",
        "⚫ Background": "#282a36",
        "⚪ Foreground": "#f8f8f2",
    }
    
    print("🎨 Dracula Color Palette:")
    for name, color in colors.items():
        print(f"   {name}: {color}")
    
    print()
    print("✨ Styling Features:")
    print("   - Responsive design")
    print("   - Smooth animations") 
    print("   - Professional typography")
    print("   - Dark theme optimized")
    print("   - High contrast for readability")
    
    # Check if CSS file exists
    css_file = Path(__file__).parent / "mmpp" / "dracula.tcss"
    if css_file.exists():
        print(f"✅ Dracula theme file found: {css_file}")
        print(f"📏 Theme file size: {css_file.stat().st_size} bytes")
    else:
        print("❌ Dracula theme file not found")
    
    return True


def demo_installation():
    """Demonstracja różnych sposobów instalacji."""
    print("📦 MMPP Installation Demo")
    print("=" * 50)
    
    install_options = [
        ("Basic", "pip install mmpp", "Core functionality only"),
        ("TUI", "pip install mmpp[tui]", "With modern Text UI"),
        ("Plotting", "pip install mmpp[plotting]", "With advanced plotting"),
        ("Interactive", "pip install mmpp[interactive]", "With Jupyter support"),
        ("Full", "pip install mmpp[full]", "All features included"),
        ("Development", "pip install mmpp[dev]", "For development"),
    ]
    
    print("💿 Installation Options:")
    for name, command, description in install_options:
        print(f"   {name:12}: {command:25} - {description}")
    
    print()
    print("🎯 Entry Points:")
    print("   mmpp         - Smart interface (auto-select)")
    print("   mmpp-tui     - Force modern TUI")
    print("   mmpp-classic - Force traditional CLI")
    
    print()
    print("🔧 Command Line Flags:")
    print("   --tui        - Use modern TUI interface")
    print("   --classic    - Use traditional CLI interface")
    print("   --cli        - Alias for --classic")
    
    return True


def main():
    """Główna funkcja demo."""
    print("🧲 MMPP Modern TUI - Interactive Demo")
    print("🎨 Professional Magnetic Analysis Interface")
    print("=" * 60)
    print()
    
    demos = [
        ("Basic TUI", demo_tui_basic),
        ("Fallback CLI", demo_fallback_cli),
        ("Smart Interface", demo_smart_interface),
        ("Dracula Theme", demo_styling),
        ("Installation", demo_installation),
    ]
    
    if len(sys.argv) > 1:
        # Run specific demo
        demo_name = sys.argv[1].lower()
        demo_map = {name.lower().replace(" ", ""): func for name, func in demos}
        
        if demo_name in demo_map:
            print(f"🎯 Running specific demo: {demo_name}")
            print()
            success = demo_map[demo_name]()
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Unknown demo: {demo_name}")
            print("Available demos:", list(demo_map.keys()))
            sys.exit(1)
    
    # Interactive menu
    print("📋 Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"   {i}. {name}")
    print("   0. Exit")
    print()
    
    try:
        choice = input("🎯 Select demo (1-5, 0 to exit): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            return
        
        try:
            demo_index = int(choice) - 1
            if 0 <= demo_index < len(demos):
                name, func = demos[demo_index]
                print(f"\n🚀 Running {name} Demo...")
                print()
                success = func()
                
                if success:
                    print(f"\n✅ {name} demo completed successfully!")
                else:
                    print(f"\n❌ {name} demo failed!")
            else:
                print("❌ Invalid choice!")
        except ValueError:
            print("❌ Please enter a number!")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    main()
