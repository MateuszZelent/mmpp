"""
Main CLI entry point for MMPP.

This module provides the main entry point and decides between:
- Modern TUI (if Textual is available) 
- Traditional CLI (fallback)
"""

import sys

def main() -> None:
    """Main entry point that selects the appropriate interface."""
    
    # Check for explicit interface selection
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    force_classic = "--classic" in args or "--cli" in args
    force_tui = "--tui" in args
    
    # Remove interface flags before passing to submodules
    if force_classic:
        sys.argv = [sys.argv[0]] + [arg for arg in args if arg not in ["--classic", "--cli"]]
    elif force_tui:
        sys.argv = [sys.argv[0]] + [arg for arg in args if arg != "--tui"]
    
    # Try TUI first if not explicitly disabled
    if not force_classic:
        try:
            import textual
            from .tui import main as tui_main
            print("🚀 Starting MMPP TUI...")
            return tui_main()
        except ImportError:
            print("📱 Textual not available, using traditional CLI...")
            print("💡 Install with: pip install textual")
        except Exception as e:
            print(f"❌ TUI error: {e}")
            print("💡 Falling back to traditional CLI...")
    
    # Use traditional CLI
    from .cli.main import main as cli_main
    return cli_main()


if __name__ == "__main__":
    main()
