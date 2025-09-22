"""
Simple CLI interface that checks for Textual availability and launches appropriate interface.
"""

import sys
from typing import Any, Dict, List, Optional

# Check for Textual availability
try:
    import textual
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

def main() -> None:
    """Main entry point that selects the best available interface."""
    
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Check for explicit interface selection
    force_classic = "--classic" in args or "--cli" in args
    force_tui = "--tui" in args
    
    # If forcing classic CLI or Textual not available, use traditional CLI
    if force_classic or not TEXTUAL_AVAILABLE:
        if force_classic:
            print("🖥️ Using traditional CLI interface...")
            # Remove the flag before passing to CLI
            sys.argv = [sys.argv[0]] + [arg for arg in args if arg not in ["--classic", "--cli"]]
        else:
            print("📱 Modern TUI not available, using traditional CLI...")
            print("💡 Install Textual for modern interface: pip install textual")
        
        # Import and use traditional CLI
        from .cli.main import main as cli_main
        return cli_main()
    
    # Textual is available, try to use TUI
    try:
        # Import TUI components only when needed
        from .cli_tui import MMPPApp, __version__
        from .cli.logging_config import get_mmpp_logger
        
        log = get_mmpp_logger("mmpp.tui")
        
        # Remove TUI flags if present
        if "--tui" in args:
            sys.argv = [sys.argv[0]] + [arg for arg in args if arg != "--tui"]
        
        # If no arguments or --tui, start the modern TUI
        if not args or force_tui:
            app = MMPPApp()
            log.info(f"Starting MMPP TUI v{__version__}")
            print("🚀 Starting Modern TUI...")
            app.run()
        else:
            # For help and other specific commands, use traditional CLI
            from .cli.main import main as cli_main
            return cli_main()
            
    except ImportError as e:
        print(f"❌ TUI Import Error: {e}")
        print("💡 Falling back to traditional CLI...")
        from .cli.main import main as cli_main
        return cli_main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ TUI Error: {e}")
        print("💡 Try using traditional CLI with: mmpp --classic")
        sys.exit(1)


if __name__ == "__main__":
    main()
