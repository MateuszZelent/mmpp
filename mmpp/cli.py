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
        sys.argv = [sys.argv[0]] + [
            arg for arg in args if arg not in ["--classic", "--cli"]
        ]
    elif force_tui:
        sys.argv = [sys.argv[0]] + [arg for arg in args if arg != "--tui"]

    # Use TUI only if:
    # 1. No arguments (interactive mode) OR
    # 2. Explicitly requested with --tui
    should_use_tui = (not args and not force_classic) or force_tui

    if should_use_tui:
        try:
            from mmpp.tui import main as tui_main

            print("🚀 Starting MMPP TUI...")
            return tui_main()
        except ImportError:
            print("📱 Textual not available, using traditional CLI...")
            print("💡 Install with: pip install textual")
        except Exception as e:
            print(f"❌ TUI error: {e}")
            print("💡 Falling back to traditional CLI...")

    # Use traditional CLI for specific commands
    from mmpp.cli.main import main as cli_main

    return cli_main()


if __name__ == "__main__":
    main()
