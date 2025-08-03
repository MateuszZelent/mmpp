"""
CLI module for MMPP - contains all command line interface functionality.

This module provides both modern TUI (Text User Interface) and traditional CLI interfaces:
- Modern TUI: Interactive interface using Textual framework with Dracula theme
- Traditional CLI: Classic argparse-based command line interface

The system automatically selects the best available interface or allows manual selection.
"""

# The main CLI interface is now handled by the parent cli.py module
# This avoids circular imports and simplifies the architecture

# Import traditional CLI as fallback
from .auth import handle_auth_command, login_to_server, logout_from_server, show_auth_status
from .jobs import handle_jobs_command
from .main import main as cli_main

# Import handle_swap_command from swap.py file
from .swap import handle_swap_command as _handle_swap_command_from_file

handle_swap_command = _handle_swap_command_from_file


def main() -> None:
    """
    Main entry point - delegates to the parent cli.py module.
    """
    # Import the main CLI handler from the parent module
    import importlib.util
    import os
    
    # Get the path to cli.py (sibling of cli/ directory)
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    cli_path = os.path.join(parent_dir, "cli.py")
    
    # Load cli.py as a module
    spec = importlib.util.spec_from_file_location("mmpp_cli", cli_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load cli.py from {cli_path}")
    
    cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_module)
    
    # Call main from the loaded module
    return cli_module.main()


__all__ = [
    "main",
    "handle_auth_command",
    "handle_jobs_command", 
    "handle_swap_command",
    "login_to_server",
    "logout_from_server",
    "show_auth_status",
    "cli_main",  # Traditional CLI
]
