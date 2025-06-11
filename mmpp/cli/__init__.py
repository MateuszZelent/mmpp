"""
CLI module for MMPP - contains all command line interface functionality.
"""

from .auth import handle_auth_command, login_to_server, logout_from_server, show_auth_status
from .jobs import handle_jobs_command
from .main import main

# Import handle_swap_command from swap.py file
from .swap import handle_swap_command as _handle_swap_command_from_file

handle_swap_command = _handle_swap_command_from_file

__all__ = [
    "main",
    "handle_auth_command",
    "handle_jobs_command",
    "handle_swap_command",
    "login_to_server",
    "logout_from_server",
    "show_auth_status",
]
