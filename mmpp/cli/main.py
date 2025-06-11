"""
Main CLI entry point for MMPP library.
"""

import argparse
import sys

try:
    import yaml
except ImportError:
    yaml = None


def main() -> None:
    """Main entry point for the mmpp CLI."""
    parser = argparse.ArgumentParser(
        description="MMPP - Micro Magnetic Post Processing Library", prog="mmpp"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show library information")

    # Auth command group
    auth_parser = subparsers.add_parser("auth", help="Server authentication utilities")
    auth_subparsers = auth_parser.add_subparsers(
        dest="auth_command", help="Authentication commands"
    )

    # Auth login command
    auth_login_parser = auth_subparsers.add_parser(
        "login", help="Authenticate with computation server"
    )
    auth_login_parser.add_argument(
        "server_url",
        nargs="?",
        help="Server URL (e.g., https://server.example.com) - will prompt if not provided",
    )
    auth_login_parser.add_argument(
        "token",
        nargs="?",
        help="CLI authentication token - will prompt if not provided",
    )

    # Auth status command
    auth_status_parser = auth_subparsers.add_parser(
        "status", help="Show current authentication status"
    )

    # Auth logout command
    auth_logout_parser = auth_subparsers.add_parser(
        "logout", help="Remove stored authentication credentials"
    )

    # Jobs command group
    jobs_parser = subparsers.add_parser("jobs", help="Job management utilities")
    jobs_subparsers = jobs_parser.add_subparsers(
        dest="jobs_command", help="Job commands"
    )

    # Jobs list command (default)
    jobs_list_parser = jobs_subparsers.add_parser(
        "list", help="List active jobs"
    )
    jobs_list_parser.add_argument(
        "--server", "-s",
        help="Server URL (uses stored credentials if not provided)"
    )

    # Swap command group
    swap_parser = subparsers.add_parser("swap", help="Simulation swapping utilities")
    swap_subparsers = swap_parser.add_subparsers(
        dest="swap_command", help="Swap commands"
    )

    # Swap init command
    swap_init_parser = swap_subparsers.add_parser(
        "init", aliases=["i"], help="Initialize a parms.yml template"
    )
    swap_init_parser.add_argument(
        "template_file",
        nargs="?",
        default="template.mx3",
        help="Template file to analyze (default: template.mx3)",
    )
    swap_init_parser.add_argument(
        "--output",
        "-o",
        default="parms.yml",
        help="Output file name (default: parms.yml)",
    )
    swap_init_parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing file"
    )
    swap_init_parser.add_argument(
        "--prefix", "-p", default="v1", help="Simulation prefix (default: v1)"
    )

    # Swap run command
    swap_run_parser = swap_subparsers.add_parser(
        "run", aliases=["r"], help="Run simulations from config file"
    )
    swap_run_parser.add_argument(
        "config_file",
        nargs="?",
        default="parms.yml",
        help="Path to the configuration file (default: parms.yml)",
    )
    swap_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # Swap info command
    swap_info_parser = swap_subparsers.add_parser(
        "info", help="Show information about config file"
    )
    swap_info_parser.add_argument(
        "config_file",
        nargs="?",
        default="parms.yml",
        help="Path to the configuration file (default: parms.yml)",
    )

    # Swap validate command
    swap_validate_parser = swap_subparsers.add_parser(
        "validate", aliases=["v"], help="Validate config file"
    )
    swap_validate_parser.add_argument(
        "config_file",
        nargs="?",
        default="parms.yml",
        help="Path to the configuration file (default: parms.yml)",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "info":
        show_info()
    elif args.command == "auth":
        from .auth import handle_auth_command
        handle_auth_command(args)
    elif args.command == "jobs":
        from .jobs import handle_jobs_command
        handle_jobs_command(args)
    elif args.command == "swap":
        from .swap import handle_swap_command
        handle_swap_command(args)
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def show_info() -> None:
    """Show library information."""
    from .. import __author__, __version__

    print(f"MMPP Library v{__version__}")
    print(f"Author: {__author__}")
    print("A library for Micro Magnetic Post Processing simulation and analysis")


if __name__ == "__main__":
    main()
