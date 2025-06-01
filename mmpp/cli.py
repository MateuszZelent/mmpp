"""
Command Line Interface for MMPP library.
"""

import argparse
import sys


def main() -> None:
    """Main entry point for the mmpp CLI."""
    parser = argparse.ArgumentParser(
        description="MMPP - Micro Magnetic Post Processing Library", prog="mmpp"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show library information")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "info":
        show_info()
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def show_info() -> None:
    """Show library information."""
    from . import __author__, __version__

    print(f"MMPP Library v{__version__}")
    print(f"Author: {__author__}")
    print("A library for Micro Magnetic Post Processing simulation and analysis")


if __name__ == "__main__":
    main()
