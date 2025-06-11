"""
Authentication command functionality for MMPP CLI.
"""

import argparse
import sys


def handle_auth_command(args: argparse.Namespace) -> None:
    """Handle authentication-related commands."""
    if args.auth_command == "login":
        login_to_server(getattr(args, "server_url", None), getattr(args, "token", None))
    elif args.auth_command == "status":
        show_auth_status()
    elif args.auth_command == "logout":
        logout_from_server()
    elif args.auth_command is None:
        print("Usage: mmpp auth <command>")
        print("Available commands:")
        print("  login         Authenticate with computation server")
        print("  status        Show current authentication status")
        print("  logout        Remove stored authentication credentials")
    else:
        print(f"Unknown auth command: {args.auth_command}")
        sys.exit(1)


def login_to_server(server_url: str = None, token: str = None) -> None:
    """Authenticate with the computation server."""
    try:
        from ..auth import login_to_server as auth_login

        # Get server URL if not provided
        if not server_url:
            print(
                "🌐 Please enter the server URL (e.g., https://containers.example.com):"
            )
            server_url = input("Server URL: ").strip()

        if not server_url:
            print("❌ Server URL cannot be empty")
            sys.exit(1)

        # Get token if not provided
        if not token:
            print("🔑 Please enter your CLI authentication token:")
            token = input("Token: ").strip()

        if not token:
            print("❌ Token cannot be empty")
            sys.exit(1)

        # Use the auth module's login function
        success = auth_login(server_url, token)

        if not success:
            sys.exit(1)

    except ImportError:
        print("❌ Error: Authentication module not available")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Login cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during login: {e}")
        sys.exit(1)


def show_auth_status() -> None:
    """Show current authentication status."""
    try:
        from ..auth import show_auth_status as auth_status

        auth_status()
    except ImportError:
        print("❌ Error: Authentication module not available")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        sys.exit(1)


def logout_from_server() -> None:
    """Remove stored authentication credentials."""
    try:
        from ..auth import logout_from_server as auth_logout

        if not auth_logout():
            sys.exit(1)
    except ImportError:
        print("❌ Error: Authentication module not available")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during logout: {e}")
        sys.exit(1)
