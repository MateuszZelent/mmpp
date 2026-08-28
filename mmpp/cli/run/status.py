"""
Cluster status checking functionality for MMPP Run module.
"""

import sys

import requests

try:
    from rich import box
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ...auth import AuthManager
from ..logging_config import get_mmpp_logger
from .utils import safe_print, safe_print_panel, safe_print_table

# Initialize logger
log = get_mmpp_logger("mmpp.cli.run.status")


def check_cluster_status() -> None:
    """Check and display cluster status and available resources."""
    try:
        # Initialize auth manager
        auth_manager = AuthManager()

        # Get authentication token
        token = auth_manager.get_token()
        if not token:
            print("❌ Authentication required. Please run 'mmpp auth login' first.")
            sys.exit(1)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Get base URL from auth manager
        base_url = auth_manager.get_base_url()
        if not base_url:
            print("❌ Server URL not configured. Please run 'mmpp auth login' first.")
            sys.exit(1)

        safe_print("🔍 [bold blue]Checking cluster status...[/bold blue]")

        # Get cluster information
        cluster_url = f"{base_url}/api/v1/cluster/stats"
        response = requests.get(cluster_url, headers=headers, timeout=30)

        if response.status_code == 200:
            cluster_data = response.json()
            _display_cluster_info(cluster_data)
        elif response.status_code == 401:
            print("❌ Authentication failed. Please run 'mmpp auth login' again.")
            sys.exit(1)
        elif response.status_code == 403:
            print("❌ Access denied. You don't have permission to view cluster status.")
            sys.exit(1)
        else:
            print(f"❌ Failed to get cluster status: {response.status_code}")
            if response.text:
                try:
                    error_data = response.json()
                    if "message" in error_data:
                        print(f"Error: {error_data['message']}")
                    else:
                        print(f"Response: {error_data}")
                except requests.JSONDecodeError:
                    safe_print(f"[red]Response: {response.text}[/red]")
            sys.exit(1)

    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        sys.exit(1)
    except (KeyError, ValueError) as e:
        log.error("Cluster status check error: %s", e)
        print(f"❌ Error processing cluster data: {e}")
        sys.exit(1)
    except Exception as e:
        log.error("Unexpected error during cluster status check: %s", e)
        print(f"❌ Unexpected error checking cluster status: {e}")
        sys.exit(1)


def _display_cluster_info(cluster_data: dict) -> None:
    """Display cluster information in a formatted way."""

    # Extract summary data from the stats response
    total_nodes = cluster_data.get("total_nodes", 0)
    free_nodes = cluster_data.get("free_nodes", 0)
    busy_nodes = cluster_data.get("busy_nodes", 0)
    sleeping_nodes = cluster_data.get("sleeping_nodes", 0)

    total_gpus = cluster_data.get("total_gpus", 0)
    free_gpus = cluster_data.get("free_gpus", 0)
    busy_gpus = cluster_data.get("busy_gpus", 0)
    active_gpus = cluster_data.get("active_gpus", 0)
    standby_gpus = cluster_data.get("standby_gpus", 0)

    timestamp = cluster_data.get("timestamp", "unknown")

    # Create nodes summary table
    if RICH_AVAILABLE:
        nodes_table = Table(title="🖥️ Cluster Nodes Summary", box=box.ROUNDED)
        nodes_table.add_column("Status", style="cyan", no_wrap=True)
        nodes_table.add_column("Count", justify="right", style="green")
        nodes_table.add_column("Percentage", justify="right")

        # Add node status rows
        nodes_table.add_row(
            "Free",
            str(free_nodes),
            f"{(free_nodes / total_nodes * 100):.1f}%" if total_nodes > 0 else "0%",
        )
        nodes_table.add_row(
            "Busy",
            str(busy_nodes),
            f"{(busy_nodes / total_nodes * 100):.1f}%" if total_nodes > 0 else "0%",
        )
        nodes_table.add_row(
            "Sleeping",
            str(sleeping_nodes),
            f"{(sleeping_nodes / total_nodes * 100):.1f}%" if total_nodes > 0 else "0%",
        )
        nodes_table.add_row("Total", str(total_nodes), "100%")

        # Create GPUs summary table
        gpus_table = Table(title="🔥 GPU Resources Summary", box=box.ROUNDED)
        gpus_table.add_column("Status", style="cyan", no_wrap=True)
        gpus_table.add_column("Count", justify="right", style="yellow")
        gpus_table.add_column("Percentage", justify="right")

        # Add GPU status rows
        gpus_table.add_row(
            "Free",
            str(free_gpus),
            f"{(free_gpus / total_gpus * 100):.1f}%" if total_gpus > 0 else "0%",
        )
        gpus_table.add_row(
            "Busy",
            str(busy_gpus),
            f"{(busy_gpus / total_gpus * 100):.1f}%" if total_gpus > 0 else "0%",
        )
        gpus_table.add_row(
            "Active",
            str(active_gpus),
            f"{(active_gpus / total_gpus * 100):.1f}%" if total_gpus > 0 else "0%",
        )
        gpus_table.add_row(
            "Standby",
            str(standby_gpus),
            f"{(standby_gpus / total_gpus * 100):.1f}%" if total_gpus > 0 else "0%",
        )
        gpus_table.add_row("Total", str(total_gpus), "100%")
    else:
        nodes_table_data = [
            "🖥️ Cluster Nodes Summary:",
            f"  Free: {free_nodes} ({(free_nodes / total_nodes * 100):.1f}%)"
            if total_nodes > 0
            else f"  Free: {free_nodes} (0%)",
            f"  Busy: {busy_nodes} ({(busy_nodes / total_nodes * 100):.1f}%)"
            if total_nodes > 0
            else f"  Busy: {busy_nodes} (0%)",
            f"  Sleeping: {sleeping_nodes} ({(sleeping_nodes / total_nodes * 100):.1f}%)"
            if total_nodes > 0
            else f"  Sleeping: {sleeping_nodes} (0%)",
            f"  Total: {total_nodes}",
        ]

        gpus_table_data = [
            "🔥 GPU Resources Summary:",
            f"  Free: {free_gpus} ({(free_gpus / total_gpus * 100):.1f}%)"
            if total_gpus > 0
            else f"  Free: {free_gpus} (0%)",
            f"  Busy: {busy_gpus} ({(busy_gpus / total_gpus * 100):.1f}%)"
            if total_gpus > 0
            else f"  Busy: {busy_gpus} (0%)",
            f"  Active: {active_gpus} ({(active_gpus / total_gpus * 100):.1f}%)"
            if total_gpus > 0
            else f"  Active: {active_gpus} (0%)",
            f"  Standby: {standby_gpus} ({(standby_gpus / total_gpus * 100):.1f}%)"
            if total_gpus > 0
            else f"  Standby: {standby_gpus} (0%)",
            f"  Total: {total_gpus}",
        ]

    # Display tables
    if RICH_AVAILABLE:
        safe_print_table(nodes_table)
        safe_print("")
        safe_print_table(gpus_table)
    else:
        for line in nodes_table_data:
            print(line)
        print()
        for line in gpus_table_data:
            print(line)
        print()

    # Calculate availability percentages
    available_nodes = free_nodes
    available_gpus = free_gpus
    online_nodes = total_nodes - sleeping_nodes  # Free + Busy nodes are online

    # Create summary
    status_info = f"""
Online Nodes: {online_nodes}/{total_nodes} ({(online_nodes / total_nodes * 100):.1f}% online)
Available Nodes: {available_nodes}/{total_nodes} ({(available_nodes / total_nodes * 100):.1f}% free)
Available GPUs: {available_gpus}/{total_gpus} ({(available_gpus / total_gpus * 100):.1f}% free)
Busy GPUs: {busy_gpus}/{total_gpus} ({(busy_gpus / total_gpus * 100):.1f}% busy)
Last updated: {timestamp}
Source: {cluster_data.get("source", "unknown")}
"""

    safe_print_panel(status_info.strip(), "📊 Cluster Summary", "blue")
