"""
Jobs command functionality for MMPP CLI.
Handles job listing and management.
"""

import argparse
import sys
from typing import Any, Optional

import requests

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..auth import AuthManager

console = Console() if RICH_AVAILABLE else None


def handle_jobs_command(args: argparse.Namespace) -> None:
    """Handle jobs-related commands."""
    if args.jobs_command == "list" or args.jobs_command is None:
        list_active_jobs(args.server_url if hasattr(args, 'server_url') else None)
    else:
        print(f"Unknown jobs command: {args.jobs_command}")
        sys.exit(1)


def list_active_jobs(server_url: Optional[str] = None) -> None:
    """List active jobs from the containers_admin2 server."""
    auth_manager = AuthManager()
    
    # Load credentials
    credentials = auth_manager.load_credentials()
    
    if not credentials:
        print("❌ Not authenticated. Please login first:")
        print("   mmpp auth login <server_url> <token>")
        sys.exit(1)
    
    # Use server URL from credentials if not provided
    if not server_url:
        server_url = credentials.get("server_url")
    
    if not server_url:
        print("❌ No server URL available. Please login first.")
        sys.exit(1)
    
    token = credentials.get("token")
    if not token:
        print("❌ No authentication token available. Please login first.")
        sys.exit(1)
    
    print(f"🔍 Fetching active jobs from {server_url}...")
    
    try:
        # Normalize server URL
        if not server_url.startswith(("http://", "https://")):
            server_url = f"https://{server_url}"
        
        # Remove /login suffix if present
        if server_url.endswith("/login"):
            server_url = server_url[:-6]
        
        # Construct API URL
        jobs_url = f"{server_url}/api/v1/jobs/active-jobs"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
        }
        
        response = requests.get(jobs_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            jobs = response.json()
            display_jobs(jobs)
        elif response.status_code == 401:
            print("❌ Authentication failed. Please login again:")
            print("   mmpp auth login")
            sys.exit(1)
        elif response.status_code == 404:
            print("❌ Jobs API not found. Check server URL.")
            sys.exit(1)
        else:
            print(f"❌ Error fetching jobs: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server: {server_url}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def display_jobs(jobs: list[dict[str, Any]]) -> None:
    """Display jobs list in a nice format using Rich if available."""
    if not jobs:
        if RICH_AVAILABLE:
            console.print("📭 [yellow]No active jobs found[/yellow]")
        else:
            print("📭 No active jobs found")
        return
    
    if RICH_AVAILABLE:
        display_jobs_rich(jobs)
    else:
        display_jobs_simple(jobs)


def display_jobs_rich(jobs: list[dict[str, Any]]) -> None:
    """Display jobs using Rich formatting."""
    # Create main table
    table = Table(
        title="🚀 Active Jobs",
        box=box.ROUNDED,
        title_style="bold blue",
        header_style="bold cyan",
        show_lines=True
    )
    
    # Add columns
    table.add_column("Job ID", style="yellow", no_wrap=True)
    table.add_column("Name", style="green", min_width=20)
    table.add_column("User", style="blue", no_wrap=True)
    table.add_column("State", style="bold", no_wrap=True)
    table.add_column("Partition", style="magenta", no_wrap=True)
    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("Time Used", style="white", no_wrap=True)
    table.add_column("Time Left", style="white", no_wrap=True)
    table.add_column("Memory", style="red", no_wrap=True)
    
    for job in jobs:
        # Color state based on value
        state = job.get("state", "UNKNOWN")
        if state == "RUNNING":
            state_colored = "[green]RUNNING[/green]"
        elif state == "PENDING":
            state_colored = "[yellow]PENDING[/yellow]"
        elif state == "FAILED":
            state_colored = "[red]FAILED[/red]"
        else:
            state_colored = f"[white]{state}[/white]"
        
        table.add_row(
            str(job.get("job_id", "N/A")),
            job.get("name", "N/A"),
            job.get("user", "N/A"),
            state_colored,
            job.get("partition", "N/A"),
            job.get("node", "N/A"),
            job.get("time_used", "N/A"),
            job.get("time_left", "N/A"),
            job.get("memory_requested", "N/A"),
        )
    
    console.print(table)
    
    # Show summary
    total_jobs = len(jobs)
    running_jobs = len([j for j in jobs if j.get("state") == "RUNNING"])
    
    summary_text = f"📊 Total: {total_jobs} jobs | 🟢 Running: {running_jobs}"
    console.print(f"\n[dim]{summary_text}[/dim]")
    
    # Show monitoring info if available
    monitored = len([j for j in jobs if j.get("monitoring_active")])
    if monitored > 0:
        console.print(f"[dim]📈 Monitoring active: {monitored} jobs[/dim]")


def display_jobs_simple(jobs: list[dict[str, Any]]) -> None:
    """Display jobs using simple text formatting (fallback when Rich is not available)."""
    print("🚀 Active Jobs")
    print("=" * 80)
    
    # Header
    print(f"{'Job ID':<10} {'Name':<20} {'User':<10} {'State':<10} {'Node':<8} {'Time Used':<12} {'Memory':<8}")
    print("-" * 80)
    
    for job in jobs:
        print(f"{str(job.get('job_id', 'N/A')):<10} "
              f"{job.get('name', 'N/A')[:19]:<20} "
              f"{job.get('user', 'N/A'):<10} "
              f"{job.get('state', 'N/A'):<10} "
              f"{job.get('node', 'N/A'):<8} "
              f"{job.get('time_used', 'N/A'):<12} "
              f"{job.get('memory_requested', 'N/A'):<8}")
    
    print("-" * 80)
    print(f"Total: {len(jobs)} jobs")
    
    running_jobs = len([j for j in jobs if j.get("state") == "RUNNING"])
    print(f"Running: {running_jobs} jobs")
