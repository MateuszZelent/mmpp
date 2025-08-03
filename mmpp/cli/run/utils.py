"""
Utility functions for MMPP Run module.
"""

try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Initialize console
console = Console() if RICH_AVAILABLE else None


def safe_print(message: str) -> None:
    """Print with rich formatting if available, fallback to plain print."""
    if RICH_AVAILABLE and console:
        console.print(message)
    else:
        print(message)


def safe_print_panel(content: str, title: str, border_style: str = "blue") -> None:
    """Print a panel with rich formatting if available, fallback to plain print."""
    if RICH_AVAILABLE and console:
        console.print(Panel(content, title=title, border_style=border_style))
    else:
        print(f"\n{title}\n{'-' * len(title)}\n{content}\n")


def safe_print_table(table) -> None:
    """Print a rich table if available, fallback to plain print."""
    if RICH_AVAILABLE and console:
        console.print(table)
    else:
        # For fallback, we'll just print a basic representation
        print("Table output (rich not available)")
        print(str(table))
