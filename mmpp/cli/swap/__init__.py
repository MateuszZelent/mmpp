"""
MMPP Swap module for simulation parameter swapping and management.
"""

from .simulation import SimulationManager, SimulationSwapper, TemplateParser

# Import handle_swap_command from the parent module
def handle_swap_command(args):
    """Handle swap-related commands - forwarded to parent module."""
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    
    # Get parent directory and import the swap.py module
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    try:
        # Import the swap module from parent
        import swap as swap_module
        return swap_module.handle_swap_command(args)
    finally:
        # Clean up sys.path
        if str(parent_dir) in sys.path:
            sys.path.remove(str(parent_dir))

__all__ = ["SimulationManager", "SimulationSwapper", "TemplateParser", "handle_swap_command"]
