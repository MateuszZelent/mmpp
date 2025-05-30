"""
Central logging configuration for MMPP using rich formatting.
"""
import logging
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

# Create a shared console instance for consistent formatting
console = Console()

# Global flag to prevent multiple handler setups
_logging_configured = False

def setup_mmpp_logging(
    debug: bool = False,
    logger_name: str = "mmpp",
    level: Optional[int] = None
) -> logging.Logger:
    """
    Set up rich logging for MMPP with consistent formatting.
    
    Args:
        debug: Enable debug level logging
        logger_name: Name of the logger (e.g., 'mmpp', 'mmpp.fft', 'mmpp.plotting')
        level: Override log level (if None, uses INFO or DEBUG based on debug flag)
    
    Returns:
        Configured logger instance
    """
    global _logging_configured
    
    logger = logging.getLogger(logger_name)
    
    # Configure root mmpp logger only once
    if not _logging_configured and logger_name == "mmpp":
        # Clear any existing handlers
        root_mmpp = logging.getLogger("mmpp")
        for handler in root_mmpp.handlers[:]:
            root_mmpp.removeHandler(handler)
        
        # Set the logger level
        if level is not None:
            root_mmpp.setLevel(level)
        elif debug:
            root_mmpp.setLevel(logging.DEBUG)
        else:
            root_mmpp.setLevel(logging.INFO)
        
        # Create rich handler with consistent formatting
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        
        # Set formatter for the handler with module name
        formatter = logging.Formatter(
            fmt="%(name)s | %(levelname)s | %(message)s",
            datefmt="[%X]"
        )
        rich_handler.setFormatter(formatter)
        
        # Add handler to root mmpp logger
        root_mmpp.addHandler(rich_handler)
        
        # Prevent propagation to avoid duplicate messages
        root_mmpp.propagate = False
        
        _logging_configured = True
    
    # For submodules, just set the level and let them inherit from parent
    if logger_name != "mmpp":
        if level is not None:
            logger.setLevel(level)
        elif debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Ensure propagation is enabled for submodules
        logger.propagate = True
    
    return logger

def get_mmpp_logger(name: str = "mmpp") -> logging.Logger:
    """
    Get an existing MMPP logger or create a basic one if it doesn't exist.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
