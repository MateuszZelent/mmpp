"""
Central logging configuration for MMPP using rich formatting optimized for dark themes.
"""

import logging
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Create a custom theme optimized for dark backgrounds
dark_theme = Theme(
    {
        "info": "bright_cyan",  # Jasny cyan dla INFO - bardzo widoczny
        "warning": "bright_yellow",  # Jasny żółty dla WARNING - bardzo widoczny
        "error": "bright_red",  # Jasny czerwony dla ERROR - bardzo widoczny
        "critical": "bold bright_red",  # Pogrubiony jasny czerwony dla CRITICAL - bardzo widoczny
        "debug": "bright_magenta",  # Jasny magenta dla DEBUG - lepiej widoczny niż bright_black
        "time": "bright_white",  # Jasny biały dla czasu - bardzo widoczny
        "name": "bright_green",  # Jasny zielony dla nazwy modułu - bardzo widoczny
        "level": "bright_blue",  # Jasny niebieski dla poziomu - bardzo widoczny
        "path": "dim bright_white",  # Przygaszony biały dla ścieżki
    }
)

# Create a shared console instance with dark theme
console = Console(theme=dark_theme, force_terminal=True)

# Global flag to prevent multiple handler setups
_logging_configured = False


def setup_mmpp_logging(
    debug: bool = False,
    logger_name: str = "mmpp",
    level: Optional[int] = None,
    use_dark_theme: bool = True,
) -> logging.Logger:
    """
    Set up rich logging for MMPP with dark theme optimization.

    Args:
        debug: Enable debug level logging
        logger_name: Name of the logger (e.g., 'mmpp', 'mmpp.fft', 'mmpp.plotting')
        level: Override log level (if None, uses INFO or DEBUG based on debug flag)
        use_dark_theme: Use colors optimized for dark backgrounds

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

        # Choose console based on theme preference
        selected_console = console if use_dark_theme else Console()

        # Create rich handler with dark theme optimization
        rich_handler = RichHandler(
            console=selected_console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            keywords=[],  # Disable keyword highlighting to avoid color conflicts
            highlighter=None,  # Disable syntax highlighting
        )

        # Custom formatter that applies colors based on log level
        class DarkThemeFormatter(logging.Formatter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.use_dark_theme = use_dark_theme

            def format(self, record):
                # First format the record with the base formatter
                if self.use_dark_theme:
                    # Apply colors based on level - optimized for dark backgrounds
                    level_colors = {
                        "DEBUG": "[bright_magenta]",  # Zmienione z bright_black na bright_magenta - lepiej widoczny
                        "INFO": "[bright_cyan]",  # Bardzo jasny cyan - doskonale widoczny na ciemnym tle
                        "WARNING": "[bright_yellow]",  # Bardzo jasny żółty - doskonale widoczny
                        "ERROR": "[bright_red]",  # Bardzo jasny czerwony - doskonale widoczny
                        "CRITICAL": "[bold bright_red]",  # Pogrubiony jasny czerwony - bardzo widoczny
                    }

                    level_color = level_colors.get(record.levelname, "[bright_white]")
                    time_color = "[bright_white]"  # Bardzo jasny biały dla czasu
                    name_color = (
                        "[bright_green]"  # Bardzo jasny zielony dla nazwy modułu
                    )

                    # Format the time properly
                    asctime = self.formatTime(record, self.datefmt)

                    # Create formatted message with colors
                    formatted = f"{time_color}{asctime}[/] | {name_color}{record.name}[/] | {level_color}{record.levelname}[/] | {record.getMessage()}"
                else:
                    # Use standard formatting for light theme
                    asctime = self.formatTime(record, self.datefmt)
                    formatted = f"{asctime} | {record.name} | {record.levelname} | {record.getMessage()}"

                return formatted

        formatter = DarkThemeFormatter(datefmt="%H:%M:%S")
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


def reset_logging_config():
    """
    Reset logging configuration to allow reconfiguration with different settings.
    Useful for switching between light and dark themes.
    """
    global _logging_configured
    _logging_configured = False

    # Clear all MMPP loggers
    root_mmpp = logging.getLogger("mmpp")
    for handler in root_mmpp.handlers[:]:
        root_mmpp.removeHandler(handler)


def configure_for_dark_theme():
    """
    Convenience function to configure logging optimized for dark terminals/themes.
    """
    reset_logging_config()
    return setup_mmpp_logging(debug=False, use_dark_theme=True)


def configure_for_light_theme():
    """
    Convenience function to configure logging optimized for light terminals/themes.
    """
    reset_logging_config()
    return setup_mmpp_logging(debug=False, use_dark_theme=False)


# Default configuration for dark theme (most common in development environments)
def get_default_logger(name: str = "mmpp", debug: bool = False) -> logging.Logger:
    """
    Get a logger with default dark theme configuration.

    Args:
        name: Logger name
        debug: Enable debug logging

    Returns:
        Configured logger instance
    """
    if not _logging_configured:
        setup_mmpp_logging(debug=debug, logger_name="mmpp", use_dark_theme=True)

    return get_mmpp_logger(name)
