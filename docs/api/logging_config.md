# Logging Configuration

```{eval-rst}
.. automodule:: mmpp.logging_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## Logging Functions

```{eval-rst}
.. autofunction:: mmpp.logging_config.setup_mmpp_logging

.. autofunction:: mmpp.logging_config.get_mmpp_logger

.. autofunction:: mmpp.logging_config.get_default_logger
```

## Usage Examples

### Basic Logging Setup

```python
from mmpp.logging_config import setup_mmpp_logging, get_mmpp_logger

# Setup logging with dark theme optimization
setup_mmpp_logging(level="INFO", dark_theme=True)

# Get logger for your module
log = get_mmpp_logger("my_module")

log.info("This is an info message")
log.warning("This is a warning")
log.error("This is an error")
```

### Advanced Configuration

```python
# Setup with custom formatting
setup_mmpp_logging(
    level="DEBUG",
    dark_theme=True,
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get logger with emoji support
log = get_mmpp_logger("mmpp.analysis")
log.info("🚀 Starting analysis...")
log.success("✅ Analysis completed successfully!")
```
