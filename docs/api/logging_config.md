# Logging Configuration API

Logging helpers are implemented in `mmpp.cli.logging_config`.

```{eval-rst}
.. automodule:: mmpp.cli.logging_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## Typical Usage

```python
from mmpp.cli.logging_config import setup_mmpp_logging, get_mmpp_logger

setup_mmpp_logging(level=None, debug=False)
log = get_mmpp_logger("mmpp")
log.warning("This is a warning")
```
