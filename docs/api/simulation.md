# Simulation API

Simulation-management helpers bundled with `mmpp`.

```{eval-rst}
.. automodule:: mmpp.simulation
   :members:
   :undoc-members:
   :show-inheritance:
```

## Example

```python
from mmpp.simulation import SimulationManager

manager = SimulationManager(
    main_path="/path/to/templates",
    destination_path="/path/to/output",
    prefix="scan_a",
)
```
