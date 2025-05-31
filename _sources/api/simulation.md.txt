# Simulation Module

```{eval-rst}
.. automodule:: mmpp.simulation
   :members:
   :undoc-members:
   :show-inheritance:
```

## Simulation Management

### SimulationManager

```{eval-rst}
.. autoclass:: mmpp.simulation.SimulationManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Parameter Sweeps

```python
from mmpp.simulation import SimulationManager
import numpy as np

# Initialize simulation manager
manager = SimulationManager(
    main_path="/path/to/simulations/",
    destination_path="/path/to/results/",
    prefix="sweep_v1"
)

# Define parameter space
params = {
    "b0": np.linspace(0.0001, 0.01, 10),
    "fcut": np.linspace(2.6, 2.8, 20),
}

# Submit all simulations
manager.submit_all_simulations(
    params=params,
    last_param_name="fcut",
    sbatch=True,
    template="template.mx3"
)
```

### Paired Parameter Studies

```python
# For paired parameters (not Cartesian product)
params = {
    "field": [0.1, 0.2, 0.3],
    "freq": [2.0, 2.5, 3.0],  # Must have same length
}

manager.submit_all_simulations(
    params=params,
    last_param_name="freq",
    pairs=True  # Use paired values instead of product
)
```
