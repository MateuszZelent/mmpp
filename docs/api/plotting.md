# Plotting Module

```{eval-rst}
.. automodule:: mmpp.plotting
   :members:
   :undoc-members:
   :show-inheritance:
```

## Plotting Classes

### MMPPlotter

```{eval-rst}
.. autoclass:: mmpp.plotting.MMPPlotter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### PlotterProxy

```{eval-rst}
.. autoclass:: mmpp.plotting.PlotterProxy
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration Classes

```{eval-rst}
.. autoclass:: mmpp.plotting.PlotConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Basic Plotting

```python
import mmpp as mp

# Load results
op = mp.open("/path/to/simulations")
results = op.find(solver=3)

# Create plots
results.plot("time", "my")
results.plot("B_ext", "frequency", comp="z")
```

### Advanced Plotting Options

```python
# Customize plot appearance
results.set_style("paper")

# Plot with custom parameters
results.plot(
    "time", "mx",
    figsize=(10, 6),
    title="Magnetization Evolution",
    xlabel="Time (ns)",
    ylabel="Mx Component"
)
```

### Snapshot Visualization

```python
# Create spatial snapshots
result = op[0]
result.matplotlib.snapshot(
    dset="m",
    z=5,
    t=-1,  # Last time step
    repeat=2
)
```
