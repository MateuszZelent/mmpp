# Core API

Core objects for scanning simulations and working with single `.zarr` results.

## `mmpp` Top-Level

```{eval-rst}
.. automodule:: mmpp
   :members:
   :undoc-members:
```

## `mmpp.core` Module

```{eval-rst}
.. automodule:: mmpp.core
   :members:
   :undoc-members:
   :show-inheritance:
```

## `MMPP`

```{eval-rst}
.. autoclass:: mmpp.core.MMPP
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __getitem__, __iter__, __len__, __repr__
```

## `ZarrJobResult`

```{eval-rst}
.. autoclass:: mmpp.core.ZarrJobResult
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __getitem__, __getattr__, __repr__
```

## Common Workflow

```python
import mmpp as mp

job = mp.open("/path/to/sims")
subset = job.find(B0=0.12)
result = subset[0]

print(result.datasets)
print(result.get_largest_m_dataset())
```
