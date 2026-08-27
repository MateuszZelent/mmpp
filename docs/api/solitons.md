# Skyrmion API

The skyrmion namespace provides topology and radial-size measurements for a
single simulation result.  The stable entry point is
`job.solitons.skyrmion`, where `job` is a `ZarrJobResult` returned by
`mmpp.open(...)`:

```python
import mmpp

job = mmpp.open("/path/to/results")[0]
sk = job.solitons.skyrmion

topology = sk.topology.detect(frame=0, z_layer=0)
same_topology = sk.detect(frame=0, z_layer=0)  # shortcut
size = sk.size.fit(method="auto", frame=0, z_layer=0)

print(topology.Q, topology.center_xy_m)
print(size.radius_m, size.diameter_m, size.model)
```

The namespace is lazy: obtaining `job.solitons.skyrmion` does not materialise
the complete magnetisation array.  Dataset selection, frame, layer, spacing,
and mask are applied when an operation runs.  Lengths in result objects are SI
values in metres; multiply by `1e9` for nm.

```{eval-rst}
.. automodule:: mmpp.solitons.skyrmion
   :members:
   :undoc-members:
   :show-inheritance:
```

## Topology

`sk.topology.detect(...)` and its `sk.detect(...)` shortcut return a
`SkyrmionTopologyResult`.  Useful fields include:

- `Q`: dimensionless integrated topological charge;
- `q_density`: local charge-density array (inverse area when physical `dx`
  and `dy` are supplied);
- `center_xy_m`: estimated centre in metres;
- `polarity`, `core_mz`, `background_mz`, and `contrast_mz`;
- `confidence`, `valid`, and `flags` for diagnostics.

```python
topology = sk.topology.detect(
    frame=12,
    z_layer=0,
    mask=analysis_mask,
)
print(f"Q = {topology.Q:.5g}")
print("centre [nm] =", tuple(value * 1e9 for value in topology.center_xy_m))
print("valid =", topology.valid, "flags =", topology.flags)
```

`Q` is a topology diagnostic, not an object counter.  It is meaningful as the
charge of one isolated, complete object only when the selected field contains
that object and the boundary treatment is appropriate.  Multiple objects can
contribute with opposite signs and cancel; a texture cut by a field boundary
or mask is not a complete object.  The sign also depends on the coordinate
orientation convention.  Record frame, layer, `dx`, `dy`, mask, convention,
and boundary assumptions with a reported value.

## Size measurements and fits

The size namespace is `sk.size`.  The direct convenience methods
`sk.measure_size(...)` and `sk.fit_size(...)` are the corresponding
measurement/fit entry points on the same object:

```python
measured = sk.measure_size(frame=0, z_layer=0, method="threshold")
fitted = sk.fit_size(frame=0, z_layer=0, method="domain_wall")
automatic = sk.size.fit(frame=0, z_layer=0, method="auto")
```

All size results use the same radius convention.  `radius_m` (also exposed as
`radius_50_m`) is `R50`, the radial position where the normalised *contrast*
between the core and background reaches 50%.  An `m_z = 0` crossing is the
same thing only for the ideal `+1/-1` core/background values.  With real
magnetisation, use the measured `core_mz` and `background_mz`; do not assume
that zero magnetisation is the midpoint.  `diameter_m` is exactly
`2 * radius_m` under this definition.  Neither a Gaussian `sigma_m` nor a
domain-wall scale is the radius by default.

The result also reports `radius_90_m`, `radius_10_m`, `wall_width_m`,
`scale_m`, `sigma_m`, `fit_method`, `fit_success`, `normalized_rmse`, `aicc`,
`quality`, and `flags`.  These diagnostics should be retained with any
published or exported size value.

### `method` choices

`sk.size.fit(method=...)` accepts `auto`, `domain_wall`, `ansatz`, `gaussian`,
and `threshold`:

| Method | Interpretation |
| --- | --- |
| `threshold` | Non-parametric radial crossings. It is a useful baseline and does not claim that a profile model is correct. |
| `domain_wall` | Fits a radial domain-wall profile. `scale_m` is its profile length parameter; `wall_width_m` is the physical 10--90 width, `radius_10_m - radius_90_m`. These are different quantities. |
| `ansatz` | Fits the configured skyrmion ansatz. Use it only when that physical profile is justified; report residuals and bounds. |
| `gaussian` | Fits a Gaussian contrast profile and reports `sigma_m`. This is a convenient heuristic for a smooth isolated spot, not a domain-wall measurement. For the normalised form `exp(-r²/(2 sigma²))`, the model-specific relation is `R50 = sigma * sqrt(2 log 2)`. |
| `auto` | Compares available profile candidates with fit diagnostics and corrected Akaike information criterion (AICc). It chooses the best acceptable fit; when candidates are statistically indistinguishable (`delta AICc <= 2`), it prefers the physical circular-wall ansatz, then the radial wall, over the heuristic Gaussian. It is model selection, not proof of physical validity. |

The `ansatz` candidate uses the circular 360-degree wall profile

```text
theta(r) = 2 atan(exp((r - R)/Delta)) + 2 atan(exp((r + R)/Delta))
m_z(r)   = offset + amplitude * (1 - cos(theta(r))) / 2
```

and fits `R`, `Delta`, offset, and amplitude.  `R` is the model radius and
`Delta` is returned as `scale_m`; the public `radius_m` still prefers the
measured R50 crossing when it is available.  This keeps the reported radius
comparable across ansatz, radial-wall, Gaussian, and threshold results.

For a domain-wall profile, the 10--90 width is measured from the radial
profile crossings and should be resolved by several cells.  For a Gaussian,
`sigma_m` is only a scale parameter for the chosen heuristic; do not silently
report it as `R50` or as a wall width.

## Analysis and batch access

The same observable dispatcher is available for one result.  A named
observable returns its native result type, while the no-argument call preserves
the combined topology-and-size workflow:

```python
size = sk.analyze("size", frame=0, z_layer=0, method="auto")
charge = sk.analyze("charge", frame=0, z_layer=0)
combined = sk.analyze(frame=0, z_layer=0, method="auto")

print(charge.Q, size.radius_nm, combined.size.model)
```

The explicit component calls remain `topology.detect`, `measure_size`, and
`size.fit`; keep their frame, layer, mask, spacing, method, and diagnostics
together when comparing results.

For a batch, use `measure_size` through the batch skyrmion namespace:

```python
jobs = mmpp.open("/path/to/results")
results = jobs[:].solitons.skyrmion.measure_size(
    frame=0,
    z_layer=0,
    method="threshold",
)
```

The return value is a tidy pandas `DataFrame`.  It contains job identity,
`status`/`error`, topology, size, selected model, residual, AICc, quality, and
flags.  A failed job remains an `error` row instead of aborting the sweep.  A
batch does not turn a full-field `Q` into an object count: each row still needs
isolated-object, mask, boundary, layer, and resolution checks.

### Generic folder sweeps

`mmpp.open` accepts a whole directory and discovers its `.zarr` results
recursively.  The batch dispatcher separates the analyzed observable from the
simulation parameter used as the sweep axis:

```python
jobs = mmpp.open("/path/to/DMI_sweep")

# Size against DMI; convert J/m² to mJ/m² for the output axis.
size_curve = jobs[:].solitons.skyrmion.analyze(
    "size",
    parameter="Dind",
    parameter_scale=1e3,
    parameter_unit="mJ/m²",
    size_metric="diameter_nm",
    frame=0,
    z_layer=0,
    method="auto",
)

# Topological charge against the same parameter.
charge_curve = jobs[:].solitons.skyrmion.analyze(
    "charge",
    parameter="Dind",
    parameter_scale=1e3,
    parameter_unit="mJ/m²",
    method="berg_luscher",
)

size_curve.plot(x="parameter_value", y="observable_value", marker="o")
charge_curve.plot(x="parameter_value", y="observable_value", marker="o")
```

The canonical names are `"size"` and `"charge"`; charge also accepts
`"topological_charge"`, `"topology"`, and `"q"`.  New observables are added to
the dispatcher rather than as new `*_vs_parameter` methods.  Inspect the live
registry with `available_analyses()`.

Omit `parameter` (or pass an empty string) to select the best varying finite
numeric attribute automatically.  Geometry, sampling, timestamp, index, and
random-seed metadata are excluded.  Candidate inspection is metadata-only and
does not run the analysis:

```python
batch_sk = jobs[:].skyrmion
display(batch_sk.available_analyses())
display(batch_sk.parameter_candidates())

curve = batch_sk.analyze("size", parameter=None, method="auto")

print(curve.attrs["parameter"])
print(curve.attrs["parameter_candidates"])
```

Every dispatched result has the common columns `analysis`, `observable_name`,
`observable_value`, and `observable_unit`, in addition to the detailed
analysis-specific columns.  For size, `size_metric` can select `radius_nm`,
`diameter_nm`, their metre variants, `wall_width_m`, `scale_m`, or `sigma_m`.
For charge, `observable_value` is `Q` and its unit is `1`.

`size_vs_parameter(...)` remains as a compatibility wrapper for
`analyze("size", ...)`.  Calling `analyze()` without an observable preserves
the older combined topology-and-size table without constructing a sweep axis.

If several independent attributes vary, MMPP emits a warning, records all
candidates in `curve.attrs`, and uses the highest-ranked one.  Pass
`parameter="exact_attr_name"` to remove that ambiguity.  Jobs without the
selected attribute remain in the table with `parameter_available=False` and a
missing `parameter_value`; per-job fitting failures likewise remain explicit
`status="error"` rows.

## Notebook helpers

Evaluating the following objects as the last value in a Jupyter cell renders
an interactive MMPP card with Overview/API tabs, live method signatures,
context, examples, and nested accessors:

- `job.solitons` or `job.m.solitons`;
- `job.solitons.skyrmion`;
- `job.solitons.skyrmion.topology` and `.size`;
- `job[:].solitons.skyrmion`;
- `SkyrmionTopologyResult`, `SkyrmionSizeResult`, and
  `SkyrmionAnalysisResult` values.

The `job.solitons` card presents both the existing vortex workflow and the
dedicated skyrmion topology/size workflow.  These cards are discovery aids;
their examples call the same public methods documented above.

## Data-quality limits

Use an explicit `mask` to exclude neighbouring textures, void cells, or a
known sample region.  The mask must describe the selected two-dimensional
layer and must be kept fixed when comparing methods.  A mask edge can truncate
the radial profile just as a field boundary can; treat missing 10--90
crossings and edge-near centres as quality flags, not as successful
measurements.

For multilayer data, select `z_layer` explicitly.  Mixing layers can blur the
core, bias `Q`, and move `R50`.  Verify that physical `dx` and `dy` are
available and that the radial profile contains enough cells across both the
core and domain wall.  A radius close to one or two cell widths is not a
well-resolved physical estimate; repeat on a refined mesh or report the
resolution limitation.

For discussion of skyrmion topological charge and size conventions, see Wang
et al., *Communications Physics* (2018),
[doi:10.1038/s42005-018-0029-0](https://doi.org/10.1038/s42005-018-0029-0),
and the corresponding [arXiv:1801.01745](https://arxiv.org/abs/1801.01745).
