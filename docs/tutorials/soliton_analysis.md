# Skyrmion analysis

This tutorial measures the topology and radial size of one isolated skyrmion
in a simulation result.  It assumes a magnetisation field with a final vector
axis `(mx, my, mz)`, physical cell sizes `dx` and `dy`, and a layer that can be
analysed as a two-dimensional field.

## Open a result

```python
import mmpp

jobs = mmpp.open("/path/to/results")
job = jobs[0]
sk = job.solitons.skyrmion
```

The namespace is lazy.  It does not load the complete magnetisation merely by
being accessed.  If a dataset must be selected explicitly, use the
dataset-aware view provided by the result, then keep that choice fixed for all
comparisons:

```python
sk = job.m.solitons.skyrmion
```

## Detect topology

Call `topology.detect` for the full result or `detect` as its shortcut:

```python
topology = sk.topology.detect(frame=0, z_layer=0)

print("Q =", topology.Q)
print("centre [nm] =", tuple(v * 1e9 for v in topology.center_xy_m))
print("polarity =", topology.polarity)
print("confidence =", topology.confidence)
print("flags =", topology.flags)

# Same operation through the shortcut:
topology_again = sk.detect(frame=0, z_layer=0)
```

`Q` is dimensionless.  `q_density` is the local density and has inverse-area
units when physical spacing is used.  Treat `Q` as the charge diagnostic for
one isolated object, not as an automatic skyrmion counter.  Several textures
can cancel, and an object intersecting the field or mask boundary is
incomplete.  Compare frames only with the same layer, mask, spacing,
coordinate convention, and boundary treatment.

## Measure a contrast-50 radius

Start with the non-parametric baseline:

```python
measured = sk.measure_size(
    frame=0,
    z_layer=0,
    method="threshold",
)

print("R50 [nm] =", measured.radius_m * 1e9)
print("diameter [nm] =", measured.diameter_m * 1e9)
print("quality =", measured.quality, measured.flags)
```

The documented radius is `R50`, the radial location where the normalised
contrast between the measured core `core_mz` and background `background_mz`
falls to 50%.  For ideal core/background values `+1` and `-1`, this is the
`m_z=0` crossing.  Real profiles need not have those extrema, so `m_z=0` is
not a universal radius definition.  The diameter is exactly `2 * R50`.

## Fit the radial profile

Use `size.fit` for a model fit; `fit_size` is the direct convenience alias:

```python
fit = sk.size.fit(
    frame=0,
    z_layer=0,
    method="domain_wall",
)

print("model =", fit.model)
print("R50 [nm] =", fit.radius_m * 1e9)
print("diameter [nm] =", fit.diameter_m * 1e9)
print("domain scale [nm] =", None if fit.scale_m is None else fit.scale_m * 1e9)
print("10--90 width [nm] =", None if fit.wall_width_m is None else fit.wall_width_m * 1e9)
print("normalised RMSE =", fit.normalized_rmse)

# Equivalent convenience call:
fit_again = sk.fit_size(frame=0, z_layer=0, method="domain_wall")
```

The domain-wall `scale_m` is the profile length parameter.  The `wall_width_m`
value is the physical distance between the 90% and 10% contrast crossings:
`radius_10_m - radius_90_m`.  Neither quantity replaces `radius_m`.

The available methods are:

| Method | Use |
| --- | --- |
| `threshold` | Robust baseline from radial contrast crossings; sensitive to noise, binning, and incomplete backgrounds. |
| `domain_wall` | Radial domain-wall model; inspect `scale_m` and `wall_width_m` separately. |
| `ansatz` | Configured analytical skyrmion profile; use only when its physical assumptions apply. |
| `gaussian` | Smooth-profile heuristic; `sigma_m` is a fit scale, not automatically the radius. |
| `auto` | Candidate model selection using AICc and fit diagnostics. |

For the normalised Gaussian contrast `c(r)=exp(-r²/(2 sigma²))`, the
model-specific conversion is `R50 = sigma * sqrt(2 log 2)`.  This conversion
does not make Gaussian `sigma_m` a domain-wall width.  Use the returned
`radius_50_m`/`radius_m` for the documented radius.

The analytical `ansatz` fit uses a circular 360-degree domain wall,
`theta(r) = 2 atan(exp((r-R)/Delta)) + 2 atan(exp((r+R)/Delta))`, with free
offset and amplitude.  It reports `Delta` as `scale_m`.  Treat this as a
physics-motivated effective profile: retain the measured R50, fitted scale,
residual, AICc, and flags rather than collapsing them into one number.

## Compare models with `auto`

```python
fit = sk.size.fit(frame=0, z_layer=0, method="auto")

print({
    "model": fit.model,
    "radius_nm": fit.radius_m * 1e9 if fit.radius_m is not None else None,
    "diameter_nm": fit.diameter_m * 1e9 if fit.diameter_m is not None else None,
    "AICc": fit.aicc,
    "normalised_RMSE": fit.normalized_rmse,
    "quality": fit.quality,
    "flags": fit.flags,
})
print("candidate diagnostics =", fit.candidate_diagnostics)
```

`auto` ranks acceptable candidate profile fits with corrected Akaike
information criterion (AICc), while fit residuals and flags indicate whether
the result is usable.  Within `delta AICc <= 2`, the models are treated as
statistically indistinguishable and the selector prefers `ansatz`, then
`domain_wall`, then `gaussian`.  A lower AICc is not proof that the selected
model is physically exact.  If candidates are close, report the ambiguity and
retain the threshold measurement as a model-independent comparison.

## Masks, layers, boundaries, and resolution

Pass a two-dimensional boolean mask when neighbouring textures, void cells, or
sample geometry would contaminate the radial profile:

```python
fit = sk.size.fit(
    frame=12,
    z_layer=0,
    mask=analysis_mask,
    method="auto",
)
```

Use the same mask for topology and size, and record it with the result.  A
mask edge can truncate an object or its radial profile, just like a field
boundary.  Reject or flag centres too close to either edge, especially when
`radius_90_m` or `radius_10_m` is missing.

For multilayer results, choose `z_layer` explicitly.  Mixing layers can blur
the object and bias both `Q` and `R50`.  Confirm physical `dx` and `dy`; do not
silently interpret pixels as metres.  The core and wall need several cells to
resolve the contrast crossings.  If the radius is only one or two cell widths,
repeat on a refined mesh or report that the estimate is resolution-limited.

## Batch measurements

The batch skyrmion namespace exposes `measure_size` for applying the same
measurement to each selected job:

```python
batch_results = jobs[:].solitons.skyrmion.measure_size(
    frame=0,
    z_layer=0,
    method="threshold",
)

print(batch_results[["path", "status", "radius_nm", "model", "quality"]])
failed = batch_results[batch_results["status"] == "error"]
```

Use a consistent dataset, layer, mask, spacing, and method across the batch.
Do not interpret a batch of full-field `Q` values as an object count without
the same isolated-object and boundary checks for every job.

### One folder as a parameter sweep

The folder itself can be the batch input.  `mmpp.open` scans nested `.zarr`
results, while the generic `analyze` dispatcher selects what is measured and
which simulation attribute forms the sweep axis:

```python
jobs = mmpp.open("/data/skyrmion_D_sweep")

# Manual sweep-axis selection and size observable.
dmi_curve = jobs[:].skyrmion.analyze(
    "size",
    parameter="Dind",
    parameter_scale=1e3,
    parameter_unit="mJ/m²",
    size_metric="diameter_nm",
    frame=0,
    z_layer=0,
    method="auto",
)

ax = dmi_curve.plot(
    x="parameter_value",
    y="observable_value",
    marker="o",
    xlabel=dmi_curve.attrs["parameter_label"],
    ylabel="skyrmion diameter [nm]",
)

# Reuse the same sweep machinery for topological charge Q.
q_curve = jobs[:].skyrmion.analyze(
    "charge",
    parameter="Dind",
    parameter_scale=1e3,
    parameter_unit="mJ/m²",
    method="berg_luscher",
)
```

For automatic selection, leave the argument empty.  It is useful to inspect
the fast metadata-only decision before running all fits:

```python
batch_sk = jobs[:].solitons.skyrmion
display(batch_sk.available_analyses())
display(batch_sk.parameter_candidates())

auto_curve = batch_sk.analyze(
    "size",
    parameter=None,
    frame=0,
    z_layer=0,
    method="auto",
)
print(auto_curve.attrs["parameter"])
```

The common output columns are `analysis`, `observable_name`,
`observable_value`, `observable_unit`, and `parameter_value`.  Detailed size
or topology columns remain beside `path`, `status`, and `error`.  Do not
drop error or low-quality rows before checking whether failures correlate with
the sweep parameter; that correlation can itself reveal a stability boundary
or a resolution problem.

For a single result, the dispatcher returns the native result of the selected
observable.  The no-argument call remains the combined result for backward
compatibility:

```python
size = sk.analyze("size", frame=0, z_layer=0, method="auto")
charge = sk.analyze("charge", frame=0, z_layer=0)
combined = sk.analyze(frame=0, z_layer=0, method="auto")

print(charge.Q, size.radius_nm, combined.size.model)
```

Keep explicit calls to `detect`, `measure_size`, and `size.fit` in
reproducible scripts and record frame, layer, mask, method, `Q`, fit
diagnostics, and quality flags alongside plots or exported tables.

### Interactive inspection

For a single simulation, open the combined snapshot/topology/profile viewer:

```python
dashboard = job[0].skyrmion.interactive(
    initial_frame=0,
    z_layer=0,
    initial_module="analysis",
    topology_method="berg_luscher",
    size_method="auto",
)
```

Switch the same dashboard to the spectrum or spatial-mode view, or open the
combined FFT viewer directly:

```python
job[0].skyrmion.interactive(initial_module="spectrum")
job[0].skyrmion.interactive(initial_module="modes")
job[0].skyrmion.interactive_spectrum(dpi=140)
job[0].vortex.interactive_modes(dpi=140)
```

For a folder, select one job for visual inspection while retaining the generic
batch tables for the complete sweep:

```python
jobs[:].skyrmion.interactive(index=0, sort_by="Dind")
curve = jobs[:].skyrmion.analyze("size", parameter="Dind")
```

Vortex core-gyration spectra have a callable notebook helper.  Evaluating it
without parentheses displays usage; calling it computes the spectrum:

```python
job[0].vortex.spectrum.gyration
spec = job[0].vortex.spectrum.gyration(method="welch")
job[0].vortex.spectrum.gyration.interactive()
```

To inspect spatial magnetisation modes at the detected gyration frequency,
use the linked FFT workflow:

```python
job[0].vortex.spectrum.gyration.interactive_modes()
mode = job[0].vortex.spectrum.gyration.mode()
mode.plot.imshow(component="z", value="magnitude")
```

This is distinct from `job[0].vortex.modes`, which classifies peaks using the
core trajectory and therefore returns labels/frequencies rather than complex
spatial mode maps.

For background on skyrmion topological charge and size conventions, see Wang
et al., *Communications Physics* (2018),
[doi:10.1038/s42005-018-0029-0](https://doi.org/10.1038/s42005-018-0029-0),
and the corresponding [arXiv:1801.01745](https://arxiv.org/abs/1801.01745).
