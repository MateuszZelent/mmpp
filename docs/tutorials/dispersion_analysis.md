# Dispersion Analysis

`result.fft.dispersion` provides spin-wave dispersion workflows: `compute_1d`,
`plot.interactive`, static heatmaps, filtering, branch tracking, and folded-mode
extraction.

## Basic Interactive Workflow

```python
disp = result.m.fft.dispersion

viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    fmax=25,
    analytical="DE",
    model="kalinikos",
    modes=True,
    lattice_constant_nm=470,
    show=False,
)

viewer.show()
```

Without an explicit time limit, the dispersion workflow uses all available time
steps. The dataset slice is part of the public path when you want a quick
preview or a workshop-sized subset. For example,
`result.m[:100, ...].fft.dispersion.plot.interactive(...)` limits the time axis
before the FFT and uses that same selection for cache keys and presets. You can
also pass `plot.interactive(tmin=0, tmax=100)` when a code-level limit is
clearer than a dataset slice.
For an offset window, use index semantics such as
`plot.interactive(tmin=100, tmax=300)`.

Long-running notebook cells report progress by default before and during the
heavy compute stage. If you want a quiet script, pass `progress=False`; if you
want to connect the messages to your own logger or UI, pass a callback:

```python
events = []
viewer = disp.plot.interactive(
    axis="x",
    show=False,
    progress_callback=events.append,
)
```

## Compute and Reuse `DispersionResult1D`

```python
res1d = disp.compute_1d(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    scaling="amplitude_squared",
    save=True,
    disk_cache=True,
)

print(res1d.shape)
print(res1d.k_range, res1d.f_range)
```

## Interactive Preview

Use `disp.plot.interactive(...)` for compute-and-view notebooks, or
`res1d.plot.interactive(...)` when you already have a result. `show=False`
returns a headless controller that can be tested, saved as a preset, or displayed
later with `.show()`.

By default, `.show()` displays the interactive toolbar without forcing a
synchronous Matplotlib draw. Press `Render / refresh dispersion` to draw the
heatmap in the notebook. If you explicitly want the old eager behavior, pass
`initial_render=True`, but the lazy default is safer for VS Code and ipympl
kernels because it avoids blocking startup.

```python
viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    fmax=25,
    analytical="DE",
    modes=True,
    lattice_constant_nm=470,
    show=False,
)

viewer = res1d.plot.interactive(show=False, fmax=25)
viewer.save_preset("/tmp/mmpp-dispersion-viewer.json")
```

The viewer state includes `result_notes`; use it to surface sampling warnings
or raw/display notes in notebook exports without recomputing the FFT.

## Raw and Display Spectra

`DispersionResult1D.S_raw` is the analysis source used by sampling and branch
tracking. `S_display` is the current display view after filters, and `result.S`
remains a backward-compatible alias for the active display array. Use
`analysis_source="display"` only when you intentionally want analysis on a
filtered visual view. When `avg_over_orthogonal=False` stores local spectra,
use `S_local_raw` for analysis-grade local slices and `S_local_display` for
filtered visual local slices; `S_local` remains the active-display alias.

## Configure Once

```python
disp.configure(
    dx=5e-9,
    dt=1e-12,
    component="perp",
)
```

`configure(tmin=..., tmax=...)` is optional and explicit. Leave both as `None`
to use all time steps selected by the dataset accessor.

For reproducible runs, pin the FFT backend and workers either in code or through
environment variables:

```python
disp = result.fft.dispersion(backend="numpy", workers=1)
```

```bash
MMPP_FFT_BACKEND=numpy MMPP_FFT_WORKERS=1 \
python scripts/analysis/verify_fft_dispersion_release_gate.py \
  --output /tmp/mmpp-dispersion-release-gate.json \
  --summary-only
```

The JSON report includes `docs_example_summary`, which should show three
time-selection paths: the full dataset with no hidden `tmax`, an explicit
dataset slice such as `m[:4, ...]`, and an explicit `tmin`/`tmax` compute
window. The top-level `summary` is the quickest status view. The `mode_policy`
section confirms that `modes=True` stores complex
spectra for reconstruction and that spectrum-only results report a clear
`store_complex=True` fallback. The `legacy_adapters` section guards old
notebook paths such as `interactive_analysis(...)` and
`dispersion_modes(...).plot_interactive(...)`. Start from the top-level
`masterplan_contracts` section when checking whether the gate covers the
interactive-dispersion plan; if the gate fails, `masterplan_failures` lists the
failed contract groups first, and `masterplan_failure_details` gives the
concrete failed checks or observed values. Use `recommended_next_steps` as the
short repair checklist for the failed contract groups. `--summary-only` keeps
stdout compact while `--output` still writes the full JSON report.

## Filtering Pipeline

```python
disp.filters(
    remove_static=True,
    window="both",
    post={
        "snr_filter": {"enabled": True, "threshold_snr": 3.0},
    },
    live={
        "gaussian_morph": {"enabled": True, "sigma_f": 1.0, "sigma_k": 1.0},
    },
)

fig, ax = disp.plot_dispersion(axis="x", live_filters={"live": {"gaussian_morph": {"enabled": True}}})
```

## Branch Tracking

```python
import numpy as np

k_path = np.linspace(res1d.k_axis.min(), res1d.k_axis.max(), 300)
branch = disp.track_branch(res1d, k_path=k_path, f_seed=8e9)

fig, (ax1, ax2) = disp.plot_branch(branch, f_units="GHz")
```

## Mode Reconstruction

Spatial mode reconstruction needs the complex spectrum and orthogonal spatial
context. Recompute with `store_complex=True` and `avg_over_orthogonal=False`
before using `.modes.at(...)` or mode-ready interactive panels.

```python
viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    store_complex=True,
    modes=True,
    lattice_constant_nm=470,
    show=False,
)

mode = viewer.mode_at_selection(k_rad_um=2.3, f_ghz=1.1, component="z")
mode.plot.imshow(mode_type="abs")
mode_viewer = mode.plot.interactive(show=False, mode_type="phase")
```

## Legacy Folded-Mode Workflow

`dispersion_modes(...).plot_interactive()` remains available for older
notebooks, but new code should prefer `disp.plot.interactive(modes=True, ...)`
for spectrum exploration, analytical overlays, selection export, and mode
extraction in one window.

```python
# Old
modes = disp.dispersion_modes(result=res1d, lattice_constant_nm=470)
modes.plot_interactive()

mode = modes.mode(k=2.3, f=1.1)
mode.plot(mode_type="abs")

# New
viewer = disp.plot.interactive(modes=True, lattice_constant_nm=470)
mode = viewer.mode_at_selection(k_rad_um=2.3, f_ghz=1.1)
```

## 2D Dispersion

`compute_2d()` is experimental. Use it when you need full `S(kx, ky, f)`, but
do not assume the full 1D raw/display/cache contract yet.

```python
res2d = disp.compute_2d(component="perp")
print(res2d.shape)
slice_x = res2d.slice_1d("kx", k_value=0.0)
```

Use `compute_2d()` when you need full `S(kx, ky, f)` instead of one propagation axis.
