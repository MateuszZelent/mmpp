# Dispersion Analysis

`result.fft.dispersion` provides spin-wave dispersion workflows: `compute_1d`, `plot_dispersion`, filtering, branch tracking, and folded-mode extraction.

## Basic Plot-First Workflow

```python
disp = result.fft.dispersion

fig, ax = disp.plot_dispersion(
    axis="x",
    component="perp",
    kscale="rad_um",
    f_units="GHz",
    fmax=25,
)
```

## Compute and Reuse `DispersionResult1D`

```python
res1d = disp.compute_1d(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    save=True,
    disk_cache=True,
)

print(res1d.shape)
print(res1d.k_range, res1d.f_range)
```

## Configure Once

```python
disp.configure(
    dx=5e-9,
    dt=1e-12,
    component="perp",
    tmax=800,
)
```

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

## Interactive Folded Modes

```python
modes = disp.dispersion_modes(
    result=res1d,
    lattice_constant_nm=470,
)

modes.plot_interactive()

mode = modes.mode(k=2.3, f=1.1)
mode.plot(mode_type="abs")
```

## 2D Dispersion

```python
res2d = disp.compute_2d(component="perp")
print(res2d.shape)
```

Use `compute_2d()` when you need full `S(kx, ky, f)` instead of one propagation axis.
