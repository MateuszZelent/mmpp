# Current-driven vortex dynamics with the Thiele model

MMPP provides a reduced collective-coordinate model for the motion of a
magnetic-vortex core in a circular nanodisk. It is useful for interpreting
thresholds, transient relaxation, nonlinear frequency shifts, and the
gyrotropic response spectrum without running a full LLGS simulation for every
parameter point.

The output is a core trajectory, not a spatial magnetization field. The model
therefore resolves the gyrotropic mode and modulation sidebands, but it cannot
predict spin-wave mode profiles, vortex nucleation, core reversal, or
annihilation.

The exhaustive public-API contract, unit table, failure behavior, and autodoc
are in [Current-driven vortex Thiele API](../api/vortex_thiele.md). This page
focuses on complete analysis workflows.

## What the reduced model gives you

The most useful output is not one fitted frequency. It is a cheap, continuous
map from a drive waveform and calibrated device parameters to the collective
vortex-core response.

| Question | Model output | Important condition |
| --- | --- | --- |
| Does a DC bias damp or pump the orbit? | linear growth and `J_threshold` | CPP sign convention and polarity must be correct |
| What steady orbit is expected? | normalized radius \(u_0\) and \(f(J)\) | only below the configured edge boundary |
| How does startup look? | \(X(t)\), \(Y(t)\), radius, phase, velocity | retain a physical seed or thermal noise |
| What happens under \(J(t)\)? | transient response, carrier, harmonics, sidebands | `dt` and duration must resolve all relevant frequencies |
| Which current reaches a target frequency? | bounded one-dimensional optimization | calibrated model, not a device guarantee |
| Which parameters explain a current sweep? | multi-current fit of \(\omega_0\), \(N\), optional slopes/scales | several distinct steady-orbit points |
| How sensitive is the result? | rapid parameter and waveform scans | validate selected points with micromagnetics |

This makes the model particularly effective as a surrogate for experiment
design. Use it to screen currents, frequencies, pulse lengths, modulation
depths, polarities, and calibration hypotheses. Then spend full LLGS runtime
on a smaller set of representative and boundary points.

## CPP: DC bias plus time-dependent current

The reduced CPP equation uses normalized core position
`s = (X/R, Y/R)` and the convention

\[
\dot{\mathbf{s}} = [\chi(J)-d(u)\omega(u,J)]\mathbf{s}
                    +p\omega(u,J)\,\hat{z}\times\mathbf{s},
\]

where `p` is core polarity and `u = |s|`. With positive material polarization
and the MMPP reduced-current convention, positive current pumps `p=-1`; always
inspect `model.J_threshold` instead of guessing the sign.

```python
import numpy as np
import matplotlib.pyplot as plt

from mmpp.analytical import (
    CPPThieleModel,
    DiskGeometry,
    MaterialParams,
    current_ac,
    omega0_novosad,
)

material = MaterialParams(
    Ms=800e3,       # A/m
    alpha=0.013,
    P=0.45,
    A=10e-12,       # J/m
)
geometry = DiskGeometry(R=128e-9, L=9e-9)

# This helper is a thin-disk asymptote. Prefer a calibrated omega0 when known.
omega0 = omega0_novosad(material, geometry)
model = CPPThieleModel(
    material,
    geometry,
    omega0=omega0,
    N=0.30,
    polarity=-1,
    domega0_dJ=0.0,  # fit Oersted/current shift when available
)

J_dc = 1.30 * model.J_threshold
J_of_t = current_ac(
    J_amp=0.15 * abs(J_dc),
    f_hz=200e6,
    J_offset=J_dc,
)

trajectory = model.simulate(
    t_span=(0.0, 80e-9),
    s0=(1e-3, 0.0),
    J_func=J_of_t,
    dt=10e-12,
    clamp_u=0.95,
)

# Reject startup before calculating the response spectrum.
frequency_hz, relative_power = trajectory.compute_spectrum(
    transient_fraction=0.60,
    signal="x",
    window="hann",
)

plt.plot(frequency_hz * 1e-9, relative_power)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Relative core-motion power")
plt.show()
```

The spectrum can contain the nonlinear gyrotropic carrier, harmonics, and
sidebands separated by the modulation frequency. It is relative FFT power of a
core coordinate, not a calibrated voltage PSD. Electrical output requires an
explicit magnetoresistance/readout model.

Any finite scalar callback is accepted as `J_func(t)`, for example a pulse or
chirp:

```python
def current_chirp(t):
    J0 = J_dc
    amplitude = 0.10 * abs(J_dc)
    f0 = 50e6
    chirp_rate = 5e15  # Hz/s
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    return J0 + amplitude * np.sin(phase)
```

### Interpret a modulated spectrum

For a sinusoidally modulated current, the response can contain:

- the nonlinear gyrotropic carrier;
- sidebands separated from the carrier by the modulation frequency;
- coordinate harmonics caused by a noncircular or field-shifted orbit;
- a low-frequency component from slow amplitude relaxation or chirping.

The following helper extracts the largest non-DC bins without pretending that
each bin is an independent physical eigenmode:

```python
non_dc = frequency_hz > 0.0
candidate_indices = np.argsort(relative_power[non_dc])[-8:][::-1]
candidate_frequency = frequency_hz[non_dc][candidate_indices]
candidate_power = relative_power[non_dc][candidate_indices]

for f_hz, power in zip(candidate_frequency, candidate_power):
    print(f"{f_hz * 1e-9:8.4f} GHz  relative power={power:.5g}")
```

Closely spaced bins from one broadened peak should not be counted as separate
modes. Increase the observation time to improve frequency resolution, and use
peak prominence/distance rules when automating a sweep.

### Choose the sampling and observation window

Three scales matter:

1. `dt` sets the Nyquist frequency \(f_\mathrm{Nyquist}=1/(2dt)\);
2. the analyzed window \(T\) sets bin spacing \(\Delta f\approx1/T\);
3. the startup rejection determines whether relaxation leaks into the
   stationary response.

For example, resolving 10 MHz sideband spacing requires substantially more
than 100 ns of stationary signal. Sampling a 1 GHz carrier at 10 ps gives 100
samples per period, but it does not by itself provide fine spectral
resolution. Run a short convergence study:

```python
settings = [
    (20e-12, 80e-9),
    (10e-12, 80e-9),
    (10e-12, 160e-9),
]

for dt_value, duration in settings:
    trial = model.simulate(
        t_span=(0.0, duration),
        s0=(1e-3, 0.0),
        J_func=J_of_t,
        dt=dt_value,
        clamp_u=0.95,
    )
    f_hz, p_rel = trial.compute_spectrum(transient_fraction=0.6)
    peak = f_hz[np.argmax(p_rel[1:]) + 1]
    print(dt_value, duration, peak)
```

Convergence of one peak location is not enough when amplitude, sideband power,
or linewidth is the reported observable; check each reported quantity.

For a perfectly centered core and a perpendicular CPP polarizer, the
Slonczewski term is multiplicative anti-damping. A deterministic trajectory at
exactly `s=(0,0)` therefore remains centered. Use a physically justified small
initial displacement or the thermal SDE solver:

```python
thermal = model.simulate_sde(
    t_span=(0.0, 80e-9),
    s0=(0.0, 0.0),
    J_func=J_of_t,
    dt=10e-12,
    temperature_k=300.0,
    seed=7,
    clamp_u=0.95,
)
```

The default stochastic diffusion follows the small-signal Thiele
fluctuation-dissipation relation. It is not a replacement for cell-wise
thermal LLGS close to core reversal or expulsion.

## CIP: translational drive

For current in the disk plane, MMPP implements the Zhang-Li generalized Thiele
equation. Here the spin-drift velocity acts as a direct translational drive.

```python
from mmpp.analytical import CIPThieleModel, current_pulse

cip = CIPThieleModel(
    material,
    geometry,
    omega0=omega0,
    polarity=1,
    current_dir=(1.0, 0.0),
)
cip_trajectory = cip.simulate(
    t_span=(0.0, 20e-9),
    r0=(1e-9, 0.0),
    J_func=current_pulse(4e10, t_on=1e-9, t_off=5e-9),
    dt=5e-12,
)
```

## Calibrate a current sweep and select an operating point

A quantitative CPP model should normally be calibrated from several
independent steady-state points, not from one trajectory. Suppose a MuMax3 or
experimental sweep provides current density and gyrotropic frequency:

```python
from mmpp.analytical import fit_omega0_N_to_fJ

J_data = np.array([1.8e10, 2.1e10, 2.4e10, 2.7e10, 3.0e10])
f_data_hz = np.array([0.51e9, 0.54e9, 0.58e9, 0.62e9, 0.67e9])

fit = fit_omega0_N_to_fJ(
    J_data,
    f_data_hz,
    material=material,
    geom=geometry,
    polarity=-1,
    initial_omega0=omega0,
    initial_N=0.25,
    fit_domega0_dJ=True,
)

print("success:", fit.success, fit.status)
print("f0 [GHz]:", fit.omega0 / (2 * np.pi * 1e9))
print("N:", fit.N)
print("RMSE [MHz]:", fit.rmse_hz * 1e-6)
fit.plt.frequency_vs_current()
```

Keep `fit.valid_mask` and the residuals with the parameters. A fit can converge
mathematically while being physically unusable because its current interval is
too narrow, points are edge limited, or the torque/current convention is
wrong.

Construct the calibrated model and solve for a target:

```python
calibrated = CPPThieleModel(
    material,
    geometry,
    omega0=fit.omega0,
    N=fit.N,
    polarity=-1,
    domega0_dJ=fit.domega0_dJ,
    chi_scale=fit.chi_scale,
)

optimum = calibrated.optimize_current_for_target_frequency(
    target_frequency_hz=0.63e9,
    J_bounds=(J_data.min(), J_data.max()),
)

print("J* [A/m²]:", optimum.current_density_a_per_m2)
print("predicted f [GHz]:", optimum.predicted_frequency_ghz)
print("mismatch [MHz]:", optimum.objective_value_hz * 1e-6)
```

Keep the optimization inside the calibrated current interval. Validate the
selected point independently before treating it as a device operating point.

## MuMax3 and field-resolved use

Use `mmpp.solitons.vortex.model.thiele.cpp(...)` or
`field_resolved_cpp(...)` when parameters should be resolved from a job. The
adapter understands MuMax3 `Pol`, `Lambda`, `epsilonprime`, fixed-layer
position, torque thickness, and polarizer direction. For `Lambda != 1`, pass a
representative `mean_m_dot_p`; the full cell-wise angular efficiency cannot be
recovered from `p_z` alone.

Geometry and current density are quantitative inputs. MMPP prefers explicit
`R`, `D`, or `Area` metadata and refuses to turn a failed automatic current
lookup into silent zero current. A radius inferred from the mesh is marked by a
warning because it assumes the disk fills the smaller box dimension.

### Dataset-aware notebook workflow

In a notebook, evaluate the namespace or a callable node without parentheses
to inspect its live helper:

```python
# Overview / Validity / API card for all current-driven models.
job[0].vortex.model.thiele

# Overview / API callable card with the current live signature.
job[0].vortex.model.thiele.cpp
```

Build and inspect an adapter before running:

```python
adapter = job[0].vortex.model.thiele.cpp(
    polarity="auto",
    N=0.30,
    mean_m_dot_p=0.0,
)

# Overview / Validity / API card: resolved R, L, Ms, alpha, f0,
# threshold, dataset, and torque convention.
adapter

# The underlying direct model has its own live card.
adapter.model

canonical_trajectory = adapter.simulate(
    t_span=(0.0, 80e-9),
    J_func=J_of_t,
    dt=10e-12,
    s0=(1e-3, 0.0),
)
```

The adapter result follows the common soliton `TrajectoryResult` contract. Use
the direct `adapter.model.simulate(...)` when the specialized
`ThieleTrajectoryResult.compute_spectrum(...)` and its dedicated helper card
are desired.

### Recommended MuMax3 qualification sequence

1. Run a zero-current, small-displacement relaxation and extract the
   small-signal gyrotropic frequency.
2. Run several DC currents on both sides of the expected threshold. Track
   polarity, orbit radius, frequency, and any edge/reversal event.
3. Record `Pol`, `Lambda`, `epsilonprime`, fixed-layer vector and position,
   torque thickness, current sign, and the averaging definition used for
   `mean_m_dot_p`.
4. Fit the reduced model only to stable, post-transient, non-edge-limited
   points.
5. Validate a held-out DC point and at least one time-dependent waveform.
6. Repeat one selected point with smaller cell size, smaller output interval,
   and longer stationary window.
7. Treat the first reversal or expulsion event as the end of the reduced
   model's calibration domain.

This separation is useful: fitting and validation points must not be the same
data if the model is being used predictively.

## Reproducibility record

Store enough context to reconstruct both the equation and the analyzed
spectrum:

```python
record = {
    "material": {
        "Ms_A_per_m": material.Ms,
        "alpha": material.alpha,
        "P": material.P,
        "A_J_per_m": material.A,
        "gamma_rad_per_s_T": material.gamma,
    },
    "geometry": {
        "R_m": geometry.R,
        "L_m": geometry.L,
    },
    "model": {
        "polarity": model.polarity,
        "omega0_rad_per_s": model.omega0,
        "N": model.N,
        "domega0_dJ": model.domega0_dJ,
        "J_threshold_A_per_m2": model.J_threshold,
    },
    "integration": {
        "t_span_s": (float(trajectory.t[0]), float(trajectory.t[-1])),
        "dt_nominal_s": 10e-12,
        "initial_s": (1e-3, 0.0),
        "edge_limited": trajectory.metadata["edge_limited"],
    },
    "spectrum": {
        "transient_fraction": 0.60,
        "signal": "x",
        "window": "hann",
    },
}
```

For dataset-aware models, retain the adapter metadata as well. In particular,
do not discard the resolved geometry source, torque thickness, polarizer
projection, or current convention.

## Numerical and physical validity checks

- Choose `dt` from the highest carrier, harmonic, or modulation frequency, not
  only from the nominal gyrotropic frequency.
- Simulate long enough that `1 / observation_time` resolves the desired peak
  separation.
- Reject the startup interval explicitly with `compute_spectrum(...)`.
- Inspect `trajectory.metadata["edge_limited"]`; an edge hit is a model-validity
  boundary, not a prediction of post-expulsion dynamics.
- Calibrate `omega0`, `N`, `domega0_dJ`, field response, and effective torque
  against independent MuMax3 sweeps for quantitative work.
- Do not extrapolate through polarity reversal, vortex expulsion, strong core
  deformation, noncircular samples, or spatial spin-wave resonances.

The separately exported `mmpp.analytical.nonlinear_stno` 4D extension is
explicitly experimental and uncalibrated. It must not be used as the production
baseline without an independent derivation and validation data set.
