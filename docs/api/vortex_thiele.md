# Current-driven vortex Thiele API

This page is the reference contract for the reduced current-driven vortex
models in MMPP. For a worked physical workflow, see
[Current-driven vortex dynamics](../tutorials/vortex_thiele_dynamics.md). For
the implementation and qualification record, see the
[2026-08-31 audit](../analysis/VORTEX_CURRENT_DRIVEN_THIELE_AUDIT_2026-08-31.md).

## What the API computes

The Thiele model replaces the complete magnetization field by the in-plane
vortex-core coordinate \(\mathbf{X}(t)=(X(t),Y(t))\). It can calculate:

- deterministic CPP and CIP core trajectories;
- stochastic CPP trajectories with reduced thermal noise;
- transient growth or decay, orbit radius, phase, velocity, and frequency;
- the relative response spectrum of \(X(t)\), \(Y(t)\), or the orbit radius;
- the CPP small-signal threshold and DC steady-orbit prediction;
- a field-resolved CPP trajectory with calibrated \(B_x\), \(B_y\), and
  \(B_z\) terms;
- multi-current \(f(J)\) calibration and target-frequency current
  optimization.

It does not return a magnetization array. Consequently, it does not calculate
spatial spin-wave eigenfunctions, core deformation, nucleation, polarity
reversal, annihilation, or post-expulsion dynamics. Coordinate FFT power is
not an electrical voltage PSD.

## Entry points

There are two supported entry paths.

### Direct analytical objects

Use this path for a standalone calculation with explicit SI parameters:

```python
from mmpp.analytical import (
    CPPThieleModel,
    CIPThieleModel,
    DiskGeometry,
    MaterialParams,
    current_ac,
    current_dc,
    current_pulse,
    field_dc,
    fit_omega0_N_to_fJ,
    omega0_novosad,
)
```

### Dataset-aware adapters

Use this path when material, geometry, current, or MuMax-compatible
spin-torque metadata should be resolved from an MMPP job:

```python
thiele = job[0].vortex.model.thiele

cpp_adapter = thiele.cpp(polarity="auto", N=0.30)
cip_adapter = thiele.cip(polarity="auto", current_dir=(1.0, 0.0))
field_adapter = thiele.field_resolved_cpp(
    polarity="auto",
    chirality=1,
)
```

The adapters expose the underlying analytical object as `adapter.model` and
convert deterministic output to the canonical vortex `TrajectoryResult` used
by the rest of the soliton API.

## Object map

| Object | Construction | Principal output |
| --- | --- | --- |
| `MaterialParams` | explicit material constants | validated SI material parameters |
| `DiskGeometry` | explicit or job-resolved geometry | radius, thickness, core scale |
| `CPPThieleModel` | perpendicular-current reduced model | `ThieleTrajectoryResult` |
| `CIPThieleModel` | in-plane-current generalized Thiele model | `ThieleTrajectoryResult` |
| `FieldResolvedCPPThieleModel` | CPP model with vector-field potential and force terms | `FieldResolvedTrajectoryResult` |
| `ThieleModelNamespace` | `job[0].vortex.model.thiele` | CPP/CIP/field adapter factories |
| `ThieleFJFitResult` | `fit_omega0_N_to_fJ(...)` | calibrated \(f(J)\) curve and diagnostics |
| `ThieleOptimizationResult` | `model.optimize_current_for_target_frequency(...)` | selected DC current and model residual |

## Units and conventions

All physical inputs use SI units unless the property name contains an explicit
suffix.

| Quantity | Unit |
| --- | --- |
| current density \(J\) | A/m² |
| current \(I\) | A |
| time | s |
| position, radius, thickness | m |
| magnetic flux density \(B\) | T |
| angular frequency \(\omega\) | rad/s |
| ordinary frequency \(f\) | Hz |
| saturation magnetization \(M_s\) | A/m |
| exchange stiffness \(A\) | J/m |

The normalized coordinate is \(\mathbf{s}=\mathbf{X}/R\), with
\(u=|\mathbf{s}|\). Core polarity is signed: `+1` and `-1` are physically
different models. The CPP pumping sign also depends on polarizer direction,
fixed-layer position, and the current convention. Do not infer the pumping
direction from a positive scalar \(J\); inspect `model.J_threshold`.

`MaterialParams.gamma` is the positive gyromagnetic-ratio magnitude in
rad/(s T). Polarity supplies the gyrotropic sense.

## Material and geometry

```python
material = MaterialParams(
    Ms=800e3,
    alpha=0.013,
    P=0.45,
    A=10e-12,
    beta_nonadiabatic=0.02,
)
geometry = DiskGeometry(
    R=128e-9,
    L=9e-9,
)
```

`MaterialParams` validates finite physical values. `P` is signed because the
sign can encode the effective polarization convention. The magnitude must not
exceed one. `DiskGeometry` requires positive radius and magnetic thickness.

`omega0_novosad(material, geometry)` implements the thin-disk asymptote

\[
\omega_0 = \frac{5}{9\pi}\gamma\mu_0 M_s\frac{L}{R}.
\]

It warns when \(L/R>0.2\). This helper is an initial estimate; use a
micromagnetically or experimentally calibrated \(\omega_0\) for quantitative
work.

## Current and field waveforms

All deterministic simulators accept a scalar callback `J_func(t)` returning a
finite current density in A/m². Built-in constructors include:

```python
dc = current_dc(2.0e10)
ac = current_ac(
    J_amp=0.2e10,
    f_hz=200e6,
    phase=0.0,
    J_offset=2.0e10,
)
pulse = current_pulse(
    J_amp=4.0e10,
    t_on=1e-9,
    t_off=6e-9,
    rise_time=0.2e-9,
)
```

Arbitrary finite callbacks are supported:

```python
import numpy as np

def chirped_current(t):
    f0 = 50e6
    chirp_rate = 5e15
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    return J_dc + J_ac * np.sin(phase)
```

The corresponding field callback `B_func(t)` returns `ExternalField` or a
scalar/vector form accepted by `ExternalField.from_any`. Public constructors
include `field_dc`, `field_ac`, `field_ac_vector`, and
`field_rotating_inplane`.

A callback returning NaN or infinity fails immediately. The dataset-aware
`"auto_from_table"` current mode also fails closed if no current can be
resolved; it is never silently converted to zero.

## CPP model

### Construction

```python
omega0 = omega0_novosad(material, geometry)

model = CPPThieleModel(
    material=material,
    geom=geometry,
    omega0=omega0,
    N=0.30,
    polarity=-1,
    domega0_dJ=0.0,
    chi_scale=1.0,
    torque_thickness=geometry.L,
)
```

For a circular perpendicular polarizer, the reduced equation is

\[
\dot{\mathbf{s}} =
[\chi(J)-d(u)\omega(u,J)]\mathbf{s}
+p\,\omega(u,J)\,\hat{\mathbf{z}}\times\mathbf{s},
\]

with

\[
\omega(u,J)=\omega_{0,\mathrm{eff}}(J)(1+Nu^2),\qquad
d(u)=d_0+d_1u^2.
\]

### Threshold and steady orbit

Use the public methods rather than reproducing sign-sensitive formulas:

```python
J_threshold = model.threshold_current_dc()
u_steady = model.steady_state_u(1.3 * J_threshold)
frequency_hz = model.predict_frequency_dc(1.3 * J_threshold)
```

The threshold implementation solves the complete linear-growth condition,
including `domega0_dJ`. A result below threshold returns no self-oscillating
steady orbit. A solution at the configured edge boundary is rejected unless
`allow_edge=True`; such a value is a reduced-model boundary, not a prediction
of the expelled state.

### Deterministic trajectory

```python
J_dc = 1.30 * model.threshold_current_dc()

result = model.simulate(
    t_span=(0.0, 80e-9),
    s0=(1e-3, 0.0),
    J_func=current_ac(
        J_amp=0.15 * abs(J_dc),
        f_hz=200e6,
        J_offset=J_dc,
    ),
    dt=10e-12,
    clamp_u=0.95,
    edge_behavior="stop",
)
```

`t_span` includes the exact final time even when it is not an integer multiple
of `dt`. `clamp_u` defines a validity event relative to the field-shifted
equilibrium. `edge_behavior="stop"` returns at the event;
`edge_behavior="freeze"` fills the remaining requested output times with the
boundary coordinate. In both cases inspect:

```python
result.metadata["edge_limited"]
result.metadata["edge_hit_time"]
result.metadata["edge_hit_kind"]
```

At exactly `s0=(0, 0)`, perpendicular CPP pumping is multiplicative and a
deterministic trajectory remains centered. Use a physically justified seed
displacement or the stochastic solver.

### Thermal trajectory

```python
thermal = model.simulate_sde(
    t_span=(0.0, 80e-9),
    s0=(0.0, 0.0),
    J_func=current_dc(J_dc),
    dt=10e-12,
    temperature_k=300.0,
    noise_scale=1.0,
    seed=7,
    clamp_u=0.95,
)
```

When `diffusion=None`, MMPP calculates the normalized-coordinate diffusion from
the reduced Thiele fluctuation-dissipation relation. `noise_scale` multiplies
the noise amplitude, so the diffusion scales with `noise_scale**2`. An explicit
`diffusion` is interpreted in normalized-coordinate²/s and recorded in result
metadata. This is not cell-wise thermal LLGS.

## Response spectrum

`ThieleTrajectoryResult.compute_spectrum` returns a tuple
`(frequency_hz, relative_power)`:

```python
frequency_hz, relative_power = result.compute_spectrum(
    transient_fraction=0.60,
    signal="x",
    window="hann",
)
```

Alternatively, provide an absolute lower time:

```python
frequency_hz, relative_power = result.compute_spectrum(
    t_min=50e-9,
    signal="radius",
    window="hann",
)
```

Accepted signals are `"x"`, `"y"`, and `"radius"`. Nonuniform time samples are
linearly resampled to a uniform grid. Time values must be finite and strictly
increasing. At least four samples must remain after transient rejection.

The spectrum can contain the nonlinear gyrotropic carrier, modulation
sidebands, and coordinate harmonics. Its bin spacing is approximately
\(1/T_\mathrm{window}\); `dt` controls the Nyquist limit. It does not contain
spatial mode profiles and is not normalized as V²/Hz.

Convenience properties use the full trajectory:

- `result.spectrum_frequencies_ghz`;
- `result.power_spectrum`;
- `result.dominant_frequency_ghz`;
- `result.linewidth_ghz`.

For publication-quality analysis, call `compute_spectrum` explicitly with a
documented transient window rather than relying on the convenience cache.

## CIP model

`CIPThieleModel` implements the generalized Thiele equation with Zhang-Li
spin-transfer torque. The current drives translation directly:

```python
cip_model = CIPThieleModel(
    material=material,
    geom=geometry,
    omega0=omega0,
    polarity=1,
    current_dir=(1.0, 0.0),
)

cip_result = cip_model.simulate(
    t_span=(0.0, 20e-9),
    r0=(1e-9, 0.0),
    J_func=current_pulse(4e10, t_on=1e-9, t_off=5e-9),
    dt=5e-12,
)
```

Unlike CPP anti-damping, a CIP pulse can displace a centered core. The
nonadiabatic coefficient is `material.beta`. `current_dir` is normalized on
construction and must be a finite nonzero vector.

## Field-resolved CPP model

Use `FieldResolvedCPPThieleModel` when vector-field effects, equilibrium shift,
Oersted stiffness, current-to-frequency calibration, or current/heating
calibration must be represented explicitly.

```python
from mmpp.analytical import (
    CurrentDrive,
    FieldResolvedCalibration,
    FieldResolvedCPPThieleModel,
    FrequencyExtractionConfig,
    OerstedCalibration,
    ThermalCalibration,
)

calibration = FieldResolvedCalibration(
    domega_dBz=2 * np.pi * 20e9,
    domega_dJ=0.0,
    current_drive=CurrentDrive(area_m2=np.pi * geometry.R**2),
    oersted=OerstedCalibration(),
    thermal=ThermalCalibration(),
)

field_model = FieldResolvedCPPThieleModel(
    material=material,
    geom=geometry,
    omega0=omega0,
    N=0.30,
    polarity=-1,
    chirality=1,
    polarizer=(0.0, 0.0, 1.0),
    calibration=calibration,
)
```

The calibrated potential is tied to the reference gyrocoefficient `G0`.
State-dependent gyro corrections affect dynamics but do not silently redefine
the conservative stiffness. Non-positive gyrocoefficient or damping values
fail instead of being hidden by absolute values.

Run one trajectory or a DC-current sweep:

```python
field_result = field_model.simulate(
    t_span=(0.0, 80e-9),
    s0=(1e-3, 0.0),
    J_func=current_dc(J_dc),
    B_func=field_dc((0.0, 0.0, 20e-3)),
    dt=10e-12,
)

table = field_model.simulate_dc_sweep(
    I_values_A=np.linspace(-2e-3, 2e-3, 21),
    B=(0.0, 0.0, 20e-3),
    t_total=80e-9,
    dt=10e-12,
    transient_fraction=0.6,
)
```

The sweep reports geometric and FFT frequency estimates, normalized amplitude,
orbit center, small-signal growth, edge status, and regime.

## MuMax-compatible CPP reduction

The dataset-aware CPP factories understand the MuMax Slonczewski parameters
`Pol`, `Lambda`, `epsilonprime`, fixed-layer direction and position, and the
magnetic thickness on which torque acts.

For \(\Lambda\ne1\), angular efficiency depends on
\(\langle\mathbf{m}\cdot\mathbf{p}\rangle\). Pass a representative
`mean_m_dot_p` when the job metadata does not supply it:

```python
adapter = job[0].vortex.model.thiele.cpp(
    polarizer=(0.0, 0.0, 1.0),
    fixed_layer_position="top",
    Lambda=2.0,
    epsilonprime=0.0,
    mean_m_dot_p=0.15,
)
```

The in-plane polarizer component is not mapped into circular CPP
auto-oscillation pumping by the reduced model. A warning makes that projection
explicit. The field-resolved adapter uses the same reduction and then restores
the field-model polarization/thickness convention exactly once.

Geometry resolution prefers explicit `R`, `D`, or `Area`. Box-size inference
assumes the disk fills the smaller in-plane dimension and emits a warning.

## Multi-current fitting

Fit \(\omega_0\) and \(N\) to several distinct steady-state frequency points:

```python
fit = fit_omega0_N_to_fJ(
    J_data,
    f_data_hz,
    material=material,
    geom=geometry,
    polarity=-1,
    initial_omega0=omega0,
    initial_N=0.25,
    fit_domega0_dJ=True,
    fit_chi_scale=False,
)

fit.plt.frequency_vs_current()
print(fit.omega0, fit.N, fit.domega0_dJ, fit.rmse_hz)
```

The fit rejects nonfinite, nonpositive, duplicate, and underdetermined data.
`valid_mask` identifies current points for which the fitted reduced model has a
valid steady orbit. One trajectory is not enough to separate stiffness,
damping, torque efficiency, and nonlinear frequency shift; the legacy
`fit_from_trajectory` name is therefore deprecated in favor of the explicitly
kinematic `summarize_trajectory_kinematics`.

## Current optimization

After calibration:

```python
optimum = model.optimize_current_for_target_frequency(
    target_frequency_hz=0.65e9,
    J_bounds=(1.05 * model.J_threshold, 4.0 * model.J_threshold),
)

print(optimum.current_density_a_per_m2)
print(optimum.predicted_frequency_hz)
print(optimum.objective_value_hz)
```

This solves a reduced-model objective. It is not an experimental guarantee.
Verify the selected point with micromagnetics or measurement and reject
edge-limited candidates.

## Notebook helper contract

Evaluating these objects as the final expression in a Jupyter cell displays a
canonical tabbed card:

| Expression | Card contents |
| --- | --- |
| `job[0].vortex.model.thiele` | model families, dataset context, live factory signatures |
| `job[0].vortex.model.thiele.cpp` | callable-node signature without executing |
| `cpp_adapter` | resolved parameters, threshold, torque mapping, simulation API |
| `cpp_adapter.model` | direct analytical-model API and validity limits |
| `result` | trajectory metrics, spectrum workflow, plotting methods |
| `fit` | fitted coefficients, valid points, RMSE, plotting |
| `optimum` | selected current, predicted frequency, residual, bounds |

Every model/result card has `Overview`, `Validity`, and `API` tabs. Callable
factory/method nodes supplied by `InteractiveNodeMixin` have `Overview` and
`API` tabs and preserve the live wrapped signature. Cards are discovery and
diagnostic helpers; the methods shown are the same public methods documented
on this page.

## Failure contract

MMPP raises rather than silently repairing these cases:

- invalid material or geometry;
- nonpositive `omega0` or torque thickness;
- invalid polarity or current direction;
- nonfinite current/field callback output;
- invalid integration window, time step, or initial state;
- automatic current lookup with no resolvable source;
- non-monotonic spectrum time samples;
- underdetermined or invalid \(f(J)\) calibration data;
- field calibration producing nonpositive gyrocoefficient or damping.

Warnings identify approximations that can still be useful but require explicit
judgment: thin-disk frequency outside its preferred aspect ratio, geometry
inferred from the box, tilted polarizer projection, frequency floors, and
experimental nonlinear-STNO extensions.

## Minimum qualification record

For a quantitative CPP or CIP result, retain:

1. material and geometry in SI units;
2. polarity, chirality, current sign, polarizer, and fixed-layer convention;
3. the complete \(J(t)\), \(B(t)\), `dt`, and observation window;
4. the rejected transient interval and spectral window;
5. the origin of \(\omega_0\), \(N\), current-frequency slope, and torque
   efficiency;
6. edge-limit and integration metadata;
7. comparison against at least one independent micromagnetic or experimental
   reference point.

## Autodoc

```{eval-rst}
.. automodule:: mmpp.analytical.thiele
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mmpp.analytical.field_resolved_thiele
   :members:
   :undoc-members:
   :show-inheritance:
```
