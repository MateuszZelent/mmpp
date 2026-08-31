# Production audit: current-driven vortex Thiele models

Date: 2026-08-31  
Repository: `containers_admin2/postprocessing/mmpp`  
Scope: analytical CPP/CIP vortex dynamics, field-resolved dynamics, stochastic
integration, MuMax3 adapters, autofit, spectra, and the public
`analytical.nonlinear_stno` extension.

## Executive decision

The reduced CPP and CIP Thiele solvers are suitable as a production-executable
collective-coordinate baseline after the remediations in this audit. They can
calculate transient core trajectories under arbitrary finite current waveforms
and the corresponding gyrotropic response spectrum.

This is a conditional scientific “go”, not a claim of full micromagnetic
equivalence:

- **Implemented:** deterministic CPP/CIP trajectories, CPP thermal SDE,
  field-resolved calibrated forces, DC thresholds/orbits, fitting, and
  transient-aware spectra, plus canonical notebook helper cards for model
  discovery, resolved parameters, results, fit diagnostics, and validity
  boundaries.
- **Production-executable:** yes for a circular, intact, rigid vortex within a
  calibrated displacement/current/field range.
- **Validated in this change:** algebraic identities, SI units, force-potential
  consistency, MuMax3 torque reduction, stochastic diffusion, edge and time
  handling, and fixed-step RK4 parity with adaptive SciPy integration.
- **Partially validated:** field/Oersted/heating coefficients remain empirical;
  they require a data set independent from the fit data.
- **Not validated:** core reversal, expulsion after the edge event, vortex
  nucleation, spatial spin-wave modes, noncircular elements, and quantitative
  linewidths near topology-changing events.
- **Experimental only:** `mmpp.analytical.nonlinear_stno`. Its extra 4D
  spin-wave/back-action terms have no cited derivation or independent
  qualification and are now labeled accordingly.

## Audited implementation surface

The primary runtime is in:

- `mmpp/analytical/thiele.py`: material/geometry contracts, CIP, reduced CPP,
  stochastic CPP, waveforms, spectra, threshold/orbit prediction, and `f(J)`
  fitting;
- `mmpp/analytical/field_resolved_thiele.py`: unitful vector-field CPP equation,
  conservative potential, calibrated forces, and frequency extraction;
- `mmpp/solitons/vortex/model/thiele/`: dataset-facing CPP/CIP/field adapters;
- `mmpp/solitons/vortex/bridge/extract.py`: MuMax/job parameter extraction;
- `mmpp/solitons/vortex/autofit/`: reference and Numba integration, fit guards,
  seeds, and objectives;
- `mmpp/solitons/vortex/nonlinear/`: force-balance and notebook-facing wrappers;
- `mmpp/analytical/nonlinear_stno/`: separate empirical 4D extension.
- `mmpp/analytical/_thiele_html.py` and
  `mmpp/solitons/vortex/model/thiele/_html.py`: canonical tabbed notebook
  helpers for direct and dataset-aware model workflows.

The audit treated source presence, a passing import, and plausible plots as
insufficient evidence. Each quantitative claim required an equation/unit
check or an executable regression.

## Physical contract

### Units and gyromagnetic convention

The package uses SI quantities:

- position in metres;
- flux density `B` in tesla;
- current density `J` in A/m²;
- angular frequency in rad/s;
- `gamma` in rad/(s T);
- the field-resolved gyrocoefficient
  \(G_0=2\pi M_s L/\gamma\) in kg/s.

The previous force-balance path mixed the H-field convention
`gamma0 = mu0*gamma` with the B-field expression for `G`, producing an error of
approximately `1/mu0`. The public path now accepts `gamma`; the legacy
`gamma0` argument is converted explicitly.

### CIP equation

The implemented current-in-plane equation follows the generalized Thiele form
used by Moon et al.:

\[
\mathbf G\times(\mathbf u-\dot{\mathbf X})
=-\nabla U-\alpha\mathbf D\dot{\mathbf X}+\beta\mathbf D\mathbf u.
\]

`current_dir` is normalized and must be nonzero. The SciPy fallback now passes
that direction and `beta_nonadiabatic` to the model instead of silently using
defaults.

### Reduced CPP equation

For normalized displacement \(\mathbf s=\mathbf X/R\), the production baseline
is

\[
\dot{\mathbf s}=
[\chi(J)-d(u)\omega(u,J)]\mathbf s
+p\omega(u,J)\hat{\mathbf z}\times\mathbf s,
\]

with

\[
d(u)=d_0+d_1u^2,\qquad
\omega(u,J)=\left(\omega_0+\omega_eJ\right)(1+Nu^2),
\]

and MMPP's reduced-current convention

\[
\chi(J)=-p\,\chi_\mathrm{scale}
\frac{\gamma\hbar P}{4eL_\mathrm{STT}M_s}J.
\]

Consequently, the complete small-signal threshold is

\[
J_\mathrm{th}=
\frac{d_0\omega_0(0)}
{-p\,\chi_\mathrm{scale}\gamma\hbar P/(4eL_\mathrm{STT}M_s)
-d_0\,d\omega_0/dJ}.
\]

The old implementation omitted the final denominator term and was wrong when
the current/Oersted frequency slope was nonzero.

### MuMax3 Slonczewski reduction

MuMax3 uses an angular efficiency

\[
\epsilon=\frac{P\Lambda^2}
{(\Lambda^2+1)+(\Lambda^2-1)(\mathbf m\cdot\mathbf p)}.
\]

For `Lambda=1`, this is `P/2`, whereas the reduced Guslienko pumping expression
uses `P`. The adapter must therefore restore a factor of two after the Gilbert
mixing of damping-like and field-like torques. It now also separates:

- `mean_m_dot_p`, used only for the angular efficiency;
- `p_z`, used for projection onto perpendicular vortex pumping;
- physical free-layer thickness `L`, used by the gyrovector;
- torque thickness `L_stt`, used by the Slonczewski prefactor;
- top/bottom fixed-layer current orientation.

A tilted polarizer cannot be reduced completely to the circular scalar CPP
equation. MMPP warns when its in-plane part is significant; quantitative
in-plane torque coefficients belong in the field-resolved calibration.

### Stochastic equation

The default normalized-coordinate diffusion is now derived from the
small-signal fluctuation-dissipation relation:

\[
D_s=\frac{k_BT\,d_0}{G_0(1+d_0^2)R^2}.
\]

`noise_scale` scales noise amplitude, so diffusion scales with its square. The
Euler-Maruyama loop uses the true fractional final step. This white-noise,
rigid-coordinate SDE is appropriate for thermal seeding and reduced linewidth
studies, not for topology-changing thermal events.

### Eigenfrequency helper

`omega0_novosad` now states and implements only the leading thin-disk
asymptote

\[
\omega_0=\frac{5}{9\pi}\gamma\mu_0M_s\frac LR.
\]

The removed multiplier `1-3(L/R)/(8*pi)` was not the finite-aspect-ratio
function in the cited Novosad equation. The full result contains a
magnetostatic integral and an exchange correction. MMPP warns for `L/R > 0.2`;
quantitative work should pass a calibrated `omega0`.

## Remediated findings

Severity definition: P0 changes the physical sign, unit scale, or torque
normalization; P1 can materially bias trajectories/fits; P2 affects robustness,
performance, or scientific interpretation.

| ID | Severity | Finding before remediation | Resolution and executable evidence |
|---|---:|---|---|
| VT-01 | P0 | MuMax `epsilon=P/2` was passed as if it were the reduced-model `P`. | Restored factor two after Gilbert mixing; regression recovers the Guslienko prefactor at `Lambda=1`. |
| VT-02 | P0 | Field adapter projected `p_z` twice and did not reconcile gyro and torque thicknesses. | Explicit inverse projection and `L/L_stt` mapping; field and reduced small-signal pumping agree for tilted polarizers. |
| VT-03 | P0 | Force balance used `gamma0=mu0*gamma` in a B-convention gyrovector. | `gamma` is canonical; legacy `gamma0` is converted; both paths reproduce `2*pi*Ms*L/gamma`. |
| VT-04 | P1 | Critical current ignored `domega0_dJ` in the loss term. | Full analytical denominator implemented and tested at zero linear growth. |
| VT-05 | P1 | Default optimization bounds reversed for negative thresholds. | Bounds are ordered independently of current sign. |
| VT-06 | P1 | CPP SDE used heuristic diffusion and a full-size final step. | FDT diffusion, squared amplitude scaling, and fractional final step implemented. |
| VT-07 | P1 | Field potential depended on local `G(X,B)` while its gradient did not; edge regularization also broke `grad U`. | Potential stiffness is referenced to `G0`; edge energy has a continuously differentiable linear continuation in `r²`; finite-difference tests pass. |
| VT-08 | P1 | Field calibration silently floored negative gyro/damping factors. | Invalid `G` or negative `D/G` now fails explicitly; configured frequency floors warn. |
| VT-09 | P1 | CIP SciPy autofit ignored `current_dir` and `beta_nonadiabatic`. | Both are propagated; CPP/CIP fast paths now honor material `gamma` and candidate polarization. |
| VT-10 | P1 | Fixed-step kernels integrated one unused step past the final output and lost fractional endpoints. | Exact endpoint grid and interval-specific RK4 step; CPP/CIP kernels match high-accuracy SciPy trajectories. |
| VT-11 | P1 | FFT assumed uniform time even after events; no transient rejection API. | Nonuniform resampling and `compute_spectrum(transient_fraction=..., t_min=...)` added. |
| VT-12 | P1 | `f(J)` fitting accepted invalid predictions, underdetermined fits, and could execute about 1.4 million coarse candidates. | Fail-closed predictions, distinct-point identifiability checks, normalized objective, bounded deterministic search, and scaled local optimization. |
| VT-13 | P1 | Missing automatic current lookup silently became `J=0`. | Explicit auto lookup now raises with an actionable message. |
| VT-14 | P1 | Radius silently defaulted to `0.45*box`, even when physical `D`/`Area` metadata existed. | Prefer `R`, `D`, and `Area`; box inference is a warned full-box assumption; the 50 nm convenience fallback also warns. |
| VT-15 | P2 | `fit_from_trajectory` implied physical parameter identification but only synthesized a circle from mean radius/frequency. | New explicit `summarize_trajectory_kinematics`; legacy alias is deprecated and result metadata states that no physical coefficients were fitted. |
| VT-16 | P2 | Public inputs accepted nonfinite or nonphysical materials, geometry, calibration, waveforms, time steps, and clamps. | Dataclass and runtime validation added at model boundaries. |
| VT-17 | P2 | Experimental `nonlinear_stno` described empirical terms as a named theory, hard-coded its wall, mutated FFT input, and lacked shape checks. | Marked experimental/uncalibrated, terminology corrected, wall parameterized, wrapper validates arrays, FFT copies input, and outputs are labeled relative signal power. |
| VT-18 | P2 | Modular autofit refactor removed a private compatibility symbol still used by the repository test suite. | Restored a direct alias to the single source of truth in `seeds.py`; autofit suite collects and passes. |
| VT-19 | P2 | Direct Thiele models, dataset-aware adapters, fit/optimization results, and the trajectory result had incomplete or structurally inconsistent notebook representations; adjacent topology/mode/event smoke helpers had also lost their exact public result-class labels. | Added escaped, collision-safe Overview/Validity/API cards using the shared MMPP helper template; cards expose live signatures, resolved units, examples, and explicit scientific boundaries. Restored exact class labels while retaining readable card titles. |

## Documentation and interactive-helper contract

The production-facing documentation now has three distinct layers:

- `docs/tutorials/vortex_thiele_dynamics.md`: an end-to-end CPP/CIP workflow
  for arbitrary \(J(t)\), spectrum interpretation, time-step/window convergence,
  thermal seeding, multi-current calibration, target-current optimization,
  MuMax3 qualification, and reproducibility metadata;
- `docs/api/vortex_thiele.md`: exhaustive public entry points, object map, SI
  units, equations, callback and failure contracts, adapter mapping,
  interactive-helper matrix, and autodoc;
- this audit: evidence, remediated findings, validation state, and independent
  qualification boundary.

Notebook discovery covers the complete natural object chain:

1. `job[0].vortex.model.thiele` renders the CPP/CIP/field factory namespace;
2. evaluating a factory method without parentheses renders its live callable
   signature;
3. a constructed adapter renders resolved geometry, material, frequency,
   threshold/torque context, methods, and validity notes;
4. `adapter.model` renders the direct analytical model contract;
5. trajectory, \(f(J)\)-fit, and target-current optimization results render
   metrics, interpretation, live API, and next actions.

All new model/result cards use the shared `node_card_html` structure:
`html_tabs` is the first outer-card child, titles are inside the Overview tab,
user/job-derived values are escaped, UUID suffixes prevent collisions, and no
`h3` heading is emitted.

## Numerical qualification

The audit adds `tests/test_vortex_thiele_audit.py` as a tracked regression gate.
It verifies:

- exact MuMax3 reduction and representative angular efficiency;
- threshold algebra for nonzero current-frequency slope;
- negative-current optimization;
- stochastic diffusion and fractional stepping;
- finite-difference equality of potential gradient and force;
- field/reduced adapter parity including tilted polarizer and torque thickness;
- propagation of field-like phase shift;
- CIP fallback direction;
- B-versus-H gyro convention;
- strict invalid-input behavior and fit identifiability;
- thin-disk eigenfrequency contract;
- CPP and CIP Numba RK4 parity against high-accuracy SciPy integration;
- custom material `gamma` in autofit threshold diagnostics;
- nonuniform and transient-aware spectra;
- explicit geometry/current resolution;
- kinematic-proxy labeling;
- experimental 4D input/output and nonmutating FFT contracts.
- canonical notebook cards for namespaces, adapters, direct CPP/CIP/field
  models, both trajectory result types, \(f(J)\) fit, and current optimization;
  helper tests also verify escaping, tab structure, unique styling, and the
  absence of `h3`.

The final validation section must distinguish four states:

1. source implemented;
2. code can execute in the current environment;
3. synthetic/unit behavior is validated;
4. independent micromagnetic or experimental agreement is validated.

Only states 1–3 are established by this repository audit. State 4 requires a
separate qualification data set.

## Required independent qualification for a device study

For each geometry/material stack:

1. Obtain low-amplitude `omega0(B)` from a zero-current displacement/relaxation
   sweep; do not fit it on the same driven-current points used for validation.
2. Measure `f(J)`, steady orbit radius, transient relaxation time, and edge/core
   switching boundaries in MuMax3.
3. Fit only identifiable coefficients (`N`, `domega0_dJ`, effective torque, and
   selected field terms), keeping material/geometry values fixed to metadata.
4. Validate on held-out currents, fields, and at least one waveform not used in
   fitting (pulse, sinusoid, or chirp).
5. Report trajectory error, frequency error, orbit-radius error, threshold
   error, and whether either solver reached its validity boundary.
6. Stop interpreting the reduced solution beyond core reversal, expulsion, or
   substantial vortex deformation.

## Primary references used for the contract

- A. A. Thiele, *Steady-State Motion of Magnetic Domains*, Phys. Rev. Lett. 30,
  230 (1973), <https://doi.org/10.1103/PhysRevLett.30.230>.
- J.-H. Moon et al., generalized Thiele equation with current-in-plane torque,
  <https://arxiv.org/abs/0809.0952>.
- K. Y. Guslienko, O. V. Sukhostavets, and D. V. Berkov, *Nonlinear magnetic
  vortex dynamics in a circular nanodot excited by spin-polarized current*,
  Nanoscale Research Letters 9, 386 (2014),
  <https://doi.org/10.1186/1556-276X-9-386>.
- V. Novosad et al., *Magnetic Vortex Resonance in Patterned Ferromagnetic
  Dots*, Phys. Rev. B 72, 024455 (2005),
  <https://doi.org/10.1103/PhysRevB.72.024455>.
- R. Dussaux et al., vortex spin-torque oscillator field/current conventions,
  <https://arxiv.org/abs/1001.4933>.
- MuMax3 Slonczewski implementation,
  <https://github.com/mumax/3/blob/master/cuda/slonczewski2.cu>.

## Final boundary

The production claim is deliberately narrow: MMPP now provides a consistent,
testable reduced model of nonstationary vortex-core motion and its response
spectrum. It does not provide a complete excitation spectrum of the magnetic
texture. Any statement about spatial modes, reversal, annihilation, or device
voltage must be supported by LLGS/micromagnetic or experimental evidence.
