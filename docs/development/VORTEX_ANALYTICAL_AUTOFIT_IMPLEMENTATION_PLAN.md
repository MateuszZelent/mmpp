# Analytical Parameter Autofit for Vortex Dynamics

## Scope
Plan for implementing physics-informed trajectory autofit in
`mmpp.solitons.vortex`.

## Core idea
Start with deterministic, single-trajectory fitting (time + spectral loss) before
introducing multi-job sweep optimization.

## Target API
```python
fit = job.vortex.autofit.thiele(
    trajectory="steady_state",
    model="auto",
    params="auto",
    fit_params=("omega0", "N", "chi_scale"),
    objective="hybrid",
)
```

## Key points
- Keep existing `plotting` and bridge APIs stable.
- Reuse existing analytical models (`mmpp.analytical.thiele`).
- Add dedicated result payload with initial parameters, loss breakdown,
  diagnostics, and comparison plots.
- Add batch sweep entry point for future shared-parameter fitting.

## Implementation phases
1. Create dedicated `autofit` namespace and config/data/result contracts.
2. Implement feature extraction from trajectory and time-windowed signals.
3. Add weighted multi-objective loss and optional global search.
4. Integrate into existing orbital overlay workflow with `fit=` option.
5. Add batch-level scaffolding and regression tests.
