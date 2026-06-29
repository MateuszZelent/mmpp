# Audit and Production Masterplan: Interactive `fft/dispersion`

**Update:** 2026-06-28

## Executive Summary

`mmpp/fft/dispersion` reached a stable production-candidate state for this module. Interactive notebook surfaces are substantially modernized, cache semantics are much stricter, and release verification now includes headless + widget smoke checks. The whole MMPP package is not yet declared production-ready until CI confirms release/install gates across wheels and extras.

## Current Position

- Interactive contract: stabilized through `disp.plot.interactive(show=False)` and compatible legacy adapters.
- Cache and recomputation: cache context now includes axis/slice/filter/backends and time-window distinctions.
- Testing: dedicated tests cover import hygiene, backend parity, raw/display split, branch metadata, mode reconstruction, and release smoke paths.
- Docs/build gates: docs smoke and release workflow checks are in place, but package-wide status still depends on CI evidence.

## Key Risks and Controls

### P0
- Missing end-to-end release artifacts installed and executed in CI for all required extras.
  - Control: wheel/sdist install matrix and smoke in release job.

### P0
- Widget ergonomics outside local/headless checks.
  - Control: `--require-widget-smoke` gate and interactive smoke checks.

### P1
- Incomplete large benchmark coverage in release.
  - Control: weekly/manual benchmark workflows for heavier profiles.

### P2
- 2D dispersion contract remains experimental.
  - Control: explicit experimental marking and minimal coverage.

## Target Architecture

```python
disp = result.fft.dispersion
viewer = disp.plot.interactive(axis='x', component='perp', fmax=25, show=False)

res = disp.compute_1d(axis='x', store_complex=False)
res.plot.interactive(show=False)

modes = res.modes.interactive(show=False)
```

Legacy paths (`dispersion_modes(...)`, `plot_interactive()`) should remain available for one full release cycle.

## Completion Conditions for this module

- all documented API smoke examples pass with and without `ipywidgets`
- release workflow validates wheel/sdist with extras (`.[fft]`, `.[interactive]`, `.[plotting]`, `.[full]`)
- docs build and module-specific benchmarks are reproducible in CI

## Final Statement

`fft/dispersion` is close to stable module status and should be treated as the local production candidate pending release gates and CI-confirmed installability.
