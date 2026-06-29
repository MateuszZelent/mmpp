# FFT Architecture Report (Spectrum-First Refactor)

**Date:** 2026-02-13
**Scope:** `mmpp/fft` (`spectrum`, `spectrum/modes`, `transmission`, `dispersion`)
**Priority:** close spectrum API consistency before further expansion

## Summary

The FFT API currently mixes legacy helpers and object-first entry points. This mixed state causes context-loss bugs and modularity friction.

Critical issues to fix first:

- component context is lost after slicing with `[..., 2]`
- duplicated `_clone()` implementation in `mmpp/fft/modes/interface.py`

## Architecture findings

- `spectrum`: partial object-oriented migration, still coupled with legacy wrappers.
- `spectrum/modes`: feature-rich but still strongly coupled to the old analyzer class.
- `transmission`: clean accessor model, lower conflict potential.
- `dispersion`: mature but large monolithic interface file.

## Selected defects (spectrum focus)

1. Component context loss on `[..., 2]`
   - affects `component_label` and single-component state flags
   - impacts downstream `spec.modes` behavior

2. Clone/state duplication in modes
   - second `_clone()` overwrites earlier implementation
   - configuration fields can be dropped when chaining `configure` and `filters`

3. Mixed canonical API surface
   - both legacy and object-first entry points remain exposed
   - docs still cross-reference both styles

4. Inconsistent `filters` helper
   - one path is property-like, another method-like
   - UX drift between `fft.spectrum` and `fft.filters` usage

## Recommended sequence

### Phase A (critical)
- align component detection for `slice(k, k+1)`
- remove duplicated mode clone implementation
- add regression tests for component retention and chain durability

### Phase B (canonical API)
- standardize on `data.fft.spectrum()` + `SpectrumResult`
- add compatibility aliases (`data`, `freqs`)
- mark legacy methods deprecated with explicit guidance

### Phase C (`filters` completion)
- provide helper shape for `data.fft.filters` aligned to spectrum flow
- preserve chainability and shared return behavior

### Phase D (documentation cleanup)
- rewrite docs/examples to one canonical style
- reduce legacy surface to thin adapters

## Acceptance criteria

- `job[0].m[:200, ..., 2].fft.spectrum().component_label == "$m_z"`
- `job[0].m[:200, ..., 2].fft.modes.component_index == 2`
- chained filters do not drop `tmax`, `filters_config`, or `cache_dir`
- `SpectrumResult` exposes `.data` and `.freqs`
- legacy entry points continue to work with deprecation warnings
