# Vortex STNO Module Implementation Plan

## Scope
Long-form implementation roadmap for `mmpp.solitons.vortex`, covering topology,
core tracking, trajectory analytics, mode/spectrum hooks, and optional extensions.

## Architecture principles
- Shared `XYConvention` for orientation and sign consistency.
- Unified `TrajectoryResult` contract for numerical and analytical trajectories.
- Accessor-based API consistent with MMPP `fft` patterns.
- Explicit metadata/caching and reproducible computation.

## Core phases
1. Foundation contracts and module skeleton (imports, accessors, registries).
2. Topology and tracking core (polarity, vorticity, chirality, core position).
3. Trajectory pipeline with orbit/phase/frequency/spectrum analysis.
4. Optional events, signals, and energy diagnostics.
5. Interactive plotting layer and dashboard UX.
6. Batch-mode map/summary tools and export workflows.

## Notes
The project remains in staged implementation; this English summary captures the
contract direction and does not replace implementation-specific technical notes.
