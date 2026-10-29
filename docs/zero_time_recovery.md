# Automatic recovery of zero-time Zarr fields

When a field has a zero-length time axis in `.zarray` and no `commit.json`,
MMPP automatically checks its `.frames` records. It exposes a pinned,
contiguous prefix starting at step 0, without modifying any source file.
Normal `job.m_layer21`, `job["m_layer21"]`, slicing, and FFT access use the
recovered shape and recorded timestamps.

Recovery requires a supported 5-D MuMax Zarr v2 field with one time sample per
chunk. Records must have a valid SHA-256 chain, finite strictly increasing
timestamps, and exactly the required spatial/component chunk keys. Physical
chunk sizes are checked while discovering the prefix. Every chunk payload is
checked against its SHA-256 when read, so corrupt data cannot silently become
Zarr fill values. A missing record or chunk ends the visible prefix.
Malformed records, inconsistent sizes, and an invalid existing commit fail
explicitly. An existing valid commit remains authoritative.

A RuntimeWarning reports the recovered sample count. Field attributes include
`mmpp_recovery`, with `anchored_to_commit=False`: the record chain is internally
consistent but has no authoritative commit head. Recovery does not establish
that a simulation finished. A new job object is needed to see frames published
later. Cache revisions identify the exact recovered prefix.

Truly empty fields remain empty. Chunk filenames alone cannot recover physical
timestamps or establish integrity; fields without frame records retain normal
Zarr behavior. Recovery never fabricates a sampling interval.

## Validation and trust boundary

Frame keys must match the expected chunk grid before any chunk path is opened;
path traversal and duplicate records are rejected. Checksums establish record
and payload consistency, not authenticity against deliberate replacement of
an entire unanchored chain. No source metadata or simulation state is rewritten.

## Zarr 3 runtime

The source reader supports Zarr 3 through a read-only Store adapter. MuMax
archives using `.zarray` remain **on-disk Zarr format 2**, independently of the
installed Python Zarr major version. The adapter routes metadata, full reads,
and partial reads through the same integrity checks; consolidated metadata
cannot bypass the snapshot. This change covers source reading, not a complete
migration of all MMPP cache/export APIs or the dependency manifest to Zarr 3.
