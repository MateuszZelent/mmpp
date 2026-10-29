"""Zarr 3 Store adapter for the verified local MuMax Zarr v2 reader."""

from __future__ import annotations

import asyncio

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.storage import LocalStore, WrapperStore


class SnapshotStore(WrapperStore):
    """Route metadata and chunk reads through one pinned integrity snapshot."""

    def __init__(self, snapshot):
        super().__init__(LocalStore(str(snapshot._root), read_only=True))
        self.snapshot = snapshot

    def _field(self, dataset):
        return self.snapshot._field(dataset)

    async def get(self, key, prototype, byte_range=None):
        try:
            data = await asyncio.to_thread(self.snapshot.__getitem__, key)
        except KeyError:
            return None
        # Verify the whole payload before applying a partial byte request.
        if isinstance(byte_range, RangeByteRequest):
            data = data[byte_range.start : byte_range.end]
        elif isinstance(byte_range, OffsetByteRequest):
            data = data[byte_range.offset :]
        elif isinstance(byte_range, SuffixByteRequest):
            data = data[-byte_range.suffix :] if byte_range.suffix else b""
        elif byte_range is not None:
            raise TypeError(f"Unsupported byte request {byte_range!r}")
        return prototype.buffer.from_bytes(data)

    async def exists(self, key):
        return await asyncio.to_thread(self.snapshot.__contains__, key)

    async def get_partial_values(self, prototype, key_ranges):
        return [await self.get(key, prototype, request) for key, request in key_ranges]

    async def _get_many(self, requests):
        for key, prototype, request in requests:
            yield key, await self.get(key, prototype, request)
