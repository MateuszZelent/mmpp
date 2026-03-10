"""Debug script to trace k3d z-axis bounds computation."""
import numpy as np
from mmpp.core.dataset import DatasetAwareWrapper
from mmpp.core.dataset_geometry import DatasetGeometry, AxisGeometry, resolve_dataset_geometry

a, b, c = 5e-9, 3e-9, 2e-9
cell = np.array((0.5e-9, 0.5e-9, 0.5e-9), dtype=float)
pmin = np.array((0.0, 0.0, 0.0), dtype=float)
pmax = np.array((2.0 * a, 2.0 * b, 2.0 * c), dtype=float)
counts = np.rint((pmax - pmin) / cell).astype(int)

xs = pmin[0] + (np.arange(counts[0], dtype=float) + 0.5) * cell[0]
ys = pmin[1] + (np.arange(counts[1], dtype=float) + 0.5) * cell[1]
zs = pmin[2] + (np.arange(counts[2], dtype=float) + 0.5) * cell[2]
zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
x_rel, y_rel, z_rel = xx - a, yy - b, zz - c
mask = ((x_rel / a) ** 2 + (y_rel / b) ** 2 + (z_rel / c) ** 2) <= 1.0
volume_zyxc = np.stack((-1e9 * y_rel, 1e9 * x_rel, 1e9 * z_rel), axis=-1).astype(np.float32)
volume_zyxc[~mask] = 0.0
data = volume_zyxc[np.newaxis, ...]

print(f"Full data shape: {data.shape}")
print(f"counts: {counts}")

geometry = DatasetGeometry(
    shape=tuple(int(v) for v in data.shape),
    spatial_axes={"x": 3, "y": 2, "z": 1},
    axes={
        "x": AxisGeometry("x", "x", 3, int(counts[0]), float(pmin[0]), float(pmax[0]), float(cell[0])),
        "y": AxisGeometry("y", "y", 2, int(counts[1]), float(pmin[1]), float(pmax[1]), float(cell[1])),
        "z": AxisGeometry("z", "z", 1, int(counts[2]), float(pmin[2]), float(pmax[2]), float(cell[2])),
    },
)

wrapper = DatasetAwareWrapper(None, "mock_data", data, geometry_override=geometry)
wrapper.attrs = {
    "dx": float(cell[0]), "dy": float(cell[1]), "dz": float(cell[2]),
    "pmin": tuple(float(v) for v in pmin), "pmax": tuple(float(v) for v in pmax),
    "x_name": "x", "y_name": "y", "z_name": "z",
}

print(f"\n--- Wrapper original ---")
print(f"  _materialized_data is None: {wrapper._materialized_data is None}")
print(f"  slice_info: {wrapper.slice_info}")
print(f"  zarr_array type: {type(wrapper.zarr_array)}, shape: {wrapper.zarr_array.shape}")

sliced7 = wrapper[0, 0:7, ...]
print(f"\n--- Sliced [0, 0:7, ...] wrapper ---")
print(f"  _materialized_data is None: {sliced7._materialized_data is None}")
print(f"  slice_info: {sliced7.slice_info}")
print(f"  _geometry_override present: {sliced7._geometry_override is not None}")

if sliced7._geometry_override:
    print(f"  _geometry_override.shape: {sliced7._geometry_override.shape}")
    for ax in ('x', 'y', 'z'):
        ag = sliced7._geometry_override.axes[ax]
        print(f"    {ax}: size={ag.size} min={ag.min_m * 1e9:.2f}nm max={ag.max_m * 1e9:.2f}nm cell={ag.cell_m * 1e9:.2f}nm")

print(f"  shape: {sliced7.shape}")

geom = resolve_dataset_geometry(sliced7, include_slice=True)
print(f"\n--- resolve_dataset_geometry(include_slice=True) ---")
print(f"  shape: {geom.shape}")
for ax in ('x', 'y', 'z'):
    ag = geom.axes[ax]
    print(f"  {ax}: size={ag.size} min={ag.min_m * 1e9:.2f}nm max={ag.max_m * 1e9:.2f}nm cell={ag.cell_m * 1e9:.2f}nm")

geom_nosli = resolve_dataset_geometry(sliced7, include_slice=False)
print(f"\n--- resolve_dataset_geometry(include_slice=False) ---")
print(f"  shape: {geom_nosli.shape}")
for ax in ('x', 'y', 'z'):
    ag = geom_nosli.axes[ax]
    print(f"  {ax}: size={ag.size} min={ag.min_m * 1e9:.2f}nm max={ag.max_m * 1e9:.2f}nm cell={ag.cell_m * 1e9:.2f}nm")

arr = sliced7.numpy(copy=False, squeeze=False)
print(f"\n--- numpy data ---")
print(f"  shape: {arr.shape}")

arr32 = np.asarray(arr, dtype=np.float32)
if arr32.ndim == 5:
    vol = arr32[-1]
    print(f"  volume[t=-1] shape: {vol.shape}")
    if vol.ndim == 4 and vol.shape[-1] <= 4:
        scalar = np.linalg.norm(vol[..., :3], axis=-1)
        print(f"  scalar shape: {scalar.shape}")
        # Check which z-layers have non-zero data
        for iz in range(scalar.shape[0]):
            nz_count = np.count_nonzero(scalar[iz])
            print(f"    z[{iz}]: {nz_count} nonzero of {scalar[iz].size}")

# Now simulate _k3d_resolve_geometry
from mmpp.core.dataset_plotting import DatasetPlotAccessor
plot_acc = DatasetPlotAccessor(sliced7)

# Call _k3d_resolve_geometry
shape_zyx = scalar.shape
bounds, axes, m, extents, cell_xyz = plot_acc._k3d_resolve_geometry(shape_zyx, multiplier=None)
print(f"\n--- _k3d_resolve_geometry ---")
print(f"  bounds: {bounds}")
print(f"  multiplier: {m}")
print(f"  extents: {extents}")
print(f"  cell_xyz: {cell_xyz}")
print(f"  z range: {bounds[4]:.2f} to {bounds[5]:.2f}")

# Check expand singleton
thin = plot_acc._k3d_expand_singleton_bounds_to_source(list(bounds), shape_zyx, multiplier=m)
print(f"\n--- _k3d_expand_singleton_bounds_to_source ---")
print(f"  expanded: {thin}")
print(f"  bounds after: {bounds}")

# Also check 0:6
sliced6 = wrapper[0, 0:6, ...]
geom6 = resolve_dataset_geometry(sliced6, include_slice=True)
print(f"\n--- 0:6 comparison ---")
for ax in ('z',):
    ag = geom6.axes[ax]
    print(f"  {ax}: size={ag.size} min={ag.min_m * 1e9:.2f}nm max={ag.max_m * 1e9:.2f}nm")
arr6 = sliced6.numpy(copy=False, squeeze=False)
arr6_32 = np.asarray(arr6, dtype=np.float32)
vol6 = arr6_32[-1]
scalar6 = np.linalg.norm(vol6[..., :3], axis=-1)
plot_acc6 = DatasetPlotAccessor(sliced6)
bounds6, _, m6, _, _ = plot_acc6._k3d_resolve_geometry(scalar6.shape, multiplier=None)
print(f"  bounds6: {bounds6}")
print(f"  z range: {bounds6[4]:.2f} to {bounds6[5]:.2f}")
