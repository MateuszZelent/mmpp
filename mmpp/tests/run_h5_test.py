"""Inline test for H5 backend — run with: python mmpp/tests/run_h5_test.py"""
import sys, json, tempfile, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
import numpy as np
import h5py
import zarr

# --- Create H5 fixture ---
tmpdir = Path(tempfile.mkdtemp())
zarr_dir = tmpdir / 'test.zarr'
zarr_dir.mkdir()
(zarr_dir / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
(zarr_dir / '.zattrs').write_text(json.dumps({
    'dx': 5e-9, 'dy': 5e-9, 'dz': 5e-9,
    'Nx': 4, 'Ny': 3, 'Nz': 2,
    'end_time': '2026-01-01'
}))

m_dir = zarr_dir / 'm'; m_dir.mkdir()
(m_dir / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
rng = np.random.default_rng(42)
nz, ny, nx, ncomp, n_steps = 2, 3, 4, 3, 5
with h5py.File(m_dir / 'm.h5', 'w') as f:
    for s in range(n_steps):
        f.create_dataset(str(s), data=rng.standard_normal((nz, ny, nx, ncomp)).astype(np.float32))
    f.create_dataset('t', data=np.linspace(0, 1e-9, n_steps))

table_dir = zarr_dir / 'table'; table_dir.mkdir()
(table_dir / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
with h5py.File(table_dir / 'table.h5', 'w') as f:
    f.create_dataset('t', data=np.linspace(0, 1e-9, 20))
    f.create_dataset('step', data=np.arange(20, dtype=np.float64))
    f.create_dataset('mx', data=rng.standard_normal(20))

# --- Test h5_backend directly ---
from mmpp.pyzfn.h5_backend import detect_h5_quantities
groups = detect_h5_quantities(zarr_dir)
assert 'm' in groups, 'FAIL: m not detected'
assert 'table' in groups, 'FAIL: table not detected'
print('PASS: detect_h5_quantities')

g = groups['m']
assert g.shape == (5, 2, 3, 4, 3), f'FAIL: shape={g.shape}'
print(f'PASS: m.shape = {g.shape}')

frame = g[0]
assert frame.shape == (2, 3, 4, 3), f'FAIL: frame shape={frame.shape}'
print(f'PASS: m[0].shape = {frame.shape}')

frames = g[0:2]
assert frames.shape == (2, 2, 3, 4, 3), f'FAIL: slice shape={frames.shape}'
print(f'PASS: m[0:2].shape = {frames.shape}')

last = g[-1]
assert last.shape == (2, 3, 4, 3)
print('PASS: m[-1] negative indexing')

t = g['t']
assert t.shape == (5,)
print(f'PASS: m["t"].shape = {t.shape}')

assert 't' in g
assert '0' in g
print('PASS: __contains__')

table = groups['table']
assert 'mx' in table
assert table['mx'].shape == (20,)
print(f'PASS: table columns: {table.keys()}')

# --- Test ZarrJobResult integration ---
from mmpp.core.job import ZarrJobResult
job = ZarrJobResult(str(zarr_dir), {'dx': 5e-9})

assert 'm' in job
assert 'table' in job
print('PASS: __contains__ on ZarrJobResult')

assert 'm' in job.keys()
assert 'table' in job.keys()
print(f'PASS: job.keys() = {job.keys()}')

m_obj = job['m']
assert m_obj.shape == (5, 2, 3, 4, 3), f'FAIL: job[m].shape={m_obj.shape}'
print(f'PASS: job["m"].shape = {m_obj.shape}')

frame0 = job['m'][0]
assert frame0.shape == (2, 3, 4, 3)
print('PASS: job["m"][0] indexing')

raw = job.get_raw('m')
assert raw.shape == (5, 2, 3, 4, 3)
print('PASS: job.get_raw("m")')

raw_sliced = job.get_raw('m', slice(0, 2))
assert raw_sliced.shape == (2, 2, 3, 4, 3)
print('PASS: job.get_raw("m", slice(0,2))')

tbl = job['table']
assert 'mx' in tbl
mx_data = tbl['mx'][:]
assert mx_data.shape == (20,)
print('PASS: job["table"]["mx"] read OK')

assert job.attrs['dx'] == 5e-9
print('PASS: job.attrs["dx"] from .zattrs')

assert job.is_finished()
print('PASS: is_finished()')

# --- Parity test: create pure zarr with same RNG seed ---
rng2 = np.random.default_rng(42)
pure_dir = tmpdir / 'pure.zarr'
z = zarr.open_group(str(pure_dir), mode='w')
z.attrs.update({'dx': 5e-9, 'dy': 5e-9, 'dz': 5e-9, 'Nx': 4, 'Ny': 3, 'Nz': 2, 'end_time': '2026-01-01'})
all_frames = [rng2.standard_normal((nz, ny, nx, ncomp)).astype(np.float32) for _ in range(n_steps)]
z.create_array('m', data=np.stack(all_frames, axis=0))
tg = z.create_group('table')
tg.create_array('t', data=np.linspace(0, 1e-9, 20))
tg.create_array('step', data=np.arange(20, dtype=np.float64))
tg.create_array('mx', data=rng2.standard_normal(20))

pure_job = ZarrJobResult(str(pure_dir), {'dx': 5e-9})

# Shape parity
assert job['m'].shape == np.asarray(pure_job['m'][:]).shape
print('PASS: shape parity H5 vs zarr')

# Data parity step 0
h5_d = job['m'][0]
zarr_d = np.asarray(pure_job['m'][0])
np.testing.assert_array_almost_equal(h5_d, zarr_d)
print('PASS: data parity step 0')

# Data parity all
h5_all = job['m'][:]
zarr_all = np.asarray(pure_job['m'][:])
np.testing.assert_array_almost_equal(h5_all, zarr_all)
print('PASS: data parity all steps')

# Table parity
for col in ['t', 'mx', 'step']:
    h5_v = job['table'][col][:]
    zarr_v = np.asarray(pure_job['table'][col][:])
    np.testing.assert_array_almost_equal(h5_v, zarr_v)
print('PASS: table data parity')

# Cleanup
import shutil
shutil.rmtree(tmpdir)

print()
print('=== ALL TESTS PASSED ===')
