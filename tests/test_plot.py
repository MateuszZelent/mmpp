import mmpp

# Enable debug logging to see detailed information
import logging
logging.basicConfig(level=logging.ERROR)

job = mmpp.MMPP("/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr")

# Test tmax parameter support
print("Testing tmax parameter...")
try:
    job[0].fft.plot_spectrum(dset="m_dot", tmax=100, z_layer=0, method=1, force=True, log_scale=False, fwhm=True)
    print("✓ Plot completed successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()