from mmpp import MMPP


op = MMPP("/zfn2/mannga/jobs/vortices/spectrum/d100_sinc4.zarr", debug=False)

# op[0].fft.plot_spectrum(
#     dset="m_z5-8",
#     z=0, # Dataset name
#     method=1, # FFT method
#     z_layer=0, # Z-layer (-1 for last layer)
#     window="hamming",
#     log_scale=False, # Use logarithmic scale
#     # normalize=True,
#     save=False, # Whether to save the plot
#     force=True, # Don't force - check if cache works
# )
op[0].fft.modes.interactive_spectrum(dset="m_z5-8", z_layer=0, method=1)
