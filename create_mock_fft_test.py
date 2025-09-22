#!/usr/bin/env python3
"""
Create mock MMPP data with FFT modes for testing characterization
"""

import sys
import os
sys.path.insert(0, '/home/kkingstoun/git/mmpp')

import numpy as np
import zarr
import matplotlib
matplotlib.use('TkAgg')

def create_mock_zarr_with_fft(zarr_path):
    """Create a mock zarr file with FFT mode data"""
    print(f"Creating mock zarr file: {zarr_path}")
    
    # Remove existing file if it exists
    if os.path.exists(zarr_path):
        import shutil
        shutil.rmtree(zarr_path)
    
    # Create root zarr group
    root = zarr.open(zarr_path, mode='w')
    
    # Simulation parameters
    nx, ny, nz = 128, 64, 1
    nt = 500
    dx = dy = 2e-9  # 2 nm resolution
    dz = 10e-9      # 10 nm thickness
    dt = 1e-12      # 1 ps timestep
    
    # Create spatial coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.array([0.0])
    t = np.arange(nt) * dt
    
    # Store basic simulation metadata
    root.attrs['nx'] = nx
    root.attrs['ny'] = ny  
    root.attrs['nz'] = nz
    root.attrs['nt'] = nt
    root.attrs['dx'] = dx
    root.attrs['dy'] = dy
    root.attrs['dz'] = dz
    root.attrs['dt'] = dt
    
    # Create m dataset with time-domain magnetization
    m_data = np.zeros((nt, nx, ny, nz, 3), dtype=np.float32)
    
    # Create mock magnetization dynamics with multiple modes
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    
    for it in range(nt):
        time = t[it]
        
        # Mode 1: Gyration mode at 8 GHz
        freq1 = 8e9  # 8 GHz
        omega1 = 2 * np.pi * freq1
        r1 = np.sqrt((x_grid - nx*dx/2)**2 + (y_grid - ny*dy/2)**2)
        envelope1 = np.exp(-(r1/(20e-9))**2)
        
        mx1 = envelope1 * np.cos(omega1 * time) * 0.1
        my1 = envelope1 * np.sin(omega1 * time) * 0.1
        
        # Mode 2: Breathing mode at 15 GHz  
        freq2 = 15e9  # 15 GHz
        omega2 = 2 * np.pi * freq2
        r2 = np.sqrt((x_grid - nx*dx/2)**2 + (y_grid - ny*dy/2)**2)
        envelope2 = np.exp(-(r2/(15e-9))**2)
        
        mx2 = envelope2 * np.cos(omega2 * time) * 0.05
        my2 = envelope2 * np.cos(omega2 * time) * 0.05
        
        # Mode 3: Azimuthal mode at 22 GHz
        freq3 = 22e9  # 22 GHz  
        omega3 = 2 * np.pi * freq3
        phi = np.arctan2(y_grid - ny*dy/2, x_grid - nx*dx/2)
        r3 = np.sqrt((x_grid - nx*dx/2)**2 + (y_grid - ny*dy/2)**2)
        envelope3 = (r3/(10e-9)) * np.exp(-(r3/(25e-9))**2)
        
        mx3 = envelope3 * np.cos(2*phi + omega3*time) * 0.08
        my3 = envelope3 * np.sin(2*phi + omega3*time) * 0.08
        
        # Ground state + small oscillations
        m_data[it, :, :, 0, 0] = (mx1 + mx2 + mx3)  # mx
        m_data[it, :, :, 0, 1] = (my1 + my2 + my3)  # my  
        m_data[it, :, :, 0, 2] = np.sqrt(1 - (mx1+mx2+mx3)**2 - (my1+my2+my3)**2)  # mz
    
    # Store m dataset
    m_dset = root.create_array('m', shape=m_data.shape, dtype=m_data.dtype, chunks=(50, 64, 32, 1, 3))
    m_dset[:] = m_data
    m_dset.attrs['dx'] = dx
    m_dset.attrs['dy'] = dy
    m_dset.attrs['dz'] = dz
    m_dset.attrs['dt'] = dt
    m_dset.attrs['unit'] = 'dimensionless'
    
    # Create FFT data
    print("Computing FFT...")
    fft_group = root.create_group('fft')
    m_fft_group = fft_group.create_group('m')
    
    # Compute FFT
    m_complex = m_data[:, :, :, 0, 0] + 1j * m_data[:, :, :, 0, 1]  # mx + i*my
    m_fft = np.fft.fft(m_complex, axis=0)
    freqs = np.fft.fftfreq(nt, dt)
    
    # Take positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask] / 1e9  # Convert to GHz
    m_fft_pos = m_fft[pos_mask]
    
    # Store frequency-domain data
    modes_data = np.zeros((len(freqs_pos), nx, ny, nz, 3), dtype=np.complex64)
    modes_data[:, :, :, :, 0] = m_fft_pos[:, :, :, np.newaxis]  # mx_fft
    modes_data[:, :, :, :, 1] = 1j * m_fft_pos[:, :, :, np.newaxis]  # my_fft (90° phase shift)
    modes_data[:, :, :, :, 2] = 0.1 * m_fft_pos[:, :, :, np.newaxis]  # mz_fft (small oscillation)
    
    # Store modes
    modes_dset = m_fft_group.create_array('modes', modes_data)
    modes_dset.attrs['dx'] = dx
    modes_dset.attrs['dy'] = dy
    modes_dset.attrs['dz'] = dz
    modes_dset.attrs['unit'] = 'dimensionless'
    
    # Store frequencies  
    freqs_dset = m_fft_group.create_array('frequencies', freqs_pos)
    freqs_dset.attrs['unit'] = 'GHz'
    
    # Create spectrum (power spectral density)
    spectrum = np.sum(np.abs(modes_data)**2, axis=(1, 2, 3, 4))
    spec_dset = m_fft_group.create_array('spec', spectrum)
    spec_dset.attrs['unit'] = 'a.u.'
    
    print(f"Created mock data with {len(freqs_pos)} frequency points")
    print(f"Frequency range: {freqs_pos.min():.1f} - {freqs_pos.max():.1f} GHz")
    print(f"Peak frequencies around: 8, 15, 22 GHz")
    
    return zarr_path

def test_characterization_with_mock_data():
    """Test characterization with mock data"""
    zarr_path = '/tmp/mock_mmpp_data.zarr'
    
    # Create mock data
    create_mock_zarr_with_fft(zarr_path)
    
    try:
        from mmpp.fft.modes import FMRModeAnalyzer
        
        print(f"\\nLoading mock data from: {zarr_path}")
        analyzer = FMRModeAnalyzer(zarr_path, debug=True)
        
        print(f"Modes available: {analyzer.modes_available}")
        if not analyzer.modes_available:
            print("Problem with mock data creation!")
            return
            
        print(f"Frequency range: {analyzer.frequencies.min():.1f} - {analyzer.frequencies.max():.1f} GHz")
        print(f"Number of frequencies: {len(analyzer.frequencies)}")
        
        # Test characterization on a few key frequencies
        test_freqs = [8.0, 15.0, 22.0]  # Our mock mode frequencies
        
        print("\\n=== Testing Mode Characterization ===")
        for freq in test_freqs:
            # Find closest available frequency
            closest_idx = np.argmin(np.abs(analyzer.frequencies - freq))
            actual_freq = analyzer.frequencies[closest_idx]
            
            print(f"\\n--- Testing frequency: {actual_freq:.2f} GHz (target: {freq} GHz) ---")
            
            try:
                char_result = analyzer.characterize_mode(actual_freq)
                
                print(f"Primary class: {char_result.primary_class}")
                print(f"m-index: {char_result.m_index}")
                print(f"Rotation sense: {char_result.rotation_sense or 'N/A'}")
                print(f"Radial nodes: {char_result.radial_nodes}")
                print(f"Confidence: {char_result.confidence:.2f}")
                print(f"Labels: {', '.join(char_result.labels)}")
                
                if char_result.notes:
                    print(f"Notes: {', '.join(char_result.notes)}")
                    
            except Exception as e:
                print(f"Error characterizing: {e}")
                import traceback
                traceback.print_exc()
        
        # Launch interactive viewer
        print("\\n=== Launching Interactive Viewer ===")
        print("Controls:")
        print("- Click spectrum to select frequency")
        print("- Press 'c' to characterize current mode")
        print("- Press 'h' for help")
        print("- Double-click mode plots for animations")
        
        analyzer.interactive_spectrum(show=True)
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(zarr_path):
            import shutil
            shutil.rmtree(zarr_path)
            print(f"\\nCleaned up: {zarr_path}")

if __name__ == "__main__":
    test_characterization_with_mock_data()