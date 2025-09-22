#!/usr/bin/env python3
"""
Test script for MMPP double-click animation functionality using real data
"""

import sys
import os
sys.path.insert(0, '/home/kkingstoun/git/mmpp')

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend

def test_with_real_data():
    """Test animation functionality with real MMPP data"""
    print("=== Test Animacji z Prawdziwymi Danymi MMPP ===")
    
    # Look for existing zarr files in the workspace
    zarr_files = []
    for root, dirs, files in os.walk('/home/kkingstoun/git/mmpp'):
        for file in files:
            if file.endswith('.zarr') or 'zarr' in file.lower():
                zarr_files.append(os.path.join(root, file))
    
    if not zarr_files:
        print("Nie znaleziono plików zarr. Tworzę test z symulowanymi danymi...")
        test_with_mock_data()
        return
    
    print(f"Znaleziono pliki zarr: {zarr_files}")
    zarr_path = zarr_files[0]
    
    try:
        from mmpp.fft.modes import FMRModeAnalyzer
        
        print(f"Ładuję dane z: {zarr_path}")
        analyzer = FMRModeAnalyzer(zarr_path, debug=True)
        
        if not analyzer.modes_available:
            print("Mody FFT nie są dostępne w tym pliku. Używam mock danych...")
            test_with_mock_data()
            return
            
        print("Tworzę interaktywny spektrogram z animacjami...")
        print("Instrukcja:")
        print("1. Kliknij na spektrum aby wybrać częstotliwość")
        print("2. Kliknij dwukrotnie na wykresach modów aby włączyć animację")
        print("3. Drugi double-click wyłącza animację")
        
        # Create interactive spectrum with animation support
        fig = analyzer.interactive_spectrum(show=True)
        
    except Exception as e:
        print(f"Błąd z prawdziwymi danymi: {e}")
        print("Przełączam na test z mock danymi...")
        test_with_mock_data()

def test_with_mock_data():
    """Fallback test with mock data"""
    print("\n=== Test z Mock Danymi ===")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        # Create simple test figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Test Animacji Double-Click\n(Mock Data)', fontsize=14)
        
        # Mock data for three different animation types
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 5, 50)
        X, Y = np.meshgrid(x, y)
        
        # Three different mode patterns
        patterns = [
            np.sin(X) * np.cos(Y),  # Magnitude pattern
            np.arctan2(Y-2.5, X-5),  # Phase pattern  
            np.cos(X/2) * np.sin(Y),  # Combined pattern
        ]
        
        titles = ['Magnitude (click 2x)', 'Phase (click 2x)', 'Combined (click 2x)']
        animations = {}
        animated_axes = set()
        
        ims = []
        for i, (ax, pattern, title) in enumerate(zip(axes, patterns, titles)):
            im = ax.imshow(pattern, extent=[0, 10, 0, 5], aspect='equal', 
                          origin='lower', cmap='viridis')
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, shrink=0.8)
            ims.append(im)
            
        def on_click(event):
            """Handle double-click events"""
            if event.dblclick and event.inaxes in axes:
                ax_idx = list(axes).index(event.inaxes)
                ax = event.inaxes
                
                print(f"Double-click na wykresie {ax_idx}")
                
                if ax in animated_axes:
                    # Stop animation
                    print(f"Zatrzymuję animację {ax_idx}")
                    if ax in animations:
                        animations[ax].event_source.stop()
                        del animations[ax]
                    animated_axes.remove(ax)
                    # Restore original
                    ims[ax_idx].set_array(patterns[ax_idx])
                    ax.figure.canvas.draw()
                else:
                    # Start animation
                    print(f"Uruchamiam animację {ax_idx}")
                    animated_axes.add(ax)
                    
                    if ax_idx == 0:  # Magnitude pulsing
                        def animate_mag(frame):
                            pulse = 1.0 + 0.3 * np.sin(frame * 0.2)
                            ims[ax_idx].set_array(patterns[ax_idx] * pulse)
                            return [ims[ax_idx]]
                        ani = animation.FuncAnimation(fig, animate_mag, frames=100, 
                                                    interval=50, blit=True, repeat=True)
                                                    
                    elif ax_idx == 1:  # Phase rotation
                        def animate_phase(frame):
                            phase_shift = frame * 0.1
                            shifted = patterns[ax_idx] + phase_shift
                            ims[ax_idx].set_array(shifted)
                            return [ims[ax_idx]]
                        ani = animation.FuncAnimation(fig, animate_phase, frames=100,
                                                    interval=50, blit=True, repeat=True)
                                                    
                    else:  # Temporal oscillation
                        def animate_temporal(frame):
                            t = frame * 0.1
                            temporal = patterns[ax_idx] * np.cos(0.5 * t)
                            ims[ax_idx].set_array(temporal)
                            return [ims[ax_idx]]
                        ani = animation.FuncAnimation(fig, animate_temporal, frames=100,
                                                    interval=50, blit=True, repeat=True)
                    
                    animations[ax] = ani
                    print(f"Animacja {ax_idx} uruchomiona!")
        
        # Connect event handler
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        plt.tight_layout()
        print("Mock test gotowy! Kliknij dwukrotnie na wykresach aby przetestować animacje.")
        plt.show()
        
    except Exception as e:
        print(f"Błąd mock testu: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("Rozpoczynam test funkcjonalności animacji double-click...")
    
    # Try with real data first
    test_with_real_data()

if __name__ == "__main__":
    main()