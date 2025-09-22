#!/usr/bin/env python3
"""
Test script for double-click animation functionality in MMPP modes
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
import matplotlib.pyplot as plt

# Create test data to simulate MMPP environment
class MockMMPP:
    def __init__(self):
        # Create mock data
        self.freqs = np.linspace(0, 50, 1000)  # GHz
        self.spectrum = np.abs(1 / (self.freqs - 10 + 1j*0.5)) + np.abs(1 / (self.freqs - 20 + 1j*0.3))
        
        # Mock mode data at frequency ~10 GHz
        x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 5, 25))
        self.mode_x = 2 * np.exp(-((x-5)**2 + (y-2.5)**2)/2) * np.cos(np.pi*x/10)
        self.mode_y = 1.5 * np.exp(-((x-5)**2 + (y-2.5)**2)/1.5) * np.sin(np.pi*y/5)
        self.mode_z = 1.0 * np.exp(-((x-5)**2 + (y-2.5)**2)/3) * np.cos(2*np.pi*x/10)
        
        self.X, self.Y = x, y
        self.selected_freq = 10.0
        self.selected_freq_idx = np.argmin(np.abs(self.freqs - self.selected_freq))
        
        # Initialize animation tracking
        self._mode_animations = {}
        self._animated_axes = set()
        
    def create_interactive_test(self):
        """Create test figure with double-click animation functionality"""
        print("Tworzę test animacji double-click...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Test Animacji - Częstotliwość: {self.selected_freq:.1f} GHz\n'
                    'Kliknij dwukrotnie na wykres aby włączyć/wyłączyć animację', fontsize=14)
        
        # Spectrum plot
        ax_spectrum = axes[0, 0]
        ax_spectrum.plot(self.freqs, self.spectrum, 'b-', alpha=0.7)
        ax_spectrum.axvline(self.selected_freq, color='red', linestyle='--', alpha=0.8)
        ax_spectrum.set_xlabel('Częstotliwość [GHz]')
        ax_spectrum.set_ylabel('Amplituda')
        ax_spectrum.set_title('Spektrum FFT')
        ax_spectrum.grid(True, alpha=0.3)
        
        # Mode plots
        mode_titles = ['Magnitude |mx|', 'Phase φy', 'Combined mz']
        mode_data = [np.abs(self.mode_x), np.angle(self.mode_y + 1j*self.mode_z), np.real(self.mode_z)]
        
        self.mode_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
        self.mode_ims = []
        
        for i, (ax, data, title) in enumerate(zip(self.mode_axes, mode_data, mode_titles)):
            im = ax.imshow(data, extent=[0, 10, 0, 5], aspect='equal', origin='lower', cmap='viridis')
            ax.set_title(title)
            ax.set_xlabel('x [μm]')
            ax.set_ylabel('y [μm]')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            if i == 1:  # Phase plot
                cbar.set_label('Phase [rad]')
            else:
                cbar.set_label('Amplitude')
                
            self.mode_ims.append(im)
            
        # Connect click handler
        self.click_connection = fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        print("Test utworzony. Kliknij dwukrotnie na wykresach modów aby przetestować animacje.")
        return fig
        
    def on_click(self, event):
        """Handle double-click events on mode plots"""
        if event.dblclick and event.inaxes in self.mode_axes:
            mode_idx = self.mode_axes.index(event.inaxes)
            mode_types = ['magnitude', 'phase', 'combined']
            mode_type = mode_types[mode_idx]
            
            print(f"Double-click wykryty na wykresie: {mode_type}")
            self._toggle_mode_animation(event.inaxes, mode_type, mode_idx)
            
    def _toggle_mode_animation(self, ax, animation_type, mode_idx):
        """Toggle animation for a specific mode axis"""
        try:
            if ax in self._animated_axes:
                # Stop animation
                print(f"Zatrzymuję animację {animation_type}...")
                self._stop_mode_animation(ax, mode_idx)
            else:
                # Start animation
                print(f"Uruchamiam animację {animation_type}...")
                self._start_mode_animation(ax, animation_type, mode_idx)
        except Exception as e:
            print(f"Błąd animacji: {e}")
            
    def _start_mode_animation(self, ax, animation_type, mode_idx):
        """Start animation on specific axis"""
        import matplotlib.animation as animation
        
        self._animated_axes.add(ax)
        
        if animation_type == 'magnitude':
            # Pulsing magnitude animation
            original_data = np.abs(self.mode_x)
            
            def animate_magnitude(frame):
                pulse = 1.0 + 0.3 * np.sin(frame * 0.2)
                new_data = original_data * pulse
                self.mode_ims[mode_idx].set_array(new_data)
                return [self.mode_ims[mode_idx]]
                
            ani = animation.FuncAnimation(ax.figure, animate_magnitude, frames=100,
                                        interval=50, blit=True, repeat=True)
                                        
        elif animation_type == 'phase':
            # Rotating phase animation
            def animate_phase(frame):
                phase_shift = frame * 0.1
                new_data = np.angle(self.mode_y * np.exp(1j * phase_shift) + 
                                   1j * self.mode_z * np.exp(1j * phase_shift))
                self.mode_ims[mode_idx].set_array(new_data)
                return [self.mode_ims[mode_idx]]
                
            ani = animation.FuncAnimation(ax.figure, animate_phase, frames=100,
                                        interval=50, blit=True, repeat=True)
                                        
        else:  # combined - temporal oscillation
            # True temporal evolution: Re[A * e^(iφ) * e^(iωt)]
            A_mag = np.abs(self.mode_z)
            A_phase = np.angle(self.mode_z + 1j * 0.1)  # Add small imaginary component
            omega = 2 * np.pi * 0.05  # Slow oscillation for visibility
            
            def animate_temporal(frame):
                t = frame * 0.1
                new_data = A_mag * np.cos(A_phase + omega * t)
                self.mode_ims[mode_idx].set_array(new_data)
                return [self.mode_ims[mode_idx]]
                
            ani = animation.FuncAnimation(ax.figure, animate_temporal, frames=100,
                                        interval=50, blit=True, repeat=True)
        
        # Store animation reference
        self._mode_animations[ax] = ani
        print(f"Animacja {animation_type} uruchomiona!")
        
    def _stop_mode_animation(self, ax, mode_idx):
        """Stop animation and restore static plot"""
        if ax in self._mode_animations:
            self._mode_animations[ax].event_source.stop()
            del self._mode_animations[ax]
            
        if ax in self._animated_axes:
            self._animated_axes.remove(ax)
            
        # Restore original static data
        mode_data = [np.abs(self.mode_x), np.angle(self.mode_y + 1j*self.mode_z), np.real(self.mode_z)]
        self.mode_ims[mode_idx].set_array(mode_data[mode_idx])
        ax.figure.canvas.draw()
        print("Animacja zatrzymana, przywrócono statyczny obraz")

def main():
    """Run the animation test"""
    print("=== Test Animacji Double-Click MMPP ===")
    print("1. Uruchamianie testu z mock danymi...")
    
    # Create mock MMPP instance
    mmpp_test = MockMMPP()
    
    # Create interactive figure
    fig = mmpp_test.create_interactive_test()
    
    print("2. Instrukcja testowania:")
    print("   - Kliknij dwukrotnie na wykres 'Magnitude |mx|' -> animacja pulsacji")
    print("   - Kliknij dwukrotnie na wykres 'Phase φy' -> animacja rotacji fazy")
    print("   - Kliknij dwukrotnie na wykres 'Combined mz' -> animacja temporalna")
    print("   - Drugi double-click zatrzymuje animację")
    print("3. Zamknij okno aby zakończyć test")
    
    plt.show()
    print("Test zakończony.")

if __name__ == "__main__":
    main()