#!/usr/bin/env python3
"""
Przykłady ręcznego uruchamiania charakteryzacji modów w MMPP
"""

# =============================================================================
# PRZYKŁAD 1: Podstawowa charakteryzacja z pliku zarr
# =============================================================================

from mmpp.fft.modes import FMRModeAnalyzer

# Załaduj analizator modów
zarr_path = "path/to/your/simulation.zarr"
analyzer = FMRModeAnalyzer(zarr_path)

# Sprawdź czy mody są dostępne
if analyzer.modes_available:
    print(f"Dostępne częstotliwości: {analyzer.frequencies.min():.1f} - {analyzer.frequencies.max():.1f} GHz")
    
    # Charakteryzuj mod przy konkretnej częstotliwości
    frequency = 15.0  # GHz
    result = analyzer.characterize_mode(frequency)
    
    print(f"Częstotliwość: {result.frequency:.2f} GHz")
    print(f"Klasa główna: {result.primary_class}")
    print(f"Indeks m: {result.m_index}")
    print(f"Rotacja: {result.rotation_sense}")
    print(f"Węzły radialne: {result.radial_nodes}")
    print(f"Pewność: {result.confidence:.2f}")
    print(f"Etykiety: {', '.join(result.labels)}")

# =============================================================================
# PRZYKŁAD 2: Charakteryzacja z dodatkowymi parametrami
# =============================================================================

from mmpp.fft.mode_characterization import ModeCharacteristicConfig

# Utwórz niestandardową konfigurację
config = ModeCharacteristicConfig(
    relative_amplitude_threshold=0.1,  # Wyższy próg amplitudy
    gyration_parallel_ratio=0.7,       # Inne kryteria klasyfikacji
    breathing_perp_ratio=0.6
)

# Charakteryzuj z niestandardową konfiguracją
result = analyzer.characterize_mode(
    frequency=12.5, 
    z_layer=0,
    config=config,
    core_position=(64, 32),  # Ręcznie podana pozycja centrum
    analysis_radius=50.0     # Promień analizy w pikselach
)

print(f"Niestandardowa analiza: {result.primary_class} (pewność: {result.confidence:.2f})")

# =============================================================================
# PRZYKŁAD 3: Batch charakteryzacja wielu częstotliwości
# =============================================================================

import numpy as np

# Wybierz częstotliwości do analizy
target_frequencies = np.linspace(8, 25, 10)  # 10 punktów między 8-25 GHz

results = []
for freq in target_frequencies:
    try:
        result = analyzer.characterize_mode(freq)
        results.append({
            'frequency': freq,
            'class': result.primary_class,
            'm_index': result.m_index,
            'confidence': result.confidence,
            'rotation': result.rotation_sense
        })
        print(f"{freq:6.2f} GHz: {result.primary_class:10s} m={result.m_index} ({result.confidence:.2f})")
    except Exception as e:
        print(f"{freq:6.2f} GHz: ERROR - {e}")

# =============================================================================
# PRZYKŁAD 4: Charakteryzacja z bezpośrednim dostępem do danych modu
# =============================================================================

# Pobierz surowe dane modu
frequency = 18.0
mode_data = analyzer.get_mode(frequency, z_layer=0)

print(f"Dane modu:")
print(f"  Częstotliwość: {mode_data.frequency:.2f} GHz")
print(f"  Kształt tablicy: {mode_data.mode_array.shape}")
print(f"  Rozdzielczość: {mode_data.metadata.get('spatial_resolution', 'N/A')}")

# Użyj bezpośrednio analizatora charakteryzacji
from mmpp.fft.mode_characterization import ModeCharacterAnalyzer

char_analyzer = ModeCharacterAnalyzer()
result = char_analyzer.analyze(mode_data)

print(f"Bezpośrednia analiza: {result.primary_class}")

# =============================================================================
# PRZYKŁAD 5: Interface FFT z indeksowaniem
# =============================================================================

from mmpp.fft import FFT

# Załaduj FFT interface
fft = FFT(zarr_path)

# Dostęp przez indeks wyniku FFT (jeśli masz wiele wyników)
modes_interface = fft[0]  # Pierwszy wynik FFT

# Dostęp przez indeks częstotliwości
freq_interface = modes_interface[50]  # 50-ty punkt częstotliwości

# Charakteryzuj bezpośrednio
result = freq_interface.characterize()
print(f"Interface wynik: {result.primary_class} przy {freq_interface.frequency:.2f} GHz")

# =============================================================================
# PRZYKŁAD 6: Charakteryzacja z wizualizacją
# =============================================================================

import matplotlib.pyplot as plt

frequency = 15.0
result = analyzer.characterize_mode(frequency)

# Pokaż mod i jego charakteryzację
fig, axes = analyzer.plot_modes(frequency)
fig.suptitle(f'Mod {result.primary_class.upper()} przy {frequency:.1f} GHz\n'
             f'm={result.m_index}, rotacja: {result.rotation_sense or "N/A"}, '
             f'pewność: {result.confidence:.2f}')

plt.tight_layout()
plt.show()

# =============================================================================
# PRZYKŁAD 7: Eksport wyników charakteryzacji
# =============================================================================

import json

# Charakteryzuj wszystkie dostępne częstotliwości
all_results = {}
for i, freq in enumerate(analyzer.frequencies[::10]):  # Co 10-ty punkt
    try:
        result = analyzer.characterize_mode(freq)
        all_results[f'freq_{i:03d}'] = {
            'frequency_ghz': float(freq),
            'primary_class': result.primary_class,
            'm_index': result.m_index,
            'rotation_sense': result.rotation_sense,
            'radial_nodes': result.radial_nodes,
            'confidence': float(result.confidence),
            'labels': result.labels,
            'notes': result.notes,
            'diagnostics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in result.diagnostics.items()}
        }
    except Exception as e:
        all_results[f'freq_{i:03d}'] = {'error': str(e), 'frequency_ghz': float(freq)}

# Zapisz do pliku JSON
with open('mode_characterization_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Wyniki zapisane do mode_characterization_results.json")

# =============================================================================
# PRZYKŁAD 8: Filtrowanie modów według typu
# =============================================================================

# Znajdź wszystkie mody gyracyjne
gyration_modes = []
breathing_modes = []
azimuthal_modes = []

test_frequencies = analyzer.frequencies[::5]  # Co 5-ty punkt dla szybkości

for freq in test_frequencies:
    try:
        result = analyzer.characterize_mode(freq)
        
        if result.primary_class == 'gyration' and result.confidence > 0.3:
            gyration_modes.append((freq, result))
        elif result.primary_class == 'breathing' and result.confidence > 0.3:
            breathing_modes.append((freq, result))
        elif result.primary_class == 'azimuthal' and result.confidence > 0.3:
            azimuthal_modes.append((freq, result))
            
    except Exception:
        continue

print(f"\nZnalezione mody (pewność > 0.3):")
print(f"  Gyracyjne: {len(gyration_modes)} modów")
print(f"  Oddechowe: {len(breathing_modes)} modów")
print(f"  Azymutalne: {len(azimuthal_modes)} modów")

# Wyświetl szczegóły najlepszych modów
print(f"\nNajlepsze mody gyracyjne:")
for freq, result in sorted(gyration_modes, key=lambda x: x[1].confidence, reverse=True)[:3]:
    print(f"  {freq:6.2f} GHz: m={result.m_index}, {result.rotation_sense}, pewność={result.confidence:.2f}")

# =============================================================================
# PRZYKŁAD 9: Porównanie różnych warstw Z
# =============================================================================

if analyzer.n_z_layers > 1:
    frequency = 15.0
    
    print(f"\nPortównanie warstw Z dla {frequency:.1f} GHz:")
    for z_layer in range(min(3, analyzer.n_z_layers)):  # Max 3 warstwy
        try:
            result = analyzer.characterize_mode(frequency, z_layer=z_layer)
            print(f"  Warstwa {z_layer}: {result.primary_class} "
                  f"(m={result.m_index}, pewność={result.confidence:.2f})")
        except Exception as e:
            print(f"  Warstwa {z_layer}: ERROR - {e}")

# =============================================================================
# PRZYKŁAD 10: Tworzenie raportu charakteryzacji
# =============================================================================

def create_characterization_report(analyzer, output_file="mode_report.txt"):
    """Utwórz szczegółowy raport charakteryzacji"""
    
    with open(output_file, 'w') as f:
        f.write("RAPORT CHARAKTERYZACJI MODÓW MMPP\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Plik danych: {analyzer.zarr_path}\n")
        f.write(f"Dataset: {analyzer.dataset_name}\n")
        f.write(f"Dostępne częstotliwości: {len(analyzer.frequencies)}\n")
        f.write(f"Zakres: {analyzer.frequencies.min():.2f} - {analyzer.frequencies.max():.2f} GHz\n")
        f.write(f"Warstwy Z: {analyzer.n_z_layers}\n\n")
        
        # Analizuj co 20-ty punkt
        sample_frequencies = analyzer.frequencies[::20]
        
        f.write("WYNIKI CHARAKTERYZACJI:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Freq [GHz]':<10} {'Klasa':<12} {'m':<5} {'Rotacja':<8} {'Pewność':<8} {'Etykiety'}\n")
        f.write("-" * 70 + "\n")
        
        for freq in sample_frequencies:
            try:
                result = analyzer.characterize_mode(freq)
                f.write(f"{freq:<10.2f} {result.primary_class:<12} "
                       f"{str(result.m_index):<5} {str(result.rotation_sense or 'N/A'):<8} "
                       f"{result.confidence:<8.2f} {', '.join(result.labels[:3])}\n")
            except Exception as e:
                f.write(f"{freq:<10.2f} ERROR: {str(e)}\n")
        
        f.write("\nKONIEC RAPORTU\n")
    
    print(f"Raport zapisany do: {output_file}")

# Utwórz raport
create_characterization_report(analyzer)

print("\n=== GOTOWE! ===")
print("Sprawdź pliki:")
print("- mode_characterization_results.json")  
print("- mode_report.txt")