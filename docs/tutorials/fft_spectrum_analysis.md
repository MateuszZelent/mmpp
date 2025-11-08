# Zaawansowana Analiza Spektrum FFT

Ten przewodnik pokazuje jak wykorzystać pełną funkcjonalność analizy FFT w MMPP do badania symulacji mikromagnetycznych.

## 🎯 Wprowadzenie do API FFT

MMPP zapewnia intuicyjny interfejs FFT dostępny poprzez `result.fft`. Po wywołaniu `print(result.fft)` otrzymasz szczegółowy przewodnik:

```python
import mmpp

# Otwórz dane symulacji
db = mmpp.open('/ścieżka/do/danych')
result = db.find(solver=3)[0]

# Pokaż dostępne metody FFT
print(result.fft)
```

Wyświetli się:
```
🔬 MMPP FFT Analysis Interface
======================================================================
📁 Job Path: /ścieżka/do/zadania  
💾 Cache Entries: 0
🎯 Mode Analysis: ✓ Available

🔧 CORE FFT METHODS:
──────────────────────────────────────────────────────
  • spectrum()      Get complex FFT spectrum
    └─ freqs, spec = result.fft.spectrum('m_z11', z_layer=-1)
  • frequencies()   Get frequency array  
    └─ result.fft.frequencies()
  • power()         Get power spectrum |FFT|²
    └─ result.fft.power()
  • magnitude()     Get magnitude |FFT|
    └─ result.fft.magnitude()
  • phase()         Get phase spectrum
    └─ result.fft.phase()
  • plot_spectrum() Plot power spectrum
    └─ fig, ax = result.fft.plot_spectrum()
  • clear_cache()  Clear computation cache
    └─ result.fft.clear_cache()

🌊 MODE ANALYSIS METHODS:
──────────────────────────────────────────────────────
  • modes                  Access mode interface
    └─ result.fft.modes.interactive_spectrum()
  • interactive_spectrum() Interactive spectrum+modes
    └─ result.fft.interactive_spectrum()
```

## 📊 Podstawowe Metody Spektrum

### 1. Spektrum Kompleksowe - `spectrum()`

```python
import numpy as np
import matplotlib.pyplot as plt

# Pobierz spektrum kompleksowe
fft = result.fft
freqs, spectrum = fft.spectrum(dset='m_z11', z_layer=-1)

print(f"Typ spektrum: {spectrum.dtype}")  # complex128
print(f"Kształt: {spectrum.shape}")
print(f"Zakres amplitud: {np.min(np.abs(spectrum)):.2e} - {np.max(np.abs(spectrum)):.2e}")
```

### 2. Częstotliwości - `frequencies()`

```python
# Pobierz tablicę częstotliwości
frequencies = fft.frequencies()
freqs_ghz = frequencies / 1e9  # Konwersja na GHz

print(f"Rozdzielczość częstotliwościowa: {(frequencies[1] - frequencies[0])/1e6:.1f} MHz")
print(f"Zakres częstotliwości: {freqs_ghz[0]:.2f} - {freqs_ghz[-1]:.2f} GHz")
print(f"Liczba punktów: {len(frequencies)}")
```

### 3. Spektrum Mocy - `power()`

```python
# Spektrum mocy |FFT|²
power = fft.power(dset='m_z11')

# Znajdź częstotliwość rezonansową
peak_idx = np.argmax(power)
resonance_freq = frequencies[peak_idx] / 1e9
peak_power = power[peak_idx]

print(f"Częstotliwość rezonansowa: {resonance_freq:.3f} GHz")
print(f"Moc rezonansu: {peak_power:.2e}")

# Oblicz współczynnik jakości Q (w przybliżeniu)
half_max = peak_power / 2
half_max_indices = np.where(power >= half_max)[0]
if len(half_max_indices) > 1:
    bandwidth = (frequencies[half_max_indices[-1]] - frequencies[half_max_indices[0]]) / 1e9
    q_factor = resonance_freq / bandwidth
    print(f"Przybliżony współczynnik Q: {q_factor:.1f}")
```

### 4. Moduł i Faza - `magnitude()`, `phase()`

```python
# Moduł spektrum
magnitude = fft.magnitude(dset='m_z11')

# Spektrum fazowe
phase = fft.phase(dset='m_z11')
phase_degrees = np.degrees(phase)

# Analiza fazowa przy rezonansie
resonance_phase = phase_degrees[peak_idx]
print(f"Faza przy rezonansie: {resonance_phase:.1f}°")

# Wykres porównawczy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.semilogy(freqs_ghz, magnitude)
ax1.set_ylabel('|FFT| (a.u.)')
ax1.set_title('Moduł Spektrum')
ax1.grid(True, alpha=0.3)

ax2.plot(freqs_ghz, phase_degrees)
ax2.set_xlabel('Częstotliwość (GHz)')
ax2.set_ylabel('Faza (°)')
ax2.set_title('Spektrum Fazowe')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🎨 Wizualizacja Spektrum

### Podstawowy wykres - `plot_spectrum()`

```python
# Automatyczny wykres z pełną funkcjonalnością
fig, ax = fft.plot_spectrum(
    dset='m_z11',
    method=1,
    z_layer=-1,
    log_scale=True,      # Skala logarytmiczna
    normalize=False,     # Bez normalizacji
    save=True,          # Zapisz dane FFT
    figsize=(12, 8)     # Rozmiar wykresu
)

# Dostosuj wykres
ax.set_xlim(0, 5)  # Ograniczenie do 0-5 GHz
ax.set_title('Spektrum FMR - Komponent Z', fontsize=16)
ax.grid(True, alpha=0.3)
plt.show()
```

### Porównanie komponentów

```python
# Porównaj wszystkie komponenty magnetyzacji
components = ['m_x11', 'm_y11', 'm_z11']
colors = ['red', 'green', 'blue']

fig, ax = plt.subplots(figsize=(12, 8))

for comp, color in zip(components, colors):
    power = fft.power(dset=comp)
    ax.semilogy(freqs_ghz, power, label=comp, color=color, linewidth=2)

ax.set_xlabel('Częstotliwość (GHz)', fontsize=14)
ax.set_ylabel('Moc (a.u.)', fontsize=14)
ax.set_title('Porównanie Komponentów Magnetyzacji', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 🌊 Analiza Modów FMR

### Dostęp do interfejsu modów

```python
# Interfejs analizy modów
modes = fft.modes

# Wyświetl informacje o modach
print(f"Analiza modów dostępna: {modes is not None}")
```

### 1. Automatyczna detekcja szczytów

```python
# Znajdź szczyty w spektrum
peaks = modes.find_peaks(
    threshold=0.1,        # 10% maksymalnej amplitudy
    min_distance=10,      # Minimalna odległość między szczytami
    component=2           # Komponent Z (0=x, 1=y, 2=z)
)

print(f"Znaleziono {len(peaks)} szczytów:")
for i, peak in enumerate(peaks):
    print(f"  {i+1}. {peak.freq:.3f} GHz (amplituda: {peak.amplitude:.2e})")
```

### 2. Interaktywna wizualizacja

```python
# Interaktywne spektrum z modami
fig = modes.interactive_spectrum(
    components=['x', 'y', 'z'],  # Wszystkie komponenty
    z_layer=0,                   # Dolna warstwa
    method=1,                    # Metoda FFT
    figsize=(16, 10),           # Duży wykres
    show=True                   # Pokaż natychmiast
)

# Instrukcje użytkowania:
print("Interakcja z wykresem:")
print("- Lewy klik: wybierz dowolną częstotliwość")
print("- Prawy klik: snap do najbliższego szczytu")
print("- Mody są automatycznie aktualizowane")
```

### 3. Wizualizacja modów przy określonej częstotliwości

```python
# Wybierz częstotliwość pierwszego szczytu
if peaks:
    target_freq = peaks[0].freq
    
    # Wizualizuj mody
    fig, axes = modes.plot_modes(
        frequency=target_freq,
        z_layer=0,
        components=['x', 'y', 'z'],
        save_path=f'modes_{target_freq:.1f}GHz.png'
    )
    
    print(f"Mody przy {target_freq:.3f} GHz zapisane do pliku")
    
    # Dostosuj wyświetlanie
    for ax in axes.flat:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
```

### 4. Obliczanie przestrzennych modów

```python
# Oblicz pełne mody przestrzenne
modes.compute_modes(
    z_slice=slice(None),  # Wszystkie warstwy Z
    window=True,          # Zastosuj okienkowanie
    save=True,           # Zapisz do zarr
    force=False          # Użyj cache jeśli możliwe
)

# Sprawdź zapisane dane
import zarr
z = zarr.open(result.path)
if 'fft' in z:
    print("Dostępne dane FFT:", list(z['fft'].keys()))
    
    # Sprawdź rozmiar danych modów
    for key in z['fft'].keys():
        data = z['fft'][key]
        print(f"  {key}: {data.shape} ({data.dtype})")
```

## 🔧 Zaawansowane Techniki

### Analiza wielowarstwowa

```python
# Analiza różnych warstw Z
z_layers = [-1, 0, 1, 2]  # Góra, dół, i warstwy pośrednie
layer_powers = {}

for z_layer in z_layers:
    try:
        power = fft.power(dset='m_z11', z_layer=z_layer)
        layer_powers[z_layer] = power
        
        # Znajdź szczyt dla tej warstwy
        peak_idx = np.argmax(power)
        peak_freq = frequencies[peak_idx] / 1e9
        print(f"Warstwa Z={z_layer}: szczyt przy {peak_freq:.3f} GHz")
        
    except Exception as e:
        print(f"Warstwa Z={z_layer}: {e}")

# Wykres porównawczy warstw
fig, ax = plt.subplots(figsize=(12, 8))
for z_layer, power in layer_powers.items():
    ax.semilogy(freqs_ghz, power, label=f'Z-layer {z_layer}')

ax.set_xlabel('Częstotliwość (GHz)')
ax.set_ylabel('Moc (a.u.)')
ax.set_title('Spektra dla różnych warstw Z')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Analiza szerokości piku i jakości

```python
def analyze_peak_quality(frequencies, power, peak_freq_ghz):
    """Analiza jakości szczytu rezonansowego."""
    
    # Znajdź indeks szczytu
    freqs_ghz = frequencies / 1e9
    peak_idx = np.argmin(np.abs(freqs_ghz - peak_freq_ghz))
    peak_power = power[peak_idx]
    
    # Znajdź punkty półmaksymalne (FWHM)
    half_max = peak_power / 2
    
    # Szukaj lewej strony
    left_idx = peak_idx
    while left_idx > 0 and power[left_idx] > half_max:
        left_idx -= 1
    
    # Szukaj prawej strony  
    right_idx = peak_idx
    while right_idx < len(power) - 1 and power[right_idx] > half_max:
        right_idx += 1
    
    # Oblicz FWHM
    if left_idx < peak_idx < right_idx:
        fwhm = freqs_ghz[right_idx] - freqs_ghz[left_idx]
        q_factor = peak_freq_ghz / fwhm
        
        return {
            'peak_freq': peak_freq_ghz,
            'peak_power': peak_power,
            'fwhm': fwhm,
            'q_factor': q_factor,
            'left_freq': freqs_ghz[left_idx],
            'right_freq': freqs_ghz[right_idx]
        }
    
    return None

# Analiza wszystkich szczytów
power = fft.power(dset='m_z11')
frequencies = fft.frequencies()

for i, peak in enumerate(peaks[:3]):  # Pierwsze 3 szczyty
    analysis = analyze_peak_quality(frequencies, power, peak.freq)
    
    if analysis:
        print(f"\n=== SZCZYT {i+1} ===")
        print(f"Częstotliwość: {analysis['peak_freq']:.3f} GHz")
        print(f"FWHM: {analysis['fwhm']:.3f} GHz") 
        print(f"Współczynnik Q: {analysis['q_factor']:.1f}")
        print(f"Zakres FWHM: {analysis['left_freq']:.3f} - {analysis['right_freq']:.3f} GHz")
```

### Batch processing wielu plików

```python
# Analiza wielu plików jednocześnie
results = db.find(solver=3, limit=5)
analysis_summary = []

for i, result in enumerate(results):
    try:
        fft = result.fft
        power = fft.power(dset='m_z11')
        frequencies = fft.frequencies()
        
        # Znajdź główny szczyt
        peak_idx = np.argmax(power)
        peak_freq = frequencies[peak_idx] / 1e9
        peak_power = power[peak_idx]
        
        analysis_summary.append({
            'result_index': i,
            'path': result.path,
            'peak_freq_ghz': peak_freq,
            'peak_power': peak_power
        })
        
        print(f"Plik {i}: {peak_freq:.3f} GHz")
        
    except Exception as e:
        print(f"Błąd w pliku {i}: {e}")

# Podsumowanie statystyczne
import pandas as pd
df = pd.DataFrame(analysis_summary)
print("\nPodsumowanie:")
print(f"Średnia częstotliwość: {df['peak_freq_ghz'].mean():.3f} ± {df['peak_freq_ghz'].std():.3f} GHz")
print(f"Zakres częstotliwości: {df['peak_freq_ghz'].min():.3f} - {df['peak_freq_ghz'].max():.3f} GHz")
```

## 🔧 Optymalizacja i Cache

### Zarządzanie cache

```python
# Sprawdź status cache
print("Wpisy w cache:", len(fft._cache))
print("Klucze cache:", list(fft._cache.keys())[:3])  # Pierwsze 3

# Wymuś przeliczenie z nowymi parametrami
freqs_forced, spectrum_forced = fft.spectrum(
    dset='m_z11',
    method=2,        # Inna metoda
    force=True       # Ignoruj cache
)

# Wyczyść cache gdy potrzeba
fft.clear_cache()
print("Cache wyczyszczony")
```

### Optymalizacja obliczeń

```python
# Użyj save=True aby zapisać wyniki do zarr
freqs, spectrum = fft.spectrum(dset='m_z11', save=True)
print("Spektrum zapisane do pliku zarr")

# Kolejne wywołania będą szybsze (z zarr)
freqs_cached, spectrum_cached = fft.spectrum(dset='m_z11')  # Ładuje z zarr
print("Spektrum załadowane z cache")

# Sprawdź co zostało zapisane
z = zarr.open(result.path)
if 'fft' in z:
    print("Zapisane zestawy FFT:", list(z['fft'].keys()))
```

## 🎯 Podsumowanie

Interfejs FFT w MMPP zapewnia:

1. **Intuicyjne API** podobne do numpy.fft
2. **Automatyczne cache'owanie** dla optymalizacji
3. **Zaawansowaną wizualizację** modów FMR
4. **Interaktywne narzędzia** do eksploracji danych
5. **Batch processing** dla wielu plików
6. **Szczegółową analizę** szczytów i jakości

Aby uzyskać pomoc w dowolnym momencie:
```python
print(result.fft)           # Pełny przewodnik
help(result.fft.spectrum)   # Pomoc dla konkretnej metody
```
