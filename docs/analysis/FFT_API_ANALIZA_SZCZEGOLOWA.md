# Szczegółowa Dokumentacja API FFT - MMPP

## 🔬 Wprowadzenie do Analizy Spektrum FFT

MMPP zapewnia zaawansowany interfejs analizy FFT podobny do `numpy.fft`, ale specjalnie dostosowany do danych z symulacji mikromagnetycznych. Główny punkt dostępu to klasa `FFT`, która dostępna jest poprzez `result.fft`.

## 🎯 Podstawowe użycie

```python
import mmpp

# Otwórz bazę danych
db = mmpp.open('/ścieżka/do/danych')
result = db.find(solver=3)[0]

# Dostęp do interfejsu FFT
fft = result.fft

# Wyświetl dostępne metody
print(fft)  # Pokaże szczegółowy opis wszystkich metod
```

Po wykonaniu `print(fft)` zostanie wyświetlony szczegółowy przewodnik:

```
🔬 MMPP FFT Analysis Interface
======================================================================
📁 Job Path: /ścieżka/do/zadania
💾 Cache Entries: 3
🎯 Mode Analysis: ✓ Available

🔧 CORE FFT METHODS:
──────────────────────────────────────────────────────
  • spectrum()      Get complex FFT spectrum
    └─ freqs, spec = op[0].fft.spectrum('m_z11', z_layer=-1)
  • frequencies()   Get frequency array
    └─ op[0].fft.frequencies()
  • power()         Get power spectrum |FFT|²
    └─ op[0].fft.power()
  • magnitude()     Get magnitude |FFT|
    └─ op[0].fft.magnitude()
  • phase()         Get phase spectrum
    └─ op[0].fft.phase()
  • plot_spectrum() Plot power spectrum
    └─ fig, ax = op[0].fft.plot_spectrum()
  • clear_cache()  Clear computation cache
    └─ op[0].fft.clear_cache()

🌊 MODE ANALYSIS METHODS:
──────────────────────────────────────────────────────
  • modes                  Access mode interface
    └─ op[0].fft.modes.interactive_spectrum()
  • [index]               Index-based mode access
    └─ op[0].fft[0][200].plot_modes()
  • plot_modes()          Plot modes at frequency
    └─ op[0].fft.plot_modes(frequency=1.5)
  • interactive_spectrum() Interactive spectrum+modes
    └─ op[0].fft.interactive_spectrum()
```

## 📊 Główne Metody FFT

### 1. `spectrum()` - Spektrum Kompleksowe

Zwraca kompletne kompleksowe spektrum FFT.

```python
# Podstawowe użycie
freqs, complex_spectrum = fft.spectrum()

# Z parametrami
freqs, complex_spectrum = fft.spectrum(
    dset='m_z11',          # Zestaw danych: 'm_z11', 'm_x11', 'm_y11'
    z_layer=-1,            # Warstwa Z: -1 (góra), 0 (dół), 1, 2, ...
    method=1,              # Metoda FFT: 1, 2, 3
    save=True,             # Zapisz wyniki do zarr
    force=False            # Wymuś ponowne obliczenie
)

print(f"Spektrum shape: {complex_spectrum.shape}")
print(f"Typ danych: {complex_spectrum.dtype}")  # complex128
```

### 2. `frequencies()` - Tablica Częstotliwości

Zwraca tablicę częstotliwości odpowiadającą spektrum.

```python
# Pobierz częstotliwości
freqs = fft.frequencies()
print(f"Zakres częstotliwości: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
print(f"Rozdzielczość: {(freqs[1] - freqs[0]):.2e} Hz")

# Konwersja na GHz (typowe dla FMR)
freqs_ghz = freqs / 1e9
print(f"Zakres w GHz: {freqs_ghz[0]:.2f} - {freqs_ghz[-1]:.2f} GHz")
```

### 3. `power()` - Spektrum Mocy

Zwraca spektrum mocy |FFT|².

```python
# Spektrum mocy
power_spectrum = fft.power(dset='m_z11')

# Znajdź częstotliwość szczytową
freqs = fft.frequencies()
peak_idx = np.argmax(power_spectrum)
peak_freq = freqs[peak_idx] / 1e9  # w GHz
peak_power = power_spectrum[peak_idx]

print(f"Szczyt przy: {peak_freq:.3f} GHz")
print(f"Moc szczytu: {peak_power:.2e}")
```

### 4. `magnitude()` - Moduł Spektrum

Zwraca moduł spektrum |FFT|.

```python
# Moduł spektrum
magnitude = fft.magnitude(dset='m_z11')

# Normalizacja
magnitude_norm = magnitude / np.max(magnitude)

# Analiza szczytów
from scipy.signal import find_peaks
peaks, properties = find_peaks(magnitude_norm, height=0.1)
freqs = fft.frequencies()
peak_freqs = freqs[peaks] / 1e9

print(f"Znalezione szczyty: {peak_freqs} GHz")
```

### 5. `phase()` - Spektrum Fazowe

Zwraca fazę spektrum.

```python
# Spektrum fazowe
phase_spectrum = fft.phase(dset='m_z11')

# Konwersja na stopnie
phase_degrees = np.degrees(phase_spectrum)

# Analiza fazowa
print(f"Zakres faz: {np.min(phase_degrees):.1f}° - {np.max(phase_degrees):.1f}°")
```

### 6. `plot_spectrum()` - Wykres Spektrum

Tworzy wykres spektrum mocy z automatycznym formatowaniem.

```python
# Podstawowy wykres
fig, ax = fft.plot_spectrum()

# Zaawansowane opcje
fig, ax = fft.plot_spectrum(
    dset='m_z11',
    method=1,
    z_layer=-1,
    log_scale=True,        # Skala logarytmiczna
    normalize=False,       # Normalizacja
    save=True,            # Zapisz dane FFT
    force=False,          # Wymuś przeliczenie
    figsize=(12, 8),      # Rozmiar figury
    title="Spektrum FMR"  # Tytuł
)

# Dostosowanie wykresu
ax.set_xlim(0, 5)  # Ograniczenie osi X (GHz)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 🌊 Analiza Modów FMR

### Dostęp do interfejsu modów

```python
# Interfejs analizy modów
modes = fft.modes

# Wyświetl dostępne metody modów
print(modes)
```

### 1. `interactive_spectrum()` - Interaktywne Spektrum

Tworzy interaktywny wykres spektrum z wizualizacją modów.

```python
# Podstawowe użycie
fig = fft.modes.interactive_spectrum()

# Z parametrami
fig = fft.modes.interactive_spectrum(
    components=['x', 'y', 'z'],  # Komponenty do pokazania
    z_layer=0,                   # Warstwa Z
    method=1,                    # Metoda FFT
    show=True,                   # Pokaż wykres
    figsize=(16, 10),           # Rozmiar figury
    dpi=100                     # Rozdzielczość
)

# Interakcja:
# - Lewy klik: wybierz dokładną częstotliwość
# - Prawy klik: snapuj do najbliższego szczytu
```

### 2. `plot_modes()` - Wykres Modów

Wizualizuje mody przy określonej częstotliwości.

```python
# Wizualizacja modów przy 2.5 GHz
fig, axes = fft.modes.plot_modes(
    frequency=2.5,              # Częstotliwość w GHz
    z_layer=0,                  # Warstwa Z
    components=['x', 'y', 'z'], # Komponenty
    save_path='modes_2_5GHz.png'  # Ścieżka zapisu
)

# Dostosowanie wyświetlania
for ax in axes.flat:
    ax.set_aspect('equal')
```

### 3. `find_peaks()` - Detekcja Szczytów

Automatycznie znajduje szczyty w spektrum.

```python
# Znajdź szczyty
peaks = fft.modes.find_peaks(
    threshold=0.1,      # Próg detekcji (względny)
    min_distance=10,    # Minimalna odległość między szczytami
    component=2         # Komponent do analizy (0=x, 1=y, 2=z)
)

# Analiza znalezionych szczytów
for i, peak in enumerate(peaks):
    print(f"Szczyt {i+1}:")
    print(f"  Częstotliwość: {peak.freq:.3f} GHz")
    print(f"  Amplituda: {peak.amplitude:.2e}")
    print(f"  Indeks: {peak.idx}")
```

### 4. `compute_modes()` - Obliczanie Modów

Oblicza przestrzenne mody FMR.

```python
# Oblicz mody
fft.modes.compute_modes(
    z_slice=slice(None),  # Wybór warstw Z
    window=True,          # Zastosuj okienkowanie
    save=True,           # Zapisz wyniki
    force=False          # Wymuś przeliczenie
)

# Po obliczeniu mody są dostępne w zarr
import zarr
z = zarr.open(result.path)
print("Dostępne dane modów:", list(z['fft'].keys()))
```

## ⚙️ Parametry Konfiguracyjne

### Wspólne parametry dla wszystkich metod:

- **`dset`**: Nazwa zestawu danych
  - `'m_z11'` (domyślne) - komponent Z
  - `'m_x11'` - komponent X  
  - `'m_y11'` - komponent Y
  - `'m_z11-14'` - zestaw warstw Z

- **`z_layer`**: Indeks warstwy Z
  - `-1` (domyślne) - górna warstwa
  - `0` - dolna warstwa
  - `1, 2, 3...` - kolejne warstwy

- **`method`**: Metoda FFT
  - `1` (domyślne) - standardowa metoda
  - `2, 3` - alternatywne metody

- **`save`**: Zapis wyników
  - `True` - zapisz do pliku zarr
  - `False` - tylko w pamięci

- **`force`**: Wymuszenie przeliczenia
  - `True` - ignoruj cache, przelicz ponownie
  - `False` - użyj cache jeśli dostępny

## 📈 Przykłady Praktyczne

### Analiza podstawowego spektrum FMR

```python
import matplotlib.pyplot as plt
import numpy as np

# Pobierz dane
fft = result.fft
freqs = fft.frequencies() / 1e9  # w GHz
power = fft.power()
phase = fft.phase()

# Utwórz wykres
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Spektrum mocy
ax1.semilogy(freqs, power)
ax1.set_xlabel('Częstotliwość (GHz)')
ax1.set_ylabel('Moc (a.u.)')
ax1.set_title('Spektrum Mocy FMR')
ax1.grid(True, alpha=0.3)

# Spektrum fazowe
ax2.plot(freqs, np.degrees(phase))
ax2.set_xlabel('Częstotliwość (GHz)')
ax2.set_ylabel('Faza (°)')
ax2.set_title('Spektrum Fazowe')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Porównanie komponentów

```python
# Porównaj wszystkie komponenty
components = ['m_x11', 'm_y11', 'm_z11']
fig, axes = plt.subplots(len(components), 1, figsize=(12, 10))

for i, comp in enumerate(components):
    power = fft.power(dset=comp)
    freqs = fft.frequencies() / 1e9
    
    axes[i].semilogy(freqs, power, label=comp)
    axes[i].set_ylabel('Moc (a.u.)')
    axes[i].set_title(f'Komponent {comp}')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[-1].set_xlabel('Częstotliwość (GHz)')
plt.tight_layout()
plt.show()
```

### Analiza szczytów i modów

```python
# Znajdź szczyty
peaks = fft.modes.find_peaks(threshold=0.1)
print(f"Znaleziono {len(peaks)} szczytów")

# Analizuj każdy szczyt
for i, peak in enumerate(peaks[:3]):  # Pierwsze 3 szczyty
    print(f"\n=== SZCZYT {i+1} ===")
    print(f"Częstotliwość: {peak.freq:.3f} GHz")
    
    # Wizualizuj mody przy tej częstotliwości
    fig, axes = fft.modes.plot_modes(
        frequency=peak.freq,
        components=['x', 'y', 'z']
    )
    
    # Zapisz wykres
    plt.savefig(f'mody_szczyt_{i+1}_{peak.freq:.1f}GHz.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
```

### Cache i optymalizacja

```python
# Sprawdź status cache
print("Cache info:", fft._cache.keys())

# Wymuś przeliczenie z nowymi parametrami
freqs_high_res, spectrum_high_res = fft.spectrum(
    method=2,      # Inna metoda
    force=True     # Wymuś przeliczenie
)

# Wyczyść cache jeśli potrzeba
fft.clear_cache()
print("Cache wyczyszczony")
```

## 🔧 Rozwiązywanie Problemów

### Sprawdzenie dostępności danych

```python
# Sprawdź czy dane istnieją
try:
    freqs, spectrum = fft.spectrum()
    print("✓ Dane FFT dostępne")
except Exception as e:
    print(f"✗ Błąd: {e}")

# Sprawdź dostępne zestawy danych
import zarr
z = zarr.open(result.path)
available_datasets = list(z.keys())
print("Dostępne zestawy:", available_datasets)
```

### Debugowanie obliczeń

```python
# Włącz tryb debug
fft._compute.debug = True

# Sprawdź metadane
freqs, spectrum = fft.spectrum()
result_info = fft._compute.get_last_result_info()
print("Informacje o obliczeniu:", result_info)
```

## 📚 Podsumowanie

Interfejs FFT w MMPP zapewnia:

1. **Podstawowe metody FFT**: `spectrum()`, `frequencies()`, `power()`, `magnitude()`, `phase()`
2. **Wizualizację**: `plot_spectrum()` z zaawansowanymi opcjami
3. **Analizę modów**: `modes.interactive_spectrum()`, `modes.plot_modes()`, `modes.find_peaks()`
4. **Optymalizację**: automatyczny cache, konfigurowalne metody
5. **Integrację**: bezpośrednia praca z plikami zarr, kompatybilność z matplotlib

Wszystkie metody są szczegółowo udokumentowane i można uzyskać pomoc poprzez `help(fft.method_name)` lub `print(fft)` dla pełnego przeglądu.
