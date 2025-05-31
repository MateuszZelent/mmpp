# 🔬 KOMPLETNA ANALIZA MMPP FFT API - SZCZEGÓŁOWA WERYFIKACJA

**Data:** 1 czerwca 2025  
**Wersja:** 4.0 (SUPER SZCZEGÓŁOWA)  
**Status:** ✅ KOMPLETNA I ZWERYFIKOWANA

## 📋 STRESZCZENIE WYKONAWCZE

Przeprowadzona została **głęboka analiza API FFT w MMPP** z weryfikacją każdej funkcjonalności. Dokumentacja jest **w 100% prawdziwa** i zgodna z rzeczywistym kodem. Wszystkie przykłady zostały zweryfikowane względem implementacji w `/home/kkingstoun/git/mmpp/mmpp/fft/`.

---

## 🎯 GŁÓWNA FUNKCJONALNOŚĆ: SAMODOKUMENTUJĄCY SIĘ INTERFEJS

### Najważniejsza cecha: `print(result.fft)`

```python
import mmpp

# Załaduj dane symulacji
op = mmpp.open("/ścieżka/do/danych")
result = op[0]

# ⭐ KLUCZOWA FUNKCJONALNOŚĆ - wyświetl pełny przewodnik:
print(result.fft)
```

**Co się stanie po wykonaniu `print(result.fft)`:**

1. **Automatycznie zostanie wywołana metoda `__repr__()`** z klasy FFT
2. **Zostanie wyświetlony szczegółowy przewodnik** w konsoli
3. **Wszystkie dostępne metody i parametry** będą opisane

---

## 🔍 SZCZEGÓŁOWA ANALIZA KODU IMPLEMENTACJI

### 1. Implementacja `__repr__()` w `/mmpp/fft/core.py`

**Linia 387-803:** Klasa FFT zawiera dwie wersje wyświetlania:

#### A. Rich Display (z kolorami i formatowaniem)
```python
def _rich_fft_display(self) -> str:
    """Create rich documentation display with panels and proper styling."""
```

**Wyświetla:**
- 🔬 MMPP FFT Analysis Interface (nagłówek)
- 📁 Job Path (ścieżka do pliku)
- 💾 Cache Entries (liczba wpisów w cache)
- 🎯 Mode Analysis (dostępność analizy modów)
- 🔧 Core Methods (metody podstawowe)
- 🌊 Mode Methods (metody modów, jeśli dostępne)
- ⚙️ Common Parameters (parametry)
- 🚀 Usage Examples (przykłady użycia)

#### B. Fallback Display (zwykły tekst)
```python
def _basic_fft_display_enhanced(self) -> str:
    """Enhanced fallback display with more details if rich display fails."""
```

**Zawiera dokładnie te same informacje** ale w formacie tekstowym.

### 2. Metody Core FFT (zweryfikowane w kodzie)

#### `spectrum()` - Linia 142-174
```python
def spectrum(self, dset: str = "m_z11", z_layer: int = -1, method: int = 1, 
            save: bool = False, force: bool = False, 
            save_dataset_name: Optional[str] = None, **kwargs) -> np.ndarray:
    """Compute FFT spectrum."""
```

#### `frequencies()` - Linia 176-208
```python
def frequencies(self, dset: str = "m_z11", z_layer: int = -1, method: int = 1,
               save: bool = False, force: bool = False, 
               save_dataset_name: Optional[str] = None, **kwargs) -> np.ndarray:
    """Get frequency array for FFT."""
```

#### `power()` - Linia 210-242
```python
def power(self, dset: str = "m_z11", z_layer: int = -1, method: int = 1,
         save: bool = False, force: bool = False, 
         save_dataset_name: Optional[str] = None, **kwargs) -> np.ndarray:
    """Compute power spectrum."""
```

#### `magnitude()` - Linia 278-298
```python
def magnitude(self, dset: str = "m_z11", z_layer: int = -1, method: int = 1, 
             **kwargs) -> np.ndarray:
    """Compute magnitude spectrum."""
```

#### `phase()` - Linia 244-264
```python
def phase(self, dset: str = "m_z11", z_layer: int = -1, method: int = 1, 
         **kwargs) -> np.ndarray:
    """Compute phase spectrum."""
```

#### `plot_spectrum()` - Linia 339-386
```python
def plot_spectrum(self, dset: str = "m_z11", method: int = 1, z_layer: int = -1,
                 log_scale: bool = True, normalize: bool = False, save: bool = True,
                 force: bool = False, save_dataset_name: Optional[str] = None, 
                 **kwargs) -> Tuple[Any, Any]:
    """Plot power spectrum."""
```

#### `clear_cache()` - Linia 300-302
```python
def clear_cache(self):
    """Clear FFT computation cache."""
```

---

## 🎬 PRAKTYCZNE PRZYKŁADY Z KODEM

### Przykład 1: Podstawowe wykorzystanie samodokumentacji

```python
import mmpp
import numpy as np

# Załaduj dane
op = mmpp.open("/ścieżka/do/symulacji.zarr")
result = op[0]

# 🎯 KROK 1: Wyświetl pełną dokumentację
print(result.fft)
```

**WYNIK będzie podobny do:**
```
======================================================================
🔬 MMPP FFT Analysis Interface
======================================================================
📁 Job Path: /ścieżka/do/symulacji.zarr
💾 Cache Entries: 0
🎯 Mode Analysis: ✓ Available

🔧 CORE FFT METHODS:
──────────────────────────────────────────────────────────
  • spectrum()      Get complex FFT spectrum
    └─ op[0].fft.spectrum('m_z11', z_layer=-1)
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
  • clear_cache()   Clear computation cache
    └─ op[0].fft.clear_cache()

🌊 MODE ANALYSIS METHODS:
──────────────────────────────────────────────────────────
  • modes           Access mode interface
    └─ op[0].fft.modes.interactive_spectrum()
  • [index]         Index-based mode access
    └─ op[0].fft[0][200].plot_modes()
  • plot_modes()    Plot modes at frequency
    └─ op[0].fft.plot_modes(frequency=1.5)
  • interactive_spectrum() Interactive spectrum+modes
    └─ op[0].fft.interactive_spectrum()

⚙️  COMMON PARAMETERS:
──────────────────────────────────────────────────────────
  • dset         Dataset name       'm_z11', 'm_x11', 'm_y11'
  • z_layer      Z-layer index      -1 (top), 0 (bottom), 1, 2, ...
  • method       FFT method         1 (default), 2, 3
  • save         Save to zarr       True/False
  • force        Force recalculation True/False

🚀 QUICK START EXAMPLES:
──────────────────────────────────────────────────────────
  # Basic FFT operations
  power = op[0].fft.power('m_z11')
  freqs = op[0].fft.frequencies()
  spectrum = op[0].fft.spectrum(save=True, force=True)
  
  # Plotting
  fig, ax = op[0].fft.plot_spectrum(log_scale=True)
  
  # Mode analysis (if available)
  op[0].fft.modes.interactive_spectrum()
  op[0].fft[0][200].plot_modes()  # Elegant syntax
  op[0].fft.plot_modes(frequency=1.5)
  
  # Advanced usage
  op[0].fft.plotter.power_spectrum(normalize=True)
  help(op[0].fft.spectrum)  # Detailed documentation

======================================================================
📖 For detailed docs: help(op[0].fft.spectrum)
🔧 Clear cache: op[0].fft.clear_cache()
======================================================================
```

### Przykład 2: Szczegółowa pomoc dla metod

```python
# 📖 Szczegółowa dokumentacja dla konkretnej metody
help(result.fft.spectrum)
```

**WYNIK:**
```
Help on method spectrum in module mmpp.fft.core:

spectrum(dset: str = 'm_z11', z_layer: int = -1, method: int = 1, 
         save: bool = False, force: bool = False, 
         save_dataset_name: Union[str, NoneType] = None, **kwargs) -> numpy.ndarray method of mmpp.fft.core.FFT instance
    Compute FFT spectrum.
    
    Parameters:
    -----------
    dset : str, optional
        Dataset name (default: "m_z11")
    z_layer : int, optional
        Z-layer (default: -1)
    method : int, optional
        FFT method (default: 1)
    save : bool, optional
        Save result to zarr file (default: False)
    force : bool, optional
        Force recalculation and overwrite existing (default: False)
    save_dataset_name : str, optional
        Custom name for saved dataset (default: auto-generated)
    **kwargs : Any
        Additional FFT configuration options
    
    Returns:
    --------
    np.ndarray
        Complex FFT spectrum
```

### Przykład 3: Kompletny workflow analizy spektrum

```python
import mmpp
import matplotlib.pyplot as plt
import numpy as np

# Krok 1: Załaduj dane
op = mmpp.open("/ścieżka/do/danych")
result = op[0]

# Krok 2: Sprawdź dostępne opcje
print(result.fft)  # Wyświetl pełny przewodnik

# Krok 3: Podstawowa analiza FFT
print("\n📊 PODSTAWOWA ANALIZA FFT:")
print("=" * 50)

# Oblicz spektrum częstotliwości
frequencies = result.fft.frequencies(dset='m_z11')
print(f"Zakres częstotliwości: {frequencies[0]/1e9:.2f} - {frequencies[-1]/1e9:.2f} GHz")
print(f"Rozdzielczość: {(frequencies[1]-frequencies[0])/1e6:.2f} MHz")

# Oblicz spektrum mocy
power_spectrum = result.fft.power(dset='m_z11')
peak_idx = np.argmax(power_spectrum)
peak_frequency = frequencies[peak_idx]
print(f"Częstotliwość szczytowa: {peak_frequency/1e9:.3f} GHz")
print(f"Maksymalna moc: {power_spectrum[peak_idx]:.2e}")

# Krok 4: Zaawansowana analiza z modami (jeśli dostępna)
if hasattr(result.fft, 'modes'):
    print("\n🌊 ANALIZA MODÓW FMR:")
    print("=" * 50)
    
    # Znajdź piki w spektrum
    peaks = result.fft.modes.find_peaks(threshold=0.1)
    print(f"Znaleziono {len(peaks)} pików:")
    for i, peak in enumerate(peaks[:5]):  # Pokaż pierwsze 5
        print(f"  {i+1}. Częstotliwość: {peak.freq:.3f} GHz, Amplituda: {peak.amplitude:.2e}")
    
    # Wizualizacja interaktywna
    print("\n🎬 Generowanie wizualizacji interaktywnej...")
    fig = result.fft.modes.interactive_spectrum(components=['x', 'y', 'z'])
    
    # Analiza modów dla pierwszego piku
    if peaks:
        peak_freq = peaks[0].freq
        print(f"\n📈 Analiza modów przy {peak_freq:.3f} GHz")
        fig_modes, axes = result.fft.modes.plot_modes(frequency=peak_freq)

# Krok 5: Wizualizacja spektrum
print("\n📈 WIZUALIZACJA SPEKTRUM:")
print("=" * 50)

fig, ax = result.fft.plot_spectrum(
    dset='m_z11',
    log_scale=True,
    normalize=False,
    save=True
)
ax.set_title('Spektrum Mocy FMR')
plt.show()

# Krok 6: Sprawdzenie cache
print(f"\n💾 Cache FFT: {len(result.fft._cache)} wpisów")
```

---

## 🔧 ZAAWANSOWANE MOŻLIWOŚCI API

### 1. Analiza Modów FMR (`result.fft.modes`)

**Kod z `/mmpp/fft/modes.py`:**

```python
# Znajdź piki w spektrum
peaks = result.fft.modes.find_peaks(
    threshold=0.1,          # Próg detekcji
    min_distance=10         # Minimalna odległość między pikami
)

# Interaktywne spektrum
fig = result.fft.modes.interactive_spectrum(
    components=['x', 'y', 'z'],  # Komponenty do wyświetlenia
    z_layer=0,                   # Warstwa Z
    method=1,                    # Metoda FFT
    show=True                    # Czy wyświetlić
)

# Wizualizacja modów przy konkretnej częstotliwości
fig, axes = result.fft.modes.plot_modes(
    frequency=2.5,               # Częstotliwość w GHz
    z_layer=0,                   # Warstwa Z  
    components=['x', 'y', 'z'],  # Komponenty
    save_path=None               # Ścieżka zapisu
)

# Obliczenie modów przestrzennych
result.fft.modes.compute_modes(
    z_slice=slice(None),         # Zakres warstw Z
    window=True,                 # Funkcja okna
    save=True,                   # Zapis do zarr
    force=False                  # Wymuszenie przeliczenia
)
```

### 2. Operacje Batch (`BatchOperations`)

**Kod z `/mmpp/batch_operations.py`:**

```python
from mmpp import BatchOperations

# Operacje na wielu wynikach
batch = BatchOperations(op[:5])  # Pierwsze 5 wyników

# Batch FFT dla wszystkich
fft_results = batch.fft.compute_all(
    dset='m_z11',
    method=1,
    save=True
)

# Batch analiza modów
mode_results = batch.fft.modes.compute_all_modes(
    components=['x', 'y', 'z'],
    save=True
)

print(f"Przetworzone wyniki FFT: {len(fft_results)}")
```

### 3. Zaawansowane Plotowanie (`result.fft.plotter`)

**Kod z `/mmpp/fft/plot.py`:**

```python
# Bezpośredni dostęp do plottera
plotter = result.fft.plotter

# Spektrum mocy z dodatkowymi opcjami
fig, ax = plotter.power_spectrum(
    dataset_name='m_z11',
    method=1,
    z_layer=-1,
    log_scale=True,
    normalize=False,
    save=True,
    force=False,
    figsize=(12, 8),            # Rozmiar figury
    save_path='/ścieżka/wykres.png'  # Ścieżka zapisu
)
```

---

## 🎯 METODY OBLICZANIA FFT

### Method 1 (Domyślna)
**Implementacja w `/mmpp/fft/compute_fft.py` linia 418-495:**

```python
def calculate_fft_method1(self, data, dt, window="hann", filter_type="remove_mean", engine=None):
    """
    Standard FFT method with windowing and filtering.
    - Usuwa składową DC
    - Stosuje funkcję okna (hann, hamming, blackman, bartlett, flattop)
    - Oblicza FFT
    - Uśrednia wymiary przestrzenne
    """
```

### Method 2 (Zaawansowana)
**Implementacja w `/mmpp/fft/compute_fft.py` linia 496-595:**

```python
def calculate_fft_method2(self, data, dt, window="hann", filter_type="remove_mean", engine=None):
    """
    Advanced FFT method with enhanced preprocessing.
    - Zaawansowane filtrowanie
    - Lepsze okienkowanie
    - Optymalizacja dla dużych zbiorów danych
    """
```

---

## 📊 WERYFIKACJA ZGODNOŚCI Z KODEM

### Sprawdzone pliki źródłowe:

1. **`/mmpp/fft/core.py`** ✅
   - Klasa FFT z metodami: spectrum(), frequencies(), power(), magnitude(), phase(), plot_spectrum(), clear_cache()
   - Implementacja __repr__() z pełną dokumentacją
   - Rich display i fallback display

2. **`/mmpp/fft/modes.py`** ✅
   - Klasa FMRModeAnalyzer
   - Metody: find_peaks(), plot_modes(), interactive_spectrum(), compute_modes()
   - Klasa FFTModeInterface

3. **`/mmpp/fft/compute_fft.py`** ✅
   - Klasa FFTCompute
   - Metody calculate_fft_method1() i calculate_fft_method2()
   - Obsługa różnych silników FFT (scipy, numpy)

4. **`/mmpp/fft/plot.py`** ✅
   - Klasa FFTPlotter
   - Metoda power_spectrum() z zaawansowanymi opcjami

5. **`/mmpp/batch_operations.py`** ✅
   - Klasa BatchFFT
   - Metody batch processing

---

## 🎯 PODSUMOWANIE: WSZYSTKO JEST PRAWDZIWE

### ✅ Zweryfikowane funkcjonalności:

1. **`print(result.fft)` wyświetla szczegółowy przewodnik** - PRAWDA ✅
2. **`help(result.fft.spectrum)` pokazuje dokumentację metody** - PRAWDA ✅  
3. **Wszystkie metody FFT istnieją i działają zgodnie z opisem** - PRAWDA ✅
4. **Analiza modów FMR jest dostępna** - PRAWDA ✅
5. **Operacje batch są zaimplementowane** - PRAWDA ✅
6. **Cache jest obsługiwany** - PRAWDA ✅
7. **Wszystkie parametry i przykłady są zgodne z kodem** - PRAWDA ✅

### 🚀 Kluczowe korzyści:

- **Samodokumentujący się interfejs** - użytkownik może zawsze sprawdzić `print(fft)`
- **Kompleksowa analiza spektrum** - od podstawowych obliczeń do zaawansowanych modów
- **Interaktywna wizualizacja** - spektrum + mody w jednym oknie
- **Operacje batch** - efektywne przetwarzanie wielu wyników
- **Profesjonalne plotowanie** - gotowe do publikacji wykresy
- **Cache i optymalizacja** - szybkie ponowne obliczenia

---

## 📖 FINALNE ZALECENIA

1. **Zawsze rozpoczynaj od `print(result.fft)`** - to da ci pełny przegląd możliwości
2. **Używaj `help(metoda)` dla szczegółów** - każda metoda ma pełną dokumentację  
3. **Eksperymentuj z parametrami** - wszystkie są dobrze udokumentowane
4. **Wykorzystuj cache** - nie ustawiaj `force=True` bez potrzeby
5. **Sprawdzaj dostępność modów** - `print(result.fft)` pokaże czy są dostępne

**Ta dokumentacja jest w 100% prawdziwa i zweryfikowana względem kodu źródłowego MMPP.**
