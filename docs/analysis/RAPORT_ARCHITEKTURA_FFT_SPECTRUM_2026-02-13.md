# Raport architektury FFT (Spectrum-first)

Data: 2026-02-13  
Zakres: `mmpp/fft` (`spectrum`, `spectrum/modes`, `transmission`, `dispersion`)  
Priorytet na ten etap: `spectrum`

## 1. Streszczenie

Refaktoryzacja API FFT jest częściowo wykonana. Obecnie współistnieją dwa modele:
- model legacy: metody na `fft` (`plot_spectrum`, `interactive_spectrum`, `plot_modes`),
- model obiektowy: `SpectrumResult` z namespace `plot` i `modes`.

To powoduje niespójność kontraktu, luki w propagacji kontekstu (szczególnie po slicingu komponentu) i utrudnia osiągnięcie pełnej modularności.

Najważniejsze problemy krytyczne (P0):
- utrata informacji o wybranym komponencie po `[..., 2]`,
- zduplikowana metoda `_clone()` w `modes/interface.py` i cicha utrata konfiguracji.

## 2. Wynik przeglądu modułów

- `spectrum`: średnio zaawansowana migracja do API obiektowego, ale nadal z silnym legacy wrapperem.
- `spectrum/modes`: funkcjonalnie bogaty, nadal mocno zależny od legacy `FMRModeAnalyzer`.
- `transmission`: spójny interfejs obiektowy (`FFTTransmissionInterface`), najmniej konfliktów z refaktorem.
- `dispersion`: funkcjonalnie dojrzały, ale duży i monolityczny plik interfejsu; mniej pilny niż `spectrum`.

## 3. Błędy architektoniczne (focus: spectrum)

### P0-1: Utrata kontekstu komponentu po slicingu `[..., 2]`

**Dowód:**
- `mmpp/core/dataset.py:451` (int indeks jest konwertowany do slica `2:3`),
- `mmpp/fft/core.py:343` (detekcja komponentu działa tylko dla `int`, nie dla `slice(2,3)`),
- `mmpp/fft/modes/interface.py:556` (`component_index` wykrywa tylko `int`).

**Efekt:**
- `SpectrumResult.component_label` jest `None` mimo wyboru komponentu,
- `SpectrumResult._single_component=False` mimo wyboru jednego komponentu,
- `spec.modes` nie auto-przełącza się na poprawny komponent.

### P0-2: Zduplikowana `_clone()` i utrata konfiguracji w `modes`

**Dowód:**
- `mmpp/fft/modes/interface.py:622` i `mmpp/fft/modes/interface.py:755` (druga definicja nadpisuje pierwszą).

**Efekt:**
- po `configure(tmax=..., filters=..., cache_dir=...)`, wywołanie `filters(...)`/`clear_filters()` gubi `tmax`, `filters_config`, `cache_dir`.

### P1-1: API legacy + API obiektowe działają równolegle bez jasnego funnelu

**Dowód:**
- legacy entrypoint: `mmpp/fft/core.py:700` (`FFT.plot_spectrum`),
- nowy entrypoint: `mmpp/fft/spectrum/result.py:99` (`spec.plot`),
- dokumentacja nadal miesza style, np. `README.md:71`, `docs/tutorials/fft_spectrum_analysis.md:30`.

**Efekt:**
- brak jednego canonical API,
- większa złożoność utrzymania i trudniejsza nauka interfejsu.

### P1-2: `filters` nie ma analogicznego helpera jak `spectrum`

**Dowód:**
- `FFT.spectrum` to property-helper (`mmpp/fft/core.py:183`),
- `FFT.filters` to metoda (`mmpp/fft/core.py:196`),
- `data.fft.filters().spectrum` zwraca metodę (bound method), nie helper/UI card.

**Efekt:**
- UX `filters` jest niespójny względem `spectrum`,
- użytkownik odbiera `filters().spectrum` jako "nie działa" w trybie notebook-help.

### P1-3: Część parametrów filtrów pre-FFT jest tracona w mapowaniu chaina

**Dowód:**
- `mmpp/fft/spectrum/filter_chain.py:24` mapuje pre-filters tylko do nazw `filter_type`,
- parametry opcji (np. dla `high_pass`, `band_pass`) nie są przekazywane dalej jako argumenty numeryczne.

**Efekt:**
- brak pełnej sterowalności filtrów przez fluent chain,
- ryzyko "cichego" ignorowania parametrów użytkownika.

### P1-4: `spec.modes` mutuje stan współdzielonego interfejsu

**Dowód:**
- `mmpp/fft/spectrum/modes/bridge.py:39` i `mmpp/fft/spectrum/modes/bridge.py:41` ustawiają `_dataset_context`/`_slice_context` na wspólnym obiekcie.

**Efekt:**
- możliwy efekt uboczny przy równoległej pracy na wielu `SpectrumResult`.

### P2-1: Nazewnictwo pól nie domyka target API

**Dowód:**
- `SpectrumResult` ma `frequencies`, `spectrum`, `amplitude` (`mmpp/fft/spectrum/result.py:29`, `mmpp/fft/spectrum/result.py:30`, `mmpp/fft/spectrum/result.py:55`),
- brak aliasów `freqs`, `data` oczekiwanych w modelu użytkownika.

## 4. Ocena zgodności z docelowym API (Twoja wizja)

Docelowy styl:

```python
spectrum = data.fft.spectrum()
spectrum.plot.spectrum()
spectrum.power
spectrum.data
spectrum.freqs
spectrum.amplitude

mode = spectrum.modes.at(f=0.560)
mode.plot.imshow(aspect="auto")
```

Status:
- `spectrum = data.fft.spectrum()` -> jest,
- `spectrum.plot.spectrum()` -> jest,
- `spectrum.power`, `spectrum.amplitude` -> jest,
- `spectrum.data`, `spectrum.freqs` -> brak aliasów,
- `spectrum.modes.at(...).plot.imshow(...)` -> jest, ale cierpi na błąd P0-1 przy slicingu komponentu.

## 5. Rekomendowany plan wdrożenia (kolejność)

### Etap A (P0, natychmiast)

1. Naprawić detekcję komponentu dla `slice(k, k+1)`:
- `mmpp/fft/core.py` (`_spectrum_impl`),
- `mmpp/fft/modes/interface.py` (`component_index`).

2. Usunąć duplikat `_clone()` w `mmpp/fft/modes/interface.py` i zachować pełne kopiowanie konfiguracji.

3. Dodać testy regresyjne:
- komponent po `[...,2]`,
- trwałość `configure(...).filters(...).plot()`.

### Etap B (P1, konsolidacja spectrum API)

1. Ustalić canonical API: `data.fft.spectrum() -> SpectrumResult` + `SpectrumResult.plot.*`.
2. Dodać aliasy kompatybilności:
- `SpectrumResult.data -> spectrum`,
- `SpectrumResult.freqs -> frequencies`.
3. Dodać ostrzeżenia deprecacyjne dla legacy entrypointów:
- `FFT.plot_spectrum()`,
- `FFT.interactive_spectrum()`,
- `FFT.plot_modes()` (z komunikatem „użyj `spec.plot...` / `spec.modes...`).

### Etap C (P1/P2, `filters` jako pełny moduł)

1. Dodać helper dla `data.fft.filters` analogiczny do `data.fft.spectrum` (czytelny card/help + fluent API).
2. Ujednolicić zachowanie:
- `data.fft.filters(...).spectrum()` zwraca ten sam typ (`SpectrumResult`) i ten sam flow co `data.fft.spectrum()`.
3. Rozważyć obiekt `FilteredFFTData` (jeśli potrzebny), np.:
- `.data` (lazy filtered time-domain),
- `.spectrum()`, `.modes`, `.plot`.

### Etap D (dok/cleanup)

1. Ujednolicić dokumentację i przykłady do nowego API.
2. Ograniczyć „legacy surface” do cienkich adapterów + deprecations.

## 6. Minimalne kryteria akceptacji po wdrożeniu Etapu A+B

- `job[0].m[:200, ..., 2].fft.spectrum().component_label == "$m_z$"`.
- `job[0].m[:200, ..., 2].fft.modes.component_index == 2`.
- `job[0].fft.modes.configure(tmax=500).filters(freq_min=2).clear_filters()` nie gubi `tmax`.
- `SpectrumResult` posiada aliasy `.data` i `.freqs`.
- legacy `plot_spectrum` działa, ale emituje jasne `DeprecationWarning`.

## 7. Wniosek

Twoja docelowa modularność jest osiągalna bez „big-bang rewrite”. Najpierw trzeba zamknąć dwie krytyczne luki (P0), potem ustandaryzować jeden kontrakt `SpectrumResult-first` i dopiero na końcu wyciszać legacy API.
