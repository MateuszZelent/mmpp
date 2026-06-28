# Audyt i perfekcyjny masterplan produkcyjny `fft/dispersion`

Data aktualizacji: 2026-06-28. Zakres: `mmpp/fft/dispersion`, interaktywne API
dyspersji, cache, semantyka widma, rekonstrukcja modów, benchmarki, testy,
dokumentacja i bramki release MMPP.

Ten dokument zastępuje wcześniejszą wersję raportu. Poprzedni audyt poprawnie
identyfikował główne ryzyka, ale mieszał trzy różne stany: stan sprzed napraw,
stan aktualnego drzewa roboczego oraz warunek pełnej produkcyjności biblioteki.
Poniżej rozdzielam je jawnie.

## Werdykt wykonawczy

Po aktualnych naprawach moduł `fft/dispersion` jest blisko statusu stabilnego
komponentu: ma lekki kontrakt `plot.interactive(show=False)`, rozdział `S_raw`
i `S_display`, skalowanie `raw_power`/`amplitude_squared`/`psd`, lepszy branch
tracking, import-hygiene testy, syntetyczne testy E2E modów, prosty benchmark
oraz lokalnie zielony release gate z wymaganym `widget_smoke`.

To nadal nie wystarcza do deklaracji "MMPP jest w pełni produkcyjne". Obecny
status powinien brzmieć:

- `fft/dispersion`: local production candidate, wymagający potwierdzenia w CI i
  artefaktach release.
- interaktywny notebook API: kontrakt headless, preset/export i widget smoke są
  lokalnie domknięte; pozostaje potwierdzenie w macierzy extras/CI oraz realnym
  notebooku użytkownika.
- cała biblioteka MMPP: nie production-ready, dopóki release nie instaluje i nie
  testuje wheel/sdist, extras, docs oraz deklarowanej macierzy Pythonów.

Najważniejsza zasada: poprawiony moduł nie jest równoznaczny z produkcyjną
biblioteką. Produkcyjność całego MMPP wymaga zielonych bramek release, clean
install wheel/sdist, extras matrix i docs build w CI.

## Stan aktualny po naprawach

### Zamknięte lub istotnie poprawione w drzewie roboczym

| Obszar | Aktualny stan | Dowód w kodzie/testach |
| --- | --- | --- |
| Import hygiene | Top-level import nie powinien wymagać Matplotlib/STNO UI. | `tests/test_import_hygiene.py`, leniwe importy w `mmpp/__init__.py` i analytical. |
| Lekki interaktywny kontrakt | Dodano `DispersionInteractiveViewer`, `res.plot.interactive(show=False)` i `disp.plot.interactive(...)`; release smoke wymusza `headless_imports.no_new_widget_modules=True` przed właściwym `widget_smoke`, a `viewer_status` robi z lifecycle/preset/export głównego viewera twardą bramkę. | `_interactive_viewer.py`, `_plotting/accessor.py`, `interface.py`, `verify_fft_dispersion_release_gate.py`. |
| Notatki i eksport w viewerze | `DispersionInteractiveViewer.state`, `export_selection()` i `_repr_html_` pokazują `result.notes`, w tym ostrzeżenia próbkowania; eksport jest bezpieczny dla `json.dumps(...)` także przy skalarach i tablicach NumPy. Normalizacja JSON jest wspólna dla głównego viewera, single-mode i animation viewerów, a release smoke eksportuje selekcje z NumPy scalar/array dla wszystkich trzech lekkich kontrolerów. `mode_viewers_status` zmienia top-level `status` na `failed`, jeśli headless single-mode albo animation viewer nie spełnia invariantów. | `_json.py`, `_interactive_viewer.py`, `modes/bridge.py`, `verify_fft_dispersion_release_gate.py`, testy `can_save_load_preset_and_export_selection`, `single_mode_plot_interactive`, `modes_plot_animation`, `reports_core_api`, `exposes_result_notes`. |
| Alias migracyjny | `interactive_analysis()` deleguje do `.plot.interactive(...)`. | `FFTDispersionInterface.interactive_analysis`. |
| Raw/display split | `DispersionResult1D` przenosi `S_raw`, `S_display`, `spectrum_for()`, `frequency_view()`. | `models.py`, testy raw/display. |
| Skalowanie | Dodano `raw_power`, `amplitude_squared`, `psd` i współczynniki gain. | `core.py`, testy scaling/window gain. |
| Cache schema | Bump do `DISPERSION_CACHE_SCHEMA_VERSION = 4`; cache zapisuje raw/display/scaling. | `interface.py`. |
| Cache backend/workers | Cache key rozróżnia bieżący backend FFT i liczbę workers. | `interface.py`, test `cache_separates_fft_worker_configuration`. |
| Cache axis/slice/filter | Cache key rozróżnia oś obliczeń, `slice_info` i konfigurację filtrów. | testy `cache_separates_axis_configuration`, `cache_separates_slice_context`, `cache_separates_filter_configuration`. |
| Oś `k` po `flipx` | Mapowanie `k -> -k` używa dopasowania po faktycznej osi, nie `[::-1]`; testy pokrywają parzyste osie i znak piku dla `flipx=True/False`. | `core.py`, testy `y_axis_uses_effective_spacing`, `flipx_mirrors_peak`. |
| Faza lokalna | `avg_over_orthogonal=False` zachowuje lokalne `S_local`/`S_complex` i fazę pomiędzy przekrojami ortogonalnymi; collapse nie eksportuje fałszywego `S_complex`. | test `local_spectra_preserve_orthogonal_phase_offsets`. |
| BZ folding | `fold_spectrum_1d` składa znane repliki `k0 +/- G` do FBZ i rozróżnia agregację `sum`/`max`; mask workflow wybiera te same repliki do rekonstrukcji profilu. | testy `fold_spectrum_1d_aggregates_known_bz_replicas`, `mode_mask_selects_bz_replicas`. |
| Slice z komponentem | `slice_info=(..., 2)` działa jako single-component input i zachowuje osie `k/f`. | test `scalar_component_slice`. |
| Explicit `tmax` | `configure(tmax=...)` ogranicza liczbę ładowanych próbek i nie współdzieli cache z innym `tmax`. | test `configured_tmax_controls_loaded_time_and_cache_key`. |
| Oś czasu | `t` z `.zarr` musi być jednorodne i rosnące; nieregularna oś czasu blokuje FFT zamiast tworzyć fałszywe `f_axis`. | `core.py`, testy `uniform_time_axis` i `nonuniform_time_axis`. |
| Osie przestrzenne | Jawne osie `x/y/z` z `.zarr` muszą być monotoniczne i równomierne; mogą też dostarczyć brakujące `dx/dy/dz`. | `core.py`, testy `uniform_spatial_axes` i `nonmonotonic_spatial_axis`. |
| Jakość próbkowania | Wynik niesie notatki/logi dla niskiego `T/X`, limitów Nyquista i `dk_max` przekraczającego Nyquista. | `core.py`, test `sampling_quality_warnings`. |
| Branch quality | Coverage, smoothness, SNR, confidence, rejected candidates, `analysis_source` i zaszumiony branch. | `_branch_linker.py`, `analyze.py`, testy branch/noisy branch. |
| Mode extraction | Poprawiono indeksowanie dodatnich częstotliwości i dodano syntetyczny E2E wave test. | `modes/bridge.py`, `tests/test_dispersion_mode_extraction.py`. |
| Podpisane fale E2E | `compute_1d` lokalizuje znane dodatnie i ujemne `k` na osiach `x/y`, także dla siatek nieparzystych. | test `signed_wave_on_x_y_even_and_odd_grids`. |
| 2D dispersion status | `compute_2d()`/`DispersionResult2D` są jawnie eksperymentalne i mają minimalny test osi/piku/slice. | `core.py`, docs API/tutorial, test `compute_2d_complex_wave`. |
| Notebook API dla modów | `res.modes.interactive(show=True)` przekazuje opcje startowe do legacy widgetu, `plot_interactive()` stosuje `fmax` i `lognorm`, pojedynczy `mode.plot.interactive(show=False)` zwraca lekki kontroler, a `res.modes.plot.animation(show=False)` nie jest już placeholderem. | `modes/bridge.py`, `modes/interactive.py`, testy `show_true`, `startup_options`, `single_mode_interactive` i `modes_plot_animation`. |
| Notebook `_repr_html_` | `DispersionResult1D`, `.plot`, `.analyze`, `.modes` i `DispersionInteractiveViewer` pokazują stabilne wskazówki API; karta `.plot` reklamuje główną ścieżkę `.interactive(show=False)`. | `models.py`, `_plotting/accessor.py`, `analyze.py`, `modes/bridge.py`, `_interactive_viewer.py`, test `notebook_repr_documents_public_accessors`. |
| Backend/workers policy | NumPy/SciPy parity jest testowane, pyFFTW ma test warunkowy i release smoke dla extras `fft/full`, a `MMPP_FFT_BACKEND`/`MMPP_FFT_WORKERS` są opisane w docs oraz sprawdzane świeżym importem subprocess. | `test_dispersion_*_backend*`, `test_dispersion_fft_backend_respects_environment_configuration`, release `extras-smoke`, docs API/tutorial. |
| Docs example smoke | Release gate wykonuje publiczny wzorzec `disp.plot.interactive(...)`, `disp.compute_1d(...)`, `res.plot.interactive(...)` oraz workflow modów `store_complex=True -> res.modes.interactive(show=False) -> mode.plot.interactive(show=False) -> res.modes.plot.animation(show=False)` na syntetycznym `.zarr`; `docs_example_status` zmienia top-level `status` na `failed`, jeśli którykolwiek invariant docs smoke nie przejdzie. | `verify_fft_dispersion_release_gate.py`, `tests/test_dispersion_release_gate.py`. |
| Benchmark | Dodano skrypt syntetycznego benchmarku z profilami `small-ci`/`medium-dev`/`research-reference`, czasem, peak memory i preflightem pamięci; release gate używa `small-ci`, release verify uruchamia preflight większych profili, a osobny weekly/manual workflow wykonuje `medium-dev` i preflight `research-reference`. | `scripts/analysis/benchmark_fft_dispersion.py`, `verify_fft_dispersion_release_gate.py`, `.github/workflows/fft-dispersion-benchmark.yml`, testy benchmark/release. |
| Widget smoke gate | Release gate raportuje `widget_smoke`; `--require-widget-smoke` zmienia brak `ipywidgets`/Matplotlib w twardy fail, a ścieżka `ok` sprawdza też, że legacy widget `.close()` czyści display i figurę. | `verify_fft_dispersion_release_gate.py`, `tests/test_dispersion_release_gate.py`. |
| Docs build gate | Release verify job buduje Sphinx docs przed publikacją artefaktów; docs/release workflows instalują `.[dev]` w cytowanej formie, bez ręcznego `linkify-it-py`. | `.github/workflows/docs.yml`, `.github/workflows/release.yml`, `tests/test_dispersion_release_gate.py`. |
| Extras matrix gate | Release workflow pobiera zbudowany artefakt `dist`, instaluje wheel z extras `.[fft]`, `.[interactive]`, `.[plotting]`, `.[full]` i uruchamia smoke w `--import-mode installed`; dla `fft/full` wymusza pyFFTW backend smoke, a dla `interactive/full` wymusza widget smoke. | `.github/workflows/release.yml`, `tests/test_dispersion_release_gate.py`. |
| Python metadata | `pyproject.toml`, `setup.py` i CI deklarują/testują Python 3.9-3.12. | metadata + test `declared_python_versions_match_ci_matrix`. |
| Release artifact smoke | Workflow instaluje wheel i wheel zbudowany ze sdist wraz z runtime dependencies, a smoke uruchamia `--import-mode installed`. Test workflow blokuje powrót do instalacji artefaktów przez `--no-deps`, w tym na etapie budowy wheel ze sdist, bo taki smoke nie dowodzi rozwiązywalności zależności pakietu. | `.github/workflows/release.yml`, `verify_fft_dispersion_release_gate.py`, `tests/test_dispersion_release_gate.py`. |
| README | Pokazuje nowy wzorzec `disp.plot.interactive(...)`. | `README.md`. |
| Internal `_interface` hygiene | Eksperymentalne helpery `_interface/cache.py` i `_interface/k0_filtering.py` nie są już eksportowane przez `mmpp.fft.dispersion._interface.__all__`; `K0Filter` pozostaje dostępny tylko przez jawny internal import. | `_interface/__init__.py`, `test_dispersion_internal_interface_exports_no_experimental_helpers`. |

### Nadal niezamknięte produkcyjnie

| Bloker | Dlaczego blokuje produkcję |
| --- | --- |
| Brak zielonego przebiegu extras gates w CI | Release workflow ma macierz wheel+extras dla `.[fft]`, `.[interactive]`, `.[plotting]`, `.[full]`, ale nie ma jeszcze dowodu z rzeczywistego zielonego joba CI. |
| Brak potwierdzenia extras/widget smoke w CI | Lokalnie `--require-widget-smoke` przechodzi w conda base z `IPython`/`ipywidgets`/Matplotlib, ale release musi jeszcze potwierdzić to po instalacji extra `interactive/full`. |
| Niepełne wykonanie dużych benchmarków | `medium-dev` ma weekly/manual workflow, a `research-reference` ma preflight pamięciowy. Pełne wykonanie FFT profilu `research-reference` pozostaje poza release gate, żeby nie blokować szybkich publikacji. |
| Niepełna walidacja backendów | NumPy/SciPy są pokryte syntetycznie; pyFFTW i workers policy wymagają bramki warunkowej. |
| Brak pełnego lokalnego docs/example matrix | README, API docs i tutorial pokazują nowy wzorzec dyspersji, ale lokalnie nie wykonano pełnego Sphinx build ani smoke wszystkich przykładów spoza `fft/dispersion`. Metadata `dev` deklaruje już `myst-parser` i `linkify-it-py`, a workflows używają cytowanego `pip install -e ".[dev]"`, więc konfiguracja `myst_enable_extensions = ["linkify"]` nie wymaga ręcznej zależności poza extra. |
| Status pakietu Alpha | `pyproject.toml` nadal deklaruje `Development Status :: 3 - Alpha`; przy tym statusie nie wolno mówić o pełnej produkcyjności całej biblioteki. |

## Definicja produkcyjności

Nie wolno używać jednego słowa "produkcyjne" dla wszystkiego. Są trzy poziomy:

| Poziom | Definicja | Status |
| --- | --- | --- |
| Production candidate `fft/dispersion` | Stabilne osie, cache, raw/display, skalowanie, syntetyczne testy fizyczne, backend parity i benchmark smoke. | Lokalnie osiągnięte dla checkoutu; wymaga potwierdzenia release/install i CI. |
| Stable notebook API | `disp.plot.interactive()`, `res.plot.interactive(show=False)`, `res.modes.interactive()`, legacy aliases, `_repr_html_`, headless tests i widget smoke. | Lokalnie osiągnięte: `--require-widget-smoke` przechodzi i `headless_imports` pozostaje puste. |
| Release-ready MMPP | CI na Python 3.9-3.12, wheel/sdist clean install, extras matrix, docs build, przykład API smoke, klasyfikatory zgodne ze statusem. | Nieosiągnięte. |

Pełna produkcyjność biblioteki może zostać zadeklarowana dopiero po trzecim
poziomie. Po aktualnych poprawkach wolno powiedzieć: "`fft/dispersion` jest
lokalnym production candidate"; nie wolno jeszcze powiedzieć, że całe MMPP jest
production-ready.

## Architektura docelowa

Docelowa ścieżka użytkownika ma być spójna z nowszym stylem `SpectrumResult`:

```python
disp = result.fft.dispersion

viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    fmax=25,
    cache="/tmp/mmpp-dispersion",
    show=False,
)

res = disp.compute_1d(axis="x", store_complex=False)
viewer = res.plot.interactive(show=False)
```

Dla modów:

```python
res = disp.compute_1d(
    axis="x",
    avg_over_orthogonal=False,
    store_complex=True,
    scaling="amplitude_squared",
)
modes = res.modes.interactive(show=False, lattice_constant_nm=470)
```

Stara ścieżka ma pozostać jako legacy przez minimum jeden cykl wydania:

```python
modes = disp.dispersion_modes(result=res, lattice_constant_nm=470)
modes.plot_interactive()
```

Ważne rozdzielenie:

- `res.plot.interactive(...)` eksploruje zapisane `S(k,f)` i może działać bez
  surowych danych.
- `res.modes.interactive(...)` rekonstruuje mody tylko, gdy dostępne są
  `S_complex` oraz kontekst źródła.
- `disp.plot.interactive(...)` jest convenience API: oblicza lub ładuje wynik,
  a potem deleguje do tego samego viewer-a.

## Rejestr ryzyk

| Priorytet | Ryzyko | Skutek | Wymagana kontrola |
| --- | --- | --- | --- |
| P0 | Wheel/sdist nie są testowane po instalacji. | Kod działa w checkoutcie, ale release może być zepsuty. | Clean install smoke dla wheel i sdist. |
| P0 | Interaktywny viewer bez zielonego widget smoke w środowisku `interactive`. | `show=False` przechodzi, ale notebook może nie działać ergonomicznie. | Lokalnie zamknięte w conda base przez `--require-widget-smoke`; utrzymać jako bramkę po instalacji extra `interactive/full` w release. |
| P0 | Cache miesza wyniki po zmianie osi/skali/filtrów. | Ciche błędy numeryczne. | Lokalnie zamknięte testami cache context dla osi, slice, scaling, filters, `store_complex` oraz tablicowych wartości filtrów; utrzymać w CI. |
| P1 | Progi wydajności w release obejmują tylko small smoke. | Duże przypadki badawcze mogą degradować bez alarmu release. | `small-ci` zostaje w release gate, a `.github/workflows/fft-dispersion-benchmark.yml` uruchamia weekly/manual `medium-dev` oraz preflight `research-reference`. |
| P1 | Niepełna backend parity. | Wyniki zależą od backendu FFT. | NumPy/SciPy zawsze, pyFFTW warunkowo. |
| P1 | Dokumentacja rozmija się z API. | Użytkownik wraca do legacy wzorców. | Docs example smoke. |
| P2 | `DispersionResult2D` ma niepełny kontrakt względem 1D. | Użytkownik może oczekiwać raw/display/cache/scaling jak w `compute_1d`. | Jawnie experimental + minimalny test 2D. |

## Masterplan perfekcyjny

Każda faza ma mierzalny rezultat, pliki do dotknięcia i bramki. Fazy można
realizować iteracyjnie, ale status produkcyjny wolno podnieść tylko po przejściu
odpowiednich bramek.

### Faza 0: baseline, importy i higiena

Cel: testy dyspersji muszą startować bez zależności UI i bez artefaktów runtime.

Status: wykonane lokalnie dla checkoutu; wymaga potwierdzenia w CI i clean
release artifact smoke.

Zakres końcowy:

- Utrzymać leniwe importy dla `mmpp`, `mmpp.fft`, `mmpp.fft.dispersion`,
  `mmpp.analytical` i `mmpp.cli`.
- Nie importować Matplotlib/IPython/ipywidgets na poziomie modułów obliczeniowych.
- Utrzymać `.gitignore` dla `__pycache__`, `.pyc`, lokalnych presetów,
  notebooków roboczych i animacji.
- Usunąć lub oznaczyć jako experimental szkielety `_interface/cache.py` i
  niepełne ścieżki `k0_filtering`, jeśli są nadal dostępne publicznie.
  Ten punkt jest domknięty: `_interface.__all__` jest puste, więc `K0Filter`
  nie jest reklamowany jako publiczny eksport; pozostaje tylko jawny import ze
  ścieżki wewnętrznej `_interface.k0_filtering`.

Bramka:

```bash
python -c "import mmpp; import mmpp.fft; import mmpp.fft.dispersion"
python -m pytest tests/test_import_hygiene.py tests/test_dispersion_mode_extraction.py -q
python -m ruff check mmpp/fft/dispersion tests/test_import_hygiene.py tests/test_dispersion_mode_extraction.py
```

Definicja done: minimalny import i testy obliczeniowe nie wymagają UI ani danych
zewnętrznych.

### Faza 1: stabilny interaktywny kontrakt notebookowy

Cel: dyspersja ma mieć taki sam standard ergonomii jak spectrum/modes.

Status: wykonane lokalnie dla checkoutu. Istnieje lekki viewer, `show=False`,
display lifecycle, testy startowych opcji `show=True` dla legacy widgetu,
publiczne `InteractiveDispersionModes.close()` oraz bramka `widget_smoke` w
release gate. Lokalny przebieg
`verify_fft_dispersion_release_gate.py --require-widget-smoke` zwraca
`status=ok`, `headless_imports.new_widget_modules=[]`, `viewer_status=ok`,
`mode_viewers_status=ok`, `docs_example_status=ok` i `widget_smoke.status=ok`.

Zakres końcowy:

- `DispersionInteractiveViewer` musi mieć stabilne `.show()`, `.close()`,
  `.state`, `_repr_html_`, `save_preset()`, `load_preset()` i
  `export_selection()`. Release smoke waliduje te invarianty przez
  `viewer_status`, w tym preset round-trip, display lifecycle cleanup i
  JSON-safe eksport selekcji.
- `DispersionInteractiveViewer.state` i `export_selection()` mają zwracać dane
  bezpieczne dla `json.dumps(...)`, również gdy opcje albo selekcje zawierają
  skalarne typy NumPy lub krótkie tablice NumPy. Ten sam prywatny helper ma być
  używany przez viewery single-mode i animation, żeby uniknąć rozjazdu
  notebookowych eksportów. Release smoke ma wykonywać ten eksport z wartościami
  NumPy dla głównego viewera, single-mode viewera i animation viewera, żeby
  regresja nie przeszła dopiero w notebooku użytkownika; `mode_viewers_status`
  robi z tego twardą bramkę.
- `show=False` nie może importować ani odpalać IPython display.
  Release smoke mierzy nowe moduły `IPython`/`ipywidgets`/`matplotlib` załadowane
  od samego początku `run_release_gate`, czyli obejmuje także przygotowanie
  import path i pierwsze importy `mmpp.fft.dispersion`; wymaga pustej listy i
  zwraca `status=failed`, jeśli ta lista nie jest pusta.
- `show=True` musi działać w notebooku i nie zostawiać żywych figur/widgetów po
  `.close()`. Legacy `InteractiveDispersionModes.close()` zatrzymuje animację,
  czyści display handle i zamyka figurę; release `widget_smoke` sprawdza ten
  cleanup, gdy opcjonalne zależności są dostępne.
- `components`, `mode_components`, `spectrum_components`, `animate` i
  `auto_animate` muszą zachować kompatybilność migracyjną ze spectrum/modes.
- `res.modes.interactive(show=False)` bez `_interface` ma działać jako tryb
  spectrum-only z jasnym powodem braku rekonstrukcji.
- `mode.plot.interactive(show=False)` ma zwracać lekki, testowalny kontroler
  pojedynczego modu zamiast `NotImplementedError`.
- `res.modes.plot.animation(show=False)` ma zwracać lekki kontroler żądania
  animacji, żeby publiczny accessor `.modes.plot` nie reklamował martwej metody.
- Lekkie kontrolery `mode.plot.interactive(show=False)` i
  `res.modes.plot.animation(show=False)` mają eksportować stan bezpieczny dla
  `json.dumps(...)`, także gdy opcje lub selekcje zawierają skalarne typy NumPy.
- `dispersion_modes(...).plot_interactive()` pozostaje legacy aliasem, ale ma
  prowadzić przez wspólny kontroler albo mieć jasno opisane różnice.

Bramka:

```bash
python -m pytest tests/test_dispersion_mode_extraction.py -q -k "interactive"
python -m pytest tests/test_spectrum_modes_bridge.py -q
python -m pytest tests/ -q -k "dispersion and interactive"
python scripts/analysis/verify_fft_dispersion_release_gate.py --require-widget-smoke
```

Bramka release: ten sam smoke musi przejść po instalacji `.[interactive]` albo
`.[full]`, a następnie po instalacji wheel/sdist w trybie `--import-mode
installed`.

Definicja done: API zwraca kontroler, jest testowalne headless i ma zielony
smoke pełnego renderowania notebookowego w środowisku z zależnościami
`interactive`.

### Faza 2: osie, slicing i cache

Cel: `f_axis`, `k_axis`, cache i metadane muszą odpowiadać faktycznie
analizowanym próbkom po slicing/subsamplingu.

Status: w dużej części wykonane. Efektywne stride, slice z pojedynczym
komponentem, explicit `tmax`, axis/slice/filter/backend/workers w cache key,
jednorodność osi czasu, walidacja jawnych osi przestrzennych, diagnostyka
jakości próbkowania oraz parzyste mapowanie `flipx` po osi `k` zostały
uwzględnione. Pozostaje utrzymanie tej macierzy jako bramki regresyjnej przy
kolejnych zmianach osi/cache.

Zakres końcowy:

- Testy dla `m[::2, ...]`, `m[:, :, ::2, ::2, :]`, osi `x` i `y`, slicing z
  komponentem oraz explicit `tmax`. Oś `y` po spatial stride ma już syntetyczny
  test piku `k/f`; slice z komponentem ma test single-component; publiczny
  `configure(tmax=...)` ma test ładowania próbek i separacji cache. Cache key ma
  bramki dla `axis`, `slice_info`, `filters`, `scaling`, `store_complex` oraz
  backend/workers.
- Cache key musi rozróżniać: dataset, slice, effective axes, `store_complex`,
  `scaling`, filters, `positive_frequencies`, backend-affecting options i schema.
  Backend/workers, oś, slice i filtry są już pokryte testami; pozostałe elementy
  utrzymać jako checklistę regresyjną przy kolejnych zmianach cache.
- Wykrywanie niejednorodnej osi czasu oraz niemonotonicznych osi przestrzennych.
  Niejednorodna oś czasu jest już blokowana; jawne osie `x/y/z` są walidowane
  pod kątem monotoniczności i równomiernego kroku.
- Ostrzeżenia dla niskiej liczby próbek, ryzyka aliasingu, zbyt grubego `df/dk`
  i branch tracking poza Nyquistem. Podstawowe notatki/logi dla `T/X`, limitów
  Nyquista i `dk_max` poza Nyquistem są już pokryte testem; docelowo warto
  wyświetlić je też w interaktywnym viewerze.

Bramka:

```bash
python -m pytest tests/ -q -k "dispersion and (slicing or cache or axis)"
python -m pytest tests/test_dispersion_mode_extraction.py -q -k "cache or slicing"
```

Definicja done: równoważny sygnał bez slicing i ze stride daje przewidywalne osie,
a cache nigdy nie zwraca wyniku dla innej konfiguracji numerycznej.

### Faza 3: semantyka widma i skalowanie

Cel: amplitudy i PSD mają być porównywalne między długościami okna, backendami,
filtrami i trybami wizualizacji.

Status: znacząco poprawione. Raw/display, scaling i notatki jakości mają testy,
a API docs oraz tutorial opisują kontrakt `S_raw`, `S_display`, alias
`result.S`, jawny `analysis_source="display"` oraz lokalne
`S_local_raw`/`S_local_display`/`S_local`. Lokalna ścieżka
`avg_over_orthogonal=False` ma teraz analogiczny kontrakt raw/display, co zamyka
ryzyko mieszania filtrowanych widoków lokalnych z analizą raw. Pozostaje
utrzymanie szerszych bramek tolerancji przy kolejnych zmianach numerycznych.

Zakres końcowy:

- `S_raw` jest źródłem analitycznym; `S_display` jest widokiem po filtrach.
- `result.S` pozostaje aliasem kompatybilności, ale dokumentacja ma jasno mówić,
  czy wskazuje display czy raw.
- `sample_at_k()` i branch tracking domyślnie pracują na raw, chyba że użytkownik
  jawnie ustawi `analysis_source="display"`.
- Lokalne przekroje mają `S_local_raw` i `S_local_display`; `S_local` pozostaje
  aliasem aktywnego widoku display dla kompatybilności.
- Zapis i odczyt cache zachowują `S_local_raw` oraz `S_local_display`, więc
  odtworzony wynik nie traci lokalnego kontraktu raw/display.
- `apply_live_filters(..., apply_to_local=True)` aktualizuje lokalny display i
  zachowuje lokalny raw, więc szybkie filtry interaktywne nie zanieczyszczają
  źródła analitycznego.
- `amplitude_squared` i `psd` mają testy tolerancji dla Hann/no-window oraz
  różnych `T` i `X`.
- `positive_frequencies=True` jest domyślne dla heatmap/viewer/branch, ale pełna
  oś pozostaje dostępna dla zastosowań fazowych.
- `DispersionInteractiveViewer.state` i eksport selekcji przenoszą `result.notes`,
  więc ostrzeżenia próbkowania są widoczne także w trybie headless.

Bramka:

```bash
python -m pytest tests/test_dispersion_mode_extraction.py -q -k "scaling or raw or filters or frequency_view"
python -m pytest tests/ -q -k "fft and scaling"
```

Definicja done: zmiana okna, długości sygnału lub live filter nie zmienia cicho
semantyki analitycznej.

### Faza 4: walidacja fizyczna, mody i branch tracking

Cel: poprawność ma być udowodniona na syntetycznych danych o znanym `k`, `f` i
fazie, a nie tylko na ręcznie skonstruowanym `S_complex`.

Status: wykonane w zakresie masterplanu. Jest syntetyczny E2E wave test, poprawka
indeksowania częstotliwości, bramka znaku `flipx`, test zachowania lokalnej
fazy przy `avg_over_orthogonal=False`, test agregacji replik BZ, E2E dla
dodatnich/ujemnych `k` na osiach `x/y` i siatkach parzystych/nieparzystych oraz
branch tracking z szumem. Workflow mask BZ jest pokryty testem rekonstrukcji
profilu z replik `k0 +/- G`.

Zakres końcowy:

- E2E dla `exp(i(kx - omega t))` na osi `x` i `y`, dodatniego i ujemnego `k`,
  parzystych i nieparzystych siatek. Pokryte testem `signed_wave_on_x_y`.
- Testy `flipx=True/False` oraz wpływu `avg_over_orthogonal` na fazę. Znak piku
  dla `flipx=True/False` i zachowanie fazy lokalnych przekrojów są już pokryte.
- Test BZ folding: znana stała sieci, kopie `k0 +/- nG`, maska BZ, liczba binów.
  Agregacja replik `k0 +/- G` przez `fold_spectrum_1d` jest już pokryta; pełny
  workflow mode mask jest pokryty testem `mode_mask_selects_bz_replicas`.
- Branch tracking z szumem: coverage, SNR, smoothness, confidence, rejected
  reason. Pokryte testem `noisy_branch`.
- `DispersionResult2D`: decyzja stable vs experimental i minimalne testy 2D albo
  jawne ograniczenie dokumentacyjne. Status jest teraz experimental; minimalny
  test pokrywa osie `kx/ky/f`, lokalizację piku i `slice_1d`.

Bramka:

```bash
python -m pytest tests/test_dispersion_mode_extraction.py -q -k "wave or mode or branch"
python -m pytest tests/ -q -k "dispersion and (mode or branch or bz)"
```

Definicja done: pipeline od `.zarr` do maksimum `S(k,f)` i profilu modu ma
kontrolowaną tolerancję amplitudy, fazy i indeksowania.

### Faza 5: wydajność i pamięć

Cel: moduł ma mieć mierzalną odporność na regresje czasu i RAM.

Status: wykonane w zakresie masterplanu dla bramek ciągłych. Benchmark ma
profile `small-ci`, `medium-dev` i `research-reference`, progi czasu/peak
memory dla small smoke, raport rozmiarów wyniku oraz preflight pamięciowy.
Release gate uruchamia profil `small-ci` z progami czasu/RAM, release verify
sprawdza preflight pamięciowy `medium-dev` i `research-reference`, a osobny
workflow `.github/workflows/fft-dispersion-benchmark.yml` uruchamia weekly/manual
pełny `medium-dev` z `store_complex` i `S_local` oraz preflight
`research-reference`. Raport JSON rozróżnia też `s_local_raw_mb` i
`s_local_display_mb`, więc koszt lokalnego kontraktu raw/display jest widoczny w
profilach bez uśredniania ortogonalnego. Pełne wykonanie FFT profilu
`research-reference` pozostaje poza release gate z powodów kosztu.

Zakres końcowy:

- Utrzymać `scripts/analysis/benchmark_fft_dispersion.py` jako oficjalny smoke.
- Dodać profile dla minimum trzech kształtów: small CI, medium developer,
  representative research. Profile są zdefiniowane jako `small-ci`,
  `medium-dev` i `research-reference`.
- Raportować: elapsed, peak memory, rozmiary `S_raw`, `S_display`, `S_complex`,
  `S_local`, `S_local_raw`, `S_local_display`, backend, workers, scaling i
  `store_complex`.
- Ustalić progi regresji w CI dla small smoke; release gate używa
  `profile="small-ci"` z limitami elapsed/peak memory. Większe benchmarki mają
  preflight pamięciowy w release verify, a weekly/manual workflow wykonuje
  `medium-dev` z `--store-complex --no-orthogonal-average`.
- Dodać preflight pamięciowy z rozbiciem na raw data, signal, spatial FFT,
  temporal FFT, power, complex cache i local spectra. Raport JSON zawiera
  `memory_preflight_mb` z tym rozbiciem, a CLI ma `--preflight-only`.
- Rozważyć chunkowaną ścieżkę `.zarr` dla długich osi czasu lub wielu przekrojów
  ortogonalnych; jeśli nie wchodzi do release, musi być jawnie wpisana jako
  ograniczenie.

Bramka:

```bash
python scripts/analysis/benchmark_fft_dispersion.py \
  --profile small-ci \
  --backend numpy \
  --workers 1 \
  --max-elapsed-s 60 \
  --max-peak-memory-mb 256 \
  --output /tmp/mmpp-dispersion-benchmark.json
python -m pytest tests/test_dispersion_benchmark.py -q
```

Definicja done: benchmark jest powtarzalny, raport JSON jest stabilny, a CI
zatrzyma oczywiste regresje pamięci/czasu.

### Faza 6: backend parity i workers policy

Cel: użytkownik nie powinien dostawać innej fizyki przez zmianę backendu FFT.

Status: w dużej części wykonane lokalnie. NumPy/SciPy parity przechodzi,
pyFFTW ma bramkę warunkową z jawnym skipem bez zależności, release `extras-smoke`
wymusza `--benchmark-backend pyfftw` dla `fft/full`, a
`MMPP_FFT_BACKEND`/`MMPP_FFT_WORKERS` są opisane w docs i testowane w świeżym
subprocess imporcie. Pełny dowód pyFFTW wymaga zielonego joba `extras-smoke` w
środowisku z extra `fft`.

Zakres końcowy:

- NumPy i SciPy muszą przechodzić zawsze.
- pyFFTW musi przechodzić, jeśli extra `fft` instaluje `pyfftw`; jeśli nie jest
  dostępny, test ma być jawnie skipped z powodem. Warunkowy test już istnieje,
  a release `extras-smoke` wymusza pyFFTW benchmark dla `fft/full`.
- `MMPP_FFT_BACKEND` i `MMPP_FFT_WORKERS` muszą być opisane w docs oraz działać
  przy świeżym imporcie. Są opisane w API docs/tutorialu i chronione testem
  `test_dispersion_fft_backend_respects_environment_configuration`.
- `workers=-1` nie może być domyślnym ukrytym zachowaniem w CI bez możliwości
  ograniczenia. Docs i benchmark/release smoke pokazują `workers=1`.

Bramka:

```bash
MMPP_FFT_BACKEND=numpy python -m pytest tests/ -q -k "dispersion and fft"
MMPP_FFT_BACKEND=scipy python -m pytest tests/ -q -k "dispersion and fft"
MMPP_FFT_BACKEND=pyfftw python -m pytest tests/ -q -k "dispersion and fft"
```

Definicja done: osie, peak location i `S_raw` mieszczą się w ustalonych
tolerancjach między backendami.

### Faza 7: dokumentacja i migracja API

Cel: publiczne przykłady mają prowadzić do nowego, stabilnego API.

Status: wykonane dla głównej ścieżki `fft/dispersion`, z warunkiem potwierdzenia
w CI/release. README, API docs i tutorial pokazują nowy wzorzec
`disp.plot.interactive(...)`, legacy workflow jest oznaczony, release gate
wykonuje docs-style smoke na syntetycznym `.zarr`, a release verify job buduje
Sphinx docs przed publikacją artefaktów. `DispersionResult1D`, `.plot`,
`.analyze`, `.modes` oraz `DispersionInteractiveViewer` mają testowany
notebookowy `_repr_html_`. Dodano uruchamialny notebook smoke
`output/jupyter-notebook/fft-dispersion-interactive-smoke.ipynb`, który tworzy
syntetyczny `.zarr` i wykonuje publiczną ścieżkę
`m.dispersion.plot.interactive(..., show=False)`, gdzie `m = job.m.fft`.
Pierwsza komórka notebooka wykonuje preflight binarnego stosu
`numpy/pandas/zarr/numcodecs/h5py` i zatrzymuje wykonanie z instrukcją naprawy
środowiska, jeśli kernel zgłasza `numpy.dtype size changed`; taki błąd oznacza
niespójne pakiety w kernelu, a nie regresję `dispersion.interactive`.
W przypadku kernela `numba_sprawna` root cause był konkretny: Python bez
`PYTHONNOUSERSITE=1` ładował `numpy`, `zarr`, `numcodecs` i `h5py` z
`~/.local/lib/python3.10/site-packages`, a `pandas` z conda env. Notebook
usuwa teraz user-site z `sys.path` zanim zaimportuje pakiety binarne; jeśli
któryś z nich został już załadowany z user-site, wymaga restartu kernela.
Lokalnie pełny Sphinx build nie został wykonany, bo sprawdzone interpretery
conda base i aktualny `python3` nie mają pakietu `sphinx`, a `.venv/bin/python`
nie jest wykonywalny w tym checkoutcie; pozostaje szerszy example smoke poza
samą dyspersją. Usunięto jednak lukę metadata i workflow: `dev` extra w
`pyproject.toml` i `setup.py` deklaruje już `linkify-it-py`, wymagane przez
MyST `linkify` w `docs/conf.py`, a workflows docs/release instalują cytowane
`.[dev]` i nie instalują `linkify-it-py` ręcznie.

Zakres końcowy:

- Zaktualizować README, `docs/api/fft/dispersion.md` i
  `docs/tutorials/dispersion_analysis.md`. Wzorzec dyspersji jest już
  zsynchronizowany.
- Dodać sekcję "legacy" dla `dispersion_modes(...).plot_interactive()`. Sekcja
  legacy jest już w tutorialu i API docs.
- Ustalić docstringi i `_repr_html_` dla `DispersionResult1D`, `.plot`,
  `.analyze`, `.modes` oraz `DispersionInteractiveViewer`. Ten punkt jest
  domknięty testem `test_dispersion_notebook_repr_documents_public_accessors`.
- Dodać example smoke: fragmenty z README i docs mają importować się i przejść na
  syntetycznym `.zarr` albo zostać oznaczone jako notebook-only. Publiczny wzorzec
  `disp.plot.interactive(...) -> compute_1d -> res.plot.interactive(...)` oraz
  docsowy workflow modów z `store_complex=True` są już wykonywane w release gate;
  ich invarianty są twardą bramką przez `docs_example_status`. Dodatkowy
  notebook smoke sprawdza skrót użytkownika `m.dispersion.plot.interactive()`
  przez `m = job.m.fft`.
- Usunąć niespójności terminologiczne: `save_complex` vs `store_complex`,
  display vs raw, Hz vs GHz, `rad/m` vs `rad/um`. Główne docs dyspersji są już
  chronione testem, który wymaga `store_complex`, `S_raw`, `S_display`,
  `S_local_raw`, `S_local_display`, `result.S` i
  `analysis_source="display"` oraz blokuje `save_complex`.

Bramka:

```bash
python -m pytest tests/ -q -k "docs or repr or dispersion"
cd docs && sphinx-build -b html . _build --keep-going
```

Definicja done: dokumentacja nie pokazuje przestarzałej ścieżki jako głównej,
release verify job buduje Sphinx docs, a publiczne przykłady mają smoke na
syntetycznym `.zarr`.

### Faza 8: release gates dla całej biblioteki

Cel: release ma być potwierdzony na artefakcie dystrybucyjnym, nie tylko na
editable install.

Status: częściowo wykonane, z mocniejszym dowodem lokalnym. Workflow release
instaluje wheel i wheel zbudowany ze sdist razem z runtime dependencies oraz
odpala `verify_fft_dispersion_release_gate.py --import-mode installed`; test
workflow blokuje instalację artefaktów przez `--no-deps`, także przy budowie
wheel ze sdist, bo wtedy smoke mógłby paść lub przejść z powodów niezwiązanych
z opublikowanym pakietem. Lokalnie
zbudowano wheel/sdist do `/tmp`, `twine check` przeszedł dla obu artefaktów, a
zainstalowany wheel oraz wheel zbudowany ze sdist przeszły
`--import-mode installed --require-widget-smoke` w tymczasowych venvach
dziedziczących zależności conda base. To potwierdza artefakty kodu i import
spoza checkoutu, ale nie zastępuje clean dependency-resolution smoke w CI,
ponieważ wcześniejsza lokalna instalacja używała `--no-deps`. Release verify
buduje Sphinx docs, release gate potrafi wymusić widget smoke przez
`--require-widget-smoke`, a `extras-smoke` zależy od joba `build`, pobiera
artefakt `dist`, instaluje wheel z extra `fft`, `interactive`, `plotting` albo
`full` i uruchamia smoke z `--import-mode installed`. Zarówno publikacja PyPI,
jak i TestPyPI czekają na `build` oraz `extras-smoke`, więc ręczna publikacja
testowa nie omija bramki wheel+extras. Pełna bramka release nadal wymaga
zielonego przebiegu tej macierzy oraz macierzy Python 3.9-3.12.

Zakres końcowy:

- Po `python -m build --sdist --wheel` utworzyć czyste środowisko i zainstalować
  wheel.
- Uruchomić import/API smoke z wheel:

```bash
python -c "import mmpp; import mmpp.fft.dispersion"
python -c "from mmpp.fft.dispersion._interactive_viewer import DispersionInteractiveViewer"
```

- Powtórzyć smoke dla sdist albo przynajmniej osobno sprawdzić, że sdist buduje
  wheel w czystym środowisku.
- Testować extras z artefaktu wheel. Release workflow ma job `extras-smoke`,
  który pobiera `dist` i instaluje zbudowany wheel z extra:

```bash
python - <<'PY'
from pathlib import Path
import subprocess
import sys

wheel = next(Path("dist").glob("mmpp-*.whl"))
for extra in ["fft", "interactive", "plotting", "full"]:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        f"{wheel}[{extra}]",
    ])
PY
```

Dla `interactive` i `full` job dodatkowo uruchamia
`verify_fft_dispersion_release_gate.py --import-mode installed
--require-widget-smoke`.

- CI ma obejmować Python 3.9, 3.10, 3.11 i 3.12 zgodnie z `pyproject.toml`.
  Klasyfikatory `pyproject.toml` i `setup.py` są teraz spójne z macierzą CI.
- Jeśli projekt ma pozostać Alpha, dokumentacja nie może obiecywać pełnej
  produkcyjności. Jeśli ma przestać być Alpha, klasyfikator musi zostać zmieniony
  dopiero po zielonych release gates.

Bramka release:

```bash
python -m ruff format --check mmpp/ tests/ scripts/
python -m ruff check mmpp/ tests/ scripts/
python -m mypy mmpp/
python -m pytest tests/ -q --cov=mmpp --cov-report=xml
python -m build --sdist --wheel
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
python scripts/analysis/verify_fft_dispersion_release_gate.py \
  --import-mode installed \
  --max-elapsed-s 60 \
  --max-peak-memory-mb 256
```

Definicja done: release workflow testuje dokładnie to, co zostanie opublikowane.

## Macierz akceptacji

| Gate | Wymóg | Dowód |
| --- | --- | --- |
| Import | `mmpp.fft.dispersion` importuje się bez UI side-effectów. | clean-env smoke + import hygiene tests |
| API | `disp.plot.interactive()`, `res.plot.interactive(show=False)`, `mode.plot.interactive(show=False)` i `res.modes.plot.animation(show=False)` zwracają viewer. | test kontraktu |
| Headless | `show=False` nie importuje IPython display. | test bez UI deps |
| Repr | `DispersionResult1D`, `.plot`, `.analyze`, `.modes` i viewer pokazują nowe API. | `test_dispersion_notebook_repr_documents_public_accessors` |
| Widget | `show=True` renderuje i `.close()` sprząta zasoby. | `--require-widget-smoke` w środowisku `interactive` |
| Legacy | `dispersion_modes().plot_interactive()` działa albo ma deprecację. | compatibility test |
| Osie | `dt/dx/dy` uwzględniają stride. | synthetic slicing tests |
| Cache | Konfiguracje numeryczne nie mieszają wyników, a cache round-trip zachowuje lokalny raw/display split. | cache-key tests + local cache round-trip |
| Raw/display | Analiza domyślnie używa raw, display tylko jawnie; dotyczy także lokalnych przekrojów `S_local_raw`/`S_local_display`. | `analysis_source` tests + local slice/cache/live-filter tests |
| Scaling | PSD/window gain mieści się w tolerancji. | scaling tests |
| Mody | E2E `.zarr -> S_complex -> mode` działa. | physics tests |
| Branch | Coverage/SNR/confidence są realne. | noisy branch tests |
| Backend | NumPy/SciPy/pyFFTW mają zgodne wyniki. | backend parity |
| Performance | Benchmark ma progi czasu i RAM. | JSON benchmark gate |
| Docs | README/docs używają live API i release verify buduje Sphinx docs. | docs/example smoke + release docs build |
| Release | Wheel/sdist instalują się i przechodzą smoke. | release workflow |

## Kolejność prac

Najkrótsza ścieżka do statusu production candidate dla samego `fft/dispersion`:

1. Zamknąć Faza 1 headless/widget contract.
2. Utrzymać Faza 2 cache/slicing matrix jako release gate przy kolejnych zmianach.
3. Domknąć Faza 3 scaling tolerances i raw/display docs.
4. Utrzymać Faza 4 physics matrix jako release gate przy kolejnych zmianach.
5. Utrzymać Faza 5 `small-ci` benchmark threshold w release gate oraz
   weekly/manual benchmark workflow dla profilu `medium-dev`.
6. Dopiero wtedy używać etykiety `production candidate: fft/dispersion`.

Najkrótsza ścieżka do release-ready MMPP:

1. Wykonać powyższe dla `fft/dispersion`.
2. Zaktualizować docs i example smoke.
3. Dodać wheel/sdist clean install gates.
4. Uruchomić CI na Python 3.9-3.12 oraz potwierdzić zielony job `extras-smoke`.
5. Dopiero wtedy rozważyć zmianę klasyfikatora z Alpha.

## Odpowiedź na pytanie: czy po tych naprawach biblioteka będzie w pełni produkcyjna?

Nie. Po aktualnych naprawach `fft/dispersion` może stać się mocnym production
candidate, ale pełna produkcyjność wymaga jeszcze bramek release i dokumentacji.
Największa różnica jest taka:

- naprawy modułu dowodzą, że kod działa w checkoutcie developerskim;
- bramki release dowodzą, że użytkownik może zainstalować opublikowany pakiet i
  dostać ten sam, stabilny kontrakt API.

Dopóki wheel/sdist, extras-smoke, widget smoke i macierz Pythonów nie są zielone,
uczciwy status to "production candidate for `fft/dispersion`", nie "fully
production-ready MMPP".

## Lokalna weryfikacja odnotowana dla tej iteracji

Dotychczasowe pozytywne sygnały z aktualnego drzewa roboczego:

```bash
~/.local/bin/pytest tests/test_dispersion_release_gate.py \
  tests/test_dispersion_benchmark.py \
  tests/test_import_hygiene.py \
  tests/test_dispersion_mode_extraction.py -q
```

Wynik odnotowany w pracy nad modułem: komenda zakończyła się kodem 0; jeden
test pyFFTW został pominięty z powodu braku opcjonalnej zależności `pyfftw`.

```bash
/home/kkingstoun/software/anaconda3/bin/python3 \
  scripts/analysis/verify_fft_dispersion_release_gate.py \
  --require-widget-smoke \
  --output /tmp/mmpp-fft-dispersion-release-gate-required-widget.json
```

Wynik odnotowany w pracy nad modułem: release gate zwrócił `status=ok`,
`viewer_status=ok`, `mode_viewers_status=ok`, `docs_example_status=ok`,
`headless_imports.new_widget_modules=[]` oraz `widget_smoke.status=ok`. Ten
przebieg potwierdza lokalnie zarówno brak eager importów UI w `show=False`, jak
i działanie legacy widget render/close w środowisku z `IPython`,
`ipywidgets`, Matplotlib i `zarr`.

```bash
/home/kkingstoun/software/anaconda3/bin/python3 -m sphinx --version
python3 -m sphinx --version
.venv/bin/python -m sphinx --version
```

Wynik odnotowany w pracy nad modułem: conda base i aktualny `python3` zwróciły
`No module named sphinx`, a `.venv/bin/python` zwrócił `permission denied`.
Pełny HTML build pozostaje więc do potwierdzenia w środowisku z `.[dev]`.
Release workflow ma teraz krok `Build documentation`, który uruchamia
`sphinx-build -b html . _build --keep-going` przed publikacją.

Release workflow ma też job `extras-smoke`, który zależy od zbudowanego joba
`build`, pobiera artefakt `dist` i instaluje wheel z extra `fft`,
`interactive`, `plotting` lub `full`; dla `fft` i `full` uruchamia
`verify_fft_dispersion_release_gate.py --import-mode installed
--benchmark-backend pyfftw`, a dla `interactive` i `full` uruchamia
`verify_fft_dispersion_release_gate.py --import-mode installed
--require-widget-smoke`. Lokalnie potwierdzono wymagany widget smoke w
istniejącym środowisku conda base, ale nie wykonano jeszcze rzeczywistego
clean install wszystkich extras z artefaktu w CI; obecny dowód release to test
struktury workflow, standardowy smoke checkoutu i lokalny wymagany widget smoke.

Release verify uruchamia też benchmark preflight dla większych profili:
`medium-dev` z limitem 1024 MB oraz `research-reference` z limitem 4096 MB.
Tryb `--preflight-only` nie tworzy `.zarr` i nie uruchamia FFT, tylko generuje
ten sam stabilny raport JSON z `memory_preflight_mb` i progami pamięci.
Lokalny preflight cięższego wariantu `research-reference --store-complex
--no-orthogonal-average` zwrócił `threshold_status=ok` oraz
`estimated_peak_mb=832.0`.

Metadata release zostały wyrównane z CI: `pyproject.toml`, `setup.py` i
`.github/workflows/ci.yml` wskazują Python 3.9, 3.10, 3.11 i 3.12, a test
`declared_python_versions_match_ci_matrix` chroni tę spójność.

```bash
/home/kkingstoun/software/anaconda3/bin/python3 - <<'PY'
import json
from pathlib import Path

path = Path("output/jupyter-notebook/fft-dispersion-interactive-smoke.ipynb")
nb = json.loads(path.read_text())
ns = {"__name__": "__notebook_smoke__"}
for index, cell in enumerate(nb["cells"], start=1):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{path}:cell{index}", "exec"), ns)
print("NOTEBOOK_SMOKE_OK")
PY
```

Wynik odnotowany w pracy nad modułem: notebook smoke zakończył się
`NOTEBOOK_SMOKE_OK`. Notatnik tworzy syntetyczny `.zarr`, otwiera go przez
`mmpp.open(...)`, ustawia `m = job.m.fft` i wykonuje
`m.dispersion.plot.interactive(..., show=False)`, `compute_1d(store_complex=True)`,
`res.plot.interactive(show=False)` oraz `res.modes.interactive(show=False)`.
Preflight wypisuje też `sys.executable`, wersje i ścieżki pakietów binarnych,
co ułatwia wykrycie mieszania pakietów conda i user-site w notebookach.
Ten sam notebook został dodatkowo wykonany interpreterem
`/home/kkingstoun/software/anaconda3/envs/numba_sprawna/bin/python` bez
zewnętrznego `PYTHONNOUSERSITE`; bootstrap usunął user-site i przebieg zakończył
się `NOTEBOOK_SMOKE_OK`.

```bash
/home/kkingstoun/software/anaconda3/bin/python3 -m build \
  --no-isolation --sdist --wheel --outdir /tmp/mmpp-artifact-smoke/dist
/home/kkingstoun/software/anaconda3/bin/python3 -m twine check \
  /tmp/mmpp-artifact-smoke/dist/*
```

Wynik odnotowany w pracy nad modułem: build zakończył się powodzeniem i utworzył
`mmpp-0.5.3-py3-none-any.whl` oraz `mmpp-0.5.3.tar.gz` w `/tmp`; `twine check`
zwrócił `PASSED` dla obu artefaktów. Build emitował ostrzeżenia deprecacyjne
setuptools dotyczące pola license, ale nie przerwał budowy.

```bash
/tmp/mmpp-artifact-smoke/wheel-venv/bin/python \
  scripts/analysis/verify_fft_dispersion_release_gate.py \
  --import-mode installed \
  --require-widget-smoke \
  --max-elapsed-s 60 \
  --max-peak-memory-mb 256 \
  --output /tmp/mmpp-artifact-smoke/wheel-installed-release-gate.json
```

Wynik odnotowany w pracy nad modułem: wheel zainstalowany w venv spoza checkoutu
importował `mmpp` z `site-packages`, a release gate zwrócił `status=ok`,
`widget_smoke.status=ok`, `headless_imports.new_widget_modules=[]` i
`benchmark.threshold_status=ok`.

```bash
/home/kkingstoun/software/anaconda3/bin/python3 -m pip wheel \
  --no-build-isolation --no-deps \
  --wheel-dir /tmp/mmpp-artifact-smoke/sdist-wheel \
  /tmp/mmpp-artifact-smoke/dist/mmpp-0.5.3.tar.gz
/tmp/mmpp-artifact-smoke/sdist-venv/bin/python \
  scripts/analysis/verify_fft_dispersion_release_gate.py \
  --import-mode installed \
  --require-widget-smoke \
  --max-elapsed-s 60 \
  --max-peak-memory-mb 256 \
  --output /tmp/mmpp-artifact-smoke/sdist-installed-release-gate.json
```

Wynik odnotowany w pracy nad modułem: wheel zbudowany ze sdist i zainstalowany w
osobnym venv także zwrócił `status=ok`, `widget_smoke.status=ok`,
`headless_imports.new_widget_modules=[]` i `benchmark.threshold_status=ok`.
Pierwsza próba `pip wheel --no-deps` bez `--no-build-isolation` była zablokowana
przez brak sieci przy pobieraniu build dependency `setuptools>=61`; dlatego
lokalny smoke użył istniejących build tools i nie dowodzi pełnej rozwiązywalności
zależności z indeksu. Ten pełny dowód pozostaje obowiązkiem CI/release.

```bash
.venv/bin/ruff check \
  mmpp/fft/dispersion/_interactive_viewer.py \
  scripts/analysis/verify_fft_dispersion_release_gate.py \
  tests/test_import_hygiene.py \
  tests/test_dispersion_release_gate.py \
  tests/test_dispersion_mode_extraction.py \
  tests/test_dispersion_benchmark.py \
  scripts/analysis/benchmark_fft_dispersion.py
git diff --check
```

Wynik odnotowany w pracy nad modułem: brak błędów.

```bash
~/.local/bin/pytest tests/test_dispersion_mode_extraction.py -q \
  -k "local_raw_display or raw_display_split or orthogonal_slice or kmax or cache or raw or filters"
```

Wynik odnotowany w pracy nad modułem: 13 testów przeszło. Ten zestaw obejmuje
nowy kontrakt `S_local_raw`/`S_local_display`, selekcję przekroju ortogonalnego,
`kmax` trim oraz regresje raw/display/cache.

```bash
~/.local/bin/pytest tests/test_dispersion_mode_extraction.py -q \
  -k "cache_roundtrip or local_raw_display or kmax"
```

Wynik odnotowany w pracy nad modułem: 3 testy przeszły. Ten zestaw potwierdza,
że `S_local_raw` i `S_local_display` przechodzą przez `kmax` trim oraz
zapis/odczyt cache.

```bash
~/.local/bin/pytest tests/test_dispersion_mode_extraction.py -q \
  -k "apply_live_filters_preserves_local_raw or cache_roundtrip or local_raw_display or kmax"
```

Wynik odnotowany w pracy nad modułem: 4 testy przeszły. Ten zestaw rozszerza
bramkę lokalnego raw/display split o `apply_live_filters(..., apply_to_local=True)`.

```bash
~/.local/bin/pytest tests/test_dispersion_mode_extraction.py -q \
  -k "single_mode_interactive or single_mode_plot_interactive or modes_at_uses_full_frequency or notebook_repr"
```

Wynik odnotowany w pracy nad modułem: 4 testy przeszły. Ten zestaw potwierdza,
że `mode.plot.interactive(show=False)` zwraca lekki kontroler, ma HTML repr,
obsługuje eksport stanu i sprząta mockowany display handle.

```bash
~/.local/bin/pytest tests/test_dispersion_mode_extraction.py -q \
  -k "modes_plot_animation or single_mode_interactive or single_mode_plot_interactive or notebook_repr"
```

Wynik odnotowany w pracy nad modułem: 5 testów przeszło. Ten zestaw potwierdza,
że `res.modes.plot.animation(show=False)` zwraca lekki kontroler żądania
animacji, ma HTML repr, eksport stanu i display lifecycle. Eksport single-mode
oraz animation viewerów przechodzi `json.dumps(...)` dla skalarów i małych tablic
NumPy w opcjach/selekcji.

```bash
/usr/bin/python3 scripts/analysis/verify_fft_dispersion_release_gate.py \
  --max-elapsed-s 60 \
  --max-peak-memory-mb 256 \
  --output /tmp/mmpp-fft-dispersion-release-gate-local-raw-display.json
```

Wynik odnotowany w pracy nad modułem: release gate zwrócił `status=ok`,
benchmark `threshold_status=ok`. To był dodatkowy headless smoke w interpreterze
systemowym bez stosu widgetowego; główny dowód widgetowy dla tej iteracji jest
zielony w opisanym wyżej przebiegu conda base z `--require-widget-smoke`.

```bash
/usr/bin/python3 scripts/analysis/benchmark_fft_dispersion.py \
  --shape 8,1,2,8,3 \
  --backend numpy \
  --workers 1 \
  --store-complex \
  --no-orthogonal-average \
  --output /tmp/mmpp-dispersion-local-spectra-benchmark.json
```

Wynik odnotowany w pracy nad modułem: benchmark zwrócił `threshold_status=ok`,
`s_local_mb=0.000488`, `s_local_raw_mb=0.000488` i
`s_local_display_mb=0.000488`, co potwierdza raportowanie kosztu lokalnych widm
raw/display.

```bash
~/.local/bin/pytest tests/test_dispersion_mode_extraction.py -q \
  -k "cache_context_hash_includes_array_filter_values or cache_roundtrip_preserves_local_raw_display_split or apply_live_filters_preserves_local_raw"
```

Wynik odnotowany w pracy nad modułem: 3 testy przeszły. Nowy test
`cache_context_hash_includes_array_filter_values` reprodukował kolizję cache
dla dwóch różnych tablic w konfiguracji filtra, a po poprawce potwierdza, że
hash kontekstu zawiera `value_sha1` tablicy. Ten dowód wzmacnia P0 kontrolę
przeciw mieszaniu wyników po zmianie filtrów.

```bash
~/.local/bin/pytest tests/test_dispersion_release_gate.py -q \
  -k "release_workflow_installs_built_artifacts_before_publish or docs_workflows_use_dev_extra_for_linkify_dependency"
```

Wynik odnotowany w pracy nad modułem: 2 testy przeszły. Test
`docs_workflows_use_dev_extra_for_linkify_dependency` potwierdza, że
`.github/workflows/docs.yml` i `.github/workflows/release.yml` polegają na
cytowanym `pip install -e ".[dev]"`, a nie na ręcznej instalacji
`linkify-it-py`. Test
`release_workflow_installs_built_artifacts_before_publish` potwierdza, że
workflow buduje wheel ze sdist bez `--no-deps`, instaluje wynik bez `--no-deps`,
a `extras-smoke` pobiera artefakt `dist`, instaluje wheel z matrix extra i
uruchamia smoke przez `--import-mode installed` zamiast importować checkout.
Ten sam test wymaga, żeby publikacja PyPI i TestPyPI miały
`needs: [build, extras-smoke]`.
Po refaktoryzacji workflow cały plik `tests/test_dispersion_release_gate.py`
przeszedł lokalnie: 18 testów zakończonych kodem 0. Dodatkowo
`/home/kkingstoun/software/anaconda3/bin/python3 -c "import yaml; ..."`
potwierdził poprawne parsowanie `.github/workflows/release.yml` i
`.github/workflows/docs.yml`.

```bash
/home/kkingstoun/software/anaconda3/bin/python3 -m build --no-isolation \
  --sdist --wheel --outdir /tmp/mmpp-docs-metadata-smoke/dist
python3 - <<'PY'
import zipfile
from pathlib import Path
wheel = next(Path("/tmp/mmpp-docs-metadata-smoke/dist").glob("*.whl"))
with zipfile.ZipFile(wheel) as zf:
    metadata = zf.read("mmpp-0.5.3.dist-info/METADATA").decode()
for line in metadata.splitlines():
    if "linkify-it-py" in line or line == "Provides-Extra: dev":
        print(line)
PY
```

Wynik odnotowany w pracy nad modułem: build wheel/sdist zakończył się kodem 0,
a wheel METADATA zawiera `Provides-Extra: dev` oraz
`Requires-Dist: linkify-it-py; extra == "dev"`. Build nadal emituje istniejące
ostrzeżenia setuptools o formacie licencji i read-only `conda-meta/history`.

Ograniczenie: pełne `pytest tests/ -q` było wcześniej blokowane przez zależności
opcjonalne w środowisku, więc nie może być użyte jako dowód pełnej produkcyjności
bez czystego środowiska release.

## Konkluzja

Masterplan jest teraz ostrzejszy niż pierwotna lista napraw. Aktualne zmiany
idą w dobrym kierunku i usuwają wiele krytycznych problemów, ale końcowa
deklaracja produkcyjności musi być oparta na bramkach, nie na ocenie jakości kodu.

Docelowy komunikat po domknięciu faz 0-7:

> `mmpp.fft.dispersion` is production candidate with stable notebook-facing API.

Docelowy komunikat po domknięciu fazy 8:

> MMPP release is production-ready for the declared Python versions and extras.

Do tego czasu raport powinien blokować każdą deklarację "w pełni produkcyjne",
jeżeli nie ma dowodu z wheel/sdist, docs, extras, widget smoke i CI matrix.
