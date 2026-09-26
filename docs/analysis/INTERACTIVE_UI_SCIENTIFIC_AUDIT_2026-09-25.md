# Audyt interaktywnych narzędzi MMPP: FFT, mody, wiry i skyrmiony

Data: 2026-09-25. Audytowany commit: `22106e8` — `fix: resample nonuniform time axes across FFT paths`. Wersja źródeł: 0.6.10. Raport dotyczy tego lokalnego stanu kodu, nie stanowi potwierdzenia publikacji tej wersji ani aktualizacji kernela użytkownika.

## 1. Ocena i granice audytu

MMPP ma rozbudowany zestaw narzędzi eksploracyjnych, ale **nie można obecnie uznać wszystkich interaktywnych wyników za ilościowo poprawne i spójne**. Najważniejsze problemy dotyczą jednostek przestrzennych, znaku czasu w animacji, normalizacji PSD, wspólnych pasków kolorów, propagacji warstwy oraz aktualności cache. Mają pierwszeństwo przed przebudową wyglądu.

Potwierdzono reprodukcjami m.in.:

- pole o rozmiarze 100 nm opisane w widoku modów jako 100 μm;
- wartość −1 zamiast +1 dla podglądu sinusa po ćwierci okresu przy dodatnim czasie;
- periodogram deklarowany jako PSD, którego pik jest 512 razy większy od referencji SciPy przy `fs=1024 Hz`;
- jeden colorbar 0–1 przy dwóch panelach o skalach 0–1 i 0–10;
- poprzednią klasyfikację skyrmionu po zmianie progu w konfiguracji;
- zmianę mierzonego promienia z 14,009 nm do 6,018 nm po przestrzennym kroku 2, przy pozostawieniu starego rozmiaru komórki;
- cztery materializacje całego wybranego szeregu podczas utworzenia dashboardu skyrmionu i analizy jednej klatki.

To audyt i raport, **bez implementowania napraw kodu biblioteki**. Dodano skrypt reprodukcyjny, ten dokument i dokładny wyjątek w `.gitignore`, aby raport nie znikał pod globalną regułą `*.md`. Nie wykonano publikacji, taga ani instalacji w środowisku użytkownika.

Oznaczenia dowodów:

| Oznaczenie | Znaczenie |
|---|---|
| R | Reprodukcja w bieżącym środowisku Python, z podanym wynikiem |
| K | Potwierdzone przez prześledzenie kodu i jego wywołań |
| H | Ryzyko lub hipoteza wymagająca dodatkowego pomiaru |
| B | Walidacja w prawdziwej przeglądarce Jupyter pozostaje otwarta |

P1 oznacza błąd mogący zmienić interpretację naukową, wynik lub istotnie utrudnić pracę; P2 — istotną niespójność, ergonomię lub koszt; P3 — porządkowanie i ulepszenie. Nie używam oceny liczbowej „gotowości”, ponieważ nie wykonano pełnego E2E Jupyter.

Przeprowadzono analizę źródeł, testy regresyjne, wykonanie rzeczywistych callbacków z kontrolowanymi danymi, porównanie ze SciPy oraz pomiary na syntetycznych teksturach. Rendering testowano przez backend Agg. Zainstalowany `ipympl` nie jest dowodem działania frontendu. Nie były dostępne Chromium/Chrome ani moduły Playwright/Selenium; nie sprawdzono faktycznego reflow, klawiatury, dotyku i komunikacji widgetów w przeglądarce. W tym audycie nie ponawiano obliczeń na pełnym zewnętrznym `0.5.zarr`; wcześniejszy test 20 klatek nie jest dowodem poprawności całego UI.

## 2. Mapa interfejsów i architektury

| Obszar | Główna implementacja | Model interakcji | Stan audytu |
|---|---|---|---|
| FFT i FMR | `mmpp/fft/modes/interactive.py`, `_interactive/` | ipywidgets + Matplotlib, spektrum i panele modów | Kod, callbacki, rendering, testy |
| Starszy FMR | `mmpp/fft/modes/visualization/interactive.py` | Matplotlib i własna animacja | Przejrzana ścieżka wyboru; brak pełnej macierzy legacy |
| Dyspersja | `mmpp/fft/dispersion/_interactive/`, `_interactive_viewer.py` | Oddzielny stan, odroczony render, filtry, ekstrakcja | Kod i testy |
| Mody dyspersji | `mmpp/fft/dispersion/modes/interactive.py`, `_interactive/` | Wybór k/f, rekonstrukcja, animacje, eksport | Kod i testy ekstrakcji |
| Wir — dashboard | `mmpp/solitons/vortex/ui/interactive_dashboard.py` | Panel modułów, przyciski Run, PNG | Kod; testy wybranych kontraktów |
| Wir — trajektoria | `mmpp/solitons/vortex/_shared/plot/interactive.py` | Suwak/Play Matplotlib, snapshot cache | Kod i test kontrolki fallback |
| Thiele | `mmpp/solitons/vortex/nonlinear/interactive.py` | Formularz modelu, Run, ODE/SDE | Kod; zakres fizyki opisany poniżej |
| Skyrmion | `mmpp/solitons/skyrmion/ui/interactive_dashboard.py` | Analiza klatki, PNG; przekierowanie do FFT | Kod, rendering, cache, geometria, koszt |
| Histereza | `mmpp/analyze/hysteresis/plot/interactive.py`, `_interactive/` | Pętla, snapshot, ROI, animacja | Przegląd pomocniczy; nie pełny audyt metryk histerezy |
| Wspólne UI | `mmpp/ui/interactive/`, `mmpp/ui/snapshot.py` | Proste helpery, stan, LRU | Kod |

Istnieją wartościowe elementy do wykorzystania: odroczone rysowanie i `diagnostics()/close()` w dyspersji, `continuous_update=False` na wielu suwakach, jawne rozdzielenie składowych spektrum i modów, cache klatek, ostrzeżenia o resamplingu, maski materiału, informacja o rzeczywiście załadowanym binie częstotliwości oraz testy zachowania kontekstu datasetu.

Jednocześnie dashboard wirów ma 2776 linii, Thiele 1054, a samo `fft/modes/_interactive/callbacks.py` 832. To miary rozmiaru, nie automatyczny dowód złej jakości. W tym przypadku łączą się jednak z mieszaniem obliczeń, stylowania, zapisu presetów, stanu i obsługi błędów w jednej klasie. `mmpp/ui/interactive` nie jest jeszcze wspólnym silnikiem cyklu życia, kolejki obliczeń i walidacji. Nie należy natomiast traktować każdego starego importu jako duplikacji: część ścieżek solitonów jest fasadą zgodności.

## 3. Rejestr problemów naukowych i funkcjonalnych

### A01 — P1: pomieszanie nm i μm w panelach FMR [R, K]

Źródła: [FMRModeData i get_mode](../../mmpp/fft/modes/__init__.py), linie 527–555 i 1175–1210; [InteractiveSpectrum._load_mode](../../mmpp/fft/modes/interactive.py), linie 603–635; [etykiety i imshow](../../mmpp/fft/modes/_interactive/mode_plots.py), linie 96–135 i 224–237.

`get_mode()` buduje extent przez `_mode_extent_nm`; kontener dokumentuje nm. Interaktywny renderer używa tego extent bez przeliczenia, a podpisuje oś `x [μm]`. Sonda wykonała rzeczywisty `InteractiveSpectrum.show()` dla extent `(0,100,0,100)`: otrzymano właśnie ten zakres oraz podpis μm. Błąd opisu długości wynosi czynnik 1000. Dodatkowo fallback `data_loader` tworzy zakres z liczby pikseli, także bez jawnej jednostki.

Naprawa: jeden kontrakt geometrii w SI, konwersja do jednostki wyświetlania w jednej warstwie; brak metadanych powinien dawać piksele, nigdy fikcyjne μm. Kryterium: pole 100 nm ma kończyć się na 100 nm albo 0,1 μm we wszystkich widokach, snapshotach i eksportach; osobny test dla anizotropowego dx/dy i cropu.

### A02 — P1: animacja pokazuje odwrócony kierunek czasu [R, K]

Źródła: [obliczanie modów](../../mmpp/fft/modes/__init__.py), linia 1722; [callbacki fazy i animacji](../../mmpp/fft/modes/_interactive/callbacks.py), linie 200, 524, 713; [animacja modów dyspersji](../../mmpp/fft/dispersion/modes/_interactive/callbacks.py), początek `on_animate`.

Współczynniki powstają z `np.fft.rfft`, natomiast odtwarzanie stosuje `M * exp(-iφ)` i opisuje je dodatnim czasem. Przy konwencji NumPy rekonstrukcja używa dodatniego znaku wykładnika. Dla `sin(ωt)` rzeczywisty callback `on_phase_index_changed` zwrócił −1 przy `φ=90°`, podczas gdy sygnał wejściowy daje +1. Tytuł pokazywał `t=0.25ns` przy 1 GHz. To może odwrócić odczyt kierunku obrotu/precesji lub propagacji. [Definicja DFT i odwrotnej DFT w NumPy](https://numpy.org/doc/stable/reference/routines.fft.html).

Naprawa: jawny, wspólny kontrakt znaku transformacji, binu i czasu. Nie zmieniać mechanicznie każdej animacji przed prześledzeniem konwencji jej danych. Zweryfikować realny sinus, cosinus, parę składowych o przesunięciu π/2 oraz falę biegnącą. Oddzielić dowolny obrót fazy od rekonstrukcji z podpisem fizycznego czasu. Ta reprodukcja dotyczy bezpośrednio FMR; analogiczny wzorzec w dyspersji wymaga osobnego testu integracyjnego.

### A03 — P1: periodogram i fallback STFT nie mają normalizacji PSD [R, K]

Źródło: [mmpp/_shared/spectral.py](../../mmpp/_shared/spectral.py), `_windowed_periodogram`, `compute_psd`, `_numpy_stft_psd`, linie około 150–313. Konsumenci obejmują widma trajektorii, proxy sygnału MTJ i dashboard Thiele.

Gałąź periodogram dzieli `|FFT|²` przez `sum(window²)`, ale pomija `fs` wymagane dla density i podwojenie dodatnich binów jednostronnego widma rzeczywistego poza DC/Nyquist. Parametr `scaling` działa w Welch/SciPy, lecz nie jest honorowany w tej gałęzi. Także `detrend` zastępuje się bezwarunkowym odjęciem średniej.

Reprodukcja: sinus amplitudy 1, N=1024, fs=1024 Hz, bin 32, identyczne symetryczne okno Hann w MMPP i SciPy. Całka PSD: MMPP 256,00000006, SciPy 0,5000000001. Stosunek piku: 512. Nie jest to uniwersalny mnożnik błędu — zależy od fs i sidedness. Sama częstotliwość piku może nadal być poprawna; normalizacja wykresu do maksimum maskuje problem. [SciPy: scaling density/spectrum i sidedness](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).

Naprawa: ujednolicić density/spectrum/raw_power, jednostki, detrend i konwencję sygnałów zespolonych. Testy Parsevala, DC, Nyquist, N parzystego/nieparzystego oraz porównanie ze SciPy dla tego samego okna. Nie przenosić tego zarzutu automatycznie na każdą ścieżkę `mmpp.fft`: dowód dotyczy wskazanego wspólnego helpera i jego konsumentów.

### A04 — P1: wybrana warstwa modów nie jest przekazywana do obliczania spektrum [K]

Źródła: [FFTModeInterfaceNew._interactive_spectrum_impl](../../mmpp/fft/modes/interface.py), linie 1246–1255 i 1350–1370; [FFT._spectrum_impl](../../mmpp/fft/core.py), linie 384–408; [kontrolki](../../mmpp/fft/modes/_interactive/controls.py), `read_controls`.

`z_layer` trafia do `viewer.show`, ale nie do wywołania `parent_fft._spectrum_impl`, którego domyślna wartość to −1. Zmiana suwaka warstwy aktualizuje `_current_z_layer` i obraz modu, lecz nie przelicza źródłowego spektrum. Dla próbki wielowarstwowej krzywa i mapa mogą dotyczyć innych warstw. Dla jednowarstwowej ten problem może pozostawać niewidoczny.

Naprawa: jawny wybór „spektrum tej warstwy” / „stałe spektrum referencyjne”, poprawne przekazanie warstwy i widoczny opis obu zakresów. Test: dwie warstwy o rozłącznych częstotliwościach, zmiana warstwy przez API i kontrolkę, ponowne użycie gotowego SpectrumResult.

### A05 — P1: jeden colorbar nie odpowiada wszystkim składowym [R, K]

Źródła: [mode_plots](../../mmpp/fft/modes/_interactive/mode_plots.py), `_resolve_plot_data` i `update_mode_plots`; [mode_layout](../../mmpp/fft/modes/_interactive/mode_layout.py), `apply_mode_colorbars`.

Każde `imshow` dostaje własną normalizację; jedna skala w wierszu jest przypięta do pierwszego obrazu. Reprodukcja: zakres mx 0–1, my 0–10, jeden colorbar 0–1. Ten sam kolor odpowiada więc różnym amplitudom, a wspólna legenda sugeruje przeciwnie.

Naprawa: domyślnie wspólna instancja Normalize i zakres liczony nad wybranymi składowymi. Alternatywnie osobne colorbary i wyraźny tryb „normalizacja osobna”. Kryterium: porównanie mx/my/mz nie może zmieniać interpretacji amplitudy wskutek niejawnego autoskalowania. Faza powinna zachować stałą skalę −π…π i cykliczną paletę.

### A06 — P1: cache skyrmionu nie uwzględnia całej konfiguracji [R, K]

Źródła: [interface._cache_key](../../mmpp/solitons/skyrmion/interface.py), linie 134–146; [topology.detect](../../mmpp/solitons/skyrmion/topology.py), linie 53–84; [size.fit](../../mmpp/solitons/skyrmion/size.py).

Klucz zawiera m.in. frame, layer, method, convention i maskę, ale pomija progi mutable config i ustawienia radialnego dopasowania. Sonda zmieniła `min_abs_q` z 0,5 na 1,5. Zwykłe `detect()` oddało ten sam obiekt: `state='skyrmion', valid=True`. `detect(force=True)` dało `state='non_skyrmion', valid=False`.

Naprawa: niezmienny snapshot efektywnej konfiguracji w kluczu albo wersjonowanie i unieważnianie wyników przy każdej zmianie. Dołączyć geometrię i wersję danych. Cache rysowania nie może zastępować cache naukowego. Kryterium: wynik cached == forced dla każdej publicznej kombinacji parametrów.

### A07 — P1: skyrmion traci skalę przestrzenną po downsamplingu [R, K]

Źródło: [interface._resolve_data i _resolve_spacing](../../mmpp/solitons/skyrmion/interface.py), linie 103–132.

Tablica jest wycinana przez `slice_info`, ale spacing czytany nadal z atrybutów całego jobu. Dla kroku 2 w x/y pozostaje 1 nm zamiast 2 nm. Sonda: R pełnego pola 14,0094 nm; R po kroku 2: 6,0181 nm. Nie jest to czysty idealny czynnik 2, bo dochodzi dyskretyzacja profilu i centrum. Osobno: brak metadanych spacing daje `(1.0,1.0)` interpretowane w metrach, bez jawnego oznaczenia braku jednostek.

Naprawa: korzystać z geometrii wybranego widoku, uwzględniać krok, początek i orientację osi. Brak długości komórki powinien wymuszać jawne dx/dy albo tryb pikselowy. Testy crop/stride/nierównego dx/dy oraz poprawności centrum. Tej obserwacji nie należy rozszerzać na całą geometrię FFT — problem wykazano w skyrmionowym resolverze.

### A08 — P1: „Fit to simulation data” nie uruchamia dopasowania [K]

Źródło: [dashboard wirów](../../mmpp/solitons/vortex/ui/interactive_dashboard.py), `_build_tab_thiele` 1008–1063, `_run_thiele_quick` 2480–2582, `_run_thiele_full` 2584–2610.

Panel ma domyślnie zaznaczone „Fit to simulation data” i „Show PSD comparison”, a przycisk nazywa się „Quick Trajectory Fit”. Callback nie odczytuje tych opcji, nie uruchamia optymalizacji ani porównania PSD. Uruchamia symulację na parametrach formularza, z `polarity=1`, ustalonym czasem 10 ns, krokiem 5 ps i początkowym promieniem 0,1R. Pełny dashboard jest tworzony bez przekazania danych jobu. To rozbieżność obietnicy UI z działaniem, niezależnie od poprawności samego solvera.

Naprawa: zmienić nazwę na „Symuluj model” i usunąć/wyłączyć nieobsługiwane opcje albo naprawdę podpiąć istniejący moduł bridge/autofit. W trybie fit pokazywać funkcję celu, jednostki, parametry dopasowane, identyfikowalność i reszty. Dla CIP nie zakładać bez opisu tego samego przelicznika `I/(πR²)` co dla CPP: potrzebna jest geometria przekroju przepływu albo bezpośrednie J.

### A09 — P2: „breathing” oznacza promień orbity, nie promień rdzenia [K]

Źródło: [compute_breathing_spectrum](../../mmpp/solitons/vortex/spectrum/gyration.py), od linii 146; dashboard `_run_spectrum`.

Implementacja liczy widmo `trajectory.r`. Dokumentacja funkcji uczciwie mówi o promieniu orbity, lecz panel „Breathing spectrum” nie rozróżnia modulacji orbity od oddechowego modu tekstury magnetyzacji. To inne obserwable; sama trajektoria środka nie wystarcza do pomiaru zmiany rozmiaru rdzenia/skyrmionu.

Naprawa: podpis „Widmo promienia orbity r(t)” i osobny pomiar R_core(t) / R_sk(t) z tekstury. Do klasyfikacji fizycznej modu potrzebna mapa przestrzenna i definicja obserwabli, nie wyłącznie obecność piku.

### A10 — P2: granice parametrów i jednostki w fallbackach [R, K]

Źródła: [guess_layer_bounds](../../mmpp/fft/modes/_interactive/controls.py), linie 17–31; [filters._to_ghz](../../mmpp/fft/modes/_interactive/filters.py), linie 130–141; [compute_psd](../../mmpp/_shared/spectral.py), około linii 223–235.

- Dla modów istniejących tylko w RAM z jedną warstwą suwak dostaje fallback −10…10, zamiast zakresu danych. Potwierdzono helperem dla `_memory_modes`.
- Heurystyka `_to_ghz([0,1000])` zwraca `[0,1000]`, mimo że wejście w Hz powinno dać `[0,1e-6]` GHz. Dotyczy ścieżki bez `frequencies_ghz`; standardowy SpectrumResult z tą właściwością omija problem.
- `compute_psd([0,1], method='welch', dt=1)` podnosi `ValueError: noverlap must be less than nperseg`: segment jest podnoszony do 8, potem SciPy skraca go do 2, ale overlap pozostaje zbyt duży.

Naprawa: zakresy z rzeczywistego obiektu, jednostki z kontraktu/metadanych, minimalny N i walidacja segmentu przed tworzeniem kontrolek. Przy niewystarczających danych UI powinien podać przyczynę oraz sensowną alternatywę. Testować 0/1/2/3/7/8/20 próbek i jedno-binowe wyniki.

## 4. Wydajność, pamięć i responsywność obliczeń

### A11 — P1: analiza jednej klatki skyrmionu ładuje cały szereg [R, K]

Źródła: [dashboard](../../mmpp/solitons/skyrmion/ui/interactive_dashboard.py), linia 44 i `run`; [resolver](../../mmpp/solitons/skyrmion/interface.py), linie 103–114.

Konstruktor materializuje dane tylko po to, aby poznać liczbę klatek i warstw. Następnie robią to topologia, rozmiar i snapshot. Reprodukcja dla `(4,1,64,64,3)` wykazała cztery wywołania zwracające pełną tablicę float64 po 393216 bajtów. To pomiar zwracanych tablic, nie liczby fizycznych odczytów dysku ani równoczesnego peak RSS — niższe warstwy mogą mieć cache.

Objętość jednej tablicy wynosi `Nt*Nz*Ny*Nx*3*itemsize`. Przykładowe 2000×1×512×512×3 to około 5,86 GiB float32 lub 11,72 GiB float64, zanim doliczy się maski, kopie i render. Sam wybór frame=0 nie ogranicza obecnej materializacji.

Naprawa: `.shape` bez odczytu; selekcja klatki i warstwy przed konwersją NumPy; jeden snapshot przekazywany do topologii, rozmiaru i renderera; LRU ograniczony bajtami. Przy wieloklatkowej analizie iteracja po chunkach. Sprawdzić też [SnapshotCache._load_raw_frame](../../mmpp/ui/snapshot.py), linie 96–99: zastosowanie slice bezpośrednio do surowego Zarr przed wyborem klatki może materializować cały widok na każdym cache miss.

### A12 — P2: domyślny Berg–Lüscher ma pętle Python po komórkach [R, K]

Źródło: [skyrmion._core._q_density](../../mmpp/solitons/skyrmion/_core.py), linie 115–151. Pomiary jednokrotne po importach, w tym samym procesie; nie są benchmarkiem p95 ani czystym badaniem zbieżności siatki.

| Siatka | Berg–Lüscher | finite_diff | Q BL | Q FD |
|---|---:|---:|---:|---:|
| 64² | 0,245 s | 0,00195 s | 0,999936 | 0,988627 |
| 128² | 1,038 s | 0,00572 s | 0,999999994 | 0,989477 |
| 256² | 3,731 s | 0,01945 s | ≈1 | 0,989636 |

Nie należy przyspieszać UI przez cichą zamianę estymatora — tabela pokazuje też różnicę wyników. Wektoryzować iloczyny potrójne, iloczyny skalarne i rozdział ładunku trójkątów; opcjonalny kompilowany kernel dopiero z testem równoważności maski i znaku. Zmierzyć RAM wersji wektorowej, ponieważ wiele tablic tymczasowych może wymagać przetwarzania pasami.

### A13 — P2: drobne zmiany kontrolek przebudowują figurę [K; czas B]

Źródła: [InteractiveSpectrum._on_controls_changed](../../mmpp/fft/modes/interactive.py), linie 456–469; [rendering.create_figure](../../mmpp/fft/modes/_interactive/rendering.py), linie 105–124; [mode_plots.update_mode_plots](../../mmpp/fft/modes/_interactive/mode_plots.py).

Wspólny callback odczytuje wszystkie parametry, filtruje spektrum i tworzy nową figurę. Zmiany niezmieniające topologii wykresu nie wymagają nowych osi/colorbarów ani utraty zoomu. Przełączanie częstotliwości i podgląd fazy mają częściowo szybsze ścieżki — warto je zachować.

Naprawa: rozdzielić invalidację danych, filtrów, wyboru, stylu i layoutu. Używać `set_data/set_xdata/set_clim`, `draw_idle`, zachować xlim/ylim. Grupować zmianę presetów w jedną transakcję; debounce tylko dla kosztownych operacji. `continuous_update=False` już ogranicza liczbę wywołań, ale nie redukuje kosztu pojedynczego renderu. [Obsługa zdarzeń i debounce ipywidgets](https://ipywidgets.readthedocs.io/en/8.1.4/examples/Widget%20Events.html).

### A14 — P2: animacja dyspersji alokuje wszystkie klatki [K]

Źródło: [dispersion modes callbacks.on_animate](../../mmpp/fft/dispersion/modes/_interactive/callbacks.py), linie 60–133.

Powstaje tablica `(n_frames,Ny,Nx)` albo RGB. Nawet stałe `abs(M)` jest powielane przez `np.repeat`. Dla 120 klatek 1024² float64 jest to około 960 MiB, a RGB 2,81 GiB, bez kopii list i Matplotlib. To wyliczenie kosztu, nie pomiar alokacji na takiej siatce.

Naprawa: przechowywać zespolony mod i generować jedną klatkę na żądanie; amplitudę pokazywać jako stałą obwiednię. Eksport strumieniować; nie blokować UI kodowaniem całego filmu. Blitting włączać warunkowo po sprawdzeniu backendu, bo nie każdy canvas go wspiera. Kryterium: pamięć podglądu nie rośnie liniowo z liczbą klatek.

### A15 — P2: niespójny cykl życia i globalny stan notebooka [K; wycieki H]

Źródła: [FMR cleanup](../../mmpp/fft/modes/interactive.py), linie 665–672; [vortex.show](../../mmpp/solitons/vortex/ui/interactive_dashboard.py), linie 428–443; [Thiele import](../../mmpp/solitons/vortex/nonlinear/interactive.py), linie 39–43; [dyspersja close](../../mmpp/fft/dispersion/_interactive/widget.py), linie 86–114; [histereza close](../../mmpp/analyze/hysteresis/plot/interactive.py), linia 577.

Dyspersja ma zamykanie kontrolek i figury. FMR zamyka starą figurę, ale nie zapewnia wspólnego publicznego `close()` z wyrejestrowaniem widget observers i timerów. Vortex/Skyrmion/Thiele również nie mają spójnego kontraktu cleanup. Histereza zatrzymuje animację i odłącza klik, lecz nie zamyka wszystkich widgetów. `plt.ioff()` i importowe `warnings.filterwarnings` zmieniają stan całego kernela.

Nie zmierzono narastającego wycieku commów, więc nie stwierdzam jego rozmiaru. Naprawa: idempotentne `close`, przechowywanie observerów/connection IDs/timerów, cleanup po ponownym wykonaniu komórki, lokalny `rc_context` i `catch_warnings`. Test: 30 cykli otwarcia/zamknięcia bez narastania figur, timerów i aktywnych callbacks.

## 5. UI, UX, czcionki i Jupyter

### A16 — P1/P2: layout nie dostosowuje się do wąskiej komórki [K, B]

Źródła: [FMR widgets](../../mmpp/fft/modes/_interactive/widgets.py), linie 568–594; [vortex layout](../../mmpp/solitons/vortex/ui/interactive_dashboard.py), linie 224–245; [skyrmion layout](../../mmpp/solitons/skyrmion/ui/interactive_dashboard.py), linie 100–127.

FMR wymaga co najmniej 315+760=1075 px przed paddingami, vortex 310+680=990 px. HBox nie ma tu reguły przejścia do kolumny. Skyrmion łączy panel minimum 300 px z obrazem `width=100%` wewnątrz tego samego HBox. To kodowe przyczyny przepełnienia albo silnego pomniejszenia wykresu, nie pomiar browser screenshot.

Naprawa: układ zależny od szerokości kontenera komórki, nie tylko ekranu. Proponowane punkty startowe: powyżej 1100 px sidebar; 760–1100 px węższy/składany panel; poniżej 760 px kontrolki nad figurą. `min_width=0` dla obszaru obrazu, kontrolowane overflow tylko tam, gdzie potrzebne. Wykres może uzasadniać dwuwymiarowe przewijanie, formularz sterowania powinien się przeorganizować. [W3C: Reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html).

### A17 — P2: globalny CSS dashboardu wirów zmienia inne widgety [K, B]

Źródło: [vortex _CSS](../../mmpp/solitons/vortex/ui/interactive_dashboard.py), linie 145–221. Selektory `.widget-label`, `.widget-readout`, `.widget-dropdown > select`, `.widget-select > select` oraz przyciski używają `!important` bez prefiksu korzenia dashboardu. Skutek obejmuje także sąsiednie widgety w notebooku; wymuszone jasne tła nie respektują tematu Jupyter.

Naprawa: własna klasa root i wszystkie selektory pod nią; tokeny jasnego/ciemnego motywu, bez globalnego resetowania. E2E: jednocześnie FFT, vortex i zwykły ipywidgets.IntSlider; otwarcie jednego panelu nie może zmieniać pozostałych.

### A18 — P2: drobne podpisy i brakujące glify [R, K, B]

Źródła: [vortex CSS](../../mmpp/solitons/vortex/ui/interactive_dashboard.py), m.in. etykiety 9–11 px; [FMR colorbary](../../mmpp/fft/modes/_interactive/mode_layout.py), etykiety 8 pt i ticki 7 pt; [plotting.py](../../mmpp/plotting.py), `setup_custom_fonts`; [paper.mplstyle](../../mmpp/paper.mplstyle).

W testach wystąpiły ostrzeżenia o braku glifów Unicode SUBSCRIPT ZERO/ONE w Arial. W sondach pojawił się fallback brakującej rodziny cursive. To faktyczne problemy renderera; nie dowodzą niedostępności całej czcionki. PNG skalowane w dół dodatkowo pomniejsza już małe ticki. Dashboard skyrmionu wstawia pełny string promienia do tytułu — może on zajmować nadmiernie dużo miejsca.

Proponowany standard: kontrolki 13–14 px, tekst pomocniczy 12 px, czytelny kontrast, jeden spójny zestaw fontów z dostępnymi symbolami naukowymi. Wykresy ekranowe: ticki około 10–11 pt, etykiety 11–12 pt; osobny preset eksportu publikacyjnego. Subskrypty przez mathtext zamiast zakładania glifów Arial; emoji nie powinny być jedynym nośnikiem znaczenia. Liczby: 3–4 cyfry znaczące, pełna precyzja w danych/tooltipie.

Zweryfikować kontrast 4,5:1 dla zwykłego tekstu, powiększenie 200%, focus i czytelność w obu motywach. To kryteria akceptacji, nie deklaracja zgodności całej aplikacji z WCAG. [W3C: Contrast Minimum](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum).

### A19 — P2: brak rozróżnienia aktualnych kontrolek i obliczonego wyniku [R, K]

Źródło: [dashboard skyrmionu](../../mmpp/solitons/skyrmion/ui/interactive_dashboard.py), konstruktor, `run_selected` i `run`.

Po zmianie Frame z 0 na 1 pozostają obraz i zielone „Done” z klatki 0. Jawny przycisk Run jest dobrym wyborem dla kosztownych obliczeń, ale wymaga stanu „parametry zmienione — wynik dla poprzedniej konfiguracji”. Kontrolka Frame nie jest przekazywana do `open_spectrum/open_modes`; te ścieżki wykorzystują cały związany widok czasowy, co UI powinien jasno wyjaśniać.

Naprawa: `pending_config` i `computed_config`, oznaczenie nieaktualnego wyniku, blokowanie duplikujących się obliczeń i prosty przebieg `gotowe → liczenie → wynik/błąd`. Dla kosztownych zadań: postęp klatek/chunków, możliwość anulowania, identyfikator zadania odrzucający spóźnione wyniki. Matplotlib i aktualizacje UI powinny pozostać w odpowiednim wątku; przeniesienie samego callbacku do wątku nie wystarczy.

### A20 — P2: preset wyglądu nie jest odtwarzalnym eksperymentem [K]

Źródła: [FMR presets](../../mmpp/fft/modes/_interactive/presets.py), `collect_preset_state`; [vortex presets](../../mmpp/solitons/vortex/ui/interactive_dashboard.py), linie 2689–2770; [dispersion presets](../../mmpp/fft/dispersion/_interactive/presets.py).

Presety zapisują kontrolki, ale nie kompletną, wersjonowaną specyfikację obliczeń. Vortex ignoruje błędy przy przywracaniu pojedynczych pól i może mimo częściowego odtworzenia zgłosić sukces. Nazwa presetu vortex jest bez sanitacji dołączana do ścieżki — np. `../name` może wyjść poza folder `vortex_dashboard`. Nie wykonywano takich zapisów w audycie. Nie jest to dowód zdalnego ataku, lecz problem integralności lokalnego zapisu.

Naprawa: oddzielić „preset UI” od eksportu analizy; schema_version, wersja MMPP, dataset, slice, warstwa, spacing, czas, metoda, okna, resampling, normalizacja, seed, źródło danych i znaczniki jakości. Walidować preset przed zastosowaniem, raportować odrzucone pola. Nazwy ograniczyć do bezpiecznej nazwy pliku, zapisywać atomowo, ścieżkę storage pozwolić skonfigurować poza read-only katalogiem danych.

### A21 — P2: niespójna obsługa błędów i możliwości środowiska [R, K]

Źródła: [skyrmion status](../../mmpp/solitons/skyrmion/ui/interactive_dashboard.py), linie 173–203 i 280–288; [FFT status](../../mmpp/fft/modes/_interactive/status.py); [dyspersja diagnostics](../../mmpp/fft/dispersion/_interactive/widget.py); [extras](../../pyproject.toml).

Skyrmion interpoluje wyjątek bez `html.escape` do widgets.HTML; FFT status już prawidłowo escapuje. Nie ustalono wykonalności skryptów w danym frontendzie, ale błąd formatowania/niespójność escaping jest potwierdzony kodowo. Dashboard wirów przechwytuje wiele wyjątków i zapisuje status; bez testu callbacka wyjątek może nie spowodować niepowodzenia testu „figura istnieje”.

Extra `interactive` nie deklaruje `ipympl`, choć ścieżka `%matplotlib widget` go wymaga; jego obecność w tym środowisku nie naprawia metadanych pakietu. [Instrukcja instalacji ipympl](https://matplotlib.org/ipympl/installing.html). W testach pojawia się także deprecacja `Layout(gap='8px')`: parametr nie jest obsługiwany przez zainstalowany layout i może w przyszłości stać się błędem.

Naprawa: wspólny, escapujący status; szczegóły błędu w rozwijanym logu; diagnostyka backendu, wersji widgets i możliwości eksportu. Rozdzielić minimum notebookowe (`ipywidgets`, `ipympl`) od ciężkich zależności 3D. Testować instalację od zera oraz kontrolowane działanie inline jako podglądu statycznego.

## 6. Dalsza kwalifikacja naukowa

### FFT i mody

UI powinien zawsze ujawniać obserwablę i redukcję przestrzenną. `|FFT(mean_space(m))|²` i `mean_space(|FFT(m)|²)` nie są tym samym pomiarem: antyfazowe obszary mogą znosić się w pierwszym. To nie jest błąd samo w sobie; nazwa metody powinna objaśniać różnicę i towarzyszyć eksportowi.

20 klatek jest dobrym testem szybkości i propagacji slice, ale nie gwarantuje rozdzielczości widmowej. Dla równomiernego próbkowania odstęp binów wynosi `1/(N*dt)`, a dla N=20 Nyquist to `1/(2*dt)`. Zero-padding zagęszcza siatkę częstotliwości, lecz nie dodaje czasu obserwacji. Pokazywać N, dt, zakres czasu, Nyquist, odstęp binów oraz okno; nie traktować interpolowanej pozycji kursora jako niezależnie zmierzonej częstotliwości. Różnicę requested/bin już częściowo pokazuje status FMR.

Resampling domyślny rozwiązuje problem odrzucania lekko nieregularnego czasu, ale nie jest bezwarunkową kwalifikacją ilościową. Potrzebne są testy błędu amplitudy, fazy i szerokości piku względem jitteru i f/fNyquist, wykrywanie dużych luk oraz jawny opis interpolacji. Filtry wizualne, wygładzanie i odejmowanie baseline nie powinny po cichu zmieniać danych używanych do raportowania linewidth/dampingu. Faza przy amplitudzie bliskiej zeru powinna być maskowana; nie ma tam stabilnego znaczenia.

Dla sumy wielu binów częstotliwości animacja jedną częstotliwością pokazuje wizualizację zredukowanego profilu, a nie dokładne odtworzenie sygnału. Rekonstrukcja fizyczna powinna używać osobnych `exp(iω_j t)` albo być jasno opisana jako przybliżenie. Dodatkowo należy rozróżniać arbitralną normalizację modu od amplitudy magnetyzacji w jednostkach fizycznych.

### Topologia i skyrmiony

Pozytywnie: kod normalizuje wektory magnetyzacji, wyklucza komórki niemagnetyczne, rozróżnia konwencję osi i ma testy zgodności znaku estymatorów. Berg–Lüscher sumuje ładunki trójkątów przez kąt bryłowy; metoda ma źródło w [pracy Berga i Lüschera, rekord CERN](https://cds.cern.ch/record/134285). Testy syntetyczne z |Q|≈1 są konieczne, ale nie wyczerpują kwalifikacji dla rzeczywistego materiału.

Do macierzy walidacji dodać: teksturę jednorodną Q≈0, odwrotną polaryzację, odbicie osi, maskę z dziurą, obiekt przy brzegu, teksturę obciętą, siatkę anizotropową i dwa skyrmiony. Nie wymuszać całkowitego Q dla niepełnego obiektu na otwartej granicy. Test zbieżności powinien utrzymywać tę samą geometrię fizyczną przy zmianie dx/dy — tabela czasu z sekcji 4 tego nie robi.

Radialne dopasowanie pojedynczego obiektu nie opisuje ogólnej elipsy, wieloskyrmionowej konfiguracji ani obiektu silnie zdeformowanego. Pokazywać coverage, residuals, quality, flagi i zakres radialny. Wybór modelu przez AICc jest wyborem statystycznym w danej rodzinie; nie dowodzi konkretnego mechanizmu fizycznego. Odchylenia systematyczne po radialnym uśrednieniu wymagają kontroli mapą 2D.

### Wir i Thiele

Oddzielić wyznaczenie trajektorii z magnetyzacji od modelu zredukowanego. Podpisy powinny jawnie pokazywać p, c, kierunek osi, znak prądu, CIP/CPP, J lub geometrię przeliczenia I→J, materiał, pole i zakres dopuszczalnego promienia. Proxy TMR nie jest bez kalibracji napięciem mierzonym ani mocą w dBm.

Pełny dashboard ma opcje ODE/SDE, temperatury i seeda — to dobry punkt wyjścia. Wiarygodna szerokość linii z szumem wymaga dostatecznie długiego przebiegu, określonego odrzucenia transjentu, kroku i estymatora PSD oraz powtórzeń. Sam pojedynczy wykres ani zgodność solverów nie dowodzą poprawności całego modelu. W niniejszym audycie nie przeprowadzano nowej kalibracji względem MuMax ani pełnego badania identyfikowalności dopasowań.

## 7. Proponowany wspólny projekt UI i modułów

Wspólny szkielet powinien ujednolicić sterowanie i zapis stanu, zachowując osobne implementacje fizyki:

```text
DatasetSelection + Geometry + TimeAxis
                 ↓
ValidatedComputeConfig → AnalysisService → Result + Provenance + Quality
                                               ↓
                                  DisplayState → Renderer
                                               ↓
                                  NotebookSession / Export
```

`ComputeConfig` obejmuje wybór danych, metody i parametry wpływające na wynik. `DisplayState` obejmuje paletę, zoom, rozmiary tekstu, układ i widoczność paneli. `NotebookSession` zarządza commami, timerami, zamykaniem, postępem, anulowaniem i odrzucaniem starych wyników. Cache obliczeń kluczować kompletnym configiem i wersją danych; cache renderu — wynikiem oraz display state.

Proponowany podział panelu:

| Grupa | Zawartość | Kiedy przeliczać |
|---|---|---|
| Dane | Dataset, slice czasu, warstwa, ROI, komponenty, geometria | Jawne Apply/Run |
| Analiza | Metoda, okno, segment, topology/fit, parametry modelu | Jawne Apply/Run, z walidacją |
| Widok | Paleta, jednostki wyświetlania, wspólna skala, podpisy | Lekka aktualizacja artystów |
| Wybór | Bin f/k, klatka, faza | Szybki odczyt cache + render |
| Jakość | Ostrzeżenia, N/dt/Δf, coverage, residuals | Razem z wynikiem |
| Eksport | PNG/SVG/PDF, dane, config, provenance, animacja | Oddzielne zadanie |

Publicznie spójny obiekt viewer: `show()`, `close()`, `diagnostics()`, `result`, `state`, `export()`. Zachować kompatybilne fasady dotychczasowego API; nie zastępować wszystkich helperów jedną wielką klasą. Migrację zacząć od kontraktów jednostek i wyników, następnie wspólnego statusu/layoutu, dopiero potem przepinać dashboardy.

Proponowane cele wydajnościowe do przyszłego pomiaru, **nie obecne wyniki**: reakcja lekkiej kontrolki p95 <100 ms, cached frame <150 ms, zmiana stylu bez odczytu Zarr i bez FFT, stan postępu widoczny przy operacji >1 s. Pomiar rozdzielić na I/O, obliczenia, przygotowanie wykresu, serializację PNG/comm i render przeglądarki. Cache budżetować w MB, nie tylko liczbie wpisów.

## 8. Plan prac i warunki odbioru

| Etap | Zakres | Warunek zamknięcia |
|---|---|---|
| 1 — poprawność wyników | A01–A08: jednostki, czas, PSD, warstwa, skale, cache, geometria, obietnice fit | Reprodukcje przechodzą jako testy poprawności; brak ukrytej zmiany konwencji bez migracji |
| 2 — koszt i funkcjonalność | A09–A15: obserwable, zakresy, lazy I/O, BL, aktualizacje figur, streaming, lifecycle | Dane czytane tylko dla potrzebnego zakresu; benchmark i cykle cleanup |
| 3 — notebook UX | A16–A21: reflow, CSS, tekst, stany, presety, diagnostyka | Macierz prawdziwego Jupyter, zrzuty i testy interakcji |
| 4 — publikowalność | Provenance, eksport, testy naukowe z sekcji 6 | Eksport odtwarza analizę; jawne jednostki i quality; artefakt instalacyjny zweryfikowany |

Najważniejsze scenariusze akceptacyjne:

1. FFT: mx, my, mz, wszystkie i kombinacje; 20 oraz długi szereg; różne warstwy; slice kroczący; regularny czas, jitter, duża luka; input read-only; raw/power/density oraz zero-padding.
2. Mody: sin/cos i rotacja kołowa, requested vs actual bin, DC bez animacji 1/f, wspólny zakres amplitudy, phase mask i zgodność czasu podglądu z eksportem.
3. Skyrmion: seria klatek, crop/stride, anisotropic grid, maska, zmiana config z cache, brak spacing, obiekt brzegowy; nieaktualny wynik widocznie oznaczony.
4. Vortex/Thiele: zmiana źródła trajektorii, p/c/J, stan przy brzegu, kontrolki fit naprawdę działające, seed i jednostki proxy.
5. UI: JupyterLab i Notebook, ipympl i inline fallback, szerokości 600/900/1200 px oraz wąski kontener 320 CSS px, jasny/ciemny motyw, zoom 100/200%, klawiatura i dwa dashboardy naraz.
6. Lifecycle/eksport: ponowne uruchomienie komórki, close, przerwanie zadania, brak SciPy/ipympl/FFmpeg, read-only cwd, uszkodzony preset, zamknięcie podczas animacji.

Nie deklarować zamknięcia punktów frontendowych na podstawie `Figure is not None` ani przejścia testów Agg. Wykres jako dwuwymiarowa treść może mieć inne ograniczenia reflow niż kontrolki, ale nie usprawiedliwia to nieosiągalnych przycisków.

## 9. Wykonane sprawdzenia i reprodukcja

Środowisko sond: Python 3.10.13, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.3, ipywidgets 8.1.8, ipympl 0.9.7, Zarr 2.18.3, backend Agg. Nie sprawdzano matrycy wszystkich wersji Pythona. Wskazówki w AGENTS o Python 3.9 różnią się od bieżącego `pyproject.toml`, który wymaga ≥3.10 — do macierzy instalacyjnej należy przyjąć rzeczywiste metadane pakietu.

Pierwszy zestaw:

```bash
python3 -m pytest tests/test_vortex_interactive.py tests/test_skyrmion_analysis.py tests/test_vortex_html_helpers.py tests/test_dispersion_release_gate.py tests/test_dispersion_mode_extraction.py tests/test_spectrum_modes_bridge.py -q -o addopts=''
```

Wynik: **286 passed, 34 warnings, 26,46 s**. Ostrzeżenia obejmowały nieobsługiwane `Layout(gap='8px')`, statyczny canvas Agg i brakujące glify Arial ₀/₁. Ostrzeżenie o niewritable IPython config dotyczy środowiska wykonania, nie jest sklasyfikowane jako błąd MMPP.

Sondy dodatkowe:

```bash
PYTHONPATH=. MPLBACKEND=Agg python3 scripts/analysis/audit_interactive_2026_09_25.py
```

Skrypt: [audit_interactive_2026_09_25.py](../../scripts/analysis/audit_interactive_2026_09_25.py). Generuje syntetyczne dane w TemporaryDirectory i wypisuje JSON, nie modyfikuje bibliotek ani zewnętrznych danych. Celowo raportuje zaobserwowane defekty, zamiast uznawać je za oczekiwane asercje poprawności. Powtórzenia czasów różnią się m.in. przez cache fontów i rozgrzanie importów; stabilne są wyniki logiczne i numeryczne opisane powyżej. Testowana ścieżka stride jest bezpośrednim dataset-bound `SkyrmionInterface(..., slice_info=...)`.

Zielony pierwszy zestaw testów i jednocześnie reprodukcje problemów opisanych w A01–A07 pokazują lukę w pokryciu kontraktów naukowych (A04 potwierdzono kodowo, pozostałe również sondami). To nie unieważnia testów; wyznacza konkretne regresje do dodania. Pełnego suite repozytorium i frontendowego E2E nie wykonano.

Dodatkowy zestaw fizyczny i API:

```bash
python3 -m pytest tests/test_vortex_spectrum.py tests/test_vortex_topology.py tests/test_vortex_modes.py tests/test_vortex_nonlinear.py tests/test_vortex_thiele_audit.py tests/test_imports_hysteresis.py -q -o addopts=''
```

Wynik: **71 passed, 33 warnings, 422,22 s**. W sumie wykonano **357 zakończonych powodzeniem testów** w dwóch zestawach. Obejmuje to istniejące regresje modeli Thiele, np. redukcji polaryzacji, potencjału/siły, FDT i zgodności integratorów, ale nie stanowi nowej pełnej walidacji modelu względem eksperymentu. Ostrzeżenia naukowe o L/R=0,444 poza zakresem asymptoty cienkiego dysku oraz o składowej polaryzatora w płaszczyźnie są pożądaną informacją o ograniczeniach modelu. Dodatkowe deprecacje UI: `description_tooltip` i `Layout(gap='20px')`.

Końcowe kontrole:

```bash
python3 -m ruff check scripts/analysis/audit_interactive_2026_09_25.py
python3 -m ruff format --check scripts/analysis/audit_interactive_2026_09_25.py
git diff --check
```

Wszystkie przeszły. Nie uruchamiano ponownie pełnego lint/mypy/build pakietu, ponieważ audyt nie zmienia jego implementacji. Linki lokalne w raporcie odnoszą się do audytowanych źródeł; numery linii są aktualne dla bazowego commita i mogą przesunąć się po naprawach.
