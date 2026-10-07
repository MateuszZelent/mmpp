# Audyt kodu MMPP: FFT, solitony i wspólne kontrakty danych

Data: 2026-10-07. Punkt odniesienia: `8ffdbef7f603f845a5608cffc86bd4267c1d474e`.

## Metoda i granice audytu

Audyt jest przeglądem statycznym: czytaniem implementacji, śledzeniem wywołań i analizą wzorów, wymiarów fizycznych, konwencji współrzędnych oraz zależności między modułami. Zgodnie z poleceniem nie uruchamiano testów, obliczeń MMPP, notebooków, kompilacji, linterów ani benchmarków. Przykłady w raporcie są kontrprzykładami wynikającymi z kodu i rachunku, a nie wynikami uruchomionych eksperymentów. Nie zmieniano implementacji.

Ścieżka robocza `compute-lib/mmpp` prowadzi do `postprocessing/mmpp`. Na początku przeglądu drzewo było czyste; w trakcie pojawiła się niezależna zmiana w `mmpp/core/attributes.py`. Audyt jej nie modyfikuje. Odwołania do linii opisują stan plików czytany podczas przeglądu, nie obietnicę zgodności z późniejszymi zmianami.

Priorytety: **P1** — cichy błąd wyniku naukowego, utrata danych cache albo niedziałająca podstawowa ścieżka API; **P2** — błąd warunkowy, metadanych, jednostek lub wiarygodności interpretacji; **P3** — utrzymanie i organizacja. „Pewne z kodu” oznacza, że wskazana ścieżka i skutek wynikają bezpośrednio z implementacji. „Ograniczenie modelu” nie jest automatycznie błędem programistycznym: wymaga jawnego kontraktu i kwalifikacji stosowalności.

## Wniosek i najważniejsze ryzyka

MMPP ma użyteczne rozdzielenie obiektów wyników, accessorów notebookowych i części silników obliczeniowych, ale nie ma jeszcze jednolitego kontraktu danych fizycznych. Ten sam wynik może być analizowany z innym czasem, geometrią, konwencją znaku lub skalowaniem zależnie od wybranego accessora. To jest poważniejszy problem niż sama wielkość plików czy powtarzający się kod.

Najpilniejsze ustalenia to:

| Obszar | Ustalenie | Skutek |
| --- | --- | --- |
| Publiczne API modów | A01: przekazanie nieobsługiwanego `dset` | `compute_modes()` kończy się `TypeError`, a wcześniej może zapisać cache |
| Widoki danych | A02–A04: utrata materializacji, czasu i geometrii | analiza innego fragmentu danych lub wynik w niewłaściwej skali fizycznej |
| Widma solitonów | A06–A07: niezgodne normalizacje i widma zespolone | moc oraz CW/CCW zależą od metody i dostępności SciPy |
| Cache | A08–A10 | pominięcie parametrów, kasowanie przy `save=False`, zwrot transmisji innej symulacji |
| Transmisja | A11 | opcjonalne wyniki zerowe wyłącznie wskutek wyboru szybszej ścieżki |
| Dyspersja | A14–A16 | błędny bin Nyquista, odbicie profilu, zastąpienie żądanego datasetu |
| Elektromagnetyzm | A20 | zastępcze pola oznaczone jako poprawnie zakończona analiza fizyczna |
| Solitony | A25–A30 | niezgodne znaki, wymyślona polaryzacja, pozorna stacjonarność i nadinterpretacja detektorów |
| Modele analityczne | A33 | błędna granica częstotliwości PSSW dla zerowego wektora falowego w płaszczyźnie |
| Autofit | A37–A39 | dopasowywanie nieaktywnych parametrów, niemiarodajne niepewności i nieścisły budżet optymalizacji |

Nie oznacza to, że każdy wykres lub każda ścieżka MMPP daje błędny wynik. Warunki wystąpienia problemów są podane poniżej. W szczególności podstawowy silnik FFT ma lepszą normalizację niż wspólny helper widm solitonów, a część modeli eksperymentalnych poprawnie komunikuje swoje ograniczenia.

## Zakres i sposób czytania

Przegląd objął rodziny algorytmów i ich rzeczywiste połączenia z API, danymi oraz cache. Nie jest deklaracją przeczytania każdej linii wszystkich rendererów, widgetów i historycznych plików. W drzewie objętych inwentaryzacją pakietów jest około 134 tys. linii Pythona; duża część to interfejsy, dokumentacja w kodzie i warstwy kompatybilności. Liczba ta opisuje rozmiar drzewa, nie liczbę linii objętych jednakowo szczegółowym audytem.

| Rodzina | Zakres analizy statycznej |
| --- | --- |
| `core` | `MMPP`, `JobResult`, dataset/view, składanie wycinków, geometria, próbkowanie, materializacja/downsampling, przejście do accessorów |
| `fft` / `spectrum` | ładowanie, wybór datasetu, metody uśredniania, silniki, skalowanie, filtry, cache, wyniki, batch/sweep, przekazanie kontekstu modom |
| `fft/modes` | publiczna fasada i faktyczny `FMRModeAnalyzer`, obliczanie i cache, dostęp do profili, maski, charakterystyka i klasyfikacja; warstwa interaktywna w zakresie przekazywania danych |
| `fft/dispersion` | FFT 1D/2D, osie, normalizacja, filtry, cache używany przez interfejs, ekstrakcja modów, folding, wykrywanie okresowości, łączenie gałęzi, analiza minimów i prędkości |
| `fft/transmission` | metody, agregacja przestrzenna, okno referencyjne, ścieżki serial/parallel/vectorized/pre-FFT, cache, batch i kontrakt importu eksperymentu |
| Pozostałe FFT | charakterystyka modów, klasyfikator wirów, analiza elektromagnetyczna, szerokość piku; rozróżnienie kodu bieżącego i legacy |
| `solitons/vortex` | tracking, tabela vs pole, topologia, trajektoria, zdarzenia, widma, sygnały, energia/pinning, health, klasyfikatory, batch |
| `solitons/vortex/model`, `bridge`, `nonlinear`, `autofit` | ekstrakcja parametrów i jednostek, CIP/CPP, dopasowanie, przygotowanie danych, funkcja straty, symulacja, optymalizacja, diagnostyka |
| `solitons/skyrmion` | wybór danych i z, maski, ładunek topologiczny, centra, promień/profil, warunki jakości, wynik i batch |
| `analytical` | FMR, modele dyspersji, Thiele CIP/CPP i field-resolved, model nieliniowego STNO, założenia i granice |
| `analyze/hysteresis` | przygotowanie danych, rozdzielanie gałęzi, metryki, niepewność i interpolacje; nie pełny audyt animacji/UI |
| Wspólna infrastruktura | `_shared/spectral`, adapter H5, klucze i reprezentacja cache, granice importów i organizacja modułów |

Nie wykonywano audytu CLI/auth/remote-runs, bezpieczeństwa całej aplikacji ani kwalifikacji wszystkich kombinacji opcjonalnych backendów wizualizacji. UI analizowano tam, gdzie zmienia interpretację lub prezentację wyniku liczbowego. Nie uruchamiano istniejących testów i nie traktowano ich obecności jako dowodu poprawności.

W trakcie pracy HEAD przesunął się do `4ab1eeae7a26ecfeea9b20dec82b29976fca1038`. Porównanie z początkowym HEAD wykazało tylko niezależną zmianę `mmpp/core/attributes.py`; pozostałe opisane tu implementacje nie zmieniły się w tym porównaniu. Plik raportu jest lokalnym artefaktem: ogólna reguła `*.md` w `.gitignore:223` go ignoruje. Audyt nie zmienia tej reguły ani nie wykonuje stagingu.

## Przepływ danych i główna przyczyna problemów

```text
MMPP / odkrywanie wyników
  -> JobResult -> Zarr albo adapter H5
      -> Dataset / DatasetView
         [dane, wycinek, osie, czas, geometria, materializacja]
          -> FFT -> spectrum / modes / dispersion / transmission
          -> solitons -> vortex / skyrmion
                       -> trajectory / topology / energy / signals
                       -> bridge / Thiele / nonlinear / autofit
          -> cache i obiekty wyników -> wykres / notebook / eksport
```

Problem pojawia się, gdy accessor otrzymuje pełny widok, lecz niższa warstwa odtwarza wejście tylko z `job.path`, nazwy datasetu i części wycinka. Drugi wspólny wzorzec to utożsamianie wyniku zastępczego z wynikiem fizycznie rozpoznanym: brak danych staje się polaryzacją `+1`, brak kryterium stacjonarności — stanem ustalonym, a brak wzbudzenia — przerwą pasmową.

## A01–A05. Publiczne API i kontrakt datasetu

### A01 — P1 — Publiczne `compute_modes()` przekazuje parametr nieobsługiwany przez wykonawcę

**Dowód:** `mmpp/fft/modes/interface.py:1694–1705`; `mmpp/fft/modes/__init__.py:1463–1472`. Fasada wywołuje `self._legacy_analyzer.compute_modes(dset=dataset, **kwargs)`, podczas gdy rzeczywista metoda nie przyjmuje `dset` ani `**kwargs`. To jednoznaczny błąd wiązania argumentów.

**Dodatkowy skutek:** samo pobranie `_legacy_analyzer` uruchamia `_ensure_modes_ready()` (`interface.py:1421–1460`). Przy braku modów może wykonać obliczenie z `save=True`, zanim jawne wywołanie użytkownika, nawet z `save=False`, dotrze do `TypeError`.

**Kierunek naprawy:** wybrany dataset powinien ustalać kontekst konstruktora/instancji; jawne obliczenie nie powinno pobierać właściwości uruchamiającej inne obliczenie z własnymi opcjami. Oddzielić utworzenie analizatora od automatycznego przygotowania danych. **Status: pewne z kodu.**

### A02 — P1 — Accessory solitonów gubią materializowany widok datasetu

**Dowód:** `mmpp/core/dataset.py:950–1018,1330–1386`; `mmpp/solitons/interface.py`; `mmpp/solitons/vortex/interface.py:1–330`; `mmpp/solitons/skyrmion/interface.py:111–144`.

Dataset przekazuje do solitonów `dataset_view`, lecz kolejne interfejsy wiru przekazują dalej przede wszystkim job, nazwę i slice. Skyrmion również ponownie czyta `job[dataset]`. Tymczasem `downsample()` tworzy dane materializowane, nadpisaną geometrię i skalę kroku czasu, a `slice_info` może być `None`. Taki widok może więc analizować pierwotny dataset, zamiast danych po redukcji.

To nie jest tylko błąd opisu osi: zmienia się zbiór próbek i komórek użytych do trackingu/topologii/rozmiaru. Ponowne wyliczenie odstępów z atrybutów joba dodatkowo omija geometrię widoku.

**Kierunek naprawy:** przenosić jeden obiekt wejściowy zawierający dane lub loader, rzeczywisty czas, geometrię i identyfikator pochodzenia. Konsument nie powinien odtwarzać widoku z nazwy. **Status: pewne z kodu dla wskazanych ścieżek.**

### A03 — P1 — Czas jest odtwarzany z niewłaściwego źródła i nie zawsze odpowiada wycinkowi

**Dowód:** `mmpp/core/dataset.py:1050–1108`; `mmpp/solitons/vortex/numerical/core/interface.py:196–485`; `numerical/core/tracking.py:296–424`; `mmpp/solitons/batch.py:435–475`; `numerical/signals/power_spectrum.py`; `vortex/_shared/analysis.py`.

Występuje kilka połączonych problemów: `Dataset.dt` preferuje globalne `t_sampl` przed czasem konkretnego datasetu i nie uwzględnia kroku wycinka; tracking tworzy `arange(nt)*dt`; ścieżka tabelowego `ext_corepos` nie stosuje wyboru czasu z datasetu. W części obliczeń sygnałów przekazywane jest samo `dt`, więc helper nie może wykryć nieregularnego rzeczywistego czasu. Awaryjne `1e-12 s` tworzy wiarygodnie wyglądającą oś mimo braku metadanych.

**Kontrprzykład:** wybranie co czwartej próbki z zachowaniem bazowego `dt` zawyża częstotliwość i prędkość czterokrotnie; wycięcie początku i ponowne ustawienie `t=0` psuje synchronizację z tabelą energii. `numerical/energy/interface.py:139–153` akceptuje kanał energii na podstawie zgodności długości, bez zgodności znaczników czasu.

**Kierunek naprawy:** czas ma pochodzić z dokładnie wybranego widoku; tabele należy dopasowywać po czasie, a nie po liczbie wierszy. Brak skali fizycznej powinien być jawnym brakiem danych. **Status: pewne z kodu.**

### A04 — P2 — Przycinanie podczas downsamplingu rozciąga zachowane dane na oryginalny obszar

**Dowód:** `mmpp/core/dataset.py:1279–1386`; `mmpp/core/dataset_geometry.py:682–716`.

Redukcja dobiera całkowity rozmiar bloków i przycina nadmiarowe komórki. Następnie `geometry.resampled()` zachowuje pełne pierwotne granice. Dla 10 komórek redukowanych do 3 powstają trzy średnie z bloków po 3 komórki, obejmujące pierwsze 9 komórek, lecz geometria przypisuje im szerokość 10 komórek. Nowy odstęp jest `10*dx/3`, zamiast `3*dx`.

Ostrzeżenie o trimie nie naprawia przypisanego zakresu. Błąd wpływa na długości, położenia i wektory falowe. **Kierunek naprawy:** najpierw wyznaczyć geometrię faktycznie przyciętego obszaru, następnie ją zredukować; ewentualnie użyć resamplingu rzeczywiście obejmującego całą domenę. **Status: pewne z kodu.**

### A05 — P2 — Składanie odwróconych wycinków gubi semantykę końca `None`

**Dowód:** `mmpp/core/dataset_geometry.py:342–352`; użycie w `mmpp/core/dataset.py:815–889`.

Algorytm składa liczby zwrócone przez `slice.indices()` w nowy obiekt `slice`. Dla pełnej osi długości 5 i `[::-1]` może w ten sposób utworzyć `slice(4, -1, -1)`. Jawne `-1` jest ponownie interpretowane względem końca tablicy; nie jest równoważne końcowi `None` w oryginalnym odwróceniu. Wynik jest pusty zamiast zawierać odwróconą oś.

**Kierunek naprawy:** zachować znaczenie sentinelów przy ujemnym kroku albo składać zakresy indeksów z jednoznaczną konwersją do slice. Jeśli backend nie obsługuje ujemnych kroków, zgłosić to jawnie, zamiast tworzyć pozornie poprawny widok. **Status: pewne z semantyki indeksowania i kodu.**

## A06–A09. Widma i cache FFT

### A06 — P1 — Wspólny periodogram nie ma normalizacji PSD

**Dowód:** `mmpp/_shared/spectral.py:145–168,223–273,276–306`.

Ścieżka NumPy zwraca `abs(FFT((x-mean)*w))**2 / sum(w**2)`. Dla gęstości widmowej względem Hz brakuje czynnika `dt=1/fs`, a dla jednostronnego widma sygnału rzeczywistego także podwojenia binów wewnętrznych. Domyślne `compute_psd(..., scaling="density")` używa tej ścieżki dla `periodogram` oraz po braku SciPy. Argumenty `scaling` i `detrend` nie są tu respektowane tak jak w Welch. Analogiczny problem dotyczy fallbacku STFT.

Wartości nie mają deklarowanej interpretacji na Hz, więc całkowanie PSD i porównywanie mocy pomiędzy różnymi `dt` lub backendami jest niewiarygodne. Sam fakt, że Welch i pojedynczy periodogram mają inną wariancję, nie wyjaśnia brakującego czynnika jednostek. Definicję jednostek i jednostronnego sumowania podaje [dokumentacja SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).

**Kierunek naprawy:** współdzielić jawne skalowanie z poprawniej zorganizowanym `mmpp/fft/_scaling.py`; zachować osobne kontrakty amplitude, power spectrum i PSD. **Status: pewne z kodu i analizy wymiarowej.**

### A07 — P2 — Widma zespolone i krótkie sygnały mają niespójne zachowanie

**Dowód:** `mmpp/_shared/spectral.py:155–162,217–253,309–375`.

Periodogram zespolony odcina ujemne częstotliwości; Welch zwraca pełne widmo dwustronne, chociaż metadane mówią o dodatnich częstotliwościach. Dla `x+i*y` ujemna częstotliwość reprezentuje przeciwny kierunek obiegu — nie jest redundantnym lustrzanym fragmentem. Dodatkowo ścieżka spektrogramu rzutuje sygnał do `float`, tracąc część urojoną. Oś czasu spektrogramu nie dodaje początkowego znacznika czasu wejścia.

Dla 2–4 próbek Welch narzuca `nperseg>=8` i domyślnie `noverlap=4`; po skróceniu segmentu przez SciPy overlap może nie być mniejszy od długości segmentu. To niezależny błąd obsługi krótkich wejść. Dwustronne zachowanie dla sygnału zespolonego jest jawnie opisane w [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).

**Kierunek naprawy:** zdefiniować podpisaną, uporządkowaną oś zespoloną, nie usuwać kwadratury; dla krótkich sygnałów stosować jawny status lub poprawne parametry segmentacji. **Status: pewne z kodu.**

### A08 — P1 — Cache modów ignoruje parametry obliczeń, a `force` kasuje dane mimo `save=False`

**Dowód:** `mmpp/fft/modes/__init__.py:630–749,1493–1520`.

Istnienie `mode_group/arr` wystarcza do zakończenia `compute_modes()` bez porównania bieżącego `window`, `t_slice`, `z_slice` i polityki resamplingu. Identyfikator widoku nie zastępuje konfiguracji konkretnego wywołania. Zmiana okna lub zakresu może więc zwrócić poprzednie mody.

Przy `force=True` metoda otwiera Zarr w trybie zapisu i usuwa grupę modów, a dla pełnego widoku także `fft/{dataset}`, niezależnie od `save`. Kasowanie odbywa się przed załadowaniem i przeliczeniem danych. Wywołanie `force=True, save=False` albo nieudane przeliczenie może zniszczyć poprawny wcześniejszy cache.

**Kierunek naprawy:** klucz powinien obejmować efektywną konfigurację i źródło; `force` ma omijać odczyt, a `save=False` wykluczać mutacje. Nowy wpis należy ukończyć przed zastąpieniem starego. **Status: pewne z kodu.**

### A09 — P2 — Wymuszone odświeżenie pozostawia stary wynik w pamięci; odczyt cache gubi etykietę skali

**Dowód:** `mmpp/fft/spectrum/compute.py:88–121`; `mmpp/fft/_compute_cache.py:67–72`; `mmpp/fft/core.py` — budowa wyniku około linii 551.

Po `force=True` obliczony wynik nie jest wkładany do słownika cache, ponieważ zapis chroni `if use_cache and not force`. Kolejne zwykłe wywołanie może ponownie zwrócić stary obiekt. To błąd semantyki odświeżenia, szczególnie przy uzupełnianiu symulacji pod tą samą ścieżką.

Osobno loader wyjmuje `scaling` z metadanych do `config` przez `pop`, a fasada odczytuje skalę ponownie z metadanych z domyślnym `raw`. Wartości po odczycie cache mogą zachować poprawne liczby, ale wynik zostaje opisany jako surowy. Nie jest to dowód, że sam loader ponownie przeskalowuje tablicę.

**Kierunek naprawy:** odświeżyć albo unieważnić wpis pamięciowy po wymuszonym obliczeniu; opis skali odczytywać z jednego autorytatywnego pola. **Status: pewne z kodu.**

## A10–A13. Transmisja

### A10 — P1 — Wspólny zewnętrzny cache może zwrócić transmisję innej symulacji

**Dowód:** `mmpp/fft/transmission/cache.py:155–163,222–259,318–337,482–488`; `interface.py:249–257`; ścieżka batch w `transmission/batch.py` około linii 2250–2300.

Zewnętrzny katalog używa stałego pliku `transmission_cache.zarr`; grupa zależy od nazwy datasetu, klucz od konfiguracji/slice i opcjonalnego `view_identity`. Dla zwykłych danych nie zawiera tożsamości joba ani rewizji źródła. Odczyt nie porównuje zapisanego `zarr_path` z aktualnym jobem.

**Scenariusz:** dwie symulacje z datasetem `m`, taką samą konfiguracją i wspólnym `cache_path`. Druga może dostać wynik pierwszej; wspólny katalog jest także przekazywany z batch. Plik nie musi być uszkodzony i tablice mogą mieć prawidłowe rozmiary.

**Kierunek naprawy:** rozdzielić namespace po tożsamości/rewizji źródła i sprawdzać zgodność przy odczycie. Nie utożsamiać tego z osobnym cache batch transmisji, który ma własne mechanizmy sygnatur źródeł. **Status: pewne z konstrukcji klucza i loadera.**

### A11 — P1 — Ścieżki przyspieszone zwracają sztuczne zera w żądanych wynikach

**Dowód:** `mmpp/fft/transmission/compute.py:3171–3214,3252–3409,3558–3576,3920–3926,4041–4044`.

`complex_accum` jest alokowany przy `keep_complex_fft=True`, lecz uzupełniany tylko w standardowej pętli serial. Wybór parallel, vectorized/sliding-window albo pre-FFT pozostawia go zerowym, a na końcu do wyniku dołączana jest zerowa średnia. Warunki wyboru szybszych ścieżek nie wykluczają `keep_complex_fft`.

W `pre_fft` analogicznie alokowane `power_plus`, `power_minus` i mapy składowych nie są uzupełniane. Użytkownik dostaje liczbowy wynik zamiast informacji, że kombinacja opcji jest nieobsługiwana. Zmiana liczby okien przekraczająca próg parallel może więc zmienić dodatkowe wyniki bez zmiany fizyki.

**Kierunek naprawy:** wspólny kontrakt wyników każdej gałęzi; brak obsługi powinien być jawny. Nie zwracać zainicjalizowanych zer jako obliczonej wielkości. **Status: pewne z przepływu zapisów do tablic.**

### A12 — P2 — `cpsd` usuwa fazę przed uśrednieniem, a tryby transmisji reprezentują różne obserwable

**Dowód:** `mmpp/fft/transmission/compute.py:2467–2732,3252–3409`.

Metoda opisana jako `cpsd` bierze moduł iloczynu widma i sprzężonej referencji przed agregacją. Operacja `abs(X*conj(Y))=abs(X)*abs(Y)` usuwa fazę wzajemną; nie jest zespoloną gęstością widma krzyżowego ani koherencją. Dla pojedynczej składowej relacja do referencji odpowiada proporcji amplitud, podczas gdy `power_ratio` używa kwadratów amplitud. [SciPy CSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html) definiuje iloczyn zespolony i osobne uśrednianie jego części.

`pre_fft` sumuje amplitudy w oknie przed FFT, a typowa agregacja `post_fft` sumuje/uśrednia moce. Antyfazy mogą skasować pierwszy wynik i pozostać w drugim. To różnica obserwabli, nie wyłącznie wydajności. Składowe kołowe mają dodatkowy współczynnik normalizacyjny wymagający jawnego kontraktu wag.

**Kierunek naprawy:** nazwać każdą wielkość zgodnie z definicją, zachować zespolone CSD, udokumentować koherentne i niekoherentne uśrednianie. `|m|²` względem referencji nie jest automatycznie transmisją strumienia energii pomiędzy różnymi ośrodkami. **Status: pewna algebra; interpretacja energetyczna wymaga modelu.**

### A13 — P2 — Callback postępu może przerwać zapis cache po zapisaniu tablic

**Dowód:** `mmpp/fft/transmission/cache.py:246–249,460–483`.

Generator klucza usuwa `progress_callback`, ale zapis metadanych wykonuje `json.dumps(asdict(result.config))` bez tego usunięcia. Funkcja przekazana jako callback nie jest serializowalna standardowym JSON. Wyjątek następuje po utworzeniu tablic i pozostawia częściowy wpis.

**Kierunek naprawy:** jedna reprezentacja konfiguracji przeznaczona do serializacji, oddzielna od callbacków i runtime; zapis do wpisu tymczasowego z oznaczeniem ukończenia. **Status: pewne z kodu dla konfiguracji z callable.**

## A14–A19. Dyspersja, rekonstrukcja i rozpoznawanie gałęzi

### A14 — P1 — Odwrócenie osi k nie zachowuje binu Nyquista

**Dowód:** `mmpp/fft/dispersion/core.py:96–102,1437–1455`.

`_mirror_k_indices()` dla każdego k szuka najbliższego punktu do `-k` przez minimalizację odległości. Na parzystej siatce FFT ujemny Nyquist nie ma osobnego dodatniego odpowiednika: powinien mapować na siebie modulo rozmiar transformacji. Dla osi proporcjonalnej do `[-2,-1,0,1]` kod szuka `+2` i wybiera `+1`. Duplikuje sąsiedni bin i gubi właściwy Nyquist. `flipx=True` jest domyślne w ścieżce 1D, a mapowanie obejmuje także `S_complex`.

**Kierunek naprawy:** stosować permutację indeksów DFT modulo N, nie najbliższy punkt na obciętej osi rzeczywistej. Sposób reprezentacji Nyquista opisuje [NumPy FFT](https://numpy.org/doc/2.2/reference/routines.fft.html). **Status: pewne z kodu; problem dotyczy parzystych siatek.**

### A15 — P1 — Profil przestrzenny jest rekonstruowany z widma o odwróconym k

**Dowód:** `mmpp/fft/dispersion/core.py:1337–1350,1445`; `mmpp/fft/dispersion/modes/extraction.py:305–340`.

Kod odwraca `S_complex` wraz z osią kierunku propagacji, po czym ekstrakcja wykonuje IFFT bez odwrócenia tej operacji. Z tożsamości DFT wynika `IFFT(F(-k))[j]=a[-j mod N]`: asymetryczna obwiednia zostaje odbita, nawet abstrahując od osobnego problemu Nyquista. Komentarz o korekcie konwencji NumPy nie zastępuje odwracalnego przekształcenia współczynników.

Ekstrakcja buduje też oś kierunku propagacji od zera, zamiast zachować początek wybranego obszaru. Dodatkowej konsekwentnej definicji wymaga znak czasowej fazy animacji; samo przestawienie k nie jest sprzężeniem zespolonym dowolnego profilu.

**Kierunek naprawy:** zachować surowe współczynniki w konwencji obliczeniowej; mapowanie do konwencji wykresu powinno być odwracalne i opisane metadanymi. Rekonstrukcja ma odtwarzać także fizyczną oś widoku. **Status: pewne z algebraicznej tożsamości transformacji.**

### A16 — P1 — Dyspersja może zastąpić jawnie żądany dataset; część FFT omija adapter H5

**Dowód:** `mmpp/fft/dispersion/core.py:298–409`; `mmpp/fft/modes/__init__.py:1526` i dalsze ładowanie; `mmpp/core/job.py:196–217,1035–1090`; `mmpp/pyzfn/h5_backend.py:338–373`.

Loader dyspersji otwiera surowy Zarr, próbuje żądanej nazwy, a następnie inne nazwy i dostępne tablice. Błąd, brak lub nieobsługiwany typ pierwszego obiektu może skończyć się analizą innego datasetu. W szczególności H5-backed quantity jest obsługiwane przez `JobResult`, ale nie przez każde bezpośrednie otwarcie Zarr. Także legacy obliczanie modów i automatyczne wyszukiwanie największego datasetu mają ścieżki oparte na surowych grupach.

Użytkownik może dostać dane z innej warstwy lub z innym próbkowaniem, przy zachowanym kontekście żądanej nazwy. Nie jest to zarzut wobec całego FFT: `_compute_loading._resolve_dataset()` potrafi przejść przez adapter joba.

**Kierunek naprawy:** jawny wybór datasetu powinien być ścisły; heurystyczne wyszukiwanie tylko przy braku wyboru. Każdy silnik powinien korzystać z tego samego protokołu źródła Zarr/H5/view. **Status: pewne z kodu; obsługa H5 jest zależna od ścieżki API.**

### A17 — P2 — Gęstość widmowa dyspersji nie określa jednoznacznie miary osi k

**Dowód:** `mmpp/fft/dispersion/core.py:1399` i blok skalowania 1D; analogiczny blok 2D około linii 1723; tworzenie osi k w `dispersion/utils.py`.

Skalowanie PSD zawiera `dt*dx` podzielone przez energię okien, a oś k jest w rad/m. To odpowiada gęstości względem częstotliwości przestrzennej w cyklach/m albo całkowaniu z miarą `dk/(2π)`. Jeśli użytkownik całkuje po wyświetlanej osi `dk`, otrzyma dodatkowy czynnik `2π`; w dwóch wymiarach przestrzennych analogicznie `(2π)^2`.

Nie każda normalizacja musi być gęstością względem `dk`; błąd kontraktu polega na braku jednoznacznego wskazania miary przy nazwie PSD. **Kierunek naprawy:** określić jednostki i miarę całki w wyniku, a następnie zastosować odpowiedni Jacobian. Rozdzielić surową moc, kwadrat amplitudy i gęstość. **Status: pewna analiza wymiarowa, wymagane doprecyzowanie zamierzonego kontraktu.**

### A18 — P2 — Folding łączy gałęzie po rankingu amplitudy, a pomocnicze detektory mają błędy metadanych/progu

**Dowód:** `mmpp/fft/dispersion/modes/folding.py:111–133,173–178,254–278,309–334`; `modes/detection.py:297–300,331–344`.

W foldingu `branch_index` pochodzi z kolejności pików posortowanych malejąco po intensywności. Następnie klucz gałęzi jest kombinacją numeru strefy i tego indeksu. Dwie fizyczne gałęzie zamieniające siłę wzbudzenia zostaną więc zamienione miejscami, nawet jeżeli częstotliwości są ciągłe. To nie jest śledzenie tożsamości modu.

Dwa dalsze błędy są lokalne: w awaryjnym foldingu poza zakresem wygenerowanych G znak zwracanego `G_applied` jest przeciwny do kontraktu `k_folded=k+G`; detektor odstępu pików przekazuje `0.3*max(S)` do helpera, który ponownie mnoży przez `max(S)`. Próg staje się kwadratowy i detekcja zależy od dowolnego przeskalowania amplitudy.

**Kierunek naprawy:** łączyć gałęzie według jawnego kryterium ciągłości/overlapu i przechowywać niepewne przypisania; ujednolicić znak G oraz rozróżnić próg względny i bezwzględny. **Status: pewne z kodu.**

### A19 — P2 — Przerwy pasmowe, okres sieci i masa efektywna są nadinterpretowane z mapy wzbudzenia

**Dowód:** `mmpp/fft/dispersion/modes/detection.py:187–197,383–501`; `dispersion/analyze.py` — wybór maksimum/centroidu około linii 590–780; `dispersion/_branch_linker.py`.

`find_band_gaps()` szuka obszarów małej sumy intensywności po k. Brak sygnału może wynikać z symetrii wzbudzenia, maski, filtru, skończonego czasu lub małej czułości na konkretną składową; nie dowodzi braku modów. Detektor okresu potrafi zwrócić ograniczoną do zadanego przedziału wartość awaryjną z zakresu osi k, mimo braku przekonującej okresowości. Ostrzega w logu, ale nadal daje zwykłą liczbę nadającą się do dalszego foldingu.

Masa efektywna jest liczona z paraboli przez najsilniejsze piki, które mogą należeć do różnych gałęzi. Sam wzór `m*=ħ/(2a)` dla `ω≈ω0+a*k²` jest spójny; nie jest dowodem, że dobrano tę samą gałąź. Podobnie minimum grzbietu największej intensywności nie musi być minimum najniższego pasma.

**Kierunek naprawy:** zwracać kandydatów, kryterium i status jakości, np. „obszar małej obserwowanej intensywności”. Wnioskowanie o pasmach wymaga pokrycia wzbudzeniem i identyfikacji gałęzi. **Status: ograniczenie identyfikowalności fizycznej, nie zarzut wobec samej FFT.**

## A20–A24. Elektromagnetyzm, charakterystyka modów i filtry

### A20 — P1 — Analiza elektromagnetyczna używa zastępczych pól i niepoprawnego Poyntinga

**Dowód:** `mmpp/fft/electromagnetic_analysis.py:81–106,128–143,503–580`.

Helper konstruuje `E=i*ω*μ0*m` i `H=m/μ0`, następnie oznacza wynik `analysis_successful=True`. Nie wyznacza pól Maxwella z geometrii, warunków brzegowych i źródeł. Te dwa pola są proporcjonalne do tego samego wektora, więc użyty niesprzężony iloczyn wektorowy jest algebraicznie zerowy, poza błędami zaokrągleń.

`compute_poynting_vector()` dodatkowo liczy `E×H/μ0`. Dla H w A/m wektor chwilowy wynosi `E×H`; dzielenie przez μ0 dotyczyłoby użycia B zamiast H. Dla amplitud zespolonych średnia po czasie wymaga `0.5*Re(E×H*)` przy konwencji amplitudy szczytowej. Własna funkcja energii traktuje drugi argument jako H, więc nie jest to tylko inne nazewnictwo. Kontekst pól harmonicznych i średniej mocy opisuje [materiał MIT o zespolonym twierdzeniu Poyntinga](https://web.mit.edu/6.013_book/www/chapter12/12.5.html).

**Kierunek naprawy:** oddzielić demonstracyjny placeholder od analizy fizycznej; przyjmować rzeczywiste E/H z jawną jednostką i konwencją. Wniosek dotyczy tego bezpośrednio importowalnego API, nie dowodzi, że każde standardowe wywołanie FFT je uruchamia. **Status: pewne z kodu i wymiarów.**

### A21 — P2 — Elektromagnetyczny Q używa binów przy piku zamiast przecięć połowy maksimum

**Dowód:** `mmpp/fft/electromagnetic_analysis.py:440–478`.

Funkcja wybiera ostatni punkt po lewej i pierwszy po prawej wśród wartości powyżej połowy maksimum. To zwykle są bezpośredni sąsiedzi piku, nie zewnętrzne przecięcia poziomu. Dla szerokiego piku szerokość zostaje zredukowana do około `2*df`, a Q zależy głównie od rozdzielczości FFT.

**Kierunek naprawy:** szukać faktycznych przejść przez poziom i interpolować tylko przy istniejącym obustronnym przecięciu; wykorzystać bardziej ostrożny kontrakt z `mmpp/fft/metrics.py:35–128`. **Status: pewne z indeksowania.**

### A22 — P2 — Siatka w obliczaniu dalekiego pola jest transponowana

**Dowód:** `mmpp/fft/electromagnetic_analysis.py:285–297` i pętla po komórkach w tej metodzie.

`meshgrid(x,y,indexing="ij")` tworzy tablice `(nx,ny)`, lecz pętla indeksuje je jak `(ny,nx)`. Dla prostokątnego pola może dojść do wyjścia poza zakres, a dla kwadratowego problem może ukryć się jako przypisanie położenia z zamienionymi osiami.

**Kierunek naprawy:** jeden jawny porządek `(y,x)` zgodny z danymi, z przestrzennymi osiami w obiekcie wejściowym. **Status: pewne z kształtów i indeksowania.**

### A23 — P2 — Klasyfikacja przestrzennego modu zależy od arbitralnej globalnej fazy FFT

**Dowód:** `mmpp/fft/mode_characterization.py:444–499`; `mmpp/fft/vortex_classifier.py:150–181,225`.

Klasyfikator buduje pole z `real(mx)+i*real(my)`, a następnie analizuje jego nawinięcie. Przedtem usuwa kwadratury. Pomnożenie całego modu przez wspólną fazę `exp(iχ)` opisuje ten sam mod, ale może zmienić klasyfikację; jeśli składowe były rzeczywiste, pomnożenie przez i zeruje obie użyte części rzeczywiste. Również lokalizacja z rzeczywistej części dynamicznego `mz` może zależeć od wybranej fazy.

**Kierunek naprawy:** przestrzenny indeks modu wyznaczać z pełnego pola zespolonego, przy jawnej definicji polaryzacji, maski i kąta. Nie utożsamiać przestrzennego windingu statycznego wektora z czasowym CW/CCW. **Status: pewny kontrprzykład fazowy; zakres dotyczy wskazanych klasyfikatorów.**

### A24 — P2 — Bezpośredni pipeline filtrów może wykonać filtr ustawiony na `False`

**Dowód:** `mmpp/fft/filters/pipeline.py:121–130,175–192,220–306`; `mmpp/fft/filters/windows.py`; `mmpp/fft/spectrum/result.py` — metoda `filtered()` około linii 430–492.

Normalizacja jawnych bloków `pre/post` zachowuje wartości `False`. Ścieżka `FilterPipeline.preprocess()` iteruje po nich bez sprawdzenia `_is_enabled`, więc np. jawne `pre.remove_mean=False` może nadal usuwać średnią. Inne ścieżki używają `split_filter_stages()`, który filtruje wyłączone opcje — stąd niezgodność między accessorami. Konwersja obiektu konfiguracji pomija też niektóre warianty detrendingu.

Przy braku SciPy część nazwanych okien jest zastępowana prostokątnym lub innym oknem bez zachowania prawdziwej tożsamości metody. Osobno logarytmowanie przez `SpectrumResult.filtered()` może zapisać wartości ujemne jako power override, jednocześnie budując amplitudę przez `sqrt(clip(power,0))`; te pola przestają opisywać ten sam fizyczny sygnał.

**Kierunek naprawy:** jedna interpretacja aktywności i parametrów filtrów; jawna informacja o efektywnym oknie; transformacje wizualne oddzielone od wyniku liczbowego. **Status: pewne z kodu, zależne od ścieżki i opcji.**

## A25–A32. Solitony: topologia, zdarzenia, energia i sygnały

### A25 — P1 — Vortex i skyrmion nie mają wspólnej konwencji orientacji ładunku

**Dowód:** `mmpp/solitons/_topology.py:1–220`; `mmpp/solitons/_coordinates.py`; `mmpp/solitons/skyrmion/_core.py:103–188`; `mmpp/solitons/vortex/topology/detection.py`.

Wspólny helper dla `y_axis="up"` odwraca tablicę i stosuje dodatkowy znak, co znosi zmianę orientacji w całce. W implementacji skyrmionu ścieżka finite-difference ma osobną korektę znaku, aby zgodzić się z lokalną implementacją Berg–Lüscher. Komentarz wprost opisuje zachowanie historycznej konwencji wiru. Zatem te same dane i nominalnie ta sama orientacja nie mają jednego kontraktu Q pomiędzy accessorami.

Detekcja vorticity po okręgu wykorzystuje rosnący indeks wiersza jako kierunek y, podczas gdy chirality korzysta z przeliczenia na fizyczne y. To może mieszać znaki `Q`, windingu, polaryzacji i współczynnika żyroskopowego.

**Kierunek naprawy:** zdefiniować orientację bazy przestrzennej i składowych magnetyzacji oraz raz stosować Jacobian orientacji. Migracja historycznego znaku może wymagać wersjonowania wyników; nie wystarczy globalna zmiana jednego minusa. **Status: pewna niezgodność kontraktów; nie twierdzenie, że istnieje tylko jedna dopuszczalna konwencja znaku.**

### A26 — P2 — Brak polaryzacji w tabeli jest zastępowany pewnym `+1`

**Dowód:** `mmpp/solitons/vortex/numerical/core/interface.py:121–170`.

Jeżeli tabela ma położenie rdzenia, ale nie ma kolumny polaryzacji, wynik dostaje `polarity=ones` i `confidence=ones`. Położenie nie wyznacza znaku rdzenia. Dalszy model może użyć wymyślonej polaryzacji do kierunku gyracji, G lub działania STT.

**Kierunek naprawy:** reprezentować brak polaryzacji osobno od poprawności położenia; umożliwić jej odczyt z pola lub jawne zadanie. Nie interpretować pewności śledzenia pozycji jako pewności wszystkich właściwości. **Status: pewne z kodu.**

### A27 — P2 — Ekstrakcja stanu ustalonego ogłasza sukces także bez spełnienia kryterium

**Dowód:** `mmpp/solitons/vortex/trajectory/steady_state.py:28–88`.

Jeżeli żaden rozważany fragment nie spełnia warunku stabilizacji, funkcja wybiera ostatnie `min_samples` i ustawia `metadata["steady_state"]=True`. Użytkownik nie odróżni wykrycia stacjonarności od awaryjnego wycięcia końca. Kryterium oparte na zmienności promienia ma ponadto własne ograniczenia: nie dowodzi stabilizacji wszystkich obserwabli ani fazy.

**Kierunek naprawy:** oddzielić wybór fragmentu od rozpoznania stanu; zwracać `detected=False/unknown`, przyczynę i miarę spełnienia kryterium. **Status: pewne z przepływu sterowania.**

### A28 — P2, P1 przy filtrowaniu zbioru — Health może pomylić obecny wir z anihilacją

**Dowód:** `mmpp/solitons/vortex/health.py:196–234,311–333,369–384`; `mmpp/solitons/batch.py:581–610`.

Health używa średniego `mz` w stałym obszarze wokół środka obrazu. Wąski rdzeń ma mały udział powierzchniowy; rdzeń przesunięty poza ten obszar jeszcze mniejszy. Mała średnia nie dowodzi zaniku struktury. Warunek końcowego `abs(mean_mz)<0.05` może więc fałszywie ogłosić anihilację.

Ocena odległości od brzegu względem środka trajektorii, zamiast środka próbki, może z kolei przeoczyć silnie przesuniętą orbitę. Nieudany odczyt pola jest przechwytywany, a brak obserwacji nie musi spowodować statusu unknown. Opcja batch `exclude_annihilated` jest domyślnie wyłączona; po włączeniu może usuwać rekordy na podstawie tej zawodnej oceny, obejmującej także inne stany unhealthy.

**Kierunek naprawy:** oceniać istnienie rdzenia lokalnie przy śledzonej pozycji, z topologią i maską materiału; oddzielić annihilated, switched, edge-contact oraz unavailable. Filtrowanie powinno zachowywać powody i listę odrzuconych rekordów. **Status: pewna nieidentyfikowalność z zastosowanej obserwabli.**

### A29 — P2 — Klasyfikatory G/C, radial/azimuthal i lokalnego stanu wykraczają poza dostępne dane

**Dowód:** `mmpp/solitons/vortex/events/state_transitions.py`; `vortex/modes/classifier.py`; `modes/radial.py`; `vortex/topology/detection.py:106–117`; podsumowania w `mmpp/solitons/batch.py:1395–1621`.

Detektor G/C normalizuje promień orbity własnym wysokim percentylem. Idealna mała orbita kołowa ma wtedy wartość bliską 1, więc próg 0.6 nie oznacza dużego odchylenia względem promienia dysku. Stan C opisuje teksturę magnetyzacji, której sama trajektoria punktu nie określa.

Klasyfikator modów korzystający z widma położenia/promienia przypisuje przestrzenne indeksy m/n na podstawie kierunku obrotu i numeru harmonicznej. Druga harmoniczna promienia orbity eliptycznej może istnieć bez oddychania samego rdzenia. To nie daje ogólnej identyfikacji radialnego modu przestrzennego. W lokalnym klasyfikatorze topologii sprawdzenie znaku windingu przed `abs(Q)>0.8` dodatkowo kieruje typowy przypadek skyrmionu do vortex/antivortex.

W batch względna moc odniesiona do maksimum całego zestawu może zmienić etykietę tej samej symulacji po dodaniu innego rekordu.

**Kierunek naprawy:** wyniki trajektorii nazywać harmonicznymi/diagnostyką orbity; rozpoznanie tekstury i indeksów wymaga pola przestrzennego. Progi powinny odnosić się do fizycznego rozmiaru lub lokalnie zdefiniowanej skali, nie przypadkowego zestawu porównań. **Status: połączone błędy warunków i ograniczenia identyfikowalności.**

### A30 — P2 — Radialna energia swobodna jest interpretowana jako potencjał i miejsca pinningu

**Dowód:** `mmpp/solitons/vortex/numerical/energy/potential.py:17–64`; `energy/interface.py:111–159`; `energy/pinning.py`.

Wzór `W(r)=-kBT*ln(P(r))` jest poprawną definicją radialnej energii swobodnej dla histogramu r. Nie jest jednak bezpośrednio energią potencjalną ruchu w dwóch wymiarach. Dla równowagi i osiowo symetrycznego `U(r)` zachodzi `P(r)dr ∝ 2πr*exp(-U(r)/kBT)dr`, więc kod otrzymuje `W=U-kBT ln(r)+const`. Nawet gładki potencjał harmoniczny ma wtedy pozorne minimum przy niezerowym r.

Następna warstwa nazywa lokalne minima miejscami pinningu, nie usuwając czynnika miary. Histogram radialny nie rozróżnia też pozycji kątowej kilku defektów. Tryb auto może użyć tej drogi przy domyślnej temperaturze 300 K bez dowodu termicznej równowagi; deterministyczna napędzana orbita nie spełnia założeń inwersji Boltzmanna.

**Kierunek naprawy:** osobno radialny PMF i energia potencjalna; dla tej drugiej uwzględnić miarę pierścieni oraz jawne założenia równowagi/temperatury. Pinning przestrzenny wymaga co najmniej analizy 2D albo wyraźnie ograniczonej interpretacji radialnej. **Status: ograniczenie fizycznej interpretacji, nie błąd samej definicji `-ln P(r)`.**

### A31 — P2 — Proxy magnetorezystancji nie zapewnia ilościowej amplitudy sygnału

**Dowód:** `mmpp/solitons/vortex/numerical/signals/magnetoresistance.py:30–97,100–139`; `signals/voltage.py`.

Fallback promienia dysku szacuje go z rozmiaru tej samej orbity. Wtedy proporcjonalne zwiększenie orbity zwiększa również mianownik i może prawie nie zmienić przewidywanej amplitudy MR. Kod odejmuje środek trajektorii, więc usuwa również informację o statycznym przesunięciu względem próbki. Dla polaryzatora z komponentem z przyjmuje średnie `mz` równe polaryzacji rdzenia, chociaż mały rdzeń nie oznacza nasycenia całego dysku w z.

Wynik poprawnie ma `method="trajectory_proxy"` — to wartościowa informacja, którą należy zachować. Nie wystarcza jednak do ilościowej interpretacji napięcia lub mocy w jednostkach fizycznych bez kalibracji rozkładu magnetyzacji i kontaktu.

**Kierunek naprawy:** wymagać fizycznej geometrii dla ilościowego trybu, stosować przestrzenne ważenie magnetyzacji/czułości kontaktu, a proxy oznaczać jako modelową estymację. **Status: ograniczenie modelu.**

### A32 — P2 — Szerokość piku jest utożsamiana z tłumieniem, a domyślna normalizacja ukrywa amplitudę

**Dowód:** `mmpp/solitons/vortex/nonlinear/slavin_tiberkevich.py` — przypisanie `Gamma_G=2π*linewidth_hz` około linii 169; `nonlinear/amplitude_equation.py`.

Szerokość widma stacjonarnego autooscylatora nie wyznacza sama w sobie dodatniego tłumienia Gilberta. Zależy m.in. od szumu, sprzężenia amplitudy z fazą, okna i czasu obserwacji. Także współczynnik `2π` nie rozstrzyga relacji linewidth–relaxation bez określenia, czy chodzi o FWHM/HWHM oraz zanik amplitudy czy energii. Związek nieliniowości z szerokością linii omawia [Kim, Tiberkevich i Slavin](https://arxiv.org/abs/cond-mat/0703317).

W równaniu amplitudy domyślny promień odniesienia jest wyznaczany z RMS danej trajektorii, więc średnia znormalizowana moc wynosi z konstrukcji około 1 dla każdej niezerowej orbity. Nie nadaje się to do porównania zmian amplitudy między prądami bez wspólnego odniesienia.

**Kierunek naprawy:** nazwać linewidth obserwablą widmową, a tłumienie identyfikować z właściwego modelu i danych; dla porównań mocy używać stałej skali fizycznej. **Status: ograniczenie identyfikacji fizycznej i kontraktu normalizacji.**

## A33–A36. Modele analityczne i identyfikacja parametrów

### A33 — P1 — Wyższe mody PSSW mają niepoprawną granicę przy k równoległym do filmu równym zero

**Dowód:** `mmpp/analytical/dispersion.py:392–422`, `kalinikos_no_approx()` dla `n>0`, `perpendicular=False`.

Kod definiuje `Fk=(1-exp(-|k|d))/(|k|d)` z granicą 1 i liczy `ω²=ω0*(ω0+ωM*(1-Fk))`. Dla `k=0` znika więc cały dynamiczny człon dipolowy i pozostaje `ω=|ω0|`.

Dla in-plane PSSW, bez anizotropii i w użytym przybliżeniu diagonalnym, granica powinna zawierać oba pola sztywności: `ω²=γ²*(B+Bex)*(B+Bex+μ0Ms)`, gdzie `Bex=2*Aex*(nπ/d)²/Ms`. Pominięcie drugiego czynnika zmienia częstotliwość także w prostym przypadku granicznym, nie tylko przy silnym mieszaniu modów. Równania diagonalnego modelu dla n są przedstawione w [Harms i Duine, równania 46–47](https://arxiv.org/html/2109.10597).

**Kierunek naprawy:** wyprowadzić tensor dynamiczny dla n i geometrii, zamiast przenosić czynnik fundamentalnego modu do innej formuły. Nazwa `no_approx` dodatkowo nie oddaje istniejących przybliżeń. **Status: błąd wzoru w określonej granicy modelu.**

### A34 — P2 — Kontrakt anizotropii i stabilnego stanu równowagi jest niejednoznaczny

**Dowód:** `mmpp/analytical/dispersion.py:80–116,259–287,718–719`; porównanie z `mmpp/analytical/fmr.py`.

`kalinikos()` dodaje `2Ku/(μ0Ms)` do wspólnego pola obu czynników, podczas gdy inne modele z Ku stosują korektę drugiej sztywności właściwą dla anizotropii prostopadłej. Dokumentacja Ku w Kalinikosie nie określa osi łatwej. Te wzory mogą opisywać różne geometrie anizotropii, ale ta sama nazwa parametru bez osi nie daje bezpiecznej wymienności przy `k→0`.

Solver orientacji kubicznej rozwiązuje warunek pierwszej pochodnej, nie sprawdzając minimum energii. Dla słabego pola, odpowiedniego ujemnego Kc1 i startu na osi symetrii może pozostać w punkcie stacjonarnym o ujemnej krzywiźnie. Późniejsze obcinanie pól/radikandu do zera może ukryć niestabilność jako zwykłą częstotliwość zero.

**Kierunek naprawy:** jawne osie i znaki stałych anizotropii, spójny Hessian energii dla obu sztywności, rozróżnienie stabilnego minimum, metastabilności i braku stosowalności liniowego modelu. Zarezerwowane `Kc2` jest obecnie udokumentowane jako nieużywane — nie traktuję tego jako ukrytego błędu. **Status: niezgodność kontraktu oraz brak kontroli stabilności.**

### A35 — P2 — CIP nie ogranicza kroku integratora zadanym `dt`

**Dowód:** `mmpp/analytical/thiele.py:1293–1477`, wywołanie `solve_ivp` około linii 1427; porównanie CPP około linii 2015–2022.

Dokumentacja CIP opisuje dt jako maksymalny krok i odstęp wyjścia, lecz wywołanie podaje siatkę `t_eval` bez `max_step`. `t_eval` określa zapis wyników, nie maksymalny krok wewnętrzny. Wąski impuls prądu/pola może zostać pominięty, nawet jeżeli wynik ma gęstą siatkę czasu. [Dokumentacja `solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) rozdziela te parametry.

CPP i `field_resolved_thiele.py:1020–1165` podają `max_step` i nie mają tego konkretnego braku. **Kierunek naprawy:** spełnić kontrakt dt; nieciągłości wymuszenia powinny ponadto wyznaczać granice odcinków integracji. **Status: pewne z wywołania API.**

### A36 — P2 — Dopasowanie siły zachowawczej nie odejmuje znanych sił napędzających

**Dowód:** `mmpp/solitons/vortex/nonlinear/nonliniearthiele.py:312–443`.

Sztywność kappa jest dopasowywana z bilansu żyroskopowego i dyssypacji, a przekazane składowe STT/Oersteda pojawiają się dopiero przy obliczeniu końcowego residuum. Jeżeli są obecne w analizowanej trajektorii, część ich działania zostaje przypisana do potencjału. Może powstać dobre dopasowanie przebiegu przy błędnej interpretacji sztywności.

**Kierunek naprawy:** przed regresją odjąć wszystkie znane składniki z tej samej wersji równania ruchu; nieznane dopasowywać wspólnie, z kontrolą identyfikowalności. Osobny `model/thiele/fit.py` oznacza stary fit kinematyczny jako deprecated/proxy i `is_physical_parameter_fit=False`; tej jawności nie należy usuwać. **Status: błąd identyfikacji dla trajektorii z niezerowymi uwzględnianymi napędami.**

## A37–A39. Autofit

### A37 — P2 — Część udostępnionych parametrów dopasowania nie wpływa na symulację

**Dowód:** `mmpp/solitons/vortex/autofit/config.py:100–121,234–254`; `autofit/simulation.py:86–280,290–328,430–462`; przykład w `autofit/interface.py` około linii 541.

Konfiguracja zawiera `phase0`, `center_x`, `center_y`; można je odblokować do fitu, a przykład wymienia `phase0`. Przygotowany kontekst symulacji ustala jednak początek i przesunięcie niezależnie od tych zmiennych; przegląd ich użyć nie pokazuje zastosowania jako zmiennych w generowaniu trajektorii. Optymalizator może więc zmieniać parametr bez zmiany przewidywanego ruchu, poza ewentualnym składnikiem regularizacji.

Własne specyfikacje mogą rozszerzać listę parametrów, lecz ścieżka przyspieszona przechowuje część materiału/geometrii w precomputowanym kontekście. Samo dopisanie np. Ms lub R do listy zmiennych nie zapewnia przeliczenia zależnych współczynników.

**Kierunek naprawy:** jawna lista wspieranych zmiennych dla danego modelu/backendu i zależności precomputacji; odrzucać parametry nieaktywne. Identyfikowalność powinna odnosić się do obserwabli, a nie wyłącznie do obecności nazwy w słowniku. **Status: pewne dla wskazanych nieużywanych parametrów; rozszerzenia wymagają kontroli backendu.**

### A38 — P2 — Skalowanie optymalizacji jest pomijane, a niepewności nie są statystycznie uzasadnione

**Dowód:** `mmpp/solitons/vortex/autofit/config.py:22,73,113–121`; `autofit/optimizers.py:70–165,193–209,234–268`.

`ParameterSpec.scale` jest deklarowane, lecz wektor optymalizacji zawiera surowe wartości fizyczne. Parametry rzędu `1e9`, 1 i `1e-9` trafiają pod wspólne tolerancje i reguły różniczkowania. To zwiększa ryzyko pozornego zbiegania lub słabej czułości na wybrane zmienne; samo w sobie nie dowodzi błędu każdego fitu.

Estymator Hessianu przycina punkty do granic, ale nadal używa symetrycznej formuły drugiej różnicy. Przy dolnej granicy i liniowej funkcji `f(x)=a*x` otrzymuje niezerową krzywiznę `4a/h`, choć prawdziwa druga pochodna wynosi zero. Następnie `1/sqrt(Hii)` jest raportowane jako niepewność parametru: pomija korelacje i nie wynika z modelu szumu dla arbitralnie ważonej funkcji strat.

**Kierunek naprawy:** optymalizacja w zmiennych bezwymiarowych, poprawne pochodne jednostronne przy granicach, pełna informacja o korelacjach; nazwę statistical uncertainty stosować dopiero przy zdefiniowanym modelu obserwacji i estymacji. **Status: pewny błąd różnic przy granicach; pozostałe punkty dotyczą wiarygodności numerycznej i statystycznej.**

### A39 — P2 — Limit ewaluacji nie obejmuje wszystkich prób i może zgubić najlepszy ostatni wynik

**Dowód:** `mmpp/solitons/vortex/autofit/single.py:298–460`; `autofit/optimizers.py:84–102`.

Nieudana symulacja zwraca karę przed zwiększeniem licznika. Wyszukiwanie seeda jawnie wyłącza zliczanie. Po udanej symulacji osiągającej limit wyjątek `_MaxEvalReached` jest podnoszony przed przekazaniem wartości do optymalizatora, więc jego własny rejestr najlepszego wyniku nie dostaje tej ostatniej oceny. Baseline korzysta z tego samego mechanizmu poza ochroną optymalizatora; mały limit, np. 1, może przerwać cały fit już tam.

**Kierunek naprawy:** jeden licznik prób solvera z rozróżnieniem success/failure/seed/diagnostics, sprawdzany przed nową próbą. Ukończony wynik należy zarejestrować przed zatrzymaniem. **Status: pewne z przepływu sterowania.**

## A40–A44. Histereza, prezentacja i połączenia modułów

### A40 — P2 — Koercja jest przypisywana po znaku pola zamiast po gałęzi pętli

**Dowód:** `mmpp/analyze/hysteresis/metrics/core.py:68–103,245–251`.

Przecięcia M=0 są dzielone na dodatnie i ujemne wartości H/B. Dla pętli przesuniętej exchange bias oba przecięcia mogą być dodatnie. Implementacja uśredni je jako `hc_plus`, pozostawi `hc_minus=NaN` i nie wyznaczy poprawnie biasu/szerokości. Fizycznie istotna jest przynależność do gałęzi rosnącej/malejącej; dopiero z dwóch pól można policzyć połowę różnicy i połowę sumy.

**Kierunek naprawy:** zachować tożsamość gałęzi oraz odróżnić pełną pętlę od minor loop. Brak wykrytego plateau nasycenia powinien również mieć jawny status, zamiast zastępować je bez kwalifikacji skrajnymi punktami. **Status: pewny kontrprzykład dla przesuniętej pętli.**

### A41 — P2 — Bootstrap pętli niszczy protokół pola, a zerowy blok powoduje nieskończoną pętlę

**Dowód:** `mmpp/analyze/hysteresis/metrics/uncertainty.py:34–80,108–120`.

Losowe bloki par `(field, magnetization)` są sklejane w nowej kolejności, po czym ponownie wyznaczane są gałęzie i całka po pętli. Skoki między odległymi blokami tworzą nowe, sztuczne odcinki i zmiany kierunku pola. Taka procedura nie zachowuje deterministycznego protokołu pomiaru; przedział dla pola powierzchni może opisywać sztuczne połączenia, zamiast niepewności obserwacji. Sam block bootstrap nie jest błędną metodą, lecz wymaga właściwego modelu resamplowanych reszt lub realizacji.

Osobny pewny błąd: brak walidacji `block_size>0`. Dla 0 lub wartości ujemnej `range(start,stop)` nie zwiększa listy indeksów, więc `while len(idx)<n_points` nie kończy się.

**Kierunek naprawy:** zachować siatkę/protokół i resamplować odpowiednie reszty w obrębie gałęzi lub całe powtórzenia; walidować dodatnie rozmiary i liczbę replik. **Status: błąd warunku zakończenia oraz nieuzasadniona interpretacja statystyczna.**

### A42 — P2 — Reprezentacja trajektorii opisuje rad/s jako GHz

**Dowód:** `mmpp/solitons/vortex/_shared/models.py:74–79,114–127`.

`instantaneous_frequency` jest pochodną fazy w rad/s. HTML mnoży jej średnią przez `1e-9` i podpisuje `mean_frequency_ghz`, bez dzielenia przez `2π`. Liczba jest zawyżona względem częstotliwości w GHz o `2π`. Dodatkowo nazwa właściwości nie odróżnia f od ω, choć docstring to robi.

**Kierunek naprawy:** oddzielne `angular_frequency` i `frequency_hz`, jedno przeliczenie w prezentacji. **Status: pewne z jednostek.**

### A43 — P2 — `spectrum.modes` nie odtwarza tej samej konfiguracji transformacji

**Dowód:** `mmpp/fft/core.py:539–553`; `mmpp/fft/spectrum/modes/bridge.py:15–49`; wykonawca w `mmpp/fft/modes/__init__.py`.

Kontekst mostka zawiera dataset, slice, preloaded data i skalę czasu, ale nie przekazuje pełnego okna, preprocessing/filterów i nfft widma. Mody są obliczane niezależnie, z własnymi domyślnymi parametrami i cache. Pik wybrany z widma po paddingu lub filtracji może wskazać najbliższy bin innej transformacji, a profil nie odpowiadać dokładnie temu samemu przygotowaniu danych.

**Kierunek naprawy:** przenosić niezmienną efektywną konfigurację analizy albo jawnie prezentować, że profil jest niezależnym przeliczeniem wraz z jego rzeczywistą częstotliwością i parametrami. Konwersja Hz/GHz w `at_peak()` nie jest tutaj wskazanym błędem. **Status: pewna niezgodność kontekstu; skutek zależy od niestandardowych opcji widma.**

### A44 — P2 — Batch widm nie utrwala kompletności, a cache nie rozpoznaje zmienionego źródła

**Dowód:** `mmpp/fft/spectrum/batch/compute.py:327–362,535–630`; `mmpp/cache/key.py:143–193`; `mmpp/fft/spectrum/batch/result.py:169` i definicja wyniku.

Batch zbiera błędy pojedynczych jobów oraz odrzucone siatki częstotliwości, lecz na końcu przekazuje do wyniku tylko udane tablice i ich ścieżki. Błędy pozostają w logach. Notebook lub zapisany wynik może więc wyglądać jak kompletny sweep, mimo brakujących punktów.

Klucz cache tego batcha identyfikuje ścieżki jobów, parametry i wycinek, bez rewizji tablic źródłowych. Przy podmienionych lub dopisanych danych i tej samej liście ścieżek odczyt może zwrócić poprzedni wynik. Kontrola kolejności zapisanych ścieżek jest obecna i przydatna, lecz nie jest kontrolą aktualności danych.

**Kierunek naprawy:** wynik powinien zawierać requested/succeeded/failed/skipped oraz powody; cache potrzebuje stabilnej tożsamości wersji źródła. Nie przenoszę tego zarzutu automatycznie na wszystkie cache MMPP: transmisja batch ma odrębne sygnatury plików. **Status: pewne z zapisanych pól i konstrukcji klucza.**

## Ocena organizacji kodu

### P3 — Odpowiedzialności są opisane katalogami, ale nie zawsze wynikają z zależności

`core/dataset.py` jest centralnym miejscem budowania widoków, lecz analizy nadal samodzielnie rozwiązują osie, czas, geometrię i backend. `fft/modes/interface.py` deleguje do dużej klasy w `fft/modes/__init__.py`, a odczyt właściwości może wykonać obliczenie i zapis. To utrudnia przewidzenie skutków nawet prostego wywołania. Podział na `analyzer`, `data_loader`, `models`, `visualization` nie zapewnia jeszcze jednokierunkowej zależności.

W solitonach część ścieżek `vortex/core`, `numerical/events`, `numerical/modes` i `numerical/nonlinear` jest warstwą re-eksportów. Same aliasy są rozsądnym mechanizmem zgodności API i nie należy liczyć ich jako niezależnych implementacji fizyki. Problemem jest brak jednej oczywistej lokalizacji kanonicznej: użytkownik i maintainer muszą znać historię przenosin. Plik o nazwie `nonliniearthiele.py` oraz jego wrapper o poprawionej pisowni dodatkowo utrwalają to rozproszenie.

Istnieją również rzeczywiście równoległe implementacje: cache w aktywnym interfejsie dyspersji i pomocniczy `dispersion/_interface/cache.py`, kilka mechanizmów widma/okien/metryk, duży `fft/old_modes.py`. Nie należy automatycznie przypisywać znalezionego błędu w nieużywanym helperze bieżącej ścieżce. W tym raporcie np. nie potraktowano alternatywnego cache dyspersji jako dowodu błędu loadera używanego przez aktualny interfejs.

**Zalecany kierunek:** najpierw wskazać właściciela każdego kontraktu i faktyczną ścieżkę wywołań, potem redukować duplikaty. Publiczne aliasy mogą pozostać jako cienkie, dobrze opisane moduły kompatybilności.

### P3 — Warstwa numeryczna zależy od prezentacji i szczegółów innego modułu

`vortex/autofit/simulation.py` korzysta z pomocników z modułu `plotting.py`; wspólny helper spektralny sięga do backendu FFT ulokowanego wewnątrz dyspersji. To odwraca pożądany kierunek zależności: solver i podstawowe FFT nie powinny potrzebować wiedzy o notebookowym wykresie ani podpakiecie konkretnej analizy.

Obiekty wyników mogą mieć `.plt` i `_repr_html_` zachowujące ergonomię notebooka, ale import i wykonanie modelu powinny być możliwe bez aktywacji warstwy prezentacji. Wspólne przeliczenia należy przenieść do małych modułów o jawnych wejściach i jednostkach. To nie jest zalecenie usunięcia fluent API ani lazy imports.

### P3 — Semantyka cache wymaga wspólnego protokołu

MMPP korzysta z wyników pamięciowych, Zarr, pickle/plików batch i lokalnych cache accessorów. `force`, `save`, `use_cache` nie zawsze znaczą to samo. Przykładowo część interfejsów opisuje `use_cache` jako wyłącznie cache pamięciowy; nie należy z samej nazwy wnioskować, że wyłącza wszystkie odczyty dyskowe. Tę różnicę trzeba jednak wyraźnie dokumentować i ograniczyć liczbę wariantów.

Wspólny kontrakt powinien zawierać:

1. Tożsamość i wersję źródła, dataset, efektywny wycinek, materializację, czas i geometrię.
2. Efektywną konfigurację numeryczną, w tym okno, normalizację, maskę, konwencje oraz wersję algorytmu/schematu.
3. Rozdzielone operacje odczytu, obliczenia i publikacji wpisu; `save=False` bez zapisu/kasowania.
4. Oznaczenie ukończenia wpisu i wymianę dopiero po udanym zapisie; opis postępowania przy równoległych writerach.
5. Aktualizację lokalnego cache po wymuszonym obliczeniu oraz status trafienia cache w wyniku.

Nie trzeba hashować całej wieloterabajtowej tablicy przy każdym wywołaniu. Źródło może dostarczać stabilny revision/generation ID; istotne jest, aby ścieżka pliku nie udawała wersji jego zawartości.

### P3 — Fallbacki mieszają brak informacji z poprawnym wynikiem

Fallback wizualizacji albo brak optional dependency może uzasadniać łagodny komunikat. W warstwie fizycznej zamiana na domyślne `dt`, spacing, polaryzację, temperaturę lub materiał oznacza już zmianę problemu naukowego. Potrzebny jest wspólny model statusu: **computed**, **estimated**, **assumed**, **unavailable**, **invalid**. Założenia powinny być widoczne przy wyniku, a nie wyłącznie w logu.

Dotyczy to także tolerancji braków w batch. Odrzucenie niezgodnej siatki jest lepsze od cichego łączenia, lecz trzeba zachować informację, że zestaw wyników jest niepełny.

### P3 — Repozytorium zawiera śledzone kopie `.orig` i `.backup`

`git ls-files '*.orig' '*.backup'` potwierdza śledzone kopie wielu aktywnych modułów, m.in. FFT, Thiele, autofitu i histerezy. W audycie fizyki analizowano aktywne pliki `.py`, nie te kopie. Ich obecność utrudnia wyszukiwanie implementacji kanonicznej i może mylić późniejsze przeglądy lub generowanie dokumentacji.

**Zalecenie:** ustalić przeznaczenie tych artefaktów i po przeglądzie historii usunąć niepotrzebne kopie w osobnej zmianie. Nie usuwać ich w ramach przypadkowego refaktoringu ani bez rozpoznania pochodzenia.

### P3 — Deklaracje wspieranych wersji i kontroli jakości wymagają uzgodnienia

Przekazane wytyczne mówią o Pythonie 3.9+, natomiast `pyproject.toml:15` deklaruje `>=3.10`, a kod używa m.in. `zip(..., strict=False)` dostępnego od 3.10. To rozbieżność dokumentacji kontraktu; nie jest dowodem, że bieżący pakiet obiecuje w metadanych obsługę 3.9.

Blok `ignore_errors=true` w `pyproject.toml` obejmuje także ważne części obliczeń i cache. Samo uruchomienie mypy w przyszłości nie obejmie ich z taką samą dokładnością jak reszty. W pierwszej kolejności warto doprecyzować typy protokołu danych i delegacji API — błąd A01 pokazuje znaczenie tej granicy. W tym audycie nie uruchamiano mypy ani żadnej innej kontroli wykonawczej.

## Ocena metod i warunków fizycznej stosowalności

| Metoda | Co może wiarygodnie opisywać | Czego nie należy zakładać automatycznie |
| --- | --- | --- |
| FFT po średnim sygnale, metoda koherentna | odpowiedź przestrzennie uśrednionej amplitudy z zachowaniem faz | brak piku nie wyklucza modu antysymetrycznego, który znosi się w średniej |
| Średnia z lokalnych mocy FFT | intensywność lokalnych oscylacji | pierwiastek z uśrednionej mocy nie odzyskuje jednej fazy ani zespolonego pola |
| Welch/periodogram | statystyka widma przy znanej osi czasu, jednostkach i oknie | szerokość piku nie zawsze jest własną linią fizyczną; padding nie poprawia rozdzielczości informacji |
| Resampling nieregularnego czasu | transformacja po jawnej interpolacji | zachowanie amplitud i faz blisko Nyquista; metadane muszą mówić o resamplingu |
| Profile FFT na wybranym f | przestrzenna odpowiedź wzbudzona w danym paśmie | kompletny zestaw modów własnych, ortogonalność ani jednoznaczność przy nakładających się rezonansach |
| Dyspersja z FFT przestrzenno-czasowej | rozkład wzbudzonej odpowiedzi w f i k | pełne pasma własne i ich luki przy arbitralnym źródle wzbudzenia |
| Transmisja `|m|²/reference` | względna intensywność konkretnego obserwowanego sygnału | współczynnik transmisji energii między różnymi materiałami/geometriami bez normalizacji strumienia |
| Tracking maksimum/centroidu/fitu rdzenia | pozycja pojedynczej rozpoznawalnej struktury przy odpowiedniej masce i rozdzielczości | działanie po anihilacji, przy wielu rdzeniach, przełączeniu polaryzacji lub kontakcie z brzegiem |
| Q finite-difference | dyskretny przybliżony całkowy ładunek z ciągłego pola | dokładna kwantyzacja przy brzegu, dziurach maski, niedorozdzielonym rdzeniu lub nieznormalizowanym m |
| Q Berg–Lüscher | ładunek z orientowanych pól na triangulacji | odporność na dowolne zdegenerowane/antypodalne trójkąty i niejednoznaczną orientację siatki |
| Rozmiar skyrmionu z radialnego profilu | promień względem przyjętej definicji i pojedynczego centrum | jeden uniwersalny promień zdeformowanej, brzegowej lub wielordzeniowej tekstury |
| Thiele CIP/CPP | model zredukowanych współrzędnych w zakresie przyjętego ansatzu i kalibracji | pełne przełączenie rdzenia, anihilacja, mody wewnętrzne i backaction poza modelem |
| Field-resolved i nonlinear STNO | przewidywania konkretnego modelu fenomenologicznego i jego parametrów | automatyczna kwalifikacja wobec dowolnej symulacji micromagnetycznej |
| Dopasowanie do trajektorii | zgodność wybranych obserwabli z parametrami modelu | jednoznaczność wszystkich parametrów materiałowych lub ich niepewność statystyczna z samej wartości loss |
| Histereza/metryki | wielkości dla poprawnie rozpoznanych gałęzi i jednostek | pełna pętla, saturacja lub niezależny szum tylko dlatego, że istnieje tablica M(H) |

Szczególnej ostrożności wymaga nazwa `damon_eshbach()` w `analytical/dispersion.py`, która prowadzi do przybliżenia fundamentalnego profilu przez grubość. Nie powinna być interpretowana jako pełne rozwiązanie powierzchniowe dla dowolnej grubości i kd. Ograniczenia diagonalnej teorii i sprzężenia modów omawia [Harms i Duine](https://arxiv.org/html/2109.10597). To kwalifikacja zakresu modelu, oddzielna od jednoznacznego błędu granicy PSSW w A33.

## Elementy implementacji, które warto zachować

- **Centralne skalowanie FFT:** `mmpp/fft/_scaling.py` rozdziela korektę amplitudy przez coherent gain i gęstość przez energię okna, z uwzględnieniem jednostronności/DC/Nyquista. Problem A06 dotyczy innej implementacji; należy konsolidować kontrakt wokół poprawnego rdzenia.
- **Jawność zespolonego wyniku:** podstawowe FFT i część obiektów wyników rozróżniają widmo zespolone od wielkości wynikającej z uśrednienia mocy i nie powinny fabrykować fazy tej drugiej.
- **Walidacja i opis resamplingu:** wspólne przygotowanie czasu sprawdza porządek i nierównomierność, ma tryb ścisły i ostrzega o konsekwencjach interpolacji. Trzeba zapewnić, aby wszystkie accessory rzeczywiście przekazywały mu pełną oś czasu.
- **Maska materiału dla modów:** `fft/modes/material_mask.py` korzysta z geometrii albo aktywności źródłowej magnetyzacji, a nie z amplitudy pojedynczego modu. Dzięki temu węzeł modu nie staje się sztucznie próżnią.
- **Skyrmiony:** implementacja ma mechanizmy kontroli maski, niepoprawnych wektorów, sąsiedztwa stencila, różnic między centrami i bliskości brzegu. Są wartościową bazą dla wyników z jakością/niepewnością, mimo problemu przekazania widoku i orientacji.
- **Przepływ batch widm:** kolejność wyników jest przywracana według kolejności wejścia, a nie czasu zakończenia wątku; niezgodne siatki są wykrywane. Brakuje utrwalenia tej diagnostyki w obiekcie wyniku, nie samego wykrywania.
- **Jawna kwalifikacja części modeli:** stary fit Thielego ma oznaczenie proxy; nonlinear STNO komunikuje eksperymentalny/niekalibrowany charakter. To właściwszy kontrakt niż sugerowanie ogólnej walidacji przez samą nazwę modelu.
- **Konwencje SI w części Thielego:** są jawne rozróżnienia `gamma` i `gamma0`, kontrole dodatniości materiału/geometrii oraz poprawnie przekazywany `max_step` w CPP/field-resolved. Nie znaleziono podstaw do uznania całej rodziny za jednostkowo niespójną.
- **Import eksperymentalnej transmisji:** odwracanie częstotliwości odwraca także wiersze danych, a konwersja jednostek i zgodność długości są jawnie kontrolowane. Nie należy odwracać samej osi podczas przyszłych porządków UI.

## Docelowe kontrakty, które usuną klasy błędów

### Wejście analizy

Jeden niezmienny kontekst powinien przenosić: źródło/rewizję, dataset, dane lub loader, semantyczne nazwy osi, rzeczywisty wektor czasu w sekundach, współrzędne i rozmiar komórek w metrach, maskę materiału, orientację bazy, składowe magnetyzacji i jej jednostkę oraz historię crop/downsample/resample. Każda analiza ma konsumować ten kontekst, zamiast odtwarzać go z globalnych atrybutów joba.

Kontekst ma działać tak samo dla Zarr, H5 i danych materializowanych. Zachowanie lazy loading jest możliwe: nie wymaga to kopiowania całej tablicy, lecz przenoszenia kompletnego opisu jej odczytu.

### Wynik naukowy

Wynik powinien rozdzielać:

- surową wielkość liczbową, jednostkę, osie i konwencję transformacji;
- definicję obserwabli: amplituda, moc, PSD, zespolone CSD, promień konkretnej definicji;
- efektywną konfigurację i użyty backend, także fallback;
- założenia, kryteria jakości, status dostępności i ewentualny powód odrzucenia;
- dane prezentacyjne, np. logarytm koloru, normalizacja do maksimum wykresu i wygładzanie do wyświetlania.

W szczególności częstotliwość f w Hz i ω w rad/s oraz k w rad/m i częstotliwość przestrzenna w cyklach/m powinny mieć odrębne nazwy. Q, winding, chirality, polarity i CW/CCW wymagają wspólnej tabeli znaków. API powinno wyjaśniać, czy pozycja jest liczona od brzegu widoku, środka próbki czy środka orbity.

### Modele i wnioskowanie

Parametr dopasowania powinien mieć jednostkę, skalę optymalizacyjną, zakres, informację o aktywności w solverze i zależnościach od innych parametrów. Parametr fizyczny, fenomenologiczny współczynnik oraz czysto numeryczny gain nie powinny otrzymywać identycznej interpretacji tylko dlatego, że wszystkie są liczbami typu float.

Detektor powinien zwracać obserwację i podstawę klasyfikacji. „Brak zaobserwowanego rdzenia” nie jest równoważne „anihilacja”, „mała intensywność” — „luka”, a „dopasowana trajektoria” — „zidentyfikowane parametry materiałowe”. Stan unknown jest użyteczną informacją naukową.

## Zalecana kolejność prac

1. **Usunąć jednoznaczne awarie i ciche podmiany:** A01, A08, A10, A11, A16; zabezpieczyć dotychczasowe cache przed kasowaniem podczas obliczeń bez zapisu.
2. **Ujednolicić dane wejściowe:** A02–A05 oraz synchronizacja tabel. To warunek sensownej naprawy częstotliwości, geometrii i dopasowań.
3. **Ujednolicić transformacje i jednostki:** A06–A07, A09, A14–A15, A17, A23–A25, A42–A43. Uzgodnić jawny kontrakt znaków przed migracją historycznych wyników.
4. **Naprawić błędy wzorów i odseparować placeholdery:** A20–A22, A33–A36. Zmiany wzorów powinny mieć opis przyjętych założeń fizycznych, nie tylko opis refaktoringu.
5. **Ograniczyć nieuzasadnione wnioskowanie:** A18–A19, A26–A32, A37–A41; uczciwie reprezentować brak identyfikacji, jakość i niepewność.
6. **Dopiero na ustalonych kontraktach porządkować strukturę:** kanoniczne moduły, cienkie kompatybilne accessory, wspólny cache i trwała kompletność batch z A44.

Ta kolejność opisuje priorytety w momencie audytu. Po jego zakończeniu rozpoczęto osobny etap napraw; jego wynik jest zapisany poniżej.

## Kryteria oceny przyszłych poprawek, wyprowadzone statycznie

Poniższe punkty nie są uruchomionymi testami. To wymagania wynikające z definicji metod i opisanych kontrprzykładów:

- Zmiana interfejsu odczytu tego samego widoku nie zmienia danych, czasu ani geometrii wejściowej.
- Odświeżenie cache daje wynik nowej rewizji także przy następnym zwykłym odczycie; wspólny katalog nie miesza jobów; obliczenie bez zapisu nie kasuje cache.
- Całka PSD ma zgodne jednostki i normalizację, a zmiana backendu nie odcina jednego kierunku obiegu sygnału zespolonego.
- Operacja odwrócenia k jest permutacją także dla parzystej siatki; rekonstrukcja zachowuje niesymetryczny profil i jego położenie.
- Globalna faza amplitudy zespolonej nie zmienia identyfikacji tego samego przestrzennego modu.
- Wyniki dodatkowe transmisji mają ten sam kontrakt przy serial, parallel i vectorized; brak implementacji jest jawny.
- Wyższy in-plane PSSW przy `k_parallel=0`, `Ku=0` zachowuje oba pola sztywności dynamicznej.
- Minimum radialnego PMF nie staje się automatycznie defektem pinningu; klasyfikacja G/C wymaga informacji o teksturze.
- Każdy deklarowany aktywny parametr fitu wpływa na przewidywane obserwable w zadanym zakresie; niepewność określa konkretny model statystyczny.
- Wynik batch utrwala listę wejść, braków i odrzuceń, tak aby zapisany sweep nie sugerował nieistniejącej kompletności.

## Granica dowodowa i wynik pracy

Raport zawiera **44 ponumerowane ustalenia**, ocenę architektury, warunków fizycznej stosowalności i kolejność napraw. Część ustaleń łączy kilka blisko powiązanych objawów jednej granicy kontraktu; liczba nie jest liczbą niezależnie odtworzonych awarii.

Pierwotny audyt był statyczny: **nie uruchamiał testów, importu/obliczeń MMPP, notebooków, linterów, mypy, buildu ani benchmarku i nie zmieniał implementacji**. Późniejszy etap napraw oraz jego zakres weryfikacji opisano w poniższej aktualizacji. Nadal nie ma tu dowodu dla konkretnego zestawu danych produkcyjnych, GPU ani całej ścieżki end-to-end w środowisku użytkownika.

## Aktualizacja: stan napraw po audycie

Data aktualizacji: 2026-10-07. Zmieniono implementację dla A01–A44, dodano regresje dla kontraktów wejścia i czasu, cache, transformacji, fizyki, stanów nieznanych, budżetu autofit i kompletności batch. Najważniejsze skutki:

- **A01–A05:** przekazanie datasetu i materializowanego widoku do analiz; czas i geometria są wyznaczane dla bieżącego widoku, redukcja uwzględnia przycięty zakres, a składanie odwróconych slice zachowuje ich semantykę.
- **A06–A13:** ujednolicono jednostki/normalizację PSD oraz zachowanie sygnałów zespolonych i krótkich; doprecyzowano tożsamość i odświeżanie cache; transmisja nie zwraca nieuzupełnionych tablic jako wyniku; CPSD zachowuje fazę, a callback postępu nie przerywa publikacji gotowego wpisu.
- **A14–A24:** poprawiono permutację osi i rekonstrukcję profilu, zachowano jawny wybór datasetu i adapter H5, opisano miarę osi k, zmieniono łączenie/detekcję gałęzi i ograniczono interpretację map wzbudzenia; analiza elektromagnetyczna, Q, siatka dalekiego pola, winding i warunki aktywacji filtrów mają jawniejsze kontrakty.
- **A25–A32:** ujednolicono orientację topologii; nieznana polaryzacja, niedostępny stan i niewykryta stacjonarność nie są zamieniane w pewny wynik; detektory, radialny PMF, magnetorezystancja, amplituda i szerokość piku komunikują granice wnioskowania.
- **A33–A39:** poprawiono granicę wyższego in-plane PSSW i rozdzielono osie anizotropii; równowaga kubiczna wybiera stabilne minimum; CIP ogranicza maksymalny krok, bilans sił uwzględnia znane napędy, autofit odrzuca nieobsługiwane parametry, pracuje w zmiennych skalowanych, a krzywizna przy granicy używa różnic jednostronnych. Baseline, seedy i optymalizacja współdzielą limit prób solvera, a porównanie końcowe wykorzystuje zapisaną najlepszą trajektorię bez dodatkowego wywołania solvera.
- **A40–A44:** koercję wyznacza się z gałęzi pętli, bootstrap zachowuje protokół pomiaru i waliduje wejście, częstotliwość rozróżnia Hz od rad/s, mostek modów raportuje faktyczną konfigurację transformacji, a batch zapisuje sukcesy, braki i sygnatury źródeł.
- **P3:** helpery numeryczne trajektorii i backend FFT przeniesiono do neutralnych modułów; usunięto historyczne kopie `.orig`/`.backup` po porównaniu z aktywnymi plikami; deklaracje Pythona, macierz CI i wybrane reguły mypy uzgodniono. Cache naprawiono na wskazanych ścieżkach, ale jego implementacje nie zostały scalone w jedną globalną klasę — pozostają oddzielone od siebie typem magazynu i publicznym kontraktem.

Weryfikacja napraw: `python -m ruff check mmpp tests`, pełne `python -m mypy mmpp/` (411 modułów) oraz śledzone testy z `tests/` zakończyły się powodzeniem. Zestaw testów obejmował 44 śledzone pliki `test_*.py`; wyłączono z niego `tests/test_swap_e2e.py`, bo jest odrębnym testem integracyjnym swap, poza zmienionymi ścieżkami MMPP. Pełne `pytest tests/` nie jest lokalnie samowystarczalne: cztery ignorowane przez Git, ręczne testy danych (`test_overlay_alpha_05.py`, `test_overlay_corrected.py`, `test_overlay_real_data.py`, `test_spectrum_integration.py`) próbują podczas importu otworzyć niedostępny dataset `m_layer13`. Po ich pominięciu szeroki przebieg zawierał także stare, ignorowane testy ręczne i testy swap; dlatego wynik całego katalogu nie jest przedstawiany jako zielony. Wykonano `git diff --check`. Nie uruchamiano buildu pakietu, GPU ani danych produkcyjnych; lokalne testy nie kwalifikują fizycznie całego MMPP.

Pozostają świadome granice organizacyjne: implementacje cache nadal używają różnych magazynów, a jednolity typ statusu (`computed`/`estimated`/`assumed`/`unavailable`/`invalid`) nie został wprowadzony globalnie. Naprawiono błędne semantyki konkretnych ścieżek z A08–A10 i A44 oraz przeniesiono dwa helpery przez granicę warstw, ale pełne rozłożenie historycznej fasady `fft/modes` i migracja wszystkich publicznych wyników do jednego statusu/protokołu cache wymagałaby osobnej, kompatybilnej zmiany API. To ograniczenia zakresu tej naprawy, nie potwierdzenie, że każda warstwa ma już zunifikowaną architekturę.

Pozycje opisane jako ograniczenia modeli wymagają ustalenia zamierzonego kontraktu i zakresu zastosowań, zanim zostaną potraktowane jako zmiany wzorów. Jednoznaczne błędy sygnatur, indeksowania, normalizacji i kluczy cache nie zależą natomiast od wyniku istniejącego zestawu testów.
