# Audyt MMPP: dane, FFT, mody, dyspersja, transmission i interaktywne API

**Data rozpoczęcia:** 2026-07-21  
**Status:** audyt aktywny — dokument jest aktualizowany wraz z dalszą analizą  
**Repozytorium:** `containers_admin2/postprocessing/mmpp`

## 1. Cel i zakres

Celem audytu jest sprawdzenie spójności logicznej i programistycznej biblioteki
MMPP, ze szczególnym uwzględnieniem:

- kontraktu widoków `DatasetAwareWrapper` i propagacji wyboru danych,
- ładowania danych z Zarr/H5 i danych zmaterializowanych,
- FFT, osi częstotliwości, skalowania i cache,
- filtrów pre-FFT, post-FFT i filtrów interaktywnych,
- modów FMR, fazy, komponentów kartezjańskich i kołowych,
- dyspersji, rekonstrukcji profili i osi przestrzennych,
- transmission i fizycznych współrzędnych okien,
- helperów notebookowych, widgetów i publicznego fluent API,
- wydajności, pamięci, opcjonalnych zależności i walidacji wydania.

Audyt nie uznaje obecności kodu ani pojedynczego zielonego testu za dowód pełnej
poprawności fizycznej. Rozróżniane są: implementacja, wykonanie testowe oraz
walidacja fizyczna na danych referencyjnych.

## 2. Oczekiwany kontrakt widoku danych

Dla kodu:

```python
view = job[0].m[:, 1, :50, ..., 2]
analysis = view.fft
```

każda dalsza operacja musi widzieć wyłącznie zaznaczony zakres. Dotyczy to
`spectrum`, `modes`, `dispersion`, `transmission`, filtrów, cache i helperów
interaktywnych. Późniejsza metoda nie może ponownie otworzyć pełnego `m` i
utracić ograniczeń `t/z/y/x/component`.

## 3. Potwierdzone błędy i naprawy

### 3.1. Dane zmaterializowane były ignorowane przez główny silnik FFT

**Waga:** krytyczna  
**Status:** naprawione, test regresyjny dodany

`DatasetSpecificFFT.spectrum` przekazywał `preloaded_data`, ale
`FFTCompute.calculate_fft_data()` nie przekazywał ich do profilowanego loadera.
W rezultacie fancy indexing lub `downsample()` mogły zakończyć się ponownym
wczytaniem pełnego datasetu ze storage.

Naprawa:

- jawne przekazanie `preloaded_data` przez warstwy spectrum i compute,
- użycie danych zmaterializowanych bez ponownego odczytu wartości ze Zarr,
- lokalna obsługa `tmax` i `z_layer` dla zmaterializowanego widoku.

### 3.2. Cache FFT nie rozróżniał zawartości widoków zmaterializowanych

**Waga:** wysoka  
**Status:** naprawione, test regresyjny dodany

Dla danych bez reprezentowalnego `slice_info` potrzebna jest tożsamość zależna
od zawartości. Dodano stabilny fingerprint obejmujący dtype, shape i skrót
Blake2b bajtów tablicy. Zapobiega to zwracaniu wyniku innego widoku o zgodnym
kształcie.

### 3.3. Stride czasu nie zmieniał efektywnego `dt`

**Waga:** krytyczna numerycznie  
**Status:** naprawione, test regresyjny dodany

Dla `m[::2, ...]` oś częstotliwości była liczona z pierwotnym `dt`, mimo że
rzeczywisty interwał próbkowania wzrasta dwukrotnie. Loader mnoży teraz `dt`
przez dodatni krok pierwszej osi. Niedodatni krok czasu jest odrzucany.

### 3.4. Mody ignorowały pełny ROI i współdzieliły globalny cache

**Waga:** krytyczna  
**Status:** naprawione w kodzie; pełny test runtime wymaga środowiska z matplotlib

Wcześniej `compute_modes()` stosował tylko osobne `t_slice` i `z_slice`.
Ograniczenia `y`, `x` i komponentu nie wpływały na tablicę modów. Dodatkowo
cache `modes/{dataset}` był wspólny dla wszystkich widoków.

Naprawa:

- `FMRModeAnalyzer` otrzymuje pełny `view_slice` lub `preloaded_data`,
- FFT modów jest liczona z pełnego widoku `t/z/y/x/component`,
- każdy widok ma odrębny identyfikator i grupę
`modes/{dataset}/views/{view_id}`,
- metadane grupy zapisują widok oraz tożsamość komponentu,
- `force=True` dla widoku nie usuwa globalnego cache FFT datasetu.

### 3.5. Wybrany `mx` albo `my` był interpretowany jako `mz`

**Waga:** krytyczna fizycznie  
**Status:** naprawione w kodzie; test integracyjny oczekuje kompletnego środowiska

Renderer traktował każdą tablicę modów z jednym komponentem jako `mz`. Dla
`m[..., 0]` i `m[..., 1]` prowadziło to do błędnej etykiety lub zerowego obrazu.
Cache przechowuje teraz `component_index`, a odczyt modów przywraca sygnał do
właściwego slotu trójskładnikowej reprezentacji kartezjańskiej.

### 3.6. Parametry filtrów pre-FFT były ignorowane

**Waga:** wysoka  
**Status:** naprawione, test regresyjny dodany

Centralny `FilterPipeline` zachowywał konfigurację parametrów, lecz wywoływał
filtr tylko po nazwie. Przykładowo `cutoff_fraction=0.125`, niestandardowe pasmo,
`polyorder` i rząd pochodnej nie docierały do implementacji.

Naprawa przekazuje opcje do:

- `high_pass`,
- `band_pass`,
- `savgol_smooth`,
- `baseline_correction`,
- `spectral_derivative`.

### 3.7. Transmission traciło początek i stride osi `x`

**Waga:** wysoka  
**Status:** naprawione dla prostych, skomponowanych slice

Po wyborze `x_start:x_stop:x_step` pozycje okien rozpoczynały się ponownie od
zera i używały pierwotnego `dx`. Obecnie `x_positions` zachowuje przesunięcie
widoku, a efektywne `dx` uwzględnia stride.

### 3.8. Dyspersja używała błędnego mapowania osi dla danych 4D

**Waga:** wysoka  
**Status:** naprawione

Stałe mapowanie `(dz,dy,dx) -> (1,2,3)` jest prawidłowe dla
`(t,z,y,x,c)`, ale nie dla `(t,y,x,c)`. Dodano mapowanie zależne od liczby
wymiarów.

### 3.9. Rekonstrukcja profilu dyspersji była odwracana dwukrotnie

**Waga:** krytyczna fizycznie  
**Status:** naprawione; pokryte istniejącymi testami fazy i IFFT

Po `ifftshift` i IFFT kod ponownie odwracał profil, jeśli `flipx=True`, mimo że
`S_complex` było już zapisane w publicznej konwencji osi `k`. Powodowało to
odwrócenie i przesunięcie fazy przestrzennej. Usunięto drugą korektę.

### 3.10. Regresje interaktywnego panelu dyspersji

**Waga:** średnia  
**Status:** naprawione

Przywrócono publiczny kontrakt:

- etykietę `Render / refresh dispersion`,
- kolejność zakładek z `Modes` pod indeksem 3,
- opisy odświeżania eksportu i podsumowania,
- komunikat lekkiego widoku HTML,
- bezpieczny eksport JSON zachowujący czytelne cudzysłowy i escapujący HTML.

### 3.11. Downsampling czasu tracił efektywny krok próbkowania

**Waga:** krytyczna numerycznie  
**Status:** naprawione, test loadera dodany

`downsample()` materializuje tablicę i usuwa `slice_info`. Oznaczało to, że po
blokowej redukcji osi czasu loader widział ponownie źródłowe `dt`. Dodano jawny
`time_step_scale`, propagowany przez spectrum, modes, dispersion i transmission.
Mnożnik jest również częścią tożsamości cache. Usunięto przy okazji podwójne
ostrzeżenie o materializacji dużej tablicy.

### 3.12. Cache transmission i dispersion nie identyfikował materializowanego widoku

**Waga:** krytyczna  
**Status:** naprawione, test klucza transmission dodany

Dwa różne widoki fancy/downsample miały `slice_info=None`, więc mogły użyć tego
samego wpisu cache. Klucze obejmują teraz fingerprint danych, efektywny krok
czasu oraz — dla dyspersji — geometrię widoku. Dotyczy to cache pamięciowego,
Zarr i zewnętrznego cache dyspersji.

### 3.13. Geometria ROI nie była wspólną częścią kontekstu analiz

**Waga:** wysoka fizycznie  
**Status:** naprawione w głównych ścieżkach; dalsze testy integracyjne trwają

Sam `slice_info` nie wystarcza dla downsamplingu i wyborów fizycznych. Istniejący
`DatasetGeometry` jest obecnie przekazywany do modes, dispersion i transmission.
Zapewnia efektywne `dx/dy/dz`, fizyczny początek osi oraz extent mapy modu.
Dyspersja nie mnoży ponownie spacingu, jeśli geometria już reprezentuje stride.

### 3.14. Niejednoznaczny layout 3D w transmission był zgadywany

**Waga:** krytyczna dla nietypowych datasetów  
**Status:** naprawione przez geometrię i fail-closed

Tablica `(t,y,x)` mogła zostać potraktowana jak `(t,y,component)`, szczególnie
gdy `x` miało 1–3 komórki. Transmission używa teraz geometrii do rozpoznania
skalarnej płaszczyzny. Jeśli layoutu nie da się rozstrzygnąć, zgłaszany jest
czytelny `ValueError` zamiast arbitralnego przypisania osi.

### 3.15. Granice filtrów były liczone w złej skali Nyquista

**Waga:** wysoka numerycznie  
**Status:** naprawione, test sinusoidalny i testy walidacji dodane

`high_fraction=0.9` oznacza ułamek Nyquista, ale implementacja porównywała tę
wartość z `rfftfreq`, którego maksimum wynosi 0.5. Oś filtra jest obecnie
normalizowana do `[0,1]` względem Nyquista. Nieprawidłowe granice są odrzucane
zamiast cicho korygowane.

### 3.16. Batch spectrum mieszał różne osie częstotliwości

**Waga:** krytyczna numerycznie  
**Status:** naprawione przez kontrolę wspólnej siatki

`BatchSpectrumResult` przechowuje jedną oś częstotliwości, ale wcześniej była to
po prostu oś pierwszego zakończonego joba. Kolejne widma z innym `dt`, `nfft`
lub rozmiarem mogły zostać do niej przypisane. Batch odrzuca obecnie niezgodny
wpis z wyjaśnieniem; dla heterogenicznych siatek właściwym API jest `overlay()`.

### 3.17. Równoległy batch miał niedeterministyczną kolejność wyników

**Waga:** wysoka programistycznie  
**Status:** naprawione

Wyniki `ThreadPoolExecutor` były dopisywane według kolejności zakończenia, przez
co `spectra`, `job_paths` i parametry zmieniały kolejność między uruchomieniami.
Każdy wynik zachowuje teraz indeks wejściowy, a kolekcje są stabilnie sortowane
przed utworzeniem rezultatu.

### 3.18. Nieznany backend FFT cicho uruchamiał NumPy i fałszował metadane

**Waga:** wysoka dla reprodukowalności  
**Status:** naprawione, testy błędnej nazwy i brakującej zależności dodane

Literówka taka jak `engine="numpyy"` trafiała do końcowego fallbacku NumPy, ale
wynik raportował żądaną nazwę jako wybrany backend. Obecnie dozwolone są tylko
`auto`, `numpy`, `scipy` i `pyfftw`; jawnie wybrany, niedostępny backend zgłasza
`ImportError`, a nie zmienia implementację bez wiedzy użytkownika.

### 3.19. Kontrola analityczna skalowania FFT

**Waga:** dowód numeryczny  
**Status:** sprawdzone dla tonu bez okna, parzystego i nieparzystego N

Dla syntetycznego cosinusa potwierdzono:

- skalowanie `amplitude` odtwarza zadaną amplitudę tonu,
- `power` daje średnią moc tonu `A²/2`,
- całka jednostronnego PSD po częstotliwości zgadza się ze średnim kwadratem
  sygnału (Parseval).

Pozostaje rozszerzenie dowodu na wszystkie okna i obie metody agregacji.

### 3.20. Fluent `fft.filters(...).spectrum()` ponownie gubił parametry pre-filtrów

**Waga:** wysoka; publiczne API notebookowe  
**Status:** naprawione, test publicznego chaina dodany

Centralny pipeline przyjmował już opcje filtrów, ale `SpectrumFilterChain`
zamieniał słownik pre-filtrów na samą listę nazw. Rozszerzono `filter_type` o
uporządkowaną mapę `nazwa -> parametry`, dzięki czemu ustawienia high-pass,
band-pass, Savitzky–Golaya, baseline i pochodnej docierają także przez fluent API.

### 3.21. `save=True` ukrywało błąd zapisu FFT

**Waga:** krytyczna kontraktowo  
**Status:** naprawione, test wymuszonego błędu zapisu dodany

Po poprawnym obliczeniu wyjątek Zarr był przechwytywany jako ostrzeżenie, a
metoda zwracała zwykły rezultat. Użytkownik mógł więc założyć, że trwały cache
istnieje. Niepowodzenie `save=True` zgłasza teraz `RuntimeError` z pierwotnym
wyjątkiem. Istniejący wpis bez `force=True` zgłasza `FileExistsError`, zamiast
kończyć metodę i pozwalać wyższej warstwie zalogować fałszywy sukces.

### 3.22. Brak wymaganych metadanych cache był uznawany za zgodność

**Waga:** krytyczna dla poprawności cache  
**Status:** naprawione, test starego wpisu bez identyfikatora slice dodany

Weryfikator porównywał `z_layer`, `source_dataset` i `slice_identifier` tylko
wtedy, gdy klucz istniał również w zapisanym wyniku. Brak atrybutu w starym lub
niepełnym cache przechodził jako match. Obecnie każde wymagane przez żądanie,
nieobecne metadane unieważniają wpis i wymuszają ponowne obliczenie.

### 3.23. Dyspersja raportowała standardowe pre-filtry, których nie wykonywała

**Waga:** krytyczna dla wiarygodności analizy  
**Status:** naprawione, test zmiany sygnału przez high-pass dodany

`apply_filter_pipeline()` obsługiwało własne filtry zaawansowane, lecz ignorowało
standardowe `high_pass`, `band_pass`, `savgol_smooth`, `baseline_correction`,
`detrend_linear`, `remove_mean` i `spectral_derivative`. Nazwy pozostawały w
`active_pre`, notatkach i cache, co sugerowało wykonanie. Filtry są teraz
rejestrowane także jako root-level pre-filters i wykonywane wspólnym backendem
z zachowaniem parametrów oraz wskazanej osi czasu.

### 3.24. Pochodna spektralna była liczona względem indeksu próbki zamiast czasu

**Waga:** wysoka dla jednostek fizycznych i porównywalności wyników  
**Status:** naprawione, dwa testy analityczne dodane

`spectral_derivative` używało `np.gradient(..., axis=0)` bez odstępu próbkowania.
Wartość pochodnej zależała przez to od liczby próbek i downsamplingu, a nie tylko
od fizycznego przebiegu. Wspólny filtr przyjmuje teraz dodatni, skończony
`spacing`; pipeline FFT i pipeline dyspersji przekazują rzeczywiste `dt`.
Test liniowego sygnału potwierdza odzyskanie zadanej pochodnej w jednostkach na
sekundę w obu ścieżkach.

### 3.25. Welch zwracał błędnie znormalizowane `amplitude_squared` i PSD

**Waga:** krytyczna dla wartości liczbowych  
**Status:** zabezpieczone fail-closed, test regresyjny dodany

Welch liczy FFT z krótszych, nakładających się segmentów i opcjonalnym lokalnym
oknem Hann, natomiast dalszy kod dzielił moc przez sumę lub energię okna całego
rekordu. Wynik oznaczony jako `amplitude_squared` albo `psd` nie miał zatem
deklarowanej normalizacji. Do czasu wprowadzenia pełnej normalizacji każdego
segmentu Welch jawnie obsługuje tylko `raw_power`; pozostałe skale zgłaszają
`ValueError`. Notatki wyniku precyzują również, że `S` jest mocą Welch, ale
`S_complex` pozostaje koherentnym FFT całego rekordu używanym do rekonstrukcji
fazy i trybów.

### 3.26. `preloaded_data` dyspersji omijało normalizację układu osi

**Waga:** krytyczna; bezpośrednio związana z propagacją slice  
**Status:** naprawione, test 1D i 2D widoku jednoskładnikowego dodany

Dane ładowane leniwie z Zarr przechodziły przez
`normalize_magnetization_components`, ale zmaterializowany widok przekazany jako
`preloaded_data` nie. Po wyborze komponenty tablica `(T,Z,Y,X)` mogła zostać
zinterpretowana tak, jakby ostatnia oś była osią komponentów. Obie ścieżki
loadera tworzą teraz kanoniczny układ `(T,Z,Y,X,C)`. Test obejmuje obliczenie
dyspersji 1D i 2D z wcześniej wybranej skalarnej komponenty.

### 3.27. Fluent helper `filters(...).compute_2d()` zawsze przekazywał błędne argumenty

**Waga:** wysoka dla publicznego API notebookowego  
**Status:** naprawione, test kontraktu dodany

Helper bezwarunkowo przekazywał `save=False` i `force=False`, mimo że niższa
funkcja ich nie przyjmowała, więc nawet domyślne wywołanie kończyło się
`TypeError`. Fałszywy przykład sugerował dodatkowo dostępny cache 2D. Domyślne
wywołanie nie przekazuje już tych flag; `save=True` i `force=True` zgłaszają
precyzyjny błąd o niezaimplementowanym cache, a karta nie reklamuje zapisu 2D.

### 3.28. Dyspersja 2D ignorowała filtry, okno przestrzenne i deklarowane skalowanie

**Waga:** krytyczna dla zgodności 1D/2D i wartości liczbowych  
**Status:** naprawione w zakresie pre-FFT i skalowania, test analityczny dodany

`compute_dispersion_2d()` wykonywało jedynie detrend, okno czasu i surową moc.
Konfiguracja fluent `.filters(...)` nie docierała do obliczeń, `space_window` z
`DispersionConfig` nie było stosowane, a wynik nie zapisywał rodzaju ani
współczynników skalowania. Obecnie:

- prefiltry wykonują się na osi czasu z fizycznym `dt`,
- okno przestrzenne jest stosowane osobno na osiach Y i X,
- `amplitude_squared` uwzględnia koherentny gain wszystkich trzech osi,
- PSD uwzględnia energię okien oraz element miary `dt*dx*dy`,
- wynik 2D przechowuje `scaling` i `scaling_factors`.

Filtry post/live pozostają niejednoznaczne dla dwóch osi k; są teraz jawnie
odrzucane z `NotImplementedError`, a nie ignorowane. Test płaskiej fali 2D z
oknami Hann odzyskuje analityczne `A²`.

### 3.29. `m+`/`m-` miały niejednoznaczne fizycznie etykiety RCP/LCP

**Waga:** średnia obliczeniowo, wysoka interpretacyjnie  
**Status:** naprawione dokumentacyjnie, test unitarności dodany

Transformacja `(mx ± i*my)/sqrt(2)` jest poprawna i unitarna, ale przypisanie jej
na stałe nazw RCP/LCP oraz clockwise/counterclockwise nie jest możliwe bez
podania znaku transformaty Fouriera i kierunku obserwacji osi. Klucze publiczne
`+` i `-` pozostają bez zmian, natomiast UI i dokumentacja pokazują teraz
jednoznaczną definicję algebraiczną. Test sprawdza znaki oraz zachowanie normy
`|m+|²+|m-|²=|mx|²+|my|²`.

### 3.30. Czyste transformacje modes wymagały Matplotlib już przy imporcie

**Waga:** wysoka dla opcjonalnych zależności i środowisk headless  
**Status:** naprawione, import potwierdzony bez Matplotlib

`modes/__init__.py` importowało cztery nieużywane moduły Matplotlib, a
`style.py` definiowało klasę dziedziczącą po niezdefiniowanym `mcolors`, gdy
Matplotlib brakowało. Nawet czysto numeryczne `VortexOptics.to_circular_basis`
nie dawało się zaimportować. Usunięto nieużywane importy, dodano bezpieczną
klasę zastępczą zgłaszającą zależność dopiero przy użyciu normalizacji kolorów,
a konwersja HSV importuje Matplotlib lokalnie wyłącznie podczas renderowania
holografii.

### 3.31. CPSD używało referencji przesuwającej się razem z oknem

**Waga:** krytyczna dla fizyki transmisji  
**Status:** naprawione, test stałej referencji dodany

W `method="cpsd"` każde lokalne widmo było mnożone przez sprzężenie pierwszego
piksela tego samego przesuwanego okna. `reference_window` nie definiowało więc
sygnału referencyjnego CPSD; referencja zmieniała położenie dla każdej kolumny.
Obecnie przed analizą okien powstaje jedno widmo referencyjne, uśrednione po
komórkach należących do wybranych okien referencyjnych, i to samo widmo jest
używane przez ścieżkę szeregową oraz równoległą. Brak lub niezgodny kształt
referencji powoduje jawny błąd.

### 3.32. Moduł transmisji wymagał `tqdm`, Matplotlib i ciężkiego `core` przy imporcie

**Waga:** wysoka dla środowisk headless i czasu startu  
**Status:** naprawione

Silnik obliczeniowy importował bezwarunkowo `tqdm` oraz helper z `mmpp.core`, co
uruchamiało m.in. zależność `rich`. Pakiet `transmission/__init__.py` dodatkowo
ładował moduły plot, experimental i batch, a więc Matplotlib. Pasek postępu ma
teraz bezpieczny fallback do zwykłego iteratora, detekcja notebooka jest lekka i
lokalna, a eksporty plot/batch/experimental są ładowane dopiero przy dostępie.
Test cache transmisji nie wymaga już pominięcia bez `tqdm`.

### 3.33. `power_ratio` i mapy `power_*` liczyły amplitudę zamiast mocy

**Waga:** krytyczna dla wartości i interpretacji transmisji  
**Status:** naprawione, test analityczny dodany

Kod sumował `abs(FFT)` mimo nazw `power_ratio`, `power_map`, `power_plus` i
`power_minus`. Wynik znormalizowany był stosunkiem amplitud, a nie mocy.
Wszystkie ścieżki wykonawcze — standardowa, sliding-window, zoptymalizowana i
równoległa — używają teraz `|FFT|²`. Wagi komponentów są wagami mocy; przed
transformacją kołową amplitudy są mnożone przez pierwiastek z wagi. CPSD nadal
używa `|X R*|`, zgodnie z wymiarem cross-power.

### 3.34. Zerowa moc referencji była zastępowana mianownikiem `1`

**Waga:** wysoka dla wiarygodności normalizacji  
**Status:** naprawione, test zerowej referencji dodany

Przy `normalize="reference"` częstotliwość z zerową mocą referencyjną była
dzielona przez sztuczne `1`, przez co niezdefiniowany stosunek wyglądał jak
zwykła wartość transmisji. Obecnie takie wiersze otrzymują `NaN`, emitowane jest
ostrzeżenie, a liczba nieważnych binów trafia do
`metadata["invalid_normalization_bins"]`. Normalizacja do maksimum zachowuje
zera dla całkowicie zerowego wiersza, ale również raportuje jego nieważność.

### 3.35. Interaktywny resolver pojedynczej komponenty zawsze przypisywał ją do `mz`

**Waga:** krytyczna; bezpośrednio związana z `m[...,component]`  
**Status:** naprawione, testy `x/y/z` dodane

Niezależnie od źródłowego slice tablica trybu `(Y,X,1)` była interpretowana jako
`mz`. Widok utworzony z `m[...,0]` lub `m[...,1]` pokazywał więc zera w panelu
oczekiwanej komponenty. Resolver wykorzystuje teraz jednoznaczną, pojedynczą
komponentę żądaną przez viewer. Próba wyznaczenia `+/-/rho/phi` z jednego kanału
zgłasza błąd, ponieważ te bazy wymagają jednocześnie `mx` i `my`.

### 3.36. Nowy viewer holografii ignorował gamma i próg szumu z konfiguracji

**Waga:** wysoka dla zgodności API i reprodukowalności obrazu  
**Status:** naprawione, test propagacji parametrów dodany

Renderer statyczny, suwak fazy, live animation oraz eksport animacji wywoływały
`complex_holography()` z zaszytymi defaultami. Parametry
`holography_gamma`/`holography_noise_threshold` są teraz rozwiązywane z jawnych
argumentów albo konfiguracji analizatora i przekazywane przez wszystkie cztery
ścieżki. Wartości niefinitywne, niedodatnia gamma i ujemny próg są odrzucane.

### 3.37. `InteractiveSpectrum.show()` bezgłośnie ignorowało literówki w argumentach

**Waga:** wysoka programistycznie  
**Status:** naprawione, test fail-fast dodany

Dowolne nieznane słowo kluczowe trafiało do `**_ignored` i znikało. Viewer
akceptuje nadal zgodne aliasy `animate`, `save_path` i `animation_save_path`,
ale pozostałe nazwy zgłaszają `TypeError` przed ładowaniem danych.

### 3.38. Holografia deformowała tablice 3D, a preview DC dzielił przez zero

**Waga:** wysoka dla poprawności renderowania  
**Status:** naprawione, cztery testy dodane

`np.dstack((H,S,V))` nie dodaje kanału RGB poprawnie dla wejścia `(Z,Y,X)`, mimo
że `TopologicalAnimator` deklaruje obsługę takich tensorów. Zastąpiono je przez
`np.stack(..., axis=-1)`, dzięki czemu wynik ma zawsze `input_shape + (3,)`.
NaN/Inf są neutralizowane przed budową HSV. Suwak fazy dla częstotliwości zero
pokazuje teraz `t=n/a (DC)` zamiast wykonywać `1/f`; błędy preview trafiają też
do statusu UI, a nie tylko do tracebacku.

### 3.39. Baza cylindryczna odwracała fizyczny znak osi Y

**Waga:** krytyczna dla chiralości `m_phi`  
**Status:** naprawione, test pola radialnego dodany

`to_cylindrical_basis()` stosowało `Y=-(y-cy)` z uzasadnieniem typowym dla
obrazów ekranowych. Dane i aktywne renderery używają jednak rosnącej osi
fizycznej Y oraz `origin="lower"`; dodatkowe odbicie odwracało znak składowej
azymutalnej i interpretację rotacji. Transformacja używa teraz `Y=y-cy`,
sprawdza zgodność kształtów i skończoność środka. Dla analitycznego pola
`(mx,my)=(x-cx,y-cy)` test otrzymuje `m_rho=r` i `m_phi=0`.

### 3.40. Import modułu testowego uruchamiał obliczenia i wypisywał raport

**Waga:** wysoka programistycznie i wydajnościowo  
**Status:** naprawione, test ciszy importu dodany

`mmpp.fft.dispersion.test_dispersion_models` zawierało bezwarunkowy kod
demonstracyjny, modyfikowało `sys.path`, wykonywało osiem modeli analitycznych i
drukowało tabelę podczas discovery/import sweep. Diagnostyka znajduje się teraz
w `main()` chronionym przez `if __name__ == "__main__"`; import nie ma efektów
ubocznych.

### 3.41. `modes.analyzer.data_access` importowało nieistniejący moduł

**Waga:** wysoka dla częściowo zrefaktoryzowanego API  
**Status:** naprawione, import objęty testem

Mixin wskazywał `..compatibility`, choć rzeczywisty moduł nazywa się `compat`.
Wyjątek był dodatkowo ukrywany przez szeroki `except` w pakiecie `analyzer`, więc
`DataAccessMixin` znikał z eksportów bez wyjaśnienia. Ścieżka została poprawiona.
Stary `modes.init.__all__` reklamował również trzy symbole, które nie były
zdefiniowane po nieudanym imporcie; lista zawiera teraz tylko dostępne nazwy.

### 3.42. Publiczne interfejsy obliczeniowe wymagały Matplotlib przy imporcie

**Waga:** wysoka dla headless/notebook discovery  
**Status:** naprawione, test czterech importów dodany

`mmpp.fft.core` ładowało `FFTPlotter`, `transmission.interface` ładowało moduł
plot, `modes.config` importowało pyplot, a moduł elektromagnetyczny importował
pyplot/GridSpec globalnie. Typy korzystają teraz z odroczonych adnotacji, a
zależności graficzne są importowane dopiero w metodach renderujących. Numeryczne
FFT, transmisja, konfiguracja modes i analiza Poyntinga dają się odkrywać bez
Matplotlib.

### 3.43. Eksperymentalne helpery transmisji wymagały Pandas przy imporcie

**Waga:** średnia dla zależności opcjonalnych  
**Status:** naprawione, objęte testem importu

`transmission.experimental` i `overlay_experimental` importowały Pandas
bezwarunkowo, więc nawet odkrywanie ich funkcji nie działało w minimalnym
środowisku. Moduły są teraz importowalne, a brak Pandas zgłasza precyzyjny
`ImportError` dopiero przy dostępie do operacji tabelarycznej. Pełny sweep 129
modułów `mmpp.fft` kończy się obecnie jednym oczekiwanym błędem — bezpośrednim
importem stricte graficznego `transmission.plot` bez Matplotlib.

### 3.44. Równoległy batch transmisji mieszał kolejność jobów i parametrów

**Waga:** krytyczna dla analiz parametrycznych  
**Status:** naprawione, test kolejności klucza cache dodany

Rezultaty z `as_completed()` były dopisywane w kolejności zakończenia. Tablice
`results`, `job_paths` i parametrów nie odpowiadały więc stabilnie wejściowemu
batchowi. Rekordy zachowują teraz indeks wejściowy i są sortowane przed budową
wyniku. Każdy worker dostaje także własną głęboką kopię mutowalnego
`TransmissionConfig`. Klucz batch cache nie sortuje już ścieżek, ponieważ
kolejność jest częścią publicznego wyniku. Częściowe błędy wraz z indeksami są
przechowywane w `BatchTransmissionResult.errors` i serializowane w formacie 1.1.

### 3.45. Batch cache nie wykrywał zmian danych pod tą samą ścieżką

**Waga:** krytyczna dla poprawności cache  
**Status:** naprawione, test zmiany chunka dodany

Hash obejmował jedynie tekst ścieżki joba. Zmiana Zarr/H5 pod istniejącą ścieżką
mogła zwrócić cały stary batch bez wejścia w weryfikację cache pojedynczego
wyniku. Klucz zawiera teraz uporządkowaną sygnaturę manifestu plików źródła:
nazwę relatywną, rozmiar i `mtime_ns`, a także liczbę plików i sumaryczny
rozmiar. Operacja nie czyta payloadu chunków, ale jej koszt jest liniowy w
liczbie plików i został świadomie zaakceptowany dla poprawności trwałego cache.

### 3.46. Heatmapa batch transmission zestawiała niezgodne częstotliwości

**Waga:** krytyczna numerycznie i interpretacyjnie  
**Status:** naprawione, trzy testy regresyjne dodane

Wyrównanie kolejnych przekrojów sprawdzało jedynie liczbę punktów osi
częstotliwości. Dwa przebiegi o tej samej długości, ale innych wartościach
częstotliwości, były więc układane w jednej macierzy bez interpolacji. Jeśli
liczby punktów się różniły, wartości poza zakresem źródła były uzupełniane
zerami, co tworzyło sztuczne minima mogące wyglądać jak zanik transmisji.

Dodano wspólny, testowalny mechanizm, który:

- porównuje kształt i rzeczywiste wartości obu siatek,
- interpoluje także siatki o tej samej długości, jeśli ich punkty się różnią,
- obsługuje ściśle rosnące i ściśle malejące osie,
- odrzuca siatki niemonotoniczne i niefinitywne,
- zapisuje `NaN` poza wspólnym zakresem zamiast wymyślonego zera,
- stosuje tę samą semantykę w heatmapie zwykłej i różnicowej,
- normalizuje względem maksimum wartości skończonych, zachowując brak danych.

### 3.47. ROI modów przekazywało extent w metrach do osi opisanej w nanometrach

**Waga:** krytyczna dla geometrii i interpretacji map modów  
**Status:** naprawione, test jednostek dodany

Po propagacji `DatasetGeometry` granice widoku `x/y` były pobierane z pól
`min_m` i `max_m`, czyli w metrach, i przekazywane bez konwersji do
`FMRModeData.extent`. Publiczny model, `spatial_extent_nm`, operacje crop oraz
wszystkie wykresy modów traktują jednak extent jako nanometry i opisują osie
`x (nm)`/`y (nm)`. Mapa zachowywała kształt obrazu, lecz wartości osi były
zaniżone miliard razy.

Dodano jeden resolver extentu w nanometrach. Dla geometrii widoku wykonuje on
jawną konwersję `m -> nm`, a dla starszej ścieżki bez geometrii zachowuje
dotychczasowe `nx*dx_nm` i `ny*dy_nm`. Test ROI `100–500 nm` potwierdza
dokładnie taki zakres publiczny.

### 3.48. Cache pojedynczych modów był zapisywany, ale nigdy odczytywany

**Waga:** wysoka wydajnościowo dla interaktywnego viewer'a  
**Status:** naprawione, test liczby odczytów Zarr dodany

`FMRModeAnalyzer.get_mode()` przy każdym wywołaniu ponownie odczytywał slice
`(frequency,z,y,x,component)` z Zarr, mimo że analizator posiadał ograniczony
cache LRU. Metoda wywoływała wyłącznie `put()`, nie wykonywała `get()`, a po
zapisie tworzyła drugi równoważny obiekt `FMRModeData` do zwrotu. Powtarzane
kliknięcia lub renderowanie tych samych modów generowały zbędne I/O i alokacje.

Cache jest teraz sprawdzany po znormalizowaniu ujemnego indeksu warstwy oraz po
wyznaczeniu rzeczywistego binu częstotliwości. Klucz używa częstotliwości binu,
dzięki czemu dwa bliskie żądania mapujące się na ten sam bin współdzielą wynik.
Jeden obiekt jest zapisywany i zwracany. Test z licznikiem backendu potwierdza
jeden odczyt dla dwóch żądań tego samego binu.

### 3.49. Oś czasu modów mogła mieć złą długość, stride albo nieregularny krok

**Waga:** krytyczna dla częstotliwości FFT i fizyki modów  
**Status:** naprawione, trzy testy osi czasu dodane

`compute_modes()` dopuszczało kilka niespójnych przypadków:

- nieregularne czasy zastępowało średnią dodatnich różnic i wykonywało RFFT,
- gdy pełna oś `t` nie pasowała do zmaterializowanego widoku, używało jej mimo
  innej liczby próbek niż tablica poddawana FFT,
- dla skalarnego `t_sampl` nie uwzględniało kroku zagnieżdżonego `t_slice`,
- `time_step_scale` mogło zostać przemnożone mimo że jawna oś czasu zawierała
  już stride widoku,
- brak metadanych kończył się arbitralnym `dt=1e-12 s`, nadając wynikowi pozorną
  fizyczną oś częstotliwości.

Jawna oś czasu jest teraz składana z slice widoku i lokalnego `t_slice`, ale
używana tylko wtedy, gdy jej długość dokładnie odpowiada analizowanej tablicy.
Musi być skończona, ściśle rosnąca i równomierna w tolerancji `1e-6`. Dla
skalarnego `dt` stosowany jest zarówno krok `t_slice`, jak i mnożnik danych
zmaterializowanych. Brak wiarygodnego `dt` zgłasza `ValueError` zamiast
produkować wynik o wymyślonych jednostkach.

### 3.50. Główne FFT wyznaczało `dt` tylko z pierwszej pary czasów

**Waga:** krytyczna dla osi częstotliwości  
**Status:** naprawione, test pełnej osi i strided slice dodany

`resolve_dt_from_metadata()` dla jawnej tablicy `attrs["t"]` sprawdzało tylko
`t[1]-t[0]`. Późniejsza zmiana kroku, cofnięcie czasu, `NaN` albo jitter nie
były wykrywane. Dodatkowo stride widoku był następnie mnożony osobno, zamiast
wyznaczyć krok bezpośrednio z aktywnej części osi.

Loader wybiera teraz czas zgodnie z pierwszym slice widoku, sprawdza wszystkie
wartości i wszystkie różnice oraz wymaga osi skończonej, ściśle rosnącej i
równomiernej w tolerancji `1e-6`. Dla jawnej osi stride wynika już z wybranych
czasów; dla skalarnego `t_sampl`/`dt` nadal jest mnożony przez krok slice.
Test potwierdza `dt=4 ps` dla źródła `2 ps` i `::2` oraz odrzucenie zmiany kroku
w drugiej części rekordu.

Przy rozszerzeniu walidacji poprawiono też zgodność istniejącego testu z Zarr 2
i Zarr 3 (`create_dataset`/`create_array`), bez zmiany zachowania biblioteki.

### 3.51. Odczyt modów nie obsługiwał legalnego cache jednowarstwowego 4D

**Waga:** wysoka dla kompatybilności cache i wejść zmaterializowanych  
**Status:** naprawione, sześć wariantów kształtu objętych testem

Główny `get_mode()` zakładał wyłącznie cache `(f,z,y,x,c)`, mimo że refaktorowany
mixin oraz starsze/zewnętrzne wyniki dopuszczają jednowarstwowy `(f,y,x,c)`.
Taki cache był interpretowany tak, jakby `y` oznaczało liczbę warstw, a następnie
indeksowany pięcioma indeksami. Podobnie wejście zmaterializowane, które nie
pochodzi ze standardowego keep-dims wrappera, mogło nie mieć osi Z lub C.

Wejście jest teraz normalizowane przed wyborem `z_slice` do kanonicznego
`(t,z,y,x,c)`. Wcześniej wybrana warstwa i/lub komponenta otrzymują osie
singleton, bez odzyskiwania danych spoza widoku. Walidowane są 1 albo 3 kanały,
dodatni krok i niepusty wybór `z_slice`. Odczyt zachowuje dodatkowo zgodność ze
starszym cache `(f,y,x,c)`. Testy obejmują pełne dane, brak osi Z/C oraz oba
warianty keep-dims używane przez `DatasetAwareWrapper`. Potwierdzono przy tym,
że standardowe `m[:,1,...]` zachowuje singleton Z i nie odzyskuje pełnej warstwy.

### 3.52. Widmo interaktywne sliced modes ładowało globalny cache i myliło amplitudę z mocą

**Waga:** krytyczna dla spójności ROI i wartości spectrum  
**Status:** naprawione, dwa testy cache/algebry dodane

Ścieżki `arr` i `freqs` analizatora sliced modes wskazywały poprawną grupę
`modes/{dataset}/views/{view_id}`, lecz fallback `power_sum`/`power_max` był
zaszyty jako globalne `modes/{dataset}`. Viewer mógł zatem zestawić mody
lokalnego ROI z widmem pełnego datasetu, a przed obliczeniem lokalnego cache
potrafił również pobrać globalne `fft/.../spectrum`.

Kandydaci widma są teraz budowani z `self.mode_group`; analizator widoku nigdy
nie przechodzi do globalnej grupy ani globalnego FFT. Legacy fallback pozostaje
wyłącznie dla analizatora pełnego datasetu.

Dodatkowo tablice nazwane `power_sum` i `power_max` były liczone z `abs(FFT)`,
czyli amplitudy. Wspólny helper używa teraz `|FFT|²`, następnie maksimum lub sumy
po osiach przestrzennych, Z i komponentach. Test analityczny dla `3+4j`
potwierdza moc `25`, a test kandydatów dowodzi braku globalnej ścieżki w widoku.
Nowy cache zapisuje `power_definition="abs_fft_squared"`; nieoznaczone albo
amplitudowe podsumowania są odrzucane z instrukcją `force=True`, dzięki czemu
stary cache nie jest cicho interpretowany według nowej definicji.

### 3.53. `modes.find_peaks()` ignorowało komponentę i jawne parametry detektora

**Waga:** wysoka dla analizy rezonansów i publicznego API  
**Status:** naprawione, test dwóch niezależnych komponent dodany

Argument `component` nie uczestniczył w obliczeniu. Dodatkowo metoda rozwiązywała
`threshold` i `min_distance`, ale wywoływała `_detect_peaks()` bez tych wartości,
więc detektor ponownie używał konfiguracji domyślnej. Wielowymiarowe widmo mogło
zostać spłaszczone do długości różnej od osi częstotliwości, a wewnętrzny szeroki
`except` zamieniał błąd indeksowania na pustą listę pików.

Publiczna metoda przyjmuje teraz wyłącznie jednoznaczny układ `(f,)` albo
`(f,component)`, sprawdza zgodność długości, zakres komponenty i skończoność
danych. Jawne `threshold=0.0` nie jest już zastępowane defaultem, a oba parametry
są przekazywane do detektora. Widmo zerowe zwraca pustą listę bez dzielenia przez
zero. Test z pikami komponent `0` i `1` przy różnych częstotliwościach potwierdza
niezależną selekcję oraz działanie jawnego progu.

### 3.54. `ModeDataLoader` nie był slice-aware i potrafił wymyślić oś `0–50 GHz`

**Waga:** krytyczna dla propagacji ROI i fizycznej osi częstotliwości  
**Status:** naprawione, cztery testy ścieżek/jednostek dodane

Pomocniczy loader używany przez interaktywny viewer deklarował obsługę
`slice_info`, ale `arr/freqs/power` rozwiązywał najpierw z globalnego
`modes/{dataset}`. Jeśli nie znalazł dokładnego spectrum, skanował dowolny klucz
z `_s` i mógł wybrać cache innego widoku. Warunek był tak szeroki, że substring
`power_sum` również uchodził za znacznik sliced FFT, a sama tablica mocy mogła
zostać potraktowana jako częstotliwości.

Przy braku osi o zgodnej długości loader tworzył liniowe `0–50 GHz`, niezależnie
od `dt` i liczby próbek. Niezgodność na końcu była ukrywana przez obcięcie obu
tablic do krótszej długości. Zespolone FFT było zmieniane w `|FFT|`, mimo że
wynik był opisywany jako power.

`ModeDataContext` przechowuje teraz dokładną `mode_group` analizatora albo
wyznacza identyczny `view_id` dla zwykłego slice. Widok nie może przejść do
globalnych modes/FFT. Fallback `_s` działa tylko dla faktycznej ścieżki
kończącej się `/spectrum`. Brak zgodnej osi i niezgodne długości zgłaszają błąd;
zespolone spectrum jest konwertowane do `|FFT|²`. Loader obsługuje też cache 4D
i singleton wybranej komponenty bez ponownego indeksowania jej numerem.

Nowy viewer korzysta ponadto z jawnego kontraktu `SpectrumResult.frequencies_ghz`
zamiast heurystyki wartości maksymalnej. Dzięki temu widmo `100–200 kHz` jest
poprawnie wyświetlane jako `0.0001–0.0002 GHz`, a nie `100000–200000 GHz`.
Filtry interaktywne odrzucają różną długość trace i osi zamiast cicho obcinać.
Loader odrzuca również `power_sum/power_max` bez znacznika aktualnej definicji
`abs_fft_squared`, tak samo jak główny analizator.

### 3.55. Pusty zakres częstotliwości w viewerze przywracał całe widmo

**Waga:** wysoka interpretacyjnie  
**Status:** naprawione, test zakresu poza danymi dodany

`apply_spectrum_filters()` po stwierdzeniu, że żaden bin nie należy do
`freq_min–freq_max`, zastępowało maskę tablicą samych `True`. Viewer wyświetlał
więc pełne widmo bez informacji, że żądane pasmo nie istnieje. Obecnie zgłaszany
jest `ValueError` zawierający żądany i dostępny zakres w GHz. Nie są już
prezentowane dane spoza filtra jako wynik filtra.

### 3.56. Fluent `modes.frequencies` i automatyczny wybór modu mieszały Hz z GHz

**Waga:** krytyczna dla wyboru fizycznej częstotliwości  
**Status:** naprawione, test jednostek i maksimum spectrum dodany

Właściwość `FFTModeInterfaceNew.frequencies` deklarowała GHz, ale zwracała
bezpośrednio `SpectrumResult.frequencies`, którego kontraktem są Hz.
`_default_mode_frequency()` także wybierało indeks maksimum na osi Hz i
przekazywało otrzymaną wartość do `FMRModeAnalyzer.get_mode()`, oczekującego
GHz. Wywołanie `modes.mode()` bez jawnego `f` mogło zatem szukać częstotliwości
większej `10⁹` razy od właściwej.

Właściwość używa teraz jawnego `frequencies_ghz`. Domyślny wybór korzysta z
`SpectrumResult.peaks[*].frequency_ghz` albo `peak_frequency_ghz`; fallback
również operuje wyłącznie na osi GHz. Test widma z maksimum przy `2e9 Hz`
potwierdza publiczne `[1,2,3] GHz` i wybór `2 GHz`.

### 3.57. Bridge `SpectrumResult.modes` współdzielił mutowalny kontekst i gubił dane materializowane

**Waga:** krytyczna dla izolacji wielu ROI i chaina spectrum→modes  
**Status:** naprawione, test izolacji obiektu i kontekstu dodany

`SpectrumModes._resolve_interface()` pobierało cache'owany `source_fft.modes`, a
następnie bezpośrednio zmieniało jego `_dataset_context` i `_slice_context`.
Dwa `SpectrumResult` utworzone z różnymi dodatkowymi widokami tego samego FFT
mogły więc nadpisywać sobie stan. Ponadto `mode_context` zawierał tylko nazwę
datasetu i slice; brakowało zmaterializowanej tablicy, `time_step_scale`,
geometrii i planu indeksowania. Chain `spec.modes.at(...)` mógł ponownie otworzyć
storage albo utracić downsampling/ROI użyte do obliczenia `spec`.

Bridge tworzy teraz prywatny klon `FFTModeInterfaceNew` przed związaniem
kontekstu. `SpectrumResult` przechowuje `preloaded_data` i mnożnik czasu, a
datasetowy wrapper dopisuje geometrię oraz `IndexPlan`. Wszystkie pola są
przenoszone do klonu; bazowy interfejs pozostaje niezmieniony. Test sprawdza
tożsamość tablicy materializowanej, skalę czasu, geometrię, slice i brak mutacji
obiektu źródłowego.

### 3.58. Bulk dispersion heatmap zestawiała różne osie `k` i dopisywała sztuczne zera

**Waga:** krytyczna numerycznie i interpretacyjnie  
**Status:** naprawione, dwa testy interpolacji dodane

`BulkMinimumPlotAccessor.heatmap()` uznawał osie falowe za zgodne, jeśli miały
taką samą liczbę punktów, bez porównania ich wartości. Dla różnej długości
wykonywał `np.interp(..., left=0, right=0)`, tworząc zerową intensywność poza
zakresem źródłowym. Nie odwracał też malejącej osi `k`, której `np.interp` nie
obsługuje zgodnie z oczekiwaniem.

Wspólny helper sprawdza długość cross-section, skończoność i ścisłą
monotoniczność obu siatek. Interpolacja zachodzi również dla przesuniętych siatek
tej samej długości, obsługuje kierunek malejący i pozostawia `NaN` poza
wsparciem danych. LogNorm korzysta wyłącznie ze skończonych dodatnich wartości i
odrzuca macierz bez legalnej domeny logarytmu. Testy potwierdzają interpolację
`15/25` oraz brak sztucznych zer na obu brzegach.

### 3.59. Wiersze bulk dispersion heatmap nie były sortowane według parametru

**Waga:** krytyczna dla map sweepów parametrycznych  
**Status:** naprawione, test nieuporządkowanego sweepu dodany

Accessor obliczał `_idx = argsort(param_values)`, ale metoda `heatmap()` nie
używała tej kolejności. Macierz pozostawała w kolejności jobów, podczas gdy
`extent` brał `params[0]` i `params[-1]` oraz zakładał ciągłą rosnącą oś.
Sweep wejściowy `[20,0,10]` przypisywał więc poszczególne cross-section do
błędnych wartości parametru. Overlay `k*` również nie był uporządkowany razem z
wierszami.

Nowy helper stabilnie sortuje parametry, wybiera siatkę referencyjną z pierwszego
punktu po sortowaniu, w tej samej kolejności wyrównuje cross-section i zwraca
permutację dla overlay. Kontroluje też zgodność liczby parametrów, osi i wierszy.
`which` oraz `kscale` są walidowane zamiast traktowania każdej literówki jako
alternatywnego trybu. Test `[20,0,10]` potwierdza wiersze `[0,10,20]`.

### 3.60. `imshow` fałszował położenie nierównomiernych punktów sweepu

**Waga:** krytyczna dla osi parametru bulk dispersion  
**Status:** naprawione, test nierównomiernych współrzędnych dodany

Nawet po uporządkowaniu wierszy `imshow(..., extent=[p_min,p_max])` rozmieszcza
każdy wiersz w równych odstępach. Dla rzeczywistych parametrów `[0,1,10]`
środkowy wynik był wizualnie położony około `5`, mimo że należał do `1`.

Heatmapa używa teraz `pcolormesh(k, param, matrix, shading="auto")`, przekazując
rzeczywiste centra obu osi. Nierównomierne odstępy są zachowane. Duplikaty
parametru są odrzucane, ponieważ nie definiują jednoznacznych oddzielnych wierszy
na osi. Test potwierdza zachowanie współrzędnych `[0,1,10]` i fail-fast dla
dwóch identycznych wartości.

### 3.61. Wynik bulk dispersion nie miał kontraktu integralności

**Waga:** krytyczna dla zapisu, heatmap i analizy sweepu  
**Status:** naprawione, testy regresyjne dodane

`BulkMinimumFrequencyResult` pozwalał utworzyć wynik, w którym tablice skalarne,
listy cross-section, osie `k` i gałęzie miały różne długości. `save()` używał
`zip(...)`, więc najkrótsza lista po cichu obcinała pozostałe punkty. Błąd mógł
ujawnić się dopiero po ponownym odczycie albo przypisać widmo do złego parametru.

Dodany `__post_init__` normalizuje tablice i wymaga dokładnie jednego rekordu na
punkt sweepu. Kontroluje długości `S(k)` względem osi `k`, pary gałęzi `f(k)`/`k`,
indeksy `errors`, skończoność parametrów oraz długości danych analitycznych.
Puste tablice są nadal legalne dla punktów, które jawnie figurują jako błąd.

### 3.62. Archiwum bulk dispersion gubiło metadane i wiele overlayów

**Waga:** wysoka  
**Status:** naprawione, round-trip dwóch modeli przetestowany

Format `.npz` nie zapisywał `meta`, a z `analytical_overlays` zachowywał tylko
legacy pola ostatniego modelu. Po `save()`/`load()` wykres porównujący kilka
modeli tracił więc wcześniejsze krzywe i kontekst obliczenia. Format otrzymał
wersję `2`; zapisuje komplet metadanych oraz każdą krzywą, etykietę, nazwę modelu
i parametry. Loader zachowuje kompatybilność ze starymi archiwami bez tych pól.

### 3.63. Parametry analityczne NumPy mogły uniemożliwić zapis wyniku

**Waga:** średnia  
**Status:** naprawione, tablica i `numpy.float64` objęte testem

`analytical_params` przechodziło bezpośrednio przez `json.dumps`. Typowe wartości
NumPy, w tym skalar `np.float64` lub tablica próbkowanych `k`, nie mają ogólnego
kodera JSON. Nowy format przechowuje słownik bez strat typów, a loader nadal
potrafi czytać legacy `an_params_json` i jawnie raportuje jego uszkodzenie.

### 3.64. Repliki BZ poza zakresem były projektowane na skrajne biny `k`

**Waga:** krytyczna fizycznie dla rekonstrukcji modów dyspersyjnych  
**Status:** naprawione, syntetyczny test masek BZ dodany

`build_bz_k_mask()` wybierał `argmin(|k-k_target|)` dla każdej repliki, również
gdy `k_target` leżał daleko poza próbkowanym zakresem. Takie repliki trafiały do
pierwszego lub ostatniego binu i wnosiły do IFFT sztuczną amplitudę brzegową.
Repliki poza nośnikiem osi są teraz pomijane. Jeżeli żadna dozwolona replika nie
ma reprezentacji w danych, operacja kończy się czytelnym `ValueError`.

### 3.65. Jawne okna `delta_k` i `delta_f` były cicho ignorowane

**Waga:** wysoka dla selektywności i interpretacji profilu  
**Status:** naprawione, testy pustych okien i parametrów dodane

Gdy dodatnie, jawnie podane okno nie obejmowało żadnej próbki, kod wracał do
najbliższego binu. Wynik nie odpowiadał więc żądanej selekcji. Obecnie puste
okno kończy się błędem, a szerokości muszą być dodatnie i skończone. Ujemne lub
ułamkowe marginesy binów, ujemne `n_bz`, nieznany `k_direction`, nieznany
reducer, nieskończone osie i cele również są odrzucane zamiast automatycznie
zmieniać znaczenie wywołania.

### 3.66. Slice ortogonalny dziedziczył złożone widmo globalne

**Waga:** krytyczna dla dalszej analizy wybranego przekroju  
**Status:** naprawione, test aktywnego źródła dodany

`DispersionResult1D.select_orthogonal_slice()` poprawnie wybierał lokalne `S`,
ale przenosił `S_folded` policzone dla całego wyniku. `get_active_data()` dawało
wtedy globalne folded spectrum zamiast wybranego slice'u. Ponieważ biblioteka
nie przechowuje folded spectrum per slice, pola folded są teraz czyszczone.
Sam model waliduje też zgodność `S`, osi, lokalnych widm, `S_complex`,
`orth_axis` oraz pary `S_folded`/`k_folded` już przy konstrukcji.

### 3.67. `DispersionResult1D.filtered()` maskował błędy i ignorował `kwargs`

**Waga:** krytyczna dla wiarygodności filtrów live  
**Status:** naprawione, test no-op i ścieżki keyword dodany

Metoda przechwytywała każdy wyjątek z filtracji, po czym zwracała wynik bez
zmian. Literówka lub niepoprawny parametr wyglądały więc jak udana operacja.
Jednocześnie udokumentowane `**kwargs` nigdy nie trafiały do silnika. Wyjątki są
teraz propagowane, a kopia `live` zostaje połączona z argumentami nazwanymi.
Test potwierdza zarówno błąd dla `gausian_morph`, jak i rzeczywistą normalizację
przez `.filtered(normalize=True)` bez modyfikacji `S_raw` i źródła.

### 3.68. Rejestry filtrów pozwalały na ciche no-op

**Waga:** wysoka, przekrojowa dla spectrum, modes i dispersion  
**Status:** naprawione, testy literówek i złych stage dodane

Oba normalizatory konfiguracji zachowywały albo ignorowały nieznane klucze.
Wspólny postprocessor dodatkowo przechwytywał wszystkie wyjątki pojedynczych
filtrów. Powodowało to pozornie poprawne wyniki bez zastosowania żądanej
operacji. Konfiguracja jest teraz sprawdzana względem rejestru właściwego dla
`pre`, `post` i `live`; zły typ stage, obca nazwa, błędny baseline/smoothing,
ujemna gamma oraz niepoprawne percentile kończą się jawnym błędem.

### 3.69. `remove_mean_and_static` ponownie wprowadzał składową DC

**Waga:** wysoka dla FFT niskoczęstotliwościowej  
**Status:** naprawione, średnia czasowa objęta testem

Kod najpierw odejmował średnią, a następnie pierwszą próbkę. Drugie odejmowanie
dodawało stały offset `-centered[0]`, więc filtr nazwany `remove_mean_and_static`
kończył z niezerową średnią i silnym binem DC. Kolejność została odwrócona:
najpierw odejmowany jest stan początkowy, a następnie końcowa średnia czasowa.

### 3.70. Filtry trace-wise obcinały wyniki dla danych całkowitych

**Waga:** średnia, programistyczna  
**Status:** naprawione, test dtype i wartości ułamkowych dodany

Helper tworzył bufor `zeros_like(input)`. Dla tablic integer wynik wygładzania
lub dzielenia był po cichu rzutowany z powrotem do integer i tracił część
ułamkową. Bufor korzysta teraz z typu wynikowego co najmniej `float64`, zachowując
również typ zespolony. Walidowana jest ponadto dodatnia całkowita długość okna i
dodatni całkowity rząd pochodnej spektralnej.

### 3.71. Wspólny postprocessor nie kontrolował osi częstotliwości

**Waga:** krytyczna dla wielowymiarowych widm  
**Status:** naprawione, test błędnej orientacji dodany

Filtry trace-wise działają wzdłuż osi `0`, lecz przekazana tablica
`frequencies` nie była używana nawet do kontroli kształtu. Widmo ułożone jako
`(N_trace, N_f)` mogło zostać wygładzone pomiędzy trace’ami zamiast po
częstotliwości. `FilterPipeline.postprocess()` wymaga teraz jednowymiarowej,
skończonej osi o długości równej `spectrum.shape[0]`. Nieznana nazwa stage także
jest odrzucana, zamiast zwracać dane bez zmian.

### 3.72. Walidacja modelu blokowała obsługiwane legacy `S_complex`

**Waga:** wysoka dla istniejących cache dyspersji  
**Status:** naprawione, test transpozycji `(Nf,Nk)` dodany

Rekonstruktor posiadał obsługę starszego zapisu kompleksowego widma jako
`(Nf,Nk)`, ale rygorystyczny konstruktor wyniku mógł odrzucić dane wcześniej.
`DispersionResult1D` kanonikalizuje teraz widmo podczas inicjalizacji do
`(Nk,Nf)` albo `(N_orth,Nk,Nf)`. Dalsze metody nie muszą zgadywać układu osi,
a zgodny legacy cache pozostaje możliwy do użycia.

### 3.73. `DispersionResult2D` nie kontrolował zgodności osi z `S`

**Waga:** krytyczna dla slice’ów `kx`/`ky`  
**Status:** naprawione, test kształtu i szerokości slice’u dodany

Model przyjmował dowolną tablicę oraz osie, a `slice_1d()` zakładał bez dowodu
układ `(Nkx,Nky,Nf)`. Niezgodność mogła przypisać częstotliwość lub kierunek do
złej osi. Konstruktor sprawdza teraz rangę, dokładny tuple długości oraz
skończoność osi. `k_value` i `dk_max` muszą być skończone, a szerokość nie może
być ujemna.

### 3.74. Flaga wygładzania prędkości grupowej była atrapą

**Waga:** wysoka dla estymacji `v_g=dω/dk`  
**Status:** naprawione, test zaszumionej gałęzi dodany

`DispersionBranch.compute_group_velocity(smooth=True)` i `smooth=False`
wykonywały identyczne `np.gradient`. Wygładzanie jest teraz rzeczywiste:
lokalny wielomian drugiego stopnia jest dopasowywany w fizycznej współrzędnej
`k`, więc działa również dla nierównomiernego próbkowania i nie cierpi na złe
uwarunkowanie dużych wartości `rad/m`. Model odrzuca różne długości tablic,
duplikaty lub niemonotoniczne `k` i zbyt krótką gałąź.

### 3.75. Wykres dyspersji mógł domyślnie ukryć większość danych

**Waga:** krytyczna wizualnie  
**Status:** naprawione; walidacja headless przechodzi, render blokuje brak Matplotlib

Heatmapa narzucała `xlim=(-10,10)` dla `rad/μm`, niezależnie od rzeczywistego
zakresu. Alias `meter` (w praktyce cycles/m po podzieleniu przez `2π`) dostawał
`(-20,20)`, co dla typowych osi rzędu `10^6 1/m` dawało pustą mapę. Sztuczne
limity usunięto. Jednostki `rad_um`, `rad_m`/`rad` i `cycles_m`/legacy `meter`
są walidowane spójnie w heatmapie, branch i overlayach. Błędny filtr live,
`fmax` bez wspólnego zakresu oraz `trim_0f` usuwające wszystkie biny nie są już
cicho ignorowane.

### 3.76. Zerowe widmo zwracało pozorne `f_min` i `k*`

**Waga:** krytyczna fizycznie dla analizy minimum gałęzi  
**Status:** naprawione, test zera i pustego gate SNR dodany

`find_lowest_possible_frequency()` wykonywało `argmax` również na samych zerach,
co wskazywało pierwszy bin po cutoff jako rzekomą gałąź. Jeśli żaden punkt nie
przechodził `min_snr`, funkcja wyłączała gate i kontynuowała. Teraz wymagana jest
dodatnia, skończona i nieujemna moc; brak punktu spełniającego gate oraz brak
sygnału przy `k≈0` kończą się błędem. Nawet `min_snr=0` nie dopuszcza wierszy o
zerowej mocy.

Metoda waliduje `side`, `peak_method`, cutoff, smoothing i okno `k`. Domyślnym
źródłem ilościowym jest `analysis_source="raw"`, dzięki czemu filtry wyłącznie
prezentacyjne nie zmieniają `f_min`, `k*` i `v_g` bez jawnej decyzji. Bulk
cross-section korzysta z tego samego źródła co ekstrakcja skalarów.

### 3.77. Fallback smoothing mógł zmienić długość gałęzi

**Waga:** wysoka w instalacji bez SciPy  
**Status:** naprawione

Box-filter używany bez SciPy ustawiał szerokość bez ograniczenia do liczby
punktów. Dla dużego `smooth_sigma`, `np.convolve(..., mode="same")` mogło zwrócić
więcej elementów niż `k_search`, a zerowe paddingi zaniżały częstotliwości na
brzegach. Okno jest teraz ograniczone, nieparzyste i używa paddingu `edge`, więc
długość oraz poziom brzegowy pozostają zachowane.

### 3.78. Multi-branch tracker obchodził wymagany prominence

**Waga:** krytyczna fizycznie dla liczby wykrytych modów  
**Status:** naprawione, test płaskiego widma dodany

Jeżeli `find_peaks` nie znalazł maksimum spełniającego
`min_prominence_log`, `_find_peaks_column()` bezwarunkowo zwracał globalny
`argmax`. Płaska kolumna lub szum zawsze produkowały więc kandydat gałęzi,
pomimo jawnego progu prominence. Fallback został usunięty: brak kwalifikowanego
peaku oznacza brak peaku w danej kolumnie.

### 3.79. Branch tracker nie walidował widma, osi ani opcji ilościowych

**Waga:** krytyczna dla stabilności i interpretacji  
**Status:** naprawione, testy zera, opcji i malejącej osi dodane

Nieznane `side`, zerowe `max_df`, ujemne smoothing/quality, błędne percentile,
pusty cutoff i widmo zerowe mogły przejść do algorytmu. Tracker wymaga teraz
skończonej, nieujemnej mocy, dodatniego sygnału oraz zgodnych osi i kształtu.
Wszystkie progi mają jawne dziedziny. Oś częstotliwości jest stabilnie sortowana
wraz z kolumnami `S`; duplikaty są odrzucane, ponieważ nie definiują
jednoznacznego sąsiedztwa dla peak detection.

### 3.80. Quality i smoothing gałęzi zależały od luk w próbkowaniu `k`

**Waga:** wysoka dla rankingu i prędkości grupowej  
**Status:** naprawione, test liniowej gałęzi na nierównym `k` dodany

Smoothness liczono ze `std(diff(f))`, ignorując `Δk`. Idealnie liniowa gałąź z
opuszczonymi kolumnami dostawała więc karę za większy skok częstotliwości. Nowa
metryka używa zmian fizycznego nachylenia `df/dk`; length score jest ograniczony
do `[0,1]`, więc confidence nie wykracza poza kontrakt.

Końcowy smoothing nie jest już gaussianem po indeksie. Lokalne dopasowanie
kwadratowe działa w rzeczywistym `k`, także po lukach, i jest ograniczane do
lokalnego zakresu obserwowanych częstotliwości, aby nie tworzyć overshootów.
`TrackedBranch` waliduje równoległe tablice, skończoność, rosnące `k` i quality.

### 3.81. Standalone branch plot nie konwertował trybu `meter`

**Waga:** wysoka wizualnie  
**Status:** naprawione, headless test konwersji dodany

Overlay przeliczał `rad/m → cycles/m`, ale `.plot.branches()` dla tego samego
`kscale` pozostawiał wartości w `rad/m`, jednocześnie etykietując je jako
`m⁻¹`. Wspólny helper obsługuje teraz `rad_um`, `rad_m`/`rad` oraz
`cycles_m`/legacy `meter` identycznie dla obu wykresów; kontroluje również
jednostkę częstotliwości i dodatnią szerokość linii.

### 3.82. Interaktywny explorer nadal ukrywał zakres `k`

**Waga:** krytyczna wizualnie i niespójna ze statycznym API  
**Status:** naprawione, headless test pełnego zakresu i jawnego limitu dodany

Interaktywny renderer posiadał osobny `_default_k_window_for_scale()` i nadal
przycinał oś do ±10 rad/µm, mimo usunięcia tego błędu ze statycznej heatmapy.
Ten sam wynik wyglądał więc inaczej zależnie od helpera. Ukryty limit usunięto;
crop zachodzi wyłącznie dla jawnego `options["k_xlim"]`, które musi zawierać
dwie skończone, rosnące liczby. Nieznane `kscale` jest odrzucane. LogNorm
pomija wartości NaN/Inf i nie jest tworzony dla stałego dodatniego widma, gdzie
`vmin == vmax`.

### 3.83. Legacy peak detector modów tracił oś częstotliwości

**Waga:** krytyczna dla automatycznego wyboru modu/FMR  
**Status:** naprawione, testy 2D, separacji i NaN dodane

`modes.utils.peak_detection.detect_peaks_scipy()` dla części tablic 2D używał
`flatten()`. Indeks peaku przestawał odpowiadać `frequencies`; późniejszy
`IndexError` był przechwytywany i zamieniany w pustą listę. Wspólny preparator
rozpoznaje teraz wyłącznie oś `0` lub ostatnią o długości `Nf`, redukuje pozostałe
wymiary średnią i odrzuca układ niejednoznaczny. Moc i częstotliwości muszą być
skończone i nieujemne.

Fallback bez SciPy ignorował `min_distance`, więc zwracał bliskie maksima,
które implementacja SciPy usuwała. Kandydaci są teraz wybierani malejąco według
amplitudy z tą samą minimalną odległością. Wyjątki z SciPy nie są już maskowane
jako „brak peaków”. Model `Peak` odrzuca boolean, NaN/Inf i typy nienumeryczne.
Próg większy od `1` pozostaje legalnym, istniejącym sposobem jawnego wygaszenia
wszystkich peaków na znormalizowanym widmie.

### 3.84. Crop modu FMR zapisywał fałszywy extent ROI

**Waga:** krytyczna dla geometrii i skalowania obrazu modu  
**Status:** naprawione, test częściowo wychodzącego ROI dodany

`FMRModeData.crop_to_region()` przycinał indeksy do granic tablicy, ale jako
`extent` wyniku wpisywał niezmieniony zakres żądany przez użytkownika. Dla ROI
`x=(-5,25)` na domenie `0..40 nm` wynikowe piksele były więc etykietowane jakby
zaczynały się od `-5 nm`; dodatkowo `int()` obcinał prawą krawędź.

Crop najpierw przecina zakres z domeną, używa `floor` dla początku i `ceil` dla
końca, a następnie wylicza extent z rzeczywistych krawędzi wybranych komórek.
Pusty lub odwrócony ROI kończy się błędem. Model waliduje skończoną dodatnią
częstotliwość, rosnący extent, niepuste wymiary, metadata oraz dodatni całkowity
`new_shape`; akceptuje poprawne indeksy komponentów typu `numpy.integer`.

### 3.85. Widgetowy prominence zależał od bezwzględnej skali widma

**Waga:** krytyczna dla automatycznego wyboru częstotliwości modu  
**Status:** naprawione, test niezmienniczości skali i fallbacku dodany

Kontrolka `peak_prom` ma zakres `[0,1]` i reprezentuje ułamek dynamiki. Kod
mnożył go przez maksimum tylko wtedy, gdy `max(S)>1`. Identyczny kształt o
amplitudzie `0.5` i `50` dawał więc inne zbiory peaków. Prominence jest teraz
zawsze `fraction * (max-min)`. Oś i trace muszą być zgodnymi tablicami 1D,
częstotliwości skończone, odległość dodatnią liczbą całkowitą, a prominence
ułamkiem `[0,1]`.

Fallback bez SciPy oblicza lokalne prominence i wybiera silniejsze maksima z
tym samym `min_distance`. Wyjątek z SciPy nie jest już maskowany jako pusta
lista. Test potwierdza identyczne pozycje dla widma pomnożonego przez `100`.

### 3.86. Preset zapisywał, ale nie odtwarzał `freq_unit`

**Waga:** wysoka dla interaktywnej osi i kliknięć  
**Status:** naprawione, round-trip stanu i błędne jednostki przetestowane

`collect_preset_state()` zawierał `freq_unit`, lecz `apply_preset_state()` je
ignorował. Po wczytaniu presetu etykieta, skalowanie osi i konwersja kliknięcia
mogły pozostać w poprzedniej jednostce. Jednostka jest teraz odtwarzana i musi
być jedną z `Hz`, `kHz`, `MHz`, `GHz`, `THz`; oba wcześniej duplikowane helpery
odrzucają obce wartości zamiast traktować je jak GHz.

Loader wymaga obiektu JSON, zachowuje snapshot i przy błędzie przywraca stan.
Przed pierwszą mutacją waliduje skończoność pól liczbowych, integralność indeksów,
kolejność percentyli i wartości enumów. Uszkodzony preset nie jest już cicho
przycinany ani raportowany jako poprawnie załadowany.

### 3.87. `SpectrumFilterState` dopuszczał niepoprawny stan widgetu

**Waga:** wysoka dla filtrowania live  
**Status:** naprawione, parametryczny test konfiguracji dodany

Ujemne `smooth_sigma`, zerowe okno, literówka nazwy filtra, NaN w zakresie i
odwrócone percentile były wykrywane dopiero w trakcie redraw albo zmieniane w
no-op. Dataclass posiada teraz własny kontrakt: skończone granice, znane tryby
smoothing/baseline, dodatnie całkowite okno, nieujemną sigmę, uporządkowane
percentile i jawne booleany.

### 3.88. Animacje modów stosowały sprzeczne konwencje znaku czasu

**Waga:** krytyczna dla interpretacji fazy i kierunku precesji  
**Status:** naprawione, test konwencji fazowej dodany

`TopologicalAnimator` realizował udokumentowaną ewolucję
`F(r,t)=F(r) exp(-iωt)`, natomiast klasyczna animacja fazy, temporalny eksport
oraz eksport aktywnego widoku używały efektywnie `exp(+iωt)`. Ten sam mod obracał
się więc w przeciwnym kierunku po przełączeniu holografii albo rodzaju eksportu.

Dodano wspólne helpery ewolucji zespolonej i zawijania fazy. Wszystkie ścieżki
używają teraz `Re[F exp(-iφ)]`, a test dla modu `1` oraz `i` sprawdza znak po
ćwierci okresu.

### 3.89. Cykl animacji zawierał dwie identyczne klatki graniczne

**Waga:** średnia dla płynności i efektywnego czasu filmu  
**Status:** naprawione, test cyklu dodany

`np.linspace(0, 2π, frames)` włączało oba końce przedziału. Pierwsza i ostatnia
klatka były fizycznie identyczne, co powodowało krótkie zatrzymanie na granicy
okresów i zmieniało efektywną szybkość fazową. Wspólny `_phase_cycle()` używa
teraz `endpoint=False`; dla czterech klatek zwraca dokładnie
`0, π/2, π, 3π/2`.

### 3.90. Eksport animacji nie miał stabilnego kontraktu ścieżki i zasobów

**Waga:** wysoka dla notebooków i automatyzacji eksportu  
**Status:** naprawione, testy rozszerzeń i walidacji dodane

Rozszerzenia `.MP4`/`.AVI` nie były rozpoznawane tak jak ich małe odpowiedniki,
zamiana rozszerzenia opierała się na tekstowym `replace()`, funkcja zwracała
`None` nawet po fallbacku MP4→GIF, a wyjątek powstały po utworzeniu figury mógł
ominąć `plt.close(fig)`.

Ścieżki są teraz rozwiązywane przez `pathlib` bez zależności od wielkości liter,
funkcje zwracają rzeczywistą ścieżkę zapisanego pliku, a figura jest zamykana w
`finally`. Walidowane są dodatnie i skończone FPS/liczba klatek/interwał,
nieujemna całkowita warstwa, skończona częstotliwość i uporządkowany zakres.
Zerowa amplituda dostaje niedegenerujący zakres normalizacji.

### 3.91. Eksport aktywnego widoku zmieniał fizycznie stałą amplitudę

**Waga:** wysoka dla poprawności wizualizacji  
**Status:** naprawione

Wiersz `magnitude` był mnożony przez sztuczny sinusoidalny impuls. Film sugerował
więc zmianę `|F(r,t)|`, mimo że dla pojedynczego liniowego modu amplituda jest
stała. Pulsowanie usunięto zarówno z eksportu widoku, jak i animacji pojedynczej
komórki. Błędy pobrania modu, indeksu osi lub artysty nie są już połykane jako
pusta klatka i pozornie udany eksport.

### 3.92. Animatory wiersza i kolumny mogły pozostawać aktywne jednocześnie

**Waga:** wysoka dla wydajności i spójności stanu widgetu  
**Status:** naprawione, testy czyszczenia stanu dodane

Uruchomienie holograficznej animacji całej kolumny nie zatrzymywało istniejących
timerów pojedynczych komórek. Dodatkowo wyjątek `event_source.stop()` pozostawiał
martwy obiekt w `_mode_animations`, a próba zatrzymania nieistniejącej animacji
kolumny usuwała znaczniki prawidłowych animacji wierszy.

Przejścia trybu zatrzymują teraz konkurujące animatory, wpis rejestru jest
usuwany w `finally`, a nieistniejąca animacja kolumny nie mutuje zbioru aktywnych
osi. Eksport rozpoznaje również aktywność przechowywaną wyłącznie w rejestrze
animacji kolumnowych.

### 3.93. Statyczny układ modów miał błędną liczbę wierszy

**Waga:** wysoka dla konfiguracji częściowych i czytelności wykresu  
**Status:** naprawione w kodzie, kontrakty układu przetestowane bez Matplotlib

`plot_modes()` tworzył trzy wiersze tylko wtedy, gdy wszystkie trzy widoki były
włączone, a dla każdej innej konfiguracji zawsze dwa. Włączenie wyłącznie fazy,
amplitudy lub części rzeczywistej zostawiało pustą oś; wyłączenie wszystkich
widoków produkowało pustą figurę zamiast błędu. Liczba i kolejność wierszy jest
teraz wyprowadzana dokładnie z flag konfiguracji, a pusty zestaw jest odrzucany.

Tytuły używają częstotliwości rzeczywiście zwróconej przez `get_mode()` zamiast
częstotliwości żądanej, która może zostać dopasowana do sąsiedniego binu.
Widok części rzeczywistej ma stałą symetryczną skalę `[-max|Re(m)|,+max|Re(m)|]`,
a zapis tworzy brakujący katalog nadrzędny. `update_single_mode_plot()` nie
połyka już błędów indeksu lub ładowania jako pozornie poprawnego redraw.

### 3.94. Interaktywny bridge `SpectrumResult` zgadywał jednostkę częstotliwości

**Waga:** krytyczna dla widm poniżej 1 MHz  
**Status:** naprawione, regresja niskiej częstotliwości przetestowana

Publiczny kontrakt `SpectrumResult.frequencies` podaje Hz i udostępnia jawne
`frequencies_ghz`. Starszy helper uznawał jednak wartości za Hz tylko wtedy,
gdy maksimum przekraczało `1e6`. Przykładowe `500 kHz` było zatem rysowane jako
`500000 GHz`, co psuło zakres, peaki i wybór modu.

Bridge używa teraz `frequencies_ghz`, jeśli właściwość istnieje, a w zgodności
ze starszym wynikiem bez tej właściwości zawsze wykonuje określoną kontraktem
konwersję Hz→GHz. Oś musi być niepustą, skończoną tablicą 1D.

### 3.95. FWHM było liczone na błędnym poziomie dla niezerowego tła

**Waga:** krytyczna dla raportowanej szerokości rezonansu  
**Status:** naprawione, testy syntetyczne z tłem i interpolacją dodane

Helper nazywany FWHM używał poziomu `peak/2`. Dla widma z niezerowym tłem nie
jest to połowa wysokości piku: prawidłowy poziom wynosi
`baseline + (peak-baseline)/2`. Dodatkowo kod odrzucał poprawne przecięcia
znajdujące się w pierwszym lub ostatnim interwale osi.

Obliczenie używa teraz połowy wysokości ponad minimalnym tłem, wyszukuje dwa
przecięcia otaczające dominujący pik i interpoluje je liniowo również w
interwałach brzegowych. Oś musi być skończona, 1D, zgodnej długości i ściśle
rosnąca; moc musi być skończona i nieujemna. Dla syntetycznego profilu
`[2,4,6,4,2]` na osi `[0,1,2,3,4]` otrzymywane są poziom `4` i szerokość `2`,
zamiast błędnej szerokości liczonej przy poziomie `3`.

### 3.96. Nowy eksport animacji materializował wszystkie zespolone klatki i dzielił przez zero dla DC

**Waga:** wysoka dla pamięci; krytyczna dla eksportu binu DC  
**Status:** naprawione, siatka fazy/czasu i przypadki DC przetestowane

Przed zapisem tworzono listę `n_frames` pełnych zespolonych kopii modu. Zużycie
pamięci rosło jak `O(n_frames × ny × nx × n_components)`, mimo że writer wymaga
tylko bieżącej klatki. Dla `actual_freq=0` wyznaczenie okresu wykonywało `1/0`.

Eksport przechowuje teraz tylko źródłowy mod oraz jednowymiarowe osie fazy i
czasu. Bieżąca projekcja jest obliczana w callbacku writera, więc pamięć danych
klatkowych ma rząd jednego modu. Eksport wymaga dodatniej skończonej
częstotliwości oraz niepustych, skończonych danych; cykl nie dubluje końca
`2π`, a etykiety czasu są wyprowadzane bezpośrednio z częstotliwości w GHz.

### 3.97. Publiczne opcje interaktywne były cicho konwertowane na inne wartości

**Waga:** wysoka dla powtarzalności notebooków  
**Status:** naprawione, walidatory typów i zakresów przetestowane

`z_layer=1.9` było obcinane do `1`, `smooth_window=4.8` do `4`, napis `"false"`
stawał się logicznym `True`, a nieznany `layout` był zastępowany przez `auto`.
Warstwa `FFTModeInterfaceNew` wykonywała dodatkowe `bool(...)`, omijając
walidację nowego explorera.

Parametry logiczne wymagają teraz rzeczywistych booli, indeksy i liczności liczb
całkowitych, DPI oraz rozmiar figury dodatnich wartości, limity osi dwóch
rosnących skończonych granic, a `aspect`/`layout` jawnych wartości enum. Metoda
FFT musi być `1` albo `2`. Walidacja następuje przed kosztownym obliczeniem
widma i tworzeniem UI, również dla opcji pochodzących z `.configure()` oraz
legacy fallbacku.

### 3.98. Wąski ROI pojedynczego komponentu był rozpoznawany jako oś komponentów

**Waga:** krytyczna dla `m[..., component]` i małych ROI  
**Status:** naprawione, regresja przestrzennego wymiaru długości 2 dodana

Reducer widma zakładał, że każdy ostatni wymiar długości `<=3` reprezentuje
komponenty `x/y/z`. Po wcześniejszym wyborze `m[...,1]` oś komponentu już nie
istnieje. Jeżeli przestrzenny ROI miał szerokość 2 lub 3 komórek, jego ostatni
wymiar był więc błędnie rozbijany na fikcyjne ślady `m_x`, `m_y` lub `m_z`,
zamiast zostać uśredniony przestrzennie jako jeden wybrany komponent.

`load_spectrum_data()` przekazuje teraz jawny znacznik `_single_component` oraz
kanoniczną etykietę komponentu do reducera. Dla pojedynczego komponentu wszystkie
osie poza częstotliwością są zawsze osiami redukowanymi, niezależnie od ich
długości. Parser etykiety rozpoznaje wyłącznie formy typu `mx`, `m_y`,
`$m_{z}$`; przypadkowa litera w napisie takim jak `complex spectrum` nie jest
już uznawana za komponent. Ten sam kontrakt zastosowano w helperze zgodności.

### 3.99. Równoległy batch był niedeterministyczny i przechowywał tensory niezgodne z heatmapą

**Waga:** krytyczna dla porównań wielu jobów; wysoka dla wydajności  
**Status:** naprawione, rzeczywisty test odwróconej kolejności workerów dodany

Pierwszy worker, który zakończył FFT, ustalał referencyjną siatkę częstotliwości.
Przy różnych `dt/nfft` wynik zależał więc od chwilowej kolejności wykonania:
ten sam batch mógł zachować inne joby w kolejnych uruchomieniach. Dodatkowo
`BatchSpectrumResult` i heatmapa deklarowały ślady 1D, lecz compute przechowywał
pełne tensory przestrzenne. Każdy worker wywoływał też globalne `gc.collect()`,
co zatrzymywało pozostałe wątki i dublowało pracę po błędzie.

Wyniki są teraz najpierw porządkowane według indeksu wejściowego. Siatkę
referencyjną wyznacza pierwszy poprawny job w tej kolejności, niezależnie od
czasu zakończenia wątku. Test celowo opóźnia pierwszy job i potwierdza identyczny
dobór wpisów. Widmo zespolone i moc są redukowane do osi częstotliwości osobno:
moc jako `mean(|F|²)`, a nie `|mean(F)|²`, co zapobiega sztucznemu zanikowi przez
kasowanie przeciwnych faz. Usunięto wymuszony GC z workerów.

`BatchSpectrumResult` oraz `SpectrumEntry` walidują rosnącą skończoną oś,
zgodne długości, jednowymiarowość śladów, nieujemną moc, liczbę ścieżek i
parametrów oraz typ warstwy. Cache pickle jest sprawdzany po odczycie, a
`to_stacked_array()` ma teraz faktycznie gwarantowany kontrakt `(jobs,freq)`.

### 3.100. Folding heatmapy rozłączał posortowane dane od współrzędnych parametru

**Waga:** krytyczna dla interpretacji map parametrycznych  
**Status:** naprawione, test zgodności wartości z indeksami dodany

Gdy zakres parametru obejmował już co najmniej 95% okresu, `apply_folding()`
zwracał oryginalne, nieposortowane wartości parametru oraz posortowane indeksy
mocy. Wiersze macierzy były więc wyświetlane przy współrzędnych należących do
innych jobów. Stały parametr powodował dzielenie przez zero podczas wyznaczania
liczby replikacji.

Helper zwraca teraz wartości i indeksy w tej samej kolejności oraz wymaga
poprawnej permutacji, skończonych wartości i dodatniego okresu. Stałego zakresu
nie można składać i kończy się on jawnym błędem. Heatmapa odrzuca nieznane
jednostki, tryby normalizacji, pusty zakres częstotliwości, niefinitywne
parametry i wielowymiarową moc. Analogiczne walidacje dodano do replikacji
punktów eksperymentalnych, łącznie z zerowym zakresem kąta i ujemnymi błędami.

### 3.101. Overlay wielu widm mógł cicho pominąć część jobów

**Waga:** wysoka dla porównań batch  
**Status:** naprawione, kontrakt osi i redukcji przetestowany

`MultiSpectrumResult.plot()` iterował przez
`zip(spectra, labels, colors)`. Krótsza lista etykiet lub kolorów bez ostrzeżenia
ucinała końcowe widma, przez co wykres nie obejmował wszystkich wyników.
Nieznana jednostka częstotliwości była traktowana jak GHz, a niezgodna lub
malejąca oś mogła dotrzeć bezpośrednio do Matplotlib.

Kolekcja wymaga teraz co najmniej jednego widma, a `batch.overlay()` jawnie
zgłasza przypadek, gdy wszystkie joby zawiodły. Długości etykiet i kolorów muszą
odpowiadać liczbie widm, jednostka jest enumem, flagi wymagają booli, a każdy
ślad przechodzi walidację rosnącej osi, zgodności pierwszego wymiaru i
nieujemnej skończonej mocy. Wielowymiarowa moc jest redukowana wyłącznie po
osiach innych niż częstotliwość.

### 3.102. `SpectrumResult` nie gwarantował integralności osi, widma i peaków

**Waga:** krytyczna dla wszystkich konsumentów widma  
**Status:** naprawione, kontrakty modelu i statycznych śladów przetestowane

Model przyjmował niezgodne długości osi i widma, niefinitywne dane, dowolny
kształt `power_override`, niezgodne surowe widmo oraz indeksy peaków poza
zakresem. Błąd ujawniał się dopiero w dalszym helperze albo prowadził do
przesuniętych markerów i niewłaściwego maksimum.

Konstruktor waliduje teraz frequency-first contract, skończoność, rosnącą oś,
zgodność surowego i transformowanego wyniku, enum skalowania/rodzaju widma oraz
kompletność `peaks_info`. Częstotliwość każdego peaku musi odpowiadać jego
indeksowi na osi. `component_label` i konfiguracje filtrów mają jawne typy.

Statyczny plotter przygotowuje wszystkie ślady przez jeden walidowany reducer.
Markery są pobierane bezpośrednio z już znormalizowanych i przeskalowanych linii
`mean(power_components)`, zamiast z osobnej wartości `(mean|F|)²`. Dzięki temu
leżą dokładnie na wykresie także po skalowaniu `×10^n`. Widmo już poddane
`log10` dostaje liniową oś z etykietą `log₁₀(...)`, zamiast drugiej osi
logarytmicznej.

### 3.103. `SpectrumResult.filtered()` maskował literówki i tracił tożsamość komponentu

**Waga:** wysoka dla reprodukowalności filtracji  
**Status:** naprawione, błędne opcje i round-trip komponentu przetestowane

Nieznany smoothing był traktowany jako Gaussian, `smooth_window=4.5` obcinano
do `4`, napisy w polach logicznych były uznawane za prawdę, nieznane kwargs
ignorowano, a sprzeczne aliasy wybierano według kolejności. Wynik filtrowania
resetował `_single_component`, więc późniejszy plot mógł ponownie uznać wąski
ROI przestrzenny za trzy komponenty.

Lista opcji jest teraz zamknięta; booleany, gamma, okno, sigma, baseline,
percentile i smoothing mają jawne kontrakty. Literówki oraz sprzeczne aliasy
kończą się błędem, a parametry okna bez wybranego smoothingu nie są ignorowane.
Filtrowany wynik zachowuje tożsamość pojedynczego komponentu.

### 3.104. Cache batch nie miał wersji schematu ani kontroli składu jobów

**Waga:** krytyczna po zmianie formatu batch na ślady 1D  
**Status:** naprawione, round-trip i odrzucenie starego schematu przetestowane

Cache pickle był akceptowany, jeśli tylko `len(cached)==len(results)`. Nie
sprawdzano wersji danych, ścieżek ani kolejności jobów. Stary cache zawierający
tensory przestrzenne mógł więc ominąć nowy kontrakt 1D albo przypisać widma do
innego zestawu wejść.

Format ma teraz `BATCH_SPECTRUM_SCHEMA_VERSION=2`, zapisywany również w Zarr.
Odczyt wymaga zgodnego typu, schematu i pełnej walidacji integralności. Compute
porównuje dokładną listę ścieżek w kolejności wejściowej; niezgodność powoduje
ponowne obliczenie zamiast użycia pozornie pasującego cache.

### 3.105. Peaki były wykrywane na innej wielkości niż publiczne widmo mocy

**Waga:** krytyczna dla wyboru rezonansów i modów  
**Status:** naprawione, test wielokomponentowy i end-to-end metadata dodane

Detekcja redukowała tensor jako `mean(|F|)`, podczas gdy `SpectrumResult.power`,
batch i plottery reprezentują `mean(|F|²)`. Operacje te nie są równoważne.
Dla dwóch komponentów pik `[10,0]` ma średnią amplitudę `5` i średnią moc `50`,
natomiast `[6,6]` ma średnią amplitudę `6`, ale moc `36`; stary kod mógł więc
wybrać inny rezonans niż maksimum publicznej mocy.

Detekcja działa teraz na mean spectral power po wszystkich osiach poza
częstotliwością. `min_prominence` jest stosowane w jednostkach tej wielkości.
`peaks_info` zapisuje `powers`, a kompatybilne pole `amplitudes` zawiera
`sqrt(power)`. `SpectrumResult.peaks` udostępnia oba pola.

Wspólny `find_peaks_1d` wymaga skończonego sygnału 1D i skończonego
nieujemnego prominence zamiast cichego przycinania błędów. Fallback bez SciPy
obsługuje płaskie wierzchołki, wybierając ich środkowy indeks. Test pełnej
ścieżki `_spectrum_impl` potwierdza wybór piku mocy `50` przy progu `40`.

### 3.106. Loader rozpoznawał layout, lecz metody FFT zgadywały go ponownie

**Waga:** krytyczna dla małych siatek i wybranych komponentów  
**Status:** naprawione, testy obu metod FFT dodane

`_select_z_layer()` korzystał z kształtu źródłowego i metadanych, aby odróżnić
`(t,z,y,x)` od `(t,y,x,c)`. Po załadowaniu ta informacja była jednak tracona.
Metody 1 i 2 wywoływały `infer_axis_layout()` ponownie na przekształconej
tablicy. Dla siatki, której wszystkie wymiary przestrzenne mają długość 1–4,
ostatni wymiar mógł zostać uznany za przestrzenny zamiast komponentowego.
Skutkiem było uśrednianie komponentów razem z przestrzenią albo pozostawienie
jednego wymiaru przestrzennego jako pozornego komponentu.

Loader zapisuje teraz rozstrzygnięte `spatial_axes` i `component_axis` w
`InputLoadMetrics`. `calculate_fft_data()` przekazuje je do metod 1/2, które
walidują osie i nie wykonują ponownej heurystyki. Metadane wyniku zawierają ten
layout. Publiczny kontrakt `load_data_from_zarr() -> (data, dt)` nie został
zmieniony. Testy obejmują obie metody na wektorze `(t,2,2,3)`.

### 3.107. Jawne metadane layoutu przegrywały z heurystyką kształtu

**Waga:** wysoka  
**Status:** naprawione, test scalar/vector dodany

`infer_axis_layout(shape, attrs)` sprawdzał najpierw `last_dim <= 4`, a dopiero
później metadane komponentów. Jawne `axis_order="tzyx"` mogło więc zostać
zignorowane, np. dla skalarnego `(t,z,y,3)`. Priorytet jest teraz jednoznaczny:
`axis_order`/`axes`/`dims`, potem liczba lub lista komponentów, a dopiero na
końcu heurystyka kształtu.

### 3.108. Brak osi z w danych planarnych był aliasowany do osi czasu

**Waga:** krytyczna dla geometrii, `sel()` i transmission  
**Status:** naprawione, test geometrii i slice'u czasu dodany

`source_spatial_axes()` zwracał dla `(t,y,x,c)` mapę z `z: 0`. Indeks `0` jest
osią czasu, zatem geometria raportowała `Nz=Nt`, a wybór `sel(z=...)` mógł ciąć
czas. Ten sam problem występował dla skalarnego `(t,y,x)` oraz w osobnej
heurystyce K3D.

Brakujące osie są teraz reprezentowane przez `None` i jedną wirtualną komórkę.
Nie uczestniczą w indeksowaniu ani resamplingu. K3D używa wspólnego resolvera,
a dane 1D `(t,x)` otrzymały geometrię `(z=1,y=1,x=Nx)` zamiast całkowitego
braku geometrii.

### 3.109. Slice czasu po downsamplingu zastępował wcześniejszą skalę dt

**Waga:** krytyczna dla osi częstotliwości  
**Status:** naprawione, testy slice/fancy/chaining dodane

Dla materializowanego widoku z `time_step_scale=2`, późniejsze `[::3]`
ustawiało skalę na `3` zamiast `2*3=6`. Przy dalszych łańcuchach używany był
ponadto skomponowany plan źródłowy zamiast lokalnego kroku bieżącego widoku.
Resolver skali przyjmuje teraz lokalny klucz indeksowania i mnoży istniejącą
skalę. Równomierny fancy indexing zachowuje się analogicznie, a nieregularny
zapisuje `nan`, dzięki czemu FFT fail-closed odrzuca niewyrażalny pojedynczym
`dt` przebieg.

### 3.110. Dyspersja wykonywała zwykłe FFT na nieregularnej osi czasu

**Waga:** krytyczna fizycznie  
**Status:** naprawione, wcześniejszy test akceptacji zastąpiony testem odrzucenia

`_time_spacing_from_axis()` przyjmował monotoniczną, lecz nieregularną oś,
zastępował ją średnim `dt` i jedynie dopisywał ostrzeżenie. Zwykłe FFT wymaga
równomiernego próbkowania; średnia nie naprawia położeń próbek, a uzyskana oś
częstotliwości, szerokości i amplitudy pików nie mają wtedy deklarowanej
interpretacji.

Dyspersja wymaga teraz równomierności z tolerancją względną `1e-6` i podaje
komunikat o konieczności resamplingu. Nie używa cichego przybliżenia średnią.

### 3.111. Transmission przypisywało złą wagę wybranemu komponentowi

**Waga:** wysoka dla wartości transmission  
**Status:** naprawione, test normalizacji wag dodany

Dimension-preserving slice komponentu jest zapisywany jako `slice(i,i+1)`, ale
transmission rozpoznawało tylko surowy `int`. Dla wejścia z jednym komponentem
trójka wag `(mx,my,mz)` była następnie przycinana od początku. Wybrane `mz`
mogło więc otrzymać wagę `mx`, a w szczególnych konfiguracjach nawet wagę zero.

Widok jednokomponentowy przy wieloelementowej konfiguracji otrzymuje teraz
jednoznaczną wagę `1`. Jawna pojedyncza waga jest zachowywana. Pojedyncza waga
dla pełnego wektora jest odrzucana zamiast cichego powielania na wszystkie
komponenty.

### 3.112. Transmission wymuszało pełne skanowania garbage collectora

**Waga:** średnia wydajnościowo  
**Status:** naprawione

Ścieżka compute wykonywała siedem `gc.collect()` po usunięciu dużych tablic
NumPy. W CPython ich bufory są zwalniane przez refcount po `del`; globalny GC
skanuje głównie obiekty cykliczne i powodował kosztowne pauzy całego notebooka,
również w ścieżce wielowątkowej. Wywołania usunięto, pozostawiając jawne `del`
w miejscach skracających czas życia buforów.

### 3.113. Granica silnika FFT akceptowała niespójne parametry i wyniki

**Waga:** wysoka dla niezawodności obliczeń  
**Status:** naprawione, testy negatywne dodane

Wspólny executor RFFT nie odrzucał `dt=nan`, pustej osi czasu, wejścia
zespolonego, boolowskiego lub niecałkowitego `nfft` ani nieboolowskiego
`zero_padding`. Część przypadków kończyła się późnym, zależnym od backendu
błędem NumPy/SciPy, a wejście zespolone mogło utracić część urojoną.

Warstwa skalowania nie sprawdzała dodatkowo, czy liczba binów jest równa
`fft_length//2+1`, czy `fft_length >= n_samples`, ani czy statystyki okna są
skończone. Niespójny backend lub mock mógł więc otrzymać pozornie poprawne
skalowanie i metadane.

Executor i scaling fail-closed walidują teraz te kontrakty przed obliczeniem.
Nie dodano kosztownego pełnego skanowania wartości dużej tablicy wejściowej;
kontrola obejmuje typ, kształt, próbnik czasu i spójność wyniku RFFT.

### 3.114. Niejednoznaczne filtry root-level trafiały do złego etapu

**Waga:** wysoka dla semantyki filtrów i cache  
**Status:** naprawione, test normalizacji konfiguracji dodany

`savgol_smooth` i `baseline_correction` są dostępne zarówno przed, jak i po
FFT. `normalize_filter_config()` klasyfikowało root-level opcję do `pre` tylko
dlatego, że zbiór pre był sprawdzany jako pierwszy. Karta notebookowa opisuje
jednak te root-level opcje jako postprocessing. Użytkownik otrzymywał inną
operację fizyczną i niepotrzebne ponowne FFT.

Root-level używa teraz udokumentowanego etapu `post`. Filtr w dziedzinie czasu
pozostaje dostępny bez niejednoznaczności jako `pre={...}`.

### 3.115. Brak SciPy cicho zastępował żądane algorytmy innymi filtrami

**Waga:** wysoka dla odtwarzalności  
**Status:** naprawione, testy bez SciPy dodane

Pre-FFT Savitzky–Golay zwracał niezmienione dane, ALS baseline przechodził w
`remove_mean`, a awaria detrendu liniowego także kończyła się usunięciem
średniej. Metadane nadal deklarowały pierwotny filtr. Oznaczało to, że ten sam
notebook dawał różną metodę zależnie od środowiska bez błędu.

Savitzky–Golay i ALS baseline zgłaszają teraz `ImportError` z instrukcją.
Detrend liniowy ma rzeczywisty, wektorowy fallback NumPy obliczający nachylenie
i wyraz wolny dla każdego śladu. Parametry `lam`, `p`, `niter`, długość okna i
rząd wielomianu są walidowane zamiast przycinane do pozornie poprawnych liczb.

### 3.116. Postprocessing krótkich widm i fallback smoothingu był niespójny

**Waga:** średnia/wysoka  
**Status:** naprawione, testy widm 1–2-binowych i braków SciPy dodane

Savitzky–Golay dla widma z jednym lub dwoma binami budował okno długości 3 i
kończył się późnym błędem SciPy. Brak SciPy zmieniał Gaussian i Savitzky–Golay
na moving average, mimo że są to inne odpowiedzi częstotliwościowe. Bezpośredni
moving average przyjmował parzyste okno i zwracał asymetrycznie przesunięty
wynik, choć dokumentował filtr centrowany.

Widma krótsze niż 3 biny pozostają teraz bez zmian, brak wymaganej implementacji
fail-closed zgłasza `ImportError`, parametry Gaussian/Savitzky są walidowane, a
centrowany moving average wymaga nieparzystego okna.

### 3.117. Faza i animacje modów kolorowały obszar niemagnetyczny

**Waga:** krytyczna dla interpretacji map fazy i klasyfikacji modów  
**Status:** naprawione, syntetyczny test dysku i round-trip Zarr dodane

FFT było wykonywane na prostokątnej macierzy obejmującej także komórki poza
materiałem. Dokładne zera, szum numeryczny lub odziedziczone wartości zespolone
w próżni trafiały następnie do `np.angle()`. Faza zera i faza szumu nie mają
znaczenia fizycznego, ale colormap nadawał im pełny kolor. Problem dotyczył
statycznych map, helpera `ModeResult.plot`, interaktywnego widoku, holografii,
animacji czasowej/fazowej oraz komponentów kołowych i cylindrycznych.

Ten sam brak maski wpływał na obliczenia. Gdy maksimum komponentu było równe
zero, próg amplitudy także wynosił zero, a warunek `amplitude >= threshold`
wybierał całą kwadratową domenę. Próżnia uczestniczyła wtedy w koherencji fazy,
profilach radialnych, winding number i klasyfikatorze vortex.

Dodano wspólny kontrakt maski materiału:

1. priorytet mają zgodne kształtem dane `geom`, `geometry`, `Msat`, `msat` lub
   `Ms`,
2. fallback wykrywa komórki, w których dowolna próbka czasu lub komponent `m`
   ma skończoną wartość powyżej względnego progu numerycznego `1e-12`,
3. maska jest wyznaczana przed odjęciem średniej i FFT, a próżnia jest dokładnie
   zerowana przed obliczeniem modów,
4. nowe cache zapisują `modes/.../material_mask` oraz źródło i aktywną frakcję,
5. stare cache rekonstruują maskę z małej próbki źródłowego `m`; przy wyborze
   jednego komponentu tylko na czas detekcji przywracane są wszystkie komponenty,
6. `FMRModeData` przechowuje maskę, surowe dane obliczeniowe mają zera poza
   materiałem, a faza i renderery korzystają z `MaskedArray`,
7. holografia zwraca RGBA z `alpha=0` poza materiałem,
8. charakterystyka fazy, profile radialne, winding i advanced vortex classifier
   jawnie przecinają własne selekcje z maską materiału.

Publiczny `ModeResult.material_mask` udostępnia maskę `(y,x)` w notebooku.
Maska nie jest wyznaczana z amplitudy pojedynczego modu, ponieważ fizyczne węzły
modu mogą mieć dokładnie zerową amplitudę wewnątrz materiału.

## 4. Dotychczasowa walidacja

Wykonane polecenia:

```text
python3 -m pytest tests/test_dispersion_mode_extraction.py -q
python3 -m pytest tests/test_dispersion_mode_extraction.py tests/test_fft_scaling_and_slicing.py -q
python3 -m pytest tests/test_dispersion_mode_extraction.py tests/test_fft_scaling_and_slicing.py tests/test_modes_interactive_filters.py tests/test_fmr_interactive_toolbar.py -q
python3 -m pytest tests/test_modes_animation_contract.py tests/test_dispersion_mode_extraction.py tests/test_fft_scaling_and_slicing.py tests/test_modes_interactive_filters.py tests/test_fmr_interactive_toolbar.py -q
python3 -m pytest tests/test_spectrum_batch_contract.py tests/test_modes_animation_contract.py tests/test_dispersion_mode_extraction.py tests/test_fft_scaling_and_slicing.py tests/test_modes_interactive_filters.py tests/test_fmr_interactive_toolbar.py -q
python3 -m pytest tests/test_fft_input_layout_contract.py tests/test_spectrum_batch_contract.py tests/test_spectrum_result_contract.py tests/test_modes_animation_contract.py tests/test_dispersion_mode_extraction.py tests/test_fft_scaling_and_slicing.py tests/test_modes_interactive_filters.py tests/test_fmr_interactive_toolbar.py -q
python3 -m compileall -q mmpp
git diff --check
```

Ostatni potwierdzony wynik testu po zmianach:

```text
335 passed, 2 skipped
```

Pominięcia dotyczą opcjonalnych `pyfftw` i `matplotlib`. Ruff nie jest
dostępny w aktualnym systemowym interpreterze. Pełny release gate jest obecnie
ograniczony także przez brak `rich`; lokalna `.venv` nie była wykonywalna.
Osobny `tests/test_spectrum_modes_bridge.py` nie przechodzi discovery w tym
środowisku, ponieważ importuje Matplotlib bezwarunkowo; równoważny test bridge
działa w headlessowym pliku celowanym.

## 5. Obszary nadal audytowane

Poniższe punkty nie są jeszcze uznane za zamknięte:

1. semantyka scalar/vector dla pozostałych nietypowych datasetów 4D,
2. skalowanie PSD/amplitude dla wszystkich okien, zeropaddingu i obu metod FFT,
3. aliasing i jakość próbkowania po downsamplingu czasu,
4. pełna ścieżka odrzucania nieregularnego fancy indexing czasu,
5. zgodność trybów holography/animation z lokalnym ROI,
6. pozostałe ścieżki batch oraz dane zmaterializowane w wielu jobach,
7. koszty pamięci, niepotrzebne kopie i wielokrotne FFT,
8. kompletność helperów `_repr_html_`, `.help`, `.plot` i interaktywnych fallbacków,
9. pełna macierz Python 3.9–3.12, Ruff, mypy, docs i build.

## 6. Stan końcowy tego dokumentu

Raport jest dokumentem roboczym. Audyt pozostaje aktywny, a brak wpisu o błędzie
nie oznacza jeszcze dowodu poprawności danego modułu. Zamknięcie wymaga
przejścia wymagań warstwa po warstwie, testów syntetycznych z analitycznym
wynikiem oraz pełnej walidacji w środowisku zależności deweloperskich.
