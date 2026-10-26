# Spójność zapisu Zarr: implementacja i obsługa

Ten opis dotyczy współpracujących wersji MuMax3, AmuMax i MMPP. Nowy protokół
wymaga zarówno aktualnego solwera, jak i zgodnego czytnika. Stary solver nadal
będzie zapisywał wyniki według starego protokołu.

## Kontrakt nowych danych

Nowe pole Zarr v2 ma `commit.json` i katalog `.frames`. Writer najpierw zapisuje
wszystkie skompresowane bloki klatki, a następnie niezmienny rekord z czasem,
nazwami, rozmiarami i SHA-256 bloków oraz hashem poprzedniego rekordu. Na końcu
atomowo publikuje `commit.json` z liczbą zatwierdzonych klatek. Zgłoszona do
kolejki klatka nie jest jeszcze zatwierdzoną klatką. Błąd zapisu zatrzymuje
publikowanie kolejnych klatek w tym polu.

MMPP przypina commit i odtwarza z rekordów wirtualne `.zarray` i `.zattrs`.
Przy odczycie bloku porównuje jego rozmiar i hash z rekordem; brakujący blok
zatwierdzonej klatki jest błędem, a nie domyślną wartością Zarr. Dla tabeli
z manifestem widoczny jest tylko prefiks `committed_rows`, a brak potrzebnego
bloku jest zgłaszany jako błąd. Manifest tabeli v1 nie zawiera jednak hashy
payloadów: nie gwarantuje wykrycia podmienionego, poprawnie dekodującego się
bloku tabeli. Proces czytający
otrzymuje przypięty snapshot — aby zobaczyć kolejne zatwierdzone klatki,
otwórz wynik ponownie. Reader przechowuje czasy i hash rekordu każdej klatki,
lecz jednocześnie buforuje opisy bloków tylko niewielkiej liczby klatek; pamięć
na czasy i hashe nadal rośnie liniowo z liczbą klatek.

Zwykłe `.zarray` i `.zattrs` pola są projekcją zgodności. Są aktualizowane po
opróżnieniu zapisu przy kontrolowanym zamknięciu. Zwykły `zarr.open()` bez
adaptera MMPP podczas pracy solvera lub po SIGKILL może zatem widzieć pustą
lub starszą projekcję, mimo że istnieją zatwierdzone klatki. Dwa pliki projekcji
nie tworzą atomowej transakcji; do bezpiecznego odczytu użyj MMPP albo eksportu
snapshotu opisanego poniżej. Konsolidacja metadanych nie zastępuje commitu.

MuMax3 zapisuje autorytatywny stan uruchomienia w `run.json`; root `.zattrs`
zawiera `run_status_file`. AmuMax publikuje stan w atomowo zapisywanych root
`.zattrs`. `complete`, `interrupted` i `failed` mają różne znaczenie. SIGKILL
nie pozwala wykonać finalizacji ani zmienić stanu na `interrupted`, a ostatni
niezatwierdzony ogon może zostać utracony. Nowe metadane zawierają `run_id`,
wersję schematu oraz dostępne hashe pliku wejściowego i binarium. Hash jest
dowodem tylko wtedy, gdy faktycznie występuje w danym wyniku.

Oba writery próbują przejąć katalog wyniku przez atomowy `mkdir` katalogu
`<wynik>.zarr-owner`. Flaga wymuszająca nie omija tej blokady. Po SIGKILL
blokada może pozostać: przed jej ręcznym usunięciem trzeba ustalić właściciela
i upewnić się, że proces naprawdę nie pracuje. Nie usuwać blokady żywego
procesu. Protokół wymaga poprawnego `mkdir`, rename i sync na danym storage;
nie jest rozproszonym fencingiem przed writerem ignorującym blokadę.

## Diagnoza i snapshot

Diagnoza jest domyślnie tylko do odczytu i weryfikuje payloady:

```bash
python -m mmpp.core.zarr_recovery /ścieżka/wynik.zarr m
```

`--metadata-only` pomija odczyt bloków i nie oznacza wyniku jako zdrowego.
`verified_prefix` bez `head_chain_valid` jest jedynie lokalnie spójnym
prefiksem, którego nie zakotwiczono w aktualnym commicie. Raport danych legacy
podaje także liczbę fizycznych bloków; specyfikacja Zarr dopuszcza ich brak
jako `fill_value`, więc sama różnica liczby bloków nie dowodzi uszkodzenia.

Całkowicie zweryfikowane pole w nowym protokole można wyeksportować do osobnego,
zwykłego Zarr v2:

```bash
python -m mmpp.core.zarr_recovery /ścieżka/wynik.zarr m \
  --export-snapshot /inna/ścieżka/m_snapshot.zarr
```

Eksport weryfikuje wszystkie bloki podczas kopiowania, nie zmienia źródła i
obejmuje wskazane pole wraz z jego czasami oraz atrybutami root. Nie obejmuje
innych pól ani tabel i nie jest checkpointem do wznowienia solvera. Wyniki bez
commitu nie kwalifikują się do takiego zweryfikowanego eksportu. Nadpisanie
eksportu używa atomowej wymiany katalogów Linux `renameat2`; poprzednia wersja
zostaje w ukrytym katalogu `.mmpp-export-*`. Na filesystemie bez tej operacji
zapis kończy się błędem, zachowując stary eksport. Odczyt z weryfikacją wymaga
Linux `/proc/self/fd`; bezpośredni `zarr.open()` nie sprawdza manifestu MMPP.

## Cache i zgodność

Cache FFT, transmission, dyspersji, modes i ogólnego przetwarzania używa
niezmiennych generacji z manifestem i atomowo publikowaną głową. Trafienie
wymaga zgodności rewizji źródła; niekompletna lub uszkodzona generacja nie jest
trafieniem. Stare generacje pozostają na dysku i wymagają osobnego, świadomego
sprzątania. Analiza nie zapisuje manifestu cache w root `.zattrs` symulacji.
Starsze źródła nadal można czytać, ale bez commitu nie używa się trwałego cache
jako wiarygodnego trafienia. W nowym API modes dla takiego źródła należy użyć
`save=False`; `Pyzfn.calc_modes` udostępnia wynik w bieżącej instancji, lecz
nie użyje go ponownie w nowej instancji. `Pyzfn.add_ndarray` udostępnia zapisy
przez Pyzfn jako generacje, nie jako bezpośrednio nadpisaną tablicę źródła.

Nieaktywne w badanym kodzie `zarr.WriteSeries` w MuMax3 zostało wycofane:
wywołanie zwraca błąd i nie zmienia danych. Nowy kod tabel powinien korzystać
z `tabular.Writer` i jego manifestu. To celowa zmiana publicznego zachowania.
Obecny adapter MMPP jest przeznaczony dla deklarowanego przez pakiet Zarr 2.x.

## Potwierdzenie i ograniczenia kwalifikacji

Przeszły ukierunkowane testy Go i Python, testy `-race` modułów zapisu,
wstrzykiwanie wybranych błędów oraz rzeczywiste lokalne uruchomienia GPU
MuMax3 i AmuMax: normalny zapis, SIGTERM i SIGKILL z weryfikacją zatwierdzonego
prefiksu. Przeszły również lokalne próby SIGINT i SIGTERM w `Relax` i
`Minimize` dla obu solverów, z jedną wcześniej zatwierdzoną klatką. Dla Lustre
wykonano
test procesu zabijanego w punktach publikacji pojedynczego pliku oraz test
blokady `flock` i konkurencyjnego `mkdir` między dwoma węzłami. Nie wykonano
testu GPU z nowym binarium na tym systemie ani testu zaniku hosta/zasilania. Nie
zweryfikowano wszystkich przypadków ENOSPC, EDQUOT, EIO, short write, błędu
close/fsync, niepewnej odpowiedzi HTTP oraz przerwań w każdej fazie Relax i
Minimize. Nie wykonano pełnej macierzy Python 3.10–3.12 ani rollout'u.

Poglądowy lokalny mikrobenchmark na `/tmp` ext4 porównał 64 klatki pola
`96×96×2×3`, 8 bloków na klatkę, trzy serie z `fsync` i finalną projekcją.
Mediany czasu: starsza ścieżka 2,864 s, commit 2,927 s. To niewielka próbka,
bez obciążenia klastra i bez pomiaru długich symulacji; nie jest prognozą dla
Lustre. Obecny AmuMax przed commitem ponownie odczytuje bloki do sprawdzenia
hasha, co może zwiększyć I/O. Potrzebny jest benchmark docelowego workloadu,
obserwowalność opóźnień i kontrolowany rollout przed wydaniem.

Zmiana nie rekonstruuje automatycznie uszkodzonych wcześniejszych wyników.
Metadanych innego uruchomienia nie należy traktować jako dowodu metadanych
archiwum, którego integralność nie została potwierdzona.
