# Raport audytu zmian v3.1 — VORTEX_STNO_MODULE_IMPLEMENTATION_PLAN.md

**Data:** 2025-07-17  
**Dokument audytowany:** `docs/VORTEX_STNO_MODULE_IMPLEMENTATION_PLAN.md` (2644 linii, wersja 3.1)  
**Zakres:** Weryfikacja 9 punktów review wykonawczego v3.1

---

## ✅ Co jest dobrze (zaimplementowane poprawnie)

### 1. XYConvention — load-bearing axis convention (§2.4, linie ~100–170)

- ✅ Pełna sekcja §2.4 z dataclassem `XYConvention(y_axis: Literal["up", "down"] = "up")`
- ✅ Funkcja `grid_xy()` z obsługą obu konwencji (y_up: `Y = np.arange(Ny) * dy`, y_down: `Y = (Ny-1-j) * dy`)
- ✅ Kontrakt: *"Wszystkie moduły solitons/vortex/ MUSZĄ przyjmować XYConvention i propagować go do _topology.py"*
- ✅ Lista 3 wymaganych testów konwencji (refleksja, rotacja, round-trip)

### 2. Faza 1 rozbita na 3 PR-y (§9, linie ~1900–2000)

- ✅ **PR1 — Scaffolding + wiring** (8 tasków: 1.1–1.8) z AC:
  - `job[0].m.solitons.vortex` istnieje, alias `.vortex` działa, `repr()` w Jupyter
- ✅ **PR2 — Topology single snapshot** (5 tasków: 1.9–1.13) z AC:
  - `detect()` → TopologyResult z `polarity`, `vorticity`, `chirality`, `chirality_confidence`, `Q`, `is_consistent`
- ✅ **PR3 — Core tracking** (7 tasków: 1.14–1.20) z AC:
  - `track()` → z, r, `rotation_sense`, `instantaneous_frequency`, `confidence`, `plt.xy()`
- ✅ Każdy PR jest mergowalny osobno z niezależnymi acceptance criteria

### 3. Normalizacja m̂ w topology (§5.1.2, linie ~546–570)

- ✅ Blok `Normalizacja m̂ (krok 0)` z kodem: `np.clip(norm, 1e-30, None)` + safe divide
- ✅ Guard: `assert m.shape[-1] >= 3`
- ✅ Komentarz w kodzie wyjaśniający, dlaczego normalizacja jest konieczna (symulacje mogą dawać `|m| ≠ 1`)

### 4. Finite-diff axis convention + arctan2 Berg-Lüscher (§5.1.2, linie ~570–598)

- ✅ Fragment o konwencji osi finite-diff: *"dm_dy interpretowane wg XYConvention.y_axis"*, nota o sign flip
- ✅ Arctan2 BL snippet z `np.einsum` — formuła $\Omega_{ij} = 2\arctan\frac{\mathbf{m}_1 \cdot (\mathbf{m}_2 \times \mathbf{m}_3)}{1 + \text{dots}}$
- ✅ Obie metody (FD i BL) wymienione z informacją kiedy którą stosować

### 5. Winding from complex phase + chirality confidence (§5.1.3–5.1.4, linie ~600–650)

- ✅ **§5.1.3** — Chirality confidence formula: `conf_C = |⟨m_φ⟩| / ⟨|m_φ|⟩` z kodem w Pythonie
- ✅ **§5.1.4** — **Nowa sekcja** "Winding number z fazy in-plane (alternatywna metoda)" z kodem:
  - `phase = np.arctan2(m_y, m_x)` na pierścieniu
  - `w = round(Δφ_total / (2π))`
  - Nota: "Numerycznie stabilniejsza niż FD dla grubych siatek"

### 6. Gaussian fallback + confidence, polarity p(t) (§5.2, linie ~700–810)

- ✅ Kontrakt "Coordinate contract" — `(x, y)` = fizyczne, `(col, row)` = pikselowe
- ✅ Gaussian fallback z kodem:
  - Guard na sigma: `if sigma_x > roi_half * dx or sigma_x < 0.3 * dx: fallback`
  - Confidence: `conf = 1.0 - residual_norm / signal_norm`
  - Automatyczny fallback do centroidu z obniżonym confidence
- ✅ **Polarity time-series** `extract_polarity_series()`:
  - Hysteresis: `threshold_up`, `threshold_down` (~0.3, ~-0.3)
  - Flagi: `p_switch_count`, `switch_times`
  - Nota o zastosowaniu w events/polarity.py

### 7. Cache key standard (§7.3, linie ~1800–1842)

- ✅ Formuła: `key = f"{method}_{hashlib.sha256(config_json.encode()).hexdigest()[:12]}"`
- ✅ Lista parametrów config_json: `mmpp_version`, `module_version`, `dataset_name`, `slice_info`, `dx/dy/dt/Nx/Ny`, parametry metody
- ✅ Example metadata JSON z pełną strukturą
- ✅ Reguła invalidacji: metadata nie zgadza się → automatyczny re-compute + overwrite
- ✅ Referencja w PR3 (task 1.18): `_cache.py` z namespace + hash(config) + metadata JSON

### 8. Trzy krytyczne testy syntetyczne (§11.4, linie ~2274–2348)

- ✅ **Test 1:** `test_berg_luscher_q_sign()` — Q ≈ +0.5 dla p=+1, w=+1; Q ≈ -0.5 dla p=-1; obie konwencje osi
- ✅ **Test 2:** `test_cwccw_from_complex_trajectory()` — CCW: ω > 0, CW: ω < 0; pełny kod z `TrajectoryResult`
- ✅ **Test 3:** `test_gaussian_fallback_at_edge()` — fallback do centroid z niskim confidence; porównanie z centered vortex
- ✅ Nota: *"Te 3 testy najlepiej amortyzują ryzyko, MUSZĄ być gotowe w Faza 1"*
- ✅ Testy przypisane do PR-ów: Test 1 → PR2 (task 1.13), Testy 2+3 → PR3 (tasks 1.19, 1.20)

### 9. Shared _topology.py — mini-wersja w Faza 1 (§4, §9)

- ✅ §4 file tree: `solitons/_topology.py` z komentarzem *"v3.1: Shared topology (mini-wersja od Fazy 1 PR2)"*
- ✅ §9 PR2 task 1.9: `solitons/_topology.py` — 200 LOC, zawiera `topological_density_fd()`, `berg_luscher_Q()`, `guiding_center()`
- ✅ §9 Faza 9 nota: *"Mini-wersja solitons/_topology.py jest tworzona już w Faza 1 PR2. Faza 9 rozszerza ten plik do pełnej wersji"*
- ✅ Task 9.2 przekreślony: *"~~Refactor vortex/topology/ to import from _topology.py~~ — już zrobione w PR2"*

### 10. Tabela podsumowująca + checklist (§13, linie ~2505–2540)

- ✅ Wiersz v3.1 w tabeli: *"XYConvention (load-bearing), Faza1→3PR, normalizacja m̂, arctan2 BL, winding z fazy, chirality confidence, Gauss fallback, polarity p(t), cache key hash+version, 3 krytyczne testy syntetyczne, _topology mini-wersja w Faza 1"*
- ✅ Checklist 15 pozycji (1–15) z rozbiciem PR1/PR2/PR3 dla Fazy 1
- ✅ Shared engine nota: *"mini-wersja od Fazy 1"*
- ✅ Changelog: *"v3.1 (2026-02-10): Review wykonawczy + doprecyzowanie"*

---

## ⚠️ Drobne problemy (nie blokujące, ale warto poprawić)

### 1. Duplikat `_utils.py` w file tree (§4)

W drzewie plików `vortex/` pojawiają się **dwa wpisy** `_utils.py`:

- **Linia ~388**: `├── _utils.py  # v3.1: XYConvention, grid_xy() — jedno źródło prawdy`
- **Linia ~508**: `├── _utils.py  # Współdzielone utility`

Powinien być **jeden plik** `vortex/_utils.py` zawierający zarówno `XYConvention`/`grid_xy()` jak i inne współdzielone utility. Sugestia: usunąć drugi wpis lub scalić komentarze.

### 2. Brak specyfikacji helpera `generate_synthetic_vortex()`

Testy syntetyczne (§11.4) używają helperów:
- `generate_synthetic_vortex(Nx, Ny, p, w)`
- `generate_vortex_mz_near_edge(Nx, Ny, core_pix)`
- `generate_vortex_mz_centered(Nx, Ny, core_pix)`

Nigdzie w planie nie ma sekcji definiującej te helpery (lokalizacja, API, implementacja). Warto dodać task w PR2 lub osobny `tests/conftest.py` / `tests/fixtures/synthetic_vortex.py`.

### 3. Ellipsis `...` w testach syntetycznych

W kodzie testów (§11.4) pojawiają się `...` jako placeholdery:
```python
traj_ccw = TrajectoryResult(time=t, x=x_ccw, y=y_ccw, ...)
result_center = core_track_single_frame(m_z_center, method="gaussian", ...)
```
To jest OK jako pseudokod w planie, ale warto dodać notę że pełna sygnatura będzie w implementacji.

---

## ❌ Czego brakuje (braki niekrytyczne — do uzupełnienia w przyszłych iteracjach)

### 1. Brak implementacji kodu — plan jest dokumentem, nie kodem

Plan v3.1 to dokument projektowy. **Żaden plik Pythona z modułu solitons/vortex nie został jeszcze stworzony.** Jedyny zaimplementowany kod to `mmpp/analytical/thiele.py` (moduł Thiele — §5.6.1).

Status plików:
| Plik | Status |
|------|--------|
| `mmpp/solitons/__init__.py` | ⬜ nie istnieje |
| `mmpp/solitons/_topology.py` | ⬜ nie istnieje |
| `mmpp/solitons/vortex/__init__.py` | ⬜ nie istnieje |
| `mmpp/solitons/vortex/_utils.py` | ⬜ nie istnieje |
| `mmpp/solitons/vortex/topology/` | ⬜ nie istnieje |
| `mmpp/solitons/vortex/core/` | ⬜ nie istnieje |
| `mmpp/analytical/thiele.py` | ✅ zaimplementowany (~900 LOC) |

To jest oczekiwane — plan jest dokumentem architektonicznym, implementacja zaczyna się od Fazy 1 PR1.

### 2. Brak timeline/deadlines

Plan estymuje ~15 tygodni ale nie podaje dat docelowych ani przypisania osób do PR-ów. Nie jest to wymagane na etapie planu, ale przydałoby się przed rozpoczęciem pracy.

### 3. Brak CI/CD pipeline definition

Plan nie definiuje konfiguracji CI (GitHub Actions / pytest workflow) dla nowych testów syntetycznych. Istniejący pipeline w MMPP może wystarczyć, ale warto to potwierdzić.

---

## 📊 Podsumowanie audytu

| Punkt review v3.1 | Sekcja planu | Status |
|---|---|---|
| XYConvention (load-bearing) | §2.4 | ✅ Pełne |
| Faza 1 → 3 PR-y z AC | §9 | ✅ Pełne |
| Normalizacja m̂ | §5.1.2 | ✅ Pełne |
| Finite-diff + arctan2 BL | §5.1.2 | ✅ Pełne |
| Winding z fazy + chirality confidence | §5.1.3–5.1.4 | ✅ Pełne |
| Gaussian fallback + polarity p(t) | §5.2 | ✅ Pełne |
| Cache key standard | §7.3 | ✅ Pełne |
| 3 krytyczne testy syntetyczne | §11.4 | ✅ Pełne |
| Shared _topology.py mini w Faza 1 | §4, §9 | ✅ Pełne |

**Wynik: 9/9 punktów review v3.1 zaimplementowanych w dokumencie planu.**

Drobne problemy (⚠️): 3 — duplikat `_utils.py`, brak specyfikacji helperów testowych, ellipsis w pseudokodzie.

Braki niekrytyczne (❌): 3 — brak kodu implementacyjnego (oczekiwane), brak timeline, brak CI definition.

---

*Raport wygenerowany automatycznie na podstawie audytu dokumentu `VORTEX_STNO_MODULE_IMPLEMENTATION_PLAN.md` v3.1.*
