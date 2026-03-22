# Plan wdrożenia autofitu parametrów analitycznych dla dynamiki vortexa

> **Cel:** zbudować produkcyjny mechanizm automatycznego dopasowania parametrów modeli analitycznych vortexa do trajektorii numerycznych z MuMax3/MMPP.
> **Zakres v1:** pojedyncza symulacja, porównanie do modelu Thiele CPP/CIP, fit oparty o fizykę i optymalizację numeryczną.
> **Zakres v2+:** fit po sweepie prądu, wspólne parametry dla wielu jobów, akceleracja GPU i ewentualny surrogate model.

---

## 1. Problem do rozwiązania

Aktualne API pozwala już:

- wyciągać trajektorię vortexa z `table`, `magnetization` albo `auto`,
- budować trajektorię analityczną przez `job.vortex.plt.orbit().add_analytics(...)`,
- porównywać orbitę numeryczną i analityczną (`VortexAnalyticalComparison` w `plotting.py`),
- rozwiązywać parametry z `attrs -> .mx3 -> manual overrides` (przez `bridge.extract.model_defaults()`).

To nie wystarcza do solidnej zgodności model-symulacja, bo:

- część parametrów modelu jest dziś ustawiana heurystycznie albo przez defaulty,
- parametry efektywnego STT są niedokalibrowane względem realnej symulacji,
- przypadki z Oerstedem i sweepem po prądzie wymagają fitu wspólnego, a nie tylko pojedynczego overlayu,
- obecny `bridge.fit.thiele_from_trajectory(...)` → `model/thiele/fit.py:fit_from_trajectory()` jest tylko prostym proxy (center = mean, radius = mean|z|, omega = median(d(phase)/dt), N=0.0 bez optymalizacji).

Najważniejszy wniosek architektoniczny:

- **najlepsza wersja nie zaczyna się od ML**, tylko od **physics-informed autofit** z dobrym loss function, ograniczeniami fizycznymi i wieloetapową optymalizacją;
- GPU i ML mają sens jako **akceleracja** albo **surrogate**, nie jako pierwszy backend.

---

## 2. Założenia projektowe

### 2.1. Zasady

- Nie psuć istniejącego API `add_analytics(...)` w `plotting.py`.
- Oddzielić:
  - ekstrakcję parametrów wejściowych (reuse `bridge/extract.py`),
  - budowę modelu analitycznego (reuse `model/thiele/cpp.py`, `cip.py` → `mmpp.analytical.thiele`),
  - ekstrakcję cech z trajektorii,
  - definicję lossu,
  - sam optimizer,
  - diagnostykę i wizualizację wyniku.
- Zachować zgodność z obecnym podziałem:
  - `bridge/` → glue i parameter resolution,
  - `model/thiele/` → adapter do `mmpp.analytical.thiele.CPPThieleModel` / `CIPThieleModel`,
  - `nonlinear/` → analityka fizyczna (ST parameters, force balance, `ThieleAnalyzer`),
  - `plotting.py` → `VortexPlotInterface` / `VortexOrbitPlotHandle` jako high-level UX.

### 2.2. Priorytety jakości

1. Poprawność fizyczna.
2. Czytelna diagnostyka, dlaczego fit się udał lub nie.
3. Reprodukowalność.
4. Szybkość.
5. GPU/ML dopiero po ustabilizowaniu modelu i lossu.

---

## 3. Istniejąca infrastruktura (stan repozytorium)

### 3.1. Łańcuch dostępu z poziomu użytkownika

```
job[i].vortex                  → VortexInterface (solitons/vortex/interface.py)
  .topology                    → TopologyInterface (numerical/topology/)
  .core                        → CoreInterface (numerical/core/)
  .trajectory                  → TrajectoryInterface (trajectory/)
  .spectrum                    → VortexSpectrumInterface (spectrum/)
  .modes                       → VortexModesInterface (numerical/modes/)
  .nonlinear                   → NonlinearInterface (numerical/nonlinear/)
    .thiele                    → ThieleAnalyzer (nonlinear/nonliniearthiele.py)
      .force_balance()
      .simulate_cpp() / .simulate_cip()
      .fit_omega0_N_to_fJ()   → mmpp.analytical.fit_omega0_N_to_fJ()
      .threshold_current_dc()
      .predict_frequency_dc()
    .slavin_tiberkevich()      → extract_st_parameters()
    .amplitude_equation()
  .events                      → EventsInterface (numerical/events/)
  .signals                     → SignalsInterface (numerical/signals/)
  .energy                      → EnergyInterface (numerical/energy/)
  .model                       → VortexModelInterface (model/)
    .thiele                    → ThieleModelNamespace
      .cpp(**kwargs)           → CPPModelAdapter → mmpp.analytical.CPPThieleModel
      .cip(**kwargs)           → CIPModelAdapter → mmpp.analytical.CIPThieleModel
  .bridge                      → BridgeInterface (bridge/)
    .compare.with_(lhs, rhs)
    .fit.thiele_from_trajectory()  → model/thiele/fit.py (prosty proxy, N=0)
    .extract.model_defaults()      → AnalyticalParameterResolution
  .plt / .plot                 → VortexPlotInterface (plotting.py)
    .orbit()                   → VortexOrbitPlotHandle
      .add_analytics(...)      → VortexAnalyticalComparison

job[:].vortex                  → BatchVortexInterface (solitons/batch.py)
  .summary() / .regimes()
  .spectrum_map()
  .plt.dashboard() / .plt.regimes() / .plt.spectrum_map()
```

### 3.2. Silnik fizyczny: `mmpp.analytical.thiele`

Główny moduł fizyczny (`mmpp/analytical/thiele.py`, ~2000 linii):

- **`MaterialParams`** — Ms, alpha, P, A, beta_nonadiabatic, gamma
- **`DiskGeometry`** — R, L, core_diameter; metoda `Rc(mat)`
- **`ExternalField`** — Bx_T, By_T, Bz_T
- **`FieldCalibration`** — domega0_dBz, seq_per_T, chirality; metody omega0_shift(), s_eq()
- **`CPPThieleModel`** — ODE Thiele'a z nieliniowym tłumieniem d(u)=d₀+d₁u², nieliniową częstotliwością ω(u)=ω₀(1+Nu²), STT pumping χ(J)=chi_scale·γσJ/2; metody `simulate()`, `steady_state_radius()`, `threshold_current()`, `predict_frequency_dc()`; symulacja SDE (Euler-Maruyama)
- **`CIPThieleModel`** — ODE Thiele'a dla geometrii Zhang-Li STT
- **`omega0_novosad()`** — estymacja ω₀ z Novosad formula
- **`fit_omega0_N_to_fJ()`** — fit ω₀ i N (opcjonalnie dω₀/dJ i chi_scale) do danych f(J); dwuetapowy: coarse grid scan + L-BFGS-B
- **`slonczewski_mtj_efficiency(Pol, Lambda, cos_theta)`** — efektywność Slonczewskiego

### 3.3. Istniejący fit-related code (podsumowanie)

| Lokalizacja | Funkcja | Co robi |
|---|---|---|
| `model/thiele/fit.py` | `fit_from_trajectory()` | Prosty proxy: center=mean, radius=mean\|z\|, omega=median(dphi/dt). **Brak optymalizacji, N=0.** |
| `bridge/fit.py` | `fit_thiele_from_trajectory()` | Thin wrapper → powyższe |
| `nonlinear/nonliniearthiele.py` | `ThieleAnalyzer.fit_omega0_N_to_fJ()` | Delegacja → `mmpp.analytical.fit_omega0_N_to_fJ()` (grid+L-BFGS-B) |
| `nonlinear/slavin_tiberkevich.py` | `extract_st_parameters()` | Fit liniowy ω(p)=ω₀+Np → STParametersResult |
| `trajectory/orbit.py` | `fit_orbit_ellipse()` | Fit elipsy do orbity → OrbitFitResult |

### 3.4. Duplikacja nonlinear

**Uwaga architektoniczna:** W drzewie istnieją dwie ścieżki `nonlinear`:
- `mmpp/solitons/vortex/nonlinear/` — importowana przez `VortexInterface` (via `numerical.nonlinear`)
- `mmpp/solitons/vortex/numerical/nonlinear/` — fizycznie ten sam katalog (symlink lub re-export)

Autofit powinien korzystać z `nonlinear/` poprzez ustabilizowane publiczne API (`VortexInterface.nonlinear`), nie poprzez bezpośrednie importy wewnętrzne.

---

## 4. Docelowe API

### 4.1. API dla pojedynczego joba

```python
fit = job.vortex.autofit.thiele(
    trajectory="steady_state",
    tracking_source="auto",
    model="auto",                          # "cpp" | "cip" | "auto"
    params="auto",                         # "auto" → bridge.extract.model_defaults()
    current="i_pillar_ma",
    fit_params=("omega0", "N", "chi_scale"),
    objective="hybrid",                    # "time" | "spectral" | "hybrid"
)
```

```python
fit.best_params                            # dict[str, float]
fit.initial_params                         # dict[str, float] — wartości startowe przed optymalizacją
fit.metrics                                # VortexAutofitMetrics
fit.loss_total                             # float
fit.loss_breakdown                         # dict[str, float]
fit.comparison.plt.orbit_overlay()         # reuse VortexAnalyticalComparison
fit.comparison.plt.time_traces()
fit.comparison.plt.st_dashboard()
fit.diagnostics                            # AutofitDiagnostics
fit.plt.convergence()                      # wykres historii lossu
fit.plt.parameter_sensitivity()            # optional
```

### 4.2. API zintegrowane z istniejącym overlayem

```python
cmp = job.vortex.plt.orbit(
    trajectory="steady_state",
    tracking_source="table",
).add_analytics(
    params="auto",
    model="auto",
    current="i_pillar_ma",
    fit="auto",                            # NOWY parametr
)
```

Interpretacja `fit=`:

- `fit=False` lub brak: obecne zachowanie, bez dopasowania.
- `fit=True` lub `fit="auto"`: physics-informed autofit z domyślnym zestawem parametrów.
- `fit={"fit_params": (...), "objective": ...}`: pełna konfiguracja dopasowania.

**Uwaga implementacyjna:** Obecny `add_analytics()` w `plotting.py` (linia ~652) buduje model adapter przez `_simulate_matching_trajectory()` i zwraca `VortexAnalyticalComparison`. Parametr `fit=` powinien wstawiać etap autofit **między** resolution parametrów a symulacją analityczną.

### 4.3. API dla sweepu po prądzie

```python
batch_fit = job[:].vortex.autofit.sweep(
    model="cpp",
    shared_params=("omega0", "N", "chi_scale"),
    per_job_params=("phase0", "center_x", "center_y"),
    trajectory="steady_state",
)
```

**Uwaga:** `res.vortex` → `BatchVortexInterface` (z `batch.py`). Sweep fit wymaga dodania `.autofit` do `BatchVortexInterface`, nie tylko do `VortexInterface`.

To API nie musi wejść w v1, ale dokument i struktura modułu mają je umożliwić bez refaktoru.

### 4.4. API a istniejący `fit_omega0_N_to_fJ`

`ThieleAnalyzer.fit_omega0_N_to_fJ()` fituje parametry z danych f(J) (dane skalarnie z wielu symulacji), natomiast autofit fituje na poziomie **pełnej trajektorii x(t), y(t)** z jednej symulacji. To są komplementarne, nie zastępowalne podejścia:

- `fit_omega0_N_to_fJ` → sweep-level, scalar f(J) relationship
- `autofit.thiele()` → single-job, full trajectory matching

W sweep fitowaniu (Faza 4), `fit_omega0_N_to_fJ` może być jednym ze składników lossu.

---

## 5. Proponowana architektura modułu

### 5.1. Nowy namespace

Dodać nowy pakiet:

```text
mmpp/solitons/vortex/autofit/
├── __init__.py          # eksport publiczny: AutofitInterface, AutofitConfig, VortexAutofitResult
├── interface.py         # AutofitInterface — accessor z .thiele() i .sweep()
├── config.py            # AutofitConfig, ParameterSpec (bounds, priors, freeze)
├── result.py            # VortexAutofitResult, AutofitDiagnostics, VortexAutofitMetrics
├── features.py          # TrajectoryFeatureExtractor — ekstrakcja cech z trajektorii
├── losses.py            # LossFunction, TimeLoss, SpectralLoss, HybridLoss, ForceBalanceLoss
├── optimizers.py        # OptimizationPipeline — init → global → local → uncertainty
├── single.py            # run_single_job_fit() — orkestracja pipeline'u dla 1 joba
├── _plotting.py         # AutofitPlotAccessor — convergence, sensitivity, param space
└── sweep.py             # run_sweep_fit() — etap późniejszy (Faza 4)
```

**Zmiana vs. oryginał:**
- `models.py` → `result.py` (uniknąć kolizji nazw z `model/thiele/models.py` i `nonlinear/models.py`)
- Dodany `_plotting.py` (konwencja `_plotting.py` jest spójna z resztą codebase: `model/thiele/_plotting.py`, `core/_plotting.py`, `topology/_plotting.py`)
- Usunięty `surrogate.py` i `gpu.py` z fazy 0 (dodawać dopiero gdy potrzebne, a nie jako puste pliki)

### 5.2. Uzasadnienie

Nie rozszerzać dalej `bridge/fit.py`, bo ten moduł jest thin wrapper (3 linie kodu) do `model/thiele/fit.py`, który sam jest prostym proxy.
Nie rozszerzać `nonlinear/nonliniearthiele.py`, bo `ThieleAnalyzer` ma już 597 linii i inną odpowiedzialność (force balance, single-shot simulation, f(J) fit).

Pełny autofit będzie miał:

- własne modele danych,
- własną konfigurację,
- własny pipeline optymalizacji,
- osobne testy i diagnostykę.

To jest już osobny subsystem, nie helper.

### 5.3. Integracja z istniejącymi modułami

```
mmpp/solitons/vortex/interface.py
  └─ dodać property .autofit → AutofitInterface (lazy import)

mmpp/solitons/vortex/plotting.py
  └─ VortexOrbitPlotHandle.add_analytics():
     dodać parametr fit= ; gdy fit≠False, wywołać autofit przed symulacją

mmpp/solitons/vortex/bridge/extract.py
  └─ używać extract_model_defaults() → AnalyticalParameterResolution
     jako źródło initial_params i priors

mmpp/analytical/thiele.py
  └─ CPPThieleModel / CIPThieleModel pozostają backendem solvera (BEZ zmian)
  └─ fit_omega0_N_to_fJ() — reuse w sweep (Faza 4)

mmpp/solitons/vortex/model/thiele/{cpp,cip}.py
  └─ adaptery modelu — autofit buduje model przez te adaptery
  └─ fit.py (fit_from_trajectory) — zachować jako legacy, nie rozszerzać

mmpp/solitons/vortex/nonlinear/
  └─ extract_st_parameters() — reuse w features.py do ekstrakcji ST features
  └─ ThieleAnalyzer.force_balance() — reuse jako składnik lossu (L_force_balance)

mmpp/solitons/batch.py
  └─ BatchVortexInterface: dodać .autofit property (Faza 4)
```

---

## 6. Modele danych

### 6.1. `AutofitConfig`

```python
@dataclass
class AutofitConfig:
    # --- Trajectory resolution ---
    trajectory: str = "steady_state"                # "full" | "steady_state"
    tracking_source: str = "auto"                   # "auto" | "table" | "magnetization"
    tracking_method: str | None = None              # None → config default

    # --- Model ---
    model: str = "auto"                             # "auto" | "cpp" | "cip"
    params: str | dict = "auto"                     # "auto" → bridge.extract.model_defaults()
    current: str | float | None = None              # key w attrs albo wartość w A/m²

    # --- Fit parameters ---
    fit_params: tuple[str, ...] = ("omega0", "N", "chi_scale")
    param_specs: dict[str, ParameterSpec] | None = None   # bounds, priors, freeze per param

    # --- Objective ---
    objective: str = "hybrid"                       # "time" | "spectral" | "hybrid"
    weights: dict[str, float] | None = None         # override domyślnych wag lossu

    # --- Optimization ---
    global_search: bool = True                      # czy robić global stage
    global_method: str = "differential_evolution"   # "differential_evolution" | "sobol"
    global_maxiter: int = 50
    local_method: str = "L-BFGS-B"                  # "L-BFGS-B" | "least_squares"
    local_maxiter: int = 200

    # --- Phase alignment ---
    align_phase: bool = True
    align_center: bool = True

    # --- Windowing ---
    windowing: str = "steady_state"                 # "full" | "steady_state" | "last_n_periods"
    n_periods: int = 10                             # dla windowing="last_n_periods"

    # --- Reproducibility ---
    random_seed: int | None = None

    # --- Field handling (Faza 2+) ---
    allow_oersted: bool = False
    allow_field_fit: bool = False
```

```python
@dataclass
class ParameterSpec:
    """Specyfikacja pojedynczego parametru fitowalnego."""
    lower: float = -np.inf
    upper: float = np.inf
    initial: float | None = None           # None → infer from physics
    prior_mean: float | None = None
    prior_std: float | None = None
    prior_type: str = "gaussian"           # "gaussian" | "log_normal" | "uniform"
    frozen: bool = False                   # True → nie fitować, użyć initial/resolved
    scale: float = 1.0                     # normalizacja dla optimizera
```

### 6.2. `VortexAutofitResult`

```python
@dataclass
class VortexAutofitResult:
    best_params: dict[str, float]
    initial_params: dict[str, float]
    param_sources: dict[str, str]          # skąd wzięto initial: "novosad", "attrs", "mx3", "heuristic"
    frozen_params: dict[str, float]
    fitted_params: tuple[str, ...]

    loss_total: float
    loss_breakdown: dict[str, float]       # {"L_xy_time": ..., "L_radius": ..., ...}
    baseline_loss: float                   # loss PRZED fitem (z initial_params)

    comparison: VortexAnalyticalComparison  # reuse istniejącego typu z plotting.py
    diagnostics: AutofitDiagnostics

    success: bool
    warnings: list[str]
    config: AutofitConfig

    @property
    def plt(self) -> AutofitPlotAccessor: ...

    @property
    def improvement_ratio(self) -> float:
        """loss_total / baseline_loss — <1 oznacza poprawę."""
        if self.baseline_loss == 0:
            return float('inf')
        return self.loss_total / self.baseline_loss
```

**Dodane vs. oryginał:**
- `baseline_loss` — kluczowe do oceny, czy fit rzeczywiście pomógł
- `improvement_ratio` — property dla szybkiego sprawdzenia
- `config` — zachowanie pełnej konfiguracji w wyniku (reprodukowalność)
- Usunięty `metrics` jako osobne pole (metryki są w `comparison.metrics`, nie duplikować)
- Usunięty `optimizer_report` (przeniesiony do `diagnostics`)

### 6.3. `AutofitDiagnostics`

```python
@dataclass
class AutofitDiagnostics:
    n_evaluations: int
    n_global_evaluations: int
    n_local_evaluations: int
    time_total_s: float
    time_global_s: float
    time_local_s: float

    # Optimizer internals
    optimizer_message: str
    optimizer_nit: int
    loss_history: list[float]              # loss po każdej ewaluacji (do wykresu convergence)

    # Parameter identifiability
    hessian_approx: np.ndarray | None      # numeryczny Hessian w optimum
    param_correlations: np.ndarray | None  # macierz korelacji parametrów
    param_uncertainties: dict[str, float] | None  # ±1σ z diagonali Hessiana
    poorly_identified: list[str]           # parametry z dużą niepewnością

    # Constraint activity
    active_bounds: dict[str, str]          # np. {"chi_scale": "lower"} gdy param na granicy
```

---

## 7. Cechy numeryczne do dopasowania

Autofit nie powinien patrzeć tylko na `x(t)` i `y(t)`. Powinien budować zestaw cech z trajektorii numerycznej i analitycznej.

### 7.1. `TrajectoryFeatureExtractor`

Centralna klasa ekstrakcji cech. Przyjmuje `TrajectoryResult` lub surowy `(t, x, y)`, zwraca `TrajectoryFeatures`.

### 7.2. Cechy czasowe (V1)

- `x(t)`, `y(t)` — po resamplingu na wspólną siatkę
- `r(t) = |z(t) - z_center|` — po odjęciu centrum
- `phi(t)` — unwrapped phase
- `f_inst(t)` — z gradientu fazy (reuse logiki z `fit_from_trajectory`)
- trend amplitudy — liniowy fit do `r(t)` w oknie steady-state
- center drift — `d(center)/dt`

### 7.3. Cechy spektralne (V1)

- częstotliwość głównego piku — z PSD (reuse `ThieleTrajectoryResult.dominant_frequency_ghz`)
- moc głównego piku
- szerokość piku (FWHM)
- harmoniczne (2f, 3f)
- zgodność PSD — norma różnicy PSD w oknie wokół głównego piku

### 7.4. Cechy geometryczne orbity (V1)

- średni promień — mean(r) w steady-state
- maksymalny promień
- eliptyczność — reuse `fit_orbit_ellipse()` z `trajectory/orbit.py`
- przesunięcie centrum — (cx, cy) vs (0, 0)
- orientacja elipsy

### 7.5. Cechy fizyczne (V2+)

- ST parameters — reuse `extract_st_parameters()` z `nonlinear/slavin_tiberkevich.py`
- force-balance residual — reuse `ThieleAnalyzer.force_balance()` z `nonlinear/nonliniearthiele.py`
- zgodność progu auto-oscylacji — `J_sim_threshold` vs `CPPThieleModel.threshold_current()`
- zgodność reżimu — stable / damped / collision / switching

---

## 8. Funkcja celu

### 8.1. Loss hybrydowy

Domyślny loss powinien być wieloskładnikowy:

```text
L_total =
  w_xy       * L_xy_time           # MSE x(t),y(t) (po align)
  + w_r      * L_radius            # MSE r(t) lub |<r_num> - <r_ana>|²
  + w_phi    * L_phase             # MSE unwrapped phi(t) (po align)
  + w_freq   * L_frequency         # |f_num - f_ana|² / f_num²
  + w_psd    * L_spectrum          # norma różnicy PSD w oknie głównego piku
  + w_ellip  * L_ellipticity       # |e_num - e_ana|² (eliptyczność orbity)
  + w_reg    * L_regularization    # prior penalties
```

**Zmiana vs. oryginał:**
- Usunięty `L_slavin_tiberkevich` i `L_force_balance` z domyślnego hybridu V1 — to ciężkie obliczeniowo cechy, które wymagają pełnej force decomposition. Dodać jako opcje w V2.
- Dodany `L_ellipticity` — tania cecha geometryczna, dobrze łapie asymetrię orbity.

### 8.2. Ważne reguły

- Wszystkie składniki lossu muszą być skalowane do porównywalnych rzędów wielkości (normalizacja per-feature).
- Porównanie czasowe musi być robione po:
  - wyborze wspólnego okna czasowego (steady-state lub last_n_periods),
  - wyrównaniu fazy (minimalizacja `|phi_num(t0) - phi_ana(t0)|`),
  - wyrównaniu centrum (odejmowanie mean(x), mean(y) z numerycznej trajektorii),
  - resamplingu na tę samą siatkę czasu.
- Trzeba obsłużyć przypadek, gdy numeryka nie jest jeszcze pełnym limit cycle (damped transient, nie osiągnięto steady-state → użyć `trajectory="full"` z krótszym oknem).

### 8.3. Tryby lossu

```python
LOSS_PRESETS = {
    "time": {
        "w_xy": 1.0, "w_r": 0.5, "w_phi": 0.3, "w_freq": 0.2,
        "w_psd": 0.0, "w_ellip": 0.1, "w_reg": 0.01,
    },
    "spectral": {
        "w_xy": 0.0, "w_r": 0.2, "w_phi": 0.0, "w_freq": 1.0,
        "w_psd": 0.8, "w_ellip": 0.1, "w_reg": 0.01,
    },
    "hybrid": {
        "w_xy": 0.5, "w_r": 0.3, "w_phi": 0.2, "w_freq": 0.5,
        "w_psd": 0.3, "w_ellip": 0.1, "w_reg": 0.01,
    },
}
```

**Uwaga:** Wagi finalne wymagają tuning'u na realnych danych. Powyższe to rozsądne wartości startowe. Użytkownik może nadpisać przez `weights={...}`.

---

## 9. Parametry fitowane i priorytety

### 9.1. V1: przypadek CPP bez Oersteda

Fitowane domyślnie:

- `omega0` — eigenfrequency ω₀ [rad/s]
- `N` — nonlinear frequency shift coefficient
- `chi_scale` — STT pumping scaling factor

Opcjonalnie (domyślnie frozen):

- `phase0` — początkowa faza trajektorii analitycznej
- `center_x`, `center_y` — przesunięcie centrum orbity
- `d0_scale` — skala efektywnego tłumienia (default 1.0, dla kalibracji d₀)

**Zmiana vs. oryginał:** `damping_eff` zmieniony na `d0_scale` — model CPP liczy `d₀` analitycznie z `alpha` i `R/Rc`. Fitowanie `alpha` bezpośrednio łamałoby fizyczny sens (alpha jest znane z symulacji). Zamiast tego fitujemy multiplikator `d0_scale` na obliczone d₀.

### 9.2. V2: CPP z Oerstedem / field

Dodać możliwość dopasowania:

- `domega0_dJ` — w modelu CPP już istnieje jako parametr; odpowiada `omega0_Oe_per_J` z planu; **użyć nazwy z kodu**: `domega0_dJ`
- `field_cal.domega0_dBz` — kalibracja Bz → ω₀
- `field_cal.seq_per_T` — kalibracja pola in-plane → equilibrium shift

**Zmiana vs. oryginał:** `omega0_Oe_per_J` → `domega0_dJ` (zgodność z `CPPThieleModel.__init__` parameter name); `By_bias_eff` → `field_cal.seq_per_T` (reuse istniejącej `FieldCalibration`); `field_calibration` → rozbite na dwa konkretne parametry.

### 9.3. V3: efektywna Slonczewski STT

Rozszerzyć parametry o:

- `Lambda` — parametr asymetrii Slonczewskiego (reuse `slonczewski_mtj_efficiency()`)
- `epsilonprime` — wtórny moment STT

**Uwaga architektoniczna:** `Lambda` i `epsilonprime` nie są dziś parametrami `CPPThieleModel`. Ich integracja wymaga:
1. Rozszerzenia `CPPThieleModel` o opcjonalny `slonczewski_efficiency_func` callback, albo
2. Przeliczenia `Lambda, epsilonprime → chi_scale_eff(theta)` w autofit i przekazania efektywnego `chi_scale`.

Rekomendacja: podejście (2) w V3 — autofit przelicza Lambda/epsilon na efektywny chi_scale i przekazuje do istniejącego modelu. Refaktor CPPThieleModel dopiero gdy podejście (2) okaże się niewystarczające.

**Usunięte z oryginału:** `pz_eff` (polaryzacja jest częścią `MaterialParams.P` i `CPPThieleModel._sigma` — fitowanie osobnego `pz_eff` jest redundantne z `chi_scale`). Parametry pinningu i asymetrii orbity — to jest beyond Thiele model, doda złożoność bez jasnego ROI.

### 9.4. Priory i bounds

Każdy parametr fitowany musi mieć:

- zakres fizyczny (hard bounds),
- wartość startową (z `bridge.extract` lub heurystyki),
- opcjonalny prior (penalty w lossu, nie hard constraint),
- możliwość zamrożenia przez użytkownika.

Domyślne bounds i inicjalizacja:

| Parametr | Lower | Upper | Initial | Prior |
|---|---|---|---|---|
| `omega0` | 0.1·ω₀_novosad | 5·ω₀_novosad | ω₀_novosad lub z PSD | log-normal(μ=ω₀_novosad, σ=0.3) |
| `N` | -0.5 | 2.0 | 0.25 (Guslienko) lub z ST fit | gaussian(μ=0.25, σ=0.3) |
| `chi_scale` | 0.1 | 10.0 | 1.0 | log-normal(μ=1.0, σ=0.5) |
| `phase0` | -π | π | z pierwszej próbki | uniform |
| `center_x` | -0.5R | 0.5R | mean(x) | gaussian(μ=mean_x, σ=0.1R) |
| `center_y` | -0.5R | 0.5R | mean(y) | gaussian(μ=mean_y, σ=0.1R) |
| `d0_scale` | 0.3 | 3.0 | 1.0 | log-normal(μ=1.0, σ=0.3) |
| `domega0_dJ` | -1e-4 | 1e-4 | 0.0 | gaussian(μ=0, σ=1e-5) |

**Zmiana vs. oryginał:** Dodana pełna tabela z konkretnymi wartościami, a nie tylko reguły ogólne. Bounds na `center` zmienione z `0.25R` na `0.5R` (numeryczna trajektoria może mieć większe przesunięcie centrum niż 0.25R, szczególnie z polem).

---

## 10. Strategia optymalizacji

### 10.1. Etap 0: physics-informed initialization

Z trajektorii numerycznej wyznaczać:

- `omega0_init` — priorytet: (1) `bridge.extract.model_defaults().omega0`, (2) z głównego piku PSD, (3) z mediany d(phi)/dt (reuse logiki z `fit_from_trajectory`), (4) `omega0_novosad()`
- `center_init` — z mean(x), mean(y) na oknie steady-state
- `phase0_init` — z pierwszej próbki w oknie
- `chi_scale_init` — 1.0 (default; ewentualnie z porównania predicted vs actual steady-state radius)
- `N_init` — (1) z `extract_st_parameters().N`, (2) 0.25 (Guslienko default)

To ma skrócić koszt późniejszego fitu i poprawić convergence rate.

### 10.2. Etap 1: global search (opcjonalny)

Dla parametrów silnie nieliniowych i gdy `global_search=True`:

- **`scipy.optimize.differential_evolution`** (domyślna) — robust, nie wymaga gradientu, dobrze radzi sobie z wieloma basinami atrakcji
- Alternatywa: Sobol / Latin hypercube + ranking top-K kandydatów

Konfiguracja:
- `maxiter=50` (domyślnie), `popsize=15`
- `workers=-1` (parallel, jeśli solver jest thread-safe — `solve_ivp` jest)
- `seed=random_seed` dla reprodukowalności

Ten etap nie musi być bardzo dokładny. Ma znaleźć dobry basin of attraction.

**Gdy `global_search=False`:** pomiń i przejdź od razu do local refinement z physics-informed startpoint. Sensowne gdy user wie, że initial params są bliskie optimum.

### 10.3. Etap 2: local refinement

Lokalny refine z najlepszego punktu z etapu 1 (lub z init jeśli brak global):

- **`scipy.optimize.minimize(method="L-BFGS-B")`** (domyślna) — bounded, gładki loss, gradient numeryczny
- Alternatywa: **`scipy.optimize.least_squares(method="trf")`** — gdy loss jest wektorowy (residual vector), lepsza convergence dla dużych residuali

Wybór automatyczny:
- `objective="time"` → `least_squares` (naturalny residual vector: `x_num(t) - x_ana(t)`)
- `objective="spectral"` lub `"hybrid"` → `L-BFGS-B` (loss skalarny)

### 10.4. Etap 3: uncertainty / identifiability (opcjonalny, V1+)

Po znalezieniu optimum:

- Hessian aproksymowany numerycznie (`scipy.optimize.approx_fprime` lub `numdifftools`)
- Macierz korelacji parametrów z odwrotności Hessiana
- Parametry słabo identyfikowalne: |correlation| > 0.95 lub σ/|param| > 1.0
- Raport w `AutofitDiagnostics`

**Nie robić** bootstrap w V1 (kosztowne, niski ROI na tym etapie).

---

## 11. GPU i ML: kiedy i jak

### 11.1. Czego nie robić na początku

Nie zaczynać od czarnej skrzynki typu:

- losowy regressor NN przewidujący `omega0`, `N`, `chi_scale`,
- pełne zastąpienie modelu fizycznego siecią,
- CUDA tylko po to, żeby „było szybciej".

To podniesie złożoność, a nie poprawi wiarygodności.

### 11.2. Sensowna ścieżka GPU

GPU ma sens dopiero wtedy, gdy:

- loss i solver są ustabilizowane,
- chcemy fitować duże sweepy lub wiele jobów równolegle,
- potrzebujemy setek lub tysięcy ewaluacji modelu.

Wtedy możliwe ścieżki:

- JAX dla zrównoleglenia solvera ODE i auto-diff (auto-grad zamiast numerycznego gradientu w L-BFGS-B)
- CuPy dla wektorowych obliczeń cech i lossu
- PyTorch/JAX jako backend surrogate modelu

### 11.3. Sensowna ścieżka ML

ML ma sens jako:

- surrogate model przyspieszający ewaluację lossu (gdy solve_ivp jest bottleneck),
- meta-model przewidujący dobre startowe wartości parametrów (zamiast grid search w global stage),
- klasyfikator jakości fitu i wykrywania outlierów.

Nie jako pierwszy solver.

---

## 12. Plan wdrożenia w fazach

### Faza 0. Porządki i fundament

**Cel:**

- wydzielić dedykowany namespace `vortex.autofit`,
- nie mieszać go z obecnym proxy `bridge.fit`.

**Prace:**

1. Dodać pakiet `mmpp/solitons/vortex/autofit/` z `__init__.py` i `interface.py`
2. Dodać `AutofitInterface` z placeholder `.thiele()` (zwraca `NotImplementedError` — sygnalizuje, że namespace działa)
3. Dodać `job.vortex.autofit` w `VortexInterface` (lazy-loaded property, wzór z `.topology`, `.model`, etc.)
4. Dodać `AutofitConfig` i `ParameterSpec` w `config.py`
5. Dodać `VortexAutofitResult`, `AutofitDiagnostics` w `result.py`
6. Dodać test: `job.vortex.autofit` nie crashuje, `AutofitConfig()` jest poprawny

**Nie modyfikować** na tym etapie: `plotting.py`, `bridge/`, `batch.py`

### Faza 1. Single-job autofit dla CPP bez Oersteda

**Cel:**

- rozwiązać najczęstszy i najczystszy przypadek użytkownika.

**Zakres:**

1. `features.py` — `TrajectoryFeatureExtractor` z cechami: r(t), phi(t), f_inst, mean_radius, dominant_freq, ellipticity
2. `losses.py` — `TimeLoss`, `SpectralLoss`, `HybridLoss` z normalizacją i wagami
3. `optimizers.py` — `OptimizationPipeline` z etapami: init → global(DE) → local(L-BFGS-B)
4. `single.py` — `run_single_job_fit()` orkestracja:
   - resolve params via `bridge.extract.model_defaults()`
   - extract features from numerical trajectory
   - build `CPPModelAdapter` via `model.thiele.cpp()`
   - run optimization pipeline
   - build `VortexAnalyticalComparison` z best params
   - return `VortexAutofitResult`
5. `interface.py` — `AutofitInterface.thiele()` deleguje do `run_single_job_fit()`
6. `_plotting.py` — `AutofitPlotAccessor.convergence()` — loss history plot

**Krytyczna ścieżka modelu w lossu:**
```
params → CPPModelAdapter(omega0, N, chi_scale, ...) → .simulate(current, t_span, s0)
       → ThieleTrajectoryResult → extract features → compute loss
```

**Akceptacja:**

- `job.vortex.autofit.thiele()` zwraca `VortexAutofitResult`
- `result.improvement_ratio < 1.0` na co najmniej 3 testowych datasetach
- `result.comparison.plt.orbit_overlay()` wizualnie poprawiony a
- działa dla `tracking_source=table` i `tracking_source=magnetization`
- reproducible z `random_seed`

### Faza 1b. Integracja z `add_analytics(fit=...)`

**Cel:**

- użytkownik nie musi zmieniać workflow — dodaje `fit="auto"` do istniejącego API.

**Zakres:**

1. `plotting.py` — dodać parametr `fit=` do `add_analytics()`:
   - `fit=False` (default) → obecne zachowanie
   - `fit=True / "auto"` → `autofit.thiele()` z domyślnymi parametrami, wynik jako overlay
   - `fit={...}` → przekazać do `AutofitConfig`
2. Zwracany `VortexAnalyticalComparison` powinien mieć opcjonalny `.autofit_result` z pełnymi diagnostykami

**Akceptacja:**

- `job.vortex.plt.orbit().add_analytics(fit="auto")` generuje poprawiony overlay
- Bez `fit=` → zachowanie identyczne jak przed

### Faza 2. Integracja z Oerstedem i polem zewnętrznym

**Cel:**

- fit dla przypadków z `addOe=1` i/lub bias field.

**Zakres:**

- Dodać `domega0_dJ` do domyślnych `fit_params` gdy `allow_oersted=True`
- Dodać `FieldCalibration` parameters do `ParameterSpec` registry
- Diagnostyka: raportować w `warnings`, czy pole jest z `attrs`, `.mx3`, czy fitowane
- Dodać `L_force_balance` i `L_slavin_tiberkevich` jako opcjonalne składniki lossu

**Akceptacja:**

- Przypadki z Oerstedem: `domega0_dJ` fitowane daje lepszy `improvement_ratio` niż `domega0_dJ=0`
- Model raportuje źródło informacji o polu

### Faza 3. Lepsza fizyka STT

**Cel:**

- przestać traktować `chi_scale` jako jedyny worek na całą niezgodność STT.

**Zakres:**

- Dodać opcjonalne fitowanie `Lambda`, `epsilonprime` → przeliczenie na efektywny `chi_scale(theta)`
- Spiąć z `slonczewski_mtj_efficiency()` z `mmpp.analytical.thiele`
- Oddzielić fizyczne `Lambda/epsilon` od resztkowego `chi_scale`

**Rozważyć:** czy warto rozszerzyć `CPPThieleModel` o theta-dependent chi, czy przeliczać externally.

### Faza 4. Multi-job / sweep fit

**Cel:**

- dopasowywać wspólne parametry modelu do całego sweepu `J`.

**Zakres:**

- Dodać `BatchVortexInterface.autofit.sweep()` w `batch.py`
- `sweep.py` — `run_sweep_fit()`:
  - wspólne parametry: `omega0`, `N`, `chi_scale`
  - per-job nuisance: `phase0`, `center_x`, `center_y`
  - loss = Σ_j L_single(j) + λ·L_consistency (np. reuse `fit_omega0_N_to_fJ` loss)
- Optimizer: hierarchiczny — outer loop po shared params, inner loop po per-job nuisance (alternating optimization) lub flat joint optimization z odpowiednią parametryzacją

### Faza 5. Akceleracja i surrogate

**Cel:**

- obniżyć koszt dużych fitów.

**Zakres:**

- Wektorowe wywołania solvera (batch `solve_ivp`)
- Batch loss computation
- JAX backend dla auto-diff i GPU solver
- Surrogate model dla warm-startu

---

## 13. Konkretne miejsca zmian w repo

### 13.1. Nowe pliki

- `mmpp/solitons/vortex/autofit/__init__.py`
- `mmpp/solitons/vortex/autofit/interface.py`
- `mmpp/solitons/vortex/autofit/config.py`
- `mmpp/solitons/vortex/autofit/result.py`
- `mmpp/solitons/vortex/autofit/features.py`
- `mmpp/solitons/vortex/autofit/losses.py`
- `mmpp/solitons/vortex/autofit/optimizers.py`
- `mmpp/solitons/vortex/autofit/single.py`
- `mmpp/solitons/vortex/autofit/_plotting.py`
- `mmpp/solitons/vortex/autofit/sweep.py` (Faza 4)
- `tests/test_vortex_autofit.py`

### 13.2. Modyfikowane pliki

- `mmpp/solitons/vortex/interface.py`
  - dodać lazy property `.autofit` → `AutofitInterface` (Faza 0)
- `mmpp/solitons/vortex/plotting.py`
  - dodać `fit=` parametr do `VortexOrbitPlotHandle.add_analytics()` (Faza 1b)
  - dodać opcjonalny `autofit_result` na `VortexAnalyticalComparison` (Faza 1b)
- `mmpp/solitons/vortex/__init__.py`
  - eksport publiczny `AutofitInterface` (Faza 0)
- `mmpp/solitons/batch.py`
  - dodać `.autofit` na `BatchVortexInterface` (Faza 4)

### 13.3. Pliki NIE modyfikowane

- `mmpp/solitons/vortex/bridge/interface.py` — **nie dodawać soft-link do autofitu** (bridge.fit ma inną odpowiedzialność)
- `mmpp/solitons/vortex/bridge/extract.py` — **bez zmian** (używać przez publiczne API)
- `mmpp/solitons/vortex/model/thiele/fit.py` — **zostawić bez zmian** (legacy proxy)
- `mmpp/analytical/thiele.py` — **bez zmian w Fazie 0-2** (rozszerzenie dopiero ewentualnie w Fazie 3)

---

## 14. Test plan

### 14.1. Testy jednostkowe (`tests/test_vortex_autofit.py`)

**Features:**
- `TrajectoryFeatureExtractor` na syntetycznej trajektorii kołowej (znane r, f, center)
- `TrajectoryFeatureExtractor` na eliptycznej trajektorii (znana eliptyczność)
- edge case: krótka trajektoria (< 2 okresy)

**Losses:**
- `TimeLoss` daje 0 dla identycznych trajektorii
- `SpectralLoss` daje 0 dla identycznego PSD
- `HybridLoss` jest ważoną sumą składników
- normalizacja: zmiana skali r nie zmienia radykalnie loss

**Config:**
- `ParameterSpec`: frozen=True → parametr nie jest w optymalizacji
- `AutofitConfig()` domyślne wartości poprawne
- bounds validation: lower < upper

**Optimizer:**
- `OptimizationPipeline` na prostej funkcji kwadratowej (znane minimum)

### 14.2. Testy integracyjne

- `job.vortex.autofit.thiele(...)` dla `tracking_source="table"` → `VortexAutofitResult`
- `job.vortex.autofit.thiele(...)` dla `tracking_source="magnetization"` → `VortexAutofitResult`
- `job.vortex.plt.orbit().add_analytics(fit="auto")` → overlay z poprawionym fitem
- Kompatybilność `tracking_source=auto/table/magnetization`
- `random_seed` → deterministic result

### 14.3. Testy regresyjne

- `add_analytics()` BEZ `fit=` → zachowanie identyczne jak przed (porównanie figury)
- `bridge.fit.thiele_from_trajectory(...)` → nadal działa, bez zmian
- `job[:].vortex.summary()` → nadal działa (batch operations nie złamane)
- `ThieleAnalyzer.fit_omega0_N_to_fJ()` → nadal działa

### 14.4. Testy jakości fitu

Na przygotowanych danych referencyjnych (syntetyczne + 2-3 prawdziwe symulacje):

- `result.improvement_ratio < 1.0` (fit lepszy niż baseline)
- `result.comparison.metrics.delta_radius_mean` maleje po fitowaniu
- `|result.comparison.metrics.delta_freq_mean|` maleje po fitowaniu
- `result.comparison.metrics.rms_xy_residual` poprawia się
- `result.success == True`

### 14.5. Testy wydajności

- Single-job autofit (3 params, hybrid, CPP) < 30s na standardowym CPU
- Limit ewaluacji modelu: < 5000 (global) + < 500 (local)
- Feature extraction: < 1s per trajectory

---

## 15. Kryteria akceptacji

Faza 0 jest gotowa, gdy:

- `job.vortex.autofit` jest dostępny i zwraca `AutofitInterface`
- `AutofitConfig()`, `ParameterSpec()` tworzą się bez błędów
- Testy importów i struktury przechodzą

Faza 1 jest gotowa, gdy:

- `job.vortex.autofit.thiele()` zwraca `VortexAutofitResult` z pełną diagnostyką
- `result.improvement_ratio < 1.0` na referencyjnych danych CPP bez Oersteda
- `result.loss_breakdown` zawiera wszystkie składniki z niezerowymi wagami
- Wynik jest reprodukowalny dla stałego `random_seed`
- Testy jednostkowe i integracyjne przechodzą

Faza 1b jest gotowa, gdy:

- `add_analytics(fit="auto")` poprawia overlay
- `add_analytics()` bez fit → zachowanie identyczne jak wcześniej

Faza 2 jest gotowa, gdy:

- Przypadki z Oerstedem nie wymagają ręcznego strojenia `omega0` / `domega0_dJ`
- Model raportuje w `param_sources`, które efekty pola były inferowane z danych, a które dopasowane

---

## 16. Rekomendowana kolejność realizacji

Najbardziej pragmatyczna kolejność:

1. **Faza 0** — `vortex.autofit` jako nowy namespace + modele danych
2. **Faza 1** — Single-job CPP no-Oersted (core autofit engine)
3. **Faza 1b** — Integracja z `add_analytics(fit=...)`
4. **Faza 2** — Oersted / bias field
5. **Faza 3** — Lepsza fizyka STT: `Lambda`, `epsilonprime`
6. **Faza 4** — Sweep fit
7. **Faza 5** — GPU / surrogate

**Zmiana vs. oryginał:** kolejność STT i Oersted odwrócona. Uzasadnienie: Oersted (Faza 2) to dodanie jednego dodatkowego parametru (`domega0_dJ`) do istniejącego pipeline'u — niski koszt, wysoki ROI. Lepsza fizyka STT (Faza 3) wymaga decyzji architektonicznych dot. modyfikacji `CPPThieleModel` — wyższy koszt, niższy ROI na początku.

Dodana **Faza 1b** jako osobny krok — integracja z plotowaniem to inna domena niż silnik fitu i powinna być testowana osobno.

---

## 17. Ryzyka i mitygacje

| Ryzyko | Prawdopodobieństwo | Wpływ | Mitygacja |
|---|---|---|---|
| `solve_ivp` za wolny w losie (setki ewaluacji) | średnie | wysoki | Profile najpierw; cache trajectory features; zmniejsz `maxiter` w global stage |
| Global search nie znajduje dobrego basinu | średnie | wysoki | Dobry physics-informed init; opcja `global_search=False` z manual startpoint |
| Loss landscape ma wiele równoważnych minimów | wysokie | średni | Raportować w diagnostyce; porównywać multiple restarts |
| `N` i `chi_scale` silnie skorelowane | wysokie | średni | Prior regularization; raport korelacji w diagnostyce; rozważyć reparametryzację |
| Trajektoria numeryczna nie jest circular → model Thiele nie pasuje | średnie | wysoki | Dodać warning w diagnostyce; `L_ellipticity` jako sygnał; w przyszłości model z eliptyczną orbitą |
| `plotting.py` add_analytics refactor złamie istniejące API | niskie | wysoki | Regresja test; `fit=` default `False` = current behavior |

---

## 18. Decyzja końcowa

Rekomendowana implementacja to:

- **physics-informed autofit jako główny backend**,
- **multi-stage optimization** (init → global DE → local L-BFGS-B → uncertainty) jako silnik dopasowania,
- **GPU i ML jako opcjonalna akceleracja w późniejszej fazie**,
- **osobny subsystem `mmpp.solitons.vortex.autofit`**, a nie dalsze rozbudowywanie proxy helperów.

Kluczowe zmiany architektoniczne vs. oryginał:
1. **Reuse istniejącego kodu** — feature extraction przez composition z `extract_st_parameters`, `fit_orbit_ellipse`, `bridge.extract.model_defaults`; NIE reimplementacja
2. **Nazwy parametrów** zgodne z istniejącym kodem (`domega0_dJ`, nie `omega0_Oe_per_J`)
3. **Plik `result.py`** zamiast `models.py` (unikanie kolizji nazw)
4. **Konwencja `_plotting.py`** (spójna z resztą codebase)
5. **Odwrócona kolejność** Faz 2↔3 (Oersted prostszy niż pełny STT refactor)
6. **Osobna Faza 1b** dla integracji z plotowaniem
7. **Sekcja ryzyk** z konkretnymi mitygacjami
8. **Baseline loss** w wyniku — kluczowe do oceny, czy fit pomógł
9. **Bez pustych plików** (`surrogate.py`, `gpu.py`) — dodawać dopiero gdy potrzebne
