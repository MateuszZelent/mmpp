# Plan wdrożenia modułu `mmpp.solitons.vortex` — Post-processing dynamiki worteksów STNO

> **Autor:** AI Copilot + Mateusz Zelent  
> **Data:** 2026-02-10 (aktualizacja v3: 2026-02-10)  
> **Wersja:** 3.0  
> **Bazuje na:** architekturze MMPP v0.5.3, module `mmpp.fft`, `mmpp.analytical`  
> **Zmiany w v2.0:** shortcut aliasy, batch API, `average_magnetization`, G/C-state, reservoir computing (Shreya et al. 2023), coupled vortex (Hamadeh et al.)  
> **Zmiany w v3.0:** korekty fizyczne wg audytu — Berg-Lüscher topology, konwencje znaków (p/w/C/Q), SI (γμ₀M_s), faza z z(t)=x+iy, CW/CCW spectrum, events/energy/signals submoduły, zarys `solitons.skyrmion`, shared topology engine (Jenkins 2021, Wittrock 2024, multi-vortex dynamics Nat. Commun. 2025)

---

## 1. Motywacja i cel

Symulacje mikromagnetyczne nano-oscylatorów z rdzeniem worteksowym (STNO — Spin Torque Nano-Oscillator) generują bogate dane przestrzenno-czasowe, których analiza wymaga wyspecjalizowanego post-processingu. Istniejące narzędzia (MuMax3 post-processing, OOMMF `avf2ovf`, Ubermag) nie oferują spójnego pipeline'u obejmującego:

- **Śledzenie rdzenia worteksu** (core tracking) w czasie
- **Analizę trajektorii** (orbit fitting, transient/steady-state separation)
- **Klasyfikację stanów topologicznych** (polarity *p*, chirality *C*, winding number *q*)
- **Analizę spektralną dynamiki gyrotropowej** (częstotliwość gyracji ω_G, mody wyższe)
- **Ekstrakcję parametrów nieliniowych** wg teorii Slavina-Tiberkevicza
- **Characteryzację modów azymuthalnych i radialnych** (m, n, l)
- **Wizualizację** w pełni zintegrowaną z API MMPP (fluent `.plt.plot()`)

Moduł `solitons.vortex` rozszerzy MMPP o te możliwości, zachowując spójność architektoniczną z istniejącym modułem `fft`.

---

## 2. Analiza istniejącej architektury MMPP

### 2.1 Wzorzec dostępu do danych

```
MMPP(path) → job[i] : ZarrJobResult
  ├── .m / .m_layer13           → DatasetAwareWrapper (zarr + slice tracking)
  │     ├── [slice]             → DatasetAwareWrapper (zachowuje wymiary!)
  │     ├── .fft                → DatasetSpecificFFT → FFT
  │     ├── .shape / .dt / .data
  │     └── .numpy()
  ├── .attrs                    → AttributesView (dx, dy, dz, Nx, Ny, ...)
  ├── .fft                      → FFT (globalny)
  └── .mpl                      → MMPPlotter  
```

### 2.2 Wzorzec modułu FFT (do naśladowania)

```
mmpp/fft/
├── __init__.py          # Publiczny eksport
├── core.py              # Klasa FFT — główny entry point, agreguje podmoduły
├── compute_fft.py       # Engine obliczeniowy (oddzielony od interfejsu!)
├── main.py              # FFTConfig, FFTResult — dataclasses  
├── plot.py              # FFTPlotter
├── spectrum/            # Sub-moduł: widmo
│   ├── result.py        # SpectrumResult z fluent .plt / .modes
│   ├── compute.py       # compute_fft_cached()
│   ├── filter_chain.py  # SpectrumFilterChain
│   └── _plotting/       # Wizualizacje
│       └── accessor.py  # SpectrumPlotAccessor  
├── dispersion/          # Sub-moduł: relacja dyspersji
│   ├── interface.py     # FFTDispersionInterface
│   ├── core.py          # SpinWaveAnalyzer
│   └── models.py        # DispersionResult1D/2D
├── modes/               # Sub-moduł: mody FMR
│   ├── interface.py     # FFTModeInterfaceNew + ModeResult
│   └── visualization/   # Renderowanie modów
├── transmission/        # Sub-moduł: transmisja
├── filters/             # Pre/post-processing pipeline
└── vortex_classifier.py # ← Istniejący prototyp (do przeniesienia!)
```

**Kluczowe wzorce architektoniczne:**

| Wzorzec | Implementacja w FFT | Zastosowanie w vortex |
|---------|--------------------|-----------------------|
| Lazy loading | `@property` z `_cache` | Tak samo — śledzenie rdzenia dopiero na żądanie |
| Fluent result | `SpectrumResult.plt.plot()` | `TrajectoryResult.plt.plot()` |  
| Accessor pattern | `SpectrumPlotAccessor` | `TrajectoryPlotAccessor` |
| Config dataclass | `FFTConfig`, `FilterConfig` | `VortexConfig`, `TrackingConfig` |
| Dataset binding | `DatasetSpecificFFT` | `DatasetSpecificVortex` |
| Cache + zarr save | `compute_fft_cached()` | `compute_tracking_cached()` |
| Rich `_repr_html_` | Wszędzie — karty w Jupyter | Konsekwentnie |

### 2.3 Istniejący kod worteksowy

Plik `mmpp/fft/vortex_classifier.py` (629 linii) zawiera **prototypowy klasyfikator** z:
- `VortexClassificationConfig` — konfiguracja progów
- `VortexModeResult` — dataclass z indeksami (m, n, l), energią, fazą
- `AdvancedVortexClassifier` — klasyfikacja modów z analizy FFT

**Status:** Prototyp — działa na modach FFT, ale:
- ❌ brak integracji z API fluent (`job[0].m.solitons.vortex...`)
- ❌ brak śledzenia rdzenia w czasie (core tracking)
- ❌ brak analizy trajektorii
- ❌ brak ekstrakcji parametrów Slavina-Tiberkevicza
- ❌ monolityczny — cała logika w jednym pliku

→ **Decyzja:** Przenieść i zrefaktoryzować do nowego modułu `solitons.vortex`.

---

## 3. Docelowa hierarchia API

### 3.1 Punkt wejścia — integracja z `ZarrJobResult`

```python
# Dostęp przez dataset (analogicznie do .fft)
job[0].m.solitons.vortex              # → VortexInterface  
job[0].m_layer13.solitons.vortex      # → VortexInterface (z kontekstem datasetu)
job[0].m[:500, ...].solitons.vortex   # → VortexInterface (z kontekstem slice'a)

# Skrótowy dostęp globalny
job[0].solitons.vortex                # → auto-select datasetu  

# ── ALIASY SKRÓTOWE (v2.0) ──────────────────────────────────────
# Dla wygody codziennej pracy — skrócone ścieżki:
job[0].m.vortex                       # → alias → job[0].m.solitons.vortex
job[0].m.vortex.track()               # → alias → job[0].m.solitons.vortex.core.track()
job[0].m.vortex.detect()              # → alias → job[0].m.solitons.vortex.topology.detect()
job[0].m.vortex.spectrum()            # → alias → job[0].m.solitons.vortex.spectrum.gyration()
```

**Implementacja aliasów** — w `DatasetAwareWrapper`:

```python
@property
def vortex(self):
    """Shortcut: self.solitons.vortex"""
    return self.solitons.vortex
```

W `VortexInterface`:

```python
def track(self, method: str = "gaussian", **kwargs) -> TrajectoryResult:
    """Shortcut: self.core.track(method, **kwargs)"""
    return self.core.track(method=method, **kwargs)

def detect(self, **kwargs) -> TopologyResult:
    """Shortcut: self.topology.detect(**kwargs)"""
    return self.topology.detect(**kwargs)
```

### 3.2 Pełne drzewo API

```
job[0].m.solitons.vortex
│
├── .topology                         → TopologyInterface
│   ├── .detect()                     → TopologyResult (p, w, C, q, core_pos) # v3.0: w explicit
│   ├── .winding_number()             → float
│   ├── .polarity()                   → int (+1/-1)
│   ├── .vorticity()                  → int (+1/-1)  # v3.0: osobne od chirality
│   ├── .chirality()                  → int (+1/-1)
│   ├── .topological_charge()         → float  # v3.0: Q from Berg-Lüscher
│   ├── .state                        → str ("vortex", "antivortex", "meron", ...)
│   ├── .classify_gc_state()          → str ("G-state" | "C-state")  # v2.0 Wittrock
│   └── .plt
│       ├── .magnetization_map()      → Axes (mapa m z oznaczeniami)
│       └── .topological_density()    → Axes (mapa gęstości topologicznej)
│
├── .core                             → CoreInterface
│   ├── .track(method="gaussian")     → TrajectoryResult
│   ├── .track_avg_m()                → TrajectoryResult  # v2.0: ⟨m_y⟩∝-X, ⟨m_x⟩∝Y
│   ├── .position(t=None)             → (x, y) lub array
│   ├── .velocity(t=None)             → (vx, vy) lub array
│   └── .plt
│       └── .position_vs_time()       → Axes
│
├── .trajectory                       → TrajectoryInterface (wymaga .core.track())
│   ├── .raw                          → TrajectoryResult (surowa trajektoria)
│   ├── .filtered(method="savgol")    → TrajectoryResult
│   ├── .steady_state(threshold=...)  → TrajectoryResult (po transjentach)
│   ├── .orbit                        → OrbitInterface
│   │   ├── .fit(model="ellipse")     → OrbitFitResult
│   │   ├── .radius                   → float (średni promień orbity)
│   │   ├── .eccentricity             → float
│   │   ├── .center                   → (x, y)
│   │   └── .plt
│   │       ├── .xy()                 → Axes (X(t), Y(t))
│   │       ├── .orbit_2d()           → Axes (X vs Y)
│   │       └── .orbit_polar()        → Axes (r(θ))
│   ├── .phase                        → PhaseInterface
│   │   ├── .instantaneous()          → np.ndarray (φ(t))
│   │   ├── .unwrapped()              → np.ndarray
│   │   ├── .frequency(method=...)    → np.ndarray (ω(t))
│   │   └── .plt
│   │       ├── .phase_portrait()     → Axes (X vs dX/dt)
│   │       └── .frequency_vs_time()  → Axes
│   └── .plt
│       ├── .overview()               → Figure (4-panel: xy, orbit, spectrum, phase)
│       └── .animation(fps=30)        → FuncAnimation / HTML
│
├── .spectrum                         → VortexSpectrumInterface
│   ├── .gyration(method="welch")     → SpectrumResult (widmo gyracji)
│   ├── .breathing()                  → SpectrumResult (widmo modów oddechowych)
│   ├── .radial_profile(f=...)        → RadialProfileResult
│   ├── .azimuthal_decomposition()    → AzimuthalResult
│   └── .plt
│       ├── .power_spectrum()         → Axes
│       └── .spectrogram()            → Axes (STFT/wavelet)
│
├── .modes                            → VortexModesInterface
│   ├── .classify(f=8.5)             → VortexModeResult (m, n, l)
│   ├── .classify_all()              → list[VortexModeResult]
│   ├── .gyration                     → ModeDetailResult (mod gyrotropowy)
│   ├── .breathing                    → ModeDetailResult (mod oddechowy)
│   ├── .azimuthal(m=2)             → ModeDetailResult
│   └── .plt
│       ├── .mode_map(f=8.5)         → Axes (mapa amplitudy/fazy)
│       └── .mode_table()            → IPython.display (tabela modów)
│
├── .nonlinear                        → NonlinearInterface
│   ├── .slavin_tiberkevich()         → STParametersResult
│   │   ├── .p_gen                    → float (moc generacji)
│   │   ├── .linewidth                → float
│   │   ├── .Q_factor                 → float
│   │   ├── .nonlinear_coeff_N        → float  
│   │   └── .plt
│   │       └── .power_vs_current()   → Axes
│   ├── .amplitude_equation()         → AmplitudeResult (|c|²(t))
│   └── .thiele                       → ThieleInterface
│       ├── .damping_force()          → np.ndarray
│       ├── .gyrovector()             → float (G)
│       ├── .restoring_force_kappa()  → float (κ)
│       └── .plt
│           └── .force_balance()      → Axes
│
├── .reservoir                        → ReservoirInterface  # v2.0 (Shreya et al. 2023)
│   ├── .memory_capacity(max_delay=50)  → MemoryCapacityResult
│   │   ├── .MC_total                 → float  (Σ MC_k)
│   │   ├── .MC_per_delay             → np.ndarray  (MC_k vs k)
│   │   └── .plt
│   │       └── .mc_vs_delay()        → Axes
│   ├── .kernel_rank(threshold=0.01)  → KernelRankResult
│   │   ├── .rank                     → int
│   │   ├── .singular_values          → np.ndarray
│   │   └── .plt
│   │       └── .singular_spectrum()  → Axes
│   ├── .nonlinear_transformation(order=3) → NLTransformResult
│   │   ├── .capacity                 → float
│   │   └── .plt.capacity_vs_order()  → Axes
│   ├── .separation(input_a, input_b) → SeparationResult
│   │   ├── .distance_vs_time         → np.ndarray
│   │   └── .plt.separation_map()     → Axes
│   ├── .readout                      → ReadoutInterface
│   │   ├── .train(X, y, method="ridge") → ReadoutModel
│   │   ├── .predict(X)               → np.ndarray
│   │   └── .nrmse                    → float
│   └── .plt
│       └── .rc_dashboard()           → Figure (4-panel: MC, rank, NL, readout)
│
├── .coupled                          → CoupledVortexInterface  # v2.0 future (Hamadeh et al.)
│   ├── .phase_locking(other_job)     → PhaseLockResult
│   │   ├── .locked                   → bool
│   │   ├── .locking_range            → tuple[float, float]
│   │   └── .plt.arnold_tongue()      → Axes
│   ├── .mutual_synchronization(jobs) → SyncResult
│   │   ├── .order_parameter          → float (Kuramoto R)
│   │   └── .plt.sync_diagram()       → Axes
│   └── .plt
│       └── .coupled_overview()       → Figure
│
├── .events                           → EventsInterface  # v3.0
│   ├── .polarity_switches()          → list[PolaritySwitchEvent]
│   ├── .state_switches()             → list[StateSwitchEvent]
│   ├── .core_expulsions()            → list[CoreExpulsionEvent]
│   ├── .dwell_times(state="G-state") → DwellTimeResult
│   └── .plt
│       ├── .event_timeline()         → Axes (vertical lines on trajectory plot)
│       └── .dwell_histogram()        → Axes
│
├── .energy                           → EnergyInterface  # v3.0
│   ├── .vs_time(components=True)     → EnergyTimeResult
│   ├── .effective_potential()         → EffectivePotentialResult
│   ├── .pinning_sites()              → list[PinningSite]
│   └── .plt
│       ├── .energy_vs_time()         → Axes
│       └── .potential_landscape()    → Axes (2D heatmap W(x,y))
│
├── .signals                          → SignalsInterface  # v3.0
│   ├── .magnetoresistance()          → ExperimentalSignalResult
│   ├── .voltage(I_dc=..., I_ac=...) → ExperimentalSignalResult
│   ├── .power_spectrum()             → np.ndarray
│   └── .plt
│       ├── .voltage_vs_time()        → Axes
│       └── .power_vs_frequency()     → Axes
│
├── .config                           → VortexConfig (mutable)
│   ├── .tracking_method = "gaussian" 
│   ├── .core_radius_estimate = None
│   └── ...
│
└── .plt                              → VortexPlotAccessor (top-level)
    ├── .summary()                    → Figure (6-panel overview)
    └── .dashboard()                  → interactive widget
```

---

## 4. Architektura modułu — struktura plików

```
mmpp/solitons/
├── __init__.py                       # Eksport: SolitonInterface
├── _base.py                          # Bazowe klasy (SolitonResult, SolitonConfig)
│
└── vortex/
    ├── __init__.py                   # Eksport: VortexInterface, VortexConfig
    ├── interface.py                  # VortexInterface — główny entry point
    ├── config.py                     # VortexConfig, TrackingConfig, ...
    │
    ├── topology/
    │   ├── __init__.py
    │   ├── detection.py              # detect_vortex_core(), winding_density()
    │   ├── invariants.py             # polarity(), chirality(), winding_number()
    │   ├── models.py                 # TopologyResult
    │   └── _plotting/
    │       └── topology_plots.py     # TopologyPlotAccessor
    │
    ├── core/
    │   ├── __init__.py
    │   ├── tracking.py               # core_track_gaussian(), core_track_weight()
    │   ├── methods.py                # TrackingMethod enum, registry
    │   ├── models.py                 # TrajectoryResult, CorePosition
    │   └── _plotting/
    │       └── core_plots.py         # CorePlotAccessor
    │
    ├── trajectory/
    │   ├── __init__.py
    │   ├── interface.py              # TrajectoryInterface
    │   ├── orbit.py                  # OrbitFitter, OrbitFitResult
    │   ├── phase.py                  # PhaseAnalyzer (Hilbert, analytic signal)
    │   ├── filtering.py              # TrajectoryFilter (Savitzky-Golay, itp.)
    │   ├── steady_state.py           # TransientDetector
    │   ├── models.py                 # OrbitFitResult, PhaseResult
    │   └── _plotting/
    │       ├── trajectory_plots.py   # TrajectoryPlotAccessor
    │       └── orbit_plots.py        # OrbitPlotAccessor
    │
    ├── spectrum/
    │   ├── __init__.py
    │   ├── gyration.py               # gyration_spectrum() — Welch/periodogrm
    │   ├── breathing.py              # breathing_spectrum()
    │   ├── radial.py                 # radial_profile(), azimuthal_decomposition()
    │   ├── spectrogram.py            # STFT, wavelet (Morlet)
    │   ├── models.py                 # VortexSpectrumResult, RadialProfileResult
    │   └── _plotting/
    │       └── spectrum_plots.py     # VortexSpectrumPlotAccessor
    │
    ├── modes/
    │   ├── __init__.py
    │   ├── classifier.py             # ← refactored z vortex_classifier.py!
    │   ├── azimuthal.py              # indeks m — phase winding
    │   ├── radial.py                 # indeks n — amplitude nodes
    │   ├── models.py                 # VortexModeResult (rozszerzony)
    │   └── _plotting/
    │       └── mode_plots.py         # VortexModePlotAccessor
    │
    ├── nonlinear/
    │   ├── __init__.py
    │   ├── slavin_tiberkevich.py     # Ekstrakcja parametrów ST
    │   ├── thiele.py                 # Równanie Thiele'a
    │   ├── amplitude_equation.py     # Równanie amplitudowe |c|²
    │   ├── models.py                 # STParametersResult, ThieleResult
    │   └── _plotting/
    │       └── nonlinear_plots.py    # NonlinearPlotAccessor
    │
    ├── reservoir/                    # v2.0 — Reservoir Computing (Shreya et al. 2023)
    │   ├── __init__.py
    │   ├── memory_capacity.py        # MC(k) — fading memory quantification
    │   ├── kernel_rank.py            # SVD rank of state matrix
    │   ├── nonlinear_transform.py    # NLMC — nonlinear memory capacity
    │   ├── separation.py             # Input separation property
    │   ├── readout.py                # Ridge/Tikhonov readout training
    │   ├── models.py                 # MemoryCapacityResult, KernelRankResult, ...
    │   └── _plotting/
    │       └── reservoir_plots.py    # ReservoirPlotAccessor
    │
    ├── coupled/                      # v2.0 future — Coupled vortex (Hamadeh et al.)
    │   ├── __init__.py
    │   ├── phase_locking.py          # Injection locking, Arnold tongue
    │   ├── synchronization.py        # Kuramoto order parameter R
    │   ├── models.py                 # PhaseLockResult, SyncResult
    │   └── _plotting/
    │       └── coupled_plots.py      # CoupledPlotAccessor
    │
    ├── events/                       # v3.0 — Event detection
    │   ├── __init__.py
    │   ├── polarity.py               # detect_polarity_switches()
    │   ├── state_transitions.py      # detect_state_switches() (G↔C)
    │   ├── core_expulsion.py         # detect_core_expulsion()
    │   ├── dwell_time.py             # dwell_time_statistics()
    │   ├── models.py                 # PolaritySwitchEvent, StateSwitchEvent, ...
    │   └── _plotting/
    │       └── event_plots.py        # EventPlotAccessor (timeline, histogram)
    │
    ├── energy/                       # v3.0 — Energy landscape
    │   ├── __init__.py
    │   ├── time_resolved.py          # energy_vs_time()
    │   ├── potential.py              # effective_potential_W(), Boltzmann inversion
    │   ├── pinning.py                # pinning_sites()
    │   ├── models.py                 # EffectivePotentialResult, PinningSite
    │   └── _plotting/
    │       └── energy_plots.py       # EnergyPlotAccessor
    │
    ├── signals/                      # v3.0 — Experimental signal generation
    │   ├── __init__.py
    │   ├── magnetoresistance.py      # R(t) from m(t) + TMR/GMR model
    │   ├── voltage.py                # V(t) = R(t) · I(t)
    │   ├── power_spectrum.py         # P(f) = |V(f)|² / R_L
    │   ├── models.py                 # ExperimentalSignalResult
    │   └── _plotting/
    │       └── signal_plots.py       # SignalPlotAccessor
    │
    ├── _cache.py                     # Cache zarr + memory (wzorzec z FFT)
    ├── _utils.py                     # Współdzielone utility
    └── _constants.py                 # Stałe fizyczne specyficzne dla worteksów

# v3.0 — Shared topology engine (vortex + skyrmion reuse)
mmpp/solitons/
├── _topology.py                      # Shared: Berg-Lüscher, guiding_center, topological_density
│                                     # Importowane przez vortex/topology/ i skyrmion/topology/
│
# v3.0 — Skyrmion module (parallel to vortex)
├── skyrmion/
│   ├── __init__.py                   # Eksport: SkyrmionInterface
│   ├── interface.py                  # SkyrmionInterface — główny entry point
│   ├── config.py                     # SkyrmionConfig
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── detection.py              # Uses _topology.berg_luscher() — Q must be ±1
│   │   ├── helicity.py               # γ₀ (Néel/Bloch/intermediate)
│   │   └── models.py                 # SkyrmionTopologyResult
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── center.py                 # Uses _topology.guiding_center() — weighted by q(r)
│   │   ├── radius.py                 # R_sk from m_z profile, shape (circular/elliptical)
│   │   └── models.py                 # SkyrmionTrajectoryResult
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── breathing.py              # Breathing mode: R(t), ω_breath
│   │   ├── hall_angle.py             # θ_Hall = atan(v_⊥/v_∥) — skyrmion Hall effect
│   │   ├── depinning.py              # Depinning events, threshold current
│   │   └── models.py                 # BreathingResult, HallAngleResult
│   └── _plotting/
│       └── skyrmion_plots.py
```

**Łącznie: ~75 plików, 12 vortex + 4 skyrmion sub-modułów + 1 shared engine** — (v3.0 expanded).

---

## 5. Szczegółowe specyfikacje fizyczne

### 5.1 Topology — detekcja i inwariants topologiczne

#### 5.1.1 Konwencje znaków — STANDARD MMPP (v3.0)

> **Uwaga krytyczna:** W literaturze definicje `w`, `C`, `Q` bywają niespójne. MMPP przyjmuje następujący standard:

| Symbol | Nazwa | Definicja | Wartości | Przykład |
|--------|-------|-----------|----------|----------|
| $p$ | **Polarity** | $p = \text{sgn}(m_z^{\text{core}})$ | $\pm 1$ | $p=+1$: rdzeń „w górę" |
| $w$ | **Vorticity** (winding) | Nawinięcie kąta in-plane $\Phi$ na konturze: $w = \frac{1}{2\pi}\oint d\Phi$ | $+1$ (vortex), $-1$ (antivortex) | |
| $C$ | **Chirality** | Kierunek rotacji in-plane: $C = \text{sgn}(\langle m_\varphi \rangle_{\text{ring}})$ | $+1$ (CCW), $-1$ (CW) | |
| $Q$ | **Topological charge** | $Q = \int q(\mathbf{r}) d^2r$ | $\pm 1/2$ (vortex), $\pm 1$ (skyrmion) | $Q = pw/2$ |
| $\gamma_0$ | **Helicity** (skyrmiony) | Kąt między $\hat{r}$ a $\hat{m}_{\text{in-plane}}$ | $[0, 2\pi)$ | Néel: $0,\pi$; Bloch: $\pm\pi/2$ |

**Relacja fundamentalna (vortex):** $Q = \frac{p \cdot w}{2}$

> W kodzie MMPP: `polarity` i `vorticity` są **zawsze** raportowane oddzielnie od `chirality`. Nigdy nie utożsamiamy $w$ i $C$ (mimo że w starszej literaturze bywa $C \equiv w$).

#### 5.1.2 Gęstość ładunku topologicznego $q(\mathbf{r})$

$$
q(\mathbf{r}) = \frac{1}{4\pi} \hat{m} \cdot \left(\frac{\partial \hat{m}}{\partial x} \times \frac{\partial \hat{m}}{\partial y}\right)
$$

> **UWAGA SI (v3.0):** Wzór jest geometryczny (bezjednostkowy per komórka). Przy całkowaniu: $Q = \sum_{i,j} q_{ij} \, dx \, dy$. Magnetyzacja $\hat{m}$ **musi** być znormalizowana (unit vector) przed obliczeniami. Wszelkie częstotliwości ($\omega_G$ itp.) w planie wyrażamy w SI: $\gamma_0 = \gamma \mu_0$, nigdy „bare" $\gamma M_s$ (CGS).

**Dwie metody dyskretyzacji (v3.0):**

| Metoda | Opis | Stabilność $Q$ | Szybkość | Rekomendacja |
|--------|------|----------------|----------|--------------|
| `"finite_diff"` | Różnice centralne $\partial_x \hat{m} \approx (\hat{m}_{i+1,j} - \hat{m}_{i-1,j})/2dx$ | Dobra dla vorteksu | ⚡⚡⚡ | Default dla trackingu |
| `"berg_luscher"` | Triangulacja: $q_\triangle$ z iloczynu na trójkątach | Doskonała ($Q \in \mathbb{Z}/2$ dokładnie) | ⚡⚡ | **Wymagane** dla skyrmionów, rekomendowane do $Q$ |

**Algorytm Berg-Lüscher:**

Dla każdego trójkąta $(m_1, m_2, m_3)$ sieci kwadratowej (2 trójkąty per komórka):

$$
q_\triangle = \frac{1}{4\pi} \cdot 2 \arctan\left(\frac{\hat{m}_1 \cdot (\hat{m}_2 \times \hat{m}_3)}{1 + \hat{m}_1 \cdot \hat{m}_2 + \hat{m}_2 \cdot \hat{m}_3 + \hat{m}_3 \cdot \hat{m}_1}\right)
$$

Daje **dokładne** $Q \in \{0, \pm 1/2, \pm 1\}$ nawet przy dużych gradientach — kluczowe dla przerzutów polaryzacji i skyrmionów.

#### 5.1.3 Chirality — stabilny estymator pierścieniowy (v3.0)

Zamiast niestabilnego konturem z iloczynem wektorowym, stosujemy **estymator annulusowy**:

1. Wyznacz pierścień wokół rdzenia: $r \in [r_{\text{core}} + \delta,\; r_{\text{core}} + R_{\text{ring}}]$
2. W układzie core-centered, oblicz składową azymutalną:
   $$m_\varphi(\mathbf{r}) = \hat{m}(\mathbf{r}) \cdot \hat{\varphi}(\mathbf{r}), \quad \hat{\varphi} = (-\sin\varphi, \cos\varphi, 0)$$
3. Chirality: $C = \text{sgn}(\langle m_\varphi \rangle_{\text{ring}})$

**Zalety:** szybkie, odporne na szum, współdzielone z `modes.azimuthal` (ten sam preprocessing).

#### 5.1.4 Algorytm detekcji rdzenia

1. Oblicz $|m_z|$ — rdzeń ma ekstremalny $m_z$ ($\pm 1$ idealnie)
2. Próg: $|m_z| > 0.9 \cdot \max(|m_z|)$ — maska rdzenia
3. Centroid ważony: $(x_c, y_c) = \sum w_i (x_i, y_i) / \sum w_i$, $w_i = |m_z(x_i, y_i)|^2$
4. Sub-pikselowa precyzja: dopasowanie Gaussa 2D do $|m_z|$ w okolicy rdzenia

#### 5.1.5 Implementacja — `topology/detection.py`

```python
@dataclass
class TopologyResult:
    """Complete topological characterization of a soliton."""
    polarity: int                       # p = ±1 (sign of m_z at core)
    vorticity: int                      # w = ±1 (+1=vortex, -1=antivortex)
    chirality: int                      # C = ±1 (+1=CCW, -1=CW in-plane)
    Q: float                            # topological charge (integrated q)
    helicity: float | None              # γ₀ [rad] — only for skyrmions
    core_position: tuple[float, float]  # (x, y) [m]
    topological_density: np.ndarray     # q(x,y) map [1/m²]
    state: str                          # "vortex" | "antivortex" | "meron" | "skyrmion"
    method: str                         # "finite_diff" | "berg_luscher"
    confidence: float                   # 0-1 (|Q_measured - Q_expected| < threshold)
    
    @property
    def is_consistent(self) -> bool:
        """Check Q ≈ p*w/2 (vortex) or |Q| ≈ 1 (skyrmion)."""
        if self.state in ("vortex", "antivortex"):
            return abs(self.Q - self.polarity * self.vorticity / 2) < 0.1
        return abs(abs(self.Q) - 1) < 0.1

def detect_topology(
    m: np.ndarray,          # shape (Ny, Nx, 3) — MUST be unit-normalized
    dx: float, dy: float,   # cell size [m] — SI!
    *,
    method: str = "finite_diff",  # or "berg_luscher"
    polarity_threshold: float = 0.5,
    chirality_ring_r: tuple[float, float] | None = None,  # (r_min, r_max) [m]
) -> TopologyResult:
    """Detect topological state from single magnetization snapshot."""
```

### 5.2 Core Tracking — śledzenie rdzenia w czasie

**Metody śledzenia rdzenia:**

| Metoda | Opis | Dokładność | Szybkość |
|--------|------|------------|----------|
| `"maximum"` | argmax $\|m_z\|$ | 1 cell | ⚡⚡⚡ |
| `"centroid"` | Centroid ważony $m_z^2$ | ~0.5 cell | ⚡⚡⚡ |
| `"gaussian"` | Dopasowanie Gaussa 2D | ~0.1 cell | ⚡⚡ |
| `"polynomial"` | Fit paraboli do $m_z$ | ~0.2 cell | ⚡⚡ |
| `"guiding_center"` | Równanie prowadzące | Fizycznie poprawny | ⚡⚡ |
| `"average_magnetization"` | Z ⟨**m**⟩ na próbce | Globalny, szybki | ⚡⚡⚡ |

**Metoda `average_magnetization` (v2.0):**

Pozycja rdzenia worteksu z uśrednionej magnetyzacji próbki (rigid vortex model):

$$
X_{\text{core}} \approx -R_{\text{dot}} \cdot \langle m_y \rangle, \qquad
Y_{\text{core}} \approx +R_{\text{dot}} \cdot C \cdot \langle m_x \rangle
$$

gdzie $R_{\text{dot}}$ to promień dysku, $C = \pm 1$ to chirality, a $\langle m_{x,y} \rangle$ to uśrednione po próbce składowe in-plane.

**Uzasadnienie:** W modelu sztywnego worteksu (rigid vortex), przesunięcie rdzenia o $\Delta X$ generuje nadwyżkę magnetyzacji $\langle m_y \rangle \propto -\Delta X / R$. Jest to najszybsza metoda śledzenia, nie wymaga dostępu do danych przestrzennych — wystarczy średnia $\langle \mathbf{m} \rangle(t)$ (MuMax3: `table.txt`). Szczególnie użyteczna dla:

- Szybkiego preview trajektorii
- Danych z eksperymentu (np. STXM daje jedynie uśredniony sygnał)
- Walidacji: porównanie z metodą sub-pikselową (Gauss/centroid)

```python
def core_track_average_magnetization(
    avg_mx: np.ndarray,         # ⟨m_x⟩(t)
    avg_my: np.ndarray,         # ⟨m_y⟩(t)
    R_dot: float,               # promień dysku [m]
    chirality: int = 1,         # C = ±1
) -> TrajectoryResult:
    """Track vortex core from spatially-averaged magnetization."""
    x = -R_dot * avg_my
    y = chirality * R_dot * avg_mx
    ...
```

**Metoda centroidu/prowadząca (guiding center):**

$$
\mathbf{R}_{\text{gc}} = \frac{\int \mathbf{r}\, q(\mathbf{r})\, d^2r}{\int q(\mathbf{r})\, d^2r}
$$

gdzie $q(\mathbf{r})$ to gęstość topologiczna — **fizycznie najbardziej poprawne** dla worteksów.

> **Krytyczna poprawka v3.0 — ROI dla guiding center:**
> W praktyce $q(x,y)$ nie jest idealnie zerowe w całym obszarze (krawędzie dysku, C-state, szum termiczny). Dlatego `guiding_center` **musi** liczyć w ROI:
> - Automatyczny: próg $|q| > \epsilon \cdot \max|q|$ lub
> - Promień od `core_guess` (np. 20–50 nm)
> - Raportuj `confidence = \sum|q|_{\text{ROI}} / \sum|q|_{\text{total}}` (jaka frakcja masy topologicznej w ROI)

**Metoda Gauss fit (najwyższa precyzja sub-pikselowa):**

1. Znajdź piksel z max $|m_z|$
2. Wytnij ROI $7 \times 7$ wokół
3. Dopasuj: $m_z(x,y) = A \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right) + B$
4. $(x_0, y_0)$ dają pozycję sub-pikselową

**Implementacja — `core/tracking.py`:**

```python
@dataclass
class TrajectoryResult:
    """Result of vortex core tracking over time."""
    time: np.ndarray           # [s]
    x: np.ndarray              # [m] — pozycja X rdzenia
    y: np.ndarray              # [m] — pozycja Y rdzenia
    polarity: np.ndarray       # p(t) — śledzenie przerzutów
    method: str                # nazwa metody
    confidence: np.ndarray     # [0-1] per-frame tracking confidence (v3.0)
    metadata: dict             # dx, dy, dataset, ...
    
    @property
    def z(self) -> np.ndarray:
        """Complex trajectory z(t) = (x-x0) + i(y-y0). v3.0: primary for phase analysis."""
        xc, yc = np.mean(self.x), np.mean(self.y)
        return (self.x - xc) + 1j * (self.y - yc)
    
    @property
    def r(self) -> np.ndarray:
        """Promień orbity r(t) = |z(t)|."""
        return np.abs(self.z)
    
    @property
    def phi(self) -> np.ndarray:
        """Kąt azymuthalny φ(t) = arg(z(t))."""
        return np.angle(self.z)
    
    @property
    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Prędkość (vx, vy) z różniczkowania numerycznego."""
        vx = np.gradient(self.x, self.time)
        vy = np.gradient(self.y, self.time)
        return vx, vy
    
    @property
    def instantaneous_frequency(self) -> np.ndarray:
        """ω(t) = dφ/dt from complex signal z(t). v3.0: preferred over Hilbert."""
        phi_unwrapped = np.unwrap(self.phi)
        return np.gradient(phi_unwrapped, self.time)
    
    @property
    def rotation_sense(self) -> str:
        """CW or CCW from sign of mean ω(t)."""
        return "CCW" if np.mean(self.instantaneous_frequency) > 0 else "CW"
    
    @property
    def plt(self) -> "TrajectoryPlotAccessor":
        return TrajectoryPlotAccessor(self)
```

### 5.3 Trajectory Analysis — orbity i faza

**Dopasowanie orbity eliptycznej:**

$$
\frac{(x - x_0)^2}{a^2} + \frac{(y - y_0)^2}{b^2} = 1
$$

z parametrami $(x_0, y_0, a, b, \theta)$ — centrum, półosie, kąt obrotu.

**Detekcja steady-state:**

1. Oblicz $r(t) = |\mathbf{R}(t) - \mathbf{R}_{\text{mean}}|$
2. Uruchomieniowe odchylenie standardowe $\sigma_r(t)$ w oknie $W$
3. Steady-state gdy $|\sigma_r(t)/\sigma_r(t-\Delta)  - 1| < \epsilon$ przez $N_{\min}$ okresów
4. *(v3.0)* Dodatkowy warunek: stabilizacja częstotliwości $|\omega(t) - \langle\omega\rangle| / \langle\omega\rangle < \epsilon_f$ — zapobiega fałszywym alarm positive w transjentach z driftem amplitudy ale niestabilną fazą.

> **v3.0 — Analiza fazy: z(t) jako metoda primary**
>
> Faza pochodzi z **sygnału kompleksowego trajektorii orbitalnej** a nie z transformaty Hilberta
> jednokomponentowego sygnału x(t). Uzasadnienie: Hilbert z $x(t)$ jest poprawny tylko dla
> wąskopasmowych oscylacji; z(t) naturalnie koduje 2D orbital motion i odróżnia CW od CCW.

**Primary method — complex orbital signal:**

$$
z(t) = (x(t) - x_0) + i\,(y(t) - y_0)
$$

$$
\phi(t) = \arg(z(t)), \quad \omega(t) = \frac{d\phi_{\text{unwrap}}}{dt}
$$

Znak $\omega(t)$ rozstrzyga kierunek gyracji: $\omega > 0 \Rightarrow$ CCW, $\omega < 0 \Rightarrow$ CW.

**Secondary method — Hilbert (`method="hilbert"`):**

$$
z_H(t) = x(t) + i\,\mathcal{H}[x(t)]
$$

Useful as fallback gdy dostępny jest tylko 1D sygnał (np. eksperymentalny MR voltage).
**Uwaga:** Hilbert **nie rozróżnia CW/CCW** — wymaga dodatkowej informacji z y(t) lub polarity.

**Orbit drift model (v3.0):**

W realistycznych symulacjach i eksperymentach centrum orbity dryfuje:

$$
\mathbf{R}_0(t) = \mathbf{R}_0^{(0)} + \mathbf{v}_{\text{drift}} t + \frac{1}{2}\mathbf{a}_{\text{drift}} t^2
$$

Model jest fitowany w oknie sliding i odejmowany przed analizą fazy/amplitudy.

```python
@dataclass
class OrbitFitResult:
    center: tuple[float, float]  # [m]
    semi_major: float            # a [m]
    semi_minor: float            # b [m]
    eccentricity: float          # e = sqrt(1 - b²/a²)
    tilt_angle: float            # θ [rad]
    residual: float              # MSE fitu
    # v3.0 additions:
    center_drift_model: Optional[np.polynomial.Polynomial]  # R₀(t) drift fit
    fit_window: tuple[float, float]   # (t_start, t_end) [s] — window used for orbit fit
    residual_time_series: Optional[np.ndarray]  # dr(t) = |R(t) - R_fit(t)| per frame
    
    @property
    def radius(self) -> float:
        """Średni promień (geometric mean)."""
        return np.sqrt(self.semi_major * self.semi_minor)
    
    @property
    def is_circular(self, tol: float = 0.1) -> bool:
        """True if eccentricity < tol."""
        return self.eccentricity < tol
```

### 5.4 Vortex Spectrum — analiza widmowa

**Widmo gyracji z trajektorii rdzenia:**

$$
S_{xx}(f) = |\text{FFT}[x(t)]|^2, \quad S_{yy}(f) = |\text{FFT}[y(t)]|^2
$$

$$
S_{\text{gyr}}(f) = S_{xx}(f) + S_{yy}(f)
$$

Metoda **Welch** z oknem Hanninga, overlap 50% — konfigurowalny.

> **v3.0 — Widmo kierunkowe CW/CCW z sygnału kompleksowego z(t)**
>
> Kluczowy insight: FFT sygnału kompleksowego $z(t) = (x-x_0) + i(y-y_0)$ daje **asymetryczne**
> widmo — peaki przy $f>0$ odpowiadają CCW, a przy $f<0$ odpowiadają CW gyracji.
> Pozwala to na rozdzielenie modów o różnym sensie rotacji bez ambiguity.

**Widmo kierunkowe (directional PSD):**

$$
Z(f) = \text{FFT}[z(t)] = \text{FFT}[(x(t)-x_0) + i(y(t)-y_0)]
$$

$$
S_+(f) = |Z(f)|^2 \text{ dla } f > 0 \quad (\text{CCW modes})
$$

$$
S_-(f) = |Z(-f)|^2 \text{ dla } f > 0 \quad (\text{CW modes})
$$

W implementacji: `S_plus = |Z[1:N//2]|²`, `S_minus = |Z[N//2+1:][::-1]|²` (po fftshift).

**Analiza azymuthalna — dekompozycja w mody $m$:**

Dla pierścienia o promieniu $r^*$ wokół rdzenia:

$$
\delta m_{\perp}(r^*, \phi, t) = \sum_{m=-\infty}^{\infty} c_m(t) e^{im\phi}
$$

$$
c_m(t) = \frac{1}{2\pi} \int_0^{2\pi} \delta m_{\perp}(r^*, \phi, t) e^{-im\phi} d\phi
$$

Widmo azymuthalne: $|c_m(f)|^2$.

**Spektrogram (STFT):**

$$
\text{STFT}[x](t, f) = \int x(\tau) w(\tau - t) e^{-2\pi i f \tau} d\tau
$$

z oknem $w$ — do śledzenia przejść transjentowych i przerzutów polaryzacji.

> **v3.0 — linewidth caveat**
>
> W symulacjach mikromagnetycznych **bez źródła szumu termicznego** (T=0), obserwowana
> szerokość linii $\Delta f$ jest zdominowana przez **rozdzielczość FFT** ($\Delta f_{\text{res}} = 1/T_{\text{sim}}$),
> a nie fizyczne mechanizmy dekoherencji. Ekstrakcja „prawdziwego" linewidth wymaga:
> (a) symulacji z termicznym polem stochastycznym, lub (b) modelowania via inverse power method.
> MMPP powinno raportować zarówno `linewidth_measured` jak i `linewidth_resolution_limited: bool`.

### 5.5 Mode Classification — rozbudowa istniejącego klasyfikatora

> **v3.0 — Core-centered coordinate frame**
>
> Klasyfikacja modów musi odbywać się w **układzie współrzędnych centrowanym na rdzeniu** (core-centered frame),
> a nie w układzie laboratoryjnym. Dla dużych orbit gyracyjnych ($r/R > 0.2$) profil $\delta\mathbf{m}$ widziany
> z centrum dysku jest silnie zmodulowany przez przesunięcie rdzenia, co zaburza dekompozycję $c_m$.
>
> Procedura: (1) Track core $(x_c(t), y_c(t))$ per frame → (2) Re-center magnetization: 
> $\delta\mathbf{m}'(\mathbf{r}, t) = \delta\mathbf{m}(\mathbf{r} + \mathbf{R}_c(t), t)$ → (3) Azimuthal decomposition on $\delta\mathbf{m}'$.
> Ref: Jenkins et al. 2021, Nat. Commun., Fig. 2.

**Indeks azymuthalny $m$ (z nawinięcia fazy):**

$$
m = \frac{1}{2\pi} \oint_{\text{ring}} d\arg(\delta m_x + i\,\delta m_y)
$$

**Indeks radialny $n$ (z węzłów amplitudy):**

$n$ = liczba zer w profilu radialnym $|\delta m|(r)$ dla $r \in (0, R_{\text{dot}})$.

**Częstotliwość gyrotropowa (Thiele):**

$$
\omega_G = \frac{\kappa}{|G|} = \frac{20}{9} \frac{\gamma_0 M_s}{R^2} \frac{L \xi}{1 + \xi/2}
$$

gdzie $\xi = R/L$ to stosunek promienia do grubości dysku, $\kappa$ — stała sprężysta odtwarzająca.
*(v3.0: jawnie $\gamma_0 = \gamma \mu_0$ w jednostkach SI — patrz §5.1.1)*

**Mody wyższe — relacja Parkina:**

$$
f_{m,n} \approx f_G \cdot \alpha_{m,n}
$$

gdzie $\alpha_{m,n}$ to stosunek częstotliwości modu $(m,n)$ do modu gyrotropowego, zależny od geometrii.

> **v3.0 — Rozszerzony `VortexModeResult`**

```python
@dataclass
class VortexModeResult:
    """Single identified vortex spin-wave mode."""
    m: int                       # azimuthal index
    n: int                       # radial index
    frequency_ghz: float         # GHz
    amplitude: float             # relative amplitude
    phase: float                 # rad — faza z fit sinusoidalnego
    profile: np.ndarray          # radial amplitude profile |δm|(r)
    # v3.0 additions:
    sense: Literal["CW", "CCW"]  # rotation sense from directional spectrum S₊/S₋
    core_offset: float           # [m] mean core offset during mode observation
    confidence: float            # [0-1] mode identification confidence (SNR-based)
    bandwidth: float             # [Hz] spectral bandwidth of the mode peak
    coupling_to_gyration: float  # correlation between mode amplitude and gyration radius
    linewidth_resolution_limited: bool  # True if bandwidth ≤ 2·Δf_res
    
    @property
    def is_gyrotropic(self) -> bool:
        return abs(self.m) == 1 and self.n == 0
    
    @property
    def label(self) -> str:
        """Human-readable label e.g. 'CCW (m=+1, n=0)'."""
        sign = "+" if self.m >= 0 else ""
        return f"{self.sense} (m={sign}{self.m}, n={self.n})"
```

### 5.6 Nonlinear Analysis — teoria Slavina-Tiberkevicza

**Uniwersalne równanie nano-oscylatora:**

$$
\frac{dc}{dt} + i\omega(|c|^2)c + \Gamma_+(|c|^2)c - \Gamma_-(|c|^2)c = f_{\text{ext}}(t)
$$

gdzie $c(t)$ — kompleksowa amplituda gyracji, i:

$$
\omega(p) = \omega_0 + Np, \quad p = |c|^2
$$

$$
\Gamma_+ = \Gamma_G(1 + Qp), \quad \Gamma_- = \sigma I (1 - p)
$$

**Parametry do ekstrakcji z symulacji:**

| Symbol | Opis | Metoda ekstrakcji |
|--------|------|-------------------|
| $\omega_0$ | Częstotliwość liniowa | Peak w widmie przy małej mocy |
| $N$ | Nieliniowy przesuw częstotliwości | $\Delta\omega / \Delta p$ z wielu amplitud |
| $\Gamma_G$ | Tłumienie Gilberta (efektywne) | Szerokość linii $\Delta f$ |
| $Q$ | Nieliniowe tłumienie | $\Delta(\Delta f) / \Delta p$ |
| $\sigma$ | Efektywność STT | Próg prądu $I_{\text{th}}$ |

**Algorytm ekstrakcji $N$ (nonlinear frequency shift):**

1. Z trajektorii: $r(t) \to p(t) = r^2(t)/R^2$
2. $\omega(t)$ z analizy fazy (Hilbert)
3. Fit liniowy: $\omega(p) = \omega_0 + Np$
4. Alternatywnie: seria widm przy różnych $I_{\text{STT}}$

**Równanie Thiele'a — siły działające na rdzeń (v3.0: explicit force decomposition):**

$$
\mathbf{G} \times \dot{\mathbf{R}} = \mathbf{F}_{\text{conservative}} + \mathbf{F}_{\text{dissipative}} + \mathbf{F}_{\text{STT}} + \mathbf{F}_{\text{Oe}}
$$

| Siła | Wyrażenie | Opis |
|------|-----------|------|
| $\mathbf{F}_{\text{conservative}}$ | $-\nabla W(\mathbf{R})$ | Potencjał odtwarzający (parabola + anharmoniczne) |
| $\mathbf{F}_{\text{dissipative}}$ | $-D\dot{\mathbf{R}}$ | Tensor tłumienia Gilberta, $D = -\alpha \eta G$ |
| $\mathbf{F}_{\text{STT}}$ | $\propto a_J M_s J$ | Spin-transfer torque z prądu spinowo-spolaryzowanego |
| $\mathbf{F}_{\text{Oe}}$ | $\propto \nabla(\mathbf{H}_{\text{Oe}} \cdot \mathbf{m})$ | Pole Oersteda od prądu — **często pomijane, ale istotne** |

Pełne wyrażenie:
- $\mathbf{G} = -G\hat{z}$, $G = 2\pi p w M_s L / \gamma_0$ — gyrovector *(v3.0: jawnie $p \cdot w$, nie $p \cdot q$)*
- $D = -\alpha \eta G$ — tensor tłumienia ($\eta$ — czynnik geometryczny)
- $W(\mathbf{R})$ — potencjał odtwarzający (parabola w przybliżeniu harmonicznym)
- $\mathbf{F}_{\text{STT}}$ — siła spinowo-transferowa
- $\mathbf{F}_{\text{Oe}}$ — siła od Oersted field: w STNO prąd $I_{DC}$ generuje niejednorodne pole circumferential $H_{\text{Oe}}(r) \propto I/r$, łamiące symetrię CW/CCW; pomijanie tego składnika prowadzi do błędu >10% w progu oscylacji (Dussaux et al. 2010)

> **v3.0 — Ekstrakcja sił z symulacji:**
> 
> MMPP pozwoli na dekompozycję sił Thiele'a z danych mikromagnetycznych:
> 1. $\mathbf{F}_{\text{conservative}} = -\nabla W$ estymowane z potencjału $W(R)$ (patrz §5.10 energy/)
> 2. $\mathbf{F}_{\text{dissipative}}$ z pomiarowych parametrów ($\alpha$, $\eta$)
> 3. $\mathbf{F}_{\text{STT}}$ z known current density profile
> 4. $\mathbf{F}_{\text{Oe}}$ z Biot-Savart dla geometry nanofilara
> 5. Resztka: $\mathbf{F}_{\text{res}} = \mathbf{G} \times \dot{\mathbf{R}} - \sum_i \mathbf{F}_i$ — wskaźnik jakości modelu

```python
@dataclass
class STParametersResult:
    """Slavin-Tiberkevich parameters extracted from simulation."""
    omega_0: float          # rad/s — linear frequency
    f_0_ghz: float          # GHz — linear frequency  
    N: float                # rad/s — nonlinear freq. shift coefficient
    Gamma_G: float          # 1/s — Gilbert damping rate
    Q: float                # dimensionless — nonlinear damping
    sigma: float            # 1/(s·A) — STT efficiency
    I_threshold: float      # A — threshold current
    generation_power: float # p_gen = |c_gen|² at steady state
    linewidth_hz: float     # Hz — spectral linewidth Δf
    quality_factor: float   # Q_osc = f_0 / Δf
    
    @property
    def plt(self) -> "STPlotAccessor":
        return STPlotAccessor(self)
```

> **v3.0 — Caveat: linewidth w symulacjach T=0**
>
> Parametr `linewidth_hz` w `STParametersResult` jest wiarygodny **wyłącznie** gdy symulacja
> zawiera stochastyczne pole termiczne. Dla T=0 mumax³/OOMMF, mierzony Δf jest artefaktem
> rozdzielczości FFT lub numerycznego szumu. MMPP automatycznie dodaje flagę
> `linewidth_resolution_limited: bool` i warning w `_repr_html_()` gdy Δf ≤ 2/T_sim.

### 5.7 Klasyfikacja G-state vs C-state (v2.0 — Wittrock et al.)

W worteksowych STNO, dynamika rdzenia wykazuje dwa jakościowo różne reżimy zależne od prądu i pola:

- **G-state (Gyrotropowy):** Rdzeń wykonuje regularną circular/elliptical orbit wokół centrum dysku. Dominuje standardowa gyracja $\omega_G$. Faza jest dobrze zdefiniowana — rdzeń zachowuje stałą rotację CW/CCW.

- **C-state (Commensurate / Chaotyczny):** Rdzeń wykazuje deformację profilu magnetyzacji — in-plane magnetyzacja w dysku przestaje być czysto curling, pojawia się składowa „C-shaped". Rdzeń może doświadczać przerzutów polaryzacji, nieregularnych trajektorii, lub przejść do stanu wielomodowego.

**Algorytm klasyfikacji G/C-state:**

$$
\text{Kryterium 1: } \sigma_r / \langle r \rangle < \epsilon_{\text{circ}} \implies \text{G-state}
$$

$$
\text{Kryterium 2: } S_2 = \frac{\langle |c_2|^2 \rangle}{\langle |c_0|^2 \rangle + \langle |c_1|^2 \rangle} > \epsilon_{\text{C}} \implies \text{C-state}
$$

gdzie $c_m$ to współczynniki dekompozycji azymutalnej, a $S_2$ mierzy wagę harmonicznej $m=2$ (deformacja C-shaped).

**Dodatkowe wskaźniki:**

| Wskaźnik | G-state | C-state |
|----------|---------|---------|
| Regularność orbity | Wysoka ($\sigma_r/\langle r \rangle < 0.1$) | Niska |
| Dominujący mod | $m=\pm 1$ (gyracja) | $m=0$ + $m=2$ mieszane |
| Widmo | Wąski peak $\omega_G$ | Szerokie / wielomodowe |
| Przerzuty polaryzacji | Brak | Mogą wystąpić |
| Profil magnetyzacji | Czysty curling | C-shaped deformacja |

```python
@dataclass
class GCStateResult:
    """G-state vs C-state classification result."""
    state: str                    # "G-state" | "C-state" | "transition"
    circularity: float            # σ_r / ⟨r⟩ — regularność orbity
    c_state_parameter: float      # S_2 — waga harmonicznej m=2
    dominant_modes: list[int]     # dominujące indeksy azymutalne
    polarity_switches: int        # liczba przerzutów polaryzacji w oknie
    confidence: float             # 0-1
    
def classify_gc_state(
    trajectory: TrajectoryResult,
    magnetization: np.ndarray,    # m(x, y, t) — opcjonalnie do dekompozycji
    *,
    circularity_threshold: float = 0.15,
    c_parameter_threshold: float = 0.1,
) -> GCStateResult:
    """Classify vortex dynamic regime as G-state or C-state (Wittrock et al.)."""
```

### 5.8 Reservoir Computing — metryki RC (v2.0 — Shreya et al. 2023)

Praca Shreya et al. (*Scientific Reports* **13**, 16553, 2023) pokazuje, że worteksowe STNO z granularną warstwą swobodną mogą służyć jako **fizyczne rezerwuary** (physical reservoir computing). Rdzeń worteksu poruszający się po nierównomiernym potencjale (grain boundaries) generuje bogaty, nieliniowy response na sygnały wejściowe.

#### 5.8.1 Architektura Reservoir Computing z STNO

```
Input signal u(t) → [Prąd I_STT(t)] → STNO vortex dynamics → [⟨m⟩(t)] → Readout W·x(t) → ŷ(t)
                                          ↑
                                    Physical Reservoir
                                    (granular vortex)
```

Kluczowe koncepcje:
- **Input encoding:** Modulacja prądu STT → wymuszanie dynamiki rdzenia
- **State vector:** Trajektoria rdzenia (X(t), Y(t)) + opcjonalnie ⟨m_x⟩, ⟨m_y⟩, ⟨m_z⟩
- **Readout:** Liniowa regresja (Ridge/Tikhonov) na state vector
- **Granularność:** Grain boundaries tworzą nierównomierny potencjał → zwiększona nieliniowość → lepsze RC

#### 5.8.2 Memory Capacity (MC)

Miara zdolności rezerwuaru do zapamiętywania przeszłych inputów:

$$
MC = \sum_{k=1}^{k_{\max}} MC_k, \quad MC_k = \frac{\text{Cov}^2(\hat{y}_k(t),\, u(t-k))}{\text{Var}(\hat{y}_k(t)) \cdot \text{Var}(u(t-k))}
$$

gdzie $\hat{y}_k(t) = \mathbf{w}_k^T \mathbf{x}(t)$ to readout wytrenowany do odtworzenia sygnału opóźnionego o $k$ kroków.

Dla idealnego rezerwuaru: $MC \leq N_{\text{nodes}}$ (Jaeger, 2001).

**Algorytm:**

```python
def compute_memory_capacity(
    state_matrix: np.ndarray,  # (T, N_features) — trajektoria rdzenia + pochodne
    input_signal: np.ndarray,  # (T,) — u(t)
    max_delay: int = 50,
    regularization: float = 1e-4,
) -> MemoryCapacityResult:
    """
    Compute MC by training separate readouts for each delay k.
    
    state_matrix: np.ndarray of shape (T, N) — reservoir states
        Can be constructed from trajectory: [X, Y, X², Y², XY, ...]
    input_signal: np.ndarray of shape (T,) — input u(t), typically i.i.d. uniform
    """
    MC_k = np.zeros(max_delay)
    for k in range(1, max_delay + 1):
        y_target = input_signal[:-k]          # u(t-k)
        X_train = state_matrix[k:]            # x(t)
        w = ridge_regression(X_train, y_target, alpha=regularization)
        y_pred = X_train @ w
        MC_k[k-1] = np.corrcoef(y_pred, y_target)[0, 1] ** 2
    return MemoryCapacityResult(MC_total=MC_k.sum(), MC_per_delay=MC_k)
```

#### 5.8.3 Kernel Rank (Information Processing Capacity)

Miara efektywnej wymiarowości przestrzeni stanów rezerwuaru:

$$
\text{rank}_\epsilon(\mathbf{X}) = \#\{i : \sigma_i > \epsilon \cdot \sigma_1\}
$$

gdzie $\sigma_i$ to wartości osobliwe macierzy stanów $\mathbf{X} \in \mathbb{R}^{T \times N}$.

**Wyższy rank = bogatszy zestaw liniowo niezależnych representacji** → lepsze RC.

W kontekście granularnych STNO: grain boundaries zwiększają rank poprzez dodanie dodatkowych nieliniowych dynamik (pinning, depinning, metastable orbits).

```python
def compute_kernel_rank(
    state_matrix: np.ndarray,
    threshold: float = 0.01,
) -> KernelRankResult:
    """Compute effective rank via SVD."""
    U, S, Vt = np.linalg.svd(state_matrix, full_matrices=False)
    rank = np.sum(S > threshold * S[0])
    return KernelRankResult(rank=rank, singular_values=S)
```

#### 5.8.4 Nonlinear Memory Capacity (NLMC)

Rozszerzenie klasycznego MC na nieliniowe transformacje:

$$
NLMC_k^{(n)} = \frac{\text{Cov}^2(\hat{y}_k^{(n)},\, L_n(u(t-k)))}{\text{Var}(\hat{y}_k^{(n)}) \cdot \text{Var}(L_n(u(t-k)))}
$$

gdzie $L_n$ to wielomian Legendre'a rzędu $n$. Mierzy zdolność do **nieliniowej transformacji** opóźnionych inputów.

$$
\text{IPC} = \sum_{n=1}^{n_{\max}} \sum_{k=1}^{k_{\max}} NLMC_k^{(n)}
$$

#### 5.8.5 Separation Property

Miara jak dobrze rezerwuar rozdziela różne sygnały wejściowe:

$$
\text{Sep}(u_A, u_B) = \frac{\| \mathbf{x}_A(t) - \mathbf{x}_B(t) \|}{\| u_A(t) - u_B(t) \|}
$$

Wysoki $\text{Sep}$ → rezerwuar zwielokrotnia różnice w inputach → łatwiejszy readout.

#### 5.8.6 Readout Training

```python
@dataclass
class ReadoutModel:
    """Trained readout layer for reservoir computing."""
    weights: np.ndarray       # (N_features,) lub (N_features, N_outputs)
    method: str               # "ridge" | "tikhonov" | "lasso"
    alpha: float              # regularization strength
    nrmse_train: float        # normalized RMSE on training set
    nrmse_test: float         # normalized RMSE on test set
    
    def predict(self, state_matrix: np.ndarray) -> np.ndarray:
        return state_matrix @ self.weights

class ReadoutInterface:
    """Train and evaluate readout layers."""
    
    def train(
        self,
        X: np.ndarray,           # (T, N) state matrix
        y: np.ndarray,           # (T,) or (T, M) targets
        method: str = "ridge",
        alpha: float = 1e-4,
        test_ratio: float = 0.3,
    ) -> ReadoutModel:
        """Train readout with cross-validation."""
        ...
    
    def nrmse(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Normalized Root Mean Square Error."""
        return np.sqrt(np.mean((y_pred - y_true)**2)) / np.std(y_true)
```

#### 5.8.7 Konstrukcja macierzy stanów z trajektorii STNO

Kluczowe dla RC — jak wyciągnąć feature vector z dynamiki rdzenia:

```python
def build_state_matrix(
    trajectory: TrajectoryResult,
    features: list[str] = ["x", "y", "r", "phi", "vx", "vy"],
    nonlinear_order: int = 2,
    time_delays: list[int] = [0, 1, 2, 5],
) -> np.ndarray:
    """
    Build RC state matrix from vortex trajectory.
    
    Features:
    - Linear: X(t), Y(t), r(t), φ(t), vx(t), vy(t)
    - Nonlinear: X², Y², XY, X³, ... (up to `nonlinear_order`)
    - Time-delayed: X(t-τ), Y(t-τ), ... (for each τ in `time_delays`)
    - Average magnetization: ⟨mx⟩, ⟨my⟩, ⟨mz⟩ (if available)
    
    Returns: state_matrix of shape (T, N_features)
    """
```

### 5.9 Events Detection — zdarzenia dynamiczne (v3.0)

Nowy submoduł `solitons/vortex/events/` — detekcja dyskretnych zdarzeń w dynamice rdzenia worteksu.

> **Motywacja:** W realnych danych eksperymentalnych i długich symulacjach rdzeń przechodzi
> przez przerzuty polaryzacji, przejścia G↔C, wyrzuty rdzenia z dysku, itd. Automatyczna
> detekcja tych zdarzeń jest **kluczowa** dla analizy statystycznej i klasyfikacji reżimów.

**5.9.1 `detect_polarity_switches(trajectory, threshold=0.5)`**

Detekcja momentów zmiany polaryzacji $p(t)$. Algorytm:
1. Filtr medianowy anti-spike na $m_z(t)$ w centrum rdzenia
2. Threshold crossing z hysterezą: $p(t) = \text{sign}(m_z(t))$ z deadband $\pm\epsilon$
3. Refractory period $\tau_{\min}$ — ignoruje oscylacje między przerzutami

```python
@dataclass
class PolaritySwitchEvent:
    time: float          # [s] moment przerzutu
    from_p: int          # polarity before (-1 or +1)
    to_p: int            # polarity after
    duration: float      # [s] transition duration (time in deadband)
    
def detect_polarity_switches(
    trajectory: TrajectoryResult,
    threshold: float = 0.5,        # |m_z| threshold for polarity
    refractory: float = 0.5e-9,    # [s] minimum time between events
    hysteresis: float = 0.1,
) -> list[PolaritySwitchEvent]: ...
```

**5.9.2 `detect_state_switches(trajectory, gc_classifier)`**

Detekcja przejść G-state ↔ C-state w sygnale ciągłym:
1. Running G/C classification w sliding window
2. Change-point detection (PELT / Binseg) na parametrze $S_2(t)$
3. Filtr minimum dwell time — ignoruje fluktuacje krótsze niż $N$ okresów

```python
@dataclass
class StateSwitchEvent:
    time: float          # [s] moment przejścia
    from_state: str      # "G-state" | "C-state"
    to_state: str
    confidence: float    # [0-1]

def detect_state_switches(
    trajectory: TrajectoryResult,
    magnetization: np.ndarray,
    window_periods: int = 5,
    min_dwell_periods: int = 3,
) -> list[StateSwitchEvent]: ...
```

**5.9.3 `detect_core_expulsion(trajectory, disk_radius)`**

Detekcja wyrzutu rdzenia poza dysk ($r > R_{\text{disk}}$):
- Zwraca czas wyrzutu, czas powrotu (jeśli nastąpił), kierunek wyrzutu
- Istotne dla eksperymentów z silnym STT

**5.9.4 `dwell_time_statistics(events)`**

Statystyki czasu przebywania w poszczególnych stanach:

$$
\langle \tau_G \rangle, \quad \langle \tau_C \rangle, \quad P(\tau > t) = e^{-t/\tau_{\text{char}}}
$$

Histogram dwell times + exponential fit → characteristic time per state.

### 5.10 Energy Landscape — potencjał efektywny (v3.0)

Nowy submoduł `solitons/vortex/energy/` — rekonstrukcja potencjału z trajektorii.

**5.10.1 `energy_vs_time(dataset, components=True)`**

Oblicz czas-rozdzielczą energię: $E_{\text{ex}}(t)$, $E_{\text{demag}}(t)$, $E_{\text{Zeeman}}(t)$, $E_{\text{total}}(t)$.
Dane z tabel energii mumax³/OOMMF (zarr: `table/E_total`, etc.)

**5.10.2 `effective_potential_W(trajectory, method="boltzmann")`**

Rekonstrukcja potencjału efektywnego $W(R)$ z histogramu pozycji rdzenia:

$$
W(R) = -k_B T_{\text{eff}} \ln P(R) + \text{const}
$$

gdzie $P(R)$ — rozkład radialny. Metoda Boltzmann inversion (wymaga T>0 symulacji).

Alternatywnie: `method="force_balance"` — z bilansu sił Thiele'a (§5.6):

$$
W(R) = -\int_0^R \left[\mathbf{G} \times \dot{\mathbf{R}} - D\dot{\mathbf{R}} - \mathbf{F}_{\text{STT}}\right] \cdot d\mathbf{R}
$$

```python
@dataclass
class EffectivePotentialResult:
    R: np.ndarray           # [m] radial positions
    W: np.ndarray           # [J] potential values
    kappa_2: float          # [J/m²] harmonic stiffness (from quadratic fit)
    kappa_4: float          # [J/m⁴] anharmonic coefficient (quartic correction)
    method: str             # "boltzmann" | "force_balance"
```

**5.10.3 `pinning_sites(trajectory, potential)`**

Detekcja miejsc pinningowych (metastabilnych minimów):
- Z histogramu pozycji: lokalne maksima $P(x,y)$
- Z potencjału $W(x,y)$: lokalne minima + bariera energetyczna
- Zbiór `PinningSite(position, depth, escape_rate)`
- Szczególnie istotne dla granularnych STNO (Shreya et al. 2023)

### 5.11 Experimental Signals — sygnały MTJ/GMR (v3.0)

Nowy submoduł `solitons/vortex/signals/` — generacja sygnałów obserwowalnych eksperymentalnie.

> **Motywacja:** Symulacje dają pełen dostęp do $\mathbf{m}(\mathbf{r}, t)$, ale eksperyment
> mierzy tylko skalar: rezystancję $R(t)$, napięcie $V(t)$, moc $P(f)$.
> MMPP powinno umożliwić bezpośrednie porównanie symulacja↔eksperyment.

**5.11.1 `magnetoresistance_signal(dataset, geometry="MTJ")`**

Rezystancja tunelowa (TMR) lub giant MR:

$$
R(t) = R_0 \left[1 + \frac{\text{TMR}}{2}\,\langle \mathbf{m}_{\text{free}} \cdot \hat{\mathbf{p}}\rangle(t)\right]^{-1}
$$

gdzie $\hat{\mathbf{p}}$ — kierunek polaryzatora (konfigurowalny), $\langle\cdot\rangle$ — średnia po volume.

**5.11.2 `voltage_signal(R_t, I_dc, I_ac=0, f_ac=0)`**

$$
V(t) = R(t) \cdot [I_{DC} + I_{AC}\sin(2\pi f_{AC} t)]
$$

**5.11.3 `power_spectrum(V_t, method="welch")`**

$$
P(f) = |V(f)|^2 / R_L
$$

z konfigurowalną impedancją obciążenia $R_L = 50\,\Omega$.

```python
@dataclass
class ExperimentalSignalResult:
    time: np.ndarray           # [s]
    resistance: np.ndarray     # [Ω] R(t)  
    voltage: np.ndarray        # [V] V(t)
    frequency: np.ndarray      # [Hz] for PSD
    power_spectrum: np.ndarray # [W/Hz] P(f)
    tmr_ratio: float           # TMR ratio used
    polarizer_direction: np.ndarray  # unit vector p̂
    
    @property
    def plt(self) -> "SignalPlotAccessor":
        return SignalPlotAccessor(self)
```

---

## 6. Integracja z istniejącym kodem

### 6.1 Punkt wejścia — rozszerzenie `DatasetAwareWrapper`

W `mmpp/core/dataset.py` — dodać property `solitons` analogicznie do `fft`:

```python
# W DatasetAwareWrapper:
@property
def solitons(self):
    """Return soliton analysis interface for this dataset."""
    if self._solitons is None:
        from ..solitons import DatasetSpecificSolitons
        self._solitons = DatasetSpecificSolitons(
            self.job_result,
            self.dataset_name,
            getattr(self.job_result, "_mmpp_ref", None),
            slice_info=self.slice_info,
        )
    return self._solitons
```

### 6.2 Punkt wejścia — rozszerzenie `ZarrJobResult`

W `mmpp/core/job.py`:

```python
@property
def solitons(self):
    """Get soliton analysis interface."""
    from ..solitons import SolitonInterface
    return SolitonInterface(self, self._mmpp_ref)
```

### 6.3 Współdzielenie z modułem FFT

Moduł `vortex.spectrum` powinien **korzystać** z istniejącego `fft.compute_fft.FFTCompute` zamiast duplikować:

```python
# W vortex/spectrum/gyration.py:
from ...fft.compute_fft import FFTCompute

def compute_gyration_spectrum(trajectory: TrajectoryResult, **kwargs) -> SpectrumResult:
    """Compute power spectrum of vortex core trajectory."""
    compute = FFTCompute()
    # Użyj trajectory.x i trajectory.y jako sygnału wejściowego
    ...
```

### 6.4 Współdzielenie z modułem analytical

Stałe fizyczne z `mmpp.analytical.constants` (MU0, GAMMA_E, gamma) — reuse, nie duplikować.

### 6.5 Migracja `vortex_classifier.py`

```
STARY: mmpp/fft/vortex_classifier.py (629 linii, monolityczny)
  │
  └── Rozbić na:
      ├── mmpp/solitons/vortex/modes/classifier.py  (logika klasyfikacji)
      ├── mmpp/solitons/vortex/modes/azimuthal.py   (indeks m)
      ├── mmpp/solitons/vortex/modes/radial.py       (indeks n)
      ├── mmpp/solitons/vortex/modes/models.py       (VortexModeResult)
      └── mmpp/solitons/vortex/topology/detection.py (find_core_center)
```

Po migracji: `mmpp/fft/vortex_classifier.py` → deprecation wrapper z `warnings.warn`.

### 6.6 Batch API — ekstrakcja parametrów po wielu symulacjach (v2.0)

Typowy workflow w STNO to **seria symulacji przy różnych $I_{\text{STT}}$**. Batch API pozwala ekstrakcję parametrów Slavina-Tiberkevicza (i innych) z wielu jobów jednocześnie:

```python
from mmpp import MMPP

# Załaduj serię symulacji
mmpp = MMPP("/path/to/stno_current_sweep/")

# ── Batch: Slavin-Tiberkevich na serii prądów ──
st_batch = mmpp.batch.vortex.slavin_tiberkevich()
# → BatchSTResult z polami:
#   .currents   → np.array([1e-3, 2e-3, ..., 10e-3])  — I_STT
#   .frequencies → np.array([0.85, 0.87, ...]) GHz       — f(I)
#   .powers     → np.array([...])                        — p_gen(I)
#   .linewidths → np.array([...])                        — Δf(I)
#   .N          → float (fit globalny)
#   .threshold  → float (I_th z fit)

st_batch.plt.frequency_vs_current()    # → f(I_STT) z fitem ST
st_batch.plt.power_vs_current()        # → p(I_STT) z fitem ST
st_batch.plt.linewidth_vs_current()    # → Δf(I_STT)

# ── Batch: trajektorie ze wszystkich symulacji ──
trajs = mmpp.batch.vortex.track(method="gaussian")
# → list[TrajectoryResult], indeksowane po job

# ── Batch: tabela parametrów ──
mmpp.batch.vortex.summary_table()
# → Rich Table / DataFrame:
# | Job  | I_STT  | f [GHz] | p_gen   | Δf [MHz] | orbit_r [nm] | state |
# |------|--------|---------|---------|----------|---------------|-------|
# | 0    | 1.0 mA | —       | —       | —        | —             | sub-th|
# | 1    | 2.0 mA | 0.85    | 0.012   | 15.3     | 2.1           | G     |
# | ...  |        |         |         |          |               |       |
```

**Implementacja — w `mmpp/solitons/_batch.py`:**

```python
class VortexBatchInterface:
    """Batch operations across multiple jobs."""
    
    def __init__(self, mmpp_instance):
        self._mmpp = mmpp_instance
    
    def track(self, method="gaussian", **kwargs) -> list[TrajectoryResult]:
        """Track vortex core across all jobs."""
        return [
            job.m.vortex.track(method=method, **kwargs)
            for job in self._mmpp
            if self._has_vortex(job)
        ]
    
    def slavin_tiberkevich(self) -> BatchSTResult:
        """Extract ST parameters from current sweep."""
        results = []
        for job in self._mmpp:
            try:
                st = job.m.vortex.nonlinear.slavin_tiberkevich()
                results.append((job.attrs.get("I_STT"), st))
            except Exception:
                continue
        return BatchSTResult.from_pairs(results)
    
    def summary_table(self) -> "rich.table.Table":
        """Generate summary table across all jobs."""
        ...
```

Integracja z `MMPP`:

```python
# W mmpp/core/mmpp.py:
class MMPP:
    @property
    def batch(self):
        if self._batch is None:
            from ..solitons._batch import BatchSolitonInterface
            self._batch = BatchSolitonInterface(self)
        return self._batch
```

---

## 7. Cache i persystencja

### 7.1 Strategia cache (wzorzec z FFT)

```
<zarr_path>/
├── m_layer13/           # dane symulacji
├── solitons/            # ← NOWY cache namespace
│   ├── vortex/
│   │   ├── topology/
│   │   │   └── t0.zarr  # TopologyResult dla t=0
│   │   ├── core/
│   │   │   ├── trajectory_gaussian.zarr  # (N_time, 2) float64
│   │   │   └── trajectory_meta.json      
│   │   ├── spectrum/
│   │   │   └── gyration_welch_<hash>.zarr
│   │   └── modes/
│   │       └── classification_<freq>.json
```

### 7.2 Implementacja cache

```python
# solitons/vortex/_cache.py
class VortexCache:
    """Zarr-based + memory cache for vortex analysis results."""
    
    def __init__(self, zarr_path: str, dataset_name: str):
        self._zarr_path = zarr_path
        self._dataset = dataset_name
        self._memory: dict[str, Any] = {}
    
    def _cache_path(self, namespace: str) -> str:
        return f"{self._zarr_path}/solitons/vortex/{namespace}"
    
    def has(self, namespace: str, key: str) -> bool: ...
    def get(self, namespace: str, key: str) -> Any: ...
    def put(self, namespace: str, key: str, data: Any) -> None: ...
    def invalidate(self, namespace: str = None) -> None: ...
```

---

## 8. Wizualizacja — wzorce plotowania

### 8.1 Accessor pattern (konsekwentnie z FFT)

```python
class TrajectoryPlotAccessor:
    """Plotting namespace for TrajectoryResult."""
    
    def __init__(self, result: TrajectoryResult):
        self._r = result
    
    def xy(self, *, ax=None, component="both", **kwargs) -> Axes:
        """Plot X(t) and Y(t) core position."""
        ...
    
    def orbit_2d(self, *, ax=None, colorby="time", **kwargs) -> Axes:
        """Plot X vs Y orbit with color-coded time."""
        ...
    
    def orbit_polar(self, *, ax=None, **kwargs) -> Axes:
        """Plot r(θ) in polar coordinates."""
        ...
    
    def phase_portrait(self, *, ax=None, component="x", **kwargs) -> Axes:
        """Phase space: X vs dX/dt."""
        ...
    
    def overview(self, *, figsize=(14, 10), **kwargs) -> Figure:
        """4-panel overview: xy, orbit, spectrum, phase."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.xy(ax=axes[0, 0])
        self.orbit_2d(ax=axes[0, 1])
        # spectrum i phase z trajectory.phase
        ...
        return fig
```

### 8.2 Rich HTML repr (Jupyter)

Każdy `*Interface` i `*Result` powinien mieć `_repr_html_()` z kartą informacyjną:

```python
class VortexInterface:
    def _repr_html_(self) -> str:
        return f"""
        <div style="...gradient...">
          <h3>🌀 Vortex Analysis Interface</h3>
          <div>Dataset: {self._dataset_name}</div>
          <table>
            <tr><td>.topology</td><td>Topological invariants</td></tr>
            <tr><td>.core.track()</td><td>Core position tracking</td></tr>
            <tr><td>.trajectory</td><td>Orbit & phase analysis</td></tr>
            ...
          </table>
        </div>
        """
```

---

## 9. Plan wdrożenia — fazy

### Faza 1: Fundament (Sprint 1-2, ~2 tyg.)

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 1.1 | Utworzenie struktury katalogów `mmpp/solitons/` | `__init__.py` na każdym poziomie | P0 |
| 1.2 | `_base.py` — `SolitonConfig`, `SolitonResult` bazowe | 1 plik | P0 |
| 1.3 | `vortex/config.py` — `VortexConfig`, `TrackingConfig` dataclasses | 1 plik | P0 |
| 1.4 | `vortex/topology/detection.py` — `detect_topology()` (z $q(\mathbf{r})$) | 1 plik, ~200 LOC | P0 |
| 1.5 | `vortex/topology/invariants.py` — `polarity()`, `chirality()`, `winding_number()` | 1 plik, ~150 LOC | P0 |
| 1.6 | `vortex/topology/models.py` — `TopologyResult` dataclass | 1 plik | P0 |
| 1.7 | `vortex/core/tracking.py` — `core_track()` z 3 metodami (max, centroid, gaussian) | 1 plik, ~300 LOC | P0 |
| 1.8 | `vortex/core/models.py` — `TrajectoryResult` z propertiami (r, phi, velocity) | 1 plik, ~150 LOC | P0 |
| 1.9 | `vortex/interface.py` — `VortexInterface` (wiring sub-modułów) | 1 plik, ~200 LOC | P0 |
| 1.10 | Integracja z `DatasetAwareWrapper.solitons.vortex` | Edycja `dataset.py`, `job.py` | P0 |
| 1.11 | Testy: `tests/test_vortex_topology.py`, `tests/test_vortex_tracking.py` | 2 pliki | P0 |

**Deliverable Fazy 1:**
```python
job[0].m.solitons.vortex.topology.detect()    # → TopologyResult
job[0].m.solitons.vortex.core.track()          # → TrajectoryResult
job[0].m.solitons.vortex.core.track().plt.xy() # → matplotlib Axes
```

### Faza 2: Trajectory + Spectrum (Sprint 3-4, ~2 tyg.)

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 2.1 | `vortex/trajectory/orbit.py` — `OrbitFitter` (ellipse fit) | 1 plik, ~200 LOC | P1 |
| 2.2 | `vortex/trajectory/phase.py` — `PhaseAnalyzer` (Hilbert transform) | 1 plik, ~150 LOC | P1 |
| 2.3 | `vortex/trajectory/steady_state.py` — `TransientDetector` | 1 plik, ~100 LOC | P1 |
| 2.4 | `vortex/trajectory/filtering.py` — filtrowanie Savgol/median | 1 plik, ~100 LOC | P1 |
| 2.5 | `vortex/trajectory/interface.py` — `TrajectoryInterface` | 1 plik, ~150 LOC | P1 |
| 2.6 | `vortex/spectrum/gyration.py` — widmo z trajektorii (Welch) | 1 plik, ~150 LOC | P1 |
| 2.7 | `vortex/spectrum/spectrogram.py` — STFT / wavelet | 1 plik, ~200 LOC | P1 |
| 2.8 | Plotting accessors dla trajectory i spectrum | 4 pliki | P1 |
| 2.9 | Cache zarr dla trajectory i spectrum | Edycja `_cache.py` | P2 |
| 2.10 | Testy: `tests/test_vortex_trajectory.py`, `tests/test_vortex_spectrum.py` | 2 pliki | P1 |

**Deliverable Fazy 2:**
```python
traj = job[0].m.solitons.vortex.core.track()
traj.plt.orbit_2d()
traj.plt.overview()

orbit = job[0].m.solitons.vortex.trajectory.orbit.fit()  
orbit.eccentricity  # → 0.15

phase = job[0].m.solitons.vortex.trajectory.phase
phase.frequency()   # → ω(t) array
phase.plt.frequency_vs_time()

spec = job[0].m.solitons.vortex.spectrum.gyration()
spec.plt.power_spectrum()
```

### Faza 3: Mode Classification (Sprint 5, ~1 tyg.)

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 3.1 | Migracja + refaktoryzacja `vortex_classifier.py` | 4 pliki | P1 |
| 3.2 | `vortex/modes/azimuthal.py` — algorytm indeksu $m$ | 1 plik | P1 |
| 3.3 | `vortex/modes/radial.py` — algorytm indeksu $n$ | 1 plik | P1 |
| 3.4 | `vortex/modes/classifier.py` — zintegrowana klasyfikacja | 1 plik, ~250 LOC | P1 |
| 3.5 | Plotting: mode maps (amplitude, phase) | 1 plik | P1 |
| 3.6 | Integracja z `fft.modes` (reuse danych FFT) | Edycja interface | P2 |
| 3.7 | Testy: `tests/test_vortex_modes.py` | 1 plik | P1 |

**Deliverable Fazy 3:**
```python
modes = job[0].m.solitons.vortex.modes.classify_all()
# → [VortexModeResult(m=0, n=0, type="gyration", f=0.85 GHz), 
#    VortexModeResult(m=0, n=1, type="breathing", f=5.2 GHz), ...]

job[0].m.solitons.vortex.modes.plt.mode_map(f=5.2)
```

### Faza 4: Nonlinear Analysis (Sprint 6-7, ~2 tyg.)

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 4.1 | `vortex/nonlinear/thiele.py` — `ThieleAnalyzer` | 1 plik, ~250 LOC | P2 |
| 4.2 | `vortex/nonlinear/slavin_tiberkevich.py` — ekstrakcja parametrów ST | 1 plik, ~300 LOC | P2 |
| 4.3 | `vortex/nonlinear/amplitude_equation.py` — $\|c\|^2(t)$ | 1 plik, ~150 LOC | P2 |
| 4.4 | Plotting: power vs current, force balance | 2 pliki | P2 |
| 4.5 | Integracja z seriami symulacji (wielu $I_{\text{STT}}$) | Edycja interface | P2 |
| 4.6 | Testy: `tests/test_vortex_nonlinear.py` | 1 plik | P2 |

**Deliverable Fazy 4:**
```python
st = job[0].m.solitons.vortex.nonlinear.slavin_tiberkevich()
st.N                    # → nonlinear frequency shift
st.linewidth_hz         # → spectral linewidth
st.quality_factor       # → Q_osc
st.plt.power_vs_current()
```

### Faza 5: Polish + Docs (Sprint 8, ~1 tyg.)

| # | Zadanie | Priorytet |
|---|---------|-----------|
| 5.1 | Rich HTML `_repr_html_()` dla wszystkich interfejsów | P1 |
| 5.2 | Dokumentacja Sphinx/MkDocs (docstrings + tutorial notebook) | P1 |
| 5.3 | CLI integration (`mmpp vortex track ...`) | P3 |
| 5.4 | Deprecation wrapper w `fft/vortex_classifier.py` | P2 |
| 5.5 | Performance optimization (vectorization, numba opcjonalnie) | P3 |
| 5.6 | Example notebook: pełny pipeline STNO | P1 |

### Faza 6: Reservoir Computing (Sprint 9-10, ~2 tyg.) — v2.0

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 6.1 | `vortex/reservoir/memory_capacity.py` — MC z trajektorii | 1 plik, ~200 LOC | P2 |
| 6.2 | `vortex/reservoir/kernel_rank.py` — SVD rank macierzy stanów | 1 plik, ~100 LOC | P2 |
| 6.3 | `vortex/reservoir/nonlinear_transform.py` — NLMC (Legendre) | 1 plik, ~150 LOC | P2 |
| 6.4 | `vortex/reservoir/separation.py` — separation property | 1 plik, ~100 LOC | P2 |
| 6.5 | `vortex/reservoir/readout.py` — Ridge/Tikhonov readout | 1 plik, ~200 LOC | P2 |
| 6.6 | `build_state_matrix()` — konstrukcja features z trajektorii | 1 plik, ~150 LOC | P2 |
| 6.7 | Plotting: RC dashboard (MC, rank, NL, readout) | 1 plik | P2 |
| 6.8 | Testy: `tests/test_vortex_reservoir.py` | 1 plik | P2 |

**Deliverable Fazy 6:**
```python
# Reservoir Computing z symulacji STNO
traj = job[0].m.vortex.track()
rc = job[0].m.vortex.reservoir

mc = rc.memory_capacity(max_delay=50)
mc.MC_total              # → 4.2 (dla granularnego STNO)
mc.plt.mc_vs_delay()     # → MC_k vs k

rank = rc.kernel_rank()
rank.rank                # → 8 (efektywna wymiarowość)
rank.plt.singular_spectrum()

# Train readout for benchmark task
readout = rc.readout.train(X_states, y_target, method="ridge")
readout.nrmse_test       # → 0.12
```

### Faza 7: Coupled Vortex Analysis (Sprint 11, ~1 tyg.) — v2.0 future

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 7.1 | `vortex/coupled/phase_locking.py` — injection locking analysis | 1 plik, ~200 LOC | P3 |
| 7.2 | `vortex/coupled/synchronization.py` — Kuramoto order parameter | 1 plik, ~150 LOC | P3 |
| 7.3 | Plotting: Arnold tongue, sync diagram | 1 plik | P3 |
| 7.4 | Testy: `tests/test_vortex_coupled.py` | 1 plik | P3 |

**Deliverable Fazy 7:**
```python
# Sprzężone worteksy (Hamadeh et al.)
lock = job[0].m.vortex.coupled.phase_locking(job_external)
lock.locked              # → True
lock.locking_range       # → (7.8, 8.2) GHz
lock.plt.arnold_tongue()
```

### Faza 8: Events + Energy + Signals (Sprint 12-13, ~2 tyg.) — v3.0

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 8.1 | `vortex/events/polarity.py` — polarity switch detection | 1 plik, ~150 LOC | P0 |
| 8.2 | `vortex/events/state_transitions.py` — G↔C transition detection | 1 plik, ~200 LOC | P1 |
| 8.3 | `vortex/events/core_expulsion.py` — core expulsion detection | 1 plik, ~100 LOC | P1 |
| 8.4 | `vortex/events/dwell_time.py` — dwell time statistics | 1 plik, ~100 LOC | P1 |
| 8.5 | `vortex/energy/potential.py` — effective potential W(R) | 1 plik, ~200 LOC | P1 |
| 8.6 | `vortex/energy/pinning.py` — pinning site detection | 1 plik, ~150 LOC | P2 |
| 8.7 | `vortex/signals/magnetoresistance.py` — R(t), V(t) | 1 plik, ~150 LOC | P1 |
| 8.8 | `vortex/signals/power_spectrum.py` — P(f) | 1 plik, ~100 LOC | P1 |
| 8.9 | Plotting accessors for events, energy, signals | 3 pliki | P1 |
| 8.10 | Testy: `tests/test_vortex_events.py`, `tests/test_vortex_energy.py` | 2 pliki | P1 |

**Deliverable Fazy 8:**
```python
# Event detection — game changer for experimental data
events = job[0].m.vortex.events.polarity_switches()
# → [PolaritySwitchEvent(time=12.4ns, from_p=+1, to_p=-1), ...]

dwell = job[0].m.vortex.events.dwell_times(state="G-state")
dwell.mean_dwell_time    # → 3.2 ns
dwell.plt.dwell_histogram()

# Energy landscape
W = job[0].m.vortex.energy.effective_potential()
W.kappa_2                # → harmonic stiffness [J/m²]
W.plt.potential_landscape()

# Experimental signal comparison
sig = job[0].m.vortex.signals.magnetoresistance(tmr_ratio=0.4)
sig.plt.voltage_vs_time()
sig.plt.power_vs_frequency()
```

### Faza 9: Skyrmion Module + Shared Topology (Sprint 14-15, ~2 tyg.) — v3.0

| # | Zadanie | Pliki | Priorytet |
|---|---------|-------|-----------|
| 9.1 | `solitons/_topology.py` — shared Berg-Lüscher, guiding_center | 1 plik, ~300 LOC | P0 |
| 9.2 | Refactor `vortex/topology/` to import from `_topology.py` | Edycja | P0 |
| 9.3 | `skyrmion/__init__.py`, `interface.py`, `config.py` | 3 pliki | P1 |
| 9.4 | `skyrmion/topology/detection.py` — Q=±1 verification | 1 plik, ~150 LOC | P1 |
| 9.5 | `skyrmion/topology/helicity.py` — γ₀ (Néel/Bloch) | 1 plik, ~100 LOC | P1 |
| 9.6 | `skyrmion/tracking/center.py` — q-weighted guiding center | 1 plik, ~100 LOC | P1 |
| 9.7 | `skyrmion/tracking/radius.py` — R_sk from m_z profile | 1 plik, ~150 LOC | P1 |
| 9.8 | `skyrmion/dynamics/breathing.py` — R(t), ω_breath | 1 plik, ~150 LOC | P2 |
| 9.9 | `skyrmion/dynamics/hall_angle.py` — θ_Hall | 1 plik, ~100 LOC | P2 |
| 9.10 | Testy: `tests/test_skyrmion_topology.py`, `tests/test_skyrmion_tracking.py` | 2 pliki | P1 |

**Deliverable Fazy 9:**
```python
# Skyrmion analysis — parallel API to vortex
sk = job[0].m.solitons.skyrmion
topo = sk.topology.detect()
topo.Q                    # → -1.0 (Berg-Lüscher)
topo.helicity             # → π/2 (Bloch-type)

traj = sk.tracking.center()
traj.plt.xy()
sk.dynamics.hall_angle()  # → θ_Hall ≈ 40°
```

---

## 10. Zależności

### 10.1 Wymagane (core)

```
numpy >= 1.24
scipy >= 1.10       # signal processing, optimize (curve_fit), ndimage
zarr >= 2.16        # cache/persystencja
```

### 10.2 Opcjonalne

```
scikit-image         # zaawansowana segmentacja (topology)
numba                # przyspieszenie core tracking (~10x)
matplotlib >= 3.7    # wizualizacja
rich                 # Jupyter repr
```

### 10.3 Nowe wpisy w `pyproject.toml`

```toml
[project.optional-dependencies]
solitons = ["scipy>=1.10"]
solitons-full = ["scipy>=1.10", "scikit-image>=0.21", "numba>=0.58"]
```

---

## 11. Testowanie

### 11.1 Strategia testów

| Poziom | Testy | Narzędzia |
|--------|-------|-----------|
| Unit | Algorytmy (tracking, topology, orbit fit) na danych syntetycznych | pytest, numpy.testing |
| Integration | Pipeline end-to-end z zarr fixtures | pytest + zarr temp dirs |
| Regression | Porównanie z wynikami referencyjnymi | numpy.testing.assert_allclose |
| Visual | Screenshot tests dla plotów (opcjonalnie) | pytest-mpl |

### 11.2 Dane syntetyczne do testów

```python
def create_synthetic_vortex(
    Nx=64, Ny=64, Nt=200, 
    R_core=3, f_gyr=0.8e9, 
    orbit_radius=2e-9
) -> zarr.Group:
    """Generuje syntetyczny worteks z orbitalną gyracją."""
    # Stan statyczny: m_z Gauss w centrum, in-plane curling
    # Dynamika: circular orbit z ω_G = 2π f_gyr
    ...
```

### 11.3 Testy specyficzne v3.0

> **Berg-Lüscher Q stability tests:**
> - Worteks na siatce 32×32, 64×64, 128×128 → Q(BL) powinno convergować do 0.5, |Q-0.5| < 0.01 dla 128×128
> - Skyrmion syntetyczny (Bloch, Q=-1) → test: |Q(BL)+1| < 0.005
> - Finite_diff vs berg_luscher: oba metody ≈ same value ± 0.05 dla smooth vortex
> - Zniekształcony worteks (edge effects) → BL stabilniejsze niż finite_diff

> **Regime transition tests:**
> - Syntetyczna trajektoria z ręcznie ustawionym przerzutem polaryzacji → `detect_polarity_switches()` musi znaleźć ≥1 event
> - Trajektoria z G→C transition (zmiana amplitudy + deformacja) → `detect_state_switches()` musi sklasyfikować oba segmenty
> - Dwell time z exponential distribution → fit w `dwell_time_statistics()` powinien odzyskać τ ± 10%

> **Invariance tests:**
> - Translational: przesunięcie siatki o (dx, dy) → Q, p, w, C bez zmian
> - Rotational: rotation siatki o 90° → Q, p bez zmian; C i w mogą się odwrócić jeśli rotacja odwraca handedness (test konsystencji)
> - phase(z(t)) CCW vortex: ω > 0; CW vortex: ω < 0 — test znaku
> - Directional spectrum: synthetic CCW orbit → S₊ ≫ S₋; CW orbit → S₋ ≫ S₊

---

## 12. Wzorzec kodu — przykład implementacji

### 12.1 `vortex/interface.py` — główny entry point

```python
"""Main Vortex analysis interface — entry point for all vortex sub-modules."""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.job import ZarrJobResult

from .config import VortexConfig


class VortexInterface:
    """
    Vortex dynamics analysis interface.
    
    Provides hierarchical access to topology, core tracking,
    trajectory analysis, spectral analysis, mode classification,
    and nonlinear parameter extraction.
    
    Access: job[0].m.solitons.vortex
    """
    
    def __init__(
        self,
        job_result: "ZarrJobResult",
        dataset_name: Optional[str] = None,
        mmpp_instance: Optional[Any] = None,
        slice_info: Optional[Any] = None,
    ):
        self._job = job_result
        self._dataset = dataset_name
        self._mmpp = mmpp_instance
        self._slice_info = slice_info
        self._config = VortexConfig()
        
        # Lazy-loaded sub-interfaces
        self._topology = None
        self._core = None
        self._trajectory = None
        self._spectrum = None
        self._modes = None
        self._nonlinear = None
        self._reservoir = None   # v2.0
        self._coupled = None     # v2.0
    
    @property
    def config(self) -> VortexConfig:
        """Mutable configuration object."""
        return self._config
    
    @property
    def topology(self):
        """Topological invariants analysis."""
        if self._topology is None:
            from .topology import TopologyInterface
            self._topology = TopologyInterface(
                self._job, self._dataset, self._slice_info, self._config
            )
        return self._topology
    
    @property
    def core(self):
        """Vortex core tracking."""
        if self._core is None:
            from .core import CoreInterface
            self._core = CoreInterface(
                self._job, self._dataset, self._slice_info, self._config
            )
        return self._core
    
    @property
    def trajectory(self):
        """Trajectory analysis (orbit, phase, steady-state)."""
        if self._trajectory is None:
            from .trajectory import TrajectoryInterface
            self._trajectory = TrajectoryInterface(
                self._job, self._dataset, self._slice_info, 
                self._config, self.core
            )
        return self._trajectory
    
    @property
    def spectrum(self):
        """Vortex-specific spectral analysis."""
        if self._spectrum is None:
            from .spectrum import VortexSpectrumInterface
            self._spectrum = VortexSpectrumInterface(
                self._job, self._dataset, self._slice_info, 
                self._config, self.core
            )
        return self._spectrum
    
    @property
    def modes(self):
        """Vortex mode classification (m, n, l indices)."""
        if self._modes is None:
            from .modes import VortexModesInterface
            self._modes = VortexModesInterface(
                self._job, self._dataset, self._slice_info, self._config
            )
        return self._modes
    
    @property
    def nonlinear(self):
        """Nonlinear analysis (Slavin-Tiberkevich, Thiele)."""
        if self._nonlinear is None:
            from .nonlinear import NonlinearInterface
            self._nonlinear = NonlinearInterface(
                self._job, self._dataset, self._slice_info,
                self._config, self.core, self.trajectory
            )
        return self._nonlinear
    
    @property
    def reservoir(self):
        """Reservoir computing metrics (v2.0 — Shreya et al. 2023)."""
        if self._reservoir is None:
            from .reservoir import ReservoirInterface
            self._reservoir = ReservoirInterface(
                self._job, self._dataset, self._slice_info,
                self._config, self.core
            )
        return self._reservoir
    
    @property
    def coupled(self):
        """Coupled vortex analysis (v2.0 future — Hamadeh et al.)."""
        if self._coupled is None:
            from .coupled import CoupledVortexInterface
            self._coupled = CoupledVortexInterface(
                self._job, self._dataset, self._slice_info, self._config
            )
        return self._coupled
    
    # ── Shortcut aliases (v2.0) ──────────────────────────────
    
    def track(self, method: str = "gaussian", **kwargs):
        """Alias: self.core.track(method, **kwargs)"""
        return self.core.track(method=method, **kwargs)
    
    def track_avg_m(self, **kwargs):
        """Alias: self.core.track(method='average_magnetization', **kwargs)"""
        return self.core.track(method="average_magnetization", **kwargs)
    
    def detect(self, **kwargs):
        """Alias: self.topology.detect(**kwargs)"""
        return self.topology.detect(**kwargs)
```

---

## 13. Podsumowanie

| Aspekt | Wartość |
|--------|--------|
| **Nowych plików** | ~75 (vortex ~60, skyrmion ~12, shared ~3) |
| **Estymowany LOC** | ~8000-10000 |
| **Sub-modułów vortex** | 12 (topology, core, trajectory, spectrum, modes, nonlinear, reservoir, coupled, events, energy, signals, _cache) |
| **Sub-modułów skyrmion** | 4 (topology, tracking, dynamics, _plotting) |
| **Shared engine** | 1 (`_topology.py` — Berg-Lüscher, guiding_center) |
| **Fazy wdrożenia** | 9 (fundamenty → trajectory → modes → nonlinear → polish → RC → coupled → events/energy/signals → skyrmion) |
| **Czas estymowany** | 14–15 sprintów (~15 tygodni) |
| **Kompatybilność wsteczna** | 100% — nowy namespace `solitons`, zero breaking changes |
| **Reuse istniejącego kodu** | `fft.compute_fft`, `analytical.constants`, `core.dataset` |
| **Nowe w v2.0** | aliasy skrótowe, batch API, `average_magnetization`, G/C-state, RC, coupled |
| **Nowe w v3.0** | Sign conventions (p,w,C,Q,γ₀), Berg-Lüscher, z(t) phase, CW/CCW spectrum, core-centered modes, Thiele decomp, F_Oe, events/, energy/, signals/, skyrmion module, shared _topology |

### Kolejne kroki

1. ✅ Plan wdrożenia v1.0 (ten dokument)
2. ✅ Review + aktualizacja do v2.0 (aliasy, batch, avg_m, G/C-state, RC, coupled)
3. ✅ Audyt fizyczny + aktualizacja do v3.0 (konwencje znaków, Berg-Lüscher, z(t) faza, CW/CCW, events, energy, signals, skyrmion)
4. ⬜ Faza 1: Utworzenie struktury + topology + core tracking
5. ⬜ Faza 2: Trajectory + Spectrum
6. ⬜ Faza 3: Mode Classification (migracja vortex_classifier.py)
7. ⬜ Faza 4: Nonlinear Analysis
8. ⬜ Faza 5: Polish + Docs + Examples
9. ⬜ Faza 6: Reservoir Computing (Shreya et al. 2023)
10. ⬜ Faza 7: Coupled Vortex Analysis (Hamadeh et al.)
11. ⬜ Faza 8: Events + Energy + Signals (v3.0)
12. ⬜ Faza 9: Skyrmion Module + Shared Topology (v3.0)

---

## Appendix A: Kluczowe referencje naukowe

1. **K.Y. Guslienko et al.**, *Eigenfrequencies of vortex state excitations in magnetic submicron-size disks*, J. Appl. Phys. **91**, 8037 (2002) — częstotliwość gyrotropowa
2. **A.A. Thiele**, *Steady-State Motion of Magnetic Domains*, Phys. Rev. Lett. **30**, 230 (1973) — równanie Thiele'a  
3. **A. Slavin & V. Tiberkevich**, *Nonlinear Auto-Oscillator Theory of Microwave Generation by Spin-Polarized Current*, IEEE Trans. Magn. **45**, 1875 (2009) — teoria ST
4. **A.S. Jenkins et al.**, *Spin torque driven higher order spin wave modes*, arXiv (2014) — wyższe mody
5. **A. Hamadeh et al.**, *Perfect and Robust Phase-Locking of a Spin Transfer Vortex Nano-Oscillator*, Appl. Phys. Lett. (2014) — sprzężone worteksy
6. **S. Wittrock et al.**, *Non-Hermiticity in the Vortex Nano-Oscillator*, (2024) — dynamiczny stan C / G-state classification
7. **S. Shreya, A. Jenkins, A. Rezaeiyan, W. Li, R. Böhnert et al.**, *Granular vortex spin-torque nano oscillator for reservoir computing*, Sci. Rep. **13**, 16553 (2023) — RC z granularnym STNO
8. **H. Jaeger**, *The "echo state" approach to analysing and training recurrent neural networks*, GMD Report **148** (2001) — fundament RC, memory capacity
9. **L. Appeltant et al.**, *Information processing using a single dynamical node as complex system*, Nat. Commun. **2**, 468 (2011) — time-multiplexed RC
10. **A.S. Jenkins et al.**, *Spin-orbit-assisted vortex-core reversal and mode selection*, Nat. Commun. **12**, 3002 (2021) — **v3.0**: mody wyższe zależne od pozycji rdzenia, core-centered analysis
11. **B. Berg & M. Lüscher**, *Definition and statistical distributions of a topological number in the lattice O(3) sigma-model*, Nucl. Phys. B **190**, 412 (1981) — **v3.0**: dyskretyzacja ładunku topologicznego
12. **Multi-vortex synchronization**, Nat. Commun. (2025), s42005-025-02006-3 — sprzężone dynamiki wielu worteksów
13. **R. Dussaux et al.**, *Large microwave generation from current-driven magnetic vortex oscillators in magnetic tunnel junctions*, Nat. Commun. **1**, 8 (2010) — **v3.0**: F_Oe importance w STNO

## Appendix B: Diagram architektury

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZarrJobResult (job[0])                        │
│ ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│ │ DatasetAware     │  │     .fft          │  │   .solitons     │ │
│ │ Wrapper          │  │  (istniejący)    │  │   (NOWY)        │ │
│ │  .m / .m_layer   │  │  .spectrum       │  │                 │ │
│ │  [slice support] │──│  .modes          │  │  .vortex ────┐  │ │
│ │                  │  │  .dispersion     │  │              │  │ │
│ │  .fft ──────────►│  │  .transmission   │  │  (.skyrmion) │  │ │
│ │  .solitons ─────►│──│                  │  │  (future)    │  │ │
│ │  .vortex (alias)─►│ │                  │  │              │  │ │
│ └─────────────────┘  └──────────────────┘  └──────────────┘  │ │
│                                                    │           │
│  ┌────────────────────────────────────────────┐    │           │
│  │   MMPP.batch.vortex (v2.0)                 │    │           │
│  │   • .track() → list[TrajectoryResult]      │    │           │
│  │   • .slavin_tiberkevich() → BatchSTResult  │    │           │
│  │   • .summary_table() → Rich Table          │    │           │
│  └────────────────────────────────────────────┘    │           │
└────────────────────────────────────────────────────┼───────────┘
                                                     │
                    ┌────────────────────────────────┘
                    ▼
    ┌──────────────────────────────────────────────────────┐
    │               VortexInterface                        │
    │  Aliasy: .track() .detect() .spectrum()              │
    │                                                      │
    │  ┌──────────┐  ┌──────────┐  ┌───────────────┐      │
    │  │ topology │  │   core   │  │  trajectory   │      │
    │  │          │  │          │  │               │      │
    │  │ • detect │  │ • track  │  │ • orbit       │      │
    │  │ • p,C,q  │  │ • avg_m  │  │ • phase       │      │
    │  │ • state  │  │ • pos(t) │  │ • steady_st.  │      │
    │  │ • G/C ◄──│──│ • vel(t) │  │               │      │
    │  └────┬─────┘  └────┬─────┘  └───────┬───────┘      │
    │       │              │                │              │
    │  ┌────┴─────┐  ┌────┴─────┐  ┌───────┴───────┐      │
    │  │ spectrum │  │  modes   │  │  nonlinear    │      │
    │  │          │  │          │  │               │      │
    │  │ • gyrat. │  │ • classif│  │ • Slavin-Tib. │      │
    │  │ • breath.│  │ • (m,n,l)│  │ • Thiele      │      │
    │  │ • STFT   │  │ • maps   │  │ • amplitude   │      │
    │  └──────────┘  └──────────┘  └───────────────┘      │
    │                                                      │
    │  ┌────────────────┐  ┌──────────────────────────┐    │
    │  │ reservoir v2.0 │  │  coupled v2.0 (future)   │    │
    │  │                │  │                          │    │
    │  │ • memory_cap.  │  │ • phase_locking          │    │
    │  │ • kernel_rank  │  │ • synchronization        │    │
    │  │ • NL transform │  │ • Kuramoto R             │    │
    │  │ • separation   │  │ • Arnold tongue          │    │
    │  │ • readout      │  │                          │    │
    │  └────────────────┘  └──────────────────────────┘    │
    │                                                      │
    │  ┌──────────┐  ┌──────────┐  ┌───────────────┐      │
    │  │events 3.0│  │energy 3.0│  │ signals 3.0   │      │
    │  │          │  │          │  │               │      │
    │  │• polarity│  │• W(R)    │  │ • R(t) TMR    │      │
    │  │• G↔C     │  │• pinning │  │ • V(t)        │      │
    │  │• expuls. │  │• E(t)    │  │ • P(f)        │      │
    │  │• dwell_t │  │          │  │               │      │
    │  └──────────┘  └──────────┘  └───────────────┘      │
    │                                                      │
    │  Każdy sub-moduł: compute.py + models.py + .plt.*    │
    └──────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │          solitons/_topology.py (shared v3.0)         │
    │  Berg-Lüscher Q | guiding_center | topological_dens │
    └───────────────────────┬──────────────────────────────┘
                            │
    ┌───────────────────────┴──────────────────────────────┐
    │            SkyrmionInterface (v3.0)                   │
    │                                                      │
    │  ┌──────────┐  ┌──────────┐  ┌───────────────┐      │
    │  │ topology │  │ tracking │  │  dynamics      │      │
    │  │          │  │          │  │               │      │
    │  │ • Q=±1   │  │ • center │  │ • breathing   │      │
    │  │ • γ₀ hel.│  │ • radius │  │ • Hall angle  │      │
    │  │ • type   │  │ • shape  │  │ • depinning   │      │
    │  └──────────┘  └──────────┘  └───────────────┘      │
    └──────────────────────────────────────────────────────┘
```
