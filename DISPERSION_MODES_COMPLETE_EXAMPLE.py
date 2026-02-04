"""
KOMPLETNY PRZYKŁAD UŻYCIA dispersion_modes() Z WSZYSTKIMI PARAMETRAMI
====================================================================

Ten plik zawiera przykładową funkcję pokazującą WSZYSTKIE dostępne parametry
dla metody dispersion_modes() wraz z domyślnymi wartościami i dostępnymi opcjami.
"""

import mmpp


def dispersion_modes_complete_example(job, dataset_index=0):
    """
    Przykład użycia dispersion_modes() z WSZYSTKIMI możliwymi parametrami.
    
    Ten przykład pokazuje pełną konfigurację - w praktyce używaj tylko
    potrzebnych parametrów!
    """
    
    modes = (
        job[dataset_index].m_layer13[...]  # lub inna nazwa datasetu
        .fft.dispersion
        
        # =====================================================================
        # FILTRY (opcjonalne - można łączyć w łańcuch)
        # =====================================================================
        .filters(
            # --- Podstawowe flagi (legacy) ---
            remove_static=False,    # Odejmij początkowy krok czasowy
            average=False,          # Odejmij średnią czasową
            window=None,            # Opcje: None, 'time', 'space', 'both', '2d', ['time', 'space']
            
            # --- Filtry PRE (surowe dane M(t,x,y,z) przed FFT) ---
            pre={
                # Envelope extraction (wykrywanie obwiedni sygnału)
                "envelope_extraction": {
                    "enabled": False,
                    "threshold_std": 2.0,      # Próg w odchyleniach standardowych
                    "margin": 10,              # Margines w punktach czasowych
                },
                
                # Wavelet denoising (odszumianie falkowe 1D w czasie)
                "wavelet_denoise": {
                    "enabled": False,
                    "wavelet": "db4",          # Opcje: 'db4', 'sym5', 'coif3', etc.
                    "level": 3,                # Poziom dekompozycji: 1-6
                    "mode": "soft",            # Opcje: 'soft', 'hard'
                },
                
                # Wiener filter (filtr czasowy)
                "wiener_time": {
                    "enabled": False,
                    "noise_power": None,       # Auto jeśli None
                },
                
                # Median morphological (filtr medianowy)
                "median_morph": {
                    "enabled": False,
                    "kernel_size": 3,          # Rozmiar jądra (nieparzyste)
                },
                
                # Amplitude equalization (wyrównanie amplitudy)
                "amplitude_equalization": {
                    "enabled": False,
                    "smoothing_fraction": 0.05,  # Frakcja dla wygładzania
                },
                
                # Dynamic compression (kompresja dynamiki)
                "dynamic_compression": {
                    "enabled": False,
                    "method": "log",           # Opcje: 'log', 'sqrt', 'asinh'
                    "alpha": 10.0,             # Parametr dla asinh
                },
                
                # PSD adaptive (adaptacyjny PSD)
                "psd_adaptive": {
                    "enabled": False,
                },
                
                # ICA denoising (Independent Component Analysis)
                "ica_denoise": {
                    "enabled": False,
                    "n_components": None,      # Auto jeśli None
                },
                
                # Sparse denoising (odszumianie rzadkie)
                "sparse_denoise": {
                    "enabled": False,
                    "threshold": 0.1,
                },
                
                # Welch average (uśrednianie Welcha - SPECJALNE, wykonywane przy FFT)
                "welch_average": {
                    "enabled": False,
                    "n_segments": 4,           # Liczba segmentów: 2-12
                    "overlap": 0.5,            # Nakładanie: 0.0-0.9
                },
            },
            
            # --- Filtry POST (na spektrum S(k,f) po FFT) ---
            post={
                # FK bandpass (pasmo-przepustowy w k-f)
                "fk_bandpass": {
                    "enabled": False,
                    "f_min": 0.0,              # Hz
                    "f_max": 10e9,             # Hz
                    "k_min": -10.0,            # rad/μm (konwertowane wewnętrznie)
                    "k_max": 10.0,             # rad/μm
                },
                
                # SNR filter (filtr signal-to-noise)
                "snr_filter": {
                    "enabled": False,
                    "threshold_snr": 3.0,      # Próg SNR
                    "noise_percentile": 10.0,  # Percentyl do estymacji szumu
                },
                
                # Gaussian morphological (morfologiczny Gaussa)
                "gaussian_morph": {
                    "enabled": False,
                    "sigma_f": 1.0,            # Sigma dla częstotliwości
                    "sigma_k": 1.0,            # Sigma dla k
                    "threshold_std": 1.5,      # Próg w odchyleniach std
                },
                
                # Wiener 2D (filtr Wienera 2D)
                "wiener2d": {
                    "enabled": False,
                    "window_size": 5,          # Rozmiar okna (nieparzyste): 1-21
                },
                
                # Wavelet 2D (falki 2D)
                "wavelet2d": {
                    "enabled": False,
                    "wavelet": "db4",
                    "level": 2,
                },
            },
            
            # --- Filtry LIVE (szybkie, bez przeładowania danych) ---
            live={
                # Te same opcje co POST - mogą być przetwarzane "na żywo"
                "gaussian_morph": {
                    "enabled": False,
                    "sigma_f": 1.0,
                    "sigma_k": 1.0,
                },
                
                "fk_bandpass": {
                    "enabled": False,
                    "f_min": 0.0,
                    "f_max": 10e9,
                },
                
                "snr_filter": {
                    "enabled": False,
                    "threshold_snr": 3.0,
                },
            },
            
            # --- Advanced (zaawansowane - łączenie wszystkich powyższych) ---
            advanced=None,  # dict łączący powyższe sekcje
        )
        
        # =====================================================================
        # DISPERSION_MODES() - główna metoda
        # =====================================================================
        .dispersion_modes(
            # --- Cache i obliczenia ---
            save=False,              # Zapisz wynik do cache (bool)
            cache=None,              # Ścieżka do external cache, np. "/tmp/" (str | None)
            force=False,             # Wymuś przeliczenie (bool)
            
            # --- Parametry sieci ---
            lattice_constant_nm=470.0,  # Stała sieciowa w nanometrach (float)
            
            # --- Parametry compute_1d (przekazywane jako **compute_kwargs) ---
            
            # Oś propagacji
            axis="x",                # Opcje: "x" | "y"
            
            # Komponent magnetyzacji
            component=None,          # Opcje: None (auto), "perp", "mx", "my", "mz", "sum"
            
            # Uśrednianie przestrzenne
            avg_over_orthogonal=False,  # WAŻNE dla modów: False zachowuje info o y!
            
            # Tryb uśredniania ortogonalnego
            orthogonal_avg_mode="fft_power",  # Opcje gdy avg_over_orthogonal=True:
                                              # - "magnetization" (legacy default)
                                              # - "fft_power" (zalecane)
                                              # - "fft_abs"
                                              # - "fft_power_max"
                                              # - "fft_power_median"
            
            # Okna czasowe i przestrzenne
            time_window=None,        # Opcje: None, "hann"
            space_window=None,       # Opcje: None, "hann"
            
            # Detrending
            detrend=None,            # Opcje: None, "mean", "initial"
            
            # Brillouin zone folding
            fold_period=None,        # Okres fałdowania w metrach (float | None)
            fold_agg=None,           # Opcje: None, "sum", "max"
            
            # FFT convention correction
            flipx=True,              # Mirror flip k-axis (bool) - zazwyczaj True
            
            # Zaawansowane
            kmax=None,               # Ogranicz wynik do |k| ≤ kmax [rad/m] (float | None)
            use_cache=True,          # Użyj cache w pamięci (bool)
            disk_cache=True,         # Użyj cache na dysku (bool)
        )
    )
    
    return modes


# =============================================================================
# PRZYKŁADY UŻYCIA
# =============================================================================

def example_basic():
    """Najprostsze użycie - domyślne parametry"""
    modes = (
        job[0].m_layer13[...]
        .fft.dispersion
        .dispersion_modes(lattice_constant_nm=470)
    )
    modes.plot_interactive()


def example_with_cache():
    """Z zapisem do zewnętrznego cache"""
    modes = (
        job[0].m_layer13[...]
        .fft.dispersion
        .dispersion_modes(
            save=True,
            cache="/tmp/",
            force=False,
            lattice_constant_nm=470
        )
    )
    modes.plot_interactive(dpi=100)


def example_with_filters():
    """Z prostymi filtrami"""
    modes = (
        job[0].m_layer13[:600, ..., 0:1]
        .fft.dispersion
        .filters(
            remove_static=True,
            window="both",  # Hann w czasie i przestrzeni
            live={
                "gaussian_morph": {"enabled": True, "sigma_f": 1.0, "sigma_k": 1.0},
                "fk_bandpass": {"enabled": True, "f_min": 0.0, "f_max": 10e9},
            }
        )
        .dispersion_modes(save=True, force=True, lattice_constant_nm=470)
    )
    modes.plot_interactive()


def example_advanced_filters():
    """Zaawansowane filtry (compute + post + live)"""
    modes = (
        job[0].m_layer13[:600, ..., 0:1]
        .fft.dispersion
        .filters(
            remove_static=True,
            pre={
                "wavelet_denoise": {"enabled": True, "wavelet": "db4", "level": 3},
                "amplitude_equalization": {"enabled": True, "smoothing_fraction": 0.05},
                "welch_average": {"enabled": True, "n_segments": 4, "overlap": 0.5},
            },
            post={
                "snr_filter": {"enabled": True, "threshold_snr": 3.0},
            },
            live={
                "gaussian_morph": {"enabled": True, "sigma_f": 1.0, "sigma_k": 1.0},
                "fk_bandpass": {"enabled": True, "f_min": 0.0, "f_max": 10e9},
            }
        )
        .dispersion_modes(
            save=True,
            force=True,
            cache="/tmp/",
            lattice_constant_nm=470,
            axis="y",
            avg_over_orthogonal=False,  # KRYTYCZNE dla modów przestrzennych!
        )
    )
    modes.plot_interactive(dpi=150)


def example_y_axis():
    """Analiza wzdłuż osi Y zamiast X"""
    modes = (
        job[0].m_layer13[...]
        .fft.dispersion
        .dispersion_modes(
            axis="y",  # Propagacja wzdłuż Y
            lattice_constant_nm=470,
            avg_over_orthogonal=False,
        )
    )
    modes.plot_interactive()


# =============================================================================
# UWAGI WAŻNE
# =============================================================================

"""
1. avg_over_orthogonal=False jest KLUCZOWE dla rekonstrukcji modów przestrzennych!
   - True: średnia po Y/X → tylko 1D dyspersja S(k,f)
   - False: zachowuje oś Y/X → możliwa rekonstrukcja m(x,y)

2. Cache:
   - save=True, cache=None → zapis w job zarr
   - save=True, cache="/tmp/" → zapis w /tmp/mmpp_cache_<hash>/
   - force=True → wymuś przeliczenie nawet jeśli cache istnieje

3. Filtry:
   - PRE: przetwarzanie surowych danych M(t,x,y,z) PRZED FFT (wolne)
   - POST: przetwarzanie spektrum S(k,f) PO FFT (szybsze)
   - LIVE: ultra-szybkie przeładowanie z cache bez FFT (interaktywne)

4. Welch average:
   - Specjalny filtr wykonywany podczas FFT, nie przed
   - Redukuje szum kosztem rozdzielczości czasowej

5. Lattice constant:
   - W nanometrach!
   - Używane do fałdowania stref Brillouina (k + nG)
   - Można auto-wykryć w interaktywnym interfejsie

6. Component:
   - None → auto (użyje config.component)
   - "perp" → prostopadły do równowagi
   - "mx", "my", "mz" → konkretna składowa
   - "sum" → suma wszystkich składowych
"""
