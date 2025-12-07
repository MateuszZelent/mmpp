"""
Mode analysis and characterization functions.

Contains functions for analyzing and classifying FMR modes:
- characterize_mode: Classify modes into gyration/breathing/azimuthal families
- characterize_vortex_mode: Advanced vortex/skyrmion mode classification
- print_characterization_details: Detailed analysis output
"""

import numpy as np
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...mode_characterization import ModeCharacterizationResult, ModeCharacteristicConfig
    from ...mode_characterization.vortex_classifier import VortexModeResult

log = logging.getLogger("mmpp.fft.modes")

# Import analyzer components
from ...mode_characterization import ModeCharacterAnalyzer


def characterize_mode(
    analyzer_instance,
    frequency: float,
    z_layer: int = 0,
    *,
    core_position: Optional[tuple[float, float]] = None,
    analysis_radius: Optional[float] = None,
    config: Optional["ModeCharacteristicConfig"] = None,
    verbose: bool = False,
) -> "ModeCharacterizationResult":
    """
    Classify the mode at ``frequency`` into gyration/breathing/azimuthal families.

    Parameters:
    -----------
    analyzer_instance : FMRModeAnalyzer
        The analyzer instance
    frequency : float
        Frequency to analyze [GHz]
    z_layer : int, optional
        Layer index (default: 0)
    core_position : tuple[float, float], optional
        Core position in pixels (default: auto-detected)
    analysis_radius : float, optional
        Analysis radius (default: auto-determined)
    config : ModeCharacteristicConfig, optional
        Configuration for analysis (default: instance config)
    verbose : bool, optional
        Show detailed calculation results and classification criteria (default: False)

    Returns:
    --------
    ModeCharacterizationResult
        Classification result with detailed metrics
    """

    analyzer = (
        analyzer_instance._character_analyzer
        if config is None
        else ModeCharacterAnalyzer(config)
    )
    mode_data = analyzer_instance.get_mode(frequency, z_layer)
    result = analyzer.analyze(
        mode_data,
        core_position=core_position,
        analysis_radius=analysis_radius,
    )

    if verbose:
        print_characterization_details(analyzer_instance, result, frequency, z_layer)

    return result


def characterize_vortex_mode(
    analyzer_instance,
    frequency: float,
    z_layer: int = 0,
    *,
    core_position: Optional[tuple[float, float]] = None,
    R_dot: Optional[float] = None,
    config: Optional["ModeCharacteristicConfig"] = None,
    verbose: bool = False,
) -> "VortexModeResult":
    """
    Advanced vortex/skyrmion mode classification.

    Implements rigorous classification based on:
    - Thiele equation dynamics for gyration modes
    - Azimuthal index m from phase winding
    - Radial index n from amplitude nodes
    - Energy partitioning and phase coherence

    Parameters:
    -----------
    analyzer_instance : FMRModeAnalyzer
        The analyzer instance
    frequency : float
        Frequency to analyze [GHz]
    z_layer : int, optional
        Layer index (default: 0)
    core_position : tuple[float, float], optional
        Core position in pixels (default: auto-detected)
    R_dot : float, optional
        Dot radius in same units as spatial resolution (default: auto-estimated)
    config : ModeCharacteristicConfig, optional
        Configuration for analysis (default: instance config)
    verbose : bool, optional
        Show detailed vortex analysis (default: False)

    Returns:
    --------
    VortexModeResult
        Advanced classification with m,n indices, energies, and physics
    """

    analyzer = (
        analyzer_instance._character_analyzer
        if config is None
        else ModeCharacterAnalyzer(config)
    )
    mode_data = analyzer_instance.get_mode(frequency, z_layer)

    try:
        result = analyzer.analyze_vortex(
            mode_data,
            core_position=core_position,  # type: ignore
            R_dot=R_dot,
            verbose=verbose,
        )
        return result

    except ImportError as e:
        log.error(f"Advanced vortex classifier not available: {e}")
        print(f"❌ Advanced vortex classifier not available.")
        print(f"   Falling back to standard characterization...")

        # Fallback to standard analysis
        std_result = analyzer.analyze(
            mode_data,
            core_position=core_position,
        )

        if verbose:
            print_characterization_details(analyzer_instance, std_result, frequency, z_layer)

        # Convert to VortexModeResult format (basic mapping)
        from ...mode_characterization.vortex_classifier import VortexModeResult

        basic_result = VortexModeResult(
            frequency=frequency,
            m_index=0,  # would need proper analysis
            n_index=0,
            mode_type=std_result.primary_class,
            confidence=std_result.confidence,
            core_position=core_position or (0, 0),
            notes=[
                "Fallback to standard analysis - advanced vortex classifier unavailable"
            ],
        )

        return basic_result


def print_characterization_details(
    analyzer_instance,
    result: "ModeCharacterizationResult",
    frequency: float,
    z_layer: int,
) -> None:
    """Print detailed characterization analysis results."""
    print("\n" + "=" * 80)
    print(f"🔍 DETAILED MODE CHARACTERIZATION ANALYSIS")
    print("=" * 80)
    print(f"Frequency: {frequency:.3f} GHz, Layer: {z_layer}")
    print(f"Final Classification: {result.primary_class.upper()}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Labels: {', '.join(result.labels)}")

    print("\n📊 ENERGY DISTRIBUTION:")
    print(f"   • In-plane energy (Ex + Ey):  {result.energy_parallel:.6e}")
    print(f"   • Out-of-plane energy (Ez):   {result.energy_perp:.6e}")
    total_energy = result.energy_parallel + result.energy_perp
    parallel_ratio = (
        result.energy_parallel / total_energy if total_energy > 0 else 0
    )
    perp_ratio = result.energy_perp / total_energy if total_energy > 0 else 0
    print(f"   • In-plane ratio:             {parallel_ratio:.3f}")
    print(f"   • Out-of-plane ratio:         {perp_ratio:.3f}")
    print(f"   • Dominant component:         {result.dominant_component}")

    print("\n🌀 GYRATION ANALYSIS:")
    if result.m_index is not None:
        print(f"   • Winding number (m):         {result.m_index}")
        print(f"   • Winding quality:            {result.m_quality:.3f}")
        print(f"   • Rotation sense:             {result.rotation_sense or 'N/A'}")
    else:
        print(f"   • Winding number (m):         Not determined")

    if result.phase_xy_mean is not None:
        phase_deg = np.degrees(result.phase_xy_mean)
        print(
            f"   • Phase difference mx-my:     {phase_deg:.1f}° ({result.phase_xy_mean:.3f} rad)"
        )
        # Klasyfikacja kwadratura
        is_quadrature = abs(abs(result.phase_xy_mean) - np.pi / 2) < np.pi / 4
        print(
            f"   • Quadrature relation:        {'✅ YES' if is_quadrature else '❌ NO'} (±90° ± 45°)"
        )
    print(f"   • Phase coherence (mx,my):    {result.phase_xy_coherence:.3f}")

    print("\n💨 BREATHING ANALYSIS:")
    print(f"   • mz phase uniformity:        {result.phase_z_uniformity:.3f}")
    print(f"   • Radial nodes:               {result.radial_nodes}")
    breathing_indicator = result.phase_z_uniformity > 0.65  # z config
    print(
        f"   • Strong breathing mode:      {'✅ YES' if breathing_indicator else '❌ NO'} (uniformity > 0.65)"
    )

    print("\n📐 SPATIAL CHARACTERISTICS:")
    if "analysis_radius" in result.diagnostics:
        print(
            f"   • Analysis radius:            {result.diagnostics['analysis_radius']:.1f} pixels"
        )
    if "core_position" in result.diagnostics:
        cx, cy = result.diagnostics["core_position"]
        print(f"   • Core position:              ({cx:.1f}, {cy:.1f}) pixels")
    if "ring_coverage" in result.diagnostics:
        print(
            f"   • Ring coverage:              {result.diagnostics['ring_coverage']:.3f}"
        )

    print("\n🎯 CLASSIFICATION CRITERIA:")
    # Gyration criteria
    print("   GYRATION requires:")
    print(f"      - Winding |m| = 1:          {abs(result.m_index or 0) == 1}")
    print(f"      - In-plane dominance:       {parallel_ratio > 0.5}")
    print(f"      - Good phase coherence:     {result.phase_xy_coherence > 0.5}")
    is_quadrature = (
        result.phase_xy_mean is not None
        and abs(abs(result.phase_xy_mean) - np.pi / 2) < 0.55
    )  # z config
    print(f"      - Quadrature phase:         {is_quadrature}")

    # Breathing criteria
    print("\n   BREATHING requires:")
    print(f"      - Out-of-plane dominance:   {perp_ratio > 0.5}")
    print(f"      - mz phase uniformity:      {result.phase_z_uniformity > 0.65}")
    print(f"      - Low winding quality:      {result.m_quality < 0.5}")

    # Configuration thresholds
    config = analyzer_instance._character_analyzer.config
    print(f"\n⚙️  CONFIGURATION THRESHOLDS:")
    print(f"   • Amplitude threshold:        {config.relative_amplitude_threshold}")
    print(f"   • Quadrature tolerance:       {config.quadrature_tolerance:.3f} rad")
    print(f"   • Breathing uniformity:       {config.breathing_phase_uniformity}")
    print(f"   • Gyration parallel ratio:   {config.gyration_parallel_ratio}")

    print("=" * 80 + "\n")
