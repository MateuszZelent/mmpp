#!/usr/bin/env python3
"""
Demo script showing how to use Advanced Vortex Classifier with real MMPP data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def demo_vortex_analysis():
    """Demonstrate advanced vortex analysis with MMPP"""
    
    print("🌀 MMPP ADVANCED VORTEX/SKYRMION ANALYSIS DEMO")
    print("="*60)
    
    print("\n📋 This demo shows how to:")
    print("• Load vortex/skyrmion data with MMPP")
    print("• Perform advanced mode classification") 
    print("• Extract physical parameters (m, n indices)")
    print("• Analyze energy distributions and phase relations")
    print("• Compare with standard MMPP analysis")
    
    print(f"\n🔧 SETUP INSTRUCTIONS")
    print("-" * 30)
    
    setup_code = '''
# 1. Import MMPP with vortex classifier
from mmpp.fft.modes import FMRModeAnalyzer
from mmpp.fft.mode_characterization import ModeCharacteristicConfig

# 2. Configure for vortex analysis  
config = ModeCharacteristicConfig(
    use_vortex_classifier=True,
    vortex_dot_radius=200e-9,        # 200 nm dot radius
    quadrature_tolerance=0.4,        # strict quadrature check
    eta_parallel_for_gyr=0.7,       # gyration energy threshold  
    eta_perp_for_breath=0.6,        # breathing energy threshold
)

# 3. Load your vortex data
analyzer = FMRModeAnalyzer('your_vortex_data.zarr', mode_character_config=config)

# 4. Display interactive spectrum
analyzer.show_interactive_spectrum()
# -> Press 'v' for verbose analysis of selected frequency
'''
    
    print("```python")
    print(setup_code.strip())  
    print("```")
    
    print(f"\n📊 ANALYSIS EXAMPLES")
    print("-" * 30)
    
    examples = [
        ("Single Frequency Analysis", '''
# Analyze specific frequency with detailed output
result = analyzer.characterize_vortex_mode(
    frequency=8.5,          # GHz - adjust to your data
    verbose=True            # shows detailed breakdown
)

print(f"Mode classification: {result.mode_type}")
print(f"Azimuthal index m: {result.m_index}")  
print(f"Radial index n: {result.n_index}")
print(f"Rotation sense: {result.rotation_sense}")
print(f"Confidence: {result.confidence:.3f}")
'''),

        ("Batch Frequency Sweep", '''
# Analyze multiple frequencies  
import numpy as np
import pandas as pd

frequencies = np.arange(7.0, 11.0, 0.1)  # 7-11 GHz, 0.1 GHz step
results = []

for freq in frequencies:
    try:
        result = analyzer.characterize_vortex_mode(freq, verbose=False)
        results.append({
            'frequency_GHz': freq,
            'mode_type': result.mode_type,
            'm_index': result.m_index, 
            'n_index': result.n_index,
            'confidence': result.confidence,
            'E_parallel_fraction': result.E_parallel_frac,
            'rotation_sense': result.rotation_sense,
        })
    except Exception as e:
        print(f"⚠️  Failed at {freq:.1f} GHz: {e}")

# Create results DataFrame
df = pd.DataFrame(results)
print(df.head())
'''),

        ("Energy and Phase Analysis", '''
# Detailed energy and phase analysis
result = analyzer.characterize_vortex_mode(8.5, verbose=False)

print(f"📊 ENERGY ANALYSIS:")
print(f"   In-plane energy: {result.E_parallel:.2e}")
print(f"   Out-of-plane energy: {result.E_perp:.2e}")  
print(f"   In-plane fraction: {result.E_parallel_frac:.3f}")

print(f"\\n🔄 PHASE ANALYSIS:")
print(f"   mx-my phase diff: {result.delta_phi_xy*180/np.pi:.1f}°")
print(f"   Distance to quadrature: {result.dist_to_quadrature*180/np.pi:.1f}°")
print(f"   Phase coherence: {result.phase_coherence_xy:.3f}")
print(f"   mz phase uniformity: {result.std_phi_mz_on_ring:.3f} rad")

print(f"\\n📍 SPATIAL ANALYSIS:")  
cx, cy = result.core_position
print(f"   Core center: ({cx:.1f}, {cy:.1f}) pixels")
print(f"   Ring radius: {result.r_star:.2f}")
print(f"   Radial nodes: {len(result.radial_nodes)} at {result.radial_nodes}")
'''),

        ("Comparison with Standard Analysis", '''
# Compare advanced vortex vs standard MMPP analysis
freq = 8.5

# Standard MMPP characterization
std_result = analyzer.characterize_mode(freq, verbose=False)

# Advanced vortex classification  
vortex_result = analyzer.characterize_vortex_mode(freq, verbose=False)

print(f"COMPARISON AT {freq:.1f} GHz:")
print(f"{'Method':<15} {'Classification':<12} {'Confidence':<10} {'Details'}")
print("-" * 60)
print(f"{'Standard':<15} {std_result.primary_class:<12} {std_result.confidence:<10.3f} {std_result.labels}")
print(f"{'Vortex':<15} {vortex_result.mode_type:<12} {vortex_result.confidence:<10.3f} m={vortex_result.m_index}, n={vortex_result.n_index}")
'''),

        ("Mode Type Statistics", '''
# Statistics of mode types across frequency range
from collections import Counter

mode_counts = Counter([r['mode_type'] for r in results])
m_index_counts = Counter([r['m_index'] for r in results])

print(f"MODE TYPE DISTRIBUTION:")
for mode_type, count in mode_counts.items():
    fraction = count / len(results)
    print(f"   {mode_type:<12}: {count:>3} ({fraction:.1%})")

print(f"\\nAZIMUTHAL INDEX DISTRIBUTION:")  
for m_index, count in sorted(m_index_counts.items()):
    fraction = count / len(results)
    print(f"   m = {m_index:>2}: {count:>3} ({fraction:.1%})")
'''),
    ]
    
    for title, code in examples:
        print(f"\n🔹 **{title}**")
        print("```python")
        print(code.strip())
        print("```")
    
    print(f"\n🎮 INTERACTIVE USAGE")
    print("-" * 30)
    
    interactive = '''
# Interactive spectrum with advanced vortex analysis
analyzer.show_interactive_spectrum()

# Available keyboard shortcuts:
#   'c' - Standard characterization  
#   'v' - VERBOSE vortex analysis (NEW!)
#   'h' - Show help
#   Click spectrum to select frequency
#   Right-click to snap to nearest peak
'''
    
    print("```python")
    print(interactive.strip())
    print("```")
    
    print(f"\n📈 EXPECTED OUTPUT EXAMPLE")
    print("-" * 30)
    
    example_output = '''
🌀 ADVANCED VORTEX/SKYRMION MODE ANALYSIS
================================================================================
Frequency: 8.500 GHz
Final Classification: GYRATION
Confidence: 0.892
Mode indices: m=1, n=0

📍 CORE ANALYSIS:
   • Core center: (64.2, 63.8) pixels
   • Ring radius r*: 18.5 nm
   • Analysis radius: 4.2 nm  
   • Rotation sense: CCW

⚡ ENERGY DISTRIBUTION:
   • In-plane energy: 2.34e-01
   • Out-of-plane energy: 1.82e-02
   • In-plane fraction: 0.928
   • Out-of-plane fraction: 0.072

🔄 PHASE RELATIONSHIPS:
   • Phase diff mx-my: 87.3° (1.523 rad)
   • Distance to quadrature: 2.7°
   • Phase coherence xy: 0.891
   • mz phase std on ring: 0.145 rad

📊 MODE INDICES:
   • Azimuthal index m: 1 (clockwise rotation around core)
   • Radial index n: 0 (fundamental radial mode)
   • Radial nodes at: []

🌀 GYRATION SPECIFICS:
   • Core orbit radius: 0.042 (relative to dot radius)
   
📝 CLASSIFICATION NOTES:
   • |m|=1 supports gyration: m=1
   • Strong in-plane energy: 0.928  
   • Good quadrature: dist=0.047 rad
================================================================================
'''
    
    print(example_output)
    
    print(f"\n⚙️  CONFIGURATION OPTIONS")
    print("-" * 30)
    
    config_options = '''
ModeCharacteristicConfig parameters for vortex analysis:

• use_vortex_classifier: bool = False
    Enable advanced vortex analysis
    
• vortex_dot_radius: float = None  
    Dot radius in same units as spatial resolution
    If None, auto-estimated from data size
    
• quadrature_tolerance: float = 0.55
    Max deviation from ±π/2 for gyration (radians)
    
• eta_parallel_for_gyr: float = 0.6
    Min in-plane energy fraction for gyration
    
• eta_perp_for_breath: float = 0.6  
    Min out-of-plane energy fraction for breathing
    
• breathing_phase_uniformity: float = 0.65
    Max mz phase standard deviation for breathing

Advanced VortexClassificationConfig options:

• ring_thickness_factor: float = 0.04
    Ring width for analysis as fraction of R_dot
    
• nbins_radial: int = 96
    Number of radial bins for profile analysis
    
• node_amplitude_threshold: float = 0.25
    Threshold for radial node detection
    
• smoothing_kernel_size: int = 3
    Kernel size for radial profile smoothing
'''
    
    print(config_options)
    
    print(f"\n🎯 TIPS FOR BEST RESULTS")  
    print("-" * 30)
    
    tips = [
        "Set vortex_dot_radius accurately for your sample geometry",
        "Use verbose=True initially to understand classification logic", 
        "Compare with standard MMPP analysis for validation",
        "Adjust energy thresholds based on your material parameters",
        "Check phase coherence values for data quality assessment",
        "Use batch analysis for frequency-dependent mode evolution",
        "Interactive 'v' key provides instant detailed analysis"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"   {i}. {tip}")

if __name__ == "__main__":
    demo_vortex_analysis()
    
    print(f"\n" + "="*60)
    print(f"🚀 READY TO ANALYZE YOUR VORTEX/SKYRMION DATA!")
    print("="*60)
    print(f"📁 Replace 'your_vortex_data.zarr' with your actual data file")
    print(f"⚙️  Adjust configuration parameters for your system")
    print(f"🔬 Use verbose=True to see detailed physics analysis")
    print(f"🎮 Try interactive mode with 'v' key for exploration")
    print(f"\n✨ Happy vortex mode hunting! 🌀")