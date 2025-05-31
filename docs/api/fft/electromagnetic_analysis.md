# Electromagnetic Analysis Module

```{eval-rst}
.. automodule:: mmpp.fft.electromagnetic_analysis
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Classes

### ElectromagneticAnalyzer

```{eval-rst}
.. autoclass:: mmpp.fft.electromagnetic_analysis.ElectromagneticAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### ElectromagneticAnalysisConfig

```{eval-rst}
.. autoclass:: mmpp.fft.electromagnetic_analysis.ElectromagneticAnalysisConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Analysis Functions

```{eval-rst}
.. autofunction:: mmpp.fft.electromagnetic_analysis.comprehensive_electromagnetic_analysis
```

```{eval-rst}
.. autofunction:: mmpp.fft.electromagnetic_analysis.visualize_electromagnetic_analysis
```

## Usage Examples

### Basic Electromagnetic Analysis

```python
import mmpp
from mmpp.fft.electromagnetic_analysis import ElectromagneticAnalyzer

# Load data and get mode
db = mmpp.open('/path/to/data')
result = db.find(solver=3)[0]

# Get mode data at specific frequency
mode_data = result.fft.modes.get_mode(frequency=2.5)

# Create electromagnetic analyzer
analyzer = ElectromagneticAnalyzer()

# Analyze electromagnetic fields
em_results = analyzer.analyze_mode(mode_data)

print(f"Electric energy: {em_results['energy_densities']['electric'].sum():.2e} J/m³")
print(f"Magnetic energy: {em_results['energy_densities']['magnetic'].sum():.2e} J/m³")
```

### Comprehensive Analysis with Visualization

```python
from mmpp.fft.electromagnetic_analysis import comprehensive_electromagnetic_analysis

# Perform full analysis with visualization
analysis_results = comprehensive_electromagnetic_analysis(
    mode_data,
    include_far_field=True,
    include_energy_analysis=True,
    include_visualization=True
)

# Results include:
# - Energy density maps
# - Poynting vector field
# - Far-field radiation patterns
# - Current density analysis
```

### Far-Field Radiation Analysis

```python
# Compute far-field radiation pattern
far_field = analyzer.compute_far_field(
    mode_data.current_density,
    frequency=2.5,
    spatial_extent=mode_data.extent
)

# Visualize radiation pattern
fig = analyzer.visualize_radiation_pattern(
    far_field,
    title="Mode Radiation Pattern at 2.5 GHz"
)
```
