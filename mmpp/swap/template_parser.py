"""
Template parser for extracting parameters from .mx3 template files.
"""

import os
import re
from typing import Any
from pathlib import Path


class TemplateParser:
    """Parser for .mx3 template files to extract swappable parameters."""
    
    def __init__(self, template_path: str):
        """
        Initialize parser with template file path.
        
        Args:
            template_path: Path to the .mx3 template file
        """
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
    
    def extract_parameters(self) -> dict[str, dict[str, Any]]:
        """
        Extract all parameters from template file.
        
        Returns:
            Dictionary with parameter info including suggested values
        """
        with open(self.template_path, encoding='utf-8') as f:
            content = f.read()
        
        # Find all {parameter} patterns
        parameter_pattern = r'\{([^}]+)\}'
        parameters = re.findall(parameter_pattern, content)
        
        # Remove duplicates while preserving order
        unique_params = list(dict.fromkeys(parameters))
        
        param_info = {}
        for param in unique_params:
            param_info[param] = self._analyze_parameter(param, content)
        
        return param_info
    
    def _analyze_parameter(self, param_name: str, content: str) -> dict[str, Any]:
        """
        Analyze a parameter to suggest appropriate values.
        
        Args:
            param_name: Name of the parameter
            content: Full template content
            
        Returns:
            Dictionary with parameter analysis
        """
        info: dict[str, Any] = {
            'suggested_values': [],
            'type': 'unknown',
            'description': '',
            'usage_context': []
        }
        
        # Find lines containing the parameter
        lines = content.split('\n')
        usage_lines = [line.strip() for line in lines if f'{{{param_name}}}' in line]
        info['usage_context'] = usage_lines
        
        # Suggest values based on parameter name and context
        if param_name.lower() in ['material']:
            info['suggested_values'] = [1, 2, 3]
            info['type'] = 'integer'
            info['description'] = 'Material type (1=Py, 2=CoFeB, 3=other)'
            
        elif 'size' in param_name.lower():
            info['suggested_values'] = [100.0, 200.0, 300.0]
            info['type'] = 'float'
            info['description'] = 'Size parameter in nanometers'
            
        elif param_name.lower() in ['rotation']:
            info['suggested_values'] = [0, 45, 90]
            info['type'] = 'float'
            info['description'] = 'Rotation angle in degrees'
            
        elif param_name.lower() in ['sq_parm']:
            info['suggested_values'] = [0.5, 1.0, 1.5]
            info['type'] = 'float'
            info['description'] = 'Squircle parameter'
            
        elif 'b0' in param_name.lower():
            info['suggested_values'] = "np.linspace(0.005, 0.05, 10)"
            info['type'] = 'numpy_array'
            info['description'] = 'Magnetic field strength in Tesla'
            
        elif 'tx' in param_name.lower() or 'thickness' in param_name.lower():
            info['suggested_values'] = "np.linspace(1000e-9, 6000e-9, 6)"
            info['type'] = 'numpy_array'
            info['description'] = 'Thickness in meters'
            
        elif 'antenna' in param_name.lower() or 'anetnna' in param_name.lower():
            info['suggested_values'] = [0, 1]
            info['type'] = 'integer'
            info['description'] = 'Antenna mode (0=uniform, 1=complex)'
            
        else:
            # Generic suggestions based on usage context
            info['suggested_values'] = [1.0]
            info['type'] = 'float'
            info['description'] = f'Parameter {param_name} - please customize'
        
        return info
    
    def generate_yaml_template(self, output_prefix: str = "v1") -> str:
        """
        Generate a YAML template based on extracted parameters.
        
        Args:
            output_prefix: Prefix for simulation outputs
            
        Returns:
            YAML template string
        """
        params = self.extract_parameters()
        
        if not params:
            raise ValueError("No parameters found in template file")
        
        # Find a suitable last parameter (prefer B0-like parameters)
        last_param = None
        for param_name in params.keys():
            if 'b0' in param_name.lower():
                last_param = param_name
                break
        if not last_param:
            last_param = list(params.keys())[-1]
        
        # Generate YAML content
        yaml_lines = [
            "# MMPP Simulation Parameters Template",
            f"# Generated from: {self.template_path.name}",
            "#",
            "# Syntax:",
            "# - Use lists for discrete values: [value1, value2, value3]",
            "# - Use numpy arrays for ranges: np.linspace(start, stop, num)",
            "# - Comment out parameters to disable them (prefix with #)",
            "",
            "# Swap parameters:",
            "swap:"
        ]
        
        # Add parameters
        for param_name, info in params.items():
            yaml_lines.append(f"  # {info['description']}")
            if info['usage_context']:
                yaml_lines.append(f"  # Used in: {info['usage_context'][0]}")
            
            if isinstance(info['suggested_values'], str):
                # numpy array
                yaml_lines.append(f"  {param_name}: {info['suggested_values']}")
            else:
                # list of values
                yaml_lines.append(f"  {param_name}: {info['suggested_values']}")
            
            yaml_lines.append("")
        
        # Add config section
        yaml_lines.extend([
            "# Configuration options:",
            "config:",
            f'  last_param_name: "{last_param}"  # Parameter for final iteration',
            f'  prefix: "{output_prefix}"              # Simulation prefix/version',
            "  sbatch: true                      # Use SLURM batch system",
            "  full_name: false                  # Use full parameter names in paths",
            "",
            "  # Execution control:",
            "  minsim: 0                         # Minimum simulation index",
            "  maxsim: null                      # Maximum simulation index (null = no limit)",
            "  pairs: false                      # Use paired parameters instead of cartesian product",
            "  cleanup: false                    # Cleanup temporary files",
            "  check: false                      # Check simulation status",
            "  force: false                      # Force re-run completed simulations",
        ])
        
        return '\n'.join(yaml_lines)
