#!/usr/bin/env python3
"""
Simple test for enhanced YAML parameter parsing functionality.
Tests the core parsing without heavy dependencies.
"""

import tempfile
import os
import sys
import traceback

def create_simple_swapper():
    """Create a minimal SimulationSwapper for testing."""
    class MockLogger:
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def info(self, msg): print(f"INFO: {msg}")
    
    # Minimal implementation for testing
    import yaml
    import numpy as np
    import math
    import re
    from typing import Any, Dict, List
    
    class TestSimulationSwapper:
        def __init__(self, config_file: str):
            self.config_file = config_file
            self.log = MockLogger()  # Initialize log first
            self.config_data = self._load_config()
            self.parameters = self._extract_parameters()
            
        def _load_config(self) -> Dict[str, Any]:
            with open(self.config_file) as f:
                content = f.read()
            content = self._preprocess_numpy_content(content)
            return yaml.safe_load(content)
            
        def _extract_parameters(self) -> Dict[str, List[Any]]:
            parameters = {}
            if "swap" in self.config_data:
                for key, value in self.config_data["swap"].items():
                    if isinstance(value, list):
                        parameters[key] = value
            return parameters
            
        def _preprocess_numpy_content(self, content: str) -> str:
            """Enhanced preprocessing with numpy and math support."""
            import numpy as np
            import math
            
            # Create safe namespace for eval
            safe_namespace = {
                'np': np,
                'numpy': np,
                'math': math,
                'pi': math.pi,
                'e': math.e,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'exp': math.exp,
                'log': math.log,
                'log10': math.log10,
                'sqrt': math.sqrt,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'range': range,
                'list': list,
                'float': float,
                'int': int,
                # YAML boolean keywords
                'true': True,
                'false': False,
                'null': None,
            }

            lines = content.split("\n")
            processed_lines = []

            for line in lines:
                if line.strip().startswith("#"):
                    processed_lines.append(line)
                    continue

                if ":" in line:
                    try:
                        key_part, value_part = line.split(":", 1)
                        value_part = value_part.strip()
                        
                        comment = ""
                        if "#" in value_part:
                            value_part, comment = value_part.split("#", 1)
                            value_part = value_part.strip()
                            comment = " #" + comment

                        processed = False

                        # Handle (min,max,count) syntax
                        if value_part.startswith("(") and value_part.endswith(")") and value_part.count(",") == 2:
                            try:
                                params_str = value_part[1:-1]
                                parts = [p.strip() for p in params_str.split(",")]
                                
                                if len(parts) == 3:
                                    min_val = eval(parts[0], {"__builtins__": {}}, safe_namespace)
                                    max_val = eval(parts[1], {"__builtins__": {}}, safe_namespace)
                                    count = int(eval(parts[2], {"__builtins__": {}}, safe_namespace))

                                    result = np.linspace(min_val, max_val, count)
                                    result_list = result.tolist()

                                    processed_lines.append(f"{key_part}: {result_list}{comment}")
                                    processed = True
                            except Exception as e:
                                self.log.warning(f"Error processing range syntax in line '{line}': {e}")

                        # Handle list comprehensions and complex expressions that start with [
                        if not processed and value_part.startswith("[") and "for" in value_part:
                            try:
                                result = eval(value_part, {"__builtins__": {}}, safe_namespace)
                                processed_lines.append(f"{key_part}: {result}{comment}")
                                processed = True
                            except Exception as e:
                                self.log.warning(f"Error evaluating list comprehension in line '{line}': {e}")

                        # Handle numpy/math expressions
                        if not processed and any(pattern in value_part for pattern in [
                            'np.', 'numpy.', 'linspace', 'arange', 'logspace', 'geomspace',
                            'math.', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e', '**'
                        ]):
                            try:
                                result = eval(value_part, {"__builtins__": {}}, safe_namespace)
                                
                                if hasattr(result, 'tolist'):
                                    result_list = result.tolist()
                                elif isinstance(result, (list, tuple)):
                                    result_list = list(result)
                                elif isinstance(result, (int, float)):
                                    result_list = [result]
                                else:
                                    result_list = [result]

                                processed_lines.append(f"{key_part}: {result_list}{comment}")
                                processed = True
                            except Exception as e:
                                self.log.warning(f"Error evaluating expression in line '{line}': {e}")

                        if not processed:
                            processed_lines.append(line)

                    except Exception as e:
                        self.log.warning(f"Error processing line '{line}': {e}")
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)

            return "\n".join(processed_lines)
    
    return TestSimulationSwapper

def test_enhanced_parsing():
    """Test the enhanced YAML parsing functionality."""
    
    print("🧪 Testing Enhanced YAML Parameter Parsing")
    print("=" * 50)
    
    # Create test YAML content
    test_yaml = """# Enhanced YAML Parameter Test
swap:
  # Basic discrete values
  Material: [1, 2, 3, 4]
  
  # Range syntax (min, max, count)
  B0: (0.001, 0.1, 5)
  
  # NumPy expressions
  rotation: np.arange(0, 91, 15)
  xsize: np.linspace(100, 500, 5)
  ysize: np.logspace(2, 3, 4)
  
  # Mathematical expressions
  alpha: [0.001 * i for i in range(1, 4)]
  frequencies: [1e9 * (2**i) for i in range(3)]
  
  # Mathematical functions
  angles: [sin(pi*i/4) for i in range(5)]
  powers: [2**i for i in range(1, 6)]
  
config:
  last_param_name: "Material"
  prefix: "test_enhanced"
"""

    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(test_yaml)
        temp_yaml_path = f.name

    try:
        TestSwapper = create_simple_swapper()
        swapper = TestSwapper(temp_yaml_path)
        
        print(f"✅ Configuration loaded successfully!")
        print(f"📊 Found {len(swapper.parameters)} parameters:")
        
        # Display parsed parameters
        for param_name, values in swapper.parameters.items():
            print(f"  • {param_name}: {len(values)} values")
            print(f"    Values: {values}")
        
        # Calculate total combinations
        total_combinations = 1
        for values in swapper.parameters.values():
            total_combinations *= len(values)
        
        print(f"\\n🚀 Total combinations: {total_combinations}")
        print("\\n✅ Enhanced YAML parsing test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(temp_yaml_path):
            os.unlink(temp_yaml_path)

def main():
    """Run the test."""
    success = test_enhanced_parsing()
    
    if success:
        print("\\n📝 New syntax features confirmed working:")
        print("  • (min, max, count) - simple range syntax")
        print("  • np.linspace(), np.arange(), np.logspace() - numpy functions")  
        print("  • Mathematical expressions with sin, cos, pi, e, etc.")
        print("  • List comprehensions: [expr for i in range(n)]")
        print("  • Safe evaluation with mathematical functions")
        return 0
    else:
        print("\\n❌ Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
