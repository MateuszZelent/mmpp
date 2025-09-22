#!/usr/bin/env python3
"""
Test script for enhanced YAML parameter parsing with numpy expressions and math.
Demonstrates the new features added to the swap functionality.
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the mmpp module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmpp'))

def test_yaml_parsing():
    """Test the enhanced YAML parsing with various syntaxes."""
    
    # Import the SimulationSwapper
    try:
        from mmpp.cli.swap.simulation import SimulationSwapper
    except ImportError as e:
        print(f"❌ Could not import SimulationSwapper: {e}")
        print("Make sure you're running this from the mmpp root directory")
        return False

    # Create test YAML content with various syntaxes
    test_yaml = """# Enhanced YAML Parameter Test
# Testing various parameter syntax options

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
  
  # Mixed expressions
  field_strength: np.array([0.01, 0.05, 0.1]) * 1.5
  
config:
  last_param_name: "Material"
  prefix: "test_enhanced"
  template_name: "template.mx3"
  sbatch: false
  pairs: false
"""

    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(test_yaml)
        temp_yaml_path = f.name

    try:
        print("🧪 Testing Enhanced YAML Parameter Parsing")
        print("=" * 50)
        
        # Create swapper instance
        print(f"📄 Loading config from: {temp_yaml_path}")
        swapper = SimulationSwapper(temp_yaml_path)
        
        # Get configuration info
        info = swapper.get_info()
        
        print(f"✅ Configuration loaded successfully!")
        print(f"📊 Found {len(info['parameters'])} parameters:")
        
        # Display parsed parameters
        for param_name in info['parameters']:
            if param_name in swapper.parameters:
                values = swapper.parameters[param_name]
                print(f"  • {param_name}: {len(values)} values")
                print(f"    Values: {values}")
        
        print(f"\n🚀 Total combinations: {info['total_combinations']}")
        
        # Validate configuration
        issues = swapper.validate_config()
        if issues:
            print("\n⚠️  Validation issues:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ Configuration is valid!")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing YAML parsing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_yaml_path):
            os.unlink(temp_yaml_path)

def test_template_generation():
    """Test template generation with new syntax examples."""
    
    try:
        from mmpp.cli.swap.simulation import TemplateParser
    except ImportError as e:
        print(f"❌ Could not import TemplateParser: {e}")
        return False
    
    # Create a test template file
    test_template = """{
    "Material": "{Material}",
    "B0": "{B0}",
    "rotation": "{rotation}",
    "xsize": "{xsize}",
    "ysize": "{ysize}",
    "alpha": "{alpha}",
    "Tx": "{Tx}"
}"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mx3', delete=False) as f:
        f.write(test_template)
        temp_template_path = f.name
    
    try:
        print("\n🔧 Testing Template Generation")
        print("=" * 50)
        
        parser = TemplateParser(temp_template_path)
        params = parser.get_parameters()
        
        print(f"📋 Found parameters: {params}")
        
        yaml_template = parser.generate_yaml_template(
            last_param="Material",
            prefix="enhanced_test",
            template_name="test_template.mx3"
        )
        
        print("📝 Generated YAML template:")
        print("-" * 30)
        print(yaml_template[:1000] + "..." if len(yaml_template) > 1000 else yaml_template)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing template generation: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(temp_template_path):
            os.unlink(temp_template_path)

def main():
    """Run all tests."""
    print("🧪 MMPP Enhanced YAML Parser Test Suite")
    print("=" * 60)
    
    # Test basic parsing
    success1 = test_yaml_parsing()
    
    # Test template generation
    success2 = test_template_generation()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed! Enhanced YAML parsing is working correctly.")
        print("\n📝 New syntax features available:")
        print("  • (min, max, count) - simple range syntax")
        print("  • np.linspace(), np.arange(), np.logspace() - numpy functions")
        print("  • Mathematical expressions with sin, cos, pi, e, etc.")
        print("  • List comprehensions: [expr for i in range(n)]")
        print("  • Safe evaluation with mathematical functions")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
