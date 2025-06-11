"""
Command Line Interface for MMPP library.
"""

import argparse
import sys

try:
    import yaml
except ImportError:
    yaml = None


def main() -> None:
    """Main entry point for the mmpp CLI."""
    parser = argparse.ArgumentParser(
        description="MMPP - Micro Magnetic Post Processing Library", prog="mmpp"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show library information")
    
    # Swap command group
    swap_parser = subparsers.add_parser("swap", help="Simulation swapping utilities")
    swap_subparsers = swap_parser.add_subparsers(dest="swap_command", help="Swap commands")
    
    # Swap init command
    swap_init_parser = swap_subparsers.add_parser("init", help="Initialize a parms.yml template")
    swap_init_parser.add_argument("--output", "-o", default="parms.yml",
                                help="Output file name (default: parms.yml)")
    swap_init_parser.add_argument("--force", "-f", action="store_true",
                                help="Overwrite existing file")
    
    # Swap run command
    swap_run_parser = swap_subparsers.add_parser("run", help="Run simulations from config file")
    swap_run_parser.add_argument("config_file", help="Path to the configuration file")
    swap_run_parser.add_argument("--dry-run", action="store_true",
                               help="Show what would be done without executing")
    
    # Swap info command
    swap_info_parser = swap_subparsers.add_parser("info", help="Show information about config file")
    swap_info_parser.add_argument("config_file", help="Path to the configuration file")
    
    # Swap validate command
    swap_validate_parser = swap_subparsers.add_parser("validate", help="Validate config file")
    swap_validate_parser.add_argument("config_file", help="Path to the configuration file")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "info":
        show_info()
    elif args.command == "swap":
        handle_swap_command(args)
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def handle_swap_command(args) -> None:
    """Handle swap-related commands."""
    if args.swap_command == "init":
        create_parms_template(args.output, args.force)
    elif args.swap_command == "run":
        run_simulations_from_config(args.config_file, args.dry_run)
    elif args.swap_command == "info":
        show_config_info(args.config_file)
    elif args.swap_command == "validate":
        validate_config_file(args.config_file)
    elif args.swap_command is None:
        print("Usage: mmpp swap <command>")
        print("Available commands:")
        print("  init      Initialize a parms.yml template")
        print("  run       Run simulations from config file")
        print("  info      Show information about config file")
        print("  validate  Validate config file")
    else:
        print(f"Unknown swap command: {args.swap_command}")
        sys.exit(1)


def create_parms_template(output_file: str, force: bool = False) -> None:
    """Create a template parms.yml file for simulation swapping."""
    import os
    from pathlib import Path

    if os.path.exists(output_file) and not force:
        print(f"Error: File '{output_file}' already exists. Use --force to overwrite.")
        sys.exit(1)

    template_content = '''# MMPP Simulation Parameters Template
# This file contains parameter definitions for simulation swapping
#
# Syntax:
# - Use lists for discrete values: [value1, value2, value3]
# - Use numpy arrays for ranges: np.linspace(start, stop, num)
# - Comment out parameters to disable them (prefix with #)
# - The last_param_name should match one of your parameter keys
#
# Required imports (automatically handled)
# import numpy as np

# Material parameters
Material: [3]

# Thickness parameters (in meters)
Tx: [6000e-9]
# Tx: np.linspace(1000e-9, 6000e-9, 6)

# Geometry parameters (in nanometers)
xsize: [100.0, 200.0]
ysize: [100.0, 200.0]
# xsize: np.linspace(350, 600, 3)
# ysize: np.linspace(350, 600, 3)

# Physical parameters
sq_parm: [1.0]
rotation: [0]
# rotation: np.linspace(0.0, 90.0, 3)

# Field parameters
# b01: [0.001]
anetnna: [0]
B0: np.linspace(0.005, 0.05, 10)

# Configuration options
config:
  last_param_name: "B0"        # Parameter name for the last iteration
  prefix: "v11"                # Simulation prefix/version
  sbatch: 1                    # Use SLURM batch system (1=true, 0=false)
  full_name: false             # Use full parameter names in paths

  # Advanced options
  template: "template.mx3"     # Template file for simulation
  main_path: "/path/to/simulations/"     # Main simulation directory
  destination_path: "/path/to/results/"  # Results destination

  # Execution control
  minsim: 0                    # Minimum simulation index
  maxsim: null                 # Maximum simulation index (null = no limit)
  pairs: false                 # Use paired parameters instead of cartesian product
  cleanup: false               # Cleanup temporary files
  check: false                 # Check simulation status
  force: false                 # Force re-run completed simulations
'''

    try:
        Path(output_file).write_text(template_content)
        print(f"✓ Created template file: {output_file}")
        print("\nNext steps:")
        print(f"1. Edit {output_file} to configure your simulation parameters")
        print("2. Run: mmpp swap run <your_params_file>")
        print("3. Validate config: mmpp swap validate <your_params_file>")
        print("4. Show info: mmpp swap info <your_params_file>")
        print("\nSee the template file for detailed configuration options.")
    except OSError as e:
        print(f"Error creating template file: {e}")
        sys.exit(1)


def show_info() -> None:
    """Show library information."""
    from . import __author__, __version__

    print(f"MMPP Library v{__version__}")
    print(f"Author: {__author__}")
    print("A library for Micro Magnetic Post Processing simulation and analysis")


def run_simulations_from_config(config_file: str, dry_run: bool = False) -> None:
    """Run simulations from a configuration file."""
    try:
        from .simulation import SimulationSwapper
        
        swapper = SimulationSwapper(config_file)
        
        # Validate config first
        issues = swapper.validate_config()
        if any(issue.startswith("ERROR") for issue in issues):
            print("❌ Configuration validation failed:")
            for issue in issues:
                if issue.startswith("ERROR"):
                    print(f"  {issue}")
            sys.exit(1)
        
        # Show warnings
        warnings = [issue for issue in issues if issue.startswith("WARNING")]
        if warnings:
            print("⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"  {warning}")
        
        if dry_run:
            info = swapper.get_info()
            print(f"📋 Dry run - would execute {info['total_combinations']} simulations")
            print(f"📁 Config file: {config_file}")
            print(f"🔧 Parameters: {', '.join(info['parameters'])}")
            print(f"📊 Parameter counts: {info['parameter_counts']}")
            print("✨ Use without --dry-run to execute")
        else:
            print(f"🚀 Starting simulation execution from: {config_file}")
            swapper.run_simulations()
            print("✅ Simulation execution completed")
            
    except ImportError:
        print("❌ Error: Required dependencies not available. Install with: pip install mmpp[dev]")
        sys.exit(1)
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"❌ Error running simulations: {e}")
        sys.exit(1)


def show_config_info(config_file: str) -> None:
    """Show detailed information about a configuration file."""
    try:
        from .simulation import SimulationSwapper
        
        swapper = SimulationSwapper(config_file)
        info = swapper.get_info()
        
        print(f"📋 Configuration Info: {config_file}")
        print("=" * 50)
        print(f"📊 Total combinations: {info['total_combinations']}")
        print(f"🔧 Parameters: {len(info['parameters'])}")
        
        print("\n📈 Parameter Details:")
        for param, count in info['parameter_counts'].items():
            print(f"  • {param}: {count} values")
        
        print("\n⚙️  Configuration Options:")
        config_opts = info['config_options']
        print(f"  • Prefix: {config_opts['prefix']}")
        print(f"  • Last parameter: {config_opts['last_param_name']}")
        print(f"  • SBATCH: {config_opts['sbatch']}")
        print(f"  • Pairs mode: {config_opts['pairs']}")
        print(f"  • Template: {config_opts['template']}")
        print(f"  • Main path: {config_opts['main_path']}")
        
        # Show validation issues
        issues = info['validation_issues']
        if issues:
            print("\n🔍 Validation Results:")
            for issue in issues:
                if issue.startswith("ERROR"):
                    print(f"  ❌ {issue}")
                elif issue.startswith("WARNING"):
                    print(f"  ⚠️  {issue}")
                else:
                    print(f"  ℹ️  {issue}")
        else:
            print("\n✅ Configuration is valid")
            
    except (ImportError, OSError, ValueError, AttributeError) as e:
        print(f"❌ Error reading configuration: {e}")
        sys.exit(1)


def validate_config_file(config_file: str) -> None:
    """Validate a configuration file and show results."""
    try:
        from .simulation import SimulationSwapper
        
        swapper = SimulationSwapper(config_file)
        issues = swapper.validate_config()
        
        print(f"🔍 Validating: {config_file}")
        print("=" * 50)
        
        if not issues:
            print("✅ Configuration is valid!")
            return
        
        errors = [issue for issue in issues if issue.startswith("ERROR")]
        warnings = [issue for issue in issues if issue.startswith("WARNING")]
        
        if errors:
            print("❌ Errors found:")
            for error in errors:
                print(f"  {error}")
        
        if warnings:
            print("⚠️  Warnings:")
            for warning in warnings:
                print(f"  {warning}")
        
        if errors:
            print("\n❌ Configuration has errors and cannot be used")
            sys.exit(1)
        else:
            print("\n✅ Configuration is valid (with warnings)")
            
    except (ImportError, OSError, ValueError, AttributeError) as e:
        print(f"❌ Error validating configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
