"""
MMPP Swap module for simulation parameter swapping and management.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[misc, assignment]

from .simulation import SimulationManager, SimulationSwapper, TemplateParser


def handle_swap_command(args: argparse.Namespace) -> None:
    """Handle swap-related commands."""
    if args.swap_command in ["init", "i"]:
        create_parms_template(args.template_file, args.output, args.force, args.prefix)
    elif args.swap_command in ["run", "r"]:
        run_simulations_from_config(args.config_file, args.dry_run)
    elif args.swap_command == "info":
        show_config_info(args.config_file)
    elif args.swap_command in ["validate", "v"]:
        validate_config_file(args.config_file)
    elif args.swap_command is None:
        print("Usage: mmpp swap <command>")
        print("Available commands:")
        print("  init (i)     Initialize a parms.yml template")
        print("  run (r)      Run simulations from config file")
        print("  info         Show information about config file")
        print("  validate (v) Validate config file")
    else:
        print(f"Unknown swap command: {args.swap_command}")
        sys.exit(1)


def create_parms_template(
    template_file: str, output_file: str, force: bool = False, prefix: str = "v1"
) -> None:
    """Create a template parms.yml file for simulation swapping by analyzing template.mx3."""
    # Check if template file exists in current directory
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        print("Make sure you're in the directory containing your template file.")
        sys.exit(1)

    if os.path.exists(output_file) and not force:
        print(f"❌ File '{output_file}' already exists. Use --force to overwrite.")
        sys.exit(1)

    try:
        parser = TemplateParser(template_file)
        parameters = parser.get_parameters()

        if not parameters:
            print(f"⚠️  No parameters found in {template_file}")
            print(
                "Make sure the template contains parameters in {parameter_name} format."
            )
            print("Example: {B0}, {xsize}, {Material}")
            sys.exit(1)

        print(f"🔍 Analyzing template: {template_file}")
        print(f"📊 Found {len(parameters)} parameters: {', '.join(parameters)}")

        # Generate YAML template with current working directory information
        current_dir = os.getcwd()
        yaml_content = parser.generate_yaml_template(
            last_param=parameters[-1] if parameters else "param1",
            prefix=prefix,
            template_name=template_file,
        )

        # Add working directory info to header
        yaml_header = f"""# MMPP Simulation Parameters Template
# Auto-generated from: {template_file}
# Working directory: {current_dir}
# Found parameters: {", ".join(parameters)}
#
# Syntax:
# - Use lists for discrete values: [value1, value2, value3]
# - Use numpy arrays for ranges: np.linspace(start, stop, num)
# - Comment out parameters to disable them (prefix with #)
# - The last_param_name should match one of your swap parameters
#
# Example numpy usage (uncomment and modify as needed):
# import numpy as np

"""

        # Replace the original header
        lines = yaml_content.split("\n")
        content_start = 0
        for i, line in enumerate(lines):
            if line.startswith("swap:"):
                content_start = i
                break

        final_content = yaml_header + "\n".join(lines[content_start:])

        Path(output_file).write_text(final_content, encoding="utf-8")

        print(f"✅ Created template file: {output_file}")
        print(f"📁 Working directory: {current_dir}")
        print("\nNext steps:")
        print(f"1. Edit {output_file} to customize your simulation parameters")
        print(f"2. Validate config: mmpp swap validate {output_file}")
        print(f"3. Show info: mmpp swap info {output_file}")
        print(f"4. Run simulations: mmpp swap run {output_file}")
        print("\nSee the generated file for detailed configuration options.")

    except ImportError:
        print("❌ Error: Template parser not available")
        sys.exit(1)
    except OSError as e:
        print(f"❌ Error creating template file: {e}")
        sys.exit(1)


def run_simulations_from_config(config_file: str, dry_run: bool = False) -> None:
    """Run simulations from a configuration file."""
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        if config_file == "parms.yml":
            print("💡 Hint: Run 'mmpp swap init' first to generate a parms.yml file")
        sys.exit(1)

    try:
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
            print(
                f"📋 Dry run - would execute {info['total_combinations']} simulations"
            )
            print(f"📁 Config file: {config_file}")
            print(f"🔧 Parameters: {', '.join(info['parameters'])}")
            print(f"📊 Parameter counts: {info['parameter_counts']}")
            print("✨ Use without --dry-run to execute")
        else:
            print(f"🚀 Starting simulation execution from: {config_file}")
            swapper.run_simulations()
            print("✅ Simulation execution completed")

    except ImportError:
        print(
            "❌ Error: Required dependencies not available. Install with: pip install mmpp[dev]"
        )
        sys.exit(1)
    except (OSError, ValueError) as e:
        print(f"❌ Error running simulations: {e}")
        sys.exit(1)


def show_config_info(config_file: str) -> None:
    """Show detailed information about a configuration file."""
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        if config_file == "parms.yml":
            print("💡 Hint: Run 'mmpp swap init' first to generate a parms.yml file")
        sys.exit(1)

    try:
        swapper = SimulationSwapper(config_file)
        info = swapper.get_info()

        print(f"📋 Configuration Info: {config_file}")
        print("=" * 50)
        print(f"📊 Total combinations: {info['total_combinations']}")
        print(f"🔧 Parameters: {len(info['parameters'])}")

        print("\n📈 Parameter Details:")
        for param, count in info["parameter_counts"].items():
            print(f"  • {param}: {count} values")

        print("\n⚙️  Configuration Options:")
        config_opts = info["config_options"]
        print(f"  • Prefix: {config_opts['prefix']}")
        print(f"  • Last parameter: {config_opts['last_param_name']}")
        print(f"  • SBATCH: {config_opts['sbatch']}")
        print(f"  • Pairs mode: {config_opts['pairs']}")
        print(f"  • Template: {config_opts['template']}")
        print(f"  • Main path: {config_opts['main_path']}")

        # Show validation issues
        issues = info["validation_issues"]
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
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        if config_file == "parms.yml":
            print("💡 Hint: Run 'mmpp swap init' first to generate a parms.yml file")
        sys.exit(1)

    try:
        swapper = SimulationSwapper(config_file)
        issues = swapper.validate_config()
        info = swapper.get_info()

        print(f"🔍 Validating: {config_file}")
        print("=" * 50)

        # Show detailed validation results with counts
        parameters = info["parameters"]
        total_combinations = info["total_combinations"]
        config_opts = info["config_options"]

        print(f"📊 Found {len(parameters)} parameters: {', '.join(parameters)}")
        print(f"🚀 Will create {total_combinations} simulations")
        print(
            f"📁 Target directory: {config_opts.get('main_path', 'current directory')}"
        )

        # Show parameter details
        if parameters:
            print("\n📈 Parameter Details:")
            for param, count in info["parameter_counts"].items():
                print(f"  • {param}: {count} values")

        if not issues:
            print("\n✅ Configuration is valid!")
            return

        errors = [issue for issue in issues if issue.startswith("ERROR")]
        warnings = [issue for issue in issues if issue.startswith("WARNING")]

        if errors:
            print("\n❌ Errors found:")
            for error in errors:
                print(f"  {error}")

        if warnings:
            print("\n⚠️  Warnings:")
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


__all__ = [
    "SimulationManager",
    "SimulationSwapper",
    "TemplateParser",
    "handle_swap_command",
]
