#!/usr/bin/env python3
"""
Test script for dark theme logging colors in MMPP.
"""

import sys
import os

sys.path.insert(0, "/home/kkingstoun/git/mmpp")


def test_dark_theme_logging():
    """Test dark theme logging with improved colors."""
    print("Testing MMPP dark theme logging colors...")
    print("=" * 60)

    try:
        # Import the updated logging configuration
        from mmpp.logging_config import (
            setup_mmpp_logging,
            get_mmpp_logger,
            configure_for_dark_theme,
            configure_for_light_theme,
            reset_logging_config,
        )

        print("✅ Logging imports successful")

        # Test dark theme configuration (default)
        print("\n🌙 Testing DARK THEME logging:")
        print("-" * 40)

        # Configure for dark theme
        configure_for_dark_theme()

        # Create loggers for different modules
        main_log = get_mmpp_logger("mmpp")
        fft_log = get_mmpp_logger("mmpp.fft")
        plot_log = get_mmpp_logger("mmpp.plotting")
        sim_log = get_mmpp_logger("mmpp.simulation")

        # Test different log levels with dark theme colors
        print("Testing all log levels with dark theme colors:")
        main_log.debug("🐛 This is a DEBUG message (bright_black/gray)")
        main_log.info("ℹ️  This is an INFO message (bright_cyan)")
        main_log.warning("⚠️  This is a WARNING message (bright_yellow)")
        main_log.error("❌ This is an ERROR message (bright_red)")
        main_log.critical("🚨 This is a CRITICAL message (bold bright_red)")

        print("\nTesting different modules:")
        fft_log.info("🔬 FFT module message (should be bright_cyan)")
        plot_log.warning("📊 Plotting module warning (should be bright_yellow)")
        sim_log.error("⚡ Simulation module error (should be bright_red)")

        # Test debug mode
        print("\n🔍 Testing DEBUG mode:")
        setup_mmpp_logging(debug=True, logger_name="mmpp", use_dark_theme=True)
        debug_log = get_mmpp_logger("mmpp.debug_test")
        debug_log.debug("🐛 This DEBUG message should now be visible")
        debug_log.info("ℹ️  Info in debug mode")

        print("\n" + "-" * 40)
        print("🌞 Testing LIGHT THEME logging:")
        print("-" * 40)

        # Configure for light theme
        configure_for_light_theme()
        light_log = get_mmpp_logger("mmpp.light")
        light_log.info("☀️  This is light theme INFO (standard colors)")
        light_log.warning("⚠️  This is light theme WARNING")
        light_log.error("❌ This is light theme ERROR")

        print("\n✅ All logging color tests completed!")
        return True

    except Exception as e:
        print(f"❌ Error testing logging colors: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mmpp_integration():
    """Test logging integration with MMPP classes."""
    print("\n🔗 Testing MMPP integration with dark theme logging:")
    print("-" * 50)

    try:
        # Test simulation module logging
        from mmpp.simulation import log as sim_log

        sim_log.info("🎯 Simulation module with dark theme colors")
        sim_log.warning("⚠️  Simulation warning message")

        # Test plotting module logging
        from mmpp.plotting import log as plot_log

        plot_log.info("📊 Plotting module with dark theme colors")
        plot_log.debug("🐛 Plotting debug message")

        # Test FFT module logging
        try:
            from mmpp.fft.modes import log as modes_log

            modes_log.info("🌊 FFT modes module with dark theme colors")
        except Exception:
            print("ℹ️  FFT modes module not available for testing")

        print("✅ MMPP integration test successful!")
        return True

    except Exception as e:
        print(f"❌ Error testing MMPP integration: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🎨 Testing MMPP Dark Theme Logging Colors")
    print("=" * 70)
    print("This test shows improved colors for dark terminal backgrounds")
    print("Colors used:")
    print("  • DEBUG: bright_black (gray)")
    print("  • INFO: bright_cyan")
    print("  • WARNING: bright_yellow")
    print("  • ERROR: bright_red")
    print("  • CRITICAL: bold bright_red")
    print("  • Time: bright_white")
    print("  • Module name: bright_green")
    print("=" * 70)

    success = True

    # Test dark theme logging
    success &= test_dark_theme_logging()

    # Test MMPP integration
    success &= test_mmpp_integration()

    if success:
        print("\n🎉 All dark theme logging tests passed!")
        print("💡 Tip: Use configure_for_light_theme() if you need light theme colors")
    else:
        print("\n❌ Some tests failed. Check output above.")
        sys.exit(1)
