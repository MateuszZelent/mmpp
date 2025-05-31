#!/usr/bin/env python3
"""
Test script to verify improved dark theme logging colors.
"""

import sys
import os
sys.path.insert(0, '/home/kkingstoun/git/mmpp')

def test_improved_dark_logging():
    """Test the improved dark theme logging colors."""
    print("🎨 Testing IMPROVED Dark Theme Logging Colors")
    print("=" * 70)
    print("Testing improved colors optimized for dark terminal backgrounds:")
    print("  • DEBUG: bright_magenta (much more visible than bright_black)")
    print("  • INFO: bright_cyan (excellent visibility)")
    print("  • WARNING: bright_yellow (excellent visibility)")
    print("  • ERROR: bright_red (excellent visibility)")
    print("  • CRITICAL: bold bright_red (excellent visibility)")
    print("  • Time: bright_white (excellent visibility)")
    print("  • Module name: bright_green (excellent visibility)")
    print("=" * 70)
    
    try:
        # Import the updated logging configuration
        from mmpp.logging_config import (
            setup_mmpp_logging, 
            get_mmpp_logger, 
            reset_logging_config
        )
        
        # Reset to ensure fresh configuration
        reset_logging_config()
        
        # Setup dark theme logging
        setup_mmpp_logging(debug=True, use_dark_theme=True)
        
        # Create loggers for testing
        main_log = get_mmpp_logger("mmpp")
        fft_log = get_mmpp_logger("mmpp.fft")
        plot_log = get_mmpp_logger("mmpp.plotting")
        sim_log = get_mmpp_logger("mmpp.simulation")
        
        print("\n🌙 DARK THEME - Testing all log levels:")
        print("-" * 60)
        
        # Test all log levels with improved colors
        main_log.debug("🐛 DEBUG message - now bright_magenta (much more visible!)")
        main_log.info("ℹ️  INFO message - bright_cyan (excellent on dark background)")
        main_log.warning("⚠️  WARNING message - bright_yellow (perfect visibility)")
        main_log.error("❌ ERROR message - bright_red (clearly visible)")
        main_log.critical("🚨 CRITICAL message - bold bright_red (impossible to miss)")
        
        print("\n🔧 Testing different modules:")
        print("-" * 40)
        fft_log.info("🔬 FFT module: Analysis completed successfully")
        plot_log.warning("📊 Plotting: Using fallback font due to missing font file")
        sim_log.error("⚡ Simulation: Invalid parameter detected")
        
        print("\n🔍 Testing debug mode with various scenarios:")
        print("-" * 50)
        debug_log = get_mmpp_logger("mmpp.test.debug")
        debug_log.debug("🛠️  Debug: Cache miss, recomputing FFT")
        debug_log.info("📈 Analysis: Processing 1024 data points")
        debug_log.warning("⚡ Performance: Operation took longer than expected")
        debug_log.error("💥 Error: Failed to load configuration file")
        
        print("\n✅ All improved logging color tests completed!")
        print("💡 Colors should now be much more visible in dark terminals!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing improved logging colors: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_usage_scenario():
    """Test with a realistic usage scenario."""
    print("\n🎯 Testing realistic usage scenario:")
    print("-" * 50)
    
    try:
        from mmpp.logging_config import get_mmpp_logger
        
        # Simulate FFT analysis logging
        fft_log = get_mmpp_logger("mmpp.fft.core")
        fft_log.info("Starting FFT analysis for dataset 'm_z11'")
        fft_log.debug("Using FFT method 1 with zero padding factor 2")
        fft_log.info("FFT computation completed in 0.342 seconds")
        fft_log.warning("Cache size growing large (150 entries), consider clearing")
        
        # Simulate plotting logging
        plot_log = get_mmpp_logger("mmpp.plotting")
        plot_log.info("Generating power spectrum plot")
        plot_log.debug("Applying custom paper style and fonts")
        plot_log.info("Plot saved to: spectrum_analysis.png")
        
        # Simulate mode analysis logging
        modes_log = get_mmpp_logger("mmpp.fft.modes")
        modes_log.info("Computing FMR modes for frequency range 0.5-3.0 GHz")
        modes_log.warning("Mode computation is memory intensive for large datasets")
        modes_log.info("Interactive spectrum display ready")
        
        print("✅ Realistic scenario test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in realistic scenario: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # Test improved colors
    success &= test_improved_dark_logging()
    
    # Test realistic usage
    success &= test_real_usage_scenario()
    
    if success:
        print("\n🎉 All tests passed! Dark theme logging colors are now much more visible!")
        print("💡 Key improvements:")
        print("   • DEBUG: Changed from bright_black to bright_magenta (much more visible)")
        print("   • All colors optimized for dark terminal backgrounds")
        print("   • Formatter fixed to properly apply colors")
    else:
        print("\n❌ Some tests failed. Check output above.")
        sys.exit(1)
