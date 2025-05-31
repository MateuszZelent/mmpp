#!/usr/bin/env python3
"""
Test script for MMPP logging system.
"""

import sys
import os

# Add the mmpp module to the path
sys.path.insert(
    0,
    "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp",
)

try:
    from mmpp.logging_config import setup_mmpp_logging, get_mmpp_logger

    # Test basic logging setup
    print("Testing MMPP logging system...")

    # Test main logger
    main_log = get_mmpp_logger("mmpp")
    setup_mmpp_logging(debug=False, logger_name="mmpp")
    main_log.info("This is an INFO message from mmpp.main")
    main_log.debug("This DEBUG message should NOT appear (debug=False)")

    # Test FFT logger
    fft_log = get_mmpp_logger("mmpp.fft")
    setup_mmpp_logging(debug=True, logger_name="mmpp.fft")
    fft_log.info("This is an INFO message from mmpp.fft")
    fft_log.debug("This DEBUG message SHOULD appear (debug=True)")

    # Test plot logger
    plot_log = get_mmpp_logger("mmpp.plot")
    setup_mmpp_logging(debug=False, logger_name="mmpp.plot")
    plot_log.info("This is an INFO message from mmpp.plot")
    plot_log.warning("This is a WARNING message from mmpp.plot")
    plot_log.error("This is an ERROR message from mmpp.plot")

    print("Logging test completed successfully!")

except Exception as e:
    print(f"Error testing logging: {e}")
    import traceback

    traceback.print_exc()
