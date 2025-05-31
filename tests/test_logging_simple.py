#!/usr/bin/env python3
"""
Simple test to verify logging configuration without loading data files.
"""

import sys
import os

# Add the mmpp module to the path
sys.path.insert(
    0,
    "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp",
)

print("Testing logging configuration...")

# Test logging setup directly
from mmpp.logging_config import setup_mmpp_logging, get_mmpp_logger

print("\n=== Testing direct logging setup ===")

# Test normal mode
print("Setting up normal logging (debug=False)...")
setup_mmpp_logging(debug=False, logger_name="mmpp")
log = get_mmpp_logger("mmpp")
log.info("This is an INFO message")
log.debug("This DEBUG message should NOT appear")
log.warning("This is a WARNING message")

print("\nSetting up debug logging (debug=True)...")
setup_mmpp_logging(debug=True, logger_name="mmpp")
log_debug = get_mmpp_logger("mmpp")
log_debug.info("This is an INFO message in debug mode")
log_debug.debug("This DEBUG message SHOULD appear")
log_debug.warning("This is a WARNING message in debug mode")

# Test sublogger
print("\nTesting sublogger...")
sublog = get_mmpp_logger("mmpp.fft")
sublog.info("This is an INFO from mmpp.fft")
sublog.debug("This is a DEBUG from mmpp.fft")

print("\nLogging test completed!")
