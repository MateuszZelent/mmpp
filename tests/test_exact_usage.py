#!/usr/bin/env python3

# Test the exact usage pattern requested: 
# import mmpp as mp
# mp.open("/path/to/directory")

import sys
import os

# Add the current directory to path to import mmpp
sys.path.insert(0, '/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp')

try:
    # Test the exact import as requested
    import mmpp as mp
    
    print("SUCCESS: import mmpp as mp - WORKING!")
    
    # Check if open function is available
    if hasattr(mp, 'open'):
        print("SUCCESS: mp.open function is available!")
        
        # Test if we can call the open function with a test path (we'll use a non-existent path)
        # This should create the MMPP instance but fail gracefully when no zarr files are found
        test_path = "/tmp/test_mmpp_path"
        if not os.path.exists(test_path):
            os.makedirs(test_path, exist_ok=True)
        
        try:
            print(f"Testing mp.open('{test_path}')...")
            result = mp.open(test_path)
            print(f"SUCCESS: mp.open() returned: {type(result)}")
            print(f"MMPP instance created successfully!")
        except Exception as e:
            print(f"ERROR in mp.open(): {e}")
        finally:
            # Clean up test directory
            if os.path.exists(test_path):
                os.rmdir(test_path)
                
    else:
        print("ERROR: mp.open function not found!")
        print(f"Available attributes: {[attr for attr in dir(mp) if not attr.startswith('_')]}")

except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {e}")

print("\nIf everything worked, you can now use:")
print("import mmpp as mp")
print("mp.open('/mnt/local/kkingstoun/admin/pcss_storage/mannga/bowtie/mateusz/sinc/solver_test/v1')")
