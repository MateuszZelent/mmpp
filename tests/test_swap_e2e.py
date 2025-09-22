#!/usr/bin/env python3
"""
Test E2E dla funkcjonalności MMPP Swap.

Ten test weryfikuje cały proces:
1. Tworzenie template.mx3
2. Inicjalizacja parms.yml
3. Walidacja konfiguracji
4. Dry run symulacji
"""

import os
import shutil
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path


class TestSwapE2E(unittest.TestCase):
    """Test End-to-End dla modułu swap."""

    def setUp(self):
        """Przygotuj środowisko testowe."""
        # Stwórz tymczasowy katalog
        self.test_dir = tempfile.mkdtemp(prefix="mmpp_swap_test_")
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Podstawowe parametry testowe
        self.template_file = "template.mx3"
        self.config_file = "parms.yml"
        
        print(f"Test directory: {self.test_dir}")

    def tearDown(self):
        """Posprzątaj po teście."""
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_test_template(self):
        """Stwórz template.mx3 do testów."""
        template_content = """// MMPP Test Template
// This is a test micromagnetic simulation template

// Simulation parameters
SetGridSize(128, 64, 1)
SetCellSize({cellsize}, {cellsize}, 3e-9)

// Material parameters
Msat = {msat}     // Saturation magnetization (A/m)
Aex = {aex}       // Exchange constant (J/m)
alpha = 0.01      // Damping parameter

// External field
B_ext = vector(0, 0, {bfield})  // External field (T)

// Initial magnetization
m = uniform(1, 0, 0)

// Save data
SaveAs(m, "m_final")
SaveAs(B_demag, "B_demag")

// Relax the system
relax()

print("Simulation completed successfully")
"""
        
        with open(self.template_file, 'w') as f:
            f.write(template_content)
        
        print(f"Created template file: {self.template_file}")

    def run_mmpp_command(self, args):
        """Uruchom komendę mmpp i zwróć wynik."""
        cmd = [sys.executable, "-m", "mmpp"] + args
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=self.test_dir
            )
            return result
        except subprocess.TimeoutExpired:
            self.fail(f"Command timed out: {' '.join(cmd)}")
        except Exception as e:
            self.fail(f"Error running command {' '.join(cmd)}: {e}")

    def test_swap_init(self):
        """Test inicjalizacji parms.yml."""
        print("\\n=== Test: swap init ===")
        
        # Stwórz template
        self.create_test_template()
        
        # Uruchom mmpp swap init
        result = self.run_mmpp_command(["swap", "init", self.template_file])
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\\n{result.stderr}")
        
        # Sprawdź czy komenda się udała
        self.assertEqual(result.returncode, 0, 
                        f"swap init failed: {result.stderr}")
        
        # Sprawdź czy plik parms.yml został utworzony
        self.assertTrue(os.path.exists(self.config_file),
                       "parms.yml file was not created")
        
        # Sprawdź zawartość pliku
        with open(self.config_file, 'r') as f:
            content = f.read()
            
        # Weryfikuj że zawiera oczekiwane parametry
        self.assertIn("cellsize", content, "cellsize parameter not found")
        self.assertIn("msat", content, "msat parameter not found") 
        self.assertIn("aex", content, "aex parameter not found")
        self.assertIn("bfield", content, "bfield parameter not found")
        self.assertIn("template.mx3", content, "template name not found")
        
        print("✅ swap init test passed")

    def test_swap_validate(self):
        """Test walidacji konfiguracji."""
        print("\\n=== Test: swap validate ===")
        
        # Przygotuj środowisko
        self.create_test_template()
        self.run_mmpp_command(["swap", "init", self.template_file])
        
        # Uruchom walidację
        result = self.run_mmpp_command(["swap", "validate", self.config_file])
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\\n{result.stderr}")
        
        # Sprawdź czy walidacja się udała
        self.assertEqual(result.returncode, 0,
                        f"swap validate failed: {result.stderr}")
        
        # Sprawdź czy output zawiera informacje o walidacji
        self.assertIn("parameters", result.stdout.lower(),
                     "Validation output missing parameter info")
        
        print("✅ swap validate test passed")

    def test_swap_info(self):
        """Test wyświetlania informacji o konfiguracji."""
        print("\\n=== Test: swap info ===")
        
        # Przygotuj środowisko
        self.create_test_template()
        self.run_mmpp_command(["swap", "init", self.template_file])
        
        # Uruchom info
        result = self.run_mmpp_command(["swap", "info", self.config_file])
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\\n{result.stderr}")
        
        # Sprawdź czy komenda się udała
        self.assertEqual(result.returncode, 0,
                        f"swap info failed: {result.stderr}")
        
        # Sprawdź czy output zawiera oczekiwane informacje
        output = result.stdout.lower()
        self.assertIn("configuration info", output,
                     "Missing configuration info header")
        self.assertIn("parameters", output,
                     "Missing parameters information")
        self.assertIn("combinations", output,
                     "Missing combinations information")
        
        print("✅ swap info test passed")

    def test_swap_dry_run(self):
        """Test dry run symulacji."""
        print("\\n=== Test: swap run --dry-run ===")
        
        # Przygotuj środowisko
        self.create_test_template()
        self.run_mmpp_command(["swap", "init", self.template_file])
        
        # Uruchom dry run
        result = self.run_mmpp_command(["swap", "run", self.config_file, "--dry-run"])
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\\n{result.stderr}")
        
        # Sprawdź czy dry run się udał
        self.assertEqual(result.returncode, 0,
                        f"swap run --dry-run failed: {result.stderr}")
        
        # Sprawdź czy output zawiera informacje o dry run
        output = result.stdout.lower()
        self.assertIn("dry run", output,
                     "Missing dry run information")
        self.assertIn("simulations", output,
                     "Missing simulations count")
        
        print("✅ swap run --dry-run test passed")

    def test_swap_workflow_complete(self):
        """Test kompletnego workflow swap."""
        print("\\n=== Test: Complete Swap Workflow ===")
        
        # 1. Stwórz template
        self.create_test_template()
        print("✓ Created template file")
        
        # 2. Inicjalizuj konfigurację
        result = self.run_mmpp_command(["swap", "init", self.template_file])
        self.assertEqual(result.returncode, 0, "Init failed")
        print("✓ Initialized configuration")
        
        # 3. Waliduj konfigurację
        result = self.run_mmpp_command(["swap", "validate", self.config_file])
        self.assertEqual(result.returncode, 0, "Validate failed")
        print("✓ Validated configuration")
        
        # 4. Pokaż info
        result = self.run_mmpp_command(["swap", "info", self.config_file])
        self.assertEqual(result.returncode, 0, "Info failed")
        print("✓ Showed configuration info")
        
        # 5. Uruchom dry run
        result = self.run_mmpp_command(["swap", "run", self.config_file, "--dry-run"])
        self.assertEqual(result.returncode, 0, "Dry run failed")
        print("✓ Executed dry run")
        
        print("✅ Complete workflow test passed")

    def test_error_handling(self):
        """Test obsługi błędów."""
        print("\\n=== Test: Error Handling ===")
        
        # Test 1: Brak template file
        result = self.run_mmpp_command(["swap", "init", "nonexistent.mx3"])
        self.assertNotEqual(result.returncode, 0, "Should fail for missing template")
        print("✓ Handles missing template file")
        
        # Test 2: Brak config file
        result = self.run_mmpp_command(["swap", "validate", "nonexistent.yml"])
        self.assertNotEqual(result.returncode, 0, "Should fail for missing config")
        print("✓ Handles missing config file")
        
        # Test 3: Nieprawidłowy plik konfiguracyjny
        with open("invalid.yml", "w") as f:
            f.write("invalid: yaml: content: [")
        
        result = self.run_mmpp_command(["swap", "validate", "invalid.yml"])
        self.assertNotEqual(result.returncode, 0, "Should fail for invalid YAML")
        print("✓ Handles invalid YAML")
        
        print("✅ Error handling test passed")


def run_tests():
    """Uruchom wszystkie testy."""
    # Sprawdź czy mmpp jest dostępne
    try:
        result = subprocess.run([sys.executable, "-m", "mmpp", "--help"], 
                               capture_output=True, timeout=10)
        if result.returncode != 0:
            print("❌ MMPP not available or not working")
            return False
    except Exception as e:
        print(f"❌ Error checking MMPP availability: {e}")
        return False
    
    print("🚀 Starting MMPP Swap E2E Tests")
    print("=" * 50)
    
    # Uruchom testy
    unittest.main(verbosity=2, exit=False)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
