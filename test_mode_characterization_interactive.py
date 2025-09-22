#!/usr/bin/env python3
"""
Test script for mode characterization in interactive viewer
"""

import sys
import os
sys.path.insert(0, '/home/kkingstoun/git/mmpp')

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend

def create_test_characterization():
    """Create test with mode characterization functionality"""
    print("=== Test Charakteryzacji Modów w Interaktywnej Przeglądarce ===")
    
    try:
        from mmpp.fft.modes import FMRModeAnalyzer
        
        # Find a zarr file to use
        zarr_path = None
        test_paths = [
            '/home/kkingstoun/git/mmpp/mmpp/cli/run/test/test.zarr',
            '/home/kkingstoun/git/mmpp/mmpp/cli/run/test/test_batch/submission_method_mmpp/aex_1e-11/msat_8e+05.zarr'
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                zarr_path = path
                break
                
        if not zarr_path:
            print("Nie znaleziono pliku zarr. Szukam dostępnych...")
            for root, dirs, files in os.walk('/home/kkingstoun/git/mmpp'):
                for file in files:
                    if file.endswith('.zarr'):
                        zarr_path = os.path.join(root, file)
                        break
                if zarr_path:
                    break
        
        if not zarr_path:
            print("Nie znaleziono żadnego pliku zarr!")
            return
            
        print(f"Używam pliku: {zarr_path}")
        
        # Create analyzer
        analyzer = FMRModeAnalyzer(zarr_path, debug=True)
        
        if not analyzer.modes_available:
            print("Mody FFT nie są dostępne w tym pliku!")
            return
            
        print("\n=== INSTRUKCJA UŻYTKOWANIA ===")
        print("1. Otwiera się okno z interaktywnym spektrum")
        print("2. Kliknij na spektrum aby wybrać częstotliwość")
        print("3. Naciśnij klawisz 'c' aby scharakteryzować aktualny mod")
        print("4. Naciśnij klawisz 'h' aby zobaczyć pomoc")
        print("5. Kliknij dwukrotnie na wykresach modów aby uruchomić animację")
        print("6. Zamknij okno aby zakończyć")
        print("\n=== URUCHAMIAM INTERAKTYWNĄ PRZEGLĄDARKE ===")
        
        # Show initial frequency range
        print(f"Dostępne częstotliwości: {analyzer.frequencies.min():.2f} - {analyzer.frequencies.max():.2f} GHz")
        print(f"Liczba częstotliwości: {len(analyzer.frequencies)}")
        
        # Create interactive plot
        fig = analyzer.interactive_spectrum(show=True)
        
        print("\nTest zakończony.")
        
    except Exception as e:
        print(f"Błąd podczas testu: {e}")
        import traceback
        traceback.print_exc()

def create_demo_characterization():
    """Create demo showing characterization results"""
    try:
        from mmpp.fft.modes import FMRModeAnalyzer
        from mmpp.fft.mode_characterization import ModeCharacterAnalyzer, ModeCharacterizationResult
        
        # Find zarr file
        zarr_path = '/home/kkingstoun/git/mmpp/mmpp/cli/run/test/test.zarr'
        if not os.path.exists(zarr_path):
            print("Demo wymaga pliku test.zarr - uruchom najpierw symulację!")
            return
            
        print("=== Demo Automatycznej Charakteryzacji ===")
        analyzer = FMRModeAnalyzer(zarr_path)
        
        if not analyzer.modes_available:
            print("Brak modów w pliku!")
            return
            
        # Test characterization on a few frequencies
        freqs_to_test = analyzer.frequencies[::len(analyzer.frequencies)//5][:3]  # 3 test frequencies
        
        for freq in freqs_to_test:
            print(f"\n--- Częstotliwość: {freq:.3f} GHz ---")
            
            try:
                char_result = analyzer.characterize_mode(freq)
                
                print(f"Klasa główna: {char_result.primary_class}")
                print(f"Indeks m: {char_result.m_index}")
                print(f"Rotacja: {char_result.rotation_sense or 'N/A'}")
                print(f"Węzły radialne: {char_result.radial_nodes}")
                print(f"Pewność: {char_result.confidence:.2f}")
                print(f"Etykiety: {', '.join(char_result.labels)}")
                
                if char_result.notes:
                    print(f"Uwagi: {', '.join(char_result.notes)}")
                    
            except Exception as e:
                print(f"Błąd charakteryzacji: {e}")
                
    except Exception as e:
        print(f"Błąd demo: {e}")

def main():
    """Main test function"""
    print("Wybierz test:")
    print("1. Interaktywna przeglądarka z charakteryzacją")
    print("2. Demo automatycznej charakteryzacji")
    
    choice = input("Wybierz (1 lub 2): ").strip()
    
    if choice == "1":
        create_test_characterization()
    elif choice == "2":
        create_demo_characterization()
    else:
        print("Uruchamiam demo automatycznej charakteryzacji...")
        create_demo_characterization()

if __name__ == "__main__":
    main()