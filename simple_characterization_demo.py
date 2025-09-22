#!/usr/bin/env python3
"""
Prosty przykład charakteryzacji modów - GOTOWY DO URUCHOMIENIA

Użycie:
python simple_characterization_demo.py path/to/your/file.zarr

Lub z mock danymi:
python simple_characterization_demo.py --mock
"""

import sys
import os
import argparse

# Dodaj ścieżkę do MMPP jeśli potrzeba
sys.path.insert(0, '/home/kkingstoun/git/mmpp')

def characterize_single_mode(zarr_path, frequency=15.0):
    """
    PRZYKŁAD 1: Charakteryzacja pojedynczego modu
    """
    print(f"\n=== CHARAKTERYZACJA POJEDYNCZEGO MODU ===")
    
    from mmpp.fft.modes import FMRModeAnalyzer
    
    try:
        # Załaduj analizator
        analyzer = FMRModeAnalyzer(zarr_path, debug=True)
        
        if not analyzer.modes_available:
            print("❌ Brak danych modów FFT w tym pliku!")
            return False
            
        print(f"✅ Załadowano dane z {len(analyzer.frequencies)} punktów częstotliwości")
        
        # Charakteryzuj mod przy konkretnej częstotliwości
        print(f"\n📊 Charakteryzuję mod przy {frequency:.1f} GHz...")
        result = analyzer.characterize_mode(frequency)
        
        # Wyświetl wyniki
        print(f"\n🎯 WYNIKI CHARAKTERYZACJI:")
        print(f"   • Częstotliwość: {result.frequency:.2f} GHz")
        print(f"   • Klasa główna: {result.primary_class.upper()}")
        print(f"   • Indeks m: {result.m_index}")
        print(f"   • Rotacja: {result.rotation_sense or 'Brak'}")
        print(f"   • Węzły radialne: {result.radial_nodes}")
        print(f"   • Pewność klasyfikacji: {result.confidence:.2f}")
        print(f"   • Etykiety: {', '.join(result.labels)}")
        
        if result.notes:
            print(f"   • Uwagi: {'; '.join(result.notes)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return False

def characterize_frequency_range(zarr_path, freq_min=8.0, freq_max=25.0, n_points=10):
    """
    PRZYKŁAD 2: Charakteryzacja zakresu częstotliwości
    """
    print(f"\n=== CHARAKTERYZACJA ZAKRESU CZĘSTOTLIWOŚCI ===")
    
    from mmpp.fft.modes import FMRModeAnalyzer
    import numpy as np
    
    try:
        analyzer = FMRModeAnalyzer(zarr_path)
        
        if not analyzer.modes_available:
            print("❌ Brak danych modów FFT!")
            return False
            
        # Wybierz częstotliwości do testowania
        test_freqs = np.linspace(freq_min, freq_max, n_points)
        
        print(f"📊 Testuję {n_points} częstotliwości między {freq_min:.1f} - {freq_max:.1f} GHz:")
        print(f"{'Freq [GHz]':<10} {'Klasa':<12} {'m':<4} {'Rot.':<6} {'Pewność':<8} {'Status'}")
        print("-" * 60)
        
        results = []
        for freq in test_freqs:
            try:
                result = analyzer.characterize_mode(freq)
                status = "✅" if result.confidence > 0.3 else "⚠️"
                print(f"{freq:<10.1f} {result.primary_class:<12} "
                      f"{str(result.m_index):<4} {str(result.rotation_sense or '-'):<6} "
                      f"{result.confidence:<8.2f} {status}")
                results.append(result)
            except Exception as e:
                print(f"{freq:<10.1f} {'ERROR':<12} {'?':<4} {'-':<6} {'0.00':<8} ❌")
                
        # Podsumowanie
        if results:
            classes = [r.primary_class for r in results if r.confidence > 0.3]
            print(f"\n📈 PODSUMOWANIE:")
            from collections import Counter
            class_counts = Counter(classes)
            for cls, count in class_counts.items():
                print(f"   • {cls.capitalize()}: {count} modów")
                
        return True
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return False

def interactive_characterization(zarr_path):
    """
    PRZYKŁAD 3: Interaktywna charakteryzacja z wyborem częstotliwości
    """
    print(f"\n=== INTERAKTYWNA CHARAKTERYZACJA ===")
    
    from mmpp.fft.modes import FMRModeAnalyzer
    
    try:
        analyzer = FMRModeAnalyzer(zarr_path)
        
        if not analyzer.modes_available:
            print("❌ Brak danych modów FFT!")
            return False
            
        print(f"✅ Załadowano dane: {analyzer.frequencies.min():.1f} - {analyzer.frequencies.max():.1f} GHz")
        
        while True:
            print(f"\n🎮 MENU INTERAKTYWNE:")
            print("1. Charakteryzuj konkretną częstotliwość")
            print("2. Pokaż dostępne częstotliwości") 
            print("3. Uruchom interaktywną przeglądarkę")
            print("4. Zakończ")
            
            choice = input("\nWybierz opcję (1-4): ").strip()
            
            if choice == "1":
                try:
                    freq_str = input("Podaj częstotliwość [GHz]: ").strip()
                    freq = float(freq_str)
                    
                    result = analyzer.characterize_mode(freq)
                    print(f"\n🎯 WYNIK dla {freq:.1f} GHz:")
                    print(f"   Klasa: {result.primary_class.upper()}")
                    print(f"   m-index: {result.m_index}")
                    print(f"   Rotacja: {result.rotation_sense or 'Brak'}")
                    print(f"   Pewność: {result.confidence:.2f}")
                    
                except ValueError:
                    print("❌ Nieprawidłowa częstotliwość!")
                except Exception as e:
                    print(f"❌ Błąd charakteryzacji: {e}")
                    
            elif choice == "2":
                print(f"\n📊 Dostępne częstotliwości:")
                print(f"   Zakres: {analyzer.frequencies.min():.2f} - {analyzer.frequencies.max():.2f} GHz")
                print(f"   Liczba punktów: {len(analyzer.frequencies)}")
                print(f"   Przykładowe częstotliwości:")
                sample_freqs = analyzer.frequencies[::len(analyzer.frequencies)//10]
                for i, freq in enumerate(sample_freqs[:8]):
                    print(f"     {freq:.2f} GHz", end="   ")
                    if (i+1) % 4 == 0:
                        print()
                print()
                
            elif choice == "3":
                print(f"\n🖥️ Uruchamiam interaktywną przeglądarkę...")
                print("STEROWANIE:")
                print("  • Kliknij na spektrum aby wybrać częstotliwość")
                print("  • Naciśnij 'c' aby scharakteryzować aktualny mod")
                print("  • Naciśnij 'h' aby zobaczyć pomoc")
                print("  • Kliknij dwukrotnie na wykres modu aby uruchomić animację")
                
                try:
                    analyzer.interactive_spectrum(show=True)
                except Exception as e:
                    print(f"❌ Błąd przeglądarki: {e}")
                    
            elif choice == "4":
                print("👋 Do widzenia!")
                break
                
            else:
                print("❌ Nieprawidłowy wybór!")
                
        return True
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return False

def create_mock_data_demo():
    """
    PRZYKŁAD 4: Demo z mock danymi (gdy nie masz pliku zarr)
    """
    print(f"\n=== DEMO Z MOCK DANYMI ===")
    
    try:
        import numpy as np
        from mmpp.fft.modes import FMRModeData
        from mmpp.fft.mode_characterization import ModeCharacterAnalyzer
        
        print("🔧 Tworzę przykładowe dane modu gyracyjnego...")
        
        # Utwórz siatkę 64x64
        n = 64
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
        
        # Mod gyracyjny (m=1)
        envelope = np.exp(-(r/0.5)**2)
        mx = envelope * np.exp(1j * phi) * 0.1
        my = envelope * np.exp(1j * (phi + np.pi/2)) * 0.1
        mz = envelope * np.exp(1j * phi) * 0.05
        
        # Utwórz dane modu
        mode_array = np.stack([mx, my, mz], axis=-1)
        metadata = {
            "spatial_resolution": (2e-9, 2e-9),  # 2 nm/piksel
            "core_position_px": (n//2, n//2)
        }
        
        mode = FMRModeData(
            frequency=15.0,  # 15 GHz
            mode_array=mode_array,
            metadata=metadata
        )
        
        # Charakteryzuj
        analyzer = ModeCharacterAnalyzer()
        result = analyzer.analyze(mode)
        
        print(f"🎯 WYNIKI MOCK ANALIZY:")
        print(f"   • Klasa: {result.primary_class.upper()}")
        print(f"   • m-index: {result.m_index}")
        print(f"   • Rotacja: {result.rotation_sense}")
        print(f"   • Pewność: {result.confidence:.2f}")
        print(f"   • Etykiety: {', '.join(result.labels)}")
        
        # Wizualizacja
        try:
            import matplotlib.pyplot as plt
            
            print(f"\n🖼️ Pokazuję wizualizację mock danych...")
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(np.abs(mx), origin='lower', cmap='viridis')
            axes[0].set_title('|mx|')
            axes[0].axis('off')
            
            axes[1].imshow(np.abs(my), origin='lower', cmap='viridis') 
            axes[1].set_title('|my|')
            axes[1].axis('off')
            
            axes[2].imshow(np.abs(mz), origin='lower', cmap='viridis')
            axes[2].set_title('|mz|')
            axes[2].axis('off')
            
            fig.suptitle(f'Mock mod {result.primary_class.upper()} (m={result.m_index})')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("📝 Matplotlib niedostępny - brak wizualizacji")
            
        return True
        
    except Exception as e:
        print(f"❌ Błąd mock demo: {e}")
        return False

def main():
    """Główna funkcja demo"""
    parser = argparse.ArgumentParser(description="Demo charakteryzacji modów MMPP")
    parser.add_argument("zarr_path", nargs="?", help="Ścieżka do pliku .zarr")
    parser.add_argument("--mock", action="store_true", help="Użyj mock danych")
    parser.add_argument("--frequency", type=float, default=15.0, help="Częstotliwość do testowania")
    parser.add_argument("--interactive", action="store_true", help="Tryb interaktywny")
    parser.add_argument("--range", nargs=3, type=float, metavar=('MIN', 'MAX', 'N'), 
                       help="Test zakresu częstotliwości: min max liczba_punktów")
    
    args = parser.parse_args()
    
    print("🔬 MMPP Mode Characterization Demo")
    print("=" * 40)
    
    success = False
    
    if args.mock:
        success = create_mock_data_demo()
        
    elif args.zarr_path and os.path.exists(args.zarr_path):
        print(f"📁 Używam pliku: {args.zarr_path}")
        
        if args.interactive:
            success = interactive_characterization(args.zarr_path)
        elif args.range:
            freq_min, freq_max, n_points = args.range
            success = characterize_frequency_range(args.zarr_path, freq_min, freq_max, int(n_points))
        else:
            success = characterize_single_mode(args.zarr_path, args.frequency)
            
    else:
        print("❌ Nie podano prawidłowego pliku .zarr!")
        print("\nUżycie:")
        print("  python simple_characterization_demo.py file.zarr")
        print("  python simple_characterization_demo.py --mock")
        print("  python simple_characterization_demo.py file.zarr --interactive")
        print("  python simple_characterization_demo.py file.zarr --range 8 25 10")
        return
    
    if success:
        print(f"\n✅ Demo zakończone pomyślnie!")
    else:
        print(f"\n❌ Demo zakończone z błędami!")

if __name__ == "__main__":
    main()