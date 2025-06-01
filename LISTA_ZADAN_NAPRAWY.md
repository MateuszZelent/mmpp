# LISTA ZADAŃ DO POPRAWY REPOZYTORIUM MMPP

## 🚨 KRYTYCZNE BŁĘDY PROGRAMISTYCZNE

### 1. Błędy Type Hints i Type Safety
**Lokalizacja:** `mmpp/core.py`
- **Linia 6 i 12:** Duplikowane importy `from pathlib import Path`
- **Linia 11 i 58:** Duplikowane importy Console z rich
- **Linia 10 i 60:** Duplikowane importy Syntax z rich
- **Status:** 🔴 KRYTYCZNE - powoduje niepotrzebne importy

### 2. Za Ogólne Exception Handling
**Lokalizacje:** Znaleziono 43+ wystąpień w całym kodzie
- `mmpp/core.py`: 12 wystąpień `except Exception as e:`
- `mmpp/batch_operations.py`: 4 wystąpienia  
- `mmpp/fft/main.py`: 4 wystąpienia
- `mmpp/fft/modes.py`: 15+ wystąpień
- `mmpp/fft/core.py`: 2 wystąpienia
- **Status:** 🔴 KRYTYCZNE - utrudnia debugging

### 3. Debug Markery w Kodzie Produkcyjnym
**Lokalizacje:**
- `mmpp/core.py` linia 158: `log.debug("Debug marker reached")`
- **Status:** 🟡 ŚREDNIE - do usunięcia przed release

### 4. Nieużywane Importy
**Lokalizacje:**
- `tests/optimized_colorbar.py`: `import warnings` (prawdopodobnie nieużywane)
- `tests/debug_attributes.py`: `import os` (może być nieużywane)
- **Status:** 🟡 ŚREDNIE - cleanup kodu

## 🧹 PROBLEMY ORGANIZACYJNE

### 5. Pliki Rozwojowe w Repozytorium Produkcyjnym
**Lokalizacje:**
- `tests/debug_legend.py` - plik rozwojowy
- `tests/debug_attributes.py` - plik rozwojowy
- `tests/optimized_colorbar.py` - może być developmentalny
- **Status:** 🟡 ŚREDNIE - należy przenieść lub usunąć

### 6. Nieaktualne Hardcoded References w Dokumentacji
**Lokalizacje znalezione:**
- `mmpp/fft/core.py`: Wiele przykładów z `"m_z11"` zamiast auto-selection
- `docs/analysis/`: Dokumenty zawierają stare API
- `tests/`: Testy używają hardcoded dataset names
- **Status:** 🟠 WYSOKIE - wprowadza użytkowników w błąd

### 7. Duplikowane Przykłady w Kodzie
**Lokalizacje:**
- `mmpp/fft/core.py` linie 474, 486, 590, 650: Przykłady z hardcoded `'m_z11'`
- **Status:** 🟡 ŚREDNIE - należy zaktualizować

## 🐛 LOGICZNE BŁĘDY I PROBLEMY

### 8. Potencjalne Problemy z Type Safety
**Wymagają szczegółowej analizy:**
- Brak sprawdzenia czy znalezione problemy są rzeczywiste
- Potrzebna inspekcja manual typowania
- **Status:** ⚪ DO ZBADANIA

### 9. Hardcoded Paths w Testach
**Lokalizacje:**
- Różne pliki testowe mogą zawierać ścieżki specyficzne dla systemu
- **Status:** 🟡 ŚREDNIE - może powodować problemy na różnych systemach

## 📚 PROBLEMY DOKUMENTACYJNE

### 10. Nieaktualna Dokumentacja API
**Lokalizacje:**
- `docs/analysis/KOMPLETNA_ANALIZA_FFT_API.md`: Używa starych defaultów
- `docs/analysis/FFT_API_ANALIZA_SZCZEGOLOWA.md`: Używa starych defaultów  
- `docs/index.md`: Przykłady z hardcoded wartościami
- **Status:** 🔴 KRYTYCZNE - wprowadza użytkowników w błąd

### 11. Brakująca Dokumentacja Nowych Funkcji
**Co brakuje:**
- Dokumentacja auto-selection functionality
- Przykłady użycia nowego API
- Migration guide dla użytkowników
- **Status:** 🟠 WYSOKIE - użytkownicy nie wiedzą o nowych możliwościach

## 🔧 PROBLEMY TECHNICZNE

### 12. Potencjalne Dependency Issues
**Do sprawdzenia:**
- Weryfikacja czy wszystkie dependencies są aktualne
- Sprawdzenie kompatybilności wersji
- **Status:** ⚪ DO ZBADANIA

### 13. Test Organization
**Problemy:**
- Pliki debug w folderze tests/
- Brak clear test structure dla nowej funkcjonalności
- **Status:** 🟡 ŚREDNIE

## 🎯 PRIORYTET NAPRAW

### KRYTYCZNE (Natychmiast):
1. ✅ **Fix duplikowanych importów w core.py**
2. ✅ **Update dokumentacji API z hardcoded examples**
3. ✅ **Replace general Exception catching z specific exceptions**

### WYSOKIE (Ten sprint):
4. ✅ **Remove debug markers z production code**
5. ✅ **Clean up development files z main repository**
6. ✅ **Update wszystkich przykładów w kodzie do auto-selection**

### ŚREDNIE (Następny sprint):
7. **Fix nieużywane importy**
8. **Improve test organization**
9. **Add comprehensive tests dla auto-selection**

### NISKIE (Future):
10. **Performance monitoring**
11. **Advanced type safety improvements**

## 📋 SZCZEGÓŁOWY PLAN DZIAŁANIA

### Faza 1: Critical Fixes (Dzisiaj)
```bash
# 1. Fix duplikowanych importów
# 2. Remove debug markers  
# 3. Update core documentation
```

### Faza 2: Documentation Update (Jutro)
```bash
# 1. Update API documentation
# 2. Fix hardcoded examples
# 3. Add migration guide
```

### Faza 3: Code Cleanup (Następny tydzień)
```bash
# 1. Improve exception handling
# 2. Clean up test files
# 3. Remove unused imports
```

### Faza 4: Testing & Validation (Continuous)
```bash
# 1. Add auto-selection tests
# 2. Performance testing
# 3. Cross-platform validation
```

## 🔍 STATYSTYKI ZNALEZIONYCH PROBLEMÓW

- **General Exception handling:** 43+ wystąpień
- **Hardcoded 'm_z11' references:** 45+ wystąpień  
- **Debug files in production:** 2 pliki
- **Duplikowane importy:** 3 pary w core.py
- **Documentation files do update:** 10+ plików

**Całkowita ocena:** Repozytorium ma solidną bazę ale wymaga cleanup'u i aktualizacji dokumentacji po wprowadzeniu auto-selection feature.
