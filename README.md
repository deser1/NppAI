# NppAI - Autorski Model i Wtyczka AI dla Notepad++

NppAI to unikalny projekt łączący w sobie wtyczkę do popularnego edytora kodu Notepad++ oraz całkowicie autorski silnik sieci neuronowej typu Transformer, napisany od zera w C++. Głównym celem NppAI nie jest integrowanie zewnętrznych płatnych usług (jak OpenAI), ale stworzenie niezależnego, uczącego się na Twoim własnym kodzie, prywatnego asystenta programistycznego.

## 🚀 Główne Cechy Projektu

*   **Własny Silnik Inferencji C++ (Zero zależności):** Wtyczka nie używa gotowych rozwiązań jak `llama.cpp` czy `Ollama`. Cała matematyka Transformera (Self-Attention, RMSNorm, SwiGLU) została napisana w czystym C++ i zintegrowana wewnątrz wtyczki. Kod jest lekki, natywny i wykonuje się niezwykle szybko.
*   **Własny Format Wag (`.nppai`):** Model używa dedykowanego binarnego formatu plików, zoptymalizowanego do błyskawicznego wczytywania przez wtyczkę.
*   **Prywatność i Globalne Uczenie:** Wtyczka posiada mechanizm `TelemetryManager`, który śledzi (Diff Tracker) jak użytkownik modyfikuje wygenerowany przez AI kod. Przed jakąkolwiek wysyłką do bazy, wtyczka używa wyrażeń regularnych (Regex) do wycinania wrażliwych danych (klucze API, IP, lokalne ścieżki).
*   **Niestandardowy Trening (PyTorch):** Dołączony skrypt `train_nppai.py` pozwala przeszukać dysk z Twoimi prywatnymi projektami (C++, Python, JS, PHP i inne), by wyekstrahować kod i wygenerować plik uczący. Następnie możesz przetrenować własny model – lokalnie lub w chmurze – tworząc asystenta, który idealnie naśladuje Twój styl programowania.

## 🛠️ Architektura Projektu

Projekt składa się z dwóch głównych środowisk:

1.  **Środowisko Klienckie (Wtyczka C++)**
    *   **NppAIEngine:** Własny silnik matematyczny AI.
    *   **AIManager:** Most łączący silnik z API Scintilli.
    *   **TelemetryManager:** Moduł odpowiedzialny za anonimizację i zbieranie poprawek uczących (Continuous Learning / RLHF).
    *   **PluginDefinition:** Zintegrowany rdzeń wtyczki DLL ładujący się w Notepad++.

2.  **Środowisko Trenowania (Skrypt Python)**
    *   **train_nppai.py:** Skrypt oparty na PyTorch. Potrafi przeszukać podaną strukturę folderów (omijając `node_modules`, `build` itp.), zapisać czysty kod do jednego dużego datasetu (`dataset_export.txt`), a następnie zbudować, wytrenować i wyeksportować wagi modelu do pliku `.nppai`.

## ⚙️ Kompilacja i Instalacja Wtyczki

Projekt używa CMake do łatwej kompilacji pod system Windows. Wtyczka wymaga architektury x64.

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Po skompilowaniu skopiuj plik `NppAI.dll` do folderu wtyczek Notepad++ (zazwyczaj `C:\Program Files\Notepad++\plugins\NppAI\NppAI.dll`).

## 🧠 Jak Wytrenować Twój Model?

1.  Upewnij się, że masz zainstalowane środowisko Python wraz z biblioteką PyTorch (`pip install torch`).
2.  Dostosuj ścieżkę do swoich projektów w pliku `train_nppai.py` w funkcji `load_dataset(folder_path="D:/Projekty/")`.
3.  Uruchom skrypt:
    ```bash
    python train_nppai.py
    ```
4.  Skrypt znajdzie Twoje pliki źródłowe, wygeneruje zagregowany plik `dataset_export.txt` (gotowy do wyniesienia w chmurę dla mocniejszych GPU) oraz od razu rozpocznie symulacyjny trening na Twoim komputerze.
5.  Gotowy model wyląduje w folderze `models/NppAI-model-v1.nppai`. Skopiuj ten plik w miejsce widoczne dla Twojej wtyczki!

## 🔮 Plany na przyszłość

*   Dodanie wsparcia dla asynchronicznego wysyłania pakietów telemetrycznych via WinHTTP do zewnętrznego serwera.
*   Rozbudowa silnika C++ o obsługę instrukcji SIMD/AVX2 dla przyspieszenia inferencji na CPU.
*   Implementacja zaawansowanego Tokenizera (np. BPE zamiast znakowego) dla zwiększenia wydajności pamięci.

---
*Stworzono z pasją do C++ i pełnej kontroli nad AI.*
