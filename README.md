# NppAI - Autorski Model i Wtyczka AI dla Notepad++

NppAI to unikalny projekt łączący w sobie wtyczkę do popularnego edytora kodu Notepad++ oraz całkowicie autorski silnik sieci neuronowej typu Transformer, napisany od zera w C++. Głównym celem NppAI nie jest integrowanie zewnętrznych płatnych usług (jak OpenAI), ale stworzenie niezależnego, uczącego się na Twoim własnym kodzie, prywatnego asystenta programistycznego.

## 🚀 Główne Cechy Projektu

- **Własny Silnik Inferencji C++ (Zero zależności):** Wtyczka nie używa gotowych rozwiązań jak `llama.cpp` czy `Ollama`. Cała matematyka Transformera (Self-Attention, RMSNorm, SwiGLU) została napisana w czystym C++ i zintegrowana wewnątrz wtyczki. Kod jest lekki, natywny i wykonuje się niezwykle szybko.
- **Własny Format Wag (`.nppai`):** Model używa dedykowanego binarnego formatu plików, zoptymalizowanego do błyskawicznego wczytywania przez wtyczkę.
- **Architektura RAG (Lokalna pamięć wektorowa):** Silnik potrafi indeksować otwarte pliki w wektorowej bazie lokalnej (tworząc pliki `.rag_db` i `.nppai_mem`), dzięki czemu podczas pisania kodu AI automatycznie czerpie kontekst z innych części Twojego projektu.
- **Prywatność i Globalne Uczenie (Federated Learning):** Wtyczka posiada mechanizm `TelemetryManager`, który śledzi (Diff Tracker) jak użytkownik modyfikuje wygenerowany przez AI kod. Przed wysyłką do bazy, wtyczka używa wyrażeń regularnych (Regex) do wycinania wrażliwych danych. Serwer odbiera pakiety, dopisuje je do centralnego zbioru i uruchamia proces dotrenowania modelu w tle.
- **Niestandardowy Trening (PyTorch):** Dołączony skrypt `train_nppai.py` ładuje bazę `instruct_dataset.txt`. Następnie możesz przetrenować własny model – lokalnie lub w chmurze – tworząc asystenta, który idealnie naśladuje Twój styl programowania.
- **Lokalizacja (I18N):** Wtyczka automatycznie reaguje na zmianę języka systemowego w Notepad++ (komunikat `NPPN_NATIVELANGCHANGED`) i w locie dynamicznie tłumaczy cały interfejs z języka angielskiego na polski.
- **Kwantyzacja i Optymalizacja (AVX2/INT8):** Wagi modelu mogą być kwantyzowane w locie do formatu INT8, drastycznie zmniejszając zużycie pamięci RAM. Wektorowe operacje matematyczne wykorzystują instrukcje SIMD (AVX2) dla procesorów x64, a całość może być poprawnie kompilowana dla urządzeń ARM64 (np. Snapdragon X Elite).
- **Asynchroniczny RAG:** Silnik i baza wektorowa operują w niezależnych wątkach, nie powodując przycięć (zamrażania) interfejsu graficznego edytora.

## 🛠️ Architektura Projektu

Projekt składa się z dwóch głównych środowisk:

1.  **Środowisko Klienckie (Wtyczka C++)**
    - **NppAIEngine:** Własny silnik matematyczny AI z obsługą tagu `<think>`.
    - **AIManager:** Most łączący silnik z API Scintilli.
    - **RAGManager:** Zarządca bazy wektorowej pobierający lokalny kontekst plików.
    - **TelemetryManager:** Moduł odpowiedzialny za anonimizację i zbieranie poprawek uczących (Continuous Learning / RLHF).
    - **PluginDefinition:** Zintegrowany rdzeń wtyczki DLL ładujący się w Notepad++ i budujący dokowany panel UI.

2.  **Środowisko Serwerowe (Skrypty Python)**
    - **server_backend.py:** Serwer oparty na FastAPI, zbierający poprawki (RLHF) od wtyczek z całego świata do pliku `instruct_dataset.txt` i serwujący nowe wagi w formacie strumieniowym, odpornym na rwanie (chunking). Automatycznie uruchamia trening w tle.
    - **train_nppai.py:** Skrypt oparty na PyTorch. Potrafi zbudować, wytrenować i wyeksportować wagi modelu do binarnego pliku `.nppai`.

## ⚙️ Kompilacja i Instalacja Wtyczki

Projekt używa CMake do łatwej kompilacji pod system Windows. Wtyczka wymaga architektury x64. Kompilator automatycznie dba o inkrementację numeru kompilacji (Auto-Versioning) oraz o Trimming / Link-Time Optimization (LTO).

```bash
cmake -B build -A x64
cmake --build build --config Release
```

Po skompilowaniu skopiuj wygenerowany plik `build/Release/NppAI.dll` do folderu wtyczek Notepad++ (zazwyczaj `C:\Program Files\Notepad++\plugins\NppAI\NppAI.dll`).

## 🧠 Jak Uruchomić Infrastrukturę AI?

1.  Upewnij się, że masz zainstalowane środowisko Python wraz z bibliotekami FastAPI, Uvicorn, Pydantic i PyTorch:
    ```bash
    pip install fastapi uvicorn pydantic torch
    ```
2.  Uruchom główny serwer backendu w chmurze (lub lokalnie):
    ```bash
    python server_backend.py
    ```
    _Serwer uruchomi się na porcie 8000 i zacznie nasłuchiwać zapytań (telemetrii) od Twojej wtyczki NppAI. Będzie też dystrybuował nowe wagi modelu._
3.  Z poziomu Notepad++ możesz wysłać zapytanie (panel dokowany u dołu) lub sprawdzić aktualizacje w locie. Gdy serwer zgromadzi odpowiednią liczbę poprawek kodu od Ciebie, wywoła w tle skrypt `train_nppai.py`, który od nowa nauczy asystenta.

## 🔮 Plany na przyszłość

- Obsługa modeli klasy Mixture of Experts (MoE) we własnym silniku C++.
- Wdrożenie kwantyzacji K-Quants (np. Q4_K) dla jeszcze większej oszczędności VRAM/RAM.
- Integracja własnego parsera języków AST do poprawy trafności wyników RAG (rozumienie struktury klas).

---

_Stworzono z pasją do C++ i pełnej kontroli nad AI._
