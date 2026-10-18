# Notepad++ i autorski model NppAI — uruchomienie na Windows

Wtyczka używa autorskiego modelu NppAI. „Online” oznacza tutaj wywołanie
serwera NppAI; serwer może działać na tym samym komputerze. Samo zainstalowanie
DLL nie uruchamia serwera ani nie dostarcza wag modelu.

## 1. Zainstaluj poprawioną wtyczkę

Zamknij Notepad++. Pobierz paczkę `plugin_dll_x64`, `plugin_dll_x86` lub
`plugin_dll_arm64` z udanego przebiegu GitHub Actions, zgodną z architekturą
Twojego Notepad++. Wypakuj **oba pliki**, `NppAI.dll` i `sandbox_validate.py`,
do katalogu wtyczki, np. `C:\Program Files\Notepad++\plugins\NppAI`.
Nowe paczki zawierają już pomocnik walidacji obok DLL.

Panel i pole zapytania używają Unicode Win32, a zapytania do modelu są UTF-8.
Polskie znaki podczas wpisywania i dołączania zaznaczonego kodu nie powinny
przechodzić przez stronę kodową ANSI. Samo przełączenie języka interfejsu
nie oznacza nawiązania połączenia z serwerem.

## 2. Przygotuj serwer z repozytorium NppAI

Pobierz aktualny kod repozytorium. Otwórz PowerShell w jego głównym katalogu,
gdzie znajduje się `server_backend.py`. Wymagany jest Python 3.12 lub nowszy.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

Do samej inferencji nie jest potrzebny PyTorch. Serwer korzysta z własnego
runtime NppAI, bez zewnętrznego modelu ani konta u dostawcy AI.

## 3. Wskaż rzeczywisty plik modelu

Przygotuj wyeksportowany **dense v2, 256-bajtowy** model `.nppai`.
Aktualny serwer Python obsługuje ten wariant architektury. Plik `.pth`,
archiwum ZIP, eksperymentalny `.nppaicopy`, model MoE lub checkpoint z innym
słownikiem nie jest zamiennikiem tego formatu. Niezgodny plik zostanie odrzucony
przez sprawdzenie gotowości; skrypt nie podmienia go na inny model.

Zastąp ścieżkę poniżej rzeczywistą lokalizacją swojego pliku i Notepad++:

```powershell
.\.venv\Scripts\python.exe scripts\start_nppai.py `
  --model "C:\MojeModele\NppAI-model-v1.nppai" `
  --notepad "C:\Program Files\Notepad++\notepad++.exe"
```

Skrypt uruchamia usługę na `127.0.0.1:8000`, ładuje faktyczny plik NppAI,
sprawdza uwierzytelniony `/api/v1/health`, pokazuje identyfikator SHA-256 modelu
i liczbę bajtów kontekstu. Dopiero po gotowości uruchamia nową instancję
Notepad++ z adresem, zgodnym kluczem i automatycznym wyborem trybu. Używa
udokumentowanego `-multiInst`: [instrukcja Notepad++](https://npp-user-manual.org/docs/command-prompt/).

Pozostaw konsolę otwartą przez czas pracy. Ctrl+C zatrzymuje usługę. Klucz jest
losowy dla tej sesji, nie jest wypisywany ani zapisywany do pliku i nie trzeba
go przepisywać ręcznie. Globalne zmienne Windows pozostają bez zmian.
Jeżeli port 8000 jest zajęty, zatrzymaj poprzednią usługę albo dodaj `--port 8001`.
Sam przycisk generowania nie instaluje ani nie uruchamia serwera.

Opcja `--notepad` jest opcjonalna przy uruchamianiu samej usługi. Notepad++
uruchomiony wcześniej z menu Start nie odziedziczy konfiguracji nowej sesji:
do pracy z wtyczką uruchom go tym samym poleceniem z `--notepad`.

## 4. Sprawdź zapytanie o silnię

W nowym Notepad++ zapisz pusty dokument jako `silnia.py` w kodowaniu UTF-8.
Otwórz panel NppAI i wpisz `wygeneruj kod na silnię`. Odpowiedź modelu będzie
pokazana jako propozycja/diff. Nie jest ona automatycznie wstawiana do pliku.
Stan połączenia albo rzeczywisty błąd pojawi się w panelu.

Dla jednoznacznej nazwy funkcji i zgodności z poniższym zestawem testów można
użyć istniejącego trybu podpisu:

```text
@python-signature silnia(n)
Wygeneruj kod na silnię dla nieujemnej liczby całkowitej n. Zwróć wynik.
```

Nagłówek funkcji jest wtedy dostarczany przez aplikację; model NppAI generuje
jej treść. Ten tryb nie stanowi dowodu, że model sam nauczył się kopiowania nazw.
Uruchomienie połączenia nie poprawia automatycznie umiejętności checkpointu:
aktualne małe modele badawcze mają udokumentowane błędy generowania kodu.

## 5. Opcjonalnie: walidacja i przycisk Zastosuj

Bez konfiguracji walidacji zobaczysz odpowiedź/diff i wskazówki konfiguracji,
a **Zastosuj pozostanie wyłączone**. Aby włączyć zatwierdzanie, potrzebujesz
Docker Desktop z kontenerami Linux, lokalnego obrazu Pythona i własnych testów.
Nie uruchamiaj wygenerowanego kodu bezpośrednio w procesie Notepad++.

Dla powyższej funkcji `silnia(n)` dodano autorskie przypadki testowe dla
0, 1, 5 i 10 w `examples/silnia-validation-cases.json`.
W tej samej konsoli PowerShell, przed uruchomieniem skryptu:

```powershell
$dockerHost = 'npipe:////./pipe/docker_engine'
docker --host $dockerHost pull python:3.12-slim
$env:NPPAI_VALIDATION_IMAGE = docker --host $dockerHost image inspect python:3.12-slim --format '{{.Id}}'
$env:NPPAI_VALIDATION_PYTHON = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:NPPAI_TEST_CASES = (Resolve-Path '.\examples\silnia-validation-cases.json').Path
# Teraz uruchom polecenie start_nppai.py z punktu 3.
```

Te testy odnoszą się do `silnia(n)`. Dla innego zadania wskaż odpowiedni własny
zestaw zamiast pozostawiać testy silni. Zastosuj odblokuje się dopiero po
poprawnym sprawdzeniu składni i wskazanych testów. Mechanizm nie oznacza
ogólnego dowodu poprawności kodu. Szczegóły: [walidacja](validated-generation.md).

## Rozwiązywanie problemów

| Komunikat | Działanie |
| --- | --- |
| Brak modelu lub konfiguracji | Sprawdź plik `.nppai` i uruchom Notepad++ przez `start_nppai.py`. |
| Port zajęty | Zatrzymaj poprzedni serwer albo zmień `--port`. |
| Readiness HTTP 503: model unavailable | Plik jest nieobecny/uszkodzony lub nie jest dense v2 byte; sprawdź eksport. |
| Błąd połączenia / Windows error 12002 | Sprawdź otwartą konsolę, działanie serwera i czas generowania modelu. |
| HTTP 401/403 | Klucze serwera i edytora różnią się; uruchom obie aplikacje tym samym skryptem. |
| HTTP 422 | Pełne zapytanie/kontekst nie mieści się w możliwościach modelu. |
| Odpowiedź jest, Zastosuj wyłączone | Skonfiguruj walidację lub sprawdź faktyczne błędy testów w panelu. |
