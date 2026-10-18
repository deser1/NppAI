# Ochrona etapów: powtórki wszystkich zadań a zamrożone planowanie

Eksperyment wykorzystuje wyłącznie własne modele NppAI. Porównuje trzy warianty
bez zmiany formatu poleceń i bez podawania wzorcowych odpowiedzi do samodzielnego
generowania. Nie włącza nowego trybu w DLL ani nie podmienia domyślnego modelu.

## Stałe warunki

[Protokół](../evaluation/all-stage-protection-v1.json) określa 512 aktualizacji,
seed 43, cztery przykłady na aktualizację, AdamW z learning rate 0.0003 i clip 1.
Wszystkie 96 przykładów treningowych pozostają niezmienione. Każda aktualizacja
nowego modelu zawiera po jednym celu: odpowiedź bezpośrednia, specyfikacja, plan,
kod; przypisanie do wylosowanych zadań rotuje. Przy treningu planu/kodu wcześniejsze
artefakty są autorskie i poprawne, a przy samodzielnej generacji pochodzą z modeli.

Źródło planującego i inicjalizacji: artefakt `12149182157`, checkpoint
`4859ad69e93ff2ca80d8951bca617476d118f8ef127abc4b313b8d2f0671c6fd`.
Osobny koder: wcześniej uczony wariant z powtórkami odpowiedzi bezpośrednich,
artefakt `12152565374`, checkpoint
`236cf897531431972120ce29a6272812dc593f187883ed06d34a9b7b99dfda76`.
Archiwa i wagi są weryfikowane pełnymi hashami; brak źródła nie wybiera zamiennika.

| Wariant | Specyfikacja i plan | Kod i odpowiedź bezpośrednia | Liczba załadowanych parametrów wag |
| --- | --- | --- | ---: |
| Początkowy | Model początkowy | Ten sam model | 180544 |
| Powtórki wszystkich etapów | Nowy model po 512 aktualizacjach | Ten sam nowy model | 180544 |
| Zamrożone planowanie + koder | Niezmieniony model początkowy | Wcześniej uczony osobny koder | 361088 |

Koder z ostatniego wariantu ma już 512 aktualizacji z poprzedniego eksperymentu.
Zweryfikowano zgodność jego inicjalizacji i kolejności czterech zadań w każdym
kroku z nowym treningiem. Różnią się jednak cele i ich udziały: wcześniejszy
koder miał po dwa cele kodu etapowego i bezpośredniego. Porównanie nie zrównuje
liczby bajtów celu, długości sekwencji, czasu ani kosztu pamięci. Podana liczba
parametrów nie jest pomiarem całkowitego RAM procesu.

Nowy `experimental_stage_router.py` ma jawną, stałą tożsamość modelu dla każdej
roli. Specyfikację i plan generuje planujący, kod — koder. W wariancie z jednym
modelem obie role mają ten sam checkpoint. Pusty, ucięty, ponadbudżetowy wynik lub
nieoczekiwana tożsamość modelu zatrzymuje proces. Nie ma ukrytej korekty nazw,
wstawiania nagłówka ani przełączania modelu po błędzie. Istniejąca kontrola
pojedynczego checkpointu w zwykłym `run_staged` pozostaje bez zmian.

## Wyniki lokalnego pomiaru

To **dopasowanie do 96 znanych przykładów treningowych**, nie test nowych zadań.

| Dokładne odpowiedzi | Początkowy | Wszystkie etapy | Zamrożone planowanie + koder |
| --- | ---: | ---: | ---: |
| Bezpośrednio z polecenia | 88/96 | 96/96 | 96/96 |
| Cały proces samodzielnie | 2/96 | 45/96 | 10/96 |
| Kod z autorską specyfikacją i planem | 16/96 | 61/96 | 86/96 |
| Dokładna wygenerowana specyfikacja | 12/96 | 55/96 | 12/96 |
| Dokładny wygenerowany plan | 11/96 | 63/96 | 11/96 |

Wszystkie mianowniki obejmują także etapy, do których proces nie doszedł. Ostatnie
dwa wiersze porównują teksty z autorskimi celami; nie są oceną każdego możliwego
poprawnego sformułowania planu. Wiersz z autorskimi artefaktami jest **wspomagany**
i nie liczy się jako samodzielne rozwiązanie.

Oba warianty zachowują wszystkie 88 wcześniejszych trafień bezpośrednich i oba
wcześniejsze trafienia całego procesu. Powtórki wszystkich etapów nie chronią
jednak każdego wcześniejszego artefaktu: zachowują **10/12** specyfikacji i
**10/11** planów oraz **12/16** trafień kodu z autorskimi artefaktami. Zamrożony
planujący zachowuje **wszystkie surowe wywołania specyfikacji i planu**, a osobny
koder wszystkie 16 wcześniejszych trafień wspomaganych. Audyt weryfikuje także
listy utraconych przypadków, więc wzrost sumy nie jest mylony z pełną retencją.

Wszystkie warianty nadal mają **0/16** pełnych odpowiedzi walidacyjnych i **0/12**
w istniejącym zbiorze dwujęzycznym, zarówno bezpośrednio, jak i etapowo. Są to już
oglądane zbiory regresyjne. Nie promujemy żadnego checkpointu na tych wynikach.

## Weryfikacja i interpretacja

Pomiar zapisuje surowe żądania/odpowiedzi, role i identyfikatory, dane poprzednich
artefaktów, każdy krok treningu, straty i ekspozycję tokenów. Modele zamrożone
muszą zachować hash stanu. To samo dotyczy wszystkich modeli po ewaluacji. Cztery
stałe wywołania porównują nowy eksport dense-v2 z runtime'em NumPy; nie jest to
certyfikacja wszystkich danych ani natywnej ścieżki INT8.

Dedykowany workflow ponawia pomiar i wykonuje wygenerowany kod dwujęzycznego
zbioru w istniejącym ograniczonym Dockerze. Wymaga przejścia wszystkich 12 autorskich
rozwiązań kontrolnych i odrzucenia znanego błędu. Błędy infrastruktury przerywają
pomiar, a puste odpowiedzi pozostają porażkami. Lokalny pomiar nie ma Dockera i
jawnie zapisuje `not_run`. Audyt offline rekonstruuje wyniki, role, retencję,
zbieżność historycznej kolejności zadań i pochodzenie kodu użytego do walidacji.

W tym pilotażu powtórki wszystkich etapów poprawiają **całą generację treningową**
bardziej niż zamrożone, słabe planowanie. Zamrożenie natomiast daje silniejszą
ochronę poprzednich artefaktów. Te obserwacje nie rozstrzygają ogólnej przewagi
jednej architektury: różnią się udziały celów treningowych i liczba modeli.

Kolejny krok powinien badać przenoszenie nazw i argumentów oraz działanie kodera
z rzeczywistymi, niedoskonałymi artefaktami planującego. Ewentualne nowe dane
wejściowe i cele muszą pochodzić tylko z treningu; artefakty maszynowe nie stają
się automatycznie prawdziwymi planami. Zamrozić nowy podział zadań przed kolejnym
treningiem, utrzymać kontrolę wszystkich wcześniejszych trafień i wymagać wzrostu
poprawności funkcjonalnej przed zmianą domyślnej wtyczki.

```sh
python -m unittest discover -s tests -p 'test_all_stage_protection*.py' -v
python scripts/all_stage_protection.py --original original.zip --coder coder.zip \
  --output measured --image sha256:LOCAL_VALIDATOR_IMAGE
python scripts/audit_all_stage_protection.py measured
```

Artefakt workflow `nppai-all-stage-protection` zawiera nowe i zamrożone wagi,
przebieg treningu, odpowiedzi, porównanie runtime'ów i diagnostykę Dockera.
Pominięcie `--image` oznacza jawne pominięcie wykonania funkcjonalnego.

Surowe pomiary lokalne: [all-stage-records.zip](evidence/staged-protocol/all-stage-records.zip)
i [hashe plików/źródeł](evidence/staged-protocol/all-stage-source.json). Po rozpakowaniu
można uruchomić audyt na wskazanym katalogu. Powtórzenie na tej samej maszynie dało
identyczne nowe wagi, ślad treningu oraz wszystkie surowe grupy odpowiedzi. Nie
jest to ogólna certyfikacja powtarzalności między środowiskami.
