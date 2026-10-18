# Kierunki rozwoju jakości NppAI

Ocena dotyczy naszego autorskiego modelu i aktualnych pomiarów repozytorium.
Nie wymaga zastąpienia NppAI zewnętrznym modelem. To plan eksperymentów, a nie
obietnica, że każdy z nich poprawi wyniki.

## Co już wynika z pomiarów

Model pilotażowy ma 180 544 parametry, dwie warstwy bloków i słownik 256 bajtów.
W tym eksperymencie okno kontekstu ma 1024 bajty, a nie 1024 słowa. Taki model
jest użytecznym prototypem badawczym; obecne pomiary nie uzasadniają przedstawiania
go jako ogólnego asystenta programisty.

[Osobny trening kodowania](staged-code-fit.md) poprawił dopasowanie kodu z
poprawnymi wcześniejszymi etapami z 16/96 do 92/96. Powtórki odpowiedzi
bezpośrednich dały wariant 86/96 i zachowanie wszystkich 88 pierwotnie poprawnych
odpowiedzi bez etapów. Jednak cały proces etapowy nie poprawił jakości: utracił
dwa wcześniejsze trafienia, a istniejące zadania spoza treningu nadal mają
0 poprawnych pełnych odpowiedzi. Dopasowanie do treningu i samodzielne rozwiązanie
nowego problemu muszą pozostać oddzielnymi miarami.

## Priorytety i kryteria decyzji

| Kolejność | Co rozwijać | Dlaczego / jak zmierzyć |
| --- | --- | --- |
| 1 | Zachowanie wszystkich etapów podczas douczania | Sam replay kodu nie ochronił specyfikacji i planu. Porównać replay wszystkich etapów z zamrożonym modelem planującym i osobnymi wagami kodera. Sprawdzać każdy wcześniejszy poprawny przypadek, nie tylko sumę trafień. |
| 2 | Przenoszenie nazw, argumentów i operacji na nowe zadania | Oddzielne testy: zmieniona nazwa, zmienione argumenty, nowe brzmienie, nowa operacja. Przy niezmienionym zadaniu funkcja ma zachować nazwę z polecenia i poprawne powiązanie zmiennych. Mechanizm kopiowania/wiązania jest hipotezą do sprawdzenia, nie gotową poprawką. |
| 3 | Różnorodne dane z niezależną weryfikacją | Autorskie lub właściwie licencjonowane zadania z testami: warunki, pętle, listy, tekst, błędy i przypadki brzegowe. Grupować podobne szablony przed podziałem train/dev/test. Samo przemianowanie 100 funkcji nie tworzy 100 nowych problemów. |
| 4 | Uczenie naprawy z wyników testów | Gromadzić: polecenie, błędny kod, diagnostykę, poprawkę i wynik niezależnych testów. Oddzielać dane treningowe od zamrożonych testów. W pierwszym etapie uczyć na sprawdzonych poprawkach, bez automatycznego uznawania własnych odpowiedzi za prawdę. |
| 5 | Jakość odpowiedzi po polsku i wykonywanie instrukcji | Dane na wyjaśnienie kodu, krótką odpowiedź, poprawkę i wskazanie brakujących wymagań. Oceniać zgodność z poleceniem, poprawność faktów, kompletność, język i nieuzasadnione twierdzenia. Testować język przy niezmienionym programie i nazwach. |
| 6 | Pojemność, tokenizacja i użyteczny kontekst | Po ustabilizowaniu danych porównać mały i większy model przy jawnym budżecie oraz tokenizer bajtowy i wariant z obsługą identyfikatorów. Mierzyć testy, RAM i czas; dłuższe okno bez odpowiednich danych nie jest dowodem lepszego rozumowania. |
| 7 | Kontekst repozytorium i ograniczone działanie agenta | Najpierw testować, czy trafnie dobrany fragment kodu poprawia odpowiedź. Potem poprawki wielu plików, kompilacja i ograniczona naprawa. Zachować przegląd różnic i zweryfikowane Apply. |

Najbliższy eksperyment powinien chronić planowanie i bezpośrednie odpowiedzi,
podczas gdy koder uczy się osobno. Najprostsze porównanie to replay wszystkich
czterech zadań kontra dwa własne checkpointy: zamrożony planujący i osobno uczony
koder. To zwiększa wymagania pamięci i routingu, więc wymaga pomiaru całego procesu.
Nie zakładamy z góry, że dwa modele wygrają z jednym.

## Jak oceniać odpowiedź, a nie tylko stratę treningową

Dla kodu raportować: poprawną składnię, właściwą nazwę i argumenty, niezależne
wykonanie testów, przypadki brzegowe, zachowanie wcześniejszych trafień oraz czas
i pamięć. Dla tekstu: poprawność, wykonanie instrukcji, jasność, kompletność i
adekwatność języka. Krótsza odpowiedź jest lepsza tylko wtedy, gdy spełnia zadanie;
dłuższy plan i więcej statusów nie oznaczają lepszego wyniku.

Każdy kolejny zbiór testowy zamrozić przed treningiem, oddzielić rodziny/szablony
i ujawnić wcześniejsze wykorzystanie. Obecne małe zbiory są już obejrzanymi
regresjami. Nie używać ich kolejnych wyników do wyboru hiperparametrów, a potem
opisywać jako niezależnego testu. Promocja modelu wymaga wzrostu poprawności
funkcjonalnej przy zachowaniu wcześniejszych umiejętności; zielone CI samo tego
nie potwierdza.

## Co wspierają badania i czego nie dowodzą dla NppAI

- [CodeRL, Le i in., 2022](https://arxiv.org/abs/2207.01780) wykorzystuje sygnały
  poprawności i testy przy uczeniu/generowaniu kodu. Uzasadnia badanie informacji
  zwrotnej z wykonania, lecz jego wyniki dotyczą innych, większych modeli i nie
  gwarantują poprawy NppAI. Na naszym etapie najpierw potrzebne są sprawdzone dane
  napraw i podstawowa skuteczność, zanim uruchomimy rozbudowane RL.
- [Ouyang i in., 2022](https://arxiv.org/abs/2203.02155) opisują demonstracje
  poprawnego zachowania i ocenę odpowiedzi przy uczeniu wykonywania instrukcji.
  Dla NppAI proponuję najpierw jawne przykłady polecenie–odpowiedź i prostą ocenę
  jakości, a nie natychmiastową implementację całego RLHF.
- [Hoffmann i in., 2022](https://arxiv.org/abs/2203.15556) badają równowagę danych,
  wielkości modelu i budżetu. To argument za mierzeniem tych czynników razem.
  Ich konkretne proporcje badano w znacznie większej skali; nie przenosimy ich
  mechanicznie na model o 180 tys. parametrów.

Powyższe priorytety są wnioskiem projektowym z naszych pomiarów i tych publikacji.
Nowe architektury, kopiowanie identyfikatorów, zmiana tokenizacji oraz uczenie z
wykonania pozostają eksperymentami do porównania, nie wdrożonymi zdolnościami.
