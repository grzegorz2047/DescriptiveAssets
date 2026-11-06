# Sightline

Sightline to prototyp silnika do tworzenia assetów 3D i poziomów gier **tekstem**. Asset (wieża,
skrzynia, rycerz) i poziom (plansza z kafelkami) to pliki YAML, które pisze agent AI — albo Ty.
Silnik daje agentowi „wzrok”: renderuje obrazki PNG (żeby ocenić całość) i drukuje **zmierzone
fakty** (wymiary w metrach, co czego dotyka, co lewituje, co jest zasłonięte, czy sylwetka zgadza
się z Twoim szkicem, gdzie gracz utknie). Twoje życzenia zapisuje się jako `checks:` — testy z
wynikiem PASS/FAIL — a agent poprawia model, dopóki wszystkie przechodzą.

Dlaczego tak? Model językowy źle ocenia proporcje i drobne błędy na obrazku, ale świetnie czyta
liczby i nazwy. Asset złożony z nazwanych części (`roof`, `door`, `shield`) można poprawiać
lokalnie: „zrób dach bardziej stromy” to zmiana jednego pola `roof.height`.

## Instalacja

Wymagany Python 3.11+ i trzy biblioteki: numpy, pillow, pyyaml (bez GPU).

```
git clone <repozytorium> sightline
cd sightline
python -m venv .venv
.venv\Scripts\pip install -e .[dev]          # Linux/macOS: .venv/bin/pip
.venv\Scripts\sightline --help               # albo: python -m sightline --help
```

Testy: `.venv\Scripts\python -m pytest tests -q`.

## Pięciominutowa wycieczka

Wszystkie wyniki poniżej to prawdziwe wydruki z przykładów w `assets/` i `levels/`.

**1. Obejrzyj asset.** `render` zapisuje widoki, mapę części i arkusz zbiorczy — z szarym
człowiekiem 1,8 m dla skali i podpisem „1 px = … m”:

```
$ sightline render watchtower
RENDER  watchtower: front, right, top, iso + parts + sheet
wrote E:/sightline/out/watchtower/front.png
...
wrote E:/sightline/out/watchtower/sheet.png
```

**2. Sprawdź życzenia projektanta.** Każdy check mówi, co zmierzył i czego oczekiwał:

```
$ sightline check watchtower
ASSET  watchtower  (E:/sightline/assets/watchtower.yaml)
  PASS  grounded           measured min y 0.00 m  expected |min y| <= 0.05 m
  PASS  size               measured height 8.80 m, width 2.90 m, depth 2.90 m  expected height 8.00..9.50 m, ...
  PASS  visible            measured door 94% in front  expected >= 80%
  PASS  readable_at        measured IoU 85% at 32 px front  expected IoU >= 80%, keep roof, shaft
  CHECKS  9: 9 passed, 0 failed
```

Gdy coś nie przechodzi, linia mówi co zrobić i przypomina Twoje „dlaczego”:

```
FAIL  contrast  measured 1.19 (hoop_high #323438 vs upper #543a21)  expected >= 1.30
      — make one material lighter or darker  (why: the iron hoops must read against the wood)
```

(to prawdziwy błąd z pierwszej wersji beczki: ciemne żelazne obręcze na ciemnym dębie — po
zmianie obręczy na stal check przechodzi).

**3. Zajrzyj do pomiarów.** `inspect` to raport, któremu agent ufa bardziej niż własnym oczom:

```
$ sightline inspect watchtower
SIZE        width 2.90 m  height 8.80 m  depth 2.90 m  = 4.89 x human height (1.8 m)
GROUND      grounded yes  min y 0.00 m  floats by 0.00 m
MASS        volume 24.237 m3  center [0.00, 3.25, -0.01]  stable yes  support margin 1.27 m
CUTS        5  (removed = primitive volume inside the cut, visible = share of the carved footprint seen)
  door           0.304    94%     0%   0%  41%  shaft
READABILITY front: ...
   32 px  IoU 85%   lost 4: lintel, band, pole, pennant
WARNINGS    5
  - at 32 px (front) these parts vanish: lintel, band, pole, pennant — make them larger or thicker ...
```

Są też sylwetki jako siatki liter z opisem w metrach, kontakty części, kontrasty kolorów i ostrzeżenia
w stylu „ball floats 0.50 m above block — attach it with on: block.top”.

**4. Zagraj w poziom.** Bot z modelem ruchu (krok, skok, spadek) przechodzi planszę:

```
$ sightline level play courtyard
PATH  spawn → goal 28.00 m (horizontal 25.00 + vertical 3.00), 1 jump(s), max rise 0.00 m, max fall 1.50 m
   3. jump E over water 2 cells (2,4)→(5,4)
   7. climb +1.50 m (ladder) E (11,2)→(12,2)
   9. pick key_red (13,2)
  13. open door_red with key_red (13,6)
SIGHT  eye height 1.70 m
  spawn → goal             BLOCKED by wall (11,8) top 3.00 m, ray at 1.70 m, 9.58 m from spawn
CHECKS  8: 8 passed, 0 failed
```

Poziom `broken_bridge` jest celowo zepsuty — bot znajduje miejsce, w którym gracz utknie, sekret
poza zasięgiem skoku i drzwi, które da się obejść (`exit code 1`). `sightline level render courtyard`
rysuje planszę w izometrii z prawdziwymi assetami (wieża, sosna, rycerz, latarnia, skrzynie) i
ścieżką gracza.

**5. Porównaj całą bibliotekę w jednej skali:** `sightline sheet` → `out/library.png`.

Pozostałe komendy: `new` (szablon nowego assetu z Twoim opisem zamienionym na listę TODO),
`diff` (co się zmieniło w pomiarach od ostatniego commita), `styles` (materiały i kolory),
`shapes` (bryły i punkty zaczepienia), `export`, `ref`, `match`. Każda drukuje ścieżki plików,
które zapisała. Kody wyjścia: 0 — OK, 1 — jakiś check nie przeszedł, 2 — błąd w pliku (z numerem
linii i podpowiedzią, np. `did you mean 'crate'?`).

## Praca ze zdjęciem lub szkicem

Najczęściej zaczynasz od obrazka. Sightline mierzy obrazek tak samo jak model, więc agent porównuje
liczby z liczbami.

1. Wrzuć obrazek do `assets/refs/` i poproś agenta o model. Agent uruchamia
   `sightline ref refs/lighthouse_concept.png --height 11.85` i dostaje: maskę obiektu, skalę,
   proporcje, profil szerokości w 10 pasach, dominujące kolory z najbliższymi materiałami i listę
   **niewiadomych** (np. jak wygląda tył, jaka jest głębokość).
2. Agent zadaje Ci pytania **w jednej wiadomości**, każde z proponowaną odpowiedzią — wystarczy
   „ok”.
3. Agent buduje bryłę z kilku dużych części, a `sightline match lighthouse` pokazuje różnice:

   ```
   FIT     IoU 0.89   missing 2% (reference not covered)   extra 10% (model outside the reference)
     30-40%     8.36..7.12  ...  model too wide by 0.97 m (left +0.48, right +0.48)
   REGIONS
     extra   0.59 x 2.05 m (4% of ref) at x 1.05..1.64, y 6.96..9.01, next to `band_3` -> `band_3` sticks out here
   ```

   oraz obrazek `out/lighthouse/match_0.png`: Twój szkic | model | nakładka (czerwone = tylko
   model, niebieskie = tylko szkic) | nazwy części na szkicu.
4. Po poprawkach check `matches_reference` przechodzi (latarnia morska z przykładów: IoU 0.95).
   Uwagi przekazujesz po nazwach części: „`dome` wyżej”, „`band_3` węższy”.

## Eksport do silnika gry

```
sightline export knight                  # out/knight/knight.glb — glTF 2.0, osobny węzeł na każdą część
sightline export crate --max-tris 1500   # limit trójkątów na cały asset
sightline export knight --quality high   # dokładniej: high | medium (domyślnie) | low | full
sightline export knight --bake-colors    # + kolory wierzchołków z deseniami materiałów (pełna siatka)
sightline export crate --format obj      # OBJ + MTL
```

Plik `.glb` jest w metrach, oś Y w górę. Godot 4 importuje go bezpośrednio, Blender przez
File → Import → glTF 2.0, Unity przez pakiet glTFast. Części zachowują nazwy z YAML, więc można je
osobno animować lub podmieniać.

Siatki są upraszczane (decymacja) część po części: płaskie ściany zostają z kilku trójkątów,
krzywizny dostają tyle, ile potrzebują, a każda część pozostaje zamknięta i zachowuje nazwę,
materiały i wymiary (z dokładnością do 5 mm). Skrzynia ma ok. 3 tys. trójkątów zamiast 400 tys.
(plik 80 KB zamiast 9,6 MB), rycerz 2,4 tys. zamiast 71 tys. `--quality` mówi, o ile może się
przesunąć powierzchnia: `high` 1/8, `medium` 1/4, `low` 1/2 komórki siatki; `full` wyłącza
upraszczanie. `--max-tris` to budżet na cały asset — jeśli jakość go nie mieści, tolerancja rośnie,
aż się zmieści. Tabela po eksporcie pokazuje trójkąty przed (`BEFORE`) i po (`TRIS`) oraz
zmierzone odchylenie od prawdziwej powierzchni (`DEV_MM`). `--bake-colors` eksportuje pełną siatkę,
bo kolory wierzchołków potrzebują gęstych wierzchołków.

## Przykłady

| asset | co pokazuje |
|---|---|
| `crate` | ramy, ukośna listwa z lustrzanym odbiciem, wycięcie (cut) w ramie |
| `barrel` | ścięte stożki zszyte płynnie (`blend`), obręcze-torusy |
| `watchtower` | drzwi i okna jako wycięcia, powtórzenia promieniowe, dach-ostrosłup, proporczyk |
| `knight` | postać low-poly: lustrzane ręce, części podążające za obrotem ręki (`follow`), tarcza i miecz |
| `pine_tree` | 5 części, czytelna nawet jako ikona 16 px |
| `street_lamp` | smukła latarnia, materiał `lamp` |
| `lighthouse` | model dopasowany do szkicu `assets/refs/lighthouse_concept.png` |

Poziomy: `levels/courtyard.yaml` (wszystkie checki przechodzą) i `levels/broken_bridge.yaml`
(trzy checki celowo nie przechodzą). Dokumentacja: `docs/FORMAT.md` (pełny opis formatów, po
angielsku), `docs/AI_GUIDE.md` (jak ma pracować agent), `SPEC.md` (kontrakt między modułami).

## Ograniczenia prototypu

- Bryły są tylko z prymitywów (prostopadłościan, kula, walec, stożek, kapsuła, torus, klin,
  elipsoida) z wycięciami i płynnym łączeniem — nie ma rzeźbienia, tekstur z obrazków ani
  animacji.
- Renderer jest programowy (numpy, bez GPU): 256 px na widok, bez antyaliasingu i bez światła
  emitowanego (lampa to po prostu jasny kolor). Cienie na samym assecie są ostre, długi cień
  wysokiej wieży może wypaść poza płytkę podłoża.
- Pomiary mają rozdzielczość: sylwetki są do pół piksela za szerokie na krawędzi, „widoczność”
  części za gęstymi kratami wychodzi zaniżona, a kontrast liczony jest z uproszczonego
  oświetlenia (ciemniej niż na PNG). Części cieńsze niż komórka wokseli mają niedokładną objętość.
- Dopasowanie do obrazka działa na sylwetce z jednego widoku i płaskich kolorach; zdjęcia z
  tłem pełnym szczegółów wymagają wyciętej maski. Zgadywanie kąta (`--fit-view`) myli się przy
  obiektach okrągłych.
- Uproszczone siatki mają wspólne wierzchołki, więc ostra krawędź cieniuje się jak wąskie
  sfazowanie, a nie twarde załamanie. Krawędzie biegnące ukośnie do siatki (listwy skrzyni pod 45°)
  są po surface nets lekko ząbkowane i po uproszczeniu zostaje na nich więcej trójkątów. Eksport
  skrzyni zajmuje ok. 13 s (samo upraszczanie ok. 10 s). Pliki `.glb` skrzyni, rycerza i wieży
  zaimportowano w Blenderze 5.2 (nazwy, wymiary, brak krawędzi non-manifold); w Godot i Unity
  jeszcze ich nie otwierano.
- Poziomy to siatka kolumn terenu z jednym modelem ruchu gracza; obiekt blokuje całe komórki
  swojego obrysu, w których jego bryła schodzi niżej niż wzrost gracza plus skok (pod skrzydłami
  wiatraka da się przejść, a linia wzroku przechodzi pod nimi). Eksport poziomu do glTF nie jest
  jeszcze zrobiony.
- Eksport siatek (surface nets) może zostawić kilka krawędzi wspólnych dla czterech trójkątów
  w wklęsłych narożnikach wycięć; `export` wypisuje je w linii `MANIFOLD`, ale ich nie naprawia
  (upraszczanie ich nie pogarsza).
