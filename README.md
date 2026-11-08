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
git clone https://github.com/grzegorz2047/DescriptiveAssets.git sightline
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
MASS        volume 24.237 m3  center [0.00, 3.25, -0.01]  stable yes  support margin 1.29 m
            ground contacts: base
CUTS        5  (removed = primitive volume inside the cut, visible = share of the carved footprint seen)
  door           0.304    94%     0%   0%  41%  shaft
READABILITY front, iso: ...
  front  32 px  IoU 85%
                lost 4: lintel, band, pole, pennant
WARNINGS    5
  - at 32 px (front, iso) these parts vanish: lintel, band, pole, pennant, post_1, ... — make them larger ...
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

5. Gdy sylwetka się zgadza, `sightline structure lighthouse` sprawdza **wnętrze**: gdzie na obrazku
   jest ciemna rama, a gdzie jasne płyty i światła, oraz które części modelu obrazek dzieli na kilka
   elementów. Raport mówi wprost, co zmienić: „obrazek jest tu ciemny, model jasny: `chest`
   (metal); daj tu ciemną część” albo „brakuje światła w x −0,05..−0,02, y 1,77..1,79”.
   `sightline structure refs/obrazek.png --height 1.9` rozkłada sam obrazek na listę elementów
   (płyty, rama, światła) z wymiarami w metrach i parami lewa–prawa.

**Płyty prosto z obrazka.** Bryła `extrude` to dowolny obrys (wielokąt) wyciągnięty na grubość —
kanciaste płyty, tarcze, ostrza, emblematy. `sightline structure robot --trace R12,R15` sam obrysowuje
wskazane elementy obrazka i wypisuje gotowe części w metrach, ustawione na powierzchni modelu.
Robot z przykładów doszedł tak od 69% do 93% zgodności materiałów i od 43% do 91% zgodności podziałów
z konceptem (`out/robot/increments_all.png` pokazuje każdy krok).

**Kilka widoków.** Do zasobu można dać kilka obrazków: przód, tył, lewy i prawy bok (lista
`reference:`). `match --index` i `structure --index` porównują model z każdym z nich. Bryła
`extrude` przyjmuje też obrys z boku (`side:`), więc część buduje się jak z rzutów: obrys z przodu
przecina się z obrysem z boku (łydka, która puchnie z tyłu i zwęża się ku kostce, płyta piersi
wygięta do obojczyka). `--trace ... --conform` obrysowuje płyty także z tyłu i z boku, wygięte wzdłuż
ciała. Robot doszedł tak z boku od IoU 0,76 do 0,89–0,90, a przód został na 0,90
(`out/robot/views4_r20.png` pokazuje koncept i każdą rundę w czterech widokach).

**Malowanie.** Ta sama bryła bywa z tyłu srebrna, a z przodu ciemna. Część z `op: paint` nie dodaje
bryły: powierzchnia innych części w jej obrysie dostaje jej materiał (strefa koloru, naklejka, linia
podziału). Obrys i wymiary się nie zmieniają, a eksport do .glb dzieli siatkę na materiały.
`sightline structure robot --index I --trace fix --paint --conform` sam maluje tylko te miejsca,
gdzie kolor modelu różni się od obrazka I. `sightline render robot --light view` oświetla tył i boki
tak jak przód, jak na arkuszu z konceptem.

**Sufit i dopasowanie.** `match` porównuje też obrazki z przeciwnych stron (przód z odbitym tyłem,
lewy bok z odbitym prawym), bo obrys każdej bryły widziany z przeciwnej strony jest lustrzanym
odbiciem. Linia `CEILING` mówi, ile w ogóle da się osiągnąć: przód i tył robota zgadzają się na
0,96, boki tylko na 0,91, więc żaden model nie przekroczy średnio 0,95 z boku. Tam rysunki są ze
sobą sprzeczne i to pytanie do projektanta. `sightline fit robot --out ...` sam przesuwa liczby
części leżących przy błędach (położenie, wymiary, obrót, szerokość obrysów) krokami 4, 2 i 1 mm.
Krok zostaje tylko wtedy, gdy rośnie średnia z obrysu, materiałów i podziałów, a żaden widok nie
spada poniżej startu. Części na osi symetrii nie przesuwają się w bok, a wynik trafia do pliku
z zachowaniem komentarzy, po sprawdzeniu, że kontrole dalej przechodzą. Przy okazji `structure`
czyta obrazek na własnej siatce, a nie na siatce modelu, więc odczyt nie zmienia się razem z
modelem. Wcześniej jeden wiersz różnicy w kadrze potrafił przesunąć wynik materiałów o kilka
punktów, a liczby rundy 21 (94–98%) były przez to zawyżone; na stabilnym odczycie wynosiły
94/91/93/90%. Po dopasowaniu i nowym przejściu kolorów robot ma IoU 0,93/0,93/0,93/0,94,
materiały 97/98/97/97% i podziały 97/85/88/91% (przód, tył, prawy, lewy;
`out/robot/views4_r22.png`).

**Rzeźbienie.** Połowa błędu obrysu to model wystający poza obrazek, a dokładanie części tego nie
usuwa. Robot dostał więc cięcia `op: cut` przez całe ciało tam, gdzie wystaje poza zachowawczy obrys
(suma przodu, odbitego tyłu i ich lustrzanych odbić; z boku suma prawego i odbitego lewego), oraz
wypełnienia tam, gdzie wszystkie obrazki pokazują bryłę, a modelu brakuje. Każde cięcie i wypełnienie
przeszło przez tego samego strażnika co fitter. Obrys wzrósł do IoU 0,95/0,95/0,94/0,95, a materiały
i podziały zostały bez zmian (`out/robot/views4_r23.png`). Kosztem są dłuższe kontrole robota
(ok. 3,5 minuty zamiast 2). Robi to polecenie `sightline carve <asset> --out ...` dla każdego zasobu
z obrazkami z przodu, z tyłu lub z boku.

**Sprzeczne rysunki.** Oba boki robota zgadzają się ze sobą tylko na 0,91 (linia `CEILING`), więc
jeden model nie dopasuje się do obu. Test z pominiętym zdjęciem (`--holdout`) to potwierdził:
rzeźbienie do prawego boku podniosło prawy do 0,954, a lewy spadło do 0,923. Projektant pozwolił
pominąć jeden bok, więc robot ma teraz trzy obrazki (przód, tył, lewy) i obrys 0,949/0,951/0,954
(`out/robot/views4_r26.png`). Obrysy cięć są wygładzane przed uproszczeniem, bo każdy schodek piksela
przeciągnięty przez całe ciało rysował się jako pasek na udzie.

**Cała runda w czterech poleceniach.** Po ręcznym ułożeniu kształtów: `sightline fit` dociąga
liczby, `sightline carve` tnie i wypełnia obrys, `sightline paint --strip` maluje od nowa z każdego
obrazka po kolei (boki, tył, przód), a `sightline score --against poprzedni.yaml --out score.png`
daje jedną tabelę ze wszystkimi miarami dla każdego obrazka oraz obrazek do pokazania: koncept na
górze, pod nim wiersz na każdą wersję, z IoU pod każdym kafelkiem. Cięcia mają własny limit 256,
osobny od brył, więc rzeźbienie nie zabiera miejsca częściom.

**Rowki i linie paneli.** `sightline detail` czyta krawędzie, które obrazek ma wewnątrz elementów
(rowki, szwy, linie paneli), a model nie, i rysuje je na powierzchni jako ciemne malowania kształtu
`strokes` (otwarte linie o zadanej szerokości, po jednym wpisie na część i widok). Wiersz DETAIL w
`structure` mówi, ile krawędzi wnętrza obrazka ma już odpowiednik w modelu; linie zostają, gdy ta
miara rośnie, a materiały i podziały nie spadają.

**Liczenie elementów.** Zgodność pikseli nie odróżni dwojga oczu od czworga: model z dodatkową lampą
potrafi lepiej pasować do kilku zdjęć naraz. Wiersz `COUNTS` w `structure` i kolumna „lights" w `score`
liczą osobne plamy każdej klasy na obrazku i w modelu i mówią, ile z nich to ten sam element; check
`spots: {view: front, count: 2, material: glow_blue, box: [-0.1, 0.1, 1.7, 1.85]}` przypina liczbę
plam materiału w widoku (w prostokącie w metrach) bez obrazka.

**Gotowy model zamiast obrazka.** `sightline import model.glb` zamienia siatkę na wersję opisową:
każdy kawałek modelu staje się nazwaną bryłą (prostopadłościan, walec, elipsoida, stożek) z
najbliższym materiałem i błędem dopasowania w centymetrach. To punkt startu do dalszej pracy z
`match` i `structure`.

Kilka rzeczy, które `match` robi za agenta:

- IoU liczy z dokładnych sylwetek (pole powierzchni, a nie piksele z zaokrągleniem), więc wynik
  prawie nie zależy od rozdzielczości (±0,01 między `--res 192` a `1024`), a cienkie skrzydła nie
  dostają fałszywego „extra”.
- Cienkie linie z obrazka (druty, linki, pręty) zgłasza jako **thin detail** — bez sugestii, że
  sąsiednia część jest za mała.
- Dół obrazka kładzie na ziemi (y = 0); część wystająca pod ziemię nie przesuwa całego porównania,
  tylko jest wymieniona w linii `ALIGN`.
- Rozpoznaje obrazek **oświetlony** (render z cieniowaniem, zdjęcie) i wtedy nie udaje, że zmierzone
  kolory to farba: białe skrzydła w ciemnym renderze mierzą #555a58, a nie „żelazo”. Odpowiedź
  projektanta zapisuje się jako `lighting: lit` albo `flat` w `reference:`.

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

Przydatne w formacie: `between: [dolne_skrzydlo.top, gorne_skrzydlo.bottom]` stawia zastrzał
(albo drut) od punktu do punktu — bez liczenia kątów i długości; `tags: [hidden]` oznacza część,
która ma być niewidoczna (np. silnik w osłonie), więc raport nie ostrzega, że jej nie widać.

Poziomy: `levels/courtyard.yaml` (wszystkie checki przechodzą) i `levels/broken_bridge.yaml`
(trzy checki celowo nie przechodzą). Dokumentacja: `docs/FORMAT.md` (pełny opis formatów, po
angielsku), `docs/AI_GUIDE.md` (jak ma pracować agent), `SPEC.md` (kontrakt między modułami).

## Ograniczenia prototypu

- Bryły są tylko z prymitywów (prostopadłościan, kula, walec, stożek, kapsuła, torus, klin,
  elipsoida) z wycięciami i płynnym łączeniem — nie ma rzeźbienia, tekstur z obrazków ani
  animacji.
- Renderer jest programowy (numpy, bez GPU): 256 px na widok, bez antyaliasingu i bez światła
  emitowanego (lampa to po prostu jasny kolor). Cienie na samym assecie są ostre, długi cień
  wysokiej wieży może wypaść poza płytkę podłoża. Naklejki (np. kokardy na skrzydłach) muszą
  wystawać ponad powierzchnię o ~0,1 piksela renderu (5 mm przy 10-metrowym assecie i 256 px),
  inaczej wygrywa część zdefiniowana wcześniej. Widoki od dołu (`orbit:30:-20`) rysują się bez
  płytki podłoża, a `render --no-ground` usuwa ją zawsze.
- Pomiary mają rozdzielczość: sylwetki w renderach są do pół piksela za szerokie na krawędzi
  (`match` liczy już z dokładnych), „widoczność” części za gęstymi kratami wychodzi zaniżona, a
  kontrast liczony jest z uproszczonego oświetlenia (ciemniej niż na PNG). Części cieńsze niż
  komórka wokseli mają niedokładną objętość; stabilność bierze jednak punkty styku z ziemią z
  dokładnych brył, więc okrągłe koła i cienka płoza się liczą, a ostrzeżenie wymienia, na czym
  model stoi.
- Dopasowanie do obrazka działa na sylwetce z jednego widoku; przy obrazku oświetlonym podaje
  tylko kolory względne (która część jest jaśniejsza od reszty), bo farby nie da się odczytać z
  cieniowania. Zdjęcia z tłem pełnym szczegółów wymagają wyciętej maski. Zgadywanie kąta (`--fit-view`) myli się przy
  obiektach okrągłych.
- Uproszczone siatki mają wspólne wierzchołki, więc ostra krawędź cieniuje się jak wąskie
  sfazowanie, a nie twarde załamanie. Krawędzie biegnące ukośnie do siatki (listwy skrzyni pod 45°)
  są po surface nets lekko ząbkowane i po uproszczeniu zostaje na nich więcej trójkątów. Eksport
  skrzyni zajmuje ok. 13 s (samo upraszczanie ok. 10 s). Pliki `.glb` przechodzą walidator
  struktury i otwierają się w Blenderze 5.2 (każda część to osobny obiekt o właściwych wymiarach,
  zamknięty, bez krawędzi non-manifold); w Godot i Unity jeszcze ich nie otwierano.
- Poziomy to siatka kolumn terenu z jednym modelem ruchu gracza; obiekt blokuje całe komórki
  swojego obrysu, w których jego bryła schodzi niżej niż wzrost gracza plus skok (pod skrzydłami
  wiatraka da się przejść, a linia wzroku przechodzi pod nimi). Eksport poziomu do glTF nie jest
  jeszcze zrobiony.
- Eksport siatek (surface nets) może zostawić kilka krawędzi wspólnych dla czterech trójkątów
  w wklęsłych narożnikach wycięć; `export` wypisuje je w linii `MANIFOLD`, ale ich nie naprawia
  (upraszczanie ich nie pogarsza).
- Analiza struktury obrazka zna trzy klasy (ciemne, jasne, akcent): dobrze działa na projektach
  mechanicznych (płyty, rama, światła), słabiej na kolorowych strojach (czerwień, złoto, skóra).
  Import `.glb` dopasowuje jedną bryłę do każdego kawałka siatki, więc duże, jednolite siatki (np.
  całe nadwozie) wychodzą jako jeden prostopadłościan.

## Licencja

MIT — zobacz plik `LICENSE`.
