# Working with Sightline as an AI agent

Sightline gives you two kinds of eyes on what you build:

- **pictures** (`render`, `match`, `level render`) for the gestalt — does it look like a
  watchtower, is it charming, does it read;
- **measured text** (`inspect`, `check`, `match`, `level play`) for facts you cannot read
  reliably from pixels — sizes in meters, what touches what, what floats, what is hidden, how the
  silhouette compares with the designer's picture, where the player can get stuck.

Trust the text for facts and the pictures for taste. Open every PNG the commands print (they print
absolute paths), and never claim something looks right without having looked.

The format is in `docs/FORMAT.md`; `sightline shapes` and `sightline styles` print the tables you
need most while editing.

## 1. The loop

```
brief ──► ask (one batch) ──► checks ──► blockout ──► render + check ──► fix ──► show ──► feedback
                                              ▲                                      │
                                              └──────────────────────────────────────┘
```

1. **Read the brief** and, if there is a picture, the reference (section 2).
2. **Ask** what you cannot know, in **one** message, each question with a proposed default
   (section 3). Do not wait idle: proceed with the defaults unless the answer could make the work
   useless.
3. **Write the intent as checks first.** `sightline new <name> --brief "..."` writes a template
   whose TODO comments map each sentence of the brief to a likely check. Keep the designer's words
   in `why:` — they are printed when the check fails, so a failure explains itself.
4. **Block out** with few large primitives, named after what they are (`roof`, `door`,
   `left_arm`), attached with `on:` so that moving one part moves its children. Get the big shapes
   right before any detail.
5. **Look and measure:** `sightline render <asset>` then open `out/<asset>/sheet.png` (all views,
   the part map and a header); `sightline check <asset>`; `sightline inspect <asset>` when a check
   fails or something looks odd.
6. **Fix the largest problem first**, re-run, repeat. After a change, `sightline diff <asset>`
   shows what moved in the measurements (against git HEAD, or `--against old_copy.yaml`).
7. **Show** `sheet.png` (and `match_0.png` when there is a reference) with a short list of what
   the checks prove, then take feedback by part names (section 5).

A command that exits **1** found a failing check; **2** means invalid input — read the `where`
(file, YAML path, line) and the `hint`, they usually contain the fix verbatim.

```
$ sightline render crat
unknown asset 'crat'
  hint: did you mean 'crate'? assets in E:/sightline/assets: barrel, crate, knight, ...
```

## 2. The reference loop (a picture from the designer)

The designer usually starts from a photo, a concept painting or a sketch. Bring the model to what
the picture shows, ask about what it cannot show, and keep the model easy to change.

1. **`sightline ref refs/picture.png [--height M] [--view front]`** and look at the picture
   yourself. The report gives you the picture in the units of your model:

   ```
   MASK       from background #c8e0ee (tolerance 24, border spread 0)  confidence high  coverage 25%
   SCALE      height 11.85 m given: 1 px = 0.0156 m
   SIZE       11.85 m tall x 5.34 m wide
   WIDTH PROFILE  (10 bands from the top; m, x from the center)
     0-10%       11.85..10.67    -0.90   +1.00    1.90
     20-30%        9.48..8.30    -1.52   +1.63    3.15
     90-100%       1.20..0.00    -2.67   +2.67    5.34
   COLORS
     #cd4036   32%  rows 0-10%, 20-40%, 60-70% (middle) ... ~ paint_red (dE 4)
   ```

   The silhouette grid (`#` filled) has meter labels on rows and columns; the width profile gives
   the left/right edge of every 10% band; colors come with the nearest style material.
   A `low` mask confidence means the background is busy: ask for a cut-out PNG first.
   Read the `LIGHTING` line: in a **lit** picture (a shaded render or photo) the colours are the
   shading, not the paint — white wings in a dim render measured #555a58 and their "nearest material"
   was iron. Choose materials by what the parts are (canvas wings, wooden struts), keep the picture's
   light-to-dark order, and record the designer's answer as `lighting: lit|flat` in the reference.
2. **Ask the `unknowns`** in one batch, adding your own (polygon budget, which details matter at
   gameplay distance). Each comes with a default you will use if nobody answers.
3. **Block out** following the grid and the width profile: one part per band of the profile
   (a frustum per tapering band, a slab for a deck, a cut for a door). Put the reference into the
   asset (`reference: {image: refs/picture.png, view: front, height: 11.85}`) and add
   `- matches_reference: {min_iou: 0.85}` to the checks.
4. **`sightline match <asset>`** and fix in this order: the **regions** (largest first), then the
   **width profile** bands, then the colors. A region line tells you what and where:

   ```
   FIT     IoU 0.89   missing 2% (reference not covered)   extra 10% (model outside the reference)
     30-40%     8.36..7.12    -1.16..+1.16  2.31    -1.64..+1.64  3.28   model too wide by 0.97 m (left +0.48, right +0.48)
   REGIONS
     extra   0.59 x 2.05 m (4% of ref) at x 1.05..1.64, y 6.96..9.01, next to `band_3` -> `band_3` sticks out here
   ```

   Here `band_3` is a 1.6 m-radius cylinder where the picture tapers to 1.0 m: change its
   `radius`/`top_radius`. Open `out/<asset>/match_0.png` (picture | model | overlay: grey both, red
   model only, blue picture only | part names over the picture).
   - A region marked `thin detail missing: strokes ~0.03 m wide (wire, rod, pole?)` is wires or rods
     the model skips: it says nothing about the size of the part next to it. Add a thin part (a
     `between:` cylinder) if it matters, or leave it and raise `max_missing` a little — thin details
     count with their real area at every `res`.
   - A width verdict `ok within 0.07 m up/down (an edge runs along the band border)` means the
     outline matches once an edge moves a few cm vertically: no width change is needed; look at the
     height of that edge instead.
   - `ALIGN` puts the picture's bottom on the ground line in front/side views. A `! below the ground:
     skid 0.04 m` line names parts poking under y = 0; they do not shift the picture, but fix them
     (`inspect` GROUND).
   - IoU and missing hold within about ±0.01 from res 192 to 1024 (they are area shares of the
     exact silhouettes), so `res` in `matches_reference` is for speed or detail, not for luck.
   - Without `height`, the picture is scaled to the model's height: if the model lacks its topmost
     part, everything looks too small. Fix big regions **top first**, or give `--height`.
   - `suggested` in the color table is the picture's colour for flat paint, and for a lit picture the
     base corrected by that part's light *relative to the other parts* (the picture's overall light
     cancels out): only a part that differs from the rest gets a different colour. Thin parts
     (railings, bars) pick up their neighbours' colors there — ignore their rows.
   - Unknown camera angle: `sightline match <asset> --fit-view` tries 121 angles and matches from
     the best. For round objects (towers, trees) many angles tie; keep `front` if the picture is an
     elevation.
5. **Repeat** until `matches_reference` passes, then add the designer's other wishes as checks,
   show `sheet.png` + `match_0.png`, and take feedback.

6. **`sightline structure <asset>`** once the outline fits. A high IoU only says the outline is
   right; the designer looks at the inside too. The report lists picture regions where the model's
   material class differs (`picture is dark here, the model bright: chest (metal); use a dark part`),
   lights the model lacks, and `SPLITS`: parts the picture divides into several plates. Work through
   FINDINGS largest first, split the parts SPLITS names, and watch `CLASSES` and `DIVISIONS` rise.
   Use `sightline structure refs/picture.png --height M` before building: its region list (ids,
   sizes and positions in meters, left/right pairs, a relief guess, every light) is the parts list
   of the concept. `assets/robot.yaml` was rebuilt this way: classes 60% -> 67%, disagreeing regions
   27 -> 11, agreeing 35 -> 50, at the same IoU.
   - Pictures pasted into chat may have transparent backgrounds: keep the alpha when you copy them
     into `refs/` (save as RGBA PNG), or the background turns black.
   - Picture colours are lit: a white wing seen from below measures dark grey. Ask about paint and
     ignore the colour suggestions for lit pictures.

`assets/lighthouse.yaml` with `assets/refs/lighthouse_concept.png` is a worked example (IoU 0.95).

### Starting from a ready model

`sightline import model.glb` turns a mesh into a descriptive asset (one fitted primitive per mesh
piece, `WORST FITS` first). Render a picture of the original (any renderer; `tools/blender_truth.py`
does it headless in Blender, with an ID map), add it as the asset's reference and refine with
`match` and `structure` like any concept. `tools/structure_bench.py` scores the structure analysis
against models whose parts and depth are known; use it before changing the algorithm.

## 3. Asking questions

Ask once, briefly, with defaults — the designer should be able to answer "ok" to all of them:

> Before I build the lighthouse, five quick questions (my default in brackets):
> 1. The picture shows only the front. Depth and back? [round tower, same from all sides]
> 2. Height? [11.9 m, matching the door at 2.4 m]
> 3. Are the colors flat paint? [yes: picture colors become the base colors; for a lit render:
>    materials by what the parts are, light-to-dark as in the picture]
> 4. Which details must read at gameplay distance? [bands, lantern, dome; railing optional]
> 5. Budget? [under 60 parts; glb for Godot]

Questions worth asking for most assets: real size (or the size of the thing next to it), the
hidden sides, the camera distance in the game (which details matter), the style (low-poly, blocky,
smooth), the polygon/part budget, the target engine.

## 4. Reading the reports

`sightline inspect <asset>` prints these sections (the same facts as `--json`):

| section | use it to |
|---|---|
| `SIZE`, `GROUND`, `MASS` | real dimensions, "x human height", floating/sinking amounts, center of mass, stability margin and the parts that touch the ground (`ground contacts:`) |
| `PARTS` | one row per part: center and size in m, share of volume, what it is attached to, **visible %** per view, lit color |
| `CONTACTS`, `FLOATING` | what touches/overlaps (with shared m³); floating pieces with the gap, side and the nearest anchor |
| `CUTS` | removed volume and visibility of each cut (a door that cuts nothing is a bug) |
| `COLORS`, `CONTRAST` | material shares in iso and WCAG contrast of neighbouring parts |
| `SILHOUETTE` | letter grids per view with meter labels; each part keeps its letter in every view |
| `READABILITY` | what survives at 32 and 16 px in front and iso (icons, far cameras) |
| `WARNINGS` | short sentences with a fix, e.g. `ball floats 0.50 m above block — attach it with on: block.top or move it 0.50 m down` (a diagonal gap gets the whole move, `move it 1.33 m left (-x) and 0.60 m down`) |

How far to trust the numbers:

- **Sizes, contacts, cuts, floating gaps** come from the exact geometry: trust them (gaps are an
  upper bound accurate to one voxel cell, shown as `voxel cell` in `COUNTS`).
- **Visible %** is measured at 128 px on the part's footprint shrunk by one pixel (silhouettes are
  up to half a pixel wider per edge), so a panel set into an opening reads its true share; parts
  behind many thin parts (bars, railings) can still read low; parts under 40 px are measured on a
  finer grid. Parts only on the back read 0% in every report view and trigger an "under 30%
  visible" warning — expected for a back rail; check `view: back` if it matters. Tag a part that is
  meant to be out of sight (an engine inside a cowling, a floor under a carpet) `tags: [hidden]`
  to silence its warning.
- **Stability** uses where the parts really touch the ground (within 2 cm of y = 0, on the exact
  shapes): a round tyre or a 3 cm skid counts at any voxel size. An `unstable` warning names the
  contacts and the side the center of mass is on, e.g. `1.40 m behind (-z) the ground contacts
  (wheel_l, wheel_r)` — add the missing support (a tail skid) there.
- **Lit colors and contrast** are a Lambert proxy without patterns or outlines; they read darker
  than the PNG. Same-material pairs are marked `=`. Use the warning as a hint and the PNG as the
  judge.
- Parts thinner than one voxel cell get unreliable volumes (`volume_share`, center of mass).
- `diff` lists every changed number (often 100+ lines for one size change); read the top block
  (size, ground, mass) and the `+ warning` lines first.

Levels: `sightline level play <level>` prints the bot's critical path step by step, reachability,
softlock regions with the reason the bot cannot leave them, sight lines with their blocker or
tightest clearance, an ASCII map with the path as `*`, the heights, the objects and then the level's
checks. Fix unreachable markers first (the line says which move fails and why, e.g.
`exit to (11,6) needs rise 2.00 m > jump_up 1.00 m`), then softlocks, then the order and sight.

## 5. Taking feedback by part names

Names make feedback local. Every part name appears in `parts.png`, in the report tables and in
the silhouette legend, so the designer can point at it.

| feedback | edit |
|---|---|
| "make the roof steeper" | `roof.height` up (keep `radius`) |
| "the tower is too thin" | `shaft.radius`; children attached with `on:` follow |
| "door a bit bigger" | `door.size`; the cut follows its `on:` anchor |
| "the shield should be on the other arm" | change `on: hand_l.front` to `hand_r.front` |
| "more crenels" | `crenel.repeat.count` |
| "less red" | swap `material:` or set a style palette key |

After an edit re-run `render` + `check` and say what changed in numbers ("roof now 2.4 m tall,
the silhouette at 32 px still keeps it"). Keep the designer's wish as a check if it is a lasting
constraint (`ratio: {a: roof.height, b: total.height, range: [0.25, 0.35], why: "steep roof"}`).

## 6. Modeling tips and pitfalls

- Start every asset with a **ground part** (`align: bottom` at the origin) and attach everything
  else with `on:`; `grounded` and `connected` then hold by construction.
- `on: target.top` puts your **bottom** on it (the default align is the opposite anchor). For a
  part centered on a point use `align: center`.
- `sides: n` prisms and cones put a flat face toward +Z; the anchor box uses the circumradius, so
  a cut on `@[0, -1, 1]` starts slightly in front of the face — make cuts deep enough.
- Rotated parts can dip below the ground (a 45° brace on a short box): the report's `GROUND`
  line and the `sinks ... below the ground` warning catch it.
- `follow: false` keeps a part upright in the world when its target is rotated (a sword in a
  hand whose arm swings).
- Struts, braces and wires between two parts: `between: [lower.top_left, "upper@[-0.6, -1, 0]"]`
  with `size: [w, d]` (box) or `radius` (cylinder) puts the part from one point to the other — no
  angles or lengths to compute, and it follows when either end moves.
- Decals (roundels, emblems, windows painted on a face) must stand about 0.1 render pixel proud of
  the face (5 mm for a 10 m asset at 256 px); flush ones lose the tie to the part defined first.
- Views from below (`orbit:30:-20`) draw no ground slab; `render --no-ground` drops it anywhere.
- `mirror: x` plus `_l`/`_r` names builds both sides from one entry; attach children to the `_l`
  part and mirror them too — their twins land on the `_r` parts.
- Cuts are invisible tools: name them after the hole (`door`, `window`) and check them with
  `visible: {part: door, view: front}`.
- The renderer has no light emission: a lamp is a bright `lamp`-material box and reads as warm
  ochre on shaded faces.
- In level renders the iso camera looks from the south-east: tall walls hide what stands
  north-west of them. Place showcase objects in the open, or show `top.png`.

## 7. Exporting for game engines

```
sightline export knight                     # out/knight/knight.glb, one node per part, decimated
sightline export crate --max-tris 1500      # a triangle budget for the whole asset
sightline export knight --quality high      # finer: high | medium (default) | low | full
sightline export knight --bake-colors       # + COLOR_0 vertex colors with the patterns (full mesh)
sightline export crate --format obj         # out/crate/crate.obj + crate.mtl
sightline export lighthouse --merged        # one node instead of one per part
```

The `.glb` is glTF 2.0 binary in meters with +Y up — the glTF convention, so no axis conversion.
Each solid part (or blend group) is its own node and mesh, named after the part, so engines can
address, hide, animate or swap parts. One material per style material (PBR base color, metallic
from `shine`). The command prints a mesh table, a `DECIMATE` line, a `MANIFOLD` line (surface
nets can leave a few edges shared by four triangles at the concave corners of a cut: engines import
it, Blender's manifold check and 3D printing flag it; decimation never makes it worse) and
`glTF structure OK` after validating the file.

- **Godot 4:** drop the `.glb` into the project; it imports as a scene with one `MeshInstance3D`
  per part. Use `--bake-colors` and a material that multiplies vertex color to keep the patterns.
- **Unity:** import with a glTF importer package (glTFast or UniGLTF); parts become child
  GameObjects. Unity is left-handed and the importer converts the axes; the asset's front is +Z in
  Sightline (glTF's forward) — check once that it faces where your game expects.
- **Blender:** File → Import → glTF 2.0; one object per part, named as in the YAML, 1 unit = 1 m.

`--max-cells` (default 128) sets the mesh resolution along the longest side; thin parts are meshed
finer automatically. The meshes are then decimated part by part: flat faces collapse to a few
triangles, curved parts keep what their shape needs, every part stays closed and keeps its name,
materials and bounding box (within 5 mm), and sharp edges keep a thin strip of triangles so they
shade crisp. At the default `--quality medium` the crate drops from about 400 000 to about 3 000
triangles (80 KB `.glb`), the knight from 71 000 to 2 400. Read the table before you hand over:

```
PART           VERTS     TRIS   BEFORE    CELL  DEV_MM    VOLUME
body              56      108   172800   0.007    0.96     0.636
brace_front      429      854     9924   0.007    1.79     0.003
...
DECIMATE  399792 -> 2950 triangles (135.5x fewer); largest distance from the SDF surface 2.29 mm (tolerance 1.67-1.80 mm, surface nets alone 2.29 mm)
```

- `TRIS` is what the file holds, `BEFORE` what surface nets made; `DEV_MM` is measured (distance of
  the vertices and triangle centers from the true surface). When it equals the "surface nets alone"
  number, the error was already in the meshing (sawtoothed edges), not added by decimation.
- A part that keeps many triangles for its shape (like the crate's 45° braces, whose sawtoothed
  edges cannot be straightened within the tolerance) is a hint: `--quality low`, a `--max-tris`
  budget, or a lower `--max-cells` all help.
- The designer gives a budget ("under 2k triangles")? Use `--max-tris 2000`: the tolerance relaxes
  until the asset fits, and the `DECIMATE` line shows the tolerance it needed.
- `--bake-colors` exports the full mesh (vertex colors need the dense vertices), so it is heavy;
  prefer plain materials for game props.

## 8. Before you hand over

- `sightline check` exits 0 for the asset (and `sightline level check` for levels);
- you opened `sheet.png` (and `match_0.png`, the level `iso.png`/`top.png`) and they look right;
- the `WARNINGS` that remain are explained (back parts, a proxy contrast) or fixed;
- part names say what parts are, and the brief is in the file;
- you exported what the designer needs and listed the paths.
