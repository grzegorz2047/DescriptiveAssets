# Sightline format reference

Everything Sightline reads is YAML: styles (`styles/*.yaml`), assets (`assets/*.yaml`) and levels
(`levels/*.yaml`), plus reference pictures (PNG/JPEG) named from assets. This file is the complete
reference; `docs/AI_GUIDE.md` explains how to work with it, and `sightline shapes` / `sightline
styles` print the most used tables in the terminal.

Contents: [1 Conventions](#1-conventions) · [2 Styles](#2-styles) · [3 Assets](#3-assets) ·
[4 Checks](#4-asset-checks) · [5 References](#5-reference-pictures) · [6 Levels](#6-levels) ·
[7 Outputs](#7-outputs-and-the-command-line) · [8 Errors](#8-errors)

## 1. Conventions

- **Units** are meters, **angles** degrees.
- **World axes:** right-handed, **+Y up**, **+X right (east)**, **+Z front** (toward the camera of
  the `front` view, south). The ground is y = 0. Assets stand on the ground, centered at x = z = 0.
- **Left and right** always follow the axes: `left` = −X, `right` = +X, as seen in the `front`
  view. A part named `arm_l` sits on −X, so for a character facing the front it is the character's
  *right* arm. Pick one convention per asset and say it in a comment.
- **Views** (orthographic): `front` (looks toward −Z), `back`, `left` (camera on −X), `right`
  (camera on +X), `top` (image up = −Z, the back), `iso` (azimuth 45° from +Z toward +X, elevation
  30°: shows the front, the right side and the top), and free views `orbit:<azimuth>:<elevation>`,
  e.g. `orbit:30:15`. One render shares one meters-per-pixel scale across its views.
- **YAML:** booleans are only `true`/`false` (YAML 1.2), so `on:`, `yes`, `no`, `off` stay text.
  Duplicate keys are errors. Unknown keys are errors with a did-you-mean hint.
- **Names** of parts, markers and doors: `^[a-z][a-z0-9_]*$`. A file's `asset:` / `level:` /
  `style:` must equal its file name without `.yaml`.
- **Project root:** the folder with `styles/`, `assets/`, `levels/`. Found from `--root`; else the
  nearest ancestor of the asset or level file with a `styles/` folder; else `$SIGHTLINE_ROOT`; else
  the nearest ancestor of the current directory with `styles/`; else the repository next to the
  installed package.
- **Reference human:** 1.8 m tall. Reports give heights as multiples of it, renders show it as a
  grey figure (body capsule plus head) to the right of and behind the asset.

## 2. Styles

`styles/<name>.yaml`. Every section is optional: missing keys fall back **key by key** to
`styles/default.yaml` (then to built-in values). A style therefore always offers every default
material, and can recolor the whole library just by overriding palette keys.

```yaml
style: dusk                    # must equal the file name
palette:                       # named colors, '#rrggbb'
  oak: "#8a5a2b"
materials:                     # name -> material
  wood:  {color: oak, pattern: planks, pattern_color: "#6f4520", scale: 0.25, axis: y}
  stone: {color: "#8c8c88", pattern: bricks, pattern_color: "#6d6d69", scale: [0.4, 0.2]}
  metal: {color: "#b8bcc2", shine: 0.6}
light:
  direction: [0.45, -1.0, -0.6]   # the direction the light TRAVELS; normalized on load
  ambient: 0.35                   # 0..1
  shadows: true
outline: {width: 1, color: "#1b1b1b"}   # px, 0..16 (0 = no lines)
background: "#dde1e6"
ground: {show: true, color: "#c8ccc2", grid: 1.0}   # grid line spacing in m (0 = none)
render: {res: 256, views: [front, right, top, iso]}  # res 16..4096 px on the long side
```

| material key | meaning |
|---|---|
| `color` | required; `#rrggbb` or a palette name |
| `pattern` | `solid` (default), `noise`, `stripes`, `planks`, `bricks`, `checker` |
| `pattern_color` | second color of the pattern; default: 80% of `color` |
| `scale` | meters, number or `[a, b]` (bricks: `[brick width, brick height]`; checker: `[x/z cell, y cell]`); default 0.25 |
| `axis` | `x`, `y` or `z`: the direction stripes and planks run along (default `y`) |
| `shine` | 0..1 specular highlight (metals, glass, water) |

Patterns live in **part-local** coordinates, so they turn with their part. Bricks become paving
on upward faces; planks get staggered joints and a tint per board; patterns finer than about 3 px
fade to their mean color instead of flickering. There is no emission: a lamp is a bright color.

**Default materials** (`styles/default.yaml`, 44):

| material | color | pattern | shine |
|---|---|---|---|
| `wood` | #9a6a3c | planks #7d5330, scale 0.25, along y | 0 |
| `dark_wood` | #5c3b24 | planks #462c1a, scale 0.25, along y | 0 |
| `planks` | #c08f57 | planks #8f6538, scale 0.3, along x | 0 |
| `bark` | #6b4a32 | stripes #4e3423, scale 0.12, along y | 0 |
| `stone` | #8e8b84 | bricks #6c6a64, scale [0.6, 0.35] | 0 |
| `brick` | #a4523a | bricks #d2c7b5, scale [0.4, 0.2] | 0 |
| `plaster` | #e8dcc4 | noise #d9cbb0, scale 0.3 | 0 |
| `tiles` | #e8dcc4 | checker #8e8b84, scale 0.5 | 0 |
| `roof` | #b4462f | stripes #8b3322, scale 0.25, along x | 0 |
| `slate_roof` | #4f5866 | stripes #3c434e, scale 0.2, along x | 0 |
| `thatch` | #d2ae62 | stripes #a9863f, scale 0.1, along y | 0 |
| `metal` | #b6bbc2 | solid | 0.6 |
| `iron` | #4b4f55 | solid | 0.4 |
| `gold` | #e0b032 | solid | 0.8 |
| `bronze` | #ad7a3c | solid | 0.5 |
| `cloth_red` | #b43a3a | noise #972f2f, scale 0.08 | 0 |
| `cloth_blue` | #2f5fa8 | noise #264f8d, scale 0.08 | 0 |
| `cloth_green` | #3f7a3a | noise #33652f, scale 0.08 | 0 |
| `cloth_yellow` | #d0a23a | noise #a9863f, scale 0.08 | 0 |
| `leather` | #7b4a2c | noise #663c23, scale 0.05 | 0 |
| `skin` | #e2ae8a | solid | 0.1 |
| `hair` | #4a3020 | stripes #36231a, scale 0.04, along y | 0 |
| `skin_orc` | #86903f | solid (olive orc skin) | 0.1 |
| `hair_black` | #2a2523 | stripes #161312, scale 0.04, along y | 0 |
| `bone` | #e9e0c9 | solid | 0.1 |
| `rope` | #bb9b67 | stripes #9a7d4f, scale 0.04, along y | 0 |
| `leaf` | #55913b | noise #41752d, scale 0.15 | 0 |
| `moss` | #6f8f3a | noise #41752d, scale 0.1 | 0 |
| `grass` | #6aa042 | noise #58893a, scale 0.2 | 0 |
| `dirt` | #8a6a48 | noise #735637, scale 0.15 | 0 |
| `sand` | #dcc58e | noise #c9b07a, scale 0.15 | 0 |
| `snow` | #f4f6f8 | solid | 0.2 |
| `water` | #3f80c0 | solid | 0.7 |
| `ice` | #cfe8f2 | solid | 0.8 |
| `lava` | #e8662a | noise #a83a1c, scale 0.3 | 0 |
| `glass` | #a9d6e8 | solid | 0.9 |
| `paint_white` | #f1efe9 | solid | 0.15 |
| `paint_black` | #27272b | solid | 0.15 |
| `paint_red` | #c23b30 | solid | 0.15 |
| `paint_blue` | #2f5fa8 | solid | 0.15 |
| `canvas` | #e4dcc6 | solid (doped linen of early aircraft) | 0.1 |
| `lamp` | #ffe08a | solid | 0.3 |
| `glow_blue` | #5cc8ff | solid (a light: no emission, a bright color) | 0.3 |
| `helper` | #9a9da3 | solid (the scale figure; not for assets) | 0 |

Stripes and planks at 0.04 m (hair, rope) fade to a flat color above about 0.015 m/px, i.e. on
anything bigger than a character. The palette has 59 named colors; `sightline styles` lists them.

## 3. Assets

`assets/<name>.yaml`. A complete example is `assets/watchtower.yaml`; every example in
`assets/` passes its own checks.

```yaml
asset: watchtower              # required, equals the file name
brief: >                       # the designer's request, verbatim (shown on sheets and reports)
  A wooden watchtower on a stone base, readable from far away.
style: default                 # optional, default "default"
tags: [building]               # optional free text
parts: [...]                   # required, ordered; a part may only refer to EARLIER parts
reference: {...}               # optional, one entry or a list (section 5)
checks: [...]                  # optional (section 4)
```

### 3.1 Part keys

| key | meaning |
|---|---|
| `name` | required, unique after expansion |
| `shape` | required, see 3.2; plus the shape's parameters |
| `material` | required for solids, forbidden for cuts |
| `at` | `[x, y, z]` world point for the part's `align` anchor (default `[0, 0, 0]`) |
| `on` | `part.anchor` or `part@[u, v, w]`: attach to a point of an earlier part (an anchor is required: `on: base` is an error, write `base.top`) |
| `between` | `[A, B]`, each a `part.anchor`, `part@[u, v, w]` or `[x, y, z]`: a strut or wire whose local +Y runs from A to B, centered on the midpoint. The length is the distance, so leave out `height` (cylinder, cone, capsule) and write `size: [width, depth]` (box, ellipsoid, wedge). Not with `at`/`on`/`align`/`follow`; `rotate` spins the part in its own frame first (e.g. `[0, 20, 0]` turns a box strut about its axis); `offset` is in the world frame |
| `align` | which of the part's own anchors goes to `at`/`on`. Default with `at`: `center`; with `on`: the **opposite** of the target anchor (`top` → `bottom`, `top_front` → `bottom_back`, `@[0,1,1]` → `@[0,-1,-1]`) |
| `offset` | `[dx, dy, dz]` added after attaching; in the target's frame when following |
| `follow` | with `on`: `true` (default) inherits the target's rotation (world matrix = target @ own); `false` keeps the world frame |
| `rotate` | `[rx, ry, rz]` degrees, R = Rz·Ry·Rx (X first), about the part's own bbox center |
| `repeat` | linear `{count: n, step: [dx, dy, dz]}` or radial `{count: n, axis: x\|y\|z, center: [x,y,z] \| part.anchor}` |
| `mirror` | `x`, `y` or `z`: adds a reflected twin across the world plane through the origin |
| `mirror_name` | name of the twin (default: swap `_l`↔`_r`, `_left`↔`_right`, else `<name>_mirror`) |
| `op` | `union` (default) or `cut` |
| `from` | cuts only: `[part, ...]` or `all` (all earlier solids). Default: the `on` target. A repeated base name (`from: [window]`) means all its copies |
| `blend` | meters: smooth union with the `on` target (or `blend_with`) |
| `blend_with` | the part to blend with when it is not the `on` target |
| `tags` | free text, e.g. `[door]`, `[walkable]`. `hidden` marks a part or cut that is meant to be out of sight (an interior or underside part): no low-visibility warning for it |

A strut between two wings, without trigonometry:

```yaml
  - name: strut_l
    shape: box
    size: [0.09, 0.14]               # width, depth: the length comes from the two points
    material: wood
    between: ["wing_lower@[-0.66, 1, 0.5]", "wing_upper@[-0.53, -1, 0.5]"]
    mirror: x
```

(Quote an `@[...]` anchor inside a flow list or mapping.)

### 3.2 Shapes

Each shape has a local frame centered on its local bounding box, with Y as its main axis.

| shape | parameters | local bbox (w, h, d) |
|---|---|---|
| `box` | `size: [w, h, d]`, `round: r` (edge radius ≤ min/2) | w, h, d |
| `sphere` | `radius` | 2r, 2r, 2r |
| `ellipsoid` | `size: [w, h, d]` (full extents) | w, h, d |
| `cylinder` | `radius`, `height`, `sides: n` (≥ 3, a prism), `round: r` | 2r, h, 2r |
| `cone` | `radius` (bottom, > 0), `height`, `top_radius` (> 0: frustum), `sides: n` | 2·max(r, top), h, 2·max(r, top) |
| `capsule` | `radius`, `height` (total, ≥ 2r) | 2r, h, 2r |
| `torus` | `radius` (ring center), `thickness` (tube radius); the ring lies in local XZ | 2(R+t), 2t, 2(R+t) |
| `wedge` | `size: [w, h, d]`: triangular prism along X, full height at the back (−Z), sloping to 0 at the front edge (+Z) | w, h, d |

- `sides: n` makes a regular n-gon with a **flat face toward +Z**; `radius` is the circumradius, so
  the flat face is at `radius · cos(180°/n)`. A door cut on `@[0, -1, 1]` of an 8-sided tower
  therefore starts 0.076·r in front of the face; make the cut deep enough.
- `sides: 4` cones are square pyramids whose faces point along ±X and ±Z (half-width
  `radius · 0.707`) — the usual hip roof.
- An upside-down cone: `rotate: [180, 0, 0]`, or a frustum with `top_radius > radius`.
- The anchor box is the *bbox*, not the visible surface: a `sides: 3` prism's flat edge is at half
  the circumradius, so place it with an explicit `offset` (see `shield_tip` in `assets/knight.yaml`).

### 3.3 Anchors

Anchors are points of a part's local bounding box moved with the part:
`center`, `top` (+Y), `bottom` (−Y), `right` (+X), `left` (−X), `front` (+Z), `back` (−Z), and
any combination with at most one word per axis joined by `_` in any order (`top_front_left`).
`@[u, v, w]` is normalized: each component −1..1 maps the bbox low..high, so `@[0, -1, 1]` is the
bottom center of the front face and `@[0, 0.45, 0]` is 45% of the half-height above the center.

### 3.4 Expansion: repeat, then mirror

- Linear `repeat` translates copy *i* by *i*·step (in the target frame when following).
- Radial `repeat` rotates copy *i* by *i*·360/n about the axis through `center` (default the
  origin); under `follow` the axis is in the target's frame.
- Copies are `name_0 … name_{n-1}`; the bare name no longer exists (except in a cut's `from:`).
- `mirror` reflects position and orientation (matrix det −1 is fine). Twins of attached parts
  attach to the twins of their targets: `hand_l` on `arm_l` with `mirror: x` gives `hand_r` on
  `arm_r`. Each repeated copy gets its own twin. `from: all` is not remapped.

### 3.5 Operations

- **Union** is the default. Parts may overlap; the report lists overlaps with their volume.
- **Blend** (`blend: k`) joins the part to its target with a smooth fillet of radius k; all parts
  joined by blends form one evaluation group (one mesh on export, named after its first member).
  Inside a group each member's cuts are applied before the smooth union.
- **Cut** (`op: cut`) subtracts the part from every part in `from` (default the `on` target). Cuts
  are never drawn and have no material. A cut may target only earlier solids.

### 3.6 Validation

Errors (with file, YAML path, line and a hint): unknown keys, unknown shape, missing/zero/negative
sizes, unknown material (did-you-mean over the style's materials), forward or unknown references,
unknown anchors, duplicate names after expansion, `round` too large, capsule height < 2r,
`count` < 1, more than 256 parts after expansion, a cut with a material, a solid without one,
`asset:` not equal to the file name.

## 4. Asset checks

`checks:` turns the designer's wishes into tests. An entry is a bare kind (`- grounded`) or a
one-key mapping. Any entry may carry `why: "..."` — inside its mapping or as a sibling key — which
is printed on failure. All entries are validated before anything runs; unknown kinds, keys, parts,
views and dimensions are errors. Each result line reads
`PASS|FAIL  kind  measured …  expected …  — what to do  (why: …)`.

| check | arguments (defaults) | passes when |
|---|---|---|
| `grounded` | `tolerance` (max(0.02 m, half a voxel cell)) | lowest point within tolerance of y = 0 |
| `connected` | – | no floating pieces |
| `stable` | – | the center of mass is over the support polygon: the hull of where parts touch the ground (material within 2 cm of y = 0, found on the exact shapes, so round tyres and a thin skid count); the report names the contact parts |
| `size` | `width`/`height`/`depth`: `v` (±2%) or `[lo, hi]` | asset AABB in range |
| `part_size` | `part`, then dims as `size` | that part's world AABB in range (cuts allowed) |
| `symmetric` | `axis` (`x`; `x` or `z`), `min` (0.9) | occupancy IoU with its mirror image |
| `visible` | `part`, `view` (`any` = best view), `min` (0.5) | share of the part's own pixels seen; works for cuts (`visible: {part: door}`) |
| `ratio` | `a`, `b`: `part.dim` or `total.dim`; `range: [lo, hi]` (a scalar means ±2%) | a / b in range; `total` is the whole asset |
| `relation` | `a`, `is`, `b`; `is` ∈ above, below, left_of, right_of, in_front_of, behind, touching, inside | AABB centers compared strictly; `touching` = a report contact; `inside` = AABB containment ± one cell |
| `contrast` | `a`, `b`, `min` (1.5) | WCAG contrast of the two parts' lit colors in iso (any pair, not only neighbours) |
| `readable_at` | `px` (32), `min_iou` (0.8), `keep` ([]), `view` (front) | the silhouette at N px keeps its shape and the listed parts (the report reads front and iso) |
| `distinct_from` | `assets` (list or one name), `view` (front), `max_iou` (0.7) | normalized silhouette IoU with each other asset stays below |
| `max_parts` | `n` or `{max: n}` | number of solid parts |
| `max_materials` | `n` or `{max: n}` | number of materials used |
| `matches_reference` | `index` (0), `min_iou` (0.85), `max_missing` (0.05), `res` (256) | `sightline match` IoU and missing share at that render size (section 5) |

Notes on the numbers behind the checks:

- **Lit colors** in `contrast` are a Lambert proxy: `color × (ambient + (1 − ambient) × max(0, n·−light))`
  averaged over the part's pixels in iso. They ignore patterns, shadows and outlines, so they read
  darker than the PNGs and dark pairs (iron on oak) score low even when the render reads well.
- **Visible fractions** are measured at the report resolution (128 px), on the part's own
  footprint shrunk by one pixel (every silhouette is up to half a pixel wider per edge, so without
  that a door panel set into its door cut would read low on a big building). Parts covering fewer
  than 40 px are measured again on a 4× finer grid, so a small part behind a propeller blade reads
  its real share. Parts behind many thin parts (bars in front of glass) can still read lower than in
  the 256 px PNGs. A failing `visible` check names the part in front.
- **Decals** (discs or plates on a face) need to stand at least ~0.1 render pixel proud of it (5 mm
  for a 10 m asset at 256 px, 2 cm at 64 px); closer than that the part defined first wins the tie.
- **readable_at** box-filters a 256 px silhouette down to N px; a part is kept when it still owns
  at least 4 low-resolution pixels.
- A part only seen from the back is "hidden" in every report view (front, right, top, iso);
  checks on back parts should name `view: back` or `left`.

## 5. Reference pictures

```yaml
reference:                      # one mapping or a list of them
  image: refs/lighthouse_concept.png   # relative to the asset file's folder, then the root, then root/assets
  view: front                   # a view name or {azimuth: 30, elevation: 15}
  height: 11.85                 # optional: real height (m) of the object in the picture
  mask: auto                    # auto, or a black/white PNG (white = object)
  lighting: auto                # auto (guessed), lit (a shaded render or photo) or flat (flat paint)
```

- **Segmentation** (`mask: auto`): PNG alpha when present; otherwise the background is the median
  color of a 3-px border and the object is everything farther than the tolerance (max(24, 3 × the
  border's spread)). Specks under 0.5% of the image are dropped. Regions of the object painted in
  the background color become holes, so concept art needs a background that differs from every
  object color (the lighthouse art uses a flat sky blue because its white bands are near-white).
- **Confidence** high / medium / low from the background's color spread, how much of the
  top/left/right border the object touches and how much of the image it covers. Low confidence
  becomes a question: "send a cut-out PNG or a plain background".
- **Scale**: `height` gives meters per pixel; without it the matcher fits the picture to the
  model's height. For iso/oblique views the picture height includes the top, and matching corrects
  for it.
- Images over 1600 px are reduced by an integer factor first.
- `sightline ref <image>` prints mask, scale, proportions, a silhouette grid (≤ 32 × 48 cells,
  cell centers labelled), a 10-band width profile, up to 6 dominant colors with the nearest style
  material, details thinner than 3 px, a `LIGHTING` line, and the **unknowns**: questions the
  picture cannot answer, each `{topic, question, default}`.
- **Lighting**: a picture is *lit* (a shaded render or a photo) when its colours spread around their
  clusters and change gently from pixel to pixel; *flat* paint keeps each region one colour. In a lit
  picture the measured colours are the shading, not the paint (white wings can measure #555a58), so
  the materials question says so and its default is to choose materials by what the parts are.
  `lighting: lit|flat` in the reference (or `ref --lighting`) records the designer's answer.
- `sightline match <asset>` renders the model from the reference view, aligns the picture's bottom
  on the ground line y = 0 in front/side views (on the model's bottom when it floats; parts below
  the ground are listed and ignored) and the best horizontal shift (`ALIGN` line), and reports IoU,
  `missing` (share of the picture not covered by the model), `extra`, a `SIZE` line (the model's
  extent vs the picture's outline at its scale), a width profile per band with a verdict, regions
  ≥ 1% of the picture (the 1-px fringe along the other outline is not counted, so a region does not
  spread along the whole silhouette) with their size in meters, nearest part and a hint ("too big on
  the top by ~X m" = the region's largest extent normal to that side), a per-part color table, a
  diff grid (`#` both, `+` model only, `-` picture only) and `out/<asset>/match_<i>.png`.
- **Match numbers**: IoU, missing and extra are area shares of the model's exact coverage (sampled
  on its silhouette edge, without the renderer's half-pixel widening) against the picture's
  coverage, so thin wings and wires count with their real width and the numbers hold within about
  ±0.01 from res 192 to 1024. Regions made of strokes up to 4 grid px wide are **thin details**
  (wires, rods: "thin detail missing … no part needs resizing") and come after the shape errors. A
  width verdict must survive moving the band borders by max(1.5 px, 2% of the height), else it reads
  `ok within … (an edge runs along the band border)`. `suggested` in the color table is the
  picture's colour for flat paint, and base × reference / rendered divided by the picture's overall
  light for lit pictures (only parts that differ from the rest change).

### 5.1 Picture structure (`sightline structure`)

`match` compares outlines. Inside the outline a concept shows how the object is divided: plates, a
dark frame between them, lights, trims. `sightline structure` reads those divisions from the picture
alone and, for an asset, says where the model's parts and materials agree with them.

- **Edges**: the Lab colour gradient (lightness plus chroma) of a lightly smoothed picture, taken
  relative to the local lightness (a step in a dark area counts more, as for the eye), thinned and
  kept by hysteresis between the 70% and 88% quantiles of the object's own gradients. The outline is
  not an edge.
- **Regions**: connected areas of one class (dark / bright / accent) cut by the edges; a boundary
  with no edge on it is shading, so the regions on both sides join. Regions under 0.15% of the object
  join the neighbour they share the most boundary with; lights and decals may stay small.
- **Classes**: dark/bright split by Otsu on lightness inside the object (fixed L 30 when the
  picture is not two-toned); **accent** = saturated (chroma >= 24), for lights, decals and gems.
- **Relief guess** (not measured depth): a bright region framed by dark seams or strong edges is a
  `raised plate`, a dark region is `recessed (frame or gap)`, an accent is a `light or decal`, else
  `surface`. `shading` says where a region gets lighter, a hint of curvature.
- Regions get ids (`R1` largest), size and position in meters (with `--height`; else % of the
  height) and a left/right `pair` when a region of the same class mirrors it.

`sightline structure <asset>` aligns the picture with the model like `match` (default `--res 512`)
and reports:

```
CLASSES    dark/bright/accent agree on 67% of the shared silhouette
DIVISIONS  44% of the picture's inner edges have a model part boundary within 2 px; 78% of the model's part boundaries lie on a picture edge
FINDINGS   11 regions disagree with the model, 50 agree
  R18  dark  0.13 x 0.23 m at x -0.06..+0.08, y 1.41..1.64: picture is dark here, the model bright: chest (metal); use a dark part (frame, gap, joint) here
  R97  accent blue  0.03 x 0.02 m at x -0.05..-0.02, y 1.77..1.79: light or decal missing: the model shows visor (iron) here; add a small blue part
SPLITS     model parts that the picture divides into several elements (split them, or add plates on them):
  chest (metal): 3 picture regions R33, R43, R50 (bright)
```

and writes `out/<asset>/structure_<i>.png` (picture regions that disagree | the model with its part
boundaries | class agreement and shared / picture-only / model-only edges).

The three classes suit mechanical designs (plates, frame, lights). On multi-coloured costumes (red
cloth, gold trim, skin) large saturated areas count as accents and the agreement figure is low.

### 5.2 Importing a ready model (`sightline import`)

`sightline import model.glb [--name N] [--max-parts 120] [--no-split]` writes `assets/<name>.yaml`:
one part per mesh primitive, split into its connected pieces (vertices welded by position), the
largest `--max-parts` kept. Each piece becomes an oriented primitive: the principal axes of its
area-weighted vertices (snapped to world axes within 4 degrees), the shape chosen by how much of that
box a closed piece fills (box ~1, cylinder ~0.79, ellipsoid ~0.52, cone ~0.26; open meshes stay boxes),
the material nearest to the base colour (or to the mean of its texture under its UVs). Skinned meshes
are read in their default node pose. The model is moved so it stands on y = 0, centred in x and z.
Every part carries a comment with its source, fill ratio, fit error (RMS distance of the piece's
vertices from the part's surface) and colour; `WORST FITS` lists the parts to refine first. The file
is a starting point: render it next to a picture of the original and refine with `match` and
`structure`.

## 6. Levels

`levels/<name>.yaml`: a grid of columns of terrain, read by a navigation bot. The two examples are
`levels/courtyard.yaml` (all checks pass) and `levels/broken_bridge.yaml` (built to fail three).

```yaml
level: courtyard               # equals the file name
brief: ...
style: default                 # tile and object colors
cell: 1.0                      # meters per cell in x and z (default 1.0)
height_unit: 0.5               # meters per step in `heights` (default 0.5)
player: {step: 0.5, jump_up: 1.0, jump_gap: 2, max_drop: 3.0, height: 1.8}   # these are the defaults
heights: |                     # rows = z (row 0 = north/back), cols = x (col 0 = west); integers
  2 2 2 2                      # trailing `# comments` are allowed; optional (all 0)
  2 0 0 2
tiles: |                       # same shape, one character per cell, space-separated; required
  # # # #
  # . k #
legend: {"%": floor}           # optional: extra or overriding char -> kind
markers: {spawn: [1, 1], goal: [2, 1], key_red: [2, 1]}   # name -> [col, row]
doors: {door_red: {at: [3, 1], key: key_red}}             # name -> {at, key (optional)}
objects:
  - {asset: watchtower, at: [6, 2], rotate: 0, solid: true}
checks: [...]
```

**Tiles** (default legend): `.` floor, `#` wall (blocks movement and sight; its top is its
`heights` value), `~` water (deadly, can be jumped over), `^` spikes (deadly, can be jumped over),
`_` pit (void, can be jumped over), `=` ladder (moving onto or off it ignores the height
difference), `D` door (standable once its key is held; a `D` not listed in `doors:` needs no key),
`k` key pickup (needs a marker on it for its name). Cell (c, r) has its center at
x = (c + 0.5)·cell, z = (r + 0.5)·cell, y = heights[r][c]·height_unit.

**Markers:** `spawn` is required; `goal` is needed by `no_softlocks`, `path_length`, `jumps` and
`max_fall`. Keys are the markers referenced by a door or standing on a `k` tile (at most 8). Doors
may not share a name with a marker.

**Objects:** `asset` (a name under `assets/`), `at: [col, row]`, `rotate` (one yaw angle in degrees,
the same as `rotate: [0, deg, 0]`), `solid` (default true). The asset's origin goes on the anchor
cell's center at terrain height. Its footprint is every cell whose center lies inside (or on the
edge of) the rotated asset AABB in x/z — a 2 × 2 m object covers 3 × 3 cells — and that holds some
of its geometry (from the asset's voxel occupancy); its height is the AABB top. A solid object
blocks walking and jumping on the footprint cells where its geometry comes lower than
`player.height + jump_up` above the ground; cells under an overhang only (windmill sails, a
bridge) stay walkable and are listed as "walkable under overhangs". Sight is blocked only where a
sight line passes through the object's geometry over a cell. Solid objects may not cover markers
or doors.
Cells hanging off the grid are dropped with a warning. A missing asset becomes a 1-cell, 1 m
placeholder box with a warning (an error with did-you-mean when a close name exists).

**Movement** (4 directions; cost = horizontal + vertical meters):

| move | when |
|---|---|
| walk | to a standable neighbour with rise ≤ `step` and drop ≤ `max_drop` |
| hop | onto an adjacent ledge with `step` < rise ≤ `jump_up` (counts as a jump) |
| jump | straight over 1..`jump_gap` non-standable cells (water, spikes, pit) to the first standable cell, rise ≤ `jump_up`, drop ≤ `max_drop`; walls only if their top ≤ takeoff + `jump_up`; never over solid objects |
| ladder | onto or off a ladder cell, any height difference |

**Line of sight:** eyes at `player.height − 0.1` m above each cell's terrain, sampled every
cell/4; blocked by walls, terrain higher than the sight line and solid objects' geometry at the
line's height (not their bounding box, so a line passes under sails). Doors do not block
sight. Limits: grid ≤ 256 × 256, cols × rows × 2^keys ≤ 2^19 bot states.

**Level checks:**

| check | forms |
|---|---|
| `reachable` | `reachable: goal` or `{to: name \| [c, r], from: spawn}` |
| `order` | `[key_red, door_red, goal]`: each item must be reached before the next is possible; fails also when an item can be bypassed |
| `no_softlocks` | no reachable state from which the goal is unreachable |
| `visible` / `not_visible` | `{from: spawn, to: goal}` (names or `[c, r]`) |
| `path_length` | `N` (= max) or `{min, max}` meters along the critical path |
| `jumps` | `N` (= exactly) or `{min, max}` jumps (hops count) |
| `max_fall` | `3.0` or `{max: 3.0}`: largest drop on the critical path |

`why:` works as for assets. Path segments read like `walk E 4 cells (1,1)→(5,1)`,
`jump E over water 2 cells (2,4)→(5,4)`, `jump up +1.00 m E`, `climb +1.50 m (ladder) E`,
`drop -1.50 m S`, `pick key_red (13,2)`, `open door_red with key_red (13,6)`, `reach goal (13,8)`.

The iso render draws objects with the asset renders (sprites), the critical path in magenta
(dashed where hidden behind nearer walls or objects), softlock cells in red and never-reached cells
in purple. The fixed iso camera looks from the south-east, so faces toward north and west and
anything behind tall walls are hidden: `top.png` shows everything.

## 7. Outputs and the command line

| command | writes / prints |
|---|---|
| `sightline new <name> [--style S] [--brief TEXT] [--force]` | `assets/<name>.yaml` template (a ground part, commented examples of on/align/repeat/mirror/cut, the brief's sentences as TODO checks) |
| `sightline render <asset> [--views a,b] [--res N] [--out DIR] [--no-human] [--no-ground]` | `out/<asset>/<view>.png` (orbit views as `orbit_30_15.png`), `parts.png`, `sheet.png`; views from below (`orbit:30:-20`) draw no ground slab |
| `sightline inspect <asset> [--json] [--res N] [--compare a,b]` | the measured report |
| `sightline check [<asset>...] [--json]` | PASS/FAIL per check; no names = every asset |
| `sightline diff <asset> [--against REV\|path]` | report changes since git HEAD (or a revision or an older copy) |
| `sightline sheet [<asset>...] [--out PATH] [--res N] [--view V]` | `out/library.png`: assets side by side at one scale with the human |
| `sightline export <asset> [--format glb\|obj] [--merged] [--bake-colors] [--max-cells N] [--quality full\|high\|medium\|low] [--max-tris N] [--out PATH]` | `out/<asset>/<asset>.glb` (or `.obj` + `.mtl`) and a mesh table with triangles before and after decimation |
| `sightline ref <image> [--view V] [--height M] [--mask PATH] [--lighting auto\|lit\|flat] [--json]` | the reference analysis |
| `sightline match <asset> [--index I] [--fit-view] [--res N] [--out DIR] [--json]` | the comparison and `out/<asset>/match_<i>.png` |
| `sightline structure <asset>\|<image> [--index I] [--res N] [--height M] [--out DIR] [--json]` | divisions inside the picture; with an asset, how its parts line up: `out/<asset>/structure_<i>.png` |
| `sightline import <file.glb> [--name N] [--max-parts N] [--no-split] [--out PATH] [--force]` | `assets/<name>.yaml`: one fitted primitive per mesh piece |
| `sightline level render\|play\|check <level> [--json] [--out DIR]` | `out/levels/<level>/iso.png`, `top.png`; the bot report; the checks |
| `sightline styles [name]`, `sightline shapes` | materials and colors; shape, anchor and placement reference |

Every command accepts `--root DIR` and prints the absolute paths of the files it wrote
(`wrote E:/…/iso.png`). Exit codes: **0** ok, **1** a check failed (`check`, `level play`,
`level check`), **2** an error (invalid file, unknown name, usage).

**Captions** under every render read `<asset> · <view> · 1 px = 0.0094 m · grid 1 m`, where the
pixel is the saved PNG's pixel (small views are upscaled ×2, which halves the stated value).

**Export:** one glTF node and mesh per solid part or blend group, named after the part
(blend groups after their first member), Y up, meters. One material per style material:
`baseColorFactor` = sRGB → linear of the color, `metallic` = shine × 0.8, `roughness` = 1 −
shine × 0.6. `--merged` joins the part meshes into one node (it is not a boolean union: faces
inside overlaps stay). `--bake-colors` adds `COLOR_0` with the material patterns, using
baseColorFactor × COLOR_0 = the baked color. Parts thinner than 3 base cells are meshed finer
(down to a quarter of the base cell). OBJ `Kd` is the style's sRGB color as written.

**Decimation:** every part mesh is then reduced on its own by quadric edge collapse; parts stay
closed, keep their names, materials and topology (holes stay holes), their bounding box shrinks by at
most 5 mm and never grows, and sharp edges keep a narrow strip of triangles so they still shade
crisp. `--quality` sets how far the surface may move, in cells of the part: `high` 1/8, `medium`
1/4 (the default), `low` 1/2; `full` keeps the surface-nets mesh as meshed (the default with
`--bake-colors`, which needs the dense vertices and cannot be combined with another quality or
`--max-tris`). `--max-tris N` is a budget for the whole asset: if the quality pass leaves more
triangles, the tolerance doubles until the budget is met. Flat faces cost almost nothing, curved
parts keep more triangles; at the defaults the crate goes from about 400 000 to about 3 000
triangles, the knight from 71 000 to 2 400. The mesh table shows `TRIS` (exported), `BEFORE`
(surface nets) and `DEV_MM`, the largest measured distance of the vertices and triangle centers from
the SDF surface; the `DECIMATE` line sums it up and also gives the surface-nets mesh's own distance
(sawtoothed edges can already be off by up to 0.42 cell).

## 8. Errors

Every invalid input is a `SightlineError`, printed as

```
assets/knight.yaml: parts[5] (sword), line 21: unknown anchor 'tip' on part 'arm_r'
  hint: did you mean 'top'? anchors: center, top, bottom, ...
```

`where` names the file (relative to the root), the YAML path and the line; `hint` says what to
write instead. Check errors use `assets/x.yaml: checks[i] (kind)`, level errors
`levels/x.yaml: objects[2] (crate).at`.
