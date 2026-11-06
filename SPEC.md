# Sightline — engineering spec v0.1

Sightline is a text-first 3D asset and level engine. An AI agent writes assets and levels as
YAML, and the engine gives it three things a model cannot do reliably from pixels alone:

1. **Edit locally** — an asset is a list of named parts built from primitives, so feedback such as
   "make the roof steeper" is a change to one field.
2. **Measure** — after every change the engine renders PNGs (for the gestalt) *and* prints a text
   report of measured facts (sizes, contacts, floating parts, visibility, contrast, symmetry,
   stability, silhouettes) that the agent can trust more than its own reading of an image.
3. **Test intent** — the designer's wishes become `checks:` that the engine evaluates
   (PASS/FAIL with the measured value), so the agent iterates until they pass.

Levels follow the same idea; the "sense" is a navigation bot with a movement model that walks the
level and prints what it found (path, jumps, softlocks, line of sight).

This file is the contract between modules. When code and this spec disagree, fix one of them in
the same change and say so.

## 0. Ground rules for contributors (humans and agents)

- Python ≥ 3.11, dependencies: numpy, pillow, pyyaml (pytest for tests). No scipy, no GPU.
- Code, identifiers, comments, CLI output, docs except `README.md`: English. `README.md`: Polish.
- Deterministic: same input → byte-identical PNGs and reports. No unseeded randomness. Noise
  patterns are hash-based on coordinates.
- Every user-facing error is a `sightline.errors.SightlineError` with `where` (file + YAML path,
  e.g. `assets/knight.yaml: parts[3] (sword)`) and a `hint`. Unknown keys are errors, with
  `did_you_mean`. Never a bare traceback for invalid input.
- All numbers in reports: meters with 2 decimals, percentages as integers, angles in degrees.
- Paths: run tools with absolute paths; the venv interpreter is `E:/sightline/.venv/Scripts/python.exe`.

## 1. Conventions

- Units: meters. Right-handed world, **+Y up, +X right (east), +Z front (toward the viewer of the
  front view, south)**. Ground plane is y = 0. Assets stand on the ground, centered at x = z = 0.
- Angles: degrees. `rotate: [rx, ry, rz]` means the matrix `R = Rz @ Ry @ Rx` (X applied first),
  about the part's own local origin (its local bounding-box center).
- Views (`sightline.camera`, already implemented): `front back left right top iso`, all
  orthographic. `iso` = azimuth 45° from +Z toward +X, elevation 30°. `top` has image-up = −Z.
  One render shares one meters-per-pixel scale across all its views.
- Reference human: 1.8 m tall, used for scale in reports and as an optional render helper.

## 2. Repository layout

```
pyproject.toml  SPEC.md  README.md (pl)
src/sightline/
  errors.py      SightlineError, did_you_mean                      (done)
  results.py     CheckResult shared by asset and level checks       (done)
  camera.py      view_basis, Camera, make_cameras, fixed_camera     (done)
  model.py       Material, Style, Part, Group, Scene, Asset dataclasses
  style.py       load_style(name_or_path, root) -> Style
  loader.py      load_asset(path_or_name, root) -> Asset
  sdf.py         primitive SDFs, smooth min, Scene.eval helpers
  march.py       trace(...) and march(scene, camera) -> ViewBuffers
  voxel.py       occupancy(scene) -> VoxelGrid
  render.py      shading, outlines, ground, scale figure, labels, part map -> PIL images/PNGs
  sheet.py       contact sheets (one asset; many assets at shared scale)
  inspect.py     measurements -> report dict -> text
  checks.py      assertion engine over the report
  export.py      surface nets mesh -> glTF 2.0 (.glb, part per node) and OBJ + MTL
  reference.py   reference image analysis, model-vs-reference matching, view fitting
  levels/        model.py loader.py nav.py sight.py report.py render.py checks.py
  cli.py         `sightline` command
styles/          default.yaml (+ more)
assets/          example assets (*.yaml)
levels/          example levels (*.yaml)
docs/            FORMAT.md (complete format reference), AI_GUIDE.md (how an agent works with it)
tests/           pytest, one file per module (test_loader.py, test_march.py, ...)
out/             generated renders (gitignored)
```

Project root (`root`): the directory containing `styles/`, `assets/`, `levels/`. CLI resolves it
from `--root`, else env `SIGHTLINE_ROOT`, else the nearest ancestor of the CWD containing
`styles/`, else the repo root next to the installed package.

## 3. Style files (`styles/<name>.yaml`)

```yaml
style: default                 # must equal the file stem
palette:                       # named colors
  oak: "#8a5a2b"
  slate: "#5b6470"
materials:                     # key = material name used by parts
  wood:  {color: oak, pattern: planks, pattern_color: "#6f4520", scale: 0.25, axis: y}
  stone: {color: "#8c8c88", pattern: bricks, pattern_color: "#6d6d69", scale: [0.4, 0.2]}
  metal: {color: "#b8bcc2", shine: 0.6}
  cloth: {color: "#a33b3b", pattern: noise, pattern_color: "#8a3030", scale: 0.1}
light:
  direction: [-0.4, -1.0, -0.5]   # direction the light TRAVELS (normalized by the loader)
  ambient: 0.35                   # 0..1
  shadows: true
outline: {width: 1, color: "#1b1b1b"}      # silhouette + part-boundary lines, width in px (0 = off)
background: "#dde1e6"
ground: {show: true, color: "#c8ccc2", grid: 1.0}   # grid line spacing in meters (0 = none)
render: {res: 256, views: [front, right, top, iso]}
```

- `color`/`pattern_color`: hex `#rrggbb` or a palette key.
- `pattern`: `solid` (default) | `noise` | `stripes` | `planks` | `bricks` | `checker`.
  `scale`: number or `[a, b]` (meters; bricks: [brick width, brick height]); `axis`: x|y|z — the
  direction stripes/planks run along. Patterns are evaluated in **part-local** coordinates so they
  rotate with the part. Noise is a deterministic hash of the cell coordinates.
- `shine`: 0..1 specular strength (Blinn-Phong, fixed exponent 24).
- Missing sections fall back to `styles/default.yaml` values (a style may set only `materials`).
- `Style.material(name)` raises SightlineError with did_you_mean on unknown material.

## 4. Asset files (`assets/<name>.yaml`)

```yaml
asset: watchtower              # required; must equal the file stem
brief: >                       # optional: the designer's request, verbatim
  A wooden watchtower on a stone base, readable from far away.
style: default                 # optional; default "default"
tags: [building]               # optional, free text
parts:                         # required, ordered; references may only point to EARLIER parts
  - name: base                 # required, ^[a-z][a-z0-9_]*$, unique after expansion
    shape: box                 # required, see 4.1
    size: [3, 0.5, 3]
    material: stone            # required for solids; forbidden for cuts
    at: [0, 0, 0]
    align: bottom              # which own anchor goes to `at` (default center)
  - name: shaft
    shape: cylinder
    radius: 1.1
    height: 6
    sides: 8                   # optional: polygonal prism
    material: wood
    on: base.top               # attach: own `align` anchor (default: opposite = bottom) to target anchor
  - name: door
    shape: box
    size: [1.0, 2.1, 1.0]
    op: cut                    # subtract from `from` (default: the `on` target)
    on: shaft@[0, -1, 1]       # normalized anchor: bottom of the front face
    align: bottom
  - name: roof
    shape: cone
    radius: 1.8
    height: 2.2
    material: roof
    on: shaft.top
  - name: crenel
    shape: box
    size: [0.3, 0.4, 0.3]
    material: stone
    on: base.top_front_left
    offset: [0.2, 0, -0.2]
    repeat: {count: 4, step: [0.85, 0, 0]}    # crenel_0 .. crenel_3
reference:                     # optional, see section 12; one entry or a list
  image: refs/watchtower.png   # relative to the asset file's directory, then root
  view: front                  # front/back/left/right/top/iso or {azimuth: 30, elevation: 15}
  height: 8.0                  # optional real height in meters of the object in the image
  mask: auto                   # auto | path to a black/white mask png
checks:                        # optional, see section 7
  - grounded
  - connected
  - size: {height: [6, 10]}
  - visible: {part: door, view: front, min: 0.8, why: "player must see the entrance"}
  - matches_reference: {min_iou: 0.85}
```

### 4.1 Shapes (local frame: centered on the local bounding box, Y = the shape's main axis)

| shape | params | local bbox (w, h, d) |
|---|---|---|
| `box` | `size: [w,h,d]`, `round: r` (edge radius, ≤ min/2, default 0) | w, h, d |
| `sphere` | `radius` | 2r, 2r, 2r |
| `ellipsoid` | `size: [w,h,d]` (full extents) | w, h, d |
| `cylinder` | `radius`, `height`, `sides: n` (≥3, optional polygon), `round: r` | 2r, h, 2r |
| `cone` | `radius` (bottom), `height`, `top_radius` (default 0 → cone, >0 → frustum), `sides: n` | 2·max(r, top), h, 2·max(r, top) |
| `capsule` | `radius`, `height` (total, ≥ 2r) | 2r, h, 2r |
| `torus` | `radius` (ring center radius), `thickness` (tube radius); ring lies in local XZ | 2(R+t), 2t, 2(R+t) |
| `wedge` | `size: [w,h,d]` — triangular prism along X: full height at the back face (−Z), slope down to 0 at the front edge (+Z) | w, h, d |

Polygon `sides`: a regular n-gon prism (or pyramid/frustum for `cone`) with a **flat face** toward
+Z, so a door cut into the front of an octagonal tower sits on a face. `radius` is the
circumradius; the local bbox stays 2r × h × 2r.
SDFs must be exact or a conservative lower bound (never overestimate distance).

### 4.2 Placement

- `at: [x, y, z]` — world point where the part's `align` anchor goes. Default `at: [0,0,0]`,
  default `align: center`.
- `on: <part>.<anchor>` or `on: <part>@[u, v, w]` — target point on an earlier part. The part's own
  `align` anchor (default: the **opposite** anchor, every component negated: `top`→`bottom`,
  `top_front`→`bottom_back`, `@[0,1,1]`→`@[0,-1,-1]`, `center`→`center`) is placed on it.
- `offset: [dx, dy, dz]` — added after attachment.
- `follow: true` (default with `on`) — the part inherits the target's orientation: its world matrix
  is `M_target @ R_own` and `offset` is expressed in the target's frame. `follow: false` → world
  frame. With `at`, rotation and offset are in the world frame.
- Anchors are points of the part's LOCAL bounding box, transformed by its world matrix:
  `center`, `top`(+Y) `bottom`(−Y) `right`(+X) `left`(−X) `front`(+Z) `back`(−Z), and any
  combination of at most one per axis joined with `_` in any order (`top_front_left`).
  `@[u, v, w]` is normalized: each component in [−1, 1] maps lo..hi of the local bbox.
- `rotate: [rx, ry, rz]` (degrees, R = Rz Ry Rx) about the local origin.

### 4.3 Expansion: `repeat` and `mirror`

Applied in this order after placement of the base part: repeat, then mirror.

- `repeat: {count: n, step: [dx,dy,dz]}` — linear; copy i is translated by `i·step` (world frame,
  or the target frame when `follow` applies).
- `repeat: {count: n, axis: y, center: [x,y,z] | <part>.<anchor>}` — radial; copy i is rotated by
  `i·360/n` degrees about the axis through `center` (default `[0,0,0]`), position and orientation.
- Copies are named `<name>_0 … <name>_{n-1}`; the bare `<name>` is not defined afterwards.
- `mirror: x | y | z` — adds a reflected twin across the world plane through the origin
  perpendicular to that axis. World matrix `S @ M`, position `S @ p` with `S = diag(±1)`.
  Matrices may therefore be orthogonal with det −1; SDF evaluation uses `M.T @ (p − pos)`, which
  stays exact. Twin name: `mirror_name` if given; else swap a suffix `_l`↔`_r`,
  `_left`↔`_right`; else `<name>_mirror`. Repeated copies each get a twin.

### 4.4 Operations

- `op: union` (default). `blend: k` (meters) — smooth union with the `on` target (or
  `blend_with: <part>`), polynomial smooth-min with radius k.
- `op: cut` — the part is subtracted, never rendered. `from: [names] | all` (default: the `on`
  target; error if neither). A cut applies to every listed solid part: `d = max(d_part, −d_cut)`.
  `all` = all earlier solid parts.
- `tags: [...]` per part (free text, e.g. `door`, `walkable`).

### 4.5 Validation (loader)

Errors (SightlineError with `where` and `hint`): unknown top-level or part keys, unknown shape,
missing/negative/zero sizes, unknown material (did_you_mean over the style's materials), forward
or unknown references, unknown anchor, duplicate names after expansion, `round` too large,
capsule height < 2r, `count` < 1, more than 256 parts after expansion, cut with material, solid
without material. `asset:` must equal the file stem.

## 5. Model (`model.py`) — the in-memory contract

```python
@dataclass
class Material:
    name: str
    color: tuple[float, float, float]          # linear 0..1 sRGB values as written (no gamma math)
    pattern: str = "solid"
    pattern_color: tuple[float, float, float] | None = None
    scale: tuple[float, float] = (0.25, 0.25)
    axis: str = "y"
    shine: float = 0.0

@dataclass
class Style:
    name: str
    palette: dict[str, tuple[float, float, float]]
    materials: dict[str, Material]
    light_dir: np.ndarray        # unit vector, direction light travels
    ambient: float
    shadows: bool
    outline_width: int
    outline_color: tuple[float, float, float]
    background: tuple[float, float, float]
    ground_show: bool
    ground_color: tuple[float, float, float]
    ground_grid: float
    render_res: int
    render_views: list[str]
    def material(self, name) -> Material

@dataclass
class Part:
    name: str
    shape: str
    params: dict                 # normalized numbers: size (3-tuple), radius, height, top_radius, sides, round, thickness
    matrix: np.ndarray           # (3,3) orthogonal local->world (det may be -1)
    position: np.ndarray         # (3,) world position of the local origin
    material: str | None         # None for cuts
    op: str = "union"            # "union" | "cut"
    cut_from: list[str] = ...    # names of solid parts this cut applies to
    blend: float = 0.0
    blend_with: str | None = None
    attached_to: str | None = None   # target part name from `on`
    source: str = ""             # "parts[3] (sword)" or "parts[3] (sword) mirror" for messages
    tags: list[str] = ...
    helper: bool = False         # render helpers (scale figure); excluded from inspect/checks
    local_half: np.ndarray       # (3,) half extents of the local bbox
    def to_local(self, pts) -> np.ndarray          # (N,3) world -> local
    def sdf(self, pts) -> np.ndarray               # (N,) float32, local primitive only (no cuts/blend)
    def anchor(self, spec) -> np.ndarray           # spec: "top_front" or (u,v,w)
    def world_aabb(self) -> tuple[np.ndarray, np.ndarray]

@dataclass
class Group:                     # evaluation unit = one solid part, or several joined by blend
    members: list[int]           # indices into Scene.parts (solids), in file order
    blend_k: dict[int, float]    # member -> k for smooth union with earlier members (0 = hard min)
    cutters: dict[int, list[int]]  # member -> cut part indices
    aabb_min: np.ndarray; aabb_max: np.ndarray     # members' AABBs, expanded by max k

class Scene:
    parts: list[Part]            # all parts incl. cuts and helpers, file order
    groups: list[Group]
    def solid_indices(self, include_helpers=False) -> list[int]
    def eval_part(self, i, pts) -> np.ndarray      # part i with its cutters applied
    def eval_group(self, g, pts) -> tuple[dist, part_index]
    def eval(self, pts) -> tuple[np.ndarray, np.ndarray]   # union of groups; part index of the nearest solid (-1 if none)
    def bounds(self, include_helpers=True) -> tuple[np.ndarray, np.ndarray]  # union of solid world AABBs
    def only(self, indices) -> "Scene"             # sub-scene with these solids + their cutters (same Part objects)
    def with_parts(self, extra_parts) -> "Scene"   # adds helper parts as their own groups

def make_part(name, shape, params: dict, position, rotate_deg=(0, 0, 0), material=None,
              helper=False, op="union") -> Part   # normalizes params, computes local_half/matrix

@dataclass
class ReferenceSpec:             # parsed `reference:` entry
    image: pathlib.Path          # resolved, must exist (SightlineError otherwise)
    view: str                    # a camera view name; {azimuth, elevation} -> camera.orbit_view(...)
    height: float | None
    mask: str | pathlib.Path     # "auto" or resolved path

@dataclass
class Asset:
    name: str
    path: pathlib.Path
    brief: str
    style: Style
    tags: list[str]
    parts: list[Part]            # expanded, file order
    checks: list                 # raw check entries from YAML (validated by checks.py)
    references: list[ReferenceSpec]
    raw: dict                    # parsed YAML
    def scene(self) -> Scene
    def part(self, name) -> Part # SightlineError with did_you_mean
```

## 6. Geometry passes

### 6.1 `march.py`

```python
@dataclass
class ViewBuffers:
    camera: Camera
    hit: np.ndarray       # (H,W) bool
    depth: np.ndarray     # (H,W) float32, distance along the ray from the ray origin; inf if no hit
    part: np.ndarray      # (H,W) int32, index into scene.parts; -1 = no hit
    point: np.ndarray     # (H,W,3) float32 world hit point
    normal: np.ndarray    # (H,W,3) float32 unit world normal
    local: np.ndarray     # (H,W,3) float32 hit point in the hit part's local frame (for patterns)

def trace(scene, origins, dirs, t_max, eps, max_steps=160, groups=None) -> tuple[t, part]
def march(scene, camera, max_steps=160) -> ViewBuffers
```

Algorithm (performance matters — numpy only): for every group, intersect all rays with the group
AABB (slab test, expanded by `2·eps`); sphere-trace **only** the rays that enter it, starting at
the entry distance, evaluating only that group's SDF; record the first hit `t_g`. The scene hit is
the minimum over groups (union ⇒ first hit = min of per-group first hits). Hit threshold
`eps = 0.5 · camera.mpp`; step factor 0.9; stop at the AABB exit. Normals: central differences of
the hit group's SDF with h = `0.5 · mpp`. Part index at a hit = argmin of member distances. Use
float32. `trace` is the same machinery for arbitrary rays (shadow rays in render.py).
Target: a 30-part asset, 4 views at 256 px in < 6 s on a laptop CPU.

### 6.2 `voxel.py`

```python
@dataclass
class VoxelGrid:
    origin: np.ndarray        # world position of cell (0,0,0)'s center
    cell: float               # cell edge in meters
    shape: tuple[int,int,int] # (nx, ny, nz)
    solids: list[int]         # scene part indices, order of the first axis below
    strict: np.ndarray        # (len(solids), nx, ny, nz) bool: eval_part < 0 at the cell center
    loose: np.ndarray         # same, eval_part < 0.5*cell  (used for contacts; tolerance = 1 cell)
    def centers(self) -> np.ndarray   # (nx,ny,nz,3)
def occupancy(scene, max_cells=96, include_helpers=False) -> VoxelGrid
```

Grid covers `scene.bounds()` expanded by one cell; `cell = longest side / max_cells`; per-part AABB
culling (evaluate a part only on cells inside its AABB ± cell).

## 7. Inspect and checks

### 7.1 Report (`inspect.py`) — `inspect_asset(asset, res=128, views=("front","right","top","iso"), max_cells=96, compare=()) -> dict`

Keys (the JSON schema; text formatting renders all of them):

- `asset`, `style`, `brief`, `part_count` (solids), `cut_count`, `voxel_cell` (m), `render_mpp`.
- `size`: `{width, height, depth, min:[x,y,z], max:[x,y,z], human_ratio}` from solid AABB union
  refined by occupancy (use the SDF-exact AABB of parts, not voxels).
- `ground`: `{min_y, grounded: bool (|min_y| ≤ max(0.02, cell/2)), floats_by}`.
- `mass`: `{volume, center:[x,y,z], stable: bool|None, support_margin}` — uniform density over
  strict union occupancy; support = convex hull (xz) of occupied cells in the lowest layer touching
  the ground; stable = COM xz inside the hull; margin = signed distance to the hull edge.
  `None` if not grounded.
- `symmetry`: `{x: IoU, z: IoU}` of union occupancy mirrored across the AABB center planes.
- `parts`: list of `{name, shape, material, center, size:[w,h,d] (world AABB), volume_share,
  attached_to, visible: {view: fraction}, tags}`. `visible[view]` = pixels where this part is the
  first hit in the full render ÷ pixels it covers when rendered alone (`scene.only([i])`), same
  camera; 0 when it covers no pixel.
- `contacts`: list `{a, b, kind: "touch"|"overlap", overlap_volume}` from `loose`
  intersections; `overlap` when strict intersection volume > 0.
- `floating`: list `{parts: [...], gap, nearest}` — contact components that do not reach the
  ground component; `gap` = min distance (m) to the grounded component (SDF sampled at the
  floating component's occupied cell centers), `nearest` = name of the nearest grounded part.
- `cuts`: list `{name, cuts: [...], removed_volume}`.
- `colors`: `{view: "iso", shares: {material: fraction}}` of visible pixels.
- `contrast`: list `{a, b, ratio}` for part pairs adjacent in the iso image (sharing ≥ 3 boundary
  pixels), WCAG relative luminance of each part's mean shaded color; sorted ascending.
- `silhouette`: `{view: {"cols": n, "rows": m, "cell": meters, "x0": .., "y0": .., "legend": {letter: part}, "grid": [str rows]}}`
  — part map downsampled to ≤ 32 columns with square cells in meters; each cell shows the letter of
  the part covering most of it, `.` empty; letters A–Z then a–z assigned by descending visible
  area. Rows carry meter labels in text output, and characters are space-separated.
- `readability`: `{view: "front", at_px: {32: {iou, parts_kept, parts_lost}, 16: {...}}}` — the
  silhouette rendered so the asset's longer image side is N px, upscaled back, IoU vs full res;
  a part is kept if it covers ≥ 4 px at that size.
- `compare`: `{other: {view: {iou_true_scale, iou_normalized}}}` silhouettes aligned at ground and
  x-center; normalized = both scaled to equal height.
- `warnings`: list of short sentences (floating parts, parts < 30% visible in every view, contrast
  < 1.5 between adjacent parts, parts lost at 32 px, ungrounded, unstable).

`format_report(report) -> str`: stable, compact, aligned columns, headings in CAPS. The JSON and
text carry the same facts. `diff_reports(old, new) -> str` lists changed metrics with old → new.

### 7.2 Checks (`checks.py`) — `run_checks(asset, report, entries=None, root=None) -> list[CheckResult]`

`CheckResult(kind, passed, measured, expected, message, why)`. Entry forms (bare string = no args):

| check | meaning |
|---|---|
| `grounded` / `{grounded: {tolerance: m}}` | min y within tolerance of 0 |
| `connected` | no floating components |
| `stable` | COM over the support polygon |
| `size: {width/height/depth: v or [lo, hi]}` | scalar v ⇒ ±2% |
| `part_size: {part: p, width/height/depth: ...}` | world AABB of that part |
| `symmetric: {axis: x, min: 0.9}` | occupancy IoU |
| `visible: {part: p, view: front/…/any, min: f}` | `any` = best view |
| `ratio: {a: "<part or total>.<width/height/depth>", b: "...", range: [lo, hi]}` | a/b |
| `relation: {a: p, is: above/below/left_of/right_of/in_front_of/behind/touching/inside, b: q}` | AABB-center/contact based |
| `contrast: {a: p, b: q, min: r}` | luminance contrast in iso |
| `readable_at: {px: 32, min_iou: 0.8, keep: [parts]}` | silhouette survives downscale; listed parts kept |
| `distinct_from: {assets: [..], view: front, max_iou: 0.7}` | normalized silhouette IoU |
| `max_parts: n`, `max_materials: n` | budgets |
| `matches_reference: {index: 0, min_iou: 0.85, max_missing: 0.05}` | via `reference.match_asset`; missing = share of reference area not covered |

Any entry may carry `why: "..."` (designer intent), printed on failure. Unknown check kinds and
keys are SightlineErrors. Text output: one line per check `PASS|FAIL  kind  measured … expected …`.

## 8. Rendering (`render.py`, `sheet.py`)

```python
def render_views(asset, views=None, res=None, scale_figure=True, outline=True) -> dict[str, PIL.Image]
def render_part_map(asset, view="front", res=None) -> PIL.Image   # flat color per part + name labels
def render_sprite(asset, view="iso", mpp=...) -> tuple[PIL.Image (RGBA), anchor_px]  # for level compositing
def write_renders(asset, out_dir, views=None, res=None) -> dict[str, Path]  # PNG per view + parts.png + sheet.png
```

- Shading: material base color × pattern, Lambert with `light_dir` + ambient, optional shadows
  (shadow rays via `march.trace` toward −light_dir, from hit point + normal·2eps), optional
  Blinn-Phong `shine`, cheap 4-sample ambient occlusion along the normal.
- Ground plane y = 0 drawn under the asset (analytic ray–plane hit, receives shadows) with grid lines
  every `ground.grid` m (thicker line every 5 m); not in top view under the asset footprint.
- Outline: pixels where `part` differs from a neighbor or depth jumps > 4·mpp, dilated to width px.
- Scale figure: a neutral grey 1.8 m human (capsule body + sphere head) placed at
  `x = asset max x + 0.6 m`, as helper parts (`helper=True`) in its own group.
- Caption strip under every image: `<asset> · <view> · 1 px = 0.023 m · grid 1 m`.
- Part map: each part a distinct flat color (golden-ratio hue spacing, fixed order), names written
  at the centroid of each part's visible pixels with a leader line if the label would overlap.
- `sheet.png` (one asset): all views + part map in a grid with a header (name, size, part count).
- `sheet_many(assets, out)`: iso views of many assets at a **shared** scale side by side with the
  human, names and heights — compares scale across the library.
- Default PNG upscale: images smaller than 384 px on the long side are upscaled ×2 with nearest
  neighbour for viewing; the caption states the true mpp.

## 9. Export (`export.py`)

Meshing: naive surface nets over an SDF on a voxel grid (cell = longest side / `max_cells`), one
vertex per sign-changing cell (mean of edge crossings, then 1–2 projection steps onto the surface
along the SDF gradient for sharper results), one quad per sign-changing edge → two triangles with
consistent outward winding (normals from the SDF gradient). Coordinates in meters, Y up — the glTF
convention, so no axis conversion.

- `export_glb(asset, out_path, max_cells=128, per_part=True, bake_colors=False) -> Path` —
  **glTF 2.0 binary**, written by hand (JSON chunk + BIN chunk, 4-byte aligned, `asset.version
  "2.0"`, `generator "sightline"`). `per_part=True`: one node + mesh per blend group / solid part,
  named after the part (blend group: first member's name), meshed from that group's SDF with its
  cuts applied, so engines can address, animate or swap parts; node translation = 0 (vertices in
  asset space) for simplicity. `per_part=False`: one merged mesh. One glTF material per style
  material: `pbrMetallicRoughness.baseColorFactor` from the color (sRGB → linear conversion),
  `metallicFactor` = shine·0.8, `roughnessFactor` = 1 − shine·0.6. `bake_colors=True`: `COLOR_0`
  vertex colors from the material pattern evaluated at each vertex (part-local coordinates).
  Attributes: POSITION (with min/max), NORMAL, indices (UNSIGNED_INT). Must load in Blender,
  Godot and three.js; tests validate the structure by parsing the file back.
- `export_obj(asset, out_path, max_cells=128) -> Path` — OBJ + MTL, `o <part>` and
  `usemtl <material>` per part, `Kd` from the style.
- Level export (stretch): `export_level_glb(level, out_path)` — terrain column boxes merged per
  tile kind + object assets as separate nodes.

## 10. Levels (`levels/`)

```yaml
level: courtyard               # must equal file stem
brief: ...
style: default
cell: 1.0                      # meters per cell in x and z
height_unit: 0.5               # meters per step in the `heights` grid
player: {step: 0.5, jump_up: 1.0, jump_gap: 2, max_drop: 3.0, height: 1.8}
heights: |                     # rows = z (row 0 = north/back, increasing toward +Z/front),
  0 0 0 0 1 1                  # columns = x (col 0 = west/left); integers, space-separated
  0 0 0 0 1 1
tiles: |                       # same shape; one char per cell, space-separated
  . . ~ ~ . .
  . . ~ ~ D .
legend: {"%": floor}           # optional additions/overrides: char -> kind
markers: {spawn: [0, 0], goal: [5, 1], key_red: [1, 1]}     # [col, row]
doors: {door_red: {at: [4, 1], key: key_red}}
objects:
  - {asset: watchtower, at: [5, 0], rotate: 90, solid: true}
checks:
  - reachable: goal                     # from spawn
  - order: [key_red, door_red, goal]    # each must be reached before the next is possible
  - no_softlocks
  - visible: {from: spawn, to: goal}    # line of sight at eye height; or `not_visible`
  - path_length: {max: 80}              # meters along the critical path
  - jumps: {min: 1, max: 5}
  - max_fall: 3.0
```

Tile kinds (default legend): `.` floor, `#` wall (solid, blocks movement and sight; its top is its
`heights` value), `~` water (deadly, not standable, can be jumped over), `^` spikes (deadly), `_`
pit (void), `=` ladder (moving onto or off a ladder cell ignores the height difference), `D` door
(standable if the door's key was collected; doors not listed in `doors:` need no key), `k` key
pickup (a marker is also needed for its name). World position of cell (c, r) center:
`x = (c + 0.5)·cell`, `z = (r + 0.5)·cell`, `y = heights[r][c]·height_unit`.

Bot (`nav.py`): state = (col, row, frozenset(keys)). Moves (4 directions): **walk** to a neighbor
that is standable when rise ≤ `step` and drop ≤ `max_drop`; **jump** straight over 1..`jump_gap`
non-standable or deadly cells (not walls taller than takeoff + jump_up) to a standable landing with
rise ≤ `jump_up` and drop ≤ `max_drop`; **ladder** as above. Picking a key = entering its marker
cell. Dijkstra on cost = meters travelled. Outputs: critical path (spawn → goal, through required
keys), segments compressed to readable steps (`walk E 4 cells (1,1)→(5,1)`, `jump E over water 2
cells`, `climb +1.0 m`, `drop −2.0 m`, `pick key_red`, `open door_red`), reachable cell count,
unreachable markers, **softlocks** (reachable states from which the goal is unreachable),
max rise/fall, jump count, path length in meters.

Line of sight (`sight.py`): 3D segment between eye points (standing height − 0.1 m) sampled every
`cell/4`; blocked when the sample is below the terrain top of a wall/solid object cell or the
heightfield of the cell it is over. Reports the first blocking cell.

Report (`report.py`): `PATH`, `REACHABILITY`, `SOFTLOCKS`, `SIGHT` (for each `visible` check and
spawn→goal), `MAP` (ASCII with column/row rulers: markers as letters in a legend, path as `*`),
`HEIGHTS` (ASCII numbers). Same facts as JSON.

Render (`render.py`): iso PNG via PIL polygons (painter's algorithm, back to front, column prisms
with top + two visible sides, shaded by tile kind with style colors), same projection as
`camera.view_basis("iso")`; objects composited from `sightline.render.render_sprite` (fallback: a
labeled box of the object's footprint); markers as labeled pins; critical path as a polyline.
Plus a top-down PNG with a grid and coordinate labels. Both with captions.

## 11. CLI (`cli.py`)

```
sightline new <asset> [--style S] [--brief TEXT]         # template with checks from the brief left as TODO comments
sightline render <asset> [--views a,b] [--res N] [--out DIR] [--no-human]
sightline inspect <asset> [--json] [--res N] [--compare A[,B]]
sightline check [<asset>...] [--json]                    # no args: all assets; exit 1 on FAIL, 2 on error
sightline diff <asset> [--against REV|path]              # default: git HEAD version of the file
sightline sheet [<asset>...] [--out PATH]                # library sheet at shared scale
sightline export <asset> [--format glb|obj] [--merged] [--bake-colors] [--out PATH]   # default glb
sightline ref <image> [--view V] [--height M] [--mask PATH] [--json]   # analyze a reference alone
sightline match <asset> [--index I] [--fit-view] [--json]               # model vs reference
sightline level render|play|check <level> [--json]
sightline styles                                          # styles, materials, palettes
sightline shapes                                          # shape and anchor reference
```

`<asset>` is a name (`assets/<name>.yaml` under root) or a path. Every command prints the paths of
files it wrote. Errors print `SightlineError.format()` to stderr, exit 2.

## 12. Reference images (`reference.py`) — the main design loop

The designer usually starts from a picture (photo, concept art, sketch). The agent must bring the
model to what the picture shows, ask the designer about what the picture cannot tell, and keep the
model easy to change. The engine therefore measures the reference the same way it measures the
model, in the same units and grid formats, so the agent compares text with text.

### 12.1 Reference analysis — `analyze_reference(image_path, view="front", height=None, mask=None) -> dict`

- Segmentation (`mask: auto`): PNG alpha if any pixel has alpha < 250 (mask = alpha ≥ 128);
  otherwise background color = median of a 3-px border; mask = color distance (RGB Euclidean)
  > `tolerance` (default: max(24, 3 × border MAD)); keep connected components ≥ 0.5% of the image
  area (4-connectivity, numpy flood fill / union-find, no scipy). `confidence`: high / medium / low
  from border color spread and how much of the border the mask touches. A low confidence becomes a
  question for the designer ("send a cut-out PNG or a plain background"). `mask: <png>` overrides.
- Scale: `height` (m) gives meters per pixel = height / mask bbox height; without it the scale is
  unknown and the report says so (and the matcher fits by height to the model).
- Report keys: `image`, `size_px`, `view`, `mask: {confidence, coverage, bbox_px, components}`,
  `scale: {mpp or null, assumed}`, `proportions: {height_to_width, …}`,
  `silhouette` (same format as inspect's: ≤ 32 columns, square cells, meter labels when scale is
  known, `#` filled, `.` empty), `width_profile` (10 horizontal bands from the top: left, right,
  width — in meters when scale is known, else in % of height), `colors` (k-means k ≤ 6 in RGB on
  mask pixels, deterministic init by sorted luminance quantiles, 8 iterations: hex, share, band
  rows where it dominates, e.g. "#5f6a78 31% — rows 0–30% (top)"), `unknowns` (list of questions
  the picture cannot answer — always includes the hidden sides for an axis view: e.g. a `front`
  reference cannot tell depth or the back; scale unknown without `height`; low mask confidence;
  thin details below 3 px).
- `format_reference(report) -> str`.

### 12.2 Matching — `match_asset(asset, index=0, res=256) -> dict`

1. Render the model's silhouette and part buffer from the reference view (same camera math; an
   orbit view if azimuth/elevation are given).
2. Align: scale the reference mask so its bbox height equals the model's projected height (or use
   the reference mpp when `height` is given, then align by the bottom edge and horizontal center
   and report the size mismatch instead of hiding it); resample both onto one grid.
3. Report: `iou`, `missing` (share of reference area not covered by the model), `extra` (share of
   model area outside the reference), `aspect` (reference vs model height/width), `width_profile`
   (10 bands: reference vs model left/right/width in meters and the difference — "model too wide by
   0.4 m in band 0–10%"), `regions` (connected components ≥ 1% of the reference area of
   reference-only ("missing") and model-only ("extra") pixels, each with bbox in meters in the
   asset frame (x, y for front views) and the nearest visible part name — "missing 0.6 × 0.9 m at
   x 1.2..1.8, y 3.0..3.9, next to `shaft` → a part is missing (balcony?)"), `parts` (for each
   visible part: reference median color under its pixels, model rendered median color, ΔE (CIE76)
   and a suggested material base color = base × reference / rendered, clipped), `grid` (diff
   silhouette, ≤ 32 columns: `#` both, `+` model only, `-` reference only, `.` neither),
   `unknowns` (from the reference analysis).
4. Images: `out/<asset>/match_<i>.png` = reference | model render | overlay (both = grey,
   model-only = red, reference-only = blue) | part map over the reference, with captions.

`fit_view(asset, index=0) -> list[{view, iou}]`: when the reference angle is unknown, render
silhouettes at 64 px for azimuth 0..345 step 15 and elevation {0, 15, 30, 45, 60} plus `top`,
normalize both silhouettes to their bboxes, return the top 5 by IoU (note mirrored ambiguity:
azimuth a and −a give mirror images of symmetric objects).

### 12.3 Reference loop the AI guide teaches

1. `sightline ref <image>`: read the measurements and look at the image itself.
2. Ask the designer the `unknowns` in **one** short batch, each with a proposed default (scale,
   back side, materials, which details matter at gameplay distance, polygon budget/style).
3. Block out with few large primitives following the silhouette grid and the width profile; name
   parts after what they are (`roof`, `door`, `left_arm`).
4. `sightline match <asset>` and fix the largest `regions` first, then the width profile, then
   colors; repeat until `matches_reference` passes.
5. Add the designer's intent as checks, show `sheet.png` + `match_0.png`, take feedback by part
   names; export `.glb`.
