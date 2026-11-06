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
- Every user-facing error is a `sightline.errors.SightlineError` with `where` (file relative to
  root + YAML path + line, e.g. `assets/knight.yaml: parts[3] (sword), line 21`) and a `hint`.
  Unknown keys are errors, with `did_you_mean`. Never a bare traceback for invalid input.
- YAML is read with YAML 1.2 booleans (only `true`/`false`), so the key `on:` and the words
  yes/no/on/off stay strings. Duplicate keys are errors.
- All numbers in reports: meters with 2 decimals, percentages as integers, angles in degrees.
- Paths: run tools with absolute paths; the venv interpreter is `E:/sightline/.venv/Scripts/python.exe`.

## 1. Conventions

- Units: meters. Right-handed world, **+Y up, +X right (east), +Z front (toward the viewer of the
  front view, south)**. Ground plane is y = 0. Assets stand on the ground, centered at x = z = 0.
- Angles: degrees. `rotate: [rx, ry, rz]` means the matrix `R = Rz @ Ry @ Rx` (X applied first),
  about the part's own local origin (its local bounding-box center).
- Views (`sightline.camera`, already implemented): `front back left right top iso`, all
  orthographic. `iso` = azimuth 45° from +Z toward +X, elevation 30°. `top` has image-up = −Z.
  Free views are named `orbit:<azimuth>:<elevation>`. One render shares one meters-per-pixel
  scale across all its views.
- Left/right follow the axes as seen in the front view: `left` = −X, so `_l` parts sit on −X.
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
  decimate.py    quadric-error edge collapse of the part meshes (export)
  reference.py   reference image analysis, model-vs-reference matching, view fitting
  levels/        model.py loader.py nav.py sight.py report.py render.py checks.py
  cli.py         `sightline` command
styles/          default.yaml (+ more)
assets/          example assets (*.yaml); refs/ holds their reference pictures
levels/          example levels (*.yaml)
docs/            FORMAT.md (complete format reference), AI_GUIDE.md (how an agent works with it)
tests/           pytest, one file per module (test_loader.py, test_march.py, ...)
out/             generated files (gitignored): out/<asset>/, out/levels/<level>/, out/library.png
```

Project root (`root`): the directory containing `styles/`, `assets/`, `levels/`
(`style.resolve_root(root, start)`). Order: `--root`; else the nearest ancestor of the asset or
level file containing `styles/`; else env `SIGHTLINE_ROOT`; else the nearest ancestor of the CWD
containing `styles/`; else the repo root next to the installed package.

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
- `pattern_color` defaults to 0.8 × `color`; `scale` defaults to 0.25. Ranges: `ambient` 0..1,
  `outline.width` integer 0..16, `render.res` 16..4096. `style:` must equal the file stem.
- Missing values fall back **key by key** (not whole sections): built-in → `styles/default.yaml`
  → the style, for the palette, the materials and every key of light/outline/ground/render.
  Default materials resolve palette names against the merged palette, so a style can recolor the
  library by overriding palette keys, and every style offers every default material.
- `styles/default.yaml` has 39 materials (see `docs/FORMAT.md`), including `paint_red`, `lamp`
  (a bright warm color; there is no emission) and `helper` (the scale figure).
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
SDFs must be exact or a conservative lower bound (never overestimate distance). All primitives
are exact except the ellipsoid (outside f(f−1)/|∇f|, inside (f−1)·min(h): a bound that is not
1-Lipschitz far away) and polygonal cones (exact inside, max of face-section and circumscribed
frustum outside). `cone` needs `radius` > 0; an inverted cone is `rotate: [180, 0, 0]` or a
frustum with `top_radius > radius`.

### 4.2 Placement

- `at: [x, y, z]` — world point where the part's `align` anchor goes. Default `at: [0,0,0]`,
  default `align: center`.
- `on: <part>.<anchor>` or `on: <part>@[u, v, w]` — target point on an earlier part. The part's own
  `align` anchor (default: the **opposite** anchor, every component negated: `top`→`bottom`,
  `top_front`→`bottom_back`, `@[0,1,1]`→`@[0,-1,-1]`, `center`→`center`) is placed on it.
- `on:` always needs an anchor: `on: base` is an error with the hint `base.top`.
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
  Under `follow` the axis is in the target's frame (like the linear step).
- Copies are named `<name>_0 … <name>_{n-1}`; the bare `<name>` is not defined afterwards, except
  in a cut's `from:`, where a repeated base name means all its copies (so cuts survive a changed
  count).
- `mirror: x | y | z` — adds a reflected twin across the world plane through the origin
  perpendicular to that axis. World matrix `S @ M`, position `S @ p` with `S = diag(±1)`.
  Matrices may therefore be orthogonal with det −1; SDF evaluation uses `M.T @ (p − pos)`, which
  stays exact. Twin name: `mirror_name` if given; else swap a suffix `_l`↔`_r`,
  `_left`↔`_right`; else `<name>_mirror`. Repeated copies each get a twin (copies are registered
  first, then their twins). A twin's `attached_to`/`blend_with`/`cut_from` point at the twins of
  its targets when those were mirrored too (`hand_l` on `arm_l` → `hand_r` on `arm_r`); `from: all`
  is not remapped.

### 4.4 Operations

- `op: union` (default). `blend: k` (meters) — smooth union with the `on` target (or
  `blend_with: <part>`), polynomial smooth-min with radius k. Parts joined by blends form one
  group; inside a group each member's cuts are applied before the smooth union, and a member's k
  applies smin against all earlier members of its group, not only its named target.
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

Implementation notes (additive API, all in use by other modules):

- `Part` also has `to_world(local_pts)`, `sdf3(p3)` (on (3, N) float32 points) and `is_solid`;
  `world_aabb()` is the exact SDF AABB of the shape, not its rotated bbox. `Style.path` is the file
  it came from; `Asset.root` is the project root and `Asset.part_index(name)` the index.
- `Scene(parts, solids=None)` has `cutters`, `aabb(i)`, `part_dist3(i, p3, cull=None)` and
  `group_dist3(g, p3, cull=None, need_part=False)`. `Scene.only(indices)` keeps the full `parts`
  list with the asset's indices and only restricts the active solids and groups, so part ids in
  buffers from a sub-scene index `asset.parts` directly.
- `make_part` takes keyword-only `source=` and `tags=`. Part, Style, Group, Asset, ViewBuffers and
  VoxelGrid are `dataclass(eq=False)` (identity equality; the generated `__eq__` fails on numpy).
- `loader.load_asset_text(text, path, root)` loads YAML text as if it were the file at `path` (old
  git revisions); `loader.list_assets(root)`, `style.list_styles(root)`, `style.read_yaml` (data
  plus a line map) and `style.resolve_root(root, start)` are public.

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
Target: a 30-part asset, 4 views at 256 px in < 6 s on a laptop CPU (measured: about 0.2 s).

Notes: groups are visited nearest-first and rays already holding a closer hit are skipped. Inside
march/trace only, a cutter is ignored for points more than 4·mpp outside its AABB (exact outside
the solid). The hit threshold stays `0.5·mpp` (lead decision), so every silhouette is up to half a
pixel wider per edge: a 1 m box at 74 px reads about +4% in area, and every part is at least one
pixel wide. Consumers compensate: inspect's readability box-filters a 256 px render, inspect's
`visible` uses footprints shrunk by one pixel, reference matching ignores the 1-px fringe and regions
thinner than 1.5 px, and tests model the area as (w/mpp + 1)(h/mpp + 1).
Grazing rays that use up `max_steps` count as misses.
Coplanar cuts: where a cutter face lies on a face of its target, `max(d_part, -d_cut)` falls to 0 on
the shared plane over the hole although no material is near (a "phantom" sheet: a notch flush with
the wall top would render with a lid). A hit of a group with cutters, near a cutter, is therefore
confirmed by `Scene.phantom3`: a step of `d + eps/2` down the field gradient (failing that, along the
six axes) must land in material (field < 0, which is exact); otherwise the ray steps on by at least
`2·eps`. Voxel `loose` cells and inspect's contact and floating distances use the same test.

### 6.2 `voxel.py`

```python
@dataclass
class VoxelGrid:
    origin: np.ndarray        # world position of cell (0,0,0)'s center
    cell: float               # cell edge in meters
    shape: tuple[int,int,int] # (nx, ny, nz)
    solids: list[int]         # scene part indices, order of the first axis below
    strict: np.ndarray        # (len(solids), nx, ny, nz) bool: eval_part < 0 at the cell center
    loose: np.ndarray         # same, eval_part < 0.5*cell  (used for contacts; tolerance = 1 cell),
                              # minus the phantom sheet of coplanar cut faces (6.1)
    def centers(self) -> np.ndarray   # (nx,ny,nz,3)
def occupancy(scene, max_cells=96, include_helpers=False) -> VoxelGrid
```

Grid covers the AABB union of the selected solids (helpers excluded unless `include_helpers`),
expanded by one cell and centered on it, so mirroring across the AABB center maps cell onto cell;
`cell = longest side / max_cells`; per-part AABB culling (evaluate a part only on cells inside its
AABB ± cell). Extra API: `union(loose=False)`, `layer(part_index)`, `volume(part_index=None)`.
Occupancy is per part, so the fillet volume of a smooth blend belongs to no part's layer.

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
  camera, both counted on the alone footprint eroded by one pixel (the whole footprint when the
  erosion leaves nothing), so the half-pixel dilation of an occluder's edge does not eat the rim of a
  part seen through a close-fitting opening; 0 when it covers no pixel.
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
text carry the same facts. `diff_reports(old, new) -> str` lists changed metrics as
`  path: old -> new` lines (ASCII `->`: piped output on Windows uses an ANSI code page) under a
`CHANGES     n` header, plus `+ part x (shape, material)`, `- part x` and `+ warning: ...` lines.

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

Any entry may carry `why: "..."` (designer intent), printed on failure, inside its mapping or as a
sibling key. Unknown check kinds and keys are SightlineErrors; all entries are validated before any
work, with `where = '<file>: checks[i] (kind)'` (no YAML line: Asset keeps no line map). Text
output: one line per check `PASS|FAIL  kind  measured … expected …  — fix  (why: …)` and a summary
`CHECKS  n: p passed, f failed` (`format_checks`).

### 7.3 Implementation notes (inspect and checks)

Report additions and rules beyond 7.1:

- Extra keys: `pieces` (contact components), `parts[].lit_color` (hex of the mean lit color in iso,
  null when not visible there), `floating[].direction` (above/below/left of/right of/in front
  of/behind), `cuts[].visible` ({view: fraction}), `silhouette[view].axes` (e.g. `['x','y']`,
  `['x','-z']` for top, `['u','v']` for iso) and `.row_labels` (world coordinate of each row center).
- `size`: exact SDF AABB per solid; a side of a part with cuts shrinks to its occupied cells only
  when a cut removed more than one cell from that side.
- Contacts: two parts touch when their loose cells meet **or** when `d_a + d_b < one cell` at some
  cell center near both (SDF on the AABB overlap box); this never undercounts a real contact when a
  face lies exactly between two cell layers. Overlap = strict intersection volume > 0.
- Floating gap = min over the floating piece's boundary cell centers of `d_grounded + d_floating`
  (an upper bound, exact for facing faces); where `d_grounded` is the phantom sheet of a coplanar cut
  (6.1) it is replaced by the distance to the grounded part's occupied cells (accurate to a cell).
  `floating[].move` = the vector [dx, dy, dz] (m) that closes the gap: the gap along the direction
  down the grounded part's SDF gradient at the closest point (toward its nearest occupied cell for a
  phantom distance). If no part reaches the ground, the lowest piece is the
  base: a lone floating object is ungrounded but not "floating", so `connected` and `grounded` stay
  independent.
- Stability: the support polygon is the convex hull of the cell corners in the lowest two occupied
  layers; `support_margin` is the signed distance of the COM to its edge (positive inside).
- `volume_share` = the part's strict voxel volume / union volume (overlaps make the sum exceed 1).
- `cuts[].removed_volume`: midpoint sampling of the target primitives inside the cutter, 40 cells
  along the longest side of their AABB overlap (not the report grid). A cut that removes nothing is
  0% visible in every view (a target face it only touches is no carved surface). Cut visibility: footprint =
  the cutter rendered alone AND its targets; a footprint pixel shows the cut when its first hit is a
  target on the carved surface or the ray passes through the hole (assumes a convex cutter).
- Silhouettes: at most 32 columns **and** 32 rows; one cell size for all views from "clean" sizes
  (multiples of 0.02 m up to 0.2, then 0.3 0.4 0.5 0.6 0.8 1 1.5 2 2.5 …); x = 0 is a column center
  and upright views start rows at y = 0; a cell is filled at ≥ 50% coverage; letters are assigned
  once for all views by total visible area, so a part keeps its letter (beyond 52 parts: `*`).
  `x0`/`y0` are the left and bottom grid edges. The text adds a column ruler every 4 columns.
- Readability: the N px silhouette is box-filtered from a 256 px render (a low pixel is on at ≥ 50%
  coverage), because the half-pixel hit threshold would keep every part at least one pixel wide.
  Parts counted = those visible at full resolution; kept = owns ≥ 4 low pixels.
- Contrast uses the lit color `color × (ambient + (1 − ambient) × Lambert)` (no patterns, shadows
  or outlines); low-contrast warnings merge into one line and skip same-material pairs.
- Warnings add `cut X removes nothing`, split ungrounded into "floats N m above" / "sinks N m below
  the ground", and name side and anchor for floating parts (`ball floats 0.50 m above block —
  attach it with on: block.top or move it 0.50 m down`): the move is `floating[].move` per axis,
  largest first (`move it 1.33 m left (-x) and 0.60 m down`), and `on: <part>.<side>` is offered only
  when that anchor's local direction faces the world side and the anchor lies on the target's surface
  within one cell (not for a cone's side, a polygon corner or a rotated part). A cut gets a
  low-visibility warning only when another part hides it; a part whose AABB center lies inside the
  cutter (a panel set into its opening) does not hide it: that view counts the cut's visibility
  without other parts. A failing `visible` check names the main occluder in that view
  (`inspect.visible_occluders`).
- Numbers in text: voxel cell, m/px and m³ with 3 decimals, other meters with 2, percentages as
  integers.

Check defaults and forms: `visible` view `any`, min 0.5; `symmetric` axis x (x or z), min 0.9;
`contrast` min 1.5 for **any** pair (computed from the reported lit colors); `readable_at` px 32,
min_iou 0.8, keep [], optional `view` (default front; another px or view is computed on the fly);
`distinct_from` view front, max_iou 0.7, `assets` a list or one name (resolved against `root`,
default the asset's root); `matches_reference` index 0, min_iou 0.85, max_missing 0.05 —
SightlineError at validation when the asset has no such reference; it calls
`reference.match_asset(asset, index)` and reads `iou`/`missing` (fractions). `max_parts` /
`max_materials` also accept `{max: n}`. `size`/`part_size` list dimensions in the order written; a
scalar `ratio.range` means ±2%, and `total.<dim>` in `ratio` is the asset. `relation`
above/below/left_of/right_of/in_front_of/behind compare AABB centers strictly; `touching` = a
report contact; `inside` = AABB containment with a one-cell tolerance; part_size, ratio and relation
accept cut parts (their exact AABB).

## 8. Rendering (`render.py`, `sheet.py`)

```python
def render_views(asset, views=None, res=None, scale_figure=True, outline=True, *, caption=True,
                 upscale=None, mpp=None) -> dict[str, PIL.Image]          # presentation images
def render_part_map(asset, view="front", res=None, *, labels=True, outline=True, caption=True,
                    upscale=None) -> PIL.Image   # flat color per part + name labels
def render_sprite(asset, view="iso", mpp=None) -> tuple[PIL.Image (RGBA), (col, row)]  # for levels
def write_renders(asset, out_dir, views=None, res=None, *, scale_figure=True) -> dict[str, Path]
def render_raw(asset, views=None, res=None, *, mpp=None, scale_figure=True, outline=True,
               ground=None, shadows=None) -> dict[str, ViewRender]   # raw buffers for inspect/reference
def render_view(scene, style, cam, ground=None, outline=True, shadows=None, ao_step=None) -> ViewRender
def surface_color(material, local_pts, local_normals, mpp) -> (N, 3)   # unlit pattern colors (export bakes with it)
def material_for(style, name) -> Material                               # 'helper' always exists
def sheet_many(assets, out_path, res=None, view="iso", *, root=None) -> Path
```

- Shading: material base color × pattern, Lambert with `light_dir` + ambient, optional shadows
  (shadow rays via `march.trace` toward −light_dir, from hit point + normal·2eps), optional
  Blinn-Phong `shine`, cheap 4-sample ambient occlusion along the normal. For readability it adds a
  fill light (0.3) from the mirrored horizontal direction, a sky tilt of the ambient (0.25, brighter
  on upward faces), exposure 1.2 with a soft shoulder above 0.8 for asset surfaces, and light
  normalized so a lit, unoccluded upward face shows the exact material (or ground) color. A cast
  shadow removes 60% of the direct light and is blurred (3×3 box, twice) on the ground only.
- Ground: a finite **slab** (top face y = 0) over the asset + figure footprint, padded by
  max(0.25 m, 6% of the size), at least 1:2, 2% of the size thick, hit analytically by ray–box
  intersection, receiving shadows, with grid lines every `ground.grid` m (a darker line every 5 m).
  The cameras frame the whole tile; front/side views show it as a strip. Parts below y = 0 hide in
  the slab, so sinking is visible. Shadows falling off the tile are not drawn.
- Outline: silhouettes and depth jumps > 4·mpp in the full outline color; part boundaries without a
  depth jump at 50%. Width w grows **inward** on the nearer surface, so a w-px line is exactly w px,
  silhouettes keep their true size and coverage equals the asset pixels.
- Scale figure: a neutral grey 1.8 m human (capsule body radius 0.17 up to 1.42 m + head sphere
  radius 0.12, top at 1.80 m) as helper parts at `x = asset max x + 0.6 m`,
  `z = asset min z − 0.45 m`, so it never hides the asset in the front, right or iso views.
- Caption strip under every image: `<asset> · <view> · 1 px = 0.023 m · grid 1 m`, where 1 px is a
  pixel of the **saved PNG** (render mpp / upscale): values ≥ 0.01 with 3 decimals, smaller ones
  with 2 significant digits.
- Default PNG upscale: images smaller than 384 px on the long side are upscaled ×2 (nearest).
- Part map: a flat golden-ratio color per solid part (file order), dark lines between parts, the
  style background, labels with leader lines that never overlap (greedy, moved to side margins when
  crowded), parts not visible in that view listed under it; upscaled to ≥ 640 px.
- `write_renders` writes `<view>.png` (orbit views as `orbit_az_el.png`), `parts.png` (the iso part
  map when iso is rendered, else front or the first view) and `sheet.png` (all views + part map
  under a header: name, w × h × d, parts, cuts, materials, style, height in humans, brief).
- `render_sprite`: asset alone (no ground, no figure), transparent background (alpha 0/255),
  outlines and self-shadows; the anchor is the fractional (col, row) of world (0, 0, 0) where
  pixel (c, r) covers [c, c+1) × [r, r+1): pasting at (X − col, Y − row) puts the asset origin on
  (X, Y). `mpp=None` uses the style's framing.
- `sheet_many`: the human first, then every asset on its own ground tile with the same grid at one
  shared scale (tallest tile = `res` px, default 320), rows wrap at 3000 px with the ground origins
  of a row on one baseline, name and h/w/d under each; accepts Asset objects or names/paths.
- Patterns: stripes/planks alternate across the first of the other local axes (order y, x, z) that
  the normal does not point along (|n| < 0.75); planks get staggered butt joints (board length 8 ×
  width) and a per-board tint; bricks a per-brick tint, paving on upward faces; noise uses t²
  shaping; checker cells are scale[0] in x/z, scale[1] in y; patterns are sampled 0.6·mpp below
  the surface (no flicker on cell borders) and fade to their mean color under a period of ~3 px.
- No anti-aliasing (one sample per pixel, measurable pixel counts). Shadows on asset surfaces are
  hard; the ground slab does not block light.

## 9. Export (`export.py`)

Meshing: naive surface nets over an SDF on a voxel grid (cell = longest side / `max_cells`), one
vertex per sign-changing cell (mean of edge crossings, then 1–2 projection steps onto the surface
along the SDF gradient for sharper results), one quad per sign-changing edge → two triangles with
consistent outward winding (normals from the SDF gradient). Coordinates in meters, Y up — the glTF
convention, so no axis conversion. Then **decimation** (`decimate.py`, section 9.1) collapses each
part mesh to game-ready triangle counts (crate 400k → ~3k triangles) unless `quality="full"`.

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
- Level export (stretch, **not implemented**): `export_level_glb(level, out_path)` — terrain
  column boxes merged per tile kind + object assets as separate nodes.

Implementation notes:

- Signatures add keyword-only `meshes=`, `quality=`, `max_tris=`: `export_glb(asset, out_path,
  max_cells=128, per_part=True, bake_colors=False, *, meshes=None, quality=None, max_tris=None)`,
  `export_obj(asset, out_path, max_cells=128, *, meshes=None, quality=None, max_tris=None)`;
  `build_meshes(asset, max_cells=128, bake_colors=False, quality=None, max_tris=None) ->
  list[PartMesh]` builds once for both (decimated), `decimate_meshes(asset, meshes, quality="medium",
  max_tris=None)` decimates `quality="full"` meshes later, and `mesh_summary`, `format_decimation`,
  `format_edges`, `read_glb`, `accessor_data`, `validate_glb(path) -> [problems]` are public.
  `out_path` may be a folder (`<folder>/<asset>.glb`) or have no suffix; another suffix is an error.
  `max_cells` must be an integer 8..512; one group lattice is capped at 40M nodes.
- `quality` is a preset of `export.QUALITY` = `{full: None, high: 1/8, medium: 1/4, low: 1/2}`: the
  most a collapse may move the surface, in cells of the part (so it follows `max_cells` and the thin
  part refinement); `full` keeps every surface-nets triangle. `None` means `medium`, or `full` with
  `bake_colors` (vertex colors need the dense vertices); `bake_colors` with another quality or with
  `max_tris` is an error, as is `max_tris` with `full`. `max_tris` (integer ≥ 4) is a budget for the
  whole asset: when the `quality` pass leaves more, the tolerance doubles (up to 6 times) until the
  budget is met.
- `PartMesh` records decimation: `source_triangles` (surface-nets count, `None` when not decimated;
  `triangles_before` falls back to `triangle_count`), `tolerance` (m, after any budget relaxing),
  `max_error` and `source_error` (m, measured: largest |SDF| at the vertices and triangle centers of
  the decimated and of the surface-nets mesh). `mesh_summary` columns: PART, VERTS, TRIS (exported),
  BEFORE (surface nets), CELL, DEV_MM (`max_error` in mm, `-` when not decimated), VOLUME; the total
  row sums VERTS, TRIS, BEFORE and takes the largest DEV_MM. `format_decimation` prints
  `DECIMATE  <before> -> <after> triangles (<n>x fewer); largest distance from the SDF surface …
  (tolerance …, surface nets alone …)` or `DECIMATE  off: …`.
- Thin parts get a finer mesh: a blend group whose thinnest member or cutter is under 3 base cells
  uses `cell = max(thinnest / 3, base / 4)`; other groups keep the SPEC cell exactly.
- Lattice: nodes centered on each group's AABB (symmetric parts mesh symmetrically), node parity per
  axis keeps flat faces off nodes, the two outer layers are forced outside (closed meshes); vertices
  get 2 Newton projection steps, clamped to their cell; normals are the central-difference gradient
  (h = cell/4), area-weighted face normals where it vanishes. Quads split along the shorter diagonal.
- `per_part=False` joins the per-part meshes into one node named after the asset, one primitive per
  material; it is not a boolean union (faces inside overlaps stay).
- A blend group with several materials is one mesh with one primitive per material, sharing the
  POSITION/NORMAL accessors; each triangle takes the material of the member nearest its center.
- `bake_colors`: colors come from `render.surface_color` (patterns match the renders), each vertex
  averaging a 3×3 tangent patch one cell wide with mpp = cell. glTF multiplies baseColorFactor by
  COLOR_0, so COLOR_0 = linear baked color / factor with factor = per-channel max(linear base,
  linear pattern_color): factor × COLOR_0 is the baked color exactly. Without baking, factors are
  sRGB → linear of the style color.
- OBJ `Kd`/`Ks` are the style's sRGB values as written; `Ks` = shine, `Ns 24` when shine > 0.
- Extras: node `extras.parts` (members), `asset.extras {asset, style}`, bufferView targets.
- Gaps: edges sharper than ~90° are slightly sawtoothed (within 0.42 cell), vertex normals are
  averaged across sharp edges (vertices are shared, so a sharp edge shades as a narrow strip, not a
  hard crease). Sawtoothed edges that do not run along the lattice (the crate's 45° braces) keep
  more triangles after decimation, since straightening them would move the surface by more than
  the tolerance.

### 9.1 Decimation (`decimate.py`)

`decimate(parts: list[DecimationInput], tolerance, max_tris=None) -> list[DecimationResult]`: each
input is one closed part mesh (positions, triangles, SDF-gradient normals, per-triangle material
labels) with its group SDF and cell; `tolerance` is in cells. All parts are decimated in one
priority order but collapses never join two parts. Results keep the surviving input vertex and
triangle ids (export carries `vertex_part`/`tri_part` through them), the normals, the tolerance
used and the measured errors. Pure numpy, no loop over edges:

- **Quadrics:** each vertex sums the area-weighted squared distance to its original triangles'
  planes in 6-D (position, normal × 1 cell; Garland & Heckbert 1998). The collapse optimum solves
  the 6 × 6 system (vectorized Cholesky, regularized toward the edge midpoint by 1e-3 × area so a
  flat quadric gives the midpoint), clamped to the part's box. The normal term makes the strip of
  triangles along a sharp edge expensive to widen, so edges stay crisp; flat faces are free.
- **Rounds:** rank candidates by RMS 6-D error in half-octave buckets with a fixed hash breaking ties
  (plain index order would give few local minima), take an independent set whose neighborhoods
  (triangles touching either end) are disjoint — local minima within two rings, 3 passes — check
  them, apply all at once. Per-edge optima are cached until an end moves; vertex arrays are compacted
  as they shrink. Candidate filter: RMS 6-D error ≤ 2 × tolerance; the checks hold the real distance.
- **Checks** (a failure retries the collapse onto either end, then waits until a vertex of its
  neighborhood moves): link condition (exactly 2 common neighbors) and valences ≥ 3 after the
  collapse; vertices of non-manifold edges or with several fans never move (`nonmanifold_vertices`);
  seam vertices between materials only collapse along the seam; the box rule (a vertex within
  min(5 mm, tolerance) of a box side stays within it, the box never grows); no degenerate triangle,
  no triangle turning by over ~78°, unit face normal · sum of its vertex normals > 0.3 (the new
  vertex's normal is the SDF gradient there); |SDF| ≤ tolerance at the new vertex and at the center
  and a near-corner point (barycentric 0.75/0.125/0.125) of every changed triangle; a changed
  triangle whose interpolated normal near a corner is over 20° off the SDF normal there is at most
  1.5 cells wide. Collapses with zero quadric error between equal normals (inside a flat face) skip
  the SDF probes.
- **Stop:** no candidate left, or the last 10 rounds removed under 1% of the triangles; with
  `max_tris`, a pass that ends above the budget doubles the tolerance (≤ 6 times) and the relaxed
  passes stop at the budget. Deterministic (fixed hash, no randomness).

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
`cell/4`; blocked when the sample is below the heightfield (walls included) of the cell it is over or
inside the geometry of a solid object over that cell (see the objects note below). Reports the first
blocking cell.

Report (`report.py`): `PATH`, `REACHABILITY`, `SOFTLOCKS`, `SIGHT` (for each `visible` check and
spawn→goal), `MAP` (ASCII with column/row rulers: markers as letters in a legend, path as `*`),
`HEIGHTS` (ASCII numbers). Same facts as JSON.

Render (`render.py`): iso PNG via PIL polygons (painter's algorithm, back to front, column prisms
with top + two visible sides, shaded by tile kind with style colors), same projection as
`camera.view_basis("iso")`; objects composited from `sightline.render.render_sprite` (fallback: a
labeled box of the object's footprint); markers as labeled pins; critical path as a polyline.
Plus a top-down PNG with a grid and coordinate labels. Both with captions.

Implementation notes (levels):

- API: `sightline.levels.load_level(path_or_name, root=None) -> Level`, `play(level) -> dict`,
  `format_play(report) -> str`, `run_level_checks(level, report, entries=None) -> [CheckResult]`,
  `render_level(level, out_dir, report=None) -> {"iso": Path, "top": Path}`.
- Defaults: `cell` 1.0, `height_unit` 0.5, player `{step 0.5, jump_up 1.0, jump_gap 2, max_drop
  3.0, height 1.8}`. `spawn` is a required marker; `goal` is required by no_softlocks,
  path_length, jumps and max_fall. `heights` rows may carry trailing `#` comments. Limits: grid
  ≤ 256 × 256, ≤ 8 keys, cols × rows × 2^keys ≤ 2^19 bot states.
- Path cost and `path_length` = horizontal + |vertical| meters of every move. An extra move
  **hop** climbs onto an adjacent ledge with step < rise ≤ jump_up and counts as a jump. Jumps cross
  only non-standable cells and land on the first standable one; walls can be cleared only when
  their top ≤ takeoff + jump_up; solid objects can never be jumped over; ladder moves ignore the
  height difference both ways, including drops beyond max_drop.
- Segment wording (2 decimals, ASCII minus): `walk E 4 cells (0,0)→(4,0)`,
  `jump E over water 2 cells (0,0)→(3,0)`, `jump up +1.00 m E (a)→(b)`,
  `climb +1.50 m (ladder) E (a)→(b)`, `drop -2.00 m E (a)→(b)`, `pick key_a (1,0)`,
  `open door_a with key_a (4,0)`, `reach goal (5,0)`.
- Softlocks are grouped into regions (cell sets per key subset) with keys held, entry and the
  closest failed exit as the reason; when the goal is unreachable from spawn the report says so
  once instead of listing every cell.
- Keys are markers referenced by a door or standing on a `k` tile (a `k` without a marker is an
  error); doors may not share names with markers.
- Check forms beyond the table: `reachable: {to, from}`; `path_length: N` = max; `jumps: N` =
  exactly N; `max_fall: {max}`; `why` as a sibling key or inside the mapping; `order` also fails
  when an item can be bypassed (the goal is reachable without the door).
- Level files are read with `style.read_yaml` like assets (YAML 1.2 booleans, duplicate keys are
  errors), and every level error location carries `, line N`.
- Line of sight: eyes at `player.height − 0.1` m, samples every cell/4; doors do not block sight;
  the report gives the tightest clearance when visible.
- Objects: the asset origin goes on the anchor cell center at terrain height; `rotate` is one yaw
  angle (= `rotate: [0, deg, 0]`); the footprint is every cell whose center lies inside or on the
  edge of the rotated asset AABB in x/z (a 2 × 2 m object blocks 3 × 3 cells) and that holds some of
  its geometry; height = AABB top. The geometry per cell comes from the asset's voxel occupancy
  (`loose`, 64 cells along its longest side, every voxel in the cell under its rotated center), as
  world-y spans clamped to the asset's exact bottom and top (`LevelObject.columns`, `cover`). A solid
  object blocks walking and jumping (`cells`) on the footprint cells where its geometry reaches below
  `player.height + jump_up` above the cell's ground; cells with geometry only higher up (sails,
  bridges) are walkable (JSON `objects[].overhead`, "walkable under overhangs"). Sight is blocked
  where a sample lies inside one of the spans of its cell (`Level.object_spans`), and the clearance
  is measured to the span below the line. Solid objects may not cover markers or doors (error).
  Off-grid cells are dropped with a warning; a missing asset becomes a 1-cell, 1 m placeholder with
  a warning (an error with did_you_mean when a close name exists).
- Report adds OBJECTS and WARNINGS sections (JSON: grid, player, keys, doors, objects, warnings);
  SIGHT covers spawn→goal and every visible/not_visible pair.
- Iso render: painter's order with a rank buffer, so path parts behind nearer walls/objects are
  drawn as thin dashes and marker pins stay on top; sprites are requested at the level's 1× mpp
  and composited on a 2× canvas that is box-reduced; images are at least 360 px wide; captions wrap
  and carry load warnings and sprite notes; tile colors come from style materials or palette
  entries named after the tile kind (or `tile_<kind>`), built-in colors otherwise. Painter's order
  for multi-cell objects is approximate, and the fixed camera hides north/west faces and anything
  behind tall walls (the top view shows everything). Objects drawn without a sprite are one box per
  blocking cell. The top view fills each blocking cell, outlines their union and marks overhang-only
  cells with a thin inner square.

## 11. CLI (`cli.py`)

```
sightline new <asset> [--style S] [--brief TEXT] [--force]   # template; the brief's sentences become TODO checks
sightline render <asset> [--views a,b] [--res N] [--out DIR] [--no-human]
sightline inspect <asset> [--json] [--res N] [--compare A[,B]]
sightline check [<asset>...] [--json]                    # no args: all assets; exit 1 on FAIL, 2 on error
sightline diff <asset> [--against REV|path]              # default: git HEAD version of the file
sightline sheet [<asset>...] [--out PATH] [--res N] [--view V]   # library sheet at shared scale
sightline export <asset> [--format glb|obj] [--merged] [--bake-colors] [--max-cells N]
                 [--quality full|high|medium|low] [--max-tris N] [--out PATH]
sightline ref <image> [--view V] [--height M] [--mask PATH] [--json]   # analyze a reference alone
sightline match <asset> [--index I] [--fit-view] [--res N] [--out DIR] [--json]
sightline level render|play|check <level> [--json] [--out DIR]
sightline styles [name]                                   # styles, materials, palettes
sightline shapes                                          # shape and anchor reference
```

`<asset>` is a name (`assets/<name>.yaml` under root) or a path. Every command accepts `--root`
(before or after the command) and prints the **absolute** paths of the files it wrote as
`wrote <path>`. Exit codes: 0 ok, 1 a check failed, 2 an error. Errors print
`SightlineError.format()` to stderr; argparse usage errors also exit 2; an unexpected exception
prints its traceback plus an "internal error" line and exits 2. stdout/stderr are reconfigured to
UTF-8 (reports contain → · — °, which the Windows console code page cannot encode).

- Default outputs: `render`/`match`/`export` → `<root>/out/<asset>/`, `sheet` →
  `<root>/out/library.png`, `level render` → `<root>/out/levels/<level>/`.
- `new` refuses to overwrite without `--force`, validates the name and the style, writes a template
  (brief, style, a ground part `base` and a `body`, commented examples of on/align/repeat/mirror/
  cut, checks grounded/connected/size and one TODO line per brief sentence with a suggested check
  kind) and loads it once to prove it is valid.
- `check` prints `ASSET <name> (<path>)` and the check lines per asset and, for several assets,
  `ASSETS n: p passed, f failed, e errors`. An asset without checks is not inspected. `--json`
  prints a list of `{asset, path, error, passed, results: [{kind, passed, measured, expected,
  message, why, data}]}`.
- `diff` uses `git -C <file dir> show <rev>:./<file>` and `loader.load_asset_text`; a file not yet
  in that revision, an unknown revision, a missing git or a refused repository are SightlineErrors
  with a hint (`--against path/to/old.yaml`). `--against` that names an existing file compares with
  that file instead. An error in the old copy is reported as `in --against <path>: ...` (or
  `in git <rev>:<file>: ...`), not against the working file.
- `export` builds the meshes once (decimated: `--quality` default medium, `full` with
  `--bake-colors`; `--max-tris` budget), prints `EXPORT  <asset>: <format>, <n> part nodes, max_cells
  N, quality Q[, max_tris N]`, `mesh_summary`, the `DECIMATE` line, a `MANIFOLD` line
  (`export.edge_report`: open edges and edges shared by more than two triangles per mesh; reported,
  not repaired) and, for glb, the result of `validate_glb` (`glTF structure OK`); `--bake-colors`
  with `--format obj`, with `--quality` other than full or with `--max-tris` is an error; `--merged`
  with obj writes one `o` object.
- `match --fit-view` prints `format_fit` and matches from the best view. `--res` of render, sheet,
  inspect and match must be an integer from 16 to 4096 (SightlineError otherwise).
- `new` lists every matching check kind per brief sentence (`matches_reference` first), adds a
  commented `reference:` entry when the brief mentions a picture (the `refs/...` path it names), and
  its `next:` line repeats `--root` when one was given.
- `level play` prints the bot report followed by the level's check lines and exits 1 when one
  fails (so a broken level fails in scripts); `level check` prints only the checks;
  `level play --json` adds `checks` to the report.

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

### 12.2a Implementation notes (reference)

- Signatures: `analyze_reference(image_path, view="front", height=None, mask=None, *, style=None,
  root=None)` (style names the nearest material per color; root shortens the image path);
  `match_asset(asset, index=0, res=256, out_dir=None, *, view=None)` writes `match_<index>.png`
  only when `out_dir` is given (the CLI passes `out/<asset>`), `view` overrides the reference's
  view; `fit_view(asset, index=0, res=64, top=5) -> [{view, azimuth, elevation, iou, alias}]`
  ('top' has azimuth None). Also public: `format_fit`, `format_grid`, `label_components`,
  `shade_lambert`, `srgb_to_lab`, `delta_e`.
- Analysis keys add `analyzed_px`, `mask.{source, reasons, dropped_specks, border_touch,
  bottom_touch, background, tolerance, border_spread}`, `scale.{height, unit 'm'|'%', note}`,
  `size`, `proportions.{fill, symmetry_lr, center, widest_band}`, `colors[].{center,
  nearest_material, material_delta_e}`, `thin_details`, and `unknowns` as `{topic, question,
  default}`. Images over 1600 px are box-reduced by an integer factor first.
- Confidence: border color spread (MAD) ≤ 6 high, ≤ 14 medium, else low; the object touching the
  top/left/right border > 2% → medium, > 15% → low; coverage > 60% → medium, > 85% → low; more than
  20 dropped specks → medium, counting only specks that are neither within 8 px of the object nor
  smaller than 0.05% of the image (JPEG edge noise). Touching only the bottom edge adds a "cropped?"
  question instead. The hidden-depth default is the median band width of the lower half when that
  is under 90% of the full width (sails or arms widen a picture). An
  explicit mask file is always high and is not component-filtered. Object regions painted in the
  background color become holes.
- Grids: ≤ 32 columns and ≤ 48 rows; the cell is a readable step (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6,
  8 × 10^k); x = 0 is a column center, y = 0 a row edge; the dict adds `unit` and `axes`, `x0`/`y0`
  are the left and **top** edges; text rows are labelled with cell centers.
- k-means: luminance-quantile init and 8 iterations plus deterministic re-seeding of empty or
  duplicate clusters at the worst-fitting pixel; clusters closer than ΔE 2 merge, clusters under 1%
  are not listed.
- Matching: bottoms aligned; horizontally the bbox centers first, then the shift with the best IoU
  within ±25% of the width, refined to a quarter pixel (`align.shift`). With `height` in elevated
  views (iso, orbit elevation ≠ 0) the picture height is multiplied by the model's projected/world
  height ratio. Regions are labelled without the 1-px band of each difference mask that touches the
  other silhouette (the dilation/resampling fringe), then get their own band pixels back, so a real
  region next to an edge does not merge with the fringe along the whole outline. They need ≥ 1% of
  the reference area **and** a mean thickness ≥ 1.5 grid px; each gets size, thickness, `depth` (its
  largest extent normal to the nearest part's side, used for "by ~X m") and a hint ("a part is
  missing here", "`p` may be too small on the <side> by ~X m", "`p` is too big …", "`p` is not in
  the reference", "`p` sticks out here"). `size` (the SIZE line and `aspect`) uses the model's exact
  AABB extent along the view axes when they are world axes and agree with the silhouette within
  1.5 px (else the silhouette bbox minus its pixel of dilation) and the reference's full-resolution
  mask bbox at the reference scale, not the resampled grid. Region bboxes use world axes for axis views and `h`/`v`
  (image-plane meters) for oblique views.
- Per-part colors add `delta_e_base` (base vs reference) next to `suggested = base × reference /
  rendered`; `rendered` is a flat Lambert from the style light (no patterns/shadows/AO/gamma). For
  flat concept art use the reference color, for lit pictures `suggested`.
- `fit_view` crops both silhouettes to their bboxes and scales the longer side to 64 px keeping the
  aspect, compares them bottom-aligned and centered; results carry alias names. Round objects give
  near-ties at many azimuths.
- The match PNG's model panel is `render.render_views(asset, [view], res, scale_figure=False)`,
  falling back to the flat Lambert with outlines on any error (`report["model_panel"]` says which).

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
