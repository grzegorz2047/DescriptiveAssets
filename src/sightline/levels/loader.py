"""Read and validate `levels/<name>.yaml` into a `Level`.

Every problem is a SightlineError pointing at the YAML path (`levels/x.yaml: markers.goal`)
with a hint; nothing invalid reaches the navigation bot.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..errors import SightlineError, did_you_mean
from .model import (
    DEFAULT_LEGEND,
    MAX_KEYS,
    STANDABLE_KINDS,
    TILE_KINDS,
    Door,
    Level,
    LevelObject,
    Player,
)

TOP_KEYS = ("level", "brief", "style", "cell", "height_unit", "player", "heights", "tiles", "legend",
            "markers", "doors", "objects", "checks")
PLAYER_KEYS = ("step", "jump_up", "jump_gap", "max_drop", "height")
DOOR_KEYS = ("at", "key")
OBJECT_KEYS = ("asset", "at", "rotate", "solid")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_SIDE = 256
MAX_STATES = 1 << 19  # cells x key subsets explored by the bot
PLACEHOLDER_HEIGHT = 1.0
VOXEL_CELLS = 64  # occupancy grid of an object's asset: cells along its longest side


def resolve_root(root=None, near: Path | None = None) -> Path:
    """Project root in the same order assets use (`sightline.style.resolve_root`): explicit `root`, the
    nearest ancestor of the level file with styles/, $SIGHTLINE_ROOT, the CWD's nearest ancestor with
    styles/, the repo next to the package."""
    from ..style import resolve_root as shared_resolve_root

    return shared_resolve_root(root, start=near)


def _resolve_path(path_or_name, root) -> tuple[Path, Path]:
    text = str(path_or_name)
    candidate = Path(text)
    is_path = candidate.suffix in (".yaml", ".yml") or "/" in text or "\\" in text
    if is_path:
        path = candidate.resolve()
        if not path.is_file():
            raise SightlineError(f"level file not found: {text}", hint="give a path to a .yaml file or a level name under levels/")
        return path, resolve_root(root, near=path.parent)
    base = resolve_root(root)
    path = base / "levels" / f"{text}.yaml"
    if not path.is_file():
        names = sorted(p.stem for p in (base / "levels").glob("*.yaml")) if (base / "levels").is_dir() else []
        raise SightlineError(
            f"level '{text}' not found in {base / 'levels'}",
            hint=did_you_mean(text, names) or (f"available: {', '.join(names)}" if names else "no levels yet"),
        )
    return path, base


def _label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _check_keys(data: dict, allowed, where: str, what: str) -> None:
    for key in data:
        if key not in allowed:
            raise SightlineError(
                f"unknown key '{key}' in {what}",
                where=where,
                hint=did_you_mean(key, allowed) or f"allowed keys: {', '.join(allowed)}",
            )


def _number(value, where: str, name: str, minimum: float | None = 0.0, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SightlineError(f"{name} must be a number, got {value!r}", where=where, hint=f"e.g. {name}: 1.0")
    if positive and value <= 0:
        raise SightlineError(f"{name} must be > 0, got {value}", where=where, hint="use a positive number of meters")
    if minimum is not None and value < minimum:
        raise SightlineError(f"{name} must be >= {minimum:g}, got {value}", where=where, hint="negative values are not allowed here")
    return float(value)


def _cell(value, where: str, rows: int, cols: int) -> tuple[int, int]:
    ok = isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(v, int) and not isinstance(v, bool) for v in value)
    if not ok:
        raise SightlineError(f"expected a cell [col, row], got {value!r}", where=where, hint="cells are two integers, e.g. [3, 1] = column 3, row 1")
    c, r = int(value[0]), int(value[1])
    if not (0 <= c < cols and 0 <= r < rows):
        raise SightlineError(
            f"cell [{c}, {r}] is outside the {cols} x {rows} grid",
            where=where,
            hint=f"columns are 0..{cols - 1} (west to east), rows are 0..{rows - 1} (north to south)",
        )
    return c, r


def _grid_lines(value, where: str, field: str, strip_comments: bool) -> list[list[str]]:
    """Split a grid block into rows of tokens; `heights` rows may carry trailing `# comments`."""
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, list):
        lines = [" ".join(str(v) for v in item) if isinstance(item, list) else str(item) for item in value]
    else:
        raise SightlineError(f"{field} must be a block of text rows", where=f"{where}: {field}",
                             hint=f"write `{field}: |` followed by one indented line per row")
    rows = []
    for line in lines:
        if strip_comments:
            line = line.split("#", 1)[0]
        tokens = line.split()
        if tokens:
            rows.append(tokens)
    if not rows:
        raise SightlineError(f"{field} is empty", where=f"{where}: {field}", hint="add at least one row of cells")
    width = len(rows[0])
    for i, tokens in enumerate(rows):
        if len(tokens) != width:
            raise SightlineError(
                f"{field} row {i} has {len(tokens)} cells but row 0 has {width}",
                where=f"{where}: {field} row {i}",
                hint="every row needs the same number of space-separated cells",
            )
    if len(rows) > MAX_SIDE or width > MAX_SIDE:
        raise SightlineError(f"{field} is {width} x {len(rows)} cells", where=f"{where}: {field}",
                             hint=f"levels are limited to {MAX_SIDE} x {MAX_SIDE} cells")
    return rows


def _parse_heights(value, where: str) -> np.ndarray:
    rows = _grid_lines(value, where, "heights", strip_comments=True)
    out = np.zeros((len(rows), len(rows[0])), dtype=np.int64)
    for r, tokens in enumerate(rows):
        for c, tok in enumerate(tokens):
            try:
                out[r, c] = int(tok)
            except ValueError:
                raise SightlineError(
                    f"height '{tok}' is not an integer",
                    where=f"{where}: heights row {r}, col {c}",
                    hint="heights are integer steps of height_unit, e.g. 2 = 2 x 0.5 m",
                ) from None
    return out


def _parse_legend(value, where: str) -> dict[str, str]:
    legend = dict(DEFAULT_LEGEND)
    if value is None:
        return legend
    if not isinstance(value, dict):
        raise SightlineError("legend must be a mapping char -> kind", where=f"{where}: legend", hint='e.g. legend: {"%": floor}')
    for char, kind in value.items():
        char = str(char)
        if len(char) != 1 or char.isspace():
            raise SightlineError(f"legend key '{char}' must be a single non-space character",
                                 where=f"{where}: legend", hint='quote special characters: legend: {"%": floor}')
        if kind not in TILE_KINDS:
            raise SightlineError(f"unknown tile kind '{kind}' for '{char}'", where=f"{where}: legend.{char}",
                                 hint=did_you_mean(kind, TILE_KINDS) or f"kinds: {', '.join(TILE_KINDS)}")
        legend[char] = kind
    return legend


def _legend_text(legend: dict[str, str]) -> str:
    return ", ".join(f"'{ch}' {kind}" for ch, kind in legend.items())


def _parse_tiles(value, where: str, legend: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    rows = _grid_lines(value, where, "tiles", strip_comments=False)
    tiles = np.empty((len(rows), len(rows[0])), dtype="<U1")
    kinds = np.empty(tiles.shape, dtype="<U6")
    for r, tokens in enumerate(rows):
        for c, tok in enumerate(tokens):
            if len(tok) != 1:
                raise SightlineError(
                    f"tile '{tok}' is {len(tok)} characters",
                    where=f"{where}: tiles row {r}, col {c}",
                    hint="write one character per cell, separated by spaces: . . # #",
                )
            if tok not in legend:
                raise SightlineError(
                    f"unknown tile '{tok}'",
                    where=f"{where}: tiles row {r}, col {c}",
                    hint=f"legend: {_legend_text(legend)}; or add it: legend: {{\"{tok}\": floor}}",
                )
            tiles[r, c] = tok
            kinds[r, c] = legend[tok]
    return tiles, kinds


def _parse_player(value, where: str) -> Player:
    if value is None:
        return Player()
    if not isinstance(value, dict):
        raise SightlineError("player must be a mapping", where=f"{where}: player", hint="e.g. player: {step: 0.5, jump_up: 1.0}")
    _check_keys(value, PLAYER_KEYS, f"{where}: player", "player")
    p = Player()
    for key in ("step", "jump_up", "max_drop"):
        if key in value:
            setattr(p, key, _number(value[key], f"{where}: player.{key}", key))
    if "height" in value:
        p.height = _number(value["height"], f"{where}: player.height", "height", positive=True)
        if p.height <= 0.1:
            raise SightlineError("player height must be > 0.1 m", where=f"{where}: player.height", hint="a human is 1.8")
    if "jump_gap" in value:
        gap = value["jump_gap"]
        if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
            raise SightlineError(f"jump_gap must be a whole number of cells >= 0, got {gap!r}",
                                 where=f"{where}: player.jump_gap", hint="e.g. jump_gap: 2")
        p.jump_gap = int(gap)
    return p


def _resolve_style(name, root: Path, where: str) -> str:
    if not isinstance(name, str):
        raise SightlineError("style must be a name", where=f"{where}: style", hint="e.g. style: default")
    styles = root / "styles"
    if name != "default" and not (styles / f"{name}.yaml").is_file():
        names = sorted(p.stem for p in styles.glob("*.yaml")) if styles.is_dir() else []
        raise SightlineError(f"unknown style '{name}'", where=f"{where}: style",
                             hint=did_you_mean(name, names) or f"available: {', '.join(names) or 'default'}")
    return name


def _load_asset(name: str, root: Path, where: str, warnings: list[str]):
    """Load an object's asset when the asset loader exists; None (placeholder) otherwise."""
    assets = root / "assets"
    if not (assets / f"{name}.yaml").is_file():
        names = sorted(p.stem for p in assets.glob("*.yaml")) if assets.is_dir() else []
        close = did_you_mean(name, names)
        if close:
            raise SightlineError(f"asset '{name}' not found", where=where, hint=close)
        warnings.append(f"{where.split(': ', 1)[-1]}: asset '{name}' not found under assets/; using a 1-cell placeholder box")
        return None
    try:
        from ..loader import load_asset
    except (ImportError, AttributeError):
        warnings.append(f"{where.split(': ', 1)[-1]}: asset loader unavailable; '{name}' drawn as a 1-cell placeholder box")
        return None
    return load_asset(name, root)


def _asset_bounds(asset) -> tuple[np.ndarray, np.ndarray] | None:
    """World AABB of an asset's solid parts (asset space), None if the asset API is not available."""
    try:
        lo, hi = asset.scene().bounds(include_helpers=False)
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    except (AttributeError, TypeError, NotImplementedError):
        pass
    try:
        boxes = [p.world_aabb() for p in asset.parts if p.op != "cut" and not getattr(p, "helper", False)]
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    except (AttributeError, TypeError, ValueError):
        return None


def footprint_cells(col: int, row: int, cell: float, rotate: float, lo, hi, rows: int,
                    cols: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Cells the rotated asset footprint (asset AABB in x/z) covers when placed on cell (col, row).

    A cell is covered when its center lies inside or on the footprint edge (it is at least half covered),
    so the bot never walks through visible geometry; the anchor cell always counts.
    Returns (covered cells inside the grid, covered cells outside the grid), both in row-major order.
    """
    reach = int(math.ceil(float(np.max(np.abs(np.concatenate([lo[[0, 2]], hi[[0, 2]]])))) * math.sqrt(2) / cell)) + 1
    cc, rr = np.meshgrid(np.arange(col - reach, col + reach + 1), np.arange(row - reach, row + reach + 1))
    dx = (cc - col) * cell
    dz = (rr - row) * cell
    th = math.radians(rotate)
    # Inverse of the yaw rotation (LevelObject.yaw_matrix transposed): world offset -> asset frame.
    lx = dx * math.cos(th) - dz * math.sin(th)
    lz = dx * math.sin(th) + dz * math.cos(th)
    tol = 1e-6
    inside = (lx >= lo[0] - tol) & (lx <= hi[0] + tol) & (lz >= lo[2] - tol) & (lz <= hi[2] + tol)
    inside |= (cc == col) & (rr == row)
    in_grid = (cc >= 0) & (cc < cols) & (rr >= 0) & (rr < rows)

    def listed(mask):
        return sorted(((int(c), int(r)) for c, r in zip(cc[mask], rr[mask])), key=lambda t: (t[1], t[0]))

    return listed(inside & in_grid), listed(inside & ~in_grid)


def _occupied_points(asset, cache: dict) -> tuple[np.ndarray, float]:
    """Asset-space centers of the voxels occupied by the asset's solids, and the voxel size."""
    if asset.name not in cache:
        from ..voxel import occupancy

        grid = occupancy(asset.scene(), max_cells=VOXEL_CELLS)
        # Voxels inside the geometry, plus the "near" voxels of parts thinner than a voxel (sails, poles); the
        # near shell around thicker geometry would leak half a voxel into the neighbouring level cells.
        strict, loose = grid.strict.any(axis=0), grid.loose.any(axis=0)
        grown = np.pad(strict, 1)
        for ax in range(3):
            grown = grown | np.roll(grown, 1, axis=ax) | np.roll(grown, -1, axis=ax)
        keep = strict | (loose & ~grown[1:-1, 1:-1, 1:-1])
        cache[asset.name] = (grid.origin + grid.cell * np.argwhere(keep), float(grid.cell))
    return cache[asset.name]


def object_columns(obj: LevelObject, level: Level, pts: np.ndarray, voxel: float, candidates: list[tuple[int, int]]
                   ) -> tuple[dict[tuple[int, int], list[tuple[float, float]]], list[tuple[int, int]]]:
    """What the object occupies over each candidate cell (footprint_cells), from its voxel occupancy.

    Returns (columns, blocking). `columns[cell]` = world-y spans of its geometry over the cell: every
    occupied voxel goes to the cell under its (rotated) center, consecutive voxel layers merge, and the
    spans are clamped to the asset's exact bottom and top. `blocking` = the candidate cells with "low"
    geometry (between the cell's ground and player.height + jump_up above it): the player walks under
    overhangs, and cells of the bounding box with no geometry at all stay free."""
    columns: dict[tuple[int, int], list[tuple[float, float]]] = {cell: [] for cell in candidates}
    if len(pts) == 0 or not candidates:
        return columns, []
    cell = level.cell
    anchor = np.array([(obj.col + 0.5) * cell, obj.base, (obj.row + 0.5) * cell])
    world = pts @ obj.yaw_matrix().T + anchor
    c = np.floor(world[:, 0] / cell).astype(np.int64)
    r = np.floor(world[:, 2] / cell).astype(np.int64)
    wanted = np.zeros((level.rows, level.cols), dtype=bool)
    for cc, rr in candidates:
        wanted[rr, cc] = True
    keep = (c >= 0) & (c < level.cols) & (r >= 0) & (r < level.rows)
    keep[keep] = wanted[r[keep], c[keep]]
    if not keep.any():
        return columns, []
    pts, world, c, r = pts[keep], world[keep], c[keep], r[keep]
    cid = r * level.cols + c
    y0 = float(pts[:, 1].min())
    iy = np.rint((pts[:, 1] - y0) / voxel).astype(np.int64)
    layers = int(iy.max()) + 2
    key = np.unique(cid * layers + iy)  # sorted by cell, then layer
    kc, ky = key // layers, key % layers
    start = np.ones(len(key), dtype=bool)
    start[1:] = (kc[1:] != kc[:-1]) | (ky[1:] - ky[:-1] > 1)
    first = np.flatnonzero(start)
    last = np.append(first[1:], len(key)) - 1
    lo_y, hi_y = obj.base + float(obj.bounds[0][1]), obj.base + float(obj.bounds[1][1])
    for a, b in zip(first.tolist(), last.tolist()):
        rr, cc = divmod(int(kc[a]), level.cols)
        span = (max(obj.base + y0 + (ky[a] - 0.5) * voxel, lo_y), min(obj.base + y0 + (ky[b] + 0.5) * voxel, hi_y))
        columns[(cc, rr)].append((round(float(span[0]), 4), round(float(span[1]), 4)))
    # Walking: a candidate cell blocks when the object has geometry there between the cell's ground and
    # player.height + jump_up above it; with geometry only higher up (an overhang) the cell stays walkable.
    ground = level.heights_m()[r, c]
    vy = world[:, 1]
    low = (vy - voxel / 2 < ground + level.player.height + level.player.jump_up) & (vy + voxel / 2 > ground + 1e-6)
    blocking = {(k % level.cols, k // level.cols) for k in np.unique(cid[low]).tolist()}
    return columns, [cell_ for cell_ in candidates if cell_ in blocking]


def _parse_objects(value, where: str, level: Level) -> list[LevelObject]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SightlineError("objects must be a list", where=f"{where}: objects", hint="e.g. objects: [{asset: crate, at: [2, 3]}]")
    objects = []
    voxels: dict = {}  # asset name -> occupied voxel centers, shared by copies of one asset
    for i, entry in enumerate(value):
        label = f"objects[{i}]"
        if not isinstance(entry, dict):
            raise SightlineError("object must be a mapping", where=f"{where}: {label}", hint="e.g. {asset: crate, at: [2, 3], solid: true}")
        _check_keys(entry, OBJECT_KEYS, f"{where}: {label}", "object")
        name = entry.get("asset")
        if not isinstance(name, str) or not name:
            raise SightlineError("object needs `asset: <name>`", where=f"{where}: {label}", hint="name of a file in assets/ without .yaml")
        label = f"{label} ({name})"
        if "at" not in entry:
            raise SightlineError("object needs `at: [col, row]`", where=f"{where}: {label}", hint="e.g. at: [2, 3]")
        c, r = _cell(entry["at"], f"{where}: {label}.at", level.rows, level.cols)
        rotate = _number(entry.get("rotate", 0), f"{where}: {label}.rotate", "rotate", minimum=None)
        solid = entry.get("solid", True)
        if not isinstance(solid, bool):
            raise SightlineError("solid must be true or false", where=f"{where}: {label}.solid", hint="solid: true blocks movement and sight")
        asset = _load_asset(name, level.root, f"{where}: {label}", level.warnings)
        bounds = _asset_bounds(asset) if asset is not None else None
        obj = LevelObject(index=i, asset=name, col=c, row=r, rotate=rotate, solid=solid, base=level.y(c, r),
                          asset_obj=asset, placeholder=bounds is None, source=label, bounds=bounds)
        if bounds is None:
            obj.cells = obj.cover = [(c, r)]
            obj.height = PLACEHOLDER_HEIGHT
            obj.columns = {(c, r): [(obj.base, obj.base + PLACEHOLDER_HEIGHT)]}
        else:
            lo, hi = bounds
            obj.height = max(float(hi[1]), 0.05)
            candidates, outside = footprint_cells(c, r, level.cell, rotate, lo, hi, level.rows, level.cols)
            pts, voxel = _occupied_points(asset, voxels)
            columns, obj.cells = object_columns(obj, level, pts, voxel, candidates)
            obj.columns = {cell: spans for cell, spans in columns.items() if spans}
            obj.cover = [cell for cell in candidates if columns[cell]]
            if outside:
                shown = " ".join(f"({oc},{orow})" for oc, orow in outside[:6]) + (" ..." if len(outside) > 6 else "")
                level.warnings.append(
                    f"{label}: hangs off the grid by {len(outside)} cell(s) {shown}; the bot and sight ignore them "
                    f"(footprint {hi[0] - lo[0]:.2f} x {hi[2] - lo[2]:.2f} m, rotate {rotate:g})")
        objects.append(obj)
    return objects


_GRID_WHERE = re.compile(r"^(tiles|heights) row (\d+)")
_PATH_PART = re.compile(r"([^\s.\[]+)(?:\[(\d+)\])?(?: \([^)]*\))?")


def _line_of(rest: str, lines: dict[tuple, int], text: str) -> int | None:
    """1-based YAML line of an error location such as `markers.goal`, `objects[2] (crate).at` or
    `tiles row 3, col 1` (the 4th row of the block), or None."""
    grid = _GRID_WHERE.match(rest)
    if grid:
        field, row = grid.group(1), int(grid.group(2))
        if (field, row) in lines:  # rows written as a YAML list
            return lines[(field, row)]
        start = lines.get((field,))
        if start is None:
            return None
        count = -1
        for number, line in enumerate(text.splitlines()[start:], start=start + 1):
            if (line.split("#", 1)[0] if field == "heights" else line).split():
                count += 1
                if count == row:
                    return number
        return start
    path: tuple = ()
    for seg in rest.split("."):
        m = _PATH_PART.fullmatch(seg.strip())
        if not m:
            break
        path += (m.group(1),) + ((int(m.group(2)),) if m.group(2) is not None else ())
    while path and path not in lines:
        path = path[:-1]
    return lines.get(path) if path else None


def _with_line(exc: SightlineError, where: str, lines: dict[tuple, int], text: str) -> SightlineError:
    """The same error with `, line N` added to its location (SPEC 0)."""
    loc = exc.where or ""
    if not loc.startswith(where) or ", line " in loc or ": line " in loc:
        return exc
    rest = loc[len(where):]
    line = _line_of(rest[2:], lines, text) if rest.startswith(": ") else None
    if line is None:
        return exc
    return SightlineError(exc.message, where=f"{loc}, line {line}", hint=exc.hint)


def load_level(path_or_name, root=None) -> Level:
    """Load and validate a level by name (`levels/<name>.yaml` under root) or by path.

    YAML is read like assets (`style.read_yaml`): YAML 1.2 booleans (`no`, `on` stay strings),
    duplicate keys are errors, and every error location carries its line number."""
    from ..style import read_yaml

    path, base = _resolve_path(path_or_name, root)
    where = _label(path, base)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SightlineError(f"cannot read level file: {exc}", where=where, hint="save the file as UTF-8 text") from None
    raw, lines = read_yaml(text, where)
    try:
        return _build_level(raw, path, base, where)
    except SightlineError as exc:
        raise _with_line(exc, where, lines, text) from None


def _build_level(raw, path: Path, base: Path, where: str) -> Level:
    if not isinstance(raw, dict):
        raise SightlineError("a level file must be a YAML mapping", where=where, hint="start with `level: <name>`")
    _check_keys(raw, TOP_KEYS, where, "level file")

    name = raw.get("level")
    if name != path.stem:
        raise SightlineError(f"level: {name!r} must equal the file name '{path.stem}'", where=f"{where}: level",
                             hint=f"write `level: {path.stem}`")
    brief = raw.get("brief") or ""
    if not isinstance(brief, str):
        raise SightlineError("brief must be text", where=f"{where}: brief", hint="brief: > A short description")
    style = _resolve_style(raw.get("style", "default"), base, where)
    cell = _number(raw.get("cell", 1.0), f"{where}: cell", "cell", positive=True)
    height_unit = _number(raw.get("height_unit", 0.5), f"{where}: height_unit", "height_unit", positive=True)
    player = _parse_player(raw.get("player"), where)

    if "tiles" not in raw:
        raise SightlineError("missing `tiles`", where=where, hint="add `tiles: |` with one row of space-separated chars per grid row")
    legend = _parse_legend(raw.get("legend"), where)
    tiles, kinds = _parse_tiles(raw["tiles"], where, legend)
    if "heights" in raw:
        heights = _parse_heights(raw["heights"], where)
        if heights.shape != tiles.shape:
            raise SightlineError(
                f"heights is {heights.shape[1]} x {heights.shape[0]} (cols x rows) but tiles is {tiles.shape[1]} x {tiles.shape[0]}",
                where=f"{where}: heights", hint="heights and tiles must describe the same grid")
    else:
        heights = np.zeros(tiles.shape, dtype=np.int64)

    level = Level(name=name, path=path, root=base, where=where, brief=brief.strip(), style=style, cell=cell,
                  height_unit=height_unit, player=player, heights=heights, tiles=tiles, kinds=kinds, legend=legend,
                  markers={}, doors={}, objects=[], checks=[], raw=raw)
    level.markers = _parse_markers(raw.get("markers"), where, level)
    level.doors = _parse_doors(raw.get("doors"), where, level)
    _check_key_tiles(level)
    level.objects = _parse_objects(raw.get("objects"), where, level)
    _check_object_cover(level)

    keys = level.key_names()
    if len(keys) > MAX_KEYS or level.rows * level.cols * (1 << len(keys)) > MAX_STATES:
        raise SightlineError(f"{len(keys)} keys on a {level.cols} x {level.rows} grid is too many states for the bot",
                             where=f"{where}: markers", hint=f"use at most {MAX_KEYS} keys and keep cols x rows x 2^keys <= {MAX_STATES}")

    checks = raw.get("checks") or []
    if not isinstance(checks, list):
        raise SightlineError("checks must be a list", where=f"{where}: checks", hint="checks:\n  - reachable: goal")
    level.checks = checks
    from .checks import validate_checks

    validate_checks(level)
    return level


def _parse_markers(value, where: str, level: Level) -> dict[str, tuple[int, int]]:
    if not isinstance(value, dict) or "spawn" not in value:
        raise SightlineError("markers must include `spawn`", where=f"{where}: markers",
                             hint="e.g. markers: {spawn: [0, 0], goal: [5, 1]}")
    markers = {}
    for name, cell in value.items():
        loc = f"{where}: markers.{name}"
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise SightlineError(f"bad marker name {name!r}", where=loc, hint="use lowercase letters, digits and _, starting with a letter")
        c, r = _cell(cell, loc, level.rows, level.cols)
        kind = level.kind(c, r)
        if kind not in STANDABLE_KINDS:
            raise SightlineError(
                f"marker '{name}' is on a {kind} cell ({c},{r})",
                where=loc,
                hint=f"markers must stand on a {', '.join(sorted(STANDABLE_KINDS))} cell; move it or change tiles row {r}, col {c}",
            )
        markers[name] = (c, r)
    return markers


def _parse_doors(value, where: str, level: Level) -> dict[str, Door]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SightlineError("doors must be a mapping name -> {at, key}", where=f"{where}: doors",
                             hint="e.g. doors: {door_red: {at: [4, 1], key: key_red}}")
    doors: dict[str, Door] = {}
    taken: dict[tuple[int, int], str] = {}
    for name, spec in value.items():
        loc = f"{where}: doors.{name}"
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise SightlineError(f"bad door name {name!r}", where=loc, hint="use lowercase letters, digits and _, starting with a letter")
        if name in level.markers:
            raise SightlineError(f"door '{name}' has the same name as a marker", where=loc, hint="checks refer to both by name; rename one")
        if not isinstance(spec, dict):
            raise SightlineError("door must be a mapping", where=loc, hint="e.g. {at: [4, 1], key: key_red}")
        _check_keys(spec, DOOR_KEYS, loc, "door")
        if "at" not in spec:
            raise SightlineError("door needs `at: [col, row]`", where=loc, hint="the cell of its D tile")
        c, r = _cell(spec["at"], f"{loc}.at", level.rows, level.cols)
        if level.kind(c, r) != "door":
            raise SightlineError(
                f"door '{name}' at ({c},{r}) is on a {level.kind(c, r)} tile '{level.tiles[r, c]}', not a door tile",
                where=f"{loc}.at",
                hint=f"put a door tile (D) at tiles row {r}, col {c} or move the door",
            )
        if (c, r) in taken:
            raise SightlineError(f"doors '{taken[(c, r)]}' and '{name}' share cell ({c},{r})", where=loc, hint="one door per cell")
        key = spec.get("key")
        if key is not None:
            if not isinstance(key, str) or key not in level.markers:
                raise SightlineError(f"door key '{key}' is not a marker", where=f"{loc}.key",
                                     hint=did_you_mean(str(key), list(level.markers)) or "add the key as a marker, e.g. markers: {key_red: [1, 1]}")
            if level.markers[key] == (c, r):
                raise SightlineError(f"key '{key}' lies on its own door", where=f"{loc}.key", hint="place the key elsewhere")
        taken[(c, r)] = name
        doors[name] = Door(name=name, col=c, row=r, key=key)
    return doors


def _check_key_tiles(level: Level) -> None:
    marked = set(level.markers.values())
    rows, cols = np.nonzero(level.kinds == "key")
    for r, c in zip(rows.tolist(), cols.tolist()):
        if (c, r) not in marked:
            raise SightlineError(f"key tile at ({c},{r}) has no marker", where=f"{level.where}: tiles row {r}, col {c}",
                                 hint=f"name the key: markers: {{key_name: [{c}, {r}]}}")


def _check_object_cover(level: Level) -> None:
    """Solid objects may not cover markers or doors: the bot could never stand there."""
    places = {**{n: c for n, c in level.markers.items()}, **{n: (d.col, d.row) for n, d in level.doors.items()}}
    for obj in level.objects:
        if not obj.solid:
            continue
        for name, cell in places.items():
            if cell in obj.cells:
                raise SightlineError(
                    f"solid object '{obj.asset}' covers '{name}' at ({cell[0]},{cell[1]})",
                    where=f"{level.where}: {obj.source}",
                    hint="move the object, move the marker, or set solid: false",
                )
