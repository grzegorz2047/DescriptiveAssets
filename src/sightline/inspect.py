"""Measurements of an asset (SPEC 7.1): the report dict (the JSON schema) and its text form.

The report is what an AI agent reads after every change, so every number is measured on the
geometry (voxel occupancy and ray-marched buffers), never taken from the YAML. `format_report`
prints every fact of the dict as compact aligned text, `diff_reports` lists what changed.

Lit color (part colors, contrast): the material base color times
`ambient + (1 - ambient) * max(0, n . -light_dir)` with the style light, averaged over the part's
visible iso pixels. No patterns, shadows or specular: a stable proxy for how light or dark a part
reads, independent of render.py.
"""

from __future__ import annotations

import dataclasses
import math
import string
import textwrap

import numpy as np

from .camera import VIEW_NAMES, Camera, fixed_camera, make_cameras, view_basis
from .errors import SightlineError
from .march import ViewBuffers, march, trace
from .model import Asset, Scene
from .style import display_path
from .voxel import VoxelGrid, occupancy

__all__ = [
    "inspect_asset", "format_report", "diff_reports", "measured_scene", "visible_fractions", "visible_occluders",
    "material_spots",
    "readability", "ground_contacts",
    "compare_silhouettes", "relative_luminance", "contrast_ratio", "hex_to_rgb", "REPORT_VIEWS",
]

REPORT_VIEWS = ("front", "right", "top", "iso")
HUMAN_HEIGHT = 1.8
COLOR_VIEW = "iso"  # colors and contrast
READ_VIEW = "front"  # readability (the report's `readability`)
READ_MORE_VIEWS = ("iso",)  # ... and these too (`readability_views`): thin parts can vanish only from the front
READ_SIZES = (32, 16)
READ_FULL_PX = 256  # long side of the full-resolution silhouette that readability downsamples
KEEP_PX = 4  # a part survives a downscale if it owns at least this many pixels
SIL_MAX = 32  # silhouette grid: at most this many columns and rows
COMPARE_PX = 128
LOW_VISIBLE = 0.30
SMALL_FOOTPRINT_PX = 40  # visibility of parts covering fewer pixels is measured on a finer grid...
FINE_STEPS = 4  # ... this many times finer
HIDDEN_TAG = "hidden"  # parts tagged hidden are expected to be out of sight: no low-visibility warning
SUPPORT_BAND = 0.02  # m: material this close above the ground (or below it) is a ground contact
SUPPORT_STEP = 0.01  # m: finest xz step of the ground contact search
LOW_CONTRAST = 1.5
SPOT_MIN_PX = 3  # a material patch smaller than this (at the check render size) is a speck, not a spot
MIN_SHARED_EDGE = 3  # boundary pixel pairs for two parts to count as adjacent
LETTERS = string.ascii_uppercase + string.ascii_lowercase
OTHER_LETTER = "*"  # parts beyond the 52 letters
# Silhouette cell sizes whose cell centers print exactly with 2 decimals.
_NICE_CELLS = sorted({round(0.02 * n, 6) for n in range(1, 10)} | {0.2, 0.3, 0.4, 0.5, 0.6, 0.8}
                     | {round(m * 10.0**k, 6) for k in range(0, 4) for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)})
_INDENT = " " * 12


# ----------------------------------------------------------------------------- small helpers


def _r(x, nd: int = 3) -> float:
    """Rounded float for JSON (no -0.0)."""
    v = round(float(x), nd)
    return 0.0 if v == 0 else v


def _vec(values, nd: int = 3) -> list[float]:
    return [_r(v, nd) for v in values]


def _label(asset: Asset) -> str:
    return display_path(asset.path, asset.root)


def _corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b)) / union if union else 1.0


def _check_views(views) -> list[str]:
    if isinstance(views, str):
        views = [v.strip() for v in views.split(",") if v.strip()]
    out = list(dict.fromkeys(views))
    if not out:
        raise SightlineError("no views to inspect", hint=f"views: {', '.join(VIEW_NAMES)}")
    for v in out:
        view_basis(v)  # SightlineError with did_you_mean on unknown views
    return out


def measured_scene(asset: Asset) -> tuple[Scene, list[int]]:
    """The asset's scene without render helpers, and its solid part indices (asset indices)."""
    scene = asset.scene()
    solids = scene.solid_indices(include_helpers=False)
    if not solids:
        raise SightlineError("the asset has no solid parts to measure", where=_label(asset),
                             hint="add a part with a shape and a material; cuts alone are never rendered")
    if len(solids) != len(scene.solid_indices(include_helpers=True)):
        scene = scene.only(solids)
    return scene, solids


# ----------------------------------------------------------------------------- tracing helpers


def _rays(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    origins, dirs = cam.rays()
    return origins.reshape(cam.height, cam.width, 3), dirs[0]


def _trace_hits(scene: Scene, cam: Camera, rays=None, window=None, eps: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(hit distance, inf = miss; first-hit part, -1 = miss) per pixel of the image or of a
    (r0, r1, c0, c1) window. `eps` is the hit threshold in pixels: 0.5 is the march threshold, which
    widens silhouettes by up to half a pixel; small values give exact footprints."""
    origins, direction = rays if rays is not None else _rays(cam)
    r0, r1, c0, c1 = window if window is not None else (0, cam.height, 0, cam.width)
    o = origins[r0:r1, c0:c1].reshape(-1, 3)
    t, part = trace(scene, o, np.broadcast_to(direction, o.shape), np.inf, eps * cam.mpp)
    return t.reshape(r1 - r0, c1 - c0), part.reshape(r1 - r0, c1 - c0)


def _trace_parts(scene: Scene, cam: Camera, rays=None, window=None, eps: float = 0.5) -> np.ndarray:
    """First-hit part index per pixel (-1 = miss); see _trace_hits."""
    return _trace_hits(scene, cam, rays, window, eps)[1]


def _window(cam: Camera, lo, hi, pad: int = 2):
    """Pixel window (r0, r1, c0, c1) covering a world box, or None when it is off-image."""
    pts, _ = cam.project(_corners(lo, hi))
    c0 = max(0, int(math.floor(pts[:, 0].min())) - pad)
    c1 = min(cam.width, int(math.ceil(pts[:, 0].max())) + pad)
    r0 = max(0, int(math.floor(pts[:, 1].min())) - pad)
    r1 = min(cam.height, int(math.ceil(pts[:, 1].max())) + pad)
    return None if c0 >= c1 or r0 >= r1 else (r0, r1, c0, c1)


def _crop(a: np.ndarray, win) -> np.ndarray:
    return a[win[0]:win[1], win[2]:win[3]]


# ----------------------------------------------------------------------------- voxel measurements


def _part_bounds(scene: Scene, grid: VoxelGrid, solids: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Exact SDF AABB per solid; for parts with cuts, shrunk to the occupied cells where a cut removed
    more than one cell from a side (a cut can take away an end of the part)."""
    out = {}
    for k, i in enumerate(solids):
        lo, hi = (np.array(v, dtype=np.float64) for v in scene.aabb(i))
        layer = grid.loose[k]
        if scene.cutters.get(i) and layer.any():
            first = np.empty(3)
            last = np.empty(3)
            for ax in range(3):
                idx = np.flatnonzero(layer.any(axis=tuple(a for a in range(3) if a != ax)))
                first[ax], last[ax] = idx[0], idx[-1]
            c_lo = grid.origin + grid.cell * first - grid.cell / 2
            c_hi = grid.origin + grid.cell * last + grid.cell / 2
            lo = np.where(c_lo > lo + grid.cell, c_lo, lo)
            hi = np.where(c_hi < hi - grid.cell, c_hi, hi)
        out[i] = (lo, hi)
    return out


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain) of 2D points."""
    pts = sorted(set(map(tuple, np.round(points, 9).tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _signed_distance(poly: np.ndarray, q: np.ndarray) -> float:
    """Distance from q to the edge of a CCW convex polygon: positive inside, negative outside."""
    if len(poly) < 3:
        return -float(np.linalg.norm(q - _closest_on_polygon(poly, q))) if len(poly) else -math.inf
    e = np.roll(poly, -1, axis=0) - poly
    length = np.linalg.norm(e, axis=1)
    inward = np.stack([-e[:, 1], e[:, 0]], axis=1) / length[:, None]
    d = ((q - poly) * inward).sum(axis=1)
    if np.all(d >= 0):
        return float(d.min())
    t = np.clip(((q - poly) * e).sum(axis=1) / length**2, 0.0, 1.0)
    closest = poly + t[:, None] * e
    return -float(np.min(np.linalg.norm(q - closest, axis=1)))


def _voxel_footprint(grid: VoxelGrid, union: np.ndarray) -> np.ndarray:
    """xz corners of the cells in the lowest two occupied layers (fallback support polygon)."""
    ys = np.flatnonzero(union.any(axis=(0, 2)))
    foot = union[:, ys[0]:ys[0] + 2, :].any(axis=1)  # (nx, nz)
    xs = np.flatnonzero(foot.any(axis=1))
    rows = foot[xs]
    z_first = rows.argmax(axis=1)
    z_last = rows.shape[1] - 1 - rows[:, ::-1].argmax(axis=1)
    h = grid.cell / 2
    x_c = grid.origin[0] + grid.cell * xs
    z_lo = grid.origin[2] + grid.cell * z_first - h
    z_hi = grid.origin[2] + grid.cell * z_last + h
    return np.concatenate([np.stack([x_c + dx, z], axis=1) for dx in (-h, h) for z in (z_lo, z_hi)])


def ground_contacts(scene: Scene, solids: list[int], base: float, start: float) -> tuple[np.ndarray, list[int]]:
    """Where the solids touch the ground, from the SDF: (xz points (N, 2), parts with a contact, file order).

    A part touches the ground where it has material below `base + SUPPORT_BAND` (`base` = the ground
    plane y = 0, or the asset's bottom when it floats within the grounded tolerance). Each part's
    slab is searched coarse to fine: square xz cells from `start` (m) halve down to SUPPORT_STEP; a
    cell is dropped when the part's distance at its center exceeds the cell's half-diagonal (the SDF
    is a lower bound, so no material is lost), and a cell that lies wholly inside material adds its
    corners without being refined. Thin contacts (a tyre's bottom, a 3 cm skid) are found at any
    voxel size, which the old two-lowest-voxel-layers footprint could not do.
    """
    top = base + SUPPORT_BAND
    points: list[np.ndarray] = []
    touching: list[int] = []
    for i in solids:
        lo, hi = (np.asarray(v, dtype=np.float64) for v in scene.aabb(i))
        if lo[1] > top:
            continue
        a, b = float(lo[1]), top
        span = np.maximum(hi[[0, 2]] - lo[[0, 2]], 1e-6)
        counts = np.maximum(np.ceil(span / max(start, SUPPORT_STEP)).astype(int), 1)
        step = span / counts  # (sx, sz)
        gx, gz = np.meshgrid(lo[0] + (np.arange(counts[0]) + 0.5) * step[0],
                             lo[2] + (np.arange(counts[1]) + 0.5) * step[1], indexing="ij")
        cols = np.stack([gx.ravel(), gz.ravel()], axis=1)
        found = 0
        while len(cols):
            ny = max(1, math.ceil((b - a) / max(float(step.min()), SUPPORT_STEP)))
            hy = (b - a) / (2 * ny)
            ys = a + (np.arange(ny) + 0.5) * 2 * hy
            p3 = np.empty((3, len(cols) * ny), dtype=np.float32)
            p3[0] = np.repeat(cols[:, 0], ny)
            p3[1] = np.tile(ys, len(cols))
            p3[2] = np.repeat(cols[:, 1], ny)
            d = scene.part_dist3(i, p3).reshape(len(cols), ny)
            r = math.sqrt(float((step / 2) @ (step / 2)) + hy * hy)
            full = (d <= -r).any(axis=1)  # the whole cell box lies in material at some level
            near = (d <= r).any(axis=1) & ~full
            if full.any():
                hx, hz = step / 2
                c = cols[full]
                points += [c + np.array([sx * hx, sz * hz]) for sx in (-1, 1) for sz in (-1, 1)]
                found += int(full.sum())
            if float(step.max()) <= SUPPORT_STEP * (1 + 1e-9):
                if near.any():
                    sel = np.flatnonzero(near)
                    # A distance near 0 on the sheet a coplanar cut face leaves over its hole is no material.
                    k = np.argmin(d[sel], axis=1)
                    p_best = p3.reshape(3, len(cols), ny)[:, sel, k]
                    d_best = d[sel, k]
                    phantom = _phantom(scene, i, np.ascontiguousarray(p_best), d_best,
                                       np.ones(len(sel), dtype=bool), SUPPORT_STEP)
                    keep = sel[~phantom]
                    points.append(cols[keep])
                    found += len(keep)
                break
            step = step / 2
            c = cols[near]
            cols = np.concatenate([c + np.array([sx * step[0] / 2, sz * step[1] / 2]) for sx in (-1, 1) for sz in (-1, 1)])
        if found:
            touching.append(i)
    pts = np.concatenate(points) if points else np.zeros((0, 2))
    return pts, touching


def _mass(grid: VoxelGrid, union: np.ndarray, grounded: bool, support: tuple[np.ndarray, list[str]] | None = None) -> dict:
    """Volume and center of mass (uniform density over strict occupancy) and stability.

    `support` = (ground contact xz points, names of the parts that make them) from `ground_contacts`;
    without points the lowest two voxel layers stand in for them.
    """
    n = int(np.count_nonzero(union))
    if n == 0:
        return {"volume": 0.0, "center": None, "stable": None, "support_margin": None, "support": None,
                "tips_toward": None}
    idx = np.nonzero(union)
    com = grid.origin + grid.cell * np.array([a.mean() for a in idx])
    stable, margin, names, tips = None, None, None, None
    if grounded:
        pts, names = support if support is not None and len(support[0]) else (_voxel_footprint(grid, union), [])
        hull = _convex_hull(pts)
        q = com[[0, 2]]
        margin = _signed_distance(hull, q)
        stable = bool(margin >= 0)
        if not stable:
            # Which way it tips: from the closest point of the support polygon toward the center of mass.
            closest = _closest_on_polygon(hull, q)
            dx, dz = q - closest
            tips = ("+x" if dx > 0 else "-x") if abs(dx) >= abs(dz) else ("+z" if dz > 0 else "-z")
    return {"volume": _r(n * grid.cell**3, 4), "center": _vec(com), "stable": stable,
            "support_margin": None if margin is None else _r(margin), "support": names, "tips_toward": tips}


def _closest_on_polygon(poly: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Closest point to q on the boundary of a polygon (or point set when it has fewer than 3 vertices)."""
    if len(poly) == 1:
        return poly[0]
    e = np.roll(poly, -1, axis=0) - poly
    t = np.clip(((q - poly) * e).sum(axis=1) / np.maximum((e * e).sum(axis=1), 1e-18), 0.0, 1.0)
    closest = poly + t[:, None] * e
    return closest[int(np.argmin(np.linalg.norm(q - closest, axis=1)))]


def _index_box(grid: VoxelGrid, lo, hi) -> list[tuple[int, int]] | None:
    """Index ranges [start, stop) of the cells whose centers lie in a world box (None if empty)."""
    shape = np.array(grid.shape)
    i0 = np.maximum(np.ceil((np.asarray(lo) - grid.origin) / grid.cell - 1e-9).astype(int), 0)
    i1 = np.minimum(np.floor((np.asarray(hi) - grid.origin) / grid.cell + 1e-9).astype(int) + 1, shape)
    return None if np.any(i1 <= i0) else [(int(a), int(b)) for a, b in zip(i0, i1)]


def _phantom(scene: Scene, i: int, p3: np.ndarray, d: np.ndarray, mask: np.ndarray, cell: float) -> np.ndarray:
    """Bool mask: points (among `mask`) where part i's distance `d` >= 0 comes from the 0-distance sheet a
    coplanar cut face leaves over its hole (see Scene.phantom3), not from material."""
    out = np.zeros(d.shape, dtype=bool)
    cutters = scene.cutters.get(i)
    sel = np.flatnonzero(mask & (d >= 0.0)) if cutters else ()
    if len(sel):
        field = (lambda q: scene.part_dist3(i, q))  # noqa: E731
        out[sel] = scene.phantom3(field, cutters, np.ascontiguousarray(p3[:, sel]), d[sel], 0.25 * cell)
    return out


def _cell_distance(grid: VoxelGrid, layer: int, pts: np.ndarray) -> np.ndarray:
    """Distance (m) from points (N, 3) to the strict cells of one layer: nearest boundary cell center
    minus half a cell (used where the SDF bound is a cut artifact)."""
    occ = grid.strict[layer]
    if not occ.any() or len(pts) == 0:
        return np.full(len(pts), np.inf)
    inner = occ.copy()
    padded = np.pad(occ, 1)
    for ax in range(3):
        for step in (-1, 1):
            inner &= np.roll(padded, step, axis=ax)[1:-1, 1:-1, 1:-1]
    src = grid.origin + grid.cell * np.argwhere(occ & ~inner)
    out = np.empty(len(pts))
    chunk = max(1, 4_000_000 // len(src))
    for s in range(0, len(pts), chunk):
        q = pts[s:s + chunk]
        d2 = ((q[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
        out[s:s + chunk] = np.sqrt(d2.min(axis=1))
    return np.maximum(out - grid.cell / 2, 0.0)


def _contacts(scene: Scene, grid: VoxelGrid) -> list[tuple[int, int, int]]:
    """(layer a, layer b, shared strict cells) for parts closer than one voxel cell.

    Two parts touch when d_a + d_b < cell at some cell center near both. The sum is never below
    the true gap and equals it between the closest points, so faces that meet exactly halfway
    between cell centers still count (a loose & loose test misses them by float noise).
    """
    cell = grid.cell
    out = []
    n = len(grid.solids)
    aabbs = [scene.aabb(i) for i in grid.solids]
    for a in range(n):
        for b in range(a + 1, n):
            lo = np.maximum(aabbs[a][0], aabbs[b][0]) - cell
            hi = np.minimum(aabbs[a][1], aabbs[b][1]) + cell
            box = _index_box(grid, lo, hi)
            if box is None:
                continue
            sl = tuple(slice(s, e) for s, e in box)
            if not (grid.loose[a][sl] & grid.loose[b][sl]).any():
                axes = [(grid.origin[k] + cell * np.arange(*box[k])).astype(np.float32) for k in range(3)]
                gx, gy, gz = np.meshgrid(*axes, indexing="ij")
                p3 = np.stack([gx.ravel(), gy.ravel(), gz.ravel()])
                ds = [scene.part_dist3(grid.solids[k], p3) for k in (a, b)]
                close = ds[0] + ds[1] < cell
                for k, d in zip((a, b), ds):
                    # A small distance on the sheet left by a coplanar cut face is not material.
                    close &= ~_phantom(scene, grid.solids[k], p3, d, close, cell)
                if not close.any():
                    continue
            shared = int(np.count_nonzero(grid.strict[a][sl] & grid.strict[b][sl]))
            out.append((a, b, shared))
    return out


def _components(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[r] for r in sorted(groups)]


def _surface_points(grid: VoxelGrid, layers: list[int]) -> np.ndarray:
    """World centers of the boundary cells of these layers' union (strict, else loose)."""
    occ = grid.strict[layers].any(axis=0)
    if not occ.any():
        occ = grid.loose[layers].any(axis=0)
    if not occ.any():
        return np.zeros((0, 3))
    inner = occ.copy()
    padded = np.pad(occ, 1)
    for ax in range(3):
        for step in (-1, 1):
            inner &= np.roll(padded, step, axis=ax)[1:-1, 1:-1, 1:-1]
    return grid.origin + grid.cell * np.argwhere(occ & ~inner)


_DIRECTIONS = ("above", "below", "right of", "left of", "in front of", "behind")
_DIR_ANCHOR = {"above": "top", "below": "bottom", "right of": "right", "left of": "left",
               "in front of": "front", "behind": "back"}
_DIR_MOVE = {"above": "down", "below": "up", "right of": "left (-x)", "left of": "right (+x)",
             "in front of": "back (-z)", "behind": "forward (+z)"}


def _direction(lo_a, hi_a, lo_b, hi_b) -> str:
    """Where box a lies relative to box b: the side with the largest separation."""
    seps = (lo_a[1] - hi_b[1], lo_b[1] - hi_a[1], lo_a[0] - hi_b[0], lo_b[0] - hi_a[0],
            lo_a[2] - hi_b[2], lo_b[2] - hi_a[2])
    return _DIRECTIONS[int(np.argmax(seps))]


def _descent(scene: Scene, i: int, p, cell: float) -> np.ndarray:
    """Unit vector from a point toward part i's nearest surface (minus the SDF gradient)."""
    h = 0.25 * cell
    offs = np.vstack([np.eye(3) * h, -np.eye(3) * h])
    d = scene.eval_part(i, np.asarray(p, dtype=np.float64)[None, :] + offs).astype(np.float64)
    grad = d[:3] - d[3:]
    length = float(np.linalg.norm(grad))
    return -grad / length if length > 1e-12 else np.array([0.0, -1.0, 0.0])


_MOVE_WORDS = (("right (+x)", "left (-x)"), ("up", "down"), ("forward (+z)", "back (-z)"))
_SIDE_DIR = {"above": (1, 1.0), "below": (1, -1.0), "right of": (0, 1.0), "left of": (0, -1.0),
             "in front of": (2, 1.0), "behind": (2, -1.0)}


def _anchor_fits(target, side: str, tol: float) -> bool:
    """Whether `on: target.<side anchor>` puts a part on the target's surface on that world side: the
    anchor's local direction must face that way and the bbox anchor must lie on the shape (not for a
    cone's side, a rotated part or a polygon corner)."""
    axis, sign = _SIDE_DIR[side]
    uvw = np.zeros(3)
    uvw[_SIDE_DIR[side][0]] = sign
    world = target.matrix @ uvw
    if world[axis] * sign < 0.999:
        return False
    return abs(float(target.sdf(target.anchor(_DIR_ANCHOR[side])[None, :])[0])) <= tol


def _move_text(move) -> str:
    """'1.32 m left (-x) and 0.62 m down': the components of a move vector, largest first."""
    v = np.asarray(move, dtype=np.float64)
    size = float(np.linalg.norm(v))
    order = sorted(range(3), key=lambda k: (-abs(v[k]), k))
    words = [f"{abs(v[k]):.2f} m {_MOVE_WORDS[k][0 if v[k] > 0 else 1]}" for k in order
             if abs(v[k]) >= 0.005 and abs(v[k]) >= 0.05 * size]
    return " and ".join(words) or "0.00 m"


def _floating(scene, grid, solids, comps, bounds, tol) -> tuple[list[dict], list[list[int]]]:
    """Contact components that do not reach the ground component, with gap and nearest part."""
    on_ground = [k for k, i in enumerate(solids) if bounds[i][0][1] <= tol]
    if on_ground:
        base = [c for c in comps if any(k in on_ground for k in c)]
    else:  # nothing reaches the ground: measure against the lowest piece (ungrounded is its own fact)
        lowest = min(range(len(solids)), key=lambda k: (bounds[solids[k]][0][1], k))
        base = [c for c in comps if lowest in c]
    base_layers = sorted(k for c in base for k in c)
    out, members = [], []
    for comp in comps:
        if comp in base:
            continue
        pts = _surface_points(grid, comp)
        if len(pts) == 0:
            pts = np.array([scene.parts[solids[k]].position for k in comp])
        p3 = np.ascontiguousarray(pts.T, dtype=np.float32)
        d_float = np.min([scene.part_dist3(solids[k], p3) for k in comp], axis=0).astype(np.float64)
        best, nearest, best_pt, toward = math.inf, base_layers[0], pts[0], None
        for k in base_layers:
            i = solids[k]
            d = scene.part_dist3(i, p3).astype(np.float64)
            fake = _phantom(scene, i, p3, d, np.ones(d.shape, dtype=bool), grid.cell)
            if fake.any():  # the bound only sees the sheet over a cut: measure to the occupied cells
                d[fake] = np.maximum(d[fake], _cell_distance(grid, k, pts[fake]))
            j = int(np.argmin(d + d_float))
            if d[j] + d_float[j] < best:
                best, nearest, best_pt = float(d[j] + d_float[j]), k, pts[j]
                toward = None if fake[j] else _descent(scene, i, best_pt, grid.cell)
        lo_c = np.min([bounds[solids[k]][0] for k in comp], axis=0)
        hi_c = np.max([bounds[solids[k]][1] for k in comp], axis=0)
        near_i = solids[nearest]
        gap = max(best, 0.0)
        if toward is None:  # toward the nearest occupied cell of the grounded part
            centers = grid.origin + grid.cell * np.argwhere(grid.strict[nearest])
            vec = centers[np.argmin(((centers - best_pt) ** 2).sum(axis=1))] - best_pt
            toward = vec / max(float(np.linalg.norm(vec)), 1e-12)
        out.append({
            "parts": [scene.parts[solids[k]].name for k in comp],
            "gap": _r(gap),
            "nearest": scene.parts[near_i].name,
            "direction": _direction(lo_c, hi_c, *bounds[near_i]),
            "move": _vec(np.where(np.abs(toward * gap) < 0.005, 0.0, toward * gap)),
        })
        members.append([solids[k] for k in comp])
    return out, members


def _removed_volume(scene: Scene, c: int, targets: list[int], samples: int = 40) -> float:
    """Volume (m^3) of the targets' primitives inside cutter c (midpoint rule on a grid of `samples`
    cells along the longest side of the cutter/targets overlap box)."""
    if not targets:
        return 0.0
    lo, hi = (np.array(v, dtype=np.float64) for v in scene.aabb(c))
    t_lo = np.min([scene.aabb(j)[0] for j in targets], axis=0)
    t_hi = np.max([scene.aabb(j)[1] for j in targets], axis=0)
    lo, hi = np.maximum(lo, t_lo), np.minimum(hi, t_hi)
    ext = hi - lo
    if np.any(ext <= 0):
        return 0.0
    n = np.maximum(1, np.ceil(ext / (ext.max() / samples) - 1e-9)).astype(int)
    step = ext / n
    axes = [(lo[a] + step[a] * (np.arange(n[a]) + 0.5)).astype(np.float32) for a in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    p3 = np.stack([gx.ravel(), gy.ravel(), gz.ravel()])
    inside_target = np.zeros(p3.shape[1], dtype=bool)
    for j in targets:
        inside_target |= scene.parts[j].sdf3(p3) < 0
    inside = inside_target & (scene.parts[c].sdf3(p3) < 0)
    return float(np.count_nonzero(inside)) * float(np.prod(step))


# ----------------------------------------------------------------------------- visibility


def _erode(mask: np.ndarray) -> np.ndarray:
    """Pixels whose 4 neighbours are set too (pixels outside the array count as unset)."""
    m = mask.copy()
    m[1:, :] &= mask[:-1, :]
    m[:-1, :] &= mask[1:, :]
    m[:, 1:] &= mask[:, :-1]
    m[:, :-1] &= mask[:, 1:]
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return m


def _solid_visibility(scene, solids, views, cams, bufs, rays) -> dict:
    """{part: {view: (fraction, pixels alone, main occluder or -1)}}.

    The fraction is measured on the part's own footprint eroded by one pixel (when anything is left):
    every silhouette is up to half a pixel wider per edge, so without the erosion an occluder's edge
    would cover the dilated rim of a part seen through a close-fitting opening (a door panel in its
    door cut) and the fraction would drop with the pixel size instead of with real occlusion.
    """
    out = {i: {} for i in solids}
    for view in views:
        cam, buf = cams[view], bufs[view]
        for i in solids:
            win = _window(cam, *scene.aabb(i))
            if win is None:
                out[i][view] = (0.0, 0, -1)
                continue
            alone = _trace_parts(scene.only([i]), cam, rays[view], win) >= 0
            n_alone = int(np.count_nonzero(alone))
            full = _crop(buf.part, win)
            if 0 < n_alone < SMALL_FOOTPRINT_PX:
                # A few pixels say little (eroded, a 3 x 3 part keeps one pixel that a propeller blade can
                # cover): measure again on a grid FINE_STEPS times finer over the part's window.
                fine = _fine_camera(cam, win, FINE_STEPS)
                alone = _trace_parts(scene.only([i]), fine) >= 0
                full = _trace_parts(scene, fine)
            core = _erode(alone)
            if not core.any():  # thinner than 3 px: the whole footprint
                core = alone
            seen = core & (full == i)
            others = full[core & ~seen]
            others = others[others >= 0]
            occluder = int(np.bincount(others).argmax()) if others.size else -1
            n_core = int(np.count_nonzero(core))
            frac = np.count_nonzero(seen) / n_core if n_core else 0.0
            out[i][view] = (float(frac), n_alone, occluder)
    return out


def _fine_camera(cam: Camera, win, k: int) -> Camera:
    """A camera over the pixel window (r0, r1, c0, c1) of `cam` with k x k pixels per pixel of it."""
    r0, r1, c0, c1 = win
    center = cam.pixel_to_world((c0 + c1) / 2.0, (r0 + r1) / 2.0)
    return fixed_camera(cam.view, center, cam.mpp / k, (c1 - c0) * k, (r1 - r0) * k, cam.near)


def _cut_visibility(scene, c, targets, views, cams, bufs, rays, removed: float | None = None) -> dict:
    """{view: (fraction, footprint pixels, main occluder or -1, fraction without other parts)} for cut c.

    Footprint = pixels covered by both the cutter (rendered alone as a solid) and its targets' primitives
    (rendered without cuts). A footprint pixel shows the cut when its first hit is a target on the
    carved surface (carved hit points lie inside the cutter), or when the ray passes through the hole:
    no hit, or a hit behind the cutter. The last value repeats the measurement with only the targets
    in the scene, which tells "faces away" apart from "hidden by another part". A cut that removes nothing
    has no carved surface (a target face it merely touches does not count): 0 in every view.
    """
    if removed is None:
        removed = _removed_volume(scene, c, targets)
    if removed <= 0:
        return {view: (0.0, 0, -1, 0.0) for view in views}
    cutter = scene.parts[c]
    alone = Scene([dataclasses.replace(cutter, op="union", material="cut", cut_from=[])])
    prims = Scene([scene.parts[j] for j in targets]) if targets else None
    own = scene.only(targets) if targets else None
    tol = 0.05  # in pixels: exact footprints, float slack for "inside the cutter"
    out = {}
    for view in views:
        cam, buf = cams[view], bufs[view]
        win = _window(cam, *scene.aabb(c))
        if win is None or prims is None:
            out[view] = (0.0, 0, -1, 0.0)
            continue
        # Exact footprints (tiny hit threshold): the march's half-pixel widening would add a border ring.
        t_enter, alone_part = _trace_hits(alone, cam, rays[view], win, eps=tol)
        foot = (alone_part >= 0) & (_trace_parts(prims, cam, rays[view], win, eps=tol) >= 0)
        n_foot = int(np.count_nonzero(foot))
        if n_foot == 0:
            out[view] = (0.0, 0, -1, 0.0)
            continue
        origins, direction = rays[view]
        o = _crop(origins, win)

        def shows_cut(part: np.ndarray, t: np.ndarray) -> np.ndarray:
            ok = foot & (part < 0)  # through the hole to the background
            sel = foot & (part >= 0)
            if sel.any():
                inside = np.zeros_like(foot)
                inside[sel] = cutter.sdf(o[sel] + t[sel][:, None] * direction) <= tol * cam.mpp
                on_target = np.isin(part, targets)
                ok |= sel & on_target & inside  # carved surface
                ok |= sel & ~on_target & ~inside & (t > t_enter)  # another part seen through the hole
            return ok

        seen = shows_cut(_crop(buf.part, win), _crop(buf.depth, win))
        full = _crop(buf.part, win)
        others = full[foot & ~seen]
        others = others[(others >= 0) & ~np.isin(others, targets)]
        occluder = int(np.bincount(others).argmax()) if others.size else -1
        t_own, part_own = _trace_hits(own, cam, rays[view], win)
        free = shows_cut(part_own, t_own)
        out[view] = (np.count_nonzero(seen) / n_foot, n_foot, occluder, np.count_nonzero(free) / n_foot)
    return out


def _paint_visibility(scene, i: int, cam, buf) -> float:
    """Share of a paint's surface on its target parts that a view shows (the rest is hidden by other parts)."""
    targets = next(t for ci, t in scene.paints if ci == i) if any(ci == i for ci, _ in scene.paints) else []
    if len(targets) == 0:
        return 0.0
    own = scene.only([int(j) for j in targets])
    alone = int((own.surface_ids(march(own, cam)) == i).sum())
    return int((scene.surface_ids(buf) == i).sum()) / alone if alone else 0.0


def visible_fractions(asset: Asset, names, views, res: int = 128) -> dict[str, dict[str, float]]:
    """Visible fraction per part name (solid or cut) and view, rendered at `res` (for checks on views
    the report does not hold)."""
    views = _check_views(views)
    scene, solids = measured_scene(asset)
    cams = make_cameras(views, *scene.bounds(include_helpers=False), res=res)
    bufs = {v: march(scene, cams[v]) for v in views}
    rays = {v: _rays(cams[v]) for v in views}
    out = {}
    for name in names:
        i = asset.part_index(name)
        if scene.parts[i].is_paint:
            out[name] = {v: _r(_paint_visibility(scene, i, cams[v], bufs[v])) for v in views}
            continue
        if scene.parts[i].is_solid:
            vis = _solid_visibility(scene, [i], views, cams, bufs, rays)[i]
        else:
            targets = [j for j in solids if i in scene.cutters.get(j, [])]
            vis = _cut_visibility(scene, i, targets, views, cams, bufs, rays)
        out[name] = {v: _r(vis[v][0]) for v in views}
    return out


def visible_occluders(asset: Asset, name: str, views, res: int = 128) -> dict[str, str | None]:
    """Per view, the part that hides most of the hidden pixels of part/cut `name` (None = nothing in
    front: the part faces away or is outside the view)."""
    views = _check_views(views)
    scene, solids = measured_scene(asset)
    cams = make_cameras(views, *scene.bounds(include_helpers=False), res=res)
    bufs = {v: march(scene, cams[v]) for v in views}
    rays = {v: _rays(cams[v]) for v in views}
    i = asset.part_index(name)
    if scene.parts[i].is_solid:
        vis = _solid_visibility(scene, [i], views, cams, bufs, rays)[i]
    else:
        targets = [j for j in solids if i in scene.cutters.get(j, [])]
        vis = _cut_visibility(scene, i, targets, views, cams, bufs, rays)
    return {v: (scene.parts[vis[v][2]].name if vis[v][2] >= 0 else None) for v in views}


# ----------------------------------------------------------------------------- color and contrast


def material_spots(asset: Asset, view: str, materials=None, res: int = 512, box=None) -> list[dict]:
    """Separate patches of the given materials that `view` shows, largest first.

    `materials` (names; None = every accent material, as `structure` classes them: lights, decals) are
    read off the surface, paints included. `box` = [h0, h1, v0, v1] in m along the view's right and up
    axes from the origin keeps only the patches whose centre lies inside. Patches under SPOT_MIN_PX
    pixels are left out. Each patch: `px`, `center` [h, v] m, `size` [w, h] m, `parts` (largest first).
    A pixel-score can be satisfied by a model with four eyes; this counts them.
    """
    from .reference import label_components
    from .structure import material_classes

    (view,) = _check_views([view])
    scene, _ = measured_scene(asset)
    cam = make_cameras([view], *scene.bounds(include_helpers=False), res=res)[view]
    buf = march(scene, cam)
    surf = buf.part.copy()
    if scene.paints and buf.hit.any():
        surf[buf.hit] = scene.paint_ids(buf.point[buf.hit], buf.part[buf.hit])
    parts = scene.parts
    if materials is None:
        wanted = np.flatnonzero(material_classes(asset, surf)[:-1] == 2)
    else:
        names = {materials} if isinstance(materials, str) else set(materials)
        wanted = np.array([i for i, p in enumerate(parts) if p.material in names], dtype=np.int64)
    sel = buf.hit & np.isin(surf, wanted)
    labels, sizes = label_components(sel)
    out = []
    for k in np.flatnonzero(sizes >= SPOT_MIN_PX):
        yy, xx = np.nonzero(labels == k)
        world = cam.pixel_to_world(xx.mean() + 0.5, yy.mean() + 0.5)
        h, v = float(world @ cam.right), float(world @ cam.up)
        if box is not None and not (box[0] <= h <= box[1] and box[2] <= v <= box[3]):
            continue
        ids, cnt = np.unique(surf[yy, xx], return_counts=True)
        order = np.argsort(-cnt)
        out.append({"px": int(sizes[k]), "center": [round(h, 3), round(v, 3)],
                    "size": [round(float(xx.max() - xx.min() + 1) * cam.mpp, 3), round(float(yy.max() - yy.min() + 1) * cam.mpp, 3)],
                    "parts": [parts[int(ids[j])].name for j in order[:3] if ids[j] >= 0]})
    out.sort(key=lambda s: -s["px"])
    return out


def hex_to_rgb(text: str) -> tuple[float, float, float]:
    text = text.lstrip("#")
    return tuple(int(text[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


def _to_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(min(max(float(c), 0.0), 1.0) * 255)):02x}" for c in rgb)


def relative_luminance(rgb) -> float:
    """WCAG 2 relative luminance of an sRGB color with channels in 0..1."""
    c = np.asarray(rgb, dtype=np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def contrast_ratio(rgb_a, rgb_b) -> float:
    """WCAG contrast ratio (1..21) of two sRGB colors."""
    la, lb = relative_luminance(rgb_a), relative_luminance(rgb_b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _lit_colors(asset: Asset, scene: Scene, solids, buf: ViewBuffers) -> tuple[dict, dict]:
    """Mean lit color per visible solid and material pixel shares (one view)."""
    n_parts = len(scene.parts)
    style = asset.style
    to_light = -np.asarray(style.light_dir, dtype=np.float64)
    to_light /= max(np.linalg.norm(to_light), 1e-12)
    pidx = buf.part[buf.hit].astype(np.int64)
    lam = np.clip(buf.normal[buf.hit].astype(np.float64) @ to_light, 0.0, None)
    shade = style.ambient + (1.0 - style.ambient) * lam
    base = np.zeros((n_parts, 3))
    for i in solids:
        base[i] = style.material(scene.parts[i].material).color
    lit = base[pidx] * shade[:, None]
    count = np.bincount(pidx, minlength=n_parts)
    sums = np.stack([np.bincount(pidx, weights=lit[:, ch], minlength=n_parts) for ch in range(3)], axis=1)
    mean = {i: sums[i] / count[i] for i in solids if count[i]}
    total = int(count.sum())
    shares: dict[str, float] = {}
    sid = scene.surface_ids(buf)[buf.hit] if scene.paints else pidx  # paints colour parts of solids
    scount = np.bincount(sid, minlength=n_parts)
    for i in np.flatnonzero(scount):
        mat = scene.parts[int(i)].material
        shares[mat] = shares.get(mat, 0.0) + scount[i] / total
    ordered = dict(sorted(shares.items(), key=lambda kv: (-kv[1], kv[0])))
    return mean, ordered


def _adjacent_pairs(part: np.ndarray) -> list[tuple[int, int]]:
    """Part pairs sharing at least MIN_SHARED_EDGE 4-neighbour boundary pixel pairs."""
    if part.size == 0 or part.max() < 0:
        return []
    n = int(part.max()) + 1
    keys = []
    for a, b in ((part[:, :-1], part[:, 1:]), (part[:-1, :], part[1:, :])):
        m = (a >= 0) & (b >= 0) & (a != b)
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        keys.append(lo * n + hi)
    uniq, counts = np.unique(np.concatenate(keys), return_counts=True)
    return [(int(k // n), int(k % n)) for k, c in zip(uniq, counts) if c >= MIN_SHARED_EDGE]


# ----------------------------------------------------------------------------- silhouettes


def _axis_name(vec, fallback: str) -> str:
    k = int(np.argmax(np.abs(vec)))
    if abs(abs(vec[k]) - 1.0) < 1e-9:
        return ("" if vec[k] > 0 else "-") + "xyz"[k]
    return fallback


def _pixel_uv(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """View-plane coordinates (m) of pixel centers: u per column (image right), v per row (image up)."""
    uc, vc = float(cam.center @ cam.right), float(cam.center @ cam.up)
    u = uc + (np.arange(cam.width) + 0.5 - cam.width / 2.0) * cam.mpp
    v = vc + (cam.height / 2.0 - np.arange(cam.height) - 0.5) * cam.mpp
    return u, v


def _row_offset(cam: Camera) -> float:
    """Rows of upright views start at the ground (y = 0 is a row boundary); others center on v = 0."""
    return 0.0 if abs(cam.up[1] - 1.0) < 1e-9 else 0.5


def _extent(cam: Camera, buf: ViewBuffers):
    u, v = _pixel_uv(cam)
    return u[buf.hit.any(axis=0)], v[buf.hit.any(axis=1)], _row_offset(cam)


def _choose_cell(extents) -> float:
    """Smallest nice cell size that fits every view's silhouette in SIL_MAX x SIL_MAX cells."""
    for cell in _NICE_CELLS:
        fits = True
        for u, v, off in extents:
            if len(u) == 0:
                continue
            cols = math.floor(u.max() / cell + 0.5) - math.floor(u.min() / cell + 0.5) + 1
            rows = math.floor(v.max() / cell + off) - math.floor(v.min() / cell + off) + 1
            if cols > SIL_MAX or rows > SIL_MAX:
                fits = False
                break
        if fits:
            return cell
    return _NICE_CELLS[-1]


def _silhouette(cam: Camera, buf: ViewBuffers, cell: float, label_of: np.ndarray, chars: np.ndarray) -> dict:
    """Part map downsampled to square cells; a cell is filled when >= 50% covered and shows the
    letter of the part covering most of it. `label_of[part + 1]` = 1-based label (0 = background)."""
    u, v = _pixel_uv(cam)
    off = _row_offset(cam)
    col_axis, row_axis = _axis_name(cam.right, "u"), _axis_name(cam.up, "v")
    base = {"cols": 0, "rows": 0, "cell": _r(cell, 4), "x0": 0.0, "y0": 0.0, "axes": [col_axis, row_axis],
            "row_labels": [], "legend": {}, "grid": []}
    if not buf.hit.any():
        return base
    k = np.floor(u / cell + 0.5).astype(np.int64)
    j = np.floor(v / cell + off).astype(np.int64)
    hit_cols, hit_rows = buf.hit.any(axis=0), buf.hit.any(axis=1)
    kmin, kmax = int(k[hit_cols].min()), int(k[hit_cols].max())
    jmin, jmax = int(j[hit_rows].min()), int(j[hit_rows].max())
    ncols, nrows = kmax - kmin + 1, jmax - jmin + 1
    inside = ((j >= jmin) & (j <= jmax))[:, None] & ((k >= kmin) & (k <= kmax))[None, :]
    cell_id = (jmax - j)[:, None] * ncols + (k - kmin)[None, :]
    total = np.bincount(cell_id[inside], minlength=nrows * ncols)
    n_labels = int(label_of.max()) + 1
    lab = label_of[buf.part + 1]
    sel = inside & buf.hit
    counts = np.bincount(cell_id[sel] * n_labels + lab[sel], minlength=nrows * ncols * n_labels)
    counts = counts.reshape(nrows * ncols, n_labels)[:, 1:]
    filled = (total > 0) & (2 * counts.sum(axis=1) >= total)
    grid = np.where(filled, chars[counts.argmax(axis=1)], ".").reshape(nrows, ncols)
    # Trim empty border rows and columns.
    rows_used = np.flatnonzero((grid != ".").any(axis=1))
    cols_used = np.flatnonzero((grid != ".").any(axis=0))
    if len(rows_used) == 0:
        return base
    r0, r1, c0, c1 = rows_used[0], rows_used[-1] + 1, cols_used[0], cols_used[-1] + 1
    grid = grid[r0:r1, c0:c1]
    kmin += int(c0)
    jmax -= int(r0)
    nrows, ncols = grid.shape
    sign = -1.0 if row_axis.startswith("-") else 1.0
    centers = [(jmax - r + 0.5 - off) * cell for r in range(nrows)]
    letters = sorted({str(ch) for ch in grid.ravel()} - {"."}, key=lambda ch: (ch == OTHER_LETTER, LETTERS.find(ch)))
    return {
        "cols": int(ncols), "rows": int(nrows), "cell": _r(cell, 4),
        "x0": _r((kmin - 0.5) * cell), "y0": _r((jmax - nrows + 1 - off) * cell),
        "axes": [col_axis, row_axis], "row_labels": [_r(sign * c) for c in centers],
        "legend": {ch: None for ch in letters}, "grid": [" ".join(str(ch) for ch in row) for row in grid],
    }


# ----------------------------------------------------------------------------- readability, compare


def _readability(scene: Scene, view: str, sizes) -> dict:
    lo, hi = scene.bounds(include_helpers=False)
    _, right, up = view_basis(view)
    corners = _corners(lo, hi)
    su, sv = float(np.ptp(corners @ right)), float(np.ptp(corners @ up))
    longest = max(su, sv, 1e-6)
    center = (lo + hi) / 2.0
    near = float(np.linalg.norm(hi - lo)) + 1.0
    n_parts = len(scene.parts)
    at = {}
    for n in sizes:
        n = int(n)
        b = max(4, math.ceil(READ_FULL_PX / n))
        mpp_n = longest / n
        cols = math.ceil(su / mpp_n - 1e-6) + 2
        rows = math.ceil(sv / mpp_n - 1e-6) + 2
        cam = fixed_camera(view, center, mpp_n / b, cols * b, rows * b, near)
        part = _trace_parts(scene, cam)
        full = part >= 0
        low = full.reshape(rows, b, cols, b).mean(axis=(1, 3)) >= 0.5
        upscaled = np.repeat(np.repeat(low, b, axis=0), b, axis=1)
        block = (np.arange(rows * b) // b)[:, None] * cols + (np.arange(cols * b) // b)[None, :]
        counts = np.bincount(block[full] * n_parts + part[full], minlength=rows * cols * n_parts)
        owner = counts.reshape(rows * cols, n_parts).argmax(axis=1)[low.ravel()]
        owned = np.bincount(owner, minlength=n_parts)
        seen = sorted({int(i) for i in np.unique(part[full])})
        at[n] = {
            "iou": _r(_iou(upscaled, full)),
            "parts_kept": [scene.parts[i].name for i in seen if owned[i] >= KEEP_PX],
            "parts_lost": [scene.parts[i].name for i in seen if owned[i] < KEEP_PX],
        }
    return {"view": view, "at_px": at}


def readability(asset: Asset, view: str = READ_VIEW, sizes=READ_SIZES) -> dict:
    """Silhouette at N px on the longer image side (box-filtered from a 256 px render, >= 50% coverage),
    upscaled back: IoU vs full resolution and the parts that keep >= 4 px."""
    view = _check_views([view])[0]
    scene, _ = measured_scene(asset)
    for n in sizes:
        if isinstance(n, bool) or not isinstance(n, int) or n < 4:
            raise SightlineError(f"readability size must be an integer >= 4 px, got {n!r}", hint="e.g. px: 32")
    return _readability(scene, view, sizes)


def _compare_view(sa: Scene, sb: Scene, view: str, px: int) -> dict:
    fwd, right, up = view_basis(view)
    info = []
    for s in (sa, sb):
        lo, hi = s.bounds(include_helpers=False)
        c = _corners(lo, hi)
        u, v = c @ right, c @ up
        info.append((u.min(), u.max(), v.min(), v.max(), float((c @ fwd).mean()), float(np.linalg.norm(hi - lo))))
    near = max(i[5] for i in info) + 1.0
    # True scale: shared mpp, horizontal centers aligned, world v (the ground for upright views) shared.
    span_u = max(i[1] - i[0] for i in info)
    v_lo, v_hi = min(i[2] for i in info), max(i[3] for i in info)
    mpp = max(span_u, v_hi - v_lo, 1e-6) / px
    w, h = math.ceil(span_u / mpp) + 4, math.ceil((v_hi - v_lo) / mpp) + 4
    masks = []
    for s, (u0, u1, _, _, fc, _) in zip((sa, sb), info):
        center = right * (u0 + u1) / 2 + up * (v_lo + v_hi) / 2 + fwd * fc
        masks.append(_trace_parts(s, fixed_camera(view, center, mpp, w, h, near)) >= 0)
    iou_true = _iou(*masks)
    # Normalized: each scaled to the same pixel height, bottoms and centers aligned.
    heights = [max(i[3] - i[2], 1e-9) for i in info]
    w = math.ceil(max((i[1] - i[0]) / hh * px for i, hh in zip(info, heights))) + 4
    masks = []
    for s, (u0, u1, v0, v1, fc, _), hh in zip((sa, sb), info, heights):
        center = right * (u0 + u1) / 2 + up * (v0 + v1) / 2 + fwd * fc
        masks.append(_trace_parts(s, fixed_camera(view, center, hh / px, w, px + 4, near)) >= 0)
    return {"iou_true_scale": _r(iou_true), "iou_normalized": _r(_iou(*masks))}


def compare_silhouettes(asset: Asset, other: Asset, views=("front",), px: int = COMPARE_PX) -> dict:
    """{view: {iou_true_scale, iou_normalized}} of two assets' silhouettes, aligned at the ground and
    the horizontal center; normalized = both scaled to equal height."""
    views = _check_views(views)
    sa, _ = measured_scene(asset)
    sb, _ = measured_scene(other)
    return {view: _compare_view(sa, sb, view, px) for view in views}


# ----------------------------------------------------------------------------- the report


def inspect_asset(asset: Asset, res: int = 128, views=REPORT_VIEWS, max_cells: int = 96, compare=()) -> dict:
    """Measure an asset and return the report dict (SPEC 7.1); JSON-serializable and deterministic.

    `compare` lists other assets (Asset objects or names/paths resolved against the asset's root).
    """
    views = _check_views(views)
    if isinstance(res, bool) or not isinstance(res, (int, np.integer)) or not 16 <= res <= 4096:
        raise SightlineError(f"res must be an integer from 16 to 4096, got {res!r}", hint="the default is 128 px")
    res = int(res)
    scene, solids = measured_scene(asset)
    parts = scene.parts
    cuts = [i for i, p in enumerate(parts) if p.is_cut and not p.helper]
    grid = occupancy(scene, max_cells=max_cells)
    cell = grid.cell
    tol = max(0.02, cell / 2)

    # Size, ground, mass, symmetry.
    bounds = _part_bounds(scene, grid, solids)
    lo = np.min([bounds[i][0] for i in solids], axis=0)
    hi = np.max([bounds[i][1] for i in solids], axis=0)
    ext = hi - lo
    min_y = float(lo[1])
    grounded = abs(min_y) <= tol
    union = grid.union()
    union_cells = max(int(np.count_nonzero(union)), 1)
    size = {"width": _r(ext[0]), "height": _r(ext[1]), "depth": _r(ext[2]), "min": _vec(lo), "max": _vec(hi),
            "human_ratio": _r(ext[1] / HUMAN_HEIGHT)}
    ground = {"min_y": _r(min_y), "grounded": bool(grounded), "floats_by": _r(max(min_y, 0.0) if not grounded else 0.0)}
    support = None
    if grounded:
        pts, touching = ground_contacts(scene, solids, max(min_y, 0.0), cell)
        support = (pts, [parts[i].name for i in touching])
    mass = _mass(grid, union, grounded, support)
    symmetry = {"x": _r(_iou(union, union[::-1, :, :])), "z": _r(_iou(union, union[:, :, ::-1]))}

    # Contacts and floating pieces.
    pairs = _contacts(scene, grid)
    contacts = [{"a": parts[solids[a]].name, "b": parts[solids[b]].name, "kind": "overlap" if shared else "touch",
                 "overlap_volume": _r(shared * cell**3, 4)} for a, b, shared in pairs]
    comps = _components(len(solids), pairs)
    floating, float_members = _floating(scene, grid, solids, comps, bounds, tol)

    # Renders: the requested views plus the color view (readability renders its own cameras).
    march_views = list(dict.fromkeys([*views, COLOR_VIEW]))
    cams = make_cameras(march_views, *scene.bounds(include_helpers=False), res=res)
    bufs = {v: march(scene, cams[v]) for v in march_views}
    rays = {v: _rays(cams[v]) for v in march_views}
    vis = _solid_visibility(scene, solids, views, cams, bufs, rays)
    mean_lit, shares = _lit_colors(asset, scene, solids, bufs[COLOR_VIEW])

    part_entries = []
    for k, i in enumerate(solids):
        p = parts[i]
        p_lo, p_hi = bounds[i]
        part_entries.append({
            "name": p.name, "shape": p.shape, "material": p.material,
            "center": _vec((p_lo + p_hi) / 2), "size": _vec(p_hi - p_lo),
            "volume_share": _r(np.count_nonzero(grid.strict[k]) / union_cells),
            "attached_to": p.attached_to,
            "visible": {v: _r(vis[i][v][0]) for v in views},
            "lit_color": _to_hex(mean_lit[i]) if i in mean_lit else None,
            "tags": list(p.tags),
        })

    cut_entries, cut_vis = [], {}
    for c in cuts:
        targets = [j for j in solids if c in scene.cutters.get(j, [])]
        removed = _removed_volume(scene, c, targets)
        cut_vis[c] = _cut_visibility(scene, c, targets, views, cams, bufs, rays, removed)
        cut_entries.append({
            "name": parts[c].name, "cuts": [parts[j].name for j in targets],
            "removed_volume": _r(removed, 4),
            "visible": {v: _r(cut_vis[c][v][0]) for v in views},
        })

    contrast = []
    for a, b in _adjacent_pairs(bufs[COLOR_VIEW].part):
        if a in mean_lit and b in mean_lit:
            # From the reported (hex) colors, so checks that recompute it from the report agree exactly.
            ratio = contrast_ratio(hex_to_rgb(_to_hex(mean_lit[a])), hex_to_rgb(_to_hex(mean_lit[b])))
            contrast.append({"a": parts[a].name, "b": parts[b].name, "ratio": _r(ratio, 2)})
    contrast.sort(key=lambda e: (e["ratio"], e["a"], e["b"]))

    # Silhouettes: one cell size for all views, letters by descending visible area over the views.
    area = np.zeros(len(parts), dtype=np.int64)
    for v in views:
        area += np.bincount(bufs[v].part[bufs[v].hit], minlength=len(parts))
    order = sorted((i for i in solids if area[i] > 0), key=lambda i: (-area[i], i))
    label_of = np.zeros(len(parts) + 1, dtype=np.int64)
    for rank, i in enumerate(order):
        label_of[i + 1] = rank + 1
    chars = np.array([LETTERS[r] if r < len(LETTERS) else OTHER_LETTER for r in range(max(len(order), 1))])
    name_of_char = {LETTERS[r]: parts[i].name for r, i in enumerate(order) if r < len(LETTERS)}
    sil_cell = _choose_cell([_extent(cams[v], bufs[v]) for v in views])
    silhouette = {}
    for v in views:
        sil = _silhouette(cams[v], bufs[v], sil_cell, label_of, chars)
        sil["legend"] = {ch: name_of_char.get(ch, "other parts") for ch in sil["legend"]}
        silhouette[v] = sil

    read = _readability(scene, READ_VIEW, READ_SIZES)
    read_more = {v: _readability(scene, v, READ_SIZES) for v in READ_MORE_VIEWS if v != READ_VIEW}

    compared = {}
    for other in compare:
        if not isinstance(other, Asset):
            from .loader import load_asset  # local import: the loader is not needed for plain reports

            other = load_asset(other, asset.root)
        compared[other.name] = compare_silhouettes(asset, other, views)

    report = {
        "asset": asset.name,
        "style": asset.style.name,
        "brief": " ".join((asset.brief or "").split()),
        "part_count": len(solids),
        "cut_count": len(cuts),
        "paint_count": sum(1 for p in parts if p.is_paint and not p.helper),
        "pieces": len(comps),
        "voxel_cell": _r(cell, 4),
        "render_mpp": _r(cams[march_views[0]].mpp, 4),
        "size": size,
        "ground": ground,
        "mass": mass,
        "symmetry": symmetry,
        "parts": part_entries,
        "contacts": contacts,
        "floating": floating,
        "cuts": cut_entries,
        "colors": {"view": COLOR_VIEW, "shares": {m: _r(s) for m, s in shares.items()}},
        "contrast": contrast,
        "silhouette": silhouette,
        "readability": read,
        "readability_views": read_more,
        "compare": compared,
    }
    report["warnings"] = _warnings(report, scene, solids, cuts, vis, cut_vis, float_members, views)
    return report


def _warnings(report, scene, solids, cuts, vis, cut_vis, float_members, views) -> list[str]:
    """Short, specific, actionable sentences."""
    parts = scene.parts
    out = []
    g = report["ground"]
    by_name = {p["name"]: p for p in report["parts"]}
    lowest = min(report["parts"], key=lambda p: p["center"][1] - p["size"][1] / 2)["name"]
    if not g["grounded"]:
        if g["min_y"] > 0:
            out.append(f"the asset floats {g['min_y']:.2f} m above the ground (lowest part {lowest}) — move it down "
                       f"{g['min_y']:.2f} m, e.g. at y 0 with align: bottom on {lowest}")
        else:
            out.append(f"the asset sinks {-g['min_y']:.2f} m below the ground (lowest part {lowest}) — raise it "
                       f"{-g['min_y']:.2f} m, e.g. at y 0 with align: bottom on {lowest}")
    for entry, members in zip(report["floating"], float_members):
        names, gap, near, side = entry["parts"], entry["gap"], entry["nearest"], entry["direction"]
        first = parts[members[0]]
        who = names[0] if len(names) == 1 else f"{', '.join(names)} (one piece)"
        verb = "floats" if len(names) == 1 else "float"
        it = "it" if len(names) == 1 else first.name
        move = _move_text(entry["move"]) if entry.get("move") else f"{gap:.2f} m {_DIR_MOVE[side]}"
        if first.attached_to == near:
            fix = f"{it} is on: {near} already, but its offset/align leaves the gap; move it {move}"
        elif _anchor_fits(next(p for p in parts if p.name == near), side, report["voxel_cell"]):
            fix = f"attach {it} with on: {near}.{_DIR_ANCHOR[side]} or move it {move}"
        else:
            fix = f"move {it} {move}"
        out.append(f"{who} {verb} {gap:.2f} m {side} {near} — {fix}")
    m = report["mass"]
    if m["stable"] is False:
        cx, _, cz = m["center"]
        side = _TIP_WORDS.get(m.get("tips_toward"), "outside")
        names = m.get("support") or []
        on = f"the ground contacts ({_name_list(names)})" if names else "the ground footprint"
        out.append(f"unstable: the center of mass (x {cx:.2f}, z {cz:.2f}) is {-m['support_margin']:.2f} m {side} "
                   f"{on} — add a support under it or move mass over the contacts")
    for entry in report["cuts"]:
        if entry["removed_volume"] <= 0:
            targets = ", ".join(entry["cuts"]) or "no solid"
            out.append(f"cut {entry['name']} removes nothing — it does not intersect {targets}; move it into the part or fix from:")
    for i in solids:
        name = parts[i].name
        if HIDDEN_TAG in parts[i].tags:  # the designer expects it hidden (an interior or underside part)
            continue
        best_view = max(views, key=lambda v: (vis[i][v][0], -views.index(v)))
        frac, _, occluder = vis[i][best_view]
        if frac >= LOW_VISIBLE:
            continue
        if all(vis[i][v][1] == 0 for v in views):
            out.append(f"{name} covers no pixel in any view at {report['render_mpp']:.3f} m/px — it is smaller than a "
                       f"pixel or cut away; enlarge it or drop it")
            continue
        best = "0% in every view" if frac == 0 else f"best {_pct(frac)} in {best_view}"
        behind = f"; mostly hidden by {parts[occluder].name}" if occluder >= 0 else ""
        out.append(f"{name} is under {round(LOW_VISIBLE * 100)}% visible in every view ({best}){behind} — "
                   f"move it into view or make it larger")
    removed = {e["name"]: e["removed_volume"] for e in report["cuts"]}
    for c in cuts:
        # Only cuts covered by another part: a cut that merely faces away from the views is not a problem.
        # A part set inside the opening (a door panel in its door cut, a pane in a window) is the inset
        # idiom: it shows the opening, so such a view counts as if the part were not there.
        inset = {v: cut_vis[c][v][2] >= 0 and _inside_cut(scene, cut_vis[c][v][2], c) for v in views}
        shown = {v: cut_vis[c][v][3] if inset[v] else cut_vis[c][v][0] for v in views}
        frac = max(shown.values())
        blocked = [v for v in views if cut_vis[c][v][3] >= LOW_VISIBLE and cut_vis[c][v][2] >= 0 and not inset[v]]
        if frac < LOW_VISIBLE and blocked and removed[parts[c].name] > 0 and HIDDEN_TAG not in parts[c].tags:
            v = blocked[0]
            out.append(f"cut {parts[c].name} is under {round(LOW_VISIBLE * 100)}% visible in every view (in {v} "
                       f"{_pct(cut_vis[c][v][0])}, {_pct(cut_vis[c][v][3])} without other parts); hidden by "
                       f"{parts[cut_vis[c][v][2]].name} — move the cut or the part in front of it")
    low = []
    for e in report["contrast"]:
        ma, mb = by_name[e["a"]]["material"], by_name[e["b"]]["material"]
        if e["ratio"] < LOW_CONTRAST and ma != mb:
            low.append(f"{e['a']}/{e['b']} {e['ratio']:.2f} ({ma}/{mb})")
    if low:
        out.append(f"low contrast (< {LOW_CONTRAST}) between adjacent parts in the iso view: {', '.join(low)} — "
                   f"make one material of each pair lighter or darker")
    # Parts that vanish at 32 px in every readability view where they are seen at all (a wing seen edge-on
    # from the front may still read from iso).
    reads = [report["readability"], *report.get("readability_views", {}).values()]
    lost_in, kept_in = {}, set()
    for read in reads:
        e = _at_px(read["at_px"]).get(32, {})
        kept_in.update(e.get("parts_kept", []))
        for name in e.get("parts_lost", []):
            lost_in.setdefault(name, []).append(read["view"])
    lost = [n for n in lost_in if n not in kept_in and HIDDEN_TAG not in parts[next(
        i for i in solids if parts[i].name == n)].tags]
    if lost:
        where = ", ".join(r["view"] for r in reads)
        out.append(f"at 32 px ({where}) these parts vanish: {', '.join(lost)} — make them larger or thicker "
                   f"if they matter at a distance")
    return out


_TIP_WORDS = {"+x": "right of (+x)", "-x": "left of (-x)", "+z": "in front of (+z)", "-z": "behind (-z)"}


def _name_list(names: list[str], limit: int = 6) -> str:
    return ", ".join(names[:limit]) + (f" +{len(names) - limit} more" if len(names) > limit else "")


def _inside_cut(scene: Scene, i: int, c: int) -> bool:
    """Whether solid part i sits in the opening of cut c (the center of its AABB is inside the cutter)."""
    lo, hi = scene.aabb(i)
    return bool(scene.parts[c].sdf(((np.asarray(lo) + np.asarray(hi)) / 2.0)[None, :])[0] < 0.0)


def _at_px(at_px: dict) -> dict:
    """Readability entries keyed by int px (JSON round trips turn the keys into strings)."""
    return {int(k): v for k, v in at_px.items()}


# ----------------------------------------------------------------------------- text


def _m(x) -> str:
    if x is None:
        return "-"
    text = f"{float(x):.2f}"
    return "0.00" if text == "-0.00" else text


def _v(values) -> str:
    return "-" if values is None else "[" + ", ".join(_m(x) for x in values) + "]"


def _pct(frac) -> str:
    return "-" if frac is None else f"{int(math.floor(float(frac) * 100 + 0.5))}%"


def _yes(flag) -> str:
    return "n/a" if flag is None else ("yes" if flag else "no")


def _num(x: float) -> str:
    text = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _table(rows: list[list[str]], right=(), indent: str = "  ") -> list[str]:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [r[c].rjust(widths[c]) if c in right else r[c].ljust(widths[c]) for c in range(len(r))]
        out.append((indent + "  ".join(cells)).rstrip())
    return out


def _wrap_items(items: list[str], width: int = 100, indent: str = "  ") -> list[str]:
    """Items joined by two spaces, packed into lines of at most `width` chars; an item is never split."""
    lines, line = [], ""
    for item in items:
        if line and len(indent) + len(line) + 2 + len(item) > width:
            lines.append(indent + line)
            line = item
        else:
            line = f"{line}  {item}" if line else item
    if line or not lines:
        lines.append((indent + line).rstrip())
    return lines


def _signed_axis(axis: str) -> tuple[str, float]:
    return (axis[1:], -1.0) if axis.startswith("-") else (axis, 1.0)


def _silhouette_lines(view: str, sil: dict) -> list[str]:
    if not sil["grid"]:
        return [f"SILHOUETTE {view}: empty"]
    cell, cols, rows = sil["cell"], sil["cols"], sil["rows"]
    (cname, csign), (rname, _) = _signed_axis(sil["axes"][0]), _signed_axis(sil["axes"][1])
    x_left, x_right = csign * sil["x0"], csign * (sil["x0"] + cols * cell)
    labels = sil["row_labels"]
    out = [f"SILHOUETTE {view}  cell {_m(cell)} m  {cols} cols x {rows} rows  cols {cname} {_m(x_left)}..{_m(x_right)}  "
           f"rows {rname} {_m(labels[0])}..{_m(labels[-1])} (labels = cell centers)"]
    out += _wrap_items([f"{ch} {name}" for ch, name in sil["legend"].items()])
    texts = [_m(x) for x in labels]
    width = max(len(t) for t in texts)
    prefix = 2 + width + 3
    for t, row in zip(texts, sil["grid"]):
        out.append(f"  {t.rjust(width)} | {row}")
    # Ruler: world value of the column center at every 4th column of the global cell lattice.
    chars = [" "] * (prefix + 2 * cols + 8)
    for k in range(cols):
        kk = round(sil["x0"] / cell + 0.5 + k)
        if kk % 4:
            continue
        text = _num(csign * kk * cell)
        pos = prefix + 2 * k - (1 if text.startswith("-") else 0)
        span = range(max(pos - 1, 0), min(pos + len(text) + 1, len(chars)))
        if pos + len(text) > len(chars) or any(chars[p] != " " for p in span):
            continue
        chars[pos:pos + len(text)] = text
    out.append(f"  {cname.rjust(width)}   " + "".join(chars[prefix:]).rstrip())
    return out


def format_report(report: dict) -> str:
    """The report as stable, compact, aligned text; carries every fact of the dict."""
    r = report
    out: list[str] = []

    def head(title: str, text: str = "") -> None:
        out.append(f"{title:<12}{text}".rstrip())

    head("ASSET", f"{r['asset']} (style {r['style']})")
    if r.get("brief"):
        lines = textwrap.wrap(r["brief"], 96) or [""]
        head("BRIEF", lines[0])
        out += [_INDENT + line for line in lines[1:]]
    pieces = r.get("pieces", 1)
    paints = f"{r['paint_count']} paints  " if r.get("paint_count") else ""
    head("COUNTS", f"{r['part_count']} solids  {r['cut_count']} cuts  {paints}{pieces} piece{'s' if pieces != 1 else ''}  "
         f"voxel cell {r['voxel_cell']:.3f} m  render {r['render_mpp']:.3f} m/px")
    s = r["size"]
    head("SIZE", f"width {_m(s['width'])} m  height {_m(s['height'])} m  depth {_m(s['depth'])} m  "
         f"= {s['human_ratio']:.2f} x human height ({HUMAN_HEIGHT} m)")
    out.append(_INDENT + f"min {_v(s['min'])}  max {_v(s['max'])}")
    g = r["ground"]
    head("GROUND", f"grounded {_yes(g['grounded'])}  min y {_m(g['min_y'])} m  floats by {_m(g['floats_by'])} m")
    m = r["mass"]
    stable = _yes(m["stable"]) + (" (not grounded)" if m["stable"] is None else "")
    head("MASS", f"volume {m['volume']:.3f} m3  center {_v(m['center'])}  stable {stable}  "
         f"support margin {_m(m['support_margin'])} m")
    if m.get("support"):
        out.append(_INDENT + f"ground contacts: {_name_list(m['support'], 12)}")
    head("SYMMETRY", f"x {_pct(r['symmetry']['x'])}  z {_pct(r['symmetry']['z'])}  "
         "(occupancy IoU with the mirror image across the bounding-box center plane)")

    views = list(r["parts"][0]["visible"]) if r["parts"] else []
    out.append("")
    head("PARTS", f"{len(r['parts'])} solids  (center and size w h d = world AABB in m, vol = share of the asset volume,")
    out.append(_INDENT + "visible = share of the part's own pixels seen per view, lit = mean lit color in iso)")
    rows = [["name", "shape", "material", "center x", "y", "z", "size w", "h", "d", "vol", "on", *views, "lit", "tags"]]
    for p in r["parts"]:
        rows.append([p["name"], p["shape"], p["material"] or "-", *(_m(x) for x in p["center"]), *(_m(x) for x in p["size"]),
                     _pct(p["volume_share"]), p["attached_to"] or "-", *(_pct(p["visible"][v]) for v in views),
                     p["lit_color"] or "-", ",".join(p["tags"]) or "-"])
    numeric = {3, 4, 5, 6, 7, 8, 9, *range(11, 11 + len(views))}
    out += _table(rows, right=numeric)

    head("CONTACTS", f"{len(r['contacts'])} pairs  (touch = within one voxel cell, overlap = shared volume in m3)")
    if r["contacts"]:
        out += _table([[c["a"], c["b"], c["kind"], f"{c['overlap_volume']:.3f}" if c["kind"] == "overlap" else ""]
                       for c in r["contacts"]], right={3})
    head("FLOATING", "none" if not r["floating"] else f"{len(r['floating'])} piece(s) not connected to the grounded piece")
    for f in r["floating"]:
        move = f"  move {_v(f['move'])}" if f.get("move") else ""
        out.append(f"  {', '.join(f['parts'])}  gap {_m(f['gap'])} m {f.get('direction', 'from')} {f['nearest']}{move}")

    head("CUTS", "none" if not r["cuts"] else f"{len(r['cuts'])}  (removed = primitive volume inside the cut, "
         "visible = share of the carved footprint seen)")
    if r["cuts"]:
        cut_views = list(r["cuts"][0]["visible"])
        rows = [["name", "removed m3", *cut_views, "from"]]
        for c in r["cuts"]:
            rows.append([c["name"], f"{c['removed_volume']:.3f}", *(_pct(c["visible"][v]) for v in cut_views),
                         ", ".join(c["cuts"]) or "-"])
        out += _table(rows, right=set(range(1, 2 + len(cut_views))))

    shares = r["colors"]["shares"]
    head("COLORS", f"{r['colors']['view']} pixel share by material: " + ("  ".join(f"{k} {_pct(v)}" for k, v in shares.items()) or "none"))
    head("CONTRAST", f"{len(r['contrast'])} adjacent pairs in {r['colors']['view']}, WCAG ratio of mean lit colors "
         "(lit = color x (ambient + (1 - ambient) x Lambert)), ascending")
    if r["contrast"]:
        mats = {p["name"]: p["material"] for p in r["parts"]}
        items = [f"{e['a']}/{e['b']} {e['ratio']:.2f}" + ("=" if mats.get(e["a"]) == mats.get(e["b"]) else "")
                 for e in r["contrast"]]
        out += _wrap_items(items)
        if any(item.endswith("=") for item in items):
            out.append("  (= marks pairs of the same material)")

    for view, sil in r["silhouette"].items():
        out.append("")
        out += _silhouette_lines(view, sil)

    reads = [r["readability"], *r.get("readability_views", {}).values()]
    out.append("")
    head("READABILITY", f"{', '.join(x['view'] for x in reads)}: silhouette at N px on the longer side, upscaled back, "
         f"IoU vs full resolution; kept = covers >= {KEEP_PX} px")
    for read in reads:
        for n, e in sorted(_at_px(read["at_px"]).items(), reverse=True):
            out.append(f"  {read['view']:<6}{n:>3} px  IoU {_pct(e['iou'])}")
            out.append(f"                kept {len(e['parts_kept'])}: {', '.join(e['parts_kept']) or '-'}")
            out.append(f"                lost {len(e['parts_lost'])}: {', '.join(e['parts_lost']) or '-'}")

    head("COMPARE", "none" if not r["compare"] else "silhouette IoU, aligned at the ground and horizontal center; "
         "normalized = scaled to equal height")
    if r["compare"]:
        rows = [["other", "view", "true scale", "normalized"]]
        for other, per_view in r["compare"].items():
            for view, e in per_view.items():
                rows.append([other, view, _pct(e["iou_true_scale"]), _pct(e["iou_normalized"])])
        out += _table(rows, right={2, 3})

    out.append("")
    head("WARNINGS", "none" if not r["warnings"] else str(len(r["warnings"])))
    out += [f"  - {w}" for w in r["warnings"]]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- diff


def _facts(r: dict) -> dict[str, str]:
    """Flat {metric path: display text} used to diff two reports."""
    f: dict[str, str] = {}
    for key in ("asset", "style", "brief", "part_count", "cut_count", "pieces"):
        if key in r:
            f[key] = str(r[key])
    f["voxel_cell"] = f"{r['voxel_cell']:.3f}"
    f["render_mpp"] = f"{r['render_mpp']:.3f}"
    s = r["size"]
    for key in ("width", "height", "depth"):
        f[f"size.{key}"] = _m(s[key])
    f["size.min"], f["size.max"], f["size.human_ratio"] = _v(s["min"]), _v(s["max"]), _m(s["human_ratio"])
    g = r["ground"]
    f["ground.min_y"], f["ground.grounded"], f["ground.floats_by"] = _m(g["min_y"]), _yes(g["grounded"]), _m(g["floats_by"])
    m = r["mass"]
    f["mass.volume"], f["mass.center"] = f"{m['volume']:.3f}", _v(m["center"])
    f["mass.stable"], f["mass.support_margin"] = _yes(m["stable"]), _m(m["support_margin"])
    f["mass.support"] = ", ".join(m.get("support") or []) or "-"
    for axis, value in r["symmetry"].items():
        f[f"symmetry.{axis}"] = _pct(value)
    for p in r["parts"]:
        base = f"parts.{p['name']}"
        f[f"{base}.shape"], f[f"{base}.material"] = p["shape"], str(p["material"])
        f[f"{base}.center"], f[f"{base}.size"] = _v(p["center"]), _v(p["size"])
        f[f"{base}.volume_share"], f[f"{base}.attached_to"] = _pct(p["volume_share"]), str(p["attached_to"] or "-")
        for view, frac in p["visible"].items():
            f[f"{base}.visible.{view}"] = _pct(frac)
        f[f"{base}.lit_color"], f[f"{base}.tags"] = str(p["lit_color"] or "-"), ",".join(p["tags"]) or "-"
    for c in r["contacts"]:
        f[f"contacts.{c['a']}+{c['b']}"] = c["kind"] + (f" {c['overlap_volume']:.3f} m3" if c["kind"] == "overlap" else "")
    for e in r["floating"]:
        f[f"floating.{'+'.join(e['parts'])}"] = (f"gap {_m(e['gap'])} m {e.get('direction', 'from')} {e['nearest']}"
                                                 + (f" move {_v(e['move'])}" if e.get("move") else ""))
    for c in r["cuts"]:
        base = f"cuts.{c['name']}"
        f[f"{base}.cuts"], f[f"{base}.removed_volume"] = ",".join(c["cuts"]) or "-", f"{c['removed_volume']:.3f}"
        for view, frac in c["visible"].items():
            f[f"{base}.visible.{view}"] = _pct(frac)
    for mat, share in r["colors"]["shares"].items():
        f[f"colors.{mat}"] = _pct(share)
    for e in r["contrast"]:
        f[f"contrast.{e['a']}+{e['b']}"] = f"{e['ratio']:.2f}"
    for read in (r["readability"], *r.get("readability_views", {}).values()):
        for n, e in _at_px(read["at_px"]).items():
            base = f"readability.{read['view']}.{n}px"
            f[f"{base}.iou"] = _pct(e["iou"])
            f[f"{base}.kept"], f[f"{base}.lost"] = ", ".join(e["parts_kept"]) or "-", ", ".join(e["parts_lost"]) or "-"
    for other, per_view in r["compare"].items():
        for view, e in per_view.items():
            f[f"compare.{other}.{view}.iou_true_scale"] = _pct(e["iou_true_scale"])
            f[f"compare.{other}.{view}.iou_normalized"] = _pct(e["iou_normalized"])
    return f


def _silhouette_change(view: str, a: dict, b: dict) -> str | None:
    if a.get("grid") == b.get("grid") and a.get("legend") == b.get("legend"):
        return None
    same_frame = all(a.get(k) == b.get(k) for k in ("cell", "x0", "y0", "cols", "rows"))
    if same_frame:
        changed = sum(ca != cb for ra, rb in zip(a["grid"], b["grid"]) for ca, cb in zip(ra[::2], rb[::2]))
        what = f"{changed} cells changed"
    else:
        what = (f"grid {a.get('cols', 0)}x{a.get('rows', 0)} -> {b.get('cols', 0)}x{b.get('rows', 0)} cells, "
                f"cell {_m(a.get('cell', 0))} -> {_m(b.get('cell', 0))} m")
    if a.get("legend") != b.get("legend"):
        what += ", legend changed"
    return f"silhouette.{view}: {what}"


def diff_reports(old: dict, new: dict) -> str:
    """Changed metrics as `path: old -> new` lines (parts and cuts added/removed listed once)."""
    a, b = _facts(old), _facts(new)
    lines: list[str] = []
    skip: list[str] = []
    for kind, key in (("part", "parts"), ("cut", "cuts")):
        old_names = [p["name"] for p in old.get(key, [])]
        new_names = [p["name"] for p in new.get(key, [])]
        for p in new.get(key, []):
            if p["name"] not in old_names:
                detail = f" ({p['shape']}, {p['material']})" if kind == "part" else f" (from {', '.join(p['cuts']) or '-'})"
                lines.append(f"+ {kind} {p['name']}{detail}")
                skip.append(f"{key}.{p['name']}.")
        for name in old_names:
            if name not in new_names:
                lines.append(f"- {kind} {name}")
                skip.append(f"{key}.{name}.")
    for key in list(a) + [k for k in b if k not in a]:
        if any(key.startswith(s) for s in skip):
            continue
        va, vb = a.get(key, "none"), b.get(key, "none")
        if va != vb:
            lines.append("brief: changed" if key == "brief" else f"{key}: {va} -> {vb}")
    for view in list(old.get("silhouette", {})) + [v for v in new.get("silhouette", {}) if v not in old.get("silhouette", {})]:
        change = _silhouette_change(view, old.get("silhouette", {}).get(view, {}), new.get("silhouette", {}).get(view, {}))
        if change:
            lines.append(change)
    old_w, new_w = old.get("warnings", []), new.get("warnings", [])
    lines += [f"+ warning: {w}" for w in new_w if w not in old_w]
    lines += [f"- warning: {w}" for w in old_w if w not in new_w]
    if not lines:
        return "CHANGES     none\n"
    return f"CHANGES     {len(lines)}\n" + "\n".join(f"  {line}" for line in lines) + "\n"
