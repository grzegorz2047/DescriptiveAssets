"""In-memory model: materials, styles, parts, blend groups, scenes and assets (SPEC section 5).

A Part is one primitive with a world placement. A Scene evaluates the signed distance of the
union of its solid parts, with cuts and smooth blends applied, for every geometry pass
(ray marching, voxels, meshing).

Internally distance code works on points laid out as (3, N) float32 arrays (contiguous rows
for x, y and z); the public methods take the usual (N, 3).
"""

from __future__ import annotations

import math
import numbers
import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import sdf
from .errors import SightlineError, did_you_mean

F32 = np.float32

SHAPES: dict[str, tuple[tuple[str, ...], dict]] = {
    # shape: (required params, optional params with defaults)
    "box": (("size",), {"round": 0.0}),
    "sphere": (("radius",), {}),
    "ellipsoid": (("size",), {}),
    "cylinder": (("radius", "height"), {"sides": 0, "round": 0.0}),
    "cone": (("radius", "height"), {"top_radius": 0.0, "sides": 0}),
    "capsule": (("radius", "height"), {}),
    "torus": (("radius", "thickness"), {}),
    "wedge": (("size",), {}),
    "extrude": (("points", "depth"), {"round": 0.0}),
}
PARAM_KEYS = ("size", "radius", "height", "top_radius", "sides", "round", "thickness", "points", "depth")
EXTRUDE_MAX_POINTS = 64
SHAPE_USAGE = {
    "box": "size: [w, h, d], optional round: r",
    "sphere": "radius: r",
    "ellipsoid": "size: [w, h, d] (full extents)",
    "cylinder": "radius: r, height: h, optional sides: n (>= 3), round: r",
    "cone": "radius: r (bottom), height: h, optional top_radius: r (frustum), sides: n",
    "capsule": "radius: r, height: h (total, >= 2r)",
    "torus": "radius: R (ring center), thickness: t (tube radius)",
    "wedge": "size: [w, h, d] (full height at the back, slope to the front)",
    "extrude": "points: [[x, y], ...] (an outline in the XY plane, as seen from the front: 3..64 points, no "
               "crossing edges), depth: d (along Z), optional round: r",
}

ANCHOR_WORDS = {
    "top": (1, 1.0),
    "bottom": (1, -1.0),
    "right": (0, 1.0),
    "left": (0, -1.0),
    "front": (2, 1.0),
    "back": (2, -1.0),
}
ANCHOR_HINT = (
    "anchors: center, top, bottom, left, right, front, back and combinations with at most one word "
    "per axis joined by '_' (top_front_left), or @[u, v, w] with each component in [-1, 1]"
)


# ----------------------------------------------------------------------------- materials, style


@dataclass
class Material:
    name: str
    color: tuple[float, float, float]  # 0..1 sRGB values as written (no gamma math)
    pattern: str = "solid"
    pattern_color: tuple[float, float, float] | None = None
    scale: tuple[float, float] = (0.25, 0.25)
    axis: str = "y"
    shine: float = 0.0


@dataclass(eq=False)
class Style:
    name: str
    palette: dict[str, tuple[float, float, float]]
    materials: dict[str, Material]
    light_dir: np.ndarray  # unit vector, direction the light travels
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
    path: pathlib.Path | None = None  # file the style was loaded from

    def material(self, name: str) -> Material:
        try:
            return self.materials[name]
        except KeyError:
            hint = did_you_mean(name, self.materials) or ""
            names = ", ".join(sorted(self.materials))
            raise SightlineError(
                f"unknown material '{name}' in style '{self.name}'",
                hint=(hint + " " if hint else "") + f"materials: {names}",
            ) from None


# ----------------------------------------------------------------------------- params and anchors


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def normalize_params(shape: str, raw: dict, where_fn: Callable[[str | None], str | None] | None = None) -> dict:
    """Validate and normalize the shape parameters (size -> 3-tuple of floats, defaults filled).

    `where_fn(key)` returns the `where` text for an error about `key` (None = the whole part).
    Keys of `raw` that are not shape parameters are ignored here (the loader checks them).
    """
    where = where_fn or (lambda key: None)

    def fail(key, message, hint=None):
        raise SightlineError(message, where=where(key), hint=hint)

    if shape not in SHAPES:
        hint = did_you_mean(shape, SHAPES) or ""
        fail("shape", f"unknown shape '{shape}'", (hint + " " if hint else "") + f"shapes: {', '.join(SHAPES)}")
    required, optional = SHAPES[shape]
    usage = f"{shape} takes {SHAPE_USAGE[shape]}"
    for key in PARAM_KEYS:
        if key in raw and key not in required and key not in optional:
            fail(key, f"shape '{shape}' does not take '{key}'", usage)
    params: dict = {}
    for key in required:
        if key not in raw or raw[key] is None:
            fail(None, f"shape '{shape}' needs '{key}'", usage)
    for key in required + tuple(optional):
        value = raw.get(key, optional.get(key))
        if key == "size":
            if not (isinstance(value, (list, tuple)) and len(value) == 3 and all(_is_number(v) for v in value)):
                fail(key, f"size must be a list of 3 numbers [w, h, d], got {value!r}", usage)
            if any(v <= 0 for v in value):
                fail(key, f"size must be > 0 in every axis, got {list(value)}", "sizes are full extents in meters")
            params[key] = tuple(float(v) for v in value)
        elif key == "points":
            ok = isinstance(value, (list, tuple)) and 3 <= len(value) <= EXTRUDE_MAX_POINTS and all(
                isinstance(pt, (list, tuple)) and len(pt) == 2 and all(_is_number(c) for c in pt) for pt in value)
            if not ok:
                fail(key, f"points must be a list of 3 to {EXTRUDE_MAX_POINTS} [x, y] pairs, got {value!r}", usage)
            pts = np.array(value, dtype=np.float64)
            area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
            if _self_intersects(pts):
                fail(key, "the outline crosses itself", "list the corners in order around the outline")
            if abs(area) < 1e-8:
                fail(key, "points enclose no area (all on one line?)", "give the outline's corners in order around it")
            center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0  # the shape's local frame is its bbox center
            params[key] = tuple((float(x), float(y)) for x, y in pts - center)
        elif key == "sides":
            if isinstance(value, bool) or not isinstance(value, int) or (value != 0 and value < 3):
                fail(key, f"sides must be an integer >= 3, got {value!r}", "omit sides for a round shape")
            params[key] = int(value)
        else:
            if not _is_number(value):
                fail(key, f"{key} must be a number, got {value!r}", usage)
            value = float(value)
            if key in ("round", "top_radius"):
                if value < 0:
                    fail(key, f"{key} must be >= 0, got {value:g}")
            elif value <= 0:
                fail(key, f"{key} must be > 0, got {value:g}", usage)
            params[key] = value
    # Cross-parameter limits.
    rnd = params.get("round", 0.0)
    if shape == "box" and rnd > min(params["size"]) / 2:
        fail("round", f"round {rnd:g} is larger than half the smallest size ({min(params['size']) / 2:g})",
             "round is an edge radius and must be <= min(size) / 2")
    if shape == "cylinder":
        limit = min(params["height"] / 2, params["radius"])
        if params["sides"]:
            limit = min(limit, params["radius"] * math.cos(math.pi / params["sides"]))
        if rnd > limit:
            fail("round", f"round {rnd:g} is larger than allowed {limit:g}", "round must be <= min(radius, height / 2)")
    if shape == "extrude":
        pts = np.array(params["points"])
        limit = min(params["depth"] / 2, float((pts.max(axis=0) - pts.min(axis=0)).min()) / 2)
        if rnd > limit:
            fail("round", f"round {rnd:g} is larger than allowed {limit:g}", "round must be <= depth / 2 and the outline's half width")
    if shape == "capsule" and params["height"] < 2 * params["radius"] - 1e-9:
        fail("height", f"capsule height {params['height']:g} is less than 2 * radius ({2 * params['radius']:g})",
             "height is the total length including both caps; use a sphere for height = 2 * radius")
    return params


def _self_intersects(pts: np.ndarray) -> bool:
    """True when two non-adjacent edges of the closed polygon cross."""
    n = len(pts)

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            if o1 * o2 < 0 and o3 * o4 < 0:
                return True
    return False


def local_half_extents(shape: str, params: dict) -> np.ndarray:
    """Half extents of the local bounding box (SPEC 4.1 table)."""
    if shape in ("box", "ellipsoid", "wedge"):
        return np.array(params["size"], dtype=np.float64) / 2.0
    if shape == "extrude":
        pts = np.array(params["points"], dtype=np.float64)
        return np.array([*(pts.max(axis=0) - pts.min(axis=0)) / 2.0, params["depth"] / 2.0])
    if shape == "sphere":
        return np.full(3, params["radius"], dtype=np.float64)
    if shape in ("cylinder", "capsule"):
        return np.array([params["radius"], params["height"] / 2.0, params["radius"]])
    if shape == "cone":
        r = max(params["radius"], params["top_radius"])
        return np.array([r, params["height"] / 2.0, r])
    if shape == "torus":
        outer = params["radius"] + params["thickness"]
        return np.array([outer, params["thickness"], outer])
    raise ValueError(f"unknown shape {shape!r}")


def parse_anchor(spec) -> np.ndarray:
    """Normalized anchor (u, v, w) in [-1, 1]^3 from a name ('top_front'), '@[u, v, w]' or a 3-sequence."""
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("@"):
            body = text[1:].strip()
            if not (body.startswith("[") and body.endswith("]")):
                raise SightlineError(f"bad anchor '{spec}'", hint="write @[u, v, w], e.g. @[0, -1, 1]")
            try:
                values = [float(v) for v in body[1:-1].split(",")]
            except ValueError:
                raise SightlineError(f"bad anchor '{spec}'", hint="write @[u, v, w] with numbers, e.g. @[0, -1, 1]") from None
            return parse_anchor(values)
        if text == "center":
            return np.zeros(3)
        uvw = np.zeros(3)
        seen: dict[int, str] = {}
        for word in text.split("_"):
            if word not in ANCHOR_WORDS:
                options = ["center", *ANCHOR_WORDS]
                guess = did_you_mean(word, options) if word else None
                raise SightlineError(f"unknown anchor '{spec}'", hint=(guess + " " if guess else "") + ANCHOR_HINT)
            axis, sign = ANCHOR_WORDS[word]
            if axis in seen:
                raise SightlineError(f"anchor '{spec}' names axis {'xyz'[axis]} twice ({seen[axis]} and {word})",
                                     hint=ANCHOR_HINT)
            seen[axis] = word
            uvw[axis] = sign
        return uvw
    if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 3 and all(_is_number(v) for v in spec):
        uvw = np.array([float(v) for v in spec])
        if np.any(np.abs(uvw) > 1.0 + 1e-9):
            raise SightlineError(f"anchor components must be in [-1, 1], got [{', '.join(f'{v:g}' for v in uvw)}]",
                                 hint=ANCHOR_HINT)
        return uvw
    raise SightlineError(f"bad anchor {spec!r}", hint=ANCHOR_HINT)


def rotation_matrix(rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> np.ndarray:
    """R = Rz @ Ry @ Rx from degrees (X applied first)."""
    ax, ay, az = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return clean(mz @ my @ mx)


def axis_rotation(axis, degrees: float) -> np.ndarray:
    """Rotation matrix about a unit axis (right-hand rule)."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    a = math.radians(degrees)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return clean(np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * (kx @ kx))


def clean(values: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Snap float noise (e.g. cos 90 deg) to exact 0 / +-1 for deterministic, tidy numbers."""
    out = np.array(values, dtype=np.float64)
    out[np.abs(out) < tol] = 0.0
    ones = np.abs(np.abs(out) - 1.0) < tol
    out[ones] = np.sign(out[ones])
    return out


# ----------------------------------------------------------------------------- parts


def _regular_polygon(circumradius: float, sides: int) -> np.ndarray:
    """(n, 2) vertices (x, z) of a regular n-gon with a flat face toward +Z."""
    k = np.arange(sides)
    ang = (k + 0.5) * (2.0 * math.pi / sides)  # vertices sit between face normals
    return np.stack([circumradius * np.sin(ang), circumradius * np.cos(ang)], axis=1)


@dataclass(eq=False)
class Part:
    name: str
    shape: str
    params: dict  # normalized: size (3-tuple), radius, height, top_radius, sides, round, thickness
    matrix: np.ndarray  # (3, 3) orthogonal local -> world (det may be -1)
    position: np.ndarray  # (3,) world position of the local origin
    material: str | None  # None for cuts
    op: str = "union"  # "union" | "cut"
    cut_from: list[str] = field(default_factory=list)  # solid parts this cut applies to
    blend: float = 0.0
    blend_with: str | None = None
    attached_to: str | None = None  # target part name from `on`
    source: str = ""  # "parts[3] (sword)" or "parts[3] (sword) mirror", for messages
    tags: list[str] = field(default_factory=list)
    helper: bool = False  # render helpers (scale figure); excluded from inspect/checks
    local_half: np.ndarray | None = None  # (3,) half extents of the local bbox (computed if None)

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        if self.local_half is None:
            self.local_half = local_half_extents(self.shape, self.params)

    @property
    def is_solid(self) -> bool:
        return self.op != "cut"

    def to_local(self, pts) -> np.ndarray:
        """World points (N, 3) -> local frame (N, 3) float32."""
        p = np.asarray(pts, dtype=F32)
        return np.ascontiguousarray(self._local3(p.T).T)

    def to_world(self, local_pts) -> np.ndarray:
        """Local points (N, 3) -> world (N, 3) float64."""
        return np.asarray(local_pts, dtype=np.float64) @ self.matrix.T + self.position

    def _local3(self, p3: np.ndarray) -> np.ndarray:
        """(3, N) world -> (3, N) local, float32: M.T @ (p - pos)."""
        rel = p3 - self.position.astype(F32)[:, None]
        if np.array_equal(self.matrix, np.eye(3)):
            return rel
        return self.matrix.T.astype(F32) @ rel

    def sdf3(self, p3: np.ndarray) -> np.ndarray:
        """Distance of the primitive alone at (3, N) world points -> (N,) float32."""
        x, y, z = self._local3(p3)
        return sdf.primitive(self.shape, self.params, x, y, z)

    def sdf(self, pts) -> np.ndarray:
        """Distance of the primitive alone (no cuts or blends) at world points (N, 3) -> (N,) float32."""
        p = np.asarray(pts, dtype=F32).reshape(-1, 3)
        return np.asarray(self.sdf3(np.ascontiguousarray(p.T)), dtype=F32)

    def anchor(self, spec) -> np.ndarray:
        """World point of an anchor: 'top_front', '@[u, v, w]' or (u, v, w)."""
        return self.position + self.matrix @ (parse_anchor(spec) * self.local_half)

    def _hull_vertices(self) -> tuple[np.ndarray, float]:
        """Local vertices whose convex hull, grown by the returned radius, is the shape (polyhedral shapes)."""
        p = self.params
        if self.shape == "box":
            r = p["round"]
            h = np.array(p["size"]) / 2.0 - r
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
            return signs * h, r
        if self.shape == "extrude":
            hz = p["depth"] / 2.0
            return np.array([[x, y, z] for z in (-hz, hz) for x, y in p["points"]], dtype=np.float64), 0.0
        if self.shape == "wedge":
            hx, hy, hz = (v / 2.0 for v in p["size"])
            tri = [(-hy, -hz), (hy, -hz), (-hy, hz)]
            return np.array([[sx * hx, y, z] for sx in (-1, 1) for (y, z) in tri]), 0.0
        if self.shape == "cylinder":
            r, n = p["round"], p["sides"]
            apothem = p["radius"] * math.cos(math.pi / n) - r
            poly = _regular_polygon(apothem / math.cos(math.pi / n), n)
            hh = p["height"] / 2.0 - r
            verts = [[x, y, z] for y in (-hh, hh) for x, z in poly]
            return np.array(verts), r
        # polygonal cone / frustum
        n, hh = p["sides"], p["height"] / 2.0
        verts = []
        for y, rad in ((-hh, p["radius"]), (hh, p["top_radius"])):
            if rad > 0:
                verts += [[x, y, z] for x, z in _regular_polygon(rad, n)]
            else:
                verts.append([0.0, y, 0.0])
        return np.array(verts), 0.0

    def world_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact world AABB of the shape (not of its rotated bounding box)."""
        m, pos, p = self.matrix, self.position, self.params
        a = m[:, 1]  # world direction of the local Y axis
        across = np.sqrt(np.maximum(0.0, 1.0 - a * a))  # extent factor of a unit circle in local XZ
        polygonal = self.shape in ("box", "wedge", "extrude") or (self.shape in ("cylinder", "cone") and p.get("sides"))
        if polygonal:
            verts, grow = self._hull_vertices()
            world = verts @ m.T
            lo, hi = world.min(axis=0) - grow, world.max(axis=0) + grow
        elif self.shape == "sphere":
            lo, hi = -np.full(3, p["radius"]), np.full(3, p["radius"])
        elif self.shape == "ellipsoid":
            ext = np.sqrt(((m * (np.array(p["size"]) / 2.0)) ** 2).sum(axis=1))
            lo, hi = -ext, ext
        elif self.shape == "cylinder":
            r = p["round"]
            ext = (p["height"] / 2.0 - r) * np.abs(a) + (p["radius"] - r) * across + r
            lo, hi = -ext, ext
        elif self.shape == "capsule":
            ext = (p["height"] / 2.0 - p["radius"]) * np.abs(a) + p["radius"]
            lo, hi = -ext, ext
        elif self.shape == "torus":
            ext = p["radius"] * across + p["thickness"]
            lo, hi = -ext, ext
        elif self.shape == "cone":
            hh = p["height"] / 2.0
            caps = [(-hh * a, p["radius"]), (hh * a, p["top_radius"])]
            lo = np.min([c - r * across for c, r in caps], axis=0)
            hi = np.max([c + r * across for c, r in caps], axis=0)
        else:
            raise ValueError(f"unknown shape {self.shape!r}")
        return clean(pos + lo, 1e-9), clean(pos + hi, 1e-9)


def make_part(name: str, shape: str, params: dict, position, rotate_deg=(0, 0, 0), material: str | None = None,
              helper: bool = False, op: str = "union", *, source: str = "", tags=None) -> Part:
    """Build a Part from raw params; `position` is the world position of the local origin (bbox center)."""
    label = source or name
    params = normalize_params(shape, params, lambda key: f"{label}" + (f".{key}" if key else ""))
    return Part(
        name=name,
        shape=shape,
        params=params,
        matrix=rotation_matrix(*rotate_deg),
        position=np.asarray(position, dtype=np.float64),
        material=None if op == "cut" else material,
        op=op,
        source=source or name,
        tags=list(tags or []),
        helper=helper,
    )


# ----------------------------------------------------------------------------- groups and scene


@dataclass(eq=False)
class Group:
    """Evaluation unit: one solid part, or several joined by smooth blends."""

    members: list[int]  # indices into Scene.parts (solids), file order
    blend_k: dict[int, float]  # member -> k for smooth union with earlier members (0 = hard min)
    cutters: dict[int, list[int]]  # member -> cut part indices
    aabb_min: np.ndarray
    aabb_max: np.ndarray


def _aabb_dist3(p3: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance from (3, N) points to an AABB (0 inside) — a lower bound for anything inside it."""
    lo32, hi32 = lo.astype(F32)[:, None], hi.astype(F32)[:, None]
    q = np.maximum(np.maximum(lo32 - p3, p3 - hi32), 0.0)
    return np.sqrt((q * q).sum(axis=0))


class Scene:
    """Union of solid parts (with cuts and blends) as one signed distance field.

    `parts` always holds every part of the source (same indices as the asset), while
    `solid_indices()` and `groups` cover only the solids active in this scene, so part indices
    from a sub-scene (`only`) still index the full asset.
    """

    def __init__(self, parts: list[Part], solids: list[int] | None = None):
        self.parts = list(parts)
        all_solids = [i for i, p in enumerate(self.parts) if p.is_solid]
        if solids is None:
            solids = all_solids
        bad = [i for i in solids if not (0 <= i < len(self.parts)) or not self.parts[i].is_solid]
        if bad:
            raise ValueError(f"not solid part indices: {bad}")
        self._solids = sorted(set(solids))
        index: dict[str, int] = {}
        for i, p in enumerate(self.parts):
            index.setdefault(p.name, i)
        self._index = index
        # Cutters per solid (for every solid, active or not).
        self.cutters: dict[int, list[int]] = {i: [] for i in all_solids}
        for ci, p in enumerate(self.parts):
            if not p.is_solid:
                for name in p.cut_from:
                    j = index.get(name)
                    if j is not None and j in self.cutters and j < ci:
                        self.cutters[j].append(ci)
        self._aabb = {i: p.world_aabb() for i, p in enumerate(self.parts)}
        self.groups = self._build_groups()

    # -- construction

    def _build_groups(self) -> list[Group]:
        active = set(self._solids)
        parent = {i: i for i in self._solids}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        targets: dict[int, int] = {}
        for i in self._solids:
            p = self.parts[i]
            if p.blend > 0:
                name = p.blend_with or p.attached_to
                j = self._index.get(name) if name else None
                if j is not None and j in active and j != i:
                    targets[i] = j
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
        members: dict[int, list[int]] = {}
        for i in self._solids:
            members.setdefault(find(i), []).append(i)
        groups = []
        for root in sorted(members):
            mem = sorted(members[root])
            first = mem[0]
            blend_k = {m: (self.parts[m].blend if m in targets and m != first else 0.0) for m in mem}
            k_max = max(blend_k.values())
            lo = np.min([self._aabb[m][0] for m in mem], axis=0) - k_max
            hi = np.max([self._aabb[m][1] for m in mem], axis=0) + k_max
            groups.append(Group(mem, blend_k, {m: list(self.cutters[m]) for m in mem}, lo, hi))
        return groups

    def solid_indices(self, include_helpers: bool = False) -> list[int]:
        return [i for i in self._solids if include_helpers or not self.parts[i].helper]

    def only(self, indices) -> "Scene":
        """Sub-scene with these solids and their cutters (same Part objects and indices)."""
        return Scene(self.parts, [i for i in indices if self.parts[i].is_solid])

    def with_parts(self, extra_parts: list[Part]) -> "Scene":
        """This scene plus extra (helper) parts, each its own group."""
        n = len(self.parts)
        extra = list(extra_parts)
        new_solids = [n + k for k, p in enumerate(extra) if p.is_solid]
        return Scene(self.parts + extra, self._solids + new_solids)

    # -- evaluation on (3, N) float32 points

    def part_dist3(self, i: int, p3: np.ndarray, cull: float | None = None) -> np.ndarray:
        """Part i with its cutters applied. `cull` (m): skip a cutter for points farther than this
        outside its AABB — exact outside the part, and only changes values deeper inside it than `cull`."""
        d = self.parts[i].sdf3(p3)
        for c in self.cutters.get(i, ()):
            cutter = self.parts[c]
            if cull is None:
                d = np.maximum(d, -cutter.sdf3(p3))
                continue
            lo, hi = self._aabb[c]
            lo32, hi32 = (lo - cull).astype(F32)[:, None], (hi + cull).astype(F32)[:, None]
            near = np.all((p3 >= lo32) & (p3 <= hi32), axis=0)
            if near.all():
                d = np.maximum(d, -cutter.sdf3(p3))
            elif near.any():
                sel = np.flatnonzero(near)
                d[sel] = np.maximum(d[sel], -cutter.sdf3(p3[:, sel]))
        return d

    def group_dist3(self, g: Group, p3: np.ndarray, cull: float | None = None, need_part: bool = False):
        """Distance of a group at (3, N) points; with need_part also the nearest member index."""
        if len(g.members) == 1:
            d = self.part_dist3(g.members[0], p3, cull)
            if need_part:
                return d, np.full(d.shape, g.members[0], dtype=np.int32)
            return d
        dists = [self.part_dist3(m, p3, cull) for m in g.members]
        acc = dists[0]
        for m, dm in zip(g.members[1:], dists[1:]):
            acc = sdf.smin(acc, dm, g.blend_k.get(m, 0.0))
        acc = acc.astype(F32, copy=False)
        if need_part:
            nearest = np.argmin(np.stack(dists), axis=0)
            return acc, np.asarray(g.members, dtype=np.int32)[nearest]
        return acc

    def phantom3(self, field: Callable[[np.ndarray], np.ndarray], cutters, p3: np.ndarray, d: np.ndarray,
                 tau: float, fallback_dir: np.ndarray | None = None) -> np.ndarray:
        """Bool mask of points (3, N) whose field value `d` >= 0 is a cut artifact, not nearby material.

        `max(d_part, -d_cut)` is a valid lower bound, but where a cutter face is coplanar with a face
        of its target it falls to 0 on the shared plane over the hole, with no material anywhere near
        (a "phantom" surface). A point counts as real when no cutter is close, or when a step of
        `d + tau` down the field gradient lands in material (field < 0, which is exact). `field`
        evaluates the part or group at (3, M) points; `cutters` are the cut part indices involved.
        Where the gradient vanishes (exactly on a shared plane), the probe goes along `fallback_dir`
        ((3, 1) or (3, N): ray directions) or, without one, the point is a phantom.
        """
        out = np.zeros(d.shape, dtype=bool)
        if not cutters:
            return out
        cand = np.flatnonzero(d >= 0.0)
        if cand.size == 0:
            return out
        q3 = p3[:, cand]
        dc = d[cand].astype(F32)
        near = np.zeros(cand.size, dtype=bool)
        for c in cutters:
            near |= self.parts[c].sdf3(q3) < dc + F32(4.0 * tau)
        if not near.any():
            return out
        cand, q3, dc = cand[near], np.ascontiguousarray(q3[:, near]), dc[near]
        m = cand.size
        h = F32(0.5 * tau)
        offsets = np.zeros((6, 3, 1), dtype=F32)
        for k in range(3):
            offsets[2 * k, k, 0] = h
            offsets[2 * k + 1, k, 0] = -h
        probe = (q3[None, :, :] + offsets).transpose(1, 0, 2).reshape(3, 6 * m)
        f = np.asarray(field(np.ascontiguousarray(probe)), dtype=F32).reshape(6, m)
        grad = np.stack([f[0] - f[1], f[2] - f[3], f[4] - f[5]])  # (3, m), scaled by 2h
        length = np.sqrt((grad * grad).sum(axis=0))
        flat = length < F32(0.05) * h  # |gradient| < 0.025: the V-shaped minimum on a shared plane
        n = grad / np.where(flat, F32(1.0), length)
        if fallback_dir is not None:
            fd = fallback_dir if fallback_dir.shape[1] == 1 else fallback_dir[:, cand]
            n = np.where(flat[None, :], -np.broadcast_to(fd, n.shape), n)
        step = dc + F32(tau)
        target = q3 - n * step
        material = np.asarray(field(np.ascontiguousarray(target, dtype=F32)), dtype=F32) < 0.0
        if fallback_dir is None:
            material &= ~flat
        rest = np.flatnonzero(~material)
        if rest.size:  # another face as close as the one the gradient sees (a hole's corner): probe the axes
            r3, rs = q3[:, rest], step[rest]
            probes = np.concatenate([r3 + offsets[k] / h * rs for k in range(6)], axis=1)
            near = (np.asarray(field(np.ascontiguousarray(probes, dtype=F32)), dtype=F32) < 0.0).reshape(6, rest.size)
            material[rest] = near.any(axis=0)
        out[cand] = ~material
        return out

    # -- public (N, 3) API

    @staticmethod
    def _p3(pts) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(pts, dtype=F32).reshape(-1, 3).T)

    def eval_part(self, i: int, pts) -> np.ndarray:
        """Part i with its cutters applied, at world points (N, 3) -> (N,) float32."""
        return np.asarray(self.part_dist3(i, self._p3(pts)), dtype=F32)

    def eval_group(self, g, pts) -> tuple[np.ndarray, np.ndarray]:
        """(distance, nearest member part index) of a group (Group or index into `groups`)."""
        group = self.groups[g] if isinstance(g, (int, np.integer)) else g
        d, part = self.group_dist3(group, self._p3(pts), need_part=True)
        return np.asarray(d, dtype=F32), part

    def eval(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """Union of all groups: (distance, part index of the nearest solid, -1 if none).

        Groups whose AABB is farther than the best distance so far are skipped (exact).
        """
        p3 = self._p3(pts)
        n = p3.shape[1]
        best = np.full(n, np.inf, dtype=F32)
        part = np.full(n, -1, dtype=np.int32)
        for g in self.groups:
            sel = np.flatnonzero(_aabb_dist3(p3, g.aabb_min, g.aabb_max) < best)
            if sel.size == 0:
                continue
            d, idx = self.group_dist3(g, p3[:, sel], need_part=True)
            better = d < best[sel]
            best[sel[better]] = d[better]
            part[sel[better]] = idx[better]
        return best, part

    def bounds(self, include_helpers: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Union of the active solids' world AABBs."""
        ids = self.solid_indices(include_helpers)
        if not ids:
            return np.zeros(3), np.zeros(3)
        lo = np.min([self._aabb[i][0] for i in ids], axis=0)
        hi = np.max([self._aabb[i][1] for i in ids], axis=0)
        return lo, hi

    def aabb(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Cached world AABB of part i."""
        return self._aabb[i]


# ----------------------------------------------------------------------------- assets


@dataclass
class ReferenceSpec:
    """A parsed `reference:` entry."""

    image: pathlib.Path  # resolved, exists
    view: str  # camera view name; {azimuth, elevation} -> camera.orbit_view(...)
    height: float | None
    mask: str | pathlib.Path  # "auto" or resolved path
    lighting: str = "auto"  # "auto" | "lit" (shaded render, photo) | "flat" (flat paint)


@dataclass(eq=False)
class Asset:
    name: str
    path: pathlib.Path
    brief: str
    style: Style
    tags: list[str]
    parts: list[Part]  # expanded, file order
    checks: list  # raw check entries from YAML
    references: list[ReferenceSpec]
    raw: dict  # parsed YAML
    root: pathlib.Path | None = None  # project root the asset was loaded against
    _scene: Scene | None = field(default=None, init=False, repr=False, compare=False)
    _scene_key: tuple = field(default=(), init=False, repr=False, compare=False)

    def scene(self) -> Scene:
        """Scene of all parts (cached while the part list holds the same Part objects)."""
        key = tuple(id(p) for p in self.parts)
        if self._scene is None or key != self._scene_key:
            self._scene, self._scene_key = Scene(self.parts), key
        return self._scene

    def part(self, name: str) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        names = [p.name for p in self.parts]
        hint = did_you_mean(name, names) or f"parts: {', '.join(names)}"
        raise SightlineError(f"asset '{self.name}' has no part '{name}'", hint=hint)

    def part_index(self, name: str) -> int:
        target = self.part(name)
        return next(i for i, p in enumerate(self.parts) if p is target)
