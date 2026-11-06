"""Mesh export (SPEC section 9): surface nets over the part SDFs -> glTF 2.0 binary, OBJ + MTL.

Meshing is naive surface nets on a lattice of SDF samples. Every cell whose corners change sign
gets one vertex: the mean of its edge crossings, then projected onto the surface along the SDF
gradient and kept inside its cell. Every sign-changing lattice edge gives one quad joining the
four cells around it, split into two triangles wound counter-clockwise seen from outside.
Coordinates are meters with +Y up, the glTF convention, so there is no axis conversion.

Resolution: the base cell is the asset's longest side / `max_cells`. A part (blend group) whose
thinnest member or cutter spans fewer than MIN_FEATURE_CELLS base cells gets a finer cell (at most
MAX_REFINE times finer), so thin banners, rails and rings stay closed solids.

The .glb is written by hand: a JSON chunk padded with spaces and a BIN chunk padded with zeros,
one tightly packed bufferView per accessor (byteStride omitted), FLOAT / UNSIGNED_INT components.
"""

from __future__ import annotations

import json
import math
import pathlib
import struct
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .errors import SightlineError
from .model import F32, Asset, Group, Material, Scene

GLB_MAGIC = b"glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT, UNSIGNED_INT = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
TRIANGLES = 4

PAD_NODES = 2  # lattice nodes beyond a group's AABB on every side (all outside the surface)
MIN_FEATURE_CELLS = 3.0  # a part's thinnest extent should span at least this many cells
MAX_REFINE = 4.0  # ...but a part's cell is never finer than the base cell / MAX_REFINE
PROJECT_STEPS = 2  # Newton steps onto the surface per vertex
MAX_NODES = 40_000_000  # lattice size limit per group (memory guard, ~0.5 GB peak)
CHUNK = 1 << 20  # SDF points evaluated per call
# Decimation presets: the largest distance from the SDF surface a collapse may add, in cells of the
# part (None: no decimation, the surface-nets mesh as is). See decimate.py.
QUALITY = {"full": None, "high": 0.125, "medium": 0.25, "low": 0.5}
DEFAULT_QUALITY = "medium"

SdfFn = Callable[[np.ndarray], np.ndarray]  # (3, N) float32 world points -> (N,) float32

# Cube corners as (dx, dy, dz) and the 12 cube edges as corner pairs.
_CORNERS = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
_EDGES = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)
                   if np.abs(_CORNERS[a] - _CORNERS[b]).sum() == 1], dtype=np.int64)
# Cells around a lattice edge along axis a, as offsets on the next two axes (b, c) = (a+1, a+2) mod 3.
# Since e_b x e_c = e_a, this order is counter-clockwise seen from the + end of the edge.
_RING = ((-1, -1), (0, -1), (0, 0), (-1, 0))
_GRAD_OFFSETS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)


# ----------------------------------------------------------------------------- surface nets


def _lattice_axis(lo: float, hi: float, cell: float) -> tuple[float, int]:
    """First node coordinate and node count along one axis.

    The nodes are centered on the box (symmetric shapes give symmetric meshes) and reach at
    least PAD_NODES cells beyond it. The node parity is chosen so the box faces fall as far
    from nodes as possible, which avoids zero samples and degenerate triangles on flat faces.
    """
    half = (hi - lo) / 2.0 / cell
    frac = half % 1.0
    if min(frac, 1.0 - frac) >= abs(frac - 0.5):
        m = math.ceil(half + PAD_NODES - 1e-9)
        n = 2 * m + 1  # nodes at integer cell offsets from the center
    else:
        m = math.ceil(half + PAD_NODES - 0.5 - 1e-9)
        n = 2 * m + 2  # nodes at half-integer offsets
    return (lo + hi) / 2.0 - (n - 1) / 2.0 * cell, n


def _sample(fn: SdfFn, origin: np.ndarray, cell: float, shape: tuple[int, int, int]) -> np.ndarray:
    """SDF at every lattice node -> (nx, ny, nz) float32, evaluated in slabs of x."""
    nx, ny, nz = shape
    axes = [(origin[a] + cell * np.arange(shape[a])).astype(F32) for a in range(3)]
    yy, zz = (g.ravel() for g in np.meshgrid(axes[1], axes[2], indexing="ij"))
    out = np.empty(shape, dtype=F32)
    per_slab = ny * nz
    step = max(1, CHUNK // per_slab)
    for i0 in range(0, nx, step):
        k = min(nx, i0 + step) - i0
        p3 = np.empty((3, k * per_slab), dtype=F32)
        p3[0] = np.repeat(axes[0][i0 : i0 + k], per_slab)
        p3[1] = np.tile(yy, k)
        p3[2] = np.tile(zz, k)
        out[i0 : i0 + k] = np.asarray(fn(p3), dtype=F32).reshape(k, ny, nz)
    return out


def _eval_points(fn: SdfFn, pts: np.ndarray) -> np.ndarray:
    """SDF at (N, 3) float64 points -> (N,) float64, in chunks."""
    out = np.empty(len(pts), dtype=np.float64)
    for i0 in range(0, len(pts), CHUNK):
        p3 = np.ascontiguousarray(pts[i0 : i0 + CHUNK].T, dtype=F32)
        out[i0 : i0 + CHUNK] = fn(p3)
    return out


def sdf_gradient(fn: SdfFn, pts: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """(value (N,), central-difference gradient (N, 3)) of the SDF at (N, 3) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    value = np.empty(len(pts))
    grad = np.empty((len(pts), 3))
    step = CHUNK // 7
    for i0 in range(0, len(pts), step):
        p = pts[i0 : i0 + step]
        stencil = np.concatenate([p[None], p[None] + h * _GRAD_OFFSETS[:, None, :]]).reshape(-1, 3)
        vals = _eval_points(fn, stencil).reshape(7, len(p))
        value[i0 : i0 + step] = vals[0]
        grad[i0 : i0 + step] = np.stack([vals[1] - vals[2], vals[3] - vals[4], vals[5] - vals[6]], axis=1) / (2.0 * h)
    return value, grad


def surface_nets(fn: SdfFn, lo, hi, cell: float, project_steps: int = PROJECT_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """Mesh the zero level set of `fn` inside the box [lo, hi] (grown by PAD_NODES cells).

    Returns (vertices (V, 3) float64, triangles (T, 3) int64) with counter-clockwise winding seen
    from outside (where fn > 0). The mesh is closed: the outer node layers are forced outside.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    firsts, counts = zip(*(_lattice_axis(lo[a], hi[a], cell) for a in range(3)))
    origin, shape = np.array(firsts), tuple(counts)
    nodes = shape[0] * shape[1] * shape[2]
    if nodes > MAX_NODES:
        raise SightlineError(f"mesh lattice of {shape[0]} x {shape[1]} x {shape[2]} nodes is too large",
                             hint="lower max_cells (e.g. 64)")
    d = _sample(fn, origin, cell, shape)
    for a in range(3):  # outer layers outside, so every surface closes inside the lattice
        for end in (0, -1):
            sl = [slice(None)] * 3
            sl[a] = end
            face = d[tuple(sl)]
            face[face < 1e-6] = 1e-6
    inside = d < 0.0
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    if not inside.any():
        return empty

    # Active cells: corners of mixed sign.
    cshape = tuple(n - 1 for n in shape)
    count = np.zeros(cshape, dtype=np.uint8)
    inside8 = inside.view(np.uint8)
    for dx, dy, dz in _CORNERS:
        count += inside8[dx : dx + cshape[0], dy : dy + cshape[1], dz : dz + cshape[2]]
    active = (count > 0) & (count < 8)
    cells = np.argwhere(active)  # (M, 3), C order
    keys = np.ravel_multi_index(cells.T, cshape)  # sorted: vertex id of a cell = its rank in `keys`
    del count, active

    # One vertex per active cell: the mean of its edge crossings (lattice units).
    corner_d = np.stack([d[cells[:, 0] + ox, cells[:, 1] + oy, cells[:, 2] + oz]
                         for ox, oy, oz in _CORNERS], axis=1)  # (M, 8)
    total = np.zeros((len(cells), 3))
    hits = np.zeros(len(cells))
    for ea, eb in _EDGES:
        da, db = corner_d[:, ea].astype(np.float64), corner_d[:, eb].astype(np.float64)
        cross = (da < 0.0) != (db < 0.0)
        t = np.clip(np.where(cross, da / np.where(cross, da - db, 1.0), 0.0), 0.0, 1.0)
        total += cross[:, None] * (_CORNERS[ea] + t[:, None] * (_CORNERS[eb] - _CORNERS[ea]))
        hits += cross
    cell_lo = origin + cell * cells
    verts = cell_lo + cell * (total / hits[:, None])

    # Project onto the surface along the gradient (sharpens edges and corners), staying in the cell.
    for _ in range(project_steps):
        val, grad = sdf_gradient(fn, verts, 0.1 * cell)
        g2 = (grad * grad).sum(axis=1)
        step = np.where(g2 > 1e-12, val / np.maximum(g2, 1e-12), 0.0)
        verts = np.clip(verts - step[:, None] * grad, cell_lo, cell_lo + cell)

    # One quad per sign-changing lattice edge, from the four cells around it.
    quads = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        sl0, sl1 = [slice(1, -1)] * 3, [slice(1, -1)] * 3
        sl0[a], sl1[a] = slice(0, -1), slice(1, None)
        s0, s1 = inside[tuple(sl0)], inside[tuple(sl1)]
        change = s0 != s1
        node = np.argwhere(change)
        node[:, b] += 1
        node[:, c] += 1
        ring = []
        for ob, oc in _RING:
            q = node.copy()
            q[:, b] += ob
            q[:, c] += oc
            ring.append(np.searchsorted(keys, np.ravel_multi_index(q.T, cshape)))
        ring = np.stack(ring, axis=1)
        outward_minus = ~s0[change]  # inside at the + end: the surface faces -a, reverse the ring
        ring[outward_minus] = ring[outward_minus, ::-1]
        quads.append(ring)
    quads = np.concatenate(quads)
    # Every cell around a sign-changing edge is active (it contains that edge), so lookups are exact.
    if len(quads) == 0:
        return empty

    # Split along the shorter diagonal (both splits keep the winding).
    p = verts[quads]
    diag02 = ((p[:, 0] - p[:, 2]) ** 2).sum(axis=1)
    diag13 = ((p[:, 1] - p[:, 3]) ** 2).sum(axis=1)
    use02 = (diag02 <= diag13)[:, None]
    tri_a = np.where(use02, quads[:, [0, 1, 2]], quads[:, [0, 1, 3]])
    tri_b = np.where(use02, quads[:, [0, 2, 3]], quads[:, [1, 2, 3]])
    tris = np.stack([tri_a, tri_b], axis=1).reshape(-1, 3)

    used = np.zeros(len(verts), dtype=bool)
    used[tris.ravel()] = True
    if not used.all():  # defensive; every active cell is part of some quad
        remap = np.cumsum(used) - 1
        verts, tris = verts[used], remap[tris]
    return verts, tris


def vertex_normals(fn: SdfFn, verts: np.ndarray, tris: np.ndarray, h: float) -> np.ndarray:
    """Unit outward normals from the SDF gradient; area-weighted face normals where it vanishes."""
    _, grad = sdf_gradient(fn, verts, h)
    length = np.linalg.norm(grad, axis=1)
    bad = length < 1e-9
    if bad.any():
        p = verts[tris]
        face = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        acc = np.zeros_like(verts)
        for k in range(3):
            acc[:, 0] += np.bincount(tris[:, k], face[:, 0], len(verts))
            acc[:, 1] += np.bincount(tris[:, k], face[:, 1], len(verts))
            acc[:, 2] += np.bincount(tris[:, k], face[:, 2], len(verts))
        grad[bad] = acc[bad]
        length = np.linalg.norm(grad, axis=1)
    length = np.where(length < 1e-20, 1.0, length)
    normals = grad / length[:, None]
    normals[np.linalg.norm(normals, axis=1) < 0.5] = (0.0, 1.0, 0.0)  # isolated degenerate vertex
    return normals


# ----------------------------------------------------------------------------- part meshes


@dataclass(eq=False)
class PartMesh:
    """Triangle mesh of one blend group / solid part (or of the merged asset)."""

    name: str  # node name: the part (blend group: its first member)
    members: list[str]  # names of the solid parts meshed together
    positions: np.ndarray  # (V, 3) float32, asset space, meters
    normals: np.ndarray  # (V, 3) float32, unit, outward
    triangles: np.ndarray  # (T, 3) uint32, counter-clockwise seen from outside
    vertex_part: np.ndarray  # (V,) int32 scene part index nearest to each vertex
    tri_part: np.ndarray  # (T,) int32 scene part index nearest to each triangle center (-> material)
    cell: float  # lattice cell edge used (m)
    colors: np.ndarray | None = field(default=None)  # (V, 3) float32 baked sRGB pattern colors
    # Decimation (None: the surface-nets mesh as is).
    source_triangles: int | None = None  # triangles of the surface-nets mesh before decimation
    tolerance: float | None = None  # m, the largest distance from the SDF surface a collapse could add
    max_error: float | None = None  # m, measured: largest |SDF| at the vertices and triangle centers
    source_error: float | None = None  # m, the same measured on the surface-nets mesh

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    @property
    def triangles_before(self) -> int:
        """Triangles before decimation (= triangle_count when not decimated)."""
        return self.triangle_count if self.source_triangles is None else self.source_triangles

    def volume(self) -> float:
        """Enclosed volume (m^3), positive for outward winding (divergence theorem)."""
        p = self.positions.astype(np.float64)[self.triangles]
        return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)

    def area(self) -> float:
        p = self.positions.astype(np.float64)[self.triangles]
        return float(np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum() / 2.0)


def _check_max_cells(max_cells) -> int:
    if isinstance(max_cells, bool) or not isinstance(max_cells, (int, np.integer)) or not 8 <= max_cells <= 512:
        raise SightlineError(f"max_cells must be an integer in 8..512, got {max_cells!r}",
                             hint="128 is the default; 64 for a quick low-poly mesh, 256 for detail")
    return int(max_cells)


def resolve_quality(quality: str | None, max_tris, bake_colors: bool = False) -> tuple[str, float | None, int | None]:
    """(preset name, decimation tolerance in cells or None for no decimation, triangle budget or None).

    quality None means DEFAULT_QUALITY, or "full" with bake_colors: vertex colors need the dense
    surface-nets vertices, so baking never decimates."""
    if quality is not None and (not isinstance(quality, str) or quality not in QUALITY):
        raise SightlineError(f"quality must be one of {', '.join(QUALITY)}, got {quality!r}",
                             hint=f"'{DEFAULT_QUALITY}' is the default; 'full' keeps every surface-nets triangle")
    if max_tris is not None and (isinstance(max_tris, bool) or not isinstance(max_tris, (int, np.integer))
                                 or max_tris < 4):
        raise SightlineError(f"max_tris must be an integer >= 4, got {max_tris!r}",
                             hint="a triangle budget for the whole asset, e.g. 2000 for a small prop")
    if bake_colors and (quality not in (None, "full") or max_tris is not None):
        raise SightlineError("bake_colors needs the full mesh: vertex colors need the dense surface-nets vertices",
                             hint="drop --quality/--max-tris (baking exports quality full) or drop --bake-colors")
    if quality is None:
        quality = "full" if bake_colors else DEFAULT_QUALITY
    if quality == "full" and max_tris is not None:
        raise SightlineError("max_tris decimates, but quality 'full' keeps every triangle",
                             hint="use max_tris with quality high, medium or low")
    return quality, QUALITY[quality], None if max_tris is None else int(max_tris)


def _thinnest(scene: Scene, g: Group) -> float:
    """Smallest full local extent among the group's members and their cutters (m)."""
    ids = list(g.members) + [c for m in g.members for c in g.cutters.get(m, [])]
    return float(min(2.0 * scene.parts[i].local_half.min() for i in ids))


def group_cell(scene: Scene, g: Group, base_cell: float) -> float:
    """Cell for one group: the base cell, refined for thin members (see module docstring)."""
    return max(min(base_cell, _thinnest(scene, g) / MIN_FEATURE_CELLS), base_cell / MAX_REFINE)


def _group_fn(scene: Scene, g: Group) -> SdfFn:
    return lambda p3: scene.group_dist3(g, p3)


def mesh_group(scene: Scene, g: Group, cell: float) -> PartMesh | None:
    """Surface-nets mesh of one group with its cuts and blends applied (None if nothing is left)."""
    fn = _group_fn(scene, g)
    verts, tris = surface_nets(fn, g.aabb_min, g.aabb_max, cell)
    if len(tris) == 0:
        return None
    normals = vertex_normals(fn, verts, tris, 0.25 * cell)
    _, vertex_part = scene.group_dist3(g, np.ascontiguousarray(verts.T, dtype=F32), need_part=True)
    centers = verts[tris].mean(axis=1)
    _, tri_part = scene.group_dist3(g, np.ascontiguousarray(centers.T, dtype=F32), need_part=True)
    return PartMesh(
        name=scene.parts[g.members[0]].name,
        members=[scene.parts[m].name for m in g.members],
        positions=verts.astype(F32),
        normals=normals.astype(F32),
        triangles=tris.astype(np.uint32),
        vertex_part=np.asarray(vertex_part, dtype=np.int32),
        tri_part=np.asarray(tri_part, dtype=np.int32),
        cell=float(cell),
    )


def build_meshes(asset: Asset, max_cells: int = 128, bake_colors: bool = False, quality: str | None = None,
                 max_tris: int | None = None) -> list[PartMesh]:
    """One mesh per blend group / solid part of the asset, in file order (helpers excluded).

    quality: decimation preset (QUALITY; None = DEFAULT_QUALITY, or "full" when baking colors).
    max_tris: triangle budget for the whole asset (decimates further, relaxing the tolerance)."""
    max_cells = _check_max_cells(max_cells)
    _, tolerance, max_tris = resolve_quality(quality, max_tris, bake_colors)
    scene = asset.scene()
    if not scene.solid_indices():
        raise SightlineError(f"asset '{asset.name}' has no solid parts to export",
                             hint="add a part with a material (cuts alone produce no geometry)")
    lo, hi = scene.bounds(include_helpers=False)
    base = float((hi - lo).max()) / max_cells
    meshes = []
    for g in scene.groups:
        if all(scene.parts[m].helper for m in g.members):
            continue
        mesh = mesh_group(scene, g, group_cell(scene, g, base))
        if mesh is not None:
            meshes.append(mesh)
    if not meshes:
        raise SightlineError(f"asset '{asset.name}': every solid part is cut away, nothing to export")
    if tolerance is not None:
        meshes = _decimate(scene, meshes, tolerance, max_tris)
    if bake_colors:
        for mesh in meshes:
            bake_mesh_colors(asset, mesh)
    return meshes


def decimate_meshes(asset: Asset, meshes: list[PartMesh], quality: str = DEFAULT_QUALITY,
                    max_tris: int | None = None) -> list[PartMesh]:
    """Decimate build_meshes(..., quality="full") output of `asset` (new meshes; the input is kept)."""
    _, tolerance, max_tris = resolve_quality(quality, max_tris)
    if tolerance is None:
        return list(meshes)
    if any(m.colors is not None for m in meshes):
        raise SightlineError("cannot decimate meshes with baked colors", hint="decimate first, then bake")
    return _decimate(asset.scene(), meshes, tolerance, max_tris)


def _decimate(scene: Scene, meshes: list[PartMesh], tolerance: float, max_tris: int | None) -> list[PartMesh]:
    """Quadric edge collapse of every part mesh (decimate.py), tolerance in cells of each part."""
    from .decimate import DecimationInput, decimate

    groups = {scene.parts[g.members[0]].name: g for g in scene.groups}
    unknown = [m.name for m in meshes if m.name not in groups]
    if unknown:
        raise SightlineError(f"cannot decimate mesh '{unknown[0]}': no part or blend group of that name",
                             hint="decimate the per-part meshes of build_meshes(..., quality='full'), then merge")
    materials = sorted({p.material for p in scene.parts if p.material is not None})
    part_material = np.array([materials.index(p.material) if p.material in materials else -1 for p in scene.parts])
    inputs = [DecimationInput(positions=m.positions, triangles=m.triangles, normals=m.normals,
                              labels=part_material[m.tri_part], fn=_group_fn(scene, groups[m.name]), cell=m.cell)
              for m in meshes]
    out = []
    for m, r in zip(meshes, decimate(inputs, tolerance, max_tris)):
        out.append(PartMesh(
            name=m.name,
            members=list(m.members),
            positions=r.positions.astype(F32),
            normals=r.normals.astype(F32),
            triangles=r.triangles.astype(np.uint32),
            vertex_part=m.vertex_part[r.vertex_ids],
            tri_part=m.tri_part[r.triangle_ids],
            cell=m.cell,
            source_triangles=m.triangles_before,
            tolerance=r.tolerance,
            max_error=r.max_error,
            source_error=r.source_error if m.source_error is None else max(m.source_error, r.source_error),
        ))
    return out


def merge_meshes(meshes: list[PartMesh], name: str) -> PartMesh:
    """All meshes joined into one (vertex indices offset; overlapping parts keep their inner faces)."""
    offsets = np.cumsum([0] + [m.vertex_count for m in meshes])[:-1]
    colors = None
    if all(m.colors is not None for m in meshes):
        colors = np.concatenate([m.colors for m in meshes])
    decimated = [m for m in meshes if m.source_triangles is not None]
    worst = lambda key: max(getattr(m, key) for m in decimated) if decimated else None  # noqa: E731
    return PartMesh(
        name=name,
        members=[n for m in meshes for n in m.members],
        positions=np.concatenate([m.positions for m in meshes]),
        normals=np.concatenate([m.normals for m in meshes]),
        triangles=np.concatenate([m.triangles + np.uint32(o) for m, o in zip(meshes, offsets)]),
        vertex_part=np.concatenate([m.vertex_part for m in meshes]),
        tri_part=np.concatenate([m.tri_part for m in meshes]),
        cell=min(m.cell for m in meshes),
        colors=colors,
        source_triangles=sum(m.triangles_before for m in meshes) if decimated else None,
        tolerance=worst("tolerance"),
        max_error=worst("max_error"),
        source_error=worst("source_error"),
    )


def _mm(value: float | None, width: int = 6) -> str:
    return f"{'-':>{width}}" if value is None else f"{value * 1000:>{width}.2f}"


def mesh_summary(meshes: list[PartMesh]) -> str:
    """Aligned text table: part, vertices, triangles exported and before decimation, cell, largest
    distance of the vertices and triangle centers from the SDF surface (mm, decimated meshes only),
    volume; plus totals."""
    width = max([5] + [len(m.name) for m in meshes])
    lines = [f"{'PART':<{width}}  {'VERTS':>7}  {'TRIS':>7}  {'BEFORE':>7}  {'CELL':>6}  {'DEV_MM':>6}  {'VOLUME':>8}"]
    for m in meshes:
        lines.append(f"{m.name:<{width}}  {m.vertex_count:>7}  {m.triangle_count:>7}  {m.triangles_before:>7}  "
                     f"{m.cell:>6.3f}  {_mm(m.max_error)}  {m.volume():>8.3f}")
    errors = [m.max_error for m in meshes if m.max_error is not None]
    lines.append(f"{'total':<{width}}  {sum(m.vertex_count for m in meshes):>7}  "
                 f"{sum(m.triangle_count for m in meshes):>7}  {sum(m.triangles_before for m in meshes):>7}  "
                 f"{'':>6}  {_mm(max(errors) if errors else None)}")
    return "\n".join(lines)


def format_decimation(meshes: list[PartMesh]) -> str:
    """One DECIMATE line for the export report: triangles before -> after and the measured accuracy."""
    after = sum(m.triangle_count for m in meshes)
    before = sum(m.triangles_before for m in meshes)
    done = [m for m in meshes if m.source_triangles is not None]
    if not done:
        return f"DECIMATE  off: {after} surface-nets triangles as meshed (quality full)"
    dev = max(m.max_error for m in done)
    src = max(m.source_error for m in done)
    tols = sorted({round(m.tolerance * 1000, 2) for m in done})
    tol = f"{tols[0]:.2f}" if len(tols) == 1 else f"{tols[0]:.2f}-{tols[-1]:.2f}"
    return (f"DECIMATE  {before} -> {after} triangles ({before / max(after, 1):.1f}x fewer); largest distance from the "
            f"SDF surface {dev * 1000:.2f} mm (tolerance {tol} mm, surface nets alone {src * 1000:.2f} mm)")


def edge_report(meshes: list[PartMesh]) -> list[tuple[str, int, int]]:
    """(mesh name, open edges, non-manifold edges) for every mesh whose edges are not all shared by exactly
    two triangles. Surface nets can join four triangles at one edge where a cut meets a face at a concave
    corner (e.g. the top corners of a door cut): engines import that, manifold checks flag it."""
    out = []
    for m in meshes:
        if m.triangle_count == 0:
            continue
        t = np.asarray(m.triangles, dtype=np.int64)
        edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        edges.sort(axis=1)
        _, uses = np.unique(edges[:, 0] * (int(t.max()) + 1) + edges[:, 1], return_counts=True)
        open_edges, shared = int((uses == 1).sum()), int((uses > 2).sum())
        if open_edges or shared:
            out.append((m.name, open_edges, shared))
    return out


def format_edges(meshes: list[PartMesh]) -> str:
    """One MANIFOLD line for the export report."""
    bad = edge_report(meshes)
    if not bad:
        return "MANIFOLD  yes: every mesh is closed and each edge joins exactly 2 triangles"
    items = "; ".join(f"{name}: {o} open, {n} non-manifold edge(s)" for name, o, n in bad)
    return (f"MANIFOLD  no: {items} — engines import this; Blender's manifold check, 3D printing and some "
            "physics cookers flag it")


# ----------------------------------------------------------------------------- colors and patterns


def srgb_to_linear(c) -> np.ndarray:
    """sRGB 0..1 -> linear 0..1 (IEC 61966-2-1)."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# 3 x 3 tangent-plane offsets (in cells) averaged per vertex, so pattern details finer than the
# vertex spacing blend into their mean color instead of aliasing.
_BAKE_OFFSETS = np.array([(a, b) for a in (-1 / 3, 0.0, 1 / 3) for b in (-1 / 3, 0.0, 1 / 3)])


def _tangents(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit tangents (N, 3) perpendicular to unit normals n (N, 3)."""
    helper = np.where((np.abs(n[:, 0]) < 0.9)[:, None], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t1 = np.cross(n, helper)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    return t1, np.cross(n, t1)


def bake_mesh_colors(asset: Asset, mesh: PartMesh) -> PartMesh:
    """Fill mesh.colors (sRGB) with the material pattern around each vertex, as render.py shades it.

    Evaluated in the part-local frame of the vertex's nearest part, averaged over a 3 x 3 patch of
    the tangent plane one cell wide; render.surface_color fades patterns finer than the cell.
    """
    from .render import material_for, surface_color  # lazy: export works without the renderer

    parts = asset.scene().parts
    colors = np.empty((mesh.vertex_count, 3), dtype=np.float64)
    k = len(_BAKE_OFFSETS)
    for pi in np.unique(mesh.vertex_part):
        sel = np.flatnonzero(mesh.vertex_part == pi)
        part = parts[int(pi)]
        local = part.to_local(mesh.positions[sel]).astype(np.float64)
        n_local = mesh.normals[sel].astype(np.float64) @ part.matrix  # M.T @ n, row form
        t1, t2 = _tangents(n_local)
        off = mesh.cell * (_BAKE_OFFSETS[None, :, 0, None] * t1[:, None] + _BAKE_OFFSETS[None, :, 1, None] * t2[:, None])
        pts = (local[:, None, :] + off).reshape(-1, 3).astype(F32)
        nrm = np.repeat(n_local, k, axis=0).astype(F32)
        rgb = surface_color(material_for(asset.style, part.material), pts, nrm, mesh.cell)
        colors[sel] = np.asarray(rgb, dtype=np.float64).reshape(len(sel), k, 3).mean(axis=1)
    mesh.colors = np.clip(colors, 0.0, 1.0).astype(F32)
    return mesh


def _material_factor(material: Material, baked: bool) -> np.ndarray:
    """Linear baseColor RGB. Baked: per-channel max of base and pattern color so COLOR_0 stays in [0, 1]."""
    lin = srgb_to_linear(material.color)
    if baked and material.pattern not in (None, "solid"):
        pattern = material.pattern_color if material.pattern_color is not None else np.asarray(material.color) * 0.8
        lin = np.maximum(lin, srgb_to_linear(pattern))
    return lin


def gltf_material(material: Material, baked: bool = False) -> dict:
    """glTF material: baseColorFactor (linear), metallic = shine * 0.8, roughness = 1 - shine * 0.6."""
    rgb = _material_factor(material, baked)
    return {
        "name": material.name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [round(float(v), 6) for v in rgb] + [1.0],
            "metallicFactor": round(material.shine * 0.8, 6),
            "roughnessFactor": round(1.0 - material.shine * 0.6, 6),
        },
    }


# ----------------------------------------------------------------------------- glTF binary


class _BinBuilder:
    """BIN chunk with one tightly packed bufferView per accessor."""

    def __init__(self):
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, array: np.ndarray, type_: str, target: int, bounds: bool = False) -> int:
        array = np.ascontiguousarray(array)
        component = FLOAT if array.dtype == np.float32 else UNSIGNED_INT
        if array.dtype not in (np.float32, np.uint32):
            raise TypeError(f"unsupported accessor dtype {array.dtype}")
        data = array.tobytes()
        self.views.append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data), "target": target})
        self.blob += data
        self.blob += b"\0" * (-len(self.blob) % 4)  # 4-byte components keep this a no-op
        acc = {"bufferView": len(self.views) - 1, "componentType": component,
               "count": int(array.shape[0]), "type": type_}
        if bounds:
            acc["min"] = [float(v) for v in array.min(axis=0)]
            acc["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def _used_materials(parts, meshes: list[PartMesh]) -> tuple[list[str], np.ndarray]:
    """Material names used by the meshes' triangles or vertices (first use in file order) and part ->
    material index (-1 only for parts no mesh element refers to)."""
    used = set(np.concatenate([m.tri_part for m in meshes] + [m.vertex_part for m in meshes]).tolist())
    names: list[str] = []
    for i, p in enumerate(parts):
        if i in used and p.material not in names:
            names.append(p.material)
    part_mat = np.array([names.index(p.material) if p.material in names else -1 for p in parts], dtype=np.int64)
    return names, part_mat


def glb_bytes(asset: Asset, meshes: list[PartMesh]) -> bytes:
    """The complete .glb file for these meshes: one node + mesh each, one primitive per material."""
    parts = asset.scene().parts
    mat_names, part_mat = _used_materials(parts, meshes)
    baked = any(m.colors is not None for m in meshes)
    styles = [asset.style.material(n) for n in mat_names]
    factors = np.array([_material_factor(s, baked) for s in styles])

    buf = _BinBuilder()
    nodes, gl_meshes = [], []
    for mesh in meshes:
        attrs = {
            "POSITION": buf.add(mesh.positions.astype(F32), "VEC3", ARRAY_BUFFER, bounds=True),
            "NORMAL": buf.add(mesh.normals.astype(F32), "VEC3", ARRAY_BUFFER),
        }
        if baked:
            colors = mesh.colors if mesh.colors is not None else np.ones((mesh.vertex_count, 3), dtype=F32)
            factor = factors[part_mat[mesh.vertex_part]]
            lin = srgb_to_linear(colors) if mesh.colors is not None else factor
            ratio = np.clip(lin / np.maximum(factor, 1e-9), 0.0, 1.0).astype(F32)
            attrs["COLOR_0"] = buf.add(ratio, "VEC3", ARRAY_BUFFER)
        tri_mat = part_mat[mesh.tri_part]
        primitives = []
        for mi in np.unique(tri_mat):
            idx = mesh.triangles[tri_mat == mi].reshape(-1).astype(np.uint32)
            primitives.append({"attributes": dict(attrs), "indices": buf.add(idx, "SCALAR", ELEMENT_ARRAY_BUFFER),
                               "material": int(mi), "mode": TRIANGLES})
        gl_meshes.append({"name": mesh.name, "primitives": primitives})
        nodes.append({"name": mesh.name, "mesh": len(gl_meshes) - 1, "extras": {"parts": list(mesh.members)}})

    doc = {
        "asset": {"version": "2.0", "generator": "sightline",
                  "extras": {"asset": asset.name, "style": asset.style.name}},
        "scene": 0,
        "scenes": [{"name": asset.name, "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": [gltf_material(s, baked) for s in styles],
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.blob)}],
    }
    js = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * (-len(js) % 4)
    blob = bytes(buf.blob) + b"\0" * (-len(buf.blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return b"".join([
        struct.pack("<4sII", GLB_MAGIC, 2, total),
        struct.pack("<II", len(js), CHUNK_JSON), js,
        struct.pack("<II", len(blob), CHUNK_BIN), blob,
    ])


def _out_file(out_path, asset: Asset, suffix: str, func: str) -> pathlib.Path:
    path = pathlib.Path(out_path)
    if path.is_dir():
        path = path / f"{asset.name}{suffix}"
    elif path.suffix == "":
        path = path.with_suffix(suffix)
    elif path.suffix.lower() != suffix:
        raise SightlineError(f"{func} writes {suffix} files, got '{path.name}'",
                             hint="use export_glb for .glb and export_obj for .obj")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SightlineError(f"cannot create the folder {path.parent}", hint=str(exc)) from None
    return path


def _write(path: pathlib.Path, data: bytes) -> None:
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise SightlineError(f"cannot write {path}", hint=str(exc)) from None


def export_glb(asset: Asset, out_path, max_cells: int = 128, per_part: bool = True, bake_colors: bool = False,
               *, meshes: list[PartMesh] | None = None, quality: str | None = None,
               max_tris: int | None = None) -> pathlib.Path:
    """Write a glTF 2.0 binary. per_part: one node + mesh per blend group / solid part, named after
    the part; else one merged node named after the asset. `meshes` reuses build_meshes() output;
    otherwise quality / max_tris decimate as in build_meshes."""
    path = _out_file(out_path, asset, ".glb", "export_glb")
    if meshes is None:
        meshes = build_meshes(asset, max_cells, bake_colors, quality, max_tris)
    elif bake_colors:
        meshes = [m if m.colors is not None else bake_mesh_colors(asset, m) for m in meshes]
    if not per_part:
        meshes = [merge_meshes(meshes, asset.name)]
    _write(path, glb_bytes(asset, meshes))
    return path


# ----------------------------------------------------------------------------- OBJ + MTL


def _rows(line: str, values: np.ndarray, decimals: int | None = None) -> str:
    """Format every row of `values` with the printf-style `line` (one row's worth of fields)."""
    if len(values) == 0:
        return ""
    if decimals is not None:
        values = np.round(values.astype(np.float64), decimals) + 0.0  # + 0.0 turns -0.0 into 0.0
    return (line * len(values)) % tuple(values.ravel().tolist())


def export_obj(asset: Asset, out_path, max_cells: int = 128, *, meshes: list[PartMesh] | None = None,
               quality: str | None = None, max_tris: int | None = None) -> pathlib.Path:
    """Write OBJ + MTL next to it: `o <part>` per part, `usemtl <material>` per material, Kd from the style.

    Kd/Ks are the style's sRGB values as written (OBJ has no color-space convention).
    """
    path = _out_file(out_path, asset, ".obj", "export_obj")
    mtl_path = path.with_suffix(".mtl")
    if meshes is None:
        meshes = build_meshes(asset, max_cells, quality=quality, max_tris=max_tris)
    parts = asset.scene().parts
    mat_names, part_mat = _used_materials(parts, meshes)

    out = [f"# Sightline OBJ export of '{asset.name}' (style {asset.style.name}); meters, +Y up\n",
           f"# objects {len(meshes)}, vertices {sum(m.vertex_count for m in meshes)}, "
           f"triangles {sum(m.triangle_count for m in meshes)}\n",
           f"mtllib {mtl_path.name}\n"]
    offset = 1
    for mesh in meshes:
        out.append(f"o {mesh.name}\n")
        out.append(_rows("v %.5f %.5f %.5f\n", mesh.positions, 5))
        out.append(_rows("vn %.4f %.4f %.4f\n", mesh.normals, 4))
        tri_mat = part_mat[mesh.tri_part]
        for mi, name in enumerate(mat_names):
            sel = tri_mat == mi
            if not sel.any():
                continue
            out.append(f"usemtl {name}\n")
            idx = mesh.triangles[sel].astype(np.int64) + offset
            out.append(_rows("f %d//%d %d//%d %d//%d\n", np.repeat(idx, 2, axis=1)))
        offset += mesh.vertex_count
    obj_text = "".join(out)

    mtl =[f"# Sightline materials of '{asset.name}' (style {asset.style.name}); Kd = sRGB as written\n"]
    for name in mat_names:
        m = asset.style.material(name)
        r, g, b = (float(v) for v in m.color)
        mtl.append(f"newmtl {name}\n")
        mtl.append(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
        mtl.append("Ka 0.0000 0.0000 0.0000\n")
        mtl.append(f"Ks {m.shine:.4f} {m.shine:.4f} {m.shine:.4f}\n")
        mtl.append(f"Ns {24.0 if m.shine > 0 else 1.0:.1f}\n")
        mtl.append("d 1.0\n")
        mtl.append(f"illum {2 if m.shine > 0 else 1}\n\n")
    _write(mtl_path, "".join(mtl).encode("utf-8"))
    _write(path, obj_text.encode("utf-8"))
    return path


# ----------------------------------------------------------------------------- reading and validation

_COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_COMPONENT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _split_glb(data: bytes) -> tuple[dict, bytes]:
    """(JSON document, BIN chunk bytes) of a .glb; ValueError with a reason if malformed."""
    if len(data) < 20:
        raise ValueError(f"file is {len(data)} bytes, too short for a GLB header")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"magic is {magic!r}, expected b'glTF'")
    if version != 2:
        raise ValueError(f"GLB version {version}, expected 2")
    if length != len(data):
        raise ValueError(f"header length {length} != file size {len(data)}")
    js_len, js_type = struct.unpack_from("<II", data, 12)
    if js_type != CHUNK_JSON:
        raise ValueError(f"first chunk type 0x{js_type:08x} is not JSON")
    if js_len % 4 or 20 + js_len > len(data):
        raise ValueError(f"JSON chunk length {js_len} is not 4-byte aligned or overruns the file")
    try:
        doc = json.loads(data[20 : 20 + js_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON chunk does not parse: {exc}") from None
    pos, blob = 20 + js_len, b""
    if pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("truncated BIN chunk header")
        bin_len, bin_type = struct.unpack_from("<II", data, pos)
        if bin_type != CHUNK_BIN:
            raise ValueError(f"second chunk type 0x{bin_type:08x} is not BIN")
        if bin_len % 4 or pos + 8 + bin_len != len(data):
            raise ValueError(f"BIN chunk length {bin_len} is not 4-byte aligned or does not end the file")
        blob = data[pos + 8 : pos + 8 + bin_len]
    return doc, blob


def read_glb(path) -> tuple[dict, bytes]:
    """Parse a .glb file into (glTF JSON document, BIN chunk bytes)."""
    path = pathlib.Path(path)
    try:
        return _split_glb(path.read_bytes())
    except OSError as exc:
        raise SightlineError(f"cannot read {path}", hint=str(exc)) from None
    except ValueError as exc:
        raise SightlineError(f"{path.name} is not a valid glTF binary: {exc}") from None


def accessor_data(doc: dict, blob: bytes, index: int) -> np.ndarray:
    """Decode accessor `index` to an array of shape (count,) or (count, components)."""
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    ncomp = _TYPE_SIZE[acc["type"]]
    dtype = np.dtype(_COMPONENT_DTYPE[acc["componentType"]]).newbyteorder("<")
    elem = ncomp * dtype.itemsize
    stride = view.get("byteStride", elem)
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    raw = np.frombuffer(blob, dtype=np.uint8, count=(count - 1) * stride + elem, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, elem), strides=(stride, 1))
    out = np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)
    return out[:, 0] if ncomp == 1 else out


def validate_glb(path_or_bytes) -> list[str]:
    """Structural checks a glTF validator makes, plus this writer's invariants. [] = valid.

    Header/chunks (magic, version, lengths, 4-byte alignment), one embedded buffer whose length
    equals the sum of its bufferViews, views and accessors in bounds and aligned, POSITION
    min/max equal to the data, unit normals, COLOR_0 in [0, 1], indices in range and a multiple
    of 3, valid mesh/material/node/scene references, material factors in [0, 1].
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        data = pathlib.Path(path_or_bytes).read_bytes()
    try:
        doc, blob = _split_glb(data)
    except ValueError as exc:
        return [str(exc)]
    problems: list[str] = []
    bad = problems.append
    if doc.get("asset", {}).get("version") != "2.0":
        bad("asset.version must be '2.0'")
    buffers = doc.get("buffers", [])
    if len(buffers) != 1 or "uri" in buffers[0]:
        bad("expected exactly one buffer embedded in the BIN chunk (no uri)")
        return problems
    buf_len = buffers[0].get("byteLength", -1)
    if not 0 <= len(blob) - buf_len < 4:
        bad(f"buffers[0].byteLength {buf_len} does not match the BIN chunk ({len(blob)} bytes incl. padding)")
    views = doc.get("bufferViews", [])
    for i, v in enumerate(views):
        off, length = v.get("byteOffset", 0), v.get("byteLength", 0)
        if v.get("buffer") != 0:
            bad(f"bufferViews[{i}].buffer must be 0")
        if off % 4:
            bad(f"bufferViews[{i}].byteOffset {off} is not 4-byte aligned")
        if length <= 0 or off + length > buf_len:
            bad(f"bufferViews[{i}] [{off}, {off + length}) is outside the buffer ({buf_len} bytes)")
        if "byteStride" in v and (v["byteStride"] % 4 or not 4 <= v["byteStride"] <= 252):
            bad(f"bufferViews[{i}].byteStride {v['byteStride']} is invalid")
        if "target" in v and v["target"] not in (ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER):
            bad(f"bufferViews[{i}].target {v['target']} is invalid")
    covered = sum(v.get("byteLength", 0) for v in views)
    if covered != buf_len:
        bad(f"bufferViews cover {covered} bytes, the buffer has {buf_len}")
    accessors = doc.get("accessors", [])
    decoded: dict[int, np.ndarray] = {}
    for i, a in enumerate(accessors):
        where = f"accessors[{i}]"
        if a.get("componentType") not in _COMPONENT_SIZE or a.get("type") not in _TYPE_SIZE:
            bad(f"{where}: bad componentType/type {a.get('componentType')}/{a.get('type')}")
            continue
        vi = a.get("bufferView")
        if not isinstance(vi, int) or not 0 <= vi < len(views):
            bad(f"{where}.bufferView {vi} is out of range")
            continue
        view = views[vi]
        csize = _COMPONENT_SIZE[a["componentType"]]
        elem = csize * _TYPE_SIZE[a["type"]]
        stride = view.get("byteStride", elem)
        count, off = a.get("count", 0), a.get("byteOffset", 0)
        if count < 1:
            bad(f"{where}.count must be >= 1")
            continue
        if (view.get("byteOffset", 0) + off) % csize:
            bad(f"{where}: data is not aligned to its {csize}-byte component size")
        if off + (count - 1) * stride + elem > view.get("byteLength", 0):
            bad(f"{where}: {count} elements overrun bufferViews[{vi}] ({view.get('byteLength', 0)} bytes)")
            continue
        arr = accessor_data(doc, blob, i)
        decoded[i] = arr
        if a["componentType"] == FLOAT and not np.isfinite(arr).all():
            bad(f"{where}: contains NaN or infinity")
        for key, reduce in (("min", np.min), ("max", np.max)):
            if key in a:
                want = reduce(arr.reshape(count, -1), axis=0)
                got = np.asarray(a[key], dtype=np.float64)
                if got.shape != want.shape or not np.array_equal(got.astype(arr.dtype), want):
                    bad(f"{where}.{key} {a[key]} != data {key} {want.tolist()}")
    materials = doc.get("materials", [])
    for i, m in enumerate(materials):
        pbr = m.get("pbrMetallicRoughness", {})
        factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
        if len(factor) != 4 or not all(0.0 <= float(v) <= 1.0 for v in factor):
            bad(f"materials[{i}].baseColorFactor {factor} must be 4 numbers in [0, 1]")
        for key in ("metallicFactor", "roughnessFactor"):
            if not 0.0 <= float(pbr.get(key, 1.0)) <= 1.0:
                bad(f"materials[{i}].{key} must be in [0, 1]")
    for mi, mesh in enumerate(doc.get("meshes", [])):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            problems += _check_primitive(f"meshes[{mi}].primitives[{pi}]", prim, accessors, views, decoded,
                                         len(materials))
    nodes = doc.get("nodes", [])
    for i, node in enumerate(nodes):
        if "mesh" in node and not 0 <= node["mesh"] < len(doc.get("meshes", [])):
            bad(f"nodes[{i}].mesh {node['mesh']} is out of range")
        for c in node.get("children", []):
            if not 0 <= c < len(nodes):
                bad(f"nodes[{i}].children has out-of-range node {c}")
    scenes = doc.get("scenes", [])
    if "scene" in doc and not 0 <= doc["scene"] < len(scenes):
        bad(f"scene {doc['scene']} is out of range")
    for i, s in enumerate(scenes):
        for n in s.get("nodes", []):
            if not 0 <= n < len(nodes):
                bad(f"scenes[{i}].nodes has out-of-range node {n}")
    return problems


def _check_primitive(where: str, prim: dict, accessors: list, views: list, decoded: dict, n_materials: int) -> list[str]:
    problems: list[str] = []
    bad = problems.append
    attrs = prim.get("attributes", {})
    if "POSITION" not in attrs:
        return [f"{where}: no POSITION attribute"]
    refs = list(attrs.values()) + ([prim["indices"]] if "indices" in prim else [])
    if any(not isinstance(v, int) or v not in decoded for v in refs):
        return [f"{where}: refers to a missing or invalid accessor"]
    pos_acc = accessors[attrs["POSITION"]]
    n_verts = pos_acc["count"]
    if "min" not in pos_acc or "max" not in pos_acc:
        bad(f"{where}: the POSITION accessor needs min and max")
    for name, ai in attrs.items():
        acc = accessors[ai]
        if acc["count"] != n_verts:
            bad(f"{where}: {name} has {acc['count']} elements, POSITION has {n_verts}")
        if views[acc["bufferView"]].get("target", ARRAY_BUFFER) != ARRAY_BUFFER:
            bad(f"{where}: the {name} bufferView target must be ARRAY_BUFFER")
        if name in ("POSITION", "NORMAL") and (acc["type"], acc["componentType"]) != ("VEC3", FLOAT):
            bad(f"{where}: {name} must be VEC3 FLOAT")
    if "NORMAL" in attrs:
        err = np.abs(np.linalg.norm(decoded[attrs["NORMAL"]].astype(np.float64), axis=1) - 1.0).max()
        if err > 1e-3:
            bad(f"{where}: NORMAL is not unit length (max error {err:.4f})")
    if "COLOR_0" in attrs:
        acc, col = accessors[attrs["COLOR_0"]], decoded[attrs["COLOR_0"]]
        if acc["type"] not in ("VEC3", "VEC4"):
            bad(f"{where}: COLOR_0 must be VEC3 or VEC4")
        elif acc["componentType"] == FLOAT and (col.min() < 0.0 or col.max() > 1.0):
            bad(f"{where}: COLOR_0 float values must be in [0, 1]")
    mode = prim.get("mode", TRIANGLES)
    if mode not in range(7):
        bad(f"{where}: bad mode {mode}")
    if "material" in prim and not 0 <= prim["material"] < n_materials:
        bad(f"{where}: material {prim['material']} is out of range")
    if "indices" in prim:
        ia, idx = accessors[prim["indices"]], decoded[prim["indices"]]
        if ia["type"] != "SCALAR" or ia["componentType"] not in (5121, 5123, 5125):
            bad(f"{where}: indices must be SCALAR unsigned byte/short/int")
        if views[ia["bufferView"]].get("target", ELEMENT_ARRAY_BUFFER) != ELEMENT_ARRAY_BUFFER:
            bad(f"{where}: the indices bufferView target must be ELEMENT_ARRAY_BUFFER")
        if mode == TRIANGLES and ia["count"] % 3:
            bad(f"{where}: {ia['count']} indices is not a multiple of 3")
        if int(idx.max()) >= n_verts:
            bad(f"{where}: index {int(idx.max())} >= vertex count {n_verts}")
    return problems
