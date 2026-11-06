"""Mesh decimation (SPEC section 9): parallel quadric-error edge collapse of closed part meshes.

Every vertex carries the quadric of its original triangles' planes in a 6-D space of position and
normal (Garland & Heckbert 1998, with the normal as an attribute scaled by NORMAL_WEIGHT cells). A
collapse is charged for moving the surface and for smearing a sharp edge's shading, so flat areas
collapse for free, edges and corners stay where they are, and the strip of triangles along a sharp
edge keeps its width. The new vertex goes to the quadric's optimum (regularized toward the edge
midpoint where the quadric is flat), clamped to the part's bounding box; if that fails the checks
below, onto one of the edge's ends.

Each round ranks the candidate collapses by error (the RMS quadric error may reach ERR_SLACK times
the tolerance; the SDF checks below hold the real distance to 1x), keeps a set of them whose
neighborhoods do not touch (every one a local minimum two rings around, three passes), checks those
and applies all of them at once with numpy: no Python loop over edges or vertices. A collapse is
legal when it

- keeps a closed 2-manifold: both ends share exactly the two opposite vertices (link condition) and
  every vertex keeps at least three neighbors; vertices on non-manifold edges or with several fans
  never move, so a mesh that is not manifold before is no worse after;
- keeps the part's bounding box: the vertex stays inside it, and a vertex within BBOX_TOL of a side
  stays within BBOX_TOL of it, so the box shrinks by at most BBOX_TOL and never grows;
- keeps material seams (seam vertices only collapse along the seam);
- flips no triangle and keeps every triangle's winding agreeing with its vertex normals (the SDF
  gradient at the new vertex);
- keeps the new vertex, and the center and a point near each corner of every changed triangle,
  within the tolerance of the SDF surface;
- smears no shading: where the normal interpolated near a corner of a changed triangle is off the
  SDF normal there by over SHADE_ANGLE, the triangle is at most SHADE_WIDTH cells wide.

Collapses inside a flat face (zero quadric error between equal normals) skip the SDF probes. Rejected
collapses wait until a vertex of their neighborhood moves. Without a budget the rounds end when no
legal collapse is left or the last STALL_ROUNDS rounds removed under STALL_FRACTION of the triangles;
with a triangle budget they end at the budget, relaxing the tolerance (doubling, up to RELAX_STEPS
times) when no legal collapse is left first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

SdfFn = Callable[[np.ndarray], np.ndarray]  # (3, N) float32 world points -> (N,) float32

NORMAL_WEIGHT = 1.0  # cells of position error that a unit change of the normal (~57 deg) costs
SHADE_ANGLE = 20.0  # deg: interpolated vs SDF normal near a triangle corner that counts as a smear
SHADE_WIDTH = 1.5  # cells: the widest a smeared triangle may become (sharp edges stay crisp)
FLIP_COS = 0.2  # a triangle may not turn by more than ~78 deg in one collapse
WINDING_MIN = 0.3  # unit face normal . (sum of its 3 vertex normals) must exceed this
BBOX_TOL = 0.005  # m: the most a part's bounding box may shrink on any side
RELAX_STEPS = 6  # a triangle budget may double the tolerance this many times
MAX_ROUNDS = 500
SELECT_PASSES = 3  # independent-set passes per round
ERR_SLACK = 2.0  # the quadric error may reach this times the tolerance (the SDF checks hold it to 1x)
EXACT = 1e-3  # quadric error under this times the tolerance, between equal normals: inside a flat face
STALL_ROUNDS = 10  # without a budget, stop once this many rounds together removed
STALL_FRACTION = 0.01  # under this fraction of the triangles
REG = 1e-3  # pull of the optimum toward the edge midpoint, relative to the quadric's area
CHUNK_EDGES = 1 << 17  # edges per quadric solve

_IU = [(i, j) for i in range(6) for j in range(i, 6)]  # upper triangle of a symmetric 6 x 6
_SYM = np.zeros((6, 6), dtype=np.int64)
for _k, (_i, _j) in enumerate(_IU):
    _SYM[_i, _j] = _SYM[_j, _i] = _k
_EYE21 = np.array([1.0 if i == j else 0.0 for i, j in _IU])
# Barycentric weights of a point near each corner: there the interpolated normal is mostly that
# corner's, so a sharp-edge normal carried into a flat face shows up, while on a smooth curve the
# interpolated normal still matches the SDF.
_NEAR = np.array([[0.75, 0.125, 0.125], [0.125, 0.75, 0.125], [0.125, 0.125, 0.75]])
NQ = 28  # 21 matrix entries, 6 linear terms, 1 constant


@dataclass(eq=False)
class DecimationInput:
    """One closed part mesh and the SDF it was meshed from."""

    positions: np.ndarray  # (V, 3) float
    triangles: np.ndarray  # (T, 3) int, counter-clockwise seen from outside
    normals: np.ndarray  # (V, 3) float, unit, outward (the SDF gradient)
    labels: np.ndarray  # (T,) int material ids: seams between labels are kept
    fn: SdfFn
    cell: float  # lattice cell the mesh was made with (m)


@dataclass(eq=False)
class DecimationResult:
    positions: np.ndarray  # (V', 3) float64
    triangles: np.ndarray  # (T', 3) int64
    normals: np.ndarray  # (V', 3) float64 unit
    vertex_ids: np.ndarray  # (V',) index of each kept vertex in the input
    triangle_ids: np.ndarray  # (T',) index of each kept triangle in the input
    tolerance: float  # m, the tolerance the last collapse was allowed (relaxed for a budget)
    max_error: float  # m, largest |SDF| at the vertices and triangle centers
    source_error: float  # m, the same on the input mesh


# ----------------------------------------------------------------------------- quadrics


def _face_quadrics(x6: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """(T, NQ) area-weighted squared distance to each triangle's plane in 6-D (position, normal)."""
    p, q, r = (x6[tris[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(q[:, :3] - p[:, :3], r[:, :3] - p[:, :3]), axis=1)
    e1 = q - p
    l1 = np.linalg.norm(e1, axis=1)
    e1 /= np.maximum(l1, 1e-30)[:, None]
    e2 = r - p
    e2 -= (e2 * e1).sum(axis=1)[:, None] * e1
    l2 = np.linalg.norm(e2, axis=1)
    e2 /= np.maximum(l2, 1e-30)[:, None]
    w = np.where((l1 > 1e-15) & (l2 > 1e-15), area, 0.0)
    ii, jj = np.array(_IU).T
    a = _EYE21 - e1[:, ii] * e1[:, jj] - e2[:, ii] * e2[:, jj]
    pe1, pe2 = (p * e1).sum(axis=1), (p * e2).sum(axis=1)
    b = pe1[:, None] * e1 + pe2[:, None] * e2 - p
    c = (p * p).sum(axis=1) - pe1**2 - pe2**2
    return np.concatenate([a, b, c[:, None]], axis=1) * w[:, None], w


def _vertex_quadrics(x6: np.ndarray, tris: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """(V, NQ) sum of the face quadrics around each vertex and (V,) their area, in face chunks."""
    quad = np.zeros((n_verts, NQ))
    weight = np.zeros(n_verts)
    for i0 in range(0, len(tris), CHUNK_EDGES):
        t = tris[i0 : i0 + CHUNK_EDGES]
        fq, area = _face_quadrics(x6, t)
        for k in range(3):
            weight += np.bincount(t[:, k], area, n_verts)
            for col in range(NQ):
                quad[:, col] += np.bincount(t[:, k], fq[:, col], n_verts)
    return quad, weight


def _solve_spd6(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve A x = y for symmetric positive definite 6 x 6 matrices: vectorized Cholesky.

    a: (21, n) upper-triangle entries in _IU order, y: (6, n). Returns x (6, n)."""
    lo = [[None] * 6 for _ in range(6)]
    for j in range(6):
        s = a[_SYM[j, j]].copy()
        for k in range(j):
            s -= lo[j][k] * lo[j][k]
        lo[j][j] = np.sqrt(np.maximum(s, 1e-300))
        inv = 1.0 / lo[j][j]
        for i in range(j + 1, 6):
            s = a[_SYM[i, j]].copy()
            for k in range(j):
                s -= lo[i][k] * lo[j][k]
            lo[i][j] = s * inv
    z = [None] * 6
    for i in range(6):
        s = y[i].copy()
        for k in range(i):
            s -= lo[i][k] * z[k]
        z[i] = s / lo[i][i]
    x = [None] * 6
    for i in range(5, -1, -1):
        s = z[i].copy()
        for k in range(i + 1, 6):
            s -= lo[k][i] * x[k]
        x[i] = s / lo[i][i]
    return np.stack(x)


_II, _JJ = np.array(_IU).T
_TWICE = np.where(_II == _JJ, 1.0, 2.0)


def _optimum_and_cost(q: np.ndarray, w: np.ndarray, mid: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """6-D minimizer of the quadric plus REG * area * |v - mid|^2 with its position clamped to
    [lo, hi], and the quadric's cost there. q (n, NQ), w (n,), mid (n, 6), lo/hi (n, 3)."""
    v = mid.copy()
    cost = np.empty(len(q))
    for i0 in range(0, len(q), CHUNK_EDGES):
        s = slice(i0, i0 + CHUNK_EDGES)
        qt = np.ascontiguousarray(q[s].T)  # (NQ, n)
        cost[s] = _cost_t(qt, np.ascontiguousarray(mid[s].T))
        # Where the midpoint is already exact (inside a flat face) it is the optimum: skip the solve.
        solve = np.flatnonzero(cost[s] > 1e-12 * np.maximum(w[s], 1e-30) * _scale2(mid[s]))
        if len(solve) == 0:
            continue
        qs = qt[:, solve]
        eps = REG * np.maximum(w[s][solve], 1e-30)
        a = qs[:21].copy()
        a[_SYM[np.arange(6), np.arange(6)]] += eps
        x = _solve_spd6(a, eps * mid[s][solve].T - qs[21:27])
        x[:3] = np.clip(x[:3], lo[s][solve].T, hi[s][solve].T)
        rows = i0 + solve
        cost[rows] = _cost_t(qs, x)
        v[rows] = x.T
    return v, np.maximum(cost, 0.0)


def _scale2(x6: np.ndarray) -> np.ndarray:
    """max(1, |position|^2) in m^2: the quadric's rounding grows with the coordinates, so 'exact'
    (an RMS error under 1 um) is measured relative to them beyond 1 m."""
    return np.maximum((x6[:, :3] ** 2).sum(axis=1), 1.0)


def _cost_t(qt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Quadric value x' A x + 2 b' x + c for transposed quadrics qt (NQ, n) at points x (6, n)."""
    return (qt[:21] * _TWICE[:, None] * x[_II] * x[_JJ]).sum(axis=0) + 2.0 * (qt[21:27] * x).sum(axis=0) + qt[27]


# ----------------------------------------------------------------------------- topology


def _ragged(ptr: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions ptr[k] .. ptr[k + 1] - 1 for every k in `keys`, and the index of their key."""
    start = ptr[keys]
    count = ptr[keys + 1] - start
    owner = np.repeat(np.arange(len(keys)), count)
    offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    return start[owner] + offset, owner


def _csr(index: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(order, ptr): order lists the positions of `index` grouped by value, ptr the group starts."""
    order = np.argsort(index)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(index, minlength=n), out=ptr[1:])
    return order, ptr


@dataclass(eq=False)
class _Edges:
    key: np.ndarray  # sorted lo * V + hi
    a: np.ndarray  # lower vertex id
    b: np.ndarray  # higher vertex id
    c: np.ndarray  # vertex opposite the edge in its first triangle
    d: np.ndarray  # ... in its second triangle
    f0: np.ndarray  # first triangle
    f1: np.ndarray  # second triangle
    count: np.ndarray  # triangles using the edge (2 on a manifold)


def _edges(tris: np.ndarray, n_verts: int) -> _Edges:
    tail = tris.reshape(-1)
    head = tris[:, [1, 2, 0]].reshape(-1)
    opp = tris[:, [2, 0, 1]].reshape(-1)
    lo, hi = np.minimum(tail, head), np.maximum(tail, head)
    key = lo * n_verts + hi
    order = np.argsort(key)
    ks = key[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    count = np.diff(np.r_[start, len(ks)])
    o0, o1 = order[start], order[np.minimum(start + 1, len(ks) - 1)]
    return _Edges(key=ks[start], a=lo[o0], b=hi[o0], c=opp[o0], d=opp[o1], f0=o0 // 3, f1=o1 // 3, count=count)


def nonmanifold_vertices(tris: np.ndarray, n_verts: int) -> np.ndarray:
    """(V,) bool: vertices on an edge not shared by exactly two consistently wound triangles, or whose
    triangles do not form one closed fan."""
    bad = np.zeros(n_verts, dtype=bool)
    if len(tris) == 0:
        return bad
    e = _edges(tris, n_verts)
    wrong = e.count != 2
    tail = tris.reshape(-1)
    head = tris[:, [1, 2, 0]].reshape(-1)
    dkey = np.sort(tail * n_verts + head)
    twice = dkey[1:][dkey[1:] == dkey[:-1]]  # a directed edge used twice: inconsistent winding
    bad[e.a[wrong]] = bad[e.b[wrong]] = True
    bad[twice // n_verts] = bad[twice % n_verts] = True
    # Fans: around vertex v each triangle gives a link edge u -> w; on a manifold they form one cycle.
    v = tail
    u = tris[:, [1, 2, 0]].reshape(-1)
    w = tris[:, [2, 0, 1]].reshape(-1)
    src, dst = v * n_verts + u, v * n_verts + w
    order = np.argsort(src)
    ss = src[order]
    dup = np.r_[ss[1:] == ss[:-1], False] | np.r_[False, ss[1:] == ss[:-1]]
    bad[v[order[dup]]] = True
    pos = np.minimum(np.searchsorted(ss, dst[order]), len(ss) - 1)
    missing = ss[pos] != dst[order]
    bad[v[order[missing]]] = True
    succ = np.where(missing, np.arange(len(ss)), pos)
    rep = np.arange(len(ss))
    longest = int(np.bincount(v).max())
    for _ in range(max(1, longest).bit_length() + 1):  # pointer jumping: min index around each cycle
        rep = np.minimum(rep, rep[succ])
        succ = succ[succ]
    cycles = np.bincount(v[order], rep == np.arange(len(ss)), n_verts)
    bad[cycles > 1] = True
    return bad


def _link_ok(a: np.ndarray, b: np.ndarray, e: _Edges, n_verts: int) -> np.ndarray:
    """(n,) bool: edge a-b has exactly two common neighbors of its ends (the link condition)."""
    src = np.concatenate([e.a, e.b])
    dst = np.concatenate([e.b, e.a])
    order, ptr = _csr(src, n_verts)
    adj = dst[order]
    rows, owner = _ragged(ptr, a)
    x, bb = adj[rows], b[owner]
    q = np.minimum(bb, x) * n_verts + np.maximum(bb, x)
    pos = np.minimum(np.searchsorted(e.key, q), len(e.key) - 1)
    common = np.bincount(owner, e.key[pos] == q, len(a))
    return common == 2


def _rank(err: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Order of collapses: by error in half-octave buckets, ties in a fixed pseudo-random order.

    Plain sorting would rank equal-error edges (all of a flat face) by vertex index, a smooth
    gradient across the face with few local minima, so few collapses would be independent."""
    bucket = np.ceil(2.0 * np.log2(np.clip(err, 2.0**-12, 2.0**8))).astype(np.int64) + 32  # 8 .. 48
    mix = (key.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(24)  # 40 bits
    return np.argsort((bucket << 40) | mix.astype(np.int64))


def _independent(a: np.ndarray, b: np.ndarray, tris: np.ndarray, n_verts: int) -> np.ndarray:
    """(n,) bool: edges (in rank order) whose neighborhoods (the triangles touching either end) are
    pairwise disjoint. Pass 1 keeps every edge ranked first among all edges within two rings; later
    passes add edges away from the ones already kept."""
    n = len(a)
    chosen = np.zeros(n, dtype=bool)
    avail = np.ones(n, dtype=bool)
    flat = tris.reshape(-1)
    for _ in range(SELECT_PASSES):
        ia = np.flatnonzero(avail)
        if len(ia) == 0:
            break
        vmin = np.full(n_verts, n, dtype=np.int64)
        np.minimum.at(vmin, a[ia], ia)
        np.minimum.at(vmin, b[ia], ia)
        fmin = vmin[tris].min(axis=1)
        vmin2 = np.full(n_verts, n, dtype=np.int64)
        np.minimum.at(vmin2, flat, np.repeat(fmin, 3))
        win = ia[(vmin2[a[ia]] == ia) & (vmin2[b[ia]] == ia)]
        chosen[win] = True
        ends = np.zeros(n_verts, dtype=bool)
        ends[a[win]] = ends[b[win]] = True
        blocked = np.zeros(n_verts, dtype=bool)
        blocked[tris[ends[tris].any(axis=1)].reshape(-1)] = True
        avail &= ~blocked[a] & ~blocked[b] & ~chosen
    return chosen


# ----------------------------------------------------------------------------- SDF probes


def _probe(fns: list[SdfFn], part: np.ndarray, pts: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SDF value (N,) and gradient (N, 3) at points of several parts (part (N,) indexes fns and h)."""
    from .export import sdf_gradient  # lazy: export imports this module

    val = np.empty(len(pts))
    grad = np.empty((len(pts), 3))
    for pi in np.unique(part):
        sel = np.flatnonzero(part == pi)
        val[sel], grad[sel] = sdf_gradient(fns[pi], pts[sel], float(h[pi]))
    return val, grad


def _values(fns: list[SdfFn], part: np.ndarray, pts: np.ndarray) -> np.ndarray:
    from .export import _eval_points

    val = np.empty(len(pts))
    for pi in np.unique(part):
        sel = np.flatnonzero(part == pi)
        val[sel] = _eval_points(fns[pi], pts[sel])
    return val


# ----------------------------------------------------------------------------- decimation


def decimate(parts: list[DecimationInput], tolerance: float, max_tris: int | None = None) -> list[DecimationResult]:
    """Decimate every part mesh on its own (collapses never join parts) in one shared priority order.

    tolerance: largest distance from the SDF surface, in cells of each part. max_tris: a budget for all
    parts together; when the pass at `tolerance` leaves more triangles, further passes double the
    tolerance until the budget is met (or RELAX_STEPS doublings did not reach it).
    """
    n_parts = len(parts)
    vcount = [len(p.positions) for p in parts]
    tcount = [len(p.triangles) for p in parts]
    voff = np.cumsum([0] + vcount)
    toff = np.cumsum([0] + tcount)
    nv = int(voff[-1])
    if nv == 0:
        return [DecimationResult(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3)),
                                 np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0.0, 0.0, 0.0)
                for _ in parts]
    pos = np.concatenate([np.asarray(p.positions, dtype=np.float64).reshape(-1, 3) for p in parts])
    nrm = np.concatenate([np.asarray(p.normals, dtype=np.float64).reshape(-1, 3) for p in parts])
    tris = np.concatenate([np.asarray(p.triangles, dtype=np.int64).reshape(-1, 3) + voff[i]
                           for i, p in enumerate(parts)])
    tri_id = np.arange(len(tris))
    label = np.concatenate([np.asarray(p.labels, dtype=np.int64).reshape(-1) for p in parts])
    vpart = np.repeat(np.arange(n_parts), vcount)
    vid = np.arange(nv)  # input vertex of each working vertex (the arrays are compacted as they shrink)
    fns = [p.fn for p in parts]
    cell = np.array([float(p.cell) for p in parts])
    tol0 = tolerance * cell
    h = 0.25 * cell  # gradient step, as export.vertex_normals
    lam = NORMAL_WEIGHT * cell
    lo = np.array([p.positions.min(axis=0) if c else np.zeros(3) for p, c in zip(parts, vcount)], dtype=np.float64)
    hi = np.array([p.positions.max(axis=0) if c else np.zeros(3) for p, c in zip(parts, vcount)], dtype=np.float64)
    btol = np.minimum(BBOX_TOL, tol0)

    x6 = np.concatenate([pos, lam[vpart, None] * nrm], axis=1)
    quad, weight = _vertex_quadrics(x6, tris, nv)
    locked = nonmanifold_vertices(tris, nv)
    source_error = _surface_error(fns, vpart, pos, tris, n_parts)

    stamp = np.zeros(nv, dtype=np.int64)  # round in which a vertex last moved
    rej_key, rej_round = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)  # rejected edges, sorted
    c_key = np.zeros(0, dtype=np.int64)  # edge cache, sorted by key
    c_err, c_p = np.zeros(0), np.zeros((0, 3))
    c_fit, c_round = np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    relax, used_relax = 1.0, 1.0
    shade_cos = math.cos(math.radians(SHADE_ANGLE))
    history = [len(tris)]

    def box_fits(a, b, p):
        """(n,) bool: moving a and b to p keeps the part's box (a side near a or b stays near p)."""
        pi = vpart[a]
        bt = btol[pi][:, None]
        near_hi = (pos[a] >= hi[pi] - bt) | (pos[b] >= hi[pi] - bt)
        near_lo = (pos[a] <= lo[pi] + bt) | (pos[b] <= lo[pi] + bt)
        return ~(near_hi & (p < hi[pi] - bt)).any(axis=1) & ~(near_lo & (p > lo[pi] + bt)).any(axis=1)

    def evaluate(a, b):
        """(RMS 6-D error, clamped optimal position, bounding-box test) of collapsing edges a-b."""
        err, p = np.empty(len(a)), np.empty((len(a), 3))
        for i0 in range(0, len(a), CHUNK_EDGES):  # chunks bound the (n, NQ) temporaries
            ca, cb = a[i0 : i0 + CHUNK_EDGES], b[i0 : i0 + CHUNK_EDGES]
            pi = vpart[ca]
            wq = weight[ca] + weight[cb]
            v, cost = _optimum_and_cost(quad[ca] + quad[cb], wq, 0.5 * (x6[ca] + x6[cb]), lo[pi], hi[pi])
            err[i0 : i0 + CHUNK_EDGES] = np.sqrt(cost / np.maximum(wq, 1e-30))
            p[i0 : i0 + CHUNK_EDGES] = v[:, :3]
        return err, p, box_fits(a, b, p)

    def endpoint_error(a, b, at):
        """RMS 6-D error of collapsing a-b onto the vertex `at` (a or b)."""
        q = np.ascontiguousarray((quad[a] + quad[b]).T)
        return np.sqrt(np.maximum(_cost_t(q, x6[at].T), 0.0) / np.maximum(weight[a] + weight[b], 1e-30))

    def next_pass() -> bool:
        """End of a pass (nothing legal left, or stalled): relax the tolerance if over the budget."""
        nonlocal relax, rej_key, rej_round, history
        if max_tris is not None and len(tris) > max_tris and relax < 2.0**RELAX_STEPS:
            relax *= 2.0
            rej_key, rej_round = rej_key[:0], rej_round[:0]
            history = [len(tris)]
            return True
        return False

    for rnd in range(1, MAX_ROUNDS + 1):
        if relax > 1.0 and len(tris) <= max_tris:
            break  # a relaxed pass reached the budget
        if len(history) > STALL_ROUNDS and history[-1 - STALL_ROUNDS] - history[-1] < STALL_FRACTION * history[-1]:
            if next_pass():  # the last rounds hardly removed anything
                continue
            break
        if len(tris) < nv and nv > 1024:  # under half the vertices are left: compact the arrays
            keep_v = np.unique(tris)
            new_id = np.full(nv, -1, dtype=np.int64)
            new_id[keep_v] = np.arange(len(keep_v))
            tris = new_id[tris]
            pos, nrm, x6, quad, weight = pos[keep_v], nrm[keep_v], x6[keep_v], quad[keep_v], weight[keep_v]
            vpart, vid, stamp, locked = vpart[keep_v], vid[keep_v], stamp[keep_v], locked[keep_v]
            ck, cs = _rekey(c_key, new_id, nv, len(keep_v))
            c_key, c_err, c_p, c_fit, c_round = ck, c_err[cs], c_p[cs], c_fit[cs], c_round[cs]
            rk, rs = _rekey(rej_key, new_id, nv, len(keep_v))
            rej_key, rej_round = rk, rej_round[rs]
            nv = len(keep_v)
        tol = relax * tol0
        e = _edges(tris, nv)
        deg = np.bincount(e.a, minlength=nv) + np.bincount(e.b, minlength=nv)
        stuck = locked.copy()
        wrong = e.count != 2
        stuck[e.a[wrong]] = stuck[e.b[wrong]] = True
        ok = ~wrong & ~stuck[e.a] & ~stuck[e.b] & ~stuck[e.c] & ~stuck[e.d]
        ok &= (deg[e.a] + deg[e.b] >= 7) & (deg[e.c] >= 4) & (deg[e.d] >= 4)
        flat = tris.reshape(-1)
        corner_label = np.repeat(label, 3)
        lmin = np.full(nv, np.iinfo(np.int64).max)
        lmax = np.full(nv, -1)
        np.minimum.at(lmin, flat, corner_label)
        np.maximum.at(lmax, flat, corner_label)
        seam = lmin != lmax
        ok &= ~(seam[e.a] | seam[e.b]) | (label[e.f0] != label[e.f1])  # seam vertices move along the seam
        if len(rej_key):  # a rejected edge waits until a vertex of its neighborhood moves
            vst = np.zeros(nv, dtype=np.int64)
            np.maximum.at(vst, flat, np.repeat(stamp[tris].max(axis=1), 3))
            at = np.minimum(np.searchsorted(rej_key, e.key), len(rej_key) - 1)
            held = (rej_key[at] == e.key) & (np.maximum(vst[e.a], vst[e.b]) <= rej_round[at])
            ok &= ~held
            rej_key, rej_round = e.key[held], rej_round[at[held]]

        idx = np.flatnonzero(ok)
        a, b = e.a[idx], e.b[idx]
        pi = vpart[a]
        ekey = e.key[idx]
        # Optimum, error and box test per edge, cached until an end moves.
        err = np.empty(len(idx))
        p = np.empty((len(idx), 3))
        fits = np.empty(len(idx), dtype=bool)
        hit = np.zeros(len(idx), dtype=bool)
        made = np.full(len(idx), rnd, dtype=np.int64)
        if len(c_key):
            at = np.minimum(np.searchsorted(c_key, ekey), len(c_key) - 1)
            hit = (c_key[at] == ekey) & (np.maximum(stamp[a], stamp[b]) < c_round[at])
            err[hit], p[hit], fits[hit], made[hit] = c_err[at[hit]], c_p[at[hit]], c_fit[at[hit]], c_round[at[hit]]
        miss = np.flatnonzero(~hit)
        err[miss], p[miss], fits[miss] = evaluate(a[miss], b[miss])
        c_key, c_err, c_p, c_fit, c_round = ekey, err, p, fits, made
        sub = np.flatnonzero(fits & (err <= ERR_SLACK * tol[pi]))
        if len(sub) == 0:
            if next_pass():
                continue
            break
        sub = sub[_rank(err[sub] / tol[pi[sub]], ekey[sub])]
        sub = sub[_independent(a[sub], b[sub], tris, nv)]
        if relax > 1.0:  # a relaxed pass stops at the budget (each collapse removes 2 triangles)
            sub = sub[: max(1, (len(tris) - max_tris + 1) // 2)]
        sa, sb, sp, spart = a[sub], b[sub], p[sub], pi[sub]
        exact = (err[sub] <= EXACT * tol[spart]) & ((nrm[sa] * nrm[sb]).sum(axis=1) > 1.0 - 1e-6)

        order, ptr = _csr(tris.reshape(-1), nv)
        ctx = _Ctx(tris=tris, face_of=order // 3, ptr=ptr, pos=pos, nrm=nrm, fns=fns, cell=cell, h=h, tol=tol,
                   width=relax * SHADE_WIDTH * cell, shade_cos=shade_cos)
        linked = _link_ok(sa, sb, e, nv)
        good, n_new = _check(ctx, sa, sb, sp, spart, exact)
        good &= linked
        for end in (sa, sb):  # a collapse that fails at the optimum may still work onto one of its ends
            retry = np.flatnonzero(~good & linked)
            if len(retry) == 0:
                break
            ra, rb, rend = sa[retry], sb[retry], end[retry]
            alt = pos[rend]
            alt_err = endpoint_error(ra, rb, rend)
            fine = box_fits(ra, rb, alt) & (alt_err <= ERR_SLACK * tol[spart[retry]])
            retry, ra, rb, alt, alt_err = retry[fine], ra[fine], rb[fine], alt[fine], alt_err[fine]
            if len(retry) == 0:
                continue
            alt_exact = (alt_err <= EXACT * tol[spart[retry]]) & exact[retry]
            g2, n2 = _check(ctx, ra, rb, alt, spart[retry], alt_exact)
            win = retry[g2]
            good[win], sp[win], n_new[win] = True, alt[g2], n2[g2]
        if (~good).any():
            rk = ekey[sub[~good]]
            both_keys = np.concatenate([rej_key, rk])
            order = np.argsort(both_keys)
            rej_key = both_keys[order]
            rej_round = np.concatenate([rej_round, np.full(len(rk), rnd)])[order]
        sa, sb, sp, n_new = sa[good], sb[good], sp[good], n_new[good]
        if len(sa):
            used_relax = max(used_relax, relax)
            pos[sa] = sp
            nrm[sa] = n_new
            x6[sa, :3] = sp
            x6[sa, 3:] = lam[vpart[sa], None] * n_new
            quad[sa] += quad[sb]
            weight[sa] += weight[sb]
            stamp[sa] = rnd
            remap = np.arange(nv)
            remap[sb] = sa
            tris = remap[tris]
            live = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])
            tris, tri_id, label = tris[live], tri_id[live], label[live]
        history.append(len(tris))

    # Split back into parts, measure the distance to the SDF surface.
    max_error = _surface_error(fns, vpart, pos, tris, n_parts)
    tpart = vpart[tris[:, 0]]
    out = []
    for i in range(n_parts):
        sel = tpart == i
        t = tris[sel]
        keep_v = np.unique(t)
        out.append(DecimationResult(
            positions=pos[keep_v],
            triangles=np.searchsorted(keep_v, t),
            normals=nrm[keep_v],
            vertex_ids=vid[keep_v] - voff[i],
            triangle_ids=tri_id[sel] - toff[i],
            tolerance=float(used_relax * tol0[i]),
            max_error=float(max_error[i]),
            source_error=float(source_error[i]),
        ))
    return out


def _rekey(keys: np.ndarray, new_id: np.ndarray, n_old: int, n_new: int) -> tuple[np.ndarray, np.ndarray]:
    """Edge keys a * n_old + b renumbered by new_id (monotonic, so the order holds); (keys, kept)."""
    na, nb = new_id[keys // n_old], new_id[keys % n_old]
    kept = np.flatnonzero((na >= 0) & (nb >= 0))
    return na[kept] * n_new + nb[kept], kept


def _surface_error(fns, vpart, pos, tris, n_parts) -> np.ndarray:
    """(n_parts,) largest |SDF| at the used vertices and the triangle centers of each part."""
    used = np.unique(tris)
    part = np.concatenate([vpart[used], vpart[tris[:, 0]]])
    dist = np.abs(_values(fns, part, np.concatenate([pos[used], pos[tris].mean(axis=1)])))
    out = np.zeros(n_parts)
    np.maximum.at(out, part, dist)
    return out


@dataclass(eq=False)
class _Ctx:
    """What the checks of one round read."""

    tris: np.ndarray
    face_of: np.ndarray  # triangles around each vertex, grouped by ptr
    ptr: np.ndarray
    pos: np.ndarray
    nrm: np.ndarray
    fns: list
    cell: np.ndarray  # per part
    h: np.ndarray  # per part, gradient step
    tol: np.ndarray  # per part, m
    width: np.ndarray  # per part, m: widest smeared triangle
    shade_cos: float


def _check(ctx: _Ctx, sa, sb, sp, spart, exact) -> tuple[np.ndarray, np.ndarray]:
    """(accept (S,) bool, new vertex normals (S, 3)) for collapsing a and b to sp, neighborhoods disjoint.

    `exact` collapses (zero quadric error between equal normals: inside a flat face) skip the SDF probes."""
    s = len(sa)
    n_new = ctx.nrm[sa].copy()
    ok = np.ones(s, dtype=bool)
    probe = np.flatnonzero(~exact)
    if len(probe):
        val, grad = _probe(ctx.fns, spart[probe], sp[probe], ctx.h)
        glen = np.linalg.norm(grad, axis=1)
        n_new[probe] = grad / np.maximum(glen, 1e-30)[:, None]
        ok[probe] = (np.abs(val) <= ctx.tol[spart[probe]]) & (glen > 1e-9)

    ra, oa = _ragged(ctx.ptr, sa)
    rb, ob = _ragged(ctx.ptr, sb)
    faces = np.concatenate([ctx.face_of[ra], ctx.face_of[rb]])
    owner = np.concatenate([oa, ob])
    t = ctx.tris[faces]
    is_a, is_b = t == sa[owner, None], t == sb[owner, None]
    keep = ~(is_a.any(axis=1) & is_b.any(axis=1))  # the two triangles on the edge disappear
    owner, t, moved = owner[keep], t[keep], (is_a | is_b)[keep]
    part = spart[owner]
    old = ctx.pos[t]
    new = np.where(moved[..., None], sp[owner][:, None, :], old)
    nsum = np.where(moved[..., None], n_new[owner][:, None, :], ctx.nrm[t]).sum(axis=1)
    cn = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
    co = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
    ln, lo_ = np.linalg.norm(cn, axis=1), np.linalg.norm(co, axis=1)
    fail = ln < 1e-4 * ctx.cell[part] ** 2  # degenerate
    fail |= (cn * co).sum(axis=1) < FLIP_COS * ln * lo_  # flipped or turned too far
    fail |= (cn * nsum).sum(axis=1) < WINDING_MIN * ln  # winding against the vertex normals
    rows = np.flatnonzero(~exact[owner])
    if len(rows):
        nr = new[rows]  # (R, 3 corners, 3)
        nv_rows = np.where(moved[rows][..., None], n_new[owner[rows]][:, None, :], ctx.nrm[t[rows]])
        # Probe the center (distance) and a point near each corner (distance and shading).
        pts = np.concatenate([nr.mean(axis=1)[:, None], np.einsum("kc,rcx->rkx", _NEAR, nr)], axis=1)
        rp = np.repeat(part[rows], 4)
        val, grad = _probe(ctx.fns, rp, pts.reshape(-1, 3), ctx.h)
        far = np.abs(val).reshape(-1, 4).max(axis=1) > ctx.tol[part[rows]]
        fail[rows] |= far  # off the surface
        ni = np.einsum("kc,rcx->rkx", _NEAR, nv_rows)
        ni /= np.maximum(np.linalg.norm(ni, axis=2, keepdims=True), 1e-30)
        nt = grad.reshape(-1, 4, 3)[:, 1:]
        nt = nt / np.maximum(np.linalg.norm(nt, axis=2, keepdims=True), 1e-30)
        smear = ((ni * nt).sum(axis=2) < ctx.shade_cos).any(axis=1)
        longest = np.sqrt(np.max([((nr[:, k] - nr[:, (k + 1) % 3]) ** 2).sum(axis=1) for k in range(3)], axis=0))
        fail[rows] |= smear & (ln[rows] / np.maximum(longest, 1e-30) > ctx.width[part[rows]])
    ok &= np.bincount(owner, fail, s) == 0
    return ok, n_new
