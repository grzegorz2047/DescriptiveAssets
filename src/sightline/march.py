"""Sphere tracing of a Scene (SPEC 6.1).

For every blend group: slab-test all rays against the group AABB, sphere-trace only the rays that
enter it (from the entry distance, stopping at the exit), evaluating only that group's SDF. The
scene hit is the minimum over groups. Groups are visited nearest-first, and a ray is traced into
a group only if it enters it before its best hit so far, which skips most hidden work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .model import F32, Group, Scene

STEP = 0.9  # sphere-tracing step factor
REFINE_EPS = 0.05  # near-tie refinement: hit threshold as a share of the coarse one (decals 1-2 cm proud)
REFINE_STEPS = 24


@dataclass(eq=False)
class ViewBuffers:
    camera: Camera
    hit: np.ndarray  # (H, W) bool
    depth: np.ndarray  # (H, W) float32, distance along the ray from its origin; inf if no hit
    part: np.ndarray  # (H, W) int32, index into scene.parts; -1 = no hit
    point: np.ndarray  # (H, W, 3) float32 world hit point
    normal: np.ndarray  # (H, W, 3) float32 unit world normal
    local: np.ndarray  # (H, W, 3) float32 hit point in the hit part's local frame


def _slab(o3: np.ndarray, inv3: np.ndarray, lo: np.ndarray, hi: np.ndarray, pad: float):
    """Entry/exit distances of rays (3, N) through an AABB grown by `pad`."""
    lo32 = (lo - pad).astype(F32)[:, None]
    hi32 = (hi + pad).astype(F32)[:, None]
    t1 = (lo32 - o3) * inv3
    t2 = (hi32 - o3) * inv3
    t_near = np.minimum(t1, t2).max(axis=0)
    t_far = np.maximum(t1, t2).min(axis=0)
    return t_near, t_far


def _trace_groups(scene: Scene, o3: np.ndarray, d3: np.ndarray, t_max, eps: float, max_steps: int,
                  groups: list[Group], cull: float) -> tuple[np.ndarray, np.ndarray]:
    """First hit distance (inf = none) and hit group number (index into `groups`, -1 = none).

    o3: (3, N) origins; d3: (3, N) directions or (3, 1) when all rays share one direction.
    """
    n = o3.shape[1]
    best_t = np.full(n, np.inf, dtype=F32)
    best_g = np.full(n, -1, dtype=np.int32)
    if n == 0 or not groups:
        return best_t, best_g
    shared_dir = d3.shape[1] == 1 and n > 1
    safe = np.where(np.abs(d3) < 1e-12, F32(1e-12), d3)
    inv3 = (1.0 / safe).astype(F32)
    limit_all = np.broadcast_to(np.asarray(t_max, dtype=F32), (n,))
    pad = 2.0 * eps

    # Slab tests first, then visit groups nearest-first (better pruning by best_t).
    slabs = []
    for gi, g in enumerate(groups):
        t_near, t_far = _slab(o3, inv3, g.aabb_min, g.aabb_max, pad)
        t_far = np.minimum(t_far, limit_all)
        enter = t_far >= np.maximum(t_near, 0.0)
        if enter.any():
            slabs.append((float(np.median(t_near[enter])), gi, t_near, t_far, enter))
    slabs.sort(key=lambda s: (s[0], s[1]))

    for _, gi, t_near, t_far, enter in slabs:
        g = groups[gi]
        idx = np.flatnonzero(enter & (t_near < best_t))
        if idx.size == 0:
            continue
        t = np.maximum(t_near[idx], 0.0).astype(F32)
        stop = np.minimum(t_far[idx], best_t[idx])
        o = o3[:, idx]
        d = d3 if shared_dir else d3[:, idx]
        cutters = sorted({c for m in g.members for c in g.cutters.get(m, ())})
        field = (lambda q, g=g: scene.group_dist3(g, q, cull))  # noqa: E731
        for _ in range(max_steps):
            p = o + d * t
            dist = scene.group_dist3(g, p, cull)
            hit = dist < eps
            advance = STEP * dist
            if cutters and hit.any():
                # A cutter face coplanar with its target's face leaves a 0-distance sheet over the hole:
                # confirm hits near cutters, and step a phantom ray past the sheet.
                sel = np.flatnonzero(hit)
                fake = scene.phantom3(field, cutters, np.ascontiguousarray(p[:, sel]), dist[sel], 0.5 * eps,
                                      d if shared_dir else d[:, sel])
                if fake.any():
                    hit[sel[fake]] = False
                    advance[sel[fake]] = np.maximum(advance[sel[fake]], F32(2.0 * eps))
            if hit.any():
                h = idx[hit]
                best_t[h] = t[hit]
                best_g[h] = gi
            t = t + advance
            alive = ~hit & (t <= stop)
            if not alive.all():
                if not alive.any():
                    break
                idx, t, stop, o = idx[alive], t[alive], stop[alive], o[:, alive]
                if not shared_dir:
                    d = d[:, alive]
    _refine(scene, o3, d3, best_t, best_g, eps, groups, cull, shared_dir, slabs)
    return best_t, best_g


def _refine(scene: Scene, o3: np.ndarray, d3: np.ndarray, best_t: np.ndarray, best_g: np.ndarray, eps: float,
            groups: list[Group], cull: float, shared_dir: bool, slabs) -> None:
    """Settle near-ties in place: where two or more groups lie within 2 eps of a hit, re-trace each of
    them from just before the hit with a hit threshold of REFINE_EPS * eps and keep the first.

    The coarse hit threshold (half a pixel in march) makes whichever group's steps cross it first win
    when two surfaces are closer than that: a disc 1-2 cm proud of a wing rendered as a crescent and
    measured 0% visible. Rays near only one group keep their hit, so silhouettes do not change.
    """
    hit_idx = np.flatnonzero(best_g >= 0)
    if hit_idx.size == 0 or len(slabs) < 2:
        return
    d_hit = d3 if shared_dir else d3[:, hit_idx]
    p = o3[:, hit_idx] + d_hit * best_t[hit_idx]
    t_hit = best_t[hit_idx]
    near_count = np.zeros(hit_idx.size, dtype=np.int32)
    near: list[tuple[int, np.ndarray]] = []
    pad = F32(2.0 * eps)
    for _, gi, t_near, t_far, enter in sorted(slabs, key=lambda sl: sl[1]):
        # The slabs are the group AABBs grown by 2 eps: a hit point within 2 eps of the box lies on the
        # ray's stretch through the grown box.
        sel = np.flatnonzero(enter[hit_idx] & (t_near[hit_idx] <= t_hit + pad) & (t_far[hit_idx] >= t_hit - pad))
        if sel.size == 0:
            continue
        sel = sel[scene.group_dist3(groups[gi], np.ascontiguousarray(p[:, sel]), cull) < 2.0 * eps]
        if sel.size:
            near.append((gi, sel))
            near_count[sel] += 1
    tie = near_count >= 2
    if not tie.any():
        return
    fine = REFINE_EPS * eps
    new_t = np.full(hit_idx.size, np.inf, dtype=F32)
    new_g = np.full(hit_idx.size, -1, dtype=np.int32)
    for gi, sel in near:
        sel = sel[tie[sel]]
        if sel.size == 0:
            continue
        rays = hit_idx[sel]
        t = (best_t[rays] - F32(2.0 * eps)).astype(F32)
        stop = t + F32(6.0 * eps)
        o = o3[:, rays]
        d = d3 if shared_dir else d3[:, rays]
        found = np.full(sel.size, np.inf, dtype=F32)
        live = np.arange(sel.size)
        for _ in range(REFINE_STEPS):
            dist = scene.group_dist3(groups[gi], o[:, live] + (d if shared_dir else d[:, live]) * t[live], cull)
            done = dist < fine
            found[live[done]] = t[live[done]]
            t[live] += F32(STEP) * dist
            live = live[~done & (t[live] <= stop[live])]
            if live.size == 0:
                break
        # Coplanar faces tie within the fine threshold: the earlier group keeps them, so overlapping
        # panels show one clean boundary instead of a scribble of alternating pixels.
        better = found < new_t[sel] - F32(2.0 * fine)
        new_t[sel[better]] = found[better]
        new_g[sel[better]] = gi
    ok = np.flatnonzero(tie & (new_g >= 0))
    best_t[hit_idx[ok]] = new_t[ok]
    best_g[hit_idx[ok]] = new_g[ok]


def _as3(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=F32).reshape(-1, 3).T)


def trace(scene: Scene, origins, dirs, t_max, eps: float, max_steps: int = 160, groups=None):
    """First hit along arbitrary rays: (t (N,) float32, inf = no hit; part (N,) int32, -1 = none).

    `dirs` should be unit vectors; `t_max` is a scalar or (N,) array; `groups` optionally limits
    the trace to some groups (Group objects or indices into scene.groups).
    """
    o3 = _as3(origins)
    dirs = np.asarray(dirs, dtype=F32).reshape(-1, 3)
    d3 = dirs[:1].T.copy() if len(dirs) > 1 and np.all(dirs == dirs[0]) else _as3(dirs)
    glist = scene.groups if groups is None else [scene.groups[g] if isinstance(g, (int, np.integer)) else g for g in groups]
    t, gidx = _trace_groups(scene, o3, d3, t_max, eps, max_steps, glist, cull=8.0 * eps)
    part = np.full(t.shape, -1, dtype=np.int32)
    for gi in np.unique(gidx[gidx >= 0]):
        sel = np.flatnonzero(gidx == gi)
        g = glist[gi]
        if len(g.members) == 1:
            part[sel] = g.members[0]
        else:
            d = d3 if d3.shape[1] == 1 else d3[:, sel]
            _, part[sel] = scene.group_dist3(g, o3[:, sel] + d * t[sel], 8.0 * eps, need_part=True)
    return t, part


def march(scene: Scene, camera: Camera, max_steps: int = 160) -> ViewBuffers:
    """Render the scene's geometry buffers for one camera."""
    h_px, w_px = camera.height, camera.width
    origins, dirs = camera.rays()
    o3 = np.ascontiguousarray(origins.T)
    d3 = np.ascontiguousarray(dirs[:1].T)  # orthographic: one shared direction
    eps = 0.5 * camera.mpp
    cull = 8.0 * eps
    groups = scene.groups
    t, gidx = _trace_groups(scene, o3, d3, np.inf, eps, max_steps, groups, cull)

    n = h_px * w_px
    hit = gidx >= 0
    point = np.zeros((n, 3), dtype=F32)
    normal = np.zeros((n, 3), dtype=F32)
    local = np.zeros((n, 3), dtype=F32)
    part = np.full(n, -1, dtype=np.int32)
    hidx = np.flatnonzero(hit)
    p3 = o3[:, hidx] + d3 * t[hidx]
    point[hidx] = p3.T

    h = F32(0.5 * camera.mpp)
    offsets = np.zeros((6, 3, 1), dtype=F32)
    for k in range(3):
        offsets[2 * k, k, 0] = h
        offsets[2 * k + 1, k, 0] = -h
    g_hit = gidx[hidx]
    for gi in np.unique(g_hit):
        sel = np.flatnonzero(g_hit == gi)
        g = groups[gi]
        p = p3[:, sel]
        m = p.shape[1]
        # Central differences: one batched evaluation of the 6 offset point sets.
        probe = (p[None, :, :] + offsets).transpose(1, 0, 2).reshape(3, 6 * m)
        dist = scene.group_dist3(g, np.ascontiguousarray(probe), cull).reshape(6, m)
        grad = np.stack([dist[0] - dist[1], dist[2] - dist[3], dist[4] - dist[5]], axis=1)
        length = np.sqrt((grad * grad).sum(axis=1))
        bad = length < 1e-12
        grad[bad] = -d3[:, 0]
        length[bad] = 1.0
        normal[hidx[sel]] = grad / length[:, None]
        if len(g.members) == 1:
            part[hidx[sel]] = g.members[0]
        else:
            _, part[hidx[sel]] = scene.group_dist3(g, np.ascontiguousarray(p), cull, need_part=True)
    for pi in np.unique(part[hidx]):
        rows = hidx[part[hidx] == pi]
        local[rows] = scene.parts[pi].to_local(point[rows])

    depth = np.where(hit, t, np.inf).astype(F32)
    return ViewBuffers(
        camera=camera,
        hit=hit.reshape(h_px, w_px),
        depth=depth.reshape(h_px, w_px),
        part=part.reshape(h_px, w_px),
        point=point.reshape(h_px, w_px, 3),
        normal=normal.reshape(h_px, w_px, 3),
        local=local.reshape(h_px, w_px, 3),
    )
