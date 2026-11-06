"""Line of sight between two cells at eye height, sampled along the 3D segment.

The segment joins the eye points (terrain + player height - 0.1 m) of both cells and is sampled
every cell/4 meters horizontally. A sample is blocked when it is below the top of the cell it is
over (the heightfield, walls included: their top is their height) or inside a span of a solid
object's geometry over that cell (from the asset's voxel occupancy, so sight passes under overhangs).
"""

from __future__ import annotations

import math

import numpy as np

from .model import Level


def blocker_tops(level: Level) -> np.ndarray:
    """(rows, cols) meters: terrain top, raised to the top of solid objects that block walking there."""
    return np.maximum(level.heights_m(), level.object_top())


def _what(level: Level, col: int, row: int) -> str:
    kind = level.kind(col, row)
    return "wall" if kind == "wall" else f"terrain ({kind})"


def line_of_sight(level: Level, a: tuple[int, int], b: tuple[int, int]) -> dict:
    """Visibility from cell `a` to cell `b`: first blocking cell, or the tightest clearance when visible."""
    eye = level.player.eye
    pa = level.world(*a)
    pb = level.world(*b)
    pa[1] += eye
    pb[1] += eye
    delta = pb - pa
    horizontal = float(math.hypot(delta[0], delta[2]))
    n = max(2, int(math.ceil(horizontal / (level.cell / 4.0))) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = pa[None, :] + t[:, None] * delta[None, :]
    cols = np.clip(np.floor(pts[:, 0] / level.cell).astype(np.int64), 0, level.cols - 1)
    rows = np.clip(np.floor(pts[:, 2] / level.cell).astype(np.int64), 0, level.rows - 1)
    y = pts[:, 1]
    lo, hi, owner = level.object_spans()
    lo, hi, owner = lo[:, rows, cols], hi[:, rows, cols], owner[:, rows, cols]  # (K, n)
    inside = (lo <= y) & (y <= hi)
    first = inside.argmax(axis=0)
    hit_obj = np.where(inside.any(axis=0), owner[first, np.arange(n)], -1)
    span_top = hi[first, np.arange(n)]  # top of the object span the sample is in
    below = np.where(hi <= y, hi, -np.inf).max(axis=0)  # top of the object geometry under the sample
    terrain = level.heights_m()[rows, cols]
    tops = np.maximum(terrain, np.where(hit_obj >= 0, span_top, below))
    margin = y - tops
    blocked = (y < terrain) | (hit_obj >= 0)
    out = {
        "from_cell": list(a),
        "to_cell": list(b),
        "distance": round(float(np.linalg.norm(delta)), 4),
        "eye_height": round(eye, 4),
        "samples": n,
    }
    if blocked.any():
        i = int(np.argmax(blocked))
        c, r = int(cols[i]), int(rows[i])
        what = f"object {level.objects[hit_obj[i]].asset}" if hit_obj[i] >= 0 and y[i] >= terrain[i] else _what(level, c, r)
        out.update(visible=False, blocked_by={
            "cell": [c, r],
            "what": what,
            "top": round(float(tops[i]), 4),
            "ray_y": round(float(pts[i, 1]), 4),
            "at_distance": round(float(t[i] * horizontal), 4),
        })
    else:
        # Tightest clearance between the endpoints (over their own cells it is just the eye height).
        between = ~(((cols == a[0]) & (rows == a[1])) | ((cols == b[0]) & (rows == b[1])))
        pool = np.where(between, margin, np.inf) if between.any() else margin
        i = int(np.argmin(pool))
        out.update(visible=True, blocked_by=None, clearance={
            "meters": round(float(margin[i]), 4),
            "cell": [int(cols[i]), int(rows[i])],
        })
    return out


def resolve_point(level: Level, value, where: str | None = None) -> tuple[str, tuple[int, int]]:
    """A check endpoint: marker/door name or [col, row] -> (label, cell)."""
    if isinstance(value, str):
        return value, level.place(value, where)
    c, r = int(value[0]), int(value[1])
    return f"({c},{r})", (c, r)


def sight_between(level: Level, a, b) -> dict:
    la, ca = resolve_point(level, a)
    lb, cb = resolve_point(level, b)
    out = line_of_sight(level, ca, cb)
    out = {"from": la, "to": lb, **out}
    return out
