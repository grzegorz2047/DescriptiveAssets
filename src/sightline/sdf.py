"""Signed distance functions of the primitive shapes, in the shape's local frame.

Local frame: centered on the shape's local bounding box, Y = the shape's main axis.
Every primitive takes coordinate arrays `x, y, z` (float32, same shape) and returns float32
distances: negative inside, positive outside. Distances are exact, or a conservative lower
bound where noted (never an overestimate), so sphere tracing never steps through a surface.

Scalar parameters must be Python floats (not numpy float64) so results stay float32.
"""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32


def _len2(a, b):
    return np.sqrt(a * a + b * b)


def _len3(a, b, c):
    return np.sqrt(a * a + b * b + c * c)


def _extrude(d2, dy):
    """Exact 3D distance of a 2D shape (signed distance d2) extruded over |axis| <= h (dy = |axis| - h)."""
    return np.minimum(np.maximum(d2, dy), 0.0) + _len2(np.maximum(d2, 0.0), np.maximum(dy, 0.0))


def box(x, y, z, hx: float, hy: float, hz: float, r: float = 0.0):
    """Box with half extents (hx, hy, hz) and edge radius r (exact)."""
    qx = np.abs(x) - (hx - r)
    qy = np.abs(y) - (hy - r)
    qz = np.abs(z) - (hz - r)
    outside = _len3(np.maximum(qx, 0.0), np.maximum(qy, 0.0), np.maximum(qz, 0.0))
    inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return outside + inside - r


def sphere(x, y, z, r: float):
    """Sphere of radius r (exact)."""
    return _len3(x, y, z) - r


def ellipsoid(x, y, z, hx: float, hy: float, hz: float):
    """Ellipsoid with semi-axes (hx, hy, hz) — conservative bound.

    Outside, f = |p/h| is convex, so (f - 1) / |grad f| is a lower bound of the distance
    (tight near the surface). Inside, f is (1/min h)-Lipschitz, so (f - 1) * min h is one.
    """
    px, py, pz = x / hx, y / hy, z / hz
    k0 = _len3(px, py, pz)
    k1 = _len3(px / hx, py / hy, pz / hz)
    outside = k0 * (k0 - 1.0) / np.maximum(k1, 1e-20)
    inside = (k0 - 1.0) * min(hx, hy, hz)
    return np.where(k0 > 1.0, outside, inside).astype(F32, copy=False)


def _fold_polygon(x, z, sides: int):
    """Fold (x, z) into the sector of a regular n-gon face centered on +Z.

    Returns (u, v): u along the face normal, |v| along the face. Faces sit at angles
    k * 360/n measured from +Z toward +X, so one flat face looks toward +Z.
    """
    sector = 2.0 * math.pi / sides
    phi = np.arctan2(x, z)
    phi = phi - sector * np.round(phi / sector)
    rho = _len2(x, z)
    return rho * np.cos(phi), np.abs(rho * np.sin(phi))


def polygon2d(x, z, apothem: float, sides: int):
    """Exact 2D signed distance to a regular n-gon in the XZ plane with a flat face toward +Z."""
    u, v = _fold_polygon(x, z, sides)
    half_edge = apothem * math.tan(math.pi / sides)
    outside = _len2(u - apothem, np.maximum(v - half_edge, 0.0))
    return np.where(u > apothem, outside, u - apothem)


def cylinder(x, y, z, r: float, hh: float, sides: int = 0, rr: float = 0.0):
    """Cylinder (radius r, half height hh), optional n-gon prism and edge radius rr (exact)."""
    if sides:
        apothem = r * math.cos(math.pi / sides)
        d2 = polygon2d(x, z, apothem - rr, sides)
    else:
        d2 = _len2(x, z) - (r - rr)
    return _extrude(d2, np.abs(y) - (hh - rr)) - rr


def _capped_cone2d(q, y, r1: float, r2: float, h: float):
    """Exact distance in the (radial q >= 0, y) half plane to a frustum section.

    Bottom radius r1 at y = -h, top radius r2 at y = +h (Inigo Quilez's capped cone).
    """
    k2x, k2y = r2 - r1, 2.0 * h
    k2dot = k2x * k2x + k2y * k2y
    cax = q - np.minimum(q, np.where(y < 0.0, F32(r1), F32(r2)))
    cay = np.abs(y) - h
    s = np.clip(((r2 - q) * k2x + (h - y) * k2y) / k2dot, 0.0, 1.0)
    cbx = q - r2 + k2x * s
    cby = y - h + k2y * s
    sign = np.where((cbx < 0.0) & (cay < 0.0), F32(-1.0), F32(1.0))
    return sign * np.sqrt(np.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby))


def cone(x, y, z, r1: float, r2: float, hh: float, sides: int = 0):
    """Cone or frustum: bottom radius r1 at y = -hh, top radius r2 at y = +hh.

    Round: exact. With `sides`: an n-gon pyramid/frustum; exact inside, a conservative bound
    outside = max(distance to the face's half-plane section, distance to the circumscribed
    frustum), both lower bounds of the true distance.
    """
    rho = _len2(x, z)
    d_round = _capped_cone2d(rho, y, r1, r2, hh)
    if not sides:
        return d_round
    c = math.cos(math.pi / sides)
    u, _ = _fold_polygon(x, z, sides)
    return np.maximum(_capped_cone2d(u, y, r1 * c, r2 * c, hh), d_round)


def capsule(x, y, z, r: float, hh: float):
    """Capsule along Y, radius r, total half height hh (>= r) (exact)."""
    a = hh - r
    yc = y - np.clip(y, -a, a)
    return _len3(x, yc, z) - r


def torus(x, y, z, ring: float, tube: float):
    """Torus in the XZ plane: ring center radius `ring`, tube radius `tube` (exact)."""
    return _len2(_len2(x, z) - ring, y) - tube


def wedge(x, y, z, hx: float, hy: float, hz: float):
    """Triangular prism along X: full height at the back (-Z), slope to 0 at the front edge (+Z) (exact).

    Cross-section in (z, y): A = back-bottom (-hz, -hy), B = back-top (-hz, hy), C = front-bottom (hz, -hy).
    """
    d_back = _len2(z + hz, np.maximum(np.abs(y) - hy, 0.0))  # segment A-B
    d_bottom = _len2(y + hy, np.maximum(np.abs(z) - hz, 0.0))  # segment A-C
    ex, ey = 2.0 * hz, -2.0 * hy  # segment B-C
    wx, wy = z + hz, y - hy
    s = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
    d_slope = _len2(wx - s * ex, wy - s * ey)
    d2 = np.minimum(np.minimum(d_back, d_bottom), d_slope)
    inside = (z >= -hz) & (y >= -hy) & (y * (1.0 / hy) + z * (1.0 / hz) <= 0.0)
    d2 = np.where(inside, -d2, d2)
    return _extrude(d2, np.abs(x) - hx)


def polygon2(x, y, verts) -> np.ndarray:
    """Signed distance to a simple 2D polygon (verts (E, 2), any winding; negative inside; exact)."""
    v = np.asarray(verts, dtype=np.float64)
    shape = np.shape(x)
    px = np.asarray(x, dtype=np.float32).reshape(-1)
    py = np.asarray(y, dtype=np.float32).reshape(-1)
    d2 = (px - F32(v[0, 0])) ** 2 + (py - F32(v[0, 1])) ** 2
    inside = np.zeros(px.shape, dtype=bool)
    n = len(v)
    for i in range(n):
        ax, ay = v[i]
        bx, by = v[(i + 1) % n]
        ex, ey = F32(bx - ax), F32(by - ay)
        wx, wy = px - F32(ax), py - F32(ay)
        ee = float(ex * ex + ey * ey)
        t = np.clip((wx * ex + wy * ey) / F32(ee if ee > 0 else 1.0), 0.0, 1.0)
        dx, dy = wx - ex * t, wy - ey * t
        d2 = np.minimum(d2, dx * dx + dy * dy)
        # crossing number: the edge straddles the horizontal line through p, to the right of p
        up = (py >= F32(ay)) & (py < F32(by))
        down = (py < F32(ay)) & (py >= F32(by))
        cross = ex * wy - ey * wx
        inside ^= (up & (cross > 0)) | (down & (cross < 0))
    d = np.sqrt(d2)
    return np.where(inside, -d, d).astype(np.float32).reshape(shape)


def extrude(x, y, z, verts, hz: float, r: float = 0.0, side=None):
    """A 2D polygon in the XY plane extruded over |z| <= hz, edges rounded by r (exact for r = 0).

    With `side` (a polygon in the ZY plane, the outline seen from the side) the shape is the
    intersection of the two extrusions, the front outline along Z and the side outline along X (a
    bound, exact outside the shared edges); `hz` is then unused.
    """
    d2 = polygon2(x, y, verts)
    if side is not None:
        ds = polygon2(z, y, side)
        if r > 0.0:
            return _extrude(d2 + F32(r), ds + F32(r)) - F32(r)
        return _extrude(d2, ds)
    if r > 0.0:
        return _extrude(d2 + F32(r), np.abs(z) - F32(hz - r)) - F32(r)
    return _extrude(d2, np.abs(z) - F32(hz))


def smin(a, b, k: float):
    """Polynomial smooth minimum with radius k (k <= 0: hard min). 1-Lipschitz, <= min(a, b)."""
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + (0.5 / k) * (b - a), 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def primitive(shape: str, params: dict, x, y, z):
    """Distance of a normalized shape (see model.normalize_params) at local coordinates."""
    if shape == "box":
        hx, hy, hz = params["size"]
        return box(x, y, z, hx * 0.5, hy * 0.5, hz * 0.5, params["round"])
    if shape == "sphere":
        return sphere(x, y, z, params["radius"])
    if shape == "ellipsoid":
        hx, hy, hz = params["size"]
        return ellipsoid(x, y, z, hx * 0.5, hy * 0.5, hz * 0.5)
    if shape == "cylinder":
        return cylinder(x, y, z, params["radius"], params["height"] * 0.5, params["sides"], params["round"])
    if shape == "cone":
        return cone(x, y, z, params["radius"], params["top_radius"], params["height"] * 0.5, params["sides"])
    if shape == "capsule":
        return capsule(x, y, z, params["radius"], params["height"] * 0.5)
    if shape == "torus":
        return torus(x, y, z, params["radius"], params["thickness"])
    if shape == "wedge":
        hx, hy, hz = params["size"]
        return wedge(x, y, z, hx * 0.5, hy * 0.5, hz * 0.5)
    if shape == "extrude":
        return extrude(x, y, z, params["points"], params["depth"] * 0.5, params["round"], params.get("side"))
    raise ValueError(f"unknown shape {shape!r}")


def evaluate(shape: str, params: dict, local_pts) -> np.ndarray:
    """Distance of a normalized shape at local points (N, 3) -> (N,) float32."""
    p = np.asarray(local_pts, dtype=F32)
    return np.asarray(primitive(shape, params, p[:, 0], p[:, 1], p[:, 2]), dtype=F32)
