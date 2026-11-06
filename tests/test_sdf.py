"""Primitive SDFs: exact values at known points, sign, 1-Lipschitz (=> never overestimates)."""

import math

import numpy as np
import pytest

from sightline import sdf
from sightline.model import make_part, normalize_params

RNG = np.random.default_rng(1234)


def at(fn, *pts, **kw):
    p = np.array(pts, dtype=np.float64)
    return fn(p[:, 0], p[:, 1], p[:, 2], **kw)


def test_box_exact():
    d = at(sdf.box, (0, 0, 0), (2, 0, 0), (1, 0, 0), (2, 1.5, 0), (0.5, 0, 0), hx=1.0, hy=0.5, hz=0.25)
    assert d == pytest.approx([-0.25, 1.0, 0.0, math.hypot(1.0, 1.0), -0.25])


def test_round_box_corner_and_faces():
    # Rounded edges: the corner is cut by a sphere of radius r around the shrunk corner.
    r = 0.2
    d = at(sdf.box, (1, 1, 1), (2, 0, 0), (0, 0, 0), hx=1.0, hy=1.0, hz=1.0, r=r)
    # (1,1,1) to the sphere of radius r around the shrunk corner (0.8, 0.8, 0.8)
    assert d == pytest.approx([math.sqrt(3) * 0.2 - r, 1.0, -1.0])


def test_sphere_and_capsule():
    assert at(sdf.sphere, (0, 0, 0), (3, 4, 0), r=2.0) == pytest.approx([-2.0, 3.0])
    # capsule radius 0.5, total height 3 -> segment y in [-1, 1]
    d = at(sdf.capsule, (0, 0, 0), (2, 0, 0), (0, 2.5, 0), (0, 1.5, 0), (1.5, 1, 0), r=0.5, hh=1.5)
    assert d == pytest.approx([-0.5, 1.5, 1.0, 0.0, 1.0])


def test_cylinder_round_and_polygon():
    d = at(sdf.cylinder, (0, 0, 0), (3, 0, 0), (0, 2, 0), (3, 2, 0), r=1.0, hh=1.0)
    assert d == pytest.approx([-1.0, 2.0, 1.0, math.hypot(2, 1)])
    # Hexagon prism, circumradius 1: flat face toward +Z at the apothem cos(30deg).
    apo = math.cos(math.pi / 6)
    d = at(sdf.cylinder, (0, 0, 2), (0, 0, apo), (2 * math.sin(math.pi / 6), 0, 2 * math.cos(math.pi / 6)),
           r=1.0, hh=1.0, sides=6)
    # (0,0,2): 2 - apothem; vertex direction (30deg from +Z): beyond the vertex at radius 1 -> 2 - 1.
    assert d == pytest.approx([2 - apo, 0.0, 1.0], abs=1e-6)
    # Rounded cylinder: the rim edge is a quarter circle of radius rr.
    rr = 0.1
    d = at(sdf.cylinder, (2, 2, 0), (0, 1.5, 0), r=1.0, hh=1.0, rr=rr)
    assert d == pytest.approx([math.hypot(1.1, 1.1) - rr, 0.5], abs=1e-6)


def test_cone_and_frustum():
    # Cone radius 1 at y=-1, apex at y=+1.
    d = at(sdf.cone, (0, 2, 0), (0, -2, 0), (0, 0, 0), (0.5, 0, 0), r1=1.0, r2=0.0, hh=1.0)
    slant = 1 / math.sqrt(5)  # distance from the axis point (0,0,0) to the side line x = (1 - y) / 2
    assert d == pytest.approx([1.0, 1.0, -slant, 0.0], abs=1e-6)
    # Frustum: bottom radius 1, top radius 0.5, the top cap is flat.
    d = at(sdf.cone, (0, 1.5, 0), (0.25, 1.0, 0), r1=1.0, r2=0.5, hh=1.0)
    assert d == pytest.approx([0.5, 0.0], abs=1e-6)


def test_torus_and_wedge():
    d = at(sdf.torus, (2, 0, 0), (0, 0, 0), (2, 1, 0), (0, 0, 3), ring=2.0, tube=0.5)
    assert d == pytest.approx([-0.5, 1.5, 0.5, 0.5])
    # Wedge 2 x 2 x 2: back face z=-1 full height, slope from (y=1, z=-1) to (y=-1, z=1).
    d = at(sdf.wedge, (0, 0, -2), (0, -2, 0), (0, 1, 1), (3, -0.5, -0.5), (0, -0.5, -0.5), hx=1, hy=1, hz=1)
    assert d == pytest.approx([1.0, 1.0, math.sqrt(2), 2.0, -0.5], abs=1e-6)


@pytest.mark.parametrize("h", [(2.0, 0.5, 1.0), (1.0, 0.15, 0.3)])
def test_ellipsoid_exact_on_axes_and_bound_elsewhere(h):
    d = at(sdf.ellipsoid, (h[0] + 1, 0, 0), (0, h[1] + 1, 0), (0, 0, 0), (h[0], 0, 0), hx=h[0], hy=h[1], hz=h[2])
    assert d == pytest.approx([1.0, 1.0, -min(h), 0.0], abs=1e-6)
    # Off-axis: compare with the true distance to a dense surface sample; never larger.
    u, v = np.meshgrid(np.linspace(0, math.pi, 300), np.linspace(0, 2 * math.pi, 600), indexing="ij")
    surf = np.stack([h[0] * np.sin(u) * np.cos(v), h[1] * np.cos(u), h[2] * np.sin(u) * np.sin(v)], -1).reshape(-1, 3)
    pts = RNG.uniform(-3, 3, (200, 3))
    true = np.array([np.sqrt(((surf - x) ** 2).sum(1)).min() for x in pts])
    inside = ((pts / h) ** 2).sum(1) < 1
    true[inside] *= -1
    est = sdf.ellipsoid(pts[:, 0], pts[:, 1], pts[:, 2], *h)
    assert np.all(np.sign(est) == np.sign(true))
    assert np.all(np.abs(est) <= np.abs(true) + 0.01)  # sampling error of the surface
    # Near the surface the bound is tight.
    near = np.abs(true) < 0.3
    assert np.allclose(est[near], true[near], atol=0.05)


def test_polygon_cone_is_a_lower_bound():
    """n-gon pyramid: exact inside; outside never larger than the true distance."""
    n, r1, hh = 5, 1.0, 1.0
    ang = (np.arange(n) + 0.5) * 2 * math.pi / n
    base = np.stack([r1 * np.sin(ang), np.full(n, -hh), r1 * np.cos(ang)], 1)
    apex = np.array([0, hh, 0])
    # Dense sample of the pyramid surface (side triangles + base polygon).
    s, t = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120), indexing="ij")
    s, t = s.ravel(), t.ravel()
    keep = s + t <= 1
    s, t = s[keep], t[keep]
    pts_surf = []
    for k in range(n):
        a, b = base[k], base[(k + 1) % n]
        pts_surf.append(a + s[:, None] * (b - a) + t[:, None] * (apex - a))
        c = np.array([0, -hh, 0])
        pts_surf.append(c + s[:, None] * (a - c) + t[:, None] * (b - c))
    surf = np.concatenate(pts_surf)
    pts = RNG.uniform(-2, 2, (150, 3))
    true = np.array([np.sqrt(((surf - x) ** 2).sum(1)).min() for x in pts])
    est = sdf.cone(pts[:, 0], pts[:, 1], pts[:, 2], r1, 0.0, hh, sides=n)
    outside = est > 0
    assert np.all(est[outside] <= true[outside] + 0.01)
    assert np.mean(est[outside] / true[outside]) > 0.85  # and not uselessly loose


SHAPES = [
    ("box", {"size": [1.0, 2.0, 0.5], "round": 0.1}),
    ("sphere", {"radius": 0.7}),
    ("ellipsoid", {"size": [2.0, 0.6, 1.2]}),
    ("cylinder", {"radius": 0.8, "height": 1.5}),
    ("cylinder", {"radius": 0.8, "height": 1.5, "sides": 8, "round": 0.1}),
    ("cone", {"radius": 1.0, "height": 2.0}),
    ("cone", {"radius": 1.0, "height": 2.0, "top_radius": 0.4, "sides": 6}),
    ("capsule", {"radius": 0.3, "height": 1.6}),
    ("torus", {"radius": 1.0, "thickness": 0.25}),
    ("wedge", {"size": [1.0, 0.8, 1.6]}),
]


LIPSCHITZ = [(s, r) for s, r in SHAPES if s != "ellipsoid"]


@pytest.mark.parametrize("shape,raw", LIPSCHITZ, ids=[f"{s}{i}" for i, (s, _) in enumerate(LIPSCHITZ)])
def test_lipschitz_and_float32(shape, raw):
    """|d(p) - d(q)| <= |p - q| everywhere: with d = 0 on the surface this means no overestimate.

    The ellipsoid is a gradient-normalized bound that is not 1-Lipschitz far away (but is still a
    lower bound); test_ellipsoid_* checks it against true distances instead.
    """
    params = normalize_params(shape, raw)
    p = RNG.uniform(-2.5, 2.5, (4000, 3))
    q = p + RNG.normal(0, 0.05, p.shape)
    dp = sdf.evaluate(shape, params, p).astype(np.float64)
    dq = sdf.evaluate(shape, params, q).astype(np.float64)
    ratio = np.abs(dp - dq) / np.linalg.norm(p - q, axis=1)
    assert ratio.max() <= 1.0 + 1e-3
    assert sdf.evaluate(shape, params, p).dtype == np.float32


@pytest.mark.parametrize("shape,raw", SHAPES, ids=[f"{s}{i}" for i, (s, _) in enumerate(SHAPES)])
def test_shape_fits_local_bbox(shape, raw):
    """The shape stays inside its local bbox; round shapes touch every bbox face."""
    part = make_part("p", shape, raw, [0, 0, 0], material="wood")
    half = part.local_half
    polygonal = shape in ("box", "wedge") or bool(part.params.get("sides"))
    for axis in range(3):
        a, b = [k for k in range(3) if k != axis]
        u, v = np.meshgrid(np.linspace(-half[a], half[a], 101), np.linspace(-half[b], half[b], 101))
        for sign in (-1, 1):
            plane = np.zeros((u.size, 3))
            plane[:, axis] = sign * (half[axis] + 0.05)
            plane[:, a], plane[:, b] = u.ravel(), v.ravel()
            closest = part.sdf(plane).min()
            assert closest >= 0.05 - 1e-5
            if not polygonal or shape in ("box", "wedge"):
                assert closest <= 0.05 + 1e-5, (axis, sign, closest)


def test_smin():
    a = np.array([0.0, 1.0, 5.0])
    b = np.array([0.0, 1.2, 0.5])
    s = sdf.smin(a, b, 0.5)
    assert np.all(s <= np.minimum(a, b) + 1e-12)
    assert s[0] == pytest.approx(-0.125)  # equal inputs: min - k/4
    assert s[2] == pytest.approx(0.5)  # far apart (|a - b| >= k): plain min
    assert np.array_equal(sdf.smin(a, b, 0.0), np.minimum(a, b))
