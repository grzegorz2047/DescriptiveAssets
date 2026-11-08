"""The extrude shape: a 2D outline extruded along Z (SDF, validation, bounds, loading, render, export)."""

import pathlib
import re
import shutil

import numpy as np
import pytest

from sightline import sdf
from sightline.errors import SightlineError
from sightline.export import build_meshes, edge_report
from sightline.loader import load_asset_text
from sightline.model import make_part, normalize_params
from sightline.render import render_raw

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "extrude"
L_SHAPE = [[0, 0], [0.4, 0], [0.4, 0.1], [0.1, 0.1], [0.1, 0.3], [0, 0.3]]


def _inside(px, py, poly):
    """Reference point-in-polygon (even-odd ray casting)."""
    inside = False
    n = len(poly)
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        if (ay > py) != (by > py) and px < (bx - ax) * (py - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def _seg_dist(px, py, poly):
    best = np.inf
    n = len(poly)
    for i in range(n):
        a, b = np.array(poly[i], float), np.array(poly[(i + 1) % n], float)
        e, w = b - a, np.array([px, py]) - a
        t = np.clip(w @ e / (e @ e), 0, 1)
        best = min(best, np.hypot(*(w - e * t)))
    return best


@pytest.mark.parametrize("poly", [L_SHAPE, L_SHAPE[::-1], [[0, 0], [1, 0.2], [0.3, 0.9], [0.5, 0.3]]])
def test_polygon_distance_matches_brute_force(poly):
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.3, 1.2, (400, 2))
    d = sdf.polygon2(pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32), np.array(poly, float))
    for (x, y), got in zip(pts, d):
        want = _seg_dist(x, y, poly) * (-1 if _inside(x, y, poly) else 1)
        assert got == pytest.approx(want, abs=1e-5)


def test_extrude_distances_and_bounds():
    p = make_part("l", "extrude", {"points": L_SHAPE, "depth": 0.1}, [0, 0, 0])
    assert p.local_half == pytest.approx([0.2, 0.15, 0.05])  # the outline is re-centred on its bbox
    q = np.array([[0, 0, 0], [-0.15, -0.1, 0], [0, 0, 0.2], [-0.15, -0.1, 0.06]], np.float32)
    assert p.sdf(q) == pytest.approx([0.05, -0.05, np.hypot(0.05, 0.15), 0.01], abs=1e-5)
    r = make_part("r", "extrude", {"points": L_SHAPE, "depth": 0.1}, [1, 2, 3], rotate_deg=(0, 90, 0))
    lo, hi = r.world_aabb()
    assert lo == pytest.approx([0.95, 1.85, 2.8]) and hi == pytest.approx([1.05, 2.15, 3.2])
    rounded = make_part("o", "extrude", {"points": L_SHAPE, "depth": 0.1, "round": 0.02}, [0, 0, 0])
    corner = np.array([[0.2 + 0.05, 0.15 + 0.05, 0.05 + 0.05]], np.float32)  # outside a convex corner
    assert rounded.sdf(corner)[0] > p.sdf(corner)[0]


# a front outline (a 0.4 x 0.6 rectangle at heights 1.0..1.6) and a side outline in the same heights:
# a wedge 0.2 deep at the bottom, 0.05 at the top, leaning back (z from -0.1 to +0.1)
FRONT = [[-0.2, 1.0], [0.2, 1.0], [0.2, 1.6], [-0.2, 1.6]]
SIDE = [[-0.1, 1.0], [0.1, 1.0], [0.0, 1.6], [-0.05, 1.6]]


def test_extrude_side_is_the_intersection_of_both_outlines():
    p = make_part("s", "extrude", {"points": FRONT, "side": SIDE}, [0, 0, 0])
    assert p.params["depth"] == pytest.approx(0.2)  # the side outline sets the depth
    assert p.local_half == pytest.approx([0.2, 0.3, 0.1])  # y centred on the shared heights 1.0..1.6
    # local frame: y 1.0..1.6 -> -0.3..0.3, side z -0.1..0.1 -> centred on 0
    q = np.array([[0, -0.29, 0.09],    # low and far forward: inside both
                  [0, 0.29, 0.09],     # high and far forward: inside the front outline, in front of the side's lean
                  [0, 0.29, -0.02],    # high, on the side outline
                  [0.25, 0, 0]], np.float32)  # beside the front outline
    d = p.sdf(q)
    assert d[0] < 0 and d[1] > 0.05 and d[2] < 0 and d[3] == pytest.approx(0.05, abs=1e-5)
    lo, hi = make_part("w", "extrude", {"points": FRONT, "side": SIDE}, [1, 2, 3]).world_aabb()
    assert lo == pytest.approx([0.8, 1.7, 2.9]) and hi == pytest.approx([1.2, 2.3, 3.1])


def test_extrude_side_uses_only_the_shared_heights():
    short = [[-0.1, 1.2], [0.1, 1.2], [0.1, 1.4], [-0.1, 1.4]]
    p = make_part("s", "extrude", {"points": FRONT, "side": short}, [0, 0, 0])
    assert p.local_half == pytest.approx([0.2, 0.1, 0.1])


def test_extrude_side_renders_its_profile(scratch):
    text = """asset: wedge2
style: default
parts:
  - name: plate
    shape: extrude
    points: [[-0.2, 1.0], [0.2, 1.0], [0.2, 1.6], [-0.2, 1.6]]
    side: [[-0.1, 1.0], [0.1, 1.0], [0.0, 1.6], [-0.05, 1.6]]
    material: metal
    at: [0, 1.3, 0]
"""
    asset = load_asset_text(text, scratch / "wedge2.yaml", ROOT)
    vr = render_raw(asset, ["right"], 128, scale_figure=False, ground=False)["right"]
    ys, xs = np.nonzero(vr.coverage)
    top, bottom = ys.min(), ys.max()
    width = lambda r: np.count_nonzero(vr.coverage[r]) * vr.mpp
    assert width(bottom - 1) == pytest.approx(0.2, abs=0.02) and width(top + 1) == pytest.approx(0.05, abs=0.02)
    meshes = build_meshes(asset, 64)
    assert all(open_ == 0 and bad == 0 for _, open_, bad in edge_report(meshes))


@pytest.mark.parametrize("raw, message", [
    ({"points": [[0, 0], [1, 0]], "depth": 0.1}, "list of 3 to 64"),
    ({"points": L_SHAPE}, "needs 'depth' (or 'side'"),
    ({"points": FRONT, "side": SIDE, "depth": 0.1}, "give 'depth' or 'side', not both"),
    ({"points": FRONT, "side": [[0, 2], [0.1, 2], [0.1, 2.5]]}, "shares no height"),
    ({"points": FRONT, "side": [[0, 1], [0.1, 1.5], [0.1, 1], [0, 1.5]]}, "crosses itself"),
    ({"points": FRONT, "side": SIDE, "round": 0.11}, "round 0.11 is larger"),
    ({"points": [[0, 0], [1, 0], [2, 0]], "depth": 0.1}, "no area"),
    ({"points": [[0, 0], [1, 1], [1, 0], [0, 1]], "depth": 0.1}, "crosses itself"),
    ({"points": L_SHAPE, "depth": 0}, "depth must be > 0"),
    ({"points": L_SHAPE, "depth": 0.1, "round": 0.06}, "round 0.06 is larger"),
    ({"points": L_SHAPE, "depth": 0.1, "size": [1, 1, 1]}, "does not take 'size'"),
])
def test_extrude_validation(raw, message):
    with pytest.raises(SightlineError, match=re.escape(message)):
        normalize_params("extrude", raw)


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


ASSET = """\
asset: shield
style: default
parts:
  - name: base
    shape: box
    size: [0.8, 0.1, 0.4]
    material: iron
    at: [0, 0, 0]
    align: bottom
  - name: shield
    shape: extrude
    points: [[-0.25, 0.6], [0.25, 0.6], [0.3, 0.3], [0, 0.05], [-0.3, 0.3]]
    depth: 0.04
    round: 0.01
    material: metal
    on: base.top
"""


def test_extrude_loads_renders_and_exports_closed(scratch):
    path = scratch / "shield.yaml"
    asset = load_asset_text(ASSET, path, ROOT)
    shield = next(p for p in asset.scene().parts if p.name == "shield")
    assert shield.anchor("bottom")[1] == pytest.approx(0.1)  # sits on the base
    vr = render_raw(asset, ["front"], 128, scale_figure=False, ground=False)["front"]
    sid = [i for i, p in enumerate(asset.scene().parts) if p.name == "shield"][0]
    ys, xs = np.nonzero(vr.part == sid)
    w = (xs.max() - xs.min() + 1) * vr.mpp
    h = (ys.max() - ys.min() + 1) * vr.mpp
    assert w == pytest.approx(0.6, abs=0.03) and h == pytest.approx(0.55, abs=0.03)
    meshes = build_meshes(asset, 64)
    assert all(open_ == 0 and bad == 0 for _, open_, bad in edge_report(meshes))


# ----------------------------------------------------------------------------- strokes


def _line_dist(px, py, lines):
    best = np.inf
    for line in lines:
        for a, b in zip(line[:-1], line[1:]):
            a, b = np.array(a, float), np.array(b, float)
            e, w = b - a, np.array([px, py]) - a
            t = np.clip(w @ e / (e @ e), 0, 1)
            best = min(best, np.hypot(*(w - e * t)))
    return best


def test_strokes_distance_matches_brute_force():
    lines = ([[0, 0], [0.4, 0], [0.4, 0.3]], [[0.1, 0.2], [0.3, 0.25]])
    rng = np.random.default_rng(5)
    pts = rng.uniform(-0.2, 0.6, (300, 2))
    d = sdf.polyline2(pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32), lines)
    for (x, y), got in zip(pts, d):
        assert got == pytest.approx(_line_dist(x, y, lines), abs=1e-5)
    p = normalize_params("strokes", {"points": [list(map(list, l)) for l in lines], "width": 0.02, "depth": 0.05})
    assert p["depth"] == 0.05 and len(p["points"]) == 2
    # the lines are re-centred on their bounding box grown by half the width: x -0.21..0.21, y -0.16..0.16
    allp = np.concatenate([np.array(l) for l in p["points"]])
    assert allp[:, 0].min() == pytest.approx(-0.2) and allp[:, 0].max() == pytest.approx(0.2)
    assert allp[:, 1].min() == pytest.approx(-0.15) and allp[:, 1].max() == pytest.approx(0.15)
    d = sdf.evaluate("strokes", p, [[0.0, -0.15, 0.0], [0.0, -0.15, 0.1], [0.0, 0.0, 0.0]])
    assert d[0] == pytest.approx(-0.01, abs=1e-5)  # on the first line, half the width inside
    assert d[1] == pytest.approx(0.075, abs=1e-5)  # 0.1 beyond the slab's half depth 0.025
    assert d[2] > 0.05
    from sightline.model import local_half_extents
    assert local_half_extents("strokes", p) == pytest.approx([0.21, 0.16, 0.025])


def test_strokes_side_bounds_the_depth_like_extrude():
    p = normalize_params("strokes", {"points": [[[0, 1.0], [0.4, 1.0]]], "width": 0.02,
                                     "side": [[0, 0.9], [0.05, 0.9], [0.05, 1.1], [0, 1.1]]})
    assert p["depth"] == pytest.approx(0.05)
    d = sdf.evaluate("strokes", p, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.03], [0.0, 0.0, -0.03]])
    assert d[0] == pytest.approx(-0.01, abs=1e-5) and d[1] == pytest.approx(0.005, abs=1e-5) and d[2] == pytest.approx(0.005, abs=1e-5)


@pytest.mark.parametrize("raw,message", [
    ({"points": [[[0, 0]]], "width": 0.01, "depth": 0.1}, "2 or more"),
    ({"points": [[[0, 0], [1, 0]]], "width": 0.01}, "needs 'depth'"),
    ({"points": [[[0, 0], [1, 0]]], "width": 0.01, "depth": 0.1, "side": [[0, 0], [1, 0], [1, 1]]}, "not both"),
    ({"points": [[[0, 0], [1, 0]]], "depth": 0.1}, "needs 'width'"),
])
def test_strokes_validation(raw, message):
    with pytest.raises(SightlineError, match=re.escape(message)):
        normalize_params("strokes", raw)


def test_strokes_paint_renders_a_dark_line(scratch):
    asset = load_asset_text(
        "asset: grooved\nstyle: default\nparts:\n"
        "  - {name: plate, shape: box, size: [1, 1, 0.1], material: paint_white, at: [0, 0, 0]}\n"
        "  - {name: seam, shape: strokes, points: [[[-0.4, 0], [0.4, 0]]], width: 0.04, depth: 0.2, op: paint,"
        " from: [plate], material: paint_black, at: [0, 0, 0]}\n",
        scratch / "grooved.yaml", ROOT)
    vr = render_raw(asset, ["front"], 128, scale_figure=False, ground=False)["front"]
    arr = vr.rgb.astype(int)
    ys, xs = np.nonzero(vr.coverage)
    r, c = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    mid = arr[r, c].sum()
    above = arr[r - 12, c].sum()
    assert mid < above * 0.5, (mid, above)  # the seam is dark, the plate around it bright
