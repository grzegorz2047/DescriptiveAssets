"""Voxel occupancy: grid layout, volumes, strict vs loose contacts, cuts and AABB culling."""

import math
import pathlib

import numpy as np
import pytest

from sightline.errors import SightlineError
from sightline.loader import load_asset
from sightline.model import Scene, make_part
from sightline.voxel import occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def box(name, size, pos, rotate=(0, 0, 0), op="union"):
    return make_part(name, "box", {"size": size}, pos, rotate, material=None if op == "cut" else "stone", op=op)


@pytest.mark.parametrize("part,volume", [
    (box("b", [2, 1, 1], [0, 0.5, 0]), 2.0),
    (box("b", [1.3, 0.7, 2.1], [0.2, 1, -0.4], rotate=(0, 30, 0)), 1.3 * 0.7 * 2.1),
    (make_part("s", "sphere", {"radius": 0.8}, [0, 0.8, 0], material="stone"), 4 / 3 * math.pi * 0.8**3),
    (make_part("c", "cylinder", {"radius": 0.5, "height": 2.0}, [0, 1, 0], material="stone"), math.pi * 0.25 * 2.0),
    (make_part("t", "torus", {"radius": 1.0, "thickness": 0.3}, [0, 0.3, 0], material="stone"), 2 * math.pi**2 * 1.0 * 0.09),
])
def test_volume_within_5_percent(part, volume):
    grid = occupancy(Scene([part]))
    assert grid.volume() == pytest.approx(volume, rel=0.05)
    assert grid.volume(0) == grid.volume()


def test_grid_layout():
    scene = Scene([box("b", [2, 1, 0.5], [0, 0.5, 0])])
    grid = occupancy(scene, max_cells=40)
    assert grid.cell == pytest.approx(2 / 40)
    assert grid.shape == (42, 22, 12)  # extent / cell + one cell of margin on each side
    centers = grid.centers()
    assert centers.shape == (*grid.shape, 3)
    assert centers[0, 0, 0] == pytest.approx(grid.origin)
    assert centers[1, 2, 3] - centers[0, 0, 0] == pytest.approx(np.array([1, 2, 3]) * grid.cell)
    lo, hi = scene.bounds()
    # Centered on the AABB, covering it grown by one cell.
    assert (centers[0, 0, 0] + centers[-1, -1, -1]) / 2 == pytest.approx((lo + hi) / 2)
    assert np.all(centers[0, 0, 0] - grid.cell / 2 <= lo - grid.cell + 1e-9)
    assert np.all(centers[-1, -1, -1] + grid.cell / 2 >= hi + grid.cell - 1e-9)
    assert grid.strict.shape == grid.loose.shape == (1, *grid.shape)
    assert not grid.strict[0, 0].any() and not grid.strict[0, -1].any()  # margin layers empty


def test_touching_vs_overlapping_vs_gap():
    touching = Scene([box("a", [1, 1, 1], [0, 0.5, 0]), box("b", [1, 1, 1], [1, 0.5, 0])])
    g = occupancy(touching, max_cells=48)
    assert not (g.strict[0] & g.strict[1]).any()
    assert (g.loose[0] & g.loose[1]).any()
    overlapping = Scene([box("a", [1, 1, 1], [0, 0.5, 0]), box("b", [1, 1, 1], [0.8, 0.5, 0])])
    g = occupancy(overlapping, max_cells=90)
    shared = (g.strict[0] & g.strict[1]).sum() * g.cell**3
    assert shared == pytest.approx(0.2 * 1 * 1, rel=0.1)
    apart = Scene([box("a", [1, 1, 1], [0, 0.5, 0]), box("b", [1, 1, 1], [1.2, 0.5, 0])])
    g = occupancy(apart, max_cells=48)  # gap 0.2 m = 4.4 cells
    assert not (g.loose[0] & g.loose[1]).any()


def test_cut_removes_volume():
    wall = box("wall", [3, 2, 0.4], [0, 1, 0])
    hole = box("hole", [1, 1, 1], [0, 1, 0], op="cut")
    hole.cut_from = ["wall"]
    grid = occupancy(Scene([wall, hole]), max_cells=96)
    assert grid.solids == [0]
    assert grid.volume() == pytest.approx(3 * 2 * 0.4 - 1 * 1 * 0.4, rel=0.05)
    center = np.round((np.array([0, 1, 0]) - grid.origin) / grid.cell).astype(int)
    assert not grid.strict[(0, *center)]


def test_flush_notch_has_no_loose_lid():
    """A notch flush with the wall top: the cells just above the notch are not 'near the wall'."""
    wall = box("wall", [4, 2, 0.5], [0, 1, 0])
    notch = box("notch", [0.4, 0.4, 1.0], [-1.5, 1.8, 0], op="cut")
    notch.cut_from = ["wall"]
    scene = Scene([wall, notch])
    grid = occupancy(scene, max_cells=96)
    pts = grid.centers()
    d = scene.eval_part(0, pts.reshape(-1, 3)).reshape(grid.shape)
    x, y = pts[..., 0], pts[..., 1]
    above = (np.abs(x + 1.5) < 0.2 - grid.cell) & (y > 2.0) & (d < 0.5 * grid.cell)
    assert above.any() and not grid.loose[0][above].any()  # the bound says "near", nothing is there
    merlon = (np.abs(x + 1.0) < 0.1) & (y > 2.0) & (d < 0.5 * grid.cell)
    assert merlon.any() and grid.loose[0][merlon].all()


def test_culling_matches_brute_force_and_symmetry():
    a = load_asset(FIX / "tower30.yaml", ROOT)
    scene = a.scene()
    grid = occupancy(scene, max_cells=40)
    pts = grid.centers().reshape(-1, 3)
    for k, i in enumerate(grid.solids):
        d = scene.eval_part(i, pts).reshape(grid.shape)
        assert np.array_equal(grid.strict[k], d < 0), a.parts[i].name
        if not scene.cutters[i]:
            assert np.array_equal(grid.loose[k], d < 0.5 * grid.cell), a.parts[i].name
        else:  # minus the 0-distance sheet of coplanar cut faces, and only right next to a cutter
            dropped = (d < 0.5 * grid.cell) & ~grid.loose[k]
            assert not (grid.loose[k] & ~(d < 0.5 * grid.cell)).any() and dropped.sum() <= 12
            near_cut = np.min([a.parts[c].sdf(pts) for c in scene.cutters[i]], axis=0).reshape(grid.shape)
            assert np.all(near_cut[dropped] < grid.cell), a.parts[i].name
    assert grid.solids == scene.solid_indices()
    assert grid.layer(a.part_index("roof")) == grid.solids.index(a.part_index("roof"))
    # A symmetric scene on a centered grid mirrors cell for cell.
    sym = Scene([box("a", [1, 1, 1], [-1, 0.5, 0]), box("b", [1, 1, 1], [1, 0.5, 0]),
                 make_part("c", "sphere", {"radius": 0.4}, [0, 1.4, 0], material="stone")])
    union = occupancy(sym, max_cells=50).union()
    assert np.array_equal(union, union[::-1])


def test_helpers_and_errors():
    body = box("body", [1, 1, 1], [0, 0.5, 0])
    human = make_part("human", "capsule", {"radius": 0.25, "height": 1.8}, [2, 0.9, 0], material="helper", helper=True)
    scene = Scene([body, human])
    assert occupancy(scene).solids == [0]
    with_h = occupancy(scene, include_helpers=True)
    assert with_h.solids == [0, 1] and with_h.volume(1) > 0
    with pytest.raises(SightlineError, match="no solid parts"):
        occupancy(Scene([human]))
