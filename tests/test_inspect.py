"""Inspect: measured sizes, volumes, ground, mass, symmetry, contacts, floating pieces, cuts,
visibility, colors, silhouettes, readability, compare, warnings and the text/diff forms."""

import json
import pathlib
import textwrap
import time

import numpy as np
import pytest

from sightline.camera import view_basis
from sightline.errors import SightlineError
from sightline.inspect import (
    compare_silhouettes,
    contrast_ratio,
    diff_reports,
    format_report,
    hex_to_rgb,
    inspect_asset,
    readability,
    visible_fractions,
)
from sightline.loader import load_asset, load_asset_text
from sightline.voxel import occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
SPEC_KEYS = {"asset", "style", "brief", "part_count", "cut_count", "voxel_cell", "render_mpp", "size", "ground", "mass",
             "symmetry", "parts", "contacts", "floating", "cuts", "colors", "contrast", "silhouette", "readability",
             "compare", "warnings"}


def load(body: str, name: str = "t"):
    text = f"asset: {name}\n" + textwrap.dedent(body).lstrip("\n")
    return load_asset_text(text, FIX / f"{name}.yaml", ROOT)


def part(report, name):
    return next(p for p in report["parts"] if p["name"] == name)


@pytest.fixture(scope="module")
def box():
    a = load("""
        parts:
          - {name: block, shape: box, size: [2, 3, 1], material: stone, align: bottom}
        """, "box")
    return a, inspect_asset(a)


@pytest.fixture(scope="module")
def tower():
    a = load_asset(FIX / "tower30.yaml", ROOT)
    t0 = time.perf_counter()
    r = inspect_asset(a)
    return a, r, time.perf_counter() - t0


# ----------------------------------------------------------------------------- size, ground, mass


def test_box_measurements(box):
    a, r = box
    assert SPEC_KEYS <= set(r)
    s = r["size"]
    assert (s["width"], s["height"], s["depth"]) == (2.0, 3.0, 1.0)
    assert s["min"] == [-1.0, 0.0, -0.5] and s["max"] == [1.0, 3.0, 0.5]
    assert s["human_ratio"] == pytest.approx(3 / 1.8, abs=1e-3)
    cell = r["voxel_cell"]
    assert cell == pytest.approx(3 / 96, abs=1e-4)
    assert r["mass"]["volume"] == pytest.approx(6.0, rel=0.05)
    assert r["mass"]["center"] == pytest.approx([0.0, 1.5, 0.0], abs=cell)
    assert r["mass"]["stable"] is True
    assert r["mass"]["support_margin"] == pytest.approx(0.5, abs=cell)  # COM to the nearest footprint edge
    assert r["ground"] == {"min_y": 0.0, "grounded": True, "floats_by": 0.0}
    assert r["symmetry"]["x"] >= 0.98 and r["symmetry"]["z"] >= 0.98
    assert (r["part_count"], r["cut_count"], r["pieces"]) == (1, 0, 1)
    assert r["contacts"] == [] and r["floating"] == [] and r["cuts"] == []
    (p,) = r["parts"]
    assert p["center"] == [0.0, 1.5, 0.0] and p["size"] == [2.0, 3.0, 1.0] and p["volume_share"] == 1.0
    assert p["visible"] == {"front": 1.0, "right": 1.0, "top": 1.0, "iso": 1.0}
    assert r["colors"] == {"view": "iso", "shares": {"stone": 1.0}}
    assert r["warnings"] == []
    json.dumps(r)  # JSON-serializable


def test_lit_color_is_lambert_over_visible_faces(box):
    """Mean lit color = base x (ambient + (1 - ambient) max(0, n.-L)), weighted by projected face area."""
    a, r = box
    style = a.style
    to_light = -np.asarray(style.light_dir) / np.linalg.norm(style.light_dir)
    forward, _, _ = view_basis("iso")
    faces = [((0, 1, 0), 2 * 1), ((0, 0, 1), 2 * 3), ((1, 0, 0), 3 * 1)]  # iso sees top, front, right
    weights = [area * abs(np.dot(n, forward)) for n, area in faces]
    shades = [style.ambient + (1 - style.ambient) * max(0.0, float(np.dot(n, to_light))) for n, _ in faces]
    shade = np.dot(weights, shades) / sum(weights)
    expected = np.array(style.material("stone").color) * shade
    measured = np.array(hex_to_rgb(r["parts"][0]["lit_color"]))
    assert measured == pytest.approx(expected, abs=0.02)


def test_whole_asset_floating():
    a = load("""
        parts:
          - {name: ball, shape: sphere, radius: 0.5, material: gold, at: [0, 1.5, 0]}
        """, "balloon")
    r = inspect_asset(a)
    assert r["ground"] == {"min_y": 1.0, "grounded": False, "floats_by": 1.0}
    assert r["mass"]["stable"] is None and r["mass"]["support_margin"] is None
    assert r["floating"] == []  # one piece; being off the ground is the ground fact
    assert r["warnings"][0].startswith("the asset floats 1.00 m above the ground (lowest part ball)")


def test_sunk_asset_warns_with_amount():
    a = load("""
        parts:
          - {name: slab, shape: box, size: [2, 1, 2], material: stone, at: [0, -0.3, 0], align: bottom}
        """, "sunk")
    r = inspect_asset(a)
    assert r["ground"]["min_y"] == -0.3 and not r["ground"]["grounded"]
    assert any(w.startswith("the asset sinks 0.30 m below the ground") for w in r["warnings"])


def test_stable_vs_unstable_overhang():
    body = """
        parts:
          - {{name: pillar, shape: box, size: [0.4, 2, 0.4], material: stone, align: bottom}}
          - {{name: slab, shape: box, size: [3, 0.5, 1], material: stone, on: pillar.top, offset: [{dx}, 0, 0]}}
        """
    centered = inspect_asset(load(body.format(dx=0), "tee"))
    m = centered["mass"]
    assert m["stable"] is True
    assert m["center"][0] == pytest.approx(0.0, abs=0.01)
    assert m["support_margin"] == pytest.approx(0.2, abs=0.035)  # half the pillar width
    over = inspect_asset(load(body.format(dx=1.4), "overhang"))
    m = over["mass"]
    com_x = (0.4 * 2 * 0.4 * 0 + 3 * 0.5 * 1 * 1.4) / (0.32 + 1.5)
    assert m["center"][0] == pytest.approx(com_x, abs=0.03)
    assert m["stable"] is False
    assert m["support_margin"] == pytest.approx(-(com_x - 0.2), abs=0.04)
    assert any(w.startswith("unstable: the center of mass") for w in over["warnings"])
    # Mirror across the AABB center x = 1.35: the slab maps onto 2.9 of its 3 m, the pillar onto nothing.
    inter = 2.9 * 0.5 * 1
    assert over["symmetry"]["x"] == pytest.approx(inter / (2 * 1.82 - inter), abs=0.02)
    assert centered["symmetry"]["x"] >= 0.97


# ----------------------------------------------------------------------------- contacts, floating


def test_contacts_touch_vs_overlap():
    a = load("""
        parts:
          - {name: a, shape: box, size: [1, 1, 1], material: stone, align: bottom}
          - {name: b, shape: box, size: [1, 1, 1], material: stone, at: [1, 0, 0], align: bottom}
          - {name: c, shape: box, size: [1, 1, 1], material: wood, at: [0, 0, 0.8], align: bottom}
          - {name: d, shape: box, size: [1, 1, 1], material: wood, at: [4, 0, 0], align: bottom}
        """, "contacts")
    r = inspect_asset(a, max_cells=128)
    got = {(c["a"], c["b"]): c for c in r["contacts"]}
    assert set(got) == {("a", "b"), ("a", "c"), ("b", "c")}  # d stands apart
    assert got[("a", "b")]["kind"] == "touch" and got[("a", "b")]["overlap_volume"] == 0.0
    assert got[("b", "c")]["kind"] == "touch"
    assert got[("a", "c")]["kind"] == "overlap"
    assert got[("a", "c")]["overlap_volume"] == pytest.approx(1 * 1 * 0.2, rel=0.15)
    assert r["pieces"] == 2 and r["floating"] == []  # d is a second piece, but on the ground


def test_resting_sphere_touches_even_between_cell_centers():
    """The base top (y = 1) lies exactly halfway between two layers of cell centers."""
    a = load("""
        parts:
          - {name: base, shape: box, size: [4, 1, 2], material: stone, align: bottom}
          - {name: tower, shape: box, size: [1, 3, 1], material: wood, on: base.top}
          - {name: knob, shape: sphere, radius: 0.2, material: gold, on: base.top_left, offset: [0.4, 0, 0]}
        """, "knob")
    r = inspect_asset(a)
    grid = occupancy(a.scene())
    assert ((1.0 - grid.origin[1]) / grid.cell) % 1 == pytest.approx(0.5, abs=1e-6)  # the hard case
    kinds = {(c["a"], c["b"]): c["kind"] for c in r["contacts"]}
    assert kinds == {("base", "tower"): "touch", ("base", "knob"): "touch"}
    assert r["floating"] == [] and r["pieces"] == 1


def test_floating_sphere_gap_within_one_voxel():
    a = load("""
        parts:
          - {name: block, shape: box, size: [2, 1, 2], material: stone, align: bottom}
          - {name: ball, shape: sphere, radius: 0.4, material: gold, at: [0, 1.9, 0]}
        """, "floater")
    r = inspect_asset(a)
    (f,) = r["floating"]
    assert f["parts"] == ["ball"] and f["nearest"] == "block" and f["direction"] == "above"
    assert f["gap"] == pytest.approx(0.5, abs=r["voxel_cell"])
    assert r["pieces"] == 2 and r["ground"]["grounded"]
    (w,) = [w for w in r["warnings"] if w.startswith("ball")]
    assert f"ball floats {f['gap']:.2f} m above block" in w and "on: block.top" in w and "down" in w


def test_floating_beside_names_side_and_anchor():
    a = load("""
        parts:
          - {name: post, shape: box, size: [0.5, 3, 0.5], material: wood, align: bottom}
          - {name: sign, shape: box, size: [0.6, 0.4, 0.1], material: wood, at: [0.85, 2, 0]}
        """, "signpost")
    r = inspect_asset(a)
    (f,) = r["floating"]
    assert f["direction"] == "right of" and f["nearest"] == "post"
    assert f["gap"] == pytest.approx(0.30, abs=r["voxel_cell"])
    assert any("sign floats" in w and "right of post" in w and "on: post.right" in w for w in r["warnings"])


def test_diagonal_floating_gets_a_move_that_closes_the_gap():
    body = """
        parts:
          - {{name: block, shape: box, size: [1, 1, 1], material: stone, at: [0, 0.5, 0]}}
          - {{name: ball, shape: sphere, radius: 0.2, material: metal, at: [{x}, {y}, 0]}}
        """
    r = inspect_asset(load(body.format(x=2.0, y=1.7), "diag"))
    (f,) = r["floating"]
    # By hand: the nearest block point is its corner edge (0.5, 1.0): |(1.5, 0.7)| - 0.2 = 1.455 m.
    assert f["gap"] == pytest.approx(1.455, abs=r["voxel_cell"])
    dx, dy, dz = f["move"]
    assert dx == pytest.approx(-1.32, abs=0.05) and dy == pytest.approx(-0.62, abs=0.05) and dz == 0.0
    (w,) = [w for w in r["warnings"] if w.startswith("ball floats")]
    assert f"move it {abs(dx):.2f} m left (-x) and {abs(dy):.2f} m down" in w
    # Following the advice closes the gap: no floating piece any more.
    moved = inspect_asset(load(body.format(x=round(2.0 + dx, 3), y=round(1.7 + dy, 3)), "diag2"))
    assert moved["floating"] == [] and moved["pieces"] == 1


def test_floating_beside_a_cone_suggests_no_bbox_anchor():
    a = load("""
        parts:
          - {name: spire, shape: cone, radius: 1, height: 3, material: stone, align: bottom}
          - {name: flag, shape: box, size: [0.2, 0.2, 0.2], material: paint_red, at: [0, 1.5, 1.2]}
        """, "spire")
    r = inspect_asset(a)
    (w,) = [w for w in r["warnings"] if w.startswith("flag floats")]
    assert "on: spire" not in w and " — move it " in w  # the bbox front anchor is off the cone's surface


def test_part_hovering_in_a_flush_notch_floats():
    """A notch flush with the wall top leaves a 0-distance sheet in the SDF bound: no contact there."""
    a = load("""
        parts:
          - {name: wall, shape: box, size: [4, 2, 0.5], material: stone, align: bottom}
          - {name: notch, shape: box, size: [0.4, 0.4, 1.0], op: cut, on: wall.top_left, align: top_left,
             offset: [0.3, 0, 0], repeat: {count: 5, step: [0.8, 0, 0]}}
          - {name: gem, shape: box, size: [0.1, 0.1, 0.1], material: gold, at: [-1.5, 2.06, 0]}
        """, "crenel")
    r = inspect_asset(a)
    assert r["contacts"] == []
    (f,) = r["floating"]
    assert f["parts"] == ["gem"] and f["gap"] == pytest.approx(0.15, abs=r["voxel_cell"])  # to the merlon corner
    notches = {c["name"]: c for c in r["cuts"]}
    assert notches["notch_0"]["removed_volume"] == pytest.approx(0.08, rel=0.02)
    assert notches["notch_0"]["visible"]["top"] >= 0.9 and notches["notch_0"]["visible"]["iso"] > 0.2


# ----------------------------------------------------------------------------- visibility and cuts


def test_occluded_part_visibility():
    a = load("""
        parts:
          - {name: wall, shape: box, size: [4, 3, 0.5], material: stone, at: [0, 0, 1], align: bottom}
          - {name: crate, shape: box, size: [1, 1, 1], material: wood, at: [0, 0, -1], align: bottom}
        """, "occluded")
    r = inspect_asset(a)
    crate = part(r, "crate")["visible"]
    assert crate["front"] <= 0.02
    assert crate["top"] >= 0.95 and crate["right"] >= 0.95
    assert part(r, "wall")["visible"]["front"] == 1.0
    assert not any(w.startswith("crate") for w in r["warnings"])  # well visible from the top
    # A view the report does not hold.
    assert visible_fractions(a, ["crate"], ["back"])["crate"]["back"] >= 0.95


def test_hidden_part_warning_names_the_occluder():
    a = load("""
        parts:
          - {name: shell, shape: box, size: [2, 2, 2], material: stone, align: bottom}
          - {name: gem, shape: box, size: [0.4, 0.4, 0.4], material: gold, at: [0, 1, 0]}
        """, "hidden")
    r = inspect_asset(a)
    assert all(v == 0.0 for v in part(r, "gem")["visible"].values())
    (c,) = r["contacts"]
    assert c["kind"] == "overlap" and c["overlap_volume"] == pytest.approx(0.4**3, rel=0.2)
    assert "gem is under 30% visible in every view (0% in every view); mostly hidden by shell" in r["warnings"][0]


def test_cut_volume_visibility_and_stray_cut():
    a = load("""
        parts:
          - {name: wall, shape: box, size: [4, 3, 0.5], material: stone, align: bottom}
          - {name: door, shape: box, size: [1, 2, 1], op: cut, on: wall.bottom, align: bottom}
          - {name: stray, shape: box, size: [0.5, 0.5, 0.5], op: cut, from: [wall], at: [5, 1, 0]}
        """, "doorway")
    r = inspect_asset(a)
    assert r["cut_count"] == 2
    door, stray = r["cuts"]
    assert door["name"] == "door" and door["cuts"] == ["wall"]
    assert door["removed_volume"] == pytest.approx(1 * 2 * 0.5, rel=0.02)
    assert door["visible"]["front"] >= 0.95  # a through-hole: the background shows through it
    assert door["visible"]["right"] == 0.0 and door["visible"]["top"] == 0.0
    assert stray["removed_volume"] == 0.0
    assert any(w.startswith("cut stray removes nothing") for w in r["warnings"])
    assert r["mass"]["volume"] == pytest.approx(4 * 3 * 0.5 - 1.0, rel=0.05)


def test_cut_that_removes_nothing_is_not_visible():
    a = load("""
        parts:
          - {name: post, shape: box, size: [1, 2, 1], material: wood, align: bottom}
          - {name: lop, shape: box, size: [2, 1, 2], op: cut, on: post.top, align: bottom}
        """, "lop")
    r = inspect_asset(a)
    (lop,) = r["cuts"]
    assert lop["removed_volume"] == 0.0 and set(lop["visible"].values()) == {0.0}
    assert any(w.startswith("cut lop removes nothing") for w in r["warnings"])


def test_panel_in_a_close_fitting_opening_is_fully_visible():
    """A door panel set 5 cm into its door cut on a 12 m wall: visible ~100%, and no 'hidden by' warning."""
    a = load("""
        parts:
          - {name: wall, shape: box, size: [4, 12, 2], material: plaster, align: bottom}
          - {name: door, shape: box, size: [1.0, 1.9, 0.6], op: cut, on: "wall@[0, -1, 1]", align: bottom}
          - {name: door_panel, shape: box, size: [0.9, 1.85, 0.1], material: dark_wood, at: [0, 0, 0.8], align: bottom}
        """, "tall_door")
    r = inspect_asset(a, views=("front", "iso"))
    assert part(r, "door_panel")["visible"]["front"] >= 0.95
    assert not any(w.startswith("cut door") for w in r["warnings"])


def test_cut_hidden_by_a_part_in_front():
    a = load("""
        parts:
          - {name: wall, shape: box, size: [4, 3, 1], material: stone, align: bottom}
          - {name: door, shape: box, size: [1, 2, 1], op: cut, on: wall.bottom_front, align: bottom}
          - {name: barrel, shape: cylinder, radius: 0.7, height: 2.5, material: wood, at: [0, 0, 1.6], align: bottom}
        """, "blocked")
    r = inspect_asset(a)
    (door,) = r["cuts"]
    assert door["removed_volume"] == pytest.approx(1 * 2 * 0.5, rel=0.02)  # a 0.5 m deep recess
    assert door["visible"]["front"] <= 0.05
    assert any(w.startswith("cut door is under 30% visible") and "hidden by barrel" in w for w in r["warnings"])


def test_tower_door_recess_is_visible_from_the_front(tower):
    _, r, _ = tower
    door = next(c for c in r["cuts"] if c["name"] == "door")
    assert door["removed_volume"] == pytest.approx(0.9 * 1.8 * 0.4, rel=0.03)  # recess 0.4 m deep
    assert door["visible"]["front"] >= 0.8 and door["visible"]["right"] <= 0.05
    windows = {c["name"]: c["removed_volume"] for c in r["cuts"] if c["name"].startswith("window")}
    assert len(set(windows.values())) == 1  # radial copies remove the same volume


# ----------------------------------------------------------------------------- colors and contrast


def test_contrast_pairs_and_colors():
    a = load("""
        parts:
          - {name: dark, shape: box, size: [1, 1, 1], material: paint_black, align: bottom}
          - {name: light, shape: box, size: [1, 1, 1], material: paint_white, at: [1, 0, 0], align: bottom}
          - {name: light2, shape: box, size: [1, 1, 1], material: paint_white, at: [1, 1, 0], align: bottom}
        """, "contrast")
    r = inspect_asset(a)
    pairs = {(e["a"], e["b"]): e["ratio"] for e in r["contrast"]}
    lit = {p["name"]: hex_to_rgb(p["lit_color"]) for p in r["parts"]}
    assert pairs[("dark", "light")] == pytest.approx(contrast_ratio(lit["dark"], lit["light"]), abs=0.01)
    assert pairs[("dark", "light")] > 3.0
    assert pairs[("light", "light2")] < 1.5  # same material: no warning
    assert [e["ratio"] for e in r["contrast"]] == sorted(e["ratio"] for e in r["contrast"])
    assert not any("contrast" in w for w in r["warnings"])
    shares = r["colors"]["shares"]
    assert sum(shares.values()) == pytest.approx(1.0, abs=0.01)
    assert shares["paint_white"] > shares["paint_black"]


def test_wcag_contrast_formula():
    assert contrast_ratio((1, 1, 1), (0, 0, 0)) == pytest.approx(21.0)
    assert contrast_ratio((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- silhouette, readability


def test_silhouette_grid_letters():
    a = load("""
        parts:
          - {name: base, shape: box, size: [4, 1, 2], material: stone, align: bottom}
          - {name: tower, shape: box, size: [1, 3, 1], material: wood, on: base.top}
        """, "stack")
    r = inspect_asset(a, views=("front",))
    sil = r["silhouette"]["front"]
    cell = sil["cell"]
    assert cell == 0.14  # smallest clean cell that fits 4 x 4 m in 32 x 32 cells
    assert sil["axes"] == ["x", "y"] and sil["legend"] == {"A": "base", "B": "tower"}
    grid = [row.split(" ") for row in sil["grid"]]
    assert len(grid) == sil["rows"] and all(len(row) == sil["cols"] for row in grid)
    assert sil["cols"] <= 32 and sil["rows"] <= 32

    def at(x, y):
        c = int(np.floor((x - sil["x0"]) / cell))
        r_ = int(np.argmin([abs(lab - y) for lab in sil["row_labels"]]))
        return grid[r_][c]

    assert at(0, 0.5) == "A" and at(-1.8, 0.5) == "A" and at(1.8, 0.5) == "A"
    assert at(0, 2.5) == "B" and at(0, 3.9) == "B"
    assert at(1.5, 2.5) == "." and at(-1.5, 3.5) == "."
    assert sil["x0"] <= -2.0 + 1e-9 and sil["x0"] + sil["cols"] * cell >= 2.0 - 1e-9
    assert sil["y0"] == 0.0 and sil["row_labels"][-1] == cell / 2  # ground-aligned rows, labels at centers
    widths = [row.count("A") * cell for row in grid if "B" not in row and "A" in row]
    assert all(abs(w - 4.0) <= cell for w in widths)
    tower_rows = [row.count("B") * cell for row in grid if "A" not in row and "B" in row]
    assert tower_rows and all(abs(w - 1.0) <= cell for w in tower_rows)
    text = format_report(r)
    assert "SILHOUETTE front  cell 0.14 m" in text and "A base  B tower" in text


def test_more_parts_than_letters():
    a = load("""
        parts:
          - {name: hub, shape: cylinder, radius: 3, height: 0.5, material: stone, align: bottom}
          - name: tooth
            shape: box
            size: [0.2, 0.6, 0.2]
            material: wood
            on: hub.top_front
            align: bottom
            offset: [0, 0, -0.2]
            repeat: {count: 60, axis: y, center: hub.top}
        """, "gear")
    r = inspect_asset(a, views=("top",))
    sil = r["silhouette"]["top"]
    letters = list(sil["legend"])
    assert letters[0] == "A" and sil["legend"]["A"] == "hub"
    assert letters[-1] == "*" and sil["legend"]["*"] == "other parts"
    assert len(letters) == 53 and all(type(ch) is str for ch in letters)
    assert sum(row.count("*") for row in sil["grid"]) > 0


def test_readability_loses_a_thin_spike():
    a = load("""
        parts:
          - {name: body, shape: box, size: [2, 4, 2], material: stone, align: bottom}
          - {name: spike, shape: cylinder, radius: 0.02, height: 1.5, material: metal, on: body.top}
          - {name: chimney, shape: box, size: [0.6, 0.8, 0.6], material: stone, on: body.top_left, align: bottom_left}
        """, "spiky")
    r = inspect_asset(a)
    read = r["readability"]
    assert read["view"] == "front"
    at32 = read["at_px"][32]
    assert at32["parts_lost"] == ["spike"]
    assert at32["parts_kept"] == ["body", "chimney"]
    assert at32["iou"] > 0.85
    assert read["at_px"][16]["iou"] <= at32["iou"] + 0.02
    assert any(w.startswith("at 32 px (front) these parts vanish: spike") for w in r["warnings"])
    # Same machinery at another size and view.
    side = readability(a, "right", (64,))
    assert side["at_px"][64]["parts_lost"] == ["spike"] or "spike" in side["at_px"][64]["parts_kept"]
    with pytest.raises(SightlineError):
        readability(a, "front", (2,))


def test_compare_true_scale_and_normalized():
    small = load("""
        parts:
          - {name: block, shape: box, size: [2, 1, 1], material: stone, align: bottom}
        """, "small")
    big = load("""
        parts:
          - {name: block, shape: box, size: [4, 2, 2], material: stone, align: bottom}
        """, "big")
    res = compare_silhouettes(small, big, ["front", "top"])
    assert res["front"]["iou_true_scale"] == pytest.approx(0.25, abs=0.03)  # 2 x 1 inside 4 x 2
    assert res["front"]["iou_normalized"] == pytest.approx(1.0, abs=0.03)
    assert res["top"]["iou_normalized"] == pytest.approx(1.0, abs=0.03)
    r = inspect_asset(small, views=("front",), compare=[big])
    assert r["compare"]["big"]["front"] == res["front"]
    assert "COMPARE" in format_report(r) and "big" in format_report(r)


# ----------------------------------------------------------------------------- text, diff, errors


def test_format_report_is_complete_and_stable(tower):
    a, r, _ = tower
    text = format_report(r)
    assert text == format_report(r)
    assert text == format_report(json.loads(json.dumps(r)))  # JSON round trip (int keys become strings)
    for heading in ("ASSET", "COUNTS", "SIZE", "GROUND", "MASS", "SYMMETRY", "PARTS", "CONTACTS", "FLOATING", "CUTS",
                    "COLORS", "CONTRAST", "SILHOUETTE front", "SILHOUETTE iso", "READABILITY", "COMPARE", "WARNINGS"):
        assert heading in text, heading
    lines = text.splitlines()
    table = lines[next(i for i, ln in enumerate(lines) if ln.startswith("PARTS")):
                  next(i for i, ln in enumerate(lines) if ln.startswith("CONTACTS"))]
    for p in r["parts"]:
        (line,) = [ln for ln in table if ln.startswith(f"  {p['name']} ")]
        cells = line.split()
        for value in p["center"] + p["size"]:
            assert f"{value:.2f}" in cells
        assert p["material"] in cells and (p["lit_color"] or "-") in cells
        for frac in p["visible"].values():
            assert f"{int(np.floor(frac * 100 + 0.5))}%" in cells
    for c in r["contacts"]:
        assert any(ln.split()[:3] == [c["a"], c["b"], c["kind"]] for ln in lines)
    for w in r["warnings"]:
        assert f"  - {w}" in lines
    assert inspect_asset(a) == r  # deterministic


def test_diff_reports(box):
    a, r = box
    assert diff_reports(r, r) == "CHANGES     none\n"
    assert diff_reports(json.loads(json.dumps(r)), r) == "CHANGES     none\n"
    taller = load("""
        parts:
          - {name: block, shape: box, size: [2, 4, 1], material: stone, align: bottom}
          - {name: flag, shape: box, size: [0.5, 0.3, 0.05], material: cloth_red, at: [0, 4.5, 0]}
        """, "box")
    d = diff_reports(r, inspect_asset(taller))
    assert "size.height: 3.00 -> 4.65" in d
    assert "+ part flag (box, cloth_red)" in d
    assert "parts.block.size: [2.00, 3.00, 1.00] -> [2.00, 4.00, 1.00]" in d
    assert "floating.flag: none -> gap" in d
    assert "+ warning: flag floats" in d
    assert "silhouette.front:" in d


def test_bad_inputs_are_sightline_errors(box):
    a, _ = box
    with pytest.raises(SightlineError) as info:
        inspect_asset(a, views=("front", "isometric"))
    assert "unknown view 'isometric'" in info.value.message
    with pytest.raises(SightlineError):
        inspect_asset(a, res=4)
    only_cut = load("""
        parts:
          - {name: block, shape: box, size: [1, 1, 1], material: stone, align: bottom}
        """, "lonely")
    only_cut.parts = [p for p in only_cut.parts if p.is_solid and False]
    with pytest.raises(SightlineError) as info:
        inspect_asset(only_cut)
    assert "no solid parts" in info.value.message


# ----------------------------------------------------------------------------- the 30-part fixture


def test_tower30_report_and_timing(tower):
    a, r, seconds = tower
    print(f"\ninspect tower30: {seconds:.2f} s")
    assert seconds < 15.0
    assert r["part_count"] == 30 and r["cut_count"] == 6
    assert r["ground"]["grounded"] and r["floating"] == []
    assert r["size"]["height"] == pytest.approx(8.9, abs=0.01)
    assert len(r["silhouette"]) == 4 and all(s["cols"] <= 32 and s["rows"] <= 32 for s in r["silhouette"].values())
    legend = r["silhouette"]["front"]["legend"]
    assert legend["A"] == "shaft" and legend["B"] == "roof"  # by descending visible area
    assert part(r, "lantern")["visible"]["front"] > 0.9 and part(r, "lantern_back")["visible"]["front"] < 0.05
