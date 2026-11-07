"""op: paint parts (material zones without geometry) and the light that turns with the camera."""

import pathlib
import re
import shutil
import textwrap

import numpy as np
import pytest

from sightline import export, inspect as ins, render
from sightline.camera import make_cameras
from sightline.errors import SightlineError
from sightline.loader import load_asset

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "paint"

CRATE = """\
asset: {name}
style: default
parts:
  - name: body_l
    shape: box
    size: [0.6, 1.0, 1.0]
    material: iron
    at: [-0.4, 0, 0]
    align: bottom
    mirror: x
  - name: lid
    shape: box
    size: [1.6, 0.1, 1.1]
    material: iron
    at: [0, 1.0, 0]
    align: bottom
"""
BADGE = """\
  - name: badge_l
    shape: cylinder
    radius: 0.2
    height: 0.2
    rotate: [90, 0, 0]
    op: paint
    from: [body_l]
    material: metal
    at: [-0.4, 0.5, 0.5]
    mirror: x
"""


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _asset(scratch, name, extra=""):
    path = scratch / f"{name}.yaml"
    path.write_text(CRATE.format(name=name) + extra, encoding="utf-8")
    return load_asset(path, ROOT)


def _front(asset, light="world"):
    return render.render_raw(asset, ["front"], res=128, scale_figure=False, ground=False, light=light)["front"]


def test_paint_colours_the_surface_and_adds_no_geometry(scratch):
    plain = _asset(scratch, "crate")
    painted = _asset(scratch, "crate_badge", BADGE)
    scene = painted.scene()
    assert len(scene.paints) == 2 and not any(scene.parts[i].is_paint for i in scene.solid_indices())
    for a, b in zip(plain.scene().bounds(include_helpers=False), scene.bounds(include_helpers=False)):
        assert np.allclose(a, b)
    r0, r1 = _front(plain), _front(painted)
    assert np.array_equal(r0.coverage, r1.coverage)
    # the badge centre on each body turns steel-bright; the lid above it keeps its iron
    cam = r1.camera
    for x in (-0.4, 0.4):
        (c, r), _ = cam.project(np.array([[x, 0.5, 0.5]]))[0][0].astype(int), None
        assert r1.rgb[r, c].mean() > r0.rgb[r, c].mean() + 40
    c, r = cam.project(np.array([[0.0, 1.05, 0.55]]))[0][0].astype(int)
    assert np.array_equal(r1.rgb[r, c], r0.rgb[r, c])


def test_paint_reports_counts_colours_and_visibility(scratch):
    asset = _asset(scratch, "crate_report", BADGE)
    rep = ins.inspect_asset(asset)
    assert rep["paint_count"] == 2 and rep["part_count"] == 3 and rep["cut_count"] == 0
    assert rep["colors"]["shares"].get("metal", 0) > 0.02
    assert "2 paints" in ins.format_report(rep)
    vis = ins.visible_fractions(asset, ["badge_l", "badge_r"], ["front", "back"])
    assert vis["badge_l"]["front"] == pytest.approx(1.0) and vis["badge_l"]["back"] == 0.0


def test_paint_from_names_its_targets_and_mirror_swaps_them(scratch):
    asset = _asset(scratch, "crate_from", BADGE)
    scene = asset.scene()
    names = {scene.parts[ci].name: sorted(scene.parts[j].name for j in t) for ci, t in scene.paints}
    assert names == {"badge_l": ["body_l"], "badge_r": ["body_r"]}
    # without from: every earlier solid, so the lid is painted too where the shape reaches it
    lid = _asset(scratch, "crate_all", "  - name: stripe\n    shape: box\n    size: [2.0, 0.2, 2.0]\n    op: paint\n"
                                       "    material: metal\n    at: [0, 1.05, 0]\n")
    targets = {lid.scene().parts[j].name for j in lid.scene().paints[0][1]}
    assert targets == {"body_l", "body_r", "lid"}


@pytest.mark.parametrize("extra, message", [
    ("    op: paint\n    from: [body_l]\n", "solid part has no material"),
    ("    op: paint\n    material: metal\n    blend: 0.02\n", "'blend' does not apply to a paint"),
    ("    material: metal\n    from: [body_l]\n", "'from' only applies to op: cut or op: paint"),
    ("    op: paint\n    material: metal\n    from: [lid_x]\n", "unknown part 'lid_x'"),
    ("    op: tint\n    material: metal\n", "unknown op 'tint'"),
])
def test_paint_validation(scratch, extra, message):
    text = "  - name: decal\n    shape: box\n    size: [0.2, 0.2, 0.2]\n    at: [0, 0.5, 0.5]\n" + extra
    with pytest.raises(SightlineError, match=re.escape(message)):
        _asset(scratch, "crate_bad", text)


def test_paint_exports_as_its_own_material(scratch):
    asset = _asset(scratch, "crate_glb", BADGE)
    meshes = export.build_meshes(asset, max_cells=64)
    body = next(m for m in meshes if m.name == "body_l")
    names = {asset.scene().parts[i].name for i in np.unique(body.tri_part)}
    assert names == {"body_l", "badge_l"}
    data = export.glb_bytes(asset, meshes)
    assert export.validate_glb(data) == []
    doc, _ = export._split_glb(data)
    assert {m["name"] for m in doc["materials"]} == {"iron", "metal"}


def test_follow_light_keeps_the_front_and_turns_with_the_camera():
    d = np.array([0.45, -1.0, -0.6])
    cams = make_cameras(["front", "back", "right", "left"], np.array([-1.0, 0, -1]), np.array([1.0, 2, 1]), res=64)
    assert np.allclose(cams["front"].follow_light(d), d)
    assert np.allclose(cams["back"].follow_light(d), [-0.45, -1.0, 0.6])
    for v in ("back", "right", "left"):  # the light falls on every view as on the front one
        cam = cams[v]
        turned = cam.follow_light(d)
        assert np.allclose([turned @ cam.right, turned @ cam.up, -(turned @ cam.forward)], d)


def test_view_light_lights_the_back_like_the_front(scratch):
    asset = _asset(scratch, "crate_light")
    front = _front(asset)
    back_world = render.render_raw(asset, ["back"], res=128, scale_figure=False, ground=False)["back"]
    back_view = render.render_raw(asset, ["back"], res=128, scale_figure=False, ground=False, light="view")["back"]

    def body_mean(r):
        return r.rgb[r.part == asset.part_index("body_l")].mean()

    assert body_mean(back_view) == pytest.approx(body_mean(front), abs=3)
    assert body_mean(back_world) < body_mean(front) - 10
    with pytest.raises(SightlineError, match="light must be 'world' or 'view'"):
        render.render_raw(asset, ["back"], res=64, light="sun")
