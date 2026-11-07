"""Picture structure: edges, regions, classes, relief guess, model comparison and the CLI."""

import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sightline import structure as st
from sightline.errors import SightlineError
from sightline.loader import load_asset

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "structure"

# A front "concept" of a 2 m tall panel: two bright plates left and right of a dark frame strip,
# a dark lower band and one blue light on the right plate. 100 px = 1 m.
W, H = 260, 240
FRAME = (40, 40, 44)
PLATE = (200, 202, 206)
BLUE = (60, 150, 250)


def draw_panel() -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 20, 229, 219), fill=FRAME + (255,))      # 2.0 x 2.0 m frame
    d.rectangle((40, 30, 120, 150), fill=PLATE + (255,))     # left plate 0.81 x 1.21 m
    d.rectangle((139, 30, 219, 150), fill=PLATE + (255,))    # right plate, mirrored
    d.ellipse((170, 80, 186, 96), fill=BLUE + (255,))        # light on the right plate
    return img


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


@pytest.fixture
def panel(scratch):
    path = scratch / "panel.png"
    draw_panel().save(path)
    return path


def test_regions_find_plates_frame_and_light(panel):
    rep = st.analyze_structure(panel, height=2.0)
    regs = rep["regions"]
    bright = [r for r in regs if r["class"] == "bright" and r["share"] > 0.05]
    assert len(bright) == 2
    left, right = sorted(bright, key=lambda r: r["center"][0])
    # the left plate spans px 40..120 x 30..150 of a frame 30..229 x 20..219 (1 px = 0.01 m)
    assert left["x"][0] == pytest.approx(-0.90, abs=0.03) and left["x"][1] == pytest.approx(-0.09, abs=0.03)
    assert left["y"][0] == pytest.approx(0.70, abs=0.03) and left["y"][1] == pytest.approx(1.90, abs=0.03)
    assert left["pair"] == right["id"] and right["pair"] == left["id"]
    assert left["relief"] == "raised plate"
    dark = [r for r in regs if r["class"] == "dark"]
    assert dark and dark[0]["relief"].startswith("recessed")
    acc = [r for r in regs if r["class"].startswith("accent")]
    assert len(acc) == 1 and acc[0]["class"] == "accent blue"
    assert acc[0]["center"][0] == pytest.approx(0.48, abs=0.03) and acc[0]["center"][1] == pytest.approx(1.31, abs=0.03)
    text = st.format_structure(rep)
    assert "ACCENTS" in text and "raised plate" in text


def test_edges_lie_on_the_plate_borders(panel):
    from sightline.reference import _analyze

    r = _analyze(panel, "front", 2.0, None, None, None)
    edge, mag, facts = st.edges(r.rgb, r.mask)
    ys, xs = np.nonzero(edge)
    assert facts["share"] > 0.01
    # every edge pixel lies within 2 px of a plate or light border, none in a flat interior
    plate_border = np.zeros(edge.shape, bool)
    for x0, y0, x1, y1 in ((40, 30, 120, 150), (139, 30, 219, 150)):
        plate_border[y0 - 2:y1 + 3, x0 - 2:x1 + 3] = True
        plate_border[y0 + 3:y1 - 2, x0 + 3:x1 - 2] = False
    plate_border[78:99, 168:189] = True
    assert plate_border[ys, xs].mean() > 0.95


def test_shading_does_not_split_a_plate(scratch):
    """A plate with a smooth top-to-bottom gradient stays one region (shading is not a border)."""
    img = draw_panel()
    a = np.asarray(img).astype(np.float64)
    for y in range(30, 151):
        k = 0.55 + 0.45 * (150 - y) / 120.0  # lighter at the top
        a[y, 40:121, :3] = np.array(PLATE) * k
    path = scratch / "shaded.png"
    Image.fromarray(a.astype(np.uint8)).save(path)
    rep = st.analyze_structure(path, height=2.0)
    left = [r for r in rep["regions"] if r["center"][0] < -0.3 and r["share"] > 0.05]
    assert len(left) == 1, [(r["id"], r["class"], r["share"]) for r in rep["regions"][:6]]
    assert "lighter toward the top" in left[0]["shading"]


def _write_asset(scratch, name, plates_material="metal", light=True):
    (scratch / "refs").mkdir(exist_ok=True)
    draw_panel().save(scratch / "refs" / "panel.png")
    text = textwrap.dedent(f"""\
        asset: {name}
        style: default
        reference: {{image: refs/panel.png, view: front, height: 2.0}}
        parts:
          - name: frame
            shape: box
            size: [2.0, 2.0, 0.2]
            material: iron
            at: [0, 0, 0]
            align: bottom
          - name: plate_l
            shape: box
            size: [0.81, 1.21, 0.06]
            material: {plates_material}
            at: [-0.495, 1.3, 0.12]
            mirror: x
        """)
    if light:
        text += ("  - name: light\n    shape: cylinder\n    radius: 0.08\n    height: 0.04\n    rotate: [90, 0, 0]\n"
                 "    material: glow_blue\n    at: [0.48, 1.31, 0.16]\n")
    path = scratch / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return load_asset(path, ROOT)


def test_compare_agrees_with_a_matching_model(scratch):
    rep = st.compare_structure(_write_asset(scratch, "panel_ok"), res=256)
    assert rep["class_agreement"] > 0.85
    assert rep["part_boundaries_on_edges"] > 0.6
    bad = [r for r in rep["regions"] if r.get("verdict") and not r["verdict"].startswith("ok")]
    assert not bad, [(r["id"], r["verdict"]) for r in bad]


def test_compare_names_wrong_materials_and_a_missing_light(scratch):
    rep = st.compare_structure(_write_asset(scratch, "panel_bad", plates_material="iron", light=False), res=256)
    assert rep["class_agreement"] < 0.7
    verdicts = [r["verdict"] for r in rep["regions"] if r.get("verdict")]
    assert any(v.startswith("picture is bright here, the model dark") and "plate_" in v for v in verdicts)
    assert any(v.startswith("light or decal missing") for v in verdicts)
    text = st.format_compare(rep)
    assert "FINDINGS" in text and "use a bright part" in text


def test_compare_lists_a_part_the_picture_divides(scratch):
    """One wide plate where the picture has two: SPLITS names it."""
    asset = _write_asset(scratch, "panel_one", light=False)
    text = (scratch / "panel_one.yaml").read_text(encoding="utf-8")
    text = text.replace("size: [0.81, 1.21, 0.06]", "size: [1.8, 1.21, 0.06]").replace("at: [-0.495, 1.3, 0.12]\n    mirror: x", "at: [0, 1.3, 0.12]")
    (scratch / "panel_one.yaml").write_text(text, encoding="utf-8")
    rep = st.compare_structure(load_asset(scratch / "panel_one.yaml", ROOT), res=256)
    assert any(s["part"] == "plate_l" and len(s["regions"]) >= 2 for s in rep["splits"]), rep["splits"]
    del asset


def _run(*args):
    return subprocess.run([sys.executable, "-m", "sightline", *args, "--root", str(ROOT)], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def test_cli_structure_image_and_asset(scratch, panel):
    proc = _run("structure", str(panel), "--height", "2.0", "--out", str(scratch / "pic.png"))
    assert proc.returncode == 0, proc.stderr
    assert "STRUCTURE" in proc.stdout and (scratch / "pic.png").is_file()
    _write_asset(scratch, "panel_cli")
    proc = _run("structure", str(scratch / "panel_cli.yaml"), "--res", "128", "--out", str(scratch))
    assert proc.returncode == 0, proc.stderr
    assert "DIVISIONS" in proc.stdout and (scratch / "structure_0.png").is_file()
    proc = _run("structure", str(scratch / "panel_cli.yaml"), "--res", "5")
    assert proc.returncode == 2 and "res must be" in proc.stderr


def test_trace_regions_writes_mirrored_extrude_plates(scratch):
    asset = _write_asset(scratch, "panel_trace")
    rep = st.compare_structure(asset, res=256)
    plates = [r["id"] for r in rep["regions"] if r["class"] == "bright" and r["share"] > 0.05]
    result = st.trace_regions(asset, plates, res=256, depth=0.03)
    assert len(result["parts"]) == 1  # the two plates are a left/right pair: one part with mirror: x
    part = result["parts"][0]
    assert part["mirror"] and part["at"][0] < 0
    pts = np.array(part["points"])
    w, h = pts.max(axis=0) - pts.min(axis=0)
    assert w == pytest.approx(0.81, abs=0.04) and h == pytest.approx(1.21, abs=0.04)
    assert part["at"][1] == pytest.approx(1.3, abs=0.03)
    assert part["at"][2] == pytest.approx(0.15 + 0.015 - 0.01, abs=0.02)  # on the plate already there, sunk 1 cm
    text = (scratch / "panel_trace.yaml").read_text(encoding="utf-8")
    traced = text.replace("checks:", "") + result["yaml"]
    path = scratch / "panel_trace.yaml"
    path.write_text(traced, encoding="utf-8")
    loaded = load_asset(path, ROOT)
    names = {p.name for p in loaded.scene().parts}
    assert {part["name"], part["name"][:-2] + "_r"} <= names
    with pytest.raises(Exception, match="no region R999"):
        st.trace_regions(asset, ["R999"], res=256)


def test_area_classes_and_region_borders_are_reported(scratch):
    rep = st.compare_structure(_write_asset(scratch, "panel_metrics"), res=256)
    assert rep["class_agreement_areas"] >= rep["class_agreement"]
    assert rep["region_borders_on_part_boundaries"] > 0.6 and rep["part_boundaries_on_borders"] > 0.6
    text = st.format_compare(rep)
    assert "of the areas" in text and "region borders" in text and "DETAIL" in text


def test_cli_trace(scratch):
    _write_asset(scratch, "panel_cli_trace")
    proc = _run("structure", str(scratch / "panel_cli_trace.yaml"), "--res", "256", "--trace", "R1,R2,R3", "--depth", "0.02")
    assert proc.returncode == 0, proc.stderr
    assert "shape: extrude" in proc.stdout and "depth: 0.02" in proc.stdout


def test_trace_skips_regions_on_curved_surfaces(scratch):
    """max_relief: a flat plate is not traced onto a surface whose depth varies more than that."""
    asset = _write_asset(scratch, "panel_relief")
    text = (scratch / "panel_relief.yaml").read_text(encoding="utf-8")
    text = text.replace("shape: box\n    size: [0.81, 1.21, 0.06]", "shape: box\n    size: [0.81, 1.21, 0.06]\n    rotate: [40, 0, 0]")
    (scratch / "panel_relief.yaml").write_text(text, encoding="utf-8")
    asset = load_asset(scratch / "panel_relief.yaml", ROOT)
    rep = st.compare_structure(asset, res=256)
    plates = [r["id"] for r in rep["regions"] if r["class"] == "bright" and r["share"] > 0.05]
    flat = st.trace_regions(asset, plates, res=256, depth=0.008)
    picky = st.trace_regions(asset, plates, res=256, depth=0.008, max_relief=0.05)
    assert flat["parts"] and not picky["parts"]
    assert picky["skipped"] and all(relief > 0.05 for _, relief in picky["skipped"])


def _write_view_asset(scratch, name, view):
    """The panel picture as a back or right reference: plates behind (back) or on +x (right) of a wall."""
    (scratch / "refs").mkdir(exist_ok=True)
    draw_panel().save(scratch / "refs" / "panel.png")
    if view == "back":
        frame, plate = "[2.0, 2.0, 0.2]", "size: [0.81, 1.21, 0.06]\n    at: [-0.495, 1.3, -0.12]\n    mirror: x"
    else:
        frame, plate = "[0.2, 2.0, 2.0]", "size: [0.06, 1.21, 0.81]\n    at: [0.12, 1.3, -0.495]\n    mirror: z"
    text = (f"asset: {name}\nstyle: default\nreference: {{image: refs/panel.png, view: {view}, height: 2.0}}\n"
            f"parts:\n  - name: frame\n    shape: box\n    size: {frame}\n    material: iron\n    at: [0, 0, 0]\n"
            f"    align: bottom\n  - name: plate_a\n    shape: box\n    {plate}\n    material: metal\n")
    path = scratch / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return load_asset(path, ROOT)


@pytest.mark.parametrize("view", ["back", "right"])
def test_trace_from_back_and_side_pictures(scratch, view):
    """Parts traced from a back or side picture face that camera and sit on the model's surface there."""
    asset = _write_view_asset(scratch, f"panel_{view}", view)
    rep = st.compare_structure(asset, res=256)
    plates = [r["id"] for r in rep["regions"] if r["class"] == "bright" and r["share"] > 0.05]
    result = st.trace_regions(asset, plates, res=256, depth=0.03)
    assert result["parts"] and all(part["mirror"] for part in result["parts"])
    for part in result["parts"]:
        assert part["name"].startswith(f"plate_{view}_")
        if view == "back":  # one mirrored pair, on the back plates (z -0.15), sunk 1 cm
            assert part["rotate"] == [0, 180.0, 0] and part["at"][0] < 0
            assert part["at"][2] == pytest.approx(-(0.15 + 0.015 - 0.01), abs=0.02)
        else:  # the camera's side (+x) with mirror: x for the other
            assert part["rotate"] == [0, 90.0, 0] and part["at"][0] == pytest.approx(0.155, abs=0.02)
    assert len(result["parts"]) == (1 if view == "back" else 2)
    path = scratch / f"panel_{view}.yaml"
    path.write_text(path.read_text(encoding="utf-8") + result["yaml"], encoding="utf-8")
    loaded = load_asset(path, ROOT)
    traced = [p for p in loaded.scene().parts if p.name.startswith(f"plate_{view}_")]
    lo, hi = traced[0].world_aabb()
    pts = np.array(result["parts"][0]["points"])
    w, h = pts.max(axis=0) - pts.min(axis=0)
    assert (hi - lo)[1] == pytest.approx(h, abs=0.01) and w == pytest.approx(0.81, abs=0.04)


def test_trace_conform_bends_the_plate_along_the_surface(scratch):
    """--conform: a side outline that follows a tilted surface instead of a flat plate in front of it."""
    _write_asset(scratch, "panel_conform")
    text = (scratch / "panel_conform.yaml").read_text(encoding="utf-8")
    text = text.replace("shape: box\n    size: [0.81, 1.21, 0.06]", "shape: box\n    size: [0.81, 1.21, 0.06]\n    rotate: [20, 0, 0]")
    (scratch / "panel_conform.yaml").write_text(text, encoding="utf-8")
    asset = load_asset(scratch / "panel_conform.yaml", ROOT)
    rep = st.compare_structure(asset, res=256)
    plates = [r["id"] for r in rep["regions"] if r["class"] == "bright" and r["share"] > 0.05]
    flat = st.trace_regions(asset, plates, res=256, depth=0.02)
    bent = st.trace_regions(asset, plates, res=256, depth=0.02, conform=True)
    assert flat["parts"][0]["side"] is None and "depth: 0.02" in flat["yaml"]
    side = np.array(bent["parts"][0]["side"])
    assert "side: [" in bent["yaml"]
    top, bottom = side[side[:, 1] > 0.4], side[side[:, 1] < -0.4]
    assert top[:, 0].max() - bottom[:, 0].max() > 0.15  # the tilted plate leans forward at the top
    assert side[:, 0].max() - side[:, 0].min() > 0.15 + 0.02
    path = scratch / "panel_conform.yaml"
    path.write_text(path.read_text(encoding="utf-8") + bent["yaml"], encoding="utf-8")
    load_asset(path, ROOT)


def test_trace_from_a_side_picture_keeps_front_and_back_apart(scratch):
    """Right view: the picture's left is the model's front (+z). One plate drawn there lands at +z, with its
    outline the right way round (a regression: side and back traces were mirrored along the image)."""
    (scratch / "refs").mkdir(exist_ok=True)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 20, 229, 219), fill=FRAME + (255,))
    d.polygon([(40, 30), (120, 30), (120, 150), (40, 90)], fill=PLATE + (255,))  # wide at the top, toward +z
    img.save(scratch / "refs" / "side.png")
    text = ("asset: wall_side\nstyle: default\nreference: {image: refs/side.png, view: right, height: 2.0}\n"
            "parts:\n  - name: frame\n    shape: box\n    size: [0.2, 2.0, 2.0]\n    material: iron\n"
            "    at: [0, 0, 0]\n    align: bottom\n")
    (scratch / "wall_side.yaml").write_text(text, encoding="utf-8")
    asset = load_asset(scratch / "wall_side.yaml", ROOT)
    rep = st.compare_structure(asset, res=256)
    bright = [r["id"] for r in rep["regions"] if r["class"] == "bright" and r["share"] > 0.05]
    result = st.trace_regions(asset, bright, res=256, depth=0.03)
    (part,) = result["parts"]
    assert part["rotate"] == [0, 90.0, 0] and part["mirror"]
    assert part["at"][2] == pytest.approx(0.495, abs=0.03)  # the plate's middle, 0.1..0.9 m in front
    pts = np.array(part["points"])  # local x = camera right = -z: the long edge is at the back (+x local)
    top = pts[pts[:, 1] > 0.5]
    assert top[:, 0].min() < -0.3 and top[:, 0].max() > 0.3
    low = pts[pts[:, 1] < -0.4]
    assert low[:, 0].min() > 0.2  # only the back part of the plate reaches down


def test_trace_fix_paints_only_where_the_model_differs(scratch):
    """--trace fix --paint: iron plates where the picture shows bright ones become painted metal; the
    outline and the size stay, and the classes then agree."""
    asset = _write_asset(scratch, "panel_fix", plates_material="iron")
    before = st.compare_structure(asset, res=256)
    result = st.trace_regions(asset, ["fix"], res=256, paint=True, conform=True)
    parts = result["parts"]
    assert parts and all(p["op"] == "paint" and not p["mirror"] for p in parts)
    assert {p["material"] for p in parts if p["region"] in ("F1", "F2")} == {"metal"}
    assert any("plate_l" in p["from"] or "plate_r" in p["from"] for p in parts)
    assert all(p["name"].startswith("paint_f") for p in parts)
    path = scratch / "panel_fix.yaml"
    path.write_text(path.read_text(encoding="utf-8") + result["yaml"], encoding="utf-8")
    fixed = load_asset(path, ROOT)
    after = st.compare_structure(fixed, res=256)
    assert after["iou"] == before["iou"]
    assert before["class_agreement_areas"] < 0.75 < 0.95 < after["class_agreement_areas"]
    with pytest.raises(SightlineError, match="no region R999"):
        st.trace_regions(asset, ["R999"], res=256)
