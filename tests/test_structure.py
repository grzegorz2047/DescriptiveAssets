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
