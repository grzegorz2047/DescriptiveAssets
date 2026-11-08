"""`sightline score` (the scoreboard and the concept-vs-model picture) and `sightline paint` (the colour pass)."""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest
from PIL import Image, ImageDraw

from sightline import paint as P, score as S
from sightline.errors import SightlineError
from sightline.loader import load_asset, load_asset_text

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "score_paint"
PX = 100  # picture pixels per meter

# a 2 m frame with a bright plate on its left half: the model's plate is iron, so the picture's bright
# class is wrong there until a paint pass colours it
ASSET = """\
asset: {name}
style: default
reference:
  - {{image: front.png, view: front, height: 2.0}}
  - {{image: back.png, view: back, height: 2.0}}
parts:
  - name: frame                       # dark frame
    shape: box
    size: [2.0, 2.0, 0.2]
    material: iron
    at: [0, 0, 0]
    align: bottom
  - name: plate_l                     # the picture shows this plate bright; its material is wrong on purpose
    shape: box
    size: [0.8, 1.2, 0.02]
    material: iron
    at: [-0.5, 0.3, 0.1]
    align: bottom
checks:
  - grounded
"""


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _picture(path, plate_on_left=True):
    img = Image.new("RGBA", (2 * PX + 60, 2 * PX + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 20, 30 + 2 * PX - 1, 20 + 2 * PX - 1), fill=(40, 40, 44, 255))
    x0 = 30 + int(0.1 * PX) if plate_on_left else 30 + int(1.1 * PX)
    d.rectangle((x0, 20 + int(0.5 * PX), x0 + int(0.8 * PX), 20 + int(1.7 * PX)), fill=(200, 202, 206, 255))
    img.save(path)


def _asset(scratch, name="panel"):
    _picture(scratch / "front.png", True)
    _picture(scratch / "back.png", False)  # seen from behind the plate is on the picture's right
    path = scratch / f"{name}.yaml"
    path.write_text(ASSET.format(name=name), encoding="utf-8")
    return path


def test_score_lists_every_reference_and_writes_the_picture(scratch):
    path = _asset(scratch)
    asset = load_asset(path, ROOT)
    rep = S.score_asset(asset, res=128)
    assert [r["view"] for r in rep["views"]] == ["front", "back"]
    front = rep["views"][0]
    assert front["iou"] > 0.9 and 0 <= front["outline_picture"] <= 1 and front["classes"] < 0.9
    assert front["ceiling"]["view"] == "back" and front["ceiling"]["pair_mean_max"] > 0.9
    assert rep["counts"] == {"solids": 2, "cuts": 0, "paints": 0}
    out = S.format_score(rep, S.score_asset(asset, res=128, ceiling=False))
    assert "SCORE  panel" in out and "(+0.000)" in out and "with back" in out
    png = S.write_score_png([("before", asset), ("now", asset)], scratch / "score.png", res=96)
    img = Image.open(png)
    assert img.size[0] > 300 and img.size[1] > 300  # 3 rows (concept, before, now) x 2 views
    with pytest.raises(SightlineError, match="res must be"):
        S.score_asset(asset, res=8)


def test_paint_colours_the_plate_from_the_pictures_in_order(scratch):
    path = _asset(scratch)
    rep = P.paint_asset(path, ROOT, res=128)
    assert rep["order"] == [1, 0]  # back before front (the default order: sides, back, front)
    assert rep["paints"] >= 1 and all(s["view"] in ("front", "back") for s in rep["steps"])
    front = rep["views"][0]
    assert front["classes_after"] > front["classes_before"] + 0.1
    painted = load_asset_text(rep["text"], path, ROOT)
    assert any(p.is_paint and p.name.startswith("paint") for p in painted.parts)
    # the second run strips what the first wrote and ends with the same count
    path.write_text(rep["text"], encoding="utf-8")
    again = P.paint_asset(path, ROOT, res=128, strip=True)
    assert again["stripped"] == rep["paints"] and again["paints"] == rep["paints"]
    out = P.format_paint(again)
    assert "STRIP  " in out and "STEPS  " in out and "WROTE  nothing (dry run)" in out
    with pytest.raises(SightlineError, match="no reference shows the left view"):
        P.paint_asset(path, ROOT, res=128, order=["left"])


def test_strip_paints_removes_only_the_prefixed_entries():
    text = ("asset: a\nstyle: default\nparts:\n  - name: body\n    shape: box\n    size: [1, 1, 1]\n    material: iron\n"
            "  # --- paints where the model's class differs from the front picture\n"
            "  - name: paint_f1\n    shape: box\n    size: [0.2, 0.2, 0.2]\n    op: paint\n    material: metal\n"
            "    at: [0, 0.5, 0.5]\n"
            "  - name: decal\n    shape: box\n    size: [0.1, 0.1, 0.1]\n    op: paint\n    material: metal\n    at: [0, 0.2, 0.5]\n"
            "  - {name: paint_f2, shape: sphere, radius: 0.1, op: paint, material: metal, at: [0, 0.8, 0.5]}\n"
            "checks:\n  - grounded\n")
    out, n = P.strip_paints(text, "paint")
    assert n == 2 and "paint_f1" not in out and "paint_f2" not in out and "# --- paints" not in out
    assert "  - name: decal\n    shape: box" in out and out.endswith("checks:\n  - grounded\n")
    load_asset_text(out, pathlib.Path("a.yaml"), ROOT)
    everything, n = P.strip_paints(text, None)
    assert n == 3 and "op: paint" not in everything


def test_cli_score_and_paint(scratch):
    path = _asset(scratch)
    run = lambda *a: subprocess.run([sys.executable, "-m", "sightline", *a, "--root", str(ROOT)], capture_output=True,
                                    text=True, encoding="utf-8", cwd=ROOT, timeout=600)
    proc = run("score", str(path), "--res", "128", "--out", str(scratch / "score.png"), "--no-ceiling")
    assert proc.returncode == 0, proc.stderr
    assert "SCORE  panel" in proc.stdout and "IMAGE  " in proc.stdout and (scratch / "score.png").is_file()
    out = scratch / "panel_painted.yaml"
    proc = run("paint", str(path), "--res", "128", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert "PAINT  panel" in proc.stdout and "WROTE" in proc.stdout
    assert load_asset(out, ROOT).name == "panel_painted"
