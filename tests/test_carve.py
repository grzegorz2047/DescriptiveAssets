"""`sightline carve`: cuts over what sticks out of the pictures, fills of what they all show."""

import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sightline import carve as C
from sightline.errors import SightlineError
from sightline.loader import load_asset, load_asset_text

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "carve"
PX = 200  # picture pixels per meter

ASSET = """\
asset: {name}
style: default
reference:
  - {{image: front.png, view: front, height: 1.0}}
parts:
  - name: body                        # the picture shows the block 1.00 m wide
    shape: box
    size: [{width}, 1.0, 0.4]
    material: iron
    at: [0, 0, 0]
    align: bottom
checks:
  - grounded
  - connected
{extra}"""


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _asset(scratch, name, width, symmetric=True):
    img = Image.new("RGBA", (PX + 80, PX + 60), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((40, 30, 40 + PX - 1, 30 + PX - 1), fill=(90, 92, 98, 255))
    img.save(scratch / "front.png")
    path = scratch / f"{name}.yaml"
    extra = "  - symmetric: {axis: x, min: 0.9}\n" if symmetric else ""
    path.write_text(ASSET.format(name=name, width=width, extra=extra), encoding="utf-8")
    return path


def test_carve_trims_a_block_wider_than_its_picture(scratch):
    path = _asset(scratch, "wide", 1.1)
    rep = C.carve_asset(path, ROOT, res=128)
    (v,) = rep["views"]
    assert v["iou_before"] < 0.93 and v["iou_after"] > 0.97
    cuts = [k for k in rep["kept"] if k["kind"] == "cut"]
    assert cuts and all(k["mirror"] for k in cuts)  # the symmetric check: one side, mirrored
    assert rep["checks"]["new_failures"] == [] and rep["symmetry"] == ["x"]
    text = rep["text"]
    assert "# the picture shows the block 1.00 m wide" in text and "op: cut, from: [body]" in text
    carved = load_asset_text(text, path, ROOT)
    assert sum(p.is_cut for p in carved.parts) == 2 * len(cuts)
    assert path.read_text(encoding="utf-8").count("op: cut") == 0  # the file itself is untouched
    out = C.format_carve(rep)
    assert "CARVE  wide" in out and "KEPT  " in out and "WROTE  nothing (dry run)" in out


def test_carve_fills_a_block_narrower_than_its_picture(scratch):
    path = _asset(scratch, "narrow", 0.9, symmetric=False)
    rep = C.carve_asset(path, ROOT, res=128, cuts=False)
    (v,) = rep["views"]
    fills = [k for k in rep["kept"] if k["kind"] == "fill"]
    assert fills and all(k["material"] in ("iron", "metal") for k in fills)  # by the picture's class there
    assert v["iou_after"] > v["iou_before"] + 0.05
    assert not any(k["mirror"] for k in rep["kept"])


def test_pieces_are_split_until_they_enclose_nothing():
    """An outside region that surrounds the outline (a ring) comes back as hole-free pieces covering it; the
    halves overlap along the split (two cuts that only touched would leave a zero-thickness sheet there)."""
    H = W = 64
    rows, cols = np.mgrid[0:H, 0:W]

    class Plane:
        world = np.stack([cols * 0.01, -rows * 0.01, np.zeros((H, W))], axis=-1)

    ring = np.zeros((H, W), dtype=bool)
    ring[4:60, 4:60] = True
    ring[20:44, 20:44] = False
    pieces = C._pieces(Plane, ring, ring.astype(np.float32), 1.0)
    assert len(pieces) >= 2
    union = np.zeros_like(ring)
    overlap = 0
    for comp, pts, w in pieces:
        assert not C._has_hole(comp) and len(pts) <= C.MAX_POINTS
        overlap += int((union & comp).sum())
        union |= comp
    assert np.array_equal(union, ring)
    assert overlap > 0  # the split seams are covered twice


def test_carve_rejects_assets_it_cannot_carve(scratch):
    path = _asset(scratch, "top_only", 1.1)
    path.write_text(path.read_text(encoding="utf-8").replace("view: front", "view: top"), encoding="utf-8")
    with pytest.raises(SightlineError, match="front, back, left or right reference"):
        C.carve_asset(path, ROOT, res=128)
    bare = scratch / "bare.yaml"
    bare.write_text("asset: bare\nstyle: default\nparts:\n  - {name: a, shape: box, size: [1, 1, 1], material: iron}\n",
                    encoding="utf-8")
    with pytest.raises(SightlineError, match="no reference picture"):
        C.carve_asset(bare, ROOT, res=128)
    with pytest.raises(SightlineError, match="res must be"):
        C.carve_asset(_asset(scratch, "wide", 1.1), ROOT, res=8)


def test_cli_carve_writes_a_copy(scratch):
    path = _asset(scratch, "wide", 1.1)
    out = scratch / "wide_carved.yaml"
    cmd = [sys.executable, "-m", "sightline", "carve", str(path), "--res", "128", "--out", str(out), "--root", str(ROOT)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "CHECKS  3 run: none that passed before fails now" in proc.stdout and "WROTE" in proc.stdout
    carved = load_asset(out, ROOT)
    assert carved.name == "wide_carved" and any(p.is_cut for p in carved.parts)
    assert "carve: start" in proc.stderr


def test_carve_holdout_back_picture_shapes_nothing_but_improves(scratch):
    path = _asset(scratch, "wide_held", 1.1)
    text = path.read_text(encoding="utf-8").replace(
        "  - {image: front.png, view: front, height: 1.0}",
        "  - {image: front.png, view: front, height: 1.0}\n  - {image: back.png, view: back, height: 1.0}")
    path.write_text(text, encoding="utf-8")
    shutil.copy(scratch / "front.png", scratch / "back.png")
    rep = C.carve_asset(path, ROOT, res=128, checks=False, holdout=[1])
    front, back = rep["views"]
    assert back["holdout"] and rep["holdout"] == [1] and rep["kept"]
    assert back["iou_after"] > back["iou_before"] + 0.03  # cut to the front picture, the back follows
    with pytest.raises(SightlineError, match="every reference is held out"):
        C.carve_asset(_asset(scratch, "only_front", 1.1), ROOT, res=128, checks=False, holdout=[0])
