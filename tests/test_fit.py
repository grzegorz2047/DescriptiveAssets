"""`sightline fit` (part numbers nudged toward the reference pictures) and the picture-agreement ceiling."""

import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sightline import fit as F, reference as R
from sightline.errors import SightlineError
from sightline.loader import load_asset

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "fit"
PX = 200  # picture pixels per meter

ASSET = """\
asset: {name}
style: default
reference:
  - {{image: front.png, view: front, height: 1.0}}{extra_ref}
parts:
  - name: body                        # the block the picture shows 1.00 m wide
    shape: box
    size: [0.94, 1.0, 0.4]
    material: iron
    at: [0, 0, 0]
    align: bottom
  - {{name: knob, shape: sphere, radius: 0.05, material: metal, at: [0, 1.0, 0]}}
checks:
  - grounded
  - connected
"""


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _picture(path, width_m=1.0, height_m=1.0, notch=False):
    """A block width_m x height_m standing on the picture's bottom, on a transparent background."""
    w, h = int(round(width_m * PX)), int(round(height_m * PX))
    img = Image.new("RGBA", (w + 80, h + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0, y1 = 40, h + 30
    d.rectangle((x0, y1 - h, x0 + w - 1, y1 - 1), fill=(90, 92, 98, 255))
    if notch:  # a bite out of the top left corner: the mirror image differs
        d.rectangle((x0, y1 - h, x0 + w // 3, y1 - h + h // 3), fill=(0, 0, 0, 0))
    img.save(path)


def _asset(scratch, name="block", extra_ref=""):
    _picture(scratch / "front.png")
    path = scratch / f"{name}.yaml"
    path.write_text(ASSET.format(name=name, extra_ref=extra_ref), encoding="utf-8")
    return path


def test_fit_widens_the_block_to_its_picture_and_keeps_the_file_layout(scratch):
    path = _asset(scratch)
    rep = F.fit_asset(path, ROOT, res=128, steps=(0.008, 0.004), max_move=0.05, checks=False)
    (v,) = rep["views"]
    assert v["iou_after"] > v["iou_before"] + 0.02 and v["iou_after"] > 0.97
    assert abs(v["iou_fitted"] - v["iou_after"]) < 2e-3  # the windowed renders measured what match measures
    widths = [c for c in rep["changes"] if c["part"] == "body" and c["param"] == "size[0]"]
    assert widths and 0.98 <= widths[0]["to"] <= 1.02
    text = rep["text"]
    assert "# the block the picture shows 1.00 m wide" in text  # comments and layout stay
    assert re.search(r"size: \[(0\.9[89]\d*|1\.0[0-2]?\d*), 1\.0, 0\.4\]", text)  # only the width is rewritten
    assert path.read_text(encoding="utf-8").count("size: [0.94, 1.0, 0.4]") == 1  # the file itself is untouched
    assert "body" in [g["part"] for g in rep["gains"]]
    out = F.format_fit(rep)
    assert "FIT  block" in out and re.search(r"  body: .*size\[0\] 0\.94 -> ", out) and "WROTE  nothing (dry run)" in out


def test_windowed_trials_equal_full_renders(scratch):
    path = _asset(scratch)
    from sightline.loader import asset_from_data
    from sightline.march import march
    from sightline.style import read_yaml

    data, lines = read_yaml(path.read_text(encoding="utf-8"), "block")
    asset = asset_from_data(data, lines, path, ROOT)
    views = [F._setup_view(asset, 0, 128)]
    v = views[0]
    assert v.iou() == pytest.approx(v.iou0, abs=6e-4)  # match rounds to 3 places
    cls, div = v.inner()  # the fitter's class and division shares are structure's (strict) ones
    before = v.before
    assert cls == pytest.approx(before["classes_strict"], abs=2e-3)
    assert div == pytest.approx((before["region_borders_on_part_boundaries"] + before["part_boundaries_on_borders"]) / 2,
                                abs=2e-3)
    state = F._State(path, ROOT, lines, data, asset, views, [v.score() for v in views],
                     {k: dict(e) for k, e in enumerate(data["parts"])})
    for prm in F.entry_params(data["parts"][0]):
        F._fit_number(state, 0, prm, 0.008)
    scene = state.asset.scene()
    vb = march(scene, v.camera)
    full = R._model_coverage(scene, v.camera, vb.hit)
    assert np.array_equal(full, v.model) and np.array_equal(vb.part, v.part)
    assert np.array_equal(scene.surface_ids(vb), v.surf)
    assert state.evals > 0 and F.changes(state)
    assert state.scores[0] == v.score()  # the kept state is what a fresh score of the arrays gives


def test_symmetric_check_keeps_centred_parts_on_the_plane(scratch):
    path = _asset(scratch, "block_sym")
    path.write_text(path.read_text(encoding="utf-8") + "  - symmetric: {axis: x, min: 0.9}\n", encoding="utf-8")
    rep = F.fit_asset(path, ROOT, res=128, steps=(0.008,), max_move=0.05, checks=False)
    assert rep["symmetry"] == {"axes": ["x"], "kept": ["body", "knob"]}
    moved = {(c["part"], c["param"]) for c in rep["changes"]}
    assert ("body", "size[0]") in moved and ("body", "at[0]") not in moved and ("knob", "at[0]") not in moved
    assert "SYMMETRY  2 centred entries kept on the x = 0 plane" in F.format_fit(rep)
    frozen = F.frozen_params({"name": "a", "at": [0, 1, 0], "rotate": [5, 0, 0]}, [0])
    assert {p.label() for p in frozen} == {"at[0]", "rotate[1]", "rotate[2]"}
    assert F.frozen_params({"name": "a", "at": [0.2, 1, 0]}, [0]) == set()
    assert F.frozen_params({"name": "a", "at": [0, 1, 0], "mirror": "x"}, [0]) == set()


def test_entry_params_and_edit_yaml():
    names = [p.label() for p in F.entry_params({"name": "a", "shape": "box", "size": [1, 2, 3], "at": [0, 1, 0],
                                                "rotate": [0, 10, 0], "round": 0.01})]
    assert names == ["at[0]", "at[1]", "at[2]", "size[0]", "size[1]", "size[2]", "round", "rotate[0]", "rotate[1]",
                     "rotate[2]"]
    assert F.entry_params({"name": "p", "shape": "box", "size": [1, 1, 1], "op": "paint", "material": "metal"}) == []
    ext = F.entry_params({"name": "e", "shape": "extrude", "points": [[0, 0], [1, 0], [0, 1]], "depth": 0.1})
    assert [p.label() for p in ext] == ["depth", "points width", "points height"]
    text = ("asset: a   # top\nparts:\n  - name: b   # keep me\n    size: [1, 2, 3]   # and me\n    at: [0, 0, 0]\n"
            "  - {name: c, shape: sphere, radius: 0.5, at: [1, 0, 0]}\n")
    out = F.edit_yaml(text, {("parts", 0, "size"): [1.0005, 2, 3], ("parts", 1, "radius"): 0.25, ("asset",): "z"})
    assert out == ("asset: z   # top\nparts:\n  - name: b   # keep me\n    size: [1.0005, 2, 3]   # and me\n"
                   "    at: [0, 0, 0]\n  - {name: c, shape: sphere, radius: 0.25, at: [1, 0, 0]}\n")
    with pytest.raises(SightlineError, match="no such value"):
        F.edit_yaml(text, {("parts", 0, "rotate"): [0, 0, 0]})


def test_fit_rejects_bad_input(scratch):
    path = _asset(scratch)
    with pytest.raises(SightlineError, match="no part entry named lid"):
        F.fit_asset(path, ROOT, res=128, parts=["lid"], checks=False)
    with pytest.raises(SightlineError, match="res must be"):
        F.fit_asset(path, ROOT, res=8, checks=False)
    with pytest.raises(SightlineError, match="max_move"):
        F.fit_asset(path, ROOT, res=128, max_move=-1, checks=False)


def test_ceiling_compares_opposite_pictures(scratch):
    path = _asset(scratch, extra_ref="\n  - {image: back.png, view: back, height: 1.0}")
    _picture(scratch / "back.png")  # the same block from behind: its mirror image is the front picture
    asset = load_asset(path, ROOT)
    same = R.picture_agreement(asset, 0, 0.01)
    assert same["other"] == 1 and same["view"] == "back" and same["iou"] > 0.99 and same["pair_mean_max"] > 0.995
    _picture(scratch / "back.png", notch=True)  # a bite out of one corner: the pictures disagree
    lower = R.picture_agreement(load_asset(path, ROOT), 0, 0.01)
    assert 0.85 < lower["iou"] < 0.92 and lower["pair_mean_max"] == pytest.approx((1 + lower["iou"]) / 2, abs=1e-4)
    assert R.picture_agreement(load_asset(_asset(scratch, "solo"), ROOT), 0, 0.01) is None
    rep = R.match_asset(load_asset(path, ROOT), 0, res=96)
    assert rep["ceiling"]["other"] == 1 and "CEILING the back picture (reference[1]) mirrored" in R.format_match(rep)
    assert R.match_asset(load_asset(path, ROOT), 0, res=96, ceiling=False)["ceiling"] is None
    assert R.opposite_view("left") == "right" and R.opposite_view("top") is None
    assert R.opposite_view("orbit:30:15") == "orbit:210:-15"


def test_cli_fit_writes_a_copy_and_runs_the_checks(scratch):
    path = _asset(scratch)
    out = scratch / "block_fitted.yaml"
    cmd = [sys.executable, "-m", "sightline", "fit", str(path), "--res", "128", "--steps", "8,4", "--max-move", "0.05",
           "--out", str(out), "--root", str(ROOT)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "CHECKS  2 run: none that passed before fails now" in proc.stdout and "WROTE" in proc.stdout
    fitted = load_asset(out, ROOT)
    assert fitted.name == "block_fitted" and fitted.part("body").params["size"][0] > 0.97
    assert "fit: round 1" in proc.stderr
