"""`sightline detail`: inner edges the model lacks become strokes paints that follow the surface."""

import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sightline import detail as D, structure as st
from sightline.loader import load_asset

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "detail"
FRAME, PLATE, LINE = (40, 40, 44), (200, 202, 206), (120, 122, 126)  # the seam must be strong enough for the edge detector (box blur) yet bright-class


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _panel(groove=True) -> Image.Image:
    """A 2 x 2 m dark frame (1 px = 0.01 m) with one bright plate; with `groove`, a thin line across the plate."""
    img = Image.new("RGBA", (260, 240), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 20, 229, 219), fill=FRAME + (255,))
    d.rectangle((50, 40, 209, 180), fill=PLATE + (255,))  # 1.6 x 1.4 m plate
    if groove:
        d.rectangle((70, 110, 189, 111), fill=LINE + (255,))  # a 1.2 m seam, 2 px tall, in the plate's middle
    return img


def _asset(scratch, groove=True):
    (scratch / "refs").mkdir(exist_ok=True)
    pic = "panel.png" if groove else "panel_plain.png"  # two files: the picture cache is keyed by path, mtime and size
    _panel(groove).save(scratch / "refs" / pic)
    text = textwrap.dedent(f"""\
        asset: panel
        style: default
        reference: {{image: refs/{pic}, view: front, height: 2.0}}
        parts:
          - name: frame
            shape: box
            size: [2.0, 2.0, 0.2]
            material: iron
            at: [0, 0, 0]
            align: bottom
          - name: plate
            shape: box
            size: [1.6, 1.4, 0.06]
            material: metal
            at: [0, 1.1, 0.12]
        checks:
          - grounded
        """)
    path = scratch / "panel.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_edges_are_the_seam_and_nothing_else(scratch):
    rep = st.compare_structure(load_asset(_asset(scratch), ROOT), res=256)
    missing = D.missing_edges(rep)
    rows, cols = np.nonzero(missing)
    assert 40 <= missing.sum() <= 400, int(missing.sum())
    v = rep["_trace"]["v_of"](rows + 0.5)
    assert np.abs(v - 1.09).max() < 0.06, (v.min(), v.max())  # all on the seam's height
    rep0 = st.compare_structure(load_asset(_asset(scratch, groove=False), ROOT), res=256)
    assert D.missing_edges(rep0).sum() < 10


def test_chains_order_pixels_and_simplify_to_a_few_points():
    mask = np.zeros((40, 60), dtype=bool)
    mask[20, 5:50] = True  # a straight run
    mask[10:18, 30] = True  # a separate short vertical run (8 px: just long enough)
    ch = D.chains(mask)
    assert len(ch) == 2 and sorted(len(c) for c in ch) == [8, 45]
    line = D.simplify_open(max(ch, key=len))
    assert len(line) == 2 and line[0][0] == 20 and {line[0][1], line[1][1]} == {5, 49}


def test_detail_draws_the_seam_and_raises_detail(scratch):
    path = _asset(scratch)
    rep = D.detail_asset(path, ROOT, res=256, checks=True)
    (row,) = rep["views"]
    assert row["entries"] == 1 and row["lines"] >= 1 and row["kept"], row
    assert row["after"]["detail"] > row["before"]["detail"] + 0.2, (row["before"], row["after"])
    assert row["after"]["classes"] >= row["before"]["classes"] - D.FLOOR
    assert "shape: strokes" in rep["text"] and "from: [plate]" in rep["text"] and "op: paint" in rep["text"]
    assert rep["checks"]["new_failures"] == []
    text = D.format_detail(rep)
    assert "DETAIL  panel" in text and "kept" in text
    # the grooved copy loads and the seam lies on the plate's middle, in the frame's dark material
    out = scratch / "grooved" / "panel.yaml"
    out.parent.mkdir()
    shutil.copytree(scratch / "refs", out.parent / "refs")
    out.write_text(rep["text"], encoding="utf-8")
    asset = load_asset(out, ROOT)
    seam = [p for p in asset.parts if p.name.startswith("groove_")]
    assert len(seam) == 1 and seam[0].is_paint and seam[0].material == "iron"
    assert abs(seam[0].position[1] - 1.09) < 0.05


def test_detail_leaves_a_model_without_lacking_edges_alone(scratch):
    rep = D.detail_asset(_asset(scratch, groove=False), ROOT, res=256, checks=False)
    assert rep["entries"] == 0 and rep["views"][0]["why"] == "nothing to draw"


def test_cli_detail(scratch):
    path = _asset(scratch)
    out = scratch / "copy" / "panel_grooved.yaml"
    proc = subprocess.run([sys.executable, "-m", "sightline", "detail", str(path), "--res", "256", "--out", str(out),
                           "--no-check", "--root", str(ROOT)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert "DETAIL  panel" in proc.stdout and "WROTE" in proc.stdout
    assert out.exists() and "asset: panel_grooved" in out.read_text(encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "sightline", "detail", str(path), "--views", "x", "--root", str(ROOT)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode != 0 and "comma-separated" in proc.stderr + proc.stdout
