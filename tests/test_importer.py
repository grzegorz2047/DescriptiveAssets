"""Import of ready .glb models as descriptive assets: fitted primitives per mesh piece."""

import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sightline import importer
from sightline.export import export_glb
from sightline.loader import load_asset, load_asset_text
from sightline.style import load_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "out" / "test_tmp" / "importer"

SOURCE = """\
asset: {name}
style: default
parts:
  - name: slab
    shape: box
    size: [1.2, 0.2, 0.8]
    material: stone
    at: [0, 0, 0]
    align: bottom
  - name: drum
    shape: cylinder
    radius: 0.3
    height: 0.6
    material: iron
    at: [-0.3, 0.2, 0]
    align: bottom
  - name: ball
    shape: sphere
    radius: 0.2
    material: paint_red
    at: [0.35, 0.4, 0.1]
  - name: beam
    shape: box
    size: [0.9, 0.08, 0.08]
    rotate: [0, 0, 30]
    material: wood
    at: [0.2, 0.75, -0.25]
"""


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


@pytest.fixture
def glb(scratch):
    src = scratch / "source.yaml"
    src.write_text(SOURCE.format(name="source"), encoding="utf-8")
    asset = load_asset(src, ROOT)
    return export_glb(asset, scratch / "source.glb", max_cells=96)


def test_import_fits_one_primitive_per_part(glb):
    result = importer.import_glb(glb, "imported", style=load_style("default", ROOT))
    parts = {p["name"]: p for p in result["parts"]}
    assert set(parts) == {"slab", "drum", "ball", "beam"}
    assert parts["slab"]["fit"]["shape"] == "box"
    assert parts["drum"]["fit"]["shape"] == "cylinder"
    assert parts["ball"]["fit"]["shape"] == "ellipsoid"
    assert parts["beam"]["fit"]["shape"] == "box"
    slab = parts["slab"]["fit"]["params"]["size"]
    assert sorted(slab) == pytest.approx(sorted([1.2, 0.2, 0.8]), abs=0.03)
    assert parts["drum"]["fit"]["params"]["radius"] == pytest.approx(0.3, abs=0.02)
    assert parts["drum"]["fit"]["params"]["height"] == pytest.approx(0.6, abs=0.03)
    # the tilted beam keeps its 30 degree tilt (about z, either sign of the long axis)
    rz = parts["beam"]["fit"]["rotate"][2] % 180
    assert min(abs(rz - 30), abs(rz - 210 % 180)) < 3 or abs(abs(rz) - 30) < 3, parts["beam"]["fit"]["rotate"]
    assert parts["slab"]["material"] == "stone" and parts["drum"]["material"] == "iron"
    assert parts["ball"]["material"] == "paint_red"
    assert all(p["error"] < 0.02 for p in parts.values()), {n: p["error"] for n, p in parts.items()}


def test_imported_yaml_loads_and_keeps_size(glb, scratch):
    result = importer.import_glb(glb, "imported")
    path = scratch / "imported.yaml"
    path.write_text(result["yaml"], encoding="utf-8")
    asset = load_asset_text(result["yaml"], path, ROOT)
    lo, hi = asset.scene().bounds(include_helpers=False)
    src_lo, src_hi = load_asset(scratch / "source.yaml", ROOT).scene().bounds(include_helpers=False)
    assert np.allclose(hi - lo, src_hi - src_lo, atol=0.04)
    assert lo[1] == pytest.approx(0.0, abs=0.02)
    text = importer.format_import(result)
    assert "IMPORT  imported: 4 parts" in text and "WORST FITS" in text


def test_split_and_max_parts(scratch):
    src = scratch / "rows.yaml"
    src.write_text("asset: rows\nstyle: default\nparts:\n  - name: post\n    shape: box\n    size: [0.1, 0.5, 0.1]\n"
                   "    material: wood\n    at: [0, 0, 0]\n    align: bottom\n    repeat: {count: 5, step: [0.3, 0, 0]}\n",
                   encoding="utf-8")
    glb = export_glb(load_asset(src, ROOT), scratch / "rows.glb", max_cells=64, per_part=False)
    merged = importer.import_glb(glb, "rows_in")
    assert len(merged["parts"]) == 5  # one merged mesh, split into its 5 connected pieces
    capped = importer.import_glb(glb, "rows_cap", max_parts=3)
    assert len(capped["parts"]) == 3 and capped["facts"]["dropped"] == 2
    whole = importer.import_glb(glb, "rows_whole", split=False)
    assert len(whole["parts"]) == 1


def test_cli_import(glb, scratch):
    out = scratch / "cli_in.yaml"
    run = [sys.executable, "-m", "sightline", "import", str(glb), "--name", "cli_in", "--out", str(out), "--root", str(ROOT)]
    proc = subprocess.run(run, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert out.is_file() and "WORST FITS" in proc.stdout
    again = subprocess.run(run, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert again.returncode == 2 and "already exists" in again.stderr
    bad = subprocess.run([sys.executable, "-m", "sightline", "import", str(scratch / "x.obj"), "--root", str(ROOT)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert bad.returncode == 2 and "glTF binary" in bad.stderr
