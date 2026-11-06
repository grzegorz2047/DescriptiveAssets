"""End-to-end tests of the `sightline` command: subprocesses on the example assets and levels."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from sightline import cli
from sightline.errors import SightlineError
from sightline.export import read_glb, validate_glb

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "test_tmp" / "cli"
EXAMPLES = ("barrel", "crate", "knight", "lighthouse", "pine_tree", "street_lamp", "watchtower")


def run(*args: str, timeout: float = 240) -> subprocess.CompletedProcess:
    """Run `python -m sightline <args> --root ROOT` and capture text output."""
    cmd = [sys.executable, "-m", "sightline", *args]
    if args and args[0] not in ("--help", "--version"):
        cmd += ["--root", str(ROOT)]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=timeout)


def wrote(proc: subprocess.CompletedProcess) -> list[pathlib.Path]:
    return [pathlib.Path(line[len("wrote "):]) for line in proc.stdout.splitlines() if line.startswith("wrote ")]


@pytest.fixture(scope="module", autouse=True)
def clean_out():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    yield


def test_help_lists_every_command():
    proc = run("--help")
    assert proc.returncode == 0
    for command in ("new", "render", "inspect", "check", "diff", "sheet", "export", "ref", "match", "level",
                    "styles", "shapes"):
        assert f"    {command}" in proc.stdout
    assert run("--version").stdout.strip().startswith("sightline ")


def test_render_writes_pngs_at_the_requested_size():
    out = OUT / "render_crate"
    proc = run("render", "crate", "--res", "96", "--views", "front,iso", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    paths = wrote(proc)
    assert [p.name for p in paths] == ["front.png", "iso.png", "parts.png", "sheet.png"]
    for p in paths:
        assert p.is_absolute() and p.is_file()
    with Image.open(paths[1]) as iso:
        # res 96 -> the iso view is 96 px on its long side, shown x2 plus a caption strip
        assert iso.width <= 2 * 96 and 2 * 96 <= iso.height <= 2 * 96 + 60


def test_check_all_examples_pass():
    proc = run("check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"ASSETS  {len(EXAMPLES)}: {len(EXAMPLES)} passed, 0 failed, 0 errors" in proc.stdout
    for name in EXAMPLES:
        assert f"ASSET  {name}  (" in proc.stdout
    assert "FAIL" not in proc.stdout


def test_check_failing_asset_exits_1_and_json_reports_it():
    path = OUT / "tall_crate.yaml"
    text = (ROOT / "assets" / "crate.yaml").read_text(encoding="utf-8").replace("asset: crate", "asset: tall_crate")
    text = text.replace("  - max_parts: 16", "  - max_parts: 16\n  - size: {height: [2, 3], why: \"test\"}")
    path.write_text(text, encoding="utf-8")
    proc = run("check", str(path))
    assert proc.returncode == 1
    fail = [line for line in proc.stdout.splitlines() if "FAIL" in line]
    assert len(fail) == 1 and "size" in fail[0] and "height 0.90 m" in fail[0] and "(why: test)" in fail[0]
    proc = run("check", str(path), "crate", "--json")
    assert proc.returncode == 1
    data = json.loads(proc.stdout)
    assert [d["asset"] for d in data] == ["tall_crate", "crate"]
    assert [d["passed"] for d in data] == [False, True]
    assert data[1]["results"] and all(r["passed"] for r in data[1]["results"])


def test_inspect_json_parses_with_measured_numbers():
    proc = run("inspect", "barrel", "--json")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["asset"] == "barrel" and report["part_count"] == 8
    assert report["size"]["height"] == pytest.approx(0.94, abs=0.02)
    assert report["ground"]["grounded"] is True and report["floating"] == []
    text = run("inspect", "barrel").stdout
    assert text.startswith("ASSET       barrel") and "SILHOUETTE front" in text


def test_export_glb_and_obj():
    out = OUT / "export"
    proc = run("export", "crate", "--max-cells", "48", "--out", str(out / "crate.glb"))
    assert proc.returncode == 0, proc.stderr
    assert "glTF structure OK" in proc.stdout and "total" in proc.stdout
    (glb,) = wrote(proc)
    assert validate_glb(glb) == []
    doc, blob = read_glb(glb)
    names = {n["name"] for n in doc["nodes"]}
    assert {"body", "post_0", "brace_front", "brace_back"} <= names
    assert doc["asset"]["version"] == "2.0" and len(blob) > 0
    proc = run("export", "crate", "--format", "obj", "--merged", "--max-cells", "32", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    obj, mtl = wrote(proc)
    assert obj.suffix == ".obj" and mtl.suffix == ".mtl" and obj.is_file() and mtl.is_file()
    assert obj.read_text(encoding="utf-8").count("\no ") == 1  # merged: one object
    bad = run("export", "crate", "--format", "obj", "--bake-colors")
    assert bad.returncode == 2 and "needs --format glb" in bad.stderr


def test_ref_and_match_on_the_lighthouse():
    proc = run("ref", "assets/refs/lighthouse_concept.png", "--height", "11.85", "--json")
    assert proc.returncode == 0, proc.stderr
    ref = json.loads(proc.stdout)
    assert ref["mask"]["confidence"] == "high"
    assert ref["size"]["height"] == pytest.approx(11.85, abs=0.05)
    out = OUT / "match"
    proc = run("match", "lighthouse", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert "FIT     IoU" in proc.stdout
    (png,) = wrote(proc)
    assert png == (out / "match_0.png").resolve() and png.is_file()
    report = json.loads(run("match", "lighthouse", "--out", str(out), "--json").stdout)
    assert report["iou"] >= 0.85 and report["missing"] <= 0.06


def test_level_commands_and_exit_codes():
    good = run("level", "play", "courtyard")
    assert good.returncode == 0, good.stderr
    assert "PATH  spawn → goal" in good.stdout and "CHECKS  8: 8 passed, 0 failed" in good.stdout
    assert "OBJECTS" in good.stdout and "watchtower at (6,2)" in good.stdout
    broken = run("level", "play", "broken_bridge")
    assert broken.returncode == 1
    assert "CHECKS  7: 4 passed, 3 failed" in broken.stdout
    check = run("level", "check", "broken_bridge", "--json")
    assert check.returncode == 1
    results = json.loads(check.stdout)
    assert sorted(r["kind"] for r in results if not r["passed"]) == ["no_softlocks", "order", "reachable"]
    out = OUT / "level"
    render = run("level", "render", "courtyard", "--out", str(out))
    assert render.returncode == 0, render.stderr
    paths = wrote(render)
    assert [p.name for p in paths] == ["iso.png", "top.png"] and all(p.is_file() for p in paths)


def test_errors_exit_2_with_where_and_hint():
    proc = run("render", "crat")
    assert proc.returncode == 2
    assert "unknown asset 'crat'" in proc.stderr and "hint: did you mean 'crate'?" in proc.stderr
    assert "Traceback" not in proc.stderr
    proc = run("level", "play", "courtyar")
    assert proc.returncode == 2 and "did you mean 'courtyard'?" in proc.stderr
    bad = OUT / "bad_box.yaml"
    bad.write_text("asset: bad_box\nparts:\n  - {name: box, shape: box, size: [1, 1, 1], material: wod}\n",
                   encoding="utf-8")
    proc = run("check", str(bad))
    assert proc.returncode == 2
    assert "bad_box.yaml: parts[0] (box), line 3" in proc.stderr and "did you mean 'wood'?" in proc.stderr
    proc = run("render", "crate", "--views", "front,isso")
    assert proc.returncode == 2 and "did you mean 'iso'?" in proc.stderr
    usage = run("render")  # argparse: missing argument
    assert usage.returncode == 2 and "usage:" in usage.stderr


def test_new_writes_a_template_that_loads_and_passes():
    path = OUT / "new" / "stone_well.yaml"
    brief = "A stone well, about 1.5 m tall. It should read at 32 px. Keep it under 20 parts."
    proc = run("new", str(path), "--brief", brief)
    assert proc.returncode == 0, proc.stderr
    assert wrote(proc) == [path.resolve()]
    text = path.read_text(encoding="utf-8")
    assert "asset: stone_well" in text and "# - name: door" in text and "op: cut" in text and "mirror: x" in text
    assert '"It should read at 32 px."  ->  readable_at' in text and "->  max_parts" in text
    assert run("check", str(path)).returncode == 0
    again = run("new", str(path))
    assert again.returncode == 2 and "already exists" in again.stderr and "--force" in again.stderr
    assert run("new", str(OUT / "new" / "Bad-Name.yaml")).returncode == 2


def test_diff_against_a_file_and_git():
    old = OUT / "crate_old.yaml"
    old.write_text((ROOT / "assets" / "crate.yaml").read_text(encoding="utf-8")
                   .replace("size: [0.86, 0.86, 0.86]", "size: [0.86, 0.7, 0.86]"), encoding="utf-8")
    proc = run("diff", "crate", "--against", str(old))
    assert proc.returncode == 0, proc.stderr
    assert "CHANGES" in proc.stdout
    assert "parts.body.size: [0.86, 0.70, 0.86] -> [0.86, 0.86, 0.86]" in proc.stdout
    assert "ground.grounded: no -> yes" in proc.stdout
    # An asset that git does not know yet: a clear message, not a crash.
    path = OUT / "untracked.yaml"
    path.write_text("asset: untracked\nparts:\n  - {name: b, shape: box, size: [1, 1, 1], material: wood}\n",
                    encoding="utf-8")
    proc = run("diff", str(path))
    assert proc.returncode == 2 and "Traceback" not in proc.stderr
    assert "untracked.yaml is not in git HEAD yet" in proc.stderr and "--against" in proc.stderr


def test_git_show_reads_a_committed_file():
    text = cli._git_show(ROOT / "src" / "sightline" / "errors.py", "HEAD")
    assert "class SightlineError" in text
    with pytest.raises(SightlineError, match="unknown git revision"):
        cli._git_show(ROOT / "src" / "sightline" / "errors.py", "no_such_branch_xyz")


def test_styles_shapes_and_sheet():
    styles = run("styles")
    assert styles.returncode == 0
    assert "STYLE  default  (styles/default.yaml)" in styles.stdout
    assert "paint_red" in styles.stdout and "#c23b30" in styles.stdout and "PALETTE" in styles.stdout
    shapes = run("shapes")
    assert shapes.returncode == 0
    for word in ("cylinder", "wedge", "ANCHORS", "top_front_left", "repeat:", "mirror:", "op: cut"):
        assert word in shapes.stdout
    out = OUT / "library.png"
    sheet = run("sheet", "crate", "barrel", "--out", str(out), "--res", "120")
    assert sheet.returncode == 0, sheet.stderr
    assert wrote(sheet) == [out.resolve()]
    with Image.open(out) as img:
        assert img.width > 200 and img.height > 120


def test_new_with_root_and_a_picture_in_the_brief():
    root = OUT / "newroot"
    (root / "styles").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "styles" / "default.yaml", root / "styles" / "default.yaml")
    brief = ("A windmill from the concept art in refs/windmill_concept.png, about 12.7 m tall. "
             "It is twice as tall as wide and must not tip over.")
    cmd = [sys.executable, "-m", "sightline", "new", "mill", "--root", str(root), "--brief", brief]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=240)
    assert proc.returncode == 0, proc.stderr
    assert f"sightline check mill --root {root.resolve().as_posix()}" in proc.stdout  # runnable as printed
    text = (root / "assets" / "mill.yaml").read_text(encoding="utf-8")
    assert "# reference:" in text and "#   image: refs/windmill_concept.png" in text
    assert "->  matches_reference: {min_iou: 0.85} | size:" in text and "| stable" in text


def test_res_limits_and_diff_against_errors():
    for args in (("match", "lighthouse", "--res", "0"), ("inspect", "crate", "--res", "100000")):
        proc = run(*args)
        assert proc.returncode == 2 and "Traceback" not in proc.stderr
        assert "res must be an integer from 16 to 4096" in proc.stderr
    bad = OUT / "crate_other.yaml"
    bad.write_text((ROOT / "assets" / "crate.yaml").read_text(encoding="utf-8").replace("asset: crate", "asset: other"),
                   encoding="utf-8")
    proc = run("diff", "crate", "--against", str(bad))
    assert proc.returncode == 2
    assert proc.stderr.startswith(f"in --against {bad.resolve().as_posix()}: asset, line 1: asset: 'other'")


def test_brief_todos_pick_the_matching_check():
    text = cli._brief_todos("Tall, about 8 m. The door must face the player. Readable at 16 px.")
    lines = text.splitlines()
    assert "size:" in lines[1] and "visible:" in lines[2] and "readable_at:" in lines[3]
    assert "size: {height: [lo, hi]}" in lines[1]


def test_brief_size_hint_names_the_dimension():
    text = cli._brief_todos("Upper wingspan 10 m, about 2.7 m tall on its wheels. A fuselage about 7.5 m long.")
    lines = text.splitlines()
    assert "size: {width: [lo, hi], height: [lo, hi]} or part_size" in lines[1]
    assert "size: {depth: [lo, hi]} or part_size" in lines[2]


def test_render_no_ground_and_views_from_below(tmp_path):
    out = tmp_path / "r"
    proc = run("render", "crate", "--views", "front,orbit:30:-20", "--no-ground", "--no-human", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    from sightline.loader import load_asset
    from sightline.render import render_raw

    crate = load_asset("crate", ROOT)
    raws = render_raw(crate, ["front", "orbit:30:-20"], 64)
    assert raws["front"].ground.any()  # the style shows the slab ...
    assert not raws["orbit:30:-20"].ground.any()  # ... but not when looking up from below the ground
    assert not render_raw(crate, ["front"], 64, ground=False)["front"].ground.any()
