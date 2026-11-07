"""Checks: every kind PASS and FAIL with the measured values, entry forms, defaults and errors."""

import pathlib
import re
import shutil
import sys
import textwrap
import types

import pytest

from sightline.checks import CHECK_KINDS, format_checks, parse_checks, run_checks
from sightline.errors import SightlineError
from sightline.inspect import contrast_ratio, hex_to_rgb, inspect_asset
from sightline.loader import load_asset, load_asset_text

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
SCRATCH = ROOT / "out" / "test_tmp" / "checks"


def load(body: str, name: str):
    text = f"asset: {name}\n" + textwrap.dedent(body).lstrip("\n")
    return load_asset_text(text, FIX / f"{name}.yaml", ROOT)


def make(body: str, name: str):
    asset = load(body, name)
    return asset, inspect_asset(asset)


def check(case, entry):
    asset, report = case
    (res,) = run_checks(asset, report, [entry])
    return res


@pytest.fixture(scope="module")
def stack():
    return make("""
        parts:
          - {name: base, shape: box, size: [4, 1, 2], material: stone, align: bottom}
          - {name: tower, shape: box, size: [1, 3, 1], material: wood, on: base.top}
          - {name: knob, shape: sphere, radius: 0.2, material: gold, on: base.top_left, offset: [0.4, 0, 0]}
          - {name: niche, shape: box, size: [0.6, 0.6, 0.4], op: cut, on: tower.front, align: front}
        checks:
          - grounded
          - size: {height: 4}
        """, "stack")


@pytest.fixture(scope="module")
def floater():
    return make("""
        parts:
          - {name: block, shape: box, size: [2, 1, 2], material: stone, align: bottom}
          - {name: ball, shape: sphere, radius: 0.4, material: gold, at: [0, 1.9, 0]}
        """, "floater")


@pytest.fixture(scope="module")
def balloon():
    return make("""
        parts:
          - {name: ball, shape: sphere, radius: 0.5, material: gold, at: [0, 1.5, 0]}
        """, "balloon")


@pytest.fixture(scope="module")
def overhang():
    return make("""
        parts:
          - {name: pillar, shape: box, size: [0.4, 2, 0.4], material: stone, align: bottom}
          - {name: slab, shape: box, size: [3, 0.5, 1], material: stone, on: pillar.top, offset: [1.4, 0, 0]}
        """, "overhang")


@pytest.fixture(scope="module")
def occluded():
    return make("""
        parts:
          - {name: wall, shape: box, size: [4, 3, 0.5], material: paint_white, at: [0, 0, 1], align: bottom}
          - {name: crate, shape: box, size: [1, 1, 1], material: paint_black, at: [0, 0, -1], align: bottom}
          - {name: crate_lid, shape: box, size: [1, 0.1, 1], material: paint_black, on: crate.top}
          - {name: plank, shape: box, size: [1, 0.2, 0.5], material: paint_white, on: wall.top_front, align: bottom_front}
        """, "occluded")


@pytest.fixture(scope="module")
def spiky():
    return make("""
        parts:
          - {name: body, shape: box, size: [2, 4, 2], material: stone, align: bottom}
          - {name: spike, shape: cylinder, radius: 0.02, height: 1.5, material: metal, on: body.top}
          - {name: chimney, shape: box, size: [0.6, 0.8, 0.6], material: stone, on: body.top_left, align: bottom_left}
        """, "spiky")


# ----------------------------------------------------------------------------- ground, connection, stability


def test_grounded(stack, balloon):
    res = check(stack, "grounded")
    assert res.passed and res.kind == "grounded"
    assert res.measured == "min y 0.00 m" and res.expected == "|min y| <= 0.02 m"
    res = check(balloon, "grounded")
    assert not res.passed and res.measured == "min y 1.00 m" and res.data["min_y"] == 1.0
    assert "1.00 m above the ground" in res.message
    assert check(balloon, {"grounded": {"tolerance": 1.2}}).passed


def test_connected(stack, floater):
    res = check(stack, "connected")
    assert res.passed and res.measured == "1 piece(s), none floating"
    res = check(floater, "connected")
    assert not res.passed and res.data["floating"] == 1
    gap = floater[1]["floating"][0]["gap"]
    assert res.measured == f"1 floating: ball {gap:.2f} m above block"
    assert abs(gap - 0.5) <= floater[1]["voxel_cell"]


def test_stable(stack, overhang, balloon):
    res = check(stack, "stable")
    assert res.passed and res.data["margin"] > 0.4
    res = check(overhang, "stable")
    assert not res.passed and res.data["margin"] == pytest.approx(-(1.154 - 0.2), abs=0.04)
    assert res.measured == f"center of mass {-res.data['margin']:.2f} m right of (+x) the footprint (ground contacts: pillar)"
    assert res.data["support"] == ["pillar"] and res.message.startswith("add a support under the center of mass")
    res = check(balloon, "stable")
    assert not res.passed and res.measured == "n/a (not grounded)"


# ----------------------------------------------------------------------------- sizes and proportions


def test_size(stack):
    res = check(stack, {"size": {"height": 4}})
    assert res.passed and res.measured == "height 4.00 m" and res.expected == "height 3.92..4.08 m"
    res = check(stack, {"size": {"height": [4.5, 6], "width": 4, "depth": [1, 3]}})
    assert not res.passed
    assert res.measured == "height 4.00 m, width 4.00 m, depth 2.00 m"
    assert res.message == "height is 0.50 m too small"
    assert res.data == {"height": 4.0, "width": 4.0, "depth": 2.0}


def test_part_size(stack):
    res = check(stack, {"part_size": {"part": "tower", "height": 3, "width": [0.9, 1.1]}})
    assert res.passed and res.measured == "tower height 3.00 m, width 1.00 m"
    res = check(stack, {"part_size": {"part": "tower", "width": [2, 3]}})
    assert not res.passed and res.message == "width is 1.00 m too small"
    res = check(stack, {"part_size": {"part": "niche", "depth": 0.4}})  # a cut: its exact AABB
    assert res.passed and res.measured == "niche depth 0.40 m"
    # +-2% of a small size rounds to one value at 2 decimals: such a range is printed with 3.
    res = check(stack, {"part_size": {"part": "knob", "width": 0.2, "height": 0.4}})
    assert res.expected == "width 0.196..0.204 m, height 0.39..0.41 m"


def test_symmetric(stack, overhang):
    res = check(stack, {"symmetric": {"axis": "z", "min": 0.99}})
    assert not res.passed and res.measured == "z IoU 0.97"  # the niche is cut into the front only
    res = check(stack, {"symmetric": {"axis": "x"}})
    assert res.data["iou"] == stack[1]["symmetry"]["x"]
    res = check(overhang, "symmetric")  # default axis x, min 0.9
    assert not res.passed and res.expected == ">= 0.90"
    assert res.measured == f"x IoU {overhang[1]['symmetry']['x']:.2f}"


def test_ratio(stack):
    res = check(stack, {"ratio": {"a": "tower.height", "b": "total.height", "range": [0.7, 0.8]}})
    assert res.passed and res.measured == "tower.height / total.height = 3.00 / 4.00 = 0.75"
    assert res.data["ratio"] == pytest.approx(0.75)
    res = check(stack, {"ratio": {"a": "tower.width", "b": "base.width", "range": [0.5, 1]}})
    assert not res.passed and res.data["ratio"] == pytest.approx(0.25) and res.message == "ratio too small"
    assert check(stack, {"ratio": {"a": "base.width", "b": "base.depth", "range": 2}}).passed  # scalar = +-2%


def test_relation(stack):
    res = check(stack, {"relation": {"a": "tower", "is": "above", "b": "base"}})
    assert res.passed and res.measured == "tower center y 2.50 vs base 0.50 (+2.00 m)"
    res = check(stack, {"relation": {"a": "base", "is": "above", "b": "tower"}})
    assert not res.passed and res.message == "base is not above tower"
    assert check(stack, {"relation": {"a": "knob", "is": "left_of", "b": "tower"}}).passed
    assert not check(stack, {"relation": {"a": "knob", "is": "right_of", "b": "tower"}}).passed
    assert check(stack, {"relation": {"a": "tower", "is": "touching", "b": "base"}}).measured == "touch"
    res = check(stack, {"relation": {"a": "tower", "is": "touching", "b": "knob"}})
    assert not res.passed and res.measured.startswith("no contact (bounding boxes")
    assert check(stack, {"relation": {"a": "niche", "is": "inside", "b": "tower"}}).passed
    res = check(stack, {"relation": {"a": "tower", "is": "inside", "b": "base"}})
    assert not res.passed and res.measured == "tower sticks out of base by 3.00 m (top)"
    assert check(stack, {"relation": {"a": "niche", "is": "in_front_of", "b": "base"}}).passed
    assert check(stack, {"relation": {"a": "base", "is": "behind", "b": "niche"}}).passed
    assert check(stack, {"relation": {"a": "base", "is": "below", "b": "knob"}}).passed


# ----------------------------------------------------------------------------- views


def test_visible(occluded):
    res = check(occluded, {"visible": {"part": "crate", "view": "front", "min": 0.5}})
    assert not res.passed and res.measured == "crate 0% in front" and res.expected == ">= 50%"
    assert res.data["fraction"] <= 0.02 and res.message == "mostly hidden by wall in front"
    res = check(occluded, {"visible": {"part": "crate", "view": "right"}})
    assert res.passed and res.data["fraction"] >= 0.95
    assert not check(occluded, {"visible": {"part": "crate", "view": "top"}}).passed  # under its lid
    res = check(occluded, {"visible": {"part": "crate", "min": 0.9}})  # default view: any
    assert res.passed and res.data["fraction"] >= 0.95  # the lid's half-pixel edge covers a sliver
    assert res.measured == f"crate {round(res.data['fraction'] * 100)}% in right (best of front, right, top, iso)"
    res = check(occluded, {"visible": {"part": "crate", "view": "back", "min": 0.9}})  # not in the report
    assert res.passed and res.data["fraction"] >= 0.95


def test_contrast(occluded, stack):
    res = check(occluded, {"contrast": {"a": "wall", "b": "crate", "min": 3}})
    parts = {p["name"]: p for p in occluded[1]["parts"]}
    expected = contrast_ratio(hex_to_rgb(parts["wall"]["lit_color"]), hex_to_rgb(parts["crate"]["lit_color"]))
    assert res.passed and res.data["ratio"] == pytest.approx(expected)
    assert res.measured.startswith(f"{expected:.2f} (wall #")
    res = check(occluded, {"contrast": {"a": "crate", "b": "crate_lid"}})  # same material: ratio ~ 1
    assert not res.passed and res.data["ratio"] < 1.5 and res.expected == ">= 1.50"
    res = check(stack, {"contrast": {"a": "niche", "b": "base"}})  # a cut has no color
    assert not res.passed and res.measured == "niche not visible in iso"


def test_readable_at(spiky):
    res = check(spiky, {"readable_at": {"px": 32, "min_iou": 0.8, "keep": ["chimney"]}})
    assert res.passed and res.measured.startswith("IoU ") and "at 32 px front" in res.measured
    assert res.data["lost"] == []
    res = check(spiky, {"readable_at": {"keep": ["spike", "chimney"]}})
    assert not res.passed and res.data["lost"] == ["spike"]
    assert res.measured.endswith(", lost: spike") and "spike vanishes at 32 px" in res.message
    res = check(spiky, {"readable_at": {"px": 48, "min_iou": 0.99}})  # size not in the report: computed
    assert not res.passed and res.data["iou"] < 0.99
    assert check(spiky, "readable_at").passed  # defaults: 32 px, IoU >= 0.8


@pytest.fixture(scope="module")
def library():
    """A root with three assets for distinct_from."""
    shutil.rmtree(SCRATCH, ignore_errors=True)
    (SCRATCH / "assets").mkdir(parents=True)
    bodies = {
        "tall": "{name: body, shape: box, size: [1, 4, 1], material: stone, align: bottom}",
        "squat": "{name: body, shape: box, size: [4, 1, 1], material: stone, align: bottom}",
        "giant": "{name: body, shape: box, size: [2, 8, 2], material: stone, align: bottom}",
    }
    for name, part in bodies.items():
        (SCRATCH / "assets" / f"{name}.yaml").write_text(f"asset: {name}\nparts:\n  - {part}\n", encoding="utf-8")
    tall = load_asset("tall", SCRATCH)
    return tall, inspect_asset(tall)


def test_distinct_from(library):
    res = check(library, {"distinct_from": {"assets": ["squat"]}})
    assert res.passed and res.data["iou"]["squat"] < 0.3 and res.expected == "<= 0.70"
    res = check(library, {"distinct_from": {"assets": ["squat", "giant"], "view": "front", "max_iou": 0.7}})
    assert not res.passed and res.data["iou"]["giant"] > 0.95  # same shape at twice the scale
    assert res.measured.startswith("squat 0.") and "giant 1.00" in res.measured and "(front, normalized IoU)" in res.measured
    assert "too similar to giant" in res.message
    with pytest.raises(SightlineError) as info:
        check(library, {"distinct_from": {"assets": ["gaint"]}})
    assert "did you mean 'giant'" in info.value.hint
    assert info.value.where.endswith("checks[0] (distinct_from)")


# ----------------------------------------------------------------------------- budgets and reference


def test_budgets(stack):
    res = check(stack, {"max_parts": 3})
    assert res.passed and res.measured == "3 solid parts" and res.expected == "<= 3"
    res = check(stack, {"max_parts": 2})
    assert not res.passed and res.message.startswith("1 parts over budget")
    res = check(stack, {"max_materials": 3})
    assert res.passed and res.measured == "3 materials (gold, stone, wood)"
    assert not check(stack, {"max_materials": 2}).passed


@pytest.fixture
def fake_reference(monkeypatch):
    calls = []
    module = types.ModuleType("sightline.reference")

    def match_asset(asset, index=0, res=256, ceiling=True):
        assert ceiling is False  # checks never pay for the picture-agreement ceiling
        calls.append(index if res == 256 else (index, res))
        return {"iou": module.iou, "missing": module.missing}

    module.match_asset = match_asset
    module.iou, module.missing = 0.9, 0.02
    monkeypatch.setitem(sys.modules, "sightline.reference", module)
    return module, calls


def test_matches_reference(fake_reference):
    module, calls = fake_reference
    case = make("""
        parts:
          - {name: block, shape: box, size: [1, 2, 1], material: stone, align: bottom}
        reference: {image: refs/tower_front.png, view: front, height: 2}
        """, "refd")
    res = check(case, "matches_reference")
    assert res.passed and res.measured == "IoU 0.90, missing 2% (reference 0)"
    assert res.expected == "IoU >= 0.85, missing <= 5%" and calls == [0]
    module.iou, module.missing = 0.7, 0.12
    res = check(case, {"matches_reference": {"min_iou": 0.8, "max_missing": 0.1}, "why": "match the concept art"})
    assert not res.passed and res.data == {"iou": 0.7, "missing": 0.12}
    assert "IoU 0.70 < 0.80" in res.message and "12% of the reference is not covered" in res.message
    assert res.why == "match the concept art" and "(why: match the concept art)" in res.line()
    with pytest.raises(SightlineError) as info:
        check(case, {"matches_reference": {"index": 1}})
    assert "out of range" in info.value.message
    # res: the model render's long side (the match is measured on that grid).
    res = check(case, {"matches_reference": {"min_iou": 0.6, "max_missing": 0.2, "res": 512}})
    assert res.passed and res.measured == "IoU 0.70, missing 12% (reference 0, res 512)" and calls[-1] == (0, 512)
    for bad in (8, 5000, 2.5):
        with pytest.raises(SightlineError):
            check(case, {"matches_reference": {"res": bad}})


def test_matches_reference_without_reference(stack):
    with pytest.raises(SightlineError) as info:
        check(stack, "matches_reference")
    assert "no reference" in info.value.message and "reference:" in info.value.hint
    assert info.value.where == "tests/fixtures/stack.yaml: checks[0] (matches_reference)"


# ----------------------------------------------------------------------------- entry forms and errors


def test_asset_checks_and_why_forms(stack):
    asset, report = stack
    results = run_checks(asset, report)  # the asset's own checks:
    assert [r.kind for r in results] == ["grounded", "size"] and all(r.passed for r in results)
    a = check(stack, {"size": {"height": 9, "why": "must tower over the walls"}})
    b = check(stack, {"size": {"height": 9}, "why": "must tower over the walls"})
    assert a.why == b.why == "must tower over the walls" and not a.passed
    assert a.line() == b.line()
    assert a.line().startswith("FAIL  size") and a.line().endswith("(why: must tower over the walls)")
    text = format_checks(run_checks(asset, report, ["grounded", {"max_parts": 1}]))
    assert text.splitlines()[0].startswith("PASS  grounded")
    assert text.splitlines()[-1] == "CHECKS  2: 1 passed, 1 failed"


@pytest.mark.parametrize("entry,message,hint", [
    ("visibel", "unknown check 'visibel'", "did you mean 'visible'?"),
    ({"sise": {"height": 3}}, "unknown check 'sise'", "did you mean 'size'?"),
    ({"visible": {"part": "tower", "veiw": "front"}}, "unknown key 'veiw' for check 'visible'", "did you mean 'view'?"),
    ({"visible": {"part": "towr"}}, "part: asset 'stack' has no part 'towr'", "did you mean 'tower'?"),
    ({"visible": {"part": "tower", "view": "frnt"}}, "unknown view 'frnt'", "did you mean 'front'?"),
    ({"visible": {"view": "front"}}, "visible needs part:", "visible: {part: door"),
    ({"size": {}}, "size needs width, height and/or depth", "size: {height"),
    ({"size": {"height": [5, 3]}}, "height range [5, 3] has lo > hi", "swap"),
    ({"relation": {"a": "tower", "is": "abov", "b": "base"}}, "unknown relation 'abov'", "did you mean 'above'?"),
    ({"ratio": {"a": "tower.hight", "b": "total.height", "range": [0, 1]}}, "a: unknown dimension 'hight'", "did you mean 'height'?"),
    ({"symmetric": {"axis": "y"}}, "symmetric axis must be x or z", "x = left/right"),
    ({"max_parts": "many"}, "max_parts must be an integer >= 0", None),
    ({"size": {"height": 3}, "grounded": None}, "a check entry names exactly one check", "one check per list item"),
    ("size", "check 'size' needs arguments", "size: {height"),
    (42, "a check must be a name or a mapping", None),
])
def test_entry_errors(stack, entry, message, hint):
    asset, report = stack
    with pytest.raises(SightlineError) as info:
        run_checks(asset, report, ["grounded", entry])
    assert info.value.message == message or info.value.message.startswith(message), info.value.message
    if hint:
        assert hint in (info.value.hint or ""), info.value.hint
    assert re.match(r"tests/fixtures/stack\.yaml: checks\[1\]", info.value.where)


def test_validation_happens_before_any_evaluation(stack, fake_reference):
    _, calls = fake_reference
    asset, report = stack
    with pytest.raises(SightlineError):
        run_checks(asset, report, ["grounded", "nope"])
    assert parse_checks(asset, ["grounded", {"max_parts": 3}])[1].args == {"max": 3}
    assert set(CHECK_KINDS) >= {"grounded", "connected", "stable", "size", "part_size", "symmetric", "visible", "ratio",
                                "relation", "contrast", "readable_at", "distinct_from", "max_parts", "max_materials",
                                "matches_reference"}


def test_tower30_fixture_checks():
    asset = load_asset(FIX / "tower30.yaml", ROOT)
    results = run_checks(asset, inspect_asset(asset))
    assert [r.kind for r in results] == ["grounded", "connected", "size", "visible"]
    assert all(r.passed for r in results), format_checks(results)
    assert results[3].measured.startswith("door ") and results[3].data["fraction"] >= 0.8
