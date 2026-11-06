"""Loader and style: placement, anchors, expansion, ops, references and precise errors."""

import math
import pathlib
import re
import shutil
import textwrap

import numpy as np
import pytest

from sightline.errors import SightlineError
from sightline.loader import load_asset, load_asset_text, twin_name
from sightline.model import make_part, rotation_matrix
from sightline.style import load_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def load(body: str, name: str = "t", root=ROOT):
    text = f"asset: {name}\n" + textwrap.dedent(body).lstrip("\n")
    return load_asset_text(text, FIX / f"{name}.yaml", root)


def error(body: str, name: str = "t", root=ROOT) -> SightlineError:
    with pytest.raises(SightlineError) as info:
        load(body, name, root)
    return info.value


def center(part):
    lo, hi = part.world_aabb()
    return (lo + hi) / 2


@pytest.fixture
def scratch(request):
    """A clean scratch directory under out/ (the E: drive, gitignored)."""
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:80]
    path = ROOT / "out" / "test_tmp" / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


@pytest.fixture(scope="module")
def tower():
    return load_asset(FIX / "tower30.yaml", ROOT)


# ----------------------------------------------------------------------------- fixture asset


def test_tower_fixture_structure(tower):
    names = [p.name for p in tower.parts]
    assert len(names) == 36 and len(set(names)) == 36
    assert names[:4] == ["base", "plinth", "shaft", "door"]
    assert [p.name for p in tower.parts if p.op == "cut"] == ["door", "window_0", "window_1", "window_2", "window_3", "drain"]
    assert tower.part("door").material is None and tower.part("door").cut_from == ["shaft"]
    assert tower.part("shaft").tags == ["walls"] and tower.tags == ["building", "fixture"]
    assert tower.style.name == "default" and tower.brief.startswith("A stone-and-wood watchtower")
    assert len(tower.checks) == 4 and tower.checks[0] == "grounded"
    (ref,) = tower.references
    assert ref.image == (FIX / "refs" / "tower_front.png").resolve() and ref.image.is_file()
    assert (ref.view, ref.height, ref.mask) == ("front", 8.4, "auto")
    assert tower.part("crenel_3").source == "parts[6] (crenel) copy 3"
    assert tower.part("banner_r").source == "parts[11] (banner_l) mirror"


def test_stacking_with_default_opposite_align(tower):
    # base 0.6 tall on the ground; plinth (on base.top, default align bottom) 0.4; shaft 5.
    assert tower.part("base").world_aabb()[0][1] == pytest.approx(0.0)
    assert tower.part("plinth").world_aabb()[0][1] == pytest.approx(0.6)
    assert tower.part("shaft").world_aabb()[0][1] == pytest.approx(1.0)
    assert tower.part("shaft").world_aabb()[1][1] == pytest.approx(6.0)
    # finial: on roof.top, default align = bottom -> its lowest point is the roof apex.
    apex = tower.part("roof").anchor("top")
    assert apex[1] == pytest.approx(6.0 + 0.4 + 2.2)
    assert tower.part("finial").anchor("bottom") == pytest.approx(apex)


def test_normalized_anchor_and_explicit_align(tower):
    door = tower.part("door")
    lo, hi = door.world_aabb()
    # shaft@[0, -1, 1] = bottom of the bbox front face (circumradius 1.3); align bottom.
    assert lo[1] == pytest.approx(1.0) and hi[1] == pytest.approx(2.8)
    assert center(door) == pytest.approx([0, 1.9, 1.3])
    ramp = tower.part("ramp")  # base@[0, 1, 1] with align top_back
    assert ramp.anchor("top_back") == pytest.approx([0, 0.6, 2.0])
    assert ramp.world_aabb()[0] == pytest.approx([-0.6, 0.0, 2.0])


def test_rotated_part_anchors():
    a = load("""
        parts:
          - {name: beam, shape: box, size: [2, 1, 0.5], material: wood, rotate: [0, 0, 90], at: [1, 2, 3]}
    """)
    beam = a.part("beam")
    # Rz(90): local +X -> world +Y, local +Y -> world -X.
    assert beam.anchor("top") == pytest.approx([1 - 0.5, 2, 3])
    assert beam.anchor("right") == pytest.approx([1, 2 + 1, 3])
    assert beam.anchor("top_right_front") == pytest.approx([0.5, 3, 3.25])
    assert beam.anchor((0, 1, 0)) == pytest.approx(beam.anchor("@[0, 1, 0]"))
    lo, hi = beam.world_aabb()
    assert lo == pytest.approx([0.5, 1, 2.75]) and hi == pytest.approx([1.5, 3, 3.25])


def test_follow_inherits_rotation_and_offset_in_target_frame():
    a = load("""
        parts:
          - {name: arm, shape: box, size: [2, 0.2, 0.2], material: wood, rotate: [0, 90, 0]}
          - {name: hand, shape: box, size: [0.2, 0.2, 0.4], material: skin, on: arm.right, offset: [0.5, 0, 0]}
          - {name: flag, shape: box, size: [0.2, 0.2, 0.4], material: skin, on: arm.right, offset: [0.5, 0, 0], follow: false}
          - {name: tip, shape: sphere, radius: 0.1, material: gold, on: hand.right}
    """)
    ry = rotation_matrix(0, 90, 0)
    assert a.part("arm").anchor("right") == pytest.approx([0, 0, -1])  # Ry(90): +X -> -Z
    hand = a.part("hand")
    assert np.allclose(hand.matrix, ry)
    # default align left; offset [0.5, 0, 0] in the arm frame = 0.5 m along world -Z
    assert hand.position == pytest.approx([0, 0, -1 - 0.1 - 0.5])
    flag = a.part("flag")
    assert np.allclose(flag.matrix, np.eye(3))
    assert flag.position == pytest.approx([0.1 + 0.5, 0, -1])
    # chains follow the full transform: the tip sits beyond the hand along world -Z
    assert a.part("tip").position == pytest.approx([0, 0, -1.6 - 0.1 - 0.1])
    assert a.part("hand").attached_to == "arm"


def test_at_with_rotation_and_world_offset():
    a = load("""
        parts:
          - {name: post, shape: box, size: [0.2, 2, 0.4], material: wood, at: [1, 0, 0], align: bottom,
             rotate: [0, 90, 0], offset: [0, 0, 1]}
    """)
    post = a.part("post")
    assert post.position == pytest.approx([1, 1, 1])  # offset stays in the world frame
    lo, hi = post.world_aabb()
    assert hi - lo == pytest.approx([0.4, 2, 0.2])


def test_repeat_linear(tower):
    xs = [center(tower.part(f"post_{k}"))[0] for k in range(3)]
    assert xs == pytest.approx([1.0, 1.4, 1.8])
    assert center(tower.part("rail")) == pytest.approx([1.4, 0.6, 2.6])
    e = error("""
        parts:
          - {name: peg, shape: box, size: [0.1, 0.1, 0.1], material: wood, repeat: {count: 3, step: [1, 0, 0]}}
          - {name: cap, shape: sphere, radius: 0.1, material: wood, on: peg.top}
    """)
    assert "'peg' was expanded by repeat into peg_0 … peg_2" in e.message
    assert "peg_0" in e.hint


def test_repeat_linear_in_target_frame():
    a = load("""
        parts:
          - {name: wall, shape: box, size: [4, 1, 0.2], material: brick, rotate: [0, 90, 0]}
          - {name: stud, shape: box, size: [0.1, 0.1, 0.1], material: wood, on: wall.top, repeat: {count: 3, step: [1, 0, 0]}}
    """)
    # step [1, 0, 0] in the wall frame = world -Z
    z = [a.part(f"stud_{k}").position[2] for k in range(3)]
    assert z == pytest.approx([0, -1, -2])


def test_repeat_radial(tower):
    for k in range(8):
        ang = math.radians(45 * k)
        c = center(tower.part(f"crenel_{k}"))
        assert c == pytest.approx([1.05 * math.sin(ang), 6.2, 1.05 * math.cos(ang)], abs=1e-9)
        assert np.allclose(tower.part(f"crenel_{k}").matrix, rotation_matrix(0, 45 * k, 0))
    assert center(tower.part("window_1")) == pytest.approx([1.2, 4.2, 0])
    with pytest.raises(SightlineError):
        tower.part("crenel")


def test_repeat_radial_axis_in_target_frame_and_center_anchor():
    a = load("""
        parts:
          - {name: wheel, shape: cylinder, radius: 0.5, height: 0.1, material: wood, rotate: [90, 0, 0], at: [0, 1, 0]}
          - name: spoke
            shape: box
            size: [0.4, 0.04, 0.04]
            material: iron
            on: wheel.center
            align: left
            repeat: {count: 4, axis: y, center: wheel.center}
    """)
    # The wheel's local Y is world +Z, so with follow the spokes turn about world Z through (0, 1, 0),
    # staying in the wheel plane (a world-Y axis would swing them out of it).
    tips = [a.part(f"spoke_{k}").anchor("right") for k in range(4)]
    assert tips == [pytest.approx(t, abs=1e-9) for t in ([0.4, 1, 0], [0, 1.4, 0], [-0.4, 1, 0], [0, 0.6, 0])]


def test_mirror_names_positions_and_det(tower):
    left, right = tower.part("banner_l"), tower.part("banner_r")
    s = np.diag([-1.0, 1, 1])
    assert right.position == pytest.approx(s @ left.position)
    assert np.allclose(right.matrix, s @ left.matrix)
    assert np.linalg.det(right.matrix) == pytest.approx(-1.0)
    assert np.linalg.det(tower.part("torch_right").matrix) == pytest.approx(-1.0)
    back = tower.part("lantern_back")
    assert back.position == pytest.approx(np.diag([1, 1, -1.0]) @ tower.part("lantern").position)
    assert [twin_name(n) for n in ("arm_l", "arm_r", "leg_left", "leg_right", "fin")] == \
        ["arm_r", "arm_l", "leg_right", "leg_left", "fin_mirror"]


def test_mirror_of_repeated_copies():
    a = load("""
        parts:
          - {name: peg_l, shape: box, size: [0.1, 0.1, 0.1], material: wood, at: [1, 0, 0],
             repeat: {count: 2, step: [0, 1, 0]}, mirror: x}
          - {name: fin, shape: wedge, size: [0.1, 0.3, 0.4], material: metal, at: [0.5, 0, 0], mirror: x}
    """)
    assert [p.name for p in a.parts] == ["peg_l_0", "peg_l_1", "peg_r_0", "peg_r_1", "fin", "fin_mirror"]
    assert a.part("peg_r_1").position == pytest.approx([-1, 1, 0])


def test_mirrored_sdf_stays_exact():
    """A det -1 matrix keeps M.T @ (p - pos) an isometry, so twin.sdf(S p) == part.sdf(p)."""
    a = load("""
        parts:
          - {name: fin_l, shape: wedge, size: [0.4, 0.8, 1.2], material: metal, at: [1.5, 0.5, 0.3],
             rotate: [20, 35, -10], mirror: x}
          - {name: horn_l, shape: cone, radius: 0.3, height: 1, sides: 5, material: bone, at: [1, 2, 0],
             rotate: [0, 0, 40], mirror: x}
    """)
    rng = np.random.default_rng(7)
    pts = rng.uniform(-3, 3, (500, 3))
    mirrored = pts * [-1, 1, 1]
    for name in ("fin", "horn"):
        orig, twin = a.part(f"{name}_l"), a.part(f"{name}_r")
        assert np.allclose(twin.sdf(mirrored), orig.sdf(pts), atol=1e-5)
    # world AABBs mirror too
    lo, hi = a.part("fin_l").world_aabb()
    tlo, thi = a.part("fin_r").world_aabb()
    assert tlo == pytest.approx([-hi[0], lo[1], lo[2]]) and thi == pytest.approx([-lo[0], hi[1], hi[2]])


def test_twin_refers_to_twins_of_its_targets():
    a = load(FIX.joinpath("knight.yaml").read_text(encoding="utf-8").split("\n", 1)[1], name="knight")
    assert a.part("arm_r").attached_to == "torso"
    assert a.part("sword").attached_to == "arm_r"
    b = load("""
        parts:
          - {name: arm_l, shape: capsule, radius: 0.1, height: 0.8, material: skin, at: [0.5, 1, 0], mirror: x}
          - {name: hand_l, shape: sphere, radius: 0.12, material: skin, on: arm_l.bottom, blend: 0.05, mirror: x}
          - {name: slot_l, shape: box, size: [0.05, 0.2, 0.3], op: cut, on: arm_l.center, mirror: x}
    """)
    assert (b.part("hand_r").attached_to, b.part("slot_r").cut_from) == ("arm_r", ["arm_r"])
    assert b.part("hand_r").blend == 0.05


def test_cut_effect_on_eval(tower):
    scene = tower.scene()
    shaft = tower.part_index("shaft")
    inside_door = np.array([[0, 1.9, 1.0]])  # inside the shaft volume and inside the door box
    assert tower.part("shaft").sdf(inside_door)[0] < 0
    assert scene.eval_part(shaft, inside_door)[0] > 0
    d, part = scene.eval(inside_door)
    assert d[0] > 0
    # drain (from: all) cuts the base: a point on its axis inside the base is now outside
    in_drain = np.array([[0, 0.3, -1.8]])
    assert tower.part("base").sdf(in_drain)[0] < 0
    assert scene.eval_part(tower.part_index("base"), in_drain)[0] > 0
    assert "drain" in [tower.parts[c].name for c in scene.cutters[tower.part_index("base")]]


def test_blend_groups_and_effect(tower):
    scene = tower.scene()
    groups = {tuple(tower.parts[m].name for m in g.members): g for g in scene.groups}
    assert ("bush", "bush_top") in groups and ("roof", "finial") in groups
    g = groups[("bush", "bush_top")]
    assert g.blend_k[tower.part_index("bush_top")] == 0.15
    # In the crease between the two ellipsoids (outside both, about equally near) the smooth
    # union lies below both parts: polynomial smin subtracts up to k/4 there.
    bush, top = tower.part("bush"), tower.part("bush_top")
    xs, ys = np.meshgrid(np.linspace(-3.2, -2.2, 201), np.linspace(0.4, 1.2, 161))
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.8)], 1)
    da, db = bush.sdf(pts), top.sdf(pts)
    score = np.where((da > 0) & (db > 0), np.abs(da - db) + da, np.inf)
    crease = pts[[np.argmin(score)]]
    d, part = scene.eval_group(g, crease)
    assert d[0] < min(bush.sdf(crease)[0], top.sdf(crease)[0]) - 0.02
    assert tower.parts[part[0]].name in ("bush", "bush_top")
    lo, _ = scene.bounds()
    assert g.aabb_min == pytest.approx(np.minimum(bush.world_aabb()[0], top.world_aabb()[0]) - 0.15)
    assert lo[1] == pytest.approx(0)


def test_scene_eval_matches_brute_force(tower):
    scene = tower.scene()
    rng = np.random.default_rng(3)
    pts = rng.uniform([-3.5, -0.5, -2.5], [3.5, 9.5, 3.5], (3000, 3))
    d, part = scene.eval(pts)
    brute = np.stack([scene.eval_group(g, pts)[0] for g in scene.groups])
    assert np.allclose(d, brute.min(axis=0))
    assert np.all(part >= 0)
    sub = scene.only([tower.part_index("door"), tower.part_index("shaft")])
    assert sub.solid_indices() == [tower.part_index("shaft")]
    assert sub.parts is not scene.parts and sub.parts[2] is scene.parts[2]


def test_yaml_on_key_and_booleans_stay_strings():
    a = load("""
        parts:
          - {name: a, shape: sphere, radius: 1, material: wood, align: bottom}
          - {name: b, shape: sphere, radius: 0.5, material: wood, on: a.top, follow: true}
    """)
    assert a.raw["parts"][1]["on"] == "a.top"
    e = error("""
        parts:
          - {name: a, shape: sphere, radius: 1, material: wood}
          - {name: b, shape: sphere, radius: 0.5, material: wood, on: a.top, follow: yes}
    """)
    assert "follow must be true or false" in e.message


def test_make_part_api():
    p = make_part("body", "capsule", {"radius": 0.25, "height": 1.4}, [3, 0.7, 0], material="helper", helper=True)
    assert p.helper and p.matrix.shape == (3, 3) and p.local_half == pytest.approx([0.25, 0.7, 0.25])
    assert p.sdf(np.array([[3, 0.7, 0.0]]))[0] == pytest.approx(-0.25)
    with pytest.raises(SightlineError, match="capsule height"):
        make_part("x", "capsule", {"radius": 0.5, "height": 0.6}, [0, 0, 0], material="wood")


# ----------------------------------------------------------------------------- errors


def test_error_unknown_part_key_with_line():
    e = error("""
        parts:
          - name: body
            shape: box
            size: [1, 1, 1]
            materail: wood
    """)
    assert e.message == "unknown key 'materail'"
    assert e.where == "tests/fixtures/t.yaml: parts[0] (body), line 6"
    assert "did you mean 'material'?" in e.hint


def test_error_unknown_material():
    e = error("""
        parts:
          - {name: body, shape: box, size: [1, 1, 1], material: stoen}
    """)
    assert "unknown material 'stoen'" in e.message and "did you mean 'stone'?" in e.hint
    assert "parts[0] (body)" in e.where


def test_anchor_out_of_range_prints_plain_numbers():
    from sightline.model import parse_anchor

    with pytest.raises(SightlineError) as info:
        parse_anchor("@[0, 2, 0.5]")
    assert info.value.message == "anchor components must be in [-1, 1], got [0, 2, 0.5]"


def test_error_unknown_anchor_names_part_and_lists_anchors():
    body = FIX.joinpath("knight.yaml").read_text(encoding="utf-8").split("\n", 1)[1]
    e = error(body.replace("on: arm_r.bottom", "on: arm_r.tip"), name="knight")
    assert e.message == "unknown anchor 'tip' on part 'arm_r'"
    assert e.where.startswith("tests/fixtures/knight.yaml: parts[5] (sword), line ")
    assert "top_front_left" in e.hint and "@[u, v, w]" in e.hint
    e = error(body.replace("on: arm_r.bottom", "on: arm_r.botom"), name="knight")
    assert "did you mean 'bottom'?" in e.hint
    e = error(body.replace("on: arm_r.bottom", "on: arm_r.top_bottom"), name="knight")
    assert e.message == "bad anchor 'top_bottom' on part 'arm_r': anchor 'top_bottom' names axis y twice (top and bottom)"


def test_error_forward_unknown_and_self_references():
    e = error("""
        parts:
          - {name: base, shape: box, size: [1, 1, 1], material: stone}
          - {name: lamp, shape: sphere, radius: 0.1, material: gold, on: roof.top}
          - {name: roof, shape: cone, radius: 1, height: 1, material: roof, on: base.top}
    """)
    assert "'roof' is defined later (parts[2])" in e.message and "parts[1] (lamp)" in e.where
    e = error("""
        parts:
          - {name: base, shape: box, size: [1, 1, 1], material: stone}
          - {name: lamp, shape: sphere, radius: 0.1, material: gold, on: bse.top}
    """)
    assert e.message == "unknown part 'bse'" and "did you mean 'base'?" in e.hint
    e = error("""
        parts:
          - {name: base, shape: box, size: [1, 1, 1], material: stone, on: base.top}
    """)
    assert "refers to itself" in e.message
    e = error("""
        parts:
          - {name: base, shape: box, size: [1, 1, 1], material: stone}
          - {name: lamp, shape: sphere, radius: 0.1, material: gold, on: base}
    """)
    assert "needs an anchor" in e.message and "base.top" in e.hint


def test_error_duplicates_and_limits():
    e = error("""
        parts:
          - {name: arm_r, shape: box, size: [1, 1, 1], material: stone}
          - {name: arm_l, shape: box, size: [1, 1, 1], material: stone, at: [2, 0, 0], mirror: x}
    """)
    assert "duplicate part name 'arm_r' (already defined by parts[0] (arm_r))" in e.message
    assert "mirror_name" in e.hint
    e = error("""
        parts:
          - {name: a, shape: box, size: [0.1, 0.1, 0.1], material: stone, repeat: {count: 200, step: [0.2, 0, 0]}}
          - {name: b, shape: box, size: [0.1, 0.1, 0.1], material: stone, repeat: {count: 60, step: [0.2, 0, 0]}}
    """)
    assert "more than 256 parts" in e.message and "260" in e.message
    e = error("""
        parts:
          - {name: a, shape: box, size: [0.1, 0.1, 0.1], material: stone, repeat: {count: 0, step: [1, 0, 0]}}
    """)
    assert "count must be an integer >= 1" in e.message


@pytest.mark.parametrize("part,needle", [
    ("{name: a, shape: box, size: [1, 1, 1], round: 0.6, material: wood}", "round 0.6 is larger"),
    ("{name: a, shape: capsule, radius: 0.5, height: 0.8, material: wood}", "capsule height 0.8 is less than 2 * radius"),
    ("{name: a, shape: box, size: [1, -1, 1], material: wood}", "size must be > 0"),
    ("{name: a, shape: sphere, radius: 0, material: wood}", "radius must be > 0"),
    ("{name: a, shape: cylindre, radius: 1, height: 1, material: wood}", "unknown shape 'cylindre'"),
    ("{name: a, shape: sphere, size: [1, 1, 1], material: wood}", "shape 'sphere' does not take 'size'"),
    ("{name: a, shape: box, material: wood}", "shape 'box' needs 'size'"),
    ("{name: a, shape: box, size: [1, 1, 1]}", "solid part has no material"),
    ("{name: a, shape: box, size: [1, 1, 1], material: wood, at: [0, 0, 0], on: x.top}", "either 'at' or 'on'"),
    ("{name: A, shape: box, size: [1, 1, 1], material: wood}", "bad part name 'A'"),
    ("{name: a, shape: cylinder, radius: 1, height: 1, sides: 2, material: wood}", "sides must be an integer >= 3"),
    ("{name: a, shape: box, size: [1, 1, 1], material: wood, rotate: 45}", "rotate must be [rx, ry, rz]"),
    ("{name: a, shape: box, size: [1, 1, 1], material: wood, mirror: w}", "mirror must be x, y or z"),
])
def test_error_part_validation(part, needle):
    e = error(f"parts:\n  - {part}\n")
    assert needle in e.message
    assert "parts[0]" in e.where


def test_error_hints():
    e = error("parts:\n  - {name: a, shape: cylindre, radius: 1, height: 1, material: wood}\n")
    assert "did you mean 'cylinder'?" in e.hint
    e = error("parts:\n  - {name: a, shape: box, size: [1, 1, 1], round: 0.6, material: wood}\n")
    assert "min(size) / 2" in e.hint


def test_error_cut_rules():
    e = error("""
        parts:
          - {name: wall, shape: box, size: [2, 2, 0.2], material: brick}
          - {name: hole, shape: box, size: [0.5, 0.5, 1], material: brick, op: cut, on: wall.center}
    """)
    assert "a cut has no material" in e.message
    e = error("""
        parts:
          - {name: wall, shape: box, size: [2, 2, 0.2], material: brick}
          - {name: hole, shape: box, size: [0.5, 0.5, 1], op: cut, at: [0, 0, 0]}
    """)
    assert "a cut needs 'from: [parts]'" in e.message
    e = error("""
        parts:
          - {name: wall, shape: box, size: [2, 2, 0.2], material: brick}
          - {name: hole, shape: box, size: [0.5, 0.5, 1], op: cut, on: wall.center}
          - {name: hole2, shape: box, size: [0.5, 0.5, 1], op: cut, from: [hole]}
    """)
    assert "'hole' is a cut, not a solid part" in e.message


def test_cut_from_repeated_base_name_means_all_copies():
    a = load("""
        parts:
          - {name: tooth, shape: box, size: [0.2, 0.5, 0.2], material: bone, repeat: {count: 3, step: [0.5, 0, 0]}}
          - {name: notch, shape: box, size: [2, 0.1, 1], op: cut, from: [tooth]}
    """)
    assert a.part("notch").cut_from == ["tooth_0", "tooth_1", "tooth_2"]


def test_error_top_level():
    e = error("part:\n  - {name: a, shape: box, size: [1, 1, 1], material: wood}\n")
    assert e.message == "unknown top-level key 'part'" and "did you mean 'parts'?" in e.hint
    with pytest.raises(SightlineError) as info:
        load_asset_text("asset: other\nparts:\n  - {name: a, shape: sphere, radius: 1, material: wood}\n", FIX / "t.yaml", ROOT)
    assert "does not match the file name 't'" in info.value.message
    e = error("parts:\n  - {name: a, shape: box, size: [1, 1, 1], material: wood}\n  -  name: b\n   shape: box\n")
    assert "YAML syntax error" in e.message and "line" in e.where
    e = error("parts:\n  - name: a\n    name: b\n    shape: box\n")
    assert "duplicate key 'name'" in e.message and e.where.endswith("line 4")


def test_error_reference_image_missing():
    e = error("""
        parts:
          - {name: a, shape: sphere, radius: 1, material: wood}
        reference: {image: refs/nope.png, view: front}
    """)
    assert e.message == "reference image not found: refs/nope.png"
    assert "fixtures/refs/nope.png" in e.hint
    e = error("""
        parts:
          - {name: a, shape: sphere, radius: 1, material: wood}
        reference: {image: refs/tower_front.png, view: fornt}
    """)
    assert "did you mean 'front'?" in e.hint


def test_reference_orbit_view_and_list():
    a = load_asset(FIX / "knight.yaml", ROOT)
    (ref,) = a.references
    assert ref.view == "orbit:30:15" and ref.mask == "auto" and ref.height is None
    assert ref.lighting == "auto"


def test_reference_resolves_under_root_assets_and_takes_lighting():
    # An asset outside assets/ (a candidate in out/, a test fixture) names the picture as the library does.
    a = load("""
        parts:
          - {name: a, shape: sphere, radius: 1, material: wood}
        reference: {image: refs/lighthouse_concept.png, view: front, lighting: lit}
    """)
    (ref,) = a.references
    assert ref.image == (ROOT / "assets" / "refs" / "lighthouse_concept.png").resolve() and ref.lighting == "lit"
    e = error("""
        parts:
          - {name: a, shape: sphere, radius: 1, material: wood}
        reference: {image: refs/lighthouse_concept.png, lighting: shaded}
    """)
    assert e.message == "lighting must be auto, lit or flat, got 'shaded'" and "flat paint" in e.hint
    e = error("""
        parts:
          - {name: a, shape: sphere, radius: 1, material: wood}
        reference: {image: refs/nope.png}
    """)
    assert "assets/refs/nope.png" in e.hint  # the third place it looked


# ----------------------------------------------------------------------------- styles and roots


def test_default_style_library():
    style = load_style("default", ROOT)
    required = """wood dark_wood planks stone brick plaster roof slate_roof thatch metal iron gold bronze cloth_red
        cloth_blue leather skin hair leaf bark grass dirt sand water glass bone rope paint_white paint_black helper""".split()
    assert set(required) <= set(style.materials)
    assert style.material("wood").color == pytest.approx((0x9a / 255, 0x6a / 255, 0x3c / 255))
    assert style.material("stone").pattern == "bricks" and style.material("stone").scale == (0.6, 0.35)
    # light from top-left-front: travels toward +x, -y, -z, normalized
    assert np.linalg.norm(style.light_dir) == pytest.approx(1.0)
    assert style.light_dir[0] > 0 and style.light_dir[1] < 0 and style.light_dir[2] < 0
    assert style.render_views == ["front", "right", "top", "iso"] and style.outline_width == 1
    with pytest.raises(SightlineError) as info:
        style.material("wod")
    assert "did you mean 'wood'?" in info.value.hint


def test_partial_style_falls_back_to_default(scratch):
    (scratch / "styles").mkdir()
    (scratch / "styles" / "mini.yaml").write_text(textwrap.dedent("""
        style: mini
        palette: {ink: "#102030"}
        materials:
          wood: {color: ink, pattern: stripes, axis: x}
          neon: {color: "#00ff00", shine: 0.5}
        light: {ambient: 0.5}
    """), encoding="utf-8")
    style = load_style("mini", scratch)
    default = load_style("default", ROOT)
    assert style.name == "mini"
    assert style.material("wood").color == pytest.approx((0x10 / 255, 0x20 / 255, 0x30 / 255))
    assert style.material("wood").pattern_color == pytest.approx(tuple(c * 0.8 for c in style.material("wood").color))
    assert style.material("stone").color == default.material("stone").color  # inherited
    assert style.ambient == 0.5 and np.allclose(style.light_dir, default.light_dir)
    assert style.background == default.background and style.render_res == 256
    (scratch / "assets").mkdir()
    (scratch / "assets" / "lamp.yaml").write_text(
        "asset: lamp\nstyle: mini\nparts:\n  - {name: bulb, shape: sphere, radius: 0.2, material: neon}\n", encoding="utf-8")
    lamp = load_asset("lamp", scratch)
    assert lamp.style.name == "mini" and lamp.root == scratch.resolve()
    with pytest.raises(SightlineError) as info:
        load_asset("lamb", scratch)
    assert "did you mean 'lamp'?" in info.value.hint


@pytest.mark.parametrize("body,needle", [
    ("materials: {wood: {color: '#12345'}}", "bad color '#12345'"),
    ("materials: {wood: {color: oka}}", "unknown palette color 'oka'"),
    ("materials: {wood: {color: '#123456', pattern: plank}}", "unknown pattern 'plank'"),
    ("materials: {wood: {color: '#123456', colour: '#000000'}}", "unknown key 'colour'"),
    ("materials: {wood: {color: '#123456', shine: 2}}", "shine must be a number in [0, 1]"),
    ("light: {direction: [0, 0, 0]}", "light.direction must be a non-zero"),
    ("render: {views: [front, sidee]}", "unknown view 'sidee'"),
    ("lights: {}", "unknown key 'lights'"),
])
def test_style_errors(scratch, body, needle):
    (scratch / "styles").mkdir()
    (scratch / "styles" / "bad.yaml").write_text("style: bad\n" + body + "\n", encoding="utf-8")
    with pytest.raises(SightlineError) as info:
        load_style("bad", scratch)
    assert needle in info.value.message
    assert info.value.where.startswith("styles/bad.yaml")


def test_style_name_errors(scratch):
    (scratch / "styles").mkdir()
    (scratch / "styles" / "toon.yaml").write_text("style: cartoon\n", encoding="utf-8")
    with pytest.raises(SightlineError, match="does not match the file name"):
        load_style("toon", scratch)
    with pytest.raises(SightlineError) as info:
        load_style("tooon", scratch)
    assert "did you mean 'toon'?" in info.value.hint
    e = error("style: defualt\nparts:\n  - {name: a, shape: sphere, radius: 1, material: wood}\n")
    assert "unknown style 'defualt'" in e.message and e.where.startswith("tests/fixtures/t.yaml: style")
