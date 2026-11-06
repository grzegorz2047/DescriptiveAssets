"""Grid levels: loader validation, navigation bot, softlocks, order checks, sight, report and renders."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sightline.camera import fixed_camera
from sightline.errors import SightlineError
from sightline.levels import format_play, load_level, play, render_level, run_level_checks
from sightline.levels.loader import footprint_cells
from sightline.levels.render import PATH_RGB, _Iso, iso_frame
from sightline.levels.sight import line_of_sight

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "out" / "test_tmp" / "levels"


@pytest.fixture
def work(request) -> Path:
    """A throwaway project root on E: (levels/, styles/, assets/) for one test."""
    base = SCRATCH / request.node.name.replace("[", "_").replace("]", "")
    shutil.rmtree(base, ignore_errors=True)
    for sub in ("levels", "styles", "assets"):
        (base / sub).mkdir(parents=True)
    return base


def make(work: Path, body: str, name: str = "t"):
    path = work / "levels" / f"{name}.yaml"
    path.write_text(f"level: {name}\n" + textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return load_level(path, root=work)


def results_by_kind(level, report=None):
    report = report or play(level)
    return [(r.kind, r.passed, r) for r in run_level_checks(level, report)]


# ---------------------------------------------------------------- movement rules

def test_flat_corridor_path_is_compressed(work):
    level = make(work, """
        tiles: |
          . . . . .
        markers: {spawn: [0, 0], goal: [4, 0]}
    """)
    path = play(level)["path"]
    assert path["found"]
    assert path["length"] == 4.0 and path["horizontal"] == 4.0 and path["vertical"] == 0.0
    assert path["jumps"] == 0
    assert path["segments"] == ["walk E 4 cells (0,0)→(4,0)", "reach goal (4,0)"]


def test_jump_over_two_water_cells(work):
    level = make(work, """
        tiles: |
          . ~ ~ . .
        markers: {spawn: [0, 0], goal: [4, 0]}
    """)
    path = play(level)["path"]
    assert path["length"] == 4.0
    assert path["jumps"] == 1
    assert path["segments"][0] == "jump E over water 2 cells (0,0)→(3,0)"


def test_gap_wider_than_jump_gap_is_explained(work):
    level = make(work, """
        tiles: |
          . ~ ~ ~ .
        markers: {spawn: [0, 0], goal: [4, 0]}
    """)
    report = play(level)
    assert not report["path"]["found"]
    goal = report["reachability"]["places"]["goal"]
    assert not goal["reachable"]
    assert goal["reason"] == "way in to (4,0) crosses 3 cells of water > jump_gap 2"


@pytest.mark.parametrize("rise_steps, expect", [
    (1, ("walk E 1 cell (0,0)→(1,0), +0.50 m", 0)),
    (2, ("jump up +1.00 m E (0,0)→(1,0)", 1)),
    (3, None),
])
def test_rise_rules_step_hop_and_too_high(work, rise_steps, expect):
    level = make(work, f"""
        heights: |
          0 {rise_steps}
        tiles: |
          . .
        markers: {{spawn: [0, 0], goal: [1, 0]}}
    """)
    report = play(level)
    if expect is None:
        assert not report["path"]["found"]
        assert report["reachability"]["places"]["goal"]["reason"] == "way in to (1,0) needs rise 1.50 m > jump_up 1.00 m"
    else:
        segment, jumps = expect
        assert report["path"]["segments"][0] == segment
        assert report["path"]["jumps"] == jumps
        assert report["path"]["max_rise"] == pytest.approx(rise_steps * 0.5)


def test_ladder_ignores_height_difference(work):
    level = make(work, """
        heights: |
          0 3
        tiles: |
          = .
        markers: {spawn: [0, 0], goal: [1, 0]}
    """)
    path = play(level)["path"]
    assert path["segments"][0] == "climb +1.50 m (ladder) E (0,0)→(1,0)"
    assert path["ladder_climb"] == 1.5 and path["max_rise"] == 0.0 and path["jumps"] == 0


@pytest.mark.parametrize("top, reachable", [(6, True), (8, False)])
def test_drop_limit(work, top, reachable):
    level = make(work, f"""
        heights: |
          {top} 0
        tiles: |
          . .
        markers: {{spawn: [0, 0], goal: [1, 0]}}
    """)
    report = play(level)
    assert report["path"]["found"] is reachable
    if reachable:
        assert report["path"]["max_fall"] == 3.0
        assert report["path"]["segments"][0] == "drop -3.00 m E (0,0)→(1,0)"
    else:
        assert report["reachability"]["places"]["goal"]["reason"] == "way in to (1,0) drops 4.00 m > max_drop 3.00 m"


@pytest.mark.parametrize("wall, ok", [(2, True), (3, False)])
def test_jump_over_low_wall_only(work, wall, ok):
    level = make(work, f"""
        heights: |
          0 {wall} 0
        tiles: |
          . # .
        markers: {{spawn: [0, 0], goal: [2, 0]}}
    """)
    report = play(level)
    assert report["path"]["found"] is ok
    if ok:
        assert report["path"]["segments"][0] == "jump E over wall 1 cell (0,0)→(2,0)"
    else:
        assert report["reachability"]["places"]["goal"]["reason"] == (
            "way in to (2,0) is blocked by wall (1,0), top 1.50 m > takeoff + jump_up 1.00 m")


def test_custom_player_limits(work):
    level = make(work, """
        player: {jump_gap: 3, jump_up: 1.5}
        heights: |
          0 0 0 0 3
        tiles: |
          . ~ ~ ~ .
        markers: {spawn: [0, 0], goal: [4, 0]}
    """)
    path = play(level)["path"]
    assert path["found"] and path["jumps"] == 1
    assert path["length"] == pytest.approx(4.0 + 1.5)


# ---------------------------------------------------------------- keys, doors, softlocks

LOCKED_LEDGE = """
    heights: |
      4 4 4 0 0 0
    tiles: |
      . k . . D .
    doors: {door_a: {at: [4, 0], key: key_a}}
    markers: {spawn: [2, 0], key_a: [1, 0], goal: [5, 0]}
"""


def test_key_then_door_path_and_locked_softlock(work):
    level = make(work, LOCKED_LEDGE)
    report = play(level)
    path = report["path"]
    assert path["segments"] == [
        "walk W 1 cell (2,0)→(1,0)",
        "pick key_a (1,0)",
        "walk E 1 cell (1,0)→(2,0)",
        "drop -2.00 m E (2,0)→(3,0)",
        "walk E 1 cell (3,0)→(4,0)",
        "open door_a with key_a (4,0)",
        "walk E 1 cell (4,0)→(5,0)",
        "reach goal (5,0)",
    ]
    assert path["length"] == 7.0 and path["keys"] == ["key_a"] and path["doors"] == ["door_a"]
    soft = report["softlocks"]
    assert soft["goal_reachable"] and len(soft["regions"]) == 1
    region = soft["regions"][0]
    assert region["cells"] == [[3, 0]]
    assert region["keys_held"] == ["none"]  # stuck only when dropping down without the key
    assert region["entered_by"] == "drop -2.00 m E (2,0)→(3,0)"
    assert region["reason"] == "exit through door_a (4,0) needs key_a"


def test_softlock_pit_reports_cells_and_rise(work):
    body = """
        heights: |
          3 3 3 3
          0 0 3 3
        tiles: |
          . . . .
          . . {ladder} .
        markers: {{spawn: [0, 0], goal: [3, 1]}}
        checks:
          - no_softlocks
    """
    level = make(work, body.format(ladder="."))
    report = play(level)
    regions = report["softlocks"]["regions"]
    assert [r["cells"] for r in regions] == [[[0, 1], [1, 1]]]
    assert regions[0]["reason"] == "exit to (0,0) needs rise 1.50 m > jump_up 1.00 m"
    assert regions[0]["entered_by"] == "drop -1.50 m S (0,0)→(0,1)"
    (kind, passed, res), = results_by_kind(level, report)
    assert kind == "no_softlocks" and not passed
    assert "needs rise 1.50 m > jump_up 1.00 m" in res.message
    # A ladder out of the pit removes the softlock.
    fixed = make(work, body.format(ladder="="), name="t2")
    assert play(fixed)["softlocks"]["regions"] == []


def test_goal_unreachable_is_not_reported_as_many_softlocks(work):
    level = make(work, """
        tiles: |
          . . # .
        heights: |
          0 0 9 0
        markers: {spawn: [0, 0], goal: [3, 0]}
        checks:
          - no_softlocks
    """)
    report = play(level)
    assert not report["softlocks"]["goal_reachable"]
    text = format_play(report)
    assert "SOFTLOCKS  goal unreachable from spawn" in text
    (_, passed, res), = results_by_kind(level, report)
    assert not passed and res.message == "goal unreachable from spawn"


# ---------------------------------------------------------------- order check

def test_order_passes_when_each_item_gates_the_next(work):
    level = make(work, """
        tiles: |
          . k . D .
        doors: {door_a: {at: [3, 0], key: key_a}}
        markers: {spawn: [0, 0], key_a: [1, 0], goal: [4, 0]}
        checks:
          - order: [key_a, door_a, goal]
    """)
    (_, passed, res), = results_by_kind(level)
    assert passed, res.line()


def test_order_fails_when_door_can_be_bypassed(work):
    level = make(work, """
        tiles: |
          . k . D .
          . . . . .
        doors: {door_a: {at: [3, 0], key: key_a}}
        markers: {spawn: [0, 0], key_a: [1, 0], goal: [4, 0]}
        checks:
          - order: [key_a, door_a, goal]
    """)
    (_, passed, res), = results_by_kind(level)
    assert not passed
    assert res.measured == "goal reachable without door_a"  # key_a -> door_a still holds
    assert res.data["not_required"] == ["goal reachable without door_a"]


def test_order_fails_when_not_reachable_in_that_order(work):
    level = make(work, """
        heights: |
          4 4 0
        tiles: |
          . . .
        markers: {spawn: [0, 0], high: [1, 0], low: [2, 0]}
        checks:
          - order: [low, high]
    """)
    (_, passed, res), = results_by_kind(level)
    assert not passed and res.measured == "high unreachable after low"


def test_check_variants_measure_numbers(work):
    level = make(work, LOCKED_LEDGE + """
    checks:
      - reachable: {to: goal, from: key_a}
      - path_length: {min: 8}
      - path_length: 7
      - jumps: 0
      - max_fall: {max: 1.5, why: "falls above 1.5 m hurt"}
      - visible: {from: [0, 0], to: key_a}
    """)
    res = run_level_checks(level, play(level))
    assert [(r.kind, r.passed) for r in res] == [("reachable", True), ("path_length", False), ("path_length", True),
                                                 ("jumps", True), ("max_fall", False), ("visible", True)]
    assert res[1].measured == "7.00 m" and res[1].expected == ">= 8 m"
    assert res[4].measured == "2.00 m" and res[4].why == "falls above 1.5 m hurt"
    assert "(why: falls above 1.5 m hurt)" in res[4].line()


def test_level_without_goal_reports_reachability_only(work):
    level = make(work, """
        tiles: |
          . . ~ .
        markers: {spawn: [0, 0], chest: [3, 0]}
        checks:
          - reachable: chest
    """)
    report = play(level)
    text = format_play(report)
    assert "PATH  none: no goal marker" in text and "SOFTLOCKS  not checked: no goal marker" in text
    assert report["reachability"]["places"]["chest"]["distance"] == 3.0
    (_, passed, _), = results_by_kind(level, report)
    assert passed


# ---------------------------------------------------------------- line of sight

@pytest.mark.parametrize("wall, visible", [(4, False), (2, True)])
def test_line_of_sight_over_walls(work, wall, visible):
    level = make(work, f"""
        heights: |
          0 0 {wall} 0 0 0
        tiles: |
          . . # . . .
        markers: {{spawn: [0, 0], goal: [5, 0]}}
        checks:
          - not_visible: {{from: spawn, to: goal}}
    """)
    s = line_of_sight(level, (0, 0), (5, 0))
    assert s["distance"] == 5.0 and s["eye_height"] == pytest.approx(1.7)
    assert s["visible"] is visible
    if visible:
        assert s["clearance"] == {"meters": pytest.approx(0.7), "cell": [2, 0]}
    else:
        assert s["blocked_by"] == {"cell": [2, 0], "what": "wall", "top": 2.0, "ray_y": pytest.approx(1.7),
                                   "at_distance": 1.5}
    (_, passed, _), = results_by_kind(level)
    assert passed is (not visible)


def test_eye_height_follows_player_height(work):
    level = make(work, """
        player: {height: 1.0}
        heights: |
          0 0 2 0 0
        tiles: |
          . . # . .
        markers: {spawn: [0, 0], goal: [4, 0]}
    """)
    s = line_of_sight(level, (0, 0), (4, 0))
    assert not s["visible"] and s["blocked_by"]["ray_y"] == pytest.approx(0.9)


# ---------------------------------------------------------------- objects

def test_solid_placeholder_object_blocks_and_warns(work):
    body = """
        tiles: |
          . . . . .
        markers: {{spawn: [0, 0], goal: [4, 0]}}
        objects:
          - {{asset: crate, at: [2, 0], solid: {solid}}}
    """
    level = make(work, body.format(solid="true"))
    assert level.objects[0].placeholder and level.objects[0].cells == [(2, 0)]
    assert any("placeholder" in w for w in level.warnings)
    report = play(level)
    assert not report["path"]["found"]
    assert report["reachability"]["places"]["goal"]["reason"] == "way in to (3,0) is blocked by solid object crate (2,0)"
    assert play(make(work, body.format(solid="false"), name="t2"))["path"]["found"]


def test_footprint_cells_follow_rotation_and_report_clipping():
    lo, hi = np.array([-1.5, 0.0, -0.5]), np.array([1.5, 1.0, 0.5])
    assert footprint_cells(5, 5, 1.0, 0, lo, hi, 10, 10) == ([(4, 5), (5, 5), (6, 5)], [])
    assert footprint_cells(5, 5, 1.0, 90, lo, hi, 10, 10) == ([(5, 4), (5, 5), (5, 6)], [])
    assert footprint_cells(0, 0, 1.0, 0, lo, hi, 10, 10) == ([(0, 0), (1, 0)], [(-1, 0)])
    # 45 degrees: the 3 x 1 m box reaches the diagonal neighbours' centers (1.41 m away, inside 1.5 m).
    cells, _ = footprint_cells(5, 5, 1.0, 45, lo, hi, 10, 10)
    assert cells == [(6, 4), (5, 5), (4, 6)]


def test_footprint_counts_half_covered_cells():
    """A 2 x 2 m object centered on a cell covers half of each neighbour: they block too (3 x 3 cells)."""
    lo, hi = np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0])
    cells, outside = footprint_cells(2, 2, 1.0, 0, lo, hi, 5, 5)
    assert cells == [(c, r) for r in (1, 2, 3) for c in (1, 2, 3)] and outside == []
    # A 1 x 1 m crate stays in its own cell.
    small = footprint_cells(2, 2, 1.0, 0, np.array([-0.5, 0, -0.5]), np.array([0.5, 1, 0.5]), 5, 5)
    assert small == ([(2, 2)], [])


def test_yaw_matrix_matches_asset_rotation():
    from sightline.levels.model import LevelObject
    from sightline.model import rotation_matrix

    for deg in (0, 30, 90, -45, 270):
        obj = LevelObject(index=0, asset="a", col=0, row=0, rotate=deg)
        assert np.allclose(obj.yaw_matrix(), rotation_matrix(0, deg, 0), atol=1e-12)


CRATE = "asset: crate\nparts:\n  - name: body\n    shape: box\n    size: [3.0, 1.2, 1.0]\n    material: wood\n    align: bottom\n"


def test_object_footprint_from_a_real_asset(work):
    """Objects load through sightline.loader.load_asset: footprint and height come from the asset's geometry."""
    (work / "assets" / "crate.yaml").write_text(CRATE, encoding="utf-8")
    level = make(work, """
        tiles: |
          . . . . . . .
          . . . . . . .
          . . . . . . .
        markers: {spawn: [0, 1], goal: [6, 1]}
        objects:
          - {asset: crate, at: [3, 1]}
    """)
    obj = level.objects[0]
    assert not obj.placeholder and obj.asset_obj.name == "crate" and level.warnings == []
    assert obj.cells == [(2, 1), (3, 1), (4, 1)] and obj.height == pytest.approx(1.2)
    assert np.allclose(obj.bounds[0], [-1.5, 0, -0.5]) and np.allclose(obj.bounds[1], [1.5, 1.2, 0.5])
    report = play(level)
    assert report["path"]["length"] == 8.0  # 6 cells east plus a detour of one row and back
    assert report["sight"][0]["clearance"]["meters"] == pytest.approx(0.5)  # eye 1.7 m over the 1.2 m crate


WINDMILL = """asset: windmill
parts:
  - {name: base, shape: box, size: [1.0, 3.0, 1.0], material: stone, align: bottom}
  - {name: sail, shape: box, size: [5.0, 1.0, 0.1], material: wood, at: [0, 4.0, 0.3], align: bottom}
"""


def test_overhang_is_walkable_and_sight_passes_under_it(work):
    """A 5 m sail 4 m up over a 1 m base: only the base cell blocks; sight at eye height passes under the sail."""
    (work / "assets" / "windmill.yaml").write_text(WINDMILL, encoding="utf-8")
    level = make(work, """
        tiles: |
          . . . . . . .
          . . . . . . .
          . . . . . . .
        markers: {spawn: [0, 1], goal: [6, 1], probe: [2, 1]}
        objects:
          - {asset: windmill, at: [3, 1]}
    """)
    obj = level.objects[0]
    assert obj.cells == [(3, 1)]  # the AABB footprint is 5 cells wide, only the base reaches the ground
    assert obj.cover == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    (lo, hi), = obj.columns[(2, 1)]  # one span: the sail, 4..5 m (to a voxel of 5/64 m; the top is exact)
    assert lo == pytest.approx(4.0, abs=5 / 64) and hi == 5.0 and obj.columns[(3, 1)][0][0] == 0.0
    report = play(level)
    assert report["objects"][0]["overhead"] == [[1, 1], [2, 1], [4, 1], [5, 1]]
    assert report["path"]["length"] == 8.0  # around the base only: (2,1) and (4,1) are walkable
    # Spawn -> probe passes under the sail at 1.7 m; a line over the base is blocked by it.
    s = line_of_sight(level, (0, 1), (2, 1))
    assert s["visible"] and s["clearance"]["meters"] == pytest.approx(1.7)
    s = line_of_sight(level, (0, 1), (6, 1))
    assert not s["visible"] and s["blocked_by"]["what"] == "object windmill" and s["blocked_by"]["cell"] == [3, 1]
    assert "+ 4 walkable under overhangs" in format_play(report)
    # top.png fills the blocking cell only: sample near each cell's lower-left corner (labels sit at the
    # centers); cell (c, r) spans x 36 + 40 c.., y 30 + 40 r.. at 1x.
    from sightline.levels.render import OBJECT_RGB

    top = Image.open(render_level(level, work / "out")["top"]).convert("RGB")
    corner = [top.getpixel((int(36 + (c + 0.1) * 40), int(30 + 1.85 * 40))) for c in range(7)]
    assert [c for c in range(7) if corner[c] == OBJECT_RGB] == [3]


def test_rotated_object_hanging_off_the_grid_warns(work):
    (work / "assets" / "crate.yaml").write_text(CRATE, encoding="utf-8")
    level = make(work, """
        tiles: |
          . . . .
          . . . .
          . . . .
        markers: {spawn: [0, 0], goal: [0, 2]}
        objects:
          - {asset: crate, at: [3, 0], rotate: 90}
    """)
    assert level.objects[0].cells == [(3, 0), (3, 1)]
    warning = "objects[0] (crate): hangs off the grid by 1 cell(s) (3,-1); the bot and sight ignore them"
    assert level.warnings[0].startswith(warning)
    report = play(level)
    assert report["warnings"][0] == level.warnings[0]
    assert "WARNINGS\n  - " + warning in format_play(report)


# ---------------------------------------------------------------- validation

INVALID = [
    ("tiles: |\n  . . .\n  . .\nmarkers: {spawn: [0, 0]}\n", "row 1 has 2 cells but row 0 has 3", "tiles row 1", "same number"),
    ("tiles: |\n  . X .\nmarkers: {spawn: [0, 0]}\n", "unknown tile 'X'", "tiles row 0, col 1", "'.' floor"),
    ("tiles: |\n  . . .\nmarkers: {spawn: [0, 0], goal: [7, 0]}\n", "outside the 3 x 1 grid", "markers.goal", "columns are 0..2"),
    ("tiles: |\n  . # .\nmarkers: {spawn: [0, 0], goal: [1, 0]}\n", "is on a wall cell (1,0)", "markers.goal", "move it"),
    ("tiles: |\n  . . .\nmarkers: {spawn: [0, 0]}\ndoors: {door_a: {at: [1, 0]}}\n", "not a door tile", "doors.door_a.at", "door tile (D)"),
    ("tiles: |\n  . . .\nmarkers: {spawn: [0, 0], goal: [2, 0]}\nchecks:\n  - reachabel: goal\n",
     "unknown check kind 'reachabel'", "checks[0]", "did you mean 'reachable'?"),
    ("tiles: |\n  . . .\nmarkers: {spawn: [0, 0], goal: [2, 0]}\nchecks:\n  - reachable: gaol\n",
     "unknown marker or door 'gaol'", "checks[0] (reachable)", "did you mean 'goal'?"),
    ("tiles: |\n  . . .\nmarkers: {spawn: [0, 0], goal: [2, 0]}\nchecks:\n  - visible: {from: spawn, too: goal}\n",
     "unknown key 'too'", "checks[0] (visible)", "did you mean 'to'?"),
    ("tile: |\n  . . .\n", "unknown key 'tile'", "levels/t.yaml", "did you mean 'tiles'?"),
    ("tiles: |\n  . . .\nheights: |\n  0 0\nmarkers: {spawn: [0, 0]}\n", "heights is 2 x 1", "heights", "same grid"),
    ("tiles: |\n  . . .\nheights: |\n  0 1.5 0\nmarkers: {spawn: [0, 0]}\n", "height '1.5' is not an integer", "heights row 0, col 1", "integer steps"),
    ("tiles: |\n  . D .\nmarkers: {spawn: [0, 0], key_red: [2, 0]}\ndoors: {door_a: {at: [1, 0], key: key_rde}}\n",
     "door key 'key_rde' is not a marker", "doors.door_a.key", "did you mean 'key_red'?"),
    ("tiles: |\n  . k .\nmarkers: {spawn: [0, 0]}\n", "key tile at (1,0) has no marker", "tiles row 0, col 1", "markers: {key_name: [1, 0]}"),
    ("tiles: |\n  % . .\nlegend: {'%': flor}\nmarkers: {spawn: [1, 0]}\n", "unknown tile kind 'flor'", "legend.%", "did you mean 'floor'?"),
    ("tiles: |\n  . . .\nplayer: {jump_upp: 2}\nmarkers: {spawn: [0, 0]}\n", "unknown key 'jump_upp'", "player", "did you mean 'jump_up'?"),
    ("tiles: |\n  . . .\nmarkers: {goal: [0, 0]}\n", "must include `spawn`", "markers", "spawn: [0, 0]"),
    ("tiles: |\n  . . .\nmarkers: {spawn: [0, 0]}\nchecks:\n  - no_softlocks\n", "no_softlocks needs a goal marker", "checks[0]", "goal"),
    ("tiles: [\n", "YAML syntax error", "line", "indentation"),
    # Shared YAML rules (SPEC 0): duplicate keys are errors with line numbers, and so are all locations.
    ("tiles: |\n  . D .\nmarkers: {spawn: [0, 0], key_a: [2, 0]}\ndoors: {door_a: {at: [1, 0], key: key_a}}\ndoors: {}\n",
     "duplicate key 'doors' (first at line 5)", "levels/t.yaml, line 6", "each key may appear once"),
    ("tiles: |\n  . . .\n  . k .\nmarkers: {spawn: [0, 0]}\n", "key tile at (1,1) has no marker",
     "tiles row 1, col 1, line 4", "markers: {key_name: [1, 1]}"),
    ("tiles: |\n  . . .\nmarkers: {spawn: [0, 0], goal: [7, 0]}\n", "outside the 3 x 1 grid", "markers.goal, line 4",
     "columns are 0..2"),
]


def test_yaml_words_stay_text_in_levels(work):
    level = make(work, """
        tiles: |
          . . .
        markers: {spawn: [0, 0], no: [1, 0], goal: [2, 0]}
    """)
    assert list(level.markers) == ["spawn", "no", "goal"]


def test_unreachable_goal_is_warned_once(work):
    level = make(work, """
        tiles: |
          . _ _ _ .
        markers: {spawn: [0, 0], goal: [4, 0]}
    """)
    warnings = play(level)["warnings"]
    assert warnings.count("goal unreachable from spawn") == 1


@pytest.mark.parametrize("body, message, where, hint", INVALID, ids=[case[1][:30] for case in INVALID])
def test_invalid_levels_raise_sightline_errors(work, body, message, where, hint):
    path = work / "levels" / "t.yaml"
    path.write_text("level: t\n" + body, encoding="utf-8")
    with pytest.raises(SightlineError) as info:
        load_level(path, root=work)
    err = info.value
    assert message in err.message
    assert err.where and where in err.where and err.where.startswith("levels/t.yaml")
    assert err.hint and hint in err.hint


def test_level_name_must_match_file(work):
    path = work / "levels" / "t.yaml"
    path.write_text("level: other\ntiles: |\n  . .\nmarkers: {spawn: [0, 0]}\n", encoding="utf-8")
    with pytest.raises(SightlineError, match="must equal the file name 't'"):
        load_level(path, root=work)


def test_unknown_level_name_suggests(work):
    make(work, "tiles: |\n  . .\nmarkers: {spawn: [0, 0]}\n", name="courtyard")
    with pytest.raises(SightlineError) as info:
        load_level("courtyrd", root=work)
    assert info.value.hint == "did you mean 'courtyard'?"


def test_spec_example_level_parses_and_passes(work):
    """The YAML from SPEC section 10 (comments inside the heights block included)."""
    level = make(work, """
        brief: spec example
        style: default
        cell: 1.0
        height_unit: 0.5
        player: {step: 0.5, jump_up: 1.0, jump_gap: 2, max_drop: 3.0, height: 1.8}
        heights: |                     # rows = z (row 0 = north/back, increasing toward +Z/front),
          0 0 0 0 1 1                  # columns = x (col 0 = west/left); integers, space-separated
          0 0 0 0 1 1
        tiles: |                       # same shape; one char per cell, space-separated
          . . ~ ~ . .
          . . ~ ~ D .
        legend: {"%": floor}
        markers: {spawn: [0, 0], goal: [5, 1], key_red: [1, 1]}
        doors: {door_red: {at: [4, 1], key: key_red}}
        objects:
          - {asset: watchtower, at: [5, 0], rotate: 90, solid: true}
        checks:
          - reachable: goal
          - order: [key_red, door_red, goal]
          - no_softlocks
          - visible: {from: spawn, to: goal}
          - path_length: {max: 80}
          - jumps: {min: 1, max: 5}
          - max_fall: 3.0
    """)
    assert level.heights.tolist() == [[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]]
    results = results_by_kind(level)
    assert [(k, p) for k, p, _ in results] == [(k, True) for k, _, _ in results], [r.line() for _, _, r in results]


# ---------------------------------------------------------------- example levels, report, renders

def test_courtyard_example_checks_pass():
    level = load_level("courtyard", root=ROOT)
    report = play(level)
    failed = [r.line() for r in run_level_checks(level, report) if not r.passed]
    assert failed == []
    path = report["path"]
    assert path["keys"] == ["key_red"] and path["doors"] == ["door_red"]
    assert path["jumps"] >= 1 and path["ladder_climb"] == 1.5 and path["max_fall"] == 1.5
    assert report["softlocks"]["regions"] == [] and report["reachability"]["unreachable"] == []


def test_broken_bridge_example_detects_its_bugs():
    level = load_level("broken_bridge", root=ROOT)
    report = play(level)
    verdicts = [(r.kind, r.passed) for r in run_level_checks(level, report)]
    assert verdicts == [("reachable", True), ("reachable", False), ("no_softlocks", False), ("order", False),
                        ("visible", True), ("jumps", True), ("max_fall", True)]
    region, = report["softlocks"]["regions"]
    assert region["count"] == 24 and region["keys_held"] == ["none", "key_blue"]
    assert region["entered_by"] == "drop -2.00 m E (2,6)→(3,6)"
    assert region["reason"] == "exit to (11,6) needs rise 2.00 m > jump_up 1.00 m"
    secret = report["reachability"]["places"]["secret"]
    assert secret["reason"] == "way in to (6,1) crosses 3 cells of pit > jump_gap 2"
    assert report["reachability"]["regions"][0]["cells"] == [[6, 1], [7, 1]]


def test_format_play_sections_rulers_and_map():
    report = play(load_level("broken_bridge", root=ROOT))
    text = format_play(report)
    for heading in ("PATH  ", "REACHABILITY  ", "SOFTLOCKS  ", "SIGHT  ", "MAP  ", "HEIGHTS  "):
        assert f"\n{heading}" in text
    lines = text.splitlines()
    m = lines.index(next(ln for ln in lines if ln.startswith("MAP  ")))
    assert lines[m + 1].rstrip().endswith("1 1 1 1")  # tens ruler for columns 10..13
    assert lines[m + 2].strip() == "0 1 2 3 4 5 6 7 8 9 0 1 2 3"
    assert lines[m + 3 + 4] == "  4  . S * * * * _ _ * * * * * *"  # row 4: spawn, path, broken bridge
    assert "S spawn (1,4)" in text and "D door_blue (12,5) needs key_blue" in text


def test_report_is_json_and_deterministic():
    level = load_level("courtyard", root=ROOT)
    a = json.dumps(play(level), sort_keys=True)
    b = json.dumps(play(load_level("courtyard", root=ROOT)), sort_keys=True)
    assert a == b
    assert json.loads(a)["level"] == "courtyard"


def test_renders_are_written_legible_and_deterministic():
    level = load_level("broken_bridge", root=ROOT)
    out1 = render_level(level, SCRATCH / "render_a")
    out2 = render_level(level, SCRATCH / "render_b")
    assert set(out1) == {"iso", "top"}
    for view in ("iso", "top"):
        assert out1[view].name == f"{view}.png"
        assert out1[view].read_bytes() == out2[view].read_bytes()
    iso = np.asarray(Image.open(out1["iso"]).convert("RGB")).astype(int)
    assert (np.abs(iso - np.array(PATH_RGB)).sum(axis=2) < 30).sum() > 200  # the critical path is drawn
    top = Image.open(out1["top"]).convert("RGB")
    # Top view layout: 36 px left margin, 30 px top margin, 40 px per cell -> center of pit cell (4, 2).
    r, g, b = top.getpixel((36 + 4 * 40 + 20, 30 + 2 * 40 + 20))
    assert max(r, g, b) < 70  # pits are dark
    r, g, b = top.getpixel((36 + 0 * 40 + 20, 30 + 8 * 40 + 20))
    assert r > b  # cliff floor is warm, not blue


def test_iso_composites_sprite_at_its_anchor(work, monkeypatch):
    """With an asset renderer present, the sprite's anchor pixel lands on the object's cell center at terrain height."""
    calls = []

    def fake_render_sprite(asset, view="iso", mpp=None):
        calls.append((view, mpp))
        img = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
        return img, (10, 30)  # anchor = bottom center = asset origin

    fake = type(sys)("sightline.render")
    fake.render_sprite = fake_render_sprite
    monkeypatch.setitem(sys.modules, "sightline.render", fake)
    level = make(work, """
        heights: |
          0 0 0 0
          0 0 2 0
          0 0 0 0
        tiles: |
          . . . .
          . . . .
          . . . .
        markers: {spawn: [0, 2], goal: [0, 0]}
        objects:
          - {asset: statue, at: [2, 1]}
    """)
    level.objects[0].asset_obj = object()  # pretend the asset loaded
    out = render_level(level, work / "out")
    iso, _ = iso_frame(level)
    assert calls == [("iso", iso.mpp)]
    img = np.asarray(Image.open(out["iso"]).convert("RGB")).astype(int)
    red = (img[..., 0] > 240) & (img[..., 1] < 25) & (img[..., 2] < 25)  # pure sprite red, not the goal pin
    ys, xs = np.nonzero(red)
    assert 400 < red.sum() < 700  # ~20 x 30 px, edges blended by the 2x downsample
    (ax, ay), = iso.px([level.world(2, 1)], scale=1)  # cell center on top of its 1.0 m column
    assert abs((xs.min() + xs.max() + 1) / 2 - ax) <= 1.0
    assert abs((ys.max() + 1) - ay) <= 1.0


def fake_renderer(monkeypatch, render_sprite=None, **attrs):
    """Install a stand-in `sightline.render` module (tests never depend on the real asset renderer)."""
    fake = type(sys)("sightline.render")
    if render_sprite is not None:
        fake.render_sprite = render_sprite
    for k, v in attrs.items():
        setattr(fake, k, v)
    monkeypatch.setitem(sys.modules, "sightline.render", fake)


def test_rotated_asset_reaches_renderer_and_stays_in_frame(work, monkeypatch):
    """The sprite is rendered from a turned copy of the asset, and the frame covers the part off the grid."""
    seen = []

    def fake_render_sprite(asset, view="iso", mpp=None):
        seen.append(asset.scene().bounds(include_helpers=False))
        return Image.new("RGBA", (12, 12), (0, 200, 0, 255)), (6, 12)

    fake_renderer(monkeypatch, fake_render_sprite)
    (work / "assets" / "crate.yaml").write_text(CRATE, encoding="utf-8")
    level = make(work, """
        tiles: |
          . . . .
          . . . .
          . . . .
        markers: {spawn: [0, 0], goal: [0, 2]}
        objects:
          - {asset: crate, at: [3, 0], rotate: 90}
    """)
    render_level(level, work / "out")
    (lo, hi), = seen
    assert np.allclose(lo, [-0.5, 0, -1.5]) and np.allclose(hi, [0.5, 1.2, 1.5])  # 3 m length now along z
    original = level.objects[0].asset_obj.scene().bounds(include_helpers=False)
    assert np.allclose(original[0], [-1.5, 0, -0.5])  # the loaded asset itself is untouched
    iso, _ = iso_frame(level)
    corners = np.array(iso.px(level.objects[0].world_corners(level.cell), scale=1))
    assert corners.min() >= 0 and (corners[:, 0] <= iso.width).all() and (corners[:, 1] <= iso.height).all()


def test_sprite_renderer_fallbacks(work, monkeypatch):
    from sightline.levels.render import _object_sprite, sprite_renderer

    monkeypatch.setitem(sys.modules, "sightline.render", None)  # module not written yet
    fn, note = sprite_renderer()
    assert fn is None and note == "objects drawn as footprint boxes: no sprite renderer (sightline.render) yet"
    fake_renderer(monkeypatch, other=1)  # module exists, render_sprite not yet
    fn, note = sprite_renderer()
    assert fn is None and "failed to import" in note and "render_sprite" in note

    def broken(asset, view="iso", mpp=None):
        raise ValueError("not enough values to unpack")

    fake_renderer(monkeypatch, broken)
    fn, note = sprite_renderer()
    assert fn is broken and note is None
    (work / "assets" / "crate.yaml").write_text(CRATE, encoding="utf-8")
    level = make(work, """
        tiles: |
          . . . . .
        markers: {spawn: [0, 0], goal: [4, 0]}
        objects:
          - {asset: crate, at: [2, 0], solid: false}
    """)
    assert _object_sprite(level.objects[0], 0.025, fn) == "crate: sprite failed (ValueError: not enough values to unpack)"
    out = render_level(level, work / "out")  # still renders, the object as a footprint box
    img = np.asarray(Image.open(out["iso"]).convert("RGB")).astype(int)
    assert (np.abs(img - np.array([190, 168, 132])).sum(axis=2) < 40).sum() > 100  # OBJECT_RGB box faces


@pytest.mark.parametrize("wall", [0, 6])
def test_iso_path_is_hidden_behind_nearer_walls(work, wall):
    """A path behind a 3 m wall is only hinted at with thin dashes; behind a flat row it is drawn in full."""
    tile = "#" if wall else "."
    level = make(work, f"""
        heights: |
          0 0 0 0 0 0
          {' '.join([str(wall)] * 6)}
        tiles: |
          . . . . . .
          {' '.join([tile] * 6)}
        markers: {{spawn: [0, 0], goal: [5, 0]}}
    """, name=f"w{wall}")
    out = render_level(level, work / "out")
    img = np.asarray(Image.open(out["iso"]).convert("RGB")).astype(int)
    iso, _ = iso_frame(level)
    xs = np.linspace(1.0, 5.0, 81)  # along row 0 at the path's 0.12 m lift, away from the pins
    pts = np.array(iso.px(np.stack([xs, np.full_like(xs, 0.12), np.full_like(xs, 0.5)], axis=1), scale=1))
    d = pts[-1] - pts[0]
    normal = np.array([-d[1], d[0]]) / np.hypot(*d)

    def sample(points):
        return img[np.round(points[:, 1] - 0.5).astype(int), np.round(points[:, 0] - 0.5).astype(int)]

    core = float(np.mean(np.abs(sample(pts) - np.array(PATH_RGB)).sum(axis=1) < 60))
    # White halo: some pixel 2..3.5 px off the centerline (between the 3 px core and the 7 px halo edge) on each side.
    halo = float(np.mean([np.any([sample(pts + sign * k * normal).min(axis=1) >= 245 for k in (2.0, 2.5, 3.0, 3.5)], axis=0)
                          for sign in (-1, 1)]))
    if wall:
        assert 0.3 < core < 0.75 and halo < 0.05  # dashes over the wall, no halo
    else:
        assert core > 0.95 and halo > 0.9


def test_labels_avoid_blocked_areas_and_each_other():
    from sightline.levels.render import _font, _Labels

    img = Image.new("RGB", (240, 120), (255, 255, 255))
    labels = _Labels(img, _font(13))
    labels.block((100, 40, 140, 80))  # e.g. an object sprite
    boxes = [labels.beside(120, 60, 7, "spawn"), labels.beside(120, 60, 7, "goal"), labels.above(120, 40, "statue"),
             labels.centered(120, 60, "crate")]
    labels.draw_all()
    for i, a in enumerate(boxes):
        assert 0 <= a[0] and 0 <= a[1] and a[2] <= 240 and a[3] <= 120
        assert labels.overlap(a) == pytest.approx((a[2] - a[0]) * (a[3] - a[1]))  # only with itself
        for b in boxes[i + 1:]:
            assert min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1])


def test_caption_lines_wrap_to_the_image_width():
    from PIL import ImageDraw

    from sightline.levels.render import _font, wrap_text

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    text = "t · iso · 1 px = 0.025 m · grid 1 m · path 2.00 m, 0 jump(s) · a_very_long_unbreakable_token_here"
    lines = wrap_text(draw, text, _font(13), 200)
    assert len(lines) > 1 and " ".join(lines).replace("  ", " ")
    assert all(draw.textlength(ln, font=_font(13)) <= 200 or " " not in ln for ln in lines)
    assert "".join(lines).replace(" · ", "").replace(" ", "") == text.replace(" · ", "").replace(" ", "")


def test_tile_colors_come_from_the_level_style(work):
    from sightline.levels.render import DEFAULT_COLORS, level_look

    (work / "styles" / "tiles.yaml").write_text(
        'style: tiles\npalette: {wall: "#405060"}\nmaterials:\n  water: {color: "#102030"}\nbackground: "#ffffff"\n',
        encoding="utf-8")
    level = make(work, "style: tiles\ntiles: |\n  . ~ #\nmarkers: {spawn: [0, 0]}\n")
    look = level_look(level)
    assert look.colors["water"] == (16, 32, 48) and look.colors["wall"] == (64, 80, 96)
    assert look.colors["floor"] == DEFAULT_COLORS["floor"] and look.background == (255, 255, 255)
    assert look.source == "style tiles"


def test_iso_projection_matches_camera_view_basis():
    level = load_level("courtyard", root=ROOT)
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 1.5, 2.0], [16.0, -0.5, 11.0], [5.5, 3.0, 0.5]])
    iso = _Iso(level, pts)
    cam = fixed_camera("iso", center=(0, 0, 0), mpp=iso.mpp, width=100, height=100, near=50)
    ours = np.array(iso.px(pts, scale=1))
    theirs, _ = cam.project(pts)
    assert np.allclose(ours - ours[0], theirs - theirs[0], atol=1e-9)
