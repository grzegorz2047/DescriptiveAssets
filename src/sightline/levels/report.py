"""`play(level)`: walk the level with the bot and collect measured facts; `format_play` prints them.

Sections: PATH, REACHABILITY, SOFTLOCKS, SIGHT, MAP (ASCII with rulers, markers as letters,
critical path as `*`), HEIGHTS. The JSON dict and the text carry the same facts.
"""

from __future__ import annotations

import string

import numpy as np

from .checks import validate_checks
from .model import Level
from .nav import analyze, path_report, reachability_report, softlock_report
from .sight import sight_between

PATH_CHAR = "*"


def _sight_pairs(level: Level) -> list[tuple]:
    """spawn -> goal plus every visible/not_visible check pair, without duplicates."""
    pairs = []
    if "goal" in level.markers:
        pairs.append(("spawn", "goal"))
    for chk in validate_checks(level):
        if chk.kind in ("visible", "not_visible"):
            pair = (chk.args["from"], chk.args["to"])
            key = tuple(tuple(v) if isinstance(v, list) else v for v in pair)
            if key not in [tuple(tuple(v) if isinstance(v, list) else v for v in p) for p in pairs]:
                pairs.append(pair)
    return pairs


def _symbols(level: Level) -> tuple[dict[str, str], str]:
    """Map letter per marker (first letter of its name when free) and the object symbol."""
    used = set(level.legend) | {PATH_CHAR}
    obj_symbol = next(ch for ch in "O@&$%" if ch not in used)
    used.add(obj_symbol)
    letters: dict[str, str] = {}
    pool = string.ascii_uppercase + string.digits
    for name in level.markers:
        options = [ch.upper() for ch in name if ch.isalpha()] + list(pool)
        ch = next(o for o in options if o not in used)
        letters[name] = ch
        used.add(ch)
    return letters, obj_symbol


def map_report(level: Level, path: dict) -> dict:
    grid = level.tiles.copy().astype("<U1")
    letters, obj_symbol = _symbols(level)
    for obj in level.objects:
        for c, r in obj.cells:
            grid[r, c] = obj_symbol
    for c, r in path.get("cells", []):
        if level.kind(c, r) != "door":
            grid[r, c] = PATH_CHAR
    for name, (c, r) in level.markers.items():
        grid[r, c] = letters[name]
    present = set(level.tiles.ravel().tolist())
    legend = [{"symbol": ch, "meaning": kind} for ch, kind in level.legend.items() if ch in present]
    if path.get("cells"):
        legend.append({"symbol": PATH_CHAR, "meaning": "critical path"})
    for obj in level.objects:
        legend.append({"symbol": obj_symbol, "meaning": f"object {obj.asset} at ({obj.col},{obj.row}), "
                       f"{len(obj.cells)} cell(s), {obj.height:.2f} m{'' if obj.solid else ', not solid'}"
                       f"{f', {len(obj.cover) - len(obj.cells)} more walkable under overhangs' if len(obj.cover) > len(obj.cells) else ''}"
                       f"{', placeholder' if obj.placeholder else ''}"})
    for name, (c, r) in level.markers.items():
        legend.append({"symbol": letters[name], "meaning": f"{name} ({c},{r})"})
    for door in level.doors.values():
        legend.append({"symbol": level.tiles[door.row, door.col],
                       "meaning": f"{door.name} ({door.col},{door.row})" + (f" needs {door.key}" if door.key else "")})
    return {"grid": [" ".join(row) for row in grid.tolist()], "legend": legend, "markers": letters}


def play(level: Level) -> dict:
    """Run the bot on the level and return the JSON-able report."""
    analysis = analyze(level)
    path = path_report(analysis)
    reach = reachability_report(analysis)
    soft = softlock_report(analysis)
    sight = [sight_between(level, a, b) for a, b in _sight_pairs(level)]
    warnings = list(level.warnings)
    if analysis.goal_cell is not None and not path["found"]:
        warnings.append("goal unreachable from spawn")
    for name in reach["unreachable"]:
        if f"{name} unreachable from spawn" not in warnings:  # the goal may be listed above already
            warnings.append(f"{name} unreachable from spawn")
    if soft.get("regions") and soft.get("goal_reachable"):
        warnings.append(f"{len(soft['regions'])} softlock region(s): the player can get stuck")
    h = level.heights_m()
    p = level.player
    return to_jsonable({
        "level": level.name,
        "brief": level.brief,
        "style": level.style,
        "grid": {"cols": level.cols, "rows": level.rows, "cell": level.cell, "height_unit": level.height_unit,
                 "width": round(level.cols * level.cell, 4), "depth": round(level.rows * level.cell, 4),
                 "min_height": round(float(h.min()), 4), "max_height": round(float(h.max()), 4)},
        "player": {"step": p.step, "jump_up": p.jump_up, "jump_gap": p.jump_gap, "max_drop": p.max_drop, "height": p.height},
        "keys": analysis.graph.keys,
        "doors": {n: {"cell": [d.col, d.row], "key": d.key} for n, d in level.doors.items()},
        "objects": [{"asset": o.asset, "cell": [o.col, o.row], "cells": [list(c) for c in o.cells],
                     "overhead": [list(c) for c in o.cover if c not in o.cells], "height": round(o.height, 4),
                     "solid": o.solid, "rotate": o.rotate, "placeholder": o.placeholder} for o in level.objects],
        "path": path,
        "reachability": reach,
        "softlocks": soft,
        "sight": sight,
        "map": map_report(level, path),
        "heights": {"grid": level.heights.tolist(), "unit": level.height_unit},
        "warnings": warnings,
    })


# ---------------------------------------------------------------- text

def _rulers(cols: int, width: int, indent: int) -> list[str]:
    """Column number rulers: a tens line (when needed) and a units line, cells `width` wide + 1 space."""
    lines = []
    if cols > 10:
        lines.append(" " * indent + " ".join((str(c // 10) if c >= 10 else " ").rjust(width) for c in range(cols)).rstrip())
    lines.append(" " * indent + " ".join(str(c % 10).rjust(width) for c in range(cols)))
    return lines


def _grid_text(rows: list[list[str]], width: int) -> list[str]:
    label = len(str(len(rows) - 1))
    indent = label + 2
    out = _rulers(len(rows[0]), width, indent)
    for r, cells in enumerate(rows):
        out.append(f"{str(r).rjust(label)}  " + " ".join(v.rjust(width) for v in cells))
    return out


def _m(v) -> str:
    return f"{v:.2f} m"


def format_play(report: dict) -> str:
    g = report["grid"]
    p = report["player"]
    lines = [
        f"LEVEL {report['level']}  {g['cols']} x {g['rows']} cells of {g['cell']:.2f} m "
        f"({g['width']:.2f} x {g['depth']:.2f} m), height unit {g['height_unit']:.2f} m, style {report['style']}",
        f"PLAYER step {p['step']:.2f} m, jump_up {p['jump_up']:.2f} m, jump_gap {p['jump_gap']} cells, "
        f"max_drop {p['max_drop']:.2f} m, height {p['height']:.2f} m",
    ]
    if report["brief"]:
        lines.append(f"BRIEF {report['brief']}")

    path = report["path"]
    lines.append("")
    if path["found"]:
        lines.append(
            f"PATH  spawn → goal {_m(path['length'])} (horizontal {path['horizontal']:.2f} + vertical {path['vertical']:.2f}), "
            f"{path['jumps']} jump(s), max rise {_m(path['max_rise'])}, max fall {_m(path['max_fall'])}, "
            f"ladders {_m(path['ladder_climb'])}")
        if path["keys"] or path["doors"]:
            lines.append(f"  keys {', '.join(path['keys']) or 'none'}; doors {', '.join(path['doors']) or 'none'}")
        width = len(str(len(path["segments"])))
        for i, seg in enumerate(path["segments"], 1):
            lines.append(f"  {str(i).rjust(width)}. {seg}")
    else:
        lines.append(f"PATH  none: {path.get('reason', 'unknown')}")

    reach = report["reachability"]
    lines.append("")
    lines.append(f"REACHABILITY  {reach['reachable_cells']} of {reach['standable_cells']} standable cells ({reach['share']}%)")
    name_w = max((len(n) for n in reach["places"]), default=4)
    for name, info in reach["places"].items():
        cell = f"({info['cell'][0]},{info['cell'][1]})"
        if info["reachable"]:
            lines.append(f"  {name.ljust(name_w)}  {cell.ljust(8)} {info['type']:<6}  {_m(info['distance'])}")
        else:
            lines.append(f"  {name.ljust(name_w)}  {cell.ljust(8)} {info['type']:<6}  UNREACHABLE: {info.get('reason', '')}")
    for reg in reach["regions"]:
        lines.append(f"  never reached: {_cells(reg['cells'])} — {reg['reason']}")

    soft = report["softlocks"]
    lines.append("")
    if not soft["checked"]:
        lines.append(f"SOFTLOCKS  not checked: {soft['reason']}")
    elif not soft["goal_reachable"]:
        spawn_reg = next((reg for reg in soft["regions"] if reg["contains_spawn"]), None)
        lines.append(f"SOFTLOCKS  goal unreachable from spawn: all {soft['cells']} reachable cell(s) are dead ends "
                     f"(fix the PATH first)")
        if spawn_reg is not None:
            lines.append(f"     {spawn_reg['reason']}")
    elif not soft["regions"]:
        lines.append("SOFTLOCKS  none")
    else:
        lines.append(f"SOFTLOCKS  {len(soft['regions'])} region(s), {soft['cells']} cell(s) where the goal can no longer be reached")
        for i, reg in enumerate(soft["regions"], 1):
            entered = f"; entered by {reg['entered_by']}" if reg["entered_by"] else ""
            lines.append(f"  {i}. {_cells(reg['cells'])} (keys held: {' | '.join(reg['keys_held'])}){entered}")
            lines.append(f"     {reg['reason']}")

    lines.append("")
    eye = report["sight"][0]["eye_height"] if report["sight"] else p["height"] - 0.1
    lines.append(f"SIGHT  eye height {_m(eye)}")
    if not report["sight"]:
        lines.append("  (no pairs)")
    for s in report["sight"]:
        pair = f"{s['from']} → {s['to']}"
        if s["visible"]:
            cl = s["clearance"]
            lines.append(f"  {pair:<24} visible, {_m(s['distance'])}, tightest clearance {_m(cl['meters'])} at ({cl['cell'][0]},{cl['cell'][1]})")
        else:
            bb = s["blocked_by"]
            lines.append(f"  {pair:<24} BLOCKED by {bb['what']} ({bb['cell'][0]},{bb['cell'][1]}) top {_m(bb['top'])}, "
                         f"ray at {_m(bb['ray_y'])}, {_m(bb['at_distance'])} from {s['from']}")

    mp = report["map"]
    lines.append("")
    lines.append("MAP  row 0 = north (back, -Z), col 0 = west (-X); one character per cell")
    lines.extend("  " + ln for ln in _grid_text([row.split(" ") for row in mp["grid"]], 1))
    lines.append("  legend: " + ", ".join(f"{e['symbol']} {e['meaning']}" for e in mp["legend"]))

    hg = report["heights"]["grid"]
    width = max(len(str(v)) for row in hg for v in row)
    lines.append("")
    lines.append(f"HEIGHTS  steps of {report['heights']['unit']:.2f} m (min {g['min_height']:.2f} m, max {g['max_height']:.2f} m)")
    lines.extend("  " + ln for ln in _grid_text([[str(v) for v in row] for row in hg], width))

    if report["objects"]:
        lines.append("")
        lines.append("OBJECTS")
        for o in report["objects"]:
            over = f" + {len(o['overhead'])} walkable under overhangs" if o.get("overhead") else ""
            lines.append(f"  {o['asset']} at ({o['cell'][0]},{o['cell'][1]}) rotate {o['rotate']:g}°, {len(o['cells'])} cell(s){over}, "
                         f"{_m(o['height'])} tall, {'solid' if o['solid'] else 'not solid'}{', placeholder box' if o['placeholder'] else ''}")
    if report["warnings"]:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"  - {w}" for w in report["warnings"])
    return "\n".join(lines)


def _cells(cells: list, limit: int = 8) -> str:
    if len(cells) <= limit:
        return " ".join(f"({c},{r})" for c, r in cells)
    cs = [c for c, _ in cells]
    rs = [r for _, r in cells]
    return f"{len(cells)} cells in ({min(cs)},{min(rs)})..({max(cs)},{max(rs)})"


def to_jsonable(value):
    """Convert numpy scalars/arrays inside a report to plain Python types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
