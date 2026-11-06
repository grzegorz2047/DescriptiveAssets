"""Level checks: designer intent evaluated against the bot's findings.

Entry forms (any entry may add `why: "..."`, as a sibling key or inside its mapping):
  - reachable: goal                     | reachable: {to: goal, from: spawn}
  - order: [key_red, door_red, goal]    each item is required for the next AND reachable in order
  - no_softlocks
  - visible: {from: spawn, to: goal}    | not_visible: {...}; endpoints are names or [col, row]
  - path_length: {max: 80, min: 10}     | path_length: 80 (= max)
  - jumps: {min: 1, max: 5}             | jumps: 2 (exactly)
  - max_fall: 3.0                       largest drop on the critical path, meters
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import SightlineError, did_you_mean
from ..results import CheckResult
from .model import Level

CHECK_KINDS = ("reachable", "order", "no_softlocks", "visible", "not_visible", "path_length", "jumps", "max_fall")
NEEDS_GOAL = ("no_softlocks", "path_length", "jumps", "max_fall")


@dataclass
class LevelCheck:
    index: int
    kind: str
    args: dict
    why: str | None
    where: str


def _fail(msg: str, where: str, hint: str) -> SightlineError:
    return SightlineError(msg, where=where, hint=hint)


def _name_or_cell(level: Level, value, where: str, field: str):
    if isinstance(value, str):
        level.place(value, f"{where}.{field}")
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        c, r = value
        if not level.in_bounds(c, r):
            raise _fail(f"{field} cell [{c}, {r}] is outside the grid", f"{where}.{field}",
                        f"columns 0..{level.cols - 1}, rows 0..{level.rows - 1}")
        return [int(c), int(r)]
    raise _fail(f"{field} must be a marker/door name or [col, row], got {value!r}", f"{where}.{field}",
                f"known names: {', '.join(level.place_names())}")


def _keys(args: dict, allowed, where: str) -> None:
    for key in args:
        if key not in allowed:
            raise _fail(f"unknown key '{key}'", where, did_you_mean(key, allowed) or f"allowed: {', '.join(allowed)}")


def _num(value, where: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise _fail(f"{field} must be a number >= 0, got {value!r}", f"{where}.{field}", f"e.g. {field}: 3")
    return float(value)


def _range(args: dict, where: str, kind: str, scalar_key: str) -> dict:
    """{min, max} from a scalar (meaning `scalar_key`) or a mapping."""
    if not isinstance(args, dict):
        if scalar_key == "exact":
            v = _num(args, where, kind)
            return {"min": v, "max": v}
        return {scalar_key: _num(args, where, kind)}
    _keys(args, ("min", "max", "why"), where)
    out = {k: _num(args[k], where, k) for k in ("min", "max") if k in args}
    if not out:
        raise _fail(f"{kind} needs min and/or max", where, f"e.g. {kind}: {{min: 1, max: 5}}")
    if "min" in out and "max" in out and out["min"] > out["max"]:
        raise _fail(f"{kind} min {out['min']:g} > max {out['max']:g}", where, "swap them")
    return out


def parse_check(level: Level, index: int, entry) -> LevelCheck:
    where = f"{level.where}: checks[{index}]"
    why = None
    if isinstance(entry, str):
        kind, value = entry, None
    elif isinstance(entry, dict):
        keys = [k for k in entry if k != "why"]
        if len(keys) != 1:
            raise _fail(f"a check entry needs exactly one kind, got {keys or 'none'}", where,
                        f"one entry per check; kinds: {', '.join(CHECK_KINDS)}")
        kind, value = keys[0], entry[keys[0]]
        why = entry.get("why")
    else:
        raise _fail(f"bad check entry {entry!r}", where, "write `- no_softlocks` or `- reachable: goal`")
    if kind not in CHECK_KINDS:
        raise _fail(f"unknown check kind '{kind}'", where, did_you_mean(kind, CHECK_KINDS) or f"kinds: {', '.join(CHECK_KINDS)}")
    where = f"{where} ({kind})"
    if isinstance(value, dict) and "why" in value:
        why = value["why"]
    if why is not None and not isinstance(why, str):
        raise _fail("why must be text", where, 'why: "player must see the gate from spawn"')
    if kind in NEEDS_GOAL and "goal" not in level.markers:
        raise _fail(f"{kind} needs a goal marker", where, "add markers: {goal: [col, row]}")

    if kind == "reachable":
        if isinstance(value, str):
            args = {"to": value, "from": "spawn"}
        elif isinstance(value, dict):
            _keys(value, ("to", "from", "why"), where)
            if "to" not in value:
                raise _fail("reachable needs `to`", where, "reachable: {to: goal}")
            args = {"to": value["to"], "from": value.get("from", "spawn")}
        else:
            raise _fail("reachable needs a target name", where, "reachable: goal")
        args = {k: _name_or_cell(level, v, where, k) for k, v in args.items()}
    elif kind == "order":
        items = value.get("items") if isinstance(value, dict) else value
        if isinstance(value, dict):
            _keys(value, ("items", "why"), where)
        if not isinstance(items, list) or len(items) < 2 or not all(isinstance(i, str) for i in items):
            raise _fail("order needs a list of at least two marker/door names", where, "order: [key_red, door_red, goal]")
        for i, item in enumerate(items):
            level.place(item, f"{where}[{i}]")
        args = {"items": list(items)}
    elif kind == "no_softlocks":
        if value not in (None, True) and not (isinstance(value, dict) and set(value) <= {"why"}):
            raise _fail("no_softlocks takes no arguments", where, "write `- no_softlocks`")
        args = {}
    elif kind in ("visible", "not_visible"):
        if not isinstance(value, dict):
            raise _fail(f"{kind} needs {{from, to}}", where, f"{kind}: {{from: spawn, to: goal}}")
        _keys(value, ("from", "to", "why"), where)
        for k in ("from", "to"):
            if k not in value:
                raise _fail(f"{kind} needs `{k}`", where, f"{kind}: {{from: spawn, to: goal}}")
        args = {k: _name_or_cell(level, value[k], where, k) for k in ("from", "to")}
    elif kind == "path_length":
        args = _range(value, where, kind, "max")
    elif kind == "jumps":
        args = _range(value, where, kind, "exact")
    else:  # max_fall
        if isinstance(value, dict):
            _keys(value, ("max", "why"), where)
            if "max" not in value:
                raise _fail("max_fall needs max", where, "max_fall: 3.0")
            value = value["max"]
        args = {"max": _num(value, where, "max_fall")}
    return LevelCheck(index=index, kind=kind, args=args, why=why, where=where)


def validate_checks(level: Level) -> list[LevelCheck]:
    """Parse every entry of `level.checks`; SightlineError on the first invalid one."""
    return [parse_check(level, i, entry) for i, entry in enumerate(level.checks)]


def _range_text(args: dict, unit: str = "") -> str:
    lo, hi = args.get("min"), args.get("max")
    if lo is not None and hi is not None:
        return f"{lo:g}{unit}" if lo == hi else f"{lo:g}..{hi:g}{unit}"
    return f"<= {hi:g}{unit}" if hi is not None else f">= {lo:g}{unit}"


def _in_range(v: float, args: dict) -> bool:
    return (args.get("min") is None or v >= args["min"] - 1e-9) and (args.get("max") is None or v <= args["max"] + 1e-9)


def _check_order(level: Level, analysis, items: list[str]) -> tuple[bool, str, str, dict]:
    from .nav import reach, states_at

    g = analysis.graph
    cells = [g.cell(*level.place(n)) for n in items]
    sources = np.array([analysis.spawn_state])
    for i, cell in enumerate(cells):
        reached = reach(g, sources)
        at = states_at(g, cell)
        at = at[reached[at]]
        if len(at) == 0:
            after = f"after {items[i - 1]}" if i else "from spawn"
            return False, f"{items[i]} unreachable {after}", f"{items[i]} not reachable in this order", {"failed_at": items[i]}
        sources = at
    not_required = []
    for i, (a, b) in enumerate(zip(items, items[1:])):
        blocked = np.zeros(g.n_cells, dtype=bool)
        blocked[cells[i]] = True
        reached = reach(g, [analysis.spawn_state], blocked_cells=blocked)
        if reached[states_at(g, cells[i + 1])].any():
            c, r = level.place(a)
            not_required.append((f"{b} reachable without {a}", f"a route to {b} avoids {a} at ({c},{r})"))
    if not_required:
        return (False, "; ".join(t for t, _ in not_required), "; ".join(m for _, m in not_required),
                {"not_required": [t for t, _ in not_required]})
    return True, "reachable in order, each item required for the next", "", {}


def run_level_checks(level: Level, report: dict, entries=None) -> list[CheckResult]:
    """Evaluate `entries` (default: the level's own checks) against a `play()` report."""
    from .nav import analyze, reach, states_at
    from .sight import sight_between

    parsed = validate_checks(level) if entries is None else [parse_check(level, i, e) for i, e in enumerate(entries)]
    analysis = None
    path = report["path"]
    results = []
    for chk in parsed:
        a, kind = chk.args, chk.kind
        data: dict = {}
        msg = ""
        if kind == "reachable":
            target = a["to"]
            info = report["reachability"]["places"].get(target) if isinstance(target, str) else None
            if a["from"] == "spawn" and info is not None:
                ok = info["reachable"]
                measured = f"{target} at {info['distance']:.2f} m" if ok else f"{target} unreachable"
                msg = info.get("reason", "")
                data = {"distance": info["distance"]}
            else:
                analysis = analysis or analyze(level)
                g = analysis.graph
                _, src_cell = _point(level, a["from"])
                _, dst_cell = _point(level, target)
                src = states_at(g, g.cell(*src_cell))
                src = src[analysis.reached[src]]
                if len(src) == 0:
                    c = g.cell(*src_cell)
                    src = np.array([c + int(g.cell_keys[c]) * g.n_cells])
                ok = bool(reach(g, src)[states_at(g, g.cell(*dst_cell))].any())
                measured = f"{_label(target)} {'reachable' if ok else 'unreachable'} from {_label(a['from'])}"
            expected = "reachable"
        elif kind == "order":
            analysis = analysis or analyze(level)
            ok, measured, msg, data = _check_order(level, analysis, a["items"])
            expected = " → ".join(a["items"])
        elif kind == "no_softlocks":
            soft = report["softlocks"]
            ok = soft["checked"] and not soft["regions"]
            measured = "none" if ok else f"{len(soft['regions'])} region(s), {soft['cells']} cell(s)"
            expected = "none"
            if soft["regions"] and not soft["goal_reachable"]:
                measured, msg = f"all {soft['cells']} reachable cell(s)", "goal unreachable from spawn"
            elif soft["regions"]:
                first = soft["regions"][0]
                msg = f"{_cells_short(first['cells'])}: {first['reason']}"
            data = {"regions": len(soft["regions"]), "cells": soft["cells"]}
        elif kind in ("visible", "not_visible"):
            s = sight_between(level, a["from"], a["to"])
            ok = s["visible"] if kind == "visible" else not s["visible"]
            if s["visible"]:
                measured = f"visible, clearance {s['clearance']['meters']:.2f} m at ({s['clearance']['cell'][0]},{s['clearance']['cell'][1]})"
            else:
                bb = s["blocked_by"]
                measured = f"blocked by {bb['what']} ({bb['cell'][0]},{bb['cell'][1]}) {bb['at_distance']:.2f} m from {s['from']}"
            expected = f"{s['from']} {'sees' if kind == 'visible' else 'does not see'} {s['to']}"
            data = {"visible": s["visible"]}
        elif not path["found"]:
            ok, measured, expected = False, "no path to goal", _expected(kind, a)
            msg = path.get("reason", "")
        elif kind == "path_length":
            ok = _in_range(path["length"], a)
            measured, expected = f"{path['length']:.2f} m", _range_text(a, " m")
            data = {"length": path["length"]}
        elif kind == "jumps":
            ok = _in_range(path["jumps"], a)
            measured, expected = f"{path['jumps']}", _range_text(a)
            data = {"jumps": path["jumps"]}
        else:  # max_fall
            ok = path["max_fall"] <= a["max"] + 1e-9
            measured, expected = f"{path['max_fall']:.2f} m", f"<= {a['max']:.2f} m"
            data = {"max_fall": path["max_fall"]}
        results.append(CheckResult(kind=kind, passed=bool(ok), measured=measured, expected=expected, message=msg,
                                   why=chk.why, data=data))
    return results


def _expected(kind: str, args: dict) -> str:
    if kind == "max_fall":
        return f"<= {args['max']:.2f} m"
    return _range_text(args, " m" if kind == "path_length" else "")


def _point(level: Level, value):
    from .sight import resolve_point

    return resolve_point(level, value)


def _label(value) -> str:
    return value if isinstance(value, str) else f"({value[0]},{value[1]})"


def _cells_short(cells: list, limit: int = 6) -> str:
    text = " ".join(f"({c},{r})" for c, r in cells[:limit])
    return text + (f" +{len(cells) - limit} more" if len(cells) > limit else "")
