"""Asset checks (SPEC 7.2): the designer's intent as assertions over the inspect report.

Entry forms (a bare string means "no arguments"; any entry may add `why: "..."`, inside its mapping
or as a sibling key):

  - grounded                              | grounded: {tolerance: 0.05}
  - connected                             no floating pieces
  - stable                                center of mass over the ground footprint
  - size: {height: [6, 10], width: 4}     scalar v = v +- 2%
  - part_size: {part: door, height: [1.8, 2.4]}
  - symmetric: {axis: x, min: 0.9}
  - visible: {part: door, view: front, min: 0.8}     view any = best view
  - ratio: {a: roof.height, b: total.height, range: [0.2, 0.4]}
  - relation: {a: flag, is: above, b: roof}
  - contrast: {a: roof, b: shaft, min: 1.5}
  - readable_at: {px: 32, min_iou: 0.8, keep: [roof]}
  - distinct_from: {assets: [knight], view: front, max_iou: 0.7}
  - max_parts: 30                         | max_materials: 6
  - matches_reference: {index: 0, min_iou: 0.85, max_missing: 0.05}
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np

from . import inspect as insp
from .camera import VIEW_NAMES
from .errors import SightlineError, did_you_mean
from .model import Asset
from .results import CheckResult
from .style import display_path

__all__ = ["run_checks", "parse_checks", "format_checks", "CHECK_KINDS", "RELATIONS"]

CHECK_KINDS: dict[str, tuple[str, ...]] = {
    "grounded": ("tolerance",),
    "connected": (),
    "stable": (),
    "size": ("width", "height", "depth"),
    "part_size": ("part", "width", "height", "depth"),
    "symmetric": ("axis", "min"),
    "visible": ("part", "view", "min"),
    "ratio": ("a", "b", "range"),
    "relation": ("a", "is", "b"),
    "contrast": ("a", "b", "min"),
    "readable_at": ("px", "min_iou", "keep", "view"),
    "distinct_from": ("assets", "view", "max_iou"),
    "max_parts": (),
    "max_materials": (),
    "matches_reference": ("index", "min_iou", "max_missing"),
}
SCALAR_KINDS = ("max_parts", "max_materials")  # take a number, not a mapping
BARE_KINDS = ("grounded", "connected", "stable", "symmetric", "readable_at", "matches_reference")
EXAMPLES = {
    "grounded": "grounded, or grounded: {tolerance: 0.05}",
    "connected": "connected",
    "stable": "stable",
    "size": "size: {height: [6, 10], width: 4}",
    "part_size": "part_size: {part: door, height: [1.8, 2.4]}",
    "symmetric": "symmetric: {axis: x, min: 0.9}",
    "visible": "visible: {part: door, view: front, min: 0.8}",
    "ratio": "ratio: {a: roof.height, b: total.height, range: [0.2, 0.4]}",
    "relation": "relation: {a: flag, is: above, b: roof}",
    "contrast": "contrast: {a: roof, b: shaft, min: 1.5}",
    "readable_at": "readable_at: {px: 32, min_iou: 0.8, keep: [roof]}",
    "distinct_from": "distinct_from: {assets: [knight], view: front, max_iou: 0.7}",
    "max_parts": "max_parts: 30",
    "max_materials": "max_materials: 6",
    "matches_reference": "matches_reference: {index: 0, min_iou: 0.85, max_missing: 0.05}",
}
RELATIONS = ("above", "below", "left_of", "right_of", "in_front_of", "behind", "touching", "inside")
DIMS = ("width", "height", "depth")
SCALAR_TOLERANCE = 0.02  # size v means v +- 2%


@dataclass
class CheckEntry:
    """One validated check entry."""

    index: int
    kind: str
    args: dict
    why: str | None
    where: str


# ----------------------------------------------------------------------------- parsing


def _fail(message: str, where: str, hint: str | None = None):
    raise SightlineError(message, where=where, hint=hint)


def _number(value, where: str, key: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{key} must be a number, got {value!r}", where)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        bounds = f"in [{lo:g}, {hi:g}]" if lo is not None and hi is not None else (f">= {lo:g}" if lo is not None else f"<= {hi:g}")
        _fail(f"{key} must be {bounds}, got {value:g}", where)
    return float(value)


def _integer(value, where: str, key: str, lo: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lo:
        _fail(f"{key} must be an integer >= {lo}, got {value!r}", where)
    return int(value)


def _range(value, where: str, key: str) -> tuple[float, float]:
    """[lo, hi], or a scalar v meaning v +- 2%."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            _fail(f"{key} must be a number or [lo, hi], got {value!r}", where, f"e.g. {key}: [6, 10]")
        lo, hi = (_number(v, where, key) for v in value)
        if lo > hi:
            _fail(f"{key} range [{lo:g}, {hi:g}] has lo > hi", where, "swap the two numbers")
        return lo, hi
    v = _number(value, where, key)
    return v * (1 - SCALAR_TOLERANCE), v * (1 + SCALAR_TOLERANCE)


def _part_name(asset: Asset, value, where: str, key: str) -> str:
    if not isinstance(value, str):
        _fail(f"{key} must be a part name, got {value!r}", where)
    try:
        asset.part(value)
    except SightlineError as exc:
        _fail(f"{key}: {exc.message}", where, exc.hint)
    return value


def _view(value, where: str, allow_any: bool = False) -> str:
    options = (*VIEW_NAMES, "any") if allow_any else VIEW_NAMES
    if value not in options:
        guess = did_you_mean(str(value), options)
        _fail(f"unknown view '{value}'", where, (guess + " " if guess else "") + f"views: {', '.join(options)}")
    return value


def _operand(asset: Asset, value, where: str, key: str) -> tuple[str, str]:
    """'<part or total>.<width|height|depth>' -> (part or 'total', dim)."""
    if not isinstance(value, str) or "." not in value:
        _fail(f"{key} must be '<part or total>.<width|height|depth>', got {value!r}", where, "e.g. roof.height or total.width")
    name, dim = value.rsplit(".", 1)
    if dim not in DIMS:
        guess = did_you_mean(dim, DIMS)
        _fail(f"{key}: unknown dimension '{dim}'", where, (guess + " " if guess else "") + "use width, height or depth")
    if name != "total":
        _part_name(asset, name, where, key)
    return name, dim


def _validate(asset: Asset, kind: str, raw, where: str) -> dict:
    """Normalized arguments of one check (defaults filled)."""
    if kind in SCALAR_KINDS:
        if isinstance(raw, dict) and set(raw) <= {"max"}:  # also accept {max: n}
            raw = raw.get("max")
        return {"max": _integer(raw, where, kind, 0)}
    if raw is None:
        if kind not in BARE_KINDS:
            _fail(f"check '{kind}' needs arguments", where, f"e.g. {EXAMPLES[kind]}")
        raw = {}
    if not isinstance(raw, dict):
        _fail(f"check '{kind}' takes a mapping, got {raw!r}", where, f"e.g. {EXAMPLES[kind]}")
    allowed = CHECK_KINDS[kind]
    for key in raw:
        if key not in allowed:
            guess = did_you_mean(str(key), allowed)
            keys = ", ".join(allowed) or "none"
            _fail(f"unknown key '{key}' for check '{kind}'", where, (guess + " " if guess else "") + f"keys: {keys}; e.g. {EXAMPLES[kind]}")
    a: dict = {}
    if kind == "grounded":
        a["tolerance"] = _number(raw["tolerance"], where, "tolerance", 0.0) if "tolerance" in raw else None
    elif kind in ("size", "part_size"):
        if kind == "part_size":
            if "part" not in raw:
                _fail("part_size needs part:", where, f"e.g. {EXAMPLES[kind]}")
            a["part"] = _part_name(asset, raw["part"], where, "part")
        dims = {d: _range(raw[d], where, d) for d in raw if d in DIMS}  # in the order written
        if not dims:
            _fail(f"{kind} needs width, height and/or depth", where, f"e.g. {EXAMPLES[kind]}")
        a["dims"] = dims
    elif kind == "symmetric":
        axis = raw.get("axis", "x")
        if axis not in ("x", "z"):
            _fail(f"symmetric axis must be x or z, got {axis!r}", where, "x = left/right mirror, z = front/back mirror")
        a["axis"] = axis
        a["min"] = _number(raw.get("min", 0.9), where, "min", 0.0, 1.0)
    elif kind == "visible":
        if "part" not in raw:
            _fail("visible needs part:", where, f"e.g. {EXAMPLES[kind]}")
        a["part"] = _part_name(asset, raw["part"], where, "part")
        a["view"] = _view(raw.get("view", "any"), where, allow_any=True)
        a["min"] = _number(raw.get("min", 0.5), where, "min", 0.0, 1.0)
    elif kind == "ratio":
        for key in ("a", "b", "range"):
            if key not in raw:
                _fail(f"ratio needs {key}:", where, f"e.g. {EXAMPLES[kind]}")
        a["a"] = _operand(asset, raw["a"], where, "a")
        a["b"] = _operand(asset, raw["b"], where, "b")
        a["range"] = _range(raw["range"], where, "range")
    elif kind == "relation":
        for key in ("a", "is", "b"):
            if key not in raw:
                _fail(f"relation needs {key}:", where, f"e.g. {EXAMPLES[kind]}")
        a["a"] = _part_name(asset, raw["a"], where, "a")
        a["b"] = _part_name(asset, raw["b"], where, "b")
        rel = raw["is"]
        if rel not in RELATIONS:
            guess = did_you_mean(str(rel), RELATIONS)
            _fail(f"unknown relation '{rel}'", where, (guess + " " if guess else "") + f"relations: {', '.join(RELATIONS)}")
        if a["a"] == a["b"]:
            _fail(f"relation compares '{a['a']}' with itself", where, "use two different parts")
        a["is"] = rel
    elif kind == "contrast":
        for key in ("a", "b"):
            if key not in raw:
                _fail(f"contrast needs {key}:", where, f"e.g. {EXAMPLES[kind]}")
        a["a"] = _part_name(asset, raw["a"], where, "a")
        a["b"] = _part_name(asset, raw["b"], where, "b")
        a["min"] = _number(raw.get("min", insp.LOW_CONTRAST), where, "min", 1.0, 21.0)
    elif kind == "readable_at":
        a["px"] = _integer(raw.get("px", 32), where, "px", 4)
        a["min_iou"] = _number(raw.get("min_iou", 0.8), where, "min_iou", 0.0, 1.0)
        keep = raw.get("keep", [])
        if isinstance(keep, str):
            keep = [keep]
        if not isinstance(keep, list):
            _fail(f"keep must be a list of part names, got {keep!r}", where, "e.g. keep: [roof, door]")
        a["keep"] = [_part_name(asset, k, where, "keep") for k in keep]
        a["view"] = _view(raw.get("view", insp.READ_VIEW), where)
    elif kind == "distinct_from":
        names = raw.get("assets")
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
            _fail("distinct_from needs assets: [names]", where, f"e.g. {EXAMPLES[kind]}")
        a["assets"] = list(names)
        a["view"] = _view(raw.get("view", "front"), where)
        a["max_iou"] = _number(raw.get("max_iou", 0.7), where, "max_iou", 0.0, 1.0)
    elif kind == "matches_reference":
        a["index"] = _integer(raw.get("index", 0), where, "index", 0)
        a["min_iou"] = _number(raw.get("min_iou", 0.85), where, "min_iou", 0.0, 1.0)
        a["max_missing"] = _number(raw.get("max_missing", 0.05), where, "max_missing", 0.0, 1.0)
        if not asset.references:
            _fail("matches_reference needs a reference image, but the asset has no reference:", where,
                  "add reference: {image: refs/<name>.png, view: front, height: <m>} to the asset, or drop this check")
        if a["index"] >= len(asset.references):
            _fail(f"reference index {a['index']} out of range: the asset has {len(asset.references)} reference(s)", where,
                  f"use index 0..{len(asset.references) - 1}")
    return a


def parse_checks(asset: Asset, entries=None) -> list[CheckEntry]:
    """Validate check entries (default: the asset's `checks:`); SightlineError on any problem."""
    entries = asset.checks if entries is None else entries
    label = display_path(asset.path, asset.root)
    if entries is None:
        return []
    if not isinstance(entries, list):
        _fail(f"checks must be a list, got {type(entries).__name__}", f"{label}: checks", "checks:\n  - grounded\n  - connected")
    out = []
    for i, entry in enumerate(entries):
        head = f"{label}: checks[{i}]"
        why = None
        if isinstance(entry, str):
            kind, raw = entry.strip(), None
        elif isinstance(entry, dict):
            body = dict(entry)
            why = body.pop("why", None)
            if len(body) != 1:
                keys = ", ".join(map(str, body)) or "none"
                _fail(f"a check entry names exactly one check, got keys: {keys}", head,
                      "write one check per list item, e.g. - size: {height: [6, 10]}")
            (kind, raw), = body.items()
            kind = str(kind)
        else:
            _fail(f"a check must be a name or a mapping, got {entry!r}", head, "e.g. - grounded  or  - size: {height: 3}")
        where = f"{head} ({kind})"
        if kind not in CHECK_KINDS:
            guess = did_you_mean(kind, CHECK_KINDS)
            _fail(f"unknown check '{kind}'", head, (guess + " " if guess else "") + f"checks: {', '.join(CHECK_KINDS)}")
        if isinstance(raw, dict) and "why" in raw:
            raw = dict(raw)
            why = raw.pop("why")
        if why is not None and not isinstance(why, str):
            _fail(f"why must be text, got {why!r}", where, 'why: "player must see the entrance"')
        out.append(CheckEntry(i, kind, _validate(asset, kind, raw, where), why, where))
    return out


# ----------------------------------------------------------------------------- evaluation helpers


def _m(x) -> str:
    return insp._m(x)


def _pct(x) -> str:
    return insp._pct(x)


def _span(lo: float, hi: float, unit: str = " m") -> str:
    if abs(hi - lo) < 1e-12:
        return f"{_m(lo)}{unit}"
    if _m(lo) == _m(hi):  # a narrow range (small value +-2%): 3 decimals so it does not read as zero width
        return f"{lo:.3f}..{hi:.3f}{unit}"
    return f"{_m(lo)}..{_m(hi)}{unit}"


class _Context:
    """Report lookups shared by the checks of one run."""

    def __init__(self, asset: Asset, report: dict, root):
        self.asset = asset
        self.report = report
        self.root = root if root is not None else asset.root
        self.solids = {p["name"]: p for p in report["parts"]}
        self.cuts = {c["name"]: c for c in report["cuts"]}
        self._others: dict[str, Asset] = {}

    def box(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World AABB of a part: the report's for solids, the exact one for cuts."""
        if name in self.solids:
            c, s = np.array(self.solids[name]["center"]), np.array(self.solids[name]["size"])
            return c - s / 2, c + s / 2
        lo, hi = self.asset.part(name).world_aabb()
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def dim(self, name: str, dim: str) -> float:
        if name == "total":
            return float(self.report["size"][dim])
        lo, hi = self.box(name)
        return float((hi - lo)[DIMS.index(dim)])

    def other(self, name: str, where: str) -> Asset:
        if name not in self._others:
            from .loader import load_asset

            try:
                self._others[name] = load_asset(name, self.root)
            except SightlineError as exc:
                raise SightlineError(f"distinct_from: {exc.message}", where=where, hint=exc.hint) from None
        return self._others[name]


def _result(entry: CheckEntry, passed: bool, measured: str, expected: str, message: str = "", **data) -> CheckResult:
    return CheckResult(entry.kind, bool(passed), measured, expected, "" if passed else message, entry.why, data)


def _dims_check(entry: CheckEntry, values: dict[str, float], subject: str) -> CheckResult:
    measured, expected, problems = [], [], []
    for dim, (lo, hi) in entry.args["dims"].items():
        v = values[dim]
        measured.append(f"{dim} {_m(v)} m")
        expected.append(f"{dim} {_span(lo, hi)}")
        if v < lo - 1e-9:
            problems.append(f"{dim} is {_m(lo - v)} m too small")
        elif v > hi + 1e-9:
            problems.append(f"{dim} is {_m(v - hi)} m too large")
    return _result(entry, not problems, subject + ", ".join(measured), ", ".join(expected), "; ".join(problems),
                   **{d: values[d] for d in entry.args["dims"]})


def _relation_measure(ctx: _Context, a: str, rel: str, b: str) -> tuple[bool, str, str]:
    lo_a, hi_a = ctx.box(a)
    lo_b, hi_b = ctx.box(b)
    ca, cb = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2
    axis = {"above": 1, "below": 1, "left_of": 0, "right_of": 0, "in_front_of": 2, "behind": 2}
    if rel in axis:
        k = axis[rel]
        sign = 1 if rel in ("above", "right_of", "in_front_of") else -1
        delta = float(ca[k] - cb[k])
        name = "xyz"[k]
        measured = f"{a} center {name} {_m(ca[k])} vs {b} {_m(cb[k])} ({'+' if delta >= 0 else ''}{_m(delta)} m)"
        ok = sign * delta > 1e-9
        return ok, measured, f"{a} center {'higher' if k == 1 and sign > 0 else 'lower' if k == 1 else ('greater' if sign > 0 else 'smaller')} in {name}"
    if rel == "touching":
        for c in ctx.report["contacts"]:
            if {c["a"], c["b"]} == {a, b}:
                vol = f" ({c['overlap_volume']:.3f} m3)" if c["kind"] == "overlap" else ""
                return True, f"{c['kind']}{vol}", "touch or overlap"
        gap = float(np.linalg.norm(np.maximum(0.0, np.maximum(lo_a - hi_b, lo_b - hi_a))))
        return False, f"no contact (bounding boxes {_m(gap)} m apart)", "touch or overlap"
    # inside: a's box within b's box, tolerance one voxel cell
    tol = float(ctx.report["voxel_cell"])
    out_lo, out_hi = lo_b - lo_a, hi_a - hi_b  # positive = sticks out on that side
    worst = max(float(out_lo.max()), float(out_hi.max()))
    if worst <= tol:
        return True, f"{a} box inside {b} box", f"{a} inside {b} (tolerance {_m(tol)} m)"
    sides = [("left", "bottom", "back")[k] for k in range(3) if out_lo[k] > tol]
    sides += [("right", "top", "front")[k] for k in range(3) if out_hi[k] > tol]
    return False, f"{a} sticks out of {b} by {_m(worst)} m ({', '.join(sides)})", f"{a} inside {b} (tolerance {_m(tol)} m)"


# ----------------------------------------------------------------------------- the checks


def _evaluate(ctx: _Context, e: CheckEntry) -> CheckResult:
    r, a, kind = ctx.report, e.args, e.kind
    if kind == "grounded":
        tol = a["tolerance"] if a["tolerance"] is not None else max(0.02, r["voxel_cell"] / 2)
        y = r["ground"]["min_y"]
        ok = abs(y) <= tol + 1e-9
        msg = f"lowest point is {_m(abs(y))} m {'above' if y > 0 else 'below'} the ground"
        return _result(e, ok, f"min y {_m(y)} m", f"|min y| <= {_m(tol)} m", msg, min_y=y, tolerance=tol)
    if kind == "connected":
        fl = r["floating"]
        if not fl:
            return _result(e, True, f"{r.get('pieces', 1)} piece(s), none floating", "no floating parts", floating=0)
        items = "; ".join(f"{', '.join(f['parts'])} {_m(f['gap'])} m {f.get('direction', 'from')} {f['nearest']}" for f in fl)
        return _result(e, False, f"{len(fl)} floating: {items}", "no floating parts",
                       "attach the floating parts with on: <part>.<anchor>", floating=len(fl))
    if kind == "stable":
        m = r["mass"]
        if m["stable"] is None:
            return _result(e, False, "n/a (not grounded)", "center of mass over the footprint",
                           "the asset does not touch the ground, so it has no support", margin=None)
        margin = m["support_margin"]
        measured = f"center of mass {_m(margin)} m inside the footprint" if margin >= 0 else \
            f"center of mass {_m(-margin)} m outside the footprint"
        return _result(e, m["stable"], measured, "center of mass over the footprint",
                       "widen the base under the center of mass or move mass back over it", margin=margin)
    if kind == "size":
        return _dims_check(e, {d: float(r["size"][d]) for d in DIMS}, "")
    if kind == "part_size":
        return _dims_check(e, {d: ctx.dim(a["part"], d) for d in DIMS}, f"{a['part']} ")
    if kind == "symmetric":
        v = float(r["symmetry"][a["axis"]])
        return _result(e, v >= a["min"] - 1e-9, f"{a['axis']} IoU {v:.2f}", f">= {a['min']:.2f}",
                       f"mirror image across {a['axis']} differs: make the parts on both sides match", iou=v)
    if kind == "visible":
        name, view = a["part"], a["view"]
        entry = ctx.solids.get(name) or ctx.cuts.get(name)
        visible = dict(entry["visible"]) if entry else {}
        if not visible or (view != "any" and view not in visible):  # measure what the report does not hold
            wanted = [view] if view != "any" else list(insp.REPORT_VIEWS)
            visible.update(insp.visible_fractions(ctx.asset, [name], wanted)[name])
        if view == "any":
            best = max(visible, key=lambda v: (visible[v], -list(visible).index(v)))
            frac, label = float(visible[best]), best
            measured = f"{name} {_pct(frac)} in {best} (best of {', '.join(visible)})"
        else:
            frac, label = float(visible[view]), view
            measured = f"{name} {_pct(frac)} in {view}"
        passed = frac >= a["min"] - 1e-9
        msg = ""
        if not passed:  # name what is in front (one extra render, only on failure)
            hider = insp.visible_occluders(ctx.asset, name, [label])[label]
            msg = (f"mostly hidden by {hider} in {label}" if hider else
                   f"nothing covers it in {label}: it faces away from the view or is cut away")
        return _result(e, passed, measured, f">= {_pct(a['min'])}", msg, fraction=frac, view=label)
    if kind == "ratio":
        (na, da), (nb, db) = a["a"], a["b"]
        va, vb = ctx.dim(na, da), ctx.dim(nb, db)
        lo, hi = a["range"]
        if vb <= 0:
            return _result(e, False, f"{na}.{da} / {nb}.{db} = {_m(va)} / 0", f"{_span(lo, hi, '')}", "divisor is zero")
        q = va / vb
        measured = f"{na}.{da} / {nb}.{db} = {_m(va)} / {_m(vb)} = {q:.2f}"
        msg = "ratio too small" if q < lo else "ratio too large"
        return _result(e, lo - 1e-9 <= q <= hi + 1e-9, measured, _span(lo, hi, ""), msg, ratio=q)
    if kind == "relation":
        ok, measured, expected = _relation_measure(ctx, a["a"], a["is"], a["b"])
        return _result(e, ok, measured, expected, f"{a['a']} is not {a['is'].replace('_', ' ')} {a['b']}")
    if kind == "contrast":
        ca, cb = ctx.solids.get(a["a"], {}).get("lit_color"), ctx.solids.get(a["b"], {}).get("lit_color")
        missing = [n for n, c in ((a["a"], ca), (a["b"], cb)) if c is None]
        if missing:
            return _result(e, False, f"{', '.join(missing)} not visible in iso", f">= {a['min']:.2f}",
                           "contrast is measured on the iso view", ratio=None)
        ratio = insp.contrast_ratio(insp.hex_to_rgb(ca), insp.hex_to_rgb(cb))
        return _result(e, ratio >= a["min"] - 1e-9, f"{ratio:.2f} ({a['a']} {ca} vs {a['b']} {cb})", f">= {a['min']:.2f}",
                       "make one material lighter or darker", ratio=ratio)
    if kind == "readable_at":
        read = r["readability"]
        at = insp._at_px(read["at_px"])
        if read["view"] == a["view"] and a["px"] in at:
            res = at[a["px"]]
        else:
            res = insp._at_px(insp.readability(ctx.asset, a["view"], (a["px"],))["at_px"])[a["px"]]
        lost = [k for k in a["keep"] if k not in res["parts_kept"]]
        measured = f"IoU {_pct(res['iou'])} at {a['px']} px {a['view']}" + (f", lost: {', '.join(lost)}" if lost else "")
        expected = f"IoU >= {_pct(a['min_iou'])}" + (f", keep {', '.join(a['keep'])}" if a["keep"] else "")
        problems = ([f"silhouette IoU {_pct(res['iou'])} < {_pct(a['min_iou'])}"] if res["iou"] < a["min_iou"] - 1e-9 else [])
        if lost:
            verb = "vanishes" if len(lost) == 1 else "vanish"
            problems.append(f"{', '.join(lost)} {verb} at {a['px']} px: make {'it' if len(lost) == 1 else 'them'} larger or thicker")
        return _result(e, not problems, measured, expected, "; ".join(problems), iou=res["iou"], lost=lost)
    if kind == "distinct_from":
        view = a["view"]
        ious = {}
        for name in a["assets"]:
            cached = r.get("compare", {}).get(name, {}).get(view)
            if cached is None:
                cached = insp.compare_silhouettes(ctx.asset, ctx.other(name, e.where), [view])[view]
            ious[name] = float(cached["iou_normalized"])
        worst = max(ious, key=lambda n: (ious[n], n))
        measured = ", ".join(f"{n} {ious[n]:.2f}" for n in ious) + f" ({view}, normalized IoU)"
        return _result(e, ious[worst] <= a["max_iou"] + 1e-9, measured, f"<= {a['max_iou']:.2f}",
                       f"silhouette too similar to {worst}: change the outline, not just details", iou=ious)
    if kind == "max_parts":
        n = int(r["part_count"])
        return _result(e, n <= a["max"], f"{n} solid parts", f"<= {a['max']}",
                       f"{n - a['max']} parts over budget: merge or drop small parts", parts=n)
    if kind == "max_materials":
        mats = sorted({p["material"] for p in r["parts"]})
        return _result(e, len(mats) <= a["max"], f"{len(mats)} materials ({', '.join(mats)})", f"<= {a['max']}",
                       f"{len(mats) - a['max']} materials over budget: reuse materials", materials=mats)
    if kind == "matches_reference":
        try:
            from .reference import match_asset
        except ImportError as exc:
            raise SightlineError(f"reference matching is not available ({exc})", where=e.where,
                                 hint="sightline.reference.match_asset is required for matches_reference") from None
        m = match_asset(ctx.asset, index=a["index"])
        iou, missing = float(m["iou"]), float(m["missing"])
        problems = []
        if iou < a["min_iou"] - 1e-9:
            problems.append(f"IoU {iou:.2f} < {a['min_iou']:.2f}")
        if missing > a["max_missing"] + 1e-9:
            problems.append(f"{_pct(missing)} of the reference is not covered")
        msg = "; ".join(problems) + ("; run sightline match and fix the largest regions first" if problems else "")
        return _result(e, not problems, f"IoU {iou:.2f}, missing {_pct(missing)} (reference {a['index']})",
                       f"IoU >= {a['min_iou']:.2f}, missing <= {_pct(a['max_missing'])}", msg, iou=iou, missing=missing)
    raise AssertionError(kind)  # unreachable: kinds are validated


def run_checks(asset: Asset, report: dict, entries=None, root=None) -> list[CheckResult]:
    """Evaluate check entries (default: the asset's `checks:`) against an inspect report.

    All entries are validated first, so a typo fails before any work. `root` resolves the asset
    names of distinct_from (default: the asset's root).
    """
    parsed = parse_checks(asset, entries)
    ctx = _Context(asset, report, pathlib.Path(root) if root is not None else None)
    return [_evaluate(ctx, e) for e in parsed]


def format_checks(results: list[CheckResult]) -> str:
    """One line per check plus a summary line."""
    lines = [res.line() for res in results]
    failed = sum(not res.passed for res in results)
    lines.append(f"CHECKS  {len(results)}: {len(results) - failed} passed, {failed} failed" if results else "CHECKS  none")
    return "\n".join(lines) + "\n"
