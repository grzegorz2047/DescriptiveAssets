"""Asset loader: YAML -> validated, placed and expanded Asset (SPEC section 4).

Every problem in the file becomes a SightlineError whose `where` names the file, the YAML path
and line (`assets/knight.yaml: parts[3] (sword), line 25`) and whose hint says how to fix it.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np

from .camera import VIEW_NAMES, orbit_view, parse_orbit
from .errors import SightlineError, did_you_mean
from .model import (
    ANCHOR_HINT,
    PARAM_KEYS,
    Asset,
    Part,
    ReferenceSpec,
    _is_number,
    axis_rotation,
    clean,
    local_half_extents,
    normalize_params,
    parse_anchor,
    rotation_matrix,
)
from .style import display_path, load_style, read_yaml, resolve_root

__all__ = ["load_asset", "load_asset_text", "asset_from_data", "list_assets", "twin_name", "read_yaml", "MAX_PARTS",
           "MAX_CUTS", "MAX_PAINTS"]

TOP_KEYS = ("asset", "brief", "style", "tags", "parts", "reference", "checks")
PART_KEYS = ("name", "shape", *PARAM_KEYS, "material", "at", "on", "between", "align", "offset", "follow", "rotate",
             "repeat", "mirror", "mirror_name", "op", "from", "blend", "blend_with", "tags")
LENGTH_SHAPES = {"box": "size", "ellipsoid": "size", "wedge": "size", "cylinder": "height", "cone": "height",
                 "capsule": "height"}  # shapes `between` can stretch along their local Y axis
REPEAT_KEYS = ("count", "step", "axis", "center")
REFERENCE_KEYS = ("image", "view", "height", "mask", "lighting")
MAX_PARTS = 256  # solids
MAX_CUTS = 256  # op: cut (carving adds many; they are never meshed)
MAX_PAINTS = 512  # op: paint zones; grooves (strokes) are many by nature
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
AXES = {"x": 0, "y": 1, "z": 2}
_TARGET_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*(?:\.\s*([A-Za-z0-9_]*)|@\s*(\[.*\]))\s*$")


def twin_name(name: str) -> str:
    """Mirror twin name: swap a suffix _left/_right or _l/_r, else append _mirror."""
    for a, b in (("_left", "_right"), ("_right", "_left"), ("_l", "_r"), ("_r", "_l")):
        if name.endswith(a):
            return name[: -len(a)] + b
    return name + "_mirror"


def _short_list(names, limit: int = 24) -> str:
    names = list(names)
    text = ", ".join(names[:limit])
    return text + (f", … ({len(names) - limit} more)" if len(names) > limit else "")


class _Ctx:
    """Error context: file label and YAML line numbers."""

    def __init__(self, label: str, lines: dict):
        self.label = label
        self.lines = lines

    def where(self, head: str, path: tuple = ()) -> str:
        line = None
        for k in range(len(path), 0, -1):
            line = self.lines.get(tuple(path[:k]))
            if line:
                break
        text = f"{self.label}: {head}" if head else self.label
        return f"{text}, line {line}" if line else text

    def fail(self, message: str, head: str, path: tuple = (), hint: str | None = None):
        raise SightlineError(message, where=self.where(head, path), hint=hint)


class _Builder:
    """Builds the expanded part list entry by entry."""

    def __init__(self, ctx: _Ctx, style, entries: list):
        self.ctx = ctx
        self.style = style
        self.entries = entries
        self.parts: list[Part] = []
        self.index: dict[str, int] = {}  # name -> index in parts
        self.origin: dict[str, str] = {}  # name -> source label (for duplicate errors)
        self.repeated: dict[str, list[str]] = {}  # base name -> copy names
        self.twin_of: dict[str, str] = {}  # part name -> its mirror twin
        self.planned = self._prescan(entries)

    # -- helpers

    @staticmethod
    def _prescan(entries) -> dict[str, int]:
        """Names each entry will define (for 'defined later' errors)."""
        planned: dict[str, int] = {}
        for i, e in enumerate(entries):
            if not isinstance(e, dict) or not isinstance(e.get("name"), str):
                continue
            base = e["name"]
            rep = e.get("repeat")
            count = rep.get("count") if isinstance(rep, dict) else None
            names = [base]
            if isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= MAX_PARTS:
                names += [f"{base}_{k}" for k in range(count)]
            if e.get("mirror") in AXES:
                tb = e["mirror_name"] if isinstance(e.get("mirror_name"), str) else twin_name(base)
                names += [tb] + ([f"{tb}_{k}" for k in range(count)] if len(names) > 1 else [])
            for n in names:
                planned.setdefault(n, i)
        return planned

    def vec3(self, value, head, path, what):
        if not (isinstance(value, (list, tuple)) and len(value) == 3 and all(_is_number(v) for v in value)):
            self.ctx.fail(f"{what} must be a list of 3 numbers [x, y, z], got {value!r}", head, path)
        return np.array([float(v) for v in value])

    def number(self, value, head, path, what, minimum=0.0, strict=False):
        if not _is_number(value) or value < minimum or (strict and value == minimum):
            op = ">" if strict else ">="
            self.ctx.fail(f"{what} must be a number {op} {minimum:g}, got {value!r}", head, path)
        return float(value)

    def str_list(self, value, head, path, what) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            self.ctx.fail(f"{what} must be a list of strings, got {value!r}", head, path)
        return list(value)

    def resolve(self, name, head, path, current: str, entry: int, solid: bool = False) -> int:
        """Index of an earlier part, or a precise error (forward/unknown/self/repeated reference)."""
        if not isinstance(name, str) or not name:
            self.ctx.fail(f"expected a part name, got {name!r}", head, path)
        if name in self.index:
            j = self.index[name]
            if solid and not self.parts[j].is_solid:
                self.ctx.fail(f"'{name}' is a {self.parts[j].op}, not a solid part", head, path,
                              "cuts, paints and blends apply to solid parts; name a solid part")
            return j
        if name in self.repeated:
            copies = self.repeated[name]
            self.ctx.fail(f"'{name}' was expanded by repeat into {copies[0]} … {copies[-1]}; the bare name is not defined",
                          head, path, f"refer to one copy, e.g. '{copies[0]}'")
        if name == current:
            self.ctx.fail(f"part '{name}' refers to itself", head, path, "refer to an earlier part")
        later = self.planned.get(name)
        if later is not None and later >= entry:
            self.ctx.fail(f"'{name}' is defined later (parts[{later}]); references may only point to earlier parts",
                          head, path, f"move '{name}' above this part, or attach '{name}' to this part instead")
        guess = did_you_mean(name, self.index)
        known = _short_list(self.index) or "none yet"
        self.ctx.fail(f"unknown part '{name}'", head, path,
                      (guess + " " if guess else "") + f"earlier parts: {known}")

    def target(self, value, head, path, current, entry) -> tuple[int, np.ndarray]:
        """Parse '<part>.<anchor>' or '<part>@[u, v, w]' -> (part index, normalized anchor)."""
        form = "use '<part>.<anchor>' (e.g. base.top) or '<part>@[u, v, w]'"
        if not isinstance(value, str):
            self.ctx.fail(f"expected a part anchor, got {value!r}", head, path, form)
        m = _TARGET_RE.match(value)
        if not m:
            if NAME_RE.match(value.strip()):
                self.ctx.fail(f"'{value}' needs an anchor", head, path,
                              f"e.g. '{value.strip()}.top' or '{value.strip()}@[0, 1, 0]'")
            self.ctx.fail(f"bad part anchor '{value}'", head, path, form)
        j = self.resolve(m[1], head, path, current, entry)
        spec = m[2] if m[3] is None else "@" + m[3]
        try:
            return j, parse_anchor(spec)
        except SightlineError as exc:
            if exc.message.startswith("unknown anchor"):
                message = f"unknown anchor '{m[2]}' on part '{m[1]}'"
            else:
                message = f"bad anchor '{spec}' on part '{m[1]}': {exc.message}"
            self.ctx.fail(message, head, path, exc.hint or ANCHOR_HINT)

    def point(self, value, head, path, current, entry) -> np.ndarray:
        """A world point: [x, y, z] or a part anchor."""
        if isinstance(value, str):
            j, uvw = self.target(value, head, path, current, entry)
            return self.parts[j].anchor(uvw)
        return self.vec3(value, head, path, "center")

    def register(self, part: Part, head, path):
        if part.name in self.index:
            self.ctx.fail(f"duplicate part name '{part.name}' (already defined by {self.origin[part.name]})", head, path,
                          "names must be unique after repeat/mirror expansion; rename the part or set mirror_name")
        self.index[part.name] = len(self.parts)
        self.origin[part.name] = part.source
        self.parts.append(part)

    # -- one entry

    def add(self, i: int, e) -> None:
        ctx = self.ctx
        base_path = ("parts", i)
        head = f"parts[{i}]"
        if not isinstance(e, dict):
            ctx.fail(f"a part must be a mapping (name:, shape:, ...), got {type(e).__name__}", head, base_path)
        name = e.get("name")
        if name is None:
            ctx.fail("part has no name", head, base_path, "add name: <lowercase_name>")
        if not isinstance(name, str) or not NAME_RE.match(name):
            ctx.fail(f"bad part name {name!r}", head, base_path + ("name",),
                     "names match ^[a-z][a-z0-9_]*$: lowercase letters, digits and _, starting with a letter")
        head = f"parts[{i}] ({name})"

        def p(key):
            return base_path + (key,)

        for key in e:
            if key not in PART_KEYS:
                guess = did_you_mean(str(key), PART_KEYS)
                ctx.fail(f"unknown key '{key}'", head, p(key),
                         (guess + " " if guess else "") + f"part keys: {', '.join(PART_KEYS)}")

        # Shape and parameters.
        shape = e.get("shape")
        if shape is None:
            ctx.fail("part has no shape", head, base_path,
                     "add shape: box | sphere | ellipsoid | cylinder | cone | capsule | torus | wedge")
        raw_params = {k: e[k] for k in PARAM_KEYS if k in e}
        span = self.between(e, raw_params, shape, head, p, name, i) if "between" in e else None
        params = normalize_params(str(shape), raw_params, lambda key: ctx.where(head, p(key) if key else base_path))
        half = local_half_extents(shape, params)

        # Operation and material.
        op = e.get("op", "union")
        if op not in ("union", "cut", "paint"):
            ctx.fail(f"unknown op {op!r}", head, p("op"),
                     "op is union (default), cut (subtracts its shape) or paint (colours the surfaces inside its shape)")
        material = e.get("material")
        if op == "cut":
            if material is not None:
                ctx.fail("a cut has no material (it is subtracted, never rendered)", head, p("material"),
                         "remove material, or drop op: cut to make it a solid part")
            for key in ("blend", "blend_with"):
                if key in e:
                    ctx.fail(f"'{key}' does not apply to a cut", head, p(key), "blends join solid parts")
        else:
            if "from" in e and op != "paint":
                ctx.fail("'from' only applies to op: cut or op: paint", head, p("from"), "add op: cut or op: paint, or remove from")
            if op == "paint":
                for key in ("blend", "blend_with"):
                    if key in e:
                        ctx.fail(f"'{key}' does not apply to a paint", head, p(key), "a paint adds no geometry to blend")
            if material is None:
                ctx.fail("solid part has no material", head, base_path,
                         f"add material: <name>; materials in style '{self.style.name}': {_short_list(sorted(self.style.materials), 40)}")
            if not isinstance(material, str) or material not in self.style.materials:
                guess = did_you_mean(str(material), self.style.materials)
                ctx.fail(f"unknown material '{material}' in style '{self.style.name}'", head, p("material"),
                         (guess + " " if guess else "") + f"materials: {_short_list(sorted(self.style.materials), 40)}")

        # Orientation.
        rotate = e.get("rotate", [0, 0, 0])
        if _is_number(rotate):
            ctx.fail(f"rotate must be [rx, ry, rz] in degrees, got {rotate!r}", head, p("rotate"),
                     f"e.g. rotate: [0, {rotate:g}, 0] to turn about the vertical axis")
        r_own = rotation_matrix(*self.vec3(rotate, head, p("rotate"), "rotate"))

        # Placement.
        if span is not None:
            a, b, attached_to = span
            frame = np.eye(3)
            matrix = clean(_align_y(b - a) @ r_own)
            offset = self.vec3(e["offset"], head, p("offset"), "offset") if "offset" in e else np.zeros(3)
            position = (a + b) / 2.0 + offset
        else:
            if "at" in e and "on" in e:
                ctx.fail("a part is placed with either 'at' or 'on', not both", head, p("on"),
                         "use on: <part>.<anchor> with offset: [...] to shift it")
            offset = self.vec3(e["offset"], head, p("offset"), "offset") if "offset" in e else np.zeros(3)
            attached_to = None
            frame = np.eye(3)
            if "on" in e:
                j, uvw = self.target(e["on"], head, p("on"), name, i)
                tgt = self.parts[j]
                attached_to = tgt.name
                follow = e.get("follow", True)
                if not isinstance(follow, bool):
                    ctx.fail(f"follow must be true or false, got {follow!r}", head, p("follow"))
                if follow:
                    frame = tgt.matrix
                point = tgt.anchor(uvw)
                default_align = -uvw
            else:
                if "follow" in e:
                    ctx.fail("'follow' only applies with 'on'", head, p("follow"), "remove follow or attach with on:")
                point = self.vec3(e["at"], head, p("at"), "at") if "at" in e else np.zeros(3)
                default_align = np.zeros(3)
            if "align" in e:
                try:
                    align = parse_anchor(e["align"])
                except SightlineError as exc:
                    ctx.fail(f"unknown align anchor {e['align']!r}", head, p("align"), exc.hint or ANCHOR_HINT)
            else:
                align = default_align
            matrix = clean(frame @ r_own)
            position = point - matrix @ (align * half) + frame @ offset

        # Operation targets.
        cut_from: list[str] = []
        blend = 0.0
        blend_with = None
        from_all = False
        if op in ("cut", "paint"):
            src = e.get("from")
            if src is None and attached_to is not None:
                self.resolve(attached_to, head, p("on"), name, i, solid=True)
                cut_from = [attached_to]
            elif src is None and op == "cut":
                ctx.fail("a cut needs 'from: [parts]' or 'from: all', or 'on:' a solid part", head, base_path,
                         "e.g. on: wall@[0, -1, 1] cuts the wall it sits on")
            elif src is None or src == "all":  # a paint without 'from' paints every earlier solid
                from_all = True
                cut_from = [q.name for q in self.parts if q.is_solid]
                if not cut_from:
                    ctx.fail(f"{'from: all' if src else 'this paint'} needs an earlier solid part", head,
                             p("from") if src else base_path)
            else:
                for k, ref in enumerate(self.str_list(src, head, p("from"), "from")):
                    if ref in self.repeated and ref not in self.index:
                        copies = self.repeated[ref]  # a repeated base name means all its copies
                    else:
                        copies = [self.parts[self.resolve(ref, head, p("from") + (k,), name, i, solid=True)].name]
                    cut_from += [c for c in copies if c not in cut_from]
        else:
            if "blend" in e:
                blend = self.number(e["blend"], head, p("blend"), "blend")
            if "blend_with" in e:
                if blend <= 0:
                    ctx.fail("'blend_with' needs blend: k (> 0, the fillet radius in meters)", head, p("blend_with"))
                blend_with = self.parts[self.resolve(e["blend_with"], head, p("blend_with"), name, i, solid=True)].name
            elif blend > 0:
                if attached_to is None:
                    ctx.fail("blend needs a partner: 'on:' a solid part or 'blend_with: <part>'", head, p("blend"))
                self.resolve(attached_to, head, p("on"), name, i, solid=True)

        tags = self.str_list(e.get("tags"), head, p("tags"), "tags")

        # Expansion: repeat, then mirror.
        placements = [(name, matrix, position)]
        copy_names: list[str] = []
        if "repeat" in e:
            placements, copy_names = self.repeat(e["repeat"], head, p("repeat"), name, i, matrix, position, frame)
        twins = []
        if "mirror" in e:
            axis = e["mirror"]
            if axis not in AXES:
                ctx.fail(f"mirror must be x, y or z, got {axis!r}", head, p("mirror"),
                         "mirror: x reflects across the plane x = 0 (left <-> right)")
            tbase = e.get("mirror_name")
            if tbase is not None and (not isinstance(tbase, str) or not NAME_RE.match(tbase)):
                ctx.fail(f"bad mirror_name {tbase!r}", head, p("mirror_name"), "names match ^[a-z][a-z0-9_]*$")
            tbase = tbase or twin_name(name)
            s = np.ones(3)
            s[AXES[axis]] = -1.0
            for k, (nm, mat, pos) in enumerate(placements):
                tname = f"{tbase}_{k}" if copy_names else tbase
                twins.append((nm, tname, clean(s[:, None] * mat), clean(s * pos, 1e-12)))
        elif "mirror_name" in e:
            ctx.fail("'mirror_name' needs mirror: x | y | z", head, p("mirror_name"))

        if op == "paint":
            total = sum(q.is_paint for q in self.parts) + len(placements) + len(twins)
            if total > MAX_PAINTS:
                ctx.fail(f"more than {MAX_PAINTS} paints after expansion (this part brings the total to {total})", head,
                         base_path, "merge neighbouring paints of one material into one outline")
        elif op == "cut":
            total = sum(q.is_cut for q in self.parts) + len(placements) + len(twins)
            if total > MAX_CUTS:
                ctx.fail(f"more than {MAX_CUTS} cuts after expansion (this part brings the total to {total})", head,
                         base_path, "merge neighbouring cuts into one outline")
        else:
            total = sum(q.is_solid for q in self.parts) + len(placements) + len(twins)
            if total > MAX_PARTS:
                ctx.fail(f"more than {MAX_PARTS} parts after expansion (this part brings the total to {total})", head,
                         base_path, "lower repeat counts or merge small details into larger parts")

        def make(nm, mat, pos, source, cfrom, bwith, att):
            return Part(name=nm, shape=shape, params=params, matrix=mat, position=clean(pos, 1e-12),
                        material=material if op != "cut" else None, op=op, cut_from=list(cfrom), blend=blend,
                        blend_with=bwith, attached_to=att, source=source, tags=list(tags), local_half=half.copy())

        for k, (nm, mat, pos) in enumerate(placements):
            source = head if not copy_names else f"{head} copy {k}"
            self.register(make(nm, mat, pos, source, cut_from, blend_with, attached_to), head, base_path)
        if copy_names:
            self.repeated[name] = copy_names
        swap = self.twin_of.get
        for k, (orig, tname, mat, pos) in enumerate(twins):
            t_from = cut_from if from_all else list(dict.fromkeys(swap(n, n) for n in cut_from))
            source = f"{head} mirror" if not copy_names else f"{head} copy {k} mirror"
            part = make(tname, mat, pos, source, t_from, swap(blend_with, blend_with) if blend_with else None,
                        swap(attached_to, attached_to) if attached_to else None)
            self.register(part, head, p("mirror"))
            self.twin_of[orig] = tname
            self.twin_of[tname] = orig

    def between(self, e, raw_params: dict, shape, head, p, name, entry):
        """`between: [A, B]`: the part's local +Y runs from point A to point B (part anchors or [x, y, z]).

        The length comes from the distance: `height` for cylinder/cone/capsule, the middle of `size` for
        box/ellipsoid/wedge, which is then written `size: [width, depth]`. Returns (A, B, the part A or B
        is on) and fills the length into `raw_params`.
        """
        ctx = self.ctx
        for key in ("at", "on", "align", "follow"):
            if key in e:
                ctx.fail(f"'{key}' does not combine with 'between'", head, p(key),
                         "between puts the part's bottom on the first point and its top on the second")
        ends = e["between"]
        form = "between: [<part>.<anchor> or [x, y, z], <part>.<anchor> or [x, y, z]], e.g. [wing_lower.top, wing_upper.bottom]"
        if not isinstance(ends, list) or len(ends) != 2:
            ctx.fail(f"between needs two points, got {ends!r}", head, p("between"), form)
        points, attached = [], None
        for k, value in enumerate(ends):
            path = p("between") + (k,)
            if isinstance(value, str):
                j, uvw = self.target(value, head, path, name, entry)
                points.append(self.parts[j].anchor(uvw))
                attached = attached or self.parts[j].name
            else:
                points.append(self.vec3(value, head, path, f"between[{k}]"))
        a, b = points
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            ctx.fail("the two 'between' points are the same point", head, p("between"), form)
        key = LENGTH_SHAPES.get(str(shape))
        if key is None:
            ctx.fail(f"between needs a shape with a length axis, not {shape!r}", head, p("between"),
                     "use " + ", ".join(sorted(LENGTH_SHAPES)))
        if key == "height":
            if "height" in raw_params:
                ctx.fail(f"between sets the height (the distance between the points, {length:.3f} m); remove height",
                         head, p("height"))
            raw_params["height"] = length
        else:
            size = raw_params.get("size")
            if not (isinstance(size, list) and len(size) == 2):
                ctx.fail(f"with between, size is [width, depth] (the length {length:.3f} m comes from the points), "
                         f"got {size!r}", head, p("size") if "size" in raw_params else p("between"),
                         "e.g. size: [0.09, 0.14]")
            raw_params["size"] = [size[0], length, size[1]]
        return a, b, attached

    def repeat(self, rep, head, path, name, entry, matrix, position, frame):
        ctx = self.ctx
        if not isinstance(rep, dict):
            ctx.fail(f"repeat must be a mapping, got {rep!r}", head, path,
                     "linear: {count: 4, step: [1, 0, 0]}; radial: {count: 8, axis: y, center: [0, 0, 0]}")
        for key in rep:
            if key not in REPEAT_KEYS:
                guess = did_you_mean(str(key), REPEAT_KEYS)
                ctx.fail(f"unknown repeat key '{key}'", head, path + (key,),
                         (guess + " " if guess else "") + f"repeat keys: {', '.join(REPEAT_KEYS)}")
        count = rep.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            ctx.fail(f"repeat count must be an integer >= 1, got {count!r}", head, path + ("count",))
        if count > MAX_PARTS:
            ctx.fail(f"repeat count {count} exceeds the {MAX_PARTS}-part limit", head, path + ("count",))
        names = [f"{name}_{k}" for k in range(count)]
        if ("step" in rep) == ("axis" in rep):
            ctx.fail("repeat needs exactly one of 'step' (linear) or 'axis' (radial)", head, path,
                     "linear: {count: 4, step: [1, 0, 0]}; radial: {count: 8, axis: y, center: [0, 0, 0]}")
        if "step" in rep:
            if "center" in rep:
                ctx.fail("'center' only applies to a radial repeat (axis:)", head, path + ("center",))
            step = frame @ self.vec3(rep["step"], head, path + ("step",), "step")
            return [(n, matrix, position + k * step) for k, n in enumerate(names)], names
        axis = rep["axis"]
        if axis not in AXES:
            ctx.fail(f"repeat axis must be x, y or z, got {axis!r}", head, path + ("axis",))
        axis_w = frame @ np.eye(3)[AXES[axis]]
        center = self.point(rep["center"], head, path + ("center",), name, entry) if "center" in rep else np.zeros(3)
        out = []
        for k, n in enumerate(names):
            rot = axis_rotation(axis_w, k * 360.0 / count)
            out.append((n, clean(rot @ matrix), center + rot @ (position - center)))
        return out, names


def _align_y(v: np.ndarray) -> np.ndarray:
    """The smallest rotation taking local +Y onto the direction of v (keeps a strut's faces as upright as it can)."""
    v = np.asarray(v, dtype=np.float64) / np.linalg.norm(v)
    c = float(v[1])
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross([0.0, 1.0, 0.0], v)
    return axis_rotation(axis / np.linalg.norm(axis), float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))


# ----------------------------------------------------------------------------- top level


def _asset_path(path_or_name, root) -> pathlib.Path:
    text = str(path_or_name)
    path = pathlib.Path(text)
    if text.endswith((".yaml", ".yml")) or "/" in text or "\\" in text or path.is_file():
        candidates = [path] if path.is_absolute() else [pathlib.Path.cwd() / path]
        if not path.is_absolute() and root is not None:
            candidates.append(pathlib.Path(root) / path)
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        raise SightlineError(f"asset file not found: {text}",
                             hint="looked in: " + ", ".join(c.as_posix() for c in candidates))
    r = resolve_root(root)
    cand = r / "assets" / f"{text}.yaml"
    if cand.is_file():
        return cand
    available = list_assets(r)
    guess = did_you_mean(text, available)
    raise SightlineError(f"unknown asset '{text}'",
                         hint=(guess + " " if guess else "") + f"assets in {(r / 'assets').as_posix()}: "
                         + (_short_list(available) or "none"))


def list_assets(root=None) -> list[str]:
    """Names of the assets under root/assets."""
    folder = resolve_root(root) / "assets"
    return sorted(p.stem for p in folder.glob("*.yaml")) if folder.is_dir() else []


def load_asset(path_or_name, root=None) -> Asset:
    """Load an asset by name (`assets/<name>.yaml` under root) or by path."""
    path = _asset_path(path_or_name, root)
    root = resolve_root(root, start=path.parent)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise SightlineError("asset file is not UTF-8 text", where=display_path(path, root)) from None
    return load_asset_text(text, path, root)


def _resolve_file(value, asset_dir: pathlib.Path, root: pathlib.Path) -> pathlib.Path | list[pathlib.Path]:
    """Existing path relative to the asset's directory, then root, then root/assets (where `refs/` lives);
    else the list of tried paths."""
    path = pathlib.Path(value)
    candidates = [path] if path.is_absolute() else [asset_dir / path, root / path, root / "assets" / path]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return candidates


def _parse_references(ctx: _Ctx, value, asset_dir, root) -> list[ReferenceSpec]:
    if value is None:
        return []
    entries = value if isinstance(value, list) else [value]
    refs = []
    for k, e in enumerate(entries):
        head = f"reference[{k}]" if isinstance(value, list) else "reference"
        path = ("reference", k) if isinstance(value, list) else ("reference",)
        if not isinstance(e, dict):
            ctx.fail(f"a reference must be a mapping with image:, got {e!r}", head, path,
                     "reference: {image: refs/x.png, view: front, height: 8.0}")
        for key in e:
            if key not in REFERENCE_KEYS:
                guess = did_you_mean(str(key), REFERENCE_KEYS)
                ctx.fail(f"unknown key '{key}'", head, path + (key,),
                         (guess + " " if guess else "") + f"reference keys: {', '.join(REFERENCE_KEYS)}")
        image = e.get("image")
        if not isinstance(image, str) or not image:
            ctx.fail("reference needs image: <path>", head, path, "relative to the asset file's directory, then root")
        found = _resolve_file(image, asset_dir, root)
        if isinstance(found, list):
            ctx.fail(f"reference image not found: {image}", head, path + ("image",),
                     "looked in: " + ", ".join(c.as_posix() for c in found))
        view = e.get("view", "front")
        if isinstance(view, dict):
            if set(view) != {"azimuth", "elevation"} or not all(_is_number(v) for v in view.values()):
                ctx.fail(f"view must be a view name or {{azimuth: deg, elevation: deg}}, got {view!r}", head,
                         path + ("view",))
            view = orbit_view(float(view["azimuth"]), float(view["elevation"]))
        if not isinstance(view, str):
            ctx.fail(f"bad view {view!r}", head, path + ("view",), f"views: {', '.join(VIEW_NAMES)}")
        if view not in VIEW_NAMES:
            try:
                ok = parse_orbit(view) is not None
            except SightlineError as exc:
                ctx.fail(exc.message, head, path + ("view",), exc.hint)
            if not ok:
                guess = did_you_mean(view, VIEW_NAMES)
                ctx.fail(f"unknown view '{view}'", head, path + ("view",),
                         (guess + " " if guess else "") + f"views: {', '.join(VIEW_NAMES)} or {{azimuth, elevation}}")
        height = e.get("height")
        if height is not None and (not _is_number(height) or height <= 0):
            ctx.fail(f"height must be a number > 0 (meters), got {height!r}", head, path + ("height",))
        mask = e.get("mask", "auto")
        if mask != "auto":
            if not isinstance(mask, str):
                ctx.fail(f"mask must be 'auto' or a path, got {mask!r}", head, path + ("mask",))
            found_mask = _resolve_file(mask, asset_dir, root)
            if isinstance(found_mask, list):
                ctx.fail(f"mask image not found: {mask}", head, path + ("mask",),
                         "looked in: " + ", ".join(c.as_posix() for c in found_mask))
            mask = found_mask
        lighting = e.get("lighting", "auto")
        if lighting not in ("auto", "lit", "flat"):
            ctx.fail(f"lighting must be auto, lit or flat, got {lighting!r}", head, path + ("lighting",),
                     "lit = a shaded render or photo, flat = flat paint (concept art), auto = guess from the shading")
        refs.append(ReferenceSpec(image=found, view=view, height=None if height is None else float(height), mask=mask,
                                  lighting=lighting))
    return refs


def load_asset_text(text: str, path, root=None) -> Asset:
    """Load an asset from YAML text as if it were the file at `path` (used for files and old revisions)."""
    path = pathlib.Path(path)
    root = resolve_root(root, start=path.parent)
    data, lines = read_yaml(text, display_path(path, root))
    return asset_from_data(data, lines, path, root)


def asset_from_data(data, lines: dict, path, root=None) -> Asset:
    """Build an asset from parsed YAML (`read_yaml`'s data and line map) as if read from the file at `path`.

    `fit` rebuilds edited copies of one parse this way instead of parsing the text again."""
    path = pathlib.Path(path)
    root = resolve_root(root, start=path.parent)
    label = display_path(path, root)
    ctx = _Ctx(label, lines)
    if data is None:
        ctx.fail("empty asset file", "", (), f"start with asset: {path.stem} and a parts: list")
    if not isinstance(data, dict):
        ctx.fail("an asset file must be a mapping with asset: and parts:", "", ())
    for key in data:
        if key not in TOP_KEYS:
            guess = did_you_mean(str(key), TOP_KEYS)
            ctx.fail(f"unknown top-level key '{key}'", str(key), (key,),
                     (guess + " " if guess else "") + f"top-level keys: {', '.join(TOP_KEYS)}")
    name = data.get("asset")
    if name is None:
        ctx.fail("missing 'asset:'", "", (), f"add asset: {path.stem}")
    if name != path.stem:
        ctx.fail(f"asset: '{name}' does not match the file name '{path.stem}'", "asset", ("asset",),
                 f"set asset: {path.stem} or rename the file to {name}.yaml")
    brief = data.get("brief") or ""
    if not isinstance(brief, str):
        ctx.fail("brief must be text", "brief", ("brief",), "use brief: > followed by indented text")
    style_name = data.get("style", "default")
    if not isinstance(style_name, str):
        ctx.fail(f"style must be a style name, got {style_name!r}", "style", ("style",))
    try:
        style = load_style(style_name, root)
    except SightlineError as exc:
        if exc.where is None:
            ctx.fail(exc.message, "style", ("style",), exc.hint)
        raise
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        ctx.fail("tags must be a list of strings", "tags", ("tags",))
    entries = data.get("parts")
    if not isinstance(entries, list) or not entries:
        ctx.fail("an asset needs a non-empty 'parts:' list", "parts", ("parts",),
                 "parts:\n    - {name: body, shape: box, size: [1, 1, 1], material: wood, align: bottom}")
    builder = _Builder(ctx, style, entries)
    for i, entry in enumerate(entries):
        builder.add(i, entry)
    if not any(p.is_solid for p in builder.parts):
        ctx.fail("the asset has no solid part (only cuts)", "parts", ("parts",))
    references = _parse_references(ctx, data.get("reference"), path.parent, root)
    checks = data.get("checks") or []
    if not isinstance(checks, list):
        ctx.fail("checks must be a list", "checks", ("checks",), "checks:\n    - grounded\n    - size: {height: [6, 10]}")
    return Asset(
        name=name,
        path=path,
        brief=brief.strip(),
        style=style,
        tags=list(tags),
        parts=builder.parts,
        checks=list(checks),
        references=references,
        raw=data,
        root=root,
    )
