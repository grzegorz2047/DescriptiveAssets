"""`sightline fit`: nudge part numbers until the model matches its reference pictures better (SPEC 12.5).

Coordinate descent over the numbers of the parts next to the outline errors: at / offset, size, radius,
height, depth, round, rotate and the width and height of extrude outlines. A step re-renders only the
window of each view that the changed parts cover (`march` on a sub-camera of the same pixel grid), so
it costs a fraction of a second instead of a full render. Each view scores the mean of three shares
as `match` and `structure` measure them: the outline's soft IoU, the material classes that agree with
the picture's and the divisions that line up (or the IoU alone with `inner=False`). A step is kept
when the mean score over all the references rises and no single view's score falls. The picture
placement (scale, ground line, sideways shift) stays as the first `match` found it; `match` and
`structure` afterwards re-measure with their own alignment. The edits go back into the YAML text in
place: comments and layout are kept.
"""

from __future__ import annotations

import copy
import math
import pathlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import yaml

from . import reference as ref_mod
from .camera import Camera, fixed_camera
from .errors import SightlineError
from .loader import asset_from_data
from .march import march
from . import structure as st_mod
from .style import _YamlLoader, display_path, read_yaml, resolve_root

STEPS = (0.004, 0.002, 0.001)  # meters per step, one round each (rotate: 1 degree per 4 mm)
DEG_PER_M = 250.0  # rotate steps: STEPS * DEG_PER_M degrees
MAX_MOVE = 0.015  # no number moves further than this from where it started (meters; degrees * 1/DEG_PER_M)
MIN_GAIN = 2e-5  # a step must raise the mean score by this much...
VIEW_TOL = 5e-5  # ... and lower no view's score by more than this (outline only: per step)
FLOOR_TOL = 5e-4  # with classes and divisions: no view's score ever falls this far below its start
MAX_REPEAT = 6  # steps in one direction per number and round
PROGRESS_EVERY = 100  # trials between progress lines
ERR_MIN = 0.25  # pixels where the picture and model shares differ by this much are outline errors
MIN_ERR_PX = 4.0  # parts next to less error than this (px of share) are left alone
MAX_PARTS = 40  # parts fitted per round (the ones next to the most error)
MIN_SIZE = 0.004  # sizes and radii stay above this
REACH_PX = 3  # a changed part's hit mask moves at most this far beyond its projected box (march eps + coverage band)
BORDER_PX = 3  # window pixels this close to its edge are not trusted (the coverage band there is cut)
LEN_KEYS = ("radius", "height", "depth", "top_radius", "thickness", "round")
_SOURCE_RE = re.compile(r"^parts\[(\d+)\]")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _poly(v) -> bool:
    return isinstance(v, list) and len(v) >= 3 and all(isinstance(p, list) and len(p) == 2 and all(_num(c) for c in p)
                                                         for p in v)


@dataclass(frozen=True)
class Param:
    """One number of a part entry: key, component (None for a scalar) and kind (move, size, round, angle, scale)."""

    key: str
    comp: int | None
    kind: str

    def label(self) -> str:
        if self.kind == "scale":
            return f"{self.key} {'width' if self.comp == 0 else 'height'}"
        return self.key if self.comp is None else f"{self.key}[{self.comp}]"


def entry_params(entry: dict) -> list[Param]:
    """The numbers of a part entry the fitter may change (only keys the entry spells out)."""
    if not isinstance(entry, dict) or entry.get("op") == "paint":
        return []
    out = []
    for key in ("at", "offset"):
        v = entry.get(key)
        if isinstance(v, list) and len(v) == 3 and all(_num(x) for x in v):
            out += [Param(key, i, "move") for i in range(3)]
    v = entry.get("size")
    if isinstance(v, list) and v and all(_num(x) for x in v):
        out += [Param("size", i, "size") for i in range(len(v))]
    elif _num(v):
        out.append(Param("size", None, "size"))
    for key in LEN_KEYS:
        if _num(entry.get(key)):
            out.append(Param(key, None, "round" if key == "round" else "size"))
    v = entry.get("rotate")
    if isinstance(v, list) and len(v) == 3 and all(_num(x) for x in v):
        out += [Param("rotate", i, "angle") for i in range(3)]
    if entry.get("shape") == "extrude":
        for key in ("points", "side"):
            if _poly(entry.get(key)):
                out += [Param(key, 0, "scale"), Param(key, 1, "scale")]
    return out


def symmetry_axes(asset) -> list[int]:
    """Mirror planes the designer asked for with `symmetric` checks: 0 (x = 0) or 2 (z = 0)."""
    axes = set()
    for c in asset.checks:
        if c == "symmetric":
            axes.add(0)
        elif isinstance(c, dict) and "symmetric" in c:
            raw = c["symmetric"]
            axis = raw.get("axis", "x") if isinstance(raw, dict) else "x"
            axes.add({"x": 0, "z": 2}.get(axis, 0))
    return sorted(axes)


def frozen_params(entry: dict, axes: list[int]) -> set[Param]:
    """Numbers that would break a mirror symmetry: a part centred on the plane (at or offset 0 across it,
    no mirror twin) keeps that 0 and its turns about the other two axes. A mirrored pair stays symmetric
    whatever it does, so it keeps everything free."""
    out: set[Param] = set()
    if not isinstance(entry, dict) or "mirror" in entry or entry.get("op") == "paint":
        return out
    for a in axes:
        for key in ("at", "offset"):
            v = entry.get(key)
            if isinstance(v, list) and len(v) == 3 and _num(v[a]) and abs(v[a]) < 1e-9:
                out.add(Param(key, a, "move"))
                out |= {Param("rotate", j, "angle") for j in range(3) if j != a}
    return out


def _boundary(ids: np.ndarray) -> np.ndarray:
    b = np.zeros(ids.shape, dtype=bool)
    b[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    b[1:, :] |= ids[1:, :] != ids[:-1, :]
    return b


def _erode2(m: np.ndarray) -> np.ndarray:
    return ref_mod._erode(ref_mod._erode(m))


@dataclass
class _View:
    index: int
    view: str
    camera: Camera
    cover: np.ndarray  # (H, W) the picture's share of each camera pixel
    outside: float  # picture share outside the camera frame (always missing)
    model: np.ndarray  # (H, W) the model's share
    part: np.ndarray  # (H, W) solid part index per pixel, -1 none
    surf: np.ndarray  # (H, W) the part that colours each pixel (a paint or the solid), -1 none
    iou0: float  # match's soft IoU before fitting
    pic: dict | None = None  # picture structure in camera pixels (inside, inner, classes, borders, edges)
    cls_of: np.ndarray | None = None  # material class per part index (+ a -1 slot)
    before: dict = field(default_factory=dict)  # structure's figures before fitting

    def iou(self, model: np.ndarray | None = None) -> float:
        m = self.model if model is None else model
        inter = float(np.minimum(self.cover, m).sum())
        union = float(np.maximum(self.cover, m).sum()) + self.outside
        return inter / max(union, 1e-9)

    def inner(self, model: np.ndarray | None = None, surf: np.ndarray | None = None) -> tuple[float, float]:
        """(class share, division share) as `structure` counts them: classes over every pixel inside both
        outlines; divisions the mean of picture borders near a part boundary and part boundaries near a
        picture border or edge (both inside the outlines' 2 px erosions, NEAR_PX tolerance)."""
        if self.pic is None:
            return 0.0, 0.0
        model = self.model if model is None else model
        surf = self.surf if surf is None else surf
        pic = self.pic
        M = model >= 0.5
        both = pic["inside"] & M
        agree = (pic["cls"] == self.cls_of[surf]) & both
        cls = float(agree.sum()) / max(1, int(both.sum()))
        inner = pic["inner"] & _erode2(M)
        pb = _boundary(surf) & inner
        rb = pic["borders"] & inner
        pe = pic["edges"] & inner
        near = st_mod.NEAR_PX
        rb_hit = float((rb & st_mod._dilate(pb, near)).sum()) / max(1, int(rb.sum()))
        pb_hit = float((pb & (st_mod._dilate(rb, near) | st_mod._dilate(pe, near))).sum()) / max(1, int(pb.sum()))
        return cls, (rb_hit + pb_hit) / 2.0

    def score(self, model=None, surf=None) -> tuple[float, float, float, float]:
        """(score, IoU, classes, divisions); the score is the IoU alone without picture structure."""
        iou = self.iou(model)
        if self.pic is None:
            return iou, iou, 0.0, 0.0
        cls, div = self.inner(model, surf)
        return (iou + cls + div) / 3.0, iou, cls, div


@dataclass
class _Trial:
    data: dict
    asset: object
    models: list
    parts: list
    surfs: list
    scores: list  # per view (score, IoU, classes, divisions)

    @property
    def ious(self) -> list:
        return [s[1] for s in self.scores]


@dataclass
class _State:
    path: object
    root: object
    lines: dict
    data: dict
    asset: object
    views: list
    scores: list  # per view (score, IoU, classes, divisions)
    orig: dict  # entry index -> original entry
    scale: dict = field(default_factory=dict)  # (entry, key) -> [fx, fy] factors of the original outline
    max_move: float = MAX_MOVE
    frozen: dict = field(default_factory=dict)  # entry -> Params kept for the asset's mirror symmetry
    gain: dict = field(default_factory=lambda: defaultdict(float))  # entry -> mean score it added
    evals: int = 0
    log: object = None  # progress callback (a line every PROGRESS_EVERY trials)
    floor: list | None = None  # per view: the lowest score allowed (start - FLOOR_TOL) when scoring classes too

    @property
    def mean(self) -> float:
        return float(np.mean([s[0] for s in self.scores]))

    @property
    def ious(self) -> list:
        return [s[1] for s in self.scores]


def _to_camera(grid: np.ndarray, origin, shape, fill=0) -> np.ndarray:
    """A grid array (as `_match` crops it) on the camera's pixels; camera pixel (0, 0) is grid `origin`."""
    pt, pl = origin
    H, W = shape
    Hg, Wg = grid.shape[:2]
    out = np.full((H, W) + grid.shape[2:], fill, dtype=grid.dtype)
    r0, r1 = max(0, -pt), min(H, Hg - pt)
    c0, c1 = max(0, -pl), min(W, Wg - pl)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = grid[r0 + pt:r1 + pt, c0 + pl:c1 + pl]
    return out


def _material_classes(asset, a: dict) -> np.ndarray:
    """Class per part index as `structure` gives the model's pixels (`structure.material_classes`)."""
    return st_mod.material_classes(asset, a["part"])


def _setup_view(asset, index: int, res: int, inner: bool = True) -> _View:
    m = ref_mod._match(asset, index, res, None)
    cam = m.camera
    shape = (cam.height, cam.width)
    cover = _to_camera(m.cover.astype(np.float32), m.origin, shape)
    model = _to_camera(m.coverage.astype(np.float32), m.origin, shape)
    scene = asset.scene()
    vb = march(scene, cam)
    view = _View(index, cam.view, cam, cover, float(m.cover.sum() - cover.sum()), model, vb.part.copy(),
                 scene.surface_ids(vb), float(m.report["iou"]))
    if inner:
        rep = st_mod.compare_structure(asset, index, res=res)
        a = rep["_arrays"]
        if a["ref"].shape != m.cover.shape:
            raise SightlineError("structure and match disagree on the comparison grid (a bug in sightline)")
        inside = _to_camera(a["ref"], m.origin, shape, False)
        view.pic = {"inside": inside, "inner": _erode2(inside), "cls": _to_camera(a["cls"], m.origin, shape, -1),
                    "borders": _boundary(_to_camera(a["labels"], m.origin, shape, -1)) & inside,
                    "edges": _to_camera(a["edge"], m.origin, shape, False)}
        view.cls_of = _material_classes(asset, a)
        view.before = {"classes": rep["class_agreement_areas"], "classes_strict": rep["class_agreement"],
                       "region_borders_on_part_boundaries": rep["region_borders_on_part_boundaries"],
                       "part_boundaries_on_borders": rep["part_boundaries_on_borders"]}
    return view


def _same_part(a, b) -> bool:
    if a.shape != b.shape or a.op != b.op or a.cut_from != b.cut_from or a.blend != b.blend:
        return False
    if not (np.allclose(a.position, b.position, atol=1e-9) and np.allclose(a.matrix, b.matrix, atol=1e-9)):
        return False
    if a.local_half is not None and b.local_half is not None and not np.allclose(a.local_half, b.local_half, atol=1e-9):
        return False
    if a.params.keys() != b.params.keys():
        return False
    for k, v in a.params.items():
        w = b.params[k]
        if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
            if not (np.shape(v) == np.shape(w) and np.allclose(v, w, atol=1e-9)):
                return False
        elif v != w:
            return False
    return True


def _boxes(scene, changed: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """World boxes that hold every surface the changed parts can move: their groups (blends), or the cut itself."""
    group_of = {m: g for g in scene.groups for m in g.members}
    out, seen = [], set()
    for i in changed:
        p = scene.parts[i]
        if p.is_paint or p.helper:
            continue
        g = group_of.get(i)
        if g is not None:
            if id(g) not in seen:
                seen.add(id(g))
                out.append((g.aabb_min, g.aabb_max))
        else:
            out.append(scene.aabb(i))
    return out


def _rect(cam: Camera, boxes) -> tuple[int, int, int, int] | None:
    """Camera pixel rows r0:r1, columns c0:c1 the boxes project to, grown by REACH_PX, clipped; None if outside."""
    if not boxes:
        return None
    corners = np.concatenate([np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                        for z in (lo[2], hi[2])]) for lo, hi in boxes])
    px, _ = cam.project(corners)
    c0 = max(0, int(math.floor(px[:, 0].min())) - REACH_PX)
    c1 = min(cam.width, int(math.ceil(px[:, 0].max())) + REACH_PX)
    r0 = max(0, int(math.floor(px[:, 1].min())) - REACH_PX)
    r1 = min(cam.height, int(math.ceil(px[:, 1].max())) + REACH_PX)
    return None if r1 <= r0 or c1 <= c0 else (r0, r1, c0, c1)


def _render_window(scene, view: _View, rect) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model share, part map and surface ids of `rect` (camera pixels), on a sub-camera of the same grid."""
    cam = view.camera
    r0, r1, c0, c1 = rect
    wr0, wr1 = max(0, r0 - BORDER_PX), min(cam.height, r1 + BORDER_PX)
    wc0, wc1 = max(0, c0 - BORDER_PX), min(cam.width, c1 + BORDER_PX)
    center = cam.pixel_to_world((wc0 + wc1) / 2.0, (wr0 + wr1) / 2.0)
    sub = fixed_camera(cam.view, center, cam.mpp, wc1 - wc0, wr1 - wr0, cam.near)
    vb = march(scene, sub)
    cov = ref_mod._model_coverage(scene, sub, vb.hit)
    inner = (slice(r0 - wr0, r1 - wr0), slice(c0 - wc0, c1 - wc0))
    return cov[inner], vb.part[inner], scene.surface_ids(vb)[inner]


def _entry_index(part) -> int | None:
    m = _SOURCE_RE.match(part.source or "")
    return int(m.group(1)) if m else None


def _with_entry(data: dict, k: int, entry: dict) -> dict:
    d = dict(data)
    parts = list(data["parts"])
    parts[k] = entry
    d["parts"] = parts
    return d


def _trial(state: _State, k: int, entry: dict) -> _Trial | None:
    data = _with_entry(state.data, k, entry)
    try:
        asset = asset_from_data(data, state.lines, state.path, state.root)
    except SightlineError:
        return None  # the step broke the asset (a size below a round, ...): not a step
    state.evals += 1
    if state.log and state.evals % PROGRESS_EVERY == 0:
        state.log(f"  {state.evals} trials, mean score {state.mean:.4f} (IoU {np.mean(state.ious):.4f}), "
                  f"fitting {_entry_name(state.data, k)}")
    old, new = state.asset.scene(), asset.scene()
    if len(old.parts) != len(new.parts):
        return None
    changed = [i for i in range(len(old.parts)) if not _same_part(old.parts[i], new.parts[i])]
    if not changed:
        return None
    boxes = _boxes(old, changed) + _boxes(new, changed)
    models, parts, surfs, scores = [], [], [], []
    for v, s in zip(state.views, state.scores):
        rect = _rect(v.camera, boxes)
        if rect is None:
            models.append(v.model)
            parts.append(v.part)
            surfs.append(v.surf)
            scores.append(s)
            continue
        cov, pm, sm = _render_window(new, v, rect)
        r0, r1, c0, c1 = rect
        m, p, q = v.model.copy(), v.part.copy(), v.surf.copy()
        m[r0:r1, c0:c1] = cov
        p[r0:r1, c0:c1] = pm
        q[r0:r1, c0:c1] = sm
        models.append(m)
        parts.append(p)
        surfs.append(q)
        scores.append(v.score(m, q))
    return _Trial(data, asset, models, parts, surfs, scores)


def _accept(state: _State, trial: _Trial) -> None:
    state.data, state.asset = trial.data, trial.asset
    for v, m, p, q in zip(state.views, trial.models, trial.parts, trial.surfs):
        v.model, v.part, v.surf = m, p, q
    state.scores = list(trial.scores)


def _better(state: _State, trial: _Trial | None) -> bool:
    if trial is None:
        return False
    if float(np.mean([s[0] for s in trial.scores])) <= state.mean + MIN_GAIN:
        return False
    if state.floor is not None:  # three noisy shares: views may trade, but none ends below its start
        return all(n[0] >= f for n, f in zip(trial.scores, state.floor))
    return all(n[0] >= o[0] - VIEW_TOL for n, o in zip(trial.scores, state.scores))


def attribute(state: _State) -> dict[int, float]:
    """Outline error (px of share) next to each part entry: extra pixels to the part that shows there,
    missing pixels to the nearest part shown."""
    mass: dict[int, float] = defaultdict(float)
    parts = state.asset.scene().parts
    entry_of = np.array([(_entry_index(p) if _entry_index(p) is not None else -1) for p in parts] + [-1])
    for v in state.views:
        diff = v.cover - v.model
        extra = diff <= -ERR_MIN
        miss = diff >= ERR_MIN
        lab = v.part
        if extra.any():
            ids = entry_of[np.where(lab[extra] >= 0, lab[extra], -1)]
            for e, w in zip(ids, -diff[extra]):
                if e >= 0:
                    mass[int(e)] += float(w)
        if miss.any():
            near = ref_mod._nearest_labels(lab, miss)
            ids = entry_of[np.where(near[miss] >= 0, near[miss], -1)]
            for e, w in zip(ids, diff[miss]):
                if e >= 0:
                    mass[int(e)] += float(w)
    return dict(mass)


def _poly_scaled(points: list, fx: float, fy: float) -> list:
    a = np.asarray(points, dtype=np.float64)
    c = (a.min(axis=0) + a.max(axis=0)) / 2.0
    out = c + (a - c) * np.array([fx, fy])
    return [[round(float(x), 4), round(float(y), 4)] for x, y in out]


def _half_extent(points: list, comp: int) -> float:
    a = np.asarray(points, dtype=np.float64)[:, comp]
    return max(1e-6, float(a.max() - a.min()) / 2.0)


def _stepped(state: _State, k: int, entry: dict, prm: Param, sign: int, step: float) -> dict | None:
    """The entry with one number moved by one step, or None where the step leaves the allowed range."""
    orig = state.orig[k]
    new = copy.deepcopy(entry)
    if prm.kind == "scale":
        f = list(state.scale.get((k, prm.key), [1.0, 1.0]))
        e = _half_extent(orig[prm.key], prm.comp)
        f[prm.comp] += sign * step / e
        if abs(f[prm.comp] - 1.0) * e > state.max_move + 1e-12 or f[prm.comp] <= 0.2:
            return None
        new[prm.key] = _poly_scaled(orig[prm.key], f[0], f[1])
        new["_fit_scale"] = f  # carried to _commit_scale, never reaches the loader
        return new
    d = step * (DEG_PER_M if prm.kind == "angle" else 1.0)
    limit = state.max_move * (DEG_PER_M if prm.kind == "angle" else 1.0)
    cur = new[prm.key] if prm.comp is None else new[prm.key][prm.comp]
    start = orig[prm.key] if prm.comp is None else orig[prm.key][prm.comp]
    value = round(float(cur) + sign * d, 6)
    if abs(value - float(start)) > limit + 1e-9:
        return None
    if prm.kind == "size" and value < MIN_SIZE:
        return None
    if prm.kind == "round" and value < 0:
        return None
    if prm.comp is None:
        new[prm.key] = value
    else:
        vec = list(new[prm.key])
        vec[prm.comp] = value
        new[prm.key] = vec
    return new


def _free_params(state: _State, k: int) -> list[Param]:
    frozen = state.frozen.get(k, set())
    return [p for p in entry_params(state.orig[k]) if p not in frozen]


def _fit_number(state: _State, k: int, prm: Param, step: float) -> int:
    """Steps kept for one number in one round (one direction, as long as it helps)."""
    kept = 0
    for sign in (1, -1):
        for _ in range(MAX_REPEAT):
            entry = state.data["parts"][k]
            new = _stepped(state, k, entry, prm, sign, step)
            if new is None:
                break
            scale = new.pop("_fit_scale", None)
            trial = _trial(state, k, new)
            if not _better(state, trial):
                break
            state.gain[k] += float(np.mean([s[0] for s in trial.scores])) - state.mean
            _accept(state, trial)
            if scale is not None:
                state.scale[(k, prm.key)] = scale
            kept += 1
        if kept:
            break
    return kept


def _entry_name(data: dict, k: int) -> str:
    e = data["parts"][k]
    return str(e.get("name", f"parts[{k}]")) if isinstance(e, dict) else f"parts[{k}]"


def _fmt_num(v: float) -> str:
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _fmt(v) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if _num(v):
        return _fmt_num(v)
    return str(v)


def changes(state: _State) -> list[dict]:
    """Every number that moved: part, key, from, to."""
    out = []
    for k, orig in state.orig.items():
        now = state.data["parts"][k]
        for prm in entry_params(orig):
            if prm.kind == "scale":
                f = state.scale.get((k, prm.key))
                if f and abs(f[prm.comp] - 1.0) > 1e-9:
                    out.append({"part": _entry_name(state.data, k), "entry": k, "key": prm.key, "param": prm.label(),
                                "from": 1.0, "to": round(f[prm.comp], 4)})
                continue
            a = orig[prm.key] if prm.comp is None else orig[prm.key][prm.comp]
            b = now[prm.key] if prm.comp is None else now[prm.key][prm.comp]
            if abs(float(a) - float(b)) > 1e-9:
                out.append({"part": _entry_name(state.data, k), "entry": k, "key": prm.key, "param": prm.label(),
                            "from": float(a), "to": float(b)})
    return out


def edit_yaml(text: str, edits: dict[tuple, object]) -> str:
    """Replace values in YAML text in place: edits[("parts", 3, "at")] = [0, 1, 2]; ("asset",) renames.

    Only values that exist are replaced (their node's text span), so comments, order and layout stay."""
    root = yaml.compose(text, Loader=_YamlLoader)
    spans = []
    for path, value in edits.items():
        node = root
        for key in path:
            if isinstance(node, yaml.MappingNode):
                node = next((v for kn, v in node.value if kn.value == key), None)
            elif isinstance(node, yaml.SequenceNode) and isinstance(key, int) and 0 <= key < len(node.value):
                node = node.value[key]
            else:
                node = None
            if node is None:
                raise SightlineError(f"cannot edit {'.'.join(map(str, path))}: no such value in the YAML text")
        if (isinstance(node, yaml.SequenceNode) and isinstance(value, list) and len(value) == len(node.value)
                and all(isinstance(n, yaml.ScalarNode) for n in node.value) and all(_num(v) for v in value)):
            for n, v in zip(node.value, value):  # a list of numbers: rewrite only the numbers that changed
                try:
                    same = abs(float(n.value) - float(v)) < 1e-12
                except ValueError:
                    same = False
                if not same:
                    spans.append((n.start_mark.index, n.end_mark.index, _fmt(v)))
            continue
        spans.append((node.start_mark.index, node.end_mark.index, _fmt(value)))
    for start, end, new in sorted(spans, reverse=True):
        text = text[:start] + new + text[end:]
    return text


def fit_asset(path, root=None, *, res: int = 512, steps=STEPS, parts=None, max_parts: int = MAX_PARTS,
              max_move: float | None = None, checks: bool = True, inner: bool = True, log=None) -> dict:
    """Fit the asset file at `path` to its reference pictures; returns the report with the new YAML text.

    `parts` names the part entries to fit (default: the `max_parts` next to the most outline error, chosen
    again each round). With `inner` (the default) a view's score is the mean of its outline IoU, class
    share and division share; without, the IoU alone. With `checks`, the asset's checks run before and after: parts whose moves
    left them floating are put back, and checks that newly fail are listed in `report["checks"]`.
    The file is not written: see `report["text"]`; `match` measures the result (`iou_after`).
    """
    t0 = time.perf_counter()
    path = pathlib.Path(path)
    root = resolve_root(root, start=path.parent)
    text = path.read_text(encoding="utf-8")
    data, lines = read_yaml(text, display_path(path, root))
    asset = asset_from_data(data, lines, path, root)
    if not asset.references:
        raise SightlineError(f"asset '{asset.name}' has no reference picture to fit", where=display_path(path, root),
                             hint="add reference: {image: refs/<name>.png, view: front, height: 2.0}")
    if isinstance(res, bool) or not isinstance(res, int) or not 64 <= res <= 2048:
        raise SightlineError("res must be an integer from 64 to 2048", hint="e.g. --res 512")
    if max_move is None:
        max_move = MAX_MOVE
    if not _num(max_move) or max_move <= 0:
        raise SightlineError(f"max_move must be a positive number of meters, got {max_move!r}", hint="e.g. --max-move 0.015")
    log = log or (lambda msg: None)
    log(f"setting up {len(asset.references)} view(s) at res {res}")
    views = [_setup_view(asset, i, res, inner) for i in range(len(asset.references))]
    state = _State(path, root, lines, data, asset, views, [v.score() for v in views],
                   {k: copy.deepcopy(e) for k, e in enumerate(data["parts"])}, max_move=float(max_move), log=log)
    axes = symmetry_axes(asset)
    state.frozen = {k: frozen_params(e, axes) for k, e in enumerate(data["parts"])}
    state.frozen = {k: f for k, f in state.frozen.items() if f}
    names = {_entry_name(data, k): k for k in range(len(data["parts"]))}
    fixed = None
    if parts:
        unknown = [n for n in parts if n not in names]
        if unknown:
            raise SightlineError(f"no part entry named {', '.join(unknown)}",
                                 hint="name the entries as written in parts: (a mirrored twin is fitted through its source)")
        fixed = [names[n] for n in parts]
    start = list(state.scores)
    if inner:
        state.floor = [s[0] - FLOOR_TOL for s in start]
    log("start  " + "  ".join(f"{v.view} {s[1]:.4f}" + (f"/{s[2]:.3f}/{s[3]:.3f}" if inner else "")
                              for v, s in zip(views, state.scores)) + f"  mean score {state.mean:.4f}")
    rounds = []
    for rnd, step in enumerate(steps):
        if fixed is None:
            mass = attribute(state)
            chosen = [k for k, w in sorted(mass.items(), key=lambda t: -t[1])
                      if w >= MIN_ERR_PX and _free_params(state, k)][:max_parts]
        else:
            chosen = fixed
        before = state.mean
        kept = 0
        for k in chosen:
            for prm in _free_params(state, k):
                kept += _fit_number(state, k, prm, step)
        rounds.append({"step": step, "parts": len(chosen), "kept": kept, "mean_before": round(before, 5),
                       "mean_after": round(state.mean, 5)})
        log(f"round {rnd + 1} step {step * 1000:g} mm: {len(chosen)} parts, {kept} steps kept, mean score "
            f"{before:.4f} -> {state.mean:.4f}, IoU {np.mean(state.ious):.4f} ({state.evals} trials, "
            f"{time.perf_counter() - t0:.0f} s)")
    check_info = None
    if checks and asset.checks:
        check_info = _keep_checks(state, asset, log)
    moved = changes(state)
    edits = {}
    for c in moved:
        k = c["entry"]
        edits[("parts", k, c["key"])] = state.data["parts"][k][c["key"]]
    new_text = edit_yaml(text, edits) if edits else text
    fitted = asset_from_data(state.data, lines, path, root)
    log("measuring the result with match" + (" and structure" if inner else ""))
    after = [ref_mod.match_asset(fitted, v.index, res, ceiling=False)["iou"] for v in views]
    after_inner = [None] * len(views)
    if inner:
        for n, v in enumerate(views):
            r = st_mod.compare_structure(fitted, v.index, res=res)
            after_inner[n] = {"classes": r["class_agreement_areas"], "classes_strict": r["class_agreement"],
                              "region_borders_on_part_boundaries": r["region_borders_on_part_boundaries"],
                              "part_boundaries_on_borders": r["part_boundaries_on_borders"]}
    gains = sorted(((state.gain[k], k) for k in state.gain if state.gain[k] > 0), reverse=True)
    out_views = []
    for v, a, b, m, s in zip(views, start, state.scores, after, after_inner):
        row = {"index": v.index, "view": v.view, "iou_before": round(v.iou0, 4), "iou_after": round(m, 4),
               "iou_start": round(a[1], 4), "iou_fitted": round(b[1], 4),
               "score_start": round(a[0], 4), "score_fitted": round(b[0], 4)}
        if inner:
            row["structure_before"], row["structure_after"] = v.before, s
        out_views.append(row)
    return {
        "asset": asset.name, "path": display_path(path, root), "res": res, "inner": inner, "views": out_views,
        "mean_before": round(float(np.mean([v.iou0 for v in views])), 4), "mean_after": round(float(np.mean(after)), 4),
        "rounds": rounds, "changes": moved, "gains": [{"part": _entry_name(state.data, k), "gain": round(g, 5)}
                                                       for g, k in gains],
        "checks": check_info, "trials": state.evals, "seconds": round(time.perf_counter() - t0, 1), "text": new_text,
        "symmetry": {"axes": ["xyz"[a] for a in axes], "kept": sorted(_entry_name(data, k) for k in state.frozen)},
    }


def _run_checks(asset) -> tuple[list, dict]:
    from .checks import run_checks
    from .inspect import inspect_asset

    report = inspect_asset(asset)
    return run_checks(asset, report), report


def _keep_checks(state: _State, asset, log) -> dict:
    """Run the checks before and after; undo the parts that left the rest (`connected`) and say what else broke."""
    log("running the checks before and after")
    before, _ = _run_checks(asset)
    reverted = []
    for _ in range(3):
        fitted = asset_from_data(state.data, state.lines, state.path, state.root)
        after, report = _run_checks(fitted)
        new_fail = [a for b, a in zip(before, after) if b.passed and not a.passed]
        moved = {c["entry"] for c in changes(state)}
        undo = set()
        if any(a.kind == "connected" for a in new_fail):
            by_name = {p.name: _entry_index(p) for p in fitted.scene().parts}
            for piece in report.get("floating", []):
                undo |= {by_name.get(n) for n in piece.get("parts", [])}
        undo = {k for k in undo if k in moved}
        if not new_fail or not undo:
            break
        for k in sorted(undo):
            state.data = _with_entry(state.data, k, copy.deepcopy(state.orig[k]))
            for key in [key for key in state.scale if key[0] == k]:
                del state.scale[key]
            state.gain.pop(k, None)
            reverted.append(_entry_name(state.data, k))
        log(f"undid {', '.join(_entry_name(state.data, k) for k in sorted(undo))}: they no longer touched the rest")
    return {"before": [r.line() for r in before if not r.passed], "new_failures": [r.line() for r in new_fail],
            "reverted": reverted, "count": len(after)}


def format_fit(rep: dict) -> str:
    lines = [f"FIT  {rep['asset']}  {rep['path']}  res {rep['res']}  ({rep['trials']} trials, {rep['seconds']:.0f} s)"]
    lines.append("VIEWS  outline soft IoU from match before -> after" + (
        "; classes (areas) and divisions (borders on boundaries / boundaries on borders) from structure"
        if rep.get("inner") else "") + "; fitting's own score (first placement) start -> fitted")
    for v in rep["views"]:
        line = (f"  reference[{v['index']}] {v['view']:<6} IoU {v['iou_before']:.3f} -> {v['iou_after']:.3f} "
                f"({v['iou_after'] - v['iou_before']:+.3f})")
        sb, sa = v.get("structure_before"), v.get("structure_after")
        if sb and sa:
            line += (f"   classes {sb['classes']:.0%} -> {sa['classes']:.0%}   divisions "
                     f"{sb['region_borders_on_part_boundaries']:.0%}/{sb['part_boundaries_on_borders']:.0%} -> "
                     f"{sa['region_borders_on_part_boundaries']:.0%}/{sa['part_boundaries_on_borders']:.0%}")
        line += f"   score {v['score_start']:.4f} -> {v['score_fitted']:.4f}"
        lines.append(line)
    lines.append(f"  mean IoU  {rep['mean_before']:.4f} -> {rep['mean_after']:.4f}  "
                 f"({rep['mean_after'] - rep['mean_before']:+.4f})")
    lines.append("ROUNDS")
    for r in rep["rounds"]:
        lines.append(f"  step {r['step'] * 1000:g} mm: {r['parts']} parts, {r['kept']} steps kept, mean score "
                     f"{r['mean_before']:.4f} -> {r['mean_after']:.4f}")
    top = rep.get("gains") or []
    if top:
        lines.append("GAINS  " + ", ".join(f"{g['part']} {g['gain'] * 100:+.2f}" for g in top[:12])
                     + ("  (mean score points)" if len(top) <= 12 else f", ... ({len(top) - 12} more; mean score points)"))
    lines.append(f"CHANGES  {len(rep['changes'])} numbers")
    by_part: dict[str, list] = defaultdict(list)
    for c in rep["changes"]:
        by_part[c["part"]].append(c)
    for name, cs in by_part.items():
        items = []
        for c in cs:
            if c["key"] in ("points", "side"):
                items.append(f"{c['param']} x{c['to']:.3f}")
            elif c["key"] == "rotate":
                items.append(f"{c['param']} {_fmt_num(c['from'])} -> {_fmt_num(c['to'])} deg")
            else:
                items.append(f"{c['param']} {_fmt_num(c['from'])} -> {_fmt_num(c['to'])} "
                             f"({(c['to'] - c['from']) * 1000:+.0f} mm)")
        lines.append(f"  {name}: " + ", ".join(items))
    sym = rep.get("symmetry") or {}
    if sym.get("kept"):
        lines.append(f"SYMMETRY  {len(sym['kept'])} centred entries kept on the {' and '.join(a + ' = 0' for a in sym['axes'])} "
                     f"plane (the asset's symmetric check): no sideways move, no turn out of the plane")
    ck = rep.get("checks")
    if ck:
        if ck["reverted"]:
            lines.append(f"CHECKS  put back (they left the rest floating): {', '.join(ck['reverted'])}")
        if ck["new_failures"]:
            lines.append(f"CHECKS  {len(ck['new_failures'])} check(s) passed before and fail now:")
            lines += [f"  {line}" for line in ck["new_failures"]]
        else:
            lines.append(f"CHECKS  {ck['count']} run: none that passed before fails now")
    if rep.get("written"):
        lines.append(f"WROTE  {rep['written']}")
    elif rep.get("refused"):
        lines.append(f"WROTE  nothing: {rep['refused']}")
    else:
        lines.append("WROTE  nothing (dry run): --write changes the asset file, --out PATH writes a copy")
    return "\n".join(lines) + "\n"
