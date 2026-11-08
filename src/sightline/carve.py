"""`sightline carve`: cut away what sticks out of the reference pictures and fill what they all show (SPEC 12.6).

Outlines are taken per viewing direction: front and back pictures share the x/y plane (prisms along z),
right and left pictures share the z/y plane (prisms along x). Cuts use a conservative outline, the union of
the pictures of that direction (and their mirror images across x = 0 when the asset has a `symmetric`
check), so a cut never removes what any picture shows. Fills use the intersection, so a fill never adds
what a picture lacks; their depth comes from the other direction's pictures. Pieces are traced at a quarter
pixel; a piece that encloses part of the outline is split in half until it does not. Every candidate is
kept only if the guarded score of `fit` rises (mean of outline IoU, classes and divisions over the views)
and no view ends below its start. A cut lists in `from:` only the solids it really reaches.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import time

import numpy as np
from PIL import Image

from . import fit as fit_mod
from . import reference as ref_mod
from . import structure as st_mod
from .errors import SightlineError
from .loader import MAX_CUTS, MAX_PARTS, asset_from_data
from .style import display_path, read_yaml, resolve_root

UP = 4  # pieces are traced at a quarter of a camera pixel
MIN_PIECE_PX = 6.0  # camera px of model to remove (cuts) or of picture to add (fills) for a piece to count
MIN_HOLE_UP_PX = 4  # enclosed outline bits smaller than this (quarter-pixel units) are noise, not islands
TRACE_EPS = 1.2  # Douglas-Peucker tolerance in quarter pixels (after SMOOTH_WINDOW averaging)
SMOOTH_WINDOW = UP  # contour points on each side averaged before simplifying (one camera pixel)
MAX_POINTS = 64  # the most an extrude outline may have: fewer, larger pieces under the part limit
MAX_SPLIT = 8  # halvings of a piece that still encloses part of the outline
SPLIT_OVERLAP = UP  # quarter pixels each half reaches into the other (no zero-thickness seam between two cuts)
DEPTH_MARGIN = 0.05  # a cut reaches this far beyond the model on both sides of its direction
TOUCH_STEP = 0.004  # grid of the test that a cut reaches a solid
PLANES = {"z": ("front", "back"), "x": ("right", "left")}  # prism direction -> the views that see that plane


def _upsample(a: np.ndarray) -> np.ndarray:
    img = Image.fromarray(a.astype(np.float32), mode="F")
    return np.asarray(img.resize((a.shape[1] * UP, a.shape[0] * UP), Image.BILINEAR))


def _sample(view, world: np.ndarray, src: np.ndarray, fill) -> np.ndarray:
    """`src` (an array on `view`'s camera pixels) at world points (..., 3), nearest pixel."""
    cam = view.camera
    px, _ = cam.project(world.reshape(-1, 3))
    c = np.floor(px[:, 0]).astype(int)
    r = np.floor(px[:, 1]).astype(int)
    ok = (c >= 0) & (c < cam.width) & (r >= 0) & (r < cam.height)
    out = np.full(len(c), fill, dtype=src.dtype)
    out[ok] = src[r[ok], c[ok]]
    return out.reshape(world.shape[:-1])


class _Plane:
    """One viewing direction on its first view's camera grid at UP x: world points, the union and intersection
    of its pictures, the model's share, and the pictures' classes."""

    def __init__(self, axis: str, views: list, mirror_x: bool):
        self.axis, self.views = axis, views
        base = views[0]
        self.view = base
        cam = base.camera
        H, W = base.cover.shape
        rows, cols = np.mgrid[0:H * UP, 0:W * UP]
        self.world = cam.pixel_to_world((cols + 0.5) / UP, (rows + 0.5) / UP)
        mirrored = self.world.copy()
        mirrored[..., 0] = -mirrored[..., 0]
        covers = [_upsample(base.cover)] + [_sample(v, self.world, v.cover, np.float32(0)) for v in views[1:]]
        if mirror_x:
            covers += [_sample(v, mirrored, v.cover, np.float32(0)) for v in views]
        self.union = np.max(covers, axis=0) >= 0.5
        self.inter = np.min(covers, axis=0) >= 0.5
        self.model = _upsample(base.model)
        self.cls = _sample(base, self.world, base.pic["cls"], np.int8(-1)) if base.pic is not None else None
        self.h = self.world @ cam.right
        self.v = self.world @ cam.up
        self.mirror_x = mirror_x and axis == "z"

    def plane_coords(self, pts_world: np.ndarray) -> np.ndarray:
        """(u, y) outline coordinates: u = x for z prisms, u = -z for x prisms (rotate [0, 90, 0])."""
        return pts_world[:, [0, 1]] if self.axis == "z" else np.stack([-pts_world[:, 2], pts_world[:, 1]], axis=1)


def _has_hole(comp: np.ndarray) -> bool:
    rows, cols = np.nonzero(comp)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    sub = np.zeros((r1 - r0 + 2, c1 - c0 + 2), dtype=bool)
    sub[1:-1, 1:-1] = comp[r0:r1, c0:c1]
    lab, sizes = ref_mod.label_components(~sub)
    outside = lab[0, 0]
    holes = (lab >= 0) & (lab != outside)
    return int(holes.sum()) >= MIN_HOLE_UP_PX


def _pieces(plane: "_Plane", mask: np.ndarray, weight: np.ndarray, min_weight: float,
            depth: int = 0) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """(mask, world outline, weight) of the pieces of `mask` carrying at least `min_weight` of `weight`. A piece
    that encloses part of the outline, or whose outline needs more than MAX_POINTS corners at TRACE_EPS, is
    halved along its longer side (up to MAX_SPLIT times)."""
    out = []
    lab, sizes = ref_mod.label_components(mask)
    for k in range(sizes.size):
        comp = lab == k
        w = float(weight[comp].sum())
        if w < min_weight:
            continue
        split = depth < MAX_SPLIT and _has_hole(comp)
        if not split:
            fine = _traced_poly(comp, 10 ** 6)
            split = depth < MAX_SPLIT and len(fine) > MAX_POINTS
        if not split:
            pts = _outline(plane, comp)
            if pts is not None:
                out.append((comp, pts, w))
            continue
        rows, cols = np.nonzero(comp)
        half = np.zeros_like(comp)
        if rows.max() - rows.min() >= cols.max() - cols.min():
            half[: (rows.min() + rows.max() + 1) // 2, :] = True
        else:
            half[:, : (cols.min() + cols.max() + 1) // 2] = True
        # the halves overlap by SPLIT_OVERLAP: two cuts that only touched would leave a sheet of zero thickness
        # on their shared face (max(d, -d1) = max(d, -d2) = 0 there), which the renderer draws as a line
        one, other = half.copy(), ~half
        for _ in range(SPLIT_OVERLAP):
            one, other = ref_mod._dilate(one), ref_mod._dilate(other)
        out += _pieces(plane, comp & one, weight, min_weight, depth + 1)
        out += _pieces(plane, comp & other, weight, min_weight, depth + 1)
    return out


def _smooth_contour(contour: np.ndarray) -> np.ndarray:
    """A closed pixel contour with its staircase averaged out (a circular moving average over SMOOTH_WINDOW
    points). Extruded through the body, every pixel step of a cut's outline becomes a ledge that shades as
    a stripe on the surface it trims."""
    k = len(contour)
    if k < 2 * SMOOTH_WINDOW + 3:
        return contour
    pad = np.concatenate([contour[-SMOOTH_WINDOW:], contour, contour[:SMOOTH_WINDOW]])
    kernel = np.ones(2 * SMOOTH_WINDOW + 1) / (2 * SMOOTH_WINDOW + 1)
    return np.stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(2)], axis=1)


def _traced_poly(comp: np.ndarray, max_points: int) -> np.ndarray:
    return st_mod.simplify(_smooth_contour(st_mod.outline(comp)), eps=TRACE_EPS, max_points=max_points)


def _outline(plane: "_Plane", comp: np.ndarray) -> np.ndarray | None:
    poly = _traced_poly(comp, MAX_POINTS)
    if len(poly) < 3:
        return None
    cam = getattr(getattr(plane, "view", None), "camera", None)
    if cam is not None:  # the smoothed corners sit between pixels: map them exactly (orthographic camera)
        return cam.pixel_to_world((poly[:, 0] + 0.5) / UP, (poly[:, 1] + 0.5) / UP)
    r = np.clip(np.round(poly[:, 1]).astype(int), 0, comp.shape[0] - 1)
    c = np.clip(np.round(poly[:, 0]).astype(int), 0, comp.shape[1] - 1)
    return plane.world[r, c]


def _fmt_pts(uv: np.ndarray, center: np.ndarray) -> list:
    return [[round(float(a), 4), round(float(b), 4)] for a, b in uv - center]


def _cut_entry(name: str, plane: _Plane, pts_world: np.ndarray, lo: np.ndarray, hi: np.ndarray, mirror: bool) -> dict:
    uv = plane.plane_coords(pts_world)
    c = (uv.min(axis=0) + uv.max(axis=0)) / 2.0
    e = {"name": name, "shape": "extrude", "points": _fmt_pts(uv, c), "op": "cut", "from": []}
    if plane.axis == "z":
        e["depth"] = round(float(hi[2] - lo[2]) + 2 * DEPTH_MARGIN, 4)
        e["at"] = [round(float(c[0]), 4), round(float(c[1]), 4), round(float(lo[2] + hi[2]) / 2.0, 4)]
    else:
        e["depth"] = round(float(hi[0] - lo[0]) + 2 * DEPTH_MARGIN, 4)
        e["at"] = [round(float(lo[0] + hi[0]) / 2.0, 4), round(float(c[1]), 4), round(float(-c[0]), 4)]
        e["rotate"] = [0, 90, 0]
    if mirror:
        e["mirror"] = "x"
    return e


def _fill_entry(name: str, plane: _Plane, other: _Plane | None, pts_world: np.ndarray, material: str,
                lo: np.ndarray, hi: np.ndarray, mirror: bool) -> dict | None:
    """An extrude filling a missing piece: its outline in this plane; across, the other direction's pictures
    (their common outline between the piece's heights), else the model's whole depth."""
    uv = plane.plane_coords(pts_world)
    y0, y1 = float(uv[:, 1].min()), float(uv[:, 1].max())
    across = None
    if other is not None:
        band = other.inter & (other.v >= y0) & (other.v <= y1)
        if band.sum() >= 4 * UP * UP:
            lab, sizes = ref_mod.label_components(band)
            pts = _outline(other, lab == int(np.argmax(sizes)))
            if pts is not None:
                across = other.plane_coords(pts)
    e = {"name": name, "shape": "extrude", "material": material}
    if across is None:
        c = (uv.min(axis=0) + uv.max(axis=0)) / 2.0
        e["points"] = _fmt_pts(uv, c)
        k = 2 if plane.axis == "z" else 0
        e["depth"] = round(float(hi[k] - lo[k]), 4)
        mid = round(float(lo[k] + hi[k]) / 2.0, 4)
        if plane.axis == "z":
            e["at"] = [round(float(c[0]), 4), round(float(c[1]), 4), mid]
        else:
            e["at"] = [mid, round(float(c[1]), 4), round(float(-c[0]), 4)]
            e["rotate"] = [0, 90, 0]
    else:
        front_uv, side_uv = (uv, across) if plane.axis == "z" else (across, uv)  # front: (x, y); side: (-z, y)
        y0, y1 = max(front_uv[:, 1].min(), side_uv[:, 1].min()), min(front_uv[:, 1].max(), side_uv[:, 1].max())
        if y1 - y0 < 1e-3:
            return None
        fp = np.stack([front_uv[:, 0], np.clip(front_uv[:, 1], y0, y1)], axis=1)
        sp = np.stack([-side_uv[:, 0], np.clip(side_uv[:, 1], y0, y1)], axis=1)  # back to (z, y)
        ch = (fp[:, 0].min() + fp[:, 0].max()) / 2.0
        cz = (sp[:, 0].min() + sp[:, 0].max()) / 2.0
        cv = (y0 + y1) / 2.0
        e["points"] = _fmt_pts(fp, np.array([ch, cv]))
        e["side"] = _fmt_pts(sp, np.array([cz, cv]))
        e["at"] = [round(float(ch), 4), round(float(cv), 4), round(float(cz), 4)]
    if mirror:
        e["mirror"] = "x"
    return e


def _class_material(asset, view, cls: int) -> str:
    """The material the model shows most on pixels of class `cls` in `view` (structure's choice otherwise)."""
    parts = asset.scene().parts
    M = view.model >= 0.5
    ids = view.surf[M & (view.cls_of[np.where(view.surf >= 0, view.surf, -1)] == cls) & (view.surf >= 0)]
    by_mat: dict[str, int] = {}
    for i, n in zip(*np.unique(ids, return_counts=True)):
        mat = parts[int(i)].material
        if mat:
            by_mat[mat] = by_mat.get(mat, 0) + int(n)
    if by_mat:
        return max(by_mat, key=by_mat.get)
    mats = asset.style.materials
    return {0: "iron" if "iron" in mats else "paint_black", 1: "metal" if "metal" in mats else "paint_white"}.get(
        cls, next((m for m in ("glow_blue", "lamp", "gold") if m in mats), "metal"))


def _touches(scene, si: int, ci: int) -> bool:
    """Whether cut `ci` reaches solid `si`: some point within ~a grid step of both (a conservative test)."""
    half = TOUCH_STEP * 0.9
    lo = np.maximum(scene.aabb(si)[0], scene.aabb(ci)[0]) - half
    hi = np.minimum(scene.aabb(si)[1], scene.aabb(ci)[1]) + half
    if np.any(hi < lo):
        return False
    axes = [np.arange(lo[k], hi[k] + TOUCH_STEP, TOUCH_STEP, dtype=np.float32) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    p3 = np.stack([gx.ravel(), gy.ravel(), gz.ravel()])
    for s in range(0, p3.shape[1], 400_000):
        chunk = np.ascontiguousarray(p3[:, s:s + 400_000])
        near = scene.parts[ci].sdf3(chunk) <= half
        if near.any() and np.any(scene.parts[si].sdf3(np.ascontiguousarray(chunk[:, near])) <= half):
            return True
    return False


def _reached(state, entry: dict) -> list[str] | None:
    """The solids a cut entry (appended after every part) reaches, for its `from:` list; None if none."""
    probe = copy.deepcopy(entry)
    probe["from"] = "all"
    d = dict(state.data)
    d["parts"] = list(state.data["parts"]) + [probe]
    try:
        a2 = asset_from_data(d, state.lines, state.path, state.root)
    except SightlineError:
        return None
    scene = a2.scene()
    ci = next(i for i, p in enumerate(scene.parts) if p.name == entry["name"])
    names = [p.name for i, p in enumerate(scene.parts[:ci]) if p.is_solid and not p.helper and _touches(scene, i, ci)]
    return names or None


def _trial_add(state, entries: list[dict]):
    """Score the asset with `entries` appended (windows re-rendered as in `fit`)."""
    d = dict(state.data)
    d["parts"] = list(state.data["parts"]) + entries
    try:
        a2 = asset_from_data(d, state.lines, state.path, state.root)
    except SightlineError as exc:
        return None, exc.message
    state.evals += 1
    old, new = state.asset.scene(), a2.scene()
    added = list(range(len(old.parts), len(new.parts)))
    boxes = [new.aabb(i) for i in added]
    res = {"data": d, "asset": a2, "models": [], "parts": [], "surfs": [], "scores": [], "cls": []}
    for v in state.views:
        cls_of = st_mod.material_classes(a2, v.surf)
        rect = fit_mod._rect(v.camera, boxes)
        if rect is None:
            m, p, q = v.model, v.part, v.surf
        else:
            cov, pm, sm = fit_mod._render_window(new, v, rect)
            r0, r1, c0, c1 = rect
            m, p, q = v.model.copy(), v.part.copy(), v.surf.copy()
            m[r0:r1, c0:c1] = cov
            p[r0:r1, c0:c1] = pm
            q[r0:r1, c0:c1] = sm
        keep, v.cls_of = v.cls_of, cls_of
        score = v.score(m, q)
        v.cls_of = keep
        for key, val in (("models", m), ("parts", p), ("surfs", q), ("scores", score), ("cls", cls_of)):
            res[key].append(val)
    return res, None


def _accept(state, res: dict) -> None:
    state.data, state.asset = res["data"], res["asset"]
    for v, m, p, q, c in zip(state.views, res["models"], res["parts"], res["surfs"], res["cls"]):
        v.model, v.part, v.surf, v.cls_of = m, p, q, c
    state.scores = list(res["scores"])


def _free_name(taken: set, stem: str, mirror: bool) -> str:
    k = 0
    while True:
        name = f"{stem}{k}" + ("_l" if mirror else "")
        twin = f"{stem}{k}_r" if mirror else None
        if name not in taken and twin not in taken:
            taken.add(name)
            if twin:
                taken.add(twin)
            return name
        k += 1


def candidates(state, asset, axes: list[int], cuts: bool = True, fills: bool = True) -> list[tuple[float, dict]]:
    """(weight in camera px, entry) for every piece to try, cuts and fills, largest first."""
    by_view = {v.view: v for v in state.views if v.index not in state.holdout}  # held-out pictures shape nothing
    lo, hi = asset.scene().bounds(include_helpers=False)
    planes = {}
    for axis, names in PLANES.items():
        views = [by_view[n] for n in names if n in by_view]
        if views:
            planes[axis] = _Plane(axis, views, mirror_x=0 in axes)
    taken = {p.name for p in asset.scene().parts}
    out = []
    for axis, plane in planes.items():
        other = planes.get("x" if axis == "z" else "z")
        # the frame: what the pictures or the model cover, plus 2 px
        frame = plane.union | (plane.model > 0.0)
        for _ in range(2 * UP):
            frame = ref_mod._dilate(frame)
        region = frame
        mirror = plane.mirror_x  # then only the -x half, written with mirror: x (the outline is symmetric)
        if mirror:  # reaching a pixel past x = 0, so a piece and its mirror twin overlap instead of meeting
            region = frame & (plane.world[..., 0] <= plane.view.camera.mpp)
        tag = "front" if axis == "z" else "side"
        if cuts:
            outside = region & ~plane.union
            for comp, pts, w in _pieces(plane, outside, np.where(outside, plane.model, 0.0) / (UP * UP), MIN_PIECE_PX):
                entry = _cut_entry(_free_name(taken, f"carve_{tag}", mirror), plane, pts, lo, hi, mirror)
                out.append((w, entry))
        if fills:
            missing = region & plane.inter & (plane.model < 0.5)
            for comp, pts, w in _pieces(plane, missing, missing.astype(np.float32) / (UP * UP), MIN_PIECE_PX):
                cls = plane.cls[comp] if plane.cls is not None else np.zeros(1, np.int8)
                vals = cls[cls >= 0].astype(np.int64)
                c = int(np.bincount(vals, minlength=3).argmax()) if vals.size else 0
                entry = _fill_entry(_free_name(taken, f"fill_{tag}", mirror), plane, other, pts,
                                    _class_material(asset, plane.view, c), lo, hi, mirror)
                if entry is not None:
                    out.append((w, entry))
    out.sort(key=lambda t: -t[0])
    return out


def _append_text(text: str, entries: list[dict]) -> str:
    """The YAML text with flow-mapping entries added at the end of `parts:` (before `checks:` if any)."""
    if not entries:
        return text
    indent = "" if "\nparts:\n- " in text else "  "
    blk = [f"{indent}# --- sightline carve: cuts over what lies outside the pictures' common outline and fills of",
           f"{indent}#     what they all show, kept where the fitting score rose"]
    blk += [f"{indent}- " + json.dumps(e, separators=(", ", ": ")).replace('"', "") for e in entries]
    head, sep, tail = text.partition("\nchecks:")
    return head.rstrip("\n") + "\n" + "\n".join(blk) + "\n" + sep + tail


def carve_asset(path, root=None, *, res: int = 512, cuts: bool = True, fills: bool = True, checks: bool = True,
                holdout=(), log=None) -> dict:
    """Carve and fill the asset file at `path` against its reference pictures; returns the report with the
    new YAML text (`report["text"]`). The file is not written. `holdout` reference indices are measured but
    shape no piece and are not scored (see `fit_asset`)."""
    t0 = time.perf_counter()
    log = log or (lambda msg: None)
    path = pathlib.Path(path)
    root = resolve_root(root, start=path.parent)
    text = path.read_text(encoding="utf-8")
    data, lines = read_yaml(text, display_path(path, root))
    asset = asset_from_data(data, lines, path, root)
    if not asset.references:
        raise SightlineError(f"asset '{asset.name}' has no reference picture to carve to",
                             where=display_path(path, root),
                             hint="add reference: {image: refs/<name>.png, view: front, height: 2.0}")
    known = {n for names in PLANES.values() for n in names}
    if not any(r.view in known for r in asset.references):
        raise SightlineError("carve needs a front, back, left or right reference",
                             where=display_path(path, root), hint="orbit and top views are not carved")
    if isinstance(res, bool) or not isinstance(res, int) or not 64 <= res <= 2048:
        raise SightlineError("res must be an integer from 64 to 2048", hint="e.g. --res 512")
    held = fit_mod.check_holdout(asset, holdout)
    if not any(r.view in known for i, r in enumerate(asset.references) if i not in held):
        raise SightlineError("every front, back, left or right reference is held out: nothing to carve to")
    log(f"setting up {len(asset.references)} view(s) at res {res}")
    views = [fit_mod._setup_view(asset, i, res) for i in range(len(asset.references))]
    state = fit_mod._State(path, root, lines, data, asset, views, [v.score() for v in views], {}, holdout=held)
    start = list(state.scores)
    floor = [s[0] - fit_mod.FLOOR_TOL for s in start]
    axes = fit_mod.symmetry_axes(asset)
    budget = MAX_PARTS - sum(1 for p in asset.parts if p.is_solid)  # fills
    cut_budget = MAX_CUTS - sum(1 for p in asset.parts if p.is_cut)
    log("start  " + "  ".join(f"{v.view} {s[1]:.4f}/{s[2]:.3f}/{s[3]:.3f}" for v, s in zip(views, start))
        + f"  mean score {state.mean:.4f}; room for {budget} solids and {cut_budget} cuts")
    cands = candidates(state, asset, axes, cuts=cuts, fills=fills)
    log(f"{len(cands)} candidate piece(s): " + ", ".join(f"{e['name']} {w:.0f} px" for w, e in cands[:12])
        + (" ..." if len(cands) > 12 else ""))
    kept, tried = [], {}
    # every piece alone first: under the part limit the pieces go in by gain per part, not by size
    ranked = []
    for weight, entry in cands:
        name = entry["name"]
        cost = 2 if entry.get("mirror") else 1
        if entry.get("op") == "cut":
            reach = _reached(state, entry)
            if reach is None:
                tried[name] = "reaches no solid"
                continue
            entry["from"] = reach
        res_, err = _trial_add(state, [entry])
        if res_ is None:
            tried[name] = f"refused by the loader: {err}"
            continue
        gain = float(np.mean([s[0] for s in state.active(res_["scores"])])) - state.mean
        if gain <= fit_mod.MIN_GAIN:
            tried[name] = f"score {gain * 100:+.3f} alone"
            continue
        ranked.append((gain / cost, weight, entry))
    ranked.sort(key=lambda t: -t[0])
    log(f"{len(ranked)} piece(s) raise the score alone; adding them by gain per part")
    for rnd in range(2):  # the second pass retries what the first refused: views may have gained slack
        for per_part, weight, entry in ranked:
            name = entry["name"]
            if any(k["name"] == name for k in kept):
                continue
            cost = 2 if entry.get("mirror") else 1
            is_cut = entry.get("op") == "cut"
            if cost > (cut_budget if is_cut else budget):
                tried[name] = "no room: the asset is at its " + ("cut" if is_cut else "part") + " limit"
                continue
            res_, err = _trial_add(state, [entry])
            if res_ is None:
                tried[name] = f"refused by the loader: {err}"
                continue
            scores = res_["scores"]
            gain = float(np.mean([s[0] for s in state.active(scores)])) - state.mean
            low = [v.view for v, s, f in zip(views, scores, floor) if s[0] < f and v.index not in held]
            if gain <= fit_mod.MIN_GAIN or low:
                tried[name] = f"score {gain * 100:+.3f}" + (f", {'/'.join(low)} below its start" if low else "")
                continue
            _accept(state, res_)
            if is_cut:
                cut_budget -= cost
            else:
                budget -= cost
            kept.append(entry)
            tried.pop(name, None)
            log(f"  + {name} ({weight:.0f} px): mean score {state.mean:.4f}, IoU "
                + " ".join(f"{s[1]:.4f}" for s in state.scores))
    check_info = _keep_checks(state, asset, kept, log) if checks and asset.checks else None
    kept_names = {e["name"] for e in kept}
    new_text = _append_text(text, kept)
    fitted = asset_from_data(state.data, lines, path, root)
    log("measuring the result with match and structure")
    rows = []
    for v, s0, s1 in zip(views, start, state.scores):
        m = ref_mod.match_asset(fitted, v.index, res, ceiling=False)
        r = st_mod.compare_structure(fitted, v.index, res=res)
        rows.append({"index": v.index, "view": v.view, "iou_before": round(v.iou0, 4), "iou_after": round(m["iou"], 4),
                     "structure_before": v.before,
                     "structure_after": {"classes": r["class_agreement_areas"], "classes_strict": r["class_agreement"],
                                         "region_borders_on_part_boundaries": r["region_borders_on_part_boundaries"],
                                         "part_boundaries_on_borders": r["part_boundaries_on_borders"]},
                     "score_start": round(s0[0], 4), "score_after": round(s1[0], 4), "holdout": v.index in held})
    return {
        "asset": asset.name, "path": display_path(path, root), "res": res, "views": rows,
        "mean_before": round(float(np.mean([v.iou0 for v in views])), 4),
        "mean_after": round(float(np.mean([r["iou_after"] for r in rows])), 4),
        "kept": [{"name": e["name"], "kind": "cut" if e.get("op") == "cut" else "fill", "mirror": bool(e.get("mirror")),
                  "material": e.get("material"), "from": len(e.get("from", []))} for e in kept],
        "refused": [{"name": n, "why": why} for n, why in tried.items() if n not in kept_names],
        "candidates": len(cands), "checks": check_info, "symmetry": ["xyz"[a] for a in axes], "holdout": sorted(held),
        "trials": state.evals, "seconds": round(time.perf_counter() - t0, 1), "text": new_text,
    }


def _keep_checks(state, asset, kept: list[dict], log) -> dict:
    """Checks before and after; parts left floating by a cut come off that cut's `from:` list (with the part
    they hung on), a floating fill is dropped (up to 3 times)."""
    log("running the checks before and after")
    before, _ = fit_mod._run_checks(asset)
    released = []
    after = before
    new_fail = []
    for _ in range(3):
        fitted = asset_from_data(state.data, state.lines, state.path, state.root)
        after, report = fit_mod._run_checks(fitted)
        new_fail = [a for b, a in zip(before, after) if b.passed and not a.passed]
        if not any(a.kind == "connected" for a in new_fail):
            break
        names = set()
        for piece in report.get("floating", []):
            names |= set(piece.get("parts", [])) | {piece.get("nearest")}
        twins = {n[:-2] + ("_r" if n.endswith("_l") else "_l") for n in names if n and n[-2:] in ("_l", "_r")}
        names |= twins
        changed = False
        parts = list(state.data["parts"])
        for k, e in enumerate(parts):
            if not isinstance(e, dict) or e not in kept:
                continue
            if e.get("op") == "cut":
                keep = [n for n in e["from"] if n not in names]
                if len(keep) != len(e["from"]):
                    released.append(f"{e['name']}: {', '.join(n for n in e['from'] if n in names)}")
                    e["from"] = keep
                    changed = True
            elif e["name"] in names:
                released.append(f"{e['name']}: dropped")
                parts[k] = None
                kept.remove(e)
                changed = True
        state.data = dict(state.data)
        state.data["parts"] = [p for p in parts if p is not None]
        for e in [e for e in kept if e.get("op") == "cut" and not e["from"]]:
            kept.remove(e)
            state.data["parts"] = [p for p in state.data["parts"] if p is not e]
            released.append(f"{e['name']}: dropped (nothing left to cut)")
        if not changed:
            break
        log(f"released floating parts: {'; '.join(released[-4:])}")
    return {"before": [r.line() for r in before if not r.passed], "new_failures": [r.line() for r in new_fail],
            "released": released, "count": len(after)}


def format_carve(rep: dict) -> str:
    lines = [f"CARVE  {rep['asset']}  {rep['path']}  res {rep['res']}  ({rep['candidates']} candidates, "
             f"{rep['trials']} trials, {rep['seconds']:.0f} s)"]
    lines.append("VIEWS  outline soft IoU from match before -> after; classes (areas) and divisions (borders on "
                 "boundaries / boundaries on borders) from structure; fitting score")
    for v in rep["views"]:
        sb, sa = v["structure_before"], v["structure_after"]
        lines.append(f"  reference[{v['index']}] {v['view']:<6} IoU {v['iou_before']:.3f} -> {v['iou_after']:.3f} "
                     f"({v['iou_after'] - v['iou_before']:+.3f})   classes {sb['classes']:.0%} -> {sa['classes']:.0%}   "
                     f"divisions {sb['region_borders_on_part_boundaries']:.0%}/{sb['part_boundaries_on_borders']:.0%} -> "
                     f"{sa['region_borders_on_part_boundaries']:.0%}/{sa['part_boundaries_on_borders']:.0%}   "
                     f"score {v['score_start']:.4f} -> {v['score_after']:.4f}"
                     + ("   (held out: measured, not carved to)" if v.get("holdout") else ""))
    lines.append(f"  mean IoU  {rep['mean_before']:.4f} -> {rep['mean_after']:.4f}  "
                 f"({rep['mean_after'] - rep['mean_before']:+.4f})")
    if rep["symmetry"]:
        lines.append(f"SYMMETRY  outlines include their mirror images ({', '.join(rep['symmetry'])}); one-sided "
                     f"pieces are written with mirror")
    kept = rep["kept"]
    lines.append(f"KEPT  {len(kept)}: " + (", ".join(
        f"{k['name']} ({k['kind']}" + (f", {k['material']}" if k["material"] else f", {k['from']} solids")
        + (", mirrored" if k["mirror"] else "") + ")" for k in kept) if kept else "nothing raised the score"))
    if rep["refused"]:
        lines.append(f"REFUSED  {len(rep['refused'])}: " + "; ".join(f"{r['name']} {r['why']}" for r in rep["refused"][:12])
                     + (" ..." if len(rep["refused"]) > 12 else ""))
    ck = rep.get("checks")
    if ck:
        if ck["released"]:
            lines.append(f"CHECKS  released (they left parts floating): {'; '.join(ck['released'])}")
        if ck["new_failures"]:
            lines.append(f"CHECKS  {len(ck['new_failures'])} check(s) passed before and fail now:")
            lines += [f"  {line}" for line in ck["new_failures"]]
        else:
            lines.append(f"CHECKS  {ck['count']} run: none that passed before fails now")
    if rep.get("written"):
        lines.append(f"WROTE  {rep['written']}")
    elif rep.get("refused_write"):
        lines.append(f"WROTE  nothing: {rep['refused_write']}")
    else:
        lines.append("WROTE  nothing (dry run): --write changes the asset file, --out PATH writes a copy")
    return "\n".join(lines) + "\n"
