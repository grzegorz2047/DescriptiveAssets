"""`sightline detail`: the grooves, seams and panel lines a picture shows inside its elements, drawn on the
model as dark `strokes` paints that follow the surface.

`structure`'s DETAIL line counts the picture's inner edges that have a model part boundary within NEAR_PX.
The edges that have none and lie on no region border are what the model lacks: chains of them become
open polylines, grouped by the solid under them, one `op: paint` `strokes` entry per solid and view
(width GROOVE_WIDTH_PX picture pixels, the dark material of the model, a side band from the surface's
depth like a conformed paint). Each view is measured before and after with `compare_structure`; its
grooves stay when DETAIL rose and classes and divisions did not fall by more than FLOOR.
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

from . import fit as fit_mod
from . import reference as ref_mod
from . import structure as st_mod
from .errors import SightlineError
from .loader import MAX_PAINTS, asset_from_data
from .model import STROKES_MAX, STROKES_MAX_POINTS
from .style import display_path, read_yaml, resolve_root

__all__ = ["detail_asset", "format_detail", "missing_edges", "chains", "simplify_open"]

GROOVE_WIDTH_PX = 1.5  # picture pixels at the comparison size
GROOVE_MIN_WIDTH = 0.004  # m
MIN_CHAIN_PX = 8  # shorter edge chains are specks
CHAIN_JUMP_PX = 2.5  # a chain breaks where the next edge pixel is farther than this
CHAIN_EPS_PX = 1.0  # Douglas-Peucker tolerance for one polyline (grows until MAX_LINE_POINTS fit)
MAX_LINE_POINTS = 24
MAX_LINES = 96  # polylines in one strokes entry (<= STROKES_MAX)
FLOOR = 0.005  # classes and divisions of a view may not drop by more than this
MARGIN = st_mod.PAINT_MARGIN


def _boundaries(ids: np.ndarray) -> np.ndarray:
    b = np.zeros(ids.shape, dtype=bool)
    b[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    b[1:, :] |= ids[1:, :] != ids[:-1, :]
    return b


def missing_edges(rep: dict) -> np.ndarray:
    """The picture's inner edges (inside both outlines' 2 px erosions) with no model part boundary and no
    picture region border within NEAR_PX: the detail the model lacks."""
    a = rep["_arrays"]
    R_, M_, part, labels, edge = a["ref"], a["model"], a["part"], a["labels"], a["edge"]
    inner = ref_mod._erode(ref_mod._erode(R_)) & ref_mod._erode(ref_mod._erode(M_))
    near = st_mod.NEAR_PX
    return edge & inner & ~st_mod._dilate(_boundaries(part), near) & ~st_mod._dilate(_boundaries(labels), near)


def _walk(px: np.ndarray) -> list[np.ndarray]:
    """Order the pixels of one edge component into chains: a nearest-neighbour walk from the pixel farthest
    from the centre, broken where the next pixel is more than CHAIN_JUMP_PX away."""
    pts = px.astype(np.float64)
    left = np.ones(len(pts), dtype=bool)
    out = []
    cur = None
    chain: list[int] = []
    centre = pts.mean(axis=0)
    while left.any():
        if cur is None:
            idx = np.flatnonzero(left)
            cur = int(idx[np.argmax(((pts[idx] - centre) ** 2).sum(axis=1))])
            left[cur] = False
            chain = [cur]
            continue
        idx = np.flatnonzero(left)
        d = np.hypot(*(pts[idx] - pts[cur]).T)
        j = int(np.argmin(d))
        if d[j] <= CHAIN_JUMP_PX:
            cur = int(idx[j])
            left[cur] = False
            chain.append(cur)
        else:
            out.append(pts[chain])
            cur = None
    if chain and cur is not None:
        out.append(pts[chain])
    return [c for c in out if len(c) >= MIN_CHAIN_PX]


def chains(mask: np.ndarray) -> list[np.ndarray]:
    """Ordered pixel chains ((n, 2) [row, col]) of an edge mask, MIN_CHAIN_PX pixels or longer."""
    labels, sizes = ref_mod.label_components(st_mod._dilate(mask, 1))
    out = []
    for k in np.flatnonzero(sizes >= MIN_CHAIN_PX):
        px = np.argwhere((labels == k) & mask)
        if len(px) >= MIN_CHAIN_PX:
            out += _walk(px)
    return out


def simplify_open(line: np.ndarray, eps: float = CHAIN_EPS_PX, max_points: int = MAX_LINE_POINTS) -> np.ndarray:
    """Douglas-Peucker on an open polyline; `eps` grows until the result has at most `max_points`."""

    def dp(seg, tol):
        if len(seg) <= 2:
            return seg
        a, b = seg[0], seg[-1]
        ab = b - a
        L = float(np.hypot(*ab)) or 1e-12
        d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        i = int(np.argmax(d))
        if d[i] <= tol:
            return np.array([a, b])
        return np.vstack([dp(seg[: i + 1], tol)[:-1], dp(seg[i:], tol)])

    while True:
        out = dp(line, eps)
        if len(out) <= max_points:
            return out
        eps *= 1.5


def _entries(asset, rep: dict, prefix: str) -> tuple[list[dict], dict]:
    """Strokes paint entries for one reference: the lacking detail chains grouped by the solid under them."""
    tr, a = rep["_trace"], rep["_arrays"]
    cam, vb, pl, pt = tr["camera"], tr["vb"], tr["pl"], tr["pt"]
    n, yaw = st_mod._view_frame(cam)
    right, up = np.asarray(cam.right, dtype=np.float64), np.asarray(cam.up, dtype=np.float64)
    side_view = abs(n[0]) > 0.5
    h_sign = ref_mod._axes(cam)[0][1]
    v_of = tr["v_of"]

    def h_of(px):
        return h_sign * tr["h_of"](px)

    mpp = float(cam.mpp)
    width = max(GROOVE_MIN_WIDTH, GROOVE_WIDTH_PX * mpp)
    s_map = vb.point.astype(np.float64) @ n
    parts_all = asset.scene().parts
    dark = st_mod._class_materials(asset, rep)["dark"]
    view_tag = "" if rep["view"] == "front" else f"{rep['view']}_"
    missing = missing_edges(rep)
    groups: dict[int, dict] = {}
    n_chains = 0
    for chain in chains(missing):
        rows, cols = chain[:, 0].astype(int), chain[:, 1].astype(int)
        vr, vc = rows - pt, cols - pl
        ok = (vr >= 0) & (vr < vb.hit.shape[0]) & (vc >= 0) & (vc < vb.hit.shape[1])
        if not ok.all():
            continue
        hit = vb.hit[vr, vc]
        if hit.sum() < MIN_CHAIN_PX:
            continue
        under = vb.part[vr[hit], vc[hit]]
        under = under[(under >= 0)]
        under = np.array([i for i in under if parts_all[int(i)].is_solid], dtype=np.int64)
        if under.size == 0:
            continue
        ids, cnt = np.unique(under, return_counts=True)
        target = int(ids[np.argmax(cnt)])
        poly = simplify_open(chain)
        pts = np.stack([h_of(poly[:, 1] + 0.5), v_of(poly[:, 0] + 0.5)], axis=1)
        g = groups.setdefault(target, {"lines": [], "rows": [], "s": []})
        g["lines"].append(pts)
        g["rows"].append(rows[hit])
        g["s"].append(s_map[vr[hit], vc[hit]])
        n_chains += 1
    entries, lines_total = [], 0
    for target, g in sorted(groups.items(), key=lambda kv: -sum(len(x) for x in kv[1]["lines"])):
        lines = g["lines"][:MAX_LINES]
        while sum(len(x) for x in lines) > STROKES_MAX_POINTS:
            lines = lines[:-1]
        rows = np.concatenate(g["rows"])
        ss = np.concatenate(g["s"])
        urows = np.unique(rows)
        side = None
        if len(urows) >= 2:
            hi_s = np.array([ss[rows == r].max() for r in urows])
            lo_s = np.array([ss[rows == r].min() for r in urows])
            band = st_mod._side_band(urows.astype(np.float64), hi_s + MARGIN, lo_s - MARGIN, v_of, mpp)
            from .model import _self_intersects
            if not _self_intersects(band):
                side = band
        allp = np.concatenate(lines)
        x0, x1 = allp[:, 0].min() - width / 2, allp[:, 0].max() + width / 2
        y0, y1 = allp[:, 1].min() - width / 2, allp[:, 1].max() + width / 2
        if side is not None:
            lo_v, hi_v = max(y0, side[:, 1].min()), min(y1, side[:, 1].max())
            if hi_v - lo_v < 1e-3:
                side = None
        if side is None:
            lo_v, hi_v = y0, y1
            depth = float(ss.max() - ss.min()) + 2.0 * MARGIN
            cs = float(ss.max() + ss.min()) / 2.0
        else:
            depth = None
            cs = float(side[:, 0].min() + side[:, 0].max()) / 2.0
        ch, cv = (x0 + x1) / 2.0, (lo_v + hi_v) / 2.0
        center = right * ch + up * cv + n * cs
        name = f"{prefix}_{view_tag}{parts_all[target].name}"
        e = {"name": name, "shape": "strokes",
             "points": [[[round(float(x - ch), 4), round(float(y - cv), 4)] for x, y in line] for line in lines],
             "width": round(width, 4)}
        if side is not None:
            e["side"] = [[round(float(z - cs), 4), round(float(y - cv), 4)] for z, y in side]
        else:
            e["depth"] = round(depth, 4)
        if abs(yaw) > 1e-6:
            e["rotate"] = [0, round(float(yaw), 3), 0]
        e.update({"op": "paint", "from": [parts_all[target].name], "material": dark,
                  "at": [round(float(c), 4) for c in center]})
        if side_view:
            e["mirror"] = "x"
        entries.append(e)
        lines_total += len(lines)
    facts = {"missing_px": int(missing.sum()), "chains": n_chains, "lines": lines_total, "width_m": round(width, 4)}
    return entries, facts


def _append_text(text: str, entries: list[dict], view: str) -> str:
    if not entries:
        return text
    indent = "" if "\nparts:\n- " in text else "  "
    blk = [f"{indent}# --- sightline detail: grooves and panel lines the {view} picture shows inside its elements"]
    blk += [f"{indent}- " + json.dumps(e, separators=(", ", ": ")).replace('"', "") for e in entries]
    head, sep, tail = text.partition("\nchecks:")
    return head.rstrip("\n") + "\n" + "\n".join(blk) + "\n" + sep + tail


def _figures(rep: dict) -> dict:
    return {"detail": rep["edges_on_part_boundaries"], "detail_model": rep["part_boundaries_on_edges"],
            "classes": rep["class_agreement_areas"],
            "divisions": (rep["region_borders_on_part_boundaries"] + rep["part_boundaries_on_borders"]) / 2.0,
            "rb": rep["region_borders_on_part_boundaries"], "pb": rep["part_boundaries_on_borders"]}


def detail_asset(path, root=None, *, res: int = 1024, views=None, checks: bool = True, prefix: str = "groove",
                 log=None) -> dict:
    """Draw the detail the reference pictures show inside their elements onto the asset at `path` as
    `strokes` paints; returns the report with the new YAML text (`report["text"]`). The file is not
    written. `views`: reference indices (default all). `res` 1024 by default: at 512 most seams of a
    concept picture fall under the edge detector's thresholds (the robot: DETAIL 90% at 512, 66% at 1024)."""
    t0 = time.perf_counter()
    log = log or (lambda msg: None)
    path = pathlib.Path(path)
    root = resolve_root(root, start=path.parent)
    text = path.read_text(encoding="utf-8")
    data, lines = read_yaml(text, display_path(path, root))
    asset = asset_from_data(data, lines, path, root)
    if not asset.references:
        raise SightlineError(f"asset '{asset.name}' has no reference picture to read detail from",
                             where=display_path(path, root),
                             hint="add reference: {image: refs/<name>.png, view: front, height: 2.0}")
    wanted = list(range(len(asset.references))) if views is None else list(views)
    for i in wanted:
        if not (isinstance(i, int) and 0 <= i < len(asset.references)):
            raise SightlineError(f"reference index {i} out of range", hint=f"0..{len(asset.references) - 1}")
    paints = sum(1 for p in asset.parts if p.is_paint)
    rows = []
    kept_entries = 0
    for i in wanted:
        log(f"reference[{i}] {asset.references[i].view}: reading the picture's inner edges")
        rep = st_mod.compare_structure(asset, i, res=res)
        before = _figures(rep)
        entries, facts = _entries(asset, rep, prefix)
        room = MAX_PAINTS - paints
        if len(entries) > room:
            facts["dropped_for_budget"] = len(entries) - room
            entries = entries[:room]
        row = {"index": i, "view": rep["view"], "before": before, "entries": len(entries), **facts, "kept": False}
        if not entries:
            row["after"] = before
            row["why"] = "nothing to draw"
            rows.append(row)
            continue
        trial_text = _append_text(text, entries, rep["view"])
        trial_data, trial_lines = read_yaml(trial_text, display_path(path, root))
        trial = asset_from_data(trial_data, trial_lines, path, root)
        log(f"reference[{i}]: {len(entries)} strokes entries, {facts['lines']} lines; measuring")
        after = _figures(st_mod.compare_structure(trial, i, res=res))
        row["after"] = after
        if after["detail"] <= before["detail"] + 1e-9:
            row["why"] = "DETAIL did not rise"
        elif after["classes"] < before["classes"] - FLOOR:
            row["why"] = f"classes fell {before['classes']:.1%} -> {after['classes']:.1%}"
        elif after["divisions"] < before["divisions"] - FLOOR:
            row["why"] = f"divisions fell {before['divisions']:.1%} -> {after['divisions']:.1%}"
        else:
            row["kept"] = True
            text, asset = trial_text, trial
            paints += len(entries)
            kept_entries += len(entries)
        rows.append(row)
    rep_out = {"asset": asset.name, "path": display_path(path, root), "res": res, "views": rows,
               "entries": kept_entries, "paints": paints, "text": text}
    if checks and kept_entries:
        log("running the checks before and after")
        base = asset_from_data(data, lines, path, root)
        before_c, _ = fit_mod._run_checks(base)
        after_c, _ = fit_mod._run_checks(asset)
        rep_out["checks"] = {"before": sum(1 for c in before_c if c.passed), "after": sum(1 for c in after_c if c.passed),
                             "total": len(after_c),
                             "new_failures": [c.line() for b, c in zip(before_c, after_c) if b.passed and not c.passed]}
    rep_out["seconds"] = round(time.perf_counter() - t0, 1)
    return rep_out


def format_detail(rep: dict) -> str:
    lines = [f"DETAIL  {rep['asset']}  {rep['path']}  res {rep['res']}  ({rep['entries']} strokes entries kept, "
             f"{rep['paints']} paints now, {rep['seconds']:.0f} s)"]
    lines.append("VIEWS  inner edges of the picture with a model boundary within 2 px (DETAIL) before -> after; "
                 "classes; divisions; lines drawn")
    for v in rep["views"]:
        b, a = v["before"], v["after"]
        lines.append(f"  reference[{v['index']}] {v['view']:<6} DETAIL {b['detail']:.0%} -> {a['detail']:.0%}   "
                     f"classes {b['classes']:.0%} -> {a['classes']:.0%}   divisions {b['divisions']:.0%} -> {a['divisions']:.0%}   "
                     f"{v['lines']} lines in {v['entries']} entries ({v['missing_px']} px lacking)   "
                     + ("kept" if v["kept"] else f"dropped: {v.get('why', '')}")
                     + (f"   ({v['dropped_for_budget']} entries over the paint budget)" if v.get("dropped_for_budget") else ""))
    ck = rep.get("checks")
    if ck:
        lines.append(f"CHECKS {ck['before']}/{ck['total']} passed before, {ck['after']}/{ck['total']} after"
                     + (": " + "; ".join(ck["new_failures"]) if ck["new_failures"] else ""))
    if rep.get("written"):
        lines.append(f"WROTE  {rep['written']}")
    if rep.get("refused_write"):
        lines.append(f"NOT WRITTEN  {rep['refused_write']}")
    return "\n".join(lines) + "\n"
