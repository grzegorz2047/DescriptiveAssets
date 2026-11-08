"""`sightline score`: every reference of an asset on one scoreboard, and the concept-vs-model picture (SPEC 12.7).

One row per reference: outline IoU and the contour agreement from `match`, classes and divisions from
`structure`, the ceiling the opposite picture allows. `--against` puts an earlier version next to it with
the differences. `--out` writes the picture the designer wants after every round: the concept in each
view on top, one row per version under it, all at one scale, with the IoU under each tile.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import reference as ref_mod
from . import render
from . import structure as st_mod
from .errors import SightlineError
from .loader import load_asset

MARGIN = 0.06  # of the tile's larger side, around the silhouettes
TILE_MPP = 0.004  # meters per pixel of the rendered tiles (the pictures are resampled to it)
OUT_PX = 0.0035  # meters per pixel of the written picture


def score_asset(asset, res: int = 512, *, ceiling: bool = True) -> dict:
    """Per reference: `match` and `structure` figures of `asset` at `res`."""
    if isinstance(res, bool) or not isinstance(res, int) or not 64 <= res <= 2048:
        raise SightlineError("res must be an integer from 64 to 2048", hint="e.g. --res 512")
    if not asset.references:
        raise SightlineError(f"asset '{asset.name}' has no reference picture to score against",
                             hint="add reference: {image: refs/<name>.png, view: front, height: 2.0}")
    rows = []
    for i, spec in enumerate(asset.references):
        m = ref_mod.match_asset(asset, i, res, ceiling=ceiling)
        s = st_mod.compare_structure(asset, i, res=res)
        rows.append({
            "index": i, "view": spec.view, "image": m["reference"]["image"],
            "iou": m["iou"], "missing": m["missing"], "extra": m["extra"],
            "outline_picture": m["outline"]["picture"], "outline_model": m["outline"]["model"],
            "classes": s["class_agreement_areas"], "classes_strict": s["class_agreement"],
            "borders_on_boundaries": s["region_borders_on_part_boundaries"],
            "boundaries_on_borders": s["part_boundaries_on_borders"],
            "ceiling": m.get("ceiling"),
        })
    solids = sum(1 for p in asset.parts if p.is_solid and not p.helper)
    cuts = sum(1 for p in asset.parts if p.is_cut)
    paints = sum(1 for p in asset.parts if p.is_paint)
    return {"asset": asset.name, "res": res, "views": rows, "mean_iou": round(float(np.mean([r["iou"] for r in rows])), 4),
            "counts": {"solids": solids, "cuts": cuts, "paints": paints}}


def _delta(a, b) -> str:
    return f" ({a - b:+.3f})" if b is not None else ""


def format_score(rep: dict, against: dict | None = None) -> str:
    c = rep["counts"]
    lines = [f"SCORE  {rep['asset']}  res {rep['res']}  {c['solids']} solids, {c['cuts']} cuts, {c['paints']} paints"
             + (f"   (against {against['asset']}: differences in brackets)" if against else "")]
    lines.append("VIEWS  outline IoU | contour within 1 px: picture's / model's | classes (areas, strict) | "
                 "divisions: picture borders on part boundaries / boundaries on borders | ceiling")
    other = {r["index"]: r for r in (against or {"views": []})["views"]}
    for r in rep["views"]:
        o = other.get(r["index"])

        def d(key):
            return _delta(r[key], o[key]) if o else ""

        ceil = r.get("ceiling")
        ceil_txt = f"{ceil['pair_mean_max']:.3f} (with {ceil['view']})" if ceil else "-"
        lines.append(f"  [{r['index']}] {r['view']:<6} IoU {r['iou']:.3f}{d('iou')} | "
                     f"{r['outline_picture']:.0%} / {r['outline_model']:.0%} | "
                     f"{r['classes']:.0%}{d('classes')}, {r['classes_strict']:.0%} | "
                     f"{r['borders_on_boundaries']:.0%}{d('borders_on_boundaries')} / "
                     f"{r['boundaries_on_borders']:.0%}{d('boundaries_on_borders')} | {ceil_txt}")
    lines.append(f"  mean IoU {rep['mean_iou']:.4f}" + (_delta(rep["mean_iou"], against["mean_iou"]) if against else ""))
    if rep.get("image"):
        lines.append(f"IMAGE  {rep['image']}  (concept on top, one row per version, IoU under each tile)")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- the picture


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _crop(rgb: np.ndarray, mask: np.ndarray, mpp: float, box_w: float, box_h: float) -> Image.Image:
    """The silhouette centred in a box of box_w x box_h meters at `mpp`, white around it."""
    ys, xs = np.nonzero(mask)
    W, H = max(1, int(round(box_w / mpp))), max(1, int(round(box_h / mpp)))
    if not xs.size:
        return Image.new("RGB", (W, H), (255, 255, 255))
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    x0, y0 = int(round(cx - W / 2.0)), int(round(cy - H / 2.0))
    out = Image.new("RGB", (W, H), (255, 255, 255))
    out.paste(Image.fromarray(rgb), (-x0, -y0))
    return out


def _concept(asset, index: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The reference picture as (rgb, mask, meters per pixel), on white."""
    spec = asset.references[index]
    mask = None if str(spec.mask) == "auto" else spec.mask
    ref = ref_mod._analyze(spec.image, spec.view, spec.height, mask, asset.style, asset.root, None)
    x0, y0, x1, y1 = ref.bbox
    height = spec.height
    if height is None:  # as match scales it: to the model's height
        lo, hi = asset.scene().bounds(include_helpers=False)
        height = float(hi[1] - lo[1])
    mpp = height / max(1, y1 - y0)
    rgb = ref.rgb.astype(np.uint8).copy()
    if ref.alpha is not None:
        a = (ref.alpha.astype(np.float32) / 255.0)[..., None]
        rgb = (rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)
    else:
        rgb[~ref.mask] = 255
    return rgb, ref.mask, mpp


def write_score_png(versions: list[tuple[str, object]], path, res: int = 256) -> pathlib.Path:
    """`versions` = [(label, asset), ...], the last one the current asset whose references set the views.
    Rows: the concept, then every version; one scale; IoU (at `res`) under each tile."""
    base = versions[-1][1]
    views = []
    for spec in base.references:
        if spec.view not in views:
            views.append(spec.view)
    if not views:
        raise SightlineError(f"asset '{base.name}' has no reference picture", hint="score needs reference: entries")
    rows = []  # (label, [(rgb, mask, mpp)] per view, [note] per view)
    tiles = []
    for i, spec in enumerate(base.references):
        if spec.view in [v for v, _ in tiles]:
            continue
        tiles.append((spec.view, _concept(base, i)))
    rows.append(("concept", [t for _, t in tiles], ["" for _ in views]))
    for label, asset in versions:
        imgs, notes = [], []
        rendered = render.render_raw(asset, views, mpp=TILE_MPP, scale_figure=False, outline=True, ground=False,
                                     light="view")
        for v in views:
            vr = rendered[v]
            rgb = vr.rgb.copy()
            rgb[~vr.coverage] = 255
            imgs.append((rgb, vr.coverage, TILE_MPP))
            k = next((n for n, r in enumerate(asset.references) if r.view == v), None)
            notes.append("not used" if k is None else f"IoU {ref_mod.match_asset(asset, k, res, ceiling=False)['iou']:.2f}")
        rows.append((label, imgs, notes))
    # one box per view: the largest silhouette extent over the rows, plus a margin
    boxes = []
    for col in range(len(views)):
        w = h = 0.0
        for _, imgs, _ in rows:
            rgb, mask, mpp = imgs[col]
            ys, xs = np.nonzero(mask)
            if xs.size:
                w = max(w, (xs.max() - xs.min() + 1) * mpp)
                h = max(h, (ys.max() - ys.min() + 1) * mpp)
        m = MARGIN * max(w, h, 0.1)
        boxes.append((w + 2 * m, h + 2 * m))
    cols = [max(8, int(round(bw / OUT_PX))) for bw, _ in boxes]
    rh = max(8, int(round(max(bh for _, bh in boxes) / OUT_PX)))
    lab_w, gap, cap, top = 150, 10, 26, 30
    out = Image.new("RGB", (lab_w + sum(cols) + gap * len(cols), top + len(rows) * (rh + cap + gap)), (255, 255, 255))
    d = ImageDraw.Draw(out)
    x = lab_w
    for v, w in zip(views, cols):
        d.text((x + 6, 6), v, fill=(0, 0, 0), font=_font(18))
        x += w + gap
    for r, (label, imgs, notes) in enumerate(rows):
        y = top + r * (rh + cap + gap)
        d.text((6, y + rh // 2), label[:18], fill=(0, 0, 0), font=_font(18))
        x = lab_w
        for (rgb, mask, mpp), note, w, (bw, bh) in zip(imgs, notes, cols, boxes):
            tile = _crop(rgb, mask, mpp, bw, bh).resize((w, rh), Image.LANCZOS)
            out.paste(tile, (x, y))
            d.rectangle((x, y, x + w - 1, y + rh - 1), outline=(220, 220, 220))
            d.text((x + 6, y + rh + 3), note, fill=(40, 40, 40), font=_font(16))
            x += w + gap
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def load_versions(specs: list[str], root) -> list[tuple[str, object]]:
    """`label=path` or `path` items -> [(label, asset)]."""
    out = []
    for s in specs:
        label, _, target = s.partition("=") if "=" in s else ("", "", s)
        asset = load_asset(target, root)
        out.append((label or asset.name, asset))
    return out
