"""Ground-truth bench for picture structure (sightline.structure).

A model whose parts, materials and depth are known is rendered as a "picture"; the structure analysis
runs on that picture alone and is scored against the truth:

- edges: picture edges vs the visible element boundaries (part changes where the material changes,
  the depth jumps or the surface creases; depth jumps inside a part), precision/recall/F1 at 2 px;
- elements: true visible elements (parts joined across boundaries nobody can see) recovered as their
  own region (best region IoU >= 0.5), and region purity;
- classes: dark/bright/accent read from the lit picture vs the same rule on the unlit base colours;
- relief: for neighbouring elements at clearly different depths, does the relief guess put the nearer
  one in front.

Pictures: `clean` (no outlines), `outline` (the engine's default look) and `paint` (no outlines,
blurred, noisy, JPEG) to stand in for painted concepts.

    python tools/structure_bench.py [asset ...] [--res 512] [--views front] [--variants clean,paint]
"""

from __future__ import annotations

import argparse
import io
import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sightline import loader, render, structure as S, reference as R  # noqa: E402

TOL = 2
EVEN = True  # even out one-sided lighting on symmetric pictures before reading classes
CREASE_DEG = 25.0


def truth(vr, scene):
    part = vr.part
    cov = vr.coverage & (part >= 0)
    depth = np.where(cov, vr.buffers.depth, np.inf)
    normal = vr.buffers.normal
    mats = [p.material for p in scene.parts]
    mpp = vr.mpp
    jump = max(3.0 * mpp, 0.01)
    H, W = part.shape
    vis = np.zeros((H, W), dtype=bool)
    pairs_invisible: dict[tuple[int, int], list[int]] = {}
    for (sa, sb) in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                     ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        pa, pb = part[sa], part[sb]
        both = cov[sa] & cov[sb]
        da, db = depth[sa], depth[sb]
        dj = np.abs(da - db) > jump
        cosang = (normal[sa] * normal[sb]).sum(-1)
        crease = cosang < math.cos(math.radians(CREASE_DEG))
        diff = pa != pb
        matdiff = np.zeros_like(diff)
        if diff.any():
            ma = np.array(mats, dtype=object)
            idx = both & diff
            matdiff[idx] = ma[pa[idx]] != ma[pb[idx]]
        b = both & ((diff & (matdiff | dj | crease)) | (~diff & dj))
        vis[sa] |= b
        vis[sb] |= b
        inv = both & diff & ~b
        for a, c in zip(pa[inv].ravel(), pb[inv].ravel()):
            k = (min(a, c), max(a, c))
            pairs_invisible.setdefault(k, [0, 0])[0] += 1
        for a, c in zip(pa[both & diff].ravel(), pb[both & diff].ravel()):
            k = (min(a, c), max(a, c))
            pairs_invisible.setdefault(k, [0, 0])[1] += 1
    # elements: parts joined across mostly-invisible shared boundaries
    n = int(part.max()) + 1
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, c), (inv, tot) in pairs_invisible.items():
        if tot and inv / tot >= 0.8:
            ra, rc = find(a), find(c)
            if ra != rc:
                parent[max(ra, rc)] = min(ra, rc)
    elem = np.full(part.shape, -1, dtype=np.int32)
    elem[cov] = np.array([find(i) for i in range(n)])[part[cov]]
    return cov, vis, elem, depth


def base_image(vr, scene, style):
    out = np.zeros(vr.part.shape + (3,), dtype=np.float32)
    for pi in np.unique(vr.part[vr.part >= 0]):
        out[vr.part == pi] = style.material(scene.parts[pi].material).color
    return out * 255.0


def degrade(rgb: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.9))
    a = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    a = a + rng.normal(0, 4.0, a.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=80)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).astype(np.float32)


def near(m: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        m = R._dilate(m)
    return m


def score(rgb, cov, vis, elem, depth, base, mpp):
    edge, mag, _ = S.edges(rgb, cov)
    labels, cls, lab, _ = S.regions(rgb, cov, edge, mag, class_rgb=S.even_lighting(rgb, cov) if EVEN else None)
    inner = R._erode(R._erode(cov))
    t_edge = vis & inner
    p = edge & inner
    prec = (p & near(t_edge, TOL)).sum() / max(1, p.sum())
    rec = (t_edge & near(p, TOL)).sum() / max(1, t_edge.sum())
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    # elements
    n_obj = cov.sum()
    ids, sizes = np.unique(elem[cov], return_counts=True)
    big = ids[sizes >= 0.002 * n_obj]
    rec_el, found = 0, 0
    for e in big:
        me = elem == e
        reg, cnt = np.unique(labels[me], return_counts=True)
        best = 0.0
        for r_, c_ in zip(reg, cnt):
            if r_ < 0:
                continue
            iou = c_ / (me.sum() + (labels == r_).sum() - c_)
            best = max(best, iou)
        found += best >= 0.5
    el_recall = found / max(1, len(big))
    # purity: share of region pixels in the region's majority element
    pur_num = 0
    for r_ in np.unique(labels[cov]):
        if r_ < 0:
            continue
        sel = labels == r_
        _, c = np.unique(elem[sel], return_counts=True)
        pur_num += c.max()
    purity = pur_num / max(1, n_obj)
    # classes: lit picture vs base colours with the same rule
    # only materials that are clearly dark (L < 35), clearly bright (L > 65) or clearly saturated count
    blab = R.srgb_to_lab(base / 255.0)
    bL, bC = blab[..., 0], np.hypot(blab[..., 1], blab[..., 2])
    bcls = np.full(cov.shape, -1)
    bcls[cov & (bL < 35)] = 0
    bcls[cov & (bL > 65)] = 1
    bcls[cov & (bC >= 40) & (bL >= 25)] = 2
    clear = bcls >= 0
    pred = np.where((cls == 2) & (bcls != 2), 1, cls)  # an accent over a bright material counts as bright
    class_acc = (pred[clear] == bcls[clear]).mean() if clear.any() else float("nan")
    # relief order between neighbouring elements
    regs = S.describe_regions(labels, cls, lab, rgb, edge, lambda x: x, lambda y: y)
    rank = {"raised plate": 2, "light or decal": 2, "surface": 1, "recessed (frame or gap)": 0}
    reg_rank = {r["label"]: rank[r["relief"]] for r in regs}
    el_rank = {}
    for e in big:
        me = elem == e
        reg, cnt = np.unique(labels[me], return_counts=True)
        okr = reg >= 0
        if okr.any():
            el_rank[int(e)] = reg_rank.get(int(reg[okr][cnt[okr].argmax()]), 1)
    good = tot = 0
    a = np.concatenate([elem[:, :-1].ravel(), elem[:-1, :].ravel()])
    b = np.concatenate([elem[:, 1:].ravel(), elem[1:, :].ravel()])
    da = np.concatenate([depth[:, :-1].ravel(), depth[:-1, :].ravel()])
    db = np.concatenate([depth[:, 1:].ravel(), depth[1:, :].ravel()])
    sel = (a != b) & (a >= 0) & (b >= 0) & np.isfinite(da) & np.isfinite(db)
    pairs = {}
    for x, y, u, v in zip(a[sel], b[sel], da[sel], db[sel]):
        k = (int(min(x, y)), int(max(x, y)))
        d = (u - v) if x < y else (v - u)  # depth(min) - depth(max)
        s = pairs.setdefault(k, [0.0, 0])
        s[0] += d
        s[1] += 1
    for (x, y), (dsum, cnt) in pairs.items():
        if cnt < 6 or x not in el_rank or y not in el_rank or el_rank[x] == el_rank[y]:
            continue
        d = dsum / cnt
        if abs(d) < max(0.01, 2 * mpp):
            continue
        tot += 1
        front_true = x if d < 0 else y
        front_guess = x if el_rank[x] > el_rank[y] else y
        good += front_true == front_guess
    relief_acc = good / tot if tot else float("nan")
    return {"edge_p": prec, "edge_r": rec, "edge_f1": f1, "elements": len(big), "el_recall": el_recall,
            "purity": purity, "class_acc": class_acc, "relief_acc": relief_acc, "relief_pairs": tot,
            "regions": int(labels.max()) + 1}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", nargs="*")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--views", default="front")
    ap.add_argument("--variants", default="clean,outline,paint")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--mode", default=None)
    args = ap.parse_args(argv)
    if args.mode:
        S.REGION_MODE = args.mode
    names = args.assets or loader.list_assets(args.root)
    variants = args.variants.split(",")
    rows = []
    for name in names:
        asset = loader.load_asset(name, args.root)
        scene = render.layout(asset, False, False).scene
        for view in args.views.split(","):
            vr_out = render.render_raw(asset, [view], args.res, scale_figure=False, outline=True, ground=False)[view]
            vr = render.render_raw(asset, [view], args.res, scale_figure=False, outline=False, ground=False)[view]
            cov, vis, elem, depth = truth(vr, scene)
            base = base_image(vr, scene, asset.style)
            for var in variants:
                rgb = {"clean": vr.rgb.astype(np.float32), "outline": vr_out.rgb.astype(np.float32),
                       "paint": degrade(vr.rgb.astype(np.float32), cov)}[var]
                m = score(rgb, cov, vis, elem, depth, base, vr.mpp)
                rows.append((name, view, var, m))
                print(f"{name:12s} {view:6s} {var:8s} edges P {m['edge_p']:.2f} R {m['edge_r']:.2f} F1 {m['edge_f1']:.2f} | "
                      f"elements {m['elements']:3d} found {m['el_recall']:.2f} purity {m['purity']:.2f} | "
                      f"class {m['class_acc']:.2f} | relief {m['relief_acc']:.2f} ({m['relief_pairs']}) | regions {m['regions']}",
                      flush=True)
    for var in variants:
        ms = [m for (_, _, v, m) in rows if v == var]
        if not ms:
            continue
        mean = {k: float(np.nanmean([m[k] for m in ms])) for k in ("edge_p", "edge_r", "edge_f1", "el_recall", "purity", "class_acc", "relief_acc")}
        print(f"MEAN {var:8s} edges P {mean['edge_p']:.2f} R {mean['edge_r']:.2f} F1 {mean['edge_f1']:.2f} | "
              f"found {mean['el_recall']:.2f} purity {mean['purity']:.2f} | class {mean['class_acc']:.2f} | relief {mean['relief_acc']:.2f}")


if __name__ == "__main__":
    main()


def bench_external(truth_dir: str) -> dict:
    """Score the structure analysis on a Blender picture with its ID map (tools/blender_truth.py)."""
    import json

    d = pathlib.Path(truth_dir)
    pic = np.asarray(Image.open(d / "picture.png").convert("RGBA")).astype(np.float32)
    ids = np.asarray(Image.open(d / "ids.png").convert("RGBA"))
    meta = json.loads((d / "ids.json").read_text())
    cov = pic[..., 3] >= 128
    rgb = pic[..., :3] * (pic[..., 3:] / 255.0) + 255.0 * (1 - pic[..., 3:] / 255.0)
    # Blender stores the flat ID colours (set as linear values) sRGB-encoded: match the nearest one
    keys = np.array([[int(k[1:3], 16), int(k[3:5], 16), int(k[5:7], 16)] for k in meta["ids"]], dtype=np.float64) / 255.0
    enc = np.where(keys <= 0.0031308, 12.92 * keys, 1.055 * keys ** (1 / 2.4) - 0.055) * 255.0
    px = ids[..., :3].reshape(-1, 3).astype(np.float64)
    dist = ((px[:, None, :] - enc[None, :, :]) ** 2).sum(-1)
    near_id = dist.argmin(1)
    elem = np.where((dist.min(1) <= 27.0) & (ids[..., 3].reshape(-1) >= 128), near_id, -1).reshape(cov.shape).astype(np.int32)
    cov &= elem >= 0
    # an element is one connected piece of one ID (both legs share a material but are two elements)
    pieces = np.full(elem.shape, -1, dtype=np.int32)
    off = 0
    for i in np.unique(elem[cov]):
        lc, sz = R.label_components(elem == i)
        pieces[lc >= 0] = lc[lc >= 0] + off
        off += int(sz.size)
    base_elem = elem
    elem = pieces
    vis = np.zeros_like(cov)
    for (sa, sb) in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                     ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        b = cov[sa] & cov[sb] & (elem[sa] != elem[sb])
        vis[sa] |= b
        vis[sb] |= b
    base = np.zeros(rgb.shape, dtype=np.float32)
    cols = list(meta["ids"].values())
    for i, v in enumerate(cols):
        lin = np.clip(np.array(v["color_linear"], dtype=np.float64), 0, 1)
        srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
        base[base_elem == i] = srgb * 255.0
    depth = np.full(cov.shape, np.inf)
    return score(rgb, cov, vis, elem, depth, base, 0.001)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--external":
    pass
