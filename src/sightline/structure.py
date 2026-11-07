"""Picture structure (SPEC 12.4): edges, regions and a relief guess inside a reference picture, and how
the model's parts line up with them.

`match` compares outlines. Inside the outline a concept shows how the object is divided: plates, a
dark frame between them, lights, trims. This module finds those divisions in the picture (edges of
colour change, closed regions between them, a dark/bright/accent class per region and a guess of
which regions stand out and which are recessed) and, for an asset, says where the model's part
boundaries and materials agree with them.

The relief guess is a heuristic, not depth: bright regions framed by dark seams or strong edges read
as raised plates, dark regions as recessed frame or gaps, saturated small regions as lights or
decals. `bench` measures it against rendered models whose real depth is known.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw

from .errors import SightlineError
from . import reference as ref_mod

# Edges: Sobel magnitude on a lightly smoothed Lab image (L plus chroma), thinned by non-maximum
# suppression, then hysteresis between two thresholds taken from the object's own gradient
# distribution, with absolute floors (Lab units per pixel) so flat pictures do not invent edges.
EDGE_LOW_Q = 0.70
EDGE_HIGH_Q = 0.88
EDGE_LOW_MIN = 2.5
EDGE_HIGH_MIN = 6.0
CHROMA_WEIGHT = 1.0
WEBER = True  # gradients relative to local lightness: the same step means more in a dark area
WEBER_K = 16.0  # L units added to the local lightness (keeps near-black noise from exploding)
SMOOTH_JOIN = True  # join neighbours whose shared boundary has no edge (a shading change, not a material one)
SMOOTH_Q = 0.7  # a boundary is smooth when its mean gradient is below SMOOTH_Q x the low edge threshold
# Regions: closed areas between edges; smaller ones are merged into the neighbour they share the
# longest boundary with (same class preferred), then neighbours with nearly the same colour and a weak
# shared boundary are joined.
REGION_MIN = 0.0015  # share of the object's pixels
REGION_MIN_PX = 24
ACCENT_MIN_PX = 6  # lights and decals are small: keep them unless they are specks
MERGE_DE = 7.0
MERGE_EDGE = 0.35  # share of a shared boundary that lies on strong edges, below which a join is allowed
# Classes: dark/bright split by Otsu on L inside the object (fixed L 30 when the picture is not
# two-toned), accents by chroma.
ACCENT_C = 24.0
DARK_FIXED = 30.0
MATERIAL_DARK_FIXED = 50.0  # unlit material colours of a one-toned model: darker than mid grey is dark
TWO_TONE_GAP = 18.0
RAISED_BORDER = 0.4  # share of a bright region's boundary next to dark regions or strong edges
SHADE_DL = 15.0  # lightness change across a region that is worth reporting
REPORT_MIN = 0.004  # regions listed in the text report: share of the object
REPORT_MAX = 40
NEAR_PX = 2  # edge alignment tolerance
CLASS_BAND = 1  # px along class borders left out of the area class agreement

_HUES = ((20, "red"), (50, "orange"), (75, "yellow"), (160, "green"), (200, "cyan"), (290, "blue"), (335, "purple"), (361, "red"))


# ----------------------------------------------------------------------------- image helpers


def _box_blur(a: np.ndarray) -> np.ndarray:
    """3x3 box blur of (H, W) or (H, W, C), edges replicated."""
    p = np.pad(a, ((1, 1), (1, 1)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / 9.0


def _sobel(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(c, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx / 8.0, gy / 8.0


def _dilate(m: np.ndarray, n: int = 1) -> np.ndarray:
    for _ in range(n):
        m = ref_mod._dilate(m)
    return m


def otsu(values: np.ndarray, lo: float = 0.0, hi: float = 100.0, bins: int = 100) -> float:
    """Otsu threshold of `values` on [lo, hi]."""
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) / max(1, hist.sum())
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * centers)
    mt = m0[-1]
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1.0
    return float(edges[int(np.argmax(between)) + 1])


def hue_name(a: float, b: float) -> str:
    h = math.degrees(math.atan2(b, a)) % 360.0
    for top, name in _HUES:
        if h < top:
            return name
    return "red"


# ----------------------------------------------------------------------------- core analysis


def edges(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Thin edges inside `mask`: (edge (H, W) bool, gradient magnitude (H, W) float32, facts).

    `rgb` is float 0..255. Pixels on the object's outline are not edges (the outline is `match`'s job).
    """
    lab = ref_mod.srgb_to_lab(_box_blur(rgb.astype(np.float32)) / 255.0).astype(np.float32)
    gxs, gys = zip(*(_sobel(lab[..., i]) for i in range(3)))
    w = np.array([1.0, CHROMA_WEIGHT, CHROMA_WEIGHT], dtype=np.float32)
    mag = np.sqrt(sum(w[i] * (gxs[i] ** 2 + gys[i] ** 2) for i in range(3))).astype(np.float32)
    if WEBER:
        mag = mag * ((50.0 + WEBER_K) / (np.maximum(lab[..., 0], 0.0) + WEBER_K))
    # Direction from the strongest channel's gradient (L unless chroma dominates).
    gx = gxs[0] + np.sign(gxs[0] + 1e-9) * 0.0
    gy = gys[0]
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    q = (((ang + 22.5) // 45).astype(np.int8)) % 4
    p = np.pad(mag, 1, mode="constant")
    h, wd = mag.shape
    c = p[1:-1, 1:-1]
    nb = {0: (p[1:-1, 2:], p[1:-1, :-2]), 1: (p[2:, 2:], p[:-2, :-2]), 2: (p[2:, 1:-1], p[:-2, 1:-1]), 3: (p[2:, :-2], p[:-2, 2:])}
    keep = np.zeros((h, wd), dtype=bool)
    for k, (a, b) in nb.items():
        sel = q == k
        keep |= sel & (c >= a) & (c >= b)
    inner = ref_mod._erode(ref_mod._erode(mask))
    vals = mag[inner]
    if vals.size == 0:
        return np.zeros_like(mask), mag, {"low": 0.0, "high": 0.0, "share": 0.0}
    lo = max(EDGE_LOW_MIN, float(np.quantile(vals, EDGE_LOW_Q)))
    hi = max(EDGE_HIGH_MIN, float(np.quantile(vals, EDGE_HIGH_Q)))
    weak = keep & inner & (mag >= lo)
    strong = weak & (mag >= hi)
    labels, sizes = ref_mod.label_components(_dilate(weak) & inner)  # 8-ish connectivity via dilation
    hit = np.unique(labels[strong & (labels >= 0)])
    good = np.zeros(sizes.size, dtype=bool)
    good[hit] = True
    edge = weak & (labels >= 0) & good[np.maximum(labels, 0)]
    facts = {"low": round(lo, 2), "high": round(hi, 2), "share": round(float(edge.sum()) / max(1, int(mask.sum())), 4)}
    return edge, mag, facts


def _classes(lab: np.ndarray, mask: np.ndarray, fixed: float = DARK_FIXED) -> tuple[np.ndarray, float]:
    """Per-pixel class: 0 dark, 1 bright, 2 accent (inside mask; -1 outside); the dark/bright threshold."""
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    vals = L[mask]
    t = otsu(vals) if vals.size else DARK_FIXED
    lo, hi = vals[vals < t], vals[vals >= t]
    if lo.size == 0 or hi.size == 0 or hi.mean() - lo.mean() < TWO_TONE_GAP:
        t = fixed
    cls = np.where(L < t, 0, 1).astype(np.int8)
    cls[(chroma >= ACCENT_C) & (L >= 25)] = 2
    cls[~mask] = -1
    return cls, t


def _adjacency(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique unordered label pairs that touch (4-neighbour) and how many pixel pairs touch."""
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    sel = (a != b) & (a >= 0) & (b >= 0)
    a, b = a[sel], b[sel]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    if lo.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    pairs, counts = np.unique(np.stack([lo, hi], axis=1), axis=0, return_counts=True)
    return pairs, counts


def _relabel(labels: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Follow `target` (label -> label) to its fixed point and renumber densely."""
    t = target.copy()
    for _ in range(64):
        nt = t[t]
        if np.array_equal(nt, t):
            break
        t = nt
    roots, dense = np.unique(t, return_inverse=True)
    out = labels.copy()
    sel = labels >= 0
    out[sel] = dense[labels[sel]]
    return out


REGION_MODE = "class"  # "rag": over-segment by edges, merge across weak boundaries; "class": class components
MERGE_COST = 1.0  # rag: merge neighbours whose boundary strength + colour difference cost is below this
L_WEIGHT = 0.5  # rag: lightness counts half in the colour difference (shading changes lightness smoothly)
ACCENT_MAX_SHARE = 0.03  # a saturated region larger than this is a colour, not a light or decal


def regions(rgb: np.ndarray, mask: np.ndarray, edge: np.ndarray | None = None, mag: np.ndarray | None = None,
            mode: str | None = None, class_rgb: np.ndarray | None = None):
    """Closed regions between edges: (labels (H, W) int32, -1 outside; per-pixel class; lab; facts).

    `class_rgb` (same shape as rgb): the colours the dark/bright/accent classes are read from, e.g. a
    picture with its one-sided lighting evened out (see `even_lighting`); edges still come from `rgb`.
    """
    if (mode or REGION_MODE) == "rag":
        return _regions_rag(rgb, mask, edge, mag)
    return _regions_class(rgb, mask, edge, mag, class_rgb)


SYMMETRY_MIN = 0.9  # silhouette left/right IoU from which one-sided lighting is evened out


def silhouette_symmetry(mask: np.ndarray) -> tuple[float, float]:
    """(left/right IoU of the mask mirrored about its bbox centre, that centre's x in pixels)."""
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return 0.0, 0.0
    c = (xs.min() + xs.max()) / 2.0
    mx = np.round(2 * c - xs).astype(int)
    ok = (mx >= 0) & (mx < mask.shape[1])
    mirrored = np.zeros_like(mask)
    mirrored[ys[ok], mx[ok]] = True
    inter = (mirrored & mask).sum()
    return float(inter / max(1, (mirrored | mask).sum())), float(c)


def even_lighting(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """For a left/right symmetric object: the picture averaged with its mirror image (inside both
    silhouettes), so light falling from one side no longer turns one side's material dark. None when the
    silhouette is not symmetric enough (then the materials may really differ between the sides)."""
    sym, c = silhouette_symmetry(mask)
    if sym < SYMMETRY_MIN:
        return None
    H, W = mask.shape
    xs = np.arange(W)
    mx = np.round(2 * c - xs).astype(int)
    ok = (mx >= 0) & (mx < W)
    mirror = np.zeros_like(rgb)
    mirror[:, ok] = rgb[:, mx[ok]]
    mmask = np.zeros_like(mask)
    mmask[:, ok] = mask[:, mx[ok]]
    out = rgb.astype(np.float32).copy()
    both = mask & mmask
    out[both] = (rgb[both].astype(np.float32) + mirror[both].astype(np.float32)) / 2.0
    return out


def _region_stats(labels: np.ndarray, lab: np.ndarray, n: int):
    sel = labels >= 0
    cnt = np.bincount(labels[sel], minlength=n).astype(np.float64)
    mean = np.zeros((n, 3))
    for i in range(3):
        mean[:, i] = np.bincount(labels[sel], weights=lab[..., i][sel], minlength=n) / np.maximum(cnt, 1)
    return cnt, mean


def _boundary_mean(labels: np.ndarray, value: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Mean of `value` over the pixel pairs that make each touching region pair's boundary."""
    n = int(labels.max()) + 1
    pcode = pairs[:, 0].astype(np.int64) * n + pairs[:, 1]
    tot = np.zeros(len(pairs))
    acc = np.zeros(len(pairs))
    for (la, lb, va, vb) in ((labels[:, :-1], labels[:, 1:], value[:, :-1], value[:, 1:]),
                             (labels[:-1, :], labels[1:, :], value[:-1, :], value[1:, :])):
        s = (la != lb) & (la >= 0) & (lb >= 0)
        a, b = la[s], lb[s]
        code = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
        idx = np.clip(np.searchsorted(pcode, code), 0, len(pcode) - 1)
        good = pcode[idx] == code
        np.add.at(tot, idx[good], 1.0)
        np.add.at(acc, idx[good], ((va[s] + vb[s]) / 2.0)[good])
    return acc / np.maximum(tot, 1.0)


def _max3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    out = a.copy()
    for dy in range(3):
        for dx in range(3):
            out = np.maximum(out, p[dy:dy + a.shape[0], dx:dx + a.shape[1]])
    return out


def _region_classes(labels: np.ndarray, lab: np.ndarray, mask: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """Class per region from its mean colour: dark/bright by Otsu over the region means (area weighted),
    accent when small and clearly more saturated than the object as a whole."""
    cnt, mean = _region_stats(labels, lab, n)
    L = mean[:, 0]
    chroma = np.hypot(mean[:, 1], mean[:, 2])
    pix_chroma = np.hypot(lab[..., 1], lab[..., 2])[mask]
    c_med = float(np.median(pix_chroma)) if pix_chroma.size else 0.0
    w = np.repeat(L, np.maximum(cnt.astype(int), 0))
    t = otsu(w) if w.size else DARK_FIXED
    lo, hi = w[w < t], w[w >= t]
    if lo.size == 0 or hi.size == 0 or hi.mean() - lo.mean() < TWO_TONE_GAP:
        t = DARK_FIXED
    rc = np.where(L < t, 0, 1)
    total = max(1.0, cnt.sum())
    accent = (chroma >= max(ACCENT_C, 2.0 * c_med)) & (L >= 25) & (cnt / total <= ACCENT_MAX_SHARE)
    rc[accent] = 2
    return rc, t


def _is_accent_label(labels: np.ndarray, acc: np.ndarray, n: int) -> np.ndarray:
    a = np.bincount(labels[acc & (labels >= 0)], minlength=n)
    tot = np.bincount(labels[labels >= 0], minlength=n)
    return a > 0.5 * np.maximum(tot, 1)


def _regions_rag(rgb, mask, edge=None, mag=None):
    if edge is None or mag is None:
        edge, mag, _ = edges(rgb, mask)
    lab = ref_mod.srgb_to_lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
    pix_cls, _ = _classes(lab, mask)
    acc = (pix_cls == 2) & mask
    # 1. over-segmentation: components between edges; saturated specks (lights) are their own pieces
    barrier = _dilate(edge) & mask
    labels, sizes = ref_mod.label_components(mask & ~barrier & ~acc)
    off = int(sizes.size)
    la, _ = ref_mod.label_components(_dilate(acc, 2) & mask)
    la[~acc] = -1
    labels[la >= 0] = la[la >= 0] + off
    keep = labels[acc]
    labels[acc] = -1
    labels = ref_mod._nearest_labels(labels, mask)
    labels[acc] = keep
    labels[~mask] = -1
    labels = _relabel(labels, np.arange(int(labels.max()) + 1))
    magd = _max3(mag)
    hi_thr = max(EDGE_HIGH_MIN, float(np.quantile(mag[mask], EDGE_HIGH_Q))) if mask.any() else EDGE_HIGH_MIN
    # 2. merge across weak boundaries: one merge per region per round, cheapest first (no chaining
    #    along a smooth gradient)
    for _ in range(40):
        n = int(labels.max()) + 1
        pairs, counts = _adjacency(labels)
        if pairs.size == 0:
            break
        cnt, mean = _region_stats(labels, lab, n)
        bmag = _boundary_mean(labels, magd, pairs)
        d = mean[pairs[:, 0]] - mean[pairs[:, 1]]
        de = np.sqrt((L_WEIGHT * d[:, 0]) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
        cost = bmag / hi_thr + de / MERGE_DE
        isacc = _is_accent_label(labels, acc, n)
        cost[isacc[pairs[:, 0]] != isacc[pairs[:, 1]]] = np.inf
        order = np.argsort(cost)
        used = np.zeros(n, dtype=bool)
        target = np.arange(n)
        merged = 0
        for i in order:
            if cost[i] >= MERGE_COST:
                break
            a, b = pairs[i]
            if used[a] or used[b]:
                continue
            used[a] = used[b] = True
            target[max(a, b)] = min(a, b)
            merged += 1
        if not merged:
            break
        labels = _relabel(labels, target)
    # 3. small regions join the neighbour they share the longest boundary with (lights may stay small)
    n_obj = int(mask.sum())
    min_px = max(REGION_MIN_PX, int(REGION_MIN * n_obj))
    for _ in range(12):
        n = int(labels.max()) + 1
        sizes = np.bincount(labels[labels >= 0], minlength=n)
        isacc = _is_accent_label(labels, acc, n)
        small = np.where(isacc, sizes < ACCENT_MIN_PX, sizes < min_px)
        if not small.any():
            break
        pairs, counts = _adjacency(labels)
        if pairs.size == 0:
            break
        target = np.arange(n)
        best = np.full(n, -1.0)
        for (a, b), c in zip(pairs, counts):
            for s, o in ((a, b), (b, a)):
                if small[s] and c > best[s]:
                    best[s], target[s] = c, o
        mutual = target[target] == np.arange(n)
        keep_root = mutual & (sizes >= sizes[target]) & (target != np.arange(n))
        target[keep_root] = np.arange(n)[keep_root]
        labels = _relabel(labels, target)
    n = int(labels.max()) + 1
    rc, t = _region_classes(labels, lab, mask, n)
    cls = np.full(mask.shape, -1, dtype=np.int8)
    cls[labels >= 0] = rc[labels[labels >= 0]]
    return labels.astype(np.int32), cls, lab, {"dark_threshold_L": round(t, 1), "regions": n}


def _regions_class(rgb: np.ndarray, mask: np.ndarray, edge: np.ndarray | None = None, mag: np.ndarray | None = None,
                   class_rgb: np.ndarray | None = None):
    if not mask.any():
        raise SightlineError("no object pixels to analyse", hint="the picture's mask is empty")
    if edge is None or mag is None:
        edge, mag, _ = edges(rgb, mask)
    lab = ref_mod.srgb_to_lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
    clab = lab if class_rgb is None else ref_mod.srgb_to_lab(np.asarray(class_rgb, np.float32) / 255.0).astype(np.float32)
    cls, t = _classes(clab, mask)
    if class_rgb is not None:  # evened colours decide dark/bright only: lights and decals come from the picture itself
        acc_orig = _classes(lab, mask)[0] == 2
        was_acc = cls == 2
        cls[was_acc] = np.where(clab[..., 0][was_acc] < t, 0, 1)
        cls[acc_orig] = 2
    barrier = _dilate(edge) & mask
    # Components of each class separately (a plate ends where the dark frame starts, even without a
    # closed edge), cut further by the edges; edge pixels then join the nearest region.
    labels = np.full(mask.shape, -1, dtype=np.int32)
    off = 0
    for c in (0, 1, 2):
        if c == 2:  # a light's bright core and darker rim are one light: join pieces 2 px apart
            acc = cls == 2
            lc, sz = ref_mod.label_components(_dilate(acc, 2) & mask)
            lc[~acc] = -1
        else:
            lc, sz = ref_mod.label_components((cls == c) & ~barrier & mask)
        sel = lc >= 0
        labels[sel] = lc[sel] + off
        off += int(sz.size)
    # Edge pixels join the nearest plate or frame region, never a light: a light framed by a ring of
    # edge pixels would otherwise take them all in and stop reading as a light.
    acc = cls == 2
    acc_labels = labels[acc]
    labels[acc] = -1
    labels = ref_mod._nearest_labels(labels, mask)
    labels[acc] = acc_labels
    labels[~mask] = -1
    n_obj = int(mask.sum())
    min_px = max(REGION_MIN_PX, int(REGION_MIN * n_obj))
    strong = edge & (mag >= np.quantile(mag[mask], EDGE_HIGH_Q)) if mask.any() else edge

    for _ in range(12):  # small regions -> best neighbour
        n = int(labels.max()) + 1
        if n <= 1:
            break
        sizes = np.bincount(labels[labels >= 0], minlength=n)
        rcls = _region_class(labels, cls, n)
        small = np.where(rcls == 2, sizes < ACCENT_MIN_PX, sizes < min_px)
        if not small.any():
            break
        pairs, counts = _adjacency(labels)
        if pairs.size == 0:
            break
        target = np.arange(n)
        best = np.full(n, -1.0)
        for (a, b), c in zip(pairs, counts):
            for s, o in ((a, b), (b, a)):
                if small[s]:
                    score = c * (2.0 if rcls[s] == rcls[o] else 1.0) + (0.5 if not small[o] else 0.0)
                    if score > best[s]:
                        best[s], target[s] = score, o
        # Break 2-cycles: the smaller of a mutual pair keeps pointing, the larger stays.
        mutual = target[target] == np.arange(n)
        keep_root = mutual & (sizes >= sizes[target]) & (target != np.arange(n))
        target[keep_root] = np.arange(n)[keep_root]
        labels = _relabel(labels, target)

    if SMOOTH_JOIN:  # a class boundary with no edge on it is shading: join across it
        lo_thr = max(EDGE_LOW_MIN, float(np.quantile(mag[ref_mod._erode(mask)], EDGE_LOW_Q))) if mask.any() else EDGE_LOW_MIN
        for _ in range(8):
            n = int(labels.max()) + 1
            pairs, counts = _adjacency(labels)
            if pairs.size == 0:
                break
            rcls = _region_class(labels, cls, n)
            bm = _boundary_mean(labels, mag, pairs)
            ok = (bm < SMOOTH_Q * lo_thr) & (rcls[pairs[:, 0]] != 2) & (rcls[pairs[:, 1]] != 2) & (counts >= 4)
            if not ok.any():
                break
            target = np.arange(n)
            for a, b in pairs[ok][np.argsort(bm[ok])]:
                ra, rb = _root(target, a), _root(target, b)
                if ra != rb:
                    target[max(ra, rb)] = min(ra, rb)
            labels = _relabel(labels, target)
        # the class of a joined region follows its majority
        n = int(labels.max()) + 1
        rc = _region_class(labels, cls, n)
        keep_acc = cls == 2
        cls = np.where(labels >= 0, rc[np.maximum(labels, 0)], -1).astype(np.int8)
        cls[keep_acc & (cls != 2)] = cls[keep_acc & (cls != 2)]
    for _ in range(6):  # join near-identical neighbours across weak boundaries
        n = int(labels.max()) + 1
        pairs, counts = _adjacency(labels)
        if pairs.size == 0:
            break
        mean = np.zeros((n, 3))
        cnt = np.bincount(labels[labels >= 0], minlength=n).astype(np.float64)
        for i in range(3):
            mean[:, i] = np.bincount(labels[labels >= 0], weights=lab[..., i][labels >= 0], minlength=n) / np.maximum(cnt, 1)
        rcls = _region_class(labels, cls, n)
        # boundary strength: share of the shared boundary pixels (on either side) that are strong edges
        bstrong = _boundary_strength(labels, strong, pairs)
        de = np.linalg.norm(mean[pairs[:, 0]] - mean[pairs[:, 1]], axis=1)
        ok = (de < MERGE_DE) & (bstrong < MERGE_EDGE) & (rcls[pairs[:, 0]] == rcls[pairs[:, 1]])
        if not ok.any():
            break
        target = np.arange(n)
        for a, b in pairs[ok][np.argsort(de[ok])]:
            ra, rb = _root(target, a), _root(target, b)
            if ra != rb:
                target[max(ra, rb)] = min(ra, rb)
        labels = _relabel(labels, target)
    facts = {"dark_threshold_L": round(t, 1), "regions": int(labels.max()) + 1}
    return labels.astype(np.int32), cls, lab, facts


def _root(t: np.ndarray, i: int) -> int:
    while t[i] != i:
        i = t[i]
    return int(i)


def _region_class(labels: np.ndarray, cls: np.ndarray, n: int) -> np.ndarray:
    """Majority class per region (accent wins when it covers >= 40% of the region)."""
    sel = labels >= 0
    counts = np.zeros((n, 3))
    for c in range(3):
        counts[:, c] = np.bincount(labels[sel], weights=(cls[sel] == c).astype(np.float64), minlength=n)
    out = counts[:, :2].argmax(axis=1)
    tot = counts.sum(axis=1)
    out[counts[:, 2] >= 0.4 * np.maximum(tot, 1)] = 2
    return out


def _boundary_strength(labels: np.ndarray, strong: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """For each touching pair: share of boundary pixel pairs where either pixel is a strong edge
    or within one pixel of one (regions are separated by the 1-px dilated edge barrier)."""
    near = _dilate(strong)
    out = np.zeros(len(pairs))
    if not len(pairs):
        return out
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(pairs)}
    tot = np.zeros(len(pairs))
    hit = np.zeros(len(pairs))
    for (la, lb, sa, sb) in ((labels[:, :-1], labels[:, 1:], near[:, :-1], near[:, 1:]),
                             (labels[:-1, :], labels[1:, :], near[:-1, :], near[1:, :])):
        sel = (la != lb) & (la >= 0) & (lb >= 0)
        a, b = la[sel], lb[sel]
        s = (sa[sel] | sb[sel]).astype(np.float64)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        n = int(labels.max()) + 1
        code = lo.astype(np.int64) * n + hi
        pcode = pairs[:, 0].astype(np.int64) * n + pairs[:, 1]
        idx = np.searchsorted(pcode, code)
        idx = np.clip(idx, 0, len(pcode) - 1)
        good = pcode[idx] == code
        np.add.at(tot, idx[good], 1.0)
        np.add.at(hit, idx[good], s[good])
    del key
    return hit / np.maximum(tot, 1.0)


def describe_regions(labels: np.ndarray, cls: np.ndarray, lab: np.ndarray, rgb: np.ndarray, edge: np.ndarray,
                     x_of, y_of, unit: str = "m") -> list[dict]:
    """Facts per region, largest first: class, colour, share, bbox, center, relief guess, shading, mirror pair."""
    n = int(labels.max()) + 1
    if n <= 0:
        return []
    sel = labels >= 0
    total = int(sel.sum())
    rcls = _region_class(labels, cls, n)
    pairs, counts = _adjacency(labels)
    near_edge = _dilate(edge)
    # boundary pixels of each region and what lies across
    border_dark = np.zeros(n)
    border_tot = np.zeros(n)
    for (la, lb) in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        s = (la != lb) & (la >= 0) & (lb >= 0)
        a, b = la[s], lb[s]
        for x, y in ((a, b), (b, a)):
            np.add.at(border_tot, x, 1.0)
            np.add.at(border_dark, x, (rcls[y] == 0).astype(np.float64))
    # strong edges on a region's own rim also count as a frame
    rim = sel & ~ref_mod._erode(sel)
    out = []
    ys, xs = np.nonzero(sel)
    lbl = labels[ys, xs]
    order = np.argsort(-np.bincount(lbl, minlength=n))
    for i in order:
        pick = lbl == i
        if not pick.any():
            continue
        yy, xx = ys[pick], xs[pick]
        area = int(pick.sum())
        m_lab = lab[yy, xx].mean(axis=0)
        m_rgb = rgb[yy, xx].mean(axis=0) / 255.0
        c = int(rcls[i])
        name = ("dark", "bright", "accent")[c]
        if c == 2:
            name = f"accent {hue_name(float(m_lab[1]), float(m_lab[2]))}"
        frac_dark = border_dark[i] / max(1.0, border_tot[i])
        mine = labels == i
        rim_i = mine & ~ref_mod._erode(mine)
        frac_edge = float((rim_i & near_edge).sum()) / max(1, int(rim_i.sum()))
        if c == 2:
            relief = "light or decal"
        elif c == 0:
            relief = "recessed (frame or gap)"
        elif max(frac_dark, frac_edge) >= RAISED_BORDER:
            relief = "raised plate"
        else:
            relief = "surface"
        shade = ""
        if area >= 30:
            A = np.stack([xx - xx.mean(), yy - yy.mean(), np.ones_like(xx, dtype=np.float64)], axis=1).astype(np.float64)
            coef, *_ = np.linalg.lstsq(A, lab[yy, xx, 0].astype(np.float64), rcond=None)
            span_x, span_y = float(xx.max() - xx.min() + 1), float(yy.max() - yy.min() + 1)
            dl_x, dl_y = coef[0] * span_x, coef[1] * span_y
            if max(abs(dl_x), abs(dl_y)) >= SHADE_DL:
                if abs(dl_y) >= abs(dl_x):
                    shade = f"lighter toward the {'bottom' if dl_y > 0 else 'top'} (dL {abs(dl_y):.0f})"
                else:
                    shade = f"lighter toward the {'right' if dl_x > 0 else 'left'} (dL {abs(dl_x):.0f})"
        x0, x1 = float(x_of(xx.min())), float(x_of(xx.max() + 1))
        y0, y1 = float(y_of(yy.max() + 1)), float(y_of(yy.min()))
        out.append({
            "label": int(i), "class": name, "color": ref_mod._hex(m_rgb), "share": area / max(1, total),
            "px": area, "x": [round(min(x0, x1), 3), round(max(x0, x1), 3)], "y": [round(min(y0, y1), 3), round(max(y0, y1), 3)],
            "center": [round(float(x_of(xx.mean() + 0.5)), 3), round(float(y_of(yy.mean() + 0.5)), 3)],
            "relief": relief, "shading": shade, "border_dark": round(frac_dark, 2), "border_edge": round(frac_edge, 2),
        })
    for k, r in enumerate(out):
        r["id"] = f"R{k + 1}"
    # mirror pairs (x about 0)
    width = max((r["x"][1] for r in out), default=1.0) - min((r["x"][0] for r in out), default=0.0)
    tol = 0.04 * max(width, 1e-9)
    for r in out:
        r["pair"] = None
    for r in out:
        if r["pair"] or abs(r["center"][0]) < tol:
            continue
        best = None
        for q in out:
            if q is r or q["pair"] or q["class"] != r["class"]:
                continue
            d = math.hypot(q["center"][0] + r["center"][0], q["center"][1] - r["center"][1])
            ratio = max(q["px"], r["px"]) / max(1, min(q["px"], r["px"]))
            if d < tol and ratio < 2.0 and (best is None or d < best[0]):
                best = (d, q)
        if best:
            r["pair"], best[1]["pair"] = best[1]["id"], r["id"]
    return out


# ----------------------------------------------------------------------------- picture only


def analyze_structure(image_path, view: str = "front", height: float | None = None, mask=None, *, root=None) -> dict:
    """Structure of a picture alone: edges, regions with a class and relief guess (SPEC 12.4)."""
    r = ref_mod._analyze(image_path, view, height, mask, None, root)
    edge, mag, efacts = edges(r.rgb, r.mask)
    labels, cls, lab, rfacts = regions(r.rgb, r.mask, edge, mag)
    regs = describe_regions(labels, cls, lab, r.rgb, edge, r.frame.x, r.frame.y, r.frame.unit)
    rep = _summary(regs, edge, r.mask)
    rep.update(image=r.label, view=view, unit=r.frame.unit, scale=r.report["scale"], edges=efacts, region_facts=rfacts,
               regions=regs)
    rep["_arrays"] = {"rgb": r.rgb, "mask": r.mask, "edge": edge, "labels": labels, "cls": cls}
    return rep


def _summary(regs: list[dict], edge: np.ndarray, mask: np.ndarray) -> dict:
    by = {}
    for r in regs:
        by[r["class"]] = by.get(r["class"], 0.0) + r["share"]
    big = [r for r in regs if r["share"] >= REPORT_MIN]
    return {"classes": {k: round(v, 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
            "count": len(regs), "listed": min(len(big), REPORT_MAX),
            "pairs": sum(1 for r in big if r["pair"]) // 2,
            "accents": sum(1 for r in regs if r["class"].startswith("accent"))}


def _fmt_range(v, unit):
    if unit == "m":
        return f"{v[0]:+.2f}..{v[1]:+.2f}"
    return f"{v[0]:+.0f}..{v[1]:+.0f}"


def format_structure(rep: dict) -> str:
    unit = rep.get("unit", "m")
    u = " m" if unit == "m" else " %"
    lines = [f"STRUCTURE  {rep['image']}  view {rep['view']}" + (f"  1 px = {rep['scale']['mpp']:.4f} m" if rep["scale"].get("mpp") else "")]
    e = rep["edges"]
    lines.append(f"EDGES      {e['share']:.1%} of the object's pixels lie on edges (thresholds {e['low']:.1f}/{e['high']:.1f} Lab per px)")
    cl = ", ".join(f"{k} {v:.0%}" for k, v in rep["classes"].items())
    lines.append(f"REGIONS    {rep['count']} ({rep['listed']} listed, >= {REPORT_MIN:.1%} of the object); {rep['pairs']} left/right pairs; {cl}")
    lines.append(f"  {'id':<4} {'class':<13} {'colour':<8} {'share':>6}  {'size' + u:<14} {'x' + u:<14} {'y' + u:<14} {'relief':<24} pair")
    shown = [r for r in rep["regions"] if r["share"] >= REPORT_MIN][:REPORT_MAX]
    for r in shown:
        w, h = r["x"][1] - r["x"][0], r["y"][1] - r["y"][0]
        size = f"{w:.2f} x {h:.2f}" if unit == "m" else f"{w:.0f} x {h:.0f}"
        lines.append(f"  {r['id']:<4} {r['class']:<13} {r['color']:<8} {r['share']:>6.1%}  {size:<14} {_fmt_range(r['x'], unit):<14} "
                     f"{_fmt_range(r['y'], unit):<14} {r['relief']:<24} {r['pair'] or ''}"
                     + (f"  {r['shading']}" if r["shading"] else ""))
    acc = [r for r in rep["regions"] if r["class"].startswith("accent")]
    if acc:
        lines.append(f"ACCENTS    {len(acc)} small saturated regions (lights, decals, gems), largest first:")
        for r in acc[:REPORT_MAX]:
            lines.append(f"  {r['id']:<4} {r['class']:<13} {r['color']:<8} at x {_fmt_range(r['x'], unit)}, y {_fmt_range(r['y'], unit)}"
                         + (f"  pair {r['pair']}" if r["pair"] else ""))
    lines.append("RELIEF is a guess from colour and edges: bright regions framed by dark seams or strong edges read as raised plates,")
    lines.append("       dark regions as recessed frame or gaps, saturated ones as lights or decals. It is not measured depth.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- PNG output


def _outline(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    return b & (labels >= 0)


def _panel_regions(rgb, mask, labels, regs, scale=1) -> Image.Image:
    img = np.full(rgb.shape, 255.0, dtype=np.float32)
    img[mask] = rgb[mask] * 0.55 + 255 * 0.45  # faded picture
    ob = _outline(labels)
    img[ob] = (20, 20, 20)
    pil = Image.fromarray(img.astype(np.uint8))
    if scale != 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(pil)
    font = ref_mod._font(11)
    for r in [r for r in regs if r["share"] >= REPORT_MIN][:REPORT_MAX]:
        ys, xs = np.nonzero(labels == r["label"])
        if not len(ys):
            continue
        j = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)  # a pixel of the region near its center
        x, y = int(xs[j]) * scale, int(ys[j]) * scale
        col = {"dark": (40, 40, 200), "bright": (200, 30, 30)}.get(r["class"], (0, 140, 0))
        d.text((x - 6, y - 6), r["id"], fill=col, font=font)
    return pil


def write_structure_png(rep: dict, path) -> pathlib.Path:
    a = rep["_arrays"]
    rgb, mask, labels, edge, cls = a["rgb"], a["mask"], a["labels"], a["edge"], a["cls"]
    ys, xs = np.nonzero(mask)
    pad = 8
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    crop = (slice(y0, y1), slice(x0, x1))
    regs = [dict(r) for r in rep["regions"]]
    p1 = _panel_regions(rgb[crop], mask[crop], labels[crop], regs)
    cm = np.full(rgb[crop].shape, 255, dtype=np.uint8)
    cc = cls[crop]
    cm[cc == 0] = (45, 45, 55)
    cm[cc == 1] = (200, 200, 205)
    cm[cc == 2] = (40, 150, 255)
    cm[edge[crop]] = (230, 40, 40)
    p2 = Image.fromarray(cm)
    W = p1.width + p2.width + 10
    H = max(p1.height, p2.height) + 22
    out = Image.new("RGB", (W, H), (255, 255, 255))
    out.paste(p1, (0, 22))
    out.paste(p2, (p1.width + 10, 22))
    d = ImageDraw.Draw(out)
    d.text((2, 4), f"structure {rep['image']}: regions (ids as in the report)", fill=(0, 0, 0), font=ref_mod._font(12))
    d.text((p1.width + 12, 4), "classes: dark / bright / accent, edges red", fill=(0, 0, 0), font=ref_mod._font(12))
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


# ----------------------------------------------------------------------------- model vs picture

CLASS_NAMES = ("dark", "bright", "accent")
VERDICT_COVER = 0.5  # a region less covered than this by the model is a silhouette matter (see `match`)
VERDICT_CLASS = 0.6  # share of a region where the model's class must differ to call it a class mismatch
SPLIT_SHARE = 0.6  # a picture region belongs to the part covering at least this much of it


def _grid_coords(asset, view: str, res: int, model_grid: np.ndarray):
    """Grid pixel -> asset meters along the view's horizontal and vertical axes (as in `match`)."""
    from .camera import make_cameras
    from .march import march

    scene = asset.scene()
    lo, hi = scene.bounds(include_helpers=False)
    cam = make_cameras([view], lo, hi, res=res)[view]
    vb = march(scene, cam)
    my, mx = np.nonzero(vb.hit)
    gy, gx = np.nonzero(model_grid)
    pl, pt = int(gx.min()) - int(mx.min()), int(gy.min()) - int(my.min())
    H, W = vb.hit.shape
    mpp = cam.mpp
    c_h = float(np.dot(cam.center, cam.right))
    c_v = float(np.dot(cam.center, cam.up))
    hax, vax = ref_mod._axes(cam)

    def h_of(px):
        return hax[1] * (c_h + (np.asarray(px, dtype=np.float64) - pl - W / 2.0) * mpp) if hax[0] != "h" else c_h + (np.asarray(px, dtype=np.float64) - pl - W / 2.0) * mpp

    def v_of(py):
        return c_v + (H / 2.0 - (np.asarray(py, dtype=np.float64) - pt)) * mpp

    names = (hax[0] if hax[1] > 0 else f"-{hax[0]}", vax[0])
    _grid_coords.last = {"vb": vb, "pl": pl, "pt": pt, "camera": cam}
    return h_of, v_of, mpp, names


def compare_structure(asset, index: int = 0, res: int = 512, view: str | None = None) -> dict:
    """How the model's parts and materials line up with the divisions inside its reference picture."""
    if isinstance(res, bool) or not isinstance(res, int) or not 16 <= res <= 4096:
        raise SightlineError("res must be an integer from 16 to 4096", hint="e.g. --res 512")
    spec = ref_mod._reference_spec(asset, index)
    view = view or spec.view
    m = ref_mod._match(asset, index, res, view)
    pic = np.clip(m.ref_rgb, 0.0, 1.0).astype(np.float32) * 255.0
    R_, M_ = m.ref, m.model
    edge, mag, efacts = edges(pic, R_)
    evened = even_lighting(pic, R_)
    labels, cls, lab, rfacts = regions(pic, R_, edge, mag, class_rgb=evened)
    rfacts["lighting_evened"] = evened is not None
    h_of, v_of, mpp, axes = _grid_coords(asset, view, res, M_)
    regs = describe_regions(labels, cls, lab, pic, edge, h_of, v_of)
    # model classes from its flat-lit render, split by its own Otsu (both images are lit)
    # the model's classes come from its materials' own colours: the picture's lighting is not the style's
    part = m.part
    parts_all = asset.scene().parts
    base = np.zeros(m.lambert.shape, dtype=np.float32)
    for pi in np.unique(part[part >= 0]):
        mat = parts_all[int(pi)].material
        base[part == pi] = asset.style.material(mat).color if mat else (0.5, 0.5, 0.5)
    mlab = ref_mod.srgb_to_lab(np.clip(base, 0, 1)).astype(np.float32)
    mcls, _ = _classes(mlab, M_, fixed=MATERIAL_DARK_FIXED)
    parts = asset.scene().parts
    both = R_ & M_
    agree = (cls[both] == mcls[both]).mean() if both.any() else 0.0
    # Materials of areas, apart from where the borders are (DIVISIONS measures those): leave out the pixels
    # within CLASS_BAND px of a class border in either image.
    def _class_border(c):
        b = np.zeros(c.shape, dtype=bool)
        b[:, 1:] |= c[:, 1:] != c[:, :-1]
        b[:, :-1] |= c[:, 1:] != c[:, :-1]
        b[1:, :] |= c[1:, :] != c[:-1, :]
        b[:-1, :] |= c[1:, :] != c[:-1, :]
        return _dilate(b, CLASS_BAND - 1) if CLASS_BAND > 1 else b
    core = both & ~_class_border(np.where(R_, cls, -1)) & ~_class_border(np.where(M_, mcls, -1))
    agree_core = (cls[core] == mcls[core]).mean() if core.any() else 0.0
    # edges vs model part boundaries (inside both silhouettes; outlines are `match`'s job)
    pb = np.zeros_like(M_)
    pb[:, 1:] |= (part[:, 1:] != part[:, :-1])
    pb[1:, :] |= (part[1:, :] != part[:-1, :])
    inner = ref_mod._erode(ref_mod._erode(R_)) & ref_mod._erode(ref_mod._erode(M_))
    pb &= inner
    pe = edge & inner
    near_pb, near_pe = _dilate(pb, NEAR_PX), _dilate(pe, NEAR_PX)
    edge_hit = float((pe & near_pb).sum()) / max(1, int(pe.sum()))
    bound_hit = float((pb & near_pe).sum()) / max(1, int(pb.sum()))
    # The picture's divisions are the borders between its regions; edges inside a region (grooves, engraving,
    # highlights) are surface detail and are reported apart.
    rb = np.zeros_like(R_)
    rb[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    rb[1:, :] |= labels[1:, :] != labels[:-1, :]
    rb &= inner
    border_hit = float((rb & near_pb).sum()) / max(1, int(rb.sum()))
    bound_on_border = float((pb & (_dilate(rb, NEAR_PX) | near_pe)).sum()) / max(1, int(pb.sum()))

    def part_label(i):
        return f"{parts[i].name} ({parts[i].material})"

    findings = []
    region_part = {}
    for r in regs:
        sel = labels == r["label"]
        n = int(sel.sum())
        cover = float((sel & M_).sum()) / max(1, n)
        ids, cnt = np.unique(part[sel & M_], return_counts=True)
        order = np.argsort(-cnt)
        top = [(int(ids[k]), float(cnt[k]) / max(1, n)) for k in order[:3] if ids[k] >= 0]
        r["cover"] = round(cover, 2)
        r["parts"] = [{"part": parts[i].name, "material": parts[i].material, "share": round(s, 2)} for i, s in top]
        pc = cls[sel & M_]
        mc = mcls[sel & M_]
        r_cls = 2 if r["class"].startswith("accent") else ("dark", "bright").index(r["class"])
        model_cls_share = np.bincount(mc[mc >= 0], minlength=3) / max(1, int((mc >= 0).sum()))
        r["model_class"] = CLASS_NAMES[int(np.argmax(model_cls_share))] if mc.size else None
        listed = r["share"] >= REPORT_MIN or r_cls == 2
        if top and top[0][1] >= SPLIT_SHARE * cover and r_cls != 2:
            region_part.setdefault(top[0][0], []).append(r)
        if not listed:
            r["verdict"] = None
            continue
        if cover < VERDICT_COVER:
            r["verdict"] = "outside the model's silhouette here (see match)"
            continue
        main = part_label(top[0][0]) if top else "nothing"
        if r_cls == 2:
            if model_cls_share[2] >= 0.3:
                r["verdict"] = f"ok: {main}"
            else:
                r["verdict"] = f"light or decal missing: the model shows {main} here; add a small {r['class'].split()[-1]} part"
                findings.append(("light", r))
            continue
        if 1.0 - model_cls_share[r_cls] >= VERDICT_CLASS:
            want = "a dark part (frame, gap, joint)" if r_cls == 0 else "a bright part (plate, panel)"
            r["verdict"] = f"picture is {r['class']} here, the model {r['model_class']}: {main}; use {want} here"
            findings.append(("class", r))
        else:
            r["verdict"] = f"ok: {main}"
    # parts that the picture divides into several elements
    splits = []
    for pi, rs in region_part.items():
        big = [r for r in rs if r["share"] >= REPORT_MIN]
        if len(big) >= 2:
            splits.append({"part": parts[pi].name, "material": parts[pi].material,
                           "regions": [r["id"] for r in big], "classes": sorted({r["class"] for r in big})})
    splits.sort(key=lambda s: -len(s["regions"]))
    rep = {
        "asset": asset.name, "index": index, "view": view, "res": res, "grid_mpp": round(mpp, 5), "axes": list(axes),
        "image": m.report["reference"]["image"],
        "iou": m.report.get("iou"),
        "class_agreement": round(float(agree), 3),
        "class_agreement_areas": round(float(agree_core), 3),
        "edges_on_part_boundaries": round(edge_hit, 3),
        "part_boundaries_on_edges": round(bound_hit, 3),
        "region_borders_on_part_boundaries": round(border_hit, 3),
        "part_boundaries_on_borders": round(bound_on_border, 3),
        "edges": efacts, "region_facts": rfacts, "regions": regs, "splits": splits,
        "summary": _summary(regs, edge, R_),
    }
    rep["_arrays"] = {"pic": pic, "ref": R_, "model": M_, "labels": labels, "cls": cls, "mcls": mcls, "edge": edge,
                      "pb": pb, "lambert": m.lambert, "part": part}
    rep["_trace"] = {"h_of": h_of, "v_of": v_of, **_grid_coords.last}
    return rep


def format_compare(rep: dict) -> str:
    h, v = rep["axes"]
    lines = [f"STRUCTURE  {rep['asset']}  reference[{rep['index']}] {rep['image']}  view {rep['view']}  grid 1 px = {rep['grid_mpp']:.4f} m"]
    lines.append(f"CLASSES    dark/bright/accent agree on {rep['class_agreement_areas']:.0%} of the areas "
                 f"({rep['class_agreement']:.0%} of every shared pixel, {CLASS_BAND} px class borders included)")
    lines.append(f"DIVISIONS  {rep['region_borders_on_part_boundaries']:.0%} of the picture's region borders have a model part boundary "
                 f"within {NEAR_PX} px; {rep['part_boundaries_on_borders']:.0%} of the model's part boundaries lie on a picture "
                 f"border or edge")
    lines.append(f"DETAIL     {rep['edges_on_part_boundaries']:.0%} of all the picture's inner edges (grooves and lines inside "
                 f"elements too) have a model part boundary within {NEAR_PX} px; {rep['part_boundaries_on_edges']:.0%} of the "
                 f"part boundaries lie on one")
    s = rep["summary"]
    cl = ", ".join(f"{k} {v_:.0%}" for k, v_ in s["classes"].items())
    lines.append(f"REGIONS    {s['count']} in the picture ({cl}); {s['pairs']} left/right pairs; {s['accents']} lights or decals")
    probs = [r for r in rep["regions"] if r.get("verdict") and not r["verdict"].startswith("ok")]
    oks = [r for r in rep["regions"] if r.get("verdict", "") and r["verdict"].startswith("ok")]
    lines.append(f"FINDINGS   {len(probs)} regions disagree with the model, {len(oks)} agree (largest first; {h} and {v} in m)")
    for r in probs[:REPORT_MAX]:
        lines.append(f"  {r['id']:<4} {r['class']:<12} {r['x'][1] - r['x'][0]:.2f} x {r['y'][1] - r['y'][0]:.2f} m at {h} {r['x'][0]:+.2f}..{r['x'][1]:+.2f}, "
                     f"{v} {r['y'][0]:.2f}..{r['y'][1]:.2f}" + (f" (pair {r['pair']})" if r["pair"] else "") + f": {r['verdict']}")
    if rep["splits"]:
        lines.append("SPLITS     model parts that the picture divides into several elements (split them, or add plates on them):")
        for sp in rep["splits"][:REPORT_MAX]:
            lines.append(f"  {sp['part']} ({sp['material']}): {len(sp['regions'])} picture regions {', '.join(sp['regions'][:8])}"
                         + (" ..." if len(sp["regions"]) > 8 else "") + f" ({', '.join(sp['classes'])})")
    lines.append(f"AGREE      {', '.join(r['id'] + ' ' + r['verdict'][4:].split(' (')[0] for r in oks[:12])}" + (" ..." if len(oks) > 12 else ""))
    lines.append("Regions come from the picture alone (edges, colour classes); `sightline structure <image>` lists them with sizes.")
    return "\n".join(lines)


def write_compare_png(rep: dict, path) -> pathlib.Path:
    a = rep["_arrays"]
    pic, R_, M_, labels, cls, mcls, edge, pb, lam = (a[k] for k in ("pic", "ref", "model", "labels", "cls", "mcls", "edge", "pb", "lambert"))
    ys, xs = np.nonzero(R_ | M_)
    pad = 6
    crop = (slice(max(0, ys.min() - pad), ys.max() + pad + 1), slice(max(0, xs.min() - pad), xs.max() + pad + 1))
    scale = max(1, int(round(700 / max(1, crop[0].stop - crop[0].start))))
    p1 = _panel_regions(pic[crop], R_[crop], labels[crop], [r for r in rep["regions"] if r.get("verdict") and not r["verdict"].startswith("ok")], scale)
    model = (np.clip(lam[crop], 0, 1) * 255).astype(np.uint8).copy()
    model[pb[crop]] = (20, 20, 20)
    p2 = Image.fromarray(model).resize((p1.width, p1.height), Image.NEAREST)
    agree = np.full(pic[crop].shape, 255, dtype=np.uint8)
    bc = R_[crop] & M_[crop]
    same = bc & (cls[crop] == mcls[crop])
    agree[bc] = (230, 120, 120)
    agree[same] = (150, 210, 150)
    agree[R_[crop] & ~M_[crop]] = (120, 150, 230)
    agree[M_[crop] & ~R_[crop]] = (250, 200, 120)
    e = edge[crop]
    b = pb[crop]
    agree[e & ~_dilate(b, NEAR_PX)] = (200, 0, 0)
    agree[b & ~_dilate(e, NEAR_PX)] = (0, 0, 200)
    agree[e & _dilate(b, NEAR_PX)] = (0, 120, 0)
    p3 = Image.fromarray(agree).resize((p1.width, p1.height), Image.NEAREST)
    W = p1.width * 3 + 20
    out = Image.new("RGB", (W, p1.height + 24), (255, 255, 255))
    for k, pnl in enumerate((p1, p2, p3)):
        out.paste(pnl, (k * (p1.width + 10), 24))
    d = ImageDraw.Draw(out)
    f = ref_mod._font(12)
    d.text((2, 5), "picture regions that disagree (ids as in the report)", fill=(0, 0, 0), font=f)
    d.text((p1.width + 12, 5), "model (flat light) with its part boundaries", fill=(0, 0, 0), font=f)
    d.text((2 * p1.width + 22, 5), "classes: green agree, red differ; edges: dark green shared, red picture only, blue model only", fill=(0, 0, 0), font=f)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path



# ----------------------------------------------------------------------------- tracing regions into parts

TRACE_MAX_POINTS = 14
TRACE_EPS_PX = 2.5  # Douglas-Peucker tolerance to start from (grows until the outline fits TRACE_MAX_POINTS)
TRACE_SMOOTH = 2  # closing then opening by this many px before tracing: pixel noise out of the outline
TRACE_MIN_SMOOTH_PX = 60  # smaller regions (lights, decals) are traced without smoothing
TRACE_SINK = 0.01  # a traced plate sinks this far into the surface it sits on (it must touch it)
TRACE_FLUSH = 0.004  # a traced dark (recessed) region stands only this far out of the surface


def outline(mask: np.ndarray) -> np.ndarray:
    """Outer contour of the largest 4-connected piece of `mask` as (K, 2) pixel (col, row) points, clockwise
    on screen (Moore-neighbour tracing)."""
    labels, sizes = ref_mod.label_components(mask)
    if not sizes.size:
        return np.zeros((0, 2))
    m = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    m[1:-1, 1:-1] = labels == int(np.argmax(sizes))
    ys, xs = np.nonzero(m)
    k = np.lexsort((xs, ys))[0]
    start = (int(ys[k]), int(xs[k]))
    nb = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]  # clockwise from west
    pts = [start]
    cur, back = start, 0  # the west neighbour of the start is background
    for _ in range(8 * int(m.sum()) + 16):
        nxt = None
        for i in range(1, 9):
            d = (back + i) % 8
            y, x = cur[0] + nb[d][0], cur[1] + nb[d][1]
            if m[y, x]:
                nxt, prev_d = (y, x), (back + i - 1) % 8
                break
        if nxt is None:  # a single pixel
            break
        by, bx = cur[0] + nb[prev_d][0], cur[1] + nb[prev_d][1]
        back = nb.index((by - nxt[0], bx - nxt[1]))
        cur = nxt
        if cur == start:
            break
        pts.append(cur)
    a = np.array(pts, dtype=np.float64) - 1.0
    return a[:, ::-1]  # (col, row)


def smooth_mask(mask: np.ndarray, n: int = TRACE_SMOOTH) -> np.ndarray:
    """Morphological closing then opening (n px): fills notches and drops one-pixel spurs."""
    m = mask.copy()
    for _ in range(n):
        m = ref_mod._dilate(m)
    for _ in range(n):
        m = ref_mod._erode(m)
    o = m.copy()
    for _ in range(n):
        o = ref_mod._erode(o)
    for _ in range(n):
        o = ref_mod._dilate(o)
    return o if o.any() else m


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    n = np.hypot(*ab)
    rel = points - a
    d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n if n > 0 else np.hypot(rel[:, 0], rel[:, 1])
    i = int(np.argmax(d))
    if d[i] <= eps:
        return np.array([a, b])
    left = _rdp(points[:i + 1], eps)
    right = _rdp(points[i:], eps)
    return np.vstack([left[:-1], right])


def simplify(contour: np.ndarray, eps: float = TRACE_EPS_PX, max_points: int = TRACE_MAX_POINTS) -> np.ndarray:
    """Closed-contour Douglas-Peucker: split at the point farthest from the first, simplify both halves;
    the tolerance grows until the polygon has at most `max_points` corners."""
    if len(contour) <= 3:
        return contour
    far = int(np.argmax(np.hypot(*(contour - contour[0]).T)))
    while True:
        a = _rdp(contour[:far + 1], eps)
        b = _rdp(np.vstack([contour[far:], contour[:1]]), eps)
        poly = np.vstack([a[:-1], b[:-1]])
        if len(poly) <= max_points or eps > 50:
            return poly
        eps *= 1.4


def _class_materials(asset, rep) -> dict[str, str]:
    """Material to use for a traced region of each class: the model's material covering most pixels of that
    class in the comparison, else a sensible default from the style."""
    a = rep["_arrays"]
    part, mcls, M_ = a["part"], a["mcls"], a["model"]
    parts = asset.scene().parts
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        sel = M_ & (mcls == c) & (part >= 0)
        if sel.any():
            ids, cnt = np.unique(part[sel], return_counts=True)
            by_mat: dict[str, int] = {}
            for i, n in zip(ids, cnt):
                mat = parts[int(i)].material
                if mat:
                    by_mat[mat] = by_mat.get(mat, 0) + int(n)
            if by_mat:
                out[name] = max(by_mat, key=by_mat.get)
    mats = asset.style.materials
    out.setdefault("bright", "metal" if "metal" in mats else "paint_white")
    out.setdefault("dark", "iron" if "iron" in mats else "paint_black")
    out.setdefault("accent", next((m for m in ("glow_blue", "lamp", "gold") if m in mats), out["bright"]))
    return out


def _view_frame(cam) -> tuple[np.ndarray, float]:
    """Direction toward the viewer and the yaw (deg) that turns an extrude's local frame onto the view:
    local x -> camera right, local y -> up, local z -> toward the viewer. Level views only."""
    n = -np.asarray(cam.forward, dtype=np.float64)
    if abs(float(cam.up[1])) < 0.999 or abs(float(n[1])) > 1e-6:
        raise SightlineError(f"tracing needs a level view (front, back, left, right), this one is '{cam.view}'",
                             hint="use a reference with view: front, back, left or right")
    yaw = math.degrees(math.atan2(n[0], n[2]))
    if yaw <= -179.999:
        yaw = 180.0
    a = math.radians(yaw)
    if not np.allclose([math.cos(a), 0.0, -math.sin(a)], cam.right, atol=1e-6):
        raise SightlineError(f"unexpected camera frame for view '{cam.view}'")
    return n, yaw


def _side_band(rows, s_front, s_back, v_of, mpp, eps_px: float = 1.0, max_points: int = 30) -> np.ndarray:
    """Side outline [(s, v)] of a plate that follows the surface: the front edge down the rows, the back
    edge up again, each simplified on its own (Douglas-Peucker on an open polyline)."""
    v_top, v_bot = float(v_of(rows[0])), float(v_of(rows[-1] + 1.0))
    v = v_of(rows + 0.5)
    front = np.stack([np.r_[s_front[0], s_front, s_front[-1]], np.r_[v_top, v, v_bot]], axis=1)
    back = np.stack([np.r_[s_back[0], s_back, s_back[-1]], np.r_[v_top, v, v_bot]], axis=1)

    def dp(line, eps):
        if len(line) <= 2:
            return line
        a, b = line[0], line[-1]
        ab = b - a
        L = float(np.hypot(*ab)) or 1e-12
        d = np.abs(ab[0] * (line[:, 1] - a[1]) - ab[1] * (line[:, 0] - a[0])) / L
        i = int(np.argmax(d))
        if d[i] <= eps:
            return np.array([a, b])
        return np.vstack([dp(line[: i + 1], eps)[:-1], dp(line[i:], eps)])

    eps = eps_px * mpp
    while True:
        f, b = dp(front, eps), dp(back, eps)
        if len(f) + len(b) <= 2 * max_points:
            break
        eps *= 1.5
    return np.vstack([f, b[::-1]])


def trace_regions(asset, ids, index: int = 0, res: int = 512, depth: float = 0.03, prefix: str = "plate",
                  max_relief: float | None = None, conform: bool = False) -> dict:
    """Extrude parts traced from a reference picture's regions (front, back, left or right view): the
    outline in meters, placed on the model's surface at that spot (turned to face the camera), material by
    the region's class. `conform`: a side outline makes the plate follow the surface from top to bottom
    instead of standing flat. Side views give one part on the camera's side with `mirror: x`.
    Returns {yaml, parts, report, skipped}."""
    if not (isinstance(depth, (int, float)) and 0.002 <= depth <= 1.0):
        raise SightlineError(f"depth must be 0.002..1 m, got {depth!r}")
    rep = compare_structure(asset, index, res)
    tr = rep["_trace"]
    cam = tr["camera"]
    n, yaw = _view_frame(cam)
    right, up = np.asarray(cam.right, dtype=np.float64), np.asarray(cam.up, dtype=np.float64)
    side_view = abs(n[0]) > 0.5
    regs = {r["id"]: r for r in rep["regions"]}
    want = []
    for key in ids:
        if key == "disagree":
            want += [r["id"] for r in rep["regions"] if r.get("verdict") and not r["verdict"].startswith("ok")
                     and "outside" not in r["verdict"]]
        elif key in regs:
            want.append(key)
        else:
            raise SightlineError(f"no region {key} in the picture", hint=f"regions are R1..R{len(regs)}; run sightline structure first")
    a = rep["_arrays"]
    labels, vb, pl, pt = a["labels"], tr["vb"], tr["pl"], tr["pt"]
    h_of, v_of = tr["h_of"], tr["v_of"]
    mpp = float(cam.mpp)
    mats = _class_materials(asset, rep)
    s_map = vb.point.astype(np.float64) @ n  # height of the surface toward the viewer
    front_s = float(np.median(s_map[vb.hit])) if vb.hit.any() else 0.0
    view_tag = "" if rep["view"] == "front" else f"{rep['view']}_"
    from .model import _self_intersects
    done, parts_out, lines, skipped = set(), [], [], []
    for rid in want:
        if rid in done:
            continue
        r = regs[rid]
        pair = None if side_view else r.get("pair")
        mirror = side_view or bool(pair and pair in want)
        if pair and mirror and right[0] * r["center"][0] > 0:  # write the -x one with mirror: x
            r, rid = regs[pair], pair
        done.add(rid)
        if pair and mirror:
            done.add(r["pair"])
        raw = labels == r["label"]
        mask = smooth_mask(raw) if raw.sum() >= TRACE_MIN_SMOOTH_PX else raw  # small lights keep their pixels
        poly_px = simplify(outline(mask))
        if len(poly_px) < 3:
            poly_px = simplify(outline(mask), eps=0.5)
        if len(poly_px) < 3 and mask.any():  # a light a few pixels across: its pixel box
            rr, cc = np.nonzero(mask)
            poly_px = np.array([[cc.min(), rr.min()], [cc.max() + 1, rr.min()], [cc.max() + 1, rr.max() + 1],
                                [cc.min(), rr.max() + 1]], dtype=np.float64) - 0.5
        if len(poly_px) < 3:
            continue
        pts = np.stack([h_of(poly_px[:, 0] + 0.5), v_of(poly_px[:, 1] + 0.5)], axis=1)
        if _self_intersects(pts):  # fall back to the convex hull
            pts = _hull(pts)
        rows, cols = np.nonzero(mask)
        vr, vc = rows - pt, cols - pl
        ok = (vr >= 0) & (vr < vb.hit.shape[0]) & (vc >= 0) & (vc < vb.hit.shape[1])
        rows, vr, vc = rows[ok], vr[ok], vc[ok]
        hit = vb.hit[vr, vc]
        cls = "accent" if r["class"].startswith("accent") else r["class"]
        q = 0.25 if cls == "dark" else 0.75  # a recessed region sits on the deeper surface it shows
        if max_relief is not None and not conform and hit.sum() >= 4:
            ss = s_map[vr[hit], vc[hit]]
            relief = float(np.quantile(ss, 0.95) - np.quantile(ss, 0.05))
            if relief > max_relief:  # a flat plate would not lie on this curved or sloped surface
                skipped.append((rid, round(relief, 3)))
                continue
        sf = float(np.quantile(s_map[vr[hit], vc[hit]], q)) if hit.any() else front_s

        def lift(s):  # plate front from the surface: dark ones only TRACE_FLUSH proud, so lights stay in front
            return s + TRACE_FLUSH if cls == "dark" else s + depth - min(TRACE_SINK, depth / 2.0)

        side = None
        if conform and hit.sum() >= 4:
            urows = np.unique(rows)
            per, wts = np.full(len(urows), np.nan), np.zeros(len(urows))
            for k, row in enumerate(urows):
                sel = (rows == row) & hit
                if sel.any():
                    per[k], wts[k] = np.quantile(s_map[vr[sel], vc[sel]], q), sel.sum()
            good = ~np.isnan(per)
            # a gently bent plate, not a staircase over the parts below it: a weighted fit of the
            # surface's height down the region (quadratic, or a line / flat for short regions)
            vv = v_of(urows + 0.5) - float(v_of(urows.mean() + 0.5))
            deg = 2 if good.sum() >= 8 else (1 if good.sum() >= 3 else 0)
            coef = np.polyfit(vv[good], per[good], deg, w=np.sqrt(wts[good]))
            per = np.clip(np.polyval(coef, vv), per[good].min() - depth, per[good].max() + depth)
            band = _side_band(urows.astype(np.float64), lift(per), lift(per) - depth, v_of, mpp)
            if not _self_intersects(band):
                side = band
        lo_v, hi_v = pts[:, 1].min(), pts[:, 1].max()
        if side is not None:
            lo_v, hi_v = max(lo_v, side[:, 1].min()), min(hi_v, side[:, 1].max())
            cs = (side[:, 0].min() + side[:, 0].max()) / 2.0
        else:
            cs = lift(sf) - depth / 2.0
        ch = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        cv = (lo_v + hi_v) / 2.0
        center = right * ch + up * cv + n * cs
        if side_view and center[0] < 0:
            suffix = "_l"
        elif side_view or mirror:
            suffix = "_r" if center[0] > 0 else "_l"
        else:
            suffix = ""
        name = f"{prefix}_{view_tag}{rid.lower()}{suffix}"
        rel = pts - [ch, cv]
        pts_txt = ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in rel)
        lines += [f"  - name: {name}                 # traced from {rep['view']} picture region {rid}"
                  + (f" (pair {r['pair']})" if pair and mirror else "") + f": {r['class']}, {r['relief']}",
                  "    shape: extrude", f"    points: [{pts_txt}]"]
        if side is not None:
            side_rel = side - [cs, cv]
            lines.append("    side: [" + ", ".join(f"[{z:.3f}, {y:.3f}]" for z, y in side_rel) + "]")
        else:
            lines.append(f"    depth: {depth:g}")
        lines += [f"    round: {min(0.004, depth / 4):g}"]
        if abs(yaw) > 1e-6:
            lines.append(f"    rotate: [0, {yaw:g}, 0]")
        lines += [f"    material: {mats[cls]}", f"    at: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"]
        if mirror:
            lines.append("    mirror: x")
        parts_out.append({"name": name, "region": rid, "points": rel.tolist(),
                          "side": None if side is None else (side - [cs, cv]).tolist(), "rotate": [0, yaw, 0],
                          "at": center.tolist(), "material": mats[cls], "mirror": mirror})
    yaml = "\n".join(lines) + ("\n" if lines else "")
    return {"yaml": yaml, "parts": parts_out, "report": rep, "skipped": skipped}


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), counter-clockwise."""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return np.array(p)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])
