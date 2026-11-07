"""Reference images: analysis, model-vs-reference matching and view fitting (SPEC section 12).

The designer usually starts from a picture. This module measures the picture the same way the
engine measures a model (silhouette grid, width profile in meters, colors) so the agent compares
text with text, and it matches a model against the picture: IoU, missing/extra regions in meters
next to the nearest part, per-band widths and per-part colors with a suggested base color.

Image frame: pixel (col, row), row grows down. Report frame of a reference alone: x from the
mask's horizontal center, y up from its bottom edge, in meters when the real height is known,
else in percent of the object's height. Match reports use the asset frame of the view's camera:
h = world point . camera.right, v = world point . camera.up (x and y for the front view).

Colour comparison renders the model with its own flat Lambert shading from the style light
(base color x (ambient + (1 - ambient) * max(0, n . -light))), no patterns, shadows or gamma,
so the numbers do not depend on render.py.
"""

from __future__ import annotations

import colorsys
import importlib
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .camera import Camera, make_cameras, orbit_view, parse_orbit, view_basis
from .errors import SightlineError
from .march import ViewBuffers, march, trace

__all__ = [
    "analyze_reference",
    "format_reference",
    "match_asset",
    "format_match",
    "fit_view",
    "format_fit",
    "format_grid",
    "label_components",
    "shade_lambert",
    "srgb_to_lab",
    "delta_e",
]

BANDS = 10  # horizontal bands of the width profile, from the top
GRID_MAX_COLS = 32
GRID_MAX_ROWS = 48
MAX_SIDE = 1600  # larger images are box-reduced by an integer factor before analysis
MIN_COMPONENT = 0.005  # share of the image area a mask component needs to be kept
MIN_REGION = 0.01  # share of the reference area a missing/extra region needs to be listed
MIN_REGION_PX = 1.5  # ... and its mean thickness in grid px (thinner = edge noise from the half-pixel dilation)
BAND_SLACK = 0.02  # width-profile verdicts must hold with band borders moved by this share of the height (>= 1.5 px)
THIN_REGION_PX = 4  # a region of strokes up to this many grid px wide is a thin detail (wires), not a size error
BORDER_PX = 3  # border width used to estimate the background color
SPECK_NEAR_PX = 8  # dropped specks this close to the object are edge noise (JPEG blocks are 8 px)...
SPECK_TINY = 0.0005  # ... and so are specks under this share of the image area
KMEANS_K = 6
KMEANS_ITERS = 8
KMEANS_SAMPLE = 60000
THIN_MIN_PX = 10  # thin-detail components smaller than this are edge noise
LIGHTING = ("auto", "lit", "flat")
LIT_SPREAD = 4.0  # lit: object pixels lie this far (median RGB distance, 0..255) from their colour cluster...
LIT_SMOOTH = 0.2  # ... and this share of neighbouring interior pixels changes gently (shading, not flat paint)
FIT_RES = 64
FIT_AZIMUTHS = tuple(range(0, 360, 15))
FIT_ELEVATIONS = (0, 15, 30, 45, 60)
SHIFT_SEARCH = 0.25  # horizontal alignment search, share of the larger silhouette width
COVER_SAMPLES, COVER_STEP = 34, 21  # Fibonacci lattice of sub-samples per pixel on the model's silhouette edge:
# every sample has its own row and column offset, so straight edges resolve to 1/34 px
EXACT_EPS = 0.005  # hit threshold of those sub-samples in pixels (the march's 0.5 px widens every edge)

_AXIS_ALIASES = {(0, 0): "front", (90, 0): "right", (180, 0): "back", (270, 0): "left", (45, 30): "iso"}

# ----------------------------------------------------------------------------- colour


def srgb_to_lab(rgb) -> np.ndarray:
    """CIE Lab (D65) of sRGB colours in 0..1, shape (..., 3)."""
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    t = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e(a, b) -> np.ndarray | float:
    """CIE76 colour difference of sRGB colours in 0..1."""
    d = np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=-1)
    return float(d) if np.ndim(d) == 0 else d


def _hex(rgb01) -> str:
    r, g, b = (int(round(float(np.clip(v, 0.0, 1.0)) * 255)) for v in rgb01)
    return f"#{r:02x}{g:02x}{b:02x}"


def _part_color(i: int) -> tuple[int, int, int]:
    """Distinct flat colour per part index (golden-ratio hue spacing)."""
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.55, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


# ----------------------------------------------------------------------------- array helpers


def label_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-connected components of a boolean mask (vectorized union-find, no scipy).

    Returns (labels (H, W) int32 with -1 for background, numbered in row-major order of each
    component's first pixel; sizes (n,) int64).
    """
    m = np.ascontiguousarray(mask, dtype=bool)
    h, w = m.shape
    flat = m.ravel()
    labels = np.full(flat.size, -1, dtype=np.int32)
    if not flat.any():
        return labels.reshape(h, w), np.zeros(0, dtype=np.int64)
    idx = np.arange(flat.size)
    # Horizontal runs are joined up front: every pixel points at the first pixel of its run.
    starts = m.copy()
    starts[:, 1:] &= ~m[:, :-1]
    parent = np.maximum.accumulate(np.where(starts.ravel(), idx, -1))
    parent = np.where(flat, parent, idx)
    # Vertical edges between runs, hooked root-to-smaller-root until every edge is inside one tree.
    a = np.flatnonzero((m[:-1, :] & m[1:, :]).ravel())
    b = parent[a + w]
    a = parent[a]
    keep = a != b
    if keep.any():
        pairs = np.unique(np.stack([a[keep], b[keep]], axis=1), axis=0)
        a, b = pairs[:, 0], pairs[:, 1]
        while a.size:
            ra, rb = parent[a], parent[b]
            diff = ra != rb
            if not diff.any():
                break
            a, b, ra, rb = a[diff], b[diff], ra[diff], rb[diff]
            np.minimum.at(parent, np.maximum(ra, rb), np.minimum(ra, rb))
            while True:  # pointer jumping until every node points at its root
                nxt = parent[parent]
                if np.array_equal(nxt, parent):
                    break
                parent = nxt
    _, inverse = np.unique(parent[flat], return_inverse=True)
    labels[flat] = inverse
    return labels.reshape(h, w), np.bincount(inverse).astype(np.int64)


def _keep_components(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Mask of the components whose `keep[label]` is True."""
    if keep.size == 0:
        return np.zeros(labels.shape, dtype=bool)
    return (labels >= 0) & keep[np.maximum(labels, 0)]


def _dilate(m: np.ndarray) -> np.ndarray:
    """3x3 square dilation."""
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    tmp = out.copy()
    out[:, 1:] |= tmp[:, :-1]
    out[:, :-1] |= tmp[:, 1:]
    return out


def _erode(m: np.ndarray) -> np.ndarray:
    """3x3 square erosion (pixels outside the array do not erode)."""
    return ~_dilate(~m)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the set pixels, ends exclusive."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _sat(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero first row and column: S[i, j] = a[:i, :j].sum()."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1) + a.shape[2:], dtype=np.float64)
    s[1:, 1:] = np.asarray(a, dtype=np.float64).cumsum(0).cumsum(1)
    return s


def _sat_at(sat: np.ndarray, ys, xs) -> np.ndarray:
    """Integral of the pixel-constant image over [0, y] x [0, x] on the grid ys x xs (bilinear in the SAT)."""
    h, w = sat.shape[0] - 1, sat.shape[1] - 1
    ys = np.clip(np.asarray(ys, dtype=np.float64), 0, h)
    xs = np.clip(np.asarray(xs, dtype=np.float64), 0, w)
    iy = np.minimum(ys.astype(np.int64), h - 1)
    ix = np.minimum(xs.astype(np.int64), w - 1)
    fy = (ys - iy)[:, None]
    fx = (xs - ix)[None, :]
    if sat.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    top, bottom = sat[iy], sat[iy + 1]
    return ((top[:, ix] * (1 - fx) + top[:, ix + 1] * fx) * (1 - fy)
            + (bottom[:, ix] * (1 - fx) + bottom[:, ix + 1] * fx) * fy)


def _box_sum(sat: np.ndarray, y_edges, x_edges) -> np.ndarray:
    """Sum of the image over the boxes between consecutive edges (fractional pixels allowed)."""
    f = _sat_at(sat, y_edges, x_edges)
    return f[1:, 1:] - f[:-1, 1:] - f[1:, :-1] + f[:-1, :-1]


def _box_cover(sat: np.ndarray, y_edges, x_edges) -> np.ndarray:
    """Mean over each box; parts of a box outside the image count as 0."""
    area = np.outer(np.diff(y_edges), np.diff(x_edges))
    return _box_sum(sat, y_edges, x_edges) / np.maximum(area, 1e-12)


def _box_color(sat_rgb: np.ndarray, y_edges, x_edges) -> tuple[np.ndarray, np.ndarray]:
    """Mean colour over the image part of each box, and that part's area share (0 = fully outside)."""
    h, w = sat_rgb.shape[0] - 1, sat_rgb.shape[1] - 1
    inside = np.outer(np.diff(np.clip(y_edges, 0, h)), np.diff(np.clip(x_edges, 0, w)))
    total = np.outer(np.diff(y_edges), np.diff(x_edges))
    color = _box_sum(sat_rgb, y_edges, x_edges) / np.maximum(inside, 1e-12)[..., None]
    return color, inside / np.maximum(total, 1e-12)


NICE_STEPS = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def _nice(x: float) -> float:
    """Smallest readable step (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 times a power of ten) >= x."""
    x = max(float(x), 1e-9)
    p = 10.0 ** math.floor(math.log10(x))
    for m in NICE_STEPS:
        if m * p >= x * (1 - 1e-9):
            return round(m * p, 12)
    return 10.0 * p


def _ranges(bands: list[int]) -> list[list[int]]:
    """Consecutive band indices -> [[start %, end %], ...]."""
    out: list[list[int]] = []
    for b in bands:
        if out and out[-1][1] == b * 100 // BANDS:
            out[-1][1] = (b + 1) * 100 // BANDS
        else:
            out.append([b * 100 // BANDS, (b + 1) * 100 // BANDS])
    return out


def _band_rows(top: int, bottom: int) -> list[tuple[int, int]]:
    """Row ranges [r0, r1) of the BANDS horizontal bands between two pixel rows."""
    edges = [top + (bottom - top) * b // BANDS for b in range(BANDS + 1)]
    return list(zip(edges[:-1], edges[1:]))


def _extent(mask: np.ndarray, r0: int, r1: int) -> tuple[int, int] | None:
    """(left, right exclusive) column extent of a mask over rows [r0, r1), None if empty."""
    if r1 <= r0:
        return None
    cols = np.flatnonzero(mask[r0:r1].any(axis=0))
    return (int(cols[0]), int(cols[-1]) + 1) if cols.size else None


# ----------------------------------------------------------------------------- image loading


def _open_image(path: pathlib.Path, label: str) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int], int]:
    """(rgb float32 0..255 (H, W, 3), alpha uint8 or None, original (w, h), reduction factor)."""
    if not path.is_file():
        raise SightlineError(f"reference image not found: {label}", hint="give the path to a PNG or JPEG file")
    try:
        with Image.open(path) as im:
            im.load()
            size = im.size
            has_alpha = im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info
            im = im.convert("RGBA" if has_alpha else "RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SightlineError(f"cannot read image: {exc}", where=label, hint="use a PNG or JPEG file") from None
    factor = max(1, math.ceil(max(size) / MAX_SIDE))
    if factor > 1:
        im = im.reduce(factor)
    arr = np.asarray(im)
    return arr[..., :3].astype(np.float32), (arr[..., 3].copy() if has_alpha else None), size, factor


def _load_mask(path: pathlib.Path, size: tuple[int, int], factor: int, label: str) -> np.ndarray:
    """Black/white (or transparent) mask PNG -> bool mask at the analysis size."""
    if not path.is_file():
        raise SightlineError(f"mask image not found: {path.as_posix()}", where=label, hint="mask: auto or a black/white PNG")
    try:
        with Image.open(path) as im:
            im.load()
            if im.size != size:
                raise SightlineError(f"mask is {im.size[0]} x {im.size[1]} px but the image is {size[0]} x {size[1]} px",
                                     where=label, hint="the mask must have exactly the size of the reference image")
            has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            im = im.convert("RGBA" if has_alpha else "L")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SightlineError(f"cannot read mask image: {exc}", where=label, hint="use a black/white PNG") from None
    if factor > 1:
        im = im.reduce(factor)
    arr = np.asarray(im)
    if arr.ndim == 3:
        alpha = arr[..., 3]
        if (alpha < 250).any():
            return alpha >= 128
        arr = np.asarray(Image.fromarray(arr).convert("L"))
    return arr >= 128


# ----------------------------------------------------------------------------- segmentation


def _segment(rgb: np.ndarray, alpha: np.ndarray | None, mask_file: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """Object mask and the facts about how it was found (SPEC 12.1 segmentation)."""
    h, w = rgb.shape[:2]
    info: dict = {"source": None, "background": None, "tolerance": None, "border_spread": None}
    reasons: list[str] = []
    levels: list[int] = []  # 0 high, 1 medium, 2 low
    dropped = 0
    if mask_file is not None:
        info["source"] = "mask file"
        mask = mask_file.copy()
    else:
        if alpha is not None and (alpha < 250).any():
            info["source"] = "alpha"
            raw = alpha >= 128
        else:
            info["source"] = "background"
            b = max(1, min(BORDER_PX, h // 4, w // 4))
            border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                                     rgb[b:-b, :b].reshape(-1, 3), rgb[b:-b, -b:].reshape(-1, 3)])
            bg = np.median(border, axis=0)
            spread = float(np.median(np.linalg.norm(border - bg, axis=1)))  # MAD of the colour distance
            tol = max(24.0, 3.0 * spread)
            raw = np.linalg.norm(rgb - bg, axis=-1) > tol
            info.update(background=_hex(bg / 255.0), tolerance=round(tol, 1), border_spread=round(spread, 1))
            level = 0 if spread <= 6 else 1 if spread <= 14 else 2
            levels.append(level)
            if level:
                reasons.append(f"the background is not plain (border colour spread {spread:.0f})")
        labels, sizes = label_components(raw)
        keep = sizes >= MIN_COMPONENT * h * w
        mask = _keep_components(labels, keep)
        dropped = int((~keep).sum())
        # Compression noise (JPEG ringing) leaves tiny specks along the object's edges: only specks that are
        # neither tiny nor next to the object say that the background is not clean.
        near = mask.copy()
        for _ in range(SPECK_NEAR_PX):
            near = _dilate(near)
        touching = np.zeros(sizes.size, dtype=bool)
        touching[np.unique(labels[near & (labels >= 0)])] = True
        stray = int((~keep & ~touching & (sizes >= SPECK_TINY * h * w)).sum())
        if stray > 20:
            levels.append(1)
            reasons.append(f"{stray} small specks were dropped")
    if not mask.any():
        raise SightlineError(
            "no object found in the reference image",
            hint="send a cut-out PNG (transparent background) or a picture on a plain background, "
                 "or give mask: <black/white png>",
        )
    labels, sizes = label_components(mask)
    # Standing on the bottom edge is normal for concept art; touching the other sides is not.
    sides = np.concatenate([mask[0], mask[1:, 0], mask[1:, -1]])
    touch = float(sides.mean())
    touch_bottom = float(mask[-1].mean())
    coverage = float(mask.mean())
    if info["source"] != "mask file":
        if touch > 0.02:
            levels.append(1 if touch <= 0.15 else 2)
            reasons.append(f"the object touches {touch:.0%} of the image's top/left/right border")
        if coverage > 0.6:
            levels.append(1 if coverage <= 0.85 else 2)
            reasons.append(f"the object fills {coverage:.0%} of the image")
    info.update(
        confidence=("high", "medium", "low")[max(levels, default=0)],
        reasons=reasons,
        coverage=round(coverage, 4),
        bbox_px=list(_bbox(mask)),
        components=int(sizes.size),
        dropped_specks=dropped,
        border_touch=round(touch, 3),
        bottom_touch=round(touch_bottom, 3),
    )
    return mask, info


# ----------------------------------------------------------------------------- reference analysis


@dataclass
class _Frame:
    """Reference pixel -> report units: x from the mask's horizontal center, y up from its bottom."""

    cx: float
    bottom: float
    upp: float  # units per pixel
    unit: str  # "m" or "%" (of the object's height)

    def x(self, px):
        return (np.asarray(px, dtype=np.float64) - self.cx) * self.upp

    def y(self, py):
        return (self.bottom - np.asarray(py, dtype=np.float64)) * self.upp


@dataclass
class _Ref:
    path: pathlib.Path
    label: str
    rgb: np.ndarray  # (H, W, 3) float32 0..255 (at the analysis size)
    alpha: np.ndarray | None
    mask: np.ndarray  # (H, W) bool
    bbox: tuple[int, int, int, int]
    frame: _Frame
    report: dict


def _elevated(view: str) -> bool:
    """True for views looking down at an angle (iso, orbit with elevation != 0)."""
    orbit = (45.0, 30.0) if view == "iso" else parse_orbit(view)
    return orbit is not None and orbit[1] != 0


def _num(v: float, unit: str) -> float:
    """Round a report coordinate: meters to 2 decimals, percent to integers."""
    return round(float(v), 2) if unit == "m" else float(round(float(v)))


def _fmt(v, unit: str, signed: bool = False) -> str:
    if v is None:
        return "-"
    if unit == "m":
        return f"{v:+.2f}" if signed else f"{v:.2f}"
    return f"{v:+.0f}" if signed else f"{v:.0f}"


def _grid_layout(h_lo: float, h_hi: float, v_lo: float, v_hi: float) -> tuple[float, float, float, int, int]:
    """Square cells over [h_lo, h_hi] x [v_lo, v_hi] (v up): (cell, x0 left edge, y0 top edge, cols, rows).

    The cell is a readable step (`_nice`), h = 0 is a column center and v = 0 a row edge, <= 32 cols, <= 48 rows.
    """
    cell = _nice(max((h_hi - h_lo) / (GRID_MAX_COLS - 2), (v_hi - v_lo) / (GRID_MAX_ROWS - 2), 1e-6))
    k_lo = math.floor(h_lo / cell + 0.5 + 1e-9)
    k_hi = math.ceil(h_hi / cell + 0.5 - 1e-9)
    r_lo = math.floor(v_lo / cell + 1e-9)
    r_hi = math.ceil(v_hi / cell - 1e-9)
    cols, rows = max(1, k_hi - k_lo), max(1, r_hi - r_lo)
    return cell, (k_lo - 0.5) * cell, r_hi * cell, cols, rows


def _grid_dict(cell, x0, y0, rows: list[str], unit: str, legend: dict, axes=("x", "y")) -> dict:
    """Grid in inspect's silhouette format; cell, x0 (left edge) and y0 (top edge) in `unit`."""
    nd = 2 if unit == "m" else 4  # cells are readable steps, so 2 decimals are exact in meters
    return {"cols": len(rows[0]) if rows else 0, "rows": len(rows), "cell": round(cell, nd), "unit": unit,
            "x0": round(x0, nd), "y0": round(y0, nd), "axes": list(axes), "legend": legend, "grid": rows}


def _silhouette(mask_sat: np.ndarray, frame: _Frame, bbox) -> dict:
    x0, y0, x1, y1 = bbox
    cell, gx0, gy0, cols, rows = _grid_layout(frame.x(x0), frame.x(x1), 0.0, frame.y(y0))
    xe = frame.cx + (gx0 + cell * np.arange(cols + 1)) / frame.upp
    ye = frame.bottom - (gy0 - cell * np.arange(rows + 1)) / frame.upp
    cover = _box_cover(mask_sat, ye, xe)
    grid = ["".join("#" if c >= 0.5 else "." for c in row) for row in cover]
    return _grid_dict(cell, gx0, gy0, grid, frame.unit, {"#": "filled", ".": "empty"})


def _kmeans(pixels: np.ndarray, k: int = KMEANS_K, iters: int = KMEANS_ITERS) -> np.ndarray:
    """Deterministic k-means in RGB: init at sorted-luminance quantiles; an empty or duplicate
    cluster is re-seeded at the pixel farthest from its center. Returns the centers (k', 3)."""
    n = len(pixels)
    step = max(1, math.ceil(n / KMEANS_SAMPLE))
    x = pixels[::step].astype(np.float64)
    lum = x @ np.array([0.2126, 0.7152, 0.0722])
    order = np.argsort(lum, kind="stable")
    k = max(1, min(k, len(x)))
    centers = x[order[((np.arange(k) + 0.5) / k * len(x)).astype(int)]].copy()
    xx = (x * x).sum(axis=1)
    for _ in range(iters):
        d = xx[:, None] - 2.0 * x @ centers.T + (centers * centers).sum(axis=1)[None, :]
        assign = d.argmin(axis=1)
        counts = np.bincount(assign, minlength=k)
        for j in range(k):
            if counts[j]:
                centers[j] = x[assign == j].mean(axis=0)
        # Re-seed empty clusters and exact duplicates at the worst-fitting pixel.
        dist = d[np.arange(len(x)), assign]
        for j in range(k):
            dup = any(np.allclose(centers[j], centers[i]) for i in range(j))
            if counts[j] == 0 or dup:
                far = int(dist.argmax())
                if dist[far] <= 1.0:
                    break
                centers[j] = x[far]
                dist[far] = 0.0
    return centers


def _colors(rgb: np.ndarray, mask: np.ndarray, bbox, frame: _Frame, style) -> list[dict]:
    """Dominant colours of the object: hex, share, bands where each dominates, center, nearest material."""
    pixels = rgb[mask]
    centers = _kmeans(pixels)
    # Assign every object pixel, then drop empty clusters and merge near-identical ones (dE < 2).
    pp = pixels.astype(np.float64)
    d = (pp * pp).sum(axis=1)[:, None] - 2.0 * pp @ centers.T + (centers * centers).sum(axis=1)[None, :]
    assign = d.argmin(axis=1)
    lab = srgb_to_lab(centers / 255.0)
    target = np.arange(len(centers))
    for j in range(len(centers)):
        for i in range(j):
            if target[i] == i and np.linalg.norm(lab[i] - lab[j]) < 2.0:
                target[j] = i
                break
    assign = target[assign]
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = bbox
    band_of = np.minimum((ys - y0) * BANDS // max(1, y1 - y0), BANDS - 1)
    counts = np.zeros((BANDS, len(centers)), dtype=np.int64)
    np.add.at(counts, (band_of, assign), 1)
    dominant = np.where(counts.sum(axis=1) > 0, counts.argmax(axis=1), -1)
    materials = []
    if style is not None:
        materials = [(name, np.array(m.color)) for name, m in style.materials.items() if name != "helper"]
    out = []
    total = len(assign)
    for j in np.unique(assign):
        sel = assign == j
        color = pixels[sel].mean(axis=0) / 255.0
        share = sel.sum() / total
        if share < 0.01:
            continue
        rng = _ranges([b for b in range(BANDS) if dominant[b] == j])
        if rng:
            where = "rows " + ", ".join(f"{a}-{b}%" for a, b in rng)
            mid = (rng[0][0] + rng[-1][1]) / 2
            where += " (top)" if mid < 100 / 3 else " (middle)" if mid < 200 / 3 else " (bottom)"
        else:
            where = "not dominant in any band"
        entry = {
            "hex": _hex(color),
            "share": round(float(share), 3),
            "rows": rng,
            "where": where,
            "center": [_num(frame.x(xs[sel].mean() + 0.5), frame.unit), _num(frame.y(ys[sel].mean() + 0.5), frame.unit)],
            "nearest_material": None,
            "material_delta_e": None,
        }
        if materials:
            des = [delta_e(color, c) for _, c in materials]
            best = int(np.argmin(des))
            entry["nearest_material"] = materials[best][0]
            entry["material_delta_e"] = round(float(des[best]), 1)
        out.append(entry)
    out.sort(key=lambda e: (-e["share"], e["hex"]))
    return out


def _thin_details(mask: np.ndarray, frame: _Frame) -> list[dict]:
    """Structures under 3 px wide: the mask minus its 3x3 opening, as components."""
    thin = mask & ~_dilate(_erode(mask))
    labels, sizes = label_components(thin)
    out = []
    for j in np.flatnonzero(sizes >= THIN_MIN_PX):
        ys, xs = np.nonzero(labels == j)
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        length = max(bw, bh)
        if length < 6:
            continue
        out.append({
            "px": int(sizes[j]),
            "length_px": length,
            "width_px": round(float(sizes[j]) / length, 1),
            "x": [_num(frame.x(xs.min()), frame.unit), _num(frame.x(xs.max() + 1), frame.unit)],
            "y": [_num(frame.y(ys.max() + 1), frame.unit), _num(frame.y(ys.min()), frame.unit)],
        })
    out.sort(key=lambda e: (-e["px"], e["x"][0]))
    return out[:5]


def _lighting(rgb: np.ndarray, mask: np.ndarray, given: str | None = None) -> dict:
    """Is the picture lit (a shaded render or photo) or flat paint (concept art)?

    Shading spreads every paint over a range of tones and changes them gently from pixel to pixel;
    flat paint keeps a region one colour (JPEG noise moves it by a step or two). So a picture is lit
    when the object's pixels lie on median >= LIT_SPREAD from their k-means colour and >= LIT_SMOOTH
    of the neighbouring interior pixel pairs differ by 0.5..8 luminance steps. `given` ("lit"/"flat",
    the reference's `lighting:` key) overrides the guess; the measurements are reported either way.
    """
    lum = rgb.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    inner = _erode(_erode(mask))
    steps = [np.abs(lum[:, 1:] - lum[:, :-1])[inner[:, 1:] & inner[:, :-1]],
             np.abs(lum[1:, :] - lum[:-1, :])[inner[1:, :] & inner[:-1, :]]]
    d = np.concatenate(steps)
    smooth = float(np.mean((d > 0.5) & (d <= 8.0))) if d.size else 0.0
    pixels = rgb[mask]
    step = max(1, math.ceil(len(pixels) / KMEANS_SAMPLE))
    sample = pixels[::step].astype(np.float64)
    centers = _kmeans(pixels)
    dist = np.sqrt(((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    spread = float(np.median(dist)) if dist.size else 0.0
    obj = float(np.median(lum[mask]))
    bg = float(np.median(lum[~mask])) if (~mask).any() else None
    guess = "lit" if spread >= LIT_SPREAD and smooth >= LIT_SMOOTH else "flat"
    mode = given if given in ("lit", "flat") else guess
    evidence = (f"{smooth:.0%} of neighbouring pixels change gently, colours spread {spread:.1f} around their "
                f"clusters" + (f"; the object (luminance {obj:.0f}) is darker than the background ({bg:.0f})"
                               if bg is not None and obj < bg - 10 else ""))
    return {"mode": mode, "source": "given" if given in ("lit", "flat") else "auto", "guess": guess,
            "smooth": round(smooth, 3), "spread": round(spread, 1), "object_luminance": round(obj, 1),
            "background_luminance": None if bg is None else round(bg, 1), "evidence": evidence}


_HIDDEN = {
    "front": ("front", "depth (front to back)", "back and sides"),
    "back": ("back", "depth (front to back)", "front and sides"),
    "left": ("left side", "width (left to right)", "right side, front and back"),
    "right": ("right side", "width (left to right)", "left side, front and back"),
}


def _unknowns(rep: dict, view: str, model_height: float | None = None) -> list[dict]:
    """Questions the picture cannot answer, each with a proposed default answer."""
    out = []
    unit = rep["scale"]["unit"]
    size = rep["size"]
    if view in _HIDDEN:
        seen, dim, rest = _HIDDEN[view]
        other = size["width"]
        default_dim = f"{other:.2f} m (same as the visible width)" if unit == "m" else "the same as the visible width"
        # Sails, arms or a canopy widen the picture; the body (median width of the lower half) is the better guess.
        lower = [b["width"] for b in rep.get("width_profile", [])[BANDS // 2:] if b.get("width")]
        body = float(np.median(lower)) if lower else other
        if unit == "m" and body < 0.9 * other:
            default_dim = f"{body:.2f} m (as wide as the body, the median width of the lower half; the full width is {other:.2f} m)"
        out.append({"topic": "hidden sides",
                    "question": f"The picture shows only the {seen}. What is the {dim}, and what do the {rest} look like?",
                    "default": f"{dim.split(' ')[0]} = {default_dim}; {rest} plain, in the same materials"})
    elif view == "top":
        out.append({"topic": "hidden sides", "question": "The picture shows only the top. How tall is it, and what do the sides look like?",
                    "default": "walls straight down to the ground, height = the shorter footprint side"})
    else:
        out.append({"topic": "hidden sides", "question": "The far sides are hidden in this view. Do they look like the visible ones?",
                    "default": "yes: mirror the visible sides"})
    if rep["scale"]["mpp"] is None:
        default = (f"{model_height:.2f} m, the model's current height (the matcher fits the picture to it)" if model_height
                   else "keep the model's height; `sightline match` fits the picture to it")
        out.append({"topic": "scale", "question": "How tall is the object in meters?", "default": default})
    mask = rep["mask"]
    if mask["confidence"] != "high":
        why = "; ".join(mask["reasons"])
        out.append({"topic": "mask",
                    "question": f"The outline is uncertain ({why}). Can you send a cut-out PNG (transparent background) "
                                "or a picture on a plain background?",
                    "default": "use the automatic outline as is and trust only the large shapes"})
    elif mask["border_touch"] > 0.0:
        out.append({"topic": "cropped", "question": "The object touches the picture's top or side edge. Is it cut off?",
                    "default": "no: the visible part is the whole object"})
    elif mask["bottom_touch"] > 0.0:
        out.append({"topic": "cropped", "question": "The object reaches the picture's bottom edge. Is that the ground, or is it cut off?",
                    "default": "that is the ground: the object stands on the bottom edge"})
    if mask["components"] > 1:
        out.append({"topic": "pieces", "question": f"The picture shows {mask['components']} separate pieces. One asset or several?",
                    "default": "one asset, each piece its own part"})
    thin = rep["thin_details"]
    if thin:
        t = thin[0]
        u = "m" if unit == "m" else "% of the height"
        where = f"x {_fmt(t['x'][0], unit)}..{_fmt(t['x'][1], unit)}, y {_fmt(t['y'][0], unit)}..{_fmt(t['y'][1], unit)} {u}"
        out.append({"topic": "thin details",
                    "question": f"{len(thin)} detail(s) are thinner than 3 px (largest at {where}). "
                                "Are they real parts (pole, rope, antenna), and do they matter at gameplay distance?",
                    "default": "skip them in the blockout; add thin parts (>= 0.05 m) later if they matter"})
    colors = rep["colors"][:4]
    light = rep.get("lighting") or {"mode": "flat", "source": "auto", "evidence": ""}
    if colors:
        listing = ", ".join(f"{c['hex']} ({c['share']:.0%})" + (f" ~ {c['nearest_material']}" if c["nearest_material"] else "")
                            for c in colors)
        if light["mode"] == "lit":
            seen = "is lit" if light["source"] == "given" else f"looks lit ({light['evidence']})"
            out.append({"topic": "materials",
                        "question": f"The picture {seen}, so its main colours are shaded, not the paint: {listing}. "
                                    "Which materials are they?",
                        "default": "lit: choose each part's material by what it is and keep the picture's light-to-dark "
                                   "order; the nearest materials listed match the shaded colours and are often too dark; "
                                   "`sightline match` suggests base colours corrected for the picture's overall light"})
        elif light["source"] == "given":
            out.append({"topic": "materials", "question": f"Which materials are the main colours: {listing}?",
                        "default": "the nearest style materials listed (flat paint: the picture colours are the base colours)"})
        else:
            out.append({"topic": "materials",
                        "question": f"The picture looks like flat paint ({light['evidence']}). Which materials are the "
                                    f"main colours: {listing}? Are they flat paint or lit (with shading)?",
                        "default": "the nearest style materials listed; flat paint, so the picture colours are the base colours"})
    return out


def _analyze(image_path, view: str = "front", height=None, mask=None, style=None, root=None, lighting=None) -> _Ref:
    path = pathlib.Path(image_path)
    label = path.as_posix()
    if root is not None:
        try:
            label = path.resolve().relative_to(pathlib.Path(root).resolve()).as_posix()
        except ValueError:
            pass
    view_basis(view)  # validates the view name (SightlineError with did_you_mean)
    if lighting is not None and lighting not in LIGHTING:
        raise SightlineError(f"lighting must be one of {', '.join(LIGHTING)}, got {lighting!r}", where=label,
                             hint="lit = a shaded render or photo, flat = flat paint (concept art), auto = guess")
    if height is not None:
        if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
            raise SightlineError(f"height must be a number > 0 (meters), got {height!r}", where=label)
        height = float(height)
    rgb, alpha, size, factor = _open_image(path, label)
    mask_arr = None
    if mask is not None and str(mask) != "auto":
        mask_arr = _load_mask(pathlib.Path(mask), size, factor, label)
    obj, minfo = _segment(rgb, alpha, mask_arr)
    x0, y0, x1, y1 = minfo["bbox_px"]
    bw, bh = x1 - x0, y1 - y0
    mpp = height / bh if height else None
    frame = _Frame(cx=(x0 + x1) / 2.0, bottom=float(y1), upp=mpp if mpp else 100.0 / bh, unit="m" if mpp else "%")
    sat = _sat(obj.astype(np.float32))

    if mpp:
        note = f"height {height:.2f} m given: 1 px = {mpp:.4f} m"
        if _elevated(view):
            note += " (approximate: in an elevated view the picture's height also includes the top)"
    else:
        note = "unknown (no height given): sizes are % of the object's height; `sightline match` fits the picture to the model's height"
    scale = {"mpp": round(mpp, 5) if mpp else None, "height": height, "assumed": mpp is None, "unit": frame.unit, "note": note}

    crop = obj[y0:y1, x0:x1]
    ys, xs = np.nonzero(obj)
    mirror = crop[:, ::-1]
    sym = float((crop & mirror).sum() / max(1, (crop | mirror).sum()))
    widths = []
    profile = []
    for b, (r0, r1) in enumerate(_band_rows(y0, y1)):
        ext = _extent(obj, r0, r1)
        entry = {"band": f"{b * 100 // BANDS}-{(b + 1) * 100 // BANDS}%", "top": _num(frame.y(r0), frame.unit),
                 "bottom": _num(frame.y(r1), frame.unit), "left": None, "right": None, "width": None}
        if ext:
            entry.update(left=_num(frame.x(ext[0]), frame.unit), right=_num(frame.x(ext[1]), frame.unit),
                         width=_num((ext[1] - ext[0]) * frame.upp, frame.unit))
            widths.append((ext[1] - ext[0], b))
        profile.append(entry)
    widest = max(widths, key=lambda t: (t[0], -t[1]))[1] if widths else 0
    proportions = {
        "height_to_width": round(bh / bw, 3),
        "fill": round(float(crop.mean()), 3),
        "symmetry_lr": round(sym, 3),
        "center": [_num(frame.x(xs.mean() + 0.5), frame.unit), _num(frame.y(ys.mean() + 0.5), frame.unit)],
        "widest_band": profile[widest]["band"],
    }
    rep = {
        "image": label,
        "size_px": [int(size[0]), int(size[1])],
        "analyzed_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "view": view,
        "mask": minfo,
        "scale": scale,
        "size": {"width": _num(bw * frame.upp, frame.unit), "height": _num(bh * frame.upp, frame.unit),
                 "width_px": bw, "height_px": bh},
        "proportions": proportions,
        "silhouette": _silhouette(sat, frame, (x0, y0, x1, y1)),
        "width_profile": profile,
        "colors": _colors(rgb, obj, (x0, y0, x1, y1), frame, style),
        "thin_details": _thin_details(obj, frame),
        "lighting": _lighting(rgb, obj, lighting),
    }
    rep["unknowns"] = _unknowns(rep, view)
    return _Ref(path=path, label=label, rgb=rgb, alpha=alpha, mask=obj, bbox=(x0, y0, x1, y1), frame=frame, report=rep)


def analyze_reference(image_path, view: str = "front", height: float | None = None, mask=None, *,
                      style=None, root=None, lighting: str | None = None) -> dict:
    """Measure a reference picture (SPEC 12.1): mask, scale, proportions, silhouette grid, width
    profile, dominant colours and the questions it cannot answer.

    `mask`: None/"auto" or a black/white PNG path. `style` (default: the default style) is used to
    name the nearest material of each colour; `root` only shortens the image path in the report.
    `lighting`: "lit" or "flat" when known (the designer said so), else guessed from the shading.
    """
    if style is None:
        try:
            from .style import load_style

            style = load_style("default", root)
        except SightlineError:
            style = None
    return _analyze(image_path, view, height, mask, style, root, lighting).report


# ----------------------------------------------------------------------------- text output


def format_grid(grid: dict) -> list[str]:
    """Grid rows with row labels (cell centers) and a column ruler, characters space-separated."""
    unit, cell = grid["unit"], grid["cell"]
    hname, vname = grid.get("axes", ["x", "y"])
    rows = grid["grid"]
    labels = [_fmt(grid["y0"] - (r + 0.5) * cell, unit) for r in range(len(rows))]
    lw = max([len(s) for s in labels] + [len(vname) + 2])
    out = [f"  {(vname + ' ' + ('m' if unit == 'm' else '%')):>{lw}}"]
    for lab, row in zip(labels, rows):
        out.append(f"  {lab:>{lw}}  " + " ".join(row))
    cols = grid["cols"]
    centers = [grid["x0"] + (c + 0.5) * cell for c in range(cols)]
    ticks = [c for c in range(cols) if round(centers[c] / cell) % 4 == 0]
    pad = " " * (lw + 4)
    tick_line = list(" " * (2 * cols))
    text_line = list(" " * (2 * cols + 8))
    last_end = -1
    for c in ticks:
        tick_line[2 * c] = "^"
        s = _fmt(centers[c], unit)
        start = 2 * c
        if start > last_end:
            text_line[start:start + len(s)] = s
            last_end = start + len(s)
    out.append(pad + "".join(tick_line).rstrip())
    out.append(pad + "".join(text_line).rstrip() + f"   {hname} {'m' if unit == 'm' else '%'}")
    return out


def _pct(v) -> str:
    return f"{v * 100:.0f}%"


def format_reference(report: dict) -> str:
    """Stable text form of `analyze_reference` (headings in CAPS, aligned columns)."""
    r = report
    unit = r["scale"]["unit"]
    u = " m" if unit == "m" else "%"
    m = r["mask"]
    lines = [f"REFERENCE  {r['image']}  {r['size_px'][0]} x {r['size_px'][1]} px  view {r['view']}"]
    if r["analyzed_px"] != r["size_px"]:
        lines.append(f"           analyzed at {r['analyzed_px'][0]} x {r['analyzed_px'][1]} px (pixel numbers below use this size)")
    src = m["source"]
    if src == "background":
        src = f"background {m['background']} (tolerance {m['tolerance']:.0f}, border spread {m['border_spread']:.0f})"
    bx0, by0, bx1, by1 = m["bbox_px"]
    lines.append(f"MASK       from {src}  confidence {m['confidence']}  coverage {_pct(m['coverage'])}  "
                 f"{m['components']} component(s)")
    lines.append(f"           bbox x {bx0}..{bx1} y {by0}..{by1} px ({bx1 - bx0} x {by1 - by0} px)"
                 + (f"; {m['dropped_specks']} specks dropped" if m["dropped_specks"] else ""))
    for why in m["reasons"]:
        lines.append(f"           ! {why}")
    lines.append(f"SCALE      {r['scale']['note']}")
    s = r["size"]
    if unit == "m":
        lines.append(f"SIZE       {s['height']:.2f} m tall x {s['width']:.2f} m wide")
    else:
        lines.append(f"SIZE       width {s['width']:.0f}% of the height ({s['width_px']} x {s['height_px']} px)")
    p = r["proportions"]
    lines.append(f"PROPORTIONS  height/width {p['height_to_width']:.2f}  fill {_pct(p['fill'])}  "
                 f"left-right symmetry {_pct(p['symmetry_lr'])}  mass center x {_fmt(p['center'][0], unit)}{u}, "
                 f"y {_fmt(p['center'][1], unit)}{u}  widest band {p['widest_band']}")
    g = r["silhouette"]
    lines.append("")
    cell = f"{g['cell']:.2f}" if unit == "m" else f"{g['cell']:g}"
    lines.append(f"SILHOUETTE  {r['view']}  cell {cell}{u}  {g['cols']} x {g['rows']}  (# filled, . empty)")
    lines += format_grid(g)
    lines.append("")
    lines.append(f"WIDTH PROFILE  ({BANDS} bands from the top; {'m' if unit == 'm' else '% of the height'}, x from the center)")
    lines.append(f"  {'band':<9}{'y':>15}  {'left':>7} {'right':>7} {'width':>7}")
    for e in r["width_profile"]:
        yr = f"{_fmt(e['top'], unit)}..{_fmt(e['bottom'], unit)}"
        lines.append(f"  {e['band']:<9}{yr:>15}  {_fmt(e['left'], unit, True):>7} {_fmt(e['right'], unit, True):>7} "
                     f"{_fmt(e['width'], unit):>7}")
    lg = r.get("lighting")
    if lg:
        how = "given" if lg["source"] == "given" else "guessed"
        lines.append(f"LIGHTING   {lg['mode']} ({how}): {lg['evidence']}")
    lines.append("")
    if lg and lg["mode"] == "lit":
        lines.append("COLORS  (k-means on the object's pixels; share of the object; lit picture: shaded colours, so the")
        lines.append("        nearest materials match the shading, not the paint)")
    else:
        lines.append("COLORS  (k-means on the object's pixels; share of the object)")
    for c in r["colors"]:
        near = f"  ~ {c['nearest_material']} (dE {c['material_delta_e']:.0f})" if c["nearest_material"] else ""
        lines.append(f"  {c['hex']}  {_pct(c['share']):>4}  {c['where']:<34} center x {_fmt(c['center'][0], unit, True)}{u}, "
                     f"y {_fmt(c['center'][1], unit)}{u}{near}")
    if r["thin_details"]:
        lines.append("")
        lines.append("THIN DETAILS  (< 3 px wide)")
        for t in r["thin_details"]:
            lines.append(f"  {t['length_px']} px long, {t['width_px']:.1f} px wide at x {_fmt(t['x'][0], unit)}..{_fmt(t['x'][1], unit)}{u}, "
                         f"y {_fmt(t['y'][0], unit)}..{_fmt(t['y'][1], unit)}{u}")
    lines.append("")
    lines += _format_unknowns(r["unknowns"])
    return "\n".join(lines) + "\n"


def _format_unknowns(unknowns: list[dict]) -> list[str]:
    lines = ["UNKNOWNS  (ask the designer in one batch; each has a proposed default)"]
    for i, q in enumerate(unknowns, 1):
        lines.append(f"  {i}. [{q['topic']}] {q['question']}")
        lines.append(f"     default: {q['default']}")
    return lines


# ----------------------------------------------------------------------------- shading


def shade_lambert(asset, buffers: ViewBuffers, background=None, cam: Camera | None = None) -> np.ndarray:
    """Flat Lambert image (H, W, 3) float 0..1: material base colour x (ambient + (1 - ambient) * n . -light).

    No patterns, shadows or gamma; pixels without a hit get `background` (default: the style's). With
    `cam`, the light turns with that camera (`Camera.follow_light`): a back or side reference is lit
    like the front one, as on a concept turnaround sheet.
    """
    style = asset.style
    light = cam.follow_light(style.light_dir) if cam is not None else np.asarray(style.light_dir)
    to_light = -np.asarray(light, dtype=np.float32)
    lam = np.clip(buffers.normal @ to_light, 0.0, 1.0)
    shade = style.ambient + (1.0 - style.ambient) * lam
    base = np.zeros(buffers.hit.shape + (3,), dtype=np.float32)
    surf = asset.scene().surface_ids(buffers)  # paints (op: paint) colour parts of the solids
    for pi in np.unique(surf[buffers.hit]):
        base[surf == pi] = style.material(asset.parts[pi].material).color
    out = base * shade[..., None]
    out[~buffers.hit] = style.background if background is None else background
    return out


# ----------------------------------------------------------------------------- matching


def _asset_label(asset) -> str:
    try:
        from .style import display_path

        return display_path(asset.path, asset.root)
    except Exception:  # noqa: BLE001 - only cosmetic
        return str(asset.path)


def _reference_spec(asset, index: int):
    refs = asset.references
    where = _asset_label(asset)
    if not refs:
        raise SightlineError(f"asset '{asset.name}' has no reference image", where=where,
                             hint="add reference: {image: refs/<name>.png, view: front, height: 8.0} to the asset file")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(refs):
        raise SightlineError(f"reference index {index!r} is out of range", where=where,
                             hint=f"the asset has {len(refs)} reference(s): use index 0..{len(refs) - 1}")
    return refs[index]


def _axes(camera: Camera) -> tuple[tuple[str, float], tuple[str, float]]:
    """Asset-frame names of the image axes: ((name, sign) for h, (name, sign) for v); h/v when oblique."""
    out = []
    for vec, fallback in ((camera.right, "h"), (camera.up, "v")):
        k = int(np.argmax(np.abs(vec)))
        out.append(("xyz"[k], float(np.sign(vec[k]))) if abs(vec[k]) > 0.9999 else (fallback, 1.0))
    return out[0], out[1]


def _axis_range(lo: float, hi: float, axis: tuple[str, float]) -> list[float]:
    a, b = sorted((axis[1] * lo, axis[1] * hi))
    return [round(a, 2), round(b, 2)]


def _side(dx: float, dy: float) -> str:
    """Image side of an offset (dy grows down)."""
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "bottom" if dy > 0 else "top"


def _nearest_labels(labels: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Spread part labels (>= 0) into unlabeled pixels, 4-neighbour steps, until every `need` pixel has one."""
    out = labels.copy()
    while (need & (out < 0)).any():
        before = out.copy()
        for sl_dst, sl_src in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                               ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                               ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                               ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            dst, src = out[sl_dst], before[sl_src]
            fill = (dst < 0) & (src >= 0)
            dst[fill] = src[fill]
        if np.array_equal(before, out):
            break
    return out


def _open(m: np.ndarray, r: int) -> np.ndarray:
    """Opening with a (2r+1) square: what is left after removing strokes narrower than 2r+1 px."""
    out = m
    for _ in range(r):
        out = _erode(out)
    for _ in range(r):
        out = _dilate(out)
    return out & m


def _stroke_width(sel: np.ndarray) -> float:
    """Mean stroke width in px: 2 x area / perimeter (pixel edges between the region and the rest)."""
    m = np.pad(sel, 1)
    edges = int((m[:, 1:] != m[:, :-1]).sum() + (m[1:, :] != m[:-1, :]).sum())
    return 2.0 * int(sel.sum()) / max(1, edges)


def _sub_extent(cov: np.ndarray, r0: int, r1: int) -> tuple[float, float] | None:
    """Sub-pixel (left, right) of the columns covered >= 50% in some row of [r0, r1), None if none.

    The end columns add their partial coverage: an edge 0.3 px into a column reads 0.3, also when the
    column is under half covered (it then joins its covered neighbour).
    """
    if r1 <= r0:
        return None
    prof = cov[r0:r1].max(axis=0)
    idx = np.flatnonzero(prof >= 0.5)
    if not idx.size:
        return None
    a, b = int(idx[0]), int(idx[-1])
    left = a + 1.0 - float(prof[a]) - (float(prof[a - 1]) if a > 0 else 0.0)
    right = b + float(prof[b]) + (float(prof[b + 1]) if b + 1 < prof.size else 0.0)
    return left, right


def _gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Signed gap between two ranges (lo, hi): > 0 when a lies above b, < 0 below, 0 when they overlap."""
    a_lo, a_hi = min(a), max(a)
    b_lo, b_hi = min(b), max(b)
    if a_lo > b_hi:
        return a_lo - b_hi
    if a_hi < b_lo:
        return a_hi - b_lo
    return 0.0


def _majority(values: np.ndarray) -> int:
    vals, counts = np.unique(values[values >= 0], return_counts=True)
    return int(vals[np.argmax(counts)]) if vals.size else -1


@dataclass
class _Match:
    report: dict
    ref_rgb: np.ndarray  # (Hg, Wg, 3) 0..1 aligned reference colours (outside the picture: background)
    ref: np.ndarray  # (Hg, Wg) bool
    model: np.ndarray  # (Hg, Wg) bool
    part: np.ndarray  # (Hg, Wg) int32: who colours each pixel, the solid or a paint over it (op: paint)
    lambert: np.ndarray  # (Hg, Wg, 3) 0..1
    cover: np.ndarray | None = None  # (Hg, Wg) 0..1 the picture's share of each grid pixel (soft)
    coverage: np.ndarray | None = None  # (Hg, Wg) 0..1 the model's share of each grid pixel (soft)
    camera: Camera | None = None  # the model camera; its pixel (0, 0) is grid pixel `origin` = (row, col)
    origin: tuple[int, int] = (0, 0)
    picture: object = None  # the analysed reference (_Ref: rgb, mask, bbox at the analysis resolution)
    pic_rows: np.ndarray | None = None  # (Hg,) picture row (analysis pixels) at each grid row's centre
    pic_cols: np.ndarray | None = None  # (Wg,) picture column at each grid column's centre


def _model_coverage(scene, cam: Camera, hit: np.ndarray) -> np.ndarray:
    """Share of each pixel the model covers (0..1), free of the march's half-pixel dilation.

    Pixels two steps inside the (dilated) march hit mask count as covered and pixels outside its
    3x3 dilation as empty; every pixel of the band between is sampled with COVER_SAMPLES rays on a
    Fibonacci lattice at a hit threshold of EXACT_EPS px. So a 3 px thin wing covers 3 px, not 4,
    and the coverage does not depend on where the pixel grid falls (the scene bounds frame the
    camera): a 2.4 px wing reads 2.4 +- 0.03 px at any phase.
    """
    cov = _erode(_erode(hit)).astype(np.float32)
    band = _dilate(hit) & (cov == 0)
    rows, cols = np.nonzero(band)
    n = rows.size
    if n:
        k = np.arange(COVER_SAMPLES)
        ox = (k + 0.5) / COVER_SAMPLES
        oy = ((k * COVER_STEP) % COVER_SAMPLES + 0.5) / COVER_SAMPLES
        cc = (cols[:, None] + ox[None, :]).ravel()
        rr = (rows[:, None] + oy[None, :]).ravel()
        origins = (cam.pixel_to_world(cc, rr) - cam.forward * cam.near).astype(np.float32)
        _, part = trace(scene, origins, cam.forward[None, :].astype(np.float32), np.inf, EXACT_EPS * cam.mpp)
        cov[rows, cols] = (part >= 0).reshape(n, COVER_SAMPLES).mean(axis=1)
    return cov


def _cov_extent(cov: np.ndarray, axis: int) -> tuple[float, float]:
    """Sub-pixel first and last edge of the covered area: rows (axis 0) or columns (axis 1).

    The outermost row (column) covers its pixels by at most its largest coverage, measured from the
    inner side: a flat edge 0.3 px into its row reads exactly 0.3.
    """
    prof = cov.max(axis=1 - axis)
    idx = np.flatnonzero(prof > 0)
    a, b = int(idx[0]), int(idx[-1])
    if a == b:
        return a + 0.5 - float(prof[a]) / 2.0, a + 0.5 + float(prof[a]) / 2.0
    return a + 1.0 - float(prof[a]), b + float(prof[b])


def _match(asset, index: int, res: int, view: str | None) -> _Match:
    spec = _reference_spec(asset, index)
    view = view or spec.view
    mask = None if str(spec.mask) == "auto" else spec.mask
    lighting = getattr(spec, "lighting", "auto")
    ref = _analyze(spec.image, view, spec.height, mask, asset.style, asset.root, None if lighting == "auto" else lighting)
    scene = asset.scene()
    lo, hi = scene.bounds(include_helpers=False)
    cam = make_cameras([view], lo, hi, res=res)[view]
    vb = march(scene, cam)
    if not vb.hit.any():
        raise SightlineError(f"the model is not visible from view '{view}'", where=_asset_label(asset))
    mpp = cam.mpp
    H, W = vb.hit.shape
    mcov = _model_coverage(scene, cam, vb.hit)
    if not mcov.any():  # thinner than the sub-samples everywhere: fall back to the dilated mask
        mcov = vb.hit.astype(np.float32)
    left, right = _cov_extent(mcov, 1)
    top, sil_bottom = _cov_extent(mcov, 0)

    # Vertical alignment: in upright views a grounded model stands on y = 0 like the picture's object,
    # so the picture's bottom goes on the ground line, not on the lowest model pixel (a part poking
    # 4 cm below the ground would shift the whole picture down). Elevated views and models that
    # float clear of the ground use the silhouette's bottom.
    c_v = float(np.dot(cam.center, cam.up))
    align_on, below = "silhouette", []
    mbot = sil_bottom
    if float(cam.up[1]) > 1.0 - 1e-9:
        base_y = max(float(lo[1]), 0.0)
        base_row = H / 2.0 - (base_y - c_v) / mpp
        if sil_bottom > base_row - 1.0:
            mbot = base_row
            align_on = "ground" if base_y == 0.0 else "bottom"
            if sil_bottom > base_row + 0.25:
                rows_below = np.arange(H)[:, None] + 0.5 > base_row + 0.25
                seen = np.unique(vb.part[rows_below & vb.hit & (mcov > 0)])
                for pi in seen[seen >= 0]:
                    depth = base_y - float(scene.aabb(int(pi))[0][1])
                    if depth > 0.0:
                        below.append({"part": asset.parts[int(pi)].name, "depth": round(depth, 3)})
                below.sort(key=lambda e: (-e["depth"], e["part"]))
    rx0, ry0, rx1, ry1 = ref.bbox
    model_h_px, ref_h_px = mbot - top, ry1 - ry0

    # Scale: model pixels per reference pixel.
    if spec.height is None:
        k = model_h_px / ref_h_px
        ratio = 1.0
        mode = "fit"
    else:
        # In an elevated view the picture's height includes the top; correct like the model's projection.
        ratio = (sil_bottom - top) * mpp / float(hi[1] - lo[1]) if _elevated(view) and hi[1] > lo[1] else 1.0
        k = spec.height * ratio / ref_h_px / mpp
        mode = "height"
    mcx = (left + right) / 2.0
    rcx, rbot = (rx0 + rx1) / 2.0, float(ry1)
    mapped = (mcx + (rx0 - rcx) * k, mbot + (ry0 - rbot) * k, mcx + (rx1 - rcx) * k, mbot)
    shift_max = max(2, math.ceil(SHIFT_SEARCH * max(right - left, mapped[2] - mapped[0])))
    grid_left = min(0, math.floor(mapped[0]) - shift_max - 2)
    grid_right = max(W, math.ceil(mapped[2]) + shift_max + 2)
    grid_top = min(0, math.floor(mapped[1]) - 2)
    grid_bottom = max(H, math.ceil(mapped[3]) + 2)
    pl, pt = -grid_left, -grid_top
    Wg, Hg = grid_right - grid_left, grid_bottom - grid_top

    mc = np.zeros((Hg, Wg), dtype=np.float32)
    part = np.full((Hg, Wg), -1, dtype=np.int32)
    lam = np.empty((Hg, Wg, 3), dtype=np.float32)
    lam[:] = asset.style.background
    mc[pt:pt + H, pl:pl + W] = mcov
    part[pt:pt + H, pl:pl + W] = vb.part
    surf = np.full((Hg, Wg), -1, dtype=np.int32)  # the part that colours each pixel (a paint, or the solid)
    surf[pt:pt + H, pl:pl + W] = scene.surface_ids(vb)
    lam[pt:pt + H, pl:pl + W] = shade_lambert(asset, vb, cam=cam)
    model = mc >= 0.5
    if not model.any():
        model = mc > 0
    if (model & (part < 0)).any():  # coverage >= 0.5 with a missed center sample: borrow the nearest label
        part = np.where(model, _nearest_labels(part, model), part)
        surf = np.where(model, _nearest_labels(surf, model), surf)

    # Reference coverage on the grid, with room for the horizontal shift search.
    sat = _sat(ref.mask.astype(np.float32))
    y_edges = rbot + (np.arange(Hg + 1) - pt - mbot) / k
    ext_cols = np.arange(-shift_max, Wg + shift_max + 1)
    cover_ext = _box_cover(sat, y_edges, rcx + (ext_cols - pl - mcx) / k)

    def score(cov) -> float:
        return _soft_fit(cov, mc)[0]

    def cover_at(s: float) -> np.ndarray:  # reference moved right by s grid px
        return _box_cover(sat, y_edges, rcx + (np.arange(Wg + 1) - s - pl - mcx) / k)

    best = None
    for s in sorted(range(-shift_max, shift_max + 1), key=lambda v: (abs(v), v)):
        iou = score(cover_ext[:, shift_max - s: shift_max - s + Wg])
        if best is None or iou > best[0] + 1e-12:
            best = (iou, s)
    for frac in (-0.5, -0.25, 0.25, 0.5):  # sub-pixel refinement around the best whole pixel
        s = best[1] + frac
        iou = score(cover_at(s))
        if iou > best[0] + 1e-12:
            best = (iou, s)
    shift = best[1]
    cover = cover_at(shift)
    x_edges = rcx + (np.arange(Wg + 1) - shift - pl - mcx) / k
    rgb, inside = _box_color(_sat(ref.rgb), y_edges, x_edges)
    rgb = rgb / 255.0
    if ref.alpha is not None:  # transparent pictures: show the object over the background
        rgb = np.where((cover > 0)[..., None], rgb, np.asarray(asset.style.background))
    rgb = np.where((inside > 0.5)[..., None], rgb, np.asarray(asset.style.background))
    refm = cover >= 0.5

    # Crop to both silhouettes plus a margin.
    ux0, uy0, ux1, uy1 = _bbox(refm | model)
    margin = max(4, int(0.06 * max(ux1 - ux0, uy1 - uy0)))
    cx0, cy0 = max(0, ux0 - margin), max(0, uy0 - margin)
    cx1, cy1 = min(Wg, ux1 + margin), min(Hg, uy1 + margin)
    crop = (slice(cy0, cy1), slice(cx0, cx1))
    refm, model, part, lam, rgb, cover, mc = (refm[crop], model[crop], part[crop], lam[crop], rgb[crop], cover[crop],
                                              mc[crop])
    surf = surf[crop]
    pl -= cx0
    pt -= cy0
    # Grid edge coordinate -> asset frame meters (h along camera.right, v along camera.up).
    c_h = float(np.dot(cam.center, cam.right))

    def h_of(gx):
        return c_h + (np.asarray(gx, dtype=np.float64) - pl - W / 2.0) * mpp

    def v_of(gy):
        return c_v + (H / 2.0 - (np.asarray(gy, dtype=np.float64) - pt)) * mpp

    hax, vax = _axes(cam)
    h_name = hax[0] if hax[1] > 0 else f"-{hax[0]}"
    # Sizes for the SIZE line, free of the grid: the model's sub-pixel silhouette extent (the same at
    # every res; its exact AABB extent along a world axis when the two agree within the sub-sample step),
    # the reference's full-resolution mask bbox at the reference scale.
    sizes = []
    for vec, px in ((cam.right, right - left), (cam.up, sil_bottom - top)):
        exact = float(np.abs(vec) @ (hi - lo))
        aligned = float(np.max(np.abs(vec))) > 1.0 - 1e-9
        sizes.append(exact if aligned and abs(exact / mpp - px) <= 0.15 else px * mpp)
    sizes += [(rx1 - rx0) * k * mpp, (ry1 - ry0) * k * mpp]
    align = {"bottom": align_on, "base": round(c_v + (H / 2.0 - mbot) * mpp, 2), "below_ground": below}
    report = _match_report(asset, index, spec, view, ref, refm, model, mc, part, lam, rgb, cover, mpp, h_of, v_of, hax,
                           vax, h_name, mode, k, ratio, shift, res, sizes, align)
    pic_rows = ((y_edges[:-1] + y_edges[1:]) / 2.0)[crop[0]]
    pic_cols = ((x_edges[:-1] + x_edges[1:]) / 2.0)[crop[1]]
    return _Match(report, rgb, refm, model, surf, lam, cover, mc, cam, (pt, pl), ref, pic_rows, pic_cols)


def _soft_fit(cover: np.ndarray, mc: np.ndarray) -> tuple[float, float, float]:
    """(IoU, missing, extra) of two coverage images (0..1 per pixel).

    intersection = sum min(r, m), union = sum max(r, m). Where both edges cross a pixel from the same
    side the partial coverages nest, so this is the area IoU, not a count of pixels over 50%: thin
    wings and wires count with their real width at every resolution.
    """
    r_sum, m_sum = float(cover.sum()), float(mc.sum())
    inter = float(np.minimum(cover, mc).sum())
    iou = inter / max(1e-9, r_sum + m_sum - inter)
    return iou, (r_sum - inter) / max(1e-9, r_sum), (m_sum - inter) / max(1e-9, m_sum)


def opposite_view(view: str) -> str | None:
    """The view from the other side (front <-> back, left <-> right, orbit az <-> az + 180 at -el), else None."""
    named = {"front": "back", "back": "front", "left": "right", "right": "left"}
    if view in named:
        return named[view]
    orbit = parse_orbit(view) if view.startswith("orbit:") else None
    if orbit is None:
        return None
    az, el = orbit
    return orbit_view((az + 180.0) % 360.0, -el)


def _same_view(a: str, b: str) -> bool:
    if a == b:
        return True
    pa = parse_orbit(a) if a.startswith("orbit:") else None
    pb = parse_orbit(b) if b.startswith("orbit:") else None
    return pa is not None and pb is not None and abs((pa[0] - pb[0] + 180.0) % 360.0 - 180.0) < 1e-6 and abs(pa[1] - pb[1]) < 1e-6


def picture_agreement(asset, index: int, mpp: float) -> dict | None:
    """How far reference `index` agrees with the reference from the opposite side, mirrored; None if there is none.

    Orthographic silhouettes of any object seen from opposite sides are mirror images, so this is a
    ceiling for every model: by the triangle inequality of the Jaccard distance (1 - IoU), the two
    views' IoUs cannot average more than (1 + agreement) / 2. Both pictures stand on the ground at
    their given heights (or the same height when either has none), on a grid of `mpp` meters per pixel,
    the other one shifted sideways to its best soft IoU as `match` shifts each picture.
    """
    spec = _reference_spec(asset, index)
    other_view = opposite_view(spec.view)
    if other_view is None:
        return None
    j = next((k for k, r in enumerate(asset.references) if k != index and _same_view(r.view, other_view)), None)
    if j is None:
        return None
    pics = []
    for k in (index, j):
        s = asset.references[k]
        mask = None if str(s.mask) == "auto" else s.mask
        ref = _analyze(s.image, s.view, s.height, mask, asset.style, asset.root, None)
        x0, y0, x1, y1 = ref.bbox
        pics.append((ref, s.height, (x0 + x1) / 2.0, float(y1), float(y1 - y0), float(x1 - x0)))
    (ra, ha, cxa, ba, hpa, wpa), (rb, hb, cxb, bb, hpb, wpb) = pics
    if ha is None or hb is None:  # no metric height: the two pictures at one (the first one's) height
        ha = hb = 1.0
    sa, sb = ha / hpa, hb / hpb  # meters per picture pixel
    shift_max = 0.25 * max(wpa * sa, wpb * sb)
    half = max(wpa * sa, wpb * sb) / 2.0 + shift_max + 2 * mpp
    ncol = 2 * math.ceil(half / mpp)
    top = max(ha, hb) + 2 * mpp
    nrow = math.ceil(top / mpp)
    x_edges = (np.arange(ncol + 1) - ncol / 2) * mpp  # symmetric about x = 0: a column flip mirrors x -> -x
    y_edges = top - np.arange(nrow + 1) * mpp  # meters above the ground, top row first
    sat_a, sat_b = _sat(ra.mask.astype(np.float32)), _sat(rb.mask.astype(np.float32))
    cov_a = _box_cover(sat_a, ba - y_edges / sa, cxa + x_edges / sa)

    def cov_b(shift: float) -> np.ndarray:  # the other picture mirrored, then moved right by `shift` meters
        xb = -(x_edges[::-1] - shift)
        return _box_cover(sat_b, bb - y_edges / sb, cxb + xb / sb)[:, ::-1]

    best = None
    steps = math.ceil(shift_max / mpp)
    for s in sorted(range(-steps, steps + 1), key=lambda v: (abs(v), v)):
        iou = _soft_fit(cov_a, cov_b(s * mpp))[0]
        if best is None or iou > best[0] + 1e-12:
            best = (iou, s * mpp)
    for frac in (-0.5, -0.25, 0.25, 0.5):
        s = best[1] + frac * mpp
        iou = _soft_fit(cov_a, cov_b(s))[0]
        if iou > best[0] + 1e-12:
            best = (iou, s)
    iou, shift = best
    a, b = cov_a >= 0.5, cov_b(shift) >= 0.5
    inter = int((a & b).sum())
    thick = int((a & ~_dilate(b)).sum() + (b & ~_dilate(a)).sum())
    return {"other": j, "view": asset.references[j].view, "image": _label(asset.references[j].image, asset.root),
            "iou": round(float(iou), 4), "within_1px": round(inter / max(1, inter + thick), 4),
            "pair_mean_max": round((1.0 + float(iou)) / 2.0, 4), "shift": round(float(shift), 4)}


def _label(path, root) -> str:
    path = pathlib.Path(path)
    if root is not None:
        try:
            return path.resolve().relative_to(pathlib.Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _match_report(asset, index, spec, view, ref, R, M, mc, part, lam, rgb, cover, mpp, h_of, v_of, hax, vax, h_name,
                  mode, k, ratio, shift, res, sizes, align) -> dict:
    r_area = int(R.sum())
    iou, missing, extra = _soft_fit(cover, mc)
    rb, mb = _bbox(R), _bbox(M)
    r_size = ((rb[2] - rb[0]) * mpp, (rb[3] - rb[1]) * mpp)  # on the grid (profile tolerance)
    m_size, size_ref = (sizes[0], sizes[1]), (sizes[2], sizes[3])  # exact-ish sizes (SIZE line, aspect)

    # Width profile over the union's vertical extent. Edges are sub-pixel (coverage of the end columns).
    # A verdict must survive moving the band's top and bottom by `slack` rows: a near-horizontal or
    # sloped edge (a wing with dihedral) that crosses the band border 4 cm higher in the picture would
    # otherwise read "model too narrow by 1.8 m" although a 4 cm vertical move fixes it.
    uy0, uy1 = min(rb[1], mb[1]), max(rb[3], mb[3])
    tol = max(1.5 * mpp, 0.01 * r_size[1])
    slack = max(1, math.ceil(max(1.5 * mpp, BAND_SLACK * r_size[1]) / mpp - 1e-9))
    n_rows = R.shape[0]
    profile = []
    for b, (r0, r1) in enumerate(_band_rows(uy0, uy1)):
        e_r, e_m = _sub_extent(cover, r0, r1), _sub_extent(mc, r0, r1)
        entry = {"band": f"{b * 100 // BANDS}-{(b + 1) * 100 // BANDS}%", "top": round(float(v_of(r0)), 2),
                 "bottom": round(float(v_of(r1)), 2), "reference": None, "model": None, "diff": None, "verdict": ""}
        for key, e in (("reference", e_r), ("model", e_m)):
            if e:
                left, right = float(h_of(e[0])), float(h_of(e[1]))
                entry[key] = {"left": round(left, 2), "right": round(right, 2), "width": round(right - left, 2)}
        if e_r and e_m:
            dl = (e_m[0] - e_r[0]) * mpp  # > 0: model's left edge further right
            dr = (e_m[1] - e_r[1]) * mpp  # > 0: model's right edge further right
            entry["diff"] = {"left": round(-dl, 2), "right": round(dr, 2), "width": round(dr - dl, 2)}
            wide = (max(0, r0 - slack), min(n_rows, r1 + slack))
            mid = (r0 + r1) // 2
            narrow = (r0 + slack, r1 - slack) if r1 - r0 > 2 * slack else (mid, mid + 1)
            rw, rn = _sub_extent(cover, *wide), _sub_extent(cover, *narrow) or e_r
            mw, mn = _sub_extent(mc, *wide), _sub_extent(mc, *narrow) or e_m
            # Each edge ranges between its narrow-band and wide-band position; a difference counts only
            # where the two ranges do not overlap.
            rdl = _gap((mw[0], mn[0]), (rw[0], rn[0])) * mpp
            rdr = _gap((mn[1], mw[1]), (rn[1], rw[1])) * mpp
            dw = rdr - rdl
            if abs(dw) > tol:
                word = "too wide" if dw > 0 else "too narrow"
                entry["verdict"] = f"model {word} by {abs(dw):.2f} m (left {-rdl:+.2f}, right {rdr:+.2f})"
            elif abs(rdl) > tol and abs(rdr) > tol and rdl * rdr > 0:
                entry["verdict"] = f"model shifted {'right' if rdl > 0 else 'left'} by {abs(rdl + rdr) / 2:.2f} m"
            elif abs(dr - dl) > tol or (abs(dl) > tol and abs(dr) > tol and dl * dr > 0):
                entry["verdict"] = f"ok within {slack * mpp:.2f} m up/down (an edge runs along the band border)"
            else:
                entry["verdict"] = "ok"
        elif e_r:
            entry["verdict"] = "model empty here"
        elif e_m:
            entry["verdict"] = "reference empty here (model only)"
        profile.append(entry)

    # Regions of reference-only and model-only pixels.
    visible = [int(i) for i in np.unique(part[M])]
    names = {i: asset.parts[i].name for i in visible}
    part_px = {i: int((part == i).sum()) for i in visible}
    part_box = {i: _bbox(part == i) for i in visible}
    miss_mask, extra_mask = R & ~M, M & ~R
    regions = []
    near_map = None
    for kind, mk, other in (("missing", miss_mask, M), ("extra", extra_mask, R)):
        # The 1-px band along the other silhouette is the half-pixel dilation (and resampling) fringe:
        # label without it, so a real region next to that edge does not merge with the whole fringe, then
        # give each kept region back its own band pixels.
        band = mk & _dilate(other)
        core = mk & ~band
        # Thin strokes (wires, rods) and thick areas are labelled apart, so a wire that touches a real gap
        # neither inflates it nor gets blamed on the part it runs past.
        thick = _open(core, (THIN_REGION_PX + 1) // 2)
        found = []
        for layer, thin in ((thick, False), (core & ~thick, True)):
            labels, sizes = label_components(layer)
            found += [(labels == j, thin) for j in np.flatnonzero(sizes >= max(1, MIN_REGION * r_area))]
        for sel, thin in found:
            sel |= _dilate(sel) & band
            x0, y0, x1, y1 = _bbox(sel)
            n = int(sel.sum())
            longest = max(x1 - x0, y1 - y0)
            thickness = n / longest
            if thickness < MIN_REGION_PX:  # edge noise: silhouettes may differ by about a pixel
                continue
            if kind == "missing":
                if near_map is None:
                    near_map = _nearest_labels(part, miss_mask)
                pi = _majority(near_map[sel])
            else:
                pi = _majority(part[sel])
            sliver = longest >= 4 * thickness
            name = names.get(pi, "?")
            if pi in part_box:
                px0, py0, px1, py1 = part_box[pi]
                side = _side(((x0 + x1) - (px0 + px1)) / max(1, px1 - px0), ((y0 + y1) - (py0 + py1)) / max(1, py1 - py0))
            else:
                side = "?"
            # Extent normal to that side: the most a part would have to grow or shrink there.
            depth = int(sel.sum(axis=0).max() if side in ("top", "bottom") else sel.sum(axis=1).max())
            width = _stroke_width(sel) * mpp
            if thin and kind == "missing":
                # Wires, rods and rigging: thin strokes say nothing about the size of the part they run past.
                hint = (f"thin detail missing: strokes ~{width:.2f} m wide (wire, rod, pole?); add a thin part "
                        "(>= 0.05 m) if it matters, no part needs resizing")
            elif thin:
                hint = f"thin model detail (~{width:.2f} m wide) not in the picture: `{name}`, check it is wanted"
            elif kind == "missing":
                hint = (f"`{name}` may be too small on the {side} by ~{depth * mpp:.2f} m" if sliver
                        else "a part is missing here (the picture has something the model lacks)")
            elif pi in part_px and n >= 0.6 * part_px[pi]:
                hint = f"`{name}` is not in the reference (remove or move it?)"
            elif sliver:
                hint = f"`{name}` is too big on the {side} by ~{depth * mpp:.2f} m"
            else:
                hint = f"`{name}` sticks out here (too big or misplaced)"
            regions.append({
                "kind": kind,
                "area": round(n * mpp * mpp, 2),
                "share": round(n / max(1, r_area), 3),
                "size": [round((x1 - x0) * mpp, 2), round((y1 - y0) * mpp, 2)],
                "thickness": round(thickness * mpp, 2),  # mean width across the region's long side
                "depth": round(depth * mpp, 2),  # largest extent normal to the nearest part's side
                "thin": bool(thin),  # made of strokes <= THIN_REGION_PX grid px wide (wires, rods)
                "width": round(width, 3),  # mean stroke width: 2 x area / perimeter
                "bbox": {hax[0]: _axis_range(float(h_of(x0)), float(h_of(x1)), hax),
                         vax[0]: _axis_range(float(v_of(y1)), float(v_of(y0)), vax)},
                "near": name,
                "hint": hint,
            })
    regions.sort(key=lambda e: (e["thin"], -e["area"], e["kind"]))  # thin details last: fix the shapes first

    # Per-part colours: reference median vs flat Lambert render, over the part's interior pixels.
    lit = ref.report["lighting"]["mode"] == "lit"
    parts, measured = [], []
    for i in sorted(visible, key=lambda i: (-part_px[i], i)):
        p = asset.parts[i]
        pix = part == i
        core = _erode(pix)
        if core.sum() < 3:
            core = pix
        valid = core & (cover >= 0.999)
        base = np.array(asset.style.material(p.material).color, dtype=np.float64)
        entry = {"part": p.name, "material": p.material, "px": part_px[i], "base": _hex(base), "rendered": None,
                 "reference": None, "delta_e": None, "delta_e_base": None, "suggested": None}
        use = valid if valid.sum() >= max(3, 0.2 * core.sum()) else core
        rendered = np.median(lam[use], axis=0)
        entry["rendered"] = _hex(rendered)
        if use is valid:
            refc = np.median(rgb[valid], axis=0)
            measured.append((entry, base, rendered, refc))
            entry.update(reference=_hex(refc), delta_e=round(delta_e(rendered, refc), 1),
                         delta_e_base=round(delta_e(base, refc), 1))
        parts.append(entry)
    # Suggested base colours. Flat paint: the picture's colour is the paint. Lit: base x reference / rendered,
    # divided by the picture's overall light (per channel, the median of that ratio over the parts), so the
    # picture's exposure, light colour and direction cancel out and only parts that differ from the rest
    # change (a dim render made every cream wing "#f2ffff" before).
    exposure = None
    if lit and measured:
        ratios = np.array([np.where(rd > 1e-3, rc / np.maximum(rd, 1e-3), 1.0) for _, _, rd, rc in measured])
        exposure = np.maximum(np.median(ratios, axis=0), 1e-3)
    for entry, base, rendered, refc in measured:
        if exposure is None:
            suggested = refc
        else:
            suggested = np.where(rendered > 1e-3, base * refc / np.maximum(rendered, 1e-3) / exposure, refc)
        entry["suggested"] = _hex(np.clip(suggested, 0.0, 1.0))

    # Diff grid over both silhouettes.
    ux0, uy0b, ux1, uy1b = _bbox(R | M)
    h_lo, h_hi = float(h_of(ux0)), float(h_of(ux1))
    v_lo, v_hi = float(v_of(uy1b)), float(v_of(uy0b))
    if -1.01 * mpp <= v_lo < 0:  # the half-pixel silhouette dilation below the ground: no extra row for it
        v_lo = 0.0
    cell, gx0, gy0, cols, rows = _grid_layout(h_lo, h_hi, v_lo, v_hi)
    xe = (gx0 + cell * np.arange(cols + 1) - h_of(0)) / mpp
    ye = (v_of(0) - (gy0 - cell * np.arange(rows + 1))) / mpp
    rc = _box_cover(_sat(R.astype(np.float32)), ye, xe) >= 0.5
    mc = _box_cover(_sat(M.astype(np.float32)), ye, xe) >= 0.5
    chars = np.where(rc & mc, "#", np.where(mc, "+", np.where(rc, "-", ".")))
    grid = _grid_dict(cell, gx0, gy0, ["".join(row) for row in chars], "m",
                      {"#": "both", "+": "model only", "-": "reference only", ".": "neither"}, (h_name, vax[0]))

    ref_rep = ref.report
    model_height = None
    if mode == "fit":
        model_height = round(m_size[1], 2)
    unknowns = _unknowns(ref_rep, view, model_height) if mode == "fit" else ref_rep["unknowns"]
    scale = {
        "mode": mode,
        "note": (f"reference fitted to the model's height ({m_size[1]:.2f} m in this view)" if mode == "fit"
                 else f"reference height {spec.height:.2f} m given"
                 + (f" (picture height corrected x{ratio:.2f} for the elevated view)" if ratio != 1.0 else "")),
        "reference_mpp": round(k * mpp, 5),
        "grid_mpp": round(mpp, 5),
    }
    return {
        "asset": asset.name,
        "reference": {"index": index, "image": ref.label, "view": view, "height": spec.height},
        "res": res,
        "scale": scale,
        "size": {"reference": [round(size_ref[0], 2), round(size_ref[1], 2)], "model": [round(m_size[0], 2), round(m_size[1], 2)]},
        "align": {"shift": round(shift * mpp, 2), **align},
        "iou": round(iou, 3),
        "missing": round(missing, 3),
        "extra": round(extra, 3),
        "aspect": {"reference": round(size_ref[1] / max(size_ref[0], 1e-9), 3), "model": round(m_size[1] / max(m_size[0], 1e-9), 3)},
        "axes": [h_name, vax[0]],
        "width_profile": profile,
        "regions": regions[:12],
        "parts": parts,
        "grid": grid,
        "mask": {"source": ref_rep["mask"]["source"], "confidence": ref_rep["mask"]["confidence"]},
        "lighting": {"mode": ref_rep["lighting"]["mode"], "source": ref_rep["lighting"]["source"],
                     "exposure": None if exposure is None else [round(float(v), 3) for v in exposure]},
        "unknowns": unknowns,
        "images": {},
        "model_panel": None,
    }


def match_asset(asset, index: int = 0, res: int = 256, out_dir=None, *, view: str | None = None,
                ceiling: bool = True) -> dict:
    """Compare the model with its reference picture `index` (SPEC 12.2).

    Renders the model from the reference view on the model camera's pixel grid, scales the picture
    to the model's height (or uses the given `height`), aligns the bottom edges and the horizontal
    position (bbox centers, then the shift with the best IoU within a quarter of the width, to a
    quarter pixel) and reports IoU, missing/extra shares and regions, the width profile, per-part
    colours and a diff grid. Silhouettes from march() are dilated by up to half a pixel per edge,
    so widths carry about +-1 grid px and regions thinner than 1.5 px are ignored as edge noise.
    With `out_dir`, writes `match_<index>.png` there. `view` overrides the reference's view (e.g.
    the best one from `fit_view`). With `ceiling` and a reference from the opposite side, `ceiling`
    holds `picture_agreement`: how far the two pictures agree, the most any model can reach.
    """
    if isinstance(res, bool) or not isinstance(res, (int, np.integer)) or not 16 <= res <= 4096:
        raise SightlineError(f"res must be an integer from 16 to 4096, got {res!r}",
                             hint="res is the pixel count of the model render's long side; the default is 256")
    res = int(res)
    m = _match(asset, index, res, view)
    m.report["ceiling"] = None
    if ceiling and (view is None or view == _reference_spec(asset, index).view):
        m.report["ceiling"] = picture_agreement(asset, index, m.camera.mpp)
    if out_dir is not None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"match_{index}.png"
        m.report["model_panel"] = _write_match_png(asset, m, path, res)
        m.report["images"] = {"match": path.as_posix()}
    return m.report


def format_match(report: dict) -> str:
    """Stable text form of `match_asset`."""
    r = report
    h_name, v_name = r["axes"]
    ref = r["reference"]
    lines = [f"MATCH  {r['asset']}  reference[{ref['index']}] {ref['image']}  view {ref['view']}"]
    sc = r["scale"]
    lines.append(f"SCALE   {sc['note']}; grid 1 px = {sc['grid_mpp']:.4f} m")
    rs, ms = r["size"]["reference"], r["size"]["model"]
    size_line = f"SIZE    reference {rs[1]:.2f} m tall x {rs[0]:.2f} m wide, model {ms[1]:.2f} x {ms[0]:.2f} m (in this view)"
    if sc["mode"] == "height":
        dh, dw = ms[1] - rs[1], ms[0] - rs[0]
        words = [f"{(big if d > 0 else small)} by {abs(d):.2f} m" if abs(d) >= 0.005 else f"same {what}"
                 for d, what, big, small in ((dh, "height", "taller", "shorter"), (dw, "width", "wider", "narrower"))]
        size_line += f" -> model {', '.join(words)}"
    lines.append(size_line)
    lines.append(f"FIT     IoU {r['iou']:.2f}   missing {_pct(r['missing'])} (reference not covered)   "
                 f"extra {_pct(r['extra'])} (model outside the reference)")
    c = r.get("ceiling")
    if c:
        lines.append(f"CEILING the {c['view']} picture (reference[{c['other']}]) mirrored agrees with this one on IoU "
                     f"{c['iou']:.2f} ({_pct(c['within_1px'])} within 1 px): no model can average more than "
                     f"{c['pair_mean_max']:.2f} over the two views")
    lines.append(f"ASPECT height/width reference {r['aspect']['reference']:.2f} vs model {r['aspect']['model']:.2f}")
    al = r["align"]
    on = {"ground": "on the ground line (y = 0.00)", "bottom": f"on the model's bottom (y = {al.get('base', 0):.2f})"}
    bottom = on.get(al.get("bottom"), f"on the model silhouette's bottom ({v_name} {al.get('base', 0):.2f})")
    lines.append(f"ALIGN   picture bottom {bottom}; horizontal shift {al['shift']:+.2f} m from the centers (best IoU)")
    if al.get("below_ground"):
        deep = ", ".join(f"{e['part']} {e['depth']:.2f} m" for e in al["below_ground"][:4])
        lines.append(f"        ! below the ground: {deep} (not used to align; `sightline inspect` GROUND)")
    if r["mask"]["confidence"] != "high":
        lines.append(f"        ! reference outline confidence {r['mask']['confidence']}: trust only the large shapes")
    g = r["grid"]
    lines.append("")
    lines.append(f"DIFF GRID  cell {g['cell']:.2f} m  {g['cols']} x {g['rows']}  (# both, + model only, - reference only, . neither)")
    lines += format_grid(g)
    lines.append("")
    lines.append(f"WIDTH PROFILE  ({BANDS} bands from the top; {h_name} in m; diff = model - reference)")
    lines.append(f"  {'band':<8}{v_name + ' m':>13}   {'reference':>19}   {'model':>19}   verdict")
    for e in r["width_profile"]:
        yr = f"{e['top']:.2f}..{e['bottom']:.2f}"
        cells = []
        for key in ("reference", "model"):
            v = e[key]
            cells.append(f"{v['left']:+.2f}..{v['right']:+.2f} {v['width']:5.2f}" if v else "-")
        lines.append(f"  {e['band']:<8}{yr:>13}   {cells[0]:>19}   {cells[1]:>19}   {e['verdict']}")
    lines.append("")
    lines.append(f"REGIONS  (>= {MIN_REGION:.0%} of the reference area, largest first)")
    if not r["regions"]:
        lines.append("  none")
    for e in r["regions"]:
        at = ", ".join(f"{axis} {lo:.2f}..{hi:.2f}" for axis, (lo, hi) in e["bbox"].items())
        lines.append(f"  {e['kind']:<7} {e['size'][0]:.2f} x {e['size'][1]:.2f} m ({_pct(e['share'])} of ref) at "
                     f"{at}, next to `{e['near']}` -> {e['hint']}")
    lines.append("")
    lines.append("PARTS  (median colours over each part; rendered = base x flat Lambert from the style light)")
    pw = max([len("part")] + [len(e["part"]) for e in r["parts"]]) + 2  # long part names keep their own column
    mw = max([len("material")] + [len(str(e["material"])) for e in r["parts"]]) + 2
    lines.append(f"  {'part':<{pw}}{'material':<{mw}}{'base':<9}{'rendered':<10}{'reference':<11}{'dE':>5}{'dE base':>9}"
                 "  suggested base")
    for e in r["parts"]:
        de = f"{e['delta_e']:.1f}" if e["delta_e"] is not None else "-"
        deb = f"{e['delta_e_base']:.1f}" if e["delta_e_base"] is not None else "-"
        lines.append(f"  {e['part']:<{pw}}{str(e['material']):<{mw}}{e['base']:<9}{e['rendered']:<10}"
                     f"{(e['reference'] or '-'):<11}{de:>5}{deb:>9}  {e['suggested'] or '- (not in the reference)'}")
    lg = r.get("lighting") or {"mode": "flat", "source": "auto", "exposure": None}
    how = "given" if lg["source"] == "given" else "guessed; set reference lighting: lit|flat if wrong"
    if lg["mode"] == "lit":
        ex = " ".join(f"{v:.2f}" for v in lg["exposure"]) if lg["exposure"] else "-"
        lines.append(f"  (picture lit ({how}): suggested = base x reference / rendered / the picture's overall light "
                     f"[{ex}],")
        lines.append("   so only parts that differ from the rest change)")
    else:
        lines.append(f"  (picture flat paint ({how}): suggested = the reference colour)")
    lines.append("")
    lines += _format_unknowns(r["unknowns"])
    if r["images"]:
        lines.append("")
        lines.append(f"IMAGES  {r['images']['match']}  (reference | model | overlay grey both, red model only, blue reference only | parts)")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- match image


def _font(size: int = 12):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _model_panel(asset, view: str, res: int, lam: np.ndarray, part: np.ndarray) -> tuple[Image.Image, str]:
    """render.py's view if available, else the flat Lambert image with part outlines."""
    try:
        render = importlib.import_module(f"{__package__}.render")  # optional, lazy
        img = render.render_views(asset, views=[view], res=res, scale_figure=False, light="view")[view]
        return img.convert("RGB"), "render.py"
    except (ImportError, AttributeError, TypeError, KeyError, SightlineError):
        pass
    rgb = lam.copy()
    edge = np.zeros(part.shape, dtype=bool)
    edge[:, 1:] |= part[:, 1:] != part[:, :-1]
    edge[1:, :] |= part[1:, :] != part[:-1, :]
    rgb[edge] = asset.style.outline_color
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)), "lambert"


def _fit_into(img: Image.Image, size: tuple[int, int], bg=(255, 255, 255)) -> Image.Image:
    w, h = size
    s = min(w / img.width, h / img.height)
    resized = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                         Image.NEAREST if s >= 1 else Image.BOX)
    out = Image.new("RGB", size, bg)
    out.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
    return out


def _write_match_png(asset, m: _Match, path: pathlib.Path, res: int) -> str:
    r = m.report
    hg, wg = m.ref.shape
    up = 2 if max(hg, wg) < 384 else 1
    size = (wg * up, hg * up)

    def big(arr01):
        return Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).resize(size, Image.NEAREST)

    reference = big(m.ref_rgb)
    model_img, source = _model_panel(asset, r["reference"]["view"], res, m.lambert, m.part)
    model_img = model_img.resize(size, Image.NEAREST) if model_img.size == (wg, hg) else _fit_into(model_img, size)
    overlay = np.full(m.ref.shape + (3,), 0.97, dtype=np.float32)
    overlay[m.ref & m.model] = (0.6, 0.6, 0.6)
    overlay[m.model & ~m.ref] = (0.86, 0.22, 0.2)
    overlay[m.ref & ~m.model] = (0.2, 0.36, 0.85)
    overlay = big(overlay)
    # Part map over the reference: flat part colours at 55% over the aligned picture, outlines, names.
    tint = m.ref_rgb.copy()
    for i in np.unique(m.part[m.model]):
        sel = m.part == i
        tint[sel] = 0.45 * tint[sel] + 0.55 * np.array(_part_color(int(i))) / 255.0
    edge = np.zeros(m.part.shape, dtype=bool)
    edge[:, 1:] |= m.part[:, 1:] != m.part[:, :-1]
    edge[1:, :] |= m.part[1:, :] != m.part[:-1, :]
    tint[edge] = (0.1, 0.1, 0.1)
    parts_img = big(tint)
    draw = ImageDraw.Draw(parts_img)
    font = _font(11)
    boxes: list[tuple[int, int, int, int]] = []
    for i in sorted(np.unique(m.part[m.model]), key=lambda i: -int((m.part == i).sum())):
        ys, xs = np.nonzero(m.part == i)
        if len(ys) < 12:
            continue
        name = asset.parts[int(i)].name
        tw = int(draw.textlength(name, font=font))
        x = int(np.clip(xs.mean() * up - tw / 2, 1, size[0] - tw - 1))
        y = int(np.clip(ys.mean() * up - 6, 1, size[1] - 13))
        for _ in range(20):  # nudge down until it does not overlap an earlier label
            box = (x - 1, y - 1, x + tw + 1, y + 13)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                break
            y = min(size[1] - 13, y + 13)
        boxes.append(box)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x, y), name, fill=(20, 20, 20), font=font)

    panels = [(reference, [f"reference {pathlib.Path(r['reference']['image']).name}", "reference"]),
              (model_img, [f"model ({source})", "model"]),
              (overlay, [f"IoU {r['iou']:.2f}: grey both, red model, blue ref", f"IoU {r['iou']:.2f} grey/red/blue",
                         f"IoU {r['iou']:.2f}"]),
              (parts_img, ["parts over reference", "parts"])]
    gap, head, cap = 8, 22, 18
    slot = max(size[0], 170)  # room for the captions under narrow panels
    canvas = Image.new("RGB", (4 * slot + 5 * gap, head + size[1] + cap + gap), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    small, title = _font(11), _font(13)
    headers = [f"{r['asset']} - match reference[{r['reference']['index']}] - view {r['reference']['view']} - "
               f"IoU {r['iou']:.2f}, missing {_pct(r['missing'])}, extra {_pct(r['extra'])} - "
               f"1 px = {r['scale']['grid_mpp'] / up:.3f} m",
               f"{r['asset']} - IoU {r['iou']:.2f}, missing {_pct(r['missing'])}, extra {_pct(r['extra'])}"]
    d.text((gap, 4), next((h for h in headers if d.textlength(h, font=title) <= canvas.width - 2 * gap), headers[-1]),
           fill=(20, 20, 20), font=title)
    for n, (img, captions) in enumerate(panels):
        x = gap + n * (slot + gap) + (slot - size[0]) // 2
        canvas.paste(img, (x, head))
        d.rectangle((x - 1, head - 1, x + size[0], head + size[1]), outline=(200, 200, 200))
        caption = next((c for c in captions if d.textlength(c, font=small) <= slot), captions[-1])
        d.text((gap + n * (slot + gap), head + size[1] + 3), caption, fill=(40, 40, 40), font=small)
    canvas.save(path)
    return source


# ----------------------------------------------------------------------------- view fitting


def _normalize(mask: np.ndarray, size: int) -> np.ndarray:
    """Crop to the bbox and resample so the longer side is `size` px (aspect kept)."""
    x0, y0, x1, y1 = _bbox(mask)
    bw, bh = x1 - x0, y1 - y0
    s = size / max(bw, bh)
    ow, oh = max(1, round(bw * s)), max(1, round(bh * s))
    sat = _sat(mask[y0:y1, x0:x1].astype(np.float32))
    return _box_cover(sat, np.linspace(0, bh, oh + 1), np.linspace(0, bw, ow + 1)) >= 0.5


def _iou_aligned(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two silhouettes placed bottom-aligned and horizontally centered on one canvas."""
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    ca = np.zeros((h, w), dtype=bool)
    cb = np.zeros((h, w), dtype=bool)
    ca[h - a.shape[0]:, (w - a.shape[1]) // 2:(w - a.shape[1]) // 2 + a.shape[1]] = a
    cb[h - b.shape[0]:, (w - b.shape[1]) // 2:(w - b.shape[1]) // 2 + b.shape[1]] = b
    return float((ca & cb).sum() / max(1, (ca | cb).sum()))


def fit_view(asset, index: int = 0, res: int = FIT_RES, top: int = 5) -> list[dict]:
    """Guess the reference's camera angle: model silhouettes at `res` px for azimuth 0..345 step 15
    and elevation 0/15/30/45/60 plus `top`. Both silhouettes are cropped to their bboxes and scaled
    so the longer side is `res` px (aspect kept), then compared bottom-aligned and centered.
    Returns the `top` best as [{view, azimuth, elevation, iou, alias}], best first.

    Azimuth a and -a give mirror images of a left-right symmetric object, so check the picture.
    """
    spec = _reference_spec(asset, index)
    mask = None if str(spec.mask) == "auto" else spec.mask
    ref = _analyze(spec.image, "front", None, mask, None, asset.root)
    target = _normalize(ref.mask, res)
    scene = asset.scene()
    lo, hi = scene.bounds(include_helpers=False)
    candidates = [(az, el) for el in FIT_ELEVATIONS for az in FIT_AZIMUTHS] + [(None, 90)]
    results = []
    for n, (az, el) in enumerate(candidates):
        view = "top" if az is None else orbit_view(az, el)
        cam = make_cameras([view], lo, hi, res=res)[view]
        vb = march(scene, cam)
        iou = _iou_aligned(target, _normalize(vb.hit, res)) if vb.hit.any() else 0.0
        results.append((-round(iou, 6), n, {"view": view, "azimuth": az, "elevation": el, "iou": round(iou, 3),
                                             "alias": "top" if az is None else _AXIS_ALIASES.get((az, el))}))
    results.sort(key=lambda t: (t[0], t[1]))
    return [r[2] for r in results[:top]]


def format_fit(results: list[dict]) -> str:
    """Text form of `fit_view`."""
    lines = ["FIT VIEW  (silhouette IoU after normalizing both to their bounding boxes)"]
    for r in results:
        alias = f" = {r['alias']}" if r["alias"] and r["alias"] != r["view"] else ""
        angle = "top" if r["azimuth"] is None else f"azimuth {r['azimuth']:>3} deg, elevation {r['elevation']:>2} deg"
        lines.append(f"  {r['view']:<14}{alias:<9} {angle:<34} IoU {r['iou']:.2f}")
    lines.append("  note: azimuth a and -a show mirror images of a left-right symmetric object; check the picture")
    lines.append("  use it: reference view {azimuth: A, elevation: E} in the asset, or `sightline match --fit-view`")
    return "\n".join(lines) + "\n"
