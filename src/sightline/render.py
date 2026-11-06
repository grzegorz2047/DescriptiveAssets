"""Rendering (SPEC 8): shaded views, part maps, level sprites and captioned PNGs.

Pipeline per view: march the scene (asset + scale figure) -> analytic hit of the ground slab
(y = 0 top face) -> base colors (material patterns in part-local coordinates, ground grid) ->
light (Lambert key + sky ambient x ambient occlusion + fill light, shadow rays via march.trace,
Blinn-Phong shine) -> outlines from the part and depth buffers -> presentation (nearest-neighbour
upscale of small images and a caption strip).

Pixels are single-sample and crisp on purpose: one image pixel = one camera pixel = `mpp` meters
before the presentation upscale, so pixel counts stay measurable.
"""

from __future__ import annotations

import colorsys
import functools
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import Camera, fixed_camera, make_cameras, parse_orbit, view_basis
from .errors import SightlineError
from .march import ViewBuffers, march, trace
from .model import F32, Material, Part, Scene, make_part

# Scale figure (SPEC 8): a neutral 1.8 m human beside the asset.
HUMAN_HEIGHT = 1.8
FIGURE_GAP_X = 0.6  # figure center x = asset max x + this
FIGURE_GAP_Z = 0.45  # figure center z = asset min z - this: never in front of the asset in front/right/iso
BODY_RADIUS = 0.17
BODY_TOP = 1.42  # shoulders; the gap up to the head reads as a neck
HEAD_RADIUS = 0.12
FIGURE_NAMES = ("scale_figure_body", "scale_figure_head")
HELPER_COLOR = (0.604, 0.616, 0.639)

# Look. Lighting is normalized so a fully lit, unoccluded upward face shows the material color.
SHADOW_STRENGTH = 0.6  # share of the direct light a cast shadow removes (soft but visible)
FILL_STRENGTH = 0.3  # fill light from the opposite side, so faces turned from the key light still read
SKY_TILT = 0.25  # ambient is brighter on upward faces: ambient * (1 - SKY_TILT + SKY_TILT * n.y)
EXPOSURE = 1.2  # asset surfaces only; the ground shows its style color when lit
SHOULDER = 0.8  # asset colors above this roll off smoothly toward 1 instead of clipping
SPEC_EXPONENT = 24.0
AO_SAMPLES = 4
AO_STRENGTH = 0.85
INNER_LINE_ALPHA = 0.5  # part boundaries without a depth jump are drawn half strength
SEAM_DEG = 8.0  # no inner line between same-material parts whose surfaces meet at less than this angle
DEPTH_JUMP_PX = 4.0  # outline where depth jumps by more than this many pixels (SPEC: 4 mpp)
UPSCALE_BELOW = 384  # images smaller than this on the long side are shown x2 (nearest neighbour)
PART_MAP_MIN = 640  # labeled part maps are upscaled to at least this long side
CAPTION_FONT = 13
LABEL_FONT = 12
TEXT_RGB = (34, 36, 40)
CAPTION_BG = (246, 247, 249)
CAPTION_RULE = (200, 204, 210)


# ----------------------------------------------------------------------------- fonts and text


@functools.lru_cache(maxsize=None)
def font(size: int):
    """Pillow's bundled TrueType font at `size` px (Pillow >= 10.1), else the old bitmap font."""
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, AttributeError, OSError, ImportError):
        return ImageFont.load_default()


def separator() -> str:
    """Caption separator: a middle dot with a TrueType font, '|' with the ASCII bitmap fallback."""
    return " · " if isinstance(font(CAPTION_FONT), ImageFont.FreeTypeFont) else " | "


def text_box(text: str, fnt) -> tuple[int, int, int, int]:
    """(left, top, width, height) of `text` drawn at (0, 0)."""
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=fnt)
    return l, t, r - l, b - t


def format_mpp(mpp: float) -> str:
    """0.023 for normal scales, two significant digits below 1 cm (0.0086, 0.00015)."""
    return f"{mpp:.3f}" if mpp >= 0.01 else f"{mpp:.2g}"


def caption_text(name: str, view: str, mpp: float, grid: float = 0.0) -> str:
    """`<asset> · <view> · 1 px = 0.023 m · grid 1 m` (grid omitted when 0)."""
    fields = [name, view, f"1 px = {format_mpp(mpp)} m"]
    if grid > 0:
        fields.append(f"grid {grid:g} m")
    return separator().join(fields)


def _wrap(fields: list[str], fnt, width: int, sep: str) -> list[str]:
    """Greedy line wrap of caption fields joined by `sep` to fit `width` px."""
    lines: list[str] = []
    for f in fields:
        if lines and text_box(lines[-1] + sep + f, fnt)[2] <= width:
            lines[-1] += sep + f
        else:
            lines.append(f)
    return lines


def upscale_factor(size: tuple[int, int], upscale: int | None = None) -> int:
    """Presentation upscale: `upscale` if given, else 2 when the long side is under 384 px."""
    if upscale is not None:
        if isinstance(upscale, bool) or not isinstance(upscale, (int, np.integer)) or not 1 <= upscale <= 8:
            raise SightlineError(f"upscale must be an integer 1..8, got {upscale!r}")
        return int(upscale)
    return 2 if max(size) < UPSCALE_BELOW else 1


def present(image: Image.Image, caption: str | list[str] | None, upscale: int = 1,
            background=(221, 225, 230)) -> Image.Image:
    """Upscale (nearest) and add a caption strip under the image. A caption string is split on the
    separator and wrapped; a list gives the lines. Narrow images are padded so the caption fits."""
    img = image.convert("RGB")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    if not caption:
        return img
    fnt = font(CAPTION_FONT)
    sep = separator()
    if isinstance(caption, str):
        fields = caption.split(sep)
        widest = max(text_box(f, fnt)[2] for f in fields)
        width = max(img.width, widest + 12)
        lines = _wrap(fields, fnt, width - 12, sep)
    else:
        lines = list(caption)
        width = max(img.width, max(text_box(s, fnt)[2] for s in lines) + 12)
    if width > img.width:  # pad narrow images with their own background color
        wide = Image.new("RGB", (width, img.height), tuple(int(c) for c in background))
        wide.paste(img, ((width - img.width) // 2, 0))
        img = wide
    line_h = CAPTION_FONT + 5
    strip_h = line_h * len(lines) + 8
    out = Image.new("RGB", (width, img.height + strip_h), CAPTION_BG)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.line([(0, img.height), (width - 1, img.height)], fill=CAPTION_RULE)
    for k, line in enumerate(lines):
        l, t, _, _ = text_box(line, fnt)
        draw.text((6 - l, img.height + 5 + k * line_h - t + 1), line, font=fnt, fill=TEXT_RGB)
    return out


# ----------------------------------------------------------------------------- small array helpers


def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx], `fill` outside the image."""
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[max(-dy, 0): h - max(dy, 0), max(-dx, 0): w - max(dx, 0)] = a[max(dy, 0): h - max(-dy, 0), max(dx, 0): w - max(-dx, 0)]
    return out


def _thicken(lines: np.ndarray, depth: np.ndarray, hit: np.ndarray, steps: int, jump: float) -> np.ndarray:
    """Grow 1-px lines by `steps` px into hit pixels on the same or a nearer surface (never across a
    depth jump to a farther one), so a width-w outline is w px wide and stays inside the silhouette."""
    out = lines.copy()
    for _ in range(max(0, steps)):
        add = np.zeros_like(out)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            add |= _shift(out, dy, dx, False) & (depth <= _shift(depth, dy, dx, np.inf) + jump)
        out |= add & hit
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        out &= _shift(mask, dy, dx, False)
    return out


def _masked_blur(values: np.ndarray, mask: np.ndarray, passes: int = 2) -> np.ndarray:
    """3x3 box blur of `values` over `mask` pixels only (normalized), repeated `passes` times."""
    v = np.where(mask, values, 0.0).astype(F32)
    m = mask.astype(F32)
    for _ in range(passes):
        acc = np.zeros_like(v)
        wgt = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += _shift(v, dy, dx, 0.0)
                wgt += _shift(m, dy, dx, 0.0)
        v = np.where(mask, acc / np.maximum(wgt, 1e-6), 0.0).astype(F32)
    return np.where(mask, v, values).astype(F32)


def _hash01(*ints: np.ndarray) -> np.ndarray:
    """Deterministic hash of integer cell coordinates -> float32 in [0, 1)."""
    mask = np.int64(0xFFFFFFFF)
    h = np.zeros(np.broadcast(*ints).shape, dtype=np.int64)
    for k, (v, prime) in enumerate(zip(ints, (73856093, 19349663, 83492791, 2654435761))):
        h ^= (np.asarray(v, dtype=np.int64) * prime + k * 0x9E37) & mask
    for _ in range(2):
        h = ((h ^ (h >> 16)) * 0x45D9F3B) & mask
    h ^= h >> 16
    return (h & 0xFFFFFF).astype(F32) / F32(0x1000000)


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- materials and patterns


def material_for(style, name: str | None) -> Material:
    """Style material; the `helper` grey exists even when a style lacks it."""
    if name in style.materials:
        return style.materials[name]
    if name == "helper":
        return Material("helper", HELPER_COLOR)
    return style.material(name)  # SightlineError with did_you_mean


def _across_axis(n: np.ndarray, along: int) -> np.ndarray:
    """Per pixel, the local axis across which stripes/planks alternate: of the two axes other than
    `along`, prefer y, then x, then z, unless the surface normal points mostly along it."""
    c1, c2 = [k for k in (1, 0, 2) if k != along]
    return np.where(np.abs(n[:, c1]) < 0.75, c1, c2)


def _pattern(mat: Material, p: np.ndarray, n: np.ndarray, mpp: float) -> tuple[np.ndarray, float, float]:
    """(t, feature size m, mean t): t in [0, 1] mixes the color toward pattern_color."""
    rows = np.arange(len(p))
    sa, sb = (float(v) for v in mat.scale)
    axis = "xyz".index(mat.axis) if mat.axis in ("x", "y", "z") else 1
    if mat.pattern == "noise":
        cells = np.floor(p / sa).astype(np.int64)
        t = _hash01(cells[:, 0], cells[:, 1], cells[:, 2])
        return t * t * 1.4, sa, 0.47
    if mat.pattern == "stripes":
        c = p[rows, _across_axis(n, axis)]
        return (np.floor(c / sa).astype(np.int64) % 2).astype(F32), sa, 0.5
    if mat.pattern == "checker":
        k = np.floor(p[:, 0] / sa) + np.floor(p[:, 1] / sb) + np.floor(p[:, 2] / sa)
        return (k.astype(np.int64) % 2).astype(F32), min(sa, sb), 0.5
    if mat.pattern == "planks":
        width = sa
        c = p[rows, _across_axis(n, axis)] / width
        board = np.floor(c).astype(np.int64)
        seam = max(0.09, 0.9 * mpp / width)  # seam width as a share of the board width, >= ~1 px
        length = 8.0 * width  # boards are 8 widths long, butt joints staggered per board
        u = p[:, axis] / length + _hash01(board, np.int64(7))
        joint = (u - np.floor(u)) < seam * width / length
        tint = 0.22 * _hash01(board)
        t = np.where((c - board < seam) | joint, 1.0, tint).astype(F32)
        return t, width, 0.11 + seam
    if mat.pattern == "bricks":
        bw, bh = sa, sb
        top = np.abs(n[:, 1]) > 0.75
        row_c = np.where(top, p[:, 2], p[:, 1])
        col_c = np.where(top | (np.abs(n[:, 0]) < 0.75), p[:, 0], p[:, 2])
        r = row_c / bh
        row = np.floor(r).astype(np.int64)
        u = col_c / bw + 0.5 * (row & 1)
        col = np.floor(u).astype(np.int64)
        mortar = max(0.07 * min(bw, bh), 0.9 * mpp)
        joint = ((r - row) < mortar / bh) | ((u - col) < mortar / bw)
        t = np.where(joint, 1.0, 0.3 * _hash01(row, col, top.astype(np.int64))).astype(F32)
        share = mortar / bh + mortar / bw
        return t, bh, share + 0.15 * (1 - share)
    return np.zeros(len(p), dtype=F32), 1.0, 0.0


def surface_color(mat: Material, local: np.ndarray, n_local: np.ndarray, mpp: float) -> np.ndarray:
    """Unlit color (N, 3) of a material at part-local points with part-local normals.

    Patterns smaller than ~3 px fade toward their mean color instead of aliasing into moire.
    """
    base = np.asarray(mat.color, dtype=F32)
    out = np.broadcast_to(base, (len(local), 3)).astype(F32)
    if mat.pattern in (None, "solid") or len(local) == 0:
        return out
    pat = np.asarray(mat.pattern_color if mat.pattern_color is not None else base * 0.8, dtype=F32)
    p = local - n_local * F32(0.6 * mpp)  # just below the surface: faces on cell borders stay stable
    t, feature, mean = _pattern(mat, p.astype(np.float64), n_local, mpp)
    fade = float(np.clip((feature / mpp - 1.5) / 2.5, 0.0, 1.0))
    t = np.clip(mean + (t - mean) * fade, 0.0, 1.0).astype(F32)
    return out + (pat - base)[None, :] * t[:, None]


# ----------------------------------------------------------------------------- scale figure and ground


def figure_parts(x: float, z: float, material: str = "helper") -> list[Part]:
    """The 1.8 m reference human (capsule body + sphere head) standing at (x, 0, z), as helper parts."""
    body = make_part(FIGURE_NAMES[0], "capsule", {"radius": BODY_RADIUS, "height": BODY_TOP},
                     [x, BODY_TOP / 2, z], material=material, helper=True, source="scale figure")
    head = make_part(FIGURE_NAMES[1], "sphere", {"radius": HEAD_RADIUS},
                     [x, HUMAN_HEIGHT - HEAD_RADIUS, z], material=material, helper=True, source="scale figure")
    return [body, head]


def scale_figure(bounds_min, bounds_max, material: str = "helper") -> list[Part]:
    """Scale figure beside an asset with these bounds: center x = asset max x + 0.6 m (SPEC 8),
    center z = asset min z - 0.45 m, so it never stands in front of the asset in the front, right
    or iso views."""
    lo = np.asarray(bounds_min, dtype=np.float64)
    hi = np.asarray(bounds_max, dtype=np.float64)
    return figure_parts(float(hi[0]) + FIGURE_GAP_X, float(lo[2]) - FIGURE_GAP_Z, material)


@dataclass
class Ground:
    """A ground slab: top face at y = 0 over `rect`, `thickness` m deep, grid lines every `grid` m."""

    rect: tuple[float, float, float, float]  # x0, x1, z0, z1
    thickness: float
    grid: float
    color: tuple[float, float, float]


@dataclass
class Layout:
    scene: Scene  # asset scene, plus the scale figure when shown
    ground: Ground | None
    frame_min: np.ndarray  # box the cameras frame (asset + figure + ground)
    frame_max: np.ndarray
    size: float  # longest side of the asset + figure bounds (m)


def scene_layout(scene: Scene, style, ground_on: bool | None = None) -> Layout:
    """Ground slab and camera framing box for a scene (helpers included)."""
    lo, hi = scene.bounds(include_helpers=True)
    size = float(max(np.max(hi - lo), 1e-3))
    ground_on = style.ground_show if ground_on is None else bool(ground_on)
    ground = None
    frame_lo, frame_hi = lo.copy(), hi.copy()
    if ground_on:
        pad = max(0.25, 0.06 * size)
        thick = 0.02 * size
        x0, x1, z0, z1 = float(lo[0] - pad), float(hi[0] + pad), float(lo[2] - pad), float(hi[2] + pad)
        grow_x = max(0.0, (z1 - z0) / 2 - (x1 - x0)) / 2  # keep the tile no thinner than 1:2
        grow_z = max(0.0, (x1 - x0) / 2 - (z1 - z0)) / 2
        rect = (x0 - grow_x, x1 + grow_x, z0 - grow_z, z1 + grow_z)
        ground = Ground(rect, thick, float(style.ground_grid), tuple(style.ground_color))
        frame_lo = np.array([rect[0], min(lo[1], -thick), rect[2]])
        frame_hi = np.array([rect[1], max(hi[1], 0.0), rect[3]])
    return Layout(scene, ground, frame_lo, frame_hi, size)


def layout(asset, scale_figure_on: bool = True, ground_on: bool | None = None) -> Layout:
    """Scene (+ scale figure), ground slab and camera framing box shared by every view of one render."""
    scene = asset.scene()
    if not scene.solid_indices():
        raise SightlineError(f"asset '{asset.name}' has no solid parts to render",
                             hint="add at least one part without `op: cut`")
    if scale_figure_on:
        lo, hi = scene.bounds(include_helpers=False)
        scene = scene.with_parts(scale_figure(lo, hi))
    return scene_layout(scene, asset.style, ground_on)


def _ground_hit(origins: np.ndarray, fwd: np.ndarray, g: Ground):
    """Analytic ray / slab intersection: (t (N,) inf = miss, normal (N, 3), face id (N,) -1 = miss).

    Face id = 2 * axis + (1 if the face normal is positive); the top face (y = 0) is 3.
    """
    x0, x1, z0, z1 = g.rect
    lo = np.array([x0, -g.thickness, z0])
    hi = np.array([x1, 0.0, z1])
    d = np.where(np.abs(fwd) < 1e-12, 1e-12, fwd)
    o = origins.astype(np.float64)
    t1 = (lo - o) / d
    t2 = (hi - o) / d
    tmin = np.minimum(t1, t2)
    t_near = tmin.max(axis=1)
    t_far = np.maximum(t1, t2).min(axis=1)
    hit = t_far >= np.maximum(t_near, 0.0)
    axis = tmin.argmax(axis=1)
    positive = d[axis] < 0  # entering through the face whose normal points back at the camera
    normal = np.zeros((len(o), 3), dtype=F32)
    normal[np.arange(len(o)), axis] = np.where(positive, 1.0, -1.0)
    face = np.where(hit, 2 * axis + positive, -1)
    return np.where(hit, t_near, np.inf), normal, face


def _ground_colors(g: Ground, point: np.ndarray, face: np.ndarray) -> np.ndarray:
    """(H, W, 3) unlit ground colors with 1-px grid lines (darker every 5 m) and slab edges."""
    h, w = face.shape
    base = np.asarray(g.color, dtype=F32)
    valid = face >= 0
    out = np.where((face == 3)[..., None], base, base * F32(0.9)).astype(F32)
    if g.grid > 0:
        big = np.int64(1 << 40)
        ix = np.where(valid, np.floor(point[..., 0] / g.grid), 0).astype(np.int64)
        iz = np.where(valid, np.floor(point[..., 2] / g.grid), 0).astype(np.int64)
        minor = np.zeros((h, w), dtype=bool)
        major = np.zeros((h, w), dtype=bool)

        def is_major(k):
            v = k.astype(np.float64) * g.grid
            return np.abs(v / 5.0 - np.round(v / 5.0)) < 1e-6

        for dy, dx in ((0, 1), (1, 0)):
            same = valid & (_shift(face, dy, dx, -1) == face)
            for idx in (ix, iz):
                nb = _shift(idx, dy, dx, big)
                cross = same & (nb != idx)
                maj = cross & is_major(np.maximum(idx, nb))
                minor |= cross
                major |= maj | _shift(maj, -dy, -dx, False)  # major lines are 2 px
        out[minor] *= F32(0.82)
        out[major & valid] = base * F32(0.66)
    # crease between the top face and the slab sides
    crease = np.zeros((h, w), dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = _shift(face, dy, dx, -1)
        crease |= valid & (nb >= 0) & (nb != face) & (face != 3)
    out[crease] = base * F32(0.7)
    return out


# ----------------------------------------------------------------------------- one view


@dataclass(eq=False)
class ViewRender:
    """One rendered view before presentation (no caption, no upscale)."""

    view: str
    camera: Camera
    buffers: ViewBuffers  # geometry of asset + scale figure (part ids index scene.parts)
    rgb: np.ndarray  # (H, W, 3) uint8: asset, ground, background and outlines
    coverage: np.ndarray  # (H, W) bool: asset / figure pixels in front of the ground (outlines lie inside)
    ground: np.ndarray  # (H, W) bool: visible ground pixels
    part: np.ndarray  # (H, W) int32: visible part per pixel (-1 = ground or background)
    light: np.ndarray  # (H, W) float32: direct light visibility 0..1 (1 = not in shadow)
    grid: float  # ground grid spacing drawn (0 = none)

    @property
    def mpp(self) -> float:
        return self.camera.mpp

    def image(self) -> Image.Image:
        return Image.fromarray(self.rgb, "RGB")

    def rgba(self) -> Image.Image:
        """Asset (+ figure) on a transparent background; the ground is dropped."""
        alpha = np.where(self.coverage, 255, 0).astype(np.uint8)
        return Image.fromarray(np.dstack([self.rgb, alpha]), "RGBA")


def depth_slopes(normal: np.ndarray, cam: Camera, hit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth change (m) per pixel step along x and y that each pixel's own surface explains.

    A plane seen at a grazing angle changes depth by mpp * |n . right| / |n . forward| per pixel:
    below about 14 degrees that exceeds the 4-px depth-jump threshold, and every pixel of a wing seen
    edge-on became an outline pixel. Capped at 50 px per step (|n . forward| >= 0.02).
    """
    n = normal.reshape(-1, 3).astype(np.float64)
    facing = np.maximum(np.abs(n @ np.asarray(cam.forward, dtype=np.float64)), 0.02)
    sx = cam.mpp * np.abs(n @ np.asarray(cam.right, dtype=np.float64)) / facing
    sy = cam.mpp * np.abs(n @ np.asarray(cam.up, dtype=np.float64)) / facing
    shape = hit.shape
    return (np.where(hit, sx.reshape(shape), 0.0).astype(F32), np.where(hit, sy.reshape(shape), 0.0).astype(F32))


def _edges(part: np.ndarray, depth: np.ndarray, jump: float, slopes=None, seam=None) -> tuple[np.ndarray, np.ndarray]:
    """(strong, soft) 1-px edge masks on the nearer side of each discontinuity.

    strong: silhouette or depth jump > `jump`; soft: part boundary without a depth jump. `slopes`
    ((H, W) x and y depth change per pixel step from `depth_slopes`) raise the threshold by the
    smaller of the two pixels' slopes: a grazing plane is continuous, while the edge of a part in
    front of it still jumps (its own pixel has a small slope). `seam` = (surface id per part index,
    normals (H, W, 3)): no soft line between two parts of one surface id whose normals agree within
    SEAM_DEG (a construction seam on a continuous surface: blended members, coplanar panels).
    """
    hit = part >= 0
    strong = np.zeros(part.shape, dtype=bool)
    soft = np.zeros(part.shape, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        pn = _shift(part, dy, dx, -1)
        dn = _shift(depth, dy, dx, np.inf)
        limit = jump
        if slopes is not None:
            sl = slopes[0] if dx else slopes[1]
            limit = jump + np.minimum(sl, _shift(sl, dy, dx, 0.0))
        with np.errstate(invalid="ignore"):
            gap = np.abs(depth - dn)
        is_jump = ~(gap <= limit)  # inf (neighbour empty) counts as a jump
        strong |= hit & is_jump & (depth < dn)
        boundary = hit & (pn != part) & ~is_jump & (depth <= dn)
        if seam is not None and boundary.any():
            surface, normal = seam
            same = (surface[part] == surface[pn]) & (pn >= 0)
            same &= (normal * _shift(normal, dy, dx, 0.0)).sum(axis=-1) > np.cos(np.radians(SEAM_DEG))
            boundary &= ~same
        soft |= boundary
    return strong, soft & ~strong


def seam_surfaces(scene: Scene) -> np.ndarray:
    """Surface id per part index (+1 slot for -1): parts of one material share an id (helpers get their own),
    so a soft line between them is drawn only where the surface turns (see _edges)."""
    ids = np.full(len(scene.parts) + 1, -1, dtype=np.int64)
    names: dict[str, int] = {}
    for i, p in enumerate(scene.parts):
        key = f"helper:{i}" if p.helper else str(p.material)
        ids[i] = names.setdefault(key, len(names))
    return ids


def _ambient_occlusion(scene: Scene, pts: np.ndarray, nrm: np.ndarray, step: float,
                       plane: np.ndarray, ground: Ground | None) -> np.ndarray:
    """Cheap AO: 4 SDF samples along the normal; `plane` marks points that also see the ground."""
    occ = np.zeros(len(pts), dtype=F32)
    lo, hi = scene.bounds(include_helpers=True)
    reach = AO_SAMPLES * step
    near = np.all((pts >= lo - reach) & (pts <= hi + reach), axis=1)
    sel = np.flatnonzero(near)
    if sel.size == 0:
        return np.ones(len(pts), dtype=F32)
    p, n = pts[sel], nrm[sel]
    use_plane = plane[sel]
    if ground is not None:
        x0, x1, z0, z1 = ground.rect
        use_plane &= (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 2] >= z0) & (p[:, 2] <= z1)
    else:
        use_plane = np.zeros(len(sel), dtype=bool)
    total = 0.0
    for i in range(1, AO_SAMPLES + 1):
        h = i * step
        wgt = 0.5 ** (i - 1)
        total += wgt
        q = p + n * F32(h)
        d = scene.eval(q)[0]
        d = np.where(use_plane, np.minimum(d, q[:, 1]), d)
        occ[sel] += F32(wgt) * np.clip((h - d) / h, 0.0, 1.0)
    return np.clip(1.0 - AO_STRENGTH * occ / total, 0.0, 1.0).astype(F32)


def render_view(scene: Scene, style, cam: Camera, ground: Ground | None = None, outline: bool = True,
                shadows: bool | None = None, ao_step: float | None = None) -> ViewRender:
    """Shade one camera view of a scene (asset parts, optional helpers) with an optional ground slab."""
    vb = march(scene, cam)
    h, w = cam.height, cam.width
    n = h * w
    mpp = cam.mpp
    eps = 0.5 * mpp
    fwd = np.asarray(cam.forward, dtype=np.float64)
    origins, _ = cam.rays()
    a_t = vb.depth.reshape(n)
    a_part = vb.part.reshape(n)
    if ground is not None:
        g_t, g_normal, g_face = _ground_hit(origins, fwd, ground)
    else:
        g_t, g_normal, g_face = np.full(n, np.inf), np.zeros((n, 3), F32), np.full(n, -1)
    asset_vis = vb.hit.reshape(n) & (a_t <= g_t + eps)
    ground_vis = np.isfinite(g_t) & ~asset_vis
    face = np.where(ground_vis, g_face, -1)

    point = np.zeros((n, 3), dtype=F32)
    normal = np.zeros((n, 3), dtype=F32)
    base = np.zeros((n, 3), dtype=F32)
    shine = np.zeros(n, dtype=F32)
    ai = np.flatnonzero(asset_vis)
    point[ai] = vb.point.reshape(n, 3)[ai]
    normal[ai] = vb.normal.reshape(n, 3)[ai]
    local = vb.local.reshape(n, 3)
    for pi in np.unique(a_part[ai]):
        sel = ai[a_part[ai] == pi]
        part = scene.parts[pi]
        mat = material_for(style, part.material)
        n_local = (normal[sel].astype(np.float64) @ part.matrix).astype(F32)
        base[sel] = surface_color(mat, local[sel], n_local, mpp)
        shine[sel] = float(mat.shine)
    gi = np.flatnonzero(ground_vis)
    if gi.size:
        point[gi] = (origins[gi] + fwd * g_t[gi][:, None]).astype(F32)
        normal[gi] = g_normal[gi]
        gcol = _ground_colors(ground, point.reshape(h, w, 3), face.reshape(h, w)).reshape(n, 3)
        base[gi] = gcol[gi]

    # Light: key (with shadow rays), sky ambient x AO, fill from the other side, Blinn-Phong shine.
    to_light = -np.asarray(style.light_dir, dtype=np.float64)
    to_light /= max(np.linalg.norm(to_light), 1e-9)
    fill_dir = np.array([-to_light[0], 0.35, -to_light[2]])
    fill_dir /= max(np.linalg.norm(fill_dir), 1e-9)
    amb = float(style.ambient)
    norm = 1.0 / (amb + (1 - amb) * max(to_light[1], 0.0) + FILL_STRENGTH * fill_dir[1])
    lit_px = asset_vis | ground_vis
    ndl = np.clip(normal @ to_light.astype(F32), 0.0, None)
    visible = np.ones(n, dtype=F32)
    if style.shadows if shadows is None else shadows:
        cand = np.flatnonzero(lit_px & (ndl > 1e-4))
        if cand.size:
            o = point[cand] + normal[cand] * F32(2 * eps)
            dirs = np.broadcast_to(to_light.astype(F32), o.shape)
            t, _ = trace(scene, o, dirs, np.inf, eps)
            visible[cand] = np.where(np.isfinite(t), 0.0, 1.0)
        top = (face == 3).reshape(h, w)
        visible = _masked_blur(visible.reshape(h, w), top).reshape(n)  # soft edges on the ground
    ao = np.ones(n, dtype=F32)
    if ao_step and ao_step > 0:
        idx = np.flatnonzero(asset_vis | (face == 3))
        ao[idx] = _ambient_occlusion(scene, point[idx], normal[idx], ao_step, asset_vis[idx], ground)
    sky = amb * ao * (1.0 - SKY_TILT + SKY_TILT * np.clip(normal[:, 1], -1.0, 1.0))
    direct = (1 - amb) * ndl * (1.0 - SHADOW_STRENGTH * (1.0 - visible))
    fill = FILL_STRENGTH * np.clip(normal @ fill_dir.astype(F32), 0.0, None) * ao
    light = (sky + direct + fill) * norm * np.where(asset_vis, EXPOSURE, 1.0)
    rgb = base * light[:, None]
    if shine.any():
        half = to_light - fwd
        half /= max(np.linalg.norm(half), 1e-9)
        spec = shine * np.clip(normal @ half.astype(F32), 0.0, None) ** SPEC_EXPONENT
        rgb += (spec * (1.0 - SHADOW_STRENGTH * (1.0 - visible)))[:, None]
    hot = asset_vis[:, None] & (rgb > SHOULDER)
    rolled = SHOULDER + (1 - SHOULDER) * (1 - np.exp(-(rgb - SHOULDER) / (1 - SHOULDER)))
    rgb = np.where(hot, rolled, rgb).astype(F32)
    rgb = rgb.reshape(h, w, 3)
    rgb[~lit_px.reshape(h, w)] = np.asarray(style.background, dtype=F32)

    gmask = ground_vis.reshape(h, w)
    if ground is not None:  # dark edge where the ground tile meets the background
        empty = ~lit_px.reshape(h, w)
        gedge = np.zeros((h, w), dtype=bool)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            gedge |= gmask & _shift(empty, dy, dx, True)
        rgb[gedge] = np.asarray(ground.color, dtype=F32) * F32(0.62)

    part_vis = np.where(asset_vis, a_part, -1).reshape(h, w).astype(np.int32)
    coverage = asset_vis.reshape(h, w).copy()
    width = int(style.outline_width)
    if outline and width > 0:
        depth = np.where(asset_vis, a_t, np.inf).reshape(h, w)
        jump = DEPTH_JUMP_PX * mpp
        strong, soft = _edges(part_vis, depth, jump, depth_slopes(normal, cam, asset_vis.reshape(h, w)),
                              (seam_surfaces(scene), normal.reshape(h, w, 3)))
        strong = _thicken(strong, depth, part_vis >= 0, width - 1, jump)
        soft = _thicken(soft, depth, part_vis >= 0, width - 2, jump) & ~strong
        line = np.asarray(style.outline_color, dtype=F32)
        rgb[soft] = rgb[soft] * (1 - INNER_LINE_ALPHA) + line * INNER_LINE_ALPHA
        rgb[strong] = line
    return ViewRender(cam.view, cam, vb, _to_u8(rgb), coverage, gmask & ~coverage, part_vis,
                      visible.reshape(h, w), ground.grid if ground is not None else 0.0)


# ----------------------------------------------------------------------------- public API


def _views(views, style) -> list[str]:
    if views is None:
        views = list(style.render_views)
    if isinstance(views, str):
        views = [v.strip() for v in views.split(",") if v.strip()]
    views = list(dict.fromkeys(views))
    if not views:
        raise SightlineError("no views to render", hint="valid views: front, back, left, right, top, iso")
    for v in views:
        if not isinstance(v, str):
            raise SightlineError(f"view names must be strings, got {v!r}", hint="e.g. views=['front', 'iso']")
        view_basis(v)  # SightlineError with did_you_mean for unknown views
    return views


def _res(res, style) -> int:
    if res is None:
        return int(style.render_res)
    if isinstance(res, bool) or not isinstance(res, (int, np.integer)) or not 16 <= res <= 4096:
        raise SightlineError(f"res must be an integer from 16 to 4096, got {res!r}",
                             hint="res is the pixel count of the long side of the largest view, e.g. 256")
    return int(res)


def _mpp(mpp) -> float | None:
    if mpp is None:
        return None
    if isinstance(mpp, bool) or not isinstance(mpp, (int, float, np.floating, np.integer)) or not mpp > 0 \
            or not math.isfinite(mpp):
        raise SightlineError(f"mpp must be a positive number of meters per pixel, got {mpp!r}")
    return float(mpp)


def render_raw(asset, views=None, res=None, *, mpp: float | None = None, scale_figure: bool = True,
               outline: bool = True, ground: bool | None = None, shadows: bool | None = None) -> dict[str, ViewRender]:
    """Render views sharing one meters-per-pixel scale; no caption, no upscale.

    `res` = pixels on the long side of the largest view (default: the style's), or a fixed `mpp`.
    `ground` / `shadows` default to the style. Views from below the ground (orbit elevation < 0) leave
    the slab out, which would hide the asset's underside. Other modules (inspect, reference) use this
    for raw pixels: `ViewRender.part` and `.rgb` give the shaded color of every visible part.
    """
    style = asset.style
    views = _views(views, style)
    res = _res(res, style)
    lay = layout(asset, scale_figure, ground)
    cams = make_cameras(views, lay.frame_min, lay.frame_max, res=res, mpp=_mpp(mpp))
    ao_step = 0.012 * lay.size
    return {v: render_view(lay.scene, style, cams[v], None if _from_below(v) else lay.ground, outline, shadows, ao_step)
            for v in views}


def _from_below(view: str) -> bool:
    """True for orbit views with a negative elevation (looking up at the asset)."""
    orbit = parse_orbit(view)
    return orbit is not None and orbit[1] < 0


def present_view(asset, r: ViewRender, upscale: int | None = None, caption: bool = True) -> Image.Image:
    """Captioned (and for small images x2 upscaled) presentation image of a ViewRender.

    The caption states the scale of the saved PNG pixels: render mpp / upscale.
    """
    k = upscale_factor((r.camera.width, r.camera.height), upscale)
    text = caption_text(asset.name, r.view, r.mpp / k, r.grid) if caption else None
    return present(r.image(), text, k, background=_rgb255(asset.style.background))


def _rgb255(color) -> tuple[int, int, int]:
    return tuple(int(round(float(c) * 255)) for c in color)


def render_views(asset, views=None, res=None, scale_figure=True, outline=True, *, caption: bool = True,
                 upscale: int | None = None, mpp: float | None = None) -> dict[str, Image.Image]:
    """Shaded views {view: RGB image} at one shared scale, with ground, shadows, outlines, the 1.8 m
    scale figure and a caption strip (`<asset> · <view> · 1 px = <m> m · grid <g> m`).

    Images under 384 px on the long side are upscaled x2 (nearest); the caption's px is the PNG pixel.
    """
    raws = render_raw(asset, views, res, mpp=mpp, scale_figure=scale_figure, outline=outline)
    return {v: present_view(asset, r, upscale, caption) for v, r in raws.items()}


def render_sprite(asset, view: str = "iso", mpp: float | None = None) -> tuple[Image.Image, tuple[float, float]]:
    """RGBA sprite of the asset alone (no ground, no scale figure, transparent background) for level
    compositing, and the anchor: the (col, row) pixel position, fractional, of the asset's ground
    origin (world 0, 0, 0) in the sprite. Pixel (c, r) covers [c, c+1) x [r, r+1), so pasting the
    sprite at (X - col, Y - row) puts the asset origin on screen point (X, Y). `mpp` defaults to the
    style's render resolution framing. Outlines and self-shadows are drawn; alpha is 255 or 0.
    """
    style = asset.style
    _views([view], style)
    scene = asset.scene()
    if not scene.solid_indices():
        raise SightlineError(f"asset '{asset.name}' has no solid parts to render")
    lo, hi = scene.bounds(include_helpers=False)
    m = _mpp(mpp)
    if m is None:
        m = make_cameras([view], lo, hi, res=int(style.render_res))[view].mpp
    tight = make_cameras([view], lo, hi, mpp=m, margin=0.0)[view]
    cam = fixed_camera(view, tight.center, m, tight.width + 4, tight.height + 4, tight.near)
    size = float(max(np.max(hi - lo), 1e-3))
    r = render_view(scene, style, cam, None, True, None, 0.012 * size)
    return r.rgba(), origin_pixel(cam)


def origin_pixel(cam: Camera) -> tuple[float, float]:
    """Fractional (col, row) of world point (0, 0, 0) in a camera's image."""
    px, _ = cam.project(np.zeros((1, 3)))
    return float(px[0, 0]), float(px[0, 1])


# ----------------------------------------------------------------------------- part map


def part_colors(count: int) -> list[tuple[int, int, int]]:
    """Distinct flat colors in a fixed order: golden-ratio hue steps, alternating saturation/value."""
    out = []
    for k in range(count):
        hue = (0.07 + k * 0.6180339887498949) % 1.0
        sat = (0.62, 0.45, 0.78)[k % 3]
        val = (0.93, 0.78)[(k // 3) % 2]
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    return out


def part_map_buffers(asset, view: str = "front", res=None) -> tuple[ViewBuffers, list[int]]:
    """Geometry buffers of the asset alone (no helpers) framed for a part map, and the solid part
    indices in file order (the color order)."""
    style = asset.style
    _views([view], style)
    scene = asset.scene()
    order = scene.solid_indices()
    if not order:
        raise SightlineError(f"asset '{asset.name}' has no solid parts to render")
    lo, hi = scene.bounds(include_helpers=False)
    cam = make_cameras([view], lo, hi, res=_res(res, style))[view]
    return march(scene, cam), order


def _interior_point(mask: np.ndarray) -> tuple[float, float]:
    """A pixel of `mask` near its centroid but away from its border (row, col) as pixel centers."""
    rows, cols = np.nonzero(mask)
    cy, cx = rows.mean(), cols.mean()
    m = mask
    for _ in range(2):
        e = _erode(m)
        if not e.any():
            break
        m = e
    rows, cols = np.nonzero(m)
    k = int(np.argmin((rows - cy) ** 2 + (cols - cx) ** 2))
    return rows[k] + 0.5, cols[k] + 0.5


def _overlaps(box, boxes, gap=2.0) -> bool:
    x0, y0, x1, y1 = box
    return any(x0 < b[2] + gap and b[0] < x1 + gap and y0 < b[3] + gap and b[1] < y1 + gap for b in boxes)


def render_part_map(asset, view: str = "front", res=None, *, labels: bool = True, outline: bool = True,
                    caption: bool = True, upscale: int | None = None) -> Image.Image:
    """Flat color per solid part (golden-ratio hues, file order) with part names.

    Each label sits at the centroid of the part's visible pixels when free; otherwise a greedy
    search tries rings of positions around it (and the side margins) and the label gets a leader
    line to a dot on the part. Labels never overlap each other. Parts not visible in this view are
    listed under the image. With labels=False and outline=False every pixel of part i has exactly
    the color part_colors(n)[k] where k is i's position among the solid parts.
    """
    style = asset.style
    vb, order = part_map_buffers(asset, view, res)
    cam = vb.camera
    colors = part_colors(len(order))
    slot = {pi: k for k, pi in enumerate(order)}
    lut = np.zeros((len(asset.scene().parts) + 1, 3), dtype=np.uint8)
    lut[0] = _rgb255(style.background)
    for pi, k in slot.items():
        lut[pi + 1] = colors[k]
    ids = vb.part.astype(np.int64)
    rgb = lut[ids + 1]
    if outline:
        strong, soft = _edges(vb.part, vb.depth, DEPTH_JUMP_PX * cam.mpp, depth_slopes(vb.normal, cam, vb.hit))
        rgb[strong | soft] = (40, 40, 44)
    img = Image.fromarray(rgb, "RGB")
    k = upscale_factor((cam.width, cam.height), upscale)
    if labels:
        k = max(k, int(math.ceil(PART_MAP_MIN / max(cam.width, cam.height))))  # room for readable labels
    img = img.resize((img.width * k, img.height * k), Image.NEAREST)
    parts = asset.scene().parts
    present_ids = set(np.unique(vb.part).tolist())
    visible = [pi for pi in order if pi in present_ids]
    hidden = [parts[pi].name for pi in order if pi not in present_ids]
    bg = _rgb255(style.background)
    if labels and visible:
        img, _ = label_parts(img, vb.part, visible, [parts[pi].name for pi in visible],
                             [colors[slot[pi]] for pi in visible], k, bg)
    if not caption:
        return img
    fnt, sep = font(CAPTION_FONT), separator()
    lines = _wrap(caption_text(asset.name, f"parts {view}", cam.mpp / k).split(sep), fnt, img.width - 12, sep)
    if hidden:
        lines += _wrap(("not visible: " + ", ".join(hidden)).split(" "), fnt, img.width - 12, " ")
    return present(img, lines, 1, bg)


def label_parts(img: Image.Image, ids: np.ndarray, parts: list[int], names: list[str],
                colors: list[tuple], k: int, background) -> tuple[Image.Image, list[tuple]]:
    """Label parts on a part map `img` (= `ids` upscaled k times): greedy placement, leader lines.

    Labels go largest part first. Each tries the part's interior point, then rings of positions
    around it, then the side margins (added to the canvas); the cheapest spot that overlaps no
    earlier label wins, where the cost is the distance plus a penalty for covering other parts
    (heavy for small ones) and their anchor dots. Returns the image and the label boxes
    (x0, y0, x1, y1) in its pixel coordinates, in the order of `parts`.
    """
    fnt = font(LABEL_FONT)
    pad, swatch = 3, 9
    sizes = []
    for name in names:
        l, t, tw, th = text_box(name, fnt)
        sizes.append((tw + 2 * pad + swatch + 4, max(th, swatch) + 2 * pad, l, t, th))
    margin = max(s[0] for s in sizes) + 16
    top = 14
    w0, h0 = img.size
    canvas_w, canvas_h = w0 + 2 * margin, h0 + 2 * top
    up = np.full((canvas_h, canvas_w), -1, dtype=np.int64)
    up[top:top + h0, margin:margin + w0] = np.repeat(np.repeat(ids, k, axis=0), k, axis=1)
    areas = {pi: int((ids == pi).sum()) * k * k for pi in parts}
    anchors = {}
    for pi in parts:
        r, c = _interior_point(ids == pi)
        anchors[pi] = (margin + c * k, top + r * k)
    order = sorted(range(len(parts)), key=lambda j: (-areas[parts[j]], j))
    placed: list[tuple] = []
    results = {}
    radii = (0, 12, 20, 30, 42, 56, 72, 92, 116, 144, 180, 224, 280)
    angles = [2 * math.pi * a / 16 for a in (0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15)]
    for j in order:
        pi = parts[j]
        bw, bh = sizes[j][:2]
        ax, ay = anchors[pi]
        cands = []
        for rad in radii:
            for ang in (angles if rad else [0.0]):
                cx, cy = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
                cands.append((cx - bw / 2, cy - bh / 2, rad))
        for dy in range(0, 400, 6):  # side margins at the anchor's height, then stepping away
            for sy in ((0,) if dy == 0 else (-dy, dy)):
                cands.append((margin - bw - 8, ay - bh / 2 + sy, 300 + dy + (ax - margin) * 0.3))
                cands.append((margin + w0 + 8, ay - bh / 2 + sy, 300 + dy + (margin + w0 - ax) * 0.3))
        cands.sort(key=lambda c: c[2])
        best = None
        for x0, y0, dist in cands:
            if best is not None and dist >= best[0]:
                break  # cost >= dist, so no later candidate can win
            x0, y0 = round(x0), round(y0)
            box = (x0, y0, x0 + bw, y0 + bh)
            if x0 < 1 or y0 < 1 or box[2] > canvas_w - 1 or box[3] > canvas_h - 1 or _overlaps(box, placed):
                continue
            counts = np.bincount((up[y0:y0 + bh, x0:x0 + bw] + 1).ravel())
            cost = float(dist)
            for q in np.flatnonzero(counts[1:]):
                if q != pi:
                    cnt = int(counts[q + 1])
                    cost += 250.0 * cnt / max(areas.get(q, cnt), 1) + 0.03 * cnt
            for qj, (qx, qy) in anchors.items():  # keep other parts' anchor dots uncovered
                if qj != pi and box[0] - 4 <= qx <= box[2] + 4 and box[1] - 4 <= qy <= box[3] + 4:
                    cost += 400.0
            if best is None or cost < best[0]:
                best = (cost, box)
        if best is None:  # no free spot: stack in the right margin below everything
            y0 = max([b[3] for b in placed] + [0]) + 3
            best = (0.0, (margin + w0 + 8, y0, margin + w0 + 8 + bw, y0 + bh))
            canvas_h = max(canvas_h, y0 + bh + 2)
        placed.append(best[1])
        results[j] = best[1]

    # Crop the canvas to the image plus the labels actually used.
    x_lo = min([margin] + [b[0] - 4 for b in placed])
    x_hi = max([margin + w0] + [b[2] + 4 for b in placed])
    y_lo = min([top] + [b[1] - 4 for b in placed])
    y_hi = max([top + h0] + [b[3] + 4 for b in placed])
    x_lo, y_lo = max(0, int(x_lo)), max(0, int(y_lo))
    out = Image.new("RGB", (int(x_hi) - x_lo, int(y_hi) - y_lo), background)
    out.paste(img, (margin - x_lo, top - y_lo))
    draw = ImageDraw.Draw(out)
    for j, box in results.items():  # leaders first, under every box
        ax, ay = anchors[parts[j]]
        ax, ay = ax - x_lo, ay - y_lo
        bx0, by0, bx1, by1 = (box[0] - x_lo, box[1] - y_lo, box[2] - x_lo, box[3] - y_lo)
        if not (bx0 <= ax <= bx1 and by0 <= ay <= by1):
            nx, ny = min(max(ax, bx0), bx1), min(max(ay, by0), by1)
            draw.line([(ax, ay), (nx, ny)], fill=(30, 30, 34), width=1)
            draw.ellipse([ax - 2.5, ay - 2.5, ax + 2.5, ay + 2.5], fill=(255, 255, 255), outline=(30, 30, 34))
    for j, box in results.items():
        bw, bh, l, t, th = sizes[j]
        x0, y0 = box[0] - x_lo, box[1] - y_lo
        draw.rectangle([x0, y0, x0 + bw - 1, y0 + bh - 1], fill=(255, 255, 255), outline=(70, 72, 78))
        sy = y0 + (bh - swatch) // 2
        draw.rectangle([x0 + pad, sy, x0 + pad + swatch - 1, sy + swatch - 1], fill=colors[j], outline=(70, 72, 78))
        draw.text((x0 + pad + swatch + 4 - l, y0 + (bh - th) / 2 - t), names[j], font=fnt, fill=TEXT_RGB)
    boxes = [(results[j][0] - x_lo, results[j][1] - y_lo, results[j][2] - x_lo, results[j][3] - y_lo)
             for j in range(len(parts))]
    return out, boxes


# ----------------------------------------------------------------------------- files


def write_renders(asset, out_dir, views=None, res=None, *, scale_figure: bool = True,
                  ground: bool | None = None) -> dict[str, pathlib.Path]:
    """Write `<view>.png` for every view (default: the style's; ':' in orbit names becomes '_'),
    `parts.png` (part map of the iso view, or front / the first view when iso is not rendered) and
    `sheet.png` (all of them under a header) into out_dir, creating it.

    Returns {view: path, ..., "parts": path, "sheet": path}. scale_figure=False is `--no-human`,
    ground=False is `--no-ground` (default: the style's ground.show).
    """
    from .sheet import make_sheet

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style = asset.style
    views = _views(views, style)
    raws = render_raw(asset, views, res, scale_figure=scale_figure, ground=ground)
    images = {v: present_view(asset, r) for v, r in raws.items()}
    map_view = "iso" if "iso" in views else ("front" if "front" in views else views[0])
    part_map = render_part_map(asset, map_view, res)
    sheet = make_sheet(asset, images, part_map)
    paths: dict[str, pathlib.Path] = {}
    for v, img in images.items():
        name = v.replace(":", "_")
        paths[v] = out / f"{name}.png"
        img.save(paths[v])
    paths["parts"] = out / "parts.png"
    part_map.save(paths["parts"])
    paths["sheet"] = out / "sheet.png"
    sheet.save(paths["sheet"])
    return paths
