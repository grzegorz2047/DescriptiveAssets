"""Level renders: an iso PNG (column prisms drawn back to front) and a top-down grid PNG.

The iso projection is the one of `camera.view_basis("iso")` (azimuth 45°, elevation 30°,
orthographic), so object sprites from `sightline.render.render_sprite` line up with the terrain.
Geometry is drawn at 2x and box-reduced; text is drawn afterwards at 1x so it stays crisp. Labels
never cover marker heads, object sprites or each other while a free spot exists.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..camera import view_basis
from .model import TILE_KINDS, Level, LevelObject

SS = 2  # supersampling factor for geometry
PX_PER_CELL = 40  # iso: pixels per cell edge length at 1x
MAX_SIDE = 1800
MIN_WIDTH = 360  # narrower images get side padding so captions and labels have room
DEFAULT_COLORS = {
    "floor": (201, 194, 176),
    "wall": (128, 134, 145),
    "water": (74, 134, 197),
    "spikes": (176, 74, 74),
    "pit": (40, 40, 46),
    "ladder": (168, 116, 61),
    "door": (138, 90, 43),
    "key": (201, 194, 176),
}
DEFAULT_BACKGROUND = (221, 225, 230)
DEFAULT_LIGHT = np.array([-0.4, -1.0, -0.5]) / np.linalg.norm([-0.4, -1.0, -0.5])
DEFAULT_AMBIENT = 0.35
PATH_RGB = (214, 36, 140)
SOFTLOCK_RGB = (224, 49, 49)
UNREACHED_RGB = (112, 72, 232)
OBJECT_RGB = (190, 168, 132)
KEY_RGB = (240, 180, 0)
SPAWN_RGB = (47, 158, 68)
GOAL_RGB = (224, 49, 49)
OTHER_RGB = (28, 126, 214)
TEXT_RGB = (27, 27, 27)
DOOR_HEIGHT = 2.0
PIN_PX = 34  # iso marker pin length at 1x
PIN_HEAD_PX = 7


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, AttributeError):
        return ImageFont.load_default()


def _rgb(value) -> tuple[int, int, int]:
    """Style color (0..1 floats, 0..255 ints or #rrggbb) -> 0..255 ints."""
    if isinstance(value, str):
        v = value.lstrip("#")
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    arr = np.asarray(value, dtype=np.float64)[:3]
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return tuple(int(round(float(x))) for x in np.clip(arr, 0, 255))


def _scale(rgb, f: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(c * f)))) for c in rgb)


def _mix(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


@dataclass
class Look:
    colors: dict
    background: tuple
    light: np.ndarray  # direction the light travels
    ambient: float
    source: str

    def shade(self, rgb, normal) -> tuple[int, int, int]:
        """Lambert with the style light at half strength: faces stay distinct but none goes black."""
        lam = max(0.0, float(np.dot(normal, -self.light)))
        return _scale(rgb, 0.5 + 0.5 * (self.ambient + (1.0 - self.ambient) * lam))


def level_look(level: Level) -> Look:
    """Tile colors and light from the level's style (built-in colors if the style module is missing).

    A style can color tiles with materials or palette entries named after the tile kind (`water`,
    `wall`, ...) or `tile_<kind>`; other kinds keep the built-in colors. An invalid style file raises.
    """
    look = Look(dict(DEFAULT_COLORS), DEFAULT_BACKGROUND, DEFAULT_LIGHT.copy(), DEFAULT_AMBIENT, "built-in colors")
    try:
        from ..style import load_style
    except ImportError:
        return look
    style = load_style(level.style, level.root)
    look.background = _rgb(style.background)
    light = np.asarray(style.light_dir, dtype=np.float64)
    look.light = light / max(np.linalg.norm(light), 1e-9)
    look.ambient = float(style.ambient)
    for kind in TILE_KINDS:
        mat = style.materials.get(kind) or style.materials.get(f"tile_{kind}")
        if mat is not None:
            look.colors[kind] = _rgb(mat.color)
        elif kind in style.palette:
            look.colors[kind] = _rgb(style.palette[kind])
    look.source = f"style {style.name}"
    return look


def _marker_rgb(level: Level, name: str):
    if name == "spawn":
        return SPAWN_RGB
    if name == "goal":
        return GOAL_RGB
    if name in level.key_names():
        return KEY_RGB
    return OTHER_RGB


# ---------------------------------------------------------------- labels and captions

class _Labels:
    """Text boxes placed at 1x after the geometry.

    Each label tries candidate spots in order and takes the first one inside the image that covers
    no earlier label and no blocked area (marker heads, object sprites); when none is free it takes
    the spot with the least overlap. A label that ends up away from its point gets a leader line.
    Nothing is drawn until `draw_all()`, which draws every leader line below every box.
    """

    def __init__(self, image: Image.Image, font):
        self.draw = ImageDraw.Draw(image)
        self.size = image.size
        self.font = font
        self.boxes: list[tuple] = []
        self.blocked: list[tuple] = []
        self.pending: list[tuple] = []  # (box, text, color, border, leader or None, text origin)

    def block(self, box) -> None:
        """Keep labels off this (x0, y0, x1, y1) area."""
        self.blocked.append(tuple(float(v) for v in box))

    def _fit(self, x0: float, y0: float, w: float, h: float) -> tuple:
        x0 = min(max(x0, 2), self.size[0] - w - 2)
        y0 = min(max(y0, 2), self.size[1] - h - 2)
        return (x0, y0, x0 + w, y0 + h)

    def overlap(self, box) -> float:
        """Area of `box` covered by placed labels and blocked areas."""
        total = 0.0
        for o in self.boxes + self.blocked:
            dx = min(box[2], o[2]) - max(box[0], o[0])
            dy = min(box[3], o[3]) - max(box[1], o[1])
            if dx > 0 and dy > 0:
                total += dx * dy
        return total

    def place(self, text: str, spots, point, near: float, color=TEXT_RGB, border=None) -> tuple:
        """Draw `text` at the best of `spots(w, h)` (top-left corners); leader line when farther than `near`."""
        l, t, r, b = self.draw.textbbox((0, 0), text, font=self.font)
        w, h = r - l + 8, b - t + 6
        best = None
        for x0, y0 in spots(w, h):
            box = self._fit(x0, y0, w, h)
            cost = self.overlap(box)
            if best is None or cost < best[0]:
                best = (cost, box)
            if cost == 0:
                break
        box = best[1]
        self.boxes.append(box)
        px, py = point
        nx, ny = min(max(px, box[0]), box[2]), min(max(py, box[1]), box[3])
        leader = [(px, py), (nx, ny)] if math.hypot(nx - px, ny - py) > near else None
        self.pending.append((box, text, color, border, leader, (box[0] + 4 - l, box[1] + 3 - t)))
        return box

    def draw_all(self) -> None:
        for _, _, _, _, leader, _ in self.pending:
            if leader:
                self.draw.line(leader, fill=(90, 90, 90), width=1)
        for box, text, color, border, _, origin in self.pending:
            self.draw.rounded_rectangle(box, radius=4, fill=(255, 255, 255), outline=border or (120, 120, 120))
            self.draw.text(origin, text, font=self.font, fill=color)
        self.pending = []

    def beside(self, x: float, y: float, radius: float, text: str, **kw) -> tuple:
        """Label next to a disc at (x, y): right, left, above, below, then stepping up/down on both sides."""
        gap = radius + 3

        def spots(w, h):
            yield x + gap, y - h / 2
            yield x - gap - w, y - h / 2
            yield x - w / 2, y - gap - h
            yield x - w / 2, y + gap
            for k in range(1, 8):
                for sign in (-1, 1):
                    yield x + gap, y - h / 2 + sign * k * (h + 2)
                    yield x - gap - w, y - h / 2 + sign * k * (h + 2)

        return self.place(text, spots, (x, y), gap + 4, **kw)

    def above(self, x: float, top: float, text: str, **kw) -> tuple:
        """Label above the screen point (x, top): centered first, then the free spot nearest to it."""

        def spots(w, h):
            grid = [(x - w / 2 + shift * (w / 2 + 6), top - 4 - h - k * (h + 2))
                    for shift in (0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0) for k in range(6)]
            # Nearest first, measured from the box's bottom center to the point (stable sort: ties keep order).
            return sorted(grid, key=lambda s: math.hypot(s[0] + w / 2 - x, s[1] + h - top))

        return self.place(text, spots, (x, top), 8, **kw)

    def centered(self, x: float, y: float, text: str, **kw) -> tuple:
        """Label centered on (x, y), else just above or below it."""

        def spots(w, h):
            for k in (0, -1, 1, -2, 2, -3, 3):
                yield x - w / 2, y - h / 2 + k * (h + 2)

        return self.place(text, spots, (x, y), 8, **kw)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, width: float) -> list[str]:
    """Greedy wrap at ' · ' separators, then at spaces, so no caption line runs past `width` pixels."""
    def greedy(parts: list[str], sep: str) -> list[str]:
        lines, cur = [], ""
        for part in parts:
            cand = f"{cur}{sep}{part}" if cur else part
            if not cur or draw.textlength(cand, font=font) <= width:
                cur = cand
            else:
                lines.append(cur)
                cur = part
        return lines + ([cur] if cur else [])

    out = []
    for line in greedy(text.split(" · "), " · "):
        out.extend([line] if draw.textlength(line, font=font) <= width else greedy(line.split(" "), " "))
    return out


def _caption(image: Image.Image, lines: list[str], legend: list[tuple], background) -> Image.Image:
    """Add a caption strip (wrapped to the image width) and a color legend under the image."""
    font = _font(13)
    small = _font(12)
    pad = 8
    line_h = 18
    probe = ImageDraw.Draw(image)
    text_lines = [w for text in lines for w in wrap_text(probe, text, font, image.width - 2 * pad)]
    # Wrap legend items into rows that fit the image width.
    rows, row, x = [], [], pad
    for rgb, text in legend:
        w = 18 + int(probe.textlength(text, font=small)) + 14
        if row and x + w > image.width - pad:
            rows.append(row)
            row, x = [], pad
        row.append((rgb, text, w))
        x += w
    if row:
        rows.append(row)
    height = pad + line_h * len(text_lines) + 20 * len(rows) + pad
    out = Image.new("RGB", (image.width, image.height + height), background)
    out.paste(image, (0, 0))
    d = ImageDraw.Draw(out)
    y = image.height
    d.line([(0, y), (image.width, y)], fill=(150, 150, 150), width=1)
    y += pad
    for text in text_lines:
        d.text((pad, y), text, font=font, fill=TEXT_RGB)
        y += line_h
    for row in rows:
        x = pad
        for rgb, text, w in row:
            d.rectangle((x, y + 3, x + 12, y + 15), fill=rgb, outline=(60, 60, 60))
            d.text((x + 18, y + 2), text, font=small, fill=TEXT_RGB)
            x += w
        y += 20
    return out


def _legend(level: Level, report: dict, look: Look, object_label: str | None) -> list[tuple]:
    present = set(level.kinds.ravel().tolist())
    items = [(look.colors[k], k) for k in TILE_KINDS if k in present]
    if report["path"]["found"]:
        items.append((PATH_RGB, "critical path"))
    if object_label:
        items.append((OBJECT_RGB, object_label))
    if report["softlocks"].get("regions") and report["softlocks"].get("goal_reachable"):
        items.append((SOFTLOCK_RGB, "softlock"))
    if report["reachability"]["regions"]:
        items.append((UNREACHED_RGB, "never reached"))
    return items


def _softlock_mask(level: Level, report: dict) -> np.ndarray:
    """Softlock cells; empty when the goal is unreachable (then every reachable cell is stuck, which says nothing)."""
    mask = np.zeros((level.rows, level.cols), dtype=bool)
    if not report["softlocks"].get("goal_reachable", False):
        return mask
    for reg in report["softlocks"].get("regions", []):
        for c, r in reg["cells"]:
            mask[r, c] = True
    return mask


def _unreached_mask(level: Level, report: dict) -> np.ndarray:
    mask = np.zeros((level.rows, level.cols), dtype=bool)
    for reg in report["reachability"]["regions"]:
        for c, r in reg["cells"]:
            mask[r, c] = True
    return mask


# ---------------------------------------------------------------- object sprites

def sprite_renderer():
    """(render_sprite, None) when `sightline.render` provides it, else (None, note for the caption).

    A missing `sightline.render` module is the normal state while it is being written: the note then
    only says that objects are drawn as footprint boxes. Any other import failure is named in the note.
    """
    try:
        from ..render import render_sprite
    except ModuleNotFoundError as exc:
        if exc.name == f"{__package__.rpartition('.')[0]}.render":
            return None, "objects drawn as footprint boxes: no sprite renderer (sightline.render) yet"
        return None, f"objects drawn as footprint boxes: sightline.render failed to import ({exc})"[:200]
    except Exception as exc:  # a half-written renderer must not take the level render down with it
        return None, f"objects drawn as footprint boxes: sightline.render failed to import ({type(exc).__name__}: {exc})"[:200]
    return render_sprite, None


def _rotated_asset(asset, obj: LevelObject):
    """Shallow copy of an asset with every part turned by the object's yaw about the asset origin.

    Uses the Part matrix/position contract (world matrix R @ M, position R @ p), so the copy is what
    `rotate: [0, deg, 0]` on every part would give. The original asset is left untouched.
    """
    rot = obj.yaw_matrix()
    out = copy.copy(asset)
    out.parts = []
    for part in asset.parts:
        twin = copy.copy(part)
        twin.matrix = rot @ np.asarray(part.matrix, dtype=np.float64)
        twin.position = rot @ np.asarray(part.position, dtype=np.float64)
        out.parts.append(twin)
    if hasattr(out, "_scene"):  # the scene cache belongs to the unrotated parts
        out._scene = None
    return out


def _object_sprite(obj: LevelObject, mpp: float, render_sprite) -> tuple[Image.Image, tuple[float, float]] | str | None:
    """(RGBA sprite, anchor px) from the asset renderer, None without one, or an error note."""
    if obj.asset_obj is None or render_sprite is None:
        return None
    try:
        asset = _rotated_asset(obj.asset_obj, obj) if obj.rotate % 360 else obj.asset_obj
        image, anchor = render_sprite(asset, view="iso", mpp=mpp)
        return image.convert("RGBA"), (float(anchor[0]), float(anchor[1]))
    except Exception as exc:  # a broken renderer must not hide the level; the image says what happened
        return f"{obj.asset}: sprite failed ({type(exc).__name__}: {exc})"[:200]


# ---------------------------------------------------------------- iso

class _Canvas:
    """The 2x color image plus a rank buffer holding the painter's-order rank of the item that drew
    each pixel last, so overlays (the critical path) can be hidden where nearer geometry covers them."""

    def __init__(self, size: tuple[int, int], background):
        self.image = Image.new("RGB", size, background)
        self.ranks = Image.new("I", size, -1)
        self._color = ImageDraw.Draw(self.image)
        self._rank = ImageDraw.Draw(self.ranks)
        self.rank = 0  # rank of the item being drawn

    def polygon(self, pts, fill, outline=None, width: int = 1) -> None:
        self._color.polygon(pts, fill=fill, outline=outline, width=width)
        self._rank.polygon(pts, fill=self.rank, outline=self.rank, width=width)

    def line(self, pts, fill, width: int = 1) -> None:
        self._color.line(pts, fill=fill, width=width)
        self._rank.line(pts, fill=self.rank, width=width)

    def paste(self, img: Image.Image, pos: tuple[int, int]) -> None:
        """Paste an RGBA image; its non-transparent pixels take the current rank."""
        self.image.paste(img, pos, img)
        solid = img.getchannel("A").point(lambda a: 255 if a > 0 else 0)
        self.ranks.paste(self.rank, (pos[0], pos[1], pos[0] + img.width, pos[1] + img.height), solid)


class _Iso:
    def __init__(self, level: Level, extent_pts: np.ndarray):
        _, self.right, self.up = view_basis("iso")
        u = extent_pts @ self.right
        v = extent_pts @ self.up
        self.mpp = level.cell / PX_PER_CELL
        longest = max(float(np.ptp(u)), float(np.ptp(v))) / self.mpp
        if longest > MAX_SIDE:
            self.mpp *= longest / MAX_SIDE
        span = int(math.ceil(np.ptp(u) / self.mpp))
        self.margin, self.top = 30, 70
        self.margin_x = max(self.margin, (MIN_WIDTH - span + 1) // 2)
        self.u0, self.v1 = float(u.min()), float(v.max())
        self.width = span + 2 * self.margin_x
        self.height = int(math.ceil(np.ptp(v) / self.mpp)) + self.top + self.margin

    def px(self, pts, scale: int = SS) -> list[tuple[float, float]]:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        x = ((p @ self.right - self.u0) / self.mpp + self.margin_x) * scale
        y = ((self.v1 - p @ self.up) / self.mpp + self.top) * scale
        return list(zip(x.tolist(), y.tolist()))


def _box(draw: _Canvas, iso: _Iso, look: Look, x0, x1, y0, y1, z0, z1, rgb, top_rgb=None, edge=0.78) -> None:
    """Draw the three camera-facing faces of an axis-aligned box: south (+Z), east (+X), top (+Y)."""
    faces = []
    if y1 - y0 > 1e-6:
        faces.append(([(x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1)], (0.0, 0.0, 1.0), rgb))
        faces.append(([(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)], (1.0, 0.0, 0.0), rgb))
    faces.append(([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0.0, 1.0, 0.0), top_rgb or rgb))
    for pts, normal, color in faces:
        fill = look.shade(color, np.array(normal))
        draw.polygon(iso.px(pts), fill=fill, outline=_scale(fill, edge), width=1)


def _box_bbox(iso: _Iso, x0, x1, y0, y1, z0, z1) -> tuple[float, float, float, float]:
    """Screen bbox (1x) of an axis-aligned world box."""
    pts = np.array(iso.px([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], scale=1))
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _object_cells(obj: LevelObject) -> list[tuple[int, int]]:
    """Every cell under the object (blocking or overhang), at least its anchor cell."""
    return list(obj.cover) or list(obj.cells) or [(obj.col, obj.row)]


def _footprint_boxes(level: Level, obj: LevelObject) -> list[tuple[float, float, float, float]]:
    """World x0, x1, z0, z1 of one box per footprint cell, back to front, inset only on the outer edges
    (the boxes read as one block that follows the footprint, also for rotated objects)."""
    cell = level.cell
    cells = set(obj.cells) or {(obj.col, obj.row)}
    pad = 0.12 * cell
    out = []
    for c, r in sorted(cells, key=lambda cr: (cr[0] + cr[1], cr[0])):
        out.append((c * cell + (0 if (c - 1, r) in cells else pad), (c + 1) * cell - (0 if (c + 1, r) in cells else pad),
                    r * cell + (0 if (c, r - 1) in cells else pad), (r + 1) * cell - (0 if (c, r + 1) in cells else pad)))
    return out


def _door_span(level: Level, c: int, r: int) -> str:
    """'x' when the door panel spans along X (walls west and east), else 'z'."""
    def blocked(cc, rr):
        return not level.in_bounds(cc, rr) or level.kind(cc, rr) == "wall"
    return "x" if blocked(c - 1, r) and blocked(c + 1, r) else ("z" if blocked(c, r - 1) and blocked(c, r + 1) else "x")


def _draw_ladder(draw: _Canvas, iso: _Iso, level: Level, c: int, r: int, h: np.ndarray) -> None:
    """Ladder rails and rungs against the highest standable neighbor's side."""
    cell = level.cell
    best = None
    for dc, dr in ((0, -1), (-1, 0), (1, 0), (0, 1)):
        cc, rr = c + dc, r + dr
        if level.in_bounds(cc, rr) and level.kind(cc, rr) not in ("pit", "water", "spikes") and h[rr, cc] > h[r, c] + 1e-6:
            if best is None or h[rr, cc] > best[2]:
                best = (dc, dr, float(h[rr, cc]))
    if best is None:
        return
    dc, dr, ytop = best
    y0 = float(h[r, c])
    inset = 0.04 * cell
    x0, z0 = c * cell, r * cell
    if dr:  # edge along X at the shared z
        z = z0 + (0 if dr < 0 else cell) + (inset if dr < 0 else -inset)
        (ax, az), (bx, bz) = (x0 + 0.25 * cell, z), (x0 + 0.75 * cell, z)
    else:
        x = x0 + (0 if dc < 0 else cell) + (inset if dc < 0 else -inset)
        (ax, az), (bx, bz) = (x, z0 + 0.25 * cell), (x, z0 + 0.75 * cell)
    dark = (92, 60, 28)
    for px, pz in ((ax, az), (bx, bz)):
        draw.line(iso.px([(px, y0, pz), (px, ytop + 0.15, pz)]), fill=dark, width=2 * SS)
    y = y0 + 0.3
    while y < ytop + 0.05:
        draw.line(iso.px([(ax, y, az), (bx, y, bz)]), fill=dark, width=SS + 1)
        y += 0.3


def _draw_spikes(draw: _Canvas, iso: _Iso, x0, z0, y, cell) -> None:
    for fx, fz in ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)):
        x, z = x0 + fx * cell, z0 + fz * cell
        s = 0.11 * cell
        tri = iso.px([(x - s, y, z + s), (x + s, y, z - s), (x, y + 0.28 * cell, z)])
        draw.polygon(tri, fill=(70, 70, 76), outline=(30, 30, 30))


def _path_pieces(level: Level, path: dict, lift: float) -> list[tuple[float, list[tuple]]]:
    """(painter's key, world polyline) per move of the critical path.

    Jumps get an arc apex, ladders and big steps a vertical leg at the shared cell edge. A move is
    drawn right after the nearer of its two cells, so walls and objects in front of it hide it.
    """
    if not path.get("found"):
        return []
    pieces = []
    for mv in path["moves"]:
        a = level.world(*mv["from"]) + (0, lift, 0)
        b = level.world(*mv["to"]) + (0, lift, 0)
        pts = [tuple(a)]
        if mv["kind"] in ("jump", "hop"):
            mid = (a + b) / 2
            mid[1] = max(a[1], b[1]) + 0.45 + 0.1 * mv["gap"]
            pts.append(tuple(mid))
        elif mv["kind"] == "ladder" or abs(mv["dy"]) > level.player.step + 1e-9:
            edge = (a + b) / 2
            pts.append((edge[0], a[1], edge[2]))
            pts.append((edge[0], b[1], edge[2]))
        pts.append(tuple(b))
        pieces.append((max(sum(mv["from"]), sum(mv["to"])) + 0.45, pts))
    return pieces


def _dashed_polyline(draw, pts, fill, width: int, dash: float) -> None:
    """Dashes along a polyline, the pattern running on across its corners."""
    phase = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        length = math.hypot(bx - ax, by - ay)
        s = -phase
        while s < length:
            t0, t1 = max(s, 0.0) / max(length, 1e-9), min(s + dash, length) / max(length, 1e-9)
            if t1 > t0:
                draw.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                          fill=fill, width=width)
            s += 2 * dash
        phase = (phase + length) % (2 * dash)


def _draw_path(canvas: _Canvas, iso: _Iso, pieces: list[tuple[int, int, list[tuple]]]) -> None:
    """Critical path: halo + line where no nearer geometry covers it, thin dashes where it is hidden.

    `pieces` are (move index, painter's rank, world polyline); the line is drawn in walking order.
    """
    if not pieces:
        return
    size = canvas.image.size
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    ranks = Image.new("I", size, -1)
    ld, rd = ImageDraw.Draw(layer), ImageDraw.Draw(ranks)
    full = []
    for _, rank, pts in sorted(pieces, key=lambda p: p[0]):
        px = iso.px(pts)
        full.extend(px if not full else px[1:])
        rd.line(px, fill=rank, width=8 * SS, joint="curve")
        for x, y in (px[0], px[-1]):
            rd.ellipse((x - 4 * SS, y - 4 * SS, x + 4 * SS, y + 4 * SS), fill=rank)
    ld.line(full, fill=(255, 255, 255, 255), width=7 * SS, joint="curve")
    ld.line(full, fill=PATH_RGB + (255,), width=3 * SS, joint="curve")
    alpha = np.asarray(layer.getchannel("A"))
    shown = (np.asarray(ranks) >= np.asarray(canvas.ranks)) & (np.asarray(ranks) >= 0)
    canvas.image.paste(layer, (0, 0), Image.fromarray(np.where(shown, alpha, 0).astype(np.uint8), "L"))
    _dashed_polyline(ImageDraw.Draw(canvas.image), full, PATH_RGB, 2 * SS, 5 * SS)


def iso_frame(level: Level) -> tuple[_Iso, float]:
    """The iso image frame of a level (projection, size, mpp) and the y of the terrain slab's base.

    The frame covers the terrain slab, door panels, placeholder boxes and every object's rotated
    asset box, including parts that hang off the grid.
    """
    cell = level.cell
    h = level.heights_m()
    ybase = float(h.min()) - max(0.3, level.height_unit)
    tops = [float(h.max())]
    tops += [o.top for o in level.objects]
    tops += [level.y(d.col, d.row) + DOOR_HEIGHT for d in level.doors.values()]
    top_y = max(tops) + 0.6
    ext = [[x, y, z] for x in (0.0, level.cols * cell) for z in (0.0, level.rows * cell) for y in (ybase, top_y)]
    for obj in level.objects:
        corners = obj.world_corners(cell)
        if corners is not None:
            ext.extend(corners.tolist())
    return _Iso(level, np.array(ext)), ybase


def render_iso(level: Level, report: dict, look: Look) -> Image.Image:
    cell = level.cell
    h = level.heights_m()
    soft = _softlock_mask(level, report)
    unreached = _unreached_mask(level, report)
    iso, ybase = iso_frame(level)
    canvas = _Canvas((iso.width * SS, iso.height * SS), look.background)

    items = []  # (sort key, tie, kind, payload); larger col + row = nearer the camera = drawn later
    for r in range(level.rows):
        for c in range(level.cols):
            items.append((c + r, c, "cell", (c, r)))
            kind = level.kind(c, r)
            if kind == "ladder":
                items.append((c + r + 0.2, c, "ladder", (c, r)))
            elif kind == "door":
                items.append((c + r + 0.3, c, "door", (c, r)))
    for i, (key, pts) in enumerate(_path_pieces(level, report["path"], 0.12)):
        items.append((key, i, "path", (i, pts)))
    notes: list[str] = []
    render_sprite, note = sprite_renderer() if any(o.asset_obj is not None for o in level.objects) else (None, None)
    if note:
        notes.append(note)
    sprites = {}
    for obj in level.objects:
        sprite = _object_sprite(obj, iso.mpp, render_sprite)
        if isinstance(sprite, str):
            notes.append(sprite)
            sprite = None
        sprites[obj.index] = sprite
        items.append((max(c + r for c, r in _object_cells(obj)) + 0.5, obj.col, "object", obj))
    items.sort(key=lambda it: (it[0], it[1]))

    object_boxes = {}  # obj.index -> screen bbox at 1x, for labels
    path_pieces = []  # (move index, rank, world polyline)
    for rank, (_, _, what, payload) in enumerate(items):
        canvas.rank = rank
        if what == "cell":
            c, r = payload
            kind = level.kind(c, r)
            x0, z0 = c * cell, r * cell
            y1 = ybase if kind == "pit" else float(h[r, c])
            rgb = look.colors[kind]
            top = rgb
            if soft[r, c]:
                top = _mix(rgb, SOFTLOCK_RGB, 0.45)
            elif unreached[r, c]:
                top = _mix(rgb, UNREACHED_RGB, 0.4)
            _box(canvas, iso, look, x0, x0 + cell, ybase, y1, z0, z0 + cell, rgb, top_rgb=top, edge=0.85)
            if kind == "spikes":
                _draw_spikes(canvas, iso, x0, z0, y1, cell)
            elif kind == "key":
                ctr = level.world(c, r)
                s = 0.2 * cell
                diamond = iso.px([(ctr[0] - s, y1, ctr[2]), (ctr[0], y1, ctr[2] - s), (ctr[0] + s, y1, ctr[2]), (ctr[0], y1, ctr[2] + s)])
                canvas.polygon(diamond, fill=KEY_RGB, outline=(120, 90, 0))
        elif what == "ladder":
            _draw_ladder(canvas, iso, level, *payload, h)
        elif what == "door":
            c, r = payload
            x0, z0, y0 = c * cell, r * cell, float(h[r, c])
            t = 0.1 * cell
            if _door_span(level, c, r) == "x":
                box = (x0 + 0.06 * cell, x0 + 0.94 * cell, z0 + cell / 2 - t, z0 + cell / 2 + t)
            else:
                box = (x0 + cell / 2 - t, x0 + cell / 2 + t, z0 + 0.06 * cell, z0 + 0.94 * cell)
            _box(canvas, iso, look, box[0], box[1], y0, y0 + DOOR_HEIGHT, box[2], box[3], look.colors["door"])
        elif what == "path":
            move, pts = payload
            path_pieces.append((move, rank, pts))
        else:
            obj = payload
            sprite = sprites[obj.index]
            if sprite is not None:
                img, (ax, ay) = sprite
                (px, py), = iso.px([level.world(obj.col, obj.row, obj.base)], scale=1)
                # Paste on whole 1x pixels with a nearest-neighbour 2x copy: after the 2x box reduce the
                # sprite comes back pixel for pixel, outlines stay crisp.
                x0, y0 = int(round(px - ax)), int(round(py - ay))
                canvas.paste(img.resize((img.width * SS, img.height * SS), Image.NEAREST), (x0 * SS, y0 * SS))
                bb = img.getchannel("A").getbbox()
                if bb is not None:
                    object_boxes[obj.index] = (x0 + bb[0], y0 + bb[1], x0 + bb[2], y0 + bb[3])
            else:
                boxes = []
                for bx0, bx1, bz0, bz1 in _footprint_boxes(level, obj):
                    _box(canvas, iso, look, bx0, bx1, obj.base, obj.top, bz0, bz1, OBJECT_RGB, edge=0.6)
                    boxes.append(_box_bbox(iso, bx0, bx1, obj.base, obj.top, bz0, bz1))
                object_boxes[obj.index] = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                                           max(b[2] for b in boxes), max(b[3] for b in boxes))

    _draw_path(canvas, iso, path_pieces)
    # Marker pins, always on top.
    draw = ImageDraw.Draw(canvas.image)
    heads = {}
    for name, (c, r) in level.markers.items():
        (bx, by), = iso.px([level.world(c, r)])
        hy = by - PIN_PX * SS
        draw.line([(bx, by), (bx, hy)], fill=(40, 40, 40), width=2 * SS)
        rad = PIN_HEAD_PX * SS
        draw.ellipse((bx - rad, hy - rad, bx + rad, hy + rad), fill=_marker_rgb(level, name), outline=(30, 30, 30), width=SS)
        heads[name] = (bx / SS, hy / SS)

    out = canvas.image.reduce(SS)
    labels = _Labels(out, _font(13))
    for x, y in heads.values():
        labels.block((x - PIN_HEAD_PX - 1, y - PIN_HEAD_PX - 1, x + PIN_HEAD_PX + 1, y + PIN_HEAD_PX + 1))
    for box in object_boxes.values():
        labels.block(box)
    # Objects and doors first: they have one natural spot (above), marker labels can go to any side.
    for obj in level.objects:
        if obj.index in object_boxes:  # label above the drawn object, never on it
            x0, y0, x1, _ = object_boxes[obj.index]
            x = (x0 + x1) / 2
        else:  # an empty sprite: label its anchor at the object's top
            (x, y0), = iso.px([level.world(obj.col, obj.row, obj.top)], scale=1)
        labels.above(x, y0, obj.asset + (" (placeholder)" if obj.placeholder else ""))
    for door in level.doors.values():
        (x, y), = iso.px([level.world(door.col, door.row) + (0, DOOR_HEIGHT, 0)], scale=1)
        labels.above(x, y, door.name + (f" ({door.key})" if door.key else ""), border=look.colors["door"])
    for name, (x, y) in heads.items():
        labels.beside(x, y, PIN_HEAD_PX, name, border=_marker_rgb(level, name))
    labels.draw_all()

    path = report["path"]
    tail = (f"path {path['length']:.2f} m, {path['jumps']} jump(s)" if path["found"] else f"no path: {path.get('reason')}")
    lines = [f"{level.name} · iso · 1 px = {iso.mpp:.3f} m · grid {cell:g} m · {tail}",
             f"heights x {level.height_unit:g} m · {look.source}"]
    lines += list(dict.fromkeys(notes)) + [f"warning: {w}" for w in level.warnings]
    boxes_drawn = any(sprites[o.index] is None for o in level.objects)
    return _caption(out, lines, _legend(level, report, look, "object box" if boxes_drawn else None), look.background)


# ---------------------------------------------------------------- top

def render_top(level: Level, report: dict, look: Look) -> Image.Image:
    cp = int(max(18, min(40, 1200 // max(level.cols, level.rows))))  # pixels per cell at 1x
    left, top, pad = 36, 30, 16
    width = max(MIN_WIDTH, left + level.cols * cp + pad)
    height = top + level.rows * cp + pad
    image = Image.new("RGB", (width * SS, height * SS), look.background)
    draw = ImageDraw.Draw(image)
    h = level.heights_m()
    hmin, hmax = float(h.min()), float(h.max())
    span = max(hmax - hmin, 1e-9)
    soft = _softlock_mask(level, report)
    unreached = _unreached_mask(level, report)

    def rect(c, r, inset=0.0):
        return ((left + (c + inset) * cp) * SS, (top + (r + inset) * cp) * SS,
                (left + (c + 1 - inset) * cp) * SS - 1, (top + (r + 1 - inset) * cp) * SS - 1)

    def center(c, r):
        return ((left + (c + 0.5) * cp) * SS, (top + (r + 0.5) * cp) * SS)

    dark = (92, 60, 28)
    for r in range(level.rows):
        for c in range(level.cols):
            kind = level.kind(c, r)
            rgb = look.colors[kind]
            if kind != "pit":  # brighter = higher
                rgb = _scale(rgb, 0.82 + 0.3 * (h[r, c] - hmin) / span)
            x0, y0, x1, y1 = rect(c, r)
            draw.rectangle((x0, y0, x1, y1), fill=rgb)
            w, hh = x1 - x0, y1 - y0
            if kind == "spikes":
                for fx in (0.3, 0.7):
                    for fy in (0.3, 0.7):
                        cx, cy, s = x0 + fx * w, y0 + fy * hh, 0.13 * cp * SS
                        draw.polygon([(cx, cy - s), (cx + s, cy + s), (cx - s, cy + s)], fill=(60, 60, 64))
            elif kind == "ladder":
                for fx in (0.3, 0.7):
                    draw.line([(x0 + fx * w, y0 + 2 * SS), (x0 + fx * w, y1 - 2 * SS)], fill=dark, width=2 * SS)
                for fy in (0.25, 0.5, 0.75):
                    draw.line([(x0 + 0.3 * w, y0 + fy * hh), (x0 + 0.7 * w, y0 + fy * hh)], fill=dark, width=SS)
            elif kind == "key":
                cx, cy = center(c, r)
                s = 0.22 * cp * SS
                draw.polygon([(cx - s, cy), (cx, cy - s), (cx + s, cy), (cx, cy + s)], fill=KEY_RGB, outline=(120, 90, 0))
            elif kind == "door":
                t = 0.12
                if _door_span(level, c, r) == "x":
                    box = (x0 + 0.06 * w, y0 + (0.5 - t) * hh, x1 - 0.06 * w, y0 + (0.5 + t) * hh)
                else:
                    box = (x0 + (0.5 - t) * w, y0 + 0.06 * hh, x0 + (0.5 + t) * w, y1 - 0.06 * hh)
                draw.rectangle(box, fill=look.colors["door"], outline=(60, 40, 20), width=SS)

    # Softlock / never-reached cells: diagonal hatching clipped to the cells (numpy mask over the image).
    arr = np.asarray(image).copy()
    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    col = np.floor((xx / SS - left) / cp).astype(np.int64)
    row = np.floor((yy / SS - top) / cp).astype(np.int64)
    inside = (col >= 0) & (col < level.cols) & (row >= 0) & (row < level.rows)
    stripe = ((xx + yy) % (8 * SS)) < (2 * SS)
    for mask, color in ((soft, SOFTLOCK_RGB), (unreached, UNREACHED_RGB)):
        if mask.any():
            cells = np.zeros(arr.shape[:2], dtype=bool)
            cells[inside] = mask[row[inside], col[inside]]
            arr[cells & stripe] = color
    image = Image.fromarray(arr)
    draw = ImageDraw.Draw(image)

    # Grid lines, thicker every 5 cells.
    for c in range(level.cols + 1):
        x = (left + c * cp) * SS
        draw.line([(x, top * SS), (x, (top + level.rows * cp) * SS)], fill=(70, 70, 70) if c % 5 == 0 else (150, 150, 150), width=SS)
    for r in range(level.rows + 1):
        y = (top + r * cp) * SS
        draw.line([(left * SS, y), ((left + level.cols * cp) * SS, y)], fill=(70, 70, 70) if r % 5 == 0 else (150, 150, 150), width=SS)
    for mask, color in ((soft, SOFTLOCK_RGB), (unreached, UNREACHED_RGB)):
        for r, c in zip(*np.nonzero(mask)):
            draw.rectangle(rect(int(c), int(r)), outline=color, width=2 * SS)

    # Objects: every footprint cell (filled when solid and in the way), an outline around their union;
    # cells under an overhang only (walkable) get a thin dashed-looking outline.
    edge = (80, 66, 40)
    for obj in level.objects:
        blocking = set(obj.cells)
        for c, r in _object_cells(obj):
            if (c, r) not in blocking:
                x0, y0, x1, y1 = rect(c, r, 0.18)
                draw.rectangle((x0, y0, x1, y1), outline=edge, width=SS)
                continue
            x0, y0, x1, y1 = rect(c, r, 0.0)
            if obj.solid:
                draw.rectangle((x0, y0, x1, y1), fill=OBJECT_RGB)
            for (dc, dr), line in (((-1, 0), (x0, y0, x0, y1)), ((1, 0), (x1, y0, x1, y1)),
                                   ((0, -1), (x0, y0, x1, y0)), ((0, 1), (x0, y1, x1, y1))):
                if (c + dc, r + dr) not in blocking:
                    draw.line(line, fill=edge, width=2 * SS)

    # Critical path: solid for walking/climbing, dashed for jumps.
    path = report["path"]
    if path.get("found"):
        run = [center(*path["cells"][0])]
        for mv in path["moves"]:
            a, b = center(*mv["from"]), center(*mv["to"])
            if mv["kind"] in ("jump", "hop"):
                if len(run) > 1:
                    draw.line(run, fill=PATH_RGB, width=3 * SS, joint="curve")
                _dashed(draw, a, b, PATH_RGB, 3 * SS, 5 * SS)
                run = [b]
            else:
                run.append(b)
        if len(run) > 1:
            draw.line(run, fill=PATH_RGB, width=3 * SS, joint="curve")

    # Marker discs.
    letters = report["map"]["markers"]
    marker_pos = {}
    for name, (c, r) in level.markers.items():
        cx, cy = center(c, r)
        rad = 0.32 * cp * SS
        draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=_marker_rgb(level, name), outline=(20, 20, 20), width=SS)
        marker_pos[name] = (cx / SS, cy / SS, rad / SS)

    out = image.reduce(SS)
    d = ImageDraw.Draw(out)
    font = _font(11)
    for c in range(level.cols):
        d.text((left + (c + 0.5) * cp, top - 9), str(c), font=font, fill=TEXT_RGB, anchor="mm")
    for r in range(level.rows):
        d.text((left - 6, top + (r + 0.5) * cp), str(r), font=font, fill=TEXT_RGB, anchor="rm")
    if cp >= 24:  # height steps in the cell corner (non-zero only)
        tiny = _font(max(8, cp // 4))
        for r, c in zip(*np.nonzero(level.heights != 0)):
            if level.kind(int(c), int(r)) != "pit":
                d.text((left + (c + 1) * cp - 2, top + (r + 1) * cp - 1), str(int(level.heights[r, c])), font=tiny,
                       fill=(30, 30, 30), anchor="rd")
    letter_font = _font(int(max(9, min(13, cp * 0.4))))
    for name, (x, y, _) in marker_pos.items():
        d.text((x, y), letters[name], font=letter_font, fill=(255, 255, 255), anchor="mm")
    labels = _Labels(out, _font(12))
    for x, y, rad in marker_pos.values():
        labels.block((x - rad - 1, y - rad - 1, x + rad + 1, y + rad + 1))
    for name, (x, y, rad) in marker_pos.items():
        labels.beside(x, y, rad, name, border=_marker_rgb(level, name))
    for door in level.doors.values():
        x, y = left + (door.col + 0.5) * cp, top + (door.row + 0.5) * cp
        labels.beside(x, y, cp * 0.5, door.name + (f" ({door.key})" if door.key else ""), border=look.colors["door"])
    for obj in level.objects:
        cs = [c for c, _ in _object_cells(obj)]
        rs = [r for _, r in _object_cells(obj)]
        labels.centered(left + (min(cs) + max(cs) + 1) / 2 * cp, top + (min(rs) + max(rs) + 1) / 2 * cp,
                        obj.asset + (" (placeholder)" if obj.placeholder else ""))
    labels.draw_all()
    lines =[f"{level.name} · top · 1 cell = {level.cell:g} m ({cp} px) · corner numbers = height steps of "
             f"{level.height_unit:g} m · row 0 = north"]
    lines += [f"warning: {w}" for w in level.warnings]
    legend = _legend(level, report, look, "object footprint" if level.objects else None)
    return _caption(out, lines, legend, look.background)


def _dashed(draw, a, b, fill, width, dash) -> None:
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    n = max(1, int(length // (2 * dash)))
    for i in range(n + 1):
        t0 = min(1.0, (2 * i * dash) / max(length, 1e-9))
        t1 = min(1.0, (2 * i * dash + dash) / max(length, 1e-9))
        if t0 >= 1.0:
            break
        draw.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=fill, width=width)


def render_level(level: Level, out_dir, report: dict | None = None) -> dict[str, Path]:
    """Write `iso.png` and `top.png` into out_dir; returns {"iso": path, "top": path}."""
    from .report import play

    report = report if report is not None else play(level)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    look = level_look(level)
    paths = {"iso": out / "iso.png", "top": out / "top.png"}
    render_iso(level, report, look).save(paths["iso"], optimize=False)
    render_top(level, report, look).save(paths["top"], optimize=False)
    return paths
