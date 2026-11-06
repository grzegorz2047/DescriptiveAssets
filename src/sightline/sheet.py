"""Contact sheets (SPEC 8): one asset (all views + part map under a header) and a library sheet
comparing many assets at one shared scale next to the 1.8 m human."""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image, ImageDraw

from . import render as R
from .camera import make_cameras, view_basis
from .errors import SightlineError
from .model import Scene

SHEET_BG = (236, 238, 241)
SUB_RGB = (88, 92, 100)
GAP = 14
EDGE = 16
TITLE_FONT = 22
SUB_FONT = 14
MAX_SHEET_WIDTH = 1800
MAX_LIBRARY_WIDTH = 3000


def _mat_count(asset, scene: Scene) -> int:
    return len({asset.parts[i].material for i in scene.solid_indices()})


def asset_summary(asset) -> dict:
    """Numbers for sheet headers: size (m), solid part, cut and material counts."""
    scene = asset.scene()
    solids = scene.solid_indices()
    lo, hi = scene.bounds(include_helpers=False) if solids else (np.zeros(3), np.zeros(3))
    w, h, d = (float(v) for v in hi - lo)
    return {
        "width": w, "height": h, "depth": d,
        "parts": len(solids),
        "cuts": sum(1 for p in asset.parts if not p.is_solid),
        "materials": _mat_count(asset, scene),
    }


def _header(lines: list[tuple[str, int, tuple]], width: int) -> Image.Image:
    """Stacked text lines (text, font size, color), wrapped to `width`."""
    rows = []
    for text, size, color in lines:
        fnt = R.font(size)
        for line in R._wrap(text.split(" "), fnt, width - 2 * EDGE, " ") if text else []:
            rows.append((line, fnt, size, color))
    height = EDGE + sum(size + 6 for _, _, size, _ in rows) + 4
    img = Image.new("RGB", (width, height), SHEET_BG)
    draw = ImageDraw.Draw(img)
    y = EDGE
    for line, fnt, size, color in rows:
        l, t, _, _ = R.text_box(line, fnt)
        draw.text((EDGE - l, y - t), line, font=fnt, fill=color)
        y += size + 6
    return img


def _shelf(tiles: list[Image.Image], max_width: int) -> Image.Image:
    """Pack tiles left to right into rows (bottom-aligned, so caption strips line up)."""
    rows: list[list[Image.Image]] = [[]]
    x = EDGE
    for tile in tiles:
        if rows[-1] and x + tile.width + EDGE > max_width:
            rows.append([])
            x = EDGE
        rows[-1].append(tile)
        x += tile.width + GAP
    width = max(EDGE * 2 + sum(t.width for t in row) + GAP * (len(row) - 1) for row in rows)
    height = EDGE + sum(max(t.height for t in row) + GAP for row in rows)
    out = Image.new("RGB", (width, height), SHEET_BG)
    y = 0
    for row in rows:
        row_h = max(t.height for t in row)
        x = EDGE
        for tile in row:
            out.paste(tile, (x, y + row_h - tile.height))
            x += tile.width + GAP
        y += row_h + GAP
    return out


def _stack(header: Image.Image, body: Image.Image) -> Image.Image:
    width = max(header.width, body.width)
    out = Image.new("RGB", (width, header.height + body.height), SHEET_BG)
    out.paste(header, (0, 0))
    out.paste(body, (0, header.height))
    return out


def make_sheet(asset, images: dict[str, Image.Image], part_map: Image.Image | None = None) -> Image.Image:
    """One-asset sheet: header (name, size, part count, style, brief) over the captioned views and
    the part map. `images` are presentation images from render.render_views / present_view."""
    tiles = list(images.values()) + ([part_map] if part_map is not None else [])
    if not tiles:
        raise SightlineError("make_sheet needs at least one image")
    body = _shelf(tiles, max(MAX_SHEET_WIDTH, max(t.width for t in tiles) + 2 * EDGE))
    s = asset_summary(asset)
    sep = R.separator()
    facts = sep.join([
        f"{s['width']:.2f} x {s['height']:.2f} x {s['depth']:.2f} m (w x h x d)",
        f"{s['parts']} parts",
        f"{s['cuts']} cuts",
        f"{s['materials']} materials",
        f"style {asset.style.name}",
        f"height = {s['height'] / R.HUMAN_HEIGHT:.2f} x human",
    ])
    lines = [(asset.name, TITLE_FONT, R.TEXT_RGB), (facts, SUB_FONT, R.TEXT_RGB)]
    if asset.brief:
        lines.append((" ".join(str(asset.brief).split()), SUB_FONT, SUB_RGB))
    return _stack(_header(lines, body.width), body)


def _frame_span(lay: R.Layout, view: str) -> tuple[float, float]:
    """Projected (width, height) in meters of a layout's framing box in a view."""
    lo, hi = lay.frame_min, lay.frame_max
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    _, right, up = view_basis(view)
    return float(np.ptp(corners @ right)), float(np.ptp(corners @ up))


def sheet_many(assets, out_path, res: int | None = None, view: str = "iso", *, root=None) -> pathlib.Path:
    """Library sheet: the `view` (default iso) of many assets side by side at ONE shared scale, each on
    its own ground tile (same grid), led by the 1.8 m human, with names and sizes under them.

    `assets`: Asset objects or names/paths (loaded with loader.load_asset(..., root)). `res`: pixel
    height of the tallest tile (default 320). Writes the PNG to `out_path` and returns that path.
    """
    sheet, _ = library(assets, res, view, root=root)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def library(assets, res: int | None = None, view: str = "iso", *, root=None) -> tuple[Image.Image, float]:
    """The library sheet image of sheet_many() and its shared meters per pixel.

    The scale makes the tallest tile `res` px high (or the widest tile fit the sheet width); tiles
    wrap into rows, and in each row every tile's ground origin sits on one baseline.
    """
    from .loader import load_asset
    from .model import Asset

    items = [a if isinstance(a, Asset) else load_asset(a, root) for a in assets]
    if not items:
        raise SightlineError("sheet_many needs at least one asset", hint="pass a list of assets or asset names")
    R._views([view], items[0].style)
    target = R._res(res, items[0].style) if res is not None else 320
    style0 = items[0].style
    human = Scene(R.figure_parts(0.0, 0.0))
    lays = [R.scene_layout(human, style0)] + [R.layout(a, scale_figure_on=False) for a in items]
    styles = [style0] + [a.style for a in items]
    spans = [_frame_span(lay, view) for lay in lays]
    margin = 0.04
    grow = 1 + 2 * margin
    mpp = max(max(h for _, h in spans) * grow / target, max(w for w, _ in spans) * grow / (MAX_LIBRARY_WIDTH - 2 * EDGE))

    tiles = []
    for lay, style in zip(lays, styles):
        cam = make_cameras([view], lay.frame_min, lay.frame_max, mpp=mpp, margin=margin)[view]
        r = R.render_view(lay.scene, style, cam, lay.ground, True, None, 0.012 * lay.size)
        alpha = np.where(r.coverage | r.ground, 255, 0).astype(np.uint8)
        tiles.append((Image.fromarray(np.dstack([r.rgb, alpha]), "RGBA"), R.origin_pixel(cam)[1], r.grid))

    labels = [("human", f"h {R.HUMAN_HEIGHT:.2f} m")]
    for a in items:
        s = asset_summary(a)
        labels.append((a.name, f"h {s['height']:.2f}  w {s['width']:.2f}  d {s['depth']:.2f} m"))
    name_font, sub_font = R.font(SUB_FONT), R.font(12)
    cols = [max(t.width, R.text_box(n, name_font)[2], R.text_box(sz, sub_font)[2]) + 2 * GAP
            for (t, _, _), (n, sz) in zip(tiles, labels)]

    rows: list[list[int]] = [[]]
    x = EDGE
    for i, col in enumerate(cols):
        if rows[-1] and x + col + EDGE > MAX_LIBRARY_WIDTH:
            rows.append([])
            x = EDGE
        rows[-1].append(i)
        x += col
    width = max(420, max(2 * EDGE + sum(cols[i] for i in row) for row in rows))

    bands = []
    for row in rows:
        # Every tile's ground origin (its anchor row) sits on one baseline, so heights compare directly.
        baseline = max(tiles[i][1] for i in row)
        tops = {i: int(round(baseline - tiles[i][1])) for i in row}
        label_y = max(tops[i] + tiles[i][0].height for i in row) + 6
        band = Image.new("RGBA", (width, label_y + 44), SHEET_BG + (255,))
        draw = ImageDraw.Draw(band)
        x = EDGE
        for i in row:
            tile, col = tiles[i][0], cols[i]
            band.alpha_composite(tile, (x + (col - tile.width) // 2, tops[i]))
            for k, (text, fnt, color) in enumerate(((labels[i][0], name_font, R.TEXT_RGB), (labels[i][1], sub_font, SUB_RGB))):
                l, t, tw, _ = R.text_box(text, fnt)
                draw.text((x + (col - tw) / 2 - l, label_y + k * 19 - t), text, font=fnt, fill=color)
            x += col
        bands.append(band.convert("RGB"))

    sep = R.separator()
    grid = tiles[1][2] if len(tiles) > 1 else tiles[0][2]
    facts = sep.join([view, f"1 px = {R.format_mpp(mpp)} m"] + ([f"grid {grid:g} m"] if grid > 0 else [])
                     + ["the grey figure is 1.80 m"])
    header = _header([(f"Library at shared scale ({len(items)} assets)", TITLE_FONT, R.TEXT_RGB),
                      (facts, SUB_FONT, R.TEXT_RGB)], width)
    out = Image.new("RGB", (width, header.height + sum(b.height for b in bands) + EDGE), SHEET_BG)
    out.paste(header, (0, 0))
    y = header.height
    for band in bands:
        out.paste(band, (0, y))
        y += band.height
    return out, mpp
