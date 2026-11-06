"""Rendering: pixels of shaded views (background, outlines, materials, shadows, grid, scale figure),
part maps, captions, sprites for levels, sheets, determinism and speed."""

import copy
import dataclasses
import pathlib
import time

import numpy as np
import pytest
from PIL import Image

from sightline.camera import view_basis
from sightline.errors import SightlineError
from sightline.loader import load_asset
from sightline.march import march
from sightline.render import (CAPTION_BG, HUMAN_HEIGHT, caption_text, label_parts, part_colors, part_map_buffers,
                              render_part_map, render_raw, render_sprite, render_views, write_renders)
from sightline.sheet import library, make_sheet, sheet_many

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def box1():
    return load_asset(FIX / "box1.yaml", root=ROOT)


@pytest.fixture(scope="module")
def tower():
    return load_asset(FIX / "tower30.yaml", root=ROOT)


def rgb255(color):
    return np.array([round(c * 255) for c in color])


def lum(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def silhouette_pixels(mask):
    """Mask pixels with at least one 4-neighbour outside the mask (image border counts as outside)."""
    pad = np.pad(mask, 1, constant_values=False)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return mask & ~inner


def asset_from_text(tmp_path, name, text):
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return load_asset(path, root=ROOT)


# ----------------------------------------------------------------------------- shaded views


def test_background_where_empty_and_ground_below(box1):
    r = render_raw(box1, ["front", "iso"], res=128)
    bg = rgb255(box1.style.background)
    for view in ("front", "iso"):
        img = r[view].rgb
        empty = ~(r[view].coverage | r[view].ground)
        assert empty.sum() > 0.2 * empty.size
        assert np.all(img[empty] == bg)  # every empty pixel is exactly the style background
        assert np.all(img[0, 0] == bg) and np.all(img[0, -1] == bg)
    # Front view: the ground slab is a strip under the box, never above the box's top.
    front = r["front"]
    rows = np.nonzero(front.ground.any(axis=1))[0]
    box_rows = np.nonzero((front.part == 0).any(axis=1))[0]
    assert rows.min() >= box_rows.max() - 1


def test_outline_on_silhouette_and_off_when_disabled(box1):
    r = render_raw(box1, ["front", "iso"], res=128, scale_figure=False)
    line = rgb255(box1.style.outline_color)
    for view, vr in r.items():
        edge = silhouette_pixels(vr.coverage)
        assert edge.sum() > 100
        assert np.all(vr.rgb[edge] == line), view  # every silhouette pixel has the outline color
        interior = vr.coverage & ~edge
        assert (np.all(vr.rgb[interior] == line, axis=1)).mean() < 0.05  # the inside is shaded, not lines
    plain = render_raw(box1, ["iso"], res=128, scale_figure=False, outline=False)["iso"]
    assert not np.all(plain.rgb == line, axis=-1).any()
    assert np.array_equal(plain.coverage, plain.part >= 0)


def test_outline_width_from_style(box1):
    thick = copy.copy(box1)
    thick.style = dataclasses.replace(box1.style, outline_width=3)
    line = rgb255(box1.style.outline_color)
    n1 = np.all(render_raw(box1, ["front"], res=128, scale_figure=False)["front"].rgb == line, axis=-1).sum()
    n3 = np.all(render_raw(thick, ["front"], res=128, scale_figure=False)["front"].rgb == line, axis=-1).sum()
    assert 2.5 * n1 < n3 < 3.1 * n1  # a 3 px band inside the silhouette: ~3x the 1 px outline


def test_red_material_renders_reddish_and_patterns_vary(tmp_path):
    asset = asset_from_text(tmp_path, "redbox", """
asset: redbox
parts:
  - {name: cloth, shape: box, size: [1, 1, 1], material: cloth_red, align: bottom}
  - {name: wall, shape: box, size: [1, 1, 1], material: brick, at: [1.5, 0, 0], align: bottom}
""")
    r = render_raw(asset, ["front"], res=128, scale_figure=False)["front"]
    edge = silhouette_pixels(r.part >= 0)
    red = r.rgb[(r.part == 0) & ~edge].astype(int)
    mean = red.mean(axis=0)
    assert mean[0] > mean[1] + 50 and mean[0] > mean[2] + 50
    assert len(np.unique(red, axis=0)) >= 2  # noise pattern: more than one shade
    # Bricks: mortar lines make a bimodal lightness inside the brick wall (cream mortar on red brick).
    wall = lum(r.rgb[(r.part == 1) & ~edge])
    assert (wall > wall.mean() + 25).mean() > 0.05


def _shadow_area_expected(rect, light, box_lo, box_hi, step=0.004):
    """Ground area (m^2) on the tile outside the box footprint whose path to the light hits the box."""
    x0, x1, z0, z1 = rect
    xs = np.arange(x0 + step / 2, x1, step)
    zs = np.arange(z0 + step / 2, z1, step)
    gx, gz = np.meshgrid(xs, zs)
    to_light = -np.asarray(light) / -light[1]  # per meter of height
    t = np.linspace(0, box_hi[1], 64)[:, None, None]
    px, pz = gx + to_light[0] * t, gz + to_light[2] * t
    inside = ((px >= box_lo[0]) & (px <= box_hi[0]) & (pz >= box_lo[2]) & (pz <= box_hi[2])).any(axis=0)
    footprint = (gx >= box_lo[0]) & (gx <= box_hi[0]) & (gz >= box_lo[2]) & (gz <= box_hi[2])
    return float((inside & ~footprint).sum()) * step * step


def test_shadow_darker_than_lit_ground_with_the_right_area(box1):
    r = render_raw(box1, ["top"], res=256, scale_figure=False)["top"]
    ground = r.ground
    shadow = ground & (r.light < 0.5)
    lit = ground & (r.light > 0.99)
    assert shadow.sum() > 200 and lit.sum() > 200
    assert lum(r.rgb[shadow]).mean() < 0.85 * lum(r.rgb[lit]).mean()  # visible ...
    assert lum(r.rgb[shadow]).mean() > 0.4 * lum(r.rgb[lit]).mean()  # ... but soft
    # The light travels +x, -z: the shadow lies right of and behind (-z) the box.
    rows, cols = np.nonzero(shadow)
    pts = r.camera.pixel_to_world(cols + 0.5, rows + 0.5)
    assert pts[:, 0].mean() > 0.3 and pts[:, 2].mean() < -0.3
    from sightline.render import layout
    rect = layout(box1, scale_figure_on=False).ground.rect
    expected = _shadow_area_expected(rect, box1.style.light_dir, [-1, 0, -0.5], [1, 1, 0.5])
    measured = shadow.sum() * r.mpp ** 2
    assert abs(measured - expected) / expected < 0.08, (measured, expected)


def test_ground_grid_lines_major_every_5m(box1):
    r = render_raw(box1, ["top"], res=256, scale_figure=False)["top"]
    cam = r.camera
    # A row across the uncovered tile strip in front of the box (z = 0.62 m).
    px, _ = cam.project(np.array([[-1.2, 0, 0.62]]))
    row = int(px[0, 1])
    profile = lum(r.rgb[row].astype(float))
    on_ground = r.ground[row]
    xs = cam.pixel_to_world(np.arange(cam.width) + 0.5, np.full(cam.width, row + 0.5))[:, 0]

    def darkest_near(x):
        sel = on_ground & (np.abs(xs - x) < 2.5 * cam.mpp)
        return profile[sel].min()

    plain = np.median(profile[on_ground])
    assert darkest_near(0.0) < 0.75 * plain  # x = 0: a major line (multiple of 5 m)
    assert 0.75 * plain <= darkest_near(1.0) < 0.92 * plain  # x = 1: a minor line
    assert darkest_near(0.5) > 0.97 * plain  # no line between grid lines


def test_scale_figure_size_place_and_no_occlusion(box1):
    r = render_raw(box1, ["front", "right", "iso"], res=256)
    helper = len(box1.parts)  # helper parts are appended after the asset's parts
    front = r["front"]
    rows, cols = np.nonzero(front.part >= helper)
    height = (rows.max() - rows.min() + 1) * front.mpp
    assert abs(height - HUMAN_HEIGHT) <= 2.5 * front.mpp
    x = front.camera.pixel_to_world(cols + 0.5, rows + 0.5)[:, 0].mean()
    assert abs(x - (1.0 + 0.6)) < 0.05  # center x = asset max x + 0.6 m
    # The figure never hides the asset: the same rays hit the box with and without it.
    for view in ("front", "right", "iso"):
        alone = march(box1.scene(), r[view].camera)
        assert (r[view].buffers.part == 0).sum() == (alone.part == 0).sum() > 1000, view
        assert np.array_equal(r[view].buffers.part == 0, alone.part == 0), view


def test_captions_upscale_and_scale_text(box1):
    raws = render_raw(box1, ["front", "iso"], res=128)
    images = render_views(box1, ["front", "iso"], res=128)
    for view, img in images.items():
        cam = raws[view].camera
        assert img.width >= 2 * cam.width  # small renders are shown x2
        assert img.height > 2 * cam.height + 12  # a caption strip under the view
        arr = np.asarray(img)
        strip = arr[2 * cam.height + 2:]
        assert (np.all(strip == CAPTION_BG, axis=-1)).mean() > 0.6
        assert (lum(strip) < 100).sum() > 30  # dark text in the strip
        # The upscaled view keeps exact pixel blocks: 2x2 blocks repeat the rendered pixel.
        x0 = (img.width - 2 * cam.width) // 2  # narrow images are padded to fit the caption
        assert np.array_equal(arr[: 2 * cam.height : 2, x0: x0 + 2 * cam.width : 2], raws[view].rgb)
    big = render_views(box1, ["front"], res=512, caption=False)["front"]
    cam = render_raw(box1, ["front"], res=512)["front"].camera
    assert big.size == (cam.width, cam.height) and max(big.size) >= 384  # no upscale for big renders
    assert caption_text("box1", "front", 0.0234, 1.0) in ("box1 · front · 1 px = 0.023 m · grid 1 m",
                                                          "box1 | front | 1 px = 0.023 m | grid 1 m")
    assert "0.0086" in caption_text("a", "top", 0.00857, 0.0) and "grid" not in caption_text("a", "top", 0.00857, 0.0)


def test_orbit_view_and_ground_off(box1):
    r = render_raw(box1, ["orbit:200:20"], res=96, ground=False)["orbit:200:20"]
    assert not r.ground.any() and r.coverage.any() and r.grid == 0.0


# ----------------------------------------------------------------------------- part map


def test_part_map_has_one_flat_color_per_part(tower):
    vb, order = part_map_buffers(tower, "front", res=200)
    img = np.asarray(render_part_map(tower, "front", res=200, labels=False, outline=False, caption=False, upscale=1))
    assert img.shape[:2] == vb.part.shape
    colors = part_colors(len(order))
    visible = [pi for pi in order if (vb.part == pi).any()]
    assert len(visible) >= 20
    for k, pi in enumerate(order):
        sel = vb.part == pi
        if sel.any():
            assert np.all(img[sel] == colors[k]), tower.parts[pi].name
    hit_colors = np.unique(img[vb.part >= 0], axis=0)
    assert len(hit_colors) == len(visible)  # exactly one color per visible part
    assert np.all(img[vb.part < 0] == rgb255(tower.style.background))


def test_part_colors_distinct_and_stable():
    a = part_colors(64)
    assert len(set(a)) == 64 and a == part_colors(64) and a[:10] == part_colors(10)
    arr = np.array(a[:32], dtype=float)
    d = np.sqrt(((arr[:, None] - arr[None]) ** 2).sum(-1)) + np.eye(32) * 999
    assert d.min() > 20  # the first 32 colors are at least 20 RGB units apart


def test_part_map_labels_do_not_overlap_and_lead_to_their_part():
    ids = np.full((60, 60), -1, dtype=np.int32)
    for i in range(12):  # 12 tiny parts in a tight cluster: most labels must move away
        r, c = divmod(i, 4)
        ids[24 + 3 * r: 26 + 3 * r, 22 + 4 * c: 25 + 4 * c] = i
    ids[0:12, 0:60] = 12  # one big part
    parts = list(range(13))
    names = [f"part_{i}" for i in parts]
    k = 4
    img = Image.fromarray(np.zeros((240, 240, 3), np.uint8))
    out, boxes = label_parts(img, ids, parts, names, part_colors(13), k, (255, 255, 255))
    assert len(boxes) == 13
    for i, a in enumerate(boxes):
        assert 0 <= a[0] < a[2] <= out.width and 0 <= a[1] < a[3] <= out.height
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)


def test_part_map_lists_hidden_parts_and_has_labels(tower):
    img = render_part_map(tower, "front", res=200)
    vb, _ = part_map_buffers(tower, "front", res=200)
    assert img.width > vb.camera.width * 2 and img.height > vb.camera.height * 2 + 20


# ----------------------------------------------------------------------------- sprites


def test_sprite_rgba_transparent_and_anchor_at_ground_origin(box1):
    mpp = 0.02
    img, (ax, ay) = render_sprite(box1, view="iso", mpp=mpp)
    assert img.mode == "RGBA"
    alpha = np.asarray(img)[..., 3]
    assert set(np.unique(alpha)) <= {0, 255}
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0
    rows, cols = np.nonzero(alpha)
    _, right, up = view_basis("iso")
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (0, 1) for z in (-0.5, 0.5)])
    u, v = corners @ right / mpp, corners @ up / mpp
    # Opaque extent = projected box (+- half a pixel of hit threshold) around the anchor.
    assert abs((cols.min() - ax) - u.min()) <= 1.5 and abs((cols.max() + 1 - ax) - u.max()) <= 1.5
    assert abs((ay - rows.min()) - v.max()) <= 1.5 and abs((ay - rows.max() - 1) - v.min()) <= 1.5
    # No ground and no scale figure in a sprite.
    assert alpha.sum() / 255 < 1.1 * (u.max() - u.min()) * (v.max() - v.min())


def test_sprite_default_mpp_and_bad_input(box1):
    img, anchor = render_sprite(box1)
    assert img.width > 50 and len(anchor) == 2
    with pytest.raises(SightlineError, match="did you mean 'iso'"):
        render_sprite(box1, view="isso", mpp=0.02)
    with pytest.raises(SightlineError, match="mpp"):
        render_sprite(box1, mpp=-1)


def test_invalid_input_raises_sightline_errors(box1):
    with pytest.raises(SightlineError, match="did you mean 'front'"):
        render_views(box1, ["frnt"])
    with pytest.raises(SightlineError, match="res"):
        render_views(box1, ["front"], res=3)
    with pytest.raises(SightlineError, match="res"):
        render_raw(box1, ["front"], res="big")
    with pytest.raises(SightlineError, match="no views"):
        render_views(box1, [])


# ----------------------------------------------------------------------------- files, sheets


def test_write_renders_files_deterministic_and_fast(tower, tmp_path):
    t0 = time.perf_counter()
    paths = write_renders(tower, tmp_path / "a", res=256)
    elapsed = time.perf_counter() - t0
    assert elapsed < 15.0, f"write_renders took {elapsed:.1f} s"
    assert set(paths) == {"front", "right", "top", "iso", "parts", "sheet"}
    sizes = {}
    for key, path in paths.items():
        assert path.is_file() and path.suffix == ".png"
        with Image.open(path) as im:
            sizes[key] = im.size
    assert sizes["sheet"][0] > sizes["iso"][0] and sizes["sheet"][1] > sizes["iso"][1]
    again = write_renders(tower, tmp_path / "b", res=256)
    for key in paths:
        assert paths[key].read_bytes() == again[key].read_bytes(), key  # byte-identical PNGs


def test_render_views_deterministic(box1):
    a = render_views(box1, res=128)
    b = render_views(box1, res=128)
    assert list(a) == ["front", "right", "top", "iso"]  # style default views, in order
    for view in a:
        assert a[view].tobytes() == b[view].tobytes()


def test_make_sheet_contains_every_tile(box1):
    images = render_views(box1, res=96)
    pm = render_part_map(box1, "iso", res=96)
    sheet = make_sheet(box1, images, pm)
    tiles = list(images.values()) + [pm]
    assert sheet.width >= max(t.width for t in tiles)
    assert sheet.height > max(t.height for t in tiles) + 40  # header above the tiles


def test_library_shares_one_scale(box1, tower, tmp_path):
    _, mpp_small = library([box1], res=200)
    _, mpp_both = library([box1, tower], res=200)
    assert mpp_both > 2.5 * mpp_small  # the tall tower sets the shared scale
    from sightline.render import layout
    lay = layout(tower, scale_figure_on=False)
    _, right, up = view_basis("iso")
    corners = np.array([[x, y, z] for x in (lay.frame_min[0], lay.frame_max[0])
                        for y in (lay.frame_min[1], lay.frame_max[1]) for z in (lay.frame_min[2], lay.frame_max[2])])
    height = np.ptp(corners @ up)
    assert mpp_both == pytest.approx(height * 1.08 / 200, rel=1e-6)  # tallest tile = res px (+ margins)
    out = sheet_many([box1, tower], tmp_path / "lib" / "library.png", res=200)
    with Image.open(out) as im:
        assert im.height > 200 and im.width > 300


def test_grazing_surfaces_are_not_filled_with_outline(tmp_path):
    """A plate seen 8 degrees above its plane changes depth by 7 px per pixel row: with a fixed 4-px jump
    threshold every row was an outline (below about 14 degrees surfaces filled black). The threshold now
    adds the depth change the surface's own slope explains."""
    asset = asset_from_text(tmp_path, "plate", "asset: plate\nparts:\n"
                            "  - {name: plate, shape: box, size: [6, 0.1, 6], material: stone, align: bottom}\n"
                            "  - {name: post, shape: box, size: [0.3, 1.5, 0.3], material: wood, on: plate.top}\n")
    view = render_raw(asset, ["orbit:20:8"], 256, scale_figure=False, ground=False)["orbit:20:8"]
    line = np.array(rgb255(asset.style.outline_color), dtype=np.uint8)
    inside = view.part == 0
    for _ in range(2):
        inside &= np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
    dark = np.all(view.rgb == line, axis=-1)
    assert inside.sum() > 500
    assert (dark & inside).sum() <= 0.02 * inside.sum()
    # A part standing in front of the grazing plate still gets its outline.
    post = view.part == 1
    edge = post & ~(np.roll(post, 1, 1) & np.roll(post, -1, 1))
    assert (dark & edge).sum() >= 0.8 * edge.sum()


def test_coplanar_overlaps_and_blend_seams_draw_no_scribbles(tmp_path):
    """Two overlapping boxes with one top face drew a scribble of alternating part pixels (whichever was
    traced first won each pixel), and members of a blend group drew their construction seam as a line."""
    asset = asset_from_text(tmp_path, "slabs", "asset: slabs\nparts:\n"
                            "  - {name: a, shape: box, size: [3, 0.4, 2], material: stone, at: [-0.8, 0, 0], align: bottom}\n"
                            "  - {name: b, shape: box, size: [3, 0.4, 2], material: stone, at: [0.8, 0, 0], align: bottom}\n"
                            "  - {name: c, shape: box, size: [1, 0.4, 2], material: wood, at: [2.8, 0, 0], align: bottom}\n"
                            "  - {name: hull, shape: sphere, radius: 0.6, material: stone, at: [0, 1.4, -2.5]}\n"
                            "  - {name: nose, shape: sphere, radius: 0.5, material: stone, at: [0.7, 1.4, -2.5], blend: 0.3,\n"
                            "     blend_with: hull}\n")
    view = render_raw(asset, ["top"], 256, scale_figure=False, ground=False)["top"]
    style = asset.style
    line = np.array(rgb255(style.outline_color), dtype=np.float64)
    part = view.part
    names = {p.name: i for i, p in enumerate(asset.parts)}
    # The overlap x -0.7..0.7 of a and b: one part owns it, in one piece (no alternating pixels).
    cam = view.camera
    lo = int(cam.project(np.array([[-0.65, 0.4, 0]]))[0][0][0])
    hi = int(cam.project(np.array([[0.65, 0.4, 0]]))[0][0][0])
    r0 = int(cam.project(np.array([[0, 0.4, -0.8]]))[0][0][1])  # top view: image up is -z
    r1 = int(cam.project(np.array([[0, 0.4, 0.8]]))[0][0][1])
    hull_rows = part[r0:r1, lo:hi]
    slab = np.isin(hull_rows, [names["a"], names["b"]])
    owners = np.unique(hull_rows[slab])
    assert len(owners) == 1, owners
    # No inner line between same-material faces that continue each other (slabs a/b, hull/nose) ...
    rgb = view.rgb.astype(np.float64)
    darkened = np.abs(rgb - line).sum(axis=-1) < np.abs(rgb - np.array(rgb255(style.material("stone").color))).sum(axis=-1)
    seam_cols = [int(cam.project(np.array([[x, 0.4, 0]]))[0][0][0]) for x in (0.7, -0.7)]
    for col in seam_cols:
        assert darkened[r0:r1, col - 1:col + 2].sum() <= 2
    # ... but a material change keeps its line (b against the wood block c at x = 2.3).
    col = int(cam.project(np.array([[2.3, 0.4, 0]]))[0][0][0])
    assert darkened[r0:r1, col - 1:col + 2].sum() >= 0.6 * (r1 - r0)
    # The blend seam between hull and nose (x = 0.35 at z = -2.5) draws no line either.
    col = int(cam.project(np.array([[0.35, 2.0, -2.5]]))[0][0][0])
    row = int(cam.project(np.array([[0.35, 2.0, -2.5]]))[0][0][1])
    assert darkened[row - 4:row + 5, col - 1:col + 2].sum() <= 1
