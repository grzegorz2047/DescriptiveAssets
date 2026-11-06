"""Reference images: segmentation, measurements, model matching, match images and view fitting."""

import pathlib
import re
import shutil
import sys
import textwrap
from collections import deque

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sightline.camera import make_cameras
from sightline.errors import SightlineError
from sightline.loader import load_asset, load_asset_text
from sightline.march import march
from sightline.reference import (
    analyze_reference,
    delta_e,
    fit_view,
    format_fit,
    format_match,
    format_reference,
    label_components,
    match_asset,
    shade_lambert,
    srgb_to_lab,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
TMP = ROOT / "out" / "test_tmp" / "reference"

# ----------------------------------------------------------------------------- concept-art fixtures

W, H = 200, 300
BG = (232, 228, 218)
COLORS = {  # element -> flat colour
    "base": (128, 128, 124),
    "shaft": (154, 106, 60),
    "roof": (180, 70, 47),
    "door": (70, 48, 36),
    "finial": (224, 176, 50),
    "pole": (40, 40, 40),
}
SHAPES = [  # drawn in this order; later shapes cover earlier ones
    ("base", "rect", (40, 260, 159, 279)),
    ("shaft", "rect", (60, 110, 139, 259)),
    ("roof", "poly", [(50, 110), (150, 110), (100, 40)]),
    ("door", "rect", (88, 215, 111, 259)),
    ("finial", "ellipse", (90, 20, 110, 40)),
]
POLE = ("pole", "rect", (141, 60, 142, 109))  # 2 px wide: a thin detail
BBOX = (40, 20, 160, 280)  # x0, y0, x1, y1 of the drawn tower (ends exclusive)


def draw_tower(mode="RGB", background=BG, pole=False, noise=None, specks=0):
    """Flat-colour 'concept art' of a tower: (image, element id per pixel, element names)."""
    if noise is not None:
        rng = np.random.default_rng(3)
        arr = np.clip(np.array(background, float) + rng.normal(0, noise, (H, W, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert(mode)
    else:
        img = Image.new(mode, (W, H), background)
    ids = Image.new("L", (W, H), 0)
    d, di = ImageDraw.Draw(img), ImageDraw.Draw(ids)
    shapes = SHAPES + ([POLE] if pole else [])
    for n, (name, kind, geom) in enumerate(shapes, 1):
        color = COLORS[name] + ((255,) if mode == "RGBA" else ())
        for draw, fill in ((d, color), (di, n)):
            {"rect": draw.rectangle, "poly": draw.polygon, "ellipse": draw.ellipse}[kind](geom, fill=fill)
    for s in range(specks):  # 3 x 3 px specks along the top, far below 0.5% of the image
        x = 10 + 30 * s
        d.rectangle((x, 5, x + 2, 7), fill=(20, 20, 20) + ((255,) if mode == "RGBA" else ()))
    return img, np.asarray(ids), [s[0] for s in shapes]


def save(img, name) -> pathlib.Path:
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / name
    img.save(path)
    return path


@pytest.fixture(scope="module")
def concept():
    img, ids, names = draw_tower()
    return save(img, "concept.png"), ids, names


@pytest.fixture
def scratch(request):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name).strip("_")[:60]
    path = TMP / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


# ----------------------------------------------------------------------------- components


def bfs_labels(mask):
    """Reference implementation: 4-connected BFS labels in row-major order of first pixel."""
    h, w = mask.shape
    out = np.full((h, w), -1, int)
    n = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and out[r, c] < 0:
                out[r, c] = n
                q = deque([(r, c)])
                while q:
                    y, x = q.popleft()
                    for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and out[yy, xx] < 0:
                            out[yy, xx] = n
                            q.append((yy, xx))
                n += 1
    return out, n


def test_label_components_matches_bfs():
    rng = np.random.default_rng(11)
    for density in (0.3, 0.5, 0.58, 0.7):
        mask = rng.random((37, 53)) < density
        labels, sizes = label_components(mask)
        expected, n = bfs_labels(mask)
        assert np.array_equal(labels, expected) and len(sizes) == n
        assert np.array_equal(sizes, np.bincount(expected[expected >= 0]))
    # A spiral corridor: one component that needs many union rounds; diagonal touches do not connect.
    spiral = np.zeros((31, 31), bool)
    lo, hi = 0, 30
    while lo <= hi:
        spiral[lo, lo:hi + 1] = spiral[lo:hi + 1, hi] = spiral[hi, lo:hi + 1] = True
        spiral[lo + 2:hi + 1, lo] = True
        lo, hi = lo + 2, hi - 2
        if lo <= hi:
            spiral[lo - 1, lo] = True
    labels, sizes = label_components(spiral)
    expected, n = bfs_labels(spiral)
    assert np.array_equal(labels, expected) and len(sizes) == n
    diag = np.eye(5, dtype=bool)
    assert len(label_components(diag)[1]) == 5
    empty_labels, empty_sizes = label_components(np.zeros((4, 4), bool))
    assert empty_sizes.size == 0 and (empty_labels == -1).all()


def test_lab_and_delta_e():
    assert np.allclose(srgb_to_lab([1, 1, 1]), [100, 0, 0], atol=0.01)
    assert np.allclose(srgb_to_lab([0, 0, 0]), [0, 0, 0], atol=0.01)
    assert srgb_to_lab([1, 0, 0])[0] == pytest.approx(53.24, abs=0.05)
    assert delta_e([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.0
    assert delta_e([0, 0, 0], [1, 1, 1]) == pytest.approx(100, abs=0.01)


# ----------------------------------------------------------------------------- analysis


def test_analyze_concept_art(concept):
    path, ids, names = concept
    r = analyze_reference(path, root=ROOT)
    m = r["mask"]
    assert r["image"] == "out/test_tmp/reference/concept.png" and r["size_px"] == [W, H] and r["view"] == "front"
    assert m["source"] == "background" and m["background"] == "#e8e4da" and m["confidence"] == "high"
    assert m["components"] == 1 and m["bbox_px"] == list(BBOX)
    assert m["coverage"] == pytest.approx((ids > 0).sum() / (W * H), abs=1e-4)
    p = r["proportions"]
    assert p["height_to_width"] == pytest.approx(260 / 120, abs=1e-3)
    assert p["symmetry_lr"] > 0.97 and p["fill"] == pytest.approx((ids > 0).sum() / (120 * 260), abs=1e-3)
    # No height: scale unknown, sizes in % of the height.
    assert r["scale"]["mpp"] is None and r["scale"]["assumed"] and r["scale"]["unit"] == "%"
    assert r["size"]["height"] == 100 and r["size"]["width"] == pytest.approx(120 / 260 * 100, abs=0.5)
    # Width profile: band 50-60% (rows 150..176) is the shaft only: 80 px wide from x = 60.
    band = r["width_profile"][5]
    assert band["band"] == "50-60%"
    assert band["width"] == pytest.approx(80 / 260 * 100, abs=0.6) and band["left"] == pytest.approx(-40 / 260 * 100, abs=0.6)
    assert r["width_profile"][9]["width"] == pytest.approx(120 / 260 * 100, abs=0.6)  # the base
    # Colours: every drawn colour is found with its share (k-means, dE < 2, share +-1%).
    total = (ids > 0).sum()
    for n, name in enumerate(names, 1):
        rgb = np.array(COLORS[name]) / 255
        best = min(r["colors"], key=lambda c: delta_e(rgb, np.array(Image.new("RGB", (1, 1), c["hex"]).getpixel((0, 0))) / 255))
        found = np.array(Image.new("RGB", (1, 1), best["hex"]).getpixel((0, 0))) / 255
        assert delta_e(rgb, found) < 2.0, name
        assert best["share"] == pytest.approx((ids == n).sum() / total, abs=0.01), name
    by_hex = {c["hex"]: c for c in r["colors"]}
    roof = by_hex["#b4462f"]
    assert roof["where"] == "rows 10-40% (top)" and roof["nearest_material"] == "roof" and roof["material_delta_e"] < 1
    door = by_hex["#463024"]
    assert door["center"] == [0.0, pytest.approx((280 - 237.5) / 260 * 100, abs=1)]
    # Unknowns: hidden sides of a front view, the scale and materials; the outline is fine.
    topics = [q["topic"] for q in r["unknowns"]]
    assert topics[:2] == ["hidden sides", "scale"] and "materials" in topics and "mask" not in topics
    assert all(q["question"].endswith("?") and q["default"] for q in r["unknowns"])


def test_silhouette_grid_measures_the_object(concept):
    path, ids, _ = concept
    r = analyze_reference(path, height=13.0)
    g = r["silhouette"]
    assert g["unit"] == "m" and g["cols"] <= 32 and g["rows"] <= 48
    assert all(len(row) == g["cols"] and set(row) <= {"#", "."} for row in g["grid"])
    cell = g["cell"]
    assert cell in (0.25, 0.5, 0.2, 0.3)  # a readable 1-2-2.5-5 step
    # Filled cells approximate the object's area (m^2); the ground is the bottom edge of the last row.
    area = (ids > 0).sum() * 0.05**2
    filled = sum(row.count("#") for row in g["grid"]) * cell**2
    assert filled == pytest.approx(area, rel=0.08)
    assert g["y0"] - g["rows"] * cell == pytest.approx(0.0, abs=1e-9)
    # The column centered on x = 0 is filled from the finial down to the ground.
    c0 = round(-g["x0"] / cell - 0.5)
    assert g["x0"] + (c0 + 0.5) * cell == pytest.approx(0.0, abs=1e-9)
    column = "".join(row[c0] for row in g["grid"])
    assert column.strip(".") == "#" * len(column.strip("."))
    text = format_reference(r)
    rows = [ln for ln in text.splitlines() if re.match(r"^\s+-?\d+\.\d\d  [#.]( [#.])*$", ln)]
    assert len(rows) == g["rows"] and all(len(ln.split()) - 1 == g["cols"] for ln in rows)
    assert re.search(r"\^ +.*\n +.*-?\d\.\d\d.*x m", text)


def test_analyze_with_height_gives_meters(concept):
    path, _, _ = concept
    r = analyze_reference(path, height=13.0)
    assert r["scale"]["mpp"] == pytest.approx(0.05) and not r["scale"]["assumed"] and r["scale"]["unit"] == "m"
    assert r["size"]["height"] == 13.0 and r["size"]["width"] == pytest.approx(6.0)
    band = r["width_profile"][5]
    assert (band["left"], band["right"], band["width"]) == (pytest.approx(-2.0), pytest.approx(2.0), pytest.approx(4.0))
    assert band["top"] == pytest.approx((280 - 150) * 0.05) and band["bottom"] == pytest.approx((280 - 176) * 0.05)
    door = next(c for c in r["colors"] if c["hex"] == "#463024")
    assert door["center"][0] == pytest.approx(0.0, abs=0.03) and door["center"][1] == pytest.approx(2.125, abs=0.03)
    assert "scale" not in [q["topic"] for q in r["unknowns"]]
    hidden = r["unknowns"][0]
    assert hidden["topic"] == "hidden sides" and "6.00 m" in hidden["default"]
    text = format_reference(r)
    assert "SIZE       13.00 m tall x 6.00 m wide" in text and "1 px = 0.0500 m" in text


def test_transparent_png_uses_alpha(concept):
    path, ids, _ = concept
    img, _, _ = draw_tower(mode="RGBA", background=(0, 0, 0, 0))
    cut = analyze_reference(save(img, "concept_cutout.png"))
    opaque = analyze_reference(path)
    assert cut["mask"]["source"] == "alpha" and cut["mask"]["confidence"] == "high"
    assert cut["mask"]["bbox_px"] == opaque["mask"]["bbox_px"] and cut["mask"]["coverage"] == opaque["mask"]["coverage"]
    assert cut["silhouette"]["grid"] == opaque["silhouette"]["grid"]
    assert [c["hex"] for c in cut["colors"]] == [c["hex"] for c in opaque["colors"]]


def test_noisy_background_gives_low_confidence():
    img, ids, _ = draw_tower(background=(150, 150, 150), noise=35)
    r = analyze_reference(save(img, "noisy.png"))
    m = r["mask"]
    assert m["source"] == "background" and m["confidence"] == "low" and m["border_spread"] > 14
    assert any("not plain" in why for why in m["reasons"])
    mask_q = next(q for q in r["unknowns"] if q["topic"] == "mask")
    assert "cut-out PNG" in mask_q["question"] and mask_q["default"]
    assert "confidence low" in format_reference(r)


def test_specks_dropped_and_bottom_edge_question():
    img, ids, _ = draw_tower(specks=5)
    r = analyze_reference(save(img, "specks.png"))
    assert r["mask"]["components"] == 1 and r["mask"]["dropped_specks"] == 5
    assert r["mask"]["coverage"] == pytest.approx((ids > 0).sum() / (W * H), abs=1e-4)
    # A picture cropped at the ground line: still high confidence, but ask whether it is cut off.
    cropped = Image.new("RGB", (W, 270), BG)
    cropped.paste(draw_tower()[0].crop((0, 0, W, 270)))
    rc = analyze_reference(save(cropped, "cropped.png"))
    assert rc["mask"]["confidence"] == "high" and rc["mask"]["bottom_touch"] > 0
    assert any(q["topic"] == "cropped" and "bottom edge" in q["question"] for q in rc["unknowns"])


def test_thin_details_are_located():
    img, ids, _ = draw_tower(pole=True)
    r = analyze_reference(save(img, "pole.png"), height=13.0)
    assert len(r["thin_details"]) == 1
    t = r["thin_details"][0]
    # The pole is 2 px wide at x = 141..142 px and free of the roof from row 60 down to about row 97.
    assert t["width_px"] == pytest.approx(2.0, abs=0.3) and 30 <= t["length_px"] <= 40
    assert t["x"] == [pytest.approx((141 - 100) * 0.05, abs=0.06), pytest.approx((143 - 100) * 0.05, abs=0.06)]
    assert t["y"][1] == pytest.approx((280 - 60) * 0.05, abs=0.06)
    q = next(q for q in r["unknowns"] if q["topic"] == "thin details")
    assert "gameplay distance" in q["question"] and "x 2.05..2.15" in q["question"]
    assert "THIN DETAILS" in format_reference(r)


def test_committed_concept_fixture_reads():
    """tests/fixtures/refs/concept_tower.png: flat concept art (with a 2 px pole) for demos and the CLI."""
    r = analyze_reference(FIX / "refs" / "concept_tower.png", root=ROOT)
    assert r["mask"]["confidence"] == "high" and r["mask"]["components"] == 1
    assert r["proportions"]["height_to_width"] == pytest.approx(260 / 120, abs=1e-3)
    assert len(r["colors"]) >= 5 and len(r["thin_details"]) == 1


def test_mask_file_and_input_errors(concept, scratch):
    path, ids, _ = concept
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle((50, 50, 149, 249), fill=255)
    mask.save(scratch / "mask.png")
    r = analyze_reference(path, mask=scratch / "mask.png")
    assert r["mask"]["source"] == "mask file" and r["mask"]["bbox_px"] == [50, 50, 150, 250]
    assert r["mask"]["coverage"] == pytest.approx(100 * 200 / (W * H), abs=1e-4)
    Image.new("L", (10, 10), 255).save(scratch / "small.png")
    cases = [
        (dict(image_path=path, mask=scratch / "small.png"), "mask is 10 x 10 px", "exactly the size"),
        (dict(image_path=scratch / "nope.png"), "not found", "PNG"),
        (dict(image_path=path, view="frnt"), "unknown view", "did you mean 'front'"),
        (dict(image_path=path, height=-2), "height must be a number > 0", None),
    ]
    (scratch / "text.png").write_text("not an image")
    cases.append((dict(image_path=scratch / "text.png"), "cannot read image", "PNG or JPEG"))
    Image.new("RGB", (40, 40), BG).save(scratch / "blank.png")
    cases.append((dict(image_path=scratch / "blank.png"), "no object found", "cut-out PNG"))
    for kwargs, message, hint in cases:
        with pytest.raises(SightlineError) as info:
            analyze_reference(**kwargs)
        assert message in info.value.message, kwargs
        if hint:
            assert hint in (info.value.hint or ""), kwargs


def test_analysis_is_deterministic(concept):
    path, _, _ = concept
    assert format_reference(analyze_reference(path)) == format_reference(analyze_reference(path))


# ----------------------------------------------------------------------------- matching

TOWER = """\
asset: {name}
parts:
  - {{name: base, shape: box, size: [3, 0.5, 3], material: stone, align: bottom}}
  - {{name: shaft, shape: box, size: [{shaft_w}, 5, 2], material: {shaft_mat}, on: base.top}}
{balcony}  - {{name: roof, shape: cone, radius: 1.6, height: 2, material: roof, on: shaft.top}}
{reference}"""
BALCONY = """\
  - name: balcony
    shape: box
    size: [1.0, 0.6, 1.0]
    material: planks
    on: shaft@[1, 0.2, 0]
    align: left
"""


def tower(name, shaft_w=2, shaft_mat="wood", balcony=True, reference="reference: {image: blocky_front.png, view: front}"):
    text = TOWER.format(name=name, shaft_w=shaft_w, shaft_mat=shaft_mat, balcony=BALCONY if balcony else "",
                        reference=reference)
    return load_asset_text(text, TMP / f"{name}.yaml", ROOT)


def render_reference(asset, view, res, name):
    """A 'perfect' reference: the asset's own flat Lambert render on a plain background."""
    lo, hi = asset.scene().bounds()
    cam = make_cameras([view], lo, hi, res=res)[view]
    rgb = shade_lambert(asset, march(asset.scene(), cam), background=np.array(BG) / 255)
    return save(Image.fromarray((rgb * 255).round().astype(np.uint8)), name)


@pytest.fixture(scope="module")
def blocky():
    """The full blocky tower rendered from the front at 320 px is the reference of every variant."""
    render_reference(tower("blocky_src", reference=""), "front", 320, "blocky_front.png")
    return tower("blocky_tower")


def region_hint(report, kind):
    return [r for r in report["regions"] if r["kind"] == kind]


def test_match_identical_model(blocky):
    out = TMP / "match_identical"
    shutil.rmtree(out, ignore_errors=True)
    r = match_asset(blocky, out_dir=out)
    # Silhouettes are dilated by up to half a pixel per edge (march eps = mpp / 2), at both resolutions.
    assert r["iou"] > 0.97 and r["missing"] < 0.02 and r["extra"] < 0.02
    assert r["regions"] == []
    assert all(e["verdict"] == "ok" for e in r["width_profile"])
    assert r["scale"]["mode"] == "fit" and abs(r["align"]["shift"]) <= 0.05
    assert r["size"]["model"][1] == pytest.approx(7.5, abs=2 * r["scale"]["grid_mpp"])
    assert r["aspect"]["reference"] == pytest.approx(r["aspect"]["model"], abs=0.03)
    assert {p["part"] for p in r["parts"]} == {"base", "shaft", "balcony", "roof"}
    for p in r["parts"]:  # same shading on both sides: the suggestion keeps the base colour
        assert p["delta_e"] < 2.0, p
        base = np.array(Image.new("RGB", (1, 1), p["base"]).getpixel((0, 0))) / 255
        sugg = np.array(Image.new("RGB", (1, 1), p["suggested"]).getpixel((0, 0))) / 255
        assert delta_e(base, sugg) < 2.0, p
    g = r["grid"]
    assert g["cols"] <= 32 and set("".join(g["grid"])) <= {"#", ".", "+", "-"}
    assert sum(row.count("#") for row in g["grid"]) > 10 * sum(row.count("+") + row.count("-") for row in g["grid"])
    png = pathlib.Path(r["images"]["match"])
    assert png == out / "match_0.png" and png.is_file()
    with Image.open(png) as im:
        assert im.width > 4 * im.height // 2  # four panels side by side
    text = format_match(r)
    for heading in ("MATCH", "FIT", "DIFF GRID", "WIDTH PROFILE", "REGIONS", "PARTS", "UNKNOWNS", "IMAGES"):
        assert re.search(rf"^{heading}\b", text, re.M), heading
    assert "IoU 0.9" in text and "  none" in text


def test_match_missing_part_is_located(blocky):
    asset = tower("no_balcony", balcony=False)
    r = match_asset(asset, out_dir=TMP / "match_missing")
    missing = region_hint(r, "missing")
    assert len(missing) == 1 and not region_hint(r, "extra")
    reg = missing[0]
    # The balcony spans x 1.0..2.0, y 3.2..3.8 m next to the shaft's right face.
    assert reg["bbox"]["x"] == [pytest.approx(1.0, abs=0.08), pytest.approx(2.0, abs=0.08)]
    assert reg["bbox"]["y"] == [pytest.approx(3.2, abs=0.08), pytest.approx(3.8, abs=0.08)]
    assert reg["near"] == "shaft" and "a part is missing here" in reg["hint"]
    assert reg["area"] == pytest.approx(0.6, rel=0.15) and reg["share"] == pytest.approx(r["missing"], abs=0.005)
    # Bbox centers differ by 0.2 m (the balcony widens the picture); the IoU search undoes that.
    assert r["align"]["shift"] == pytest.approx(0.2, abs=0.04)
    band = next(e for e in r["width_profile"] if e["band"] == "50-60%")
    assert band["diff"]["width"] == pytest.approx(-1.0, abs=0.1) and band["diff"]["left"] == pytest.approx(0, abs=0.05)
    assert re.match(r"model too narrow by (0\.9[5-9]|1\.0[0-5]) m", band["verdict"])
    text = format_match(r)
    assert re.search(r"missing +(0\.9\d|1\.0\d) x 0\.6\d m .* at x 1\.\d\d\.\.2\.\d\d, y 3\.\d\d\.\.3\.\d\d, next to `shaft` "
                     r"-> a part is missing here", text)


def test_match_too_wide_model(blocky):
    asset = tower("wide_shaft", shaft_w=2.6)
    r = match_asset(asset)
    cell, mpp = r["grid"]["cell"], r["scale"]["grid_mpp"]
    shaft_bands = [e for e in r["width_profile"] if e["band"] in ("30-40%", "60-70%", "70-80%", "80-90%")]
    assert len(shaft_bands) == 4
    for e in shaft_bands:  # shaft 2.6 m instead of 2.0 m: too wide by 0.6 m, 0.3 m per side
        assert e["diff"]["width"] == pytest.approx(0.6, abs=min(cell, 3 * mpp)), e
        assert e["diff"]["left"] == pytest.approx(0.3, abs=2 * mpp) and e["diff"]["right"] == pytest.approx(0.3, abs=2 * mpp)
        assert re.match(r"model too wide by 0\.[56]\d m", e["verdict"]), e  # +-1 grid px per edge
    base = next(e for e in r["width_profile"] if e["band"] == "90-100%")
    assert base["verdict"] == "ok"
    # Slivers on both sides (the right one is split by the balcony, which moved out with the face).
    slivers = [e for e in region_hint(r, "extra") if e["near"] == "shaft"]
    assert len(slivers) in (2, 3)
    assert {("left" if "left" in e["hint"] else "right") for e in slivers} == {"left", "right"}
    assert all("too big" in e["hint"] and e["thickness"] == pytest.approx(0.3, abs=2 * mpp) for e in slivers)
    left = next(e for e in slivers if "left" in e["hint"])
    assert left["bbox"]["y"] == [pytest.approx(0.5, abs=0.05), pytest.approx(5.5, abs=0.05)]
    balcony = [e for e in region_hint(r, "extra") if e["near"] == "balcony"]
    assert len(balcony) == 1 and balcony[0]["bbox"]["x"][1] == pytest.approx(2.3, abs=0.05)
    assert re.search(r"model too wide by 0\.[56]\d m", format_match(r))


def test_match_suggests_base_color(blocky):
    asset = tower("grey_shaft", shaft_mat="stone")
    r = match_asset(asset)
    shaft = next(p for p in r["parts"] if p["part"] == "shaft")
    wood = np.array(asset.style.material("wood").color)
    sugg = np.array(Image.new("RGB", (1, 1), shaft["suggested"]).getpixel((0, 0))) / 255
    assert shaft["delta_e"] > 10 and delta_e(wood, sugg) < 2.0
    assert r["iou"] > 0.97  # colour does not change the silhouette


def test_match_with_given_height_reports_size_mismatch(blocky):
    asset = tower("tall_ref", reference="reference: {image: blocky_front.png, view: front, height: 9.0}")
    r = match_asset(asset)
    assert r["scale"]["mode"] == "height"
    assert r["size"]["reference"][1] == pytest.approx(9.0, abs=0.05)
    assert r["size"]["model"][1] == pytest.approx(7.5, abs=0.1)
    assert r["size"]["reference"][0] == pytest.approx(3.6 * 9.0 / 7.5, abs=0.1)  # roof -1.6 .. balcony 2.0
    assert r["iou"] < 0.8 and r["missing"] > 0.25
    assert "scale" not in [q["topic"] for q in r["unknowns"]]
    assert re.search(r"model shorter by 1\.(4[5-9]|5[0-5]) m", format_match(r))


def test_match_elevated_view_with_height(blocky):
    """In an elevated view the picture's height includes the top: the given height is corrected like the model's."""
    render_reference(tower("blocky_src2", reference=""), "orbit:30:15", 300, "blocky_30_15.png")
    asset = tower("orbit_ref", reference="reference: {image: blocky_30_15.png, view: {azimuth: 30, elevation: 15}, height: 7.5}")
    r = match_asset(asset)
    assert r["scale"]["mode"] == "height" and "corrected" in r["scale"]["note"]
    assert r["axes"] == ["h", "v"] and r["iou"] > 0.95
    assert r["size"]["reference"][1] == pytest.approx(r["size"]["model"][1], abs=0.05)
    assert r["size"]["reference"][0] == pytest.approx(r["size"]["model"][0], abs=0.05)
    fitted = match_asset(tower("orbit_fit", reference="reference: {image: blocky_30_15.png, view: {azimuth: 30, elevation: 15}}"))
    assert fitted["iou"] == pytest.approx(r["iou"], abs=0.02)


def test_reports_are_json(concept, blocky):
    import json

    for rep in (analyze_reference(concept[0]), match_asset(blocky, res=96)):
        assert json.loads(json.dumps(rep)) == rep


def test_match_png_lambert_fallback_and_view_override(blocky, monkeypatch):
    monkeypatch.setitem(sys.modules, "sightline.render", None)  # render.py unavailable -> own shading
    out = TMP / "match_fallback"
    r = match_asset(blocky, out_dir=out, res=128)
    assert r["model_panel"] == "lambert"
    with Image.open(out / "match_0.png") as im:
        panel_w = (im.width - 5 * 8) / 4  # four equal panels with 8 px gaps
        assert panel_w == int(panel_w) and im.width > 1.5 * im.height and im.height > 200
    # A forced side view of a front reference fits poorly but still works.
    side = match_asset(blocky, view="right", res=128)
    assert side["reference"]["view"] == "right" and side["axes"] == ["-z", "y"] and side["iou"] < r["iou"]


def test_extra_region_next_to_the_outline_keeps_its_own_size(scratch):
    """A real extra region touching the 1-px dilation fringe must not merge with the fringe of the whole outline."""
    img = Image.new("RGB", (300, 500), BG)
    ImageDraw.Draw(img).rectangle((50, 50, 249, 449), fill=(120, 120, 120))  # exactly 200 x 400 px
    img.save(scratch / "box_ref.png")
    body = textwrap.dedent("""\
        asset: tipbox
        parts:
          - {name: block, shape: box, size: [4, 8, 2], material: stone, align: bottom}
          - {name: tip, shape: box, size: [1.0, 0.6, 1.0], material: roof, on: block.top_right, align: bottom_right}
        reference: {image: box_ref.png, view: front, height: 8.0}
        """)
    r = match_asset(load_asset_text(body, scratch / "tipbox.yaml", ROOT))
    (extra,) = region_hint(r, "extra")
    mpp = r["scale"]["grid_mpp"]
    # The tip spans x 1.0..2.0, y 8.0..8.6; before, the region was the whole 4 x 8.6 m outline fringe.
    assert extra["size"][0] == pytest.approx(1.0, abs=3 * mpp) and extra["size"][1] == pytest.approx(0.6, abs=3 * mpp)
    assert extra["bbox"]["x"][0] == pytest.approx(1.0, abs=3 * mpp) and extra["bbox"]["y"][0] == pytest.approx(8.0, abs=3 * mpp)
    assert extra["near"] == "tip"
    # SIZE: the model's exact extent and the picture's own bbox at the given height, not grid pixels.
    assert r["size"]["model"] == [4.0, 8.6] and r["size"]["reference"] == [4.0, 8.0]
    assert "-> model taller by 0.60 m, same width" in format_match(r)


def test_jpeg_noise_specks_keep_high_confidence(concept, scratch):
    img = Image.open(concept[0])
    path = scratch / "concept.jpg"
    img.convert("RGB").save(path, quality=80)
    r = analyze_reference(path)
    assert r["mask"]["dropped_specks"] > 0 and r["mask"]["confidence"] == "high"


def test_match_errors(blocky):
    bare = tower("bare", reference="")
    with pytest.raises(SightlineError) as info:
        match_asset(bare)
    assert "no reference image" in info.value.message and "reference:" in info.value.hint
    with pytest.raises(SightlineError) as info:
        match_asset(blocky, index=2)
    assert "out of range" in info.value.message and "index 0..0" in info.value.hint
    with pytest.raises(SightlineError):
        fit_view(bare)
    for res in (0, 3, 100000, 2.5):
        with pytest.raises(SightlineError, match="res must be an integer from 16 to 4096"):
            match_asset(blocky, res=res)


def test_match_is_deterministic(blocky):
    asset = tower("no_balcony2", balcony=False)
    assert format_match(match_asset(asset)) == format_match(match_asset(asset))


def test_existing_fixture_reference_matches_tower30():
    """tower30's own reference (core fixture, height 8.4 m): runs end to end with a believable score."""
    a = load_asset(FIX / "tower30.yaml", ROOT)
    r = match_asset(a)
    assert 0.6 < r["iou"] < 0.95 and r["scale"]["mode"] == "height"
    names = {e["near"] for e in r["regions"]}
    assert {"bush", "barrel"} <= names  # props the picture does not show
    assert any(e["near"] == "bush" and "not in the reference" in e["hint"] for e in r["regions"])


# ----------------------------------------------------------------------------- view fitting

LSHAPE = """\
asset: {name}
parts:
  - {{name: body, shape: box, size: [2, 1, 1], material: stone, align: bottom}}
  - {{name: arm, shape: box, size: [0.6, 0.6, 1.6], material: wood, at: [0.7, 1.0, 0.3], align: bottom}}
  - {{name: post, shape: cylinder, radius: 0.2, height: 2.5, material: metal, at: [-0.8, 0, -0.3], align: bottom}}
{reference}"""


def test_fit_view_finds_the_orbit_angle():
    src = load_asset_text(LSHAPE.format(name="lshape_src", reference=""), TMP / "lshape_src.yaml", ROOT)
    render_reference(src, "orbit:60:15", 200, "lshape_60_15.png")
    asset = load_asset_text(LSHAPE.format(name="lshape", reference="reference: {image: lshape_60_15.png}"),
                            TMP / "lshape.yaml", ROOT)
    results = fit_view(asset)
    assert len(results) == 5
    assert [r["iou"] for r in results] == sorted((r["iou"] for r in results), reverse=True)
    top3 = [(r["azimuth"], r["elevation"]) for r in results[:3]]
    assert (60, 15) in top3, results
    assert results[0]["iou"] > 0.9
    text = format_fit(results)
    assert "orbit:60:15" in text and "mirror" in text
    # The fitted view feeds match_asset directly.
    best = next(r for r in results if (r["azimuth"], r["elevation"]) == (60, 15))
    assert match_asset(asset, view=best["view"])["iou"] > 0.95
