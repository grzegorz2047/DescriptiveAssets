"""Ray marching: coverage, depth, normals, part ids, local coordinates, trace() and speed."""

import math
import pathlib
import time

import numpy as np
import pytest

from sightline.camera import fixed_camera, make_cameras
from sightline.loader import load_asset
from sightline.march import march, trace
from sightline.model import Scene, make_part

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def render(scene, view, res=256):
    lo, hi = scene.bounds()
    cam = make_cameras([view], lo, hi, res=res)[view]
    return cam, march(scene, cam)


def pixel_of(cam, x, y, z):
    (col, row) = cam.project(np.array([[x, y, z]]))[0][0]
    return int(row), int(col)


def erode(mask):
    """Pixels whose 4 neighbours are also set (away from silhouette edges)."""
    m = mask.copy()
    m[1:, :] &= mask[:-1, :]
    m[:-1, :] &= mask[1:, :]
    m[:, 1:] &= mask[:, :-1]
    m[:, :-1] &= mask[:, 1:]
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return m


def test_single_box_area_depth_normals_local():
    box = make_part("block", "box", {"size": [2, 1, 1]}, [0, 0.5, 0], material="stone")
    cam, vb = render(Scene([box]), "front")
    expected_px = 2.0 * 1.0 / cam.mpp**2
    assert abs(vb.hit.sum() - expected_px) / expected_px < 0.03
    assert vb.hit.dtype == bool and vb.part.dtype == np.int32
    assert vb.depth.dtype == np.float32 and vb.normal.dtype == np.float32 and vb.local.dtype == np.float32
    # Depth: the front face z = 0.5, measured from the ray origins (hit within eps = mpp / 2).
    _, face_depth = cam.project(np.array([[0.0, 0.5, 0.5]]))
    assert np.abs(vb.depth[vb.hit] - face_depth[0]).max() <= 0.5 * cam.mpp + 1e-5
    assert np.all(np.isinf(vb.depth[~vb.hit]))
    assert np.abs(vb.point[vb.hit][:, 2] - 0.5).max() <= 0.5 * cam.mpp + 1e-5
    inner = erode(vb.hit)
    assert np.allclose(vb.normal[inner], [0, 0, 1], atol=1e-3)
    assert np.allclose(np.linalg.norm(vb.normal[vb.hit], axis=1), 1.0, atol=1e-4)
    assert np.all(vb.part[vb.hit] == 0) and np.all(vb.part[~vb.hit] == -1)
    # Local coordinates: the box is centered at its origin, so the front face is local z = 0.5.
    loc = vb.local[vb.hit]
    assert np.abs(loc[:, 2] - 0.5).max() <= 0.5 * cam.mpp + 1e-5
    # Rays grazing a side face within eps count as hits: silhouettes grow by <= eps = mpp / 2.
    assert loc[:, 0].min() >= -1.0 - 0.5 * cam.mpp and loc[:, 0].max() <= 1.0 + 0.5 * cam.mpp
    # The image is centered on the box: hit pixels are left-right symmetric.
    assert np.array_equal(vb.hit, vb.hit[:, ::-1])


def test_iso_box_faces_and_areas():
    cube = make_part("cube", "box", {"size": [1, 1, 1]}, [0, 0.5, 0], material="stone")
    cam, vb = render(Scene([cube]), "iso")
    fwd = cam.forward
    n = vb.normal[erode(vb.hit)]
    faces = {"top": [0, 1, 0], "front": [0, 0, 1], "right": [1, 0, 0]}
    total = sum(abs(np.dot(fwd, f)) for f in faces.values())  # projected area of the 3 faces (m^2)
    assert abs(vb.hit.sum() * cam.mpp**2 - total) / total < 0.03
    for name, f in faces.items():
        share = np.mean(np.all(np.abs(n - f) < 1e-3, axis=1))
        assert share == pytest.approx(abs(np.dot(fwd, f)) / total, abs=0.02), name


def test_two_parts_ids_and_areas():
    a = load_asset(FIX / "two_boxes.yaml", ROOT)
    cam, vb = render(a.scene(), "front")
    left, right = a.part_index("left_box"), a.part_index("right_box")
    px = {i: int((vb.part == i).sum()) for i in (left, right)}
    for i, (w, h) in ((left, (1.0, 1.0)), (right, (1.0, 2.0))):
        area = w * h / cam.mpp**2
        assert abs(px[i] - area) / area < 0.05
        # The hit threshold eps = mpp / 2 widens each silhouette edge by up to half a pixel.
        dilated = (w / cam.mpp + 1) * (h / cam.mpp + 1)
        assert abs(px[i] - dilated) / dilated < 0.02
    assert vb.part[pixel_of(cam, -1.0, 0.5, 0.5)] == left
    assert vb.part[pixel_of(cam, 1.0, 1.5, 0.5)] == right
    # top view: the taller right box is nearer to the camera
    cam_t, vb_t = render(a.scene(), "top")
    assert vb_t.depth[vb_t.part == right].mean() == pytest.approx(vb_t.depth[vb_t.part == left].mean() - 1.0, abs=cam_t.mpp)


def test_sphere_area_and_normals():
    ball = make_part("ball", "sphere", {"radius": 1.0}, [0, 1, 0], material="stone")
    cam, vb = render(Scene([ball]), "front")
    expected = math.pi / cam.mpp**2
    assert abs(vb.hit.sum() - expected) / expected < 0.03
    # Normal = direction from the center to the hit point.
    radial = vb.point[vb.hit] - np.array([0, 1, 0], dtype=np.float32)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    assert np.mean(np.sum(radial * vb.normal[vb.hit], axis=1) > 0.995) > 0.98


def test_rotated_and_mirrored_parts_render_symmetric():
    parts = [
        make_part("body", "box", {"size": [0.6, 1.2, 0.4]}, [0, 0.6, 0], material="wood"),
        make_part("arm_l", "capsule", {"radius": 0.1, "height": 0.8}, [0.6, 0.9, 0], (0, 0, 35), material="skin"),
    ]
    twin = make_part("arm_r", "capsule", {"radius": 0.1, "height": 0.8}, [-0.6, 0.9, 0], (0, 0, 35), material="skin")
    twin.matrix = np.diag([-1.0, 1, 1]) @ parts[1].matrix  # det -1 reflection, like the loader's mirror
    scene = Scene(parts + [twin])
    cam, vb = render(scene, "front")
    assert np.linalg.det(twin.matrix) == pytest.approx(-1)
    assert np.mean(vb.hit != vb.hit[:, ::-1]) < 0.002
    left, right = (vb.part == 1), (vb.part == 2)
    assert abs(int(left.sum()) - int(right.sum())) <= 0.01 * left.sum()
    # Mirrored normals: x flips sign between the twins.
    nl = vb.normal[left].mean(axis=0)
    nr = vb.normal[right].mean(axis=0)
    assert nl[0] == pytest.approx(-nr[0], abs=0.02) and nl[2] == pytest.approx(nr[2], abs=0.02)


def test_cuts_blends_and_local_hits_in_tower():
    a = load_asset(FIX / "tower30.yaml", ROOT)
    scene = a.scene()
    cam, vb = render(scene, "front")

    door = pixel_of(cam, 0.0, 1.9, 0.0)
    wall = pixel_of(cam, 0.0, 3.3, 0.0)
    assert a.parts[vb.part[door]].name == "shaft" and a.parts[vb.part[wall]].name == "shaft"
    # The door recess is 0.4 m deep: shaft face at z = 1.2 (apothem), door box back at z = 0.8.
    assert vb.point[wall][2] == pytest.approx(1.3 * math.cos(math.pi / 8), abs=cam.mpp)
    assert vb.point[door][2] == pytest.approx(0.8, abs=cam.mpp)
    assert vb.depth[door] - vb.depth[wall] == pytest.approx(0.4, abs=cam.mpp)
    assert a.parts[vb.part[pixel_of(cam, 0.0, 8.85, 0.0)]].name == "finial"  # blended into the roof, still its own id
    # Local hit points lie on each part's own surface.
    for i in np.unique(vb.part[vb.hit]):
        part = a.parts[i]
        loc = vb.local[vb.part == i]
        assert np.allclose(part.to_world(loc), vb.point[vb.part == i], atol=1e-4)
    # only(): the shaft alone covers more pixels than it shows in the full render
    shaft = a.part_index("shaft")
    vb_alone = march(scene.only([shaft]), cam)
    assert set(np.unique(vb_alone.part[vb_alone.hit])) == {shaft}
    assert vb_alone.hit.sum() > (vb.part == shaft).sum()


def _flush_crenel():
    """A 4 x 2 x 0.5 m wall with a 0.4 m notch whose top face lies ON the wall's top face (coplanar)."""
    wall = make_part("wall", "box", {"size": [4, 2, 0.5]}, [0, 1, 0], material="stone")
    notch = make_part("notch", "box", {"size": [0.4, 0.4, 1.0]}, [-1.5, 1.8, 0], op="cut")
    notch.cut_from = ["wall"]
    return Scene([wall, notch])


def test_coplanar_cut_leaves_no_phantom_lid():
    """max(d_part, -d_cut) is 0 on the shared plane over a flush notch; the march must not stop there."""
    scene = _flush_crenel()
    cam, vb = render(scene, "top")
    over = pixel_of(cam, -1.5, 2.0, 0.0)
    assert vb.hit[over] and vb.point[over][1] == pytest.approx(1.6, abs=0.5 * cam.mpp + 1e-4)  # the notch floor
    beside = pixel_of(cam, -1.0, 2.0, 0.0)
    assert vb.point[beside][1] == pytest.approx(2.0, abs=0.5 * cam.mpp + 1e-4)  # the merlon top is still there
    # Every top-view pixel over the notch opening sees the floor, none a lid at y = 2.
    origins, _ = cam.rays()
    x, z = origins[:, 0].reshape(vb.hit.shape), origins[:, 2].reshape(vb.hit.shape)
    inner = (np.abs(x + 1.5) < 0.2 - cam.mpp) & (np.abs(z) < 0.25 - cam.mpp)
    assert inner.sum() > 50 and np.all(np.abs(vb.point[inner][:, 1] - 1.6) <= 0.5 * cam.mpp + 1e-4)


def test_coplanar_corner_cut_shows_its_back_wall():
    """A corner notch flush with three faces of a slab (top, front, right): the front view sees its back wall."""
    base = make_part("base", "box", {"size": [2, 0.5, 2]}, [0, 0.25, 0], material="stone")
    corner = make_part("corner", "box", {"size": [0.5, 0.25, 0.5]}, [0.75, 0.375, 0.75], op="cut")
    corner.cut_from = ["base"]
    scene = Scene([base, corner])
    cam, vb = render(scene, "front")
    px = pixel_of(cam, 0.75, 0.375, 1.0)
    assert vb.point[px][2] == pytest.approx(0.5, abs=0.5 * cam.mpp + 1e-4)  # not a phantom face at z = 1
    assert vb.point[pixel_of(cam, -0.5, 0.25, 1.0)][2] == pytest.approx(1.0, abs=0.5 * cam.mpp + 1e-4)


def test_trace_arbitrary_rays():
    ball = make_part("ball", "sphere", {"radius": 1.0}, [0, 0, 0], material="stone")
    block = make_part("block", "box", {"size": [1, 1, 1]}, [5, 0, 0], material="stone")
    scene = Scene([ball, block])
    origins = np.array([[0, 0, 5], [0, 0.5, 5], [3, 0, 0], [0, 3, 5], [-3, 0, 0], [0, 0, 5]], dtype=np.float32)
    dirs = np.array([[0, 0, -1], [0, 0, -1], [1, 0, 0], [0, 0, -1], [1, 0, 0], [0, 0, -1]], dtype=np.float32)
    t, part = trace(scene, origins, dirs, t_max=np.array([100, 100, 100, 100, 100, 3.5]), eps=1e-3)
    assert t[0] == pytest.approx(4.0, abs=1e-3) and part[0] == 0
    assert t[1] == pytest.approx(5 - math.sqrt(0.75), abs=1e-3)
    assert t[2] == pytest.approx(1.5, abs=1e-3) and part[2] == 1  # ray starting between the two
    assert np.isinf(t[3]) and part[3] == -1  # miss
    assert t[4] == pytest.approx(2.0, abs=1e-3) and part[4] == 0  # nearest of two groups
    assert np.isinf(t[5])  # t_max stops it before the surface
    t_only, part_only = trace(scene, origins[4:5], dirs[4:5], 100.0, 1e-3, groups=[1])
    assert t_only[0] == pytest.approx(7.5, abs=1e-3) and part_only[0] == 1
    # Shared-direction rays (orthographic) take the same path as per-ray directions.
    many = np.repeat(origins[:2], 3, axis=0)
    t_many, _ = trace(scene, many, np.repeat(dirs[:2], 3, axis=0), 100.0, 1e-3)
    assert t_many == pytest.approx(np.repeat(t[:2], 3), abs=1e-5)


def test_with_parts_adds_helper_group_and_empty_scene():
    box = make_part("block", "box", {"size": [1, 1, 1]}, [0, 0.5, 0], material="stone")
    scene = Scene([box])
    human = make_part("human", "capsule", {"radius": 0.25, "height": 1.8}, [1.5, 0.9, 0], material="helper", helper=True)
    both = scene.with_parts([human])
    assert len(both.groups) == 2 and both.solid_indices() == [0] and both.solid_indices(True) == [0, 1]
    cam, vb = render(both, "front")
    assert set(np.unique(vb.part[vb.hit])) == {0, 1}
    empty = Scene([box], solids=[])
    vb_empty = march(empty, fixed_camera("front", [0, 0, 0], 0.05, 16, 16, 5.0))
    assert not vb_empty.hit.any() and np.all(vb_empty.part == -1)


def test_benchmark_tower30_four_views_256px():
    """SPEC 6.1 target: a ~30-part asset, 4 views at 256 px in < 6 s."""
    a = load_asset(FIX / "tower30.yaml", ROOT)
    scene = a.scene()
    lo, hi = scene.bounds()
    cams = make_cameras(["front", "right", "top", "iso"], lo, hi, res=256)
    march(scene, cams["iso"])  # warm-up (imports, allocator)
    start = time.perf_counter()
    hits = {v: int(march(scene, c).hit.sum()) for v, c in cams.items()}
    elapsed = time.perf_counter() - start
    print(f"\ntower30 ({len(scene.solid_indices())} solids, {len(scene.groups)} groups): 4 views at 256 px in {elapsed:.2f} s")
    assert all(n > 1000 for n in hits.values())
    assert elapsed < 6.0


def test_stacked_decal_discs_render_whole_not_as_crescents():
    """Roundels of 1.2 cm discs stacked on a tilted wing: the coarse half-pixel hit threshold (4.5 cm here)
    let whichever surface's steps crossed it first win, so a ring showed as a crescent and measured
    0% visible from the top. Near-ties are re-traced with a fine threshold."""
    rot = (3.0, 0.0, -5.0)
    wing = make_part("wing", "box", {"size": [10, 0.14, 1.5], "round": 0.06}, [0, 1.5, 0], rot, material="stone")
    top = wing.to_world(np.array([[0.0, 0.07, 0.0]]))[0]
    normal = wing.matrix @ np.array([0.0, 1.0, 0.0])
    discs = [make_part(name, "cylinder", {"radius": r, "height": 0.012}, top + normal * (0.006 + 0.012 * k), rot,
                       material="paint_red") for k, (name, r) in enumerate((("blue", 0.55), ("white", 0.37), ("red", 0.18)))]
    scene = Scene([wing, *discs])
    cam, vb = render(scene, "top", 128)  # mpp 0.09 m: the discs are an eighth of a pixel thick
    counts = {name: int((vb.part == i).sum()) for i, name in enumerate(("wing", "blue", "white", "red"))}
    ring = lambda r_out, r_in: math.pi * (r_out ** 2 - r_in ** 2) / cam.mpp ** 2  # noqa: E731
    assert counts["blue"] == pytest.approx(ring(0.55, 0.37), rel=0.12), counts
    assert counts["white"] == pytest.approx(ring(0.37, 0.18), rel=0.12), counts
    assert counts["red"] == pytest.approx(ring(0.18, 0.0), rel=0.15), counts
    # The ring is whole: blue pixels on both sides of the white disc's center.
    r, c = pixel_of(cam, *top)
    blue_cols = np.flatnonzero((vb.part == 1).any(axis=0))
    assert blue_cols.min() < c - 0.4 / cam.mpp and blue_cols.max() > c + 0.4 / cam.mpp
    # Silhouettes keep the coarse threshold: the wing alone covers what it covered before.
    _, alone = render(Scene([wing, *discs], solids=[0]), "top", 128)
    assert abs(int(alone.hit.sum()) - int(vb.hit.sum())) <= 4
