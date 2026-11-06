"""Mesh export: surface-nets geometry, glTF 2.0 binary structure, OBJ + MTL, baked colors, errors."""

import json
import math
import pathlib
import re
import struct
import time

import numpy as np
import pytest

from sightline.decimate import BBOX_TOL, SHADE_ANGLE, SHADE_WIDTH, DecimationInput, decimate, nonmanifold_vertices
from sightline.errors import SightlineError
from sightline.export import (
    QUALITY,
    accessor_data,
    PartMesh,
    build_meshes,
    decimate_meshes,
    edge_report,
    export_glb,
    format_decimation,
    format_edges,
    export_obj,
    mesh_summary,
    read_glb,
    surface_nets,
    validate_glb,
    vertex_normals,
)
from sightline.loader import load_asset, load_asset_text

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = ROOT / "out" / "test_tmp" / "export"


def asset_from(name: str, parts_yaml: str):
    return load_asset_text(f"asset: {name}\nparts:\n{parts_yaml}", FIX / f"{name}.yaml", ROOT)


def srgb_lin(c: float) -> float:
    """Independent sRGB -> linear reference."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ----------------------------------------------------------------------------- independent GLB parsing


def parse_glb(path):
    """Header fields, JSON document, raw JSON chunk and BIN chunk, parsed with struct only."""
    data = pathlib.Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    js_len, js_type = struct.unpack_from("<II", data, 12)
    js = data[20 : 20 + js_len]
    bin_len, bin_type = struct.unpack_from("<II", data, 20 + js_len)
    blob = data[28 + js_len : 28 + js_len + bin_len]
    head = dict(magic=magic, version=version, length=length, size=len(data), js_len=js_len, js_type=js_type,
                bin_len=bin_len, bin_type=bin_type, end=28 + js_len + bin_len)
    return head, json.loads(js.decode("utf-8")), js, blob


def decode(doc, blob, i):
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    dtype = {5126: "<f4", 5125: "<u4"}[a["componentType"]]
    n = {"SCALAR": 1, "VEC3": 3}[a["type"]]
    arr = np.frombuffer(blob, dtype=dtype, count=a["count"] * n, offset=v.get("byteOffset", 0) + a.get("byteOffset", 0))
    return arr.reshape(a["count"], n) if n > 1 else arr


def rebuild_glb(doc, blob) -> bytes:
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob = bytes(blob) + b"\0" * (-len(blob) % 4)
    return struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(blob)) + struct.pack("<II", len(js), 0x4E4F534A) + js \
        + struct.pack("<II", len(blob), 0x004E4942) + blob


def glb_triangles(doc, blob):
    """All (positions, triangles) per mesh decoded from the file."""
    out = []
    for mesh in doc["meshes"]:
        pos = decode(doc, blob, mesh["primitives"][0]["attributes"]["POSITION"]).astype(np.float64)
        tris = np.concatenate([decode(doc, blob, p["indices"]).reshape(-1, 3) for p in mesh["primitives"]])
        out.append((pos, tris.astype(np.int64)))
    return out


# ----------------------------------------------------------------------------- mesh topology helpers


def edge_use(tris):
    """(undirected edge -> triangle count histogram, max uses of one directed edge)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]).astype(np.int64)
    key = np.sort(e, axis=1)
    _, und = np.unique(key[:, 0] * 2**31 + key[:, 1], return_counts=True)
    _, dirc = np.unique(e[:, 0] * 2**31 + e[:, 1], return_counts=True)
    return und, dirc.max()


def assert_closed(tris):
    und, dmax = edge_use(tris)
    assert (und == 2).all(), f"edges used by {sorted(set(und.tolist()))} triangles"
    assert dmax == 1  # every directed edge once: consistent orientation


def euler(tris) -> int:
    v = len(np.unique(tris))
    und, _ = edge_use(tris)
    return v - len(und) + len(tris)


def volume(pos, tris) -> float:
    p = np.asarray(pos, dtype=np.float64)[tris]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


def face_normals(pos, tris):
    p = np.asarray(pos, dtype=np.float64)[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True), p.mean(axis=1)


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tower():
    return load_asset(FIX / "tower30.yaml")


@pytest.fixture(scope="module")
def tower_full(tower):
    """The surface-nets meshes as meshed (quality full)."""
    return build_meshes(tower, quality="full")


@pytest.fixture(scope="module")
def tower_meshes(tower, tower_full):
    """The default export: the same meshes decimated (quality medium), as build_meshes(tower) makes them."""
    return decimate_meshes(tower, tower_full)


@pytest.fixture(scope="module")
def tower_glb(tower, tower_meshes):
    return export_glb(tower, OUT / "tower30.glb", meshes=tower_meshes)


SPHERE = "  - {name: ball, shape: sphere, radius: 1.0, material: stone, at: [0, 1, 0]}\n"


# ----------------------------------------------------------------------------- GLB structure


def test_glb_header_and_chunks(tower_glb):
    head, doc, js, blob = parse_glb(tower_glb)
    assert head["magic"] == b"glTF" and head["version"] == 2
    assert head["length"] == head["size"] == head["end"]
    assert head["js_type"] == 0x4E4F534A and head["bin_type"] == 0x004E4942
    assert head["js_len"] % 4 == 0 and head["bin_len"] % 4 == 0
    body = js.rstrip(b" ")
    assert set(js[len(body):]) <= {0x20}  # JSON padded with spaces
    json.loads(body)
    assert set(blob[doc["buffers"][0]["byteLength"]:]) <= {0}  # BIN padded with zeros
    assert doc["asset"]["version"] == "2.0" and doc["asset"]["generator"] == "sightline"
    assert len(doc["buffers"]) == 1 and "uri" not in doc["buffers"][0]
    assert doc["scenes"][doc["scene"]]["nodes"] == list(range(len(doc["nodes"])))


def test_glb_accessors_and_views_like_a_validator(tower_glb):
    _, doc, _, blob = parse_glb(tower_glb)
    views, accessors = doc["bufferViews"], doc["accessors"]
    # One buffer exactly covered by contiguous, 4-byte aligned, tightly packed views.
    assert sum(v["byteLength"] for v in views) == doc["buffers"][0]["byteLength"]
    offset = 0
    for v in views:
        assert v["byteOffset"] == offset and v["byteOffset"] % 4 == 0
        assert "byteStride" not in v
        offset += v["byteLength"]
    for a in accessors:
        v = views[a["bufferView"]]
        ncomp = {"SCALAR": 1, "VEC3": 3}[a["type"]]
        assert a["componentType"] in (5126, 5125)
        assert v["byteLength"] == a["count"] * ncomp * 4  # count matches the buffer bytes
        assert (v["byteOffset"] + a.get("byteOffset", 0)) % 4 == 0
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            pos_acc = accessors[attrs["POSITION"]]
            pos = decode(doc, blob, attrs["POSITION"])
            assert pos_acc["min"] == pos.min(axis=0).tolist() and pos_acc["max"] == pos.max(axis=0).tolist()
            assert accessors[attrs["NORMAL"]]["count"] == pos_acc["count"]
            nrm = decode(doc, blob, attrs["NORMAL"]).astype(np.float64)
            assert np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max() < 1e-5
            idx = decode(doc, blob, prim["indices"])
            ia = accessors[prim["indices"]]
            assert (ia["componentType"], ia["type"]) == (5125, "SCALAR")
            assert views[ia["bufferView"]]["target"] == 34963
            assert views[pos_acc["bufferView"]]["target"] == 34962
            assert len(idx) % 3 == 0 and idx.max() < pos_acc["count"]
            assert prim["mode"] == 4 and 0 <= prim["material"] < len(doc["materials"])


def test_glb_nodes_are_parts_and_blend_groups(tower, tower_glb):
    _, doc, _, _ = parse_glb(tower_glb)
    scene = tower.scene()
    expected = [scene.parts[g.members[0]].name for g in scene.groups]
    names = [n["name"] for n in doc["nodes"]]
    assert names == expected and len(names) == 28
    assert [m["name"] for m in doc["meshes"]] == names
    assert all(n.get("translation") is None and n["mesh"] == i for i, n in enumerate(doc["nodes"]))
    for cut in ("door", "window_0", "window_3", "drain"):  # cuts are never meshes
        assert cut not in names
    by_name = {n["name"]: n for n in doc["nodes"]}
    assert by_name["roof"]["extras"]["parts"] == ["roof", "finial"]  # finial blends into the roof
    assert by_name["bush"]["extras"]["parts"] == ["bush", "bush_top"]
    assert "finial" not in names and "bush_top" not in names
    roof_mats = [doc["materials"][p["material"]]["name"] for p in doc["meshes"][names.index("roof")]["primitives"]]
    assert roof_mats == ["roof", "gold"]


def test_glb_materials_pbr_factors(tower, tower_glb):
    _, doc, _, _ = parse_glb(tower_glb)
    mats = {m["name"]: m["pbrMetallicRoughness"] for m in doc["materials"]}
    used = []
    for p in tower.parts:
        if p.is_solid and p.material not in used:
            used.append(p.material)
    assert [m["name"] for m in doc["materials"]] == used
    stone = tower.style.material("stone")  # granite #8e8b84
    assert mats["stone"]["baseColorFactor"] == pytest.approx([srgb_lin(c) for c in stone.color] + [1.0], abs=1e-6)
    assert mats["stone"]["baseColorFactor"][0] == pytest.approx(0.2705, abs=1e-4)
    assert (mats["stone"]["metallicFactor"], mats["stone"]["roughnessFactor"]) == (0.0, 1.0)
    assert mats["gold"]["metallicFactor"] == pytest.approx(0.64) and mats["gold"]["roughnessFactor"] == pytest.approx(0.52)
    assert mats["iron"]["metallicFactor"] == pytest.approx(0.32) and mats["iron"]["roughnessFactor"] == pytest.approx(0.76)


def test_validator_accepts_export_and_catches_corruption(tower_glb):
    assert validate_glb(tower_glb) == []
    data = tower_glb.read_bytes()
    _, doc, _, blob = parse_glb(tower_glb)
    assert validate_glb(rebuild_glb(doc, blob)) == []  # the test's rebuild is itself valid

    assert "header length" in validate_glb(data[:8] + struct.pack("<I", len(data) + 4) + data[12:])[0]
    assert "version" in validate_glb(data[:4] + struct.pack("<I", 1) + data[8:])[0]
    assert "magic" in validate_glb(b"GLTF" + data[4:])[0]
    assert validate_glb(data[:-8]) != []  # truncated

    bad = json.loads(json.dumps(doc))
    bad["accessors"][0]["count"] += 1
    assert any("overrun" in p for p in validate_glb(rebuild_glb(bad, blob)))
    bad = json.loads(json.dumps(doc))
    bad["accessors"][0]["max"][1] += 0.5
    assert any("max" in p for p in validate_glb(rebuild_glb(bad, blob)))
    bad = json.loads(json.dumps(doc))
    bad["buffers"][0]["byteLength"] -= 4
    assert validate_glb(rebuild_glb(bad, blob)) != []
    bad = json.loads(json.dumps(doc))
    bad["nodes"][0]["mesh"] = 999
    assert any("out of range" in p for p in validate_glb(rebuild_glb(bad, blob)))
    # An index pointing past the vertex count.
    prim = doc["meshes"][0]["primitives"][0]
    view = doc["bufferViews"][doc["accessors"][prim["indices"]]["bufferView"]]
    corrupt = bytearray(blob)
    corrupt[view["byteOffset"] : view["byteOffset"] + 4] = struct.pack("<I", 10**7)
    assert any(">= vertex count" in p for p in validate_glb(rebuild_glb(doc, corrupt)))


def test_read_glb_and_accessor_data_roundtrip(tower_glb, tower_meshes):
    doc, blob = read_glb(tower_glb)
    pos = accessor_data(doc, blob, doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
    assert np.array_equal(pos, tower_meshes[0].positions)
    with pytest.raises(SightlineError, match="not a valid glTF binary"):
        read_glb(FIX / "tower30.yaml")


# ----------------------------------------------------------------------------- geometry


@pytest.mark.parametrize("max_cells,tol", [(16, 0.02), (32, 0.01), (64, 0.005)])
def test_sphere_closed_outward_and_volume(max_cells, tol):
    (mesh,) = build_meshes(asset_from("sphere", SPHERE), max_cells=max_cells, quality="full")
    pos, tris = mesh.positions.astype(np.float64), mesh.triangles.astype(np.int64)
    center = np.array([0.0, 1.0, 0.0])
    assert mesh.cell == pytest.approx(2.0 / max_cells)
    assert_closed(tris)
    assert euler(tris) == 2
    rel = pos - center
    assert (np.einsum("ij,ij->i", mesh.normals, rel) > 0).all()  # vertex normals point outward
    fn, centroid = face_normals(pos, tris)
    assert (np.einsum("ij,ij->i", fn, centroid - center) > 0).all()  # so does the winding
    assert np.abs(np.linalg.norm(rel, axis=1) - 1.0).max() < 0.01 * mesh.cell  # projected onto the surface
    assert volume(pos, tris) == pytest.approx(4 / 3 * math.pi, rel=tol)
    assert mesh.volume() == pytest.approx(volume(pos, tris))


def test_edge_report_finds_open_and_non_manifold_edges():
    (mesh,) = build_meshes(asset_from("sphere", SPHERE), max_cells=24)
    assert edge_report([mesh]) == [] and format_edges([mesh]).startswith("MANIFOLD  yes")
    # Two tetrahedra sharing one edge (0-1): that edge joins 4 triangles; plus one loose triangle.
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
                   dtype=np.float32)
    tris = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2], [0, 4, 1], [0, 1, 5], [0, 5, 4], [1, 4, 5],
                     [6, 7, 8]], dtype=np.uint32)
    odd = PartMesh(name="odd", members=["odd"], positions=pos, normals=np.zeros_like(pos), triangles=tris,
                   vertex_part=np.zeros(9, dtype=np.int32), tri_part=np.zeros(9, dtype=np.int32), cell=0.1)
    assert edge_report([mesh, odd]) == [("odd", 3, 1)]
    assert "odd: 3 open, 1 non-manifold edge(s)" in format_edges([odd])


def test_sphere_volume_from_the_glb_file():
    asset = asset_from("sphere", SPHERE)
    path = export_glb(asset, OUT / "sphere.glb", max_cells=48, quality="full")
    _, doc, _, blob = parse_glb(path)
    ((pos, tris),) = glb_triangles(doc, blob)
    assert_closed(tris)
    assert volume(pos, tris) == pytest.approx(4 / 3 * math.pi, rel=0.05)
    assert doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["min"] == \
        pytest.approx([-1.0, 0.0, -1.0], abs=1e-4)


def test_box_keeps_sharp_extents():
    (mesh,) = build_meshes(load_asset(FIX / "box1.yaml"), max_cells=32, quality="full")  # 2 x 1 x 1 m, cell 1/16 m
    lo, hi = mesh.positions.min(axis=0), mesh.positions.max(axis=0)
    assert np.abs(lo - [-1.0, 0.0, -0.5]).max() < mesh.cell and np.abs(hi - [1.0, 1.0, 0.5]).max() < mesh.cell
    assert np.abs(lo - [-1.0, 0.0, -0.5]).max() < 0.01 * mesh.cell  # projection snaps edges and corners
    assert mesh.volume() == pytest.approx(2.0, rel=0.005)
    assert_closed(mesh.triangles)
    # Every vertex sits on the box surface (exact box SDF), corners and edges included.
    q = np.abs(mesh.positions.astype(np.float64) - [0.0, 0.5, 0.0]) - [1.0, 0.5, 0.5]
    sdf = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
    assert np.abs(sdf).max() < 0.05 * mesh.cell
    # Flat faces stay flat: away from the edges, +x face vertices lie on x = 1 exactly.
    face = (mesh.positions[:, 0] > 1.0 - 0.5 * mesh.cell) & (np.abs(mesh.positions[:, 1] - 0.5) < 0.4) \
        & (np.abs(mesh.positions[:, 2]) < 0.4)
    assert face.sum() > 50 and np.abs(mesh.positions[face, 0] - 1.0).max() < 1e-5


def test_rotated_box_extents_within_one_cell():
    asset = asset_from("tilted", "  - {name: slab, shape: box, size: [2, 0.5, 1], material: stone, rotate: [10, 30, 0], at: [0, 1, 0]}\n")
    (mesh,) = build_meshes(asset, max_cells=48, quality="full")
    lo, hi = asset.parts[0].world_aabb()
    assert np.abs(mesh.positions.min(axis=0) - lo).max() < mesh.cell
    assert np.abs(mesh.positions.max(axis=0) - hi).max() < mesh.cell
    assert mesh.volume() == pytest.approx(1.0, rel=0.01)
    assert_closed(mesh.triangles)


def test_symmetric_shape_gives_symmetric_mesh():
    (mesh,) = build_meshes(load_asset(FIX / "box1.yaml"), max_cells=24, quality="full")
    pts = np.round(mesh.positions.astype(np.float64), 5)
    mirrored = pts * [-1, 1, 1]
    key = lambda a: a[np.lexsort(a.T[::-1])]  # noqa: E731
    assert np.array_equal(key(pts), key(mirrored + 0.0))


def test_through_cut_makes_a_hole():
    asset = asset_from("holed", "  - {name: block, shape: box, size: [2, 1, 1], material: stone, align: bottom}\n"
                                "  - {name: hole, shape: cylinder, radius: 0.25, height: 2, rotate: [90, 0, 0], op: cut,"
                                " from: [block], at: [0, 0.5, 0]}\n")
    (mesh,) = build_meshes(asset, max_cells=64, quality="full")
    tris = mesh.triangles.astype(np.int64)
    assert_closed(tris)
    assert euler(tris) == 0  # genus 1: a tunnel through the block
    pos = mesh.positions.astype(np.float64)
    radial = np.hypot(pos[:, 0], pos[:, 1] - 0.5)
    assert radial.min() > 0.25 - 0.01 * mesh.cell  # nothing left inside the hole
    tunnel = np.abs(radial - 0.25) < 0.01 * mesh.cell
    assert np.ptp(pos[tunnel, 2]) > 0.9  # the tunnel wall runs through the whole depth
    assert mesh.volume() == pytest.approx(2.0 - math.pi * 0.25**2 * 1.0, rel=0.01)


def test_blind_cut_makes_a_recess():
    asset = asset_from("niche", "  - {name: wall, shape: box, size: [2, 2, 1], material: stone, align: bottom}\n"
                                "  - {name: niche, shape: box, size: [0.6, 1.0, 0.4], op: cut, on: wall.front, align: center}\n")
    (mesh,) = build_meshes(asset, max_cells=64, quality="full")
    assert_closed(mesh.triangles)
    assert euler(mesh.triangles.astype(np.int64)) == 2  # a recess, not a hole
    assert mesh.volume() == pytest.approx(4.0 - 0.6 * 1.0 * 0.2, rel=0.005)
    pos = mesh.positions
    back = (np.abs(pos[:, 0]) < 0.25) & (np.abs(pos[:, 1] - 1.0) < 0.4) & (pos[:, 2] > 0)
    assert back.any() and np.abs(pos[back, 2] - 0.3).max() < 1e-3  # recess back wall at z = 0.5 - 0.2


def test_surface_nets_on_an_analytic_sdf():
    def two_balls(p3):
        a = np.sqrt((p3[0] + 1.0) ** 2 + p3[1] ** 2 + p3[2] ** 2) - 0.6
        b = np.sqrt((p3[0] - 1.0) ** 2 + p3[1] ** 2 + p3[2] ** 2) - 0.6
        return np.minimum(a, b)

    verts, tris = surface_nets(two_balls, [-1.6, -0.6, -0.6], [1.6, 0.6, 0.6], 0.05)
    assert_closed(tris)
    assert euler(tris) == 4  # two separate spheres
    assert volume(verts, tris) == pytest.approx(2 * 4 / 3 * math.pi * 0.6**3, rel=0.01)
    empty_v, empty_t = surface_nets(lambda p3: np.ones(p3.shape[1], np.float32), [0, 0, 0], [1, 1, 1], 0.1)
    assert len(empty_v) == 0 and len(empty_t) == 0


def test_tower30_meshes_closed_outward_with_measured_volumes(tower, tower_full):
    by_name = {m.name: m for m in tower_full}
    for m in tower_full:
        tris = m.triangles.astype(np.int64)
        assert_closed(tris)
        assert m.volume() > 0, m.name
        fn, _ = face_normals(m.positions, tris)
        # Face winding agrees with the SDF-gradient vertex normals.
        assert (np.einsum("ij,ij->i", fn, m.normals[tris].sum(axis=1)) > 0).all(), m.name
    assert by_name["crenel_0"].volume() == pytest.approx(0.3 * 0.4 * 0.3, rel=0.1)
    assert by_name["post_0"].volume() == pytest.approx(0.12 * 0.8 * 0.12, rel=0.05)
    hoop = 2 * math.pi**2 * 0.34 * 0.03**2  # torus volume 2 pi^2 R t^2
    assert by_name["hoop"].volume() == pytest.approx(hoop, rel=0.1)


def test_thin_parts_get_a_finer_cell(tower, tower_full):
    by_name = {m.name: m for m in tower_full}
    base = by_name["base"].cell
    lo, hi = tower.scene().bounds(include_helpers=False)
    assert base == pytest.approx(float((hi - lo).max()) / 128)
    banner = by_name["banner_l"]  # 0.05 m thick cloth
    assert banner.cell == pytest.approx(max(0.05 / 3, base / 4)) and banner.cell < base  # here the base/4 cap
    assert np.ptp(banner.positions[:, 0]) == pytest.approx(0.05, abs=0.01)
    assert banner.volume() == pytest.approx(0.05 * 1.2 * 0.6, rel=0.05)
    assert by_name["band"].cell < base and by_name["shaft"].cell == base
    assert all(m.cell >= base / 4 - 1e-12 for m in tower_full)


def test_mirrored_parts_keep_outward_winding():
    knight = load_asset(FIX / "knight.yaml")
    meshes = build_meshes(knight, max_cells=96)
    mirrored = [i for i, p in enumerate(knight.parts) if p.is_solid and np.linalg.det(p.matrix) < 0]
    assert mirrored  # the fixture has det -1 twins
    for m in meshes:
        assert m.volume() > 0, m.name
        _, dmax = edge_use(m.triangles)
        assert dmax == 1, m.name
    path = export_glb(knight, OUT / "knight.glb", meshes=meshes)
    assert validate_glb(path) == []


# ----------------------------------------------------------------------------- merged, colors, OBJ


def test_merged_export_is_one_node(tower, tower_meshes):
    path = export_glb(tower, OUT / "tower30_merged.glb", per_part=False, meshes=tower_meshes)
    assert validate_glb(path) == []
    _, doc, _, blob = parse_glb(path)
    assert [n["name"] for n in doc["nodes"]] == ["tower30"] and len(doc["meshes"]) == 1
    prims = doc["meshes"][0]["primitives"]
    assert len(prims) == len(doc["materials"])  # one primitive per material
    n_tris = sum(doc["accessors"][p["indices"]]["count"] for p in prims) // 3
    assert n_tris == sum(m.triangle_count for m in tower_meshes)
    ((pos, tris),) = glb_triangles(doc, blob)
    assert volume(pos, tris) == pytest.approx(sum(m.volume() for m in tower_meshes), rel=1e-6)


def test_baked_colors(tower, tower_meshes):
    path = export_glb(tower, OUT / "tower30_baked.glb", bake_colors=True)
    assert validate_glb(path) == []
    _, doc, _, blob = parse_glb(path)
    names = [n["name"] for n in doc["nodes"]]
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            assert "COLOR_0" in attrs
            col = decode(doc, blob, attrs["COLOR_0"])
            assert len(col) == doc["accessors"][attrs["POSITION"]]["count"]
            assert col.min() >= 0.0 and col.max() <= 1.0
    mats = {m["name"]: np.array(m["pbrMetallicRoughness"]["baseColorFactor"][:3]) for m in doc["materials"]}
    # Stone bricks: factor x COLOR_0, back in sRGB, lies on the brick -> mortar segment and uses both ends.
    base_prim = doc["meshes"][names.index("base")]["primitives"][0]
    lin = decode(doc, blob, base_prim["attributes"]["COLOR_0"]).astype(np.float64) * mats["stone"]
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    stone = tower.style.material("stone")
    brick, mortar = np.array(stone.color), np.array(stone.pattern_color)
    t = (srgb - brick) @ (mortar - brick) / ((mortar - brick) @ (mortar - brick))
    off_line = np.linalg.norm(srgb - (brick + t[:, None] * (mortar - brick)), axis=1)
    assert off_line.max() < 1e-4 and t.min() > -1e-3 and t.max() < 1 + 1e-3
    assert (t < 0.1).mean() > 0.1 and (t > 0.8).mean() > 0.03  # brick faces and mortar joints
    brick = np.array([srgb_lin(c) for c in stone.color])
    # Solid gold finial (blended into the roof mesh) keeps COLOR_0 = 1: its color is the factor.
    roof_mesh = doc["meshes"][names.index("roof")]
    roof_col = decode(doc, blob, roof_mesh["primitives"][0]["attributes"]["COLOR_0"])
    roof_pos = decode(doc, blob, roof_mesh["primitives"][0]["attributes"]["POSITION"])
    tip = roof_pos[:, 1] > tower.part("finial").position[1] + 0.05
    assert tip.any() and np.abs(roof_col[tip] - 1.0).max() < 1e-6
    # Darker pattern colors keep the style color as baseColorFactor.
    assert mats["stone"] == pytest.approx(brick, abs=1e-6)


def test_baked_brighter_pattern_raises_the_factor():
    # brick: red base with a cream mortar that is brighter in every channel.
    asset = asset_from("wall", "  - {name: wall, shape: box, size: [2, 1.2, 0.4], material: brick, align: bottom}\n")
    (mesh,) = build_meshes(asset, max_cells=64, bake_colors=True)
    path = export_glb(asset, OUT / "wall_baked.glb", meshes=[mesh])
    assert validate_glb(path) == []
    _, doc, _, blob = parse_glb(path)
    brick = asset.style.material("brick")
    factor = np.array(doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"][:3])
    expected = np.maximum([srgb_lin(c) for c in brick.color], [srgb_lin(c) for c in brick.pattern_color])
    assert factor == pytest.approx(expected, abs=1e-6)
    col = decode(doc, blob, doc["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]).astype(np.float64)
    baked = np.array([[srgb_lin(c) for c in row] for row in mesh.colors.astype(np.float64)])
    assert col * factor == pytest.approx(baked, abs=1e-5)  # factor x COLOR_0 = the baked color exactly


def test_obj_and_mtl(tower, tower_meshes):
    path = export_obj(tower, OUT / "tower30.obj", meshes=tower_meshes)
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text
    lines = text.splitlines()
    v = [ln for ln in lines if ln.startswith("v ")]
    vn = [ln for ln in lines if ln.startswith("vn ")]
    f = [ln for ln in lines if ln.startswith("f ")]
    assert len(v) == len(vn) == sum(m.vertex_count for m in tower_meshes)
    assert len(f) == sum(m.triangle_count for m in tower_meshes)
    assert [ln[2:] for ln in lines if ln.startswith("o ")] == [m.name for m in tower_meshes]
    refs = np.array([[int(x) for x in pair] for ln in f[:2000] for pair in re.findall(r"(\d+)//(\d+)", ln)])
    assert refs.shape == (3 * min(2000, len(f)), 2)  # three v//vn references per face line
    assert (refs[:, 0] == refs[:, 1]).all() and refs.min() >= 1 and refs.max() <= len(v)
    all_idx = np.array([int(t.split("//")[0]) for ln in f for t in ln.split()[1:]])
    assert all_idx.min() == 1 and all_idx.max() == len(v)
    mtllib = next(ln for ln in lines if ln.startswith("mtllib "))[7:]
    mtl = (path.parent / mtllib).read_text(encoding="utf-8")
    newmtl = re.findall(r"^newmtl (\w+)$", mtl, re.M)
    usemtl = {ln[7:] for ln in lines if ln.startswith("usemtl ")}
    assert usemtl == set(newmtl) and "stone" in newmtl
    kd = re.search(r"newmtl stone\nKd ([\d.]+) ([\d.]+) ([\d.]+)", mtl).groups()
    assert [float(x) for x in kd] == pytest.approx(tower.style.material("stone").color, abs=1e-4)


def test_obj_sphere_parses_back():
    asset = asset_from("sphere", SPHERE)
    path = export_obj(asset, OUT / "sphere", max_cells=32, quality="full")
    assert path.name == "sphere.obj" and path.with_suffix(".mtl").is_file()
    lines = path.read_text(encoding="utf-8").splitlines()
    pos = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("v ")])
    tris = np.array([[int(t.split("//")[0]) - 1 for t in ln.split()[1:]] for ln in lines if ln.startswith("f ")])
    assert_closed(tris)
    assert volume(pos, tris) == pytest.approx(4 / 3 * math.pi, rel=0.01)


def test_exports_are_deterministic(tower):
    a = build_meshes(tower, max_cells=48)
    b = build_meshes(tower, max_cells=48)
    for kind, export in (("glb", export_glb), ("obj", export_obj)):
        pa = export(tower, OUT / "det_a" / f"tower30.{kind}", meshes=a)
        pb = export(tower, OUT / "det_b" / f"tower30.{kind}", meshes=b)
        assert pa.read_bytes() == pb.read_bytes(), kind
    assert (OUT / "det_a" / "tower30.mtl").read_bytes() == (OUT / "det_b" / "tower30.mtl").read_bytes()


# ----------------------------------------------------------------------------- decimation


def box1_sdf(p):
    """Exact SDF of the box1 fixture (2 x 1 x 1 m standing on y = 0), points (N, 3)."""
    q = np.abs(np.asarray(p, dtype=np.float64) - [0.0, 0.5, 0.0]) - [1.0, 0.5, 0.5]
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_decimated_tower_parts_stay_closed_named_and_sized(tower_full, tower_meshes):
    assert [m.name for m in tower_meshes] == [m.name for m in tower_full]
    assert [m.members for m in tower_meshes] == [m.members for m in tower_full]
    before = sum(m.triangle_count for m in tower_full)
    after = sum(m.triangle_count for m in tower_meshes)
    assert after * 10 < before
    assert edge_report(tower_meshes) == [] and format_edges(tower_meshes).startswith("MANIFOLD  yes")
    for full, m in zip(tower_full, tower_meshes):
        assert m.triangles_before == full.triangle_count and m.triangle_count < full.triangle_count, m.name
        tris = m.triangles.astype(np.int64)
        assert_closed(tris)
        assert euler(tris) == euler(full.triangles.astype(np.int64)), m.name  # same topology (holes, handles)
        # The bounding box shrinks by at most BBOX_TOL and never grows.
        assert (m.positions.min(axis=0) >= full.positions.min(axis=0) - 1e-6).all(), m.name
        assert (m.positions.max(axis=0) <= full.positions.max(axis=0) + 1e-6).all(), m.name
        assert np.abs(m.positions.min(axis=0) - full.positions.min(axis=0)).max() <= BBOX_TOL + 1e-6, m.name
        assert np.abs(m.positions.max(axis=0) - full.positions.max(axis=0)).max() <= BBOX_TOL + 1e-6, m.name
        assert m.volume() == pytest.approx(full.volume(), rel=0.1), m.name
        assert np.abs(np.linalg.norm(m.normals, axis=1) - 1.0).max() < 1e-5
        fn, _ = face_normals(m.positions, tris)
        assert (np.einsum("ij,ij->i", fn, m.normals[tris].sum(axis=1)) > 0).all(), m.name  # winding = normals
        assert m.tolerance == pytest.approx(QUALITY["medium"] * m.cell)
        # Measured against the SDF: within the tolerance, or no worse than the surface-nets mesh itself
        # (collapses inside flat faces are not probed, hence the small margin).
        assert m.max_error <= max(m.tolerance, m.source_error) + 0.05 * m.tolerance, m.name
        assert set(np.unique(m.tri_part)) <= set(np.unique(full.tri_part))


def test_decimation_keeps_material_seams(tower, tower_full, tower_meshes):
    parts = tower.scene().parts

    def area_by_material(m):
        p = m.positions.astype(np.float64)[m.triangles]
        area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
        out = {}
        for a, pi in zip(area, m.tri_part):
            out[parts[pi].material] = out.get(parts[pi].material, 0.0) + a
        return out

    full = area_by_material(next(m for m in tower_full if m.name == "roof"))
    dec = area_by_material(next(m for m in tower_meshes if m.name == "roof"))
    assert set(dec) == set(full) == {"roof", "gold"}
    assert dec["gold"] == pytest.approx(full["gold"], rel=0.25)  # the finial seam stays where it was
    assert dec["roof"] == pytest.approx(full["roof"], rel=0.05)


def test_decimated_box_keeps_corners_and_crisp_edges():
    box = load_asset(FIX / "box1.yaml")
    (full,) = build_meshes(box, max_cells=64, quality="full")
    (m,) = decimate_meshes(box, [full])
    assert m.triangle_count < 400 and full.triangle_count > 10_000
    assert_closed(m.triangles)
    assert m.volume() == pytest.approx(2.0, rel=1e-3)
    assert np.abs(m.positions.min(axis=0) - [-1.0, 0.0, -0.5]).max() < 1e-4
    assert np.abs(m.positions.max(axis=0) - [1.0, 1.0, 0.5]).max() < 1e-4
    pos = m.positions.astype(np.float64)
    assert np.abs(box1_sdf(pos)).max() < 0.05 * m.cell  # every vertex on the box, corners included
    # Shading: where the interpolated normal near a triangle corner is off the true normal, the
    # triangle is a thin strip along a sharp edge, not a whole face smeared.
    tri = pos[m.triangles]
    nrm = m.normals.astype(np.float64)[m.triangles]
    near = np.array([[0.75, 0.125, 0.125], [0.125, 0.75, 0.125], [0.125, 0.125, 0.75]])
    pts = np.einsum("kc,tcx->tkx", near, tri).reshape(-1, 3)
    h = 0.25 * m.cell
    grad = np.stack([(box1_sdf(pts + h * e) - box1_sdf(pts - h * e)) / (2 * h) for e in np.eye(3)], axis=1)
    off = (unit(np.einsum("kc,tcx->tkx", near, nrm)) * unit(grad).reshape(-1, 3, 3)).sum(axis=2)
    smeared = (off < math.cos(math.radians(SHADE_ANGLE))).any(axis=1)
    edge = np.stack([np.linalg.norm(tri[:, k] - tri[:, (k + 1) % 3], axis=1) for k in range(3)], axis=1)
    width = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / edge.max(axis=1)
    assert smeared.any() and (width[smeared] <= SHADE_WIDTH * m.cell + 1e-6).all()
    assert width[~smeared].max() > 10 * m.cell  # the faces themselves are a few big triangles


def test_decimated_sphere_stays_within_tolerance():
    asset = asset_from("sphere", SPHERE)
    (full,) = build_meshes(asset, max_cells=64, quality="full")
    (m,) = decimate_meshes(asset, [full])
    tol = QUALITY["medium"] * full.cell
    assert m.tolerance == pytest.approx(tol)
    assert 100 < m.triangle_count and m.triangle_count * 10 < full.triangle_count
    tris = m.triangles.astype(np.int64)
    assert_closed(tris)
    assert euler(tris) == 2
    pos, center = m.positions.astype(np.float64), np.array([0.0, 1.0, 0.0])
    assert np.abs(np.linalg.norm(pos - center, axis=1) - 1.0).max() <= tol + 1e-6
    centroid_r = np.linalg.norm(pos[tris].mean(axis=1) - center, axis=1)
    assert np.abs(centroid_r - 1.0).max() <= tol + 1e-6  # the flat triangles stay within tolerance too
    assert m.max_error == pytest.approx(max(np.abs(np.linalg.norm(pos - center, axis=1) - 1).max(),
                                            np.abs(centroid_r - 1).max()), abs=1e-5)
    assert (np.einsum("ij,ij->i", m.normals, pos - center) > 0).all()  # normals still point outward
    fn, cen = face_normals(pos, tris)
    assert (np.einsum("ij,ij->i", fn, cen - center) > 0).all()
    assert m.volume() == pytest.approx(4 / 3 * math.pi, rel=0.02)


def test_decimation_keeps_holes_and_recesses():
    holed = asset_from("holed", "  - {name: block, shape: box, size: [2, 1, 1], material: stone, align: bottom}\n"
                                "  - {name: hole, shape: cylinder, radius: 0.25, height: 2, rotate: [90, 0, 0], op: cut,"
                                " from: [block], at: [0, 0.5, 0]}\n")
    niche = asset_from("niche", "  - {name: wall, shape: box, size: [2, 2, 1], material: stone, align: bottom}\n"
                                "  - {name: niche, shape: box, size: [0.6, 1.0, 0.4], op: cut, on: wall.front, align: center}\n")
    for asset, genus_euler, volume_ in ((holed, 0, 2.0 - math.pi * 0.25**2), (niche, 2, 4.0 - 0.6 * 1.0 * 0.2)):
        (m,) = build_meshes(asset, max_cells=64)  # the default: quality medium
        assert m.source_triangles is not None and m.triangle_count * 10 < m.source_triangles
        assert_closed(m.triangles)
        assert euler(m.triangles.astype(np.int64)) == genus_euler
        assert m.volume() == pytest.approx(volume_, rel=0.02)


def test_quality_presets_and_triangle_budget():
    asset = asset_from("sphere", SPHERE)
    (full,) = build_meshes(asset, max_cells=48, quality="full")
    assert full.source_triangles is None and full.triangles_before == full.triangle_count
    counts = {q: decimate_meshes(asset, [full], q)[0].triangle_count for q in ("high", "medium", "low")}
    assert full.triangle_count > counts["high"] > counts["medium"] > counts["low"]
    assert decimate_meshes(asset, [full], "full")[0] is full  # nothing to do
    (small,) = decimate_meshes(asset, [full], max_tris=120)
    assert small.triangle_count <= 120 < counts["medium"]
    assert_closed(small.triangles)
    assert euler(small.triangles.astype(np.int64)) == 2
    assert small.tolerance > QUALITY["medium"] * full.cell  # the budget relaxed the tolerance
    assert small.volume() == pytest.approx(4 / 3 * math.pi, rel=0.1)
    (same,) = build_meshes(asset, max_cells=48, quality="medium", max_tris=10**6)
    assert same.triangle_count <= counts["medium"]
    path = export_glb(asset, OUT / "sphere_budget.glb", max_cells=48, max_tris=120)
    assert validate_glb(path) == []
    _, doc, _, blob = parse_glb(path)
    ((pos, tris),) = glb_triangles(doc, blob)
    assert len(tris) <= 120
    assert_closed(tris)


def test_quality_errors():
    box = load_asset(FIX / "box1.yaml")
    with pytest.raises(SightlineError, match="quality must be one of") as exc:
        build_meshes(box, max_cells=16, quality="ultra")
    assert exc.value.hint
    for bad in (0, 3, "100", True, 12.5):
        with pytest.raises(SightlineError, match="max_tris must be an integer"):
            build_meshes(box, max_cells=16, max_tris=bad)
    with pytest.raises(SightlineError, match="bake_colors needs the full mesh"):
        build_meshes(box, max_cells=16, bake_colors=True, quality="low")
    with pytest.raises(SightlineError, match="bake_colors needs the full mesh"):
        export_glb(box, OUT / "baked_budget.glb", max_cells=16, bake_colors=True, max_tris=100)
    with pytest.raises(SightlineError, match="max_tris decimates"):
        build_meshes(box, max_cells=16, quality="full", max_tris=100)
    (baked,) = build_meshes(box, max_cells=16, bake_colors=True)  # baking exports the full mesh
    assert baked.source_triangles is None and baked.colors is not None


def test_summary_reports_triangles_before_and_after(tower_full, tower_meshes):
    lines = mesh_summary(tower_meshes).splitlines()
    assert lines[0].split() == ["PART", "VERTS", "TRIS", "BEFORE", "CELL", "DEV_MM", "VOLUME"]
    base = lines[1].split()
    m = tower_meshes[0]
    assert base[0] == m.name and int(base[2]) == m.triangle_count and int(base[3]) == tower_full[0].triangle_count
    assert float(base[5]) == pytest.approx(m.max_error * 1000, abs=0.01)
    total = lines[-1].split()
    after, before = sum(x.triangle_count for x in tower_meshes), sum(x.triangle_count for x in tower_full)
    assert total[0] == "total" and int(total[2]) == after and int(total[3]) == before
    line = format_decimation(tower_meshes)
    assert line.startswith(f"DECIMATE  {before} -> {after} triangles ({before / after:.1f}x fewer)")
    assert "tolerance" in line and "surface nets alone" in line
    assert format_decimation(tower_full).startswith("DECIMATE  off")
    full_rows = mesh_summary(tower_full).splitlines()
    assert full_rows[1].split()[2] == full_rows[1].split()[3] and full_rows[1].split()[5] == "-"


def test_nonmanifold_vertices_found_and_kept():
    # Two tetrahedra sharing edge 0-1 (4 triangles on it) and a loose triangle (open edges).
    tris = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2], [0, 4, 1], [0, 1, 5], [0, 5, 4], [1, 4, 5],
                     [6, 7, 8]])
    assert np.flatnonzero(nonmanifold_vertices(tris, 9)).tolist() == [0, 1, 6, 7, 8]

    # Two closed balls welded at one vertex: two fans around it, every edge still joins 2 triangles.
    ball = lambda c: (lambda p3: np.sqrt((p3[0] - c) ** 2 + p3[1] ** 2 + p3[2] ** 2) - 0.6)  # noqa: E731
    verts, tris1 = surface_nets(ball(0.0), [-0.6] * 3, [0.6] * 3, 0.08)
    assert not nonmanifold_vertices(tris1, len(verts)).any()
    tip, tail = int(np.argmax(verts[:, 0])), int(np.argmin(verts[:, 0]))
    shift = verts[tip, 0] - verts[tail, 0]
    tris2 = tris1 + len(verts)
    tris2[tris2 == tail + len(verts)] = tip
    pos = np.concatenate([verts, verts + [shift, 0, 0]])
    tris = np.concatenate([tris1, tris2])
    used = np.unique(tris)
    pos, tris = pos[used], np.searchsorted(used, tris)
    weld = int(np.flatnonzero(used == tip)[0])
    fn = lambda p3: np.minimum(ball(0.0)(p3), ball(shift)(p3)).astype(np.float32)  # noqa: E731
    assert np.flatnonzero(nonmanifold_vertices(tris, len(pos))).tolist() == [weld]
    normals = vertex_normals(fn, pos, tris, 0.02)
    (out,) = decimate([DecimationInput(pos, tris, normals, np.zeros(len(tris), dtype=np.int64), fn, 0.08)], 0.25)
    assert len(out.triangles) * 4 < len(tris)
    t = out.triangles
    und, dmax = edge_use(t)
    assert (und == 2).all() and dmax == 1  # still closed, nothing new became non-manifold
    assert np.flatnonzero(nonmanifold_vertices(t, len(out.positions))).tolist() == [
        int(np.flatnonzero(out.vertex_ids == weld)[0])]
    assert np.allclose(out.positions[out.vertex_ids == weld], pos[weld])  # the weld never moved


# ----------------------------------------------------------------------------- errors and paths


def test_errors():
    box = load_asset(FIX / "box1.yaml")
    with pytest.raises(SightlineError, match="max_cells") as exc:
        build_meshes(box, max_cells=4)
    assert exc.value.hint
    with pytest.raises(SightlineError, match="max_cells"):
        build_meshes(box, max_cells="64")
    with pytest.raises(SightlineError, match=r"writes \.glb"):
        export_glb(box, OUT / "box1.gltf")
    with pytest.raises(SightlineError, match=r"writes \.obj"):
        export_obj(box, OUT / "box1.glb")
    gone = asset_from("gone", "  - {name: pebble, shape: sphere, radius: 0.2, material: stone}\n"
                              "  - {name: eraser, shape: box, size: [1, 1, 1], op: cut, from: [pebble]}\n")
    with pytest.raises(SightlineError, match="cut away"):
        build_meshes(gone)


def test_directory_out_path_uses_asset_name():
    box = load_asset(FIX / "box1.yaml")
    folder = OUT / "folder"
    folder.mkdir(parents=True, exist_ok=True)
    path = export_glb(box, folder, max_cells=16)
    assert path == folder / "box1.glb" and validate_glb(path) == []


# ----------------------------------------------------------------------------- size and speed


def test_tower30_counts_and_timing(tower, capsys):
    t0 = time.perf_counter()
    full = build_meshes(tower, quality="full")
    t1 = time.perf_counter()
    meshes = decimate_meshes(tower, full)
    t2 = time.perf_counter()
    glb = export_glb(tower, OUT / "tower30_timed.glb", meshes=meshes)
    t3 = time.perf_counter()
    obj = export_obj(tower, OUT / "tower30_timed.obj", meshes=meshes)
    t4 = time.perf_counter()
    before = sum(m.triangle_count for m in full)
    tris = sum(m.triangle_count for m in meshes)
    with capsys.disabled():
        print(f"\ntower30 max_cells=128: {len(meshes)} meshes, {before} -> {tris} triangles "
              f"({sum(m.vertex_count for m in meshes)} vertices); mesh {t1 - t0:.2f} s, decimate {t2 - t1:.2f} s, "
              f"glb {t3 - t2:.2f} s ({glb.stat().st_size / 1e6:.3f} MB), obj {t4 - t3:.2f} s "
              f"({obj.stat().st_size / 1e6:.3f} MB)")
    assert 50_000 < before < 250_000
    assert sum(m.triangles_before for m in meshes) == before
    assert 1_000 < tris < before / 10
    assert t1 - t0 < 20.0 and t2 - t1 < 40.0
    assert "total" in mesh_summary(meshes).splitlines()[-1]
