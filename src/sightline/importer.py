"""Import a ready mesh model (glTF .glb) as a descriptive Sightline asset (SPEC 13).

Every mesh primitive (and, with splitting, every connected piece of it) becomes one named part: an
oriented primitive fitted to its vertices. The shape is chosen by how much of its oriented bounding
box the piece fills (box ~1, cylinder ~0.79, ellipsoid ~0.52, cone ~0.26), the material is the style
material nearest to the piece's base colour (or the mean of its texture under its UVs), and every
part carries its fit error. The result is a starting point an agent can render, measure and refine
part by part; it is not a faithful copy of the mesh.
"""

from __future__ import annotations

import io
import math
import pathlib
import re

import numpy as np
from PIL import Image

from .errors import SightlineError
from .export import accessor_data, read_glb
from .model import make_part, rotation_matrix

FILL_SHAPES = (("box", 1.0), ("cylinder", math.pi / 4), ("ellipsoid", math.pi / 6), ("cone", math.pi / 12))
ROUND_AXIS = 0.12  # two extents within 12% of each other make a round cross-section
SNAP_DEG = 4.0  # principal axes within this of a world axis snap to it (no rotate for axis-aligned pieces)
MIN_PIECE = 0.004  # pieces smaller than this share of the model's bounding-box diagonal are dropped
MAX_PARTS = 120
WELD = 1e-5  # vertices closer than this (m) are one vertex when finding connected pieces


# ----------------------------------------------------------------------------- glTF reading


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


def _world_matrices(doc: dict) -> dict[int, np.ndarray]:
    nodes = doc.get("nodes", [])
    scene = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {"nodes": list(range(len(nodes)))}
    out: dict[int, np.ndarray] = {}

    def walk(i, parent):
        m = parent @ _node_matrix(nodes[i])
        out[i] = m
        for c in nodes[i].get("children", []):
            walk(c, m)

    for root in scene.get("nodes", []):
        walk(root, np.eye(4))
    return out


def _accessor(doc, blob, index) -> np.ndarray:
    a = accessor_data(doc, blob, index).astype(np.float64)
    acc = doc["accessors"][index]
    if acc.get("normalized"):
        a = a / {5121: 255.0, 5123: 65535.0, 5120: 127.0, 5122: 32767.0}.get(acc["componentType"], 1.0)
    return a


def _texture_image(doc, blob, tex_index) -> np.ndarray | None:
    try:
        src = doc["textures"][tex_index]["source"]
        img = doc["images"][src]
        view = doc["bufferViews"][img["bufferView"]]
        data = blob[view.get("byteOffset", 0): view.get("byteOffset", 0) + view["byteLength"]]
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB")).astype(np.float64) / 255.0
    except (KeyError, IndexError, OSError, ValueError):
        return None


def _srgb_from_linear(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def read_pieces(path) -> tuple[list[dict], dict]:
    """Mesh primitives of a .glb in world space: [{name, verts (N,3), tris (M,3), color sRGB, uv}]."""
    doc, blob = read_glb(path)
    world = _world_matrices(doc)
    nodes = doc.get("nodes", [])
    materials = doc.get("materials", [])
    textures: dict[int, np.ndarray | None] = {}
    pieces = []
    skinned = 0
    for ni, node in enumerate(nodes):
        if "mesh" not in node or ni not in world:
            continue
        mesh = doc["meshes"][node["mesh"]]
        skin = doc["skins"][node["skin"]] if "skin" in node else None
        if skin is not None:
            skinned += 1
            inv = _accessor(doc, blob, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1) \
                if "inverseBindMatrices" in skin else np.tile(np.eye(4), (len(skin["joints"]), 1, 1))
            joints = np.stack([world.get(j, np.eye(4)) @ inv[k] for k, j in enumerate(skin["joints"])])
        for pi, prim in enumerate(mesh["primitives"]):
            if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                continue
            v = _accessor(doc, blob, prim["attributes"]["POSITION"])
            vh = np.concatenate([v, np.ones((len(v), 1))], axis=1)
            if skin is not None and "JOINTS_0" in prim["attributes"]:
                j = accessor_data(doc, blob, prim["attributes"]["JOINTS_0"]).astype(np.int64)
                w = _accessor(doc, blob, prim["attributes"]["WEIGHTS_0"])
                w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
                skinm = np.einsum("nk,nkij->nij", w, joints[j])
                v = np.einsum("nij,nj->ni", skinm, vh)[:, :3]
            else:
                v = (vh @ world[ni].T)[:, :3]
            tris = (accessor_data(doc, blob, prim["indices"]).astype(np.int64).reshape(-1, 3)
                    if "indices" in prim else np.arange(len(v)).reshape(-1, 3))
            color = np.array([0.7, 0.7, 0.7])
            mat_name = None
            uv = _accessor(doc, blob, prim["attributes"]["TEXCOORD_0"]) if "TEXCOORD_0" in prim["attributes"] else None
            tex = None
            if "material" in prim and prim["material"] < len(materials):
                mat = materials[prim["material"]]
                mat_name = mat.get("name")
                pbr = mat.get("pbrMetallicRoughness", {})
                color = _srgb_from_linear(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3])
                if "baseColorTexture" in pbr and uv is not None:
                    ti = pbr["baseColorTexture"]["index"]
                    if ti not in textures:
                        textures[ti] = _texture_image(doc, blob, ti)
                    tex = textures[ti]
            name = node.get("name") or mesh.get("name") or f"mesh{node['mesh']}"
            if len(mesh["primitives"]) > 1:
                name = mat_name or f"{name}_{pi}"
            pieces.append({"name": name, "verts": v, "tris": tris, "color": color, "uv": uv, "tex": tex,
                           "factor": color, "source": f"node {ni} '{node.get('name', '')}' primitive {pi}"
                           + (f" material '{mat_name}'" if mat_name else "")})
    return pieces, {"nodes": len(nodes), "meshes": len(doc.get("meshes", [])), "skinned": skinned,
                    "materials": len(materials)}


# ----------------------------------------------------------------------------- pieces and fitting


def _components(verts: np.ndarray, tris: np.ndarray) -> list[np.ndarray]:
    """Triangle index arrays of the connected pieces (vertices welded by position)."""
    q = np.round(verts / WELD).astype(np.int64)
    _, weld = np.unique(q, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    t = weld[tris]
    n = int(weld.max()) + 1 if weld.size else 0
    parent = np.arange(n)

    def find_all(p):
        while True:
            nxt = p[p]
            if np.array_equal(nxt, p):
                return p
            p = nxt

    for a, b in ((0, 1), (1, 2), (0, 2)):
        x, y = t[:, a], t[:, b]
        while True:
            parent = find_all(parent)
            rx, ry = parent[x], parent[y]
            diff = rx != ry
            if not diff.any():
                break
            np.minimum.at(parent, np.maximum(rx[diff], ry[diff]), np.minimum(rx[diff], ry[diff]))
    parent = find_all(parent)
    root = parent[t[:, 0]]
    out = []
    for r in np.unique(root):
        out.append(np.flatnonzero(root == r))
    return out


def _tri_areas(v, t):
    return 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)


def _volume(v, t) -> float:
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _closed(t) -> bool:
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool((counts == 2).all())


def _euler_xyz(r: np.ndarray) -> tuple[float, float, float]:
    """Degrees (rx, ry, rz) with R = Rz Ry Rx."""
    sb = -r[2, 0]
    b = math.asin(max(-1.0, min(1.0, sb)))
    if abs(sb) < 0.9999:
        a = math.atan2(r[2, 1], r[2, 2])
        g = math.atan2(r[1, 0], r[0, 0])
    else:
        a = math.atan2(-r[1, 2], r[1, 1])
        g = 0.0
    return tuple(round(math.degrees(x), 1) + 0.0 for x in (a, b, g))


def fit_piece(verts: np.ndarray, tris: np.ndarray) -> dict:
    """Oriented primitive for one piece: shape, params, center, rotate (deg), fill ratio."""
    used = np.unique(tris)
    p = verts[used]
    w = np.zeros(len(verts))
    np.add.at(w, tris.ravel(), np.repeat(_tri_areas(verts, tris), 3) / 3.0)
    w = w[used] + 1e-12
    mean = (p * w[:, None]).sum(0) / w.sum()
    cov = ((p - mean) * w[:, None]).T @ (p - mean) / w.sum()
    _, vecs = np.linalg.eigh(cov)
    axes = vecs[:, ::-1]  # columns: principal axes, largest first
    # snap to world axes when close
    snapped = np.zeros((3, 3))
    used_axes = set()
    snap_ok = True
    for k in range(3):
        j = int(np.argmax(np.abs(axes[:, k])))
        if j in used_axes or abs(axes[j, k]) < math.cos(math.radians(SNAP_DEG)):
            snap_ok = False
            break
        used_axes.add(j)
        snapped[j, k] = 1.0
    if snap_ok:
        axes = snapped
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    local = (p - mean) @ axes
    lo, hi = local.min(0), local.max(0)
    ext = np.maximum(hi - lo, 1e-4)
    center = mean + axes @ ((lo + hi) / 2.0)
    closed = _closed(tris)
    fill = _volume(verts, tris) / float(np.prod(ext)) if closed else 1.0
    # choose a shape; round cross-sections need two similar extents
    order = np.argsort(ext)  # small..large
    pairs = {"0-1": (order[0], order[1], order[2]), "1-2": (order[1], order[2], order[0])}
    round_pair = None
    for a_, b_, c_ in pairs.values():
        if abs(ext[a_] - ext[b_]) <= ROUND_AXIS * max(ext[a_], ext[b_]):
            round_pair = (a_, b_, c_)
            break
    best = min(FILL_SHAPES, key=lambda s: abs(s[1] - fill))[0] if closed else "box"
    if best in ("cylinder", "cone") and round_pair is None:
        best = "box" if fill > 0.65 else "ellipsoid"
    if best in ("cylinder", "cone"):
        a_, b_, c_ = round_pair  # c_ is the main axis -> local Y
        cols = [axes[:, a_], axes[:, c_], axes[:, b_]]
        size = [ext[a_], ext[c_], ext[b_]]
    else:
        cols = [axes[:, 0], axes[:, 1], axes[:, 2]]
        size = list(ext)
    rmat = np.stack(cols, axis=1)
    if np.linalg.det(rmat) < 0:
        rmat[:, 2] *= -1
    rx, ry, rz = _euler_xyz(rmat)
    if best == "box":
        params = {"size": [round(float(s), 4) for s in size]}
    elif best == "ellipsoid":
        params = {"size": [round(float(s), 4) for s in size]}
    elif best == "cylinder":
        params = {"radius": round(float(size[0] + size[2]) / 4.0, 4), "height": round(float(size[1]), 4)}
    else:  # cone: wide end where the cross-section is wider
        along = (p - center) @ rmat[:, 1]
        across = np.linalg.norm((p - center) - np.outer(along, rmat[:, 1]), axis=1)
        top = across[along > 0].mean() if (along > 0).any() else 0.0
        bottom = across[along <= 0].mean() if (along <= 0).any() else 0.0
        if top > bottom:  # wide end up: flip so the radius is at the bottom, as the shape expects
            rmat[:, 1] *= -1
            rmat[:, 0] *= -1
            rx, ry, rz = _euler_xyz(rmat)
        params = {"radius": round(float(size[0] + size[2]) / 4.0, 4), "height": round(float(size[1]), 4)}
    return {"shape": best, "params": params, "center": center, "rotate": (rx, ry, rz), "fill": fill, "closed": closed}


def _fit_error(fit: dict, verts: np.ndarray) -> float:
    part = make_part("p", fit["shape"], dict(fit["params"]), fit["center"], fit["rotate"], "metal")
    d = part.sdf3(np.ascontiguousarray(verts.T.astype(np.float32)))
    return float(np.sqrt(np.mean(np.asarray(d, dtype=np.float64) ** 2)))


def _piece_color(piece: dict, tris: np.ndarray) -> np.ndarray:
    if piece["tex"] is None or piece["uv"] is None:
        return piece["color"]
    tex = piece["tex"]
    idx = np.unique(tris)
    uv = piece["uv"][idx] % 1.0
    h, w = tex.shape[:2]
    cols = tex[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
    return np.clip(cols.mean(0) * np.asarray(piece["factor"]), 0, 1)


def _nearest_material(color: np.ndarray, style) -> tuple[str, float]:
    from .reference import delta_e

    best = None
    for name in style.materials:
        if name == "helper":
            continue
        d = float(delta_e(color, np.asarray(style.material(name).color)))
        if best is None or d < best[1]:
            best = (name, d)
    return best


def _slug(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s or "part").strip("_").lower()
    s = re.sub(r"_effect$", "", s)
    if not s or not s[0].isalpha():
        s = "p_" + s
    return s[:40]


def import_glb(path, name: str | None = None, *, style=None, split: bool = True, max_parts: int = MAX_PARTS) -> dict:
    """Fit primitives to a .glb's pieces. Returns {name, parts: [...], facts, yaml}."""
    path = pathlib.Path(path)
    if style is None:
        from .style import load_style

        style = load_style("default")
    pieces, facts = read_pieces(path)
    if not pieces:
        raise SightlineError(f"{path.name} has no triangle meshes", hint="export the model with mesh data (glb)")
    allv = np.concatenate([pc["verts"] for pc in pieces])
    lo, hi = allv.min(0), allv.max(0)
    diag = float(np.linalg.norm(hi - lo))
    shift = np.array([-(lo[0] + hi[0]) / 2.0, -lo[1], -(lo[2] + hi[2]) / 2.0])
    items = []
    for pc in pieces:
        v = pc["verts"] + shift
        comps = _components(v, pc["tris"]) if split else [np.arange(len(pc["tris"]))]
        comps.sort(key=lambda c: -_tri_areas(v, pc["tris"][c]).sum())
        for k, ti in enumerate(comps):
            t = pc["tris"][ti]
            pts = v[np.unique(t)]
            size = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            if size < MIN_PIECE * diag:
                continue
            items.append((pc, k, len(comps), t, v, size))
    dropped = 0
    if len(items) > max_parts:
        items.sort(key=lambda it: -it[5])
        dropped = len(items) - max_parts
        items = items[:max_parts]
    parts = []
    names: dict[str, int] = {}
    for pc, k, ncomp, t, v, size in items:
        fit = fit_piece(v, t)
        err = _fit_error(fit, v[np.unique(t)])
        col = _piece_color(pc, t)
        mat, de = _nearest_material(col, style)
        base = _slug(pc["name"]) + (f"_{k}" if ncomp > 1 else "")
        n = names.get(base, 0)
        names[base] = n + 1
        nm = base if n == 0 else f"{base}_{n}"
        parts.append({"name": nm, "fit": fit, "error": err, "material": mat, "color": col, "de": de,
                      "source": pc["source"] + (f" piece {k + 1}/{ncomp}" if ncomp > 1 else ""), "size": size})
    name = name or _slug(path.stem)
    yaml = _to_yaml(name, path, parts, facts, dropped, hi - lo)
    return {"name": name, "parts": parts, "facts": {**facts, "dropped": dropped, "size": (hi - lo).tolist()}, "yaml": yaml}


def _num(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _to_yaml(name, path, parts, facts, dropped, size) -> str:
    from .reference import _hex

    lines = [f"asset: {name}", "brief: >",
             f"  Imported from {path.name}: {len(parts)} parts, each an oriented primitive fitted to one mesh piece",
             "  (shape by how much of its bounding box it fills, material nearest to its colour). A starting point:",
             "  render it, compare it with the original and refine the parts with the largest fit error first.",
             "style: default", f"# import: {facts['meshes']} meshes, {facts['materials']} materials"
             + (f", {facts['skinned']} skinned (bind pose)" if facts["skinned"] else "")
             + (f"; {dropped} smallest pieces dropped" if dropped else ""),
             f"# size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m (w x h x d); fit error = RMS distance of the piece's vertices from the part's surface",
             "parts:"]
    for p in parts:
        f = p["fit"]
        lines.append(f"  - name: {p['name']}  # {p['source']}; fill {f['fill']:.2f}; fit error {p['error'] * 100:.1f} cm; colour {_hex(p['color'])} (dE {p['de']:.0f})")
        lines.append(f"    shape: {f['shape']}")
        for k, v in f["params"].items():
            lines.append(f"    {k}: " + (f"[{', '.join(_num(x) for x in v)}]" if isinstance(v, list) else _num(v)))
        if any(abs(a) > 0.05 for a in f["rotate"]):
            lines.append(f"    rotate: [{', '.join(_num(a) for a in f['rotate'])}]")
        lines.append(f"    material: {p['material']}")
        c = f["center"]
        lines.append(f"    at: [{_num(c[0])}, {_num(c[1])}, {_num(c[2])}]")
    lines += ["checks:", "  - grounded",
              f"  - size: {{width: [{size[0] * 0.95:.2f}, {size[0] * 1.05:.2f}], height: [{size[1] * 0.95:.2f}, {size[1] * 1.05:.2f}], why: \"the imported model's size\"}}"]
    return "\n".join(lines) + "\n"


def format_import(result: dict) -> str:
    f = result["facts"]
    parts = result["parts"]
    lines = [f"IMPORT  {result['name']}: {len(parts)} parts from {f['meshes']} meshes / {f['materials']} materials"
             + (f", {f['skinned']} skinned mesh(es) in bind pose" if f["skinned"] else "")
             + (f"; {f['dropped']} smallest pieces dropped" if f["dropped"] else "")]
    lines.append(f"SIZE    {f['size'][0]:.2f} x {f['size'][1]:.2f} x {f['size'][2]:.2f} m (w x h x d)")
    shapes = {}
    for p in parts:
        shapes[p["fit"]["shape"]] = shapes.get(p["fit"]["shape"], 0) + 1
    lines.append("SHAPES  " + ", ".join(f"{k} {v}" for k, v in sorted(shapes.items(), key=lambda kv: -kv[1])))
    errs = sorted(parts, key=lambda p: -p["error"] / max(p["size"], 1e-6))
    lines.append("WORST FITS (error relative to the piece's size; refine these first):")
    for p in errs[:10]:
        lines.append(f"  {p['name']:<28} {p['fit']['shape']:<9} fit error {p['error'] * 100:5.1f} cm on a {p['size']:.2f} m piece  ({p['source']})")
    return "\n".join(lines)
