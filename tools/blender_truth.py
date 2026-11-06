"""Render a ready model in headless Blender as a reference "picture" plus a ground-truth ID map.

    blender -b --factory-startup --python tools/blender_truth.py -- model.glb out_dir [view] [px]

Writes out_dir/picture.png (Workbench studio light, material colours, transparent background),
out_dir/ids.png (every material slot of every object a unique flat colour, no anti-aliasing) and
out_dir/ids.json ({"#rrggbb": {"object", "material", "color"}}), plus the model's world bounds.
view: front (default; looks toward -Y in Blender, i.e. toward -Z in glTF/Sightline) or right.
Runs on a factory-startup scene: nothing of the user's is opened or changed.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
src, out = os.path.abspath(args[0]), os.path.abspath(args[1])
view = args[2] if len(args) > 2 else "front"
px = int(args[3]) if len(args) > 3 else 1024
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, guess_original_bind_pose=False)
scene = bpy.context.scene
for a in [o for o in scene.objects if o.type == "ARMATURE"]:
    a.data.pose_position = "REST"  # the bind pose, as `sightline import` reads skinned meshes
bpy.context.view_layer.update()
objs = [o for o in scene.objects if o.type == "MESH"]
deps = bpy.context.evaluated_depsgraph_get()
lo = Vector((1e9, 1e9, 1e9))
hi = Vector((-1e9, -1e9, -1e9))
for o in objs:
    ev = o.evaluated_get(deps)
    me = ev.to_mesh()
    for v in me.vertices:
        w = ev.matrix_world @ v.co
        lo = Vector(map(min, lo, w))
        hi = Vector(map(max, hi, w))
    ev.to_mesh_clear()
center = (lo + hi) / 2
size = hi - lo

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"
if view == "right":
    d = Vector((1, 0, 0))
    w_, h_ = size.y, size.z
else:  # front: glTF +Z is Blender -Y; the camera sits on -Y looking toward +Y
    d = Vector((0, -1, 0))
    w_, h_ = size.x, size.z
margin = 1.06
aspect = w_ / max(h_, 1e-6)
scene.render.resolution_x = px if aspect >= 1 else max(16, int(px * aspect))
scene.render.resolution_y = max(16, int(px / aspect)) if aspect >= 1 else px
cam.data.ortho_scale = max(w_, h_) * margin
cam.location = center + d * (max(size) * 4 + 1)
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.clip_end = max(size) * 10 + 10
scene.render.film_transparent = True
scene.render.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light = "STUDIO"
sh.color_type = "MATERIAL"
sh.show_cavity = True
sh.show_shadows = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = os.path.join(out, "picture.png")
bpy.ops.render.render(write_still=True)

# ID pass: a unique flat colour per (object, material slot)
ids = {}
k = 0
for o in objs:
    for si, slot in enumerate(o.material_slots or [None]):
        k += 1
        r, g, b = (k * 37) % 251 + 2, (k * 91) % 247 + 4, (k * 173) % 241 + 8
        mat = bpy.data.materials.new(f"id_{k}")
        mat.diffuse_color = (r / 255.0, g / 255.0, b / 255.0, 1.0)
        orig = slot.material if slot else None
        base = list(orig.diffuse_color[:3]) if orig else [0.8, 0.8, 0.8]
        if orig and orig.use_nodes:
            bsdf = next((n for n in orig.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf:
                base = list(bsdf.inputs["Base Color"].default_value[:3])
        ids[f"#{r:02x}{g:02x}{b:02x}"] = {"object": o.name, "material": orig.name if orig else None, "color_linear": base}
        if slot:
            slot.material = mat
        else:
            o.data.materials.append(mat)
sh.light = "FLAT"
sh.show_cavity = False
sh.show_shadows = False
sh.show_specular_highlight = False
scene.display.render_aa = "OFF"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.filepath = os.path.join(out, "ids.png")
bpy.ops.render.render(write_still=True)
with open(os.path.join(out, "ids.json"), "w") as f:
    json.dump({"ids": ids, "bounds": [list(lo), list(hi)], "view": view}, f, indent=1)
print("TRUTH", out, len(ids), "ids", [round(s, 3) for s in size])
