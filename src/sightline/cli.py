"""The `sightline` command (SPEC 11).

Every command prints what it measured and the absolute paths of the files it wrote, so an agent
can open them right away. Exit codes: 0 ok, 1 a check failed, 2 an error (a SightlineError with
its `where` and `hint` on stderr, or a usage error).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import textwrap
import traceback

from . import __version__
from .errors import SightlineError

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ----------------------------------------------------------------------------- helpers


def _utf8_streams() -> None:
    """Reports contain '→', '·', '—' and '°'; the Windows console code page cannot encode them."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _show(path) -> str:
    """Absolute posix path, ready for an agent's file tools."""
    return pathlib.Path(path).resolve().as_posix()


def _wrote(paths) -> None:
    for p in paths:
        print(f"wrote {_show(p)}")


def _root(args, start=None) -> pathlib.Path:
    from .style import resolve_root

    return resolve_root(getattr(args, "root", None), start=start)


def _load(name_or_path, args):
    from .loader import load_asset

    return load_asset(name_or_path, getattr(args, "root", None))


def _out_dir(asset, args) -> pathlib.Path:
    """--out, else <root>/out/<asset>."""
    if getattr(args, "out", None):
        return pathlib.Path(args.out)
    return pathlib.Path(asset.root) / "out" / asset.name


def _split_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    items = [t.strip() for t in text.split(",") if t.strip()]
    return items or None


def _plain(value):
    """json.dumps fallback for numpy scalars/arrays and paths inside check data."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, pathlib.PurePath):
        return value.as_posix()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=_plain))


def _result_dict(res) -> dict:
    return {"kind": res.kind, "passed": res.passed, "measured": res.measured, "expected": res.expected,
            "message": res.message, "why": res.why, "data": res.data}


def _hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


# ----------------------------------------------------------------------------- new

_TEMPLATE = """\
asset: {name}
brief: >
{brief}
style: {style}
tags: []
# Meters; +Y up, +X right, +Z toward the viewer of the front view; the ground is y = 0.
# `sightline shapes` lists shapes and anchors, `sightline styles` the materials and colors.
parts:
  - name: base                 # the ground part: its bottom anchor goes to at [0, 0, 0]
    shape: box
    size: [1.2, 0.2, 1.2]
    material: stone
    align: bottom
  - name: body                 # on: its own bottom (the opposite anchor) sits on base's top
    shape: box
    size: [0.9, 1.0, 0.9]
    material: wood
    on: base.top
  # ---- examples: uncomment and adapt (parts may only refer to EARLIER parts) ----------------
  # - name: roof               # a 4-sided pyramid on the body; sides: 4 puts a flat face to +Z
  #   shape: cone
  #   radius: 0.8
  #   height: 0.6
  #   sides: 4
  #   material: roof
  #   on: body.top
  # - name: door               # op: cut subtracts it from the `on` target; it is never drawn
  #   shape: box
  #   size: [0.4, 0.7, 0.2]
  #   op: cut
  #   on: body@[0, -1, 1]      # normalized anchor: bottom center of the front face
  #   align: bottom
  # - name: post               # repeat: post_0 .. post_3, 0.33 m apart along x
  #   shape: cylinder
  #   radius: 0.04
  #   height: 0.5
  #   material: wood
  #   on: base.top_front_left
  #   align: bottom
  #   offset: [0.1, 0, -0.1]
  #   repeat: {{count: 4, step: [0.33, 0, 0]}}
  # - name: lamp_l             # mirror: x adds the twin lamp_r across x = 0
  #   shape: sphere
  #   radius: 0.08
  #   material: glass
  #   on: body.left
  #   align: right
  #   mirror: x
{reference}checks:
  - grounded
  - connected
  - size: {{height: [1.0, 1.4]}}   # TODO: the real size range
{todos}"""

_TODO_HINTS = (  # every match is listed, in this order: a sentence of the brief -> checks that express it
    (r"reference|picture|photo|concept|sketch|drawing|refs/", "matches_reference: {min_iou: 0.85}"),
    (r"\b\d+\s*px\b|icon|thumbnail|far away|from afar|a distance|silhouette", "readable_at: {px: 32, keep: [p]}"),
    (r"\bparts?\b|budget|low[- ]?poly|simple|cheap", "max_parts: 30"),
    (r"\b\d+(\.\d+)?\s*m\b|tall|high|wide|deep|long|span|height|width|depth|length|size", "SIZE"),
    (r"visible|\bsee|seen|spot|notice|fac(e|es|ing)\b|entrance|door", "visible: {part: p, view: front, min: 0.8}"),
    (r"symmetr|mirror|balanced", "symmetric: {axis: x, min: 0.9}"),
    (r"stable|stand|tip over|fall", "stable"),
    (r"contrast|pop|stand out|colou?r", "contrast: {a: p, b: q, min: 1.5}"),
    (r"above|below|on top|under|behind|in front of|left of|right of", "relation: {a: p, is: above, b: q}"),
    (r"unlike|different|distinct|confus", "distinct_from: {assets: [other], max_iou: 0.7}"),
)
_PICTURE = re.compile(r"\b(?:reference|picture|photo|concept|sketch|drawing|image)\b|refs/", re.IGNORECASE)
_REF_PATH = re.compile(r"refs/[\w./-]+\.(?:png|jpe?g)", re.IGNORECASE)


def _reference_stub(brief: str, name: str) -> str:
    """A commented `reference:` entry when the brief mentions a picture (the reference loop needs one)."""
    if not _PICTURE.search(brief):
        return ""
    found = _REF_PATH.search(brief)
    image = found.group(0) if found else f"refs/{name}.png"
    return ("# reference:                   # the picture of the brief: uncomment, then `sightline ref` / `sightline match`\n"
            f"#   image: {image}\n"
            "#   view: front                # the side the picture shows: front/back/left/right/top/iso\n"
            "#   height: 2.0                # TODO: real height in meters of the object in the picture, if known\n")


_DIMENSIONS = (  # brief words -> the size dimension they measure
    ("width", r"span|wide|width|across|broad"),
    ("height", r"tall|high|height"),
    ("depth", r"long|length|deep|depth"),
)


def _size_hint(sentence: str) -> str:
    """size: with the dimensions the sentence names (a wingspan is a width, a fuselage length a depth)."""
    dims = [d for d, pattern in _DIMENSIONS if re.search(pattern, sentence)] or ["height"]
    return "size: {" + ", ".join(f"{d}: [lo, hi]" for d in dims) + "} or part_size"


def _brief_todos(brief: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", brief.strip()) if s.strip()]
    if not sentences:
        return ("  # TODO: turn each wish of the designer into a check, e.g.\n"
                "  #   - visible: {part: door, view: front, min: 0.8, why: \"the entrance must read\"}\n")
    lines = ["  # TODO: one check per wish in the brief (`why:` keeps the designer's words):"]
    for s in sentences:
        low = s.lower()
        hints = [_size_hint(low) if h == "SIZE" else h for pattern, h in _TODO_HINTS if re.search(pattern, low)]
        quoted = s.replace('"', "'")
        lines.append(f'  #   "{quoted}"' + (f"  ->  {' | '.join(hints)}" if hints else ""))
    return "\n".join(lines) + "\n"


def cmd_new(args) -> int:
    from .loader import load_asset
    from .style import load_style

    text = args.asset
    looks_like_path = text.endswith((".yaml", ".yml")) or "/" in text or "\\" in text
    if looks_like_path:
        path = pathlib.Path(text)
        if path.suffix not in (".yaml", ".yml"):
            path = path.with_suffix(".yaml")
        path = path if path.is_absolute() else pathlib.Path.cwd() / path
        root = _root(args, start=path.parent)
    else:
        root = _root(args)
        path = root / "assets" / f"{text}.yaml"
    name = path.stem
    if not NAME_RE.match(name):
        raise SightlineError(f"bad asset name '{name}'",
                             hint="use lowercase letters, digits and _, starting with a letter, e.g. stone_well")
    if path.exists() and not args.force:
        raise SightlineError(f"{_show(path)} already exists",
                             hint="pick another name, or pass --force to overwrite it")
    load_style(args.style, root)  # unknown style -> error with did-you-mean before writing anything
    brief = (args.brief or "TODO: the designer's request, verbatim.").strip()
    brief_lines = textwrap.wrap(brief, width=92) or [brief]
    body = _TEMPLATE.format(name=name, style=args.style, brief="\n".join(f"  {ln}" for ln in brief_lines),
                            todos=_brief_todos(args.brief or ""), reference=_reference_stub(args.brief or "", name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    asset = load_asset(path, root)  # the template must load; fail loudly if it does not
    _wrote([path])
    print(f"NEW  {asset.name}: {len(asset.parts)} parts, style {asset.style.name}")
    ref = name if path.resolve() == (root / "assets" / f"{name}.yaml").resolve() else _show(path)
    if getattr(args, "root", None) is not None:  # the printed commands must find the asset as given
        ref += f" --root {_show(root)}"
    print(f"next: edit the parts, then  sightline render {ref}  and  sightline check {ref}")
    return EXIT_OK


# ----------------------------------------------------------------------------- render / inspect / check


def cmd_render(args) -> int:
    from .render import write_renders

    asset = _load(args.asset, args)
    out = _out_dir(asset, args)
    paths = write_renders(asset, out, _split_list(args.views), args.res, scale_figure=not args.no_human,
                          ground=False if args.no_ground else None)
    print(f"RENDER  {asset.name}: {', '.join(k for k in paths if k not in ('parts', 'sheet'))} + parts + sheet")
    _wrote(paths.values())
    return EXIT_OK


def cmd_inspect(args) -> int:
    from .inspect import format_report, inspect_asset

    asset = _load(args.asset, args)
    report = inspect_asset(asset, res=args.res, compare=_split_list(args.compare) or ())
    if args.json:
        _json(report)
    else:
        sys.stdout.write(format_report(report))
    return EXIT_OK


def _check_one(name, args) -> dict:
    from .checks import run_checks
    from .inspect import inspect_asset

    asset = _load(name, args)
    entry = {"asset": asset.name, "path": _show(asset.path), "error": None, "results": []}
    if asset.checks:
        report = inspect_asset(asset)
        entry["results"] = run_checks(asset, report)
    return entry


def cmd_check(args) -> int:
    from .checks import format_checks
    from .loader import list_assets

    names = args.assets or list_assets(getattr(args, "root", None))
    if not names:
        root = _root(args)
        raise SightlineError(f"no assets in {(root / 'assets').as_posix()}",
                             hint="create one with `sightline new <name>` or pass asset paths")
    entries = []
    for name in names:
        try:
            entries.append(_check_one(name, args))
        except SightlineError as exc:
            entries.append({"asset": str(name), "path": None, "error": exc, "results": []})
    errors = sum(e["error"] is not None for e in entries)
    failed = sum(e["error"] is None and any(not r.passed for r in e["results"]) for e in entries)
    if args.json:
        _json([{**e, "error": e["error"].format() if e["error"] else None,
                "passed": e["error"] is None and all(r.passed for r in e["results"]),
                "results": [_result_dict(r) for r in e["results"]]} for e in entries])
    else:
        for e in entries:
            if e["error"] is not None:
                print(f"ASSET  {e['asset']}  ERROR")
                print(textwrap.indent(e["error"].format(), "  "))
                continue
            print(f"ASSET  {e['asset']}  ({e['path']})")
            sys.stdout.write(textwrap.indent(format_checks(e["results"]), "  "))
        if len(entries) > 1:
            print(f"ASSETS  {len(entries)}: {len(entries) - failed - errors} passed, {failed} failed, {errors} errors")
    if errors:
        for e in entries:
            if e["error"] is not None and not args.json:
                print(e["error"].format(), file=sys.stderr)
        return EXIT_ERROR
    return EXIT_FAIL if failed else EXIT_OK


# ----------------------------------------------------------------------------- diff


def _git_show(path: pathlib.Path, rev: str) -> str:
    """Text of `path` at git revision `rev` (`git -C <dir> show <rev>:./<file>`)."""
    path = pathlib.Path(path).resolve()
    cmd = ["git", "-C", str(path.parent), "show", f"{rev}:./{path.name}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
    except FileNotFoundError:
        raise SightlineError("git is not installed or not on PATH",
                             hint="compare with a saved copy instead: --against path/to/old.yaml") from None
    except subprocess.TimeoutExpired:
        raise SightlineError("git show timed out", hint="compare with a saved copy: --against path/to/old.yaml") from None
    err = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        if "exists on disk, but not in" in err or "does not exist in" in err:
            raise SightlineError(f"{path.name} is not in git {rev} yet (a new file has no old version)",
                                 hint="commit it once to get a baseline, or compare with a saved copy: "
                                      "--against path/to/old.yaml")
        if "not a git repository" in err:
            raise SightlineError(f"{path.parent.as_posix()} is not inside a git repository",
                                 hint="compare with a saved copy: --against path/to/old.yaml")
        if "dubious ownership" in err:
            raise SightlineError("git refuses the repository (dubious ownership)",
                                 hint=f"git config --global --add safe.directory {path.parent.as_posix()}")
        if "unknown revision" in err or "invalid object name" in err or "bad revision" in err:
            raise SightlineError(f"unknown git revision '{rev}'", hint="e.g. HEAD, HEAD~1, a branch or a commit hash")
        first = err.splitlines()[0] if err else f"exit code {proc.returncode}"
        raise SightlineError(f"git show {rev}:{path.name} failed: {first}",
                             hint="compare with a saved copy: --against path/to/old.yaml")
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise SightlineError(f"{path.name} at {rev} is not UTF-8 text") from None


def cmd_diff(args) -> int:
    from .inspect import diff_reports, inspect_asset
    from .loader import load_asset_text

    asset = _load(args.asset, args)
    against = args.against
    old_label = against or "HEAD"
    candidate = None
    if against:
        for cand in (pathlib.Path(against), pathlib.Path(asset.root) / against):
            if cand.is_file():
                candidate = cand
                break
    if candidate is not None:
        old_text = candidate.read_text(encoding="utf-8")
        old_label = _show(candidate)
    else:
        old_text = _git_show(asset.path, against or "HEAD")
    try:
        old = load_asset_text(old_text, asset.path, asset.root)
    except SightlineError as exc:  # the error is in the old copy: say so instead of naming the working file
        from .style import display_path

        label, loc = display_path(asset.path, asset.root), exc.where or ""
        rest = loc[len(label):] if loc.startswith(label) else (f": {loc}" if loc else "")
        source = f"--against {old_label}" if candidate is not None else f"git {old_label}:{asset.path.name}"
        raise SightlineError(exc.message, where=f"in {source}{rest}", hint=exc.hint) from None
    print(f"DIFF  {asset.name}: {old_label} -> working copy")
    sys.stdout.write(diff_reports(inspect_asset(old), inspect_asset(asset)))
    return EXIT_OK


# ----------------------------------------------------------------------------- sheet / export


def cmd_sheet(args) -> int:
    from .loader import list_assets
    from .sheet import sheet_many

    root = _root(args)
    names = args.assets or list_assets(root)
    if not names:
        raise SightlineError(f"no assets in {(root / 'assets').as_posix()}", hint="create one with `sightline new <name>`")
    out = pathlib.Path(args.out) if args.out else root / "out" / "library.png"
    if out.suffix == "" or out.is_dir():
        out = out / "library.png"
    path = sheet_many(names, out, res=args.res, view=args.view, root=getattr(args, "root", None))
    print(f"SHEET  {len(names)} assets at one shared scale, {args.view} view, with the 1.8 m human")
    _wrote([path])
    return EXIT_OK


def cmd_export(args) -> int:
    from .export import (build_meshes, export_glb, export_obj, format_decimation, format_edges, merge_meshes,
                         mesh_summary, resolve_quality, validate_glb)

    asset = _load(args.asset, args)
    fmt = args.format
    if fmt == "obj" and args.bake_colors:
        raise SightlineError("--bake-colors needs --format glb", hint="OBJ has no standard vertex colors; use glb")
    if args.bake_colors and (args.quality not in (None, "full") or args.max_tris is not None):
        raise SightlineError("--bake-colors needs the full mesh: vertex colors need the dense surface-nets vertices",
                             hint="drop --quality/--max-tris (baking exports --quality full) or drop --bake-colors")
    quality, _, _ = resolve_quality(args.quality, args.max_tris, args.bake_colors)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(asset.root) / "out" / asset.name / f"{asset.name}.{fmt}"
    meshes = build_meshes(asset, args.max_cells, args.bake_colors, quality, args.max_tris)
    if fmt == "glb":
        path = export_glb(asset, out, args.max_cells, per_part=not args.merged, bake_colors=args.bake_colors,
                          meshes=meshes)
        written = [path]
    else:
        path = export_obj(asset, out, args.max_cells, meshes=[merge_meshes(meshes, asset.name)] if args.merged else meshes)
        written = [path, path.with_suffix(".mtl")]
    mode = "one merged node" if args.merged else f"{len(meshes)} part nodes"
    budget = f", max_tris {args.max_tris}" if args.max_tris is not None else ""
    print(f"EXPORT  {asset.name}: {fmt}, {mode}, max_cells {args.max_cells}, quality {quality}{budget}"
          + (", baked colors" if args.bake_colors else ""))
    print(mesh_summary(meshes))
    print(format_decimation(meshes))
    print(format_edges(meshes))
    if fmt == "glb":
        problems = validate_glb(path)
        print("glTF structure OK" if not problems else "glTF PROBLEMS:\n" + "\n".join(f"  {p}" for p in problems))
    _wrote(written)
    return EXIT_OK if fmt != "glb" or not problems else EXIT_ERROR


# ----------------------------------------------------------------------------- ref / match


def _image_path(text: str, args) -> pathlib.Path:
    path = pathlib.Path(text)
    if path.is_absolute() or path.is_file():
        return path
    root = _root(args)
    for cand in (root / path, root / "assets" / path):
        if cand.is_file():
            return cand
    return path  # analyze_reference reports the missing file


def cmd_ref(args) -> int:
    from .reference import analyze_reference, format_reference

    root = _root(args)
    image = _image_path(args.image, args)
    report = analyze_reference(image, view=args.view, height=args.height, mask=args.mask, root=root,
                               lighting=None if args.lighting == "auto" else args.lighting)
    if args.json:
        _json(report)
    else:
        sys.stdout.write(format_reference(report))
    return EXIT_OK



IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def _strip_arrays(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def cmd_structure(args) -> int:
    from . import structure as st

    target = args.target
    if pathlib.Path(target).suffix.lower() in IMAGE_SUFFIXES:
        root = _root(args)
        image = _image_path(target, args)
        rep = st.analyze_structure(image, view=args.view, height=args.height, mask=args.mask, root=root)
        out = pathlib.Path(args.out) if args.out else root / "out" / "structure" / f"{pathlib.Path(target).stem}.png"
        path = st.write_structure_png(rep, out if out.suffix else out / f"{pathlib.Path(target).stem}.png")
        if args.json:
            _json(_strip_arrays(rep))
        else:
            print(st.format_structure(rep))
        _wrote([path])
        return EXIT_OK
    asset = _load(target, args)
    rep = st.compare_structure(asset, args.index, res=args.res)
    path = st.write_compare_png(rep, _out_dir(asset, args) / f"structure_{args.index}.png")
    if args.json:
        _json(_strip_arrays(rep))
    else:
        print(st.format_compare(rep))
    _wrote([path])
    return EXIT_OK


def cmd_import(args) -> int:
    from .importer import format_import, import_glb
    from .style import load_style

    root = _root(args)
    src = pathlib.Path(args.file)
    if src.suffix.lower() != ".glb":
        raise SightlineError(f"cannot import {src.name}", hint="give a glTF binary (.glb); convert .gltf/.fbx/.obj in Blender first")
    if not src.is_file():
        raise SightlineError(f"file not found: {src.as_posix()}", hint="give the path of a .glb file")
    style = load_style("default", root)
    result = import_glb(src, args.name, style=style, split=not args.no_split, max_parts=args.max_parts)
    out = pathlib.Path(args.out) if args.out else root / "assets" / f"{result['name']}.yaml"
    if out.exists() and not args.force:
        raise SightlineError(f"{_show(out)} already exists", hint="pass --force to overwrite it, or --name / --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["yaml"], encoding="utf-8", newline="\n")
    print(format_import(result))
    print(f"next: render it and compare it with a picture of the original, e.g.  sightline render {result['name']}")
    _wrote([out])
    return EXIT_OK

def cmd_match(args) -> int:
    from .reference import fit_view, format_fit, format_match, match_asset

    asset = _load(args.asset, args)
    out = _out_dir(asset, args)
    view = None
    fit = None
    if args.fit_view:
        fit = fit_view(asset, args.index)
        view = fit[0]["view"]
    report = match_asset(asset, args.index, res=args.res, out_dir=out, view=view)
    if args.json:
        _json({**report, "fit_view": fit} if fit is not None else report)
    else:
        if fit is not None:
            sys.stdout.write(format_fit(fit))
            print(f"matching from the best view: {view}")
        sys.stdout.write(format_match(report))
        _wrote([report["images"]["match"]])
    return EXIT_OK


# ----------------------------------------------------------------------------- levels


def cmd_level(args) -> int:
    from .levels import format_play, load_level, play, render_level, run_level_checks

    level = load_level(args.level, getattr(args, "root", None))
    report = play(level)
    if args.action == "render":
        out = pathlib.Path(args.out) if args.out else pathlib.Path(level.root) / "out" / "levels" / level.name
        paths = render_level(level, out, report)
        print(f"LEVEL  {level.name}: iso + top renders")
        _wrote(paths.values())
        return EXIT_OK
    results = run_level_checks(level, report)
    failed = sum(not r.passed for r in results)
    if args.json:
        payload = [_result_dict(r) for r in results]
        _json({**report, "checks": payload} if args.action == "play" else payload)
    else:
        if args.action == "play":
            sys.stdout.write(format_play(report).rstrip("\n") + "\n\n")
        lines = [r.line() for r in results]
        lines.append(f"CHECKS  {len(results)}: {len(results) - failed} passed, {failed} failed" if results
                     else "CHECKS  none")
        print("\n".join(lines))
    return EXIT_FAIL if failed else EXIT_OK


# ----------------------------------------------------------------------------- styles / shapes


def cmd_styles(args) -> int:
    from .style import display_path, list_styles, load_style

    root = _root(args)
    names = [args.name] if args.name else list_styles(root)
    if not names:
        names = ["default"]
    for n, name in enumerate(names):
        style = load_style(name, root)
        if n:
            print()
        src = display_path(style.path, root) if style.path else "built-in"
        print(f"STYLE  {style.name}  ({src})")
        d = style.light_dir
        print(f"  light       travels [{d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}], ambient {style.ambient:.2f}, "
              f"shadows {'on' if style.shadows else 'off'}")
        print(f"  outline     {style.outline_width} px {_hex(style.outline_color)}    background {_hex(style.background)}")
        print(f"  ground      {'shown' if style.ground_show else 'hidden'} {_hex(style.ground_color)}, "
              f"grid {style.ground_grid:g} m    render {style.render_res} px: {', '.join(style.render_views)}")
        mats = sorted(style.materials.values(), key=lambda m: m.name)
        print(f"MATERIALS  {len(mats)}")
        width = max(len(m.name) for m in mats)
        patterns = {}
        for m in mats:
            pattern = "solid" if m.pattern == "solid" else m.pattern
            if m.pattern != "solid":
                scale = f"{m.scale[0]:g}" if m.scale[0] == m.scale[1] else f"{m.scale[0]:g} x {m.scale[1]:g}"
                pattern += f" {_hex(m.pattern_color) if m.pattern_color else ''} scale {scale} m"
                if m.pattern in ("stripes", "planks"):
                    pattern += f" along {m.axis}"
            patterns[m.name] = pattern
        pw = max(len(p) for p in patterns.values())
        for m in mats:
            shine = f"shine {m.shine:.2f}" if m.shine else ""
            print(f"  {m.name:<{width}}  {_hex(m.color)}  {patterns[m.name]:<{pw}}  {shine}".rstrip())
        pal = sorted(style.palette.items())
        print(f"PALETTE  {len(pal)}")
        cells = [f"{k} {_hex(v)}" for k, v in pal]
        cw = max((len(c) for c in cells), default=0) + 3
        per = max(1, 100 // max(cw, 1))
        for i in range(0, len(cells), per):
            print("  " + "".join(c.ljust(cw) for c in cells[i:i + per]).rstrip())
    return EXIT_OK


_SHAPE_BBOX = {
    "box": "w, h, d", "sphere": "2r, 2r, 2r", "ellipsoid": "w, h, d", "cylinder": "2r, h, 2r",
    "cone": "2 max(r, top_radius), h, 2 max(r, top_radius)", "capsule": "2r, h, 2r",
    "torus": "2(R+t), 2t, 2(R+t)", "wedge": "w, h, d",
}

_SHAPES_TEXT = """\
ANCHORS  points of a part's local bounding box, moved with the part
  center, top (+Y), bottom (-Y), right (+X), left (-X), front (+Z), back (-Z)
  combinations with at most one word per axis, joined by _ in any order: top_front_left
  @[u, v, w]: normalized, each component -1..1 maps the bbox low..high (@[0, -1, 1] = bottom of the front face)

PLACEMENT
  at: [x, y, z]          world point for the part's `align` anchor (default align: center)
  on: part.anchor        attach to an EARLIER part; own align defaults to the opposite anchor (top -> bottom)
  on: part@[u, v, w]     same with a normalized anchor
  between: [A, B]        strut/wire: local +Y from A to B (anchors or [x, y, z]); length = distance,
                         so no height (cylinder/cone/capsule) and size: [width, depth] (box/ellipsoid/wedge)
  align: anchor          which own anchor goes to `at` / `on`
  offset: [dx, dy, dz]   added after attaching (in the target's frame when following)
  follow: true|false     with on: inherit the target's rotation (default true)
  rotate: [rx, ry, rz]   degrees, R = Rz Ry Rx, about the part's own bbox center

EXPANSION  (repeat first, then mirror)
  repeat: {count: n, step: [dx, dy, dz]}                 copies name_0 .. name_{n-1}
  repeat: {count: n, axis: y, center: [x, y, z] | part.anchor}   radial, i * 360/n degrees
  mirror: x|y|z          adds a twin across the world plane through the origin
                         (name_l <-> name_r, _left <-> _right, else name_mirror; mirror_name: overrides)

OPERATIONS
  op: union (default)    blend: k smooth-joins with the on target (or blend_with: part), k in meters
  op: cut                subtracted, never drawn; from: [parts] | all (default: the on target); no material

PART KEYS  name shape size radius height top_radius sides round thickness material at on between align
           offset follow rotate repeat mirror mirror_name op from blend blend_with tags

VIEWS  front back left right top iso (azimuth 45, elevation 30), orbit:<azimuth>:<elevation>
WORLD  meters; +Y up, +X right (east), +Z front (toward the front-view camera, south); ground y = 0
"""


def cmd_shapes(args) -> int:
    from .model import SHAPE_USAGE, SHAPES

    print("SHAPES  local frame centered on the local bbox; Y is the shape's main axis")
    width = max(len(s) for s in SHAPES)
    for shape in SHAPES:
        print(f"  {shape:<{width}}  {SHAPE_USAGE[shape]}")
        print(f"  {'':<{width}}  local bbox (w, h, d): {_SHAPE_BBOX.get(shape, '?')}")
    print("  sides: n makes a regular n-gon prism / pyramid with a flat face toward +Z; radius is the circumradius")
    print("  wedge: full height at the back (-Z), sloping to 0 at the front edge (+Z)")
    print()
    sys.stdout.write(_SHAPES_TEXT)
    return EXIT_OK


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS,
                        help="project root with styles/, assets/, levels/ (default: $SIGHTLINE_ROOT or the "
                             "nearest ancestor with styles/)")
    p = argparse.ArgumentParser(
        prog="sightline",
        description="Text-first 3D assets and levels: render, measure and test YAML models.",
        epilog="<asset> is a name (assets/<name>.yaml under the root) or a path. "
               "Exit codes: 0 ok, 1 a check failed, 2 an error.")
    p.add_argument("--root", help="project root (also accepted after the command)")
    p.add_argument("--version", action="version", version=f"sightline {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    def add(name, func, help_text):
        sp = sub.add_parser(name, parents=[common], help=help_text, description=help_text)
        sp.set_defaults(func=func)
        return sp

    sp = add("new", cmd_new, "write a new asset template (assets/<name>.yaml)")
    sp.add_argument("asset", help="asset name or path of the new .yaml")
    sp.add_argument("--style", default="default", help="style name (default: default)")
    sp.add_argument("--brief", help="the designer's request, verbatim; its sentences become TODO checks")
    sp.add_argument("--force", action="store_true", help="overwrite an existing file")

    sp = add("render", cmd_render, "render views, the part map and a contact sheet to PNG")
    sp.add_argument("asset")
    sp.add_argument("--views", help="comma-separated views, e.g. front,iso or orbit:30:15 (default: the style's)")
    sp.add_argument("--res", type=int, help="pixels on the long side of the largest view (default: the style's)")
    sp.add_argument("--out", help="output folder (default: <root>/out/<asset>)")
    sp.add_argument("--no-human", action="store_true", help="leave out the 1.8 m scale figure")
    sp.add_argument("--no-ground", action="store_true",
                    help="leave out the ground slab (views from below, orbit elevation < 0, never draw it)")

    sp = add("inspect", cmd_inspect, "print the measured report (sizes, contacts, visibility, silhouettes ...)")
    sp.add_argument("asset")
    sp.add_argument("--json", action="store_true", help="print the report as JSON")
    sp.add_argument("--res", type=int, default=128, help="render size of the measuring views (default 128)")
    sp.add_argument("--compare", help="comma-separated other assets to compare silhouettes with")

    sp = add("check", cmd_check, "run the assets' checks (no names: every asset under <root>/assets)")
    sp.add_argument("assets", nargs="*")
    sp.add_argument("--json", action="store_true")

    sp = add("diff", cmd_diff, "what changed in the measured report since git HEAD (or another version)")
    sp.add_argument("asset")
    sp.add_argument("--against", help="git revision (default HEAD) or the path of an older copy of the file")

    sp = add("sheet", cmd_sheet, "library sheet: many assets side by side at one shared scale")
    sp.add_argument("assets", nargs="*")
    sp.add_argument("--out", help="PNG path (default: <root>/out/library.png)")
    sp.add_argument("--res", type=int, help="pixel height of the tallest tile (default 320)")
    sp.add_argument("--view", default="iso", help="view of every tile (default iso)")

    sp = add("export", cmd_export, "export a mesh: glTF 2.0 binary (.glb, one node per part) or OBJ + MTL")
    sp.add_argument("asset")
    sp.add_argument("--format", choices=("glb", "obj"), default="glb")
    sp.add_argument("--merged", action="store_true", help="one node instead of one per part")
    sp.add_argument("--bake-colors", action="store_true", help="bake material patterns into COLOR_0 (glb)")
    sp.add_argument("--max-cells", type=int, default=128, help="mesh resolution: cells on the longest side (8-512)")
    sp.add_argument("--quality", choices=("full", "high", "medium", "low"),
                    help="decimation: how far from the surface a part may move, 1/8 (high), 1/4 (medium, the "
                         "default) or 1/2 (low) of its cell; full keeps every triangle (the default with --bake-colors)")
    sp.add_argument("--max-tris", type=int, help="triangle budget for the whole asset (decimates further if needed)")
    sp.add_argument("--out", help="file or folder (default: <root>/out/<asset>/<asset>.<format>)")

    sp = add("ref", cmd_ref, "measure a reference picture alone (mask, proportions, colors, questions)")
    sp.add_argument("image")
    sp.add_argument("--view", default="front", help="view the picture shows (default front)")
    sp.add_argument("--height", type=float, help="real height in meters of the object in the picture")
    sp.add_argument("--mask", help="black/white mask PNG (default: automatic)")
    sp.add_argument("--lighting", choices=("auto", "lit", "flat"), default="auto",
                    help="lit = shaded render or photo, flat = flat paint (default: guessed from the shading)")
    sp.add_argument("--json", action="store_true")

    sp = add("match", cmd_match, "compare the model with its reference picture; writes match_<i>.png")
    sp.add_argument("asset")
    sp.add_argument("--index", type=int, default=0, help="which reference: entry (default 0)")
    sp.add_argument("--fit-view", action="store_true", help="first guess the picture's camera angle")
    sp.add_argument("--res", type=int, default=256, help="render size (default 256)")
    sp.add_argument("--out", help="output folder (default: <root>/out/<asset>)")
    sp.add_argument("--json", action="store_true")

    sp = add("structure", cmd_structure, "divisions inside a picture (edges, regions, relief guess); with an asset: "
                                         "how its parts and materials line up with them")
    sp.add_argument("target", help="an asset (compares with its reference picture) or an image file (analyses it alone)")
    sp.add_argument("--index", type=int, default=0, help="which reference: entry (default 0)")
    sp.add_argument("--res", type=int, default=512, help="render size for the comparison (default 512)")
    sp.add_argument("--view", default="front", help="image only: the view it shows (default front)")
    sp.add_argument("--height", type=float, help="image only: real height in meters (sizes in m instead of %%)")
    sp.add_argument("--mask", help="image only: black/white mask PNG (default: automatic)")
    sp.add_argument("--out", help="output folder or PNG path")
    sp.add_argument("--json", action="store_true")

    sp = add("import", cmd_import, "import a ready model (.glb) as a descriptive asset: one fitted primitive per mesh piece")
    sp.add_argument("file", help="path of a .glb file")
    sp.add_argument("--name", help="asset name (default: the file name)")
    sp.add_argument("--out", help="YAML path (default: <root>/assets/<name>.yaml)")
    sp.add_argument("--max-parts", type=int, default=120, help="keep at most this many pieces, largest first (default 120)")
    sp.add_argument("--no-split", action="store_true", help="one part per mesh primitive (do not split it into connected pieces)")
    sp.add_argument("--force", action="store_true", help="overwrite an existing file")

    sp = add("level", cmd_level, "levels: render (iso + top PNG), play (bot report + checks), check")
    sp.add_argument("action", choices=("render", "play", "check"))
    sp.add_argument("level", help="level name (levels/<name>.yaml under the root) or path")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--out", help="render folder (default: <root>/out/levels/<level>)")

    sp = add("styles", cmd_styles, "list styles with their materials, colors and palettes")
    sp.add_argument("name", nargs="?", help="only this style")

    add("shapes", cmd_shapes, "shape, anchor and placement reference")
    return p


def main(argv=None) -> int:
    _utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return EXIT_ERROR
    try:
        return int(args.func(args) or 0)
    except SightlineError as exc:
        sys.stdout.flush()
        print(exc.format(), file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # a bug, not the user's input: show it, but keep the exit-code contract
        sys.stdout.flush()
        traceback.print_exc()
        print(f"sightline: internal error ({type(exc).__name__}: {exc}); this is a bug in sightline, not in "
              f"your files", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
