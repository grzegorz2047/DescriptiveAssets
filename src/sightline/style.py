"""Style files: palette, materials, light, outline, ground and render defaults (SPEC section 3).

Also home of the shared YAML reader (YAML 1.2 booleans, so the key `on:` stays a string, plus
line numbers for error messages) and of the project-root resolution.
"""

from __future__ import annotations

import functools
import os
import pathlib
import re

import numpy as np
import yaml

from .camera import VIEW_NAMES, parse_orbit
from .errors import SightlineError, did_you_mean
from .model import Material, Style, _is_number

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root next to src/
PATTERNS = ("solid", "noise", "stripes", "planks", "bricks", "checker")
STYLE_KEYS = ("style", "palette", "materials", "light", "outline", "background", "ground", "render")
MATERIAL_KEYS = ("color", "pattern", "pattern_color", "scale", "axis", "shine")
SECTION_KEYS = {
    "light": ("direction", "ambient", "shadows"),
    "outline": ("width", "color"),
    "ground": ("show", "color", "grid"),
    "render": ("res", "views"),
}
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# Last-resort values when default.yaml itself lacks a section.
_BUILTIN = {
    "palette": {},
    "materials": {"helper": {"color": "#9a9da3"}},
    "light": {"direction": [0.45, -1.0, -0.6], "ambient": 0.35, "shadows": True},
    "outline": {"width": 1, "color": "#1b1b1b"},
    "background": "#dde1e6",
    "ground": {"show": True, "color": "#c8ccc2", "grid": 1.0},
    "render": {"res": 256, "views": ["front", "right", "top", "iso"]},
}


# ----------------------------------------------------------------------------- YAML


class _YamlLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false (so `on:`, `yes`, `off` stay strings)."""


_YamlLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def read_yaml(text: str, label: str) -> tuple[object, dict[tuple, int]]:
    """Parse YAML text -> (data, lines). `lines[path]` = 1-based line of the value at a key path
    such as ("parts", 3, "on"). Duplicate mapping keys are errors."""
    try:
        node = yaml.compose(text, Loader=_YamlLoader)
        data = yaml.load(text, Loader=_YamlLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"{label}, line {mark.line + 1}" if mark is not None else label
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        raise SightlineError(
            f"YAML syntax error: {problem}",
            where=where,
            hint="check indentation and that every 'key: value' has a space after the colon; quote strings with ': ' or '#'",
        ) from None
    lines: dict[tuple, int] = {}

    def walk(n, path):
        lines[path] = n.start_mark.line + 1
        if isinstance(n, yaml.MappingNode):
            seen: dict[str, int] = {}
            for key_node, value_node in n.value:
                key = key_node.value if isinstance(key_node, yaml.ScalarNode) else str(key_node.value)
                if key in seen:
                    raise SightlineError(
                        f"duplicate key '{key}' (first at line {seen[key]})",
                        where=f"{label}, line {key_node.start_mark.line + 1}",
                        hint="each key may appear once per mapping; merge or rename them",
                    )
                seen[key] = key_node.start_mark.line + 1
                walk(value_node, path + (key,))
                lines[path + (key,)] = key_node.start_mark.line + 1
        elif isinstance(n, yaml.SequenceNode):
            for i, item in enumerate(n.value):
                walk(item, path + (i,))

    if node is not None:
        walk(node, ())
    return data, lines


# ----------------------------------------------------------------------------- root


def _has_styles(path: pathlib.Path) -> bool:
    return (path / "styles").is_dir()


def resolve_root(root=None, start=None) -> pathlib.Path:
    """Project root (directory with styles/, assets/, levels/).

    Order: explicit `root`; nearest ancestor of `start` (e.g. the asset file) with styles/;
    env SIGHTLINE_ROOT; nearest ancestor of the CWD with styles/; the repo next to the package.
    """
    if root is not None:
        return pathlib.Path(root).resolve()
    if start is not None:
        here = pathlib.Path(start).resolve()
        for cand in ([here] if here.is_dir() else []) + list(here.parents):
            if _has_styles(cand):
                return cand
    env = os.environ.get("SIGHTLINE_ROOT")
    if env:
        return pathlib.Path(env).resolve()
    cwd = pathlib.Path.cwd().resolve()
    for cand in [cwd, *cwd.parents]:
        if _has_styles(cand):
            return cand
    return PACKAGE_ROOT


def display_path(path: pathlib.Path, root: pathlib.Path | None) -> str:
    """Path relative to root when inside it (posix), else as given."""
    path = pathlib.Path(path)
    if root is not None:
        try:
            return path.resolve().relative_to(pathlib.Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


# ----------------------------------------------------------------------------- parsing helpers


def parse_hex(value, where: str | None = None) -> tuple[float, float, float]:
    """'#rrggbb' -> (r, g, b) in 0..1."""
    if not isinstance(value, str) or not _HEX.match(value):
        raise SightlineError(f"bad color {value!r}", where=where,
                             hint="use '#rrggbb' in quotes (YAML treats an unquoted # as a comment) or a palette name")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


class _StyleParser:
    def __init__(self, label: str, lines: dict):
        self.label = label
        self.lines = lines

    def where(self, *path) -> str:
        line = None
        for k in range(len(path), -1, -1):
            line = self.lines.get(tuple(path[:k]))
            if line:
                break
        dotted = ".".join(str(p) for p in path)
        text = f"{self.label}: {dotted}" if dotted else self.label
        return f"{text}, line {line}" if line else text

    def color(self, value, palette: dict, *path) -> tuple[float, float, float]:
        if isinstance(value, str) and not value.startswith("#"):
            if value in palette:
                return palette[value]
            hint = did_you_mean(value, palette) or ""
            raise SightlineError(f"unknown palette color '{value}'", where=self.where(*path),
                                 hint=(hint + " " if hint else "") + "use '#rrggbb' or a key of `palette:`")
        return parse_hex(value, self.where(*path))

    def number(self, value, lo, hi, *path, integer=False):
        ok = _is_number(value) and (not integer or float(value).is_integer())
        if not ok or not (lo <= value <= hi):
            kind = "an integer" if integer else "a number"
            raise SightlineError(f"{path[-1]} must be {kind} in [{lo:g}, {hi:g}], got {value!r}", where=self.where(*path))
        return int(value) if integer else float(value)

    def boolean(self, value, *path) -> bool:
        if not isinstance(value, bool):
            raise SightlineError(f"{path[-1]} must be true or false, got {value!r}", where=self.where(*path))
        return value

    def mapping(self, value, *path) -> dict:
        if not isinstance(value, dict):
            raise SightlineError(f"{path[-1] if path else 'style'} must be a mapping, got {type(value).__name__}",
                                 where=self.where(*path))
        return value

    def keys(self, data: dict, allowed, *path):
        for key in data:
            if key not in allowed:
                hint = did_you_mean(key, allowed) or ""
                raise SightlineError(f"unknown key '{key}'", where=self.where(*path, key),
                                     hint=(hint + " " if hint else "") + f"valid keys: {', '.join(allowed)}")

    def material(self, name: str, spec, palette: dict) -> Material:
        path = ("materials", name)
        spec = self.mapping(spec, *path)
        self.keys(spec, MATERIAL_KEYS, *path)
        if "color" not in spec:
            raise SightlineError(f"material '{name}' needs a color", where=self.where(*path))
        color = self.color(spec["color"], palette, *path, "color")
        pattern = spec.get("pattern", "solid")
        if pattern not in PATTERNS:
            hint = did_you_mean(str(pattern), PATTERNS) or ""
            raise SightlineError(f"unknown pattern {pattern!r}", where=self.where(*path, "pattern"),
                                 hint=(hint + " " if hint else "") + f"patterns: {', '.join(PATTERNS)}")
        if "pattern_color" in spec:
            pattern_color = self.color(spec["pattern_color"], palette, *path, "pattern_color")
        elif pattern != "solid":
            pattern_color = tuple(c * 0.8 for c in color)  # a darker shade of the base color
        else:
            pattern_color = None
        scale = spec.get("scale", 0.25)
        if _is_number(scale):
            scale = [scale, scale]
        if not (isinstance(scale, list) and len(scale) == 2 and all(_is_number(v) and v > 0 for v in scale)):
            raise SightlineError(f"scale must be a number > 0 or [a, b], got {spec.get('scale')!r}",
                                 where=self.where(*path, "scale"), hint="meters; bricks: [brick width, brick height]")
        axis = spec.get("axis", "y")
        if axis not in ("x", "y", "z"):
            raise SightlineError(f"axis must be x, y or z, got {axis!r}", where=self.where(*path, "axis"))
        shine = self.number(spec.get("shine", 0.0), 0.0, 1.0, *path, "shine")
        return Material(name, color, pattern, pattern_color, (float(scale[0]), float(scale[1])), axis, shine)


# ----------------------------------------------------------------------------- loading


def _style_path(name_or_path, root: pathlib.Path) -> pathlib.Path:
    text = str(name_or_path)
    if text.endswith((".yaml", ".yml")) or "/" in text or "\\" in text:
        path = pathlib.Path(text)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise SightlineError(f"style file not found: {path.as_posix()}")
        return path
    path = root / "styles" / f"{text}.yaml"
    if path.is_file():
        return path
    if text == "default":
        return PACKAGE_ROOT / "styles" / "default.yaml"
    available = sorted(p.stem for p in (root / "styles").glob("*.yaml")) if (root / "styles").is_dir() else []
    hint = did_you_mean(text, available) or ""
    raise SightlineError(f"unknown style '{text}'",
                         hint=(hint + " " if hint else "") + f"styles in {(root / 'styles').as_posix()}: "
                         + (", ".join(available) or "none"))


@functools.lru_cache(maxsize=32)
def _read_style_file(path_str: str, mtime_ns: int, label: str) -> tuple[object, dict]:
    text = pathlib.Path(path_str).read_text(encoding="utf-8")
    return read_yaml(text, label)


def _load_raw(path: pathlib.Path, label: str) -> tuple[dict, dict]:
    data, lines = _read_style_file(str(path), path.stat().st_mtime_ns, label)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SightlineError("a style file must be a mapping", where=label)
    return data, lines


def load_style(name_or_path="default", root=None) -> Style:
    """Load `styles/<name>.yaml` (or a path); missing sections and keys fall back to default.yaml.

    Palette and materials are merged by key over the default library, so a style that sets only a
    few materials still offers every default material.
    """
    root = resolve_root(root)
    path = _style_path(name_or_path, root)
    label = display_path(path, root)
    data, lines = _load_raw(path, label)
    parser = _StyleParser(label, lines)
    parser.keys(data, STYLE_KEYS)
    if "style" in data and data["style"] != path.stem:
        raise SightlineError(f"style: '{data['style']}' does not match the file name '{path.stem}'",
                             where=parser.where("style"), hint=f"set style: {path.stem} or rename the file")

    default_path = root / "styles" / "default.yaml"
    if not default_path.is_file():
        default_path = PACKAGE_ROOT / "styles" / "default.yaml"
    layers = [(_BUILTIN, _StyleParser("built-in defaults", {}))]
    if default_path.is_file() and default_path.resolve() != path.resolve():
        d_label = display_path(default_path, root)
        d_data, d_lines = _load_raw(default_path, d_label)
        layers.append((d_data, _StyleParser(d_label, d_lines)))
    layers.append((data, parser))

    # Merge layer by layer; validation errors point into the file that set the value.
    palette: dict[str, tuple] = {}
    materials: dict[str, Material] = {}
    merged: dict[str, object] = {}
    owner: dict[str, _StyleParser] = {}
    for layer, prs in layers:
        for key, value in prs.mapping(layer.get("palette") or {}, "palette").items():
            palette[str(key)] = parse_hex(value, prs.where("palette", key))
    for layer, prs in layers:
        for name, spec in prs.mapping(layer.get("materials") or {}, "materials").items():
            materials[str(name)] = prs.material(str(name), spec, palette)
        for section, allowed in SECTION_KEYS.items():
            if section in layer:
                values = prs.mapping(layer[section], section)
                prs.keys(values, allowed, section)
                for key, value in values.items():
                    merged[f"{section}.{key}"] = value
                    owner[f"{section}.{key}"] = prs
        if "background" in layer:
            merged["background"] = layer["background"]
            owner["background"] = prs

    def get(key):
        return merged[key], owner[key], key.split(".")

    value, prs, path_keys = get("light.direction")
    if not (isinstance(value, list) and len(value) == 3 and all(_is_number(v) for v in value)) or not any(value):
        raise SightlineError(f"light.direction must be a non-zero [x, y, z], got {value!r}", where=prs.where(*path_keys),
                             hint="the direction the light travels, e.g. [0.45, -1.0, -0.6] (from top-left-front)")
    light = np.array(value, dtype=np.float64)
    light /= np.linalg.norm(light)
    ambient = get("light.ambient")[1].number(merged["light.ambient"], 0.0, 1.0, "light", "ambient")
    shadows = owner["light.shadows"].boolean(merged["light.shadows"], "light", "shadows")
    outline_width = owner["outline.width"].number(merged["outline.width"], 0, 16, "outline", "width", integer=True)
    outline_color = owner["outline.color"].color(merged["outline.color"], palette, "outline", "color")
    background = owner["background"].color(merged["background"], palette, "background")
    ground_show = owner["ground.show"].boolean(merged["ground.show"], "ground", "show")
    ground_color = owner["ground.color"].color(merged["ground.color"], palette, "ground", "color")
    ground_grid = owner["ground.grid"].number(merged["ground.grid"], 0.0, 1e6, "ground", "grid")
    res = owner["render.res"].number(merged["render.res"], 16, 4096, "render", "res", integer=True)
    views, prs, path_keys = get("render.views")
    if not isinstance(views, list) or not views:
        raise SightlineError(f"render.views must be a list of views, got {views!r}", where=prs.where(*path_keys))
    for view in views:
        if view not in VIEW_NAMES:
            try:
                if not isinstance(view, str) or parse_orbit(view) is None:
                    raise SightlineError(f"unknown view {view!r}")
            except SightlineError:
                hint = did_you_mean(str(view), VIEW_NAMES) or ""
                raise SightlineError(f"unknown view {view!r}", where=prs.where(*path_keys),
                                     hint=(hint + " " if hint else "") + f"views: {', '.join(VIEW_NAMES)}") from None

    return Style(
        name=path.stem,
        palette=palette,
        materials=materials,
        light_dir=light,
        ambient=ambient,
        shadows=shadows,
        outline_width=outline_width,
        outline_color=outline_color,
        background=background,
        ground_show=ground_show,
        ground_color=ground_color,
        ground_grid=ground_grid,
        render_res=res,
        render_views=[str(v) for v in views],
        path=path,
    )


def list_styles(root=None) -> list[str]:
    """Names of the styles under root/styles."""
    root = resolve_root(root)
    folder = root / "styles"
    return sorted(p.stem for p in folder.glob("*.yaml")) if folder.is_dir() else []
