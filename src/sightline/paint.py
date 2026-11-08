"""`sightline paint`: the colour pass over every reference picture in one go (SPEC 12.8).

For each reference in turn (sides, back, front by default: the views that see the fewest plates first),
`structure --trace fix --paint --conform` is run against the asset as painted so far and its `op: paint`
parts are appended, so a later picture only paints what the earlier ones left wrong. With `strip`, the
`op: paint` entries whose names carry the prefix are removed first, so the colours are read again from
scratch on the current shapes (after `fit` or `carve` moved or cut them). The classes of every view are
measured before and after.
"""

from __future__ import annotations

import pathlib
import time

import yaml

from . import structure as st_mod
from .errors import SightlineError
from .loader import MAX_PAINTS, asset_from_data, load_asset_text
from .style import _YamlLoader, display_path, read_yaml, resolve_root

ORDER = ("right", "left", "back", "front")  # default order of the views: the sparsest first


def strip_paints(text: str, prefix: str | None) -> tuple[str, int]:
    """The YAML text without the `op: paint` entries whose name starts with `prefix` (every paint when `prefix`
    is None) and the `# --- paints` comment lines above them; returns (text, removed)."""
    root = yaml.compose(text, Loader=_YamlLoader)
    seq = next((v for k, v in root.value if k.value == "parts"), None) if isinstance(root, yaml.MappingNode) else None
    if seq is None or not isinstance(seq, yaml.SequenceNode):
        return text, 0
    spans = []
    items = seq.value
    for n, item in enumerate(items):
        if not isinstance(item, yaml.MappingNode):
            continue
        fields = {k.value: v.value for k, v in item.value if isinstance(v, yaml.ScalarNode)}
        if fields.get("op") == "paint" and (prefix is None or str(fields.get("name", "")).startswith(prefix)):
            start = text.rfind("\n", 0, item.start_mark.index) + 1  # the item's own line (with its "- ")
            if n + 1 < len(items):  # up to the next item's line: the end mark of a block mapping sits inside it
                end = text.rfind("\n", 0, items[n + 1].start_mark.index) + 1
            else:
                end = item.end_mark.index
                nl = text.find("\n", max(end - 1, start))
                end = nl + 1 if nl >= 0 else len(text)
            spans.append((start, end))
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + text[b:]
    lines = text.split("\n")
    kept = [ln for ln in lines if not ln.lstrip().startswith("# --- paints where")]
    return "\n".join(kept), len(spans)


def _order(asset, order) -> list[int]:
    if order:
        idx = []
        for o in order:
            if isinstance(o, int) or str(o).isdigit():
                k = int(o)
                if not 0 <= k < len(asset.references):
                    raise SightlineError(f"no reference {k} to paint from", hint=f"the asset has references 0..{len(asset.references) - 1}")
                idx.append(k)
            else:
                k = next((n for n, r in enumerate(asset.references) if r.view == o), None)
                if k is None:
                    raise SightlineError(f"no reference shows the {o} view", hint="name views the asset has, or their indices")
                idx.append(k)
        return idx
    ranked = sorted(range(len(asset.references)),
                    key=lambda n: (ORDER.index(asset.references[n].view) if asset.references[n].view in ORDER else 9, n))
    return ranked


def paint_asset(path, root=None, *, res: int = 512, order=None, strip: bool = False, strip_all: bool = False,
                prefix: str = "paint", log=None) -> dict:
    """Paint the asset file at `path` from its references; returns the report with the new YAML text."""
    t0 = time.perf_counter()
    log = log or (lambda msg: None)
    path = pathlib.Path(path)
    root = resolve_root(root, start=path.parent)
    label = display_path(path, root)
    text = path.read_text(encoding="utf-8")
    asset = load_asset_text(text, path, root)
    if not asset.references:
        raise SightlineError(f"asset '{asset.name}' has no reference picture to paint from", where=label,
                             hint="add reference: {image: refs/<name>.png, view: front, height: 2.0}")
    known = {"front", "back", "left", "right"}
    if not any(r.view in known for r in asset.references):
        raise SightlineError("paint needs a front, back, left or right reference", where=label)
    if isinstance(res, bool) or not isinstance(res, int) or not 64 <= res <= 2048:
        raise SightlineError("res must be an integer from 64 to 2048", hint="e.g. --res 512")
    order_idx = [k for k in _order(asset, order) if asset.references[k].view in known]
    removed = 0
    if strip or strip_all:
        text, removed = strip_paints(text, None if strip_all else prefix)
        asset = load_asset_text(text, path, root)
        log(f"stripped {removed} paint(s)" + ("" if strip_all else f" named {prefix}*"))
    before = [st_mod.compare_structure(asset, i, res=res) for i in range(len(asset.references))]
    log("start  " + "  ".join(f"{r['view']} {r['class_agreement_areas']:.3f}" for r in before))
    steps = []
    for k in order_idx:
        res_ = st_mod.trace_regions(asset, ["fix"], k, res=res, paint=True, conform=True, prefix=prefix)
        n = len(res_["parts"])
        view = asset.references[k].view
        if n:
            block = (f"  # --- paints where the model's class differs from the {view} picture "
                     f"(structure --trace fix --paint --conform)\n" + res_["yaml"])
            if "\nparts:\n- " in text:  # a parts list without indentation
                block = "\n".join(ln[2:] if ln.startswith("  ") else ln for ln in block.split("\n"))
            head, sep, tail = text.partition("\nchecks:")
            text = head.rstrip("\n") + "\n" + block + sep + tail
            asset = load_asset_text(text, path, root)
        steps.append({"index": k, "view": view, "paints": n, "skipped": res_["skipped"]})
        log(f"  {view}: {n} paint(s)" + (f", skipped {len(res_['skipped'])}" if res_["skipped"] else ""))
    after = [st_mod.compare_structure(asset, i, res=res) for i in range(len(asset.references))]
    total = sum(1 for p in asset.parts if p.is_paint)
    rows = [{"index": i, "view": b["view"], "classes_before": b["class_agreement_areas"],
             "classes_after": a["class_agreement_areas"], "strict_before": b["class_agreement"],
             "strict_after": a["class_agreement"],
             "divisions_before": [b["region_borders_on_part_boundaries"], b["part_boundaries_on_borders"]],
             "divisions_after": [a["region_borders_on_part_boundaries"], a["part_boundaries_on_borders"]]}
            for i, (b, a) in enumerate(zip(before, after))]
    return {"asset": asset.name, "path": label, "res": res, "order": order_idx, "stripped": removed,
            "steps": steps, "views": rows, "paints": total, "max_paints": MAX_PAINTS,
            "seconds": round(time.perf_counter() - t0, 1), "text": text}


def format_paint(rep: dict) -> str:
    lines = [f"PAINT  {rep['asset']}  {rep['path']}  res {rep['res']}  ({rep['seconds']:.0f} s)"]
    if rep["stripped"]:
        lines.append(f"STRIP  {rep['stripped']} earlier paint(s) removed first")
    lines.append("STEPS  " + ", ".join(f"{s['view']} +{s['paints']}" for s in rep["steps"])
                 + f"  -> {rep['paints']} paints in the asset (limit {rep['max_paints']})")
    lines.append("VIEWS  classes (areas) before -> after, strict in brackets; divisions before -> after")
    for v in rep["views"]:
        db, da = v["divisions_before"], v["divisions_after"]
        lines.append(f"  [{v['index']}] {v['view']:<6} {v['classes_before']:.0%} -> {v['classes_after']:.0%} "
                     f"({v['strict_before']:.0%} -> {v['strict_after']:.0%})   divisions {db[0]:.0%}/{db[1]:.0%} -> "
                     f"{da[0]:.0%}/{da[1]:.0%}")
    if rep.get("written"):
        lines.append(f"WROTE  {rep['written']}")
    else:
        lines.append("WROTE  nothing (dry run): --write changes the asset file, --out PATH writes a copy")
    return "\n".join(lines) + "\n"
