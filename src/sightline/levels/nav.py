"""Navigation bot: moves on the grid, Dijkstra over (cell, keys) states, softlocks.

A cell index is `row * cols + col`; a state index is `cell + keyset * n_cells`, where bit i of
`keyset` means the i-th key of `Level.key_names()` is held. Geometric moves (walk, ladder, hop,
jump) are computed once per cell with numpy; keys and doors are applied per key subset.

Movement rules (player in meters, jump_gap in cells):
- walk to a standable 4-neighbor with rise <= step and drop <= max_drop;
- ladder: moving onto or off a ladder cell ignores the height difference;
- hop: jump up onto an adjacent ledge with step < rise <= jump_up (counts as a jump);
- jump: straight over 1..jump_gap non-standable cells (water, spikes, pit, or walls/solid objects whose
  top is <= takeoff + jump_up) to the first standable cell, rise <= jump_up, drop <= max_drop.
Path cost = horizontal meters + vertical meters of every move.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..errors import SightlineError
from .model import DIRECTIONS, STANDABLE_KINDS, Level

EPS = 1e-9
WALK, LADDER, HOP, JUMP = 0, 1, 2, 3
MOVE_NAMES = ("walk", "ladder", "hop", "jump")


@dataclass
class Moves:
    """Key-independent moves between cells (parallel arrays)."""

    src: np.ndarray
    dst: np.ndarray
    kind: np.ndarray
    direction: np.ndarray
    gap: np.ndarray  # cells jumped over
    dy: np.ndarray  # meters, landing - takeoff
    cost: np.ndarray  # meters (horizontal + vertical)
    over: list[str]  # what a jump clears, "" otherwise


@dataclass
class NavGraph:
    level: Level
    moves: Moves
    n_cells: int
    keys: list[str]
    cell_keys: np.ndarray  # (n_cells,) key bits picked up on entering
    door_need: np.ndarray  # (n_cells,) key bits needed to stand on a door
    standable: np.ndarray  # (n_cells,) standable ignoring locks
    ladder: np.ndarray
    heights: np.ndarray  # (n_cells,) meters
    obstacle_top: np.ndarray  # (n_cells,) wall / solid object top, -inf elsewhere
    n_states: int
    e_src: np.ndarray
    e_dst: np.ndarray
    e_move: np.ndarray
    out_ptr: np.ndarray
    out_edges: np.ndarray
    in_ptr: np.ndarray
    in_edges: np.ndarray

    def cell(self, col: int, row: int) -> int:
        return row * self.level.cols + col

    def col_row(self, cell: int) -> tuple[int, int]:
        r, c = divmod(int(cell), self.level.cols)
        return c, r

    def valid_states(self) -> np.ndarray:
        """(n_states,) bool: the bot can be in this state (standable, door open, key on its cell held)."""
        ks = np.arange(self.n_states) // self.n_cells
        cell = np.arange(self.n_states) % self.n_cells
        return self.standable[cell] & ((self.door_need[cell] & ~ks) == 0) & ((self.cell_keys[cell] & ~ks) == 0)

    def key_label(self, keyset: int) -> str:
        names = [k for i, k in enumerate(self.keys) if keyset >> i & 1]
        return ", ".join(names) if names else "none"


def _over_text(level: Level, cells: list[int]) -> str:
    names = []
    for cell in cells:
        r, c = divmod(cell, level.cols)
        obj = level.object_at(c, r)
        kind = level.kind(c, r)
        name = obj.asset if obj is not None else kind
        if name not in names:
            names.append(name)
    return "+".join(names)


def jump_obstacles(level: Level) -> np.ndarray:
    """(n_cells,) top a jump must clear: wall height, +inf for solid objects (never jumped over), -inf elsewhere."""
    kinds = level.kinds.ravel()
    top = np.where(kinds == "wall", level.heights_m().ravel(), -np.inf)
    return np.where(level.solid_object_mask().ravel(), np.inf, top)


def build_moves(level: Level) -> Moves:
    rows, cols, cell = level.rows, level.cols, level.cell
    p = level.player
    n = rows * cols
    h = level.heights_m().ravel()
    kinds = level.kinds.ravel()
    solid = level.solid_object_mask().ravel()
    standable = np.isin(kinds, sorted(STANDABLE_KINDS)) & ~solid
    ladder = (kinds == "ladder") & ~solid
    obstacle_top = jump_obstacles(level)
    rr, cc = np.divmod(np.arange(n), cols)
    out = {k: [] for k in ("src", "dst", "kind", "direction", "gap", "dy")}
    over: list[str] = []

    def add(src, dst, kind, d, gap):
        out["src"].append(src)
        out["dst"].append(dst)
        out["kind"].append(np.broadcast_to(kind, src.shape).astype(np.int64))
        out["direction"].append(np.full(src.shape, d, dtype=np.int64))
        out["gap"].append(np.full(src.shape, gap, dtype=np.int64))
        out["dy"].append(h[dst] - h[src])

    for d, (_, dc, dr) in enumerate(DIRECTIONS):
        # Adjacent moves: walk, ladder, hop up.
        nr, nc = rr + dr, cc + dc
        inb = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
        nb = np.clip(nr, 0, rows - 1) * cols + np.clip(nc, 0, cols - 1)
        src = np.nonzero(standable & inb & standable[nb])[0]
        dst = nb[src]
        dy = h[dst] - h[src]
        walk = (dy <= p.step + EPS) & (-dy <= p.max_drop + EPS)
        lad = ~walk & (ladder[src] | ladder[dst])
        hop = ~walk & ~lad & (dy > p.step) & (dy <= p.jump_up + EPS)
        ok = walk | lad | hop
        kind = np.where(walk, WALK, np.where(lad, LADDER, HOP))
        add(src[ok], dst[ok], kind[ok], d, 0)
        over.extend([""] * int(ok.sum()))
        # Jumps over k non-standable cells to the first standable cell.
        alive = standable.copy()
        for k in range(1, p.jump_gap + 1):
            mr, mc = rr + k * dr, cc + k * dc
            inb_m = (mr >= 0) & (mr < rows) & (mc >= 0) & (mc < cols)
            mid = np.clip(mr, 0, rows - 1) * cols + np.clip(mc, 0, cols - 1)
            alive &= inb_m & ~standable[mid] & (obstacle_top[mid] <= h + p.jump_up + EPS)
            lr, lc = rr + (k + 1) * dr, cc + (k + 1) * dc
            inb_l = (lr >= 0) & (lr < rows) & (lc >= 0) & (lc < cols)
            land = np.clip(lr, 0, rows - 1) * cols + np.clip(lc, 0, cols - 1)
            src = np.nonzero(alive & inb_l & standable[land])[0]
            dst = land[src]
            dy = h[dst] - h[src]
            ok = (dy <= p.jump_up + EPS) & (-dy <= p.max_drop + EPS)
            src, dst = src[ok], dst[ok]
            add(src, dst, JUMP, d, k)
            step = dr * cols + dc
            over.extend(_over_text(level, [int(s) + i * step for i in range(1, k + 1)]) for s in src)

    arrays = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for k, v in out.items()}
    dy = arrays["dy"].astype(np.float64)
    horizontal = (arrays["gap"] + 1) * cell
    return Moves(src=arrays["src"].astype(np.int64), dst=arrays["dst"].astype(np.int64), kind=arrays["kind"],
                 direction=arrays["direction"], gap=arrays["gap"], dy=dy, cost=horizontal + np.abs(dy), over=over)


def build_graph(level: Level) -> NavGraph:
    moves = build_moves(level)
    n_cells = level.rows * level.cols
    keys = level.key_names()
    bit = {k: 1 << i for i, k in enumerate(keys)}
    cell_keys = np.zeros(n_cells, dtype=np.int64)
    for k in keys:
        c, r = level.markers[k]
        cell_keys[r * level.cols + c] |= bit[k]
    door_need = np.zeros(n_cells, dtype=np.int64)
    for door in level.doors.values():
        if door.key:
            door_need[door.row * level.cols + door.col] = bit[door.key]
    kinds = level.kinds.ravel()
    solid = level.solid_object_mask().ravel()
    standable = np.isin(kinds, sorted(STANDABLE_KINDS)) & ~solid
    h = level.heights_m().ravel()
    obstacle_top = jump_obstacles(level)

    n_layers = 1 << len(keys)
    srcs, dsts, mids = [], [], []
    move_ids = np.arange(len(moves.src))
    for ks in range(n_layers):
        ok = ((door_need[moves.dst] & ~ks) == 0) & ((door_need[moves.src] & ~ks) == 0) & ((cell_keys[moves.src] & ~ks) == 0)
        new_ks = ks | cell_keys[moves.dst[ok]]
        srcs.append(moves.src[ok] + ks * n_cells)
        dsts.append(moves.dst[ok] + new_ks * n_cells)
        mids.append(move_ids[ok])
    e_src = np.concatenate(srcs).astype(np.int64)
    e_dst = np.concatenate(dsts).astype(np.int64)
    e_move = np.concatenate(mids).astype(np.int64)
    n_states = n_cells * n_layers
    out_edges = np.argsort(e_src, kind="stable")
    out_ptr = np.searchsorted(e_src[out_edges], np.arange(n_states + 1))
    in_edges = np.argsort(e_dst, kind="stable")
    in_ptr = np.searchsorted(e_dst[in_edges], np.arange(n_states + 1))
    return NavGraph(level=level, moves=moves, n_cells=n_cells, keys=keys, cell_keys=cell_keys, door_need=door_need,
                    standable=standable, ladder=(kinds == "ladder") & ~solid, heights=h, obstacle_top=obstacle_top,
                    n_states=n_states, e_src=e_src, e_dst=e_dst, e_move=e_move, out_ptr=out_ptr, out_edges=out_edges,
                    in_ptr=in_ptr, in_edges=in_edges)


def dijkstra(graph: NavGraph, source: int) -> tuple[np.ndarray, np.ndarray]:
    """Shortest distances (meters) from one state and the predecessor edge of each state (-1 = none)."""
    n = graph.n_states
    dist = [float("inf")] * n
    pred = [-1] * n
    ptr = graph.out_ptr.tolist()
    order = graph.out_edges.tolist()
    dst = graph.e_dst.tolist()
    cost = graph.moves.cost[graph.e_move].tolist()
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, s = heapq.heappop(heap)
        if d > dist[s]:
            continue
        for j in range(ptr[s], ptr[s + 1]):
            e = order[j]
            t = dst[e]
            nd = d + cost[e]
            if nd < dist[t]:
                dist[t] = nd
                pred[t] = e
                heapq.heappush(heap, (nd, t))
    return np.array(dist), np.array(pred, dtype=np.int64)


def reach(graph: NavGraph, sources, blocked_cells: np.ndarray | None = None, reverse: bool = False) -> np.ndarray:
    """(n_states,) bool: states reachable from `sources` (or that reach them when `reverse`), never entering blocked cells."""
    ptr = (graph.in_ptr if reverse else graph.out_ptr).tolist()
    order = (graph.in_edges if reverse else graph.out_edges).tolist()
    nxt = (graph.e_src if reverse else graph.e_dst).tolist()
    blocked = None if blocked_cells is None else blocked_cells.tolist()
    n_cells = graph.n_cells
    seen = np.zeros(graph.n_states, dtype=bool)
    queue = deque()
    for s in sources:
        s = int(s)
        if not seen[s] and (blocked is None or not blocked[s % n_cells]):
            seen[s] = True
            queue.append(s)
    seen_list = seen.tolist()
    while queue:
        s = queue.popleft()
        for j in range(ptr[s], ptr[s + 1]):
            t = nxt[order[j]]
            if not seen_list[t] and (blocked is None or not blocked[t % n_cells]):
                seen_list[t] = True
                queue.append(t)
    return np.array(seen_list, dtype=bool)


def states_at(graph: NavGraph, cell: int) -> np.ndarray:
    """All state indices standing on `cell` (one per key subset), valid or not."""
    return cell + np.arange(graph.n_states // graph.n_cells) * graph.n_cells


def label_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected components of a 2D bool mask: labels (-1 outside) numbered in row-major order of first cell."""
    rows, cols = mask.shape
    big = rows * cols
    lab = np.where(mask, np.arange(big).reshape(rows, cols), big)
    while True:
        prev = lab
        nb = lab.copy()
        nb[1:, :] = np.minimum(nb[1:, :], lab[:-1, :])
        nb[:-1, :] = np.minimum(nb[:-1, :], lab[1:, :])
        nb[:, 1:] = np.minimum(nb[:, 1:], lab[:, :-1])
        nb[:, :-1] = np.minimum(nb[:, :-1], lab[:, 1:])
        lab = np.where(mask, nb, big)
        if np.array_equal(lab, prev):
            break
    roots = np.unique(lab[mask])  # sorted = row-major order of each component's first cell
    out = np.full(mask.shape, -1, dtype=np.int64)
    out[mask] = np.searchsorted(roots, lab[mask])
    return out, len(roots)


@dataclass
class Analysis:
    level: Level
    graph: NavGraph
    spawn_state: int
    dist: np.ndarray  # (n_states,) meters from spawn, inf = unreachable
    pred: np.ndarray  # predecessor edge per state
    reached: np.ndarray  # (n_states,) bool
    goal_cell: int | None
    finish: np.ndarray | None  # (n_states,) bool: the goal is reachable from this state
    goal_state: int | None  # cheapest reached goal state

    def reached_cells(self) -> np.ndarray:
        """(rows, cols) bool: cell reached with any key subset."""
        g = self.graph
        return self.reached.reshape(-1, g.n_cells).any(axis=0).reshape(self.level.rows, self.level.cols)

    def cell_distance(self, cell: int) -> float:
        return float(self.dist[states_at(self.graph, cell)].min())


def analyze(level: Level) -> Analysis:
    graph = build_graph(level)
    sc, sr = level.markers["spawn"]
    spawn_cell = graph.cell(sc, sr)
    spawn_state = spawn_cell + int(graph.cell_keys[spawn_cell]) * graph.n_cells
    if graph.door_need[spawn_cell] & ~graph.cell_keys[spawn_cell]:
        raise SightlineError("spawn stands on a locked door", where=f"{level.where}: markers.spawn",
                             hint="move spawn off the door or remove the door's key")
    dist, pred = dijkstra(graph, spawn_state)
    reached = np.isfinite(dist)
    goal_cell = finish = goal_state = None
    if "goal" in level.markers:
        gc, gr = level.markers["goal"]
        goal_cell = graph.cell(gc, gr)
        goal_states = states_at(graph, goal_cell)
        goal_states = goal_states[graph.valid_states()[goal_states]]
        finish = reach(graph, goal_states, reverse=True)
        reached_goals = goal_states[reached[goal_states]]
        if len(reached_goals):
            goal_state = int(reached_goals[np.argmin(dist[reached_goals])])
    return Analysis(level=level, graph=graph, spawn_state=spawn_state, dist=dist, pred=pred, reached=reached,
                    goal_cell=goal_cell, finish=finish, goal_state=goal_state)


# ---------------------------------------------------------------- path

def path_edges(analysis: Analysis, target_state: int) -> list[int]:
    edges = []
    s = target_state
    while s != analysis.spawn_state:
        e = int(analysis.pred[s])
        if e < 0:
            raise RuntimeError("broken predecessor chain")
        edges.append(e)
        s = int(analysis.graph.e_src[e])
    return edges[::-1]


def _fmt_cell(cell_cr) -> str:
    return f"({cell_cr[0]},{cell_cr[1]})"


def describe_move(graph: NavGraph, m: int) -> str:
    """One move as a readable step, e.g. `jump E over water 2 cells (1,2)→(4,2)`."""
    mv = graph.moves
    d = DIRECTIONS[int(mv.direction[m])][0]
    a, b = graph.col_row(mv.src[m]), graph.col_row(mv.dst[m])
    dy = float(mv.dy[m])
    route = f"{_fmt_cell(a)}→{_fmt_cell(b)}"
    kind = int(mv.kind[m])
    if kind == LADDER:
        return f"climb {dy:+.2f} m (ladder) {d} {route}" if dy >= 0 else f"climb down {dy:+.2f} m (ladder) {d} {route}"
    if kind == HOP:
        return f"jump up {dy:+.2f} m {d} {route}"
    if kind == JUMP:
        tail = f", {dy:+.2f} m" if abs(dy) > EPS else ""
        return f"jump {d} over {mv.over[m]} {int(mv.gap[m])} cell{'s' if mv.gap[m] > 1 else ''} {route}{tail}"
    if dy < -graph.level.player.step - EPS:
        return f"drop {dy:+.2f} m {d} {route}"
    return f"walk {d} 1 cell {route}" + (f", {dy:+.2f} m" if abs(dy) > EPS else "")


def path_report(analysis: Analysis) -> dict:
    """The critical path spawn -> goal: cells, compressed segments, jumps, rises and falls."""
    level, g = analysis.level, analysis.graph
    if analysis.goal_cell is None:
        return {"found": False, "reason": "no goal marker", "segments": [], "cells": [], "moves": []}
    if analysis.goal_state is None:
        return {"found": False, "reason": "goal unreachable from spawn", "segments": [], "cells": [], "moves": []}
    edges = path_edges(analysis, analysis.goal_state)
    mv = g.moves
    sc = level.markers["spawn"]
    cells = [list(sc)]
    segments: list[str] = []
    moves_out = []
    keys_picked: list[str] = []
    doors_opened: list[str] = []
    held = int(g.cell_keys[g.cell(*sc)])
    for k in g.keys:
        if held >> g.keys.index(k) & 1:
            keys_picked.append(k)
            segments.append(f"pick {k} at spawn {_fmt_cell(sc)}")
    run = None  # (direction, start cell, end cell, cells, dy) of plain walking
    step = level.player.step

    def flush():
        nonlocal run
        if run is not None:
            d, a, b, n, dy = run
            tail = f", {dy:+.2f} m" if abs(dy) > EPS else ""
            segments.append(f"walk {d} {n} cell{'s' if n > 1 else ''} {_fmt_cell(a)}→{_fmt_cell(b)}{tail}")
            run = None

    horizontal = vertical = 0.0
    jumps = 0
    max_rise = max_fall = climbed = 0.0
    for e in edges:
        m = int(g.e_move[e])
        a, b = g.col_row(mv.src[m]), g.col_row(mv.dst[m])
        kind = int(mv.kind[m])
        dy = float(mv.dy[m])
        d = DIRECTIONS[int(mv.direction[m])][0]
        horizontal += (int(mv.gap[m]) + 1) * level.cell
        vertical += abs(dy)
        if kind in (HOP, JUMP):
            jumps += 1
        if kind == LADDER:
            climbed += abs(dy)
        else:
            max_rise = max(max_rise, dy)
            max_fall = max(max_fall, -dy)
        cells.append(list(b))
        moves_out.append({"kind": MOVE_NAMES[kind], "dir": d, "from": list(a), "to": list(b), "dy": round(dy, 4),
                          "gap": int(mv.gap[m]), "over": mv.over[m]})
        plain = kind == WALK and dy >= -step - EPS
        if plain and run is not None and run[0] == d and run[2] == a:
            run = (d, run[1], b, run[3] + 1, run[4] + dy)
        else:
            flush()
            if plain:
                run = (d, a, b, 1, dy)
            else:
                segments.append(describe_move(g, m))
        dst_cell = g.cell(*b)
        door = level.door_at(*b)
        if door is not None:
            flush()
            label = door.name + (f" with {door.key}" if door.key else "")
            if not doors_opened or doors_opened[-1] != door.name:
                segments.append(f"open {label} {_fmt_cell(b)}")
                doors_opened.append(door.name)
        new = int(g.cell_keys[dst_cell]) & ~held
        if new:
            flush()
            for i, k in enumerate(g.keys):
                if new >> i & 1:
                    segments.append(f"pick {k} {_fmt_cell(b)}")
                    keys_picked.append(k)
            held |= new
    flush()
    gc = level.markers["goal"]
    segments.append(f"reach goal {_fmt_cell(gc)}")
    return {
        "found": True,
        "length": round(horizontal + vertical, 4),
        "horizontal": round(horizontal, 4),
        "vertical": round(vertical, 4),
        "jumps": jumps,
        "max_rise": round(max_rise, 4),
        "max_fall": round(max_fall, 4),
        "ladder_climb": round(climbed, 4),
        "keys": keys_picked,
        "doors": doors_opened,
        "cells": cells,
        "segments": segments,
        "moves": moves_out,
    }


# ---------------------------------------------------------------- explanations

def _barrier(graph: NavGraph, a: int, ks: int, d: int, is_target) -> tuple[float, int, str] | None:
    """Why the bot at cell `a` holding `ks` cannot move in direction `d` to a target cell; None if no target there.

    Returns (excess over the player's limit in meters, cells jumped, text) so the closest miss sorts first.
    """
    level, p = graph.level, graph.level.player
    _, dc, dr = DIRECTIONS[d]
    ac, ar = graph.col_row(a)
    h = graph.heights
    path = []
    for k in range(1, p.jump_gap + 4):
        c, r = ac + k * dc, ar + k * dr
        if not level.in_bounds(c, r):
            return None
        path.append(r * level.cols + c)
        if graph.standable[path[-1]]:
            break
    else:
        return None
    b = path[-1]
    need = int(graph.door_need[b]) & ~ks
    if not is_target(b, ks | int(graph.cell_keys[b]) | need):
        return None
    dy = float(h[b] - h[a])
    to = f"to {_fmt_cell(graph.col_row(b))}"
    door = level.door_at(*graph.col_row(b))
    gap = len(path) - 1
    lock = (0.0, gap, f"through {door.name} {_fmt_cell(graph.col_row(b))} needs {door.key}") if need and door else None
    if gap == 0:
        if graph.ladder[a] or graph.ladder[b]:
            return lock
        if dy > max(p.step, p.jump_up) + EPS:
            return dy - p.jump_up, 0, f"{to} needs rise {dy:.2f} m > jump_up {p.jump_up:.2f} m"
        if -dy > p.max_drop + EPS:
            return -dy - p.max_drop, 0, f"{to} drops {-dy:.2f} m > max_drop {p.max_drop:.2f} m"
        return lock
    over = _over_text(level, path[:-1])
    if gap > p.jump_gap:
        return (gap - p.jump_gap) * level.cell, gap, f"{to} crosses {gap} cells of {over} > jump_gap {p.jump_gap}"
    for m in path[:-1]:
        top = float(graph.obstacle_top[m])
        if top > h[a] + p.jump_up + EPS:
            what = _over_text(level, [m])
            if not np.isfinite(top):
                return level.cell, gap, f"{to} is blocked by solid object {what} {_fmt_cell(graph.col_row(m))}"
            return top - h[a] - p.jump_up, gap, (f"{to} is blocked by {what} {_fmt_cell(graph.col_row(m))}, top {top:.2f} m "
                                                 f"> takeoff + jump_up {h[a] + p.jump_up:.2f} m")
    if dy > p.jump_up + EPS:
        return dy - p.jump_up, gap, f"{to} needs a jump rise {dy:.2f} m > jump_up {p.jump_up:.2f} m"
    if -dy > p.max_drop + EPS:
        return -dy - p.max_drop, gap, f"{to} drops {-dy:.2f} m > max_drop {p.max_drop:.2f} m"
    return lock


def explain(graph: NavGraph, states, is_target) -> str | None:
    """The closest-to-working failed move from any of `states` toward a target cell.

    Smallest excess over the player's limits wins, then the shorter move, then the text (deterministic).
    """
    best = None
    for s in sorted(int(x) for x in states):
        a, ks = s % graph.n_cells, s // graph.n_cells
        for d in range(4):
            found = _barrier(graph, a, ks, d, is_target)
            if found is not None and (best is None or found < best):
                best = found
    return best[2] if best else None


# ---------------------------------------------------------------- reachability and softlocks

def _near(mask: np.ndarray, k: int) -> np.ndarray:
    """Cells within k straight steps (N/E/S/W) of the mask: the only places a move into it can start."""
    out = mask.copy()
    for i in range(1, k + 1):
        out[i:, :] |= mask[:-i, :]
        out[:-i, :] |= mask[i:, :]
        out[:, i:] |= mask[:, :-i]
        out[:, :-i] |= mask[:, i:]
    return out


def reachability_report(analysis: Analysis) -> dict:
    level, g = analysis.level, analysis.graph
    reached_cells = analysis.reached_cells()
    standable = g.standable.reshape(level.rows, level.cols)
    places = {}
    for name in level.place_names():
        c, r = level.place(name)
        cell = g.cell(c, r)
        dist = analysis.cell_distance(cell)
        places[name] = {"cell": [c, r], "reachable": bool(np.isfinite(dist)),
                        "distance": round(dist, 4) if np.isfinite(dist) else None,
                        "type": "door" if name in level.doors else "marker"}
    unreached = standable & ~reached_cells
    labels, count = label_regions(unreached)
    regions = []
    for i in range(count):
        rr, cc = np.nonzero(labels == i)
        cells = sorted(zip(cc.tolist(), rr.tolist()), key=lambda t: (t[1], t[0]))
        region_mask = (labels == i).ravel()
        near = _near(labels == i, level.player.jump_gap + 3).ravel()
        sources = np.nonzero(analysis.reached & np.tile(near, g.n_states // g.n_cells))[0]
        names = [n for n, info in places.items() if tuple(info["cell"]) in set(cells)]
        reason = explain(g, sources, lambda b, ks, m=region_mask: bool(m[b]))
        regions.append({"cells": [list(c) for c in cells], "count": len(cells), "places": names,
                        "reason": f"way in {reason}" if reason else "no move leads in (walled off or out of jump range)"})
        for n in names:
            places[n]["reason"] = regions[-1]["reason"]
    for name, info in places.items():
        if not info["reachable"] and "reason" not in info:
            info["reason"] = "locked: its key is unreachable" if name in level.doors else "unreachable"
    total = int(standable.sum())
    n_reached = int((reached_cells & standable).sum())
    return {
        "reachable_cells": n_reached,
        "standable_cells": total,
        "share": int(round(100.0 * n_reached / total)) if total else 0,
        "places": places,
        "unreachable": [n for n, info in places.items() if not info["reachable"]],
        "regions": regions,
    }


def softlock_report(analysis: Analysis) -> dict:
    """Reachable states from which the goal can no longer be reached, grouped into regions per key subset."""
    level, g = analysis.level, analysis.graph
    if analysis.goal_cell is None:
        return {"checked": False, "reason": "no goal marker", "regions": [], "cells": 0, "states": 0}
    stuck = analysis.reached & ~analysis.finish
    goal_reachable = analysis.goal_state is not None
    layers = stuck.reshape(-1, g.n_cells)
    merged: dict[tuple, dict] = {}
    for ks in range(layers.shape[0]):
        if not layers[ks].any():
            continue
        labels, count = label_regions(layers[ks].reshape(level.rows, level.cols))
        for i in range(count):
            rr, cc = np.nonzero(labels == i)
            cells = tuple(sorted(zip(cc.tolist(), rr.tolist()), key=lambda t: (t[1], t[0])))
            entry = merged.setdefault(cells, {"keysets": [], "states": []})
            entry["keysets"].append(ks)
            entry["states"].extend((np.array([g.cell(c, r) for c, r in cells]) + ks * g.n_cells).tolist())
    finish = analysis.finish
    good_reached = analysis.reached & finish
    regions = []
    for cells, entry in merged.items():
        states = np.array(sorted(entry["states"]), dtype=np.int64)
        in_region = np.zeros(g.n_states, dtype=bool)
        in_region[states] = True
        has_spawn = bool(in_region[analysis.spawn_state])
        reason = explain(g, states, lambda b, ks: bool(finish[b + ks * g.n_cells]))
        entry_edges = np.nonzero(in_region[g.e_dst] & good_reached[g.e_src])[0]
        entered = describe_move(g, int(g.e_move[entry_edges[0]])) if len(entry_edges) else None
        if has_spawn and not goal_reachable:
            why = "goal unreachable from spawn" + (f"; closest exit {reason}" if reason else "")
        elif reason:
            why = f"exit {reason}"
        else:
            why = "no exit toward the goal (walled off or out of jump range)"
        regions.append({
            "cells": [list(c) for c in cells],
            "count": len(cells),
            "keys_held": [g.key_label(ks) for ks in entry["keysets"]],
            "entered_by": entered,
            "contains_spawn": has_spawn,
            "reason": why,
        })
    regions.sort(key=lambda reg: (not reg["contains_spawn"], reg["cells"][0][1], reg["cells"][0][0]))
    stuck_cells = layers.any(axis=0)
    return {
        "checked": True,
        "goal_reachable": goal_reachable,
        "states": int(stuck.sum()),
        "cells": int(stuck_cells.sum()),
        "regions": regions,
    }
