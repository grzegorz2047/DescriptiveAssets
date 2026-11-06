"""Level data model: a grid of tile kinds and integer heights plus markers, doors and objects.

Grid frame: column `c` grows east (+X), row `r` grows south (+Z, toward the front view).
Cell (c, r) has its center at x = (c + 0.5)·cell, z = (r + 0.5)·cell, y = heights[r][c]·height_unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import SightlineError, did_you_mean

TILE_KINDS = ("floor", "wall", "water", "spikes", "pit", "ladder", "door", "key")
DEFAULT_LEGEND = {
    ".": "floor",
    "#": "wall",
    "~": "water",
    "^": "spikes",
    "_": "pit",
    "=": "ladder",
    "D": "door",
    "k": "key",
}
STANDABLE_KINDS = frozenset({"floor", "ladder", "door", "key"})
DEADLY_KINDS = frozenset({"water", "spikes"})
GAP_KINDS = frozenset({"water", "spikes", "pit"})  # not standable, may be jumped over
# (name, dcol, drow): N = toward row 0 (-Z), E = +X, S = +Z, W = -X
DIRECTIONS = (("N", 0, -1), ("E", 1, 0), ("S", 0, 1), ("W", -1, 0))
MAX_KEYS = 8


@dataclass
class Player:
    """Movement model of the navigation bot (meters, cells for jump_gap)."""

    step: float = 0.5  # largest rise walked without jumping
    jump_up: float = 1.0  # largest rise reached by a jump
    jump_gap: int = 2  # widest gap (cells) a jump clears
    max_drop: float = 3.0  # largest survivable drop
    height: float = 1.8

    @property
    def eye(self) -> float:
        """Eye height above the standing surface."""
        return self.height - 0.1


@dataclass
class Door:
    name: str
    col: int
    row: int
    key: str | None = None  # marker name of the key, None = opens freely


@dataclass
class LevelObject:
    index: int
    asset: str
    col: int
    row: int
    rotate: float = 0.0  # degrees about +Y
    solid: bool = True
    cells: list[tuple[int, int]] = field(default_factory=list)  # cells it blocks for walking (col, row)
    cover: list[tuple[int, int]] = field(default_factory=list)  # cells with any of its geometry above (footprint)
    columns: dict = field(default_factory=dict)  # (col, row) -> [(y_lo, y_hi)] world-y spans of its geometry
    height: float = 1.0  # meters above `base`
    base: float = 0.0  # terrain y under the anchor cell
    asset_obj: Any = None  # sightline Asset when it could be loaded
    placeholder: bool = True  # True when the asset is unknown: 1 cell, PLACEHOLDER_HEIGHT tall
    source: str = ""  # "objects[0] (watchtower)" for messages
    bounds: tuple[np.ndarray, np.ndarray] | None = None  # asset-space AABB of the solids (before rotate)

    @property
    def top(self) -> float:
        return self.base + self.height

    def yaw_matrix(self) -> np.ndarray:
        """(3, 3) rotation by `rotate` degrees about +Y, the same matrix as an asset's `rotate: [0, deg, 0]`."""
        th = np.radians(self.rotate)
        c, s = float(np.cos(th)), float(np.sin(th))
        m = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        m[np.abs(m) < 1e-12] = 0.0
        return m

    def world_corners(self, cell: float) -> np.ndarray | None:
        """(8, 3) world corners of the rotated asset AABB placed on its anchor cell; None for placeholders."""
        if self.bounds is None:
            return None
        lo, hi = self.bounds
        local = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        anchor = np.array([(self.col + 0.5) * cell, self.base, (self.row + 0.5) * cell])
        return local @ self.yaw_matrix().T + anchor


@dataclass
class Level:
    name: str
    path: Path
    root: Path
    where: str  # file label used in error messages
    brief: str
    style: str
    cell: float
    height_unit: float
    player: Player
    heights: np.ndarray  # (rows, cols) int64 steps
    tiles: np.ndarray  # (rows, cols) <U1 tile chars as written
    kinds: np.ndarray  # (rows, cols) kind names
    legend: dict[str, str]
    markers: dict[str, tuple[int, int]]  # name -> (col, row), file order
    doors: dict[str, Door]
    objects: list[LevelObject]
    checks: list  # raw check entries (validated by levels.checks)
    raw: dict
    warnings: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return int(self.heights.shape[0])

    @property
    def cols(self) -> int:
        return int(self.heights.shape[1])

    def heights_m(self) -> np.ndarray:
        """(rows, cols) terrain top in meters."""
        return self.heights.astype(np.float64) * self.height_unit

    def y(self, col: int, row: int) -> float:
        return float(self.heights[row, col]) * self.height_unit

    def world(self, col: float, row: float, y: float | None = None) -> np.ndarray:
        """World point of a cell center (fractional cells allowed) at terrain height unless `y` is given."""
        if y is None:
            y = self.y(int(col), int(row))
        return np.array([(col + 0.5) * self.cell, y, (row + 0.5) * self.cell])

    def kind(self, col: int, row: int) -> str:
        return str(self.kinds[row, col])

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def solid_object_mask(self) -> np.ndarray:
        mask = np.zeros(self.heights.shape, dtype=bool)
        for obj in self.objects:
            if obj.solid:
                for c, r in obj.cells:
                    mask[r, c] = True
        return mask

    def object_top(self) -> np.ndarray:
        """(rows, cols) top of solid objects in meters, -inf where there is none."""
        top = np.full(self.heights.shape, -np.inf)
        for obj in self.objects:
            if obj.solid:
                for c, r in obj.cells:
                    top[r, c] = max(top[r, c], obj.top)
        return top

    def object_spans(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Occupied world-y spans of solid objects per cell: (lo, hi, owner) arrays of shape (K, rows, cols),
        padded with lo = +inf, hi = -inf, owner = -1 (owner = index into `objects`)."""
        k = max([len(spans) for obj in self.objects if obj.solid for spans in obj.columns.values()] or [0])
        lo = np.full((max(k, 1) * max(len(self.objects), 1), *self.heights.shape), np.inf)
        hi = np.full(lo.shape, -np.inf)
        owner = np.full(lo.shape, -1, dtype=np.int64)
        fill = np.zeros(self.heights.shape, dtype=np.int64)
        for n, obj in enumerate(self.objects):
            if not obj.solid:
                continue
            for (c, r), spans in obj.columns.items():
                for y0, y1 in spans:
                    lo[fill[r, c], r, c], hi[fill[r, c], r, c], owner[fill[r, c], r, c] = y0, y1, n
                    fill[r, c] += 1
        used = max(int(fill.max()), 1)
        return lo[:used], hi[:used], owner[:used]

    def object_at(self, col: int, row: int) -> LevelObject | None:
        for obj in self.objects:
            if obj.solid and (col, row) in obj.cells:
                return obj
        return None

    def key_names(self) -> list[str]:
        """Markers that act as keys: referenced by a door or standing on a `key` tile, in marker order."""
        used = {d.key for d in self.doors.values() if d.key}
        return [n for n, (c, r) in self.markers.items() if n in used or self.kinds[r, c] == "key"]

    def door_at(self, col: int, row: int) -> Door | None:
        for door in self.doors.values():
            if (door.col, door.row) == (col, row):
                return door
        return None

    def place_names(self) -> list[str]:
        return list(self.markers) + list(self.doors)

    def place(self, name: str, where: str | None = None) -> tuple[int, int]:
        """Cell of a marker or door name; SightlineError with did_you_mean otherwise."""
        if name in self.markers:
            return self.markers[name]
        if name in self.doors:
            d = self.doors[name]
            return (d.col, d.row)
        names = self.place_names()
        raise SightlineError(
            f"unknown marker or door '{name}'",
            where=where or self.where,
            hint=did_you_mean(name, names) or f"known: {', '.join(names) or '(none)'}",
        )
