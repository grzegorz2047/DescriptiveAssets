"""Per-part voxel occupancy (SPEC 6.2), the basis for volumes, contacts and floating-part tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import SightlineError
from .model import F32, Scene


@dataclass(eq=False)
class VoxelGrid:
    origin: np.ndarray  # world position of cell (0, 0, 0)'s center
    cell: float  # cell edge in meters
    shape: tuple[int, int, int]  # (nx, ny, nz)
    solids: list[int]  # scene part indices, order of the first axis below
    strict: np.ndarray  # (len(solids), nx, ny, nz) bool: eval_part < 0 at the cell center
    loose: np.ndarray  # same, eval_part < 0.5 * cell (contacts; tolerance = 1 cell)

    def centers(self) -> np.ndarray:
        """(nx, ny, nz, 3) world cell centers."""
        axes = [self.origin[k] + self.cell * np.arange(self.shape[k]) for k in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def union(self, loose: bool = False) -> np.ndarray:
        """(nx, ny, nz) occupancy of all solids together."""
        grid = self.loose if loose else self.strict
        return grid.any(axis=0) if len(self.solids) else np.zeros(self.shape, dtype=bool)

    def layer(self, part_index: int) -> int:
        """Position of a scene part index along the first axis of strict/loose."""
        return self.solids.index(part_index)

    def volume(self, part_index: int | None = None) -> float:
        """Strict volume (m^3) of one part, or of the union when None."""
        cells = self.union() if part_index is None else self.strict[self.layer(part_index)]
        return float(cells.sum()) * self.cell**3


def occupancy(scene: Scene, max_cells: int = 96, include_helpers: bool = False) -> VoxelGrid:
    """Sample every solid part (with its cuts) at the cell centers of a grid around the scene.

    The grid covers the solids' AABB union grown by one cell, is centered on it (so mirroring across
    the AABB center maps cells onto cells), and has `cell = longest side / max_cells`. Each part is
    evaluated only on the cells within its own AABB +- one cell.
    """
    solids = scene.solid_indices(include_helpers)
    if not solids:
        raise SightlineError("nothing to voxelize: the scene has no solid parts")
    if max_cells < 1:
        raise SightlineError(f"max_cells must be >= 1, got {max_cells}")
    lo = np.min([scene.aabb(i)[0] for i in solids], axis=0)
    hi = np.max([scene.aabb(i)[1] for i in solids], axis=0)
    extent = hi - lo
    cell = float(extent.max()) / max_cells
    if cell <= 0:
        raise SightlineError("nothing to voxelize: the solids have zero size")
    shape = tuple(int(math.ceil(e / cell - 1e-6)) + 2 for e in extent)
    origin = (lo + hi) / 2.0 - (np.array(shape) - 1) / 2.0 * cell
    strict = np.zeros((len(solids), *shape), dtype=bool)
    loose = np.zeros_like(strict)
    for k, i in enumerate(solids):
        a_lo, a_hi = scene.aabb(i)
        i0 = np.clip(np.ceil((a_lo - cell - origin) / cell - 1e-9).astype(int), 0, np.array(shape) - 1)
        i1 = np.clip(np.floor((a_hi + cell - origin) / cell + 1e-9).astype(int), 0, np.array(shape) - 1)
        axes = [origin[a] + cell * np.arange(i0[a], i1[a] + 1) for a in range(3)]
        sub = tuple(len(ax) for ax in axes)
        gx, gy, gz = np.meshgrid(*(ax.astype(F32) for ax in axes), indexing="ij")
        p3 = np.stack([gx.ravel(), gy.ravel(), gz.ravel()])
        d = scene.part_dist3(i, p3)
        near = d < 0.5 * cell
        if scene.cutters.get(i):
            # Coplanar cut faces leave a 0-distance sheet over the hole: no loose cells there.
            sel = np.flatnonzero(near & (d >= 0.0))
            if sel.size:
                field = (lambda q, i=i: scene.part_dist3(i, q))  # noqa: E731
                fake = scene.phantom3(field, scene.cutters[i], np.ascontiguousarray(p3[:, sel]), d[sel], 0.25 * cell)
                near[sel[fake]] = False
        box = (k, slice(i0[0], i1[0] + 1), slice(i0[1], i1[1] + 1), slice(i0[2], i1[2] + 1))
        strict[box] = (d < 0.0).reshape(sub)
        loose[box] = near.reshape(sub)
    return VoxelGrid(origin=origin, cell=cell, shape=shape, solids=list(solids), strict=strict, loose=loose)
