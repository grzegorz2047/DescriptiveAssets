"""Orthographic cameras for the fixed named views.

World frame: meters, right-handed, +Y up, +X right, +Z toward the viewer of the
front view. Image frame: pixel (col, row), col grows right, row grows down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import SightlineError, did_you_mean

VIEW_NAMES = ("front", "back", "left", "right", "top", "iso")
ISO_AZIMUTH_DEG = 45.0  # from +Z toward +X
ISO_ELEVATION_DEG = 30.0

_AXIS_VIEWS = {
    "front": (0.0, 0.0, -1.0),  # camera on +Z looking toward -Z
    "back": (0.0, 0.0, 1.0),
    "left": (1.0, 0.0, 0.0),  # camera on -X: shows the asset's left side
    "right": (-1.0, 0.0, 0.0),
}


def orbit_view(azimuth: float, elevation: float) -> str:
    """Name of a free view: camera at `azimuth` degrees from +Z toward +X, `elevation` above the ground."""
    return f"orbit:{azimuth:g}:{elevation:g}"


def parse_orbit(view: str) -> tuple[float, float] | None:
    """(azimuth, elevation) of an `orbit:<az>:<el>` view name, else None."""
    if not view.startswith("orbit:"):
        return None
    try:
        az, el = (float(v) for v in view.split(":")[1:3])
    except ValueError:
        raise SightlineError(f"bad orbit view '{view}'", hint="use orbit:<azimuth deg>:<elevation deg>, e.g. orbit:30:15")
    if not -90.0 < el < 90.0:
        raise SightlineError(f"orbit elevation {el:g} out of range", hint="elevation must be in (-90, 90); use view 'top' for 90")
    return az, el


def view_basis(view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (forward, right, up) unit vectors for a named view; forward is the ray direction.

    Besides the named views, `orbit:<az>:<el>` gives a free orthographic view; `iso` is orbit:45:30.
    """
    if view == "top":
        # Looking down, image right = +X, image up = -Z (back of the asset at the top).
        return np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    orbit = (ISO_AZIMUTH_DEG, ISO_ELEVATION_DEG) if view == "iso" else parse_orbit(view)
    if orbit is not None:
        az = math.radians(orbit[0])
        el = math.radians(orbit[1])
        forward = -np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    elif view in _AXIS_VIEWS:
        forward = np.array(_AXIS_VIEWS[view])
    else:
        raise SightlineError(
            f"unknown view '{view}'",
            hint=did_you_mean(view, VIEW_NAMES) or f"valid views: {', '.join(VIEW_NAMES)}, or orbit:<az>:<el>",
        )
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


@dataclass(frozen=True)
class Camera:
    view: str
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    center: np.ndarray  # world point that projects to the image center
    mpp: float  # meters per pixel
    width: int
    height: int
    near: float  # distance from the image plane through `center` back to the ray origins

    def pixel_to_world(self, cols, rows) -> np.ndarray:
        """World points on the plane through `center` for (fractional) pixel coordinates."""
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        dx = (cols - self.width / 2.0) * self.mpp
        dy = (self.height / 2.0 - rows) * self.mpp
        return self.center + dx[..., None] * self.right + dy[..., None] * self.up

    def rays(self) -> tuple[np.ndarray, np.ndarray]:
        """Ray origins and directions, one per pixel center, row-major, float32 (H*W, 3)."""
        rows, cols = np.mgrid[0 : self.height, 0 : self.width]
        plane = self.pixel_to_world(cols.ravel() + 0.5, rows.ravel() + 0.5)
        origins = plane - self.forward * self.near
        dirs = np.broadcast_to(self.forward, origins.shape)
        return origins.astype(np.float32), np.ascontiguousarray(dirs, dtype=np.float32)

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to fractional pixel coords (N, 2) as (col, row) and depth (N,)."""
        rel = np.asarray(points, dtype=np.float64) - self.center
        cols = rel @ self.right / self.mpp + self.width / 2.0
        rows = self.height / 2.0 - rel @ self.up / self.mpp
        depth = rel @ self.forward + self.near
        return np.stack([cols, rows], axis=-1), depth

    def follow_light(self, direction) -> np.ndarray:
        """The light `direction` as it falls on the front view, turned with this camera.

        Concept sheets light every view of a turnaround from the viewer's side; with the light fixed in
        the world, a back or side view would show its faces in ambient light only. Components of
        `direction` along x, y, z become components along the camera's right, up and toward-viewer axes,
        so the front view keeps the style's light unchanged.
        """
        d = np.asarray(direction, dtype=np.float64)
        return d[0] * self.right + d[1] * self.up - d[2] * self.forward


def _corners(bounds_min, bounds_max) -> np.ndarray:
    lo = np.asarray(bounds_min, dtype=np.float64)
    hi = np.asarray(bounds_max, dtype=np.float64)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def fixed_camera(view: str, center, mpp: float, width: int, height: int, near: float) -> Camera:
    forward, right, up = view_basis(view)
    return Camera(view, forward, right, up, np.asarray(center, dtype=np.float64), float(mpp), int(width), int(height), float(near))


def make_cameras(views, bounds_min, bounds_max, res: int = 256, margin: float = 0.08, mpp: float | None = None) -> dict[str, Camera]:
    """Cameras framing the box for each view, sharing one meters-per-pixel scale.

    The largest view gets `res` pixels on its long side unless `mpp` is given, so the
    same object has the same pixel size in every view of one render.
    """
    corners = _corners(bounds_min, bounds_max)
    center = corners.mean(axis=0)
    near = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))) + 1.0
    spans = {}
    for view in views:
        _, right, up = view_basis(view)
        rel = corners - center
        spans[view] = (float(np.ptp(rel @ right)), float(np.ptp(rel @ up)))
    if mpp is None:
        longest = max(max(sx, sy) for sx, sy in spans.values())
        mpp = max(longest, 1e-6) * (1.0 + 2.0 * margin) / res
    cameras = {}
    for view in views:
        sx, sy = spans[view]
        width = max(8, math.ceil(sx * (1.0 + 2.0 * margin) / mpp))
        height = max(8, math.ceil(sy * (1.0 + 2.0 * margin) / mpp))
        cameras[view] = fixed_camera(view, center, mpp, width, height, near)
    return cameras
