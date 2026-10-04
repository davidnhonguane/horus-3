"""Coordinate utilities.

Horus works internally in a metric, projected frame (metres).  Two georeferencing
modes are supported without any third-party dependency:

* ``LocalFrame``   - local tangent plane around a lat/lon origin (synthetic sites)
* ``TM35FIN``      - ETRS89 / TM35FIN (EPSG:3067), the national grid used by the
                     National Land Survey of Finland and most Finnish LiDAR data.

If ``pyproj`` is installed any other EPSG code can be used via ``PyprojFrame``.
"""
from __future__ import annotations

import math
import numpy as np

R_EARTH = 6371008.8


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH / 1000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class LocalFrame:
    """Equirectangular local tangent plane; error < 1 cm over a few km."""

    name = "local"

    def __init__(self, lat0: float, lon0: float, x0: float = 0.0, y0: float = 0.0):
        self.lat0, self.lon0, self.x0, self.y0 = lat0, lon0, x0, y0
        self._ky = 180.0 / (math.pi * R_EARTH)
        self._kx = self._ky / math.cos(math.radians(lat0))

    def to_lonlat(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.lon0 + (x - self.x0) * self._kx, self.lat0 + (y - self.y0) * self._ky

    def from_lonlat(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return self.x0 + (lon - self.lon0) / self._kx, self.y0 + (lat - self.lat0) / self._ky

    def describe(self):
        return {"type": "local", "lat0": self.lat0, "lon0": self.lon0}


class TM35FIN:
    """ETRS-TM35FIN (EPSG:3067): Transverse Mercator, GRS80, CM 27E, k0 0.9996, FE 500 km.

    Uses the Krueger series (as in the JHS 154 recommendation); sub-millimetre accuracy
    inside Finland.
    """

    name = "EPSG:3067"
    a = 6378137.0
    f = 1 / 298.257222101
    k0 = 0.9996
    lon0 = math.radians(27.0)
    E0 = 500000.0

    def __init__(self):
        f = self.f
        n = f / (2 - f)
        self.n = n
        self.A1 = self.a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
        self.e2 = 2 * f - f * f
        self.e = math.sqrt(self.e2)
        self.h = [
            n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180,
            13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440,
            61 * n ** 3 / 240 - 103 * n ** 4 / 140,
            49561 * n ** 4 / 161280,
        ]
        self.hi = [
            n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96 - n ** 4 / 360,
            n ** 2 / 48 + n ** 3 / 15 - 437 * n ** 4 / 1440,
            17 * n ** 3 / 480 - 37 * n ** 4 / 840,
            4397 * n ** 4 / 161280,
        ]

    def from_lonlat(self, lon, lat):
        lon = np.radians(np.asarray(lon, dtype=float))
        lat = np.radians(np.asarray(lat, dtype=float))
        e = self.e
        Q = np.arcsinh(np.tan(lat)) - e * np.arctanh(e * np.sin(lat))
        beta = np.arctan(np.sinh(Q))
        l = lon - self.lon0
        eta0 = np.arctanh(np.cos(beta) * np.sin(l))
        xi0 = np.arcsin(np.sin(beta) * np.cosh(eta0))
        xi, eta = xi0.copy(), eta0.copy()
        for j, hj in enumerate(self.h, start=1):
            xi = xi + hj * np.sin(2 * j * xi0) * np.cosh(2 * j * eta0)
            eta = eta + hj * np.cos(2 * j * xi0) * np.sinh(2 * j * eta0)
        N = self.A1 * xi * self.k0
        E = self.A1 * eta * self.k0 + self.E0
        return E, N

    def to_lonlat(self, E, N):
        E = np.asarray(E, dtype=float)
        N = np.asarray(N, dtype=float)
        xi = N / (self.A1 * self.k0)
        eta = (E - self.E0) / (self.A1 * self.k0)
        xi0, eta0 = xi.copy(), eta.copy()
        for j, hj in enumerate(self.hi, start=1):
            xi0 = xi0 - hj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
            eta0 = eta0 - hj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        beta = np.arcsin(np.sin(xi0) / np.cosh(eta0))
        l = np.arcsin(np.tanh(eta0) / np.cos(beta))
        Q = np.arcsinh(np.tan(beta))
        Qp = Q.copy()
        e = self.e
        for _ in range(6):
            Qp = Q + e * np.arctanh(e * np.tanh(Qp))
        lat = np.arctan(np.sinh(Qp))
        lon = self.lon0 + l
        return np.degrees(lon), np.degrees(lat)

    def describe(self):
        return {"type": "EPSG:3067"}


class PyprojFrame:  # pragma: no cover - optional dependency
    def __init__(self, epsg: str):
        from pyproj import Transformer

        self.name = epsg
        self._inv = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        self._fwd = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)

    def to_lonlat(self, x, y):
        return self._inv.transform(np.asarray(x), np.asarray(y))

    def from_lonlat(self, lon, lat):
        return self._fwd.transform(np.asarray(lon), np.asarray(lat))

    def describe(self):
        return {"type": self.name}


def frame_for_crs(crs: str | None, x_mean: float | None = None, y_mean: float | None = None):
    """Pick a frame for an input CRS string (e.g. 'EPSG:3067')."""
    if crs is None or crs.upper() in ("EPSG:3067", "TM35FIN", "ETRS-TM35FIN"):
        # Heuristic: TM35FIN coordinates look like E 50k..800k, N 6.6M..7.8M
        return TM35FIN()
    try:
        return PyprojFrame(crs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"CRS {crs} needs pyproj (pip install pyproj)") from exc


def point_segment_distance(px, py, ax, ay, bx, by):
    """Vectorised distance from points (N,) to segments (M,) -> (N, M) plus param t."""
    px = np.asarray(px)[:, None]
    py = np.asarray(py)[:, None]
    dx = (bx - ax)[None, :]
    dy = (by - ay)[None, :]
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - ax[None, :]) * dx + (py - ay[None, :]) * dy) / L2, 0, 1)
    cx = ax[None, :] + t * dx
    cy = ay[None, :] + t * dy
    return np.hypot(px - cx, py - cy), cx, cy


def polyline_distance(px, py, line: np.ndarray, chunk: int = 20000):
    """Distance from points to a polyline (K,2) plus the closest point coordinates."""
    ax, ay = line[:-1, 0], line[:-1, 1]
    bx, by = line[1:, 0], line[1:, 1]
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    out_d = np.empty(px.shape[0])
    out_cx = np.empty(px.shape[0])
    out_cy = np.empty(px.shape[0])
    for s in range(0, px.shape[0], chunk):
        d, cx, cy = point_segment_distance(px[s:s + chunk], py[s:s + chunk], ax, ay, bx, by)
        k = np.argmin(d, axis=1)
        r = np.arange(k.shape[0])
        out_d[s:s + chunk] = d[r, k]
        out_cx[s:s + chunk] = cx[r, k]
        out_cy[s:s + chunk] = cy[r, k]
    return out_d, out_cx, out_cy
