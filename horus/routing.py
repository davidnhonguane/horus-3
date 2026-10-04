"""Emergency routing for different vehicle classes.

The surveyed area is turned into a 2.5 m travel-time surface per vehicle class:
roads (from Digiroad/NLS vectors), off-road speed from LiDAR slope and vegetation
density, water/bog constraints. Storm blockages (road pieces with high fall-blockage
probability) and the simulated fire front are injected as penalties / barriers.
Shortest paths: Dijkstra (scipy.sparse.csgraph) on an 8-neighbour graph.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

VEHICLES = {
    "fire_engine": dict(width_m=2.55, height_m=3.3, mass_t=18, min_road_w=3.0, label="Fire engine (rescue service)", main=60, forest=30, off=None, clear_min=15,
                        max_slope=0, bog=None, note="carries chainsaws; clears a fallen tree in ~15 min"),
    "ambulance": dict(width_m=2.3, height_m=2.9, mass_t=5, min_road_w=2.8, label="Ambulance", main=70, forest=30, off=None, clear_min=None, max_slope=0, bog=None,
                      note="cannot pass a blocked road"),
    "pickup_4x4": dict(width_m=2.0, height_m=1.9, mass_t=3.5, min_road_w=2.6, label="4x4 pickup (utility crew)", main=70, forest=40, off=None, clear_min=20, max_slope=0,
                       bog=None, note="line crew, chainsaw"),
    "atv": dict(width_m=1.2, height_m=1.3, mass_t=0.5, min_road_w=1.5, label="ATV / quad", main=40, forest=30, off=12, clear_min=5, max_slope=20, bog=None,
                note="drives around fallen trees off-road where forest is open"),
    "forwarder": dict(width_m=2.9, height_m=3.8, mass_t=20, min_road_w=3.0, label="Forest machine (forwarder)", main=20, forest=20, off=5, clear_min=3, max_slope=27,
                      bog=2.0, note="crosses bog slowly, clears trees fast"),
    "foot": dict(width_m=0.8, height_m=1.9, mass_t=0.1, min_road_w=0.8, label="On foot (rescue team)", main=5, forest=5, off=3.5, clear_min=1, max_slope=45, bog=2.0,
                 note="Tobler slope function, slowed by dense undergrowth"),
}

_OFF8 = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]


class Router:
    def __init__(self, layers: dict, res: float, x0: float, y0: float):
        """layers: 2.5 m grids: road_cls (0 none,1 main,2 forest), slope_deg, cover, water, bog."""
        self.L = layers
        self.res, self.x0, self.y0 = res, x0, y0
        self.ny, self.nx = layers["road_cls"].shape
        self._cache = {}

    # -- speeds -----------------------------------------------------------------------------
    def speed(self, vehicle: str):
        v = VEHICLES[vehicle]
        rc, slope, cover = self.L["road_cls"], self.L["slope_deg"], self.L["cover"]
        sp = np.zeros(rc.shape)
        sp[rc == 1] = v["main"]
        sp[rc == 2] = v["forest"]
        off = rc == 0
        rw = self.L.get("road_w")
        if rw is not None and v.get("min_road_w"):
            narrow = (rc > 0) & (rw < v["min_road_w"])   # road too narrow for this vehicle: treat as terrain
            sp[narrow] = 0
            off = off | narrow
        if v["off"]:
            if vehicle == "foot":
                tob = 6 * np.exp(-3.5 * np.abs(np.tan(np.radians(slope)) * 0.7 + 0.05)) / 5.0
                s_off = v["off"] * tob * (1 - 0.4 * cover)
            else:
                s_off = v["off"] * (1 - 0.75 * cover) * (1 - slope / max(v["max_slope"], 1))
                s_off = np.where(cover > 0.85, 0, s_off)  # dense stand: no machine access without cutting
            s_off = np.where(slope > v["max_slope"], 0, s_off)
            sp[off] = np.maximum(s_off[off], 0)
        if v["bog"] is None:
            sp[self.L["bog"] & off] = 0
        else:
            sp[self.L["bog"] & off] = np.minimum(sp[self.L["bog"] & off], v["bog"]) if v["off"] else 0
        sp[self.L["water"]] = 0
        return sp  # km/h

    def graph(self, vehicle: str, block_cells=None, block_penalty=None, barrier=None):
        sp = self.speed(vehicle).copy()
        if barrier is not None:
            sp[barrier] = 0
        pen = np.zeros(sp.shape)
        band = np.zeros(sp.shape, bool)
        clear = VEHICLES[vehicle]["clear_min"]
        if block_cells is not None and block_cells.any():
            if clear is None:
                sp[block_cells] = 0
            else:
                band = block_cells
                pen = np.where(block_cells, clear, 0.0)
        ny, nx = sp.shape
        N = nx * ny
        idx = np.arange(N).reshape(ny, nx)
        mpm = sp * 1000 / 60.0  # m/min
        R, C, W = [], [], []
        for dx, dy in _OFF8:
            ys = slice(max(0, -dy), ny - max(0, dy))
            xs = slice(max(0, -dx), nx - max(0, dx))
            yd = slice(ys.start + dy, ys.stop + dy)
            xd = slice(xs.start + dx, xs.stop + dx)
            va, vb = mpm[ys, xs].ravel(), mpm[yd, xd].ravel()
            ok = (va > 0) & (vb > 0)
            d = self.res * math.hypot(dx, dy)
            w = 0.5 * d * (1 / np.maximum(va, 1e-6) + 1 / np.maximum(vb, 1e-6))
            w = w + (band[yd, xd].ravel() & ~band[ys, xs].ravel()) * pen[yd, xd].ravel()
            R.append(idx[ys, xs].ravel()[ok]), C.append(idx[yd, xd].ravel()[ok]), W.append(w[ok])
        R, C, W = np.concatenate(R), np.concatenate(C), np.concatenate(W)
        return sparse.csr_matrix((W, (R, C)), shape=(N, N)), sp, band

    def cell(self, x, y):
        ix = int(np.clip((x - self.x0) / self.res, 0, self.nx - 1))
        iy = int(np.clip((y - self.y0) / self.res, 0, self.ny - 1))
        return ix, iy

    def snap(self, x, y, sp, radius_m=60):
        """Nearest passable cell to (x, y)."""
        ix, iy = self.cell(x, y)
        r = int(radius_m / self.res)
        y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
        x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
        sub = sp[y0:y1, x0:x1] > 0
        if not sub.any():
            return None
        yy, xx = np.nonzero(sub)
        k = np.argmin((yy + y0 - iy) ** 2 + (xx + x0 - ix) ** 2)
        return int(xx[k] + x0), int(yy[k] + y0)

    def _path(self, pred, si, ei):
        path = [ei]
        while path[-1] != si:
            path.append(pred[path[-1]])
        return np.array(path[::-1])

    def _foot_leg(self, a_cell, b_cell):
        """On-foot leg between two cells (used when a road vehicle cannot reach the point)."""
        if a_cell == b_cell:
            return 0.0, np.array([a_cell[1] * self.nx + a_cell[0]])
        if "foot" not in self._cache:
            self._cache["foot"] = self.graph("foot")[0]
        G = self._cache["foot"]
        ai = a_cell[1] * self.nx + a_cell[0]
        bi = b_cell[1] * self.nx + b_cell[0]
        d, pred = dijkstra(G, directed=True, indices=ai, return_predecessors=True)
        if not np.isfinite(d[bi]):
            return None, None
        return float(d[bi]), self._path(pred, ai, bi)

    def route(self, vehicle, start_xy, end_xy, block_cells=None, barrier=None, max_walk_m=800):
        sp0 = self.speed(vehicle)
        legs = []
        cells = {}
        for key, xy in (("start", start_xy), ("end", end_xy)):
            c = self.snap(*xy, sp0, radius_m=30)
            walk = None
            if c is None:  # e.g. lake cabin without road access: drive to nearest road, then walk
                c = self.snap(*xy, sp0, radius_m=max_walk_m)
                if c is None:
                    return dict(ok=False, reason=f"{key} is more than {max_walk_m} m from anything this vehicle can use")
                walk = self.cell(*xy)
            cells[key] = (c, walk)
        if barrier is not None:
            barrier = barrier.copy()
            for c, _ in cells.values():  # people are at the start/end - keep those cells usable
                barrier[max(0, c[1] - 3):c[1] + 4, max(0, c[0] - 3):c[0] + 4] = False
        G, sp, band = self.graph(vehicle, block_cells, barrier=barrier)
        (s, ws), (e, we) = cells["start"], cells["end"]
        si = s[1] * self.nx + s[0]
        ei = e[1] * self.nx + e[0]
        dist, pred = dijkstra(G, directed=True, indices=si, return_predecessors=True)
        if not np.isfinite(dist[ei]):
            return dict(ok=False, reason="no passable route (blocked roads, fire or terrain)")
        drive = self._path(pred, si, ei)
        total = float(dist[ei])
        walk_min = 0.0
        pieces = []
        if ws is not None:
            t, pth = self._foot_leg(ws, s)
            if t is None:
                return dict(ok=False, reason="start not reachable on foot from the road")
            walk_min += t
            pieces.append(pth)
        pieces.append(drive)
        if we is not None:
            t, pth = self._foot_leg(e, we)
            if t is None:
                return dict(ok=False, reason="destination not reachable on foot from the road")
            walk_min += t
            pieces.append(pth)
        path = np.concatenate(pieces)
        py, px = np.divmod(path, self.nx)
        xs = self.x0 + (px + 0.5) * self.res
        ys = self.y0 + (py + 0.5) * self.res
        seg = np.hypot(np.diff(xs), np.diff(ys))
        dpy, dpx = np.divmod(drive, self.nx)
        rc = self.L["road_cls"][dpy, dpx]
        on_band = band[dpy, dpx]
        clear_events = int(np.sum(on_band[1:] & ~on_band[:-1]))
        def plen(pth):
            yy, xx = np.divmod(pth, self.nx)
            return float(np.hypot(np.diff(xx), np.diff(yy)).sum() * self.res)
        walk_m = sum(plen(pc) for pc in pieces) - plen(drive)
        return dict(ok=True, minutes=total + walk_min, drive_minutes=total, walk_minutes=walk_min, walk_m=walk_m,
                    length_m=float(seg.sum()), road_share=float(np.mean(rc > 0)), clearings=clear_events,
                    clearing_minutes=clear_events * (VEHICLES[vehicle]["clear_min"] or 0),
                    x=xs, y=ys, snapped_start=s, snapped_end=e)

    def snap_any(self, x, y, sp):
        """Nearest passable cell anywhere (real surveys: the entry point can be far from any track)."""
        s = self.snap(x, y, sp, radius_m=60)
        if s is not None:
            return s
        ok = np.argwhere(sp > 0)
        if ok.size == 0:
            return None
        ix, iy = self.cell(x, y)
        k = int(np.argmin((ok[:, 0] - iy) ** 2 + (ok[:, 1] - ix) ** 2))
        return int(ok[k, 1]), int(ok[k, 0])

    def travel_time_map(self, vehicle, start_xy, block_cells=None):
        G, sp, _ = self.graph(vehicle, block_cells)
        s = self.snap_any(*start_xy, sp)
        if s is None:  # this vehicle cannot move anywhere in the survey (no roads)
            return np.full((self.ny, self.nx), np.inf)
        d = dijkstra(G, directed=True, indices=s[1] * self.nx + s[0])
        return d.reshape(self.ny, self.nx)

    def response_time_map(self, depot_xy, block_cells=None, max_hose_walk=True):
        """Fire-engine drive time from the rescue access point + walking from the road
        (crew carrying hoses / backpack pumps) to every cell."""
        drive = self.travel_time_map("fire_engine", depot_xy, block_cells)
        Gf, spf, _ = self.graph("foot")
        N = self.nx * self.ny
        reach = np.isfinite(drive.ravel()) & (self.L["road_cls"].ravel() > 0)
        src = np.nonzero(reach)[0]
        if src.size == 0:  # no drivable road in the survey: crews walk in from the entry point
            s = self.snap_any(*depot_xy, spf)
            if s is None:
                return np.full((self.ny, self.nx), np.inf)
            return dijkstra(Gf, directed=True, indices=s[1] * self.nx + s[0]).reshape(self.ny, self.nx)
        # super source node N with edges to every reachable road cell weighted by drive time
        G2 = sparse.vstack([sparse.hstack([Gf, sparse.csr_matrix((N, 1))]),
                            sparse.csr_matrix((drive.ravel()[src] + 1e-6, (np.zeros(src.size), src)),
                                              shape=(1, N + 1))]).tocsr()
        d = dijkstra(G2, directed=True, indices=N)[:N]
        return d.reshape(self.ny, self.nx)
