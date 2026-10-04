"""Storm module: windthrow probability per tree and strike probability on assets.

Per tree failure model (logistic, ForestGALES-inspired, coefficients set from
literature trends and meant to be re-calibrated with utility outage records in the pilot):
  gust speed, slenderness H/D, species (shallow-rooted spruce), upwind stand edge,
  emergence above neighbours, peat soil, health (dead/stressed from multispectral),
  wet & unfrozen soil (the autumn-storm situation that drives most Finnish outages).

Strike model: the fall direction follows a von Mises distribution around the downwind
bearing. For every candidate tree we cast 36 rays of length H (power line: horizontal
reach sqrt(H^2 - h_conductor^2)) over 1 m asset rasters, giving the probability that the
tree hits each road piece / line span / building if it fails.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from .synth import _densify

N_DIRS = 36
KAPPA = 2.0
DESIGN_GUST = 25.0  # m/s, utility planning storm (roughly a 1-in-10-year gust inland)


def failure_probability(T: dict, gust: float, soil_wet: float, frozen: bool):
    sp, hl = T["species"], T["health"]
    logit = (-10.5 + 0.30 * (gust - 12) + 0.022 * (np.clip(T["hd"], 30, 160) - 70)
             + 1.0 * (sp == 1) - 0.3 * (sp == 2)
             + 1.2 * T["edge"] + 0.5 * T["emergent"] + 0.8 * T["peat"]
             + 1.8 * (hl == 2) + 0.7 * (hl == 1) + 0.03 * (T["h"] - 15)
             + (0.6 * soil_wet if not frozen else -0.8))
    return 1 / (1 + np.exp(-logit))


def exposure_features(T: dict, chm: np.ndarray, grid, wind_from_deg: float):
    a = math.radians(90 - wind_from_deg)  # unit vector pointing upwind
    ux, uy = math.cos(a), math.sin(a)
    # openings = canopy cover < 30 % in a 10 m window (crown gaps inside a stand do not count)
    kc = max(3, int(round(10 / grid.res)) | 1)
    opening = ndi.uniform_filter((chm > 0.5 * np.percentile(chm[chm > 2], 75) if np.any(chm > 2) else chm > 2)
                                 .astype(np.float32), size=kc) < 0.3
    edge = np.zeros(T["x"].shape[0])
    for d in (6, 10, 14, 18, 22, 26):
        ix, iy = grid.index(T["x"] + ux * d, T["y"] + uy * d)
        edge = np.maximum(edge, opening[iy, ix] * (1.0 - (d - 6) / 30.0))
    k = max(3, int(round(20 / grid.res)) | 1)
    mean_nb = ndi.uniform_filter(chm, size=k)
    ix, iy = grid.index(T["x"], T["y"])
    emergent = np.clip((T["h"] - mean_nb[iy, ix]) / np.maximum(T["h"], 1), 0, 1)
    return edge, emergent


class AssetRaster:
    """1 m rasters with piece ids for roads, power-line spans and buildings."""

    def __init__(self, x0, y0, nx, ny, roads, powerlines, buildings, piece_len=25.0):
        self.x0, self.y0, self.nx, self.ny = x0, y0, nx, ny
        self.road = np.full((ny, nx), -1, np.int32)
        self.power = np.full((ny, nx), -1, np.int32)
        self.bld = np.full((ny, nx), -1, np.int32)
        self.road_pieces = []   # dict(road, cls, line, mid)
        self.power_spans = []
        yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
        for ri, r in enumerate(roads):
            line = _densify(r["line"], 1.0)
            seg = np.hypot(*np.diff(line, axis=0).T)
            s = np.concatenate([[0], np.cumsum(seg)])
            npc = max(1, int(round(s[-1] / piece_len)))
            edges = np.linspace(0, s[-1], npc + 1)
            for k in range(npc):
                m = (s >= edges[k]) & (s <= edges[k + 1])
                pl = line[m]
                pid = len(self.road_pieces)
                self.road_pieces.append(dict(road=r["name"], cls=r["cls"], line=pl, mid=pl[len(pl) // 2],
                                             width=r["width"]))
                self._burn(self.road, pl, r["width"] / 2 + 0.5, pid)
        for li, p in enumerate(powerlines):
            line = _densify(p["line"], 1.0)
            seg = np.hypot(*np.diff(line, axis=0).T)
            s = np.concatenate([[0], np.cumsum(seg)])
            nsp = max(1, int(round(s[-1] / 60.0)))
            edges = np.linspace(0, s[-1], nsp + 1)
            for k in range(nsp):
                m = (s >= edges[k]) & (s <= edges[k + 1])
                pl = line[m]
                pid = len(self.power_spans)
                self.power_spans.append(dict(line_name=p["name"], line=pl, mid=pl[len(pl) // 2],
                                             conductor_h=p["conductor_h"]))
                self._burn(self.power, pl, 1.8, pid)
        for bi, b in enumerate(buildings):
            m = np.hypot(xx - (b["x"] - x0), yy - (b["y"] - y0)) <= b["r"]
            self.bld[m] = bi

    def _burn(self, arr, line, half, pid):
        mask = np.zeros(arr.shape, bool)
        ix = np.floor(line[:, 0] - self.x0).astype(int)
        iy = np.floor(line[:, 1] - self.y0).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < self.nx) & (iy < self.ny)  # parts outside the survey are dropped
        if not ok.any():
            return
        mask[iy[ok], ix[ok]] = True
        r = int(math.ceil(half))
        if r > 0:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            mask = ndi.binary_dilation(mask, structure=(xx ** 2 + yy ** 2) <= half ** 2)
        arr[mask & (arr < 0)] = pid

    def dist_rasters(self):
        out = {}
        for name, arr in (("road", self.road), ("power", self.power), ("bld", self.bld)):
            out[name] = ndi.distance_transform_edt(arr < 0) if (arr >= 0).any() else np.full(arr.shape, 1e4)
        return out


def strike_analysis(T: dict, assets: AssetRaster, p_fail: np.ndarray, wind_from_deg: float, buildings):
    """Returns per-tree hit weights by asset type and per-piece hit probabilities."""
    n = T["x"].shape[0]
    dist = assets.dist_rasters()
    ix = np.clip((T["x"] - assets.x0).astype(int), 0, assets.nx - 1)
    iy = np.clip((T["y"] - assets.y0).astype(int), 0, assets.ny - 1)
    out = {"w_road": np.zeros(n), "w_power": np.zeros(n), "w_bld": np.zeros(n),
           "target": np.full(n, "", dtype=object), "d_target": np.full(n, np.inf)}
    thetas = np.linspace(0, 2 * np.pi, N_DIRS, endpoint=False)
    down = math.radians(90 - (wind_from_deg + 180))
    vm = np.exp(KAPPA * np.cos(thetas - down))
    vm /= vm.sum()
    road_p = np.zeros(len(assets.road_pieces))   # sum log(1-p)
    power_p = np.zeros(len(assets.power_spans))
    bld_p = np.zeros(len(buildings))
    contrib = {"road": [], "power": [], "bld": []}
    for kind, arr, dmap in (("road", assets.road, dist["road"]), ("power", assets.power, dist["power"]),
                            ("bld", assets.bld, dist["bld"])):
        if kind == "power" and assets.power_spans:
            hc = assets.power_spans[0]["conductor_h"]
            reach = np.sqrt(np.maximum(T["h"] ** 2 - hc ** 2, 0))
        else:
            reach = T["h"].copy()
        d0 = dmap[iy, ix]
        cand = np.nonzero(d0 < reach)[0]
        if cand.size == 0:
            continue
        L = int(np.ceil(reach[cand].max()))
        steps = np.arange(1, L + 1, dtype=float)
        tx, ty, rr = T["x"][cand], T["y"][cand], reach[cand]
        W = np.zeros(cand.size)
        for k, th in enumerate(thetas):
            px = tx[:, None] + np.cos(th) * steps[None, :]
            py = ty[:, None] + np.sin(th) * steps[None, :]
            jx = np.clip((px - assets.x0).astype(int), 0, assets.nx - 1)
            jy = np.clip((py - assets.y0).astype(int), 0, assets.ny - 1)
            ids = arr[jy, jx]
            ids = np.where(steps[None, :] <= rr[:, None], ids, -1)
            hit = ids >= 0
            anyhit = hit.any(1)
            first = np.where(anyhit, ids[np.arange(cand.size), np.argmax(hit, 1)], -1)
            W += anyhit * vm[k]
            sel = anyhit
            if sel.any():
                contrib[kind].append((cand[sel], first[sel], np.full(sel.sum(), vm[k])))
        out["w_" + kind][cand] = W
        upd = (d0 < out["d_target"]) & (d0 < reach)
        out["target"][upd] = kind
        out["d_target"][upd] = d0[upd]
    for kind, acc in (("road", road_p), ("power", power_p), ("bld", bld_p)):
        if not contrib[kind]:
            continue
        trees = np.concatenate([c[0] for c in contrib[kind]])
        pieces = np.concatenate([c[1] for c in contrib[kind]])
        w = np.concatenate([c[2] for c in contrib[kind]])
        # aggregate weight per (tree, piece), then combine trees independently
        key = trees.astype(np.int64) * 100000 + pieces
        uk, inv = np.unique(key, return_inverse=True)
        wsum = np.bincount(inv, weights=w)
        t_u, p_u = uk // 100000, uk % 100000
        sizef = np.clip((T["h"][t_u] - 8.0) / 10.0, 0.1, 1.0) if kind == "road" else 1.0
        p = np.clip(p_fail[t_u] * wsum * sizef, 0, 0.999999)
        np.add.at(acc, p_u, np.log1p(-p))
    out["road_size_factor"] = np.clip((T["h"] - 8.0) / 10.0, 0.1, 1.0)
    return out, 1 - np.exp(road_p), 1 - np.exp(power_p), 1 - np.exp(bld_p)
