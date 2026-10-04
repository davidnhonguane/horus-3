"""Wildfire module.

* Fuel mapping from LiDAR-derived forest structure -> Canadian FBP fuel types
  (C-2 boreal spruce, C-3 mature pine, C-4 immature pine, D-1 deciduous,
  M-1 mixedwood, O-1a open/grass, NF non-fuel)
* Rate of spread from ISI/BUI (FBP system, Forestry Canada 1992), head-fire
  intensity (Byram) and suppression-difficulty classes used by rescue services
* Static fire-risk index (hazard x ignition likelihood x exposure x access)
* Fire-spread simulation: minimum-travel-time on a 16-neighbour grid with elliptical
  wind-driven spread and slope effect (FARSITE/FlamMap-style MTT approach)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

FUELS = ["NF", "O1", "C2", "C3", "C4", "D1", "M1"]
# a, b, c, q, BUI0, total fuel consumption (kg/m2)
FBP = {
    "C2": (110, 0.0282, 1.5, 0.70, 64, 3.6),
    "C3": (110, 0.0444, 3.0, 0.75, 62, 2.6),
    "C4": (110, 0.0293, 1.5, 0.80, 66, 3.0),
    "D1": (30, 0.0232, 1.6, 0.90, 32, 1.0),
    "O1": (190, 0.0310, 1.4, 1.00, 1, 0.35),
}
FUEL_LABEL = {"NF": "Non-fuel (water/road/building)", "O1": "O-1a open / grass / clear-cut",
              "C2": "C-2 boreal spruce", "C3": "C-3 mature pine", "C4": "C-4 immature conifer",
              "D1": "D-1 deciduous (birch)", "M1": "M-1 mixedwood"}


def _rsi(fuel, isi):
    a, b, c, *_ = FBP[fuel]
    return a * (1 - math.exp(-b * isi)) ** c


def _be(fuel, bui):
    _, _, _, q, bui0, _ = FBP[fuel]
    if fuel == "O1" or bui <= 0:
        return 1.0
    return math.exp(50 * math.log(q) * (1 / bui - 1 / bui0))


def fuel_map(g: dict) -> np.ndarray:
    """g: 5 m analysis grid dict (cover, mean_h, conifer, spruce, pine, water, road_main, building)."""
    f = np.full(g["cover"].shape, FUELS.index("O1"), np.int8)
    forest = g["cover"] >= 0.25
    con = g["conifer"]
    young = g["mean_h"] < 10
    f[forest & (con >= 0.65) & young] = FUELS.index("C4")
    f[forest & (con >= 0.65) & ~young & (g["spruce"] >= g["pine"])] = FUELS.index("C2")
    f[forest & (con >= 0.65) & ~young & (g["spruce"] < g["pine"])] = FUELS.index("C3")
    f[forest & (con < 0.30)] = FUELS.index("D1")
    f[forest & (con >= 0.30) & (con < 0.65)] = FUELS.index("M1")
    f[g["water"] | g["building"] | g["road_main"]] = FUELS.index("NF")
    return f


def head_ros(fuel: np.ndarray, g: dict, cond) -> tuple[np.ndarray, np.ndarray]:
    """Head-fire ROS (m/min) and head-fire intensity (kW/m) per cell."""
    isi, bui = cond.isi, cond.bui
    ros = np.zeros(fuel.shape)
    tfc = np.zeros(fuel.shape)
    for name in ("C2", "C3", "C4", "D1", "O1"):
        m = fuel == FUELS.index(name)
        r = _rsi(name, isi) * _be(name, bui)
        if name == "O1":
            r *= 0.70  # curing ~85 %
        ros[m] = r
        tfc[m] = FBP[name][5]
    m = fuel == FUELS.index("M1")
    pc = np.clip(g["conifer"][m], 0, 1)
    ros[m] = pc * _rsi("C2", isi) * _be("C2", bui) + (1 - pc) * 0.2 * _rsi("D1", isi) * _be("D1", bui)
    tfc[m] = pc * FBP["C2"][5] + (1 - pc) * FBP["D1"][5]
    # dead / beetle-killed trees: more dry fine fuel & ladder fuels
    ros *= 1 + 0.5 * np.clip(g["dead"], 0, 1)
    tfc *= 1 + 0.3 * np.clip(g["dead"], 0, 1)
    # wet peat bog and forest roads slow the fire
    ros = np.where(g["bog"], ros * 0.5, ros)
    ros = np.where(g["road_forest"], ros * 0.3, ros)
    ros = np.where(fuel == 0, 0.0, ros)
    hfi = 300.0 * tfc * ros
    return ros, hfi


def intensity_class(hfi):
    """Suppression capability classes (kW/m)."""
    return np.digitize(hfi, [10, 500, 2000, 4000, 10000])  # 0 none,1 low (hand tools) ... 5 extreme crown fire


INTENSITY_LABELS = ["no spread", "<500 kW/m hand tools", "500-2000 pumps & hoses",
                    "2000-4000 heavy equipment / aircraft", "4000-10000 crown fire, indirect attack only",
                    ">10000 extreme, evacuate"]


def risk_index(g: dict, hfi, response_min):
    """0..100 static fire-risk index. Transparent weighted product:
    hazard (fire intensity) x ignition likelihood x exposure x suppression delay."""
    hazard = np.clip(np.log10(hfi + 1) / 4.0, 0, 1)
    ignition = 0.3 + 0.7 * np.exp(-g["d_human"] / 150.0)
    exposure = np.clip(np.exp(-g["d_building"] / 200.0) * 1.0 + 0.5 * np.exp(-g["d_power"] / 60.0), 0, 1)
    access = np.clip(np.nan_to_num(response_min, nan=60.0) / 45.0, 0, 1)
    r = 100 * hazard * (0.5 + 0.5 * ignition) * (0.55 + 0.30 * exposure + 0.15 * access)
    return np.where(hfi <= 0, 0, r)


# ---------------------------------------------------------------- spread simulation
_OFFS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1),
         (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]


def length_to_breadth(ws_kmh, grass=False):
    if grass:
        return 1.1 * max(ws_kmh, 1) ** 0.464
    return 1.0 + 8.729 * (1 - math.exp(-0.030 * ws_kmh)) ** 2.155


def spread_graph(ros, dtm, res, wind_from_deg, ws_kmh):
    ny, nx = ros.shape
    N = nx * ny
    ang_to = math.radians(90 - (wind_from_deg + 180))  # math angle the fire heads to
    lb = length_to_breadth(ws_kmh)
    e = math.sqrt(max(1 - 1 / lb ** 2, 0))
    rows, cols, w = [], [], []
    idx = np.arange(N).reshape(ny, nx)
    for dx, dy in _OFFS:
        ys = slice(max(0, -dy), ny - max(0, dy))
        xs = slice(max(0, -dx), nx - max(0, dx))
        yd = slice(ys.start + dy, ys.stop + dy)
        xd = slice(xs.start + dx, xs.stop + dx)
        a = idx[ys, xs].ravel()
        b = idx[yd, xd].ravel()
        dist = res * math.hypot(dx, dy)
        phi = math.atan2(dy, dx) - ang_to
        dirf = (1 - e) / (1 - e * math.cos(phi))
        ra = ros[ys, xs].ravel() * dirf
        rb = ros[yd, xd].ravel() * dirf
        tan = (dtm[yd, xd].ravel() - dtm[ys, xs].ravel()) / dist
        sf = np.where(tan > 0, np.exp(3.533 * np.clip(tan, 0, 1.0) ** 1.2), np.exp(-0.5 * np.abs(tan)))
        ok = (ra > 1e-3) & (rb > 1e-3)
        t = 0.5 * dist * (1 / np.maximum(ra, 1e-3) + 1 / np.maximum(rb, 1e-3)) / sf
        rows.append(a[ok]), cols.append(b[ok]), w.append(t[ok])
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    return sparse.csr_matrix((w, (rows, cols)), shape=(N, N))


def simulate(ros, dtm, res, cond, ignition_cells, max_minutes=480):
    G = spread_graph(ros, dtm, res, cond.wind_dir_deg, cond.wind_kmh)
    src = [iy * ros.shape[1] + ix for ix, iy in ignition_cells]
    d = dijkstra(G, directed=True, indices=src, min_only=True, limit=max_minutes)
    return d.reshape(ros.shape)


# ---------------------------------------------------------------- tree-level fire hazard
# crown base height as a share of tree height (boreal stands, Finnish inventory practice):
# spruce keeps green branches near the ground (ladder fuel), pine self-prunes, birch in between
CBH_RATIO = np.array([0.45, 0.15, 0.35])       # pine, spruce, birch
SPECIES_FLAM = np.array([0.7, 1.0, 0.25])      # resin / needle-bed flammability, birch is the fire-break tree
FMC = np.array([100.0, 80.0, 35.0])            # foliar moisture % by health: healthy, stressed, dead (dry needles / twigs)


def dryness(cond) -> float:
    """0..1 dryness of fine + deeper fuels from the FWI system (FFMC = surface litter / grass, BUI = duff)."""
    return float(np.clip((cond.ffmc - 70) / 25, 0, 1) * 0.6 + np.clip(cond.bui / 80, 0, 1) * 0.4)


def tree_fire_hazard(T: dict, hfi_at_tree, grass_share, cond):
    """Per tree: how easily it burns (0..100) and P(torching = crown fire) if a fire reaches it today.

    Torching uses Van Wagner's (1977) critical surface intensity
        I0 = (0.010 * CBH * (460 + 25.9 * FMC)) ** 1.5      [kW/m]
    compared with the surface-fire intensity expected under the tree (~20 % of the cell's head-fire
    intensity, more where dry grass / open ground surrounds it)."""
    sp = np.clip(T["species"], 0, 2)
    hl = np.clip(T["health"], 0, 2)
    cbh = T["h"] * CBH_RATIO[sp] * np.where(hl == 2, 0.6, 1.0)      # dead trees keep dead lower branches
    cbh = np.clip(cbh, 0.3, None)
    i0 = (0.010 * cbh * (460 + 25.9 * FMC[hl])) ** 1.5
    i_surf = hfi_at_tree * (0.20 + 0.15 * grass_share)
    p_torch = 1 - np.exp(-(i_surf / np.maximum(i0, 1.0)) ** 1.5)
    dry = dryness(cond)
    tree_dry = np.array([0.15, 0.55, 1.0])[hl]                        # dead = standing dry fuel
    ladder = np.clip(1 - cbh / 6.0, 0, 1)                             # branches within ~6 m of the ground
    score = 100 * np.clip(0.30 * tree_dry + 0.20 * SPECIES_FLAM[sp] + 0.15 * ladder
                          + 0.15 * grass_share + 0.20 * dry * (0.5 + 0.5 * tree_dry), 0, 1)
    return score, p_torch, cbh, i0


def hazard_reason(sp, hl, cbh, grass, p_torch):
    r = []
    if hl == 2:
        r.append("dead, dry standing fuel")
    elif hl == 1:
        r.append("stressed / drying crown")
    if sp == 1:
        r.append("spruce: resinous, branches to the ground")
    if cbh < 3:
        r.append(f"low crown base {cbh:.1f} m (ladder fuel)")
    if grass > 0.4:
        r.append("dry grass / open ground around")
    if p_torch > 0.5:
        r.append("likely to torch into a crown fire today")
    return r
