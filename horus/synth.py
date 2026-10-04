"""Deterministic synthetic forest estate with a DJI-Zenmuse-L3-like point cloud.

Used when no real LAS file is supplied, and as ground truth to *validate* the LiDAR
pipeline (tree detection rate, height error, species/health accuracy).

The estate mimics a southern-Savonian (Kuopio) forest holding:
  * rolling moraine terrain, a lake, a pine bog
  * forest stands: mature spruce / pine, mixed, birch, young dense, clear-cut,
    and one bark-beetle damaged spruce stand next to the 20 kV line and village
  * infrastructure: regional road, forest roads, 20 kV overhead line (class 14
    wire points in the cloud), houses (class 6 roof points), lake cabins,
    a care home, and the rescue-service access point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

SPECIES = ("pine", "spruce", "birch")
HEALTH = ("healthy", "stressed", "dead")

# Naslund height curve parameters h = 1.3 + (d / (a + b d))^2  (d in cm)
NASLUND = {"pine": (1.07, 0.185), "spruce": (1.20, 0.177), "birch": (0.69, 0.197)}
CROWN_R = {"pine": (0.6, 0.070), "spruce": (0.5, 0.060), "birch": (0.7, 0.080)}  # r = c0 + c1*dbh
CROWN_LEN = {"pine": 0.38, "spruce": 0.68, "birch": 0.50}

STAND_TYPES = {
    #               p(pine, spruce, birch)   Hmean dens/ha dead stressed
    "spruce_mature": ((0.10, 0.80, 0.10), 22.0, 750, 0.01, 0.03),
    "pine_mature":   ((0.85, 0.05, 0.10), 19.0, 600, 0.01, 0.02),
    "mixed":         ((0.35, 0.35, 0.30), 18.0, 800, 0.01, 0.03),
    "birch":         ((0.10, 0.15, 0.75), 17.0, 750, 0.01, 0.02),
    "young":         ((0.60, 0.25, 0.15), 6.5, 2200, 0.00, 0.02),
    "clearcut":      ((0.80, 0.00, 0.20), 18.0, 12, 0.02, 0.02),
    "beetle_spruce": ((0.05, 0.90, 0.05), 23.0, 700, 0.30, 0.25),
    "bog_pine":      ((0.95, 0.00, 0.05), 7.0, 350, 0.03, 0.08),
}

SPECTRA = {  # mean R, G, B, NIR on 0..255
    "pine": (45, 60, 40, 165), "spruce": (35, 55, 35, 150), "birch": (60, 95, 45, 210),
    "stressed_delta": (35, 10, 5, -60), "dead": (110, 95, 80, 85), "ground": (95, 85, 62, 120),
}


@dataclass
class SyntheticSite:
    size: int
    lat0: float
    lon0: float
    dtm: np.ndarray
    water: np.ndarray
    bog: np.ndarray
    water_level: float
    stands: np.ndarray
    stand_types: list
    trees: dict
    roads: list            # list of dict(name, cls, width, line (K,2))
    powerlines: list       # list of dict(name, kv, conductor_h, row_half, line)
    buildings: list        # list of dict(name, kind, x, y, r, weight)
    depot: tuple
    points: dict = field(default_factory=dict)
    hidden_buildings: list = field(default_factory=list)   # in the forest, NOT in the map data (ground truth)
    hidden_tracks: list = field(default_factory=list)      # cleared tracks / skid trails NOT in the map data


def _norm_noise(rng, n, sigma):
    a = ndi.gaussian_filter(rng.normal(size=(n, n)), sigma, mode="reflect")
    return (a - a.mean()) / a.std()


def _densify(line, step=2.0):
    line = np.asarray(line, dtype=float)
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        L = np.hypot(*(b - a))
        k = max(1, int(np.ceil(L / step)))
        for i in range(1, k + 1):
            out.append(a + (b - a) * i / k)
    return np.array(out)


def _chaikin(line, it=3):
    p = np.asarray(line, dtype=float)
    for _ in range(it):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((q.shape[0] * 2, 2))
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([p[0], mid, p[-1]])
    return p


def _dist_to_line_grid(n, line, res=1.0):
    """Distance raster (n x n, 1 m) to a polyline via rasterise + EDT."""
    mask = np.zeros((n, n), bool)
    d = _densify(line, 0.5)
    ix = np.clip((d[:, 0] / res).astype(int), 0, n - 1)
    iy = np.clip((d[:, 1] / res).astype(int), 0, n - 1)
    mask[iy, ix] = True
    return ndi.distance_transform_edt(~mask) * res


def naslund_dbh(h, species):
    a, b = NASLUND[species]
    s = np.sqrt(np.maximum(h - 1.3, 0.01))
    return a * s / np.maximum(1 - b * s, 0.05)


def generate(seed: int = 20260901, size: int = 800, lat0: float = 62.747066, lon0: float = 27.259548,
             pts_per_m2_crown: float = 6.0, ground_pts_per_m2: float = 2.0) -> SyntheticSite:
    rng = np.random.default_rng(seed)
    n = size
    yy, xx = np.mgrid[0:n, 0:n].astype(float) + 0.5

    # ---------------- terrain ----------------
    dtm = 118 + 9.0 * _norm_noise(rng, n, 110) + 3.0 * _norm_noise(rng, n, 35) + 0.5 * _norm_noise(rng, n, 10)
    dtm += 0.012 * (yy - n / 2)                        # gentle regional trend to the north
    lake_c = np.array([0.78 * n, 0.2 * n])
    rl = np.hypot(xx - lake_c[0], yy - lake_c[1])
    dtm -= 14 * np.exp(-(rl / (0.16 * n)) ** 2)
    region = rl < 0.32 * n
    level = float(np.quantile(dtm[region], 0.30))
    water = region & (dtm < level)
    lab, nl = ndi.label(water)
    if nl > 1:  # keep the main lake body only
        sizes = ndi.sum(water, lab, range(1, nl + 1))
        water = lab == (1 + int(np.argmax(sizes)))
    water = ndi.binary_opening(water, iterations=2)
    dtm = np.where(water, level - 0.3, dtm)
    gy, gx = np.gradient(dtm)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    dist_water = ndi.distance_transform_edt(~water)
    bog = (~water) & (dtm < level + 2.2) & (dist_water < 0.22 * n) & (slope < 4)
    bog = ndi.binary_opening(ndi.binary_closing(bog, iterations=3), iterations=3)

    # ---------------- infrastructure (vector layers ~ NLS topographic DB / Digiroad) -------------
    S = n / 800.0
    main_road = _chaikin(np.array([(60, 0), (90, 150), (150, 300), (140, 450), (170, 620), (180, 800)]) * S)
    fr1 = _chaikin(np.array([(150, 300), (300, 330), (450, 320), (560, 400), (640, 520)]) * S)
    fr2 = _chaikin(np.array([(170, 620), (300, 650), (380, 640), (520, 700), (640, 770)]) * S)
    fr3 = _chaikin(np.array([(450, 320), (430, 420), (420, 520), (380, 640)]) * S)
    fr4 = _chaikin(np.array([(90, 150), (220, 130), (330, 120), (420, 190)]) * S)
    roads = [
        dict(name="Regional road 5501", cls="main", width=7.0, line=main_road),
        dict(name="Forest road A", cls="forest", width=4.0, line=fr1),
        dict(name="Forest road B", cls="forest", width=4.0, line=fr2),
        dict(name="Forest road A-B link", cls="forest", width=4.0, line=fr3),
        dict(name="Lake road", cls="forest", width=4.0, line=fr4),
    ]
    pl = np.array([(0, 500), (205, 470), (480, 395), (800, 300)]) * S
    powerlines = [dict(name="20 kV feeder Kuopio-Vehmersalmi", kv=20, conductor_h=9.0, row_half=5.0, line=pl)]

    buildings = []
    for i, (bx, by) in enumerate([(205, 440), (232, 452), (215, 492), (246, 505), (205, 525), (262, 470)]):
        buildings.append(dict(name=f"House {i + 1}", kind="house", x=bx * S, y=by * S, r=6.0, weight=1.0))
    buildings.append(dict(name="Care home Kotirinne", kind="care_home", x=235 * S, y=575 * S, r=12.0, weight=2.0))
    buildings.append(dict(name="Farm Ahola", kind="house", x=335 * S, y=110 * S, r=8.0, weight=1.0))
    # lake cabins: closest dry points ~15 m from shore towards NW of lake
    shore_d = dist_water
    for k, ang in enumerate((135, 170)):
        a = np.radians(ang)
        for r in np.arange(40, 0.45 * n, 2):
            px, py = lake_c[0] + r * np.cos(a), lake_c[1] + r * np.sin(a)
            ix, iy = int(px), int(py)
            if 0 <= ix < n and 0 <= iy < n and not water[iy, ix] and shore_d[iy, ix] >= 15:
                buildings.append(dict(name=f"Lake cabin {k + 1}", kind="cabin", x=px, y=py, r=5.0, weight=0.7))
                break
    depot = (main_road[0, 0] + 2, main_road[0, 1] + 3)
    # unmapped features the LiDAR should discover: a hunting cabin and a summer cottage deep in the
    # forest, an old forest track to the cottage and a skid trail from a past thinning
    hidden_buildings = [dict(name="Unmapped hunting cabin", kind="cabin", x=520 * S, y=565 * S, r=5.0, weight=0.7),
                        dict(name="Unmapped cottage", kind="cabin", x=705 * S, y=690 * S, r=6.0, weight=0.7)]
    hidden_tracks = [dict(name="Unmapped forest track", cls="forest", width=4.0,
                          line=_chaikin(np.array([(640, 520), (660, 590), (690, 650), (705, 682)]) * S)),
                     dict(name="Unmapped skid trail", cls="forest", width=3.5,
                          line=_chaikin(np.array([(420, 520), (380, 555), (330, 600)]) * S))]

    # distance rasters for clearing
    d_road = np.full((n, n), np.inf)
    road_halfwidth = np.zeros((n, n))
    for r in roads + hidden_tracks:
        d = _dist_to_line_grid(n, r["line"])
        upd = d < d_road
        d_road = np.where(upd, d, d_road)
        road_halfwidth = np.where(upd, r["width"] / 2, road_halfwidth)
    d_pl = _dist_to_line_grid(n, pl)
    d_bld = np.full((n, n), np.inf)
    for b in buildings + hidden_buildings:
        d_bld = np.minimum(d_bld, np.hypot(xx - b["x"], yy - b["y"]) - b["r"])

    # ---------------- stands ----------------
    n_st = 24
    seeds = rng.uniform(0, n, size=(n_st, 2))
    # nearest-seed (Voronoi) labelling with jittered boundaries
    jx = xx + 25 * _norm_noise(rng, n, 30)
    jy = yy + 25 * _norm_noise(rng, n, 30)
    dist = np.stack([(jx - sx) ** 2 + (jy - sy) ** 2 for sx, sy in seeds])
    stands = np.argmin(dist, axis=0).astype(np.int16)
    mean_elev = ndi.mean(dtm, stands, range(n_st))
    q = np.quantile(mean_elev, [0.3, 0.7])
    types = []
    for s in range(n_st):
        if mean_elev[s] > q[1]:
            t = "pine_mature"
        elif mean_elev[s] < q[0]:
            t = rng.choice(["birch", "mixed"])
        else:
            t = rng.choice(["spruce_mature", "spruce_mature", "mixed"])
        types.append(str(t))
    # management events
    spare = [s for s in range(n_st)]
    rng.shuffle(spare)
    for s in spare[:2]:
        types[s] = "clearcut"
    for s in spare[2:4]:
        types[s] = "young"
    # beetle-damaged spruce stand: the stand that covers the line next to the village
    vx, vy = int(330 * S), int(450 * S)
    types[int(stands[vy, vx])] = "beetle_spruce"
    cx2, cy2 = int(300 * S), int(560 * S)
    if types[int(stands[cy2, cx2])] in ("clearcut", "young"):
        types[int(stands[cy2, cx2])] = "spruce_mature"
    stand_type_grid = np.array(types, dtype=object)[stands]
    stand_type_grid = np.where(bog, "bog_pine", stand_type_grid)

    # ---------------- trees ----------------
    tx, ty, th, tsp, thl, tst = [], [], [], [], [], []
    site_index = 1 + 0.06 * _norm_noise(rng, n, 60)
    for tname, (pmix, hmean, dens, pdead, pstress) in STAND_TYPES.items():
        cells = stand_type_grid == tname
        if not cells.any():
            continue
        sp = np.sqrt(10000.0 / dens)
        gx_, gy_ = np.meshgrid(np.arange(sp / 2, n, sp), np.arange(sp / 2, n, sp))
        px = (gx_ + rng.uniform(-0.45, 0.45, gx_.shape) * sp).ravel()
        py = (gy_ + rng.uniform(-0.45, 0.45, gy_.shape) * sp).ravel()
        ok = (px >= 0) & (px < n) & (py >= 0) & (py < n)
        px, py = px[ok], py[ok]
        ix, iy = px.astype(int), py.astype(int)
        keep = cells[iy, ix]
        px, py, ix, iy = px[keep], py[keep], ix[keep], iy[keep]
        m = px.shape[0]
        under = rng.random(m) < (0.22 if tname not in ("clearcut", "young") else 0.05)
        h = hmean * site_index[iy, ix] * np.exp(rng.normal(0, 0.12, m))
        h = np.where(under, h * rng.uniform(0.35, 0.6, m), h)
        spc = rng.choice(3, size=m, p=pmix)
        u = rng.random(m)
        hl = np.where(u < pdead, 2, np.where(u < pdead + pstress, 1, 0))
        tx.append(px), ty.append(py), th.append(h), tsp.append(spc), thl.append(hl)
        tst.append(np.full(m, list(STAND_TYPES).index(tname)))
    tx, ty, th = np.concatenate(tx), np.concatenate(ty), np.concatenate(th)
    tsp, thl, tst = np.concatenate(tsp), np.concatenate(thl), np.concatenate(tst)
    ix, iy = tx.astype(int), ty.astype(int)
    clear = (water[iy, ix]
             | (d_road[iy, ix] < road_halfwidth[iy, ix] + 3.0)
             | (d_pl[iy, ix] < powerlines[0]["row_half"])
             | (d_bld[iy, ix] < 14.0))
    k = ~clear
    tx, ty, th, tsp, thl, tst = tx[k], ty[k], th[k], tsp[k], thl[k], tst[k]
    sp_names = np.array(SPECIES)[tsp]
    hmax = np.array([1.3 + 1 / NASLUND[s][1] ** 2 - 1.0 for s in SPECIES])[tsp]
    th = np.clip(th, 1.6, hmax)
    dbh = np.empty_like(th)
    cr = np.empty_like(th)
    for i, s in enumerate(SPECIES):
        mm = tsp == i
        dbh[mm] = naslund_dbh(th[mm], s) * np.exp(rng.normal(0, 0.12, mm.sum()))
        cr[mm] = CROWN_R[s][0] + CROWN_R[s][1] * dbh[mm]
    cr = np.where(thl == 2, cr * 0.75, cr)
    trees = dict(x=tx, y=ty, h=th, species=tsp, health=thl, dbh=dbh, crown_r=cr, stand=tst,
                 ground=dtm[ty.astype(int), tx.astype(int)])

    site = SyntheticSite(size=n, lat0=lat0, lon0=lon0, dtm=dtm, water=water, bog=bog, water_level=level,
                         stands=stands, stand_types=types, trees=trees, roads=roads, powerlines=powerlines,
                         buildings=buildings, depot=depot, hidden_buildings=hidden_buildings,
                         hidden_tracks=hidden_tracks)
    site.points = _sample_points(rng, site, pts_per_m2_crown, ground_pts_per_m2)
    return site


def _sample_points(rng, site: SyntheticSite, crown_density: float, ground_density: float) -> dict:
    n = site.size
    T = site.trees
    m = T["x"].shape[0]
    sp = T["species"]
    hl = T["health"]
    area = np.pi * T["crown_r"] ** 2
    lam = area * crown_density * np.where(hl == 2, 0.45, 1.0)
    cnt = rng.poisson(lam)
    tid = np.repeat(np.arange(m), cnt)
    N = tid.shape[0]
    t = rng.random(N) ** 0.8
    spn = sp[tid]
    clen = T["h"][tid] * np.array([CROWN_LEN[s] for s in SPECIES])[spn]
    crr = T["crown_r"][tid]
    r = np.where(spn == 1, crr * t,                                   # spruce: cone
                 np.where(spn == 0, crr * np.sqrt(t),                 # pine: paraboloid / flat top
                          crr * 2 * np.sqrt(t * (1 - t) + 1e-6)))     # birch: ellipsoid
    rr = r * (0.6 + 0.4 * np.sqrt(rng.random(N)))
    ang = rng.uniform(0, 2 * np.pi, N)
    vx = T["x"][tid] + rr * np.cos(ang)
    vy = T["y"][tid] + rr * np.sin(ang)
    vz = T["ground"][tid] + T["h"][tid] - t * clen + rng.normal(0, 0.05, N)
    # spectra
    base = np.array([SPECTRA[s] for s in SPECIES], dtype=float)[spn]
    hv = hl[tid]
    base = np.where((hv == 1)[:, None], base + np.array(SPECTRA["stressed_delta"]), base)
    base = np.where((hv == 2)[:, None], np.array(SPECTRA["dead"], dtype=float), base)
    spec = base * np.exp(rng.normal(0, 0.10, (N, 4)))

    # ground: thinned under canopy, water mostly absorbed
    ng = rng.poisson(n * n * ground_density)
    gx = rng.uniform(0, n, ng)
    gy = rng.uniform(0, n, ng)
    gix, giy = gx.astype(int), gy.astype(int)
    vcount = np.bincount(np.clip(vy.astype(int), 0, n - 1) * n + np.clip(vx.astype(int), 0, n - 1),
                         minlength=n * n).reshape(n, n)
    under = vcount[giy, gix] >= 2
    # roofs block the laser: no ground returns inside building footprints
    roof = np.zeros(gx.shape[0], bool)
    for b in site.buildings + site.hidden_buildings:
        roof |= (gx - b["x"]) ** 2 + (gy - b["y"]) ** 2 < b["r"] ** 2
    wat = site.water[giy, gix]
    keep = np.where(under, rng.random(ng) < 0.45, True) & np.where(wat, rng.random(ng) < 0.03, True) & ~roof
    gx, gy, gix, giy, wat = gx[keep], gy[keep], gix[keep], giy[keep], wat[keep]
    gz = site.dtm[giy, gix] + rng.normal(0, 0.04, gx.shape[0])
    gspec = np.array(SPECTRA["ground"], dtype=float) * np.exp(rng.normal(0, 0.12, (gx.shape[0], 4)))
    gspec[wat] = (20, 30, 40, 10)

    # buildings: roof points (class 6)
    bx, by, bz = [], [], []
    for b in site.buildings + site.hidden_buildings:
        k = int(np.pi * b["r"] ** 2 * 4)
        a = rng.uniform(0, 2 * np.pi, k)
        rr_ = b["r"] * np.sqrt(rng.random(k))
        x_ = b["x"] + rr_ * np.cos(a)
        y_ = b["y"] + rr_ * np.sin(a)
        g = site.dtm[np.clip(y_.astype(int), 0, n - 1), np.clip(x_.astype(int), 0, n - 1)]
        bx.append(x_), by.append(y_), bz.append(g + 6.5 - 1.5 * rr_ / b["r"])
    bx, by, bz = np.concatenate(bx), np.concatenate(by), np.concatenate(bz)

    # power line conductors (class 14): 3 phases, 1.2 m apart, sag between poles every 60 m
    wx, wy, wz = [], [], []
    for p in site.powerlines:
        line = _densify(p["line"], 0.4)
        seg = np.hypot(*np.diff(line, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        dirv = np.gradient(line, axis=0)
        nrm = np.stack([-dirv[:, 1], dirv[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        span = 60.0
        u = (s % span) / span
        sag = 1.2 * 4 * u * (1 - u)
        g = site.dtm[np.clip(line[:, 1].astype(int), 0, n - 1), np.clip(line[:, 0].astype(int), 0, n - 1)]
        for off in (-1.2, 0, 1.2):
            keep_w = rng.random(line.shape[0]) < 0.6
            wx.append(line[keep_w, 0] + off * nrm[keep_w, 0])
            wy.append(line[keep_w, 1] + off * nrm[keep_w, 1])
            wz.append(g[keep_w] + p["conductor_h"] + 1.0 - sag[keep_w])
    wx, wy, wz = np.concatenate(wx), np.concatenate(wy), np.concatenate(wz)

    hag = vz - T["ground"][tid]
    cls_v = np.where(hag > 2.0, 5, 3).astype(np.uint8)
    cls_g = np.where(wat, 9, 2).astype(np.uint8)
    x = np.concatenate([vx, gx, bx, wx])
    y = np.concatenate([vy, gy, by, wy])
    z = np.concatenate([vz, gz, bz, wz])
    cls = np.concatenate([cls_v, cls_g, np.full(bx.shape, 6, np.uint8), np.full(wx.shape, 14, np.uint8)])
    spec_all = np.concatenate([spec, gspec, np.tile([150, 140, 135, 120], (bx.shape[0], 1)),
                               np.tile([60, 60, 60, 40], (wx.shape[0], 1))])
    spec_all = np.clip(spec_all * 256, 0, 65535).astype(np.uint16)
    rn = np.ones(x.shape[0], np.uint8)
    inside = (x >= 0) & (x < n) & (y >= 0) & (y < n)
    truth_tree = np.concatenate([tid, np.full(gx.shape[0] + bx.shape[0] + wx.shape[0], -1)])
    out = dict(x=x[inside], y=y[inside], z=z[inside], classification=cls[inside],
               red=spec_all[inside, 0], green=spec_all[inside, 1], blue=spec_all[inside, 2], nir=spec_all[inside, 3],
               intensity=(spec_all[inside, 3] // 2).astype(np.uint16), return_number=rn[inside],
               number_of_returns=np.ones(inside.sum(), np.uint8), truth_tree=truth_tree[inside])
    return out


def render_ortho(site: "SyntheticSite", res: float = 0.5, seed: int = 3):
    """Top-down RGB orthomosaic of the estate (what a drone RGB camera / aerial orthophoto shows):
    ground cover colours + every tree crown painted from shortest to tallest. Rows north -> south."""
    rng = np.random.default_rng(seed)
    n = site.size
    W = H = int(n / res)
    xs = (np.arange(W) + 0.5) * res
    ys = n - (np.arange(H) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    ix = np.clip(X.astype(int), 0, n - 1)
    iy = np.clip(Y.astype(int), 0, n - 1)
    noise = ndi.gaussian_filter(rng.normal(size=(H, W)), 1.5)
    # forest floor / meadow
    img = np.zeros((H, W, 3))
    img[:] = (92, 104, 58)
    img += noise[..., None] * 9
    bog = site.bog[iy, ix]
    img[bog] = (150, 140, 92)
    # roads (gravel), hidden tracks (ruts: gravel + grass), power-line ROW (meadow / shrub)
    from .geo import polyline_distance
    flatx, flaty = X.ravel(), Y.ravel()
    def dist(line):
        d, _, _ = polyline_distance(flatx, flaty, line, chunk=40000)
        return d.reshape(H, W)
    for r in site.roads:
        d = dist(r["line"])
        img[d < r["width"] / 2] = (172, 164, 148)
        img[(d >= r["width"] / 2) & (d < r["width"] / 2 + 2.5)] = (128, 132, 96)
    for r in site.hidden_tracks:
        d = dist(r["line"])
        m = d < r["width"] / 2
        img[m] = (150, 140, 112)
        img[(d < 0.6)] = (118, 128, 82)  # grass ridge between wheel ruts
    for p in site.powerlines:
        d = dist(p["line"])
        img[d < p["row_half"] + 2] = (126, 148, 78)
    yy, xx = Y, X
    for b in site.buildings + site.hidden_buildings:
        m = (xx - b["x"]) ** 2 + (yy - b["y"]) ** 2 < b["r"] ** 2
        img[m] = (122, 70, 58) if b["kind"] != "care_home" else (110, 110, 115)
        yard = ((xx - b["x"]) ** 2 + (yy - b["y"]) ** 2 < (b["r"] + 9) ** 2) & ~m
        img[yard] = img[yard] * 0.5 + np.array([120, 140, 80]) * 0.5
    img[site.water[iy, ix]] = (52, 78, 96)
    # tree crowns, short to tall
    T = site.trees
    col = {0: (58, 78, 44), 1: (34, 58, 38), 2: (88, 122, 58)}
    order = np.argsort(T["h"])
    rr = T["crown_r"]
    for i in order[T["h"][order] > 2.0]:
        r = rr[i]
        c0 = int((T["x"][i] - r) / res); c1 = int((T["x"][i] + r) / res) + 1
        r0 = int((n - T["y"][i] - r) / res); r1 = int((n - T["y"][i] + r) / res) + 1
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, W), min(r1, H)
        if c0 >= c1 or r0 >= r1:
            continue
        sx = xx[r0:r1, c0:c1] - T["x"][i]
        sy = yy[r0:r1, c0:c1] - T["y"][i]
        d2 = (sx * sx + sy * sy) / (r * r)
        m = d2 < 1
        h = T["health"][i]
        base = np.array((132, 118, 96) if h == 2 else (96, 98, 50) if h == 1 else col[int(T["species"][i])], float)
        shade = (1.15 - 0.45 * d2[m])[:, None]       # sunlit top, darker rim
        img[r0:r1, c0:c1][m] = base * shade
    return np.clip(img, 0, 255).astype(np.uint8), 0.0, float(n), res
