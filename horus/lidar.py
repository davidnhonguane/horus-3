"""LiDAR / multispectral processing: point cloud -> terrain + individual trees.

Pipeline
--------
1. ground filter (use LAS class 2 if the producer classified it, else a
   progressive grid-minimum filter)
2. DTM (1 m) from ground returns, gaps filled by nearest neighbour
3. DSM from first/highest vegetation returns, CHM = DSM - DTM (buildings/wires/water removed)
4. individual tree detection: variable-window local maxima on a smoothed CHM,
   crowns delineated with marker-controlled watershed
5. per-tree features: height, crown area, point-height distribution, spectra/NDVI
6. species (Gaussian naive Bayes, calibrated on field-plot trees) and health (NDVI)
7. DBH from height (Naslund curves), slenderness H/D
8. infrastructure extraction: power-line conductors (class 14) -> vector polyline
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .synth import SPECIES, naslund_dbh

CLASS_GROUND, CLASS_LOWVEG, CLASS_HIGHVEG, CLASS_BUILDING, CLASS_WATER, CLASS_WIRE = 2, 3, 5, 6, 9, 14


@dataclass
class Grid:
    x0: float
    y0: float
    res: float
    nx: int
    ny: int

    def index(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return ix, iy

    def centers(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def coarsen(self, f: int) -> "Grid":
        return Grid(self.x0, self.y0, self.res * f, self.nx // f, self.ny // f)


def ground_filter(x, y, z, cell=5.0, max_slope=0.35, tol=0.5):
    """Progressive grid-minimum filter: coarse minimum surface, then accept points
    within a slope-dependent tolerance of the smoothed surface."""
    x0, y0 = x.min(), y.min()
    nx = int(np.ceil((x.max() - x0) / cell)) + 1
    ny = int(np.ceil((y.max() - y0) / cell)) + 1
    ix = ((x - x0) / cell).astype(int)
    iy = ((y - y0) / cell).astype(int)
    zmin = np.full(nx * ny, np.inf)
    np.minimum.at(zmin, iy * nx + ix, z)
    zmin = zmin.reshape(ny, nx)
    # remove low outliers & fill empty cells
    bad = ~np.isfinite(zmin)
    if bad.any():
        _, (iy_, ix_) = ndi.distance_transform_edt(bad, return_indices=True)
        zmin = zmin[iy_, ix_]
    # morphological opening removes cells sitting on vegetation (no ground hit in the cell)
    surf = ndi.grey_opening(zmin, size=(5, 5))
    surf = ndi.gaussian_filter(surf, 1.0)
    gy, gx = np.gradient(surf, cell)
    slope = np.hypot(gx, gy)
    thr = tol + max_slope * cell * np.minimum(slope, 1.0)
    return (z - surf[iy, ix]) < thr[iy, ix]


def _fill_nearest(a):
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
    return a[iy, ix]


def build_surfaces(pts: dict, grid: Grid, use_classes=True):
    x, y, z = pts["x"], pts["y"], pts["z"]
    cls = pts.get("classification")
    if use_classes and cls is not None and np.mean(cls == CLASS_GROUND) > 0.02:
        ground = (cls == CLASS_GROUND) | (cls == CLASS_WATER)
        method = "producer classification (LAS class 2)"
    else:
        ground = ground_filter(x, y, z)
        method = "Horus progressive grid-minimum ground filter"
    ix, iy = grid.index(x, y)
    flat = iy * grid.nx + ix
    N = grid.nx * grid.ny
    gs = np.bincount(flat[ground], weights=z[ground], minlength=N)
    gc = np.bincount(flat[ground], minlength=N)
    dtm = np.where(gc > 0, gs / np.maximum(gc, 1), np.nan).reshape(grid.ny, grid.nx)
    ground_cov = (gc > 0).reshape(grid.ny, grid.nx)
    dtm = _fill_nearest(dtm)
    dtm = ndi.gaussian_filter(dtm, 0.7)

    veg = ~ground
    if cls is not None:
        veg &= ~np.isin(cls, (CLASS_BUILDING, CLASS_WIRE, CLASS_WATER, 7, 18))
    dsm = np.full(N, -np.inf)
    np.maximum.at(dsm, flat[veg], z[veg])
    dsm = dsm.reshape(grid.ny, grid.nx)
    veg_density = float(veg.sum() / max(np.isfinite(dsm).sum() * grid.res ** 2, 1))
    if veg_density < 2.5 and grid.res < 1.0:
        # sparse national-scan data (e.g. NLS 5 p): build the surface at 1 m, then resample to the grid
        f = int(round(1.0 / grid.res))
        ny2, nx2 = grid.ny // f, grid.nx // f
        d1 = dsm[: ny2 * f, : nx2 * f].reshape(ny2, f, nx2, f).max(axis=(1, 3))
        for _ in range(2):
            e = ~np.isfinite(d1)
            if not e.any():
                break
            nb = ndi.maximum_filter(np.where(e, -np.inf, d1), size=3)
            cnt = ndi.uniform_filter((~e).astype(float), size=3) * 9
            d1 = np.where(e & (cnt >= 3), nb, d1)
        up = np.repeat(np.repeat(d1, f, 0), f, 1)
        dsm = np.full((grid.ny, grid.nx), -np.inf)
        dsm[: up.shape[0], : up.shape[1]] = up
        dsm = np.where(np.isfinite(dsm), ndi.gaussian_filter(np.where(np.isfinite(dsm), dsm, 0), 0.6), dsm)
    # fill empty cells inside crowns (sparser returns than grid cells) from neighbours
    for _ in range(2):
        empty = ~np.isfinite(dsm)
        if not empty.any():
            break
        nb = ndi.maximum_filter(np.where(empty, -np.inf, dsm), size=3)
        nbcnt = ndi.uniform_filter((~empty).astype(float), size=3) * 9
        fill = empty & (nbcnt >= 3)
        dsm = np.where(fill, nb, dsm)
    chm = np.where(np.isfinite(dsm), dsm - dtm, 0.0)
    chm = np.clip(chm, 0, 45)
    # pit filling: replace cells far below their 3x3 median (laser fell through the crown)
    med = ndi.median_filter(chm, size=3)
    chm = np.where(chm < med - 2.0, med, chm)

    mask_bld = np.zeros((grid.ny, grid.nx), bool)
    water = np.zeros((grid.ny, grid.nx), bool)
    if cls is not None:
        b = cls == CLASS_BUILDING
        if b.any():
            mask_bld.ravel()[np.unique(flat[b])] = True
            mask_bld = ndi.binary_dilation(mask_bld, iterations=1)
        w = cls == CLASS_WATER
        k10 = max(3, int(round(10 / grid.res)) | 1)
        allc = np.bincount(flat, minlength=N).reshape(grid.ny, grid.nx)
        # water: sparse class-9 returns, or large voids (laser absorbed by open water)
        void = ndi.uniform_filter((allc > 0).astype(np.float32), k10) < 0.15
        wet = np.zeros((grid.ny, grid.nx), bool)
        if w.any():
            wc = np.bincount(flat[w], minlength=N).reshape(grid.ny, grid.nx)
            wet = ndi.uniform_filter((wc > 0).astype(np.float32), k10) > 0.004
        water = (wet | void) & (ndi.uniform_filter(chm, k10) < 1.0)
        water = ndi.binary_opening(ndi.binary_closing(water, iterations=6), iterations=4)
    stats = dict(points=int(x.shape[0]), ground_points=int(ground.sum()), ground_method=method,
                 ground_cell_coverage=float(ground_cov.mean()),
                 density_pts_m2=float(x.shape[0] / (grid.nx * grid.ny * grid.res ** 2)))
    return dtm, chm, ground, mask_bld, water, stats


def detect_trees(chm: np.ndarray, grid: Grid, min_h=3.0):
    from skimage.segmentation import watershed

    # windows tuned on the synthetic reference (F1 0.73 for dominant trees, 0.5 m CHM)
    sm = ndi.gaussian_filter(chm, max(0.25 / grid.res, 0.3))
    wins_m = (1.5, 2.5, 3.5)
    cells = [max(3, int(round(w / grid.res)) | 1) for w in wins_m]
    mf = [ndi.maximum_filter(sm, size=c) for c in cells]
    mfv = np.where(sm < 8, mf[0], np.where(sm < 18, mf[1], mf[2]))
    peaks = (sm >= mfv - 1e-6) & (sm >= min_h)
    markers, nm = ndi.label(peaks)
    # merge plateau duplicates: one marker per connected peak region (already by label)
    mask = sm > 2.0
    labels = watershed(-sm, markers, mask=mask)
    idx = np.arange(1, nm + 1)
    # treetop location = centroid of the peak region, height = max raw CHM inside crown
    py, px = np.array(ndi.center_of_mass(peaks, markers, idx)).T
    h = ndi.maximum(chm, labels, idx)
    area = ndi.sum(np.ones_like(chm), labels, idx) * grid.res ** 2
    x = grid.x0 + (px + 0.5) * grid.res
    y = grid.y0 + (py + 0.5) * grid.res
    ok = np.isfinite(h) & (h >= min_h) & (area >= 1.0)
    remap = np.zeros(nm + 1, np.int32) - 1
    remap[idx[ok]] = np.arange(ok.sum())
    labels = remap[labels]
    return dict(x=x[ok], y=y[ok], h=h[ok], crown_area=area[ok],
                crown_r=np.sqrt(area[ok] / np.pi)), labels


def tree_features(pts: dict, ground_mask, dtm, labels, grid: Grid, trees: dict):
    """Aggregate per-tree point features (height distribution + spectra)."""
    x, y, z = pts["x"], pts["y"], pts["z"]
    ix, iy = grid.index(x, y)
    hag = z - dtm[iy, ix]
    lab = labels[iy, ix]
    sel = (~ground_mask) & (lab >= 0) & (hag > 1.0)
    cls = pts.get("classification")
    if cls is not None:
        sel &= ~np.isin(cls, (CLASS_BUILDING, CLASS_WIRE, CLASS_WATER))
    L = lab[sel]
    n_t = trees["x"].shape[0]
    H = trees["h"][L]
    rel = hag[sel] / np.maximum(H, 1)
    cnt = np.bincount(L, minlength=n_t).astype(float)
    feats = {}
    feats["pts"] = cnt
    feats["pt_density"] = cnt / np.maximum(trees["crown_area"], 1)
    feats["low_frac"] = np.bincount(L, weights=(rel < 0.6), minlength=n_t) / np.maximum(cnt, 1)
    feats["rel_mean"] = np.bincount(L, weights=rel, minlength=n_t) / np.maximum(cnt, 1)
    feats["crown_ratio"] = trees["crown_r"] / np.maximum(trees["h"], 1)
    has_spec = all(k in pts for k in ("red", "green", "nir"))
    feats["has_multispectral"] = bool(has_spec and np.any(pts["nir"][sel]))
    if feats["has_multispectral"]:
        top = rel > 0.5  # sunlit upper crown, what the multispectral camera sees
        w = top.astype(float)
        wc = np.maximum(np.bincount(L, weights=w, minlength=n_t), 1)
        R = np.bincount(L, weights=w * pts["red"][sel], minlength=n_t) / wc
        G = np.bincount(L, weights=w * pts["green"][sel], minlength=n_t) / wc
        NIR = np.bincount(L, weights=w * pts["nir"][sel], minlength=n_t) / wc
        feats["ndvi"] = (NIR - R) / np.maximum(NIR + R, 1)
        feats["gr_ratio"] = G / np.maximum(R, 1)
        feats["nir"] = NIR / 256.0
    inten = pts.get("intensity")
    feats["has_intensity"] = bool(inten is not None and np.std(inten[sel]) > 0) if sel.any() else False
    if feats["has_intensity"]:
        # laser intensity (1064 nm, near-infrared) of the upper crown: low for dead / defoliated crowns
        w = (rel > 0.5).astype(float)
        wc = np.maximum(np.bincount(L, weights=w, minlength=n_t), 1)
        feats["intensity_top"] = np.bincount(L, weights=w * inten[sel].astype(float), minlength=n_t) / wc
    return feats


class GaussianNB:
    """Tiny Gaussian naive Bayes (no sklearn needed)."""

    def fit(self, X, y, n_classes):
        self.mu = np.zeros((n_classes, X.shape[1]))
        self.var = np.ones((n_classes, X.shape[1]))
        self.prior = np.full(n_classes, 1e-3)
        for c in range(n_classes):
            Xc = X[y == c]
            if len(Xc) >= 2:
                self.mu[c] = Xc.mean(0)
                self.var[c] = Xc.var(0) + 1e-3
                self.prior[c] = len(Xc) / len(X)
        return self

    def predict_proba(self, X):
        ll = -0.5 * (((X[:, None, :] - self.mu[None]) ** 2) / self.var[None] + np.log(2 * np.pi * self.var[None])).sum(-1)
        ll += np.log(self.prior)[None]
        ll -= ll.max(1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(1, keepdims=True)


def species_matrix(f):
    cols = [f["crown_ratio"], f["low_frac"], f["rel_mean"]]
    if f.get("has_multispectral"):
        cols += [f["gr_ratio"], f["nir"] / 100.0]
    return np.stack(cols, 1)


def _default_model(multispectral: bool):
    """Pre-trained species model shipped in horus/models (see tools/train_species.py)."""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parent / "models" / "species_nb.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["multispectral" if multispectral else "lidar_only"]
    m = GaussianNB()
    m.mu, m.var, m.prior = np.array(d["mu"]), np.array(d["var"]), np.array(d["prior"])
    return m


def classify(trees, f, calib_idx=None, calib_species=None):
    X = species_matrix(f)
    if calib_idx is not None and len(calib_idx) >= 15:
        model = GaussianNB().fit(X[calib_idx], calib_species, 3)
        sp_source = f"Gaussian NB calibrated on {len(calib_idx)} field-plot trees"
    else:
        model, sp_source, p = _default_model(bool(f.get("has_multispectral"))), None, None
        if model is not None:
            sp_source = "default Gaussian NB (pre-trained reference model; calibrate with field plots)"
        else:  # rule-based fallback: narrow conical crowns with long crowns = spruce
            p = np.zeros((X.shape[0], 3))
            p[:, 1] = (f["crown_ratio"] < 0.11) & (f["low_frac"] > 0.35)
            if f.get("has_multispectral"):
                p[:, 2] = (f["gr_ratio"] > 1.45)
            p[:, 0] = 1 - np.clip(p[:, 1] + p[:, 2], 0, 1)
            sp_source = "rule-based crown-shape fallback"
    proba = model.predict_proba(X) if model else p
    species = np.argmax(proba, 1)
    conf = proba.max(1)

    if f.get("has_multispectral"):
        ndvi = f["ndvi"]
        thr_dead = np.where(species == 2, 0.20, 0.15)
        thr_str = np.where(species == 2, 0.45, 0.40)
        health = np.where(ndvi < thr_dead, 2, np.where(ndvi < thr_str, 1, 0))
        # dead conifers also lose needle mass -> fewer returns per m2
        health_source = "multispectral NDVI"
    else:
        dens = f["pt_density"]
        med = np.median(dens[dens > 0]) if np.any(dens > 0) else 1
        sparse = dens < 0.45 * med  # defoliated crowns: fewer returns
        if f.get("has_intensity"):
            # robust z-score of crown intensity within each species (1064 nm laser = NIR proxy)
            I = f["intensity_top"]
            z = np.zeros_like(I)
            for k in range(3):
                m = (species == k) & (I > 0)
                if m.sum() >= 20:
                    mu = np.median(I[m])
                    mad = np.median(np.abs(I[m] - mu)) * 1.4826 + 1e-6
                    z[m] = (I[m] - mu) / mad
            health = np.where((z < -3.2) | (sparse & (z < -2.0)), 2, np.where(z < -2.0, 1, 0))
            health_source = "LiDAR intensity (near-infrared 1064 nm) + return density - no multispectral, lower confidence"
        else:
            health = np.where(sparse, 2, 0)
            health_source = "LiDAR return density only (no multispectral: lower confidence)"
    dbh = np.zeros_like(trees["h"])
    for i, s in enumerate(SPECIES):
        m = species == i
        dbh[m] = naslund_dbh(trees["h"][m], s)
    return dict(species=species, species_conf=conf, health=health, dbh=dbh,
                hd=trees["h"] / np.maximum(dbh / 100.0, 0.02),
                species_source=sp_source, health_source=health_source)


def extract_powerlines(pts: dict, min_points=200, chunk=40.0):
    """Vectorise class-14 conductor points into polylines (one per connected corridor)."""
    cls = pts.get("classification")
    if cls is None:
        return []
    w = cls == CLASS_WIRE
    if w.sum() < min_points:
        return []
    x, y, z = pts["x"][w], pts["y"][w], pts["z"][w]
    c = np.stack([x, y], 1)
    mu = c.mean(0)
    u, s, vt = np.linalg.svd(c - mu, full_matrices=False)
    t = (c - mu) @ vt[0]
    bins = np.floor((t - t.min()) / chunk).astype(int)
    nb = bins.max() + 1
    cnt = np.bincount(bins, minlength=nb)
    mx = np.bincount(bins, weights=x, minlength=nb) / np.maximum(cnt, 1)
    my = np.bincount(bins, weights=y, minlength=nb) / np.maximum(cnt, 1)
    ok = cnt > 5
    line = np.stack([mx[ok], my[ok]], 1)
    # extend ends to first / last points
    i0, i1 = np.argmin(t), np.argmax(t)
    line = np.vstack([c[i0], line, c[i1]])
    hz = float(np.percentile(z - 0, 50))
    return [dict(name="Overhead line (detected from LiDAR class 14)", line=line, n_points=int(w.sum()), z_median=hz)]


def validate(det: dict, truth: dict, max_dist=2.0, max_dh=4.0, min_truth_h=10.0):
    """Match detected trees to reference trees (greedy nearest neighbour)."""
    from scipy.spatial import cKDTree

    # reference = trees visible from above (dominant / co-dominant): no taller stem within 1.5 m
    kd0 = cKDTree(np.stack([truth["x"], truth["y"]], 1))
    nbrs = kd0.query_ball_point(np.stack([truth["x"], truth["y"]], 1), r=1.5)
    visible = np.array([all(truth["h"][j] <= truth["h"][i] for j in nb) for i, nb in enumerate(nbrs)])
    tsel = (truth["h"] >= min_truth_h) & visible
    tx, ty, th = truth["x"][tsel], truth["y"][tsel], truth["h"][tsel]
    tidx = np.nonzero(tsel)[0]
    kd = cKDTree(np.stack([tx, ty], 1))
    d, j = kd.query(np.stack([det["x"], det["y"]], 1), k=3, distance_upper_bound=max_dist)
    used = np.zeros(tx.shape[0], bool)
    match = np.full(det["x"].shape[0], -1)
    order = np.argsort(d[:, 0])
    for i in order:
        for k in range(3):
            jj = j[i, k]
            if jj >= tx.shape[0] or not np.isfinite(d[i, k]):
                break
            if not used[jj] and abs(det["h"][i] - th[jj]) <= max_dh:
                used[jj] = True
                match[i] = tidx[jj]
                break
    m = match >= 0
    # detections of trees below the threshold are not false positives
    kd_all = cKDTree(np.stack([truth["x"], truth["y"]], 1))
    dd, _ = kd_all.query(np.stack([det["x"], det["y"]], 1))
    fp = (~m) & ((dd > max_dist) | (det["h"] >= min_truth_h + 2))
    recall = used.mean()
    precision = m.sum() / max(m.sum() + fp.sum(), 1)
    dh = det["h"][m] - truth["h"][match[m]]
    return match, dict(
        reference_trees=int(tsel.sum()), detected=int(det["x"].shape[0]), matched=int(m.sum()),
        recall=float(recall), precision=float(precision),
        f1=float(2 * recall * precision / max(recall + precision, 1e-9)),
        height_bias_m=float(dh.mean()), height_rmse_m=float(np.sqrt((dh ** 2).mean())),
        min_reference_height_m=min_truth_h,
    )
