"""Discover infrastructure in the point cloud that the map data does not have.

* Potential forest roads / tracks / skid trails: long, narrow (2-9 m) corridors of open, smooth,
  flat ground through forest. Open-mask -> distance transform -> skeleton -> keep corridor-width,
  low-roughness, gentle-slope, elongated components -> vectorise the longest path of each.
* Buildings in the forest: compact objects 2.5-15 m high with a smooth (planar) top and no laser
  penetration (no ground returns underneath) - tree crowns are rough and porous. LAS class 6 is used
  directly when the producer classified buildings.

Anything not within a few metres of a known road / building is reported as UNMAPPED.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .geo import polyline_distance
from .lidar import CLASS_BUILDING, CLASS_GROUND


def _block(a, f, how="mean"):
    ny, nx = a.shape[0] // f, a.shape[1] // f
    return getattr(a[: ny * f, : nx * f].reshape(ny, f, nx, f), how)(axis=(1, 3))


# ------------------------------------------------------------------------------------------- tracks
def _longest_path(pix):
    """pix: (k,2) skeleton pixels (row, col) of one component -> ordered longest path (BFS twice)."""
    idx = {tuple(p): i for i, p in enumerate(pix)}
    nbrs = [[] for _ in range(len(pix))]
    for i, (r, c) in enumerate(pix):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in idx:
                    nbrs[i].append(idx[(r + dr, c + dc)])

    def bfs(s):
        prev = {s: None}
        q = [s]
        for u in q:
            for v in nbrs[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        return q[-1], prev
    a, _ = bfs(0)
    b, prev = bfs(a)
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return pix[np.array(path)]


def find_tracks(chm, dtm, ground_cov, grid, known_lines=(), power_lines=(), min_len=50.0):
    """chm/dtm/ground_cov at 1 m. Returns list of dict(line (K,2) projected, length, width, mapped)."""
    res = grid.res
    smooth_chm = ndi.median_filter(chm, 3)
    open_ = smooth_chm < 1.5
    # bridge single overhanging crowns so a track is not cut into pieces
    open_ = ndi.binary_closing(open_, structure=np.ones((3, 3)), iterations=2)
    open_ = ndi.binary_opening(open_, iterations=1)
    dist = ndi.distance_transform_edt(open_) * res          # half-width of the opening
    # surroundings must be forest (a track runs THROUGH trees, unlike fields or clear-cuts)
    forest = ndi.uniform_filter((chm > 5).astype(float), size=int(25 / res)) > 0.35
    from skimage.morphology import skeletonize
    sk = skeletonize(open_ & (dist >= 1.0)) & (dist <= 6.5) & forest
    # ground: smooth and gentle along the corridor
    m1 = ndi.uniform_filter(dtm, 3)
    rough = np.sqrt(np.maximum(ndi.uniform_filter(dtm * dtm, 3) - m1 * m1, 0))
    gy, gx = np.gradient(dtm, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    sk &= (rough < 0.25) & (slope < 15) & (ndi.uniform_filter(ground_cov.astype(float), 3) > 0.3)
    # exclude power-line clearings (also long, open and narrow)
    if power_lines:
        pl = np.zeros(chm.shape, bool)
        for line in power_lines:
            d = _line_mask(line, grid, chm.shape)
            pl |= d
        sk &= ~ndi.binary_dilation(pl, iterations=int(12 / res))
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        pix = np.argwhere(lab == k)
        if len(pix) * res < min_len * 0.8:
            continue
        path = _longest_path(pix)
        xy = np.stack([grid.x0 + (path[:, 1] + 0.5) * res, grid.y0 + (path[:, 0] + 0.5) * res], 1)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        L = float(seg.sum())
        if L < min_len:
            continue
        # elongation: path length vs component spread (rejects blobs / small clearings)
        span = float(np.hypot(*(xy.max(0) - xy.min(0))))
        if span < 0.5 * min_len:
            continue
        w = float(np.median(dist[path[:, 0], path[:, 1]]) * 2)
        xy = _simplify(_smooth(xy, 7), 2.0)
        # split into mapped / unmapped stretches, then test each unmapped stretch on its own
        dk = np.full(len(path), np.inf)
        full = np.stack([grid.x0 + (path[:, 1] + 0.5) * res, grid.y0 + (path[:, 0] + 0.5) * res], 1)
        for kl in known_lines:
            d, _, _ = polyline_distance(full[:, 0], full[:, 1], kl)
            dk = np.minimum(dk, d)
        un = dk > 9
        mapped_share = float(1 - un.mean())
        if mapped_share > 0.85:
            out.append(dict(line=xy, length_m=round(L), width_m=round(w, 1), mapped=True, mapped_share=round(mapped_share, 2),
                            confidence=1.0))
            continue
        lab1, n1 = ndi.label(un)
        for j in range(1, n1 + 1):
            part = full[lab1 == j]
            if len(part) < 3:
                continue
            Lp = float(np.hypot(*np.diff(part, axis=0).T).sum())
            if Lp < min_len:
                continue
            ps = _smooth(part, 9)
            straight = float(np.hypot(*(ps[-1] - ps[0])) / max(Lp, 1))
            side = _side_contrast(ps, chm, grid, w)
            conf = float(np.clip(0.5 * side + 0.5 * min(straight / 0.8, 1), 0, 1))
            if side < 0.55 or straight < 0.55:
                continue
            out.append(dict(line=_simplify(ps, 2.0), length_m=round(Lp), width_m=round(w, 1), mapped=False,
                            mapped_share=0.0, confidence=round(conf, 2), forest_both_sides=round(side, 2),
                            straightness=round(straight, 2)))
    out.sort(key=lambda t: -t["length_m"])
    return out


# ---------------------------------------------------------------- full road map (no map data needed)
_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _shift(a, dr, dc):
    out = np.zeros_like(a)
    H, W = a.shape
    out[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)] = a[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)]
    return out


def _junctions(sk):
    """Skeleton pixels where >= 3 separate branches meet (crossing number; diagonal staircases are not junctions)."""
    n = [_shift(sk, dr, dc) for dr, dc in _RING]
    cn = sum(((~n[i]) & n[(i + 1) % 8]).astype(int) for i in range(8))
    return sk & (cn >= 3)


def _ends(sk):
    return sk & (ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1 <= 1)


def _prune(sk, n):
    """Remove side spurs shorter than n pixels (rough crown edges), then regrow genuine road ends."""
    p = sk.copy()
    for _ in range(n):
        p &= ~_ends(p)
    for _ in range(n):
        grow = sk & ~p & ndi.binary_dilation(_ends(p), structure=np.ones((3, 3)))
        if not grow.any():
            break
        p |= grow
    return p


def _close_gaps(sk, passable, res, max_gap_m=40.0, max_angle=25.0):
    """Join a road end to the road it points at (a crown over the road, a power-line crossing or a
    culvert can interrupt the open corridor): target within max_gap_m, inside a +-max_angle cone of
    the end's heading, and the gap runs over smooth, gentle ground."""
    from skimage.draw import line as draw_line
    lab, _ = ndi.label(sk, structure=np.ones((3, 3)))
    ey, ex = np.nonzero(_ends(sk))
    py, px = np.nonzero(sk)
    if not len(ey) or not len(py):
        return sk
    out = sk.copy()
    gap = max_gap_m / res
    look = int(8 / res)
    for y0, x0 in zip(ey, ex):
        comp = lab[y0, x0]
        # heading of the end: from the pixel ~8 m back along its own branch to the end pixel
        near = (lab[py, px] == comp) & (np.abs(py - y0) <= look) & (np.abs(px - x0) <= look)
        if near.sum() < 4:
            continue
        back = np.array([py[near].mean(), px[near].mean()])
        hd = np.array([y0, x0]) - back
        nh = np.hypot(*hd)
        if nh < 1:
            continue
        hd /= nh
        dy, dx = py - y0, px - x0
        dd = np.hypot(dy, dx)
        cand = (dd > 2) & (dd <= gap) & ((lab[py, px] != comp) | (dd > look * 2))
        if not cand.any():
            continue
        cosang = (dy * hd[0] + dx * hd[1]) / np.maximum(dd, 1e-9)
        cand &= cosang >= np.cos(np.radians(max_angle))
        if not cand.any():
            continue
        k = np.nonzero(cand)[0][np.argmin(dd[cand])]
        rr, cc = draw_line(int(y0), int(x0), int(py[k]), int(px[k]))
        if passable[rr, cc].mean() >= 0.8:
            out[rr, cc] = True
    return out


def find_road_network(chm, dtm, ground_cov, grid, known_lines=(), power_lines=(), min_network_m=50.0, max_width=16.0,
                      debug=None):
    """Road map from the point cloud alone: every open, smooth, gently sloping corridor 3-16 m wide that
    forms a long linear network through the forest. Unlike find_tracks (one unmapped track next to a
    mapped network) this keeps junctions and branches, so it can draw the whole road network of a scan
    that comes without map data. Returns list of dict(line, length_m, width_m, mapped, confidence, cls)."""
    from skimage.morphology import skeletonize
    res = grid.res
    open_ = ndi.median_filter(chm, 3) < 1.5
    open_ = ndi.binary_closing(open_, structure=np.ones((3, 3)), iterations=2)
    open_ = ndi.binary_opening(open_, iterations=1)
    dist = ndi.distance_transform_edt(open_) * res
    m1 = ndi.uniform_filter(dtm, 3)
    rough = np.sqrt(np.maximum(ndi.uniform_filter(dtm * dtm, 3) - m1 * m1, 0))
    gy, gx = np.gradient(dtm, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    sk = skeletonize(open_ & (dist >= 1.0)) & (dist <= max_width / 2)
    sk &= (rough < 0.25) & (slope < 15) & (ndi.uniform_filter(ground_cov.astype(float), 3) > 0.3)
    if power_lines:
        pl = np.zeros(chm.shape, bool)
        for line in power_lines:
            pl |= _line_mask(line, grid, chm.shape)
        sk &= ~ndi.binary_dilation(pl, iterations=int(12 / res))
    # bridge short breaks (a crown leaning over the road) and re-thin, then drop little spurs
    sk = skeletonize(ndi.binary_dilation(sk, structure=np.ones((3, 3)), iterations=max(1, int(2 / res))) & open_)
    sk = _prune(sk, int(10 / res))
    sk = _close_gaps(sk, (rough < 0.35) & (slope < 18), res)
    junc = _junctions(sk)
    jd = ndi.binary_dilation(junc, structure=np.ones((3, 3)))
    comp, nc = ndi.label(sk, structure=np.ones((3, 3)))
    seg_lab, ns = ndi.label(sk & ~jd, structure=np.ones((3, 3)))
    jy, jx = np.nonzero(junc)
    jxy = np.stack([grid.x0 + (jx + 0.5) * res, grid.y0 + (jy + 0.5) * res], 1)
    out = []
    for c in range(1, nc + 1):
        cm = comp == c
        cpix = np.argwhere(cm)
        cxy = np.stack([grid.x0 + (cpix[:, 1] + 0.5) * res, grid.y0 + (cpix[:, 0] + 0.5) * res], 1)
        if cm.sum() * res < min_network_m:
            if debug is not None and cm.sum() * res > 15:
                debug.append(dict(why="short", m=float(cm.sum() * res), at=cxy.mean(0).round().tolist()))
            continue
        hw = dist[cpix[:, 0], cpix[:, 1]]
        width = float(np.median(hw) * 2)
        segs = []
        for k in np.unique(seg_lab[cm]):
            if k == 0:
                continue
            pix = np.argwhere(seg_lab == k)
            if len(pix) < 3:
                continue
            path = _longest_path(pix)
            xy = np.stack([grid.x0 + (path[:, 1] + 0.5) * res, grid.y0 + (path[:, 0] + 0.5) * res], 1)
            # reconnect both ends to the junction they were cut from
            if len(jxy):
                for end in (0, -1):
                    d = np.hypot(*(jxy - xy[end]).T)
                    j = int(np.argmin(d))
                    if d[j] <= 3 * res + 1e-6:
                        xy = np.vstack([jxy[j], xy]) if end == 0 else np.vstack([xy, jxy[j]])
            L = float(np.hypot(*np.diff(xy, axis=0).T).sum())
            seg_w = float(np.median(dist[path[:, 0], path[:, 1]]) * 2)
            free_end = any(np.min(np.hypot(*(jxy - xy[e]).T)) > 3 * res + 1e-6 for e in (0, -1)) if len(jxy) else True
            if L < 8 or (free_end and L < 15 and len(jxy)):          # leftover dangling stub
                continue
            segs.append((xy, L, seg_w))
        if not segs:
            continue
        total = sum(sg[1] for sg in segs)
        if total < min_network_m:
            continue
        # network-level evidence: forest on both sides, steady width, smooth course
        side = float(np.mean([_side_contrast(_smooth(xy, 5), chm, grid, width) for xy, _, _ in segs if len(xy) >= 4] or [0]))
        cv = float(np.std(hw) / max(np.median(hw), 0.5))
        ps = [_smooth(xy, 9) for xy, L, _ in segs if L >= 25]
        straight = float(np.mean([np.hypot(*(p[-1] - p[0])) / max(np.hypot(*np.diff(p, axis=0).T).sum(), 1) for p in ps])) if ps else 0.8
        conf = float(np.clip(0.45 * side + 0.25 * min(straight / 0.85, 1) + 0.3 * (1 - min(cv / 0.6, 1)), 0, 1))
        if debug is not None:
            debug.append(dict(why="scored", m=round(total), side=round(side, 2), straight=round(straight, 2), cv=round(cv, 2),
                              conf=round(conf, 2), at=cxy.mean(0).round().tolist()))
        if conf < 0.6 or side < 0.35:
            continue
        for xy, L, sw in segs:
            # the open corridor = carriageway + cleared verges / ditches (~5-6 m in Finnish practice)
            cls = "main" if sw >= 11.5 and L >= 60 else "forest"
            road_w = float(np.clip(sw - 5.5, 3.0, 8.0))
            dk = np.full(len(xy), np.inf)
            for kl in known_lines:
                d, _, _ = polyline_distance(xy[:, 0], xy[:, 1], kl)
                dk = np.minimum(dk, d)
            mapped = bool(np.mean(dk < 9) > 0.7)
            sside = _side_contrast(_smooth(xy, 5), chm, grid, sw) if len(xy) >= 4 else 0.0
            p9 = _smooth(xy, 9)
            sstr = float(np.hypot(*(p9[-1] - p9[0])) / max(L, 1))
            dead = (any(np.min(np.hypot(*(jxy - xy[e]).T)) > 3 * res + 1e-6 for e in (0, -1)) if len(jxy) else True)
            if debug is not None:
                debug.append(dict(why="segment", line=xy, side=round(sside, 2), straight=round(sstr, 2), L=round(L), dead=dead, w=round(sw, 1)))
            if sside < 0.5 or sw < 5.5:       # a road runs through forest in a cleared corridor at least ~6 m wide
                continue
            out.append(dict(line=_simplify(xy, 1.5), length_m=round(L), width_m=round(road_w, 1), corridor_m=round(sw, 1), mapped=mapped,
                            confidence=round(conf, 2), forest_both_sides=round(side, 2), straightness=round(straight, 2),
                            cls=cls, network=int(c), network_m=round(total)))
    out += _link_networks(out, chm, dtm, ground_cov, grid)
    out.sort(key=lambda t: -t["length_m"])
    return out


def _link_networks(segs, chm, dtm, ground_cov, grid, max_gap=70.0, max_angle=35.0):
    """Join road fragments that the scan shows as separate networks: a free road end pointing at another
    network within max_gap, across open (no canopy), flat ground that returned laser hits (not water).
    Such links are marked inferred - the road almost certainly continues across a clearing or field."""
    res = grid.res
    gy, gx = np.gradient(dtm, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    links = []
    allpts = [(i, s_["line"]) for i, s_ in enumerate(segs)]

    def sample(a, xy):
        ix = np.clip(((xy[:, 0] - grid.x0) / res).astype(int), 0, a.shape[1] - 1)
        iy = np.clip(((xy[:, 1] - grid.y0) / res).astype(int), 0, a.shape[0] - 1)
        return a[iy, ix]
    linked = set()
    for i, sg in enumerate(segs):
        line = sg["line"]
        if len(line) < 2 or sg["length_m"] < 15:
            continue
        for end in (0, -1):
            p = line[end]
            # free end: nothing else of the road map within 6 m
            if any(j != i and np.min(np.hypot(*(l2 - p).T)) < 6 for j, l2 in allpts):
                continue
            q = line[1] if end == 0 else line[-2]
            k = 1
            while np.hypot(*(p - q)) < 8 and k + 1 < len(line):
                k += 1
                q = line[k] if end == 0 else line[-1 - k]
            hd = (p - q) / max(np.hypot(*(p - q)), 1e-9)
            best = None
            for j, l2 in allpts:
                if segs[j]["network"] == sg["network"] or (min(i, j), max(i, j)) in linked:
                    continue
                v = l2 - p
                d = np.hypot(*v.T)
                ok = (d > 3) & (d <= max_gap) & ((v @ hd) / np.maximum(d, 1e-9) >= np.cos(np.radians(max_angle)))
                if ok.any():
                    m = int(np.argmin(np.where(ok, d, np.inf)))
                    if best is None or d[m] < best[0]:
                        best = (float(d[m]), j, l2[m])
            if best is None:
                continue
            gap = np.linspace(p, best[2], max(3, int(best[0] / res)))
            open_share = float(np.mean(sample(chm, gap) < 2.0))
            firm = float(np.mean(ndi.uniform_filter(ground_cov.astype(float), 5)[np.clip(((gap[:, 1] - grid.y0) / res).astype(int), 0, chm.shape[0] - 1),
                                                                              np.clip(((gap[:, 0] - grid.x0) / res).astype(int), 0, chm.shape[1] - 1)] > 0.3))
            if open_share >= 0.6 and firm >= 0.8 and float(sample(slope, gap).max()) < 12:
                linked.add((min(i, best[1]), max(i, best[1])))
                other = segs[best[1]]
                links.append(dict(line=np.array([p, best[2]]), length_m=round(best[0]), width_m=min(sg["width_m"], other["width_m"]),
                                  corridor_m=None, mapped=False, confidence=0.5, forest_both_sides=0.0, straightness=1.0,
                                  cls="forest" if "forest" in (sg["cls"], other["cls"]) else "main", network=sg["network"],
                                  network_m=sg["network_m"], inferred=True))
    return links


def _smooth(xy, k):
    if len(xy) <= k:
        return xy
    ker = np.ones(k) / k
    xs = np.convolve(np.pad(xy[:, 0], k // 2, mode="edge"), ker, "valid")
    ys = np.convolve(np.pad(xy[:, 1], k // 2, mode="edge"), ker, "valid")
    return np.stack([xs, ys], 1)[: len(xy)]


def _side_contrast(xy, chm, grid, width):
    """Share of samples along the path where it is open in the middle and forest on BOTH sides."""
    if len(xy) < 4:
        return 0.0
    tang = np.gradient(xy, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = width / 2 + 5.0

    def sample(p):
        ix = np.clip(((p[:, 0] - grid.x0) / grid.res).astype(int), 0, chm.shape[1] - 1)
        iy = np.clip(((p[:, 1] - grid.y0) / grid.res).astype(int), 0, chm.shape[0] - 1)
        return ndi.maximum_filter(chm, 5)[iy, ix] if False else chm[iy, ix]
    mid = sample(xy)
    left = np.maximum(sample(xy + nrm * off), sample(xy + nrm * (off + 3)))
    right = np.maximum(sample(xy - nrm * off), sample(xy - nrm * (off + 3)))
    ok = (mid < 2.0) & (left > 5) & (right > 5)
    return float(ok.mean())


def _side_contrast_mask(xy, forest_mask, grid, width):
    """Like _side_contrast but on a boolean forest mask (from imagery)."""
    if len(xy) < 4:
        return 0.0
    tang = np.gradient(xy, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = width / 2 + 5.0
    fm = ndi.uniform_filter(forest_mask.astype(float), 5) > 0.4

    def sample(p):
        ix = np.clip(((p[:, 0] - grid.x0) / grid.res).astype(int), 0, fm.shape[1] - 1)
        iy = np.clip(((p[:, 1] - grid.y0) / grid.res).astype(int), 0, fm.shape[0] - 1)
        return fm[iy, ix]
    left = sample(xy + nrm * off) | sample(xy + nrm * (off + 3))
    right = sample(xy - nrm * off) | sample(xy - nrm * (off + 3))
    return float((left & right).mean())


# ------------------------------------------------------------------------------------ power lines
def find_wires(pts, dtm05, grid05, known_lines=(), min_len=80.0):
    """Unclassified conductors: sparse points 5-35 m above ground with almost nothing below them
    (cleared corridor), lying on a straight line. Returns list of dict(line, length_m, height_m, mapped)."""
    from .lidar import CLASS_WIRE
    cls = pts.get("classification")
    x, y, z = pts["x"], pts["y"], pts["z"]
    if cls is not None and (cls == CLASS_WIRE).sum() > 200:
        cand = cls == CLASS_WIRE
    else:
        ix, iy = grid05.index(x, y)
        hag = z - dtm05[iy, ix]
        g1 = grid05.coarsen(4)            # 2 m cells
        jx, jy = g1.index(x, y)
        flat = jy * g1.nx + jx
        N = g1.nx * g1.ny
        lowveg = np.bincount(flat, weights=(hag > 1.5) & (hag < 4.0), minlength=N)
        high = (hag > 5) & (hag < 35)
        nhigh = np.bincount(flat, weights=high, minlength=N)
        # wire cells: few high points, empty band below (no crown), i.e. not a tree
        hz = np.full(N, np.inf)
        np.minimum.at(hz, flat[high], hag[high])
        thin = (nhigh > 0) & (nhigh <= 6) & (lowveg == 0)
        mid = np.zeros(N)
        np.add.at(mid, flat, (hag > 2) & (hag < np.maximum(hz[flat] - 2.5, 2)))
        thin &= mid == 0
        cand = high & thin[flat]
    if cand.sum() < 100:
        return []
    px, py, pz = x[cand], y[cand], z[cand]
    ix_, iy_ = grid05.index(px, py)
    ph = pz - dtm05[iy_, ix_]
    rng = np.random.default_rng(0)
    out = []
    remaining = np.ones(px.size, bool)
    for _ in range(6):                    # sequential RANSAC: up to 6 lines
        idx = np.nonzero(remaining)[0]
        if idx.size < 100:
            break
        best, best_in = None, None
        for _t in range(400):
            a, b = rng.choice(idx, 2, replace=False)
            d = np.array([px[b] - px[a], py[b] - py[a]])
            L = np.hypot(*d)
            if L < 20:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            dist = np.abs((px[idx] - px[a]) * nrm[0] + (py[idx] - py[a]) * nrm[1])
            # conductors hang at a near-constant height above ground (sag only a few metres)
            inl = idx[(dist < 1.8) & (np.abs(ph[idx] - ph[a]) < 2.5)]
            if best_in is None or inl.size > best_in.size:
                best, best_in = (a, d / L), inl
        if best_in is None or best_in.size < 80:
            break
        a, u = best
        t = (px[best_in] - px[a]) * u[0] + (py[best_in] - py[a]) * u[1]
        # keep the longest run without gaps > 40 m (spans between poles are fully strung)
        o = np.argsort(t)
        ts = t[o]
        gaps = np.nonzero(np.diff(ts) > 40)[0]
        segs = np.split(np.arange(ts.size), gaps + 1)
        sg = max(segs, key=len)
        t0, t1 = ts[sg[0]], ts[sg[-1]]
        if t1 - t0 < min_len:
            remaining[best_in] = False
            continue
        sel = best_in[o[sg]]
        hs = ph[sel]
        good = np.abs(hs - np.median(hs)) < 2.5
        # evenly strung: points along the whole run (wire every few metres), not scattered tree tops
        occ = np.unique(np.floor((ts[sg] - t0) / 10)).size / max((t1 - t0) / 10, 1)
        if good.mean() < 0.8 or occ < 0.7 or sel.size / max(t1 - t0, 1) < 0.5:
            remaining[best_in] = False
            continue
        p0 = np.array([px[a], py[a]]) + u * t0
        p1 = np.array([px[a], py[a]]) + u * t1
        line = np.vstack([p0, p1])
        ix, iy = grid05.index(px[sel], py[sel])
        h = float(np.median(pz[sel] - dtm05[iy, ix]))
        mapped = False
        for kl in known_lines:
            d, _, _ = polyline_distance(np.linspace(p0[0], p1[0], 20), np.linspace(p0[1], p1[1], 20), kl)
            mapped |= bool(np.mean(d < 10) > 0.5)
        dup = False
        for q in out:  # same conductor found twice (parallel phases) -> keep the first
            d, _, _ = polyline_distance(np.linspace(p0[0], p1[0], 10), np.linspace(p0[1], p1[1], 10), q["line"])
            dup |= bool(np.mean(d < 6) > 0.6)
        if not dup:
            out.append(dict(line=line, length_m=round(float(t1 - t0)), height_m=round(h, 1), points=int(sel.size),
                            mapped=mapped))
        remaining[best_in] = False
    return out


def _line_mask(line, grid, shape):
    m = np.zeros(shape, bool)
    from .synth import _densify
    d = _densify(line, 0.5)
    ix = ((d[:, 0] - grid.x0) / grid.res).astype(int)
    iy = ((d[:, 1] - grid.y0) / grid.res).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < shape[1]) & (iy < shape[0])
    m[iy[ok], ix[ok]] = True
    return m


def _simplify(xy, tol):
    """Douglas-Peucker."""
    if len(xy) < 3:
        return xy
    a, b = xy[0], xy[-1]
    ab = b - a
    L = np.hypot(*ab) or 1e-9
    d = np.abs(ab[0] * (xy[:, 1] - a[1]) - ab[1] * (xy[:, 0] - a[0])) / L
    i = int(np.argmax(d))
    if d[i] > tol:
        return np.vstack([_simplify(xy[: i + 1], tol)[:-1], _simplify(xy[i:], tol)])
    return np.array([a, b])


# ---------------------------------------------------------------------------------------- buildings
def find_buildings(pts, dtm05, grid05, known_buildings=(), min_area=20.0, max_area=800.0):
    """Returns list of dict(x, y, area_m2, height_m, mapped, method)."""
    cls = pts.get("classification")
    x, y, z = pts["x"], pts["y"], pts["z"]
    g1 = grid05.coarsen(2)
    ny, nx = g1.ny, g1.nx
    ix, iy = g1.index(x, y)
    flat = iy * nx + ix
    N = nx * ny
    dtm1 = _block(dtm05, 2)
    if cls is not None and (cls == CLASS_BUILDING).sum() > 50:
        b = cls == CLASS_BUILDING
        mask = (np.bincount(flat[b], minlength=N) > 0).reshape(ny, nx)
        mask = ndi.binary_closing(mask, iterations=2)
        method = "LAS class 6 (producer classified)"
        hag_max = np.full(N, -np.inf)
        np.maximum.at(hag_max, flat[b], z[b] - dtm1.ravel()[flat[b]])
        hmax = hag_max.reshape(ny, nx)
    else:
        ground = cls == CLASS_GROUND if cls is not None else np.zeros(x.shape, bool)
        hag = z - dtm1.ravel()[flat]
        nong = ~ground & (hag > 2.0)
        dsm = np.full(N, -np.inf)
        np.maximum.at(dsm, flat[nong], hag[nong])
        hmax = dsm.reshape(ny, nx)
        cnt_all = np.bincount(flat, minlength=N).reshape(ny, nx)
        cnt_g = np.bincount(flat[ground], minlength=N).reshape(ny, nx)
        tall = np.isfinite(hmax) & (hmax > 2.5) & (hmax < 15)
        hm = np.where(np.isfinite(hmax), hmax, 0)
        # roofs are smooth (planar): small residual against a 3x3 plane-ish smoothing
        resid = np.abs(hm - ndi.uniform_filter(hm, 3))
        smooth = ndi.uniform_filter((resid < 0.35).astype(float), 3) > 0.75
        # and opaque: almost no ground returns below the object
        opaque = ndi.uniform_filter(cnt_g.astype(float), 3) / np.maximum(ndi.uniform_filter(cnt_all.astype(float), 3), 1e-6) < 0.04
        mask = ndi.binary_opening(tall & smooth & opaque, iterations=1)
        method = "LiDAR shape: smooth, opaque roof (no producer classification)"
    lab, n = ndi.label(mask)
    out = []
    for k in range(1, n + 1):
        cells = lab == k
        area = float(cells.sum() * g1.res ** 2)
        if not (min_area <= area <= max_area):
            continue
        rr, cc = np.nonzero(cells)
        # compactness: buildings are blocky, not thin
        h_ = rr.max() - rr.min() + 1
        w_ = cc.max() - cc.min() + 1
        if cells.sum() / (h_ * w_) < 0.45 or max(h_, w_) / max(min(h_, w_), 1) > 4:
            continue
        bx = g1.x0 + (cc.mean() + 0.5) * g1.res
        by = g1.y0 + (rr.mean() + 0.5) * g1.res
        hh = float(np.nanmax(np.where(np.isfinite(hmax[cells]), hmax[cells], np.nan)))
        d_known = min([np.hypot(bx - q["x"], by - q["y"]) for q in known_buildings] or [1e9])
        out.append(dict(x=float(bx), y=float(by), area_m2=round(area), height_m=round(hh, 1),
                        r=float(np.sqrt(area / np.pi)), mapped=bool(d_known < 15), method=method))
    return out
