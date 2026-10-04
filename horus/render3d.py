"""Geolocated 3D pictures: perspective snapshots of the point cloud around points of interest
(hazard trees, risky line spans, dead-tree clusters, fire hot-spots, likely road blockages).

Pure numpy software rendering (painter's algorithm with depth sorting), no GPU needed, so the
pictures can also go into reports / work orders.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw

from .lidar import CLASS_BUILDING, CLASS_GROUND, CLASS_WATER, CLASS_WIRE


def _rgb_from_points(P, sel, hag, highlight):
    n = sel.size
    cls = P["classification"][sel] if "classification" in P else np.zeros(n, np.uint8)
    if all(k in P for k in ("red", "green", "blue")) and np.any(P["red"][sel]):
        rgb = np.stack([P["red"][sel], P["green"][sel], P["blue"][sel]], 1).astype(float)
        rgb = np.clip(rgb / max(np.percentile(rgb, 99), 1) * 255 * 1.25, 0, 255)
    else:  # intensity / height shading for intensity-only national scans
        t = np.clip(hag / 25.0, 0, 1)
        rgb = np.stack([40 + 50 * t, 95 + 90 * t, 50 + 30 * t], 1)
        if "intensity" in P:
            it = P["intensity"][sel].astype(float)
            it = np.clip(it / max(np.percentile(it, 98), 1), 0.3, 1.2)
            rgb *= it[:, None]
    g = cls == CLASS_GROUND
    rgb[g] = np.array([125, 108, 80]) * (0.85 + 0.3 * np.random.default_rng(0).random(g.sum()))[:, None]
    rgb[cls == CLASS_WATER] = (70, 110, 150)
    rgb[cls == CLASS_BUILDING] = (190, 185, 180)
    rgb[cls == CLASS_WIRE] = (90, 230, 255)
    if highlight is not None:
        rgb[highlight] = (255, 60, 40)
    return np.clip(rgb, 0, 255)


def snapshot(site, x, y, radius=32.0, az_deg=None, elev_deg=32.0, size=(640, 420), title="", subtitle="",
             highlight_tree=None):
    """Render the cloud around (x, y) [projected metres]. az_deg: direction the camera looks FROM
    (compass bearing); default = from the south-west, i.e. looking along a typical storm wind."""
    P = site.pts
    W, H = size
    r2 = radius * 1.35
    m = (np.abs(P["x"] - x) < r2) & (np.abs(P["y"] - y) < r2)
    sel = np.nonzero(m)[0]
    dx, dy = P["x"][sel] - x, P["y"][sel] - y
    inside = dx * dx + dy * dy < r2 * r2
    sel, dx, dy = sel[inside], dx[inside], dy[inside]
    ix, iy = site.g05.index(P["x"][sel], P["y"][sel])
    ground = site.dtm05[iy, ix]
    hag = P["z"][sel] - ground
    gi, gj = site.g05.index(x, y)
    z0 = float(site.dtm05[gj, gi])
    dz = P["z"][sel] - z0
    hl = None
    if highlight_tree is not None:
        T = site.trees
        i = highlight_tree
        d = np.hypot(P["x"][sel] - T["x"][i], P["y"][sel] - T["y"][i])
        hl = (d < max(T["crown_r"][i], 1.2) * 1.1) & (hag > 1.5) & (hag < T["h"][i] + 1.5)
    rgb = _rgb_from_points(P, sel, hag, hl)

    az = math.radians(225 if az_deg is None else az_deg)
    el = math.radians(elev_deg)
    # camera position (bearing az measured clockwise from north)
    D = radius * 2.7
    cam = np.array([math.sin(az) * math.cos(el) * D, math.cos(az) * math.cos(el) * D, math.sin(el) * D + 8])
    tgt = np.array([0.0, 0.0, 8.0])
    f = tgt - cam
    f /= np.linalg.norm(f)
    rgt = np.cross(f, [0, 0, 1.0])
    rgt /= np.linalg.norm(rgt)
    up = np.cross(rgt, f)
    pts = np.stack([dx, dy, dz], 1) - cam
    xc, yc, zc = pts @ rgt, pts @ up, pts @ f
    ok = zc > 1
    focal = W * 0.62
    u = (W / 2 + focal * xc[ok] / zc[ok]).astype(int)
    v = (H / 2 - focal * yc[ok] / zc[ok]).astype(int)
    depth = zc[ok]
    col = np.clip(rgb[ok] * np.clip(1.2 - (depth - D * 0.6) / (D * 1.8), 0.55, 1.05)[:, None], 0, 255)
    isg = (P["classification"][sel][ok] == CLASS_GROUND) if "classification" in P else np.zeros(depth.shape, bool)
    # sky
    img = np.zeros((H, W, 3))
    sky = np.linspace(0, 1, H)[:, None]
    img[:] = (np.array([200, 222, 240]) * (1 - sky) + np.array([150, 175, 195]) * sky)[:, :, None].transpose(0, 2, 1)
    size_px = np.clip((focal * 0.30 / depth).astype(int) + 1, 2, 6)
    size_px = np.clip(np.where(isg, size_px + 2, size_px), 1, 8)  # ground splats overlap into a surface
    pix, dep, cid = [], [], []
    for ox in range(-4, 5):
        for oy in range(-4, 5):
            k = (ox >= -(size_px // 2)) & (ox < size_px - size_px // 2) & (oy >= -(size_px // 2)) & (oy < size_px - size_px // 2)
            uu, vv = u[k] + ox, v[k] + oy
            good = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            idx = np.nonzero(k)[0][good]
            pix.append(vv[good] * W + uu[good]), dep.append(depth[idx]), cid.append(idx)
    pix, dep, cid = np.concatenate(pix), np.concatenate(dep), np.concatenate(cid)
    order = np.argsort(-dep, kind="stable")       # far -> near, nearest written last (z-buffer)
    flat = img.reshape(-1, 3)
    flat[pix[order]] = col[cid[order]]
    img = flat.reshape(H, W, 3)
    im = Image.fromarray(img.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    if title:
        d.rectangle([0, 0, W, 44 if subtitle else 26], fill=(11, 18, 16))
        d.text((10, 6), title, fill=(233, 162, 59))
        if subtitle:
            d.text((10, 24), subtitle, fill=(220, 230, 225))
    # north arrow + scale
    na = math.radians(-(0 - (az_deg if az_deg is not None else 225)) + 180)
    cx_, cy_ = W - 30, H - 34
    d.ellipse([cx_ - 16, cy_ - 16, cx_ + 16, cy_ + 16], outline=(240, 240, 240), width=2)
    nx, ny = cx_ + 12 * math.sin(na), cy_ - 12 * math.cos(na)
    d.line([cx_, cy_, nx, ny], fill=(255, 80, 60), width=3)
    d.text((nx - 3, ny - 14), "N", fill=(255, 255, 255))
    d.text((10, H - 18), f"Horus 3D view - {radius * 2:.0f} m across - LiDAR points: {sel.size:,}", fill=(30, 30, 30))
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return b.getvalue()


def points_of_interest(site, n_trees=6):
    """Pick the places worth a picture."""
    T = site.trees
    lon_lat = site.frame.to_lonlat
    out = []

    def add(kind, title, sub, x, y, tree=None, az=None, radius=30.0):
        lo, la = lon_lat(x, y)
        out.append(dict(id=len(out), kind=kind, title=title, subtitle=sub, x=float(x), y=float(y),
                        lon=float(lo), lat=float(la), tree=tree, az=az, radius=radius))

    for r in site.workorder(n_trees):
        i = r["tree_id"]
        tgt = {"power": "the power line", "road": "the road", "bld": "a building"}.get(r["threatens"], "-")
        # look at the tree from the side opposite the asset so the asset is behind it
        add("hazard_tree", f"Hazard tree #{r['rank']}: {r['species']}, {r['height_m']} m, {r['health']}",
            f"{r['distance_m']} m from {tgt} - P(fail, design storm) {r['p_fail_design_storm']:.0%}",
            T["x"][i], T["y"][i], tree=i, radius=22.0)
    if len(site.power_p):
        # span with the most design-storm strike risk from trees around it
        hit = T["w_power"] > 0
        best, bk = -1.0, 0
        for k, sp in enumerate(site.assets.power_spans):
            mx, my = sp["mid"]
            near = hit & (np.hypot(T["x"] - mx, T["y"] - my) < 35)
            sc = float((T["p_fail_design"][near] * T["w_power"][near]).sum())
            if sc > best:
                best, bk = sc, k
        mx, my = site.assets.power_spans[bk]["mid"]
        add("power_span", f"Line span with the most hazard trees ({best:.2f} expected strikes, design storm)",
            f"{site.powerlines[0]['name']} - conductor ~{site.powerlines[0]['conductor_h']} m", mx, my, radius=35.0)
    if len(site.road_p):
        k = int(np.argmax(site.road_p))
        mx, my = site.assets.road_pieces[k]["mid"]
        add("road_block", f"Most likely road blockage ({site.road_p[k]:.0%})", site.assets.road_pieces[k]["road"],
            mx, my, radius=30.0)
    dead = T["health"] == 2
    if dead.sum() >= 5:
        g = site.g5
        cnt = np.zeros((g.ny // 4 + 1, g.nx // 4 + 1))
        ix, iy = g.index(T["x"][dead], T["y"][dead])
        np.add.at(cnt, (iy // 4, ix // 4), 1)
        cy, cx = np.unravel_index(np.argmax(cnt), cnt.shape)
        x = g.x0 + (cx * 4 + 2) * g.res
        y = g.y0 + (cy * 4 + 2) * g.res
        add("dead_cluster", f"Dead-tree cluster ({int(cnt.max())} dead trees in 20 x 20 m)",
            "bark beetle / drought damage - fuel and fall hazard", x, y, radius=30.0)
    if site.ctx.get("buildings"):
        r = np.where(site.G5["d_building"] < 150, site.risk, 0)
        if r.max() > 0:
            cy, cx = np.unravel_index(np.argmax(r), r.shape)
            add("fire_hotspot", f"Highest fire risk near homes (index {r.max():.0f}/100)",
                f"conditions: {site.cond.note or site.cond.scenario}", site.g5.x0 + (cx + .5) * 5, site.g5.y0 + (cy + .5) * 5,
                radius=35.0)
    for b in site.ctx.get("buildings", []):
        if b.get("discovered"):
            add("found_building", f"Unmapped building found in the forest",
                f"{b['name']} - not in the map data; people may be here during a fire or storm", b["x"], b["y"],
                radius=22.0)
    for r in site.ctx.get("roads", []):
        if r.get("discovered"):
            mid = r["line"][len(r["line"]) // 2]
            L = float(np.hypot(*np.diff(r["line"], axis=0).T).sum())
            add("found_road", f"Potential forest road found in the LiDAR ({L:.0f} m)", r["name"], mid[0], mid[1], radius=30.0)
    return out
