"""2-D imagery: orthophotos / satellite / drone orthomosaics -> roads and power-line corridors.

Image sources (any georeferenced RGB image):
  * drone orthomosaic (DJI L3 / RGB camera)  - synthetic one for the demo estate
  * NLS orthophoto (Maanmittauslaitos WMTS, needs the NLS key)
  * any PNG/JPG + world file (.pgw/.jgw/.wld) or GeoTIFF in ETRS-TM35FIN given with --image

Detection (1 m working resolution):
  * colour indices: excess-green ExG = (2G-R-B)/(R+G+B), brightness, local texture
  * road surface   = not green, not water, bright (gravel / asphalt / bare ruts)
  * cleared strip  = green but smooth and brighter than forest canopy (grass / shrub under a line)
  * roads: road-surface corridors 2-12 m wide through forest -> skeleton -> vectorised paths
  * power-line corridors: long (>=150 m) STRAIGHT cleared strips 6-45 m wide through forest
    (probabilistic Hough on the strip skeleton) - conductors are too thin to see, the clearing is not
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .geo import polyline_distance


class Ortho:
    """RGB image (rows north -> south) with georeferencing in the site's projected frame."""

    def __init__(self, rgb: np.ndarray, x0: float, y1: float, res: float, source: str):
        self.rgb = rgb.astype(np.uint8)
        self.x0, self.y1, self.res, self.source = x0, y1, res, source
        self.h, self.w = rgb.shape[:2]

    @property
    def bounds_xy(self):
        return self.x0, self.y1 - self.h * self.res, self.x0 + self.w * self.res, self.y1

    def png(self, max_px=1600):
        from PIL import Image
        im = Image.fromarray(self.rgb, "RGB")
        if max(im.size) > max_px:
            f = max_px / max(im.size)
            im = im.resize((int(im.size[0] * f), int(im.size[1] * f)))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=85)
        return b.getvalue()

    def resample(self, grid):
        """Nearest-neighbour onto a site Grid (rows south -> north like all Horus rasters)."""
        xs, ys = grid.centers()
        c = ((xs - self.x0) / self.res).astype(int)
        r = ((self.y1 - ys) / self.res).astype(int)
        ok = (c >= 0) & (c < self.w) & (r >= 0) & (r < self.h)
        out = np.zeros(xs.shape + (3,), np.uint8)
        out[ok] = self.rgb[r[ok], c[ok]]
        return out, ok


# ------------------------------------------------------------------------------------ loading
def load_image(path, frame=None):
    """PNG/JPG/TIF + world file (.pgw/.jgw/.tfw/.wld), or GeoTIFF with ModelTiepoint/PixelScale tags."""
    from PIL import Image
    p = Path(path)
    im = Image.open(p)
    rgb = np.asarray(im.convert("RGB"))
    for ext in (p.suffix[:2] + p.suffix[-1] + "w", ".wld", p.suffix + "w", ".pgw", ".jgw", ".tfw"):
        wf = p.with_suffix(ext)
        if wf.exists():
            a, d, b, e, c, f = [float(v) for v in wf.read_text().split()[:6]]
            return Ortho(rgb, c - a / 2, f - e / 2 if e < 0 else f, abs(a), f"image {p.name} (world file)")
    tags = getattr(im, "tag_v2", {})
    if 33922 in tags and 33550 in tags:
        tp = tags[33922]
        sc = tags[33550]
        return Ortho(rgb, float(tp[3]), float(tp[4]), float(sc[0]), f"GeoTIFF {p.name}")
    raise ValueError(f"{p.name}: no georeferencing (add a world file .pgw/.jgw in ETRS-TM35FIN)")


def nls_ortho(bbox_xy, frame, z=17):
    """Mosaic NLS orthophoto WMTS tiles (Web Mercator) over the site and resample to the site frame."""
    from PIL import Image
    from . import online
    x0, y0, x1, y1 = bbox_xy
    lon0, lat0 = frame.to_lonlat(x0, y0)
    lon1, lat1 = frame.to_lonlat(x1, y1)
    tx0, ty0 = online._tile_xy(float(lon0), float(lat1), z)
    tx1, ty1 = online._tile_xy(float(lon1), float(lat0), z)
    tiles = {}
    for tx in range(int(tx0), int(tx1) + 1):
        for ty in range(int(ty0), int(ty1) + 1):
            data = online.http(online.nls_tile_url("ortokuva", z, tx, ty), ttl=60 * 86400)
            tiles[(tx, ty)] = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    res = 0.5
    w = int((x1 - x0) / res)
    h = int((y1 - y0) / res)
    cx = x0 + (np.arange(w) + 0.5) * res
    cy = y1 - (np.arange(h) + 0.5) * res
    X, Y = np.meshgrid(cx, cy)
    lo, la = frame.to_lonlat(X, Y)
    n = 2 ** z
    fx = (lo + 180) / 360 * n
    fy = (1 - np.log(np.tan(np.radians(la)) + 1 / np.cos(np.radians(la))) / math.pi) / 2 * n
    out = np.zeros((h, w, 3), np.uint8)
    for (tx, ty), arr in tiles.items():
        m = (fx.astype(int) == tx) & (fy.astype(int) == ty)
        px = ((fx[m] - tx) * arr.shape[1]).astype(int).clip(0, arr.shape[1] - 1)
        py = ((fy[m] - ty) * arr.shape[0]).astype(int).clip(0, arr.shape[0] - 1)
        out[m] = arr[py, px]
    return Ortho(out, x0, y1, res, "NLS orthophoto (Maanmittauslaitos, CC BY 4.0)")


# ------------------------------------------------------------------------------------ detection
def _indices(rgb):
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1e-6
    exg = (2 * g - r - b) / s
    bright = s / 3
    m = ndi.uniform_filter(bright, 5)
    tex = np.sqrt(np.maximum(ndi.uniform_filter(bright * bright, 5) - m * m, 0))
    blue = b / s
    return exg, bright, tex, blue


def classify(rgb):
    exg, bright, tex, blue = _indices(rgb)
    water = (blue > 0.37) & (bright < 0.45) & (exg < 0.07)
    water = ndi.binary_opening(water, iterations=2)
    veg = (exg > 0.045) & ~water
    bs = ndi.uniform_filter(bright, 9)            # ~4.5 m: averages the light/dark mosaic of crowns
    canopy_b = np.median(bs[veg]) if veg.any() else 0.25
    road = (~veg) & (~water) & (bright > 0.33)
    cleared = veg & (bs > 1.3 * canopy_b)         # grass / shrub under a line: green and much brighter
    forest = veg & ~cleared
    return dict(water=water, veg=veg, road=road, cleared=cleared, forest=forest, exg=exg)


def _to_1m(mask_05):
    ny, nx = mask_05.shape[0] // 2, mask_05.shape[1] // 2
    return mask_05[: ny * 2, : nx * 2].reshape(ny, 2, nx, 2).mean(axis=(1, 3)) > 0.5


def find_roads_in_image(cls1, grid1, known_lines=(), exclude_lines=(), min_len=50.0):
    """cls1: classify() output at 1 m on grid1 (rows south -> north). Roads = road-surface corridors."""
    from skimage.morphology import skeletonize
    from .discover import _longest_path, _side_contrast_mask, _simplify, _smooth, _line_mask
    road = ndi.binary_closing(cls1["road"], iterations=2)
    road = ndi.binary_opening(road, iterations=1)
    dist = ndi.distance_transform_edt(road)
    forest = ndi.uniform_filter(cls1["forest"].astype(float), 25) > 0.3
    sk = skeletonize(road) & (dist >= 1.0) & (dist <= 6.5) & forest
    for line in exclude_lines:
        sk &= ~ndi.binary_dilation(_line_mask(line, grid1, sk.shape), iterations=12)
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        pix = np.argwhere(lab == k)
        if len(pix) < min_len * 0.8:
            continue
        path = _longest_path(pix)
        xy = np.stack([grid1.x0 + (path[:, 1] + 0.5) * grid1.res, grid1.y0 + (path[:, 0] + 0.5) * grid1.res], 1)
        L = float(np.hypot(*np.diff(xy, axis=0).T).sum())
        if L < min_len:
            continue
        ps = _smooth(xy, 9)
        straight = float(np.hypot(*(ps[-1] - ps[0])) / max(L, 1))
        side = _side_contrast_mask(ps, cls1["forest"], grid1, float(np.median(dist[path[:, 0], path[:, 1]]) * 2))
        if side < 0.5 or straight < 0.5:
            continue
        mapped = 0.0
        for kl in known_lines:
            d, _, _ = polyline_distance(ps[:, 0], ps[:, 1], kl)
            mapped = max(mapped, float(np.mean(d < 9)))
        out.append(dict(line=_simplify(ps, 2.0), length_m=round(L), width_m=round(float(np.median(dist[path[:, 0], path[:, 1]]) * 2), 1),
                        mapped=mapped > 0.6, confidence=round(0.5 * side + 0.5 * min(straight / 0.8, 1), 2)))
    return out


def find_corridors_in_image(cls1, grid1, min_len=150.0):
    """Long straight cleared strips through forest = probable power-line rights-of-way."""
    from skimage.morphology import skeletonize
    from skimage.transform import probabilistic_hough_line
    strip = ndi.binary_closing(cls1["cleared"] | cls1["road"], iterations=2)
    strip = ndi.binary_opening(strip, iterations=2)
    dist = ndi.distance_transform_edt(strip)
    forest = ndi.uniform_filter(cls1["forest"].astype(float), 41) > 0.25
    sk = skeletonize(strip) & (dist >= 3) & (dist <= 22) & forest
    segs = probabilistic_hough_line(sk, threshold=8, line_length=int(min_len / grid1.res * 0.5), line_gap=12,
                                    rng=np.random.default_rng(0))
    lines = []
    for (c0, r0), (c1, r1) in segs:
        p0 = np.array([grid1.x0 + (c0 + .5) * grid1.res, grid1.y0 + (r0 + .5) * grid1.res])
        p1 = np.array([grid1.x0 + (c1 + .5) * grid1.res, grid1.y0 + (r1 + .5) * grid1.res])
        lines.append([p0, p1])
    # merge collinear, overlapping segments
    merged = []
    for p0, p1 in sorted(lines, key=lambda l: -np.hypot(*(l[1] - l[0]))):
        d = p1 - p0
        u = d / (np.hypot(*d) + 1e-9)
        joined = False
        for m in merged:
            q0, q1 = m
            v = (q1 - q0) / (np.hypot(*(q1 - q0)) + 1e-9)
            if abs(u @ v) > 0.995:
                nrm = np.array([-v[1], v[0]])
                if abs((p0 - q0) @ nrm) < 8 and abs((p1 - q0) @ nrm) < 8:
                    t = sorted([(q0 - q0) @ v, (q1 - q0) @ v, (p0 - q0) @ v, (p1 - q0) @ v])
                    m[0], m[1] = q0 + v * t[0], q0 + v * t[-1]
                    joined = True
                    break
        if not joined:
            merged.append([p0.copy(), p1.copy()])
    out = []
    for p0, p1 in merged:
        L = float(np.hypot(*(p1 - p0)))
        if L < min_len:
            continue
        ts = np.linspace(0, 1, 30)
        pts = p0[None] + (p1 - p0)[None] * ts[:, None]
        ix = ((pts[:, 0] - grid1.x0) / grid1.res).astype(int).clip(0, dist.shape[1] - 1)
        iy = ((pts[:, 1] - grid1.y0) / grid1.res).astype(int).clip(0, dist.shape[0] - 1)
        w = float(np.median(dist[iy, ix]) * 2 * grid1.res)
        # a line right-of-way is vegetated (grass / shrub), a road is gravel: reject gravel-centred strips
        rd = ndi.binary_dilation(cls1["road"], iterations=3)[iy, ix].mean()
        cl = ndi.binary_dilation(cls1["cleared"], iterations=2)[iy, ix].mean()
        if rd > 0.3 or cl < 0.6:
            continue
        out.append(dict(line=np.vstack([p0, p1]), length_m=round(L), width_m=round(w, 1),
                        confidence=round(float(min(1.0, cl * (1 - rd) * min(L / 300, 1) + 0.2)), 2)))
    return out


def analyse(ortho: Ortho, grid05, known_roads=(), power_lines=()):
    """Run the image detectors on an ortho resampled to the site grid. Returns dict + 1 m classes."""
    rgb05, ok = ortho.resample(grid05)
    rgb05 = np.flipud(np.flipud(rgb05))  # already south->north via grid centres
    cls = classify(rgb05)
    grid1 = grid05.coarsen(2)
    cls1 = {k: _to_1m(v) for k, v in cls.items() if k != "exg"}
    corridors = find_corridors_in_image(cls1, grid1)
    roads = find_roads_in_image(cls1, grid1, known_roads, [c["line"] for c in corridors] + list(power_lines))
    cover = {k: float(v.mean()) for k, v in cls1.items()}
    return dict(roads=roads, corridors=corridors, cover=cover, source=ortho.source, coverage=float(ok.mean()))
