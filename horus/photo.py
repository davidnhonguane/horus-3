"""Analyse a single photo (JPG / PNG / TIF) without a 3D scan.

Two kinds of photo:

* FROM ABOVE (drone / aerial / satellite, camera pointing down)
    - placed on the map: world file / GeoTIFF, else DJI EXIF + XMP (GPS position, flight height above
      take-off, camera focal length -> ground sampling distance, gimbal yaw -> rotated north-up), else at a
      lat/lon given by the user with an assumed scale
    - land cover: forest canopy, open vegetation, dry grass / bare ground, road surface, water
    - individual tree crowns (local maxima of crown brightness + watershed), each rated from its colour:
      healthy (green), stressed (yellowing / pale), dead (brown, red-grey or grey - beetle-killed or dry)
    - roads and power-line clearings (horus.imagery detectors) and the trees standing next to them
    - fire hazard: dead crowns + dry grass, scaled by today's fuel dryness

* FROM THE GROUND (phone / camera photo into the forest)
    - sky removed, then the share of green, yellowing and brown / grey (dead or dry) foliage and of dry
      grass / litter in the lower part of the picture -> crown-health and fire-fuel indicators

What a single photo can NOT give: tree heights, storm-fall reach, exact positions without GPS. Those need a
3D scan (upload it together with the photo).
"""
from __future__ import annotations

import io
import math
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HEALTH = ("healthy", "stressed", "dead")


# ------------------------------------------------------------------------------------------ metadata
def read_meta(path) -> dict:
    """EXIF GPS + camera + DJI XMP (RelativeAltitude, GimbalPitch/Yaw) from a JPG."""
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS
    im = Image.open(path)
    out = dict(width=im.size[0], height=im.size[1])
    try:
        ex = im.getexif()
        base = {TAGS.get(k, k): v for k, v in ex.items()}
        sub = {}
        try:
            sub = {TAGS.get(k, k): v for k, v in ex.get_ifd(0x8769).items()}
        except Exception:  # noqa: BLE001
            pass
        gps = {}
        try:
            gps = {GPSTAGS.get(k, k): v for k, v in ex.get_ifd(0x8825).items()}
        except Exception:  # noqa: BLE001
            pass
        out["make"] = str(base.get("Make", "")).strip("\x00 ")
        out["model"] = str(base.get("Model", "")).strip("\x00 ")
        fl = sub.get("FocalLength") or base.get("FocalLength")
        f35 = sub.get("FocalLengthIn35mmFilm") or base.get("FocalLengthIn35mmFilm")
        if fl:
            out["focal_mm"] = float(fl)
        if f35:
            out["focal35_mm"] = float(f35)

        def dms(v, ref):
            d, m, s = (float(x) for x in v)
            r = d + m / 60 + s / 3600
            return -r if str(ref).upper().startswith(("S", "W")) else r
        if "GPSLatitude" in gps and "GPSLongitude" in gps:
            out["lat"] = dms(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
            out["lon"] = dms(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        if "GPSAltitude" in gps:
            out["gps_alt"] = float(gps["GPSAltitude"])
    except Exception:  # noqa: BLE001
        pass
    # DJI XMP (drone-dji:...) lives as plain text inside the JPEG
    try:
        raw = Path(path).read_bytes()[:400_000]
        i = raw.find(b"<x:xmpmeta")
        if i >= 0:
            xmp = raw[i:raw.find(b"</x:xmpmeta>", i) + 12].decode("utf-8", "replace")
            for key, name in (("RelativeAltitude", "rel_alt"), ("GimbalPitchDegree", "pitch"), ("GimbalYawDegree", "yaw"),
                              ("FlightYawDegree", "flight_yaw"), ("GpsLatitude", "xmp_lat"), ("GpsLongitude", "xmp_lon"),
                              ("GpsLongtitude", "xmp_lon")):
                m = re.search(r'drone-dji:' + key + r'\s*=\s*"([-+0-9.eE]+)"', xmp) or \
                    re.search(r'<drone-dji:' + key + r'>([-+0-9.eE]+)<', xmp)
                if m:
                    out[name] = float(m.group(1))
            if "lat" not in out and "xmp_lat" in out:
                out["lat"], out["lon"] = out["xmp_lat"], out.get("xmp_lon")
    except Exception:  # noqa: BLE001
        pass
    return out


def _gsd(meta) -> float | None:
    """Ground sampling distance (m/px) of a nadir photo: height x (36 mm / f35) / image width."""
    alt = meta.get("rel_alt")
    if not alt or alt <= 0:
        return None
    f35 = meta.get("focal35_mm")
    if not f35 and meta.get("focal_mm"):
        f35 = meta["focal_mm"] * 4.8        # typical 1" drone sensor crop factor (DJI Mavic / Phantom / M3E)
    if not f35:
        f35 = 24.0
    return float(alt * 36.0 / f35 / meta["width"])


# ------------------------------------------------------------------------------------------ colours
def _channels(rgb):
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1e-6
    return r, g, b, s, (2 * g - r - b) / s


def _hsv(rgb):
    from skimage.color import rgb2hsv
    return rgb2hsv(rgb)


def classify_pixels(rgb):
    """Per-pixel classes for vegetation state: green (healthy), yellow (stressed / drying),
    brown-red-grey (dead / dry), plus sky, water, road/bare."""
    r, g, b, s, exg = _channels(rgb)
    hsv = _hsv(rgb)
    h, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    sky = (b > r * 1.05) & (b > g * 0.95) & (val > 0.45) & (sat < 0.6)
    sky |= (val > 0.85) & (sat < 0.12)                            # overcast white sky
    water = (b / s > 0.37) & (val < 0.45) & (exg < 0.05) & ~sky
    green = (exg > 0.05) & (h > 65) & (h < 170) & (sat > 0.12) & ~sky
    yellow = (h >= 40) & (h <= 65) & (sat > 0.25) & (val > 0.25) & ~green & ~sky
    brown = (((h < 40) | (h > 330)) & (sat > 0.18) & (val > 0.12) & (val < 0.8)) & ~sky & ~water
    grey = (sat < 0.12) & (val > 0.25) & (val < 0.75) & ~sky & ~water
    dark = val < 0.12
    return dict(sky=sky, water=water, green=green, yellow=yellow, brown=brown, grey=grey, dark=dark, exg=exg, val=val)


# ------------------------------------------------------------------------------------------ from above
def analyse_aerial(rgb, gsd, place=None):
    """rgb: north-up photo; gsd m/px. place: dict(lat, lon) centre. Returns stats + crowns + overlay."""
    from skimage.segmentation import watershed
    H, W = rgb.shape[:2]
    # work at ~0.2 m (crowns) - photos are often 1-3 cm/px
    f = max(1, int(round(0.2 / max(gsd, 1e-3))))
    small = rgb[::f, ::f] if f > 1 else rgb
    res = gsd * f
    px = classify_pixels(small)
    veg = px["green"] | px["yellow"] | ((px["brown"] | px["grey"]) & ~px["water"])
    # canopy = vegetation that is textured / darker than open grass (grass is smooth and bright)
    val = px["val"]
    tex = np.sqrt(np.maximum(ndi.uniform_filter(val * val, 7) - ndi.uniform_filter(val, 7) ** 2, 0))
    smooth_bright = (ndi.uniform_filter(tex, 9) < 0.035) & (ndi.uniform_filter(val, 9) > 0.35)
    open_veg = veg & smooth_bright
    canopy = veg & ~open_veg
    canopy = ndi.binary_opening(canopy, iterations=1)
    # tree crowns: a crown top is the brightest point of its (sunlit) crown; window ~ 2.5 m
    sm = ndi.gaussian_filter(np.where(canopy, val + 0.5 * np.clip(px["exg"], 0, 1), 0), max(1.0, 0.5 / res))
    win = max(3, int(2.5 / res) | 1)
    peaks = (sm == ndi.maximum_filter(sm, win)) & canopy & (sm > 0.15)
    markers, n = ndi.label(peaks)
    crowns = []
    labels = np.zeros(canopy.shape, np.int32)
    if n:
        labels = watershed(-sm, markers, mask=canopy)
        idx = np.arange(1, n + 1)
        area = ndi.sum(np.ones_like(sm), labels, idx) * res * res
        cy, cx = np.array(ndi.center_of_mass(peaks, markers, idx)).T
        share = {k: ndi.mean(px[k].astype(float), labels, idx) for k in ("green", "yellow", "brown", "grey")}
        for i in range(n):
            if area[i] < 0.8:
                continue
            g, y, br, gr = share["green"][i], share["yellow"][i], share["brown"][i], share["grey"][i]
            dead_share = br + gr
            state = 2 if dead_share > 0.45 else 1 if (y + 0.5 * dead_share) > 0.3 else 0
            crowns.append(dict(px=float(cx[i] * f), py=float(cy[i] * f), area_m2=round(float(area[i]), 1),
                               diameter_m=round(float(2 * math.sqrt(area[i] / math.pi)), 1), state=state,
                               green=round(float(g), 2), dead=round(float(dead_share), 2)))
    # dry grass / bare: open, not green
    dry_grass = (open_veg & (px["yellow"] | px["brown"])) | (~veg & ~px["water"] & ~px["sky"] & (px["val"] > 0.3) & ~px["grey"])
    dry_grass &= small.max(axis=2) > 8
    vs = small.max(axis=2) > 8
    nv = max(int(vs.sum()), 1)
    cover = dict(canopy=float((canopy & vs).sum() / nv), open_vegetation=float((open_veg & vs).sum() / nv),
                 dry_grass_bare=float((dry_grass & vs).sum() / nv), water=float((px["water"] & vs).sum() / nv))
    # roads + power-line clearings via the map detectors on a 0.5 m north-up grid
    roads, corridors = [], []
    try:
        from .imagery import Ortho, _to_1m, classify, find_corridors_in_image, find_roads_in_image
        from .lidar import Grid
        w_m, h_m = W * gsd, H * gsd
        if min(w_m, h_m) >= 40:
            o = Ortho(rgb, 0.0, h_m, gsd, "photo")
            g05 = Grid(0.0, 0.0, 0.5, int(w_m / 0.5), int(h_m / 0.5))
            r05, _ = o.resample(g05)
            c05 = classify(r05)
            g1 = g05.coarsen(2)
            c1 = {k: _to_1m(v) for k, v in c05.items() if k != "exg"}
            corridors = find_corridors_in_image(c1, g1, min_len=min(150.0, 0.6 * max(w_m, h_m)))
            roads = find_roads_in_image(c1, g1, (), [c["line"] for c in corridors], min_len=min(50.0, 0.4 * max(w_m, h_m)))
            for item in roads + corridors:                     # metres (y up) -> photo pixels (y down)
                ln = np.asarray(item["line"], float)
                item["px"] = np.stack([ln[:, 0] / gsd, (h_m - ln[:, 1]) / gsd], 1).round(1).tolist()
    except Exception as exc:  # noqa: BLE001
        roads, corridors = [], []
        cover["detector_note"] = str(exc)
    # trees next to roads / clearings (a tree ~15-25 m tall reaches that far when it falls)
    near = []
    lines_px = [np.asarray(it["px"]) for it in roads + corridors]
    if lines_px and crowns:
        from .geo import polyline_distance
        cx = np.array([c["px"] for c in crowns])
        cyy = np.array([c["py"] for c in crowns])
        dmin = np.full(len(crowns), np.inf)
        kind = np.full(len(crowns), "", dtype=object)
        for it, ln in zip(roads + corridors, lines_px):
            d, _, _ = polyline_distance(cx, cyy, ln)
            d = d * gsd
            upd = d < dmin
            dmin[upd] = d[upd]
            kind[upd] = "power-line clearing" if it in corridors else "road"
        for i, c in enumerate(crowns):
            if dmin[i] <= 20:
                c["near"] = kind[i]
                c["dist_m"] = round(float(dmin[i]), 1)
                near.append(i)
    n_state = [sum(1 for c in crowns if c["state"] == k) for k in range(3)]
    valid = rgb.max(axis=2) > 8                              # ignore black corners of a rotated photo
    area_ha = float(valid.sum()) * gsd * gsd / 1e4
    danger = sorted([crowns[i] for i in near], key=lambda c: (-c["state"], c["dist_m"]))
    return dict(view="above", gsd_m=round(gsd, 4), size_m=[round(W * gsd, 1), round(H * gsd, 1)], area_ha=round(area_ha, 3),
                cover=cover, trees=dict(count=len(crowns), per_ha=round(len(crowns) / max(area_ha, 1e-6)),
                                        healthy=n_state[0], stressed=n_state[1], dead=n_state[2],
                                        mean_crown_d_m=round(float(np.mean([c["diameter_m"] for c in crowns])), 1) if crowns else None),
                crowns=crowns, roads=[_strip(r) for r in roads], corridors=[_strip(c) for c in corridors],
                danger=[dict(state=HEALTH[c["state"]], near=c["near"], dist_m=c["dist_m"], diameter_m=c["diameter_m"],
                             px=c["px"], py=c["py"]) for c in danger[:30]],
                masks=dict(dry=dry_grass, canopy=canopy, f=f))


def _strip(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items() if k != "line"}


# ------------------------------------------------------------------------------------------ from the ground
def analyse_ground(rgb):
    f = max(1, int(max(rgb.shape[:2]) / 900))
    small = rgb[::f, ::f]
    px = classify_pixels(small)
    H = small.shape[0]
    notsky = ~px["sky"]
    upper = np.zeros(notsky.shape, bool)
    upper[: int(H * 0.7)] = True                                # trees / crowns; the lower 30 % is the ground layer
    fol = (px["green"] | px["yellow"] | px["brown"] | px["grey"]) & notsky & upper
    n = max(int(fol.sum()), 1)
    green = float((px["green"] & fol).sum() / n)
    yellow = float((px["yellow"] & fol).sum() / n)
    deadf = float(((px["brown"] | px["grey"]) & fol).sum() / n)
    gl = ~upper & notsky
    ng = max(int(gl.sum()), 1)
    dry_ground = float(((px["yellow"] | px["brown"]) & gl).sum() / ng)
    state = "dead / dry" if deadf > 0.35 else "stressed" if (yellow + 0.5 * deadf) > 0.3 else "mostly healthy"
    return dict(view="ground", sky_share=round(float(px["sky"].mean()), 3),
                foliage=dict(green=round(green, 3), yellowing=round(yellow, 3), brown_grey=round(deadf, 3)),
                dry_ground_share=round(dry_ground, 3), overall=state,
                masks=dict(dead=(px["brown"] | px["grey"]) & fol, yellow=px["yellow"] & fol, dry=(px["yellow"] | px["brown"]) & gl,
                           f=f))


# ------------------------------------------------------------------------------------------ overlay
def overlay_png(rgb, res, max_px=1600) -> bytes:
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb, "RGB")
    sc = min(1.0, max_px / max(im.size))
    if sc < 1:
        im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)))
    base = np.asarray(im).astype(np.float32) * 0.8
    m = res["masks"]

    def tint(mask, color, a):
        mm = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(im.size, Image.NEAREST)) > 0
        base[mm] = base[mm] * (1 - a) + np.array(color, np.float32) * a
    if res["view"] == "above":
        tint(m["dry"], (255, 210, 60), 0.45)
    else:
        tint(m["dry"], (255, 210, 60), 0.4)
        tint(m["yellow"], (255, 170, 30), 0.45)
        tint(m["dead"], (230, 40, 30), 0.55)
    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    if res["view"] == "above":
        col = {0: (60, 230, 110), 1: (255, 190, 40), 2: (255, 50, 40)}
        for it, c in [(r, (255, 150, 40)) for r in res["roads"]] + [(c, (80, 210, 255)) for c in res["corridors"]]:
            pts = [(x * sc, y * sc) for x, y in it["px"]]
            if len(pts) > 1:
                d.line(pts, fill=c, width=max(3, int(5 * sc * 2)))
        rr = max(2, int(3 * sc * 2))
        for c in res["crowns"]:
            x, y = c["px"] * sc, c["py"] * sc
            r = max(rr, c["diameter_m"] / res["gsd_m"] * sc / 2 * 0.8)
            w = 3 if c.get("near") else 2
            d.ellipse([x - r, y - r, x + r, y + r], outline=col[c["state"]], width=w)
    b = io.BytesIO()
    small = np.asarray(im)
    empty = small.max(axis=2) <= 8                         # black corners of a north-up rotated photo
    if empty.mean() > 0.01:
        rgba = np.dstack([np.asarray(out), np.where(empty, 0, 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(b, "PNG", optimize=True)
    else:
        out.save(b, "JPEG", quality=85)
    return b.getvalue()


# ------------------------------------------------------------------------------------------ entry point
def analyse_photo(path, view="auto", lat=None, lon=None, gsd=None, anchor=(62.747066, 27.259548)):
    """Returns (result dict for the API, overlay JPEG bytes, north-up photo JPEG bytes)."""
    from PIL import Image, ImageOps
    path = Path(path)
    meta = read_meta(path)
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rgb = np.asarray(im)
    placement, bounds = None, None
    world = None
    try:
        from .imagery import load_image
        world = load_image(path)                         # world file / GeoTIFF in ETRS-TM35FIN
    except Exception:  # noqa: BLE001
        world = None
    pitch = meta.get("pitch")
    if view == "auto":
        if world is not None or (pitch is not None and pitch < -60):
            view = "above"
        elif pitch is not None and pitch > -45:
            view = "ground"
        else:
            px = classify_pixels(rgb[:: max(1, rgb.shape[0] // 300), :: max(1, rgb.shape[1] // 300)])
            top = px["sky"][: px["sky"].shape[0] // 3]
            view = "ground" if top.mean() > 0.12 else "above"
    if view == "ground":
        res = analyse_ground(rgb)
        placement = "ground photo: no map position" + (f" (taken at {meta['lat']:.5f} N, {meta['lon']:.5f} E)" if meta.get("lat") else "")
        if meta.get("lat"):
            res["taken_at"] = [meta["lon"], meta["lat"]]
    else:
        g = None
        if world is not None:
            from .geo import TM35FIN
            g = world.res
            rgb = world.rgb
            x0, y0, x1, y1 = world.bounds_xy
            tm = TM35FIN()
            lo0, la0 = tm.to_lonlat(x0, y0)
            lo1, la1 = tm.to_lonlat(x1, y1)
            bounds = [[float(la0), float(lo0)], [float(la1), float(lo1)]]
            placement = f"georeferenced ({world.source})"
        else:
            g = gsd or _gsd(meta)
            yaw = meta.get("yaw", meta.get("flight_yaw"))
            if yaw is not None and abs(yaw) > 0.5:      # rotate so north is up (camera top edge pointed at yaw)
                im2 = Image.fromarray(rgb).rotate(-yaw, expand=True, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                rgb = np.asarray(im2)
            clat = lat if lat not in (None, "") else meta.get("lat")
            clon = lon if lon not in (None, "") else meta.get("lon")
            if g is None:
                g = 0.05
                scale_note = "scale unknown - assumed 5 cm per pixel (enter the flight height or m/pixel for real sizes)"
            else:
                scale_note = f"scale {g * 100:.1f} cm/pixel" + (f" from flight height {meta['rel_alt']:.0f} m" if meta.get("rel_alt") and not gsd else "")
            if clat is None:
                clat, clon = anchor
                placement = "no GPS in the photo - shown at the Kuopio demo point; " + scale_note
            else:
                placement = (f"GPS {float(clat):.5f} N, {float(clon):.5f} E" + (" (drone photo)" if meta.get("rel_alt") else "")
                             + "; " + scale_note)
            h_m, w_m = rgb.shape[0] * g, rgb.shape[1] * g
            dlat = h_m / 2 / 111_320.0
            dlon = w_m / 2 / (111_320.0 * math.cos(math.radians(float(clat))))
            bounds = [[float(clat) - dlat, float(clon) - dlon], [float(clat) + dlat, float(clon) + dlon]]
        res = analyse_aerial(rgb, g)
    ov = overlay_png(rgb, res)
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG", quality=85)
    res.pop("masks", None)
    res.update(name=path.name, placement=placement, bounds=bounds, size_px=[int(rgb.shape[1]), int(rgb.shape[0])],
               camera=" ".join(x for x in (meta.get("make"), meta.get("model")) if x) or None,
               flight_height_m=meta.get("rel_alt"), gimbal_pitch=meta.get("pitch"))
    if res["view"] == "above":
        for c in res["crowns"]:                        # crown list is large; keep positions compact
            c["px"], c["py"] = round(c["px"]), round(c["py"])
    return res, ov, b.getvalue()
