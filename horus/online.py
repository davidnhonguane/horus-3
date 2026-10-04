"""Live / open data connectors for Horus.

Every connector returns ``{"ok": bool, "source": str, "attribution": str, ...}`` and never
raises - a failing source must not break the demo. Responses are cached on disk
(``data/cache``) so a site that was fetched once keeps working offline; pre-warm with
``python -m horus fetch``.

Sources (all usable around Kuopio):

=====================  ===========================================  ===========================
connector              what                                         key
=====================  ===========================================  ===========================
osm_context            roads, power lines, buildings, fire          none (Overpass API, ODbL)
                       stations, water, wetlands (OpenStreetMap)
sentinel2_scenes       latest low-cloud Sentinel-2 L2A scenes        none (Microsoft Planetary
sentinel2_change       NDVI + year-over-year canopy change           Computer STAC + tiler)
nls_*                  National Land Survey of Finland: laser       NLS_API_KEY (free)
                       scanning point clouds (LAZ), 2 m DEM,
                       orthophoto / topographic map tiles
metsakeskus_stands     Finnish Forest Centre forest stands          none (CC BY 4.0)
fmi_observations       nearest FMI weather station                   none (CC BY 4.0)
fmi_warnings           FMI CAP warnings (wind, forest-fire weather)  none
commons_photos         geotagged photos (Wikimedia Commons)          none
=====================  ===========================================  ===========================
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

UA = ("Horus/1.2 (Forey Forest Intelligence Challenge forest-risk demo; python-urllib"
      + (f"; contact: {os.environ['HORUS_CONTACT']}" if os.environ.get("HORUS_CONTACT") else "") + ")")
CACHE = Path(os.environ.get("HORUS_CACHE", Path(__file__).resolve().parent.parent / "data" / "cache"))
FIXTURES = os.environ.get("HORUS_ONLINE_FIXTURES")  # tests only: folder of canned responses

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_TILES = "https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{z}/{x}/{y}@1x.png"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
NLS_PROC = "https://avoin-paikkatieto.maanmittauslaitos.fi/tiedostopalvelu/ogcproc/v1"
NLS_WMTS = "https://avoin-karttakuva.maanmittauslaitos.fi/avoin/wmts/1.0.0/{layer}/default/WGS84_Pseudo-Mercator/{z}/{y}/{x}.{fmt}"
NLS_FEATURES = "https://avoin-paikkatieto.maanmittauslaitos.fi/maastotiedot/features/v1/collections/{coll}/items"
METSAKESKUS_WFS = "https://avoin.metsakeskus.fi/rajapinnat/v1/stand/ows"
FMI_WFS = "https://opendata.fmi.fi/wfs"
FMI_CAP = "https://alerts.fmi.fi/cap/feed/atom_en-GB.xml"
COMMONS = "https://commons.wikimedia.org/w/api.php"


# ------------------------------------------------------------------------------- http + cache
def _key(url, body=None):
    h = hashlib.sha1((url + "|" + (body.decode() if isinstance(body, bytes) else str(body or ""))).encode()).hexdigest()
    return h[:24]


def http(url, body=None, headers=None, timeout=25, ttl=6 * 3600, binary=False, cache=True):
    """GET/POST with disk cache. Secrets (api-key) are stripped from the cache key file name."""
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
    k = _key(url.split("api-key=")[0], body)
    if FIXTURES:
        return _fixture(url, body)
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / k
    if cache and f.exists() and time.time() - f.stat().st_mtime < ttl:
        return f.read_bytes()
    h = {"User-Agent": UA, "Accept-Encoding": "identity"}
    if "maanmittauslaitos.fi" in url and "api-key=" in url:
        # NLS also accepts the key as HTTP Basic user name (some proxies drop query keys)
        import base64
        k_ = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("api-key", [""])[0]
        h["Authorization"] = "Basic " + base64.b64encode(f"{k_}:".encode()).decode()
    if body is not None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method="POST" if body is not None else "GET")
    data = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503, 504) and attempt < 2:
                wait = exc.headers.get("Retry-After") if exc.headers else None
                time.sleep(min(float(wait) if wait and str(wait).isdigit() else 3 * (attempt + 1), 20))
                continue
            if cache and f.exists():  # stale cache beats nothing
                return f.read_bytes()
            raise
        except Exception:
            if cache and f.exists():
                return f.read_bytes()
            raise
    if cache:
        f.write_bytes(data)
    return data


def _fixture(url, body):
    """Test mode: answer from canned responses in the services' real formats (tests/fixtures)."""
    F = Path(FIXTURES)
    if "overpass" in url:
        return (F / "overpass.json").read_bytes()
    if "stac/v1/search" in url:
        b = json.loads(body) if body else {}
        year = int(b.get("datetime", "2026")[:4])
        js = json.loads((F / "stac.json").read_text())
        tag = "PRE" if year < dt.date.today().year - (0 if dt.date.today().month >= 7 else 1) else "POST"
        for f in js["features"]:
            f["id"] = f["id"].replace("{TAG}", tag)
            f["properties"]["datetime"] = f"{year}" + f["properties"]["datetime"][4:]
        return json.dumps(js).encode()
    if "/item/tiles/" in url:
        return (F / ("tile_pre.png" if "PRE" in url else "tile_post.png")).read_bytes()
    if "opendata.fmi.fi" in url:
        return (F / "fmi.xml").read_bytes()
    if "alerts.fmi.fi" in url:
        return (F / "cap.xml").read_bytes()
    if "maastotiedot" in url:
        coll = url.split("/collections/")[1].split("/")[0]
        f = F / f"nls_{coll}.json"
        if f.exists():
            return f.read_bytes()
        return json.dumps({"type": "FeatureCollection", "features": []}).encode()
    if "metsakeskus" in url:
        return (F / "stands.json").read_bytes()
    if "commons.wikimedia" in url:
        return (F / "commons.json").read_bytes()
    raise urllib.error.URLError(f"no fixture for {url[:90]}")


def jget(url, body=None, **kw):
    return json.loads(http(url, body, **kw).decode("utf-8"))


def _fail(source, exc, attribution=""):
    return {"ok": False, "source": source, "attribution": attribution,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def bbox_around(lat, lon, radius_m):
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]  # minlon, minlat, maxlon, maxlat


# ------------------------------------------------------------------------------- OpenStreetMap
MAIN_HW = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link",
           "primary_link", "secondary_link", "tertiary_link"}
LOCAL_HW = {"unclassified", "residential", "service", "living_street", "road"}
TRACK_HW = {"track"}


def osm_context(bbox, station_radius_km=30):
    """Infrastructure inside bbox (+ fire stations within station_radius_km of its centre)."""
    s, w, n, e = bbox[1], bbox[0], bbox[3], bbox[2]
    clat, clon = (s + n) / 2, (w + e) / 2
    sb = bbox_around(clat, clon, station_radius_km * 1000)
    b = f"({s},{w},{n},{e})"
    sbx = f"({sb[1]},{sb[0]},{sb[3]},{sb[2]})"
    # several small queries instead of one big one: a busy Overpass server times out on the big one
    parts = {
        "roads": f'way["highway"]{b};',
        "power": f'way["power"~"^(line|minor_line|cable)$"]{b};nwr["power"="substation"]{b};'
                 f'nwr["man_made"~"^(mast|tower|water_tower|telecom)$"]{b};',
        "buildings": f'way["building"]{b};',
        "nature": f'way["natural"~"^(water|wetland)$"]{b};',
        "stations": f'nwr["amenity"="fire_station"]{sbx};',
    }
    elements, failed, used = [], [], set()
    for name, body in parts.items():
        q = f"[out:json][timeout:90];({body});out geom;"
        ok = False
        last = None
        for ep in OVERPASS:
            try:
                js = jget(ep, ("data=" + urllib.parse.quote(q)).encode(),
                          headers={"Content-Type": "application/x-www-form-urlencoded"}, ttl=7 * 86400, timeout=100)
                if js.get("remark", "").lower().startswith("runtime error"):
                    raise RuntimeError(js["remark"][:120])
                elements += js.get("elements", [])
                used.add(ep)
                ok = True
                break
            except Exception as exc:  # next mirror
                last = exc
        if not ok:
            failed.append(f"{name}: {type(last).__name__}")
    if not elements and failed:
        return _fail("OpenStreetMap (Overpass)", RuntimeError("; ".join(failed)), "© OpenStreetMap contributors, ODbL")
    seen, uniq = set(), []
    for el in elements:  # the same feature can come back from two queries
        k_ = (el.get("type"), el.get("id"))
        if k_ not in seen:
            seen.add(k_)
            uniq.append(el)
    out = parse_overpass({"elements": uniq}, clat, clon) | {"endpoint": ", ".join(sorted(used))}
    if failed:
        out["partial"] = failed
    return out


def parse_overpass(js, clat, clon):
    roads, power, buildings, stations, water, wetland = [], [], [], [], [], []
    for el in js.get("elements", []):
        t = el.get("tags", {})
        geom = el.get("geometry")
        if el["type"] == "node":
            if t.get("amenity") == "fire_station":
                stations.append(dict(name=t.get("name", "Fire station"), lon=el["lon"], lat=el["lat"]))
            elif t.get("power") == "substation" or t.get("man_made") in ("mast", "tower", "water_tower", "telecom"):
                kind = "substation" if t.get("power") == "substation" else "mast"
                buildings.append(dict(name=t.get("name") or kind.replace("_", " "), kind=kind, lon=el["lon"], lat=el["lat"]))
            continue
        if not ("highway" in t or "power" in t and t.get("power") != "substation") and \
                (t.get("power") == "substation" or t.get("man_made") in ("mast", "tower", "water_tower", "telecom")) and geom:
            kind = "substation" if t.get("power") == "substation" else "mast"
            buildings.append(dict(name=t.get("name") or kind, kind=kind, lon=float(np.mean([g["lon"] for g in geom])),
                                  lat=float(np.mean([g["lat"] for g in geom]))))
            continue
        if not geom:
            continue
        path = [[g["lon"], g["lat"]] for g in geom]
        if "highway" in t:
            hw = t["highway"]
            if hw in MAIN_HW:
                cls, width = "main", 7.0
            elif hw in LOCAL_HW:
                cls, width = "forest", 5.0
            elif hw in TRACK_HW:
                cls, width = "forest", 3.5
            else:
                cls, width = "path", 1.5
            roads.append(dict(name=t.get("name") or t.get("ref") or hw, cls=cls, highway=hw, width=width,
                              surface=t.get("surface"), path=path))
        elif "power" in t:
            v = t.get("voltage", "")
            try:
                kv = max(int(x) for x in v.split(";") if x.strip().isdigit()) / 1000.0
            except ValueError:
                kv = 20.0 if t["power"] == "minor_line" else None
            power.append(dict(name=t.get("name") or f"{t['power'].replace('_', ' ')}{f' {kv:g} kV' if kv else ''}",
                              kind=t["power"], kv=kv, path=path))
        elif t.get("amenity") == "fire_station":
            lo = float(np.mean([p[0] for p in path]))
            la = float(np.mean([p[1] for p in path]))
            stations.append(dict(name=t.get("name", "Fire station"), lon=lo, lat=la))
        elif "building" in t:
            lo = float(np.mean([p[0] for p in path]))
            la = float(np.mean([p[1] for p in path]))
            kind = t.get("building")
            am = t.get("amenity")
            if am in ("nursing_home", "social_facility") or kind == "nursing_home":
                kind = "care_home"
            elif am in ("hospital", "clinic") or kind == "hospital":
                kind = "hospital"
            elif am in ("school", "kindergarten") or kind in ("school", "kindergarten"):
                kind = "school"
            elif t.get("power") == "substation":
                kind = "substation"
            buildings.append(dict(name=t.get("name") or t.get("addr:street", "") + " " + t.get("addr:housenumber", ""),
                                  kind=kind, lon=lo, lat=la, path=path))
        elif t.get("natural") == "water":
            water.append(dict(name=t.get("name", "water"), path=path))
        elif t.get("natural") == "wetland":
            wetland.append(dict(name=t.get("name", t.get("wetland", "wetland")), path=path))
    for s_ in stations:
        s_["distance_km"] = round(float(_hav(clat, clon, s_["lat"], s_["lon"])), 1)
    stations.sort(key=lambda s_: s_["distance_km"])
    return {"ok": True, "source": "OpenStreetMap (Overpass API)", "attribution": "© OpenStreetMap contributors, ODbL",
            "roads": roads, "power": power, "buildings": buildings, "fire_stations": stations[:5],
            "water": water, "wetland": wetland,
            "counts": dict(roads=len(roads), power_lines=len(power), buildings=len(buildings),
                           water=len(water), wetlands=len(wetland), fire_stations=len(stations))}


def _hav(lat1, lon1, lat2, lon2):
    from .geo import haversine_km
    return haversine_km(lat1, lon1, lat2, lon2)


def osm_to_context(osm: dict, frame):
    """Convert OSM result to the Site context format (projected coordinates)."""
    def xy(path):
        a = np.array(path, float)
        x, y = frame.from_lonlat(a[:, 0], a[:, 1])
        return np.stack([x, y], 1)
    roads = [dict(name=r["name"], cls=r["cls"], width=r["width"], line=xy(r["path"]))
             for r in osm.get("roads", []) if r["cls"] != "path" and len(r["path"]) >= 2]
    pls = [dict(name=p["name"], kv=p.get("kv") or 20, conductor_h=9.0 if (p.get("kv") or 20) <= 45 else 15.0,
                row_half=5.0 if (p.get("kv") or 20) <= 45 else 13.0, line=xy(p["path"]))
           for p in osm.get("power", []) if p["kind"] != "cable" and len(p["path"]) >= 2]
    blds = []
    for b in osm.get("buildings", []):
        x, y = frame.from_lonlat(b["lon"], b["lat"])
        k = b["kind"] if b["kind"] in SENSITIVITY else "house"
        blds.append(dict(name=b["name"].strip() or "Building", kind=k, x=float(x), y=float(y),
                         r=8.0 if k in ("substation", "school", "hospital", "care_home") else 6.0, weight=SENSITIVITY[k]))
    depot = None
    if osm.get("fire_stations"):
        st = osm["fire_stations"][0]
        x, y = frame.from_lonlat(st["lon"], st["lat"])
        depot = (float(x), float(y))
    bog = [xy(w["path"]) for w in osm.get("wetland", []) if len(w["path"]) >= 3]
    return dict(roads=roads, powerlines_vector=pls, buildings=blds, depot=depot, bog_polys=bog,
                source="OpenStreetMap (live, ODbL)")


# how much it matters if a structure is hit / burned / cut off (people at risk, or service to many people)
SENSITIVITY = {"hospital": 3.0, "care_home": 3.0, "school": 2.5, "substation": 2.5, "mast": 1.5,
               "house": 1.0, "cabin": 0.8}


# ------------------------------------------------------------------------------- Sentinel-2
def growing_season_windows(today: dt.date | None = None):
    today = today or dt.date.today()
    y = today.year if today >= dt.date(today.year, 7, 1) else today.year - 1
    post = (f"{y}-06-01", min(f"{y}-09-15", today.isoformat()))
    pre = (f"{y - 1}-06-01", f"{y - 1}-09-15")
    return pre, post


def sentinel2_scenes(bbox, start, end, max_cloud=25, limit=12):
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": max_cloud}}, "limit": limit,
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]}
    try:
        js = jget(PC_STAC, body, ttl=86400)
    except Exception as exc:
        return _fail("Sentinel-2 (Microsoft Planetary Computer)", exc, "Contains modified Copernicus Sentinel data")
    scenes = []
    for f in js.get("features", []):
        ib = f.get("bbox") or [-180, -90, 180, 90]
        covers = ib[0] <= bbox[0] and ib[1] <= bbox[1] and ib[2] >= bbox[2] and ib[3] >= bbox[3]
        scenes.append(dict(id=f["id"], date=f["properties"]["datetime"][:10],
                           cloud=round(f["properties"].get("eo:cloud_cover", 0), 1), covers=covers))
    scenes.sort(key=lambda s: (not s["covers"], s["cloud"], s["date"]))
    return {"ok": bool(scenes), "source": "Sentinel-2 L2A via Microsoft Planetary Computer",
            "attribution": "Contains modified Copernicus Sentinel data", "scenes": scenes,
            **({} if scenes else {"error": "no low-cloud scene in the window"})}


def s2_tile_templates(item_id):
    base = PC_TILES + "?collection=sentinel-2-l2a&item=" + urllib.parse.quote(item_id)
    tc = base + "&assets=B04&assets=B03&assets=B02&nodata=0&color_formula=" + urllib.parse.quote(
        "Gamma RGB 3.2 Saturation 0.8 Sigmoidal RGB 25 0.35")
    ndvi = base + "&expression=" + urllib.parse.quote("(B08-B04)/(B08+B04)") + \
        "&asset_as_band=True&rescale=-0.1,0.9&colormap_name=rdylgn&nodata=0"
    return {"truecolor": tc, "ndvi": ndvi}


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def s2_ndvi_grid(item_id, bbox, z=14):
    """Numeric NDVI mosaic for bbox from the tiler (grayscale PNG, rescale -1..1). Returns
    (ndvi array [rows north->south], lon edges, lat edges)."""
    from PIL import Image

    x0, y0 = _tile_xy(bbox[0], bbox[3], z)
    x1, y1 = _tile_xy(bbox[2], bbox[1], z)
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    W = (tx1 - tx0 + 1) * 256
    H = (ty1 - ty0 + 1) * 256
    out = np.full((H, W), np.nan, np.float32)
    base = PC_TILES + "?collection=sentinel-2-l2a&item=" + urllib.parse.quote(item_id) + \
        "&expression=" + urllib.parse.quote("(B08-B04)/(B08+B04)") + "&asset_as_band=True&rescale=-1,1&nodata=0"
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            png = http(base.format(z=z, x=tx, y=ty), ttl=30 * 86400, binary=True)
            im = Image.open(io.BytesIO(png))
            a = np.asarray(im.convert("LA"), dtype=np.float32)
            v = a[..., 0] / 255.0 * 2 - 1
            v[a[..., 1] == 0] = np.nan
            out[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = v
    # crop to bbox
    px0, py0 = int((x0 - tx0) * 256), int((y0 - ty0) * 256)
    px1, py1 = int((x1 - tx0) * 256), int((y1 - ty0) * 256)
    return out[py0:py1, px0:px1]


def sentinel2_change(bbox, pre=None, post=None, max_cloud=25):
    """Year-over-year canopy change: NDVI(post) - NDVI(pre) on the best scene of each window."""
    if pre is None or post is None:
        pre, post = growing_season_windows()
    a = sentinel2_scenes(bbox, *pre, max_cloud=max_cloud)
    b = sentinel2_scenes(bbox, *post, max_cloud=max_cloud)
    if not a["ok"] or not b["ok"]:
        return _fail("Sentinel-2 change", RuntimeError((a.get("error") or "") + " " + (b.get("error") or "")),
                     "Contains modified Copernicus Sentinel data")
    sa, sb = a["scenes"][0], b["scenes"][0]
    try:
        n0 = s2_ndvi_grid(sa["id"], bbox)
        n1 = s2_ndvi_grid(sb["id"], bbox)
    except Exception as exc:
        return _fail("Sentinel-2 change", exc, "Contains modified Copernicus Sentinel data")
    h = min(n0.shape[0], n1.shape[0])
    w = min(n0.shape[1], n1.shape[1])
    n0, n1 = n0[:h, :w], n1[:h, :w]
    d = n1 - n0
    forest = n0 > 0.55
    loss = forest & (d < -0.15)
    gain = (n0 < 0.4) & (d > 0.15)
    lat_c = (bbox[1] + bbox[3]) / 2
    px_m = 40075016.7 * math.cos(math.radians(lat_c)) / (2 ** 14 * 256)
    px_ha = px_m * px_m / 1e4
    return {"ok": True, "source": "Sentinel-2 L2A NDVI change (Planetary Computer)",
            "attribution": "Contains modified Copernicus Sentinel data", "pre": sa, "post": sb,
            "pre_window": pre, "post_window": post, "pixel_m": round(px_m, 1),
            "forest_ha": float(forest.sum() * px_ha), "canopy_loss_ha": float(loss.sum() * px_ha),
            "regrowth_ha": float(gain.sum() * px_ha), "mean_ndvi_post": float(np.nanmean(n1)),
            "_dndvi": d, "_ndvi_post": n1, "_bbox": bbox}


def change_png(res):
    from .render import to_png
    d = res["_dndvi"]
    rgba = np.zeros(d.shape + (4,), np.uint8)
    loss = d < -0.15
    gain = d > 0.15
    t = np.clip((-d - 0.15) / 0.4, 0, 1)
    rgba[loss] = np.stack([235 + 0 * t[loss], 60 - 40 * t[loss], 40 + 0 * t[loss], 120 + 120 * t[loss]], 1).astype(np.uint8)
    rgba[gain] = (80, 200, 255, 150)
    return to_png(np.flipud(rgba))  # to_png flips again -> rows stay north-up


# ------------------------------------------------------------------------------- NLS (Maanmittauslaitos)
NLS_KEY_FILES = ("nls_key.txt", ".nls_key")
NLS_LASER_CANDIDATES = ("laserkeilausaineisto_5p_bbox", "laserkeilausaineisto_5p_karttalehti", "laserkeilausaineisto_5p",
                        "laserkeilausaineisto_05_bbox", "laserkeilausaineisto_05_karttalehti")
NLS_DEM_CANDIDATES = ("korkeusmalli_2m_bbox",)


def nls_key():
    """NLS API key from env NLS_API_KEY, or a one-line file nls_key.txt / .nls_key in the project folder."""
    k = os.environ.get("NLS_API_KEY")
    if k:
        return k.strip()
    root = Path(__file__).resolve().parent.parent
    for n in NLS_KEY_FILES:
        f = root / n
        if f.exists():
            k = f.read_text().strip()
            if k:
                return k
    return None


def nls_tile_url(layer, z, x, y):
    fmt = "jpg" if layer.startswith("ortokuva") else "png"
    return NLS_WMTS.format(layer=layer, z=z, x=x, y=y, fmt=fmt) + f"?api-key={nls_key()}"


def nls_processes():
    """Available NLS processes. The full list may need extra rights (it returned 401 for a normal open-data
    key), so known process ids are also probed one by one."""
    k = nls_key()
    if not k:
        return {"ok": False, "source": "NLS file service", "error": "set NLS_API_KEY or create nls_key.txt (free key at maanmittauslaitos.fi)"}
    ids, descs = [], {}
    try:
        js = jget(f"{NLS_PROC}/processes?api-key={k}", ttl=7 * 86400)
        procs = js.get("processes", js if isinstance(js, list) else [])
        ids = [p.get("id") for p in procs if p.get("id")]
    except Exception:
        pass
    for pid in NLS_LASER_CANDIDATES + NLS_DEM_CANDIDATES:
        if pid in ids:
            continue
        try:
            descs[pid] = jget(f"{NLS_PROC}/processes/{pid}?api-key={k}", ttl=7 * 86400)
            ids.append(pid)
        except Exception:
            continue
    if not ids:
        return {"ok": False, "source": "NLS file service", "error": "key rejected or service unreachable "
                "(new keys can take a while to activate)"}
    return {"ok": True, "source": "NLS OGC API Processes", "processes": ids,
            "laser": [i for i in ids if "laser" in i], "dem": [i for i in ids if "korkeusmalli" in i]}


def nls_execute(process_id, inputs, out_dir: Path, poll_s=4, max_wait_s=900):
    """Run an NLS file-service process and download every result file into out_dir."""
    k = nls_key()
    if not k:
        raise RuntimeError("NLS_API_KEY not set")
    url = f"{NLS_PROC}/processes/{process_id}/execution?api-key={k}"
    req = urllib.request.Request(url, data=json.dumps({"id": process_id, "inputs": inputs}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA,
                                          "Authorization": "Basic " + __import__("base64").b64encode(f"{k}:".encode()).decode()},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        loc = r.headers.get("Location")
        body = r.read()
    js = json.loads(body or b"{}")
    job = js.get("jobID") or js.get("jobId") or (loc.rstrip("/").split("/")[-1].split("?")[0] if loc else None)
    if not job:
        raise RuntimeError(f"no job id in response: {str(js)[:200]}")
    t0 = time.time()
    while True:
        st = jget(f"{NLS_PROC}/jobs/{job}?api-key={k}", cache=False)
        status = st.get("status")
        if status == "successful":
            break
        if status in ("failed", "dismissed"):
            raise RuntimeError(f"NLS job {status}: {st.get('message')}")
        if time.time() - t0 > max_wait_s:
            raise TimeoutError("NLS job did not finish in time")
        time.sleep(poll_s)
    res = jget(f"{NLS_PROC}/jobs/{job}/results?api-key={k}", cache=False)
    links = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            t = o.strip()
            if t.startswith(("{", "[")):  # results are sometimes a JSON document inside a string
                try:
                    walk(json.loads(t))
                    return
                except ValueError:
                    pass
            if t.startswith("http") and "/jobs/" not in t and "/processes" not in t:
                links.append(t)
    walk(res)
    links = list(dict.fromkeys(links))
    if not links:
        raise RuntimeError(f"no download links in job result: {str(res)[:300]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for href in links:
        u = href if "api-key=" in href else href + ("&" if "?" in href else "?") + f"api-key={k}"
        name = urllib.parse.urlparse(href).path.rsplit("/", 1)[-1] or "result.bin"
        p = out_dir / name
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA, "Authorization": "Basic " + __import__("base64").b64encode(f"{k}:".encode()).decode()}), timeout=600) as r, open(p, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                z.extractall(out_dir)
                files += [out_dir / n for n in z.namelist() if not n.endswith("/")]
        else:
            files.append(p)
    # name point clouds by content: NLS links do not always end in .laz
    fixed = []
    for f in files:
        f = Path(f)
        try:
            with open(f, "rb") as fh:
                head = fh.read(4)
        except OSError:
            continue
        if head == b"LASF" and f.suffix.lower() not in (".las", ".laz"):
            # compressed LAZ has the 'laszip' VLR; laspy decides, so .laz is the safe name
            nf = f.with_suffix(".laz")
            f.rename(nf)
            f = nf
        fixed.append(f)
    return fixed


def nls_laser(lat, lon, out_dir: Path, map_sheet: str | None = None, radius_m=500):
    """Download NLS laser scanning (prefers 5 p) around a point. Uses bounding-box input when the
    process supports it, otherwise the given map sheet code (find it in MapSite)."""
    from .geo import TM35FIN

    if FIXTURES:  # tests: synthesise an NLS-quality LAS (5 p, intensity only) at the requested place
        return _fixture_laser(lat, lon, out_dir)
    pr = nls_processes()
    if not pr["ok"]:
        return pr
    cands = sorted(pr["laser"], key=lambda i: (("5p" not in i and "_5" not in i), "bbox" not in i))
    if not cands:
        return {"ok": False, "source": "NLS laser", "error": f"no laser process among {pr['processes']}"}
    tm = TM35FIN()
    E, N = tm.from_lonlat(lon, lat)
    last = RuntimeError("no usable laser process")
    for pid in cands:
        try:
            desc = jget(f"{NLS_PROC}/processes/{pid}?api-key={nls_key()}", ttl=7 * 86400)
            names = set((desc.get("inputs") or {}).keys())
            inputs = {}
            if "fileFormatInput" in names:
                allowed = ((desc["inputs"]["fileFormatInput"].get("schema") or {}).get("enum")) or ["LAZ"]
                inputs["fileFormatInput"] = "LAZ" if "LAZ" in allowed else allowed[0]
            if "boundingBoxInput" in names:
                inputs["boundingBoxInput"] = [float(E - radius_m), float(N - radius_m), float(E + radius_m), float(N + radius_m)]
            elif "mapSheetInput" in names and map_sheet:
                inputs["mapSheetInput"] = [map_sheet]
            else:
                sheet_hint = json.dumps((desc.get("inputs") or {}).get("mapSheetInput", {}))[:300]
                last = RuntimeError(f"{pid} needs a map sheet code (--nls-sheet). Input spec: {sheet_hint}")
                continue
            files = nls_execute(pid, inputs, out_dir)
            laz = [str(f) for f in files if str(f).lower().endswith((".laz", ".las"))]
            return {"ok": bool(laz), "source": f"NLS laser scanning ({pid})", "files": laz,
                    "attribution": "© Maanmittauslaitos, laser scanning data, CC BY 4.0"}
        except Exception as exc:  # try next candidate
            last = exc
    return {"ok": False, "source": "NLS laser", "error": str(last) + " - find the sheet code on MapSite "
            "(asiointi.maanmittauslaitos.fi/karttapaikka/tiedostopalvelu > Laser scanning data 5 p)"}


def _fixture_laser(lat, lon, out_dir: Path):
    from .geo import LocalFrame, TM35FIN
    from .lasio import write_las
    from .synth import generate

    s = generate(lat0=lat, lon0=lon)
    rng = np.random.default_rng(5)
    P = s.points
    keep = rng.random(P["x"].shape[0]) < 0.75
    P = {k: v[keep] for k, v in P.items() if k not in ("red", "green", "blue", "nir", "truth_tree")}
    P["classification"] = np.where(np.isin(P["classification"], (3, 5, 6, 14)), 1, P["classification"]).astype(np.uint8)
    lo, la = LocalFrame(lat, lon, s.size / 2, s.size / 2).to_lonlat(P["x"], P["y"])
    P["x"], P["y"] = TM35FIN().from_lonlat(lo, la)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "FIXTURE_nls_5p.las"
    write_las(f, P)
    return {"ok": True, "source": "TEST FIXTURE (synthetic NLS-quality scan)", "files": [str(f)],
            "attribution": "test fixture - not real data"}


# ------------------------------------------------------------------------------- NLS topographic database
# Maastotietokanta feature classes (kohdeluokka) for power lines
NLS_POWER_CLASSES = {22311: ("high-voltage line", 110.0), 22312: ("distribution line", 20.0)}


def nls_features(collection, bbox, max_features=5000):
    """OGC API Features items (GeoJSON, lon/lat) from the NLS topographic database, following 'next' links."""
    k = nls_key()
    if not k:
        raise RuntimeError("NLS_API_KEY not set")
    url = NLS_FEATURES.format(coll=collection) + "?" + urllib.parse.urlencode(
        {"bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": 1000, "f": "json"}) + f"&api-key={k}"
    feats = []
    for _ in range(20):
        js = jget(url, ttl=14 * 86400, timeout=60)
        feats += js.get("features", [])
        nxt = [l["href"] for l in js.get("links", []) if l.get("rel") == "next"]
        if not nxt or len(feats) >= max_features:
            break
        url = nxt[0] if "api-key=" in nxt[0] else nxt[0] + ("&" if "?" in nxt[0] else "?") + f"api-key={k}"
    return feats


def _lines(geom):
    if not geom:
        return []
    if geom["type"] == "LineString":
        return [geom["coordinates"]]
    if geom["type"] == "MultiLineString":
        return geom["coordinates"]
    return []


def _voltage(props):
    for kk, v in props.items():
        kl = kk.lower()
        if "jannite" in kl or "voltage" in kl:
            try:
                fv = float(v)
                return fv / 1000.0 if fv > 1000 else fv
            except (TypeError, ValueError):
                pass
    kl = props.get("kohdeluokka")
    try:
        return NLS_POWER_CLASSES.get(int(kl), (None, None))[1]
    except (TypeError, ValueError):
        return None


# NLS topographic database (Maastotietokanta) class codes. Two numbering generations exist in
# published lists, so both are mapped; unknown codes are kept and shown raw.
NLS_ROAD_CLASSES = {12111: ("main", 8.0, "road Ia"), 12112: ("main", 8.0, "road Ib"), 12121: ("main", 7.0, "road IIa"),
                    12122: ("main", 7.0, "road IIb"), 12131: ("forest", 5.0, "road IIIa"), 12132: ("forest", 4.5, "road IIIb"),
                    12141: ("forest", 4.0, "drive road (ajotie)"), 12316: ("forest", 3.0, "track (ajopolku)"),
                    12312: ("forest", 4.0, "winter road (talvitie)"), 12313: ("path", 1.5, "path (polku)"),
                    12314: ("path", 2.5, "foot/cycle way")}
NLS_BUILDING_GROUPS = [((41100, 42210, 42211, 42212, 42213, 42214), "house", "residential"),
                       ((41200, 42220, 42221, 42222, 42223, 42224), "public", "commercial / public"),
                       ((41300, 42230, 42231, 42232, 42233, 42234), "cabin", "holiday cabin"),
                       ((41400, 42240, 42241, 42242, 42243, 42244), "industrial", "industrial"),
                       ((41500, 41510, 42250, 42251, 42252, 42253, 42254, 42270), "church", "religious"),
                       ((41600, 42260, 42261, 42262, 42263, 42264), "other", "other building")]
NLS_CRITICAL = {"muuntoasema": ("substation", "transformer substation"), "masto": ("mast", "mast / tower"),
                "vesitorni": ("water_tower", "water tower"), "muuntaja": ("transformer", "transformer")}


def _building_type(p):
    kt = p.get("kayttotarkoitus")
    by_use = {1: ("house", "residential"), 2: ("public", "commercial / public"), 3: ("cabin", "holiday cabin"),
              4: ("industrial", "industrial"), 5: ("church", "religious"), 8: ("other", "other building")}
    try:
        if kt is not None and int(kt) in by_use:
            return by_use[int(kt)]
    except (TypeError, ValueError):
        pass
    try:
        k = int(p.get("kohdeluokka"))
    except (TypeError, ValueError):
        return ("house", "building")
    for codes, kind, label in NLS_BUILDING_GROUPS:
        if k in codes:
            return (kind, label)
    return ("house", f"building (class {k})")


def _centroid(g):
    c = g.get("coordinates")
    if g.get("type") == "Point":
        return float(c[0]), float(c[1])
    if g.get("type") == "Polygon":
        a = np.array(c[0], float)
        return float(a[:, 0].mean()), float(a[:, 1].mean())
    if g.get("type") == "MultiPolygon":
        a = np.array(c[0][0], float)
        return float(a[:, 0].mean()), float(a[:, 1].mean())
    if g.get("type") in ("LineString", "MultiPoint"):
        a = np.array(c, float)
        return float(a[:, 0].mean()), float(a[:, 1].mean())
    return None


def nls_topography(bbox):
    """Houses (by use), critical infrastructure, roads (by class) and power lines from the NLS topographic
    database (Maastotietokanta, OGC API Features)."""
    src = "National Land Survey of Finland - topographic database"
    attr = "© Maanmittauslaitos, Maastotietokanta, CC BY 4.0"
    if not nls_key():
        return {"ok": False, "source": src, "attribution": attr, "error": "NLS key missing (nls_key.txt / NLS_API_KEY)"}
    out = {"ok": True, "source": src, "attribution": attr, "power": [], "buildings": [], "roads": [],
           "critical": [], "errors": {}}
    try:
        for f in nls_features("sahkolinja", bbox):
            p = f.get("properties") or {}
            kv = _voltage(p)
            try:
                label = NLS_POWER_CLASSES.get(int(p.get("kohdeluokka")), ("power line",))[0]
            except (TypeError, ValueError):
                label = "power line"
            for line in _lines(f.get("geometry")):
                out["power"].append(dict(name=f"{label}{f' {kv:g} kV' if kv else ''} (NLS)", kind=label, kv=kv,
                                         path=[[c[0], c[1]] for c in line]))
    except Exception as exc:
        out["errors"]["sahkolinja"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    try:
        for f in nls_features("rakennus", bbox):
            c = _centroid(f.get("geometry") or {})
            if not c:
                continue
            kind, label = _building_type(f.get("properties") or {})
            out["buildings"].append(dict(name=f"{label} (NLS)", kind=kind, label=label, lon=c[0], lat=c[1]))
    except Exception as exc:
        out["errors"]["rakennus"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    try:
        for f in nls_features("tieviiva", bbox):
            p = f.get("properties") or {}
            try:
                tl = int(p.get("kohdeluokka", 0))
            except (TypeError, ValueError):
                tl = 0
            cls, width, label = NLS_ROAD_CLASSES.get(tl, ("forest", 4.0, f"road (class {tl})"))
            for line in _lines(f.get("geometry")):
                out["roads"].append(dict(name=p.get("nimi_suomi") or p.get("tienumero") or label, cls=cls, label=label,
                                         width=width, path=[[c[0], c[1]] for c in line]))
    except Exception as exc:
        out["errors"]["tieviiva"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    for coll, (kind, label) in NLS_CRITICAL.items():
        try:
            for f in nls_features(coll, bbox):
                c = _centroid(f.get("geometry") or {})
                if c:
                    out["critical"].append(dict(name=f"{label} (NLS)", kind=kind, label=label, lon=c[0], lat=c[1]))
        except Exception as exc:
            out["errors"][coll] = f"{type(exc).__name__}: {str(exc)[:120]}"
    out["counts"] = dict(power_lines=len(out["power"]), buildings=len(out["buildings"]), roads=len(out["roads"]),
                         critical=len(out["critical"]),
                         by_building_type={k: sum(b["kind"] == k for b in out["buildings"])
                                           for k in sorted({b["kind"] for b in out["buildings"]})})
    core = [k for k in ("sahkolinja", "rakennus", "tieviiva") if k in out["errors"]]
    if len(core) == 3:
        out["ok"] = False
        out["error"] = "; ".join(f"{k}: {out['errors'][k]}" for k in core)
        if "401" in out["error"]:
            out["error"] = "NLS refused the key (HTTP 401). Check it is active in your NLS account (omatili.maanmittauslaitos.fi)."
        elif "URLError" in out["error"] or "timed out" in out["error"]:
            out["error"] = "Could not reach the National Land Survey (no internet connection, or a firewall blocks it)."
    return out


def nls_topo_to_context(topo: dict, frame, base: dict | None = None):
    """Merge NLS power lines / buildings / roads into a Site context (NLS wins over OSM for power lines)."""
    ctx = dict(base or {})
    blds = [dict(b, kind={"public": "school", "church": "house", "industrial": "house", "other": "house"}.get(b["kind"], b["kind"]))
            for b in topo.get("buildings", [])]
    blds += [dict(c, kind="substation" if c["kind"] in ("substation", "transformer") else "mast") for c in topo.get("critical", [])]
    conv = osm_to_context({"roads": topo.get("roads", []), "power": [dict(p, kind="line") for p in topo.get("power", [])],
                           "buildings": blds}, frame)
    for b, src_b in zip(conv["buildings"], blds):
        b["name"] = src_b["name"]
    if conv["powerlines_vector"]:
        ctx["powerlines_vector"] = conv["powerlines_vector"]
    if conv["roads"]:
        ctx["roads"] = conv["roads"] + [r for r in ctx.get("roads", []) if r.get("discovered")]
    if conv["buildings"]:
        ctx["buildings"] = conv["buildings"]
    ctx["source"] = (ctx.get("source", "") + " + NLS topographic database").strip(" +")
    return ctx


# ------------------------------------------------------------------------------- Metsäkeskus
def metsakeskus_stands(bbox, limit=400):
    from .geo import TM35FIN

    tm = TM35FIN()
    x0, y0 = tm.from_lonlat(bbox[0], bbox[1])
    x1, y1 = tm.from_lonlat(bbox[2], bbox[3])
    last = None
    for tn in ("v1:stand", "stand"):
        q = urllib.parse.urlencode({"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": tn,
                                    "outputFormat": "application/json", "srsName": "EPSG:3067", "count": limit,
                                    "bbox": f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f},EPSG:3067"})
        try:
            js = jget(f"{METSAKESKUS_WFS}?{q}", ttl=7 * 86400, timeout=40)
            feats = []
            for f in js.get("features", []):
                g = f.get("geometry") or {}
                rings = g.get("coordinates", [])
                if g.get("type") == "MultiPolygon":
                    rings = [r for poly in rings for r in poly[:1]]
                else:
                    rings = rings[:1]
                ll_rings = []
                for r in rings:
                    a = np.array(r, float)
                    lo, la = tm.to_lonlat(a[:, 0], a[:, 1])
                    ll_rings.append(np.round(np.stack([lo, la], 1), 6).tolist())
                props = {k: v for k, v in (f.get("properties") or {}).items() if v not in (None, "")}
                feats.append(dict(rings=ll_rings, props=props, summary=_stand_summary(props)))
            return {"ok": True, "source": "Finnish Forest Centre - forest stands (Metsäkeskus)",
                    "attribution": "© Suomen metsäkeskus, CC BY 4.0", "stands": feats, "count": len(feats)}
        except Exception as exc:
            last = exc
    return _fail("Metsäkeskus forest stands", last, "© Suomen metsäkeskus, CC BY 4.0")


def _stand_summary(p):
    """Pick the interesting fields whatever the exact schema version calls them."""
    out = {}
    for k, v in p.items():
        kl = k.lower()
        for want, keys in (("species", ("maintreespecies", "species", "puulaji")),
                           ("height_m", ("meanheight", "height", "pituus")),
                           ("age", ("meanage", "age", "ika")),
                           ("volume_m3ha", ("volume", "tilavuus")),
                           ("development", ("developmentclass", "kehitys")),
                           ("fertility", ("fertilityclass", "kasvupaikka")),
                           ("soil", ("soiltype", "maalaji"))):
            if want not in out and any(kk in kl for kk in keys):
                out[want] = v
    return out


# ------------------------------------------------------------------------------- FMI
_NS = {"wfs": "http://www.opengis.net/wfs/2.0", "BsWfs": "http://xml.fmi.fi/schema/wfs/2.0",
       "gml": "http://www.opengis.net/gml/3.2"}


def fmi_observations(lat, lon, hours=6, place="Kuopio"):
    """Nearest FMI station. Tries latlon, then a bounding box around the site, then a named place,
    because the latlon lookup returns nothing when no station is within its default search radius."""
    end = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(hours=hours)
    base = {"service": "WFS", "version": "2.0.0", "request": "getFeature",
            "storedquery_id": "fmi::observations::weather::simple",
            "parameters": "t2m,rh,ws_10min,wg_10min,wd_10min,r_1h", "timestep": 10,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "endtime": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    tries = [{"latlon": f"{lat},{lon}", "maxlocations": 1},
             {"bbox": f"{lon - 0.8:.3f},{lat - 0.4:.3f},{lon + 0.8:.3f},{lat + 0.4:.3f}", "maxlocations": 1},
             {"place": place}]
    last = None
    for t in tries:
        try:
            xml = http(f"{FMI_WFS}?{urllib.parse.urlencode(base | t)}", ttl=1800)
            r = parse_fmi_simple(xml, lat, lon)
            if r.get("ok"):
                r["lookup"] = next(iter(t))
                return r
            last = RuntimeError(r.get("error"))
        except Exception as exc:
            last = exc
    return _fail("FMI observations", last, "© Finnish Meteorological Institute, CC BY 4.0")


def parse_fmi_simple(xml, lat, lon):
    root = ET.fromstring(xml)
    rows = []
    pos = None
    for el in root.iter("{http://xml.fmi.fi/schema/wfs/2.0}BsWfsElement"):
        p = el.find(".//gml:pos", _NS)
        if p is not None:
            pos = [float(v) for v in p.text.split()[:2]]
        rows.append((el.findtext("BsWfs:Time", namespaces=_NS), el.findtext("BsWfs:ParameterName", namespaces=_NS),
                     el.findtext("BsWfs:ParameterValue", namespaces=_NS)))
    latest = {}
    for t, k, v in rows:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        if k not in latest or t > latest[k][0]:
            latest[k] = (t, fv)
    if not latest:
        return {"ok": False, "source": "FMI observations", "error": "no observations returned"}
    out = {k: v[1] for k, v in latest.items()}
    return {"ok": True, "source": "FMI open data - nearest weather station",
            "attribution": "© Finnish Meteorological Institute, CC BY 4.0",
            "time": max(v[0] for v in latest.values()), "station_latlon": pos,
            "station_distance_km": round(float(_hav(lat, lon, *pos)), 1) if pos else None,
            "temp_c": out.get("t2m"), "rh": out.get("rh"), "wind_ms": out.get("ws_10min"),
            "gust_ms": out.get("wg_10min"), "wind_dir": out.get("wd_10min"), "rain_1h": out.get("r_1h")}


def fmi_warnings(lat, lon, region_words=("Pohjois-Savo", "North Savo", "Kuopio")):
    try:
        xml = http(FMI_CAP, ttl=900)
        return parse_cap_feed(xml, lat, lon, region_words)
    except Exception as exc:
        return _fail("FMI warnings", exc, "© Finnish Meteorological Institute")


def _point_in_poly(lat, lon, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        (la1, lo1), (la2, lo2) = poly[i], poly[(i + 1) % n]
        if (lo1 > lon) != (lo2 > lon):
            t = (lon - lo1) / (lo2 - lo1 + 1e-12)
            if lat < la1 + t * (la2 - la1):
                inside = not inside
    return inside


def parse_cap_feed(xml, lat, lon, region_words):
    root = ET.fromstring(xml)
    out = []
    for e in root.iter():
        if not e.tag.endswith("entry"):
            continue
        txt = {c.tag.split("}")[-1]: (c.text or "") for c in e}
        title = txt.get("title", "")
        summary = txt.get("summary", "") or txt.get("content", "")
        polys = []
        for c in e.iter():
            if c.tag.split("}")[-1] == "polygon" and c.text:
                v = [float(x) for x in c.text.replace(",", " ").split()]
                polys.append(list(zip(v[0::2], v[1::2])))
        hit = any(_point_in_poly(lat, lon, p) for p in polys) if polys else \
            any(w.lower() in (title + summary).lower() for w in region_words)
        if not hit:
            continue
        low = (title + " " + summary).lower()
        kind = "forest_fire" if ("fire" in low or "palo" in low) else "wind" if ("wind" in low or "tuuli" in low) else "other"
        out.append(dict(title=title.strip(), summary=summary.strip()[:300], kind=kind, updated=txt.get("updated")))
    return {"ok": True, "source": "FMI warnings (CAP)", "attribution": "© Finnish Meteorological Institute",
            "warnings": out, "count": len(out)}


# ------------------------------------------------------------------------------- photos
def commons_photos(lat, lon, radius_m=10000, limit=40):
    q = urllib.parse.urlencode({"action": "query", "format": "json", "generator": "geosearch",
                                "ggscoord": f"{lat}|{lon}", "ggsradius": min(radius_m, 10000), "ggslimit": limit,
                                "ggsnamespace": 6, "prop": "imageinfo|coordinates", "iiprop": "url|extmetadata",
                                "iiurlwidth": 480, "iiextmetadatafilter": "Artist|LicenseShortName|DateTimeOriginal"})
    try:
        js = jget(f"{COMMONS}?{q}", ttl=30 * 86400)
        return parse_commons(js)
    except Exception as exc:
        return _fail("Wikimedia Commons photos", exc, "Wikimedia Commons (individual licences)")


def parse_commons(js):
    import re
    out = []
    for p in (js.get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        co = (p.get("coordinates") or [{}])[0]
        if "lat" not in co or not ii.get("thumburl"):
            continue
        md = ii.get("extmetadata", {})
        strip = lambda s: re.sub("<[^>]+>", "", s or "").strip()
        out.append(dict(title=p.get("title", "").replace("File:", ""), lat=co["lat"], lon=co["lon"],
                        thumb=ii["thumburl"], page=ii.get("descriptionurl"),
                        artist=strip(md.get("Artist", {}).get("value"))[:80],
                        license=strip(md.get("LicenseShortName", {}).get("value")),
                        date=strip(md.get("DateTimeOriginal", {}).get("value"))[:10]))
    return {"ok": True, "source": "Wikimedia Commons (geotagged photos)",
            "attribution": "Photos: Wikimedia Commons contributors, licences per image", "photos": out,
            "count": len(out)}


# ------------------------------------------------------------------------------- orchestration
SOURCES = ("osm", "nlstopo", "sentinel2", "s2change", "stands", "fmi", "warnings", "photos", "nls")


def fetch_all(lat, lon, radius_m=1500, which=SOURCES):
    bbox = bbox_around(lat, lon, radius_m)
    res = {"center": [lon, lat], "bbox": bbox, "fetched": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"}
    if "osm" in which:
        res["osm"] = osm_context(bbox)
    if "nlstopo" in which:
        res["nlstopo"] = nls_topography(bbox)
    if "sentinel2" in which:
        pre, post = growing_season_windows()
        s = sentinel2_scenes(bbox, *post)
        if s["ok"]:
            s["tiles"] = s2_tile_templates(s["scenes"][0]["id"])
            s["window"] = post
        res["sentinel2"] = s
    if "s2change" in which:
        res["s2change"] = sentinel2_change(bbox)
    if "stands" in which:
        res["stands"] = metsakeskus_stands(bbox)
    if "fmi" in which:
        res["fmi"] = fmi_observations(lat, lon)
    if "warnings" in which:
        res["warnings"] = fmi_warnings(lat, lon)
    if "photos" in which:
        res["photos"] = commons_photos(lat, lon, radius_m=max(radius_m * 4, 5000))
    if "nls" in which:
        k = nls_key()
        res["nls"] = {"ok": bool(k), "source": "National Land Survey of Finland (Maanmittauslaitos)",
                      "attribution": "© Maanmittauslaitos, CC BY 4.0",
                      "layers": ["ortokuva", "maastokartta", "taustakartta"] if k else [],
                      **({} if k else {"error": "set NLS_API_KEY to enable orthophotos, DEM and laser scanning"})}
        if k:
            pr = nls_processes()
            res["nls"]["processes"] = pr
            res["nls"]["ok"] = pr["ok"]
            if not pr["ok"]:
                res["nls"]["error"] = pr["error"]
    return res


def public(res):
    """Strip numpy payloads before JSON."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return clean(res)
