"""Generate canned API responses (real formats, invented content) for offline tests.
NOT real data - used only when HORUS_ONLINE_FIXTURES points here."""
import io, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from horus.online import bbox_around, _tile_xy
from horus.geo import TM35FIN

F = Path(__file__).resolve().parent / "fixtures"
F.mkdir(exist_ok=True)
LAT, LON = 62.747066, 27.259548
b = bbox_around(LAT, LON, 1500)
rng = np.random.default_rng(1)

def ll(dx, dy):  # metres -> lon/lat
    return [LON + dx / (111320 * math.cos(math.radians(LAT))), LAT + dy / 111320]

els = []
els.append({"type": "way", "id": 1, "tags": {"highway": "secondary", "name": "Fixture road 5501", "ref": "5501"},
            "geometry": [dict(zip(("lon", "lat"), ll(-1400 + 100 * i, -300 + 30 * i))) for i in range(29)]})
els.append({"type": "way", "id": 2, "tags": {"highway": "track"},
            "geometry": [dict(zip(("lon", "lat"), ll(-200 + 60 * i, 500 - 40 * i))) for i in range(15)]})
els.append({"type": "way", "id": 3, "tags": {"power": "minor_line", "voltage": "20000"},
            "geometry": [dict(zip(("lon", "lat"), ll(-1500 + 150 * i, 600 - 50 * i))) for i in range(21)]})
for i in range(12):
    x, y = rng.uniform(-900, 900, 2)
    els.append({"type": "way", "id": 100 + i, "tags": {"building": "house"},
                "geometry": [dict(zip(("lon", "lat"), ll(x + dx, y + dy))) for dx, dy in ((0, 0), (10, 0), (10, 8), (0, 8), (0, 0))]})
els.append({"type": "way", "id": 200, "tags": {"natural": "wetland", "wetland": "bog"},
            "geometry": [dict(zip(("lon", "lat"), ll(dx, dy))) for dx, dy in ((300, -800), (700, -800), (700, -400), (300, -400), (300, -800))]})
els.append({"type": "node", "id": 300, "lat": 62.80, "lon": 27.45, "tags": {"amenity": "fire_station", "name": "Fixture fire station"}})
(F / "overpass.json").write_text(json.dumps({"elements": els}))

stac = {"type": "FeatureCollection", "features": [
    {"id": "S2_FIXTURE_{TAG}_A", "bbox": [b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1], "properties": {"datetime": "2026-07-14T09:50:31Z", "eo:cloud_cover": 2.4}},
    {"id": "S2_FIXTURE_{TAG}_B", "bbox": [b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1], "properties": {"datetime": "2026-08-02T09:50:31Z", "eo:cloud_cover": 11.0}}]}
(F / "stac.json").write_text(json.dumps(stac))

# NDVI tiles: forest ~0.75; "post" has a 'clear-cut' patch with NDVI 0.25
def tile(post):
    yy, xx = np.mgrid[0:256, 0:256]
    v = 0.72 + 0.06 * np.sin(xx / 17) * np.cos(yy / 23)
    if post:
        v = np.where((xx - 128) ** 2 / 900 + (yy - 110) ** 2 / 400 < 1, 0.25, v)
    g = np.clip((v + 1) / 2 * 255, 0, 255).astype(np.uint8)
    a = np.full_like(g, 255)
    bio = io.BytesIO(); Image.fromarray(np.stack([g, a], -1), "LA").save(bio, "PNG"); return bio.getvalue()
(F / "tile_pre.png").write_bytes(tile(False))
(F / "tile_post.png").write_bytes(tile(True))

fmi = """<?xml version="1.0" encoding="UTF-8"?>
<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:BsWfs="http://xml.fmi.fi/schema/wfs/2.0" xmlns:gml="http://www.opengis.net/gml/3.2">
""" + "".join(f"""<wfs:member><BsWfs:BsWfsElement gml:id="e{i}"><BsWfs:Location><gml:Point><gml:pos>62.8920 27.6340 </gml:pos></gml:Point></BsWfs:Location>
<BsWfs:Time>2026-10-03T10:00:00Z</BsWfs:Time><BsWfs:ParameterName>{k}</BsWfs:ParameterName><BsWfs:ParameterValue>{v}</BsWfs:ParameterValue></BsWfs:BsWfsElement></wfs:member>
""" for i, (k, v) in enumerate((("t2m", 7.4), ("rh", 81), ("ws_10min", 6.2), ("wg_10min", 11.8), ("wd_10min", 230), ("r_1h", 0.2)))) + "</wfs:FeatureCollection>"
(F / "fmi.xml").write_text(fmi)
cap = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Yellow wind warning: Pohjois-Savo (fixture)</title><updated>2026-10-03T08:00:00Z</updated>
<summary>Gusts 20 m/s possible in North Savo.</summary></entry>
<entry><title>Sea wind warning: Bothnian Sea (fixture)</title><updated>2026-10-03T08:00:00Z</updated><summary>Sea area.</summary></entry></feed>"""
(F / "cap.xml").write_text(cap)
tm = TM35FIN()
feats = []
for i in range(6):
    cx, cy = tm.from_lonlat(*ll(-900 + 350 * i, 300 - 120 * i))
    ring = [[cx - 120, cy - 90], [cx + 120, cy - 90], [cx + 120, cy + 90], [cx - 120, cy + 90], [cx - 120, cy - 90]]
    feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[float(x), float(y)] for x, y in ring]]},
                  "properties": {"standid": 1000 + i, "maintreespecies": [1, 2, 3][i % 3], "meanheight": 12 + 2 * i,
                                 "meanage": 30 + 10 * i, "developmentclass": "04", "fertilityclass": 3}})
(F / "stands.json").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
pages = {}
for i in range(5):
    lo, la = ll(rng.uniform(-3000, 3000), rng.uniform(-3000, 3000))
    pages[str(i)] = {"title": f"File:Fixture forest photo {i}.jpg", "coordinates": [{"lat": la, "lon": lo}],
                     "imageinfo": [{"thumburl": "/fixture-photo.svg", "descriptionurl": "https://commons.wikimedia.org/",
                                    "extmetadata": {"Artist": {"value": "Fixture"}, "LicenseShortName": {"value": "CC BY-SA 4.0"}}}]}
(F / "commons.json").write_text(json.dumps({"query": {"pages": pages}}))
print("fixtures written to", F)

# NLS topographic database: power line (sahkolinja) through the demo estate, OGC API Features GeoJSON
from horus.synth import generate as _g
from horus.geo import LocalFrame as _LF
_s = _g()
_f = _LF(LAT, LON, _s.size / 2, _s.size / 2)
_lo, _la = _f.to_lonlat(_s.powerlines[0]["line"][:, 0], _s.powerlines[0]["line"][:, 1])
(F / "nls_sahkolinja.json").write_text(json.dumps({"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"kohdeluokka": 22312, "jannite": 20000},
     "geometry": {"type": "LineString", "coordinates": [[float(a), float(b)] for a, b in zip(_lo, _la)]}}], "links": []}))
_bl = []
for b in _s.buildings:
    lo, la = _f.to_lonlat(b["x"], b["y"])
    _bl.append({"type": "Feature", "properties": {"kohdeluokka": 42211}, "geometry": {"type": "Point", "coordinates": [float(lo), float(la)]}})
(F / "nls_rakennus.json").write_text(json.dumps({"type": "FeatureCollection", "features": _bl, "links": []}))
print("NLS fixtures written")

# NLS critical infrastructure + roads (official class codes)
_lo, _la = _f.to_lonlat(np.array([480.0]), np.array([396.0]))
(F / "nls_muuntoasema.json").write_text(json.dumps({"type": "FeatureCollection", "links": [], "features": [
    {"type": "Feature", "properties": {"kohdeluokka": 22200}, "geometry": {"type": "Point", "coordinates": [float(_lo[0]), float(_la[0])]}}]}))
_rf = []
for r in _s.roads:
    lo, la = _f.to_lonlat(r["line"][:, 0], r["line"][:, 1])
    _rf.append({"type": "Feature", "properties": {"kohdeluokka": 12121 if r["cls"] == "main" else 12141},
                "geometry": {"type": "LineString", "coordinates": [[float(a), float(b)] for a, b in zip(lo, la)]}})
(F / "nls_tieviiva.json").write_text(json.dumps({"type": "FeatureCollection", "features": _rf, "links": []}))
_bl = []
for k, b in enumerate(_s.buildings):
    lo, la = _f.to_lonlat(b["x"], b["y"])
    code = 42221 if b["kind"] == "care_home" else 42231 if b["kind"] == "cabin" else 42211
    _bl.append({"type": "Feature", "properties": {"kohdeluokka": code}, "geometry": {"type": "Point", "coordinates": [float(lo), float(la)]}})
(F / "nls_rakennus.json").write_text(json.dumps({"type": "FeatureCollection", "features": _bl, "links": []}))
print("NLS infrastructure fixtures written")
