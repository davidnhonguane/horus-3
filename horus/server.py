"""Zero-dependency HTTP server: JSON API + static web UI.

Security: binds to 127.0.0.1 by default. Set HORUS_TOKEN to require a bearer token
(header ``Authorization: Bearer <token>`` or ``?token=``) for every API call; the
web UI forwards the token from its own URL. Uploaded/loaded data never leaves the
machine except the optional weather request (lat/lon of the site centre only).
"""
from __future__ import annotations

import gzip
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import dispatch, online

WEB = Path(__file__).resolve().parent.parent / "web"
UPLOADS = Path(os.environ.get("HORUS_UPLOADS") or (Path(__file__).resolve().parent.parent / "data" / "uploads"))
from .scan3d import SCAN_EXT  # noqa: E402  (.las .laz .ply .obj .glb .xyz .txt .csv .pts .e57)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
SIDE_EXT = (".jgw", ".pgw", ".tfw", ".wld", ".geojson", ".json", ".bin")   # .bin = .gltf buffer
MAX_UPLOAD = int(float(os.environ.get("HORUS_MAX_UPLOAD_GB", "8")) * 1024 ** 3)


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if not np.isfinite(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def _clean(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _ensure_laz_support(log):
    """.laz is compressed: it needs laspy + lazrs. Install them into this Python if missing."""
    try:
        import laspy  # noqa: F401
        import lazrs  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    import sys
    log("This is a compressed .laz file: installing LAZ support (pip install \"laspy[lazrs]\") ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laspy[lazrs]>=2.4"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("could not install LAZ support automatically. Stop Horus and run:  "
                           f"{sys.executable} -m pip install \"laspy[lazrs]\"  then start Horus again "
                           f"(pip said: {(r.stderr or r.stdout).strip()[-300:]})")
    import importlib
    importlib.invalidate_caches()
    import laspy  # noqa: F401,F811
    log("LAZ support installed.")


class App:
    def __init__(self, site, dataset: dispatch.Dataset | None):
        self.site = site
        self.ds = dataset
        self.token = os.environ.get("HORUS_TOKEN")
        self._fleet = None
        self._lock = threading.Lock()
        self.online = None          # last fetch_all result (with numpy payloads)
        self.online_busy = False
        self.online_radius = 1500
        self.nls_job = {"state": "idle"}
        self.demo = site            # the site Horus started with ("back to demo")
        self.photo_store = {}       # (batch, index) -> (overlay jpeg, north-up photo jpeg)
        self.upload_job = {"state": "idle"}

    # ---------------------------------------------------------------- uploaded 3D scans
    def save_upload(self, batch: str, name: str, stream, length: int) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,40}", batch or ""):
            raise ValueError("bad batch id")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "").name)[-120:].lstrip(".")
        ext = Path(safe).suffix.lower()
        if not safe or ext not in SCAN_EXT + IMAGE_EXT + SIDE_EXT:
            hint = {".usdz": " - iPhone/iPad scans: export as .glb, .obj or .ply from the scanning app",
                     ".fbx": " - export the model as .obj, .glb or .ply", ".3mf": " - export as .stl, .obj or .ply",
                     ".dae": " - export as .obj, .glb or .ply", ".osgb": " - export as .las or .ply from the photogrammetry software",
                     ".b3dm": " - export as .las or .ply from the photogrammetry software",
                     ".zip": " - unzip it first and upload the 3D file inside"}.get(ext, "")
            raise ValueError(f"file type {ext or '?'} not supported{hint}. Upload a 3D scan ({' '.join(SCAN_EXT)}), "
                             "a .jpg/.png/.tif photo (+ .jgw/.pgw/.tfw world file) or a .geojson context")
        if length <= 0 or length > MAX_UPLOAD:
            raise ValueError(f"file size must be 1 byte .. {MAX_UPLOAD // 1024 ** 3} GB")
        d = UPLOADS / batch
        d.mkdir(parents=True, exist_ok=True)
        f = d / safe
        left = length
        with open(f, "wb") as fh:
            while left > 0:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
        if left:
            f.unlink(missing_ok=True)
            raise ValueError("upload interrupted")
        magic = {".las": b"LASF", ".laz": b"LASF", ".ply": b"ply", ".glb": b"glTF", ".e57": b"ASTM"}.get(ext)
        if magic:
            with open(f, "rb") as fh:
                if fh.read(len(magic)) != magic:
                    f.unlink(missing_ok=True)
                    raise ValueError(f"{safe} is not a valid {ext} 3D file")
        return dict(name=safe, bytes=length, kind="scan" if ext in SCAN_EXT else "image" if ext in IMAGE_EXT else "side")

    def start_upload_analysis(self, batch: str, opts: dict):
        """Background job: read the uploaded scan(s), find roads / power lines / buildings, recognise trees
        (species, health), add NLS / OSM infrastructure if asked, run fire / storm / routing, swap the site."""
        from .site import Site

        if not re.fullmatch(r"[A-Za-z0-9_-]{6,40}", batch or ""):
            raise ValueError("bad batch id")
        d = UPLOADS / batch
        files = sorted(d.iterdir()) if d.is_dir() else []
        scans = [f for f in files if f.suffix.lower() in SCAN_EXT]
        images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
        ctxs = [f for f in files if f.suffix.lower() in (".geojson", ".json")]
        if not scans:
            if not images:
                raise ValueError(f"upload a 3D scan ({' '.join(SCAN_EXT)}) or a photo (.jpg .png .tif)")
            return self.start_photo_analysis(batch, images, opts)

        def log(msg, **kw):
            self.upload_job.update(kw)
            self.upload_job.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}")

        def work():
            try:
                if any(f.suffix.lower() == ".laz" for f in scans):
                    _ensure_laz_support(log)
                lat, lon = opts.get("lat"), opts.get("lon")
                anchor = (float(lat), float(lon)) if lat not in (None, "") and lon not in (None, "") else None
                crs = opts.get("crs") or "auto"
                mp = int(opts.get("max_points") or 4_000_000)
                log(f"Reading {', '.join(f.name for f in scans)} ...", step="read")
                site = Site.from_las(scans, crs=crs, context_geojson=ctxs[0] if ctxs else None, max_points=mp,
                                     osm=bool(opts.get("osm")), image=images[0] if images else None,
                                     nls=bool(opts.get("nls")), anchor=anchor)
                h = site.pts.get("header") or {}
                log(f"{site.pts['x'].shape[0]:,} points"
                    + (f" (every {h['decimation']}th of {h['points_in_file']:,})" if h.get("decimation", 1) > 1 else "")
                    + f" · coordinates: {site.meta.get('crs')}", step="analyse")
                if site.meta.get("placement"):
                    log(site.meta["placement"])
                if images:
                    log(f"Image: {images[0].name}" + ("" if getattr(site, "ortho", None) is not None else " (no georeference - skipped)"))
                log("Finding roads, power lines, buildings; recognising trees; fire, storm and routing models ...")
                site.name = opts.get("name") or scans[0].stem
                site.run(opts.get("scenario") or "normal")
                log("Simulating a storm and a fire on the scan ...", step="simulate")
                try:
                    sims = site.auto_simulations()
                    site.auto_sims = sims
                    self.upload_job["sims"] = sims
                    st, fr = sims["storm"], sims["fire"]
                    log(f"Storm {st['gust_ms']:.0f} m/s: {st['fallen']['mean']:.0f} trees fall, "
                        f"P(power-line outage) {st['p_line_outage']:.0%}. Fire ({fr['hours']:.0f} h): "
                        f"{fr['trees']['reached']} trees reached, {fr['trees']['torching']} torch")
                except Exception as exc:  # noqa: BLE001  (simulations must not lose the analysis)
                    traceback.print_exc()
                    log(f"Simulations skipped: {exc}")
                self.site = site
                T = site.trees
                d_ = site.summary().get("discovered") or {}
                u = d_.get("unmapped") or {}
                log(f"Done: {T['x'].shape[0]:,} trees, {int((T['health'] == 2).sum())} dead, "
                    f"{len(site.ctx.get('roads', []))} roads ({u.get('roads', 0)} new), "
                    f"{len(site.powerlines)} power line(s), {u.get('buildings', 0)} new building(s)",
                    state="done", step="done")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                log(f"Failed: {exc}", state="failed", error=str(exc))

        self.upload_job = {"state": "running", "batch": batch, "files": [f.name for f in files], "log": []}
        threading.Thread(target=work, daemon=True).start()
        return self.upload_job

    def start_photo_analysis(self, batch, images, opts):
        """Photos only (no 3D scan): analyse each photo on its own (horus.photo)."""
        from . import photo

        def log(msg, **kw):
            self.upload_job.update(kw)
            self.upload_job.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}")

        def work():
            try:
                out = []
                for k, f in enumerate(images):
                    log(f"Analysing photo {f.name} ...", step="photo")
                    gsd = opts.get("photo_gsd")
                    hgt = opts.get("photo_height_m")
                    gsd = float(gsd) if gsd not in (None, "") else None
                    if gsd is None and hgt not in (None, ""):
                        meta = photo.read_meta(f)
                        f35 = meta.get("focal35_mm") or 24.0
                        gsd = float(hgt) * 36.0 / f35 / meta["width"]
                    lat, lon = opts.get("lat"), opts.get("lon")
                    res, ov, img = photo.analyse_photo(f, view=opts.get("photo_view") or "auto",
                                                       lat=float(lat) if lat not in (None, "") else None,
                                                       lon=float(lon) if lon not in (None, "") else None, gsd=gsd)
                    self.photo_store[(batch, k)] = (ov, img)
                    res["overlay_url"] = f"/api/photo/{batch}/{k}/overlay.jpg"
                    res["image_url"] = f"/api/photo/{batch}/{k}/photo.jpg"
                    out.append(res)
                    if res["view"] == "above":
                        t = res["trees"]
                        log(f"{f.name}: {t['count']} tree crowns ({t['dead']} dead, {t['stressed']} stressed), "
                            f"{len(res['roads'])} road(s), {len(res['corridors'])} power-line clearing(s)")
                    else:
                        fo = res["foliage"]
                        log(f"{f.name}: photo from the ground - foliage {fo['green']:.0%} green, "
                            f"{fo['yellowing']:.0%} yellowing, {fo['brown_grey']:.0%} brown/grey")
                for k in [key for key in self.photo_store if key[0] != batch]:
                    del self.photo_store[k]                 # keep only the latest batch in memory
                self.upload_job["photos"] = out
                log("Done.", state="done", step="done")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                log(f"Failed: {exc}", state="failed", error=str(exc))

        self.upload_job = {"state": "running", "batch": batch, "files": [f.name for f in images], "log": [], "kind": "photo"}
        threading.Thread(target=work, daemon=True).start()
        return self.upload_job

    def clear_upload(self, batch: str):
        if re.fullmatch(r"[A-Za-z0-9_-]{6,40}", batch or ""):
            shutil.rmtree(UPLOADS / batch, ignore_errors=True)

    def start_nls_load(self, lat, lon, radius_m=400, sheet=None, scenario="normal"):
        """Background job: download NLS laser scanning around (lat, lon), add NLS + OSM infrastructure,
        run tree recognition and all risk models, then swap the active site."""
        from pathlib import Path
        from .geo import TM35FIN
        from .site import Site

        def log(msg, **kw):
            self.nls_job.update(kw)
            self.nls_job.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}")

        def work():
            try:
                out = Path(__file__).resolve().parent.parent / "data" / "nls" / f"{lat:.4f}_{lon:.4f}"
                log(f"Requesting NLS laser scanning around {lat:.4f} N, {lon:.4f} E ...", step="download")
                r = online.nls_laser(lat, lon, out, map_sheet=sheet, radius_m=max(radius_m, 500))
                if not r.get("ok"):
                    raise RuntimeError(r.get("error"))
                log(f"Downloaded {len(r['files'])} file(s)", step="read")
                E, N = TM35FIN().from_lonlat(lon, lat)
                site = Site.from_las(r["files"], crs="EPSG:3067", osm=True,
                                     crop=(E - radius_m, N - radius_m, E + radius_m, N + radius_m))
                log("Adding NLS topographic power lines / buildings / roads ...", step="context")
                bb = online.bbox_around(lat, lon, radius_m + 100)
                topo = online.nls_topography(bb)
                if topo.get("ok"):
                    site.ctx.update(online.nls_topo_to_context(topo, site.frame, site.ctx))
                log(f"Tree recognition + risk models on {site.pts['x'].shape[0]:,} points ...", step="analyse")
                site.name = f"NLS laser scan {lat:.4f} N {lon:.4f} E"
                site.run(scenario)
                self.site = site
                self._fleet = self._fleet
                log(f"Done: {site.trees['x'].shape[0]:,} trees recognised", state="done", step="done")
            except Exception as exc:
                log(f"Failed: {exc}", state="failed", error=str(exc))

        self.nls_job = {"state": "running", "lat": lat, "lon": lon, "log": []}
        threading.Thread(target=work, daemon=True).start()

    def fetch_online(self, radius_m=None, which=online.SOURCES):
        lon, lat = self.site.center_lonlat()
        self.online_busy = True
        try:
            r = online.fetch_all(lat, lon, radius_m or self.online_radius, which)
            if self.online and set(which) != set(online.SOURCES):
                r = {**self.online, **r}
            self.online = r
        finally:
            self.online_busy = False
        return r

    def online_status(self):
        lon, lat = self.site.center_lonlat()
        base = {"center": [lon, lat], "nls_key": bool(online.nls_key()), "busy": self.online_busy,
                "fixtures": bool(online.FIXTURES)}
        if self.online is None:
            return base | {"fetched": None}
        out = base | online.public(self.online)
        out["stand_check"] = self.stand_check()
        return out

    def stand_check(self):
        """Independent validation for real surveys: Horus' dominant detected species per Metsäkeskus
        stand vs the stand register's main species (1 pine, 2 spruce, 3/4 birch)."""
        st = (self.online or {}).get("stands") or {}
        s = self.site
        if not st.get("ok") or s.meta.get("kind") == "synthetic":
            return None
        import numpy as np
        lon, lat = s.frame.to_lonlat(s.trees["x"], s.trees["y"])
        pts = np.stack([lon, lat], 1)
        code = {1: 0, 2: 1, 3: 2, 4: 2}
        n = agree = 0
        rows = []
        for f in st["stands"]:
            sp = f["summary"].get("species")
            try:
                sp = int(sp)
            except (TypeError, ValueError):
                continue
            if sp not in code or not f["rings"]:
                continue
            inside = _pip(pts, np.array(f["rings"][0], float))
            if inside.sum() < 20:
                continue
            area = np.bincount(s.trees["species"][inside], weights=s.trees["crown_area"][inside], minlength=3)
            dom = int(np.argmax(area))
            n += 1
            agree += dom == code[sp]
            rows.append(dict(stand=f["props"].get("standid"), register=code[sp], horus=dom, trees=int(inside.sum())))
        return dict(stands=n, agreement=(agree / n) if n else None, rows=rows[:30]) if n else None

    def fleet(self):
        with self._lock:
            if self._fleet is None and self.ds is not None:
                sch, k = dispatch.baseline(self.ds)
                ops = [dict(name=o["name"], lat=o["homeLocation"]["latitude"], lon=o["homeLocation"]["longitude"],
                            home=o["homeLocation"]["municipality"],
                            multispectral=dispatch.MS in o["supportedDroneConfigurations"],
                            days_available=sum(a["status"] == "available" for a in o["availability"]))
                       for o in self.ds.ops]
                tasks = [dict(name=t["name"], lat=t["location"]["latitude"], lon=t["location"]["longitude"],
                              municipality=t["location"]["municipality"], area=t["areaHectares"], priority=t["priority"],
                              ms=t["requiredDroneConfiguration"] == dispatch.MS, due=t["dueDate"],
                              done=int(sch.done[i]) >= 0, on_time=bool(0 <= sch.done[i] <= sch.due[i]))
                         for i, t in enumerate(self.ds.tasks)]
                self._fleet = dict(operators=ops, tasks=tasks, baseline=k, period=self.ds.period,
                                   travel_model=dict(minutes_per_km=self.ds.fit[0], intercept=self.ds.fit[1],
                                                     mae_min=self.ds.fit_mae),
                                   dataset=self.ds.summary)
            return self._fleet


def _pip(pts, ring):
    """Vectorised point-in-polygon (even-odd)."""
    x, y = pts[:, 0], pts[:, 1]
    ins = np.zeros(len(pts), bool)
    xj, yj = ring[-1]
    for xi, yi in ring:
        c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        ins ^= c
        xj, yj = xi, yi
    return ins


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "Horus/1.0"

        def log_message(self, fmt, *args):
            if os.environ.get("HORUS_QUIET"):
                return
            super().log_message(fmt, *args)

        # ---------------------------------------------------------------- helpers
        def _send(self, code, body: bytes, ctype="application/json", cache=False):
            gz = "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 2048 and not ctype.startswith("image/png")
            if gz:
                body = gzip.compress(body, compresslevel=5)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "max-age=300" if cache else "no-store")
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_clean(obj), cls=_Enc, separators=(",", ":")).encode())

        def _authorized(self, q):
            if not app.token:
                return True
            h = self.headers.get("Authorization", "")
            return h == f"Bearer {app.token}" or q.get("token", [None])[0] == app.token

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > 1_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        # ---------------------------------------------------------------- routes
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            p = u.path
            try:
                if not p.startswith("/api/"):
                    return self._static(p)
                if not self._authorized(q):
                    return self._json({"error": "unauthorized"}, 401)
                s = app.site
                if p == "/api/health":
                    from . import __version__
                    return self._json({"ok": True, "site": s.name, "version": __version__, "time": time.time(),
                                       "upload_formats": list(SCAN_EXT)})
                if p == "/api/site/summary":
                    return self._json(s.summary())
                if p == "/api/site/vectors":
                    return self._json(s.vectors())
                if p == "/api/site/trees":
                    return self._json(s.tree_table())
                if p.startswith("/api/site/layer/") and p.endswith(".png"):
                    name = p[len("/api/site/layer/"):-4]
                    try:
                        return self._send(200, s.layer_png(name), "image/png")
                    except KeyError:
                        return self._json({"error": f"layer {name} not available"}, 404)
                if p == "/api/site/infrastructure":
                    return self._json(s.infrastructure())
                if p == "/api/site/ortho.jpg":
                    if getattr(s, "ortho", None) is None:
                        return self._json({"error": "no orthophoto"}, 404)
                    if not hasattr(s, "_ortho_jpg"):
                        s._ortho_jpg = s.ortho.png()
                    return self._send(200, s._ortho_jpg, "image/jpeg", cache=True)
                if p == "/api/site/nls":
                    if getattr(s, "nls_topo", None) is None:
                        lon, lat = s.center_lonlat()
                        b = s.bounds_lonlat()
                        s.nls_topo = online.nls_topography([b[0][1] - 0.003, b[0][0] - 0.0015, b[1][1] + 0.003, b[1][0] + 0.0015])
                    return self._json(s.infrastructure()["nls"])
                if p == "/api/site/fire_trees":
                    return self._json(s.fire_prone_trees(int(q.get("top", ["25"])[0])))
                if p == "/api/site/structures":
                    return self._json(s.structures_at_risk())
                if p == "/api/site/views":
                    return self._json(s.views())
                if p.startswith("/api/site/view3d/") and p.endswith(".png"):
                    try:
                        vid = int(p[len("/api/site/view3d/"):-4])
                        return self._send(200, s.view_png(vid), "image/png", cache=True)
                    except (ValueError, IndexError):
                        return self._json({"error": "unknown view"}, 404)
                if p == "/api/site/view3d.png":
                    lon, lat = float(q["lon"][0]), float(q["lat"][0])
                    az = float(q["az"][0]) if "az" in q else None
                    rad = min(max(float(q.get("r", ["30"])[0]), 10), 80)
                    return self._send(200, s.view_png(lon=lon, lat=lat, az=az, radius=rad), "image/png")
                if p == "/api/nls/status":
                    return self._json(app.nls_job)
                if p.startswith("/api/photo/"):
                    parts = p.split("/")              # /api/photo/<batch>/<k>/<overlay|photo>.jpg
                    try:
                        item = app.photo_store[(parts[3], int(parts[4]))]
                    except (KeyError, ValueError, IndexError):
                        return self._json({"error": "not found"}, 404)
                    body = item[0] if parts[5].startswith("overlay") else item[1]
                    return self._send(200, body, "image/png" if body[:4] == b"\x89PNG" else "image/jpeg")
                if p == "/api/site/costs":
                    from . import costs as _c
                    return self._json(_c.public(s.costs()))
                if p == "/api/site/sims":
                    return self._json(getattr(s, "auto_sims", None) or {})
                if p == "/api/upload/status":
                    return self._json(dict(app.upload_job, site=s.name, demo=s is app.demo,
                                           max_upload_gb=MAX_UPLOAD // 1024 ** 3))
                if p == "/api/site/points_sim.bin":
                    return self._send(200, s.points_sim_bin(), "application/octet-stream")
                if p == "/api/site/highlights":
                    return self._json(s.highlights())
                if p == "/api/site/points.bin":
                    if not hasattr(s, "_pc_cache"):
                        s._pc_cache = s.point_cloud_bin()
                    return self._send(200, s._pc_cache, "application/octet-stream", cache=True)
                if p == "/api/site/workorder.csv":
                    body = s.workorder_csv().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv")
                    self.send_header("Content-Disposition", "attachment; filename=horus_hazard_tree_workorder.csv")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if p == "/api/site/workorder":
                    top = int(q.get("top", ["50"])[0])
                    return self._json(s.workorder(top))
                if p == "/api/online":
                    return self._json(app.online_status())
                if p == "/api/online/s2change.png":
                    ch = (app.online or {}).get("s2change") or {}
                    if not ch.get("ok"):
                        return self._json({"error": "no change analysis yet"}, 404)
                    return self._send(200, online.change_png(ch), "image/png")
                if p.startswith("/api/tiles/nls/"):
                    parts = p.split("/")[4:]
                    if len(parts) != 4 or parts[0] not in ("ortokuva", "maastokartta", "taustakartta", "selkokartta") \
                            or not all(x.isdigit() for x in parts[1:]):
                        return self._json({"error": "bad tile"}, 400)
                    if not online.nls_key():
                        return self._json({"error": "NLS_API_KEY not set"}, 404)
                    try:
                        data = online.http(online.nls_tile_url(parts[0], *map(int, parts[1:])), ttl=30 * 86400)
                    except Exception as exc:
                        return self._json({"error": str(exc)}, 502)
                    return self._send(200, data, "image/jpeg" if parts[0] == "ortokuva" else "image/png", cache=True)
                if p == "/api/fleet":
                    if app.ds is None:
                        return self._json({"error": "Forey dataset not loaded"}, 404)
                    return self._json(app.fleet())
                return self._json({"error": "not found"}, 404)
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                return self._json({"error": str(exc)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authorized(q):
                return self._json({"error": "unauthorized"}, 401)
            try:
                if u.path == "/api/upload/file":
                    if app.upload_job.get("state") == "running":
                        return self._json({"error": "an analysis is running - wait for it to finish"}, 409)
                    n = int(self.headers.get("Content-Length", 0) or 0)
                    return self._json(app.save_upload(q.get("batch", [""])[0], q.get("name", [""])[0], self.rfile, n))
                b = self._body()
                s = app.site
                if u.path == "/api/upload/analyze":
                    if app.upload_job.get("state") == "running":
                        return self._json(app.upload_job, 409)
                    return self._json(app.start_upload_analysis(b.get("batch", ""), b))
                if u.path == "/api/upload/clear":
                    app.clear_upload(b.get("batch", ""))
                    return self._json({"ok": True})
                if u.path == "/api/site/demo":
                    app.site = app.demo
                    return self._json({"ok": True, "site": app.demo.name})
                if u.path == "/api/site/scenario":
                    sc = b.get("scenario", "normal")
                    if sc not in ("live", "normal", "heatwave", "storm"):
                        return self._json({"error": "unknown scenario"}, 400)
                    gust = b.get("gust")
                    wd = b.get("wind_dir")
                    return self._json(s.set_scenario(sc, None if gust in (None, "") else float(gust),
                                                     None if wd in (None, "") else float(wd)))
                if u.path == "/api/site/costs":
                    from . import costs as _c
                    cfg = s.costs()
                    for k, val in (b.get("values") or {}).items():
                        if k in cfg and val not in (None, ""):
                            cfg[k]["value"] = float(val)
                            cfg[k]["basis"] = "user"
                    if b.get("reset"):
                        s._costs = cfg = _c.load(path="/nonexistent")
                    _c.save(cfg)
                    return self._json(_c.public(cfg))
                if u.path == "/api/site/plan":
                    af = b.get("avoid_fire_min")
                    return self._json(s.plan_emergency(b.get("end", "depot"), b.get("start", "depot"),
                                                       None if af in (None, "") else float(af), b.get("mission", "any")))
                if u.path == "/api/site/storm_sim":
                    return self._json(s.simulate_storm(b.get("gust"), b.get("wind_dir"), int(b.get("runs", 300))))
                if u.path == "/api/site/fire":
                    return self._json(s.simulate_fire(float(b["lon"]), float(b["lat"]), float(b.get("hours", 4))))
                if u.path == "/api/site/route":
                    start = b.get("start", "depot")
                    end = b.get("end", "depot")
                    af = b.get("avoid_fire_min")
                    return self._json(s.route(b.get("vehicle", "fire_engine"), start, end,
                                              float(b.get("block_threshold", 0.5)),
                                              None if af in (None, "") else float(af),
                                              storm=bool(b.get("storm", True))))
                if u.path == "/api/nls/load":
                    if app.nls_job.get("state") == "running":
                        return self._json(app.nls_job, 409)
                    app.start_nls_load(float(b["lat"]), float(b["lon"]), float(b.get("radius_m", 400)),
                                       b.get("sheet") or None, b.get("scenario", "normal"))
                    return self._json(app.nls_job)
                if u.path == "/api/online/fetch":
                    srcs = tuple(x for x in b.get("sources", online.SOURCES) if x in online.SOURCES)
                    app.fetch_online(int(b.get("radius_m", app.online_radius)), srcs or online.SOURCES)
                    return self._json(app.online_status())
                if u.path == "/api/online/apply-osm":
                    if s.meta.get("kind") == "synthetic":
                        return self._json({"error": "The demo estate is synthetic: its trees do not stand on the real "
                                                    "OSM roads. Load a real survey (--las or --nls-laser) to apply OSM context."}, 400)
                    osm = (app.online or {}).get("osm") or online.osm_context(online.bbox_around(*s.center_lonlat()[::-1], 1500))
                    if not osm.get("ok"):
                        return self._json({"error": osm.get("error", "OSM fetch failed")}, 502)
                    s.ctx.update(online.osm_to_context(osm, s.frame))
                    if s.ctx.get("depot") is None:
                        s.ctx["depot"] = (s.g05.x0 + 5, s.g05.y0 + 5)
                    s.run(s.cond.scenario if s.cond else "normal")
                    if hasattr(s, "_pc_cache"):
                        del s._pc_cache
                    return self._json(s.summary())
                if u.path == "/api/fleet/storm":
                    if app.ds is None:
                        return self._json({"error": "Forey dataset not loaded"}, 404)
                    r = dispatch.storm_response(app.ds, b.get("date", "2026-10-12"), float(b.get("lat", 62.9)),
                                                float(b.get("lon", 27.7)), float(b.get("radius_km", 110)),
                                                float(b.get("gust", 32)))
                    return self._json(r)
                return self._json({"error": "not found"}, 404)
            except (KeyError, ValueError) as exc:
                return self._json({"error": f"bad request: {exc}"}, 400)
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                return self._json({"error": str(exc)}, 500)

        def _static(self, p):
            if p in ("", "/"):
                p = "/index.html"
            f = (WEB / p.lstrip("/")).resolve()
            if WEB not in f.parents or not f.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            return self._send(200, f.read_bytes(), ctype)

    return H


def serve(app: App, host="127.0.0.1", port=8000):
    from . import __version__
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), make_handler(app))
            break
        except OSError:
            if p == port:
                print(f"\n  Port {port} is busy - probably an older Horus still running. Close that window"
                      f" (or press Ctrl+C in it) to use {port}. Trying the next free port ...")
    if httpd is None:
        raise SystemExit(f"no free port in {port}-{port + 19}")
    p = httpd.server_address[1]
    print(f"\n  Horus {__version__} is running:  http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{p}/"
          + (f"?token={app.token}" if app.token else "")
          + "\n  Upload 3D scans with the  [Upload 3D scan]  button (top right) or drop files on the page."
          + "\n  (Ctrl+C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("bye")
