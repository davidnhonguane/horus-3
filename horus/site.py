"""Site pipeline: one surveyed forest area (a Forey flight task) end-to-end.

    point cloud (LAS / synthetic)
      -> terrain + canopy rasters, individual trees, species, health, power line
      -> analysis grids (0.5 m CHM, 1 m assets, 2 m routing, 5 m fire)
      -> weather scenario (live / heatwave / storm)
      -> fire hazard, intensity, risk, response-time map
      -> windthrow, hazard trees, road-blockage & line-outage probabilities
      -> on demand: fire spread simulation, routes per vehicle class
"""
from __future__ import annotations

import csv
import io
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import fire, lidar, render, routing, synth, weather, wind
from .geo import LocalFrame, frame_for_crs, polyline_distance

SPECIES = ("pine", "spruce", "birch")
HEALTH = ("healthy", "stressed", "dead")


def _block(a, f, how="mean"):
    ny, nx = a.shape[0] // f, a.shape[1] // f
    b = a[: ny * f, : nx * f].reshape(ny, f, nx, f)
    return getattr(b, how)(axis=(1, 3))


class Site:
    def __init__(self, name: str, pts: dict, frame, context: dict, truth: dict | None = None,
                 meta: dict | None = None):
        self.name = name
        self.pts = pts
        self.frame = frame
        self.ctx = context  # roads, powerlines, buildings, depot, bog (optional raster fn)
        self.truth = truth
        self.meta = meta or {}
        self.lock = threading.RLock()
        self.timings = {}
        self.cond = None
        self.fire_run = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def synthetic(cls, seed=20260901, lat0=62.747066, lon0=27.259548, task=None, degrade=None):
        """degrade='nls': make the cloud look like NLS national laser scanning 5 p - ~5 pts/m2, laser
        intensity only (no RGB / multispectral), wires and roofs left unclassified."""
        s = synth.generate(seed=seed, lat0=lat0, lon0=lon0)
        if degrade == "nls":
            P = s.points
            rng = np.random.default_rng(5)
            keep = rng.random(P["x"].shape[0]) < 0.75
            P = {k: v[keep] for k, v in P.items()}
            P["classification"] = np.where(np.isin(P["classification"], (3, 5, 6, 14)), 1, P["classification"]).astype(np.uint8)
            P["intensity"] = np.clip(P["intensity"].astype(float) * rng.normal(1, 0.12, P["x"].shape[0]), 0, 65535).astype(np.uint16)
            for k in ("red", "green", "blue", "nir"):
                P.pop(k, None)
            s.points = P
        frame = LocalFrame(lat0, lon0, x0=s.size / 2, y0=s.size / 2)
        bog = s.bog
        ctx = dict(roads=s.roads, powerlines_vector=s.powerlines, buildings=s.buildings, depot=s.depot,
                   bog_fn=lambda x, y: bog[np.clip(y.astype(int), 0, s.size - 1), np.clip(x.astype(int), 0, s.size - 1)],
                   source="synthetic estate (vectors as from NLS topographic DB / Digiroad)")
        meta = dict(kind="synthetic", seed=seed, task=task, degrade=degrade,
                    description=("Synthetic survey degraded to NLS 5 p quality (intensity only)" if degrade == "nls" else
                                 "Synthetic DJI Zenmuse L3 + multispectral survey (deterministic; used for validation)"),
                    stand_types=s.stand_types)
        site = cls(name=f"Demo estate - {task or 'Kuopio'}", pts=s.points, frame=frame, context=ctx,
                   truth=s.trees, meta=meta)
        site.truth_dtm = s.dtm
        site.ortho = _synthetic_ortho(s, seed)
        return site

    @classmethod
    def from_las(cls, path, crs="EPSG:3067", context_geojson=None, max_points=None, osm=False, crop=None,
                 image=None, nls=False, anchor=None):
        """path: one LAS/LAZ file or a list (tiles are merged). crop: (E0, N0, E1, N1) to cut a window.
        osm=True: fetch roads / power lines / buildings / fire station from OpenStreetMap.
        crs="auto": TM35FIN when the coordinates fit Finland and the file has no local CRS, else "local".
        crs="local": scan in its own coordinates (no georeference) placed at anchor=(lat, lon)."""
        from .scan3d import read_points

        paths = [path] if isinstance(path, (str, Path)) else list(path)
        parts = [read_points(p, max_points=max_points) for p in paths]
        keys = set.intersection(*[set(k for k, v in q.items() if isinstance(v, np.ndarray)) for q in parts])
        pts = {k: np.concatenate([q[k] for q in parts]) for k in keys}
        pts["header"] = parts[0].get("header")
        if crop is not None:
            m = (pts["x"] >= crop[0]) & (pts["y"] >= crop[1]) & (pts["x"] < crop[2]) & (pts["y"] < crop[3])
            pts = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in pts.items()}
        if (crs or "").lower() == "auto":
            from .lasio import is_local_crs
            wkt = (pts.get("header") or {}).get("wkt", "")
            mx, my = float(np.median(pts["x"])), float(np.median(pts["y"]))
            fits_fin = 40_000 <= mx <= 800_000 and 6_600_000 <= my <= 7_800_000
            crs = "local" if (is_local_crs(wkt) or not fits_fin) else "EPSG:3067"
        placement = None
        if (crs or "").lower() == "local":
            lat0, lon0 = anchor if anchor else (62.747066, 27.259548)
            cx = float((pts["x"].min() + pts["x"].max()) / 2)
            cy = float((pts["y"].min() + pts["y"].max()) / 2)
            frame = LocalFrame(float(lat0), float(lon0), x0=cx, y0=cy)
            placement = ("local coordinates placed at the given point" if anchor else
                         "local coordinates (no georeference) - shown at the Kuopio demo point; give lat/lon to place it")
            if not anchor and (nls or osm):
                print("  scan has no georeference: NLS / OSM context skipped (give lat/lon to place it)")
                nls = osm = False
        else:
            frame = frame_for_crs(crs)
        ctx = dict(roads=[], powerlines_vector=[], buildings=[], depot=None, bog_fn=None,
                   source="user context GeoJSON" if context_geojson else "none (LiDAR only)")
        if context_geojson:
            ctx.update(load_context(context_geojson, frame))
        if osm:
            from . import online
            cx, cy = float(np.mean(pts["x"])), float(np.mean(pts["y"]))
            lon, lat = frame.to_lonlat(cx, cy)
            half = max(np.ptp(pts["x"]), np.ptp(pts["y"])) / 2 + 100
            o = online.osm_context(online.bbox_around(float(lat), float(lon), half))
            if o.get("ok"):
                ctx.update(online.osm_to_context(o, frame))
            else:
                print("  OpenStreetMap context unavailable:", o.get("error"))
        if ctx["depot"] is None:
            ctx["depot"] = (float(np.min(pts["x"]) + 5), float(np.min(pts["y"]) + 5))
            ctx["depot_auto"] = True      # moved to where the road network enters the scan once roads are known
        elif not (np.min(pts["x"]) <= ctx["depot"][0] <= np.max(pts["x"]) and np.min(pts["y"]) <= ctx["depot"][1] <= np.max(pts["y"])):
            # depot (e.g. fire station) is km away: enter where a road crosses the survey edge
            dx = float(np.clip(ctx["depot"][0], np.min(pts["x"]) + 3, np.max(pts["x"]) - 3))
            dy = float(np.clip(ctx["depot"][1], np.min(pts["y"]) + 3, np.max(pts["y"]) - 3))
            ctx["depot_far"] = ctx["depot"]
            ctx["depot"] = (dx, dy)
        name = Path(paths[0]).stem + (f" (+{len(paths) - 1} tiles)" if len(paths) > 1 else "")
        site = cls(name=name, pts=pts, frame=frame, context=ctx,
                   meta=dict(kind="las", path=[str(p) for p in paths], header=pts.get("header"), crs=crs,
                             placement=placement))
        bb = (float(np.min(pts["x"])), float(np.min(pts["y"])), float(np.max(pts["x"])), float(np.max(pts["y"])))
        from . import imagery, online
        if nls:  # houses, critical infrastructure, roads, power lines from the National Land Survey
            lo0, la0 = frame.to_lonlat(bb[0], bb[1])
            lo1, la1 = frame.to_lonlat(bb[2], bb[3])
            topo = online.nls_topography([float(lo0) - 0.002, float(la0) - 0.001, float(lo1) + 0.002, float(la1) + 0.001])
            site.nls_topo = topo
            if topo.get("ok"):
                ctx.update(online.nls_topo_to_context(topo, frame, ctx))
                if ctx.get("depot") is not None and not (bb[0] <= ctx["depot"][0] <= bb[2] and bb[1] <= ctx["depot"][1] <= bb[3]):
                    ctx["depot_far"] = ctx["depot"]
            else:
                print("  NLS topographic database unavailable:", topo.get("error"))
        try:
            if image:
                site.ortho = imagery.load_image(image, frame)
            elif nls and online.nls_key():
                site.ortho = imagery.nls_ortho(bb, frame)
        except Exception as exc:
            print("  orthophoto unavailable:", exc)
        return site

    # ------------------------------------------------------------------ main pipeline
    def run(self, scenario="normal"):
        t0 = time.time()
        P = self.pts
        x0 = math.floor(P["x"].min())
        y0 = math.floor(P["y"].min())
        w = P["x"].max() - x0
        h = P["y"].max() - y0
        nx = int(math.ceil(w / 5.0)) * 10
        ny = int(math.ceil(h / 5.0)) * 10
        self.g05 = lidar.Grid(x0, y0, 0.5, nx, ny)
        self.g1 = self.g05.coarsen(2)
        self.g2 = self.g05.coarsen(4)
        self.g5 = self.g05.coarsen(10)

        self._label_wires(P)
        dtm, chm, gmask, bld, water, st = lidar.build_surfaces(P, self.g05)
        if self._discover(P, dtm, chm, gmask):          # newly found wires -> rebuild canopy without them
            dtm, chm, gmask, bld, water, st = lidar.build_surfaces(P, self.g05)
        self.dtm05, self.chm05, self.water05 = dtm, chm, water
        self.lidar_stats = st
        self._t("surfaces", t0)

        det, labels = lidar.detect_trees(chm, self.g05)
        det, labels = self._drop_roof_trees(det, labels)
        feats = lidar.tree_features(P, gmask, dtm, labels, self.g05, det)
        calib_idx = calib_sp = None
        self.validation = None
        if self.truth is not None:
            match, v = lidar.validate(det, self.truth)
            m = np.nonzero(match >= 0)[0]
            calib_idx = m[::20]  # ~5 % of matched trees act as field-plot calibration trees
            calib_sp = self.truth["species"][match[calib_idx]]
            self.validation = v
        cl = lidar.classify(det, feats, calib_idx, calib_sp)
        T = dict(det)
        T.update(species=cl["species"], species_conf=cl["species_conf"], health=cl["health"], dbh=cl["dbh"],
                 hd=cl["hd"], ndvi=feats.get("ndvi", np.full(det["x"].shape, np.nan)))
        ix, iy = self.g05.index(T["x"], T["y"])
        T["ground"] = dtm[iy, ix]
        if self.ctx.get("bog_fn") is None and self.ctx.get("bog_polys"):
            self.ctx["bog_fn"] = _poly_mask_fn(self.ctx["bog_polys"], self.g1)
        bogfn = self.ctx.get("bog_fn")
        T["peat"] = bogfn(T["x"], T["y"]).astype(float) if bogfn else np.zeros(T["x"].shape)
        self.trees = T
        self.class_info = dict(species_source=cl["species_source"], health_source=cl["health_source"],
                               multispectral=bool(feats.get("has_multispectral")))
        if self.truth is not None:
            mm = match >= 0
            test = np.ones(mm.sum(), bool)
            test[::20] = False  # exclude calibration trees from accuracy
            ts = self.truth["species"][match[mm]][test]
            ps = T["species"][mm][test]
            th = self.truth["health"][match[mm]][test]
            ph = T["health"][mm][test]
            conf = np.zeros((3, 3), int)
            np.add.at(conf, (ts, ps), 1)
            self.validation.update(species_accuracy=float((ts == ps).mean()), species_confusion=conf.tolist(),
                                   health_accuracy=float((th == ph).mean()),
                                   dead_tree_recall=float(((ph == 2) & (th == 2)).sum() / max((th == 2).sum(), 1)),
                                   dead_tree_precision=float(((ph == 2) & (th == 2)).sum() / max((ph == 2).sum(), 1)),
                                   dtm_mae_m=self._dtm_error(dtm, water))
        self._t("trees", t0)

        # infrastructure: power line from LiDAR (class 14) if present, else vectors
        pls = lidar.extract_powerlines(P)
        vec = self.ctx.get("powerlines_vector") or []
        self.powerlines = []
        if pls:
            for p in pls:
                d, _, _ = polyline_distance(p["line"][:, 0], p["line"][:, 1], p["line"])
                ixp, iyp = self.g05.index(p["line"][:, 0], p["line"][:, 1])
                wz = P["z"][P["classification"] == lidar.CLASS_WIRE]
                wx = P["x"][P["classification"] == lidar.CLASS_WIRE]
                wy = P["y"][P["classification"] == lidar.CLASS_WIRE]
                gx, gy = self.g05.index(wx, wy)
                hc = float(np.percentile(wz - dtm[gy, gx], 20)) if wz.size else 9.0
                meta = vec[0] if vec else {}
                self.powerlines.append(dict(name=meta.get("name", p["name"]), kv=meta.get("kv", 20),
                                            conductor_h=round(hc, 1), row_half=meta.get("row_half", 5.0),
                                            line=p["line"], source=f"LiDAR class 14 ({p['n_points']} conductor points)"))
        else:
            for p in vec:
                q = dict(p)
                q["source"] = "vector data"
                self.powerlines.append(q)
        self.assets = wind.AssetRaster(self.g1.x0, self.g1.y0, self.g1.nx, self.g1.ny, self.ctx["roads"],
                                       self.powerlines, self.ctx["buildings"])
        self._build_grids()
        self._t("grids", t0)
        self.set_scenario(scenario)
        self._t("scenario", t0)
        return self

    def _discover(self, P, dtm, chm, gmask):
        """Find unmapped forest roads, buildings and power lines in the point cloud and add them to the
        context so routing, evacuation, fire and storm models use them."""
        from . import discover
        t0 = time.time()
        g1 = self.g1
        chm1 = _block(chm, 2, "max")
        dtm1 = _block(dtm, 2)
        ix, iy = g1.index(P["x"][gmask], P["y"][gmask])
        gc = np.zeros((g1.ny, g1.nx), bool)
        gc[iy, ix] = True
        known_roads = [r["line"] for r in self.ctx.get("roads", [])]
        known_pl = [p["line"] for p in self.ctx.get("powerlines_vector", []) or []]
        wires = discover.find_wires(P, dtm, self.g05, known_pl)
        road_map = not known_roads           # no map data: draw the whole road network from the scan
        if road_map:
            tracks = discover.find_road_network(chm1, dtm1, gc, g1, (), known_pl + [w["line"] for w in wires])
            new_tracks = [t for t in tracks if not t["mapped"]]
        else:
            tracks = discover.find_tracks(chm1, dtm1, gc, g1, known_roads, known_pl + [w["line"] for w in wires])
            new_tracks = [t for t in tracks if not t["mapped"] and t["confidence"] >= 0.9]
        blds = discover.find_buildings(P, dtm, self.g05, self.ctx.get("buildings", []))
        new_blds = [b for b in blds if not b["mapped"]]
        new_wires = [w for w in wires if not w["mapped"]]
        self.ctx.setdefault("roads", [])
        for t in new_tracks:  # connect track ends to the road network they branch from (< 25 m gap)
            line = t["line"]
            for end in (0, -1):
                best = None
                for kl in known_roads:
                    d, cx, cy = polyline_distance(line[end:end + 1, 0] if end == 0 else line[-1:, 0],
                                                  line[end:end + 1, 1] if end == 0 else line[-1:, 1], kl)
                    if d[0] < 25 and (best is None or d[0] < best[0]):
                        best = (d[0], cx[0], cy[0])
                if best:
                    pt = np.array([[best[1], best[2]]])
                    line = np.vstack([pt, line]) if end == 0 else np.vstack([line, pt])
            t["line"] = line
        for k, t in enumerate(new_tracks, 1):
            if road_map:
                self.ctx["roads"].append(dict(name=f"{'Road' if t['cls'] == 'main' else 'Forest road'} found in 3D scan #{k}"
                                              + (" (inferred link across open ground)" if t.get("inferred") else ""),
                                              cls=t["cls"], width=t["width_m"], line=t["line"], discovered=True))
            else:
                self.ctx["roads"].append(dict(name=f"Potential forest road #{k} (LiDAR, unmapped)", cls="forest",
                                              width=max(3.0, min(t["width_m"], 5.0)), line=t["line"], discovered=True))
        self.ctx.setdefault("buildings", [])
        for k, b in enumerate(new_blds, 1):
            self.ctx["buildings"].append(dict(name=f"Building in forest #{k} (LiDAR, unmapped)", kind="cabin",
                                              x=b["x"], y=b["y"], r=max(4.0, b["r"]), weight=0.8, discovered=True))
        relabel = False
        if new_wires and not known_pl and not (P.get("classification") is not None and
                                              (P["classification"] == lidar.CLASS_WIRE).sum() > 200):
            self.ctx["powerlines_vector"] = [dict(name=f"Power line #{k} (discovered in LiDAR, unmapped)", kv=20,
                                                  conductor_h=w["height_m"], row_half=5.0, line=w["line"])
                                             for k, w in enumerate(new_wires, 1)]
            relabel = self._label_wires(P) > 0
        # ---- 2-D imagery: roads + power-line corridors, fused with the LiDAR finds
        img = None
        if getattr(self, "ortho", None) is not None:
            from . import imagery
            try:
                img = imagery.analyse(self.ortho, self.g05, known_roads, known_pl + [w["line"] for w in wires])
            except Exception as exc:  # imagery must never break the LiDAR pipeline
                img = dict(error=str(exc), roads=[], corridors=[])
        img_roads = (img or {}).get("roads", [])
        used = set()
        for t in tracks:
            t["evidence"] = "LiDAR"
            for j, r in enumerate(img_roads):
                d, _, _ = polyline_distance(t["line"][:, 0], t["line"][:, 1], r["line"])
                if np.mean(d < 8) > 0.5:
                    t["evidence"] = "LiDAR + image"
                    t["confidence"] = round(min(1.0, (t.get("confidence") or 0.9) + 0.05), 2)
                    r["evidence"] = "LiDAR + image"
                    used.add(j)
        img_only = [r for j, r in enumerate(img_roads) if j not in used and not r["mapped"] and r["confidence"] >= 0.9]
        for r in img_roads:
            r.setdefault("evidence", "image")
        k0 = len([r for r in self.ctx["roads"] if r.get("discovered")])
        for k, r in enumerate(img_only, k0 + 1):
            self.ctx["roads"].append(dict(name=f"Potential forest road #{k} (image, unmapped)", cls="forest",
                                          width=max(3.0, min(r["width_m"], 5.0)), line=r["line"], discovered=True))
        corridors = (img or {}).get("corridors", [])
        for c in corridors:
            c["evidence"] = "image"
            for line in [w["line"] for w in wires] + known_pl:
                d, _, _ = polyline_distance(np.linspace(c["line"][0, 0], c["line"][1, 0], 15),
                                            np.linspace(c["line"][0, 1], c["line"][1, 1], 15), line)
                if np.mean(d < 15) > 0.6:
                    c["evidence"] = "image + LiDAR wires" if any(line is w["line"] for w in wires) else "image + map"
            c["mapped"] = c["evidence"] != "image"
        for w in wires:
            w["evidence"] = "LiDAR"
            for c in corridors:
                d, _, _ = polyline_distance(np.linspace(c["line"][0, 0], c["line"][1, 0], 15),
                                            np.linspace(c["line"][0, 1], c["line"][1, 1], 15), w["line"])
                if np.mean(d < 15) > 0.6:
                    w["evidence"] = "LiDAR + image"
        self.image_analysis = img
        self.discovered = dict(tracks=tracks, buildings=blds, wires=wires, image_roads=img_roads, corridors=corridors,
                               road_map_from_scan=road_map,
                               road_km=round(sum(t["length_m"] for t in new_tracks) / 1000, 2),
                               unmapped=dict(roads=len(new_tracks) + len(img_only), buildings=len(new_blds),
                                             power_lines=len(new_wires),
                                             line_corridors=len([c for c in corridors if not c["mapped"]])),
                               seconds=round(time.time() - t0, 1))
        return relabel

    def _label_wires(self, P):
        """National scans leave conductors unclassified: points hanging 4-25 m up within 2.5 m of a known
        power line (NLS topographic DB / OSM) are labelled LAS class 14 so they are not taken for trees."""
        cls = P.get("classification")
        vec = self.ctx.get("powerlines_vector") or []
        if cls is None or not vec or (cls == lidar.CLASS_WIRE).sum() > 200:
            return 0
        n = 0
        g = cls == lidar.CLASS_GROUND
        gz = np.percentile(P["z"][g], 50) if g.any() else np.percentile(P["z"], 5)
        cand = np.nonzero(~g & ~np.isin(cls, (lidar.CLASS_WATER,)))[0]
        for p in vec:
            line = p["line"]
            x0, y0 = line.min(0) - 5
            x1, y1 = line.max(0) + 5
            c = cand[(P["x"][cand] > x0) & (P["x"][cand] < x1) & (P["y"][cand] > y0) & (P["y"][cand] < y1)]
            if c.size == 0:
                continue
            d, _, _ = polyline_distance(P["x"][c], P["y"][c], line)
            # local ground from a coarse minimum is good enough to test the conductor height band
            hz = P["z"][c] - gz
            w = c[(d < 2.5) & (hz > 4) & (hz < 40)]
            # keep genuine tree crowns: wires are sparse, crowns dense - require few neighbours above
            cls[w] = lidar.CLASS_WIRE
            n += w.size
        self.wire_points_labelled = int(n)
        return n

    def _drop_roof_trees(self, det, labels):
        """Unclassified roofs look like tree crowns: remove detections on known building footprints."""
        b = self.ctx.get("buildings") or []
        if not b or self.pts.get("classification") is not None and (self.pts["classification"] == lidar.CLASS_BUILDING).any():
            return det, labels
        bx = np.array([q["x"] for q in b])
        by = np.array([q["y"] for q in b])
        br = np.array([q.get("r", 6.0) for q in b]) + 2.0
        from scipy.spatial import cKDTree
        kd = cKDTree(np.stack([bx, by], 1))
        d, j = kd.query(np.stack([det["x"], det["y"]], 1))
        keep = d > br[j]
        self.roof_trees_removed = int((~keep).sum())
        if keep.all():
            return det, labels
        idx = np.nonzero(keep)[0]
        remap = np.full(det["x"].shape[0] + 1, -1, np.int32)
        remap[idx] = np.arange(idx.size)
        labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1)
        return {k: v[keep] for k, v in det.items()}, labels

    def _dtm_error(self, dtm, water):
        ref = getattr(self, "truth_dtm", None)
        if ref is None:
            return None
        d = _block(dtm, 2)
        ny, nx = min(d.shape[0], ref.shape[0]), min(d.shape[1], ref.shape[1])
        dry = ~_block(water, 2, "max")[:ny, :nx]
        return float(np.abs(d[:ny, :nx] - ref[:ny, :nx])[dry].mean())

    def _t(self, k, t0):
        self.timings[k] = round(time.time() - t0, 2)

    def _build_grids(self):
        g5, g2 = self.g5, self.g2
        chm, dtm = self.chm05, self.dtm05
        G = {}
        G["cover"] = _block((chm > 2).astype(float), 10)
        hsum = _block(np.where(chm > 2, chm, 0), 10, "sum")
        G["mean_h"] = hsum / np.maximum(_block((chm > 2).astype(float), 10, "sum"), 1)
        G["dtm"] = _block(dtm, 10)
        G["water"] = _block(self.water05.astype(float), 10) > 0.5
        T = self.trees
        ix, iy = g5.index(T["x"], T["y"])
        wgt = T["crown_area"]
        tot = np.zeros((g5.ny, g5.nx))
        np.add.at(tot, (iy, ix), wgt)
        sm = lambda a: ndi.gaussian_filter(a, 1.0)
        tot_s = sm(tot) + 1e-6
        for k, sp in enumerate(SPECIES):
            a = np.zeros_like(tot)
            np.add.at(a, (iy, ix), wgt * (T["species"] == k))
            G[sp] = sm(a) / tot_s
        G["conifer"] = G["pine"] + G["spruce"]
        a = np.zeros_like(tot)
        np.add.at(a, (iy, ix), wgt * (T["health"] == 2))
        G["dead"] = sm(a) / tot_s
        a = np.zeros_like(tot)
        np.add.at(a, (iy, ix), 1.0)
        G["stems_ha"] = sm(a) * 10000 / (g5.res ** 2)
        xs, ys = g5.centers()
        bogfn = self.ctx.get("bog_fn")
        G["bog"] = bogfn(xs, ys).astype(bool) if bogfn else np.zeros(xs.shape, bool)
        A = self.assets
        cls_of_piece = np.array([1 if p["cls"] == "main" else 2 for p in A.road_pieces] + [0], np.int8)
        road_cls1 = np.where(A.road >= 0, cls_of_piece[A.road], 0).astype(np.int8)
        self.road_cls1 = road_cls1
        G["road_main"] = _block((road_cls1 == 1).astype(float), 5) > 0.35
        G["road_forest"] = _block((road_cls1 == 2).astype(float), 5) > 0.2
        G["building"] = _block((A.bld >= 0).astype(float), 5) > 0.3
        human = (road_cls1 > 0) | (A.bld >= 0) | (A.power >= 0)
        G["d_human"] = _block(ndi.distance_transform_edt(~human), 5) if human.any() else np.full(G["cover"].shape, 1e3)
        G["d_building"] = (_block(ndi.distance_transform_edt(A.bld < 0), 5) if (A.bld >= 0).any()
                           else np.full(G["cover"].shape, 1e3))
        G["d_power"] = (_block(ndi.distance_transform_edt(A.power < 0), 5) if (A.power >= 0).any()
                        else np.full(G["cover"].shape, 1e3))
        gy_, gx_ = np.gradient(G["dtm"], g5.res)
        G["slope_deg"] = np.degrees(np.arctan(np.hypot(gx_, gy_)))
        self.G5 = G
        self.fuel = fire.fuel_map(G)
        # open / grass fuel share within ~15 m (cured grass carries fast surface fire into crowns)
        G["grass"] = ndi.uniform_filter((self.fuel == fire.FUELS.index("O1")).astype(float), 3)

        # 2 m routing layers
        dtm2 = _block(dtm, 4)
        gy2, gx2 = np.gradient(dtm2, g2.res)
        R = dict(road_cls=np.maximum(_block((road_cls1 == 1).astype(float), 2, "max") * 1,
                                     _block((road_cls1 == 2).astype(float), 2, "max") * 2).astype(np.int8),
                 slope_deg=np.degrees(np.arctan(np.hypot(gx2, gy2))),
                 cover=ndi.uniform_filter(_block((chm > 2).astype(float), 4), 3),
                 water=_block(self.water05.astype(float), 4) > 0.5,
                 bog=np.repeat(np.repeat(G["bog"], 5, 0), 5, 1)[: g2.ny, : g2.nx] if G["bog"].shape[0] * 5 >= g2.ny else np.zeros((g2.ny, g2.nx), bool))
        pw = np.array([float(p.get("width", 4.0)) for p in A.road_pieces] + [np.inf])
        road_w1 = np.where(A.road >= 0, pw[A.road], np.inf)
        R["road_w"] = _block(road_w1, 2, "min")                       # narrowest road width in each 2 m cell
        self.road_w1 = road_w1
        main = _block((road_cls1 == 1).astype(float), 2, "max") > 0
        R["road_cls"] = np.where(main, 1, R["road_cls"]).astype(np.int8)
        R["water"] &= R["road_cls"] == 0  # bridges / culverts
        self.router = routing.Router(R, g2.res, g2.x0, g2.y0)
        if self.ctx.get("depot_far") is not None and (R["road_cls"] > 0).any():
            # rescue arrives from a fire station outside the survey: enter where a road crosses the edge
            edge = np.zeros_like(R["road_cls"], bool)
            edge[:2, :] = edge[-2:, :] = True
            edge[:, :2] = edge[:, -2:] = True
            cand = np.argwhere(edge & (R["road_cls"] > 0))
            if cand.size:
                xs = g2.x0 + (cand[:, 1] + 0.5) * g2.res
                ys = g2.y0 + (cand[:, 0] + 0.5) * g2.res
                fx, fy = self.ctx["depot_far"]
                k = int(np.argmin((xs - fx) ** 2 + (ys - fy) ** 2 - 1e6 * (R["road_cls"][cand[:, 0], cand[:, 1]] == 1)))
                self.ctx["depot"] = (float(xs[k]), float(ys[k]))
        if self.ctx.get("depot_auto"):
            self.ctx["depot"] = self._auto_depot(R)
        self.depot = self.ctx["depot"]
        self.response = self.router.response_time_map(self.depot)  # minutes on 2 m grid
        self.response5 = self._resp_to_5m()

    def _auto_depot(self, R):
        """No map data: rescue arrives where the largest road network meets the scan edge (else the edge
        cell of the largest area a 4x4 / ATV can drive)."""
        g2 = self.g2
        roads = R["road_cls"] > 0
        if roads.any():
            lab, n = ndi.label(roads, structure=np.ones((3, 3)))
            size = np.bincount(lab.ravel(), weights=np.where(R["road_cls"] == 1, 3.0, 1.0).ravel())
            size[0] = 0
            comp = lab == int(np.argmax(size))
        else:
            sp = self.router.speed("atv") > 0
            lab, n = ndi.label(sp)
            if n == 0:
                return self.ctx["depot"]
            size = np.bincount(lab.ravel())
            size[0] = 0
            comp = lab == int(np.argmax(size))
        yy, xx = np.nonzero(comp)
        edge_d = np.minimum.reduce([yy, xx, g2.ny - 1 - yy, g2.nx - 1 - xx])
        k = int(np.argmin(edge_d))
        return (float(g2.x0 + (xx[k] + 0.5) * g2.res), float(g2.y0 + (yy[k] + 0.5) * g2.res))

    def _resp_to_5m(self):
        r = np.where(np.isfinite(self.response), self.response, 240.0)
        # 2 m -> 5 m: sample cell centres
        xs, ys = self.g5.centers()
        ix, iy = self.g2.index(xs, ys)
        return r[iy, ix]

    # ------------------------------------------------------------------ scenario-dependent
    def set_scenario(self, scenario="normal", gust=None, wind_dir=None):
        with self.lock:
            lon, lat = self.center_lonlat()
            self.cond = weather.get_conditions(scenario, lat, lon, gust=gust, wind_dir=wind_dir)
            c = self.cond
            T = self.trees
            self.ros, self.hfi = fire.head_ros(self.fuel, self.G5, c)
            self.risk = fire.risk_index(self.G5, self.hfi, self.response5)
            T["edge"], T["emergent"] = wind.exposure_features(T, self.chm05, self.g05, c.wind_dir_deg)
            T["p_fail_now"] = wind.failure_probability(T, c.gust_ms, c.soil_wet, c.soil_frozen)
            T["p_fail_design"] = wind.failure_probability(T, wind.DESIGN_GUST, 0.9, False)
            hits, self.road_p, self.power_p, self.bld_p = wind.strike_analysis(
                T, self.assets, T["p_fail_now"], c.wind_dir_deg, self.ctx["buildings"])
            T.update(hits)
            bw = np.array([b.get("weight", 1.0) for b in self.ctx["buildings"]] or [1.0])
            # consequence weights: line outage 1.0, building 1.0 (x building weight), road blockage 0.5 x size
            wr = 0.5 * T["w_road"] * T["road_size_factor"]
            T["risk_design"] = T["p_fail_design"] * (1.0 * T["w_power"] + 1.0 * T["w_bld"] + wr)
            T["risk_now"] = T["p_fail_now"] * (1.0 * T["w_power"] + 1.0 * T["w_bld"] + wr)
            ix5, iy5 = self.g5.index(T["x"], T["y"])
            T["fire_hazard"], T["p_torch"], T["cbh"], _ = fire.tree_fire_hazard(
                T, self.hfi[iy5, ix5], self.G5["grass"][iy5, ix5], c)
            self.fire_run = None
            self._views = None
            self._view_png = {}
            return self.summary()

    # ------------------------------------------------------------------ on-demand analyses
    def blocked_cells(self, threshold=0.5):
        """2 m raster of road cells belonging to pieces whose blockage probability >= threshold
        (band of +-3 m around the most likely fall point = piece midpoint)."""
        A = self.assets
        band = np.zeros((self.g2.ny, self.g2.nx), bool)
        for pid, p in enumerate(A.road_pieces):
            if self.road_p[pid] >= threshold:
                mx, my = p["mid"]
                ix, iy = self.g2.index(mx, my)
                r = int(math.ceil((p["width"] / 2 + 3) / self.g2.res))
                y0, y1 = max(0, iy - r), min(self.g2.ny, iy + r + 1)
                x0, x1 = max(0, ix - r), min(self.g2.nx, ix + r + 1)
                band[y0:y1, x0:x1] |= self.router.L["road_cls"][y0:y1, x0:x1] > 0
        return band

    def simulate_fire(self, lon, lat, hours=4.0):
        with self.lock:
            x, y = self.frame.from_lonlat(lon, lat)
            ix, iy = self.g5.index(float(x), float(y))
            arr = fire.simulate(self.ros, self.G5["dtm"], self.g5.res, self.cond, [(int(ix), int(iy))],
                                max_minutes=hours * 60)
            self.fire_run = dict(arrival=arr, lon=lon, lat=lat, hours=hours)
            burned = np.isfinite(arr)
            area = {f"{h}h": float(np.sum(arr <= h * 60) * self.g5.res ** 2 / 1e4) for h in (0.5, 1, 2, 4) if h <= hours}
            impacts = []
            for b in self.ctx["buildings"]:
                bx, by = self.g5.index(b["x"], b["y"])
                r = int(math.ceil((b["r"] + 15) / self.g5.res))
                sub = arr[max(0, by - r):by + r + 1, max(0, bx - r):bx + r + 1]
                t = float(np.nanmin(np.where(np.isfinite(sub), sub, np.nan))) if np.isfinite(sub).any() else None
                impacts.append(dict(asset=b["name"], kind=b["kind"], minutes=t))
            for p in self.powerlines:
                xs, ys = p["line"][:, 0], p["line"][:, 1]
                from .synth import _densify
                d = _densify(p["line"], 5)
                gx, gy = self.g5.index(d[:, 0], d[:, 1])
                tt = arr[gy, gx]
                impacts.append(dict(asset=p["name"], kind="power_line",
                                    minutes=float(tt[np.isfinite(tt)].min()) if np.isfinite(tt).any() else None))
            for r in self.ctx["roads"]:
                from .synth import _densify
                d = _densify(r["line"], 5)
                gx, gy = self.g5.index(d[:, 0], d[:, 1])
                tt = arr[gy, gx]
                impacts.append(dict(asset=r["name"], kind="road_cut",
                                    minutes=float(tt[np.isfinite(tt)].min()) if np.isfinite(tt).any() else None))
            impacts.sort(key=lambda d: (d["minutes"] is None, d["minutes"] or 0))
            ii = fire.intensity_class(self.hfi[burned]) if burned.any() else np.array([], int)
            trees = self._fire_trees(arr)
            from . import costs
            burned_ha = float(np.sum(np.isfinite(arr)) * self.g5.res ** 2 / 1e4)
            dmg = costs.fire_cost(self.costs(), self.trees, np.array(trees["ids"], int), trees["p_torch"], burned_ha,
                                  impacts, self.ctx["buildings"])
            dmg["burned_ha"] = round(burned_ha, 2)
            return dict(ignition=[lon, lat], hours=hours, area_ha=area, trees=trees, damage_eur=dmg,
                        max_intensity_kw_m=float(self.hfi[burned].max()) if burned.any() else 0,
                        intensity_share={fire.INTENSITY_LABELS[k]: float(np.mean(ii == k)) for k in range(6)} if ii.size else {},
                        impacts=impacts, png_url=f"/api/site/layer/fire_arrival.png?t={time.time():.0f}")

    def _fire_trees(self, arr):
        """Trees the simulated fire reaches, when, and which ones torch (crown fire)."""
        T = self.trees
        ix, iy = self.g5.index(T["x"], T["y"])
        t = arr[iy, ix]
        hit = np.nonzero(np.isfinite(t))[0]
        pt = T["p_torch"][hit]
        top = hit[np.argsort(-(T["fire_hazard"][hit] + 50 * pt))][:15]
        lon, lat = self.frame.to_lonlat(T["x"][top], T["y"][top])
        g = self.G5["grass"][iy, ix]
        return dict(
            reached=int(hit.size), torching=int((pt >= 0.5).sum()), expected_crown_fires=float(pt.sum()),
            dead_reached=int((T["health"][hit] == 2).sum()),
            by_species={sp: int((T["species"][hit] == k).sum()) for k, sp in enumerate(SPECIES)},
            ids=hit.tolist(), minutes=np.round(t[hit], 1).tolist(), p_torch=np.round(pt, 3).tolist(),
            worst=[dict(id=int(i), lon=float(lo), lat=float(la), species=SPECIES[T["species"][i]], health=HEALTH[T["health"][i]],
                        height_m=round(float(T["h"][i]), 1), crown_base_m=round(float(T["cbh"][i]), 1),
                        minutes=round(float(t[i])), p_torch=round(float(T["p_torch"][i]), 2),
                        fire_hazard=round(float(T["fire_hazard"][i])),
                        why=fire.hazard_reason(T["species"][i], T["health"][i], T["cbh"][i], g[i], T["p_torch"][i]))
                   for i, lo, la in zip(top, lon, lat)])

    def fire_prone_trees(self, top=25):
        """Trees that burn most easily today: dead / dry, resinous, low crowns, dry grass around."""
        T = self.trees
        ix, iy = self.g5.index(T["x"], T["y"])
        g = self.G5["grass"][iy, ix]
        order = np.argsort(-(T["fire_hazard"] + 30 * T["p_torch"]))[:top]
        lon, lat = self.frame.to_lonlat(T["x"][order], T["y"][order])
        near = lambda i: (T["target"][i] or "-")  # noqa: E731
        return dict(
            dryness=round(fire.dryness(self.cond), 2), ffmc=self.cond.ffmc, bui=self.cond.bui,
            very_high=int((T["fire_hazard"] >= 70).sum()), high=int(((T["fire_hazard"] >= 50) & (T["fire_hazard"] < 70)).sum()),
            torch_if_reached=int((T["p_torch"] >= 0.5).sum()), dead=int((T["health"] == 2).sum()),
            grass_ha=float((self.G5["grass"] > 0.5).sum() * self.g5.res ** 2 / 1e4 * 1.0),
            trees=[dict(id=int(i), lon=float(lo), lat=float(la), species=SPECIES[T["species"][i]], health=HEALTH[T["health"][i]],
                        height_m=round(float(T["h"][i]), 1), crown_base_m=round(float(T["cbh"][i]), 1),
                        fire_hazard=round(float(T["fire_hazard"][i])), p_torch=round(float(T["p_torch"][i]), 2),
                        near=near(i), why=fire.hazard_reason(T["species"][i], T["health"][i], T["cbh"][i], g[i], T["p_torch"][i]))
                   for i, lo, la in zip(order, lon, lat)])

    def simulate_storm(self, gust=None, wind_dir=None, runs=300, seed=None):
        """Monte-Carlo storm: which trees fall at this gust, where they land, what they hit.
        Every run draws each tree's failure (windthrow model) and, for fallen trees, whether it lands
        on a power line / building / road (fall-direction distribution around the downwind bearing)."""
        with self.lock:
            c = self.cond
            gust = float(c.gust_ms if gust in (None, "") else gust)
            wdir = float(c.wind_dir_deg if wind_dir in (None, "") else wind_dir)
            T = self.trees
            Tc = dict(T)
            Tc["edge"], Tc["emergent"] = wind.exposure_features(T, self.chm05, self.g05, wdir)
            p = wind.failure_probability(Tc, gust, c.soil_wet, c.soil_frozen)
            hits, road_p, power_p, bld_p = wind.strike_analysis(Tc, self.assets, p, wdir, self.ctx["buildings"])
            n = p.size
            rng = np.random.default_rng(seed if seed is not None else 1)
            runs = int(np.clip(runs, 20, 2000))
            from . import costs
            cfg = self.costs()
            vol = costs.tree_volume_m3(T)
            n_line = np.zeros(runs)
            fvol = np.zeros(runs)
            fallen_n = np.zeros(runs)
            line_hit = np.zeros(runs, bool)
            bld_hit = np.zeros(runs)
            road_hit = np.zeros(runs)
            freq = np.zeros(n)
            best = None
            for r in range(runs):
                f = rng.random(n) < p
                lands = rng.random(n)
                hp = f & (lands < hits["w_power"])
                hb = f & ~hp & (lands < hits["w_power"] + hits["w_bld"])
                hr = f & ~hp & ~hb & (lands < hits["w_power"] + hits["w_bld"] + hits["w_road"])
                fallen_n[r], line_hit[r], bld_hit[r], road_hit[r] = f.sum(), hp.any(), hb.sum(), hr.sum()
                n_line[r], fvol[r] = hp.sum(), vol[f].sum()
                freq += f
                if r == 0 or abs(f.sum() - np.median(fallen_n[:r + 1])) < abs(best[0].sum() - np.median(fallen_n[:r + 1])):
                    best = (f, hp, hb, hr)
            # one realisation for the map: base -> crown tip in the fall direction
            f, hp, hb, hr = best
            self.storm_run = dict(p=p, fallen=f, gust=gust, wind_from=wdir)
            idx = np.nonzero(f)[0][:4000]
            down = math.radians(90 - (wdir + 180))
            th = down + rng.vonmises(0, 2.0, idx.size)
            tipx = T["x"][idx] + np.cos(th) * T["h"][idx]
            tipy = T["y"][idx] + np.sin(th) * T["h"][idx]
            lo0, la0 = self.frame.to_lonlat(T["x"][idx], T["y"][idx])
            lo1, la1 = self.frame.to_lonlat(tipx, tipy)
            kind = np.where(hp[idx], 2, np.where(hb[idx], 3, np.where(hr[idx], 1, 0)))
            damage = costs.summarise(costs.storm_run_cost(cfg, line_hit.astype(float), n_line, road_hit, bld_hit, fvol))
            dmg_tree = costs.tree_expected_damage(cfg, T, p, hits["w_power"], hits["w_bld"], hits["w_road"])
            fell = v_fell = costs.v(cfg, "felling_cost_per_tree")
            worth = dmg_tree > fell
            top = np.argsort(-dmg_tree)[:10]
            tlo, tla = self.frame.to_lonlat(T["x"][top], T["y"][top])
            damage.update(
                trees_worth_felling=int(worth.sum()),
                felling_cost=float(worth.sum() * fell),
                damage_avoided_by_felling=float(dmg_tree[worth].sum()),
                most_costly_trees=[dict(id=int(i), lon=float(lo), lat=float(la), species=SPECIES[T["species"][i]],
                                        height_m=round(float(T["h"][i]), 1), p_fall=round(float(p[i]), 3),
                                        threatens=hits["target"][i] or "-", damage_eur=round(float(dmg_tree[i])),
                                        net_benefit_eur=round(float(dmg_tree[i] - v_fell)))
                                   for i, lo, la in zip(top, tlo, tla)])
            gusts = np.arange(10, 42, 2.0)
            curve = []
            for gg in gusts:
                pg = wind.failure_probability(Tc, gg, c.soil_wet, c.soil_frozen)
                curve.append(dict(gust=float(gg), fallen=float(pg.sum()),
                                  p_line_outage=float(1 - np.prod(1 - np.clip(pg * hits["w_power"], 0, 1)))))
            order = np.argsort(-p)[:15]
            lon, lat = self.frame.to_lonlat(T["x"][order], T["y"][order])
            why = []
            for i in order:
                w = []
                if T["health"][i] == 2:
                    w.append("dead")
                elif T["health"][i] == 1:
                    w.append("stressed")
                if T["species"][i] == 1:
                    w.append("shallow-rooted spruce")
                if T["hd"][i] > 90:
                    w.append(f"slender (H/D {T['hd'][i]:.0f})")
                if Tc["edge"][i] > 0.5:
                    w.append("exposed edge facing the wind")
                if Tc["emergent"][i] > 0.5:
                    w.append("taller than neighbours")
                if T["peat"][i] > 0:
                    w.append("peat soil")
                why.append(w)
            return dict(
                gust_ms=gust, wind_from=wdir, runs=runs, damage_eur=damage,
                fallen=dict(mean=float(fallen_n.mean()), p5=float(np.percentile(fallen_n, 5)), p95=float(np.percentile(fallen_n, 95))),
                p_line_outage=float(line_hit.mean()), buildings_hit_mean=float(bld_hit.mean()),
                p_any_building_hit=float((bld_hit > 0).mean()), trees_on_roads_mean=float(road_hit.mean()),
                road_pieces_over_50=int((road_p >= 0.5).sum()),
                easily_fall=dict(over_50=int((p >= 0.5).sum()), over_20=int((p >= 0.2).sum()), over_5=int((p >= 0.05).sum())),
                curve=curve,
                fall_lines=dict(lon0=np.round(lo0, 7).tolist(), lat0=np.round(la0, 7).tolist(), lon1=np.round(lo1, 7).tolist(),
                                lat1=np.round(la1, 7).tolist(), kind=kind.tolist()),
                p_fall=np.round(p, 4).tolist(),
                worst=[dict(id=int(i), lon=float(lo), lat=float(la), species=SPECIES[T["species"][i]], health=HEALTH[T["health"][i]],
                            height_m=round(float(T["h"][i]), 1), p_fall=round(float(p[i]), 3),
                            threatens=hits["target"][i] or "-", why=w) for i, lo, la, w in zip(order, lon, lat, why)])

    def costs(self):
        if getattr(self, "_costs", None) is None:
            from . import costs
            self._costs = costs.load()
        return self._costs

    # ------------------------------------------------------------------ simulations shown on the 3D scan
    def auto_simulations(self, hours=2.0, runs=200):
        """Run on every new scan: a storm at the design gust and a fire started at the most dangerous spot
        (highest fire-risk cell, preferring cells near houses, roads and lines) in the current conditions."""
        with self.lock:
            c = self.cond
            risk = np.where(self.fuel > 0, self.risk, -1)
            dh = self.G5["d_human"]
            near = (dh >= 40) & (dh < 150)                  # in the forest, close enough to threaten people / lines
            score = risk + 25 * near - 50 * (dh < 25)
            iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
            x = self.g5.x0 + (ix + 0.5) * self.g5.res
            y = self.g5.y0 + (iy + 0.5) * self.g5.res
            lon, lat = self.frame.to_lonlat(x, y)
        fire_r = self.simulate_fire(float(lon), float(lat), hours)
        storm_r = self.simulate_storm(max(wind.DESIGN_GUST, c.gust_ms), c.wind_dir_deg, runs)
        first = [i for i in fire_r["impacts"] if i["minutes"] is not None][:5]
        return dict(
            fire=dict(ignition=fire_r["ignition"], hours=hours, area_ha=fire_r["area_ha"], trees=fire_r["trees"],
                      first_reached=first, max_intensity_kw_m=fire_r["max_intensity_kw_m"], png_url=fire_r["png_url"],
                      damage_eur=fire_r["damage_eur"],
                      conditions=c.scenario, fwi=c.fwi),
            storm=storm_r)

    def points_sim_bin(self):
        """Per 3D point (same order as points.bin): uint8 fire (0 = not reached, 1..255 = arrival time
        from early to late) + uint8 storm (255 = tree falls in the simulated storm, else P(fall) x 200)."""
        if getattr(self, "_pc", None) is None:
            self.point_cloud_bin()
        sel, j, veg = self._pc["sel"], self._pc["j"], self._pc["veg"]
        P = self.pts
        n = sel.size
        fire_b = np.zeros(n, np.uint8)
        if self.fire_run is not None:
            gx, gy = self.g5.index(P["x"][sel], P["y"][sel])
            a = self.fire_run["arrival"][gy, gx]
            ok = np.isfinite(a)
            fire_b[ok] = 1 + np.clip(a[ok] / max(self.fire_run["hours"] * 60, 1) * 254, 0, 254).astype(np.uint8)
        storm_b = np.zeros(n, np.uint8)
        sr = getattr(self, "storm_run", None)
        if sr is not None:
            jj = j[veg]
            v = np.where(sr["fallen"][jj], 255, np.clip(sr["p"][jj] * 200, 0, 200)).astype(np.uint8)
            storm_b[veg] = v
        head = json.dumps(dict(n=int(n), fire=self.fire_run is not None, storm=sr is not None,
                               fire_hours=self.fire_run["hours"] if self.fire_run is not None else None,
                               storm_gust=sr["gust"] if sr else None)).encode()
        return np.uint32(len(head)).tobytes() + head + fire_b.tobytes() + storm_b.tobytes()

    def highlights(self):
        """Newly found infrastructure (not in any map data) for highlighting: 2D (lon/lat) and 3D (viewer xyz)."""
        if getattr(self, "_pc", None) is None:
            self.point_cloud_bin()
        out = []
        from .synth import _densify
        for r in self.ctx.get("roads", []):
            if not r.get("discovered"):
                continue
            ln = _densify(np.asarray(r["line"], float), 1.0)
            ix, iy = self.g05.index(ln[:, 0], ln[:, 1])
            z = self.dtm05[iy, ix] + 0.5
            out.append(dict(kind="road", name=r["name"], width_m=r.get("width"), length_m=round(float(np.hypot(*np.diff(ln, axis=0).T).sum())),
                            path=self.ll(ln[:: max(1, len(ln) // 60)]), xyz=np.round(_viewer_xyz(self, ln[:, 0], ln[:, 1], z), 2).tolist()))
        wires = [w for w in (getattr(self, "discovered", None) or {}).get("wires", []) if not w.get("mapped")]
        lines = [dict(line=w["line"], h=w.get("height_m", 9.0), name="Power line found in scan") for w in wires]
        if not lines:
            lines = [dict(line=p["line"], h=p.get("conductor_h", 9.0), name=p["name"]) for p in self.powerlines if "discovered" in p["name"]]
        for w in lines:
            ln = _densify(np.asarray(w["line"], float), 1.0)
            ix, iy = self.g05.index(ln[:, 0], ln[:, 1])
            z = self.dtm05[iy, ix] + float(w["h"] or 9.0)
            out.append(dict(kind="power", name=w["name"], height_m=round(float(w["h"] or 9.0), 1),
                            length_m=round(float(np.hypot(*np.diff(ln, axis=0).T).sum())),
                            path=self.ll(ln[:: max(1, len(ln) // 60)]), xyz=np.round(_viewer_xyz(self, ln[:, 0], ln[:, 1], z), 2).tolist()))
        return out

    def route(self, vehicle, start, end, block_threshold=0.5, avoid_fire_min=None, storm=True):
        with self.lock:
            sx, sy = (self.depot if start == "depot" else self.frame.from_lonlat(*start))
            ex, ey = (self.depot if end == "depot" else self.frame.from_lonlat(*end))
            blocks = self.blocked_cells(block_threshold) if storm else None
            barrier = None
            fire_note = None
            if avoid_fire_min is not None and self.fire_run is not None:
                # leaving now: avoid every cell the fire reaches within the safety margin
                arr = self.fire_run["arrival"]
                b5 = ndi.binary_dilation(arr <= avoid_fire_min, iterations=2)
                xs, ys = self.g2.centers()
                ix, iy = self.g5.index(xs, ys)
                barrier = b5[iy, ix]
                fire_note = f"avoids area burning within {avoid_fire_min:.0f} min (+10 m buffer)"
            r = self.router.route(vehicle, (float(sx), float(sy)), (float(ex), float(ey)), blocks, barrier)
            if not r["ok"]:
                return r
            r["risk"] = self._route_risk(r["x"], r["y"], vehicle, r["minutes"])
            lon, lat = self.frame.to_lonlat(r.pop("x"), r.pop("y"))
            r["path"] = np.round(np.stack([lon, lat], 1), 7)[::2].tolist() + [[float(lon[-1]), float(lat[-1])]]
            r.pop("snapped_start"), r.pop("snapped_end")
            r["vehicle"] = vehicle
            r["vehicle_label"] = routing.VEHICLES[vehicle]["label"]
            r["note"] = routing.VEHICLES[vehicle]["note"]
            r["fire_note"] = fire_note
            if self.fire_run is not None and start != "depot":
                # latest safe departure: fire arrival at the start minus travel time minus margin
                ix, iy = self.g5.index(float(sx), float(sy))
                a = self.fire_run["arrival"][max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
                if np.isfinite(a).any():
                    r["fire_reaches_start_min"] = float(np.nanmin(np.where(np.isfinite(a), a, np.nan)))
                    r["leave_within_min"] = max(0.0, r["fire_reaches_start_min"] - r["minutes"] - 5)
            return r

    def _route_risk(self, x, y, vehicle, minutes):
        """Danger along a route: falling trees (storm model), blocked road pieces, fire, slope, vehicle fit."""
        from scipy.spatial import cKDTree
        x, y = np.asarray(x, float), np.asarray(y, float)
        v = routing.VEHICLES[vehicle]
        T = self.trees
        A = self.assets
        seg = np.hypot(np.diff(x), np.diff(y))
        length = float(seg.sum())
        cum = np.concatenate([[0], np.cumsum(seg)])
        # road pieces driven over -> P(at least one is blocked by a fallen tree now)
        jx = np.clip((x - A.x0).astype(int), 0, A.nx - 1)
        jy = np.clip((y - A.y0).astype(int), 0, A.ny - 1)
        pid = A.road[jy, jx]
        on_road = pid >= 0
        pieces = np.unique(pid[on_road])
        pb = self.road_p[pieces] if pieces.size else np.zeros(0)
        p_block = float(1 - np.prod(1 - np.clip(pb, 0, 1))) if pb.size else 0.0
        clear = v["clear_min"]
        exp_delay = float((pb * (clear or 0)).sum()) if clear else 0.0
        # trees that can reach the route if they fall (crews off-road are exposed too)
        kd = cKDTree(np.stack([x, y], 1))
        d, _ = kd.query(np.stack([T["x"], T["y"]], 1), distance_upper_bound=40)
        reach = np.isfinite(d) & (d < T["h"])
        pf = T["p_fail_now"]
        haz = reach & (pf >= 0.1)
        exp_across = float((pf[reach] * 0.25).sum())        # ~1/4 of fall directions cross a line
        # fire: risk index along the route, torching trees beside it, simulated fire front
        gx, gy = self.g5.index(x, y)
        fr = self.risk[gy, gx]
        torch = np.isfinite(d) & (d < 15) & (T["p_torch"] >= 0.5)
        fire_margin = None
        if self.fire_run is not None:
            arr = self.fire_run["arrival"][gy, gx]
            t_pass = cum / max(length, 1) * minutes
            m = np.isfinite(arr)
            if m.any():
                fire_margin = float((arr[m] - t_pass[m]).min())
        # vehicle fit: road width, slope, branches over the road
        g2x, g2y = self.g2.index(x, y)
        slope = self.router.L["slope_deg"][g2y, g2x]
        rw = self.road_w1[jy, jx]
        narrow_m = float(seg[(rw[:-1] < v["width_m"] + 1.0) & np.isfinite(rw[:-1])].sum()) if seg.size else 0.0
        hx, hy = self.g05.index(x, y)
        over = on_road & (self.chm05[hy, hx] > 2)
        overhang_m = float(seg[over[:-1]].sum()) if seg.size else 0.0
        off_m = float(seg[~on_road[:-1]].sum()) if seg.size else 0.0
        warn = []
        if p_block > 0.05:
            warn.append(f"{pct_(p_block)} chance a fallen tree blocks it" + (
                " (walkers climb over)" if vehicle == "foot" else f" (crew can clear, ~{exp_delay:.0f} min)" if clear
                else " - this vehicle cannot clear trees"))
        if haz.sum():
            warn.append(f"{int(haz.sum())} storm-prone trees can fall onto it")
        if torch.sum():
            warn.append(f"{int(torch.sum())} trees beside it would likely torch if a fire reached them")
        if fire_margin is not None and fire_margin < 15:
            warn.append(f"fire front within {max(fire_margin, 0):.0f} min of the route")
        if narrow_m > 0:
            warn.append(f"{narrow_m:.0f} m of narrow road (< {v['width_m'] + 1.0:.1f} m)")
        if overhang_m > 20 and v["height_m"] > 2.5:
            warn.append(f"branches over the road on {overhang_m:.0f} m - {v['height_m']} m tall vehicle may need clearing")
        if slope.max() > 12 and vehicle not in ("foot", "atv", "forwarder"):
            warn.append(f"steep section {slope.max():.0f} deg")
        stuck = 0.0 if clear else p_block
        fire_term = 1.0 if (fire_margin is not None and fire_margin < 5) else (0.5 if fire_margin is not None and fire_margin < 15 else 0.0)
        danger = 100 * min(1.0, 0.35 * stuck + 0.15 * min(p_block, 1) + 0.15 * min(exp_across / 2, 1)
                           + 0.15 * float(np.clip((np.percentile(fr, 90) - 20) / 60, 0, 1)) + 0.1 * min(torch.sum() / 30, 1)
                           + 0.25 * fire_term + 0.05 * min(off_m / max(length, 1), 1))
        return dict(danger=round(danger), level="high" if danger >= 50 else "medium" if danger >= 25 else "low",
                    length_m=round(length), offroad_m=round(off_m), p_blocked=round(p_block, 3),
                    expected_clearing_min=round(exp_delay, 1), storm_trees_in_reach=int(haz.sum()),
                    expected_trees_across=round(exp_across, 2), fire_risk_p90=round(float(np.percentile(fr, 90))),
                    torching_trees_beside=int(torch.sum()), fire_margin_min=None if fire_margin is None else round(fire_margin),
                    max_slope_deg=round(float(slope.max()), 1), narrow_m=round(narrow_m), overhang_m=round(overhang_m),
                    warnings=warn)

    MISSIONS = {
        "any": ("any emergency", None),
        "fire": ("fire suppression", ("fire_engine", "pickup_4x4", "atv", "forwarder", "foot")),
        "medical": ("medical / evacuation", ("ambulance", "pickup_4x4", "atv", "foot")),
        "power": ("power-line repair", ("pickup_4x4", "forwarder", "atv", "foot")),
    }

    PRIMARY = {"fire": "fire_engine", "medical": "ambulance", "power": "pickup_4x4"}

    def plan_emergency(self, end, start="depot", avoid_fire_min=None, mission="any"):
        """Try every vehicle class to the destination; rank by safety and arrival time; say which vehicles fit.
        mission: any | fire | medical | power - which vehicles can do the job."""
        mlabel, allowed = self.MISSIONS.get(mission, self.MISSIONS["any"])
        opts = []
        for veh, v in routing.VEHICLES.items():
            r = self.route(veh, start, end, 0.5, avoid_fire_min)
            fit = dict(width_m=v["width_m"], height_m=v["height_m"], mass_t=v["mass_t"], min_road_w=v["min_road_w"])
            if not r.get("ok"):
                opts.append(dict(vehicle=veh, label=v["label"], ok=False, reason=r.get("reason"), fit=fit))
                continue
            rk = r["risk"]
            stuck = 0.0 if v["clear_min"] else rk["p_blocked"]
            eta = r["minutes"] + rk["expected_clearing_min"]
            score = eta * (1 + rk["danger"] / 40) * (1 + 3 * stuck)
            if allowed and veh not in allowed:
                score *= 1e6                 # can reach, but cannot do this job
            opts.append(dict(vehicle=veh, label=v["label"], ok=True, suits_mission=not allowed or veh in allowed,
                             minutes=round(r["minutes"], 1),
                             expected_minutes=round(eta, 1), walk_min=r.get("walk_min"), p_arrive=round(1 - stuck, 3),
                             danger=rk["danger"], level=rk["level"], risk=rk, fit=fit, note=v["note"], path=r["path"],
                             score=score, fire_note=r.get("fire_note")))
        ok = [o for o in opts if o["ok"] and o["suits_mission"]]
        best = min(ok, key=lambda o: o["score"]) if ok else None
        fastest = min(ok, key=lambda o: o["expected_minutes"]) if ok else None
        safest = min(ok, key=lambda o: (o["danger"], o["expected_minutes"])) if ok else None
        art = lambda w: ("an " if w[:1].lower() in "aeiou" else "a ") + w  # noqa: E731
        drive = [o for o in opts if o["ok"] and o["vehicle"] != "foot" and o["p_arrive"] >= 0.9]
        if best is None:
            text = (f"No vehicle suited to {mlabel} can reach the destination: roads blocked, fire or terrain. "
                    "Consider helicopter / drone.")
        else:
            veh = [o for o in ok if o["vehicle"] != "foot"]
            bv = min(veh, key=lambda o: o["score"]) if veh else None
            prim = next((o for o in veh if o["vehicle"] == self.PRIMARY.get(mission)), None)
            if (bv is not None and prim is not None and prim["p_arrive"] >= 0.9 and prim["danger"] < 50
                    and prim["expected_minutes"] <= 2 * bv["expected_minutes"] + 10):
                bv = prim                     # the vehicle built for the job, when it gets through safely

            def line(o):
                return (f"{art(o['label'].lower())} - about {o['expected_minutes']:.0f} min, {o['level']} danger"
                        + (f"; {o['risk']['warnings'][0]}" if o["risk"]["warnings"] else ""))
            if bv is None:
                text = f"For {mlabel}: no vehicle gets through - send the rescue team on foot, about {best['expected_minutes']:.0f} min."
            elif best["vehicle"] == "foot" and best["expected_minutes"] < 0.67 * bv["expected_minutes"]:
                text = (f"For {mlabel}: first on scene is the rescue team on foot (about {best['expected_minutes']:.0f} min). "
                        f"Best vehicle: {line(bv)}.")
                best = bv if mission in ("fire", "medical", "power") else best
            else:
                best = bv
                text = f"For {mlabel}: send {line(bv)}."
            if (safest is not None and safest is not best and safest["vehicle"] != "foot"
                    and safest["danger"] <= best["danger"] - 5):
                text += f" Safest alternative: {safest['label'].lower()} ({safest['expected_minutes']:.0f} min, danger {safest['danger']}/100)."
        for o in opts:
            o.pop("score", None)
        ex, ey = (self.depot if end == "depot" else self.frame.from_lonlat(*end))
        return dict(recommendation=text, mission=mission, mission_label=mlabel, best=best and best["vehicle"], fastest=fastest and fastest["vehicle"],
                    safest=safest and safest["vehicle"], vehicles_that_fit=[o["vehicle"] for o in drive],
                    options=sorted(opts, key=lambda o: (not o["ok"], o.get("expected_minutes", 1e9))),
                    destination=[float(c) for c in self.frame.to_lonlat(float(ex), float(ey))],
                    conditions=dict(scenario=self.cond.scenario, gust_ms=self.cond.gust_ms, fire=self.fire_run is not None))

    # ------------------------------------------------------------------ outputs
    def center_lonlat(self):
        cx = self.g05.x0 + self.g05.nx * self.g05.res / 2
        cy = self.g05.y0 + self.g05.ny * self.g05.res / 2
        lon, lat = self.frame.to_lonlat(cx, cy)
        return float(lon), float(lat)

    def bounds_lonlat(self):
        g = self.g05
        lon0, lat0 = self.frame.to_lonlat(g.x0, g.y0)
        lon1, lat1 = self.frame.to_lonlat(g.x0 + g.nx * g.res, g.y0 + g.ny * g.res)
        return [[float(lat0), float(lon0)], [float(lat1), float(lon1)]]

    def ll(self, line):
        lon, lat = self.frame.to_lonlat(line[:, 0], line[:, 1])
        return np.round(np.stack([lon, lat], 1), 7).tolist()

    def summary(self):
        T = self.trees
        n = T["x"].shape[0]
        reach_line = T["w_power"] > 0
        reach_any = (T["w_power"] > 0) | (T["w_bld"] > 0) | (T["w_road"] > 0)
        rd = np.sort(T["risk_design"][reach_any])[::-1]
        pareto = {}
        if rd.size and rd.sum() > 0:
            cum = np.cumsum(rd) / rd.sum()
            for share in (0.5, 0.8):
                k = int(np.searchsorted(cum, share) + 1)
                pareto[f"{int(share * 100)}%"] = dict(trees=k, share_of_reaching=k / rd.size)
        fwi_hist = [dict(date=h["date"], fwi=round(h["fwi"], 1)) for h in (self.cond.history or [])[-21:]]
        burnable = self.fuel > 0
        fuel_share = {fire.FUEL_LABEL[f]: float(np.mean(self.fuel[burnable] == i)) for i, f in enumerate(fire.FUELS) if i > 0}
        hfi_b = self.hfi[burnable]
        ic = fire.intensity_class(hfi_b)
        resp = self.response5[burnable]
        high_risk = self.risk >= 60
        rt = self.response5
        return dict(
            name=self.name, meta={k: v for k, v in self.meta.items() if k != "header"},
            bounds=self.bounds_lonlat(), center=self.center_lonlat(),
            area_ha=float(self.g05.nx * self.g05.ny * 0.25 / 1e4),
            lidar=self.lidar_stats, classification=self.class_info, validation=self.validation,
            timings=self.timings,
            weather=self.cond.to_dict() | {"fwi_history": fwi_hist},
            trees=dict(count=n, per_ha=n / (self.g05.nx * self.g05.ny * 0.25 / 1e4),
                       species={s: int(np.sum(T["species"] == i)) for i, s in enumerate(SPECIES)},
                       health={s: int(np.sum(T["health"] == i)) for i, s in enumerate(HEALTH)},
                       mean_h=float(T["h"].mean()), p95_h=float(np.percentile(T["h"], 95))),
            fire=dict(fuel_share=fuel_share, head_ros_m_min_p90=float(np.percentile(self.ros[burnable], 90)),
                      hfi_p90=float(np.percentile(hfi_b, 90)),
                      intensity_share={fire.INTENSITY_LABELS[k]: float(np.mean(ic == k)) for k in range(6)},
                      high_risk_ha=float(high_risk.sum() * 25 / 1e4),
                      high_risk_slow_response_ha=float((high_risk & (rt > 20)).sum() * 25 / 1e4),
                      response_p50_min=float(np.median(resp)), response_p90_min=float(np.percentile(resp, 90)),
                      trees_very_high_hazard=int((T["fire_hazard"] >= 70).sum()),
                      trees_torch_if_reached=int((T["p_torch"] >= 0.5).sum()),
                      dry_grass_ha=float((self.G5["grass"] > 0.5).sum() * self.g5.res ** 2 / 1e4)),
            storm=dict(gust_ms=self.cond.gust_ms, wind_from=self.cond.wind_dir_deg,
                       trees_reaching_line=int(reach_line.sum()), trees_reaching_any=int(reach_any.sum()),
                       expected_fallen_trees=float(T["p_fail_now"].sum()),
                       expected_line_hits=float(-np.log1p(-np.clip(self.power_p, 0, 0.999999)).sum()),
                       p_line_outage=float(1 - np.prod(1 - self.power_p)) if self.power_p.size else 0,
                       expected_road_blockages=float(self.road_p.sum()),
                       road_pieces_over_50=int((self.road_p >= 0.5).sum()),
                       trees_easily_fall_now=int((T["p_fail_now"] >= 0.2).sum()),
                       trees_easily_fall_design=int((T["p_fail_design"] >= 0.2).sum()),
                       pareto_design=pareto, design_gust_ms=wind.DESIGN_GUST),
            powerlines=[dict(name=p["name"], kv=p.get("kv"), conductor_h=p["conductor_h"], source=p["source"])
                        for p in self.powerlines],
            context_source=self.ctx.get("source"),
            discovered=self._discovered_summary(),
        )

    def infrastructure(self):
        """Everything known about roads, power lines, houses and critical infrastructure, by source."""
        d = getattr(self, "discovered", None) or {}
        ll = self.ll
        out = dict(
            image=None, nls=None,
            lidar_roads=[dict(path=ll(t["line"]), length_m=t["length_m"], width_m=t["width_m"], mapped=t["mapped"],
                              confidence=t.get("confidence"), evidence=t.get("evidence", "LiDAR"))
                         for t in d.get("tracks", []) if t["mapped"] or t.get("network") or (t.get("confidence") or 0) >= 0.9],
            image_roads=[dict(path=ll(r["line"]), length_m=r["length_m"], width_m=r["width_m"], mapped=r["mapped"],
                              confidence=r["confidence"], evidence=r.get("evidence", "image"))
                         for r in d.get("image_roads", [])],
            wires=[dict(path=ll(w["line"]), length_m=w["length_m"], height_m=w["height_m"], mapped=w["mapped"],
                        evidence=w.get("evidence", "LiDAR")) for w in d.get("wires", [])],
            corridors=[dict(path=ll(c["line"]), length_m=c["length_m"], width_m=c["width_m"], mapped=c["mapped"],
                            confidence=c.get("confidence"), evidence=c["evidence"]) for c in d.get("corridors", [])],
            buildings=[dict(pos=ll(np.array([[b["x"], b["y"]]]))[0], area_m2=b["area_m2"], height_m=b["height_m"],
                            mapped=b["mapped"], method=b["method"]) for b in d.get("buildings", [])],
            unmapped=d.get("unmapped"))
        o = getattr(self, "ortho", None)
        if o is not None:
            x0, y0, x1, y1 = o.bounds_xy
            lo0, la0 = self.frame.to_lonlat(x0, y0)
            lo1, la1 = self.frame.to_lonlat(x1, y1)
            out["image"] = dict(source=o.source, bounds=[[float(la0), float(lo0)], [float(la1), float(lo1)]],
                                cover=(getattr(self, "image_analysis", None) or {}).get("cover"),
                                error=(getattr(self, "image_analysis", None) or {}).get("error"))
        t = getattr(self, "nls_topo", None)
        if t is not None:
            out["nls"] = {k: v for k, v in t.items() if k != "errors"} | {"errors": t.get("errors", {})}
        return out

    def _discovered_summary(self):
        d = getattr(self, "discovered", None)
        if not d:
            return None
        return dict(unmapped=d["unmapped"], seconds=d["seconds"],
                    tracks=[dict(length_m=t["length_m"], width_m=t["width_m"], mapped=t["mapped"],
                                 confidence=t.get("confidence"), path=self.ll(t["line"]))
                            for t in d["tracks"] if t["mapped"] or t.get("network") or t.get("confidence", 0) >= 0.9],
                    buildings=[dict(area_m2=b["area_m2"], height_m=b["height_m"], mapped=b["mapped"], method=b["method"],
                                    pos=self.ll(np.array([[b["x"], b["y"]]]))[0]) for b in d["buildings"]],
                    wires=[dict(length_m=w["length_m"], height_m=w["height_m"], mapped=w["mapped"], path=self.ll(w["line"]))
                           for w in d["wires"]])

    def vectors(self):
        A = self.assets
        return dict(
            roads=[dict(name=r["name"], cls=r["cls"], width=r["width"], path=self.ll(r["line"]),
                        discovered=bool(r.get("discovered"))) for r in self.ctx["roads"]],
            road_pieces=[dict(id=i, road=p["road"], cls=p["cls"], p_block=round(float(self.road_p[i]), 4),
                              path=self.ll(p["line"][:: max(1, len(p["line"]) // 6)] if len(p["line"]) > 2 else p["line"]))
                         for i, p in enumerate(A.road_pieces)],
            powerlines=[dict(name=p["name"], kv=p.get("kv"), path=self.ll(p["line"]),
                             discovered="discovered" in p["name"]) for p in self.powerlines],
            power_spans=[dict(id=i, p_hit=round(float(self.power_p[i]), 4), path=self.ll(s["line"][[0, -1]]))
                         for i, s in enumerate(A.power_spans)],
            buildings=[dict(name=b["name"], kind=b["kind"], weight=b.get("weight", 1), discovered=bool(b.get("discovered")),
                            p_hit=round(float(self.bld_p[i]), 4) if i < len(self.bld_p) else 0,
                            pos=self.ll(np.array([[b["x"], b["y"]]]))[0], r=b["r"])
                       for i, b in enumerate(self.ctx["buildings"])],
            depot=self.ll(np.array([self.depot]))[0],
        )

    def tree_table(self):
        T = self.trees
        lon, lat = self.frame.to_lonlat(T["x"], T["y"])
        tcode = {"": 0, "road": 1, "power": 2, "bld": 3}
        return dict(
            fields=["lon", "lat", "h", "species", "health", "p_fail_now", "risk_now", "risk_design", "target"],
            lon=np.round(lon, 6).tolist(), lat=np.round(lat, 6).tolist(), h=np.round(T["h"], 1).tolist(),
            species=T["species"].tolist(), health=T["health"].tolist(),
            p_fail_now=np.round(T["p_fail_now"], 4).tolist(), risk_now=np.round(T["risk_now"], 4).tolist(),
            risk_design=np.round(T["risk_design"], 4).tolist(),
            target=[tcode.get(t, 0) for t in T["target"]],
            fire_hazard=np.round(T["fire_hazard"], 0).astype(int).tolist(), p_torch=np.round(T["p_torch"], 3).tolist(),
        )

    def workorder(self, top=None):
        T = self.trees
        lon, lat = self.frame.to_lonlat(T["x"], T["y"])
        reach = (T["w_power"] > 0) | (T["w_bld"] > 0) | (T["w_road"] > 0)
        order = np.argsort(-T["risk_design"])
        order = order[reach[order]]
        if top:
            order = order[:top]
        from . import costs
        cfg = self.costs()
        dmg = costs.tree_expected_damage(cfg, T, T["p_fail_design"], T["w_power"], T["w_bld"], T["w_road"])
        fell = costs.v(cfg, "felling_cost_per_tree")
        rows = []
        for rank, i in enumerate(order, 1):
            rows.append(dict(rank=rank, tree_id=int(i), lat=round(float(lat[i]), 6), lon=round(float(lon[i]), 6),
                             species=SPECIES[T["species"][i]], health=HEALTH[T["health"][i]],
                             height_m=round(float(T["h"][i]), 1), dbh_cm=round(float(T["dbh"][i]), 1),
                             slenderness_hd=round(float(T["hd"][i]), 0), threatens=T["target"][i] or "-",
                             distance_m=round(float(T["d_target"][i]), 1),
                             p_fail_design_storm=round(float(T["p_fail_design"][i]), 3),
                             p_hit_power=round(float(T["p_fail_design"][i] * T["w_power"][i]), 3),
                             p_hit_building=round(float(T["p_fail_design"][i] * T["w_bld"][i]), 3),
                             p_block_road=round(float(T["p_fail_design"][i] * T["w_road"][i] * T["road_size_factor"][i]), 3),
                             risk_score=round(float(T["risk_design"][i]), 4),
                             expected_damage_eur=round(float(dmg[i])), net_benefit_eur=round(float(dmg[i] - fell)),
                             action="fell" if T["risk_design"][i] > 0.08 or T["health"][i] == 2 else "inspect / top"))
        return rows

    def workorder_csv(self):
        rows = self.workorder()
        b = io.StringIO()
        if rows:
            w = csv.DictWriter(b, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return b.getvalue()

    def layer_png(self, name):
        g5 = self.G5
        if name == "hillshade":
            return render.to_png(render.hillshade(self.dtm05[::2, ::2], 1.0,
                                                  chm=self.chm05[::2, ::2], water=self.water05[::2, ::2]))
        if name == "chm":
            return render.to_png(render.scalar(self.chm05[::2, ::2], "chm", 0, 30, self.chm05[::2, ::2] > 0.5))
        if name == "fire_risk":
            return render.to_png(render.upsample(render.scalar(self.risk, "risk", 0, 100, self.risk > 3), 2))
        if name == "fuel":
            return render.to_png(render.upsample(render.categorical(self.fuel, render.FUEL_COLORS), 2))
        if name == "intensity":
            ic = fire.intensity_class(self.hfi)
            return render.to_png(render.upsample(render.scalar(ic.astype(float), "risk", 0, 5, ic > 0), 2))
        if name == "response":
            r = np.where(np.isfinite(self.response), self.response, np.nan)
            return render.to_png(render.scalar(np.nan_to_num(r, nan=120), "time", 0, 60, np.isfinite(r)))
        if name == "dry_fuel":
            # dry grass / open ground + dead trees, scaled by today's fuel dryness (FFMC / BUI)
            dry = fire.dryness(self.cond)
            v = (0.6 * g5["grass"] + 1.2 * np.clip(g5["dead"], 0, 1)) * (0.35 + 0.65 * dry)
            v = np.where(self.fuel == 0, 0, np.clip(v, 0, 1))
            return render.to_png(render.upsample(render.scalar(v, "risk", 0, 1, v > 0.08), 2))
        if name == "fire_arrival" and self.fire_run is not None:
            a = self.fire_run["arrival"]
            m = np.isfinite(a)
            # banded isochrones every 30 min
            v = np.floor(np.where(m, a, 0) / 30) * 30
            return render.to_png(render.upsample(render.scalar(v, "fire", 0, max(self.fire_run["hours"] * 60, 60), m), 2))
        if name == "ndvi" and self.class_info["multispectral"]:
            T = self.trees
            img = np.zeros((self.g1.ny, self.g1.nx))
            cnt = np.zeros_like(img)
            ix, iy = self.g1.index(T["x"], T["y"])
            np.add.at(img, (iy, ix), np.nan_to_num(T["ndvi"]))
            np.add.at(cnt, (iy, ix), 1)
            img = ndi.maximum_filter(np.where(cnt > 0, img / np.maximum(cnt, 1), -1), 3)
            return render.to_png(render.scalar(img, "ndvi", 0.0, 0.7, img > -0.5))
        raise KeyError(name)

    # ------------------------------------------------------------------ geolocated 3D pictures
    def viewer_origin(self):
        cx = self.g05.x0 + self.g05.nx * self.g05.res / 2
        cy = self.g05.y0 + self.g05.ny * self.g05.res / 2
        return cx, cy, float(np.percentile(self.pts["z"], 5))

    def structures_at_risk(self):
        """Rank every structure (mapped, OSM / NLS and discovered in the LiDAR) by combined risk:
        fire hazard around it, falling-tree strikes in the design storm, access for rescue, sensitivity."""
        from .online import SENSITIVITY
        T = self.trees
        bl = self.ctx.get("buildings", [])
        if not bl:
            return []
        hit = (T["w_bld"] > 0) & (T["target"] == "bld")
        road_cells = self.road_cls1 > 0
        d_road = ndi.distance_transform_edt(~road_cells) if road_cells.any() else np.full(road_cells.shape, 1e4)
        rows = []
        for i, b in enumerate(bl):
            gx, gy = self.g5.index(b["x"], b["y"])
            r5 = 6
            fire_r = float(self.risk[max(0, gy - r5):gy + r5 + 1, max(0, gx - r5):gx + r5 + 1].max())   # within ~30 m
            near = hit & (np.hypot(T["x"] - b["x"], T["y"] - b["y"]) < 35)
            pf = np.maximum(T["p_fail_design"][near], T["p_fail_now"][near])   # design storm, or today's storm if worse
            strikes = float((pf * T["w_bld"][near]).sum())
            n_reach = int(near.sum())
            resp = float(self.response5[gy, gx])
            jx, jy = self.g1.index(b["x"], b["y"])
            droad = float(d_road[jy, jx])
            sens = max(float(b.get("weight") or 0), SENSITIVITY.get(b.get("kind"), 1.0))
            f_n = float(np.clip((fire_r - 20) / 60.0, 0, 1))
            s_n = min(strikes / 0.05, 1.0)
            no_road = droad > 150
            a_n = min(max(resp - 3, 0) / 15.0, 1.0) * 0.5 + (0.5 if no_road else 0.0)
            hazard = 0.4 * f_n + 0.35 * s_n + 0.25 * a_n
            score = 0.35 * min(sens / 3.0, 1.0) + 0.65 * hazard
            reasons = []
            if f_n > 0.6:
                reasons.append(f"high fire risk around ({fire_r:.0f}/100)")
            if s_n > 0.4:
                reasons.append(f"{n_reach} trees can fall on it ({strikes:.2f} expected strikes)")
            if no_road:
                reasons.append(f"no road access ({droad:.0f} m from nearest road)")
            elif resp > 10:
                reasons.append(f"slow rescue access ({resp:.0f} min)")
            if sens >= 2.5:
                reasons.append(f"sensitive: {b['kind'].replace('_', ' ')}")
            if b.get("discovered"):
                reasons.append("not in map data (found in LiDAR)")
            lon, lat = self.frame.to_lonlat(b["x"], b["y"])
            rows.append(dict(name=b["name"], kind=b["kind"], discovered=bool(b.get("discovered")), lon=float(lon), lat=float(lat),
                             score=round(float(score), 3), fire_risk=round(fire_r), trees_in_reach=n_reach,
                             expected_strikes=round(strikes, 3), response_min=round(resp, 1), road_dist_m=round(droad),
                             sensitivity=sens, reasons=reasons))
        rows.sort(key=lambda r: -r["score"])
        for r in rows:
            r["level"] = "high" if r["score"] >= 0.55 else "medium" if r["score"] >= 0.35 else "low"
        return rows

    def views(self):
        from . import render3d
        with self.lock:
            if getattr(self, "_views", None) is None:
                cx, cy, cz = self.viewer_origin()
                v = render3d.points_of_interest(self)
                for p in v:
                    ix, iy = self.g05.index(p["x"], p["y"])
                    p["viewer"] = [p["x"] - cx, float(self.dtm05[iy, ix]) - cz, -(p["y"] - cy)]
                    p["url"] = f"/api/site/view3d/{p['id']}.png"
                self._views = v
            return self._views

    def view_png(self, vid=None, lon=None, lat=None, az=None, radius=30.0):
        from . import render3d
        if vid is not None:
            cache = getattr(self, "_view_png", None)
            if cache is None:
                self._view_png = cache = {}
            if vid not in cache:
                p = self.views()[vid]
                cache[vid] = render3d.snapshot(self, p["x"], p["y"], p["radius"], p["az"], title=p["title"],
                                               subtitle=p["subtitle"], highlight_tree=p["tree"])
            return cache[vid]
        x, y = self.frame.from_lonlat(lon, lat)
        lo, la = float(lon), float(lat)
        return render3d.snapshot(self, float(x), float(y), radius, az,
                                 title=f"3D view at {la:.5f} N, {lo:.5f} E", subtitle=self.name)

    def point_cloud_bin(self, max_points=600_000):
        """Binary: header(json len u32 + json) + float32 xyz (local, z up) + uint8 class + uint8 hazard + uint8 rgb[3]."""
        P = self.pts
        n = P["x"].shape[0]
        rng = np.random.default_rng(1)
        sel = np.sort(rng.choice(n, size=min(n, max_points), replace=False))
        cx = self.g05.x0 + self.g05.nx * self.g05.res / 2
        cy = self.g05.y0 + self.g05.ny * self.g05.res / 2
        cz = float(np.percentile(P["z"], 5))
        xyz = np.stack([P["x"][sel] - cx, P["z"][sel] - cz, -(P["y"][sel] - cy)], 1).astype(np.float32)
        cls = P["classification"][sel].astype(np.uint8)
        # hazard per point: map to tree via 0.5 m crown label (nearest tree top within crown)
        T = self.trees
        from scipy.spatial import cKDTree
        kd = cKDTree(np.stack([T["x"], T["y"]], 1))
        d, j = kd.query(np.stack([P["x"][sel], P["y"][sel]], 1), distance_upper_bound=4.0)
        j = np.where(np.isfinite(d), j, -1)
        veg = np.isin(cls, (3, 4, 5, 1)) & (j >= 0)
        rmax = max(float(np.percentile(T["risk_design"][T["risk_design"] > 0], 95)) if np.any(T["risk_design"] > 0) else 1, 1e-6)
        hz = np.zeros(sel.size, np.uint8)
        hz[veg] = np.clip(T["risk_design"][j[veg]] / rmax * 255, 0, 255).astype(np.uint8)
        dead = np.zeros(sel.size, np.uint8)
        dead[veg] = T["health"][j[veg]].astype(np.uint8)
        if "red" in P:
            rgb = np.stack([P["red"][sel], P["green"][sel], P["blue"][sel]], 1)
            rgb = np.clip(rgb / max(np.percentile(rgb, 99), 1) * 255, 0, 255).astype(np.uint8)
        else:
            rgb = np.full((sel.size, 3), 160, np.uint8)
        fh = np.zeros(sel.size, np.uint8)                          # tree fire hazard 0..100 -> 0..255
        if "fire_hazard" in T:
            fh[veg] = np.clip(T["fire_hazard"][j[veg]] * 2.55, 0, 255).astype(np.uint8)
        self._pc = dict(sel=sel, j=j, veg=veg, center=(cx, cy, cz))          # for simulation overlays in 3D
        head = json.dumps(dict(n=int(sel.size), center=[cx, cy, cz], fields=["xyz", "cls", "hazard", "health", "rgb", "fire"])).encode()
        return (np.uint32(len(head)).tobytes() + head + xyz.tobytes() + cls.tobytes() + hz.tobytes()
                + dead.tobytes() + rgb.tobytes() + fh.tobytes())


def _viewer_xyz(site, x, y, z):
    cx, cy, cz = site._pc["center"]
    return np.stack([np.asarray(x) - cx, np.asarray(z) - cz, -(np.asarray(y) - cy)], 1)


def pct_(v):
    return f"{v * 100:.0f}%"


def _synthetic_ortho(s, seed):
    """Drone RGB orthomosaic of the synthetic estate (cached: rendering takes ~20 s)."""
    from .imagery import Ortho
    cache = Path(__file__).resolve().parent.parent / "data" / "cache" / f"synthetic_ortho_{seed}.npy"
    try:
        rgb = np.load(cache)
    except Exception:
        rgb, _, _, _ = synth.render_ortho(s)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, rgb)
        except OSError:
            pass
    return Ortho(rgb, 0.0, float(s.size), 0.5, "drone RGB orthomosaic (synthetic, DJI L3 RGB camera)")


def _poly_mask_fn(polys, grid):
    """Rasterise polygons (projected coords) on a 1 m grid and return a lookup fn(x, y) -> bool."""
    from skimage.draw import polygon as skpoly

    m = np.zeros((grid.ny, grid.nx), bool)
    for P in polys:
        rr, cc = skpoly((P[:, 1] - grid.y0) / grid.res, (P[:, 0] - grid.x0) / grid.res, shape=m.shape)
        m[rr, cc] = True

    def fn(x, y):
        ix, iy = grid.index(x, y)
        return m[iy, ix]
    return fn


def load_context(path, frame):
    """GeoJSON (lon/lat) with feature properties.kind in {road, powerline, building, depot, bog}."""
    js = json.loads(Path(path).read_text())
    roads, pls, blds, depot = [], [], [], None
    for f in js["features"]:
        k = f["properties"].get("kind")
        geom = f["geometry"]
        if geom["type"] == "LineString":
            c = np.array(geom["coordinates"])
            x, y = frame.from_lonlat(c[:, 0], c[:, 1])
            line = np.stack([x, y], 1)
            if k == "road":
                cls_ = f["properties"].get("class", "forest")
                roads.append(dict(name=f["properties"].get("name", "road"), cls=cls_,
                                  width=f["properties"].get("width", 7.0 if cls_ == "main" else 4.0), line=line))
            elif k == "powerline":
                pls.append(dict(name=f["properties"].get("name", "power line"), kv=f["properties"].get("kv", 20),
                                conductor_h=f["properties"].get("conductor_h", 9.0), row_half=5.0, line=line))
        elif geom["type"] == "Point":
            x, y = frame.from_lonlat(*geom["coordinates"][:2])
            if k == "depot":
                depot = (float(x), float(y))
            else:
                blds.append(dict(name=f["properties"].get("name", "building"), kind=f["properties"].get("type", "house"),
                                 x=float(x), y=float(y), r=f["properties"].get("radius", 6.0),
                                 weight=f["properties"].get("weight", 1.0)))
    return dict(roads=roads, powerlines_vector=pls, buildings=blds, depot=depot)
