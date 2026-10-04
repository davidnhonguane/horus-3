"""Horus command line.

  python -m horus serve                       # demo: synthetic estate + Forey dataset, opens on :8000
  python -m horus serve --las survey.las --crs EPSG:3067 --context context.geojson
  python -m horus serve --scan forest_model.ply            # any 3D scan: las laz ply obj glb xyz txt csv pts e57
  (or start  python -m horus serve  and use the [Upload 3D scan] button in the web page)
  python -m horus analyze --out results/      # batch: JSON summary, work order CSV, GeoJSON, PNG layers
  python -m horus synth --out demo.las        # export the synthetic survey as LAS 1.4 (view in CloudCompare)
  python -m horus dispatch --storm 2026-10-12 --lat 62.9 --lon 27.7 --radius 110 --gust 32

Live open data (see horus/online.py):
  python -m horus serve --online              # also fetch Sentinel-2, OSM, FMI, Metsäkeskus, photos at start
  python -m horus fetch --lat 62.89 --lon 27.68   # pre-warm the cache before a demo (works offline afterwards)
  NLS_API_KEY=... python -m horus serve --nls-laser --lat 62.89 --lon 27.62 --osm
                                              # real national LiDAR (Maanmittauslaitos) + OSM infrastructure
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "forey"


def _site(args):
    from .site import Site

    t = time.time()
    las = getattr(args, "las", None)
    if getattr(args, "nls_key", None):
        import os
        os.environ["NLS_API_KEY"] = args.nls_key
    if getattr(args, "nls_laser", False):
        from . import online

        if args.lat is None or args.lon is None:
            sys.exit("--nls-laser needs --lat and --lon (centre of the forest area), e.g. --lat 62.7471 --lon 27.2595")

        out = ROOT / "data" / "nls" / f"{args.lat:.4f}_{args.lon:.4f}"
        print(f"Downloading NLS laser scanning around {args.lat}, {args.lon} ...")
        r = online.nls_laser(args.lat, args.lon, out, map_sheet=args.nls_sheet, radius_m=args.radius_m)
        if not r.get("ok"):
            sys.exit(f"NLS laser download failed: {r.get('error')}")
        las = r["files"]
        args.crs = "EPSG:3067"
        print(f"  {len(las)} file(s): {', '.join(Path(f).name for f in las)}")
    if las:
        print(f"Loading {las} ...")
        crop = None
        if getattr(args, "lat", None) is not None and getattr(args, "lon", None) is not None \
                and str(args.crs).upper() in ("EPSG:3067", "TM35FIN", "ETRS-TM35FIN"):
            from .geo import TM35FIN
            if True:
                E, N = TM35FIN().from_lonlat(args.lon, args.lat)
                crop = (E - args.radius_m, N - args.radius_m, E + args.radius_m, N + args.radius_m)
        anchor = (args.lat, args.lon) if getattr(args, "lat", None) is not None and getattr(args, "lon", None) is not None else None
        s = Site.from_las(las, crs=args.crs, context_geojson=args.context, osm=getattr(args, "osm", False), crop=crop,
                          image=getattr(args, "image", None), nls=getattr(args, "nls", False), anchor=anchor)
    else:
        print("Generating synthetic DJI L3 survey of the demo estate (Flight Task 0001, Kuopio) ...")
        s = Site.synthetic(task="Flight Task 0001")
    print("Running LiDAR + risk pipeline ...")
    s.run(args.scenario)
    v = s.validation
    print(f"  done in {time.time() - t:.1f} s - {s.trees['x'].shape[0]} trees detected"
          + (f", detection F1 {v['f1']:.2f}, height RMSE {v['height_rmse_m']:.2f} m" if v else ""))
    if las:                                   # every real scan: simulate a storm and a fire on it straight away
        try:
            sims = s.auto_simulations()
            s.auto_sims = sims
            st, fr = sims["storm"], sims["fire"]
            print(f"  storm {st['gust_ms']:.0f} m/s: {st['fallen']['mean']:.0f} trees fall, P(line outage) {st['p_line_outage']:.0%}"
                  f" | fire {fr['hours']:.0f} h: {fr['trees']['reached']} trees reached, {fr['trees']['torching']} torch")
        except Exception as exc:  # noqa: BLE001
            print("  simulations skipped:", exc)
    hl = s.highlights() if las else []
    if hl:
        print("  NEW (not in map data): " + ", ".join(f"{h['kind']} {h['length_m']} m" for h in hl[:8]))
    return s


def _dataset(path):
    from .dispatch import Dataset

    p = Path(path)
    if (p / "flight-tasks.json").exists():
        return Dataset(p)
    print(f"(Forey dataset not found in {p}; dispatch view disabled)")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="horus", description="Horus forest risk & emergency intelligence")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("serve", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--las", "--scan", dest="las", nargs="+",
                       help="3D scan(s): .las .laz .ply .obj .glb .xyz .txt .csv .pts .e57")
        p.add_argument("--crs", default="auto",
                       help="auto (default: TM35FIN if the coordinates fit Finland, else local), EPSG:3067, or local")
        p.add_argument("--context", help="GeoJSON with roads / powerlines / buildings / depot (lon/lat)")
        p.add_argument("--scenario", default="normal", choices=["live", "normal", "heatwave", "storm"])
        p.add_argument("--data", default=str(DEFAULT_DATA), help="folder with Forey JSON dataset")
        p.add_argument("--osm", action="store_true", help="fetch roads/power lines/buildings from OpenStreetMap")
        p.add_argument("--nls", action="store_true",
                       help="NLS topographic database: houses, critical infrastructure, roads, power lines (+ NLS orthophoto)")
        p.add_argument("--image", help="orthophoto / aerial / satellite image (PNG/JPG/TIF + world file, ETRS-TM35FIN)")
        p.add_argument("--nls-laser", action="store_true", help="download NLS laser scanning (needs NLS_API_KEY)")
        p.add_argument("--nls-sheet", help="NLS map sheet code if the laser process needs one")
        p.add_argument("--nls-key", help="NLS API key (or set NLS_API_KEY, or put it in nls_key.txt)")
        p.add_argument("--lat", type=float, help="site centre (with --nls-laser, or to crop a LAS)")
        p.add_argument("--lon", type=float)
        p.add_argument("--radius-m", type=float, default=400, help="half-size of the analysed window (m)")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--online", action="store_true", help="fetch live open data at start-up")
        else:
            p.add_argument("--out", default="horus_out")
    p = sub.add_parser("fetch", help="fetch + cache live open data for a location")
    p.add_argument("--lat", type=float, default=62.747066)
    p.add_argument("--lon", type=float, default=27.259548)
    p.add_argument("--radius", type=float, default=1500)
    p.add_argument("--nls-key", help="NLS API key")
    p = sub.add_parser("synth")
    p.add_argument("--out", default="horus_demo_estate.las")
    p = sub.add_parser("dispatch")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--storm", help="storm date YYYY-MM-DD")
    p.add_argument("--lat", type=float, default=62.9)
    p.add_argument("--lon", type=float, default=27.7)
    p.add_argument("--radius", type=float, default=110)
    p.add_argument("--gust", type=float, default=32)
    args = ap.parse_args(argv)

    if args.cmd in (None, "serve"):
        if args.cmd is None:
            args = ap.parse_args(["serve"] + (argv or sys.argv[1:]))
        from .server import App, serve

        s = _site(args)
        app = App(s, _dataset(args.data))
        if args.online:
            import threading
            print("Fetching live open data in the background ...")
            threading.Thread(target=app.fetch_online, daemon=True).start()
        serve(app, args.host, args.port)
    elif args.cmd == "analyze":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        s = _site(args)
        (out / "summary.json").write_text(json.dumps(_jsonable(s.summary()), indent=1))
        (out / "hazard_tree_workorder.csv").write_text(s.workorder_csv())
        (out / "trees.geojson").write_text(json.dumps(_trees_geojson(s)))
        for L in ("hillshade", "chm", "fire_risk", "fuel", "intensity", "response", "ndvi"):
            try:
                (out / f"{L}.png").write_bytes(s.layer_png(L))
            except KeyError:
                pass
        (out / "layers_bounds.json").write_text(json.dumps({"bounds_latlon": s.bounds_lonlat()}))
        print(f"Wrote results to {out.resolve()}")
    elif args.cmd == "fetch":
        import os
        if args.nls_key:
            os.environ["NLS_API_KEY"] = args.nls_key
        from . import online

        r = online.fetch_all(args.lat, args.lon, args.radius)
        for k in online.SOURCES:
            v = r.get(k, {})
            extra = v.get("counts") or v.get("count") or (v.get("scenes", [{}])[0] if v.get("scenes") else "") \
                or ({kk: round(v[kk], 1) for kk in ("canopy_loss_ha", "forest_ha") if kk in v})
            print(f"  {'OK ' if v.get('ok') else 'ERR'} {k:<10} {v.get('source', '')}  {extra or v.get('error', '')}")
        print(f"Cached in {online.CACHE}")
    elif args.cmd == "synth":
        import numpy as np
        from .geo import LocalFrame, TM35FIN
        from .lasio import write_las
        from .synth import generate

        s = generate()
        loc = LocalFrame(s.lat0, s.lon0, s.size / 2, s.size / 2)
        tm = TM35FIN()
        P = dict(s.points)
        lon, lat = loc.to_lonlat(P["x"], P["y"])
        P["x"], P["y"] = tm.from_lonlat(lon, lat)       # real-world ETRS-TM35FIN coordinates
        write_las(args.out, P)
        # context layers as they would come from Digiroad / NLS topographic database
        feats = []
        for r in s.roads:
            lo, la = loc.to_lonlat(r["line"][:, 0], r["line"][:, 1])
            feats.append({"type": "Feature", "properties": {"kind": "road", "name": r["name"], "class": r["cls"],
                                                            "width": r["width"]},
                          "geometry": {"type": "LineString", "coordinates": np.round(np.c_[lo, la], 7).tolist()}})
        for b in s.buildings:
            lo, la = loc.to_lonlat(b["x"], b["y"])
            feats.append({"type": "Feature", "properties": {"kind": "building", "name": b["name"], "type": b["kind"],
                                                            "radius": b["r"], "weight": b["weight"]},
                          "geometry": {"type": "Point", "coordinates": [round(float(lo), 7), round(float(la), 7)]}})
        lo, la = loc.to_lonlat(*s.depot)
        feats.append({"type": "Feature", "properties": {"kind": "depot", "name": "Rescue access point"},
                      "geometry": {"type": "Point", "coordinates": [float(lo), float(la)]}})
        ctx = Path(args.out).with_name(Path(args.out).stem + "_context.geojson")
        ctx.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1))
        print(f"Wrote {P['x'].shape[0]:,} points to {args.out} (LAS 1.4 PDRF 8, EPSG:3067) and context layers to {ctx}")
    elif args.cmd == "dispatch":
        from . import dispatch

        ds = dispatch.Dataset(args.data)
        _, k = dispatch.baseline(ds)
        print("Baseline plan:", json.dumps(k, indent=1))
        if args.storm:
            r = dispatch.storm_response(ds, args.storm, args.lat, args.lon, args.radius, args.gust)
            print("Storm response:", json.dumps(r["kpi"], indent=1))
            for a in r["assignments"][:15]:
                print(f"  {a['date']} {a['start']}  {a['operator']:<22} -> {a['task']:<40} data in {a['data_ready_h_after_storm']} h")


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _trees_geojson(s):
    t = s.tree_table()
    feats = []
    sp = ("pine", "spruce", "birch")
    hl = ("healthy", "stressed", "dead")
    tg = ("", "road", "power", "building")
    for i in range(len(t["lon"])):
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [t["lon"][i], t["lat"][i]]},
                      "properties": {"h": t["h"][i], "species": sp[t["species"][i]], "health": hl[t["health"][i]],
                                     "p_fail_now": t["p_fail_now"][i], "risk_design": t["risk_design"][i],
                                     "threatens": tg[t["target"][i]]}})
    return {"type": "FeatureCollection", "features": feats}


if __name__ == "__main__":
    main()
