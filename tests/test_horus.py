"""Horus test-suite.  Run:  python -m pytest -q   (or: python tests/test_horus.py)"""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from horus import dispatch, fire, geo, lasio, synth, weather  # noqa: E402
from horus.site import Site  # noqa: E402

_SITE = None


def site():
    global _SITE
    if _SITE is None:
        _SITE = Site.synthetic().run("normal")
    return _SITE


def test_tm35fin_roundtrip_and_reference():
    f = geo.TM35FIN()
    E, N = f.from_lonlat(27.0, 60.0)
    assert abs(E - 500000) < 1e-6                      # central meridian
    lon, lat = f.to_lonlat(E, N)
    assert abs(lon - 27.0) < 1e-9 and abs(lat - 60.0) < 1e-9
    E, N = f.from_lonlat(24.9525, 60.1695)             # Helsinki Senate Square
    assert abs(E - 385850) < 1000 and abs(N - 6672100) < 1000
    lon, lat = f.to_lonlat(E, N)
    assert abs(lon - 24.9525) < 1e-8 and abs(lat - 60.1695) < 1e-8


def test_fwi_matches_van_wagner_reference():
    r = weather.fwi_series([dict(date="2026-04-13", T=17, RH=42, W=25, rain=0)])[0]
    ref = dict(ffmc=87.69, dmc=8.55, dc=19.01, isi=10.85, bui=8.49, fwi=10.10)
    for k, v in ref.items():
        assert abs(r[k] - v) < 0.02, (k, r[k], v)


def test_las_roundtrip():
    s = synth.generate(size=200, seed=3)
    P = s.points
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.las"
        lasio.write_las(p, P)
        Q = lasio.read_las(p)
    assert Q["header"]["point_format"] == 8 and Q["header"]["version"] == "1.4"
    assert np.allclose(Q["x"], P["x"], atol=1e-3) and np.allclose(Q["z"], P["z"], atol=1e-3)
    assert (Q["classification"] == P["classification"]).all()
    assert (Q["nir"] == P["nir"]).all()


def test_lidar_pipeline_accuracy():
    v = site().validation
    assert v["f1"] > 0.65, v
    assert v["precision"] > 0.85
    assert v["height_rmse_m"] < 1.0
    assert v["species_accuracy"] > 0.85
    assert v["dtm_mae_m"] < 0.1
    assert site().powerlines and "LiDAR" in site().powerlines[0]["source"]


def test_fire_spreads_downwind():
    s = site()
    s.set_scenario("heatwave", wind_dir=270)            # wind from the west -> fire runs east
    lon, lat = s.frame.to_lonlat(400, 600)
    s.simulate_fire(float(lon), float(lat), 2)
    a = s.fire_run["arrival"]
    iy, ix = 120, 80
    east = a[iy, ix + 20]
    west = a[iy, ix - 20]
    assert np.isfinite(east) and (not np.isfinite(west) or east < west)
    s.set_scenario("normal")


def test_storm_increases_risk_and_blocks_roads():
    s = site()
    calm = s.set_scenario("normal")["storm"]
    storm = s.set_scenario("storm", gust=32)["storm"]
    assert storm["expected_fallen_trees"] > 20 * calm["expected_fallen_trees"]
    assert storm["road_pieces_over_50"] > 0
    blds = s.vectors()["buildings"]
    amb = s.route("ambulance", "depot", blds[8]["pos"], block_threshold=0.2)
    eng = s.route("fire_engine", "depot", blds[8]["pos"], block_threshold=0.2)
    foot = s.route("foot", "depot", blds[8]["pos"], block_threshold=0.2)
    assert foot["ok"]
    assert (not amb["ok"]) or amb["clearings"] == 0
    assert eng["ok"]
    s.set_scenario("normal")


def test_workorder_ranked():
    rows = site().workorder(50)
    assert rows and rows[0]["rank"] == 1
    scores = [r["risk_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_dispatch_baseline_and_storm():
    ds = dispatch.Dataset(ROOT / "data" / "forey")
    _, k = dispatch.baseline(ds)
    assert k["on_time_rate"] > 0.9
    r = dispatch.storm_response(ds, "2026-10-12", 62.9, 27.7, 110, 32)
    kp = r["kpi"]
    assert kp["completed"] == kp["emergency_tasks"] > 0
    assert kp["exposure_weighted_hours"] <= kp["business_as_usual_exposure_weighted_hours"]


def test_api_smoke():
    from http.server import ThreadingHTTPServer
    from horus.server import App, make_handler

    os.environ["HORUS_QUIET"] = "1"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(site(), None)))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    js = json.loads(urllib.request.urlopen(base + "/api/site/summary").read())
    assert js["trees"]["count"] > 1000
    png = urllib.request.urlopen(base + "/api/site/layer/fire_risk.png").read()
    assert png[:4] == b"\x89PNG"
    req = urllib.request.Request(base + "/api/site/route", json.dumps({"vehicle": "foot", "end": [27.26, 62.747]}).encode(),
                                 {"Content-Type": "application/json"})
    assert json.loads(urllib.request.urlopen(req).read())["ok"]
    try:
        urllib.request.urlopen(base + "/../horus/server.py")
        raise AssertionError("path traversal not blocked")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    httpd.shutdown()


def test_online_connectors_with_fixtures():
    """Parsers + NDVI change pipeline on canned responses in each service's real format."""
    import importlib
    os.environ["HORUS_ONLINE_FIXTURES"] = str(ROOT / "tests" / "fixtures")
    from horus import online
    importlib.reload(online)
    try:
        r = online.fetch_all(62.747066, 27.259548, 1500)
        assert r["osm"]["ok"] and r["osm"]["counts"]["power_lines"] == 1 and r["osm"]["fire_stations"]
        assert r["sentinel2"]["ok"] and "{z}" in r["sentinel2"]["tiles"]["ndvi"]
        ch = r["s2change"]
        assert ch["ok"] and ch["canopy_loss_ha"] > 1 and ch["forest_ha"] > ch["canopy_loss_ha"]
        assert online.change_png(ch)[:4] == b"\x89PNG"
        assert r["fmi"]["ok"] and abs(r["fmi"]["gust_ms"] - 11.8) < 1e-6
        assert r["warnings"]["count"] == 1 and r["warnings"]["warnings"][0]["kind"] == "wind"
        assert r["stands"]["ok"] and r["stands"]["stands"][0]["summary"]["species"] == 1
        assert r["photos"]["ok"] and r["photos"]["count"] == 5
        ctx = online.osm_to_context(r["osm"], geo.TM35FIN())
        assert ctx["roads"] and ctx["powerlines_vector"] and ctx["buildings"] and ctx["depot"] and ctx["bog_polys"]
        json.dumps(online.public(r))  # serialisable
    finally:
        os.environ.pop("HORUS_ONLINE_FIXTURES", None)
        importlib.reload(online)


def test_nls_quality_recognition_and_3d_views():
    """NLS-like scan (5 p, intensity only, wires/roofs unclassified) + NLS power-line vector."""
    from horus import render3d
    s = Site.synthetic(degrade="nls").run("normal")
    v = s.validation
    assert v["f1"] > 0.65 and v["species_accuracy"] > 0.6 and v["health_accuracy"] > 0.9
    assert s.wire_points_labelled > 1000 and "LiDAR" in s.powerlines[0]["source"]
    assert "intensity" in s.class_info["health_source"]
    views = s.views()
    assert len(views) >= 8 and {"hazard_tree", "power_span", "dead_cluster"} <= {x["kind"] for x in views}
    png = s.view_png(0)
    assert png[:4] == b"\x89PNG" and len(png) > 20000


def test_discovers_unmapped_roads_buildings_power_lines():
    """The synthetic forest hides 2 cabins and 2 tracks that are NOT in the map data; the NLS-quality
    variant also has its power line removed from the map. Horus must find them in the point cloud."""
    s = site()
    d = s.summary()["discovered"]
    assert d["unmapped"]["roads"] == 2 and d["unmapped"]["buildings"] == 2
    truth_b = [(520, 565), (705, 690)]
    found = [b for b in s.discovered["buildings"] if not b["mapped"]]
    assert all(min(np.hypot(b["x"] - tx, b["y"] - ty) for b in found) < 5 for tx, ty in truth_b)
    cab = [b for b in s.vectors()["buildings"] if b["discovered"]][-1]
    assert s.route("fire_engine", "depot", cab["pos"])["ok"]       # reachable via the discovered track
    n = Site.synthetic(degrade="nls")
    n.ctx["powerlines_vector"] = []
    n.run("normal")
    assert n.discovered["unmapped"]["power_lines"] >= 1
    assert "discovered" in n.powerlines[0]["name"] and n.summary()["storm"]["trees_reaching_line"] > 500


def test_structures_at_risk_ranking():
    s = site()
    R = s.structures_at_risk()
    assert R and R[0]["kind"] == "care_home" and R[0]["level"] == "high"
    assert any(r["discovered"] for r in R)            # unmapped cabins found in LiDAR are ranked too
    assert all(R[i]["score"] >= R[i + 1]["score"] for i in range(len(R) - 1))


def test_image_roads_and_power_corridors_fused_with_lidar():
    s = site()
    I = s.infrastructure()
    assert I["image"] and "orthomosaic" in I["image"]["source"]
    # hidden forest track: found in both the 3D scan and the image
    both = [r for r in I["lidar_roads"] if not r["mapped"] and r["evidence"] == "LiDAR + image"]
    assert both
    # power line: image clearings on the real line, no false corridors
    assert I["corridors"] and all(c["mapped"] for c in I["corridors"])
    assert any(w["evidence"] == "LiDAR + image" for w in I["wires"])


def test_nls_register_houses_critical_roads_power():
    import importlib
    os.environ["HORUS_ONLINE_FIXTURES"] = str(ROOT / "tests" / "fixtures")
    from horus import online
    importlib.reload(online)
    try:
        b = online.bbox_around(62.747066, 27.259548, 600)
        t = online.nls_topography(b)
        assert t["ok"] and t["counts"]["critical"] == 1 and t["critical"][0]["kind"] == "substation"
        assert t["counts"]["by_building_type"].get("cabin") == 2 and t["counts"]["by_building_type"].get("public") == 1
        assert t["counts"]["power_lines"] == 1 and t["counts"]["roads"] >= 5
    finally:
        os.environ.pop("HORUS_ONLINE_FIXTURES", None)
        importlib.reload(online)


def _scan_without_map(tmp, size=400, seed=11):
    s0 = synth.generate(size=size, seed=seed)
    f = Path(tmp) / "scan.las"
    lasio.write_las(f, s0.points)
    return s0, f


def test_road_map_from_scan_without_map_data():
    """A scan with no roads given: Horus draws the road network itself (junctions, branches, inferred
    links across clearings) and puts the rescue access point where the main road enters."""
    from horus.geo import polyline_distance
    from horus.synth import _densify
    with tempfile.TemporaryDirectory() as d:
        s0, f = _scan_without_map(d)
        s = Site.from_las(f, crs="auto").run()
    assert s.meta["crs"] == "local" and s.discovered["road_map_from_scan"]
    truth = [r["line"] for r in s0.roads + s0.hidden_tracks]
    found = [t for t in s.discovered["tracks"] if not t.get("inferred")]
    tot = sum(t["length_m"] for t in found)
    on = sum(t["length_m"] * np.mean(np.min([polyline_distance(*_densify(t["line"], 2).T, L)[0] for L in truth], 0) < 6)
             for t in found)
    covered = 0.0
    total = 0.0
    for r in s0.roads + s0.hidden_tracks:
        q = _densify(r["line"], 2)
        dd = np.min([polyline_distance(q[:, 0], q[:, 1], t["line"])[0] for t in s.discovered["tracks"]], 0)
        L = float(np.hypot(*np.diff(r["line"], axis=0).T).sum())
        covered += L * np.mean(dd < 6)
        total += L
    assert on / tot > 0.9, on / tot              # what it draws is road
    assert covered / total > 0.6, covered / total
    main_entry = s0.roads[0]["line"][0]
    assert np.hypot(*(np.array(s.depot) - main_entry)) < 15
    blds = s.vectors()["buildings"]
    assert sum(bool(s.route("ambulance", "depot", b["pos"]).get("ok")) for b in blds) >= len(blds) // 2


def test_tree_fire_hazard_and_storm_simulation():
    s = site()
    T = s.trees
    assert T["fire_hazard"][T["health"] == 2].mean() > T["fire_hazard"][T["health"] == 0].mean() + 25
    n_norm = int((T["p_torch"] >= 0.5).sum())
    s.set_scenario("heatwave")
    assert (s.trees["p_torch"] >= 0.5).sum() > 2 * n_norm          # dry fuels: far more trees torch
    lon, lat = s.frame.to_lonlat(400, 600)
    fr = s.simulate_fire(float(lon), float(lat), 1)["trees"]
    assert fr["reached"] > 100 and 0 < fr["torching"] <= fr["reached"] and fr["worst"][0]["why"]
    s.set_scenario("normal")
    ft = s.fire_prone_trees(10)
    assert ft["trees"][0]["fire_hazard"] >= ft["trees"][-1]["fire_hazard"] - 30
    lo, hi = s.simulate_storm(15, 240, 60), s.simulate_storm(32, 240, 60)
    assert hi["fallen"]["mean"] > 10 * max(lo["fallen"]["mean"], 1) and hi["p_line_outage"] > lo["p_line_outage"]
    c = [p["fallen"] for p in hi["curve"]]
    assert all(c[i] <= c[i + 1] for i in range(len(c) - 1))
    assert len(hi["fall_lines"]["kind"]) > 100 and hi["worst"][0]["p_fall"] >= hi["worst"][-1]["p_fall"]


def test_emergency_plan_safest_route_and_vehicle_fit():
    from horus import routing
    s = site()
    b = s.vectors()["buildings"][8]["pos"]
    P = s.plan_emergency(b, mission="medical")
    assert P["best"] == "ambulance" and "ambulance" in P["vehicles_that_fit"]
    opt = {o["vehicle"]: o for o in P["options"]}
    assert all(0 <= o["danger"] <= 100 for o in opt.values() if o["ok"])
    # a road narrower than the vehicle needs is not driven by it
    R = s.router
    rw = R.L["road_w"]
    narrow = (R.L["road_cls"] > 0) & np.isfinite(rw)
    old = rw.copy()
    try:
        R.L["road_w"] = np.where(narrow, 2.7, rw)      # fits a 4x4 (needs 2.6 m), not a fire engine (3.0 m)
        assert (R.speed("fire_engine")[narrow] == 0).all() and (R.speed("pickup_4x4")[narrow] > 0).any()
    finally:
        R.L["road_w"] = old
    assert routing.VEHICLES["fire_engine"]["width_m"] > routing.VEHICLES["atv"]["width_m"]
    s.set_scenario("storm", gust=30)
    try:
        Q = s.plan_emergency(b, mission="medical")
        q = {o["vehicle"]: o for o in Q["options"]}
        assert not q["ambulance"]["ok"] or q["ambulance"]["p_arrive"] < 1     # storm-blocked roads
        assert Q["best"] != "ambulance" and "foot" in Q["recommendation"]
    finally:
        s.set_scenario("normal")


def test_upload_scan_and_analyse_via_api():
    from http.server import ThreadingHTTPServer
    from horus import server as srv

    os.environ["HORUS_QUIET"] = "1"
    with tempfile.TemporaryDirectory() as d:
        old = srv.UPLOADS
        srv.UPLOADS = Path(d) / "uploads"
        app = srv.App(site(), None)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(app))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            s0, f = _scan_without_map(d, size=300, seed=5)
            up = lambda name, data: urllib.request.urlopen(urllib.request.Request(  # noqa: E731
                f"{base}/api/upload/file?batch=test123&name={name}", data, {"Content-Type": "application/octet-stream"}))
            assert json.loads(up("forest.las", f.read_bytes()).read())["kind"] == "scan"
            assert json.loads(up("model.glb", b"glTF" + b"\0" * 64).read())["kind"] == "scan"   # 3D model accepted
            try:
                up("bad.ply", b"nope")
                raise AssertionError("bad ply accepted")
            except urllib.error.HTTPError as e:
                assert e.code == 400
            post_ = urllib.request.urlopen(urllib.request.Request(f"{base}/api/upload/clear", json.dumps({"batch": "test123"}).encode(),
                                                                  {"Content-Type": "application/json"}))
            assert json.loads(post_.read())["ok"]
            up("forest.las", f.read_bytes())
            for bad in ("x.exe", "fake.las"):
                try:
                    up(bad, b"nope")
                    raise AssertionError(bad + " accepted")
                except urllib.error.HTTPError as e:
                    assert e.code == 400
            post = lambda p, b: json.loads(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
                base + p, json.dumps(b).encode(), {"Content-Type": "application/json"})).read())
            post("/api/upload/analyze", {"batch": "test123", "crs": "auto", "nls": False})
            for _ in range(300):
                st = json.loads(urllib.request.urlopen(base + "/api/upload/status").read())
                if st["state"] != "running":
                    break
                time.sleep(0.2)
            assert st["state"] == "done" and not st["demo"], st
            js = json.loads(urllib.request.urlopen(base + "/api/site/summary").read())
            assert js["trees"]["count"] > 500 and js["discovered"]["unmapped"]["roads"] > 0
            assert post("/api/site/storm_sim", {"gust": 28, "runs": 40})["fallen"]["mean"] > 0
            from PIL import Image
            Image.fromarray(np.full((200, 300, 3), (40, 100, 45), np.uint8)).save(Path(d) / "p.jpg")
            up2 = urllib.request.Request(f"{base}/api/upload/file?batch=photo01&name=p.jpg", (Path(d) / "p.jpg").read_bytes(),
                                         {"Content-Type": "application/octet-stream"})
            urllib.request.urlopen(up2)
            post("/api/upload/analyze", {"batch": "photo01", "photo_view": "above", "photo_height_m": 60})
            for _ in range(300):
                st = json.loads(urllib.request.urlopen(base + "/api/upload/status").read())
                if st["state"] != "running":
                    break
                time.sleep(0.2)
            assert st["state"] == "done" and st["photos"][0]["view"] == "above", st
            assert urllib.request.urlopen(base + st["photos"][0]["overlay_url"]).read()[:2] in (b"\xff\xd8", b"\x89P")
            assert post("/api/site/demo", {})["ok"] and json.loads(urllib.request.urlopen(base + "/api/upload/status").read())["demo"]
        finally:
            httpd.shutdown()
            srv.UPLOADS = old


def test_reads_any_3d_scan_format():
    """PLY (binary/ASCII), XYZ, CSV, OBJ (Z-up and Y-up) and GLB give the same forest as the point cloud."""
    sys.path.insert(0, str(ROOT / "tests"))
    import make3d
    from horus.scan3d import read_points
    s0 = synth.generate(size=200, seed=3)
    P = s0.points
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make3d.write_ply(d / "a.ply", P)
        make3d.write_ply(d / "b.ply", P, binary=False, n=20000)
        make3d.write_xyz(d / "c.xyz", P)
        make3d.write_xyz(d / "d.csv", P, header=True, sep=",")
        v, f = make3d.surface_mesh(P)
        make3d.write_obj(d / "e.obj", v, f)
        make3d.write_obj(d / "f.obj", v, f, y_up=True)
        make3d.write_glb(d / "g.glb", v, f)
        zr = (P["z"].min(), P["z"].max())
        for name in ("a.ply", "b.ply", "c.xyz", "d.csv", "e.obj", "f.obj", "g.glb"):
            Q = read_points(d / name, max_points=600_000)
            assert zr[0] - 0.5 < Q["z"].min() and Q["z"].max() < zr[1] + 0.5, name       # upright, metres
            if name != "b.ply":                                                        # b.ply = partial file
                assert abs(Q["z"].max() - zr[1]) < 0.5, name
            assert Q["header"]["up_axis"] == ("y" if name in ("f.obj", "g.glb") else "z"), name
        assert "red" in read_points(d / "a.ply")
        n_pts = len(Site.from_las(d / "a.ply", crs="auto").run().trees["x"])
        n_mesh = len(Site.from_las(d / "g.glb", crs="auto", max_points=600_000).run().trees["x"])
        assert n_pts > 300 and 0.6 < n_mesh / n_pts < 1.3, (n_pts, n_mesh)


def test_photo_only_analysis():
    """A photo without a 3D scan: drone photo from above (DJI GPS / height / yaw) and a photo from the ground."""
    from PIL import Image
    from horus import photo
    s0 = synth.generate(size=200, seed=4)
    rgb, *_ = synth.render_ortho(s0, res=0.25)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        im = Image.fromarray(rgb).rotate(30, resample=Image.BILINEAR)
        alt = 0.25 * 24 * im.size[0] / 36
        ex = Image.Exif()
        ex[0x010F] = "DJI"
        ex[0x8825] = {1: "N", 2: (62.0, 44.0, 49.0), 3: "E", 4: (27.0, 15.0, 34.0)}
        ex[0x8769] = {0xA405: 24}
        xmp = (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
               f'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" drone-dji:RelativeAltitude="+{alt:.2f}" '
               f'drone-dji:GimbalPitchDegree="-90.0" drone-dji:GimbalYawDegree="+30.0"/></rdf:RDF></x:xmpmeta>').encode()
        im.save(d / "dji.jpg", quality=92, exif=ex, xmp=xmp)
        m = photo.read_meta(d / "dji.jpg")
        assert abs(m["lat"] - 62.7469) < 1e-3 and m["pitch"] == -90 and m["yaw"] == 30
        r, ov, _ = photo.analyse_photo(d / "dji.jpg")
        assert r["view"] == "above" and abs(r["gsd_m"] - 0.25) < 0.01 and r["bounds"]
        visible = int((s0.trees["h"] > 10).sum())
        assert 0.5 * visible < r["trees"]["count"] < 1.6 * visible, (r["trees"], visible)
        assert r["trees"]["dead"] > 0 and ov[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xdb")
        g = np.zeros((300, 450, 3), np.uint8)
        g[:90] = (135, 180, 230)
        g[90:210] = (40, 95, 45)
        g[90:210, :150] = (120, 75, 45)              # one third of the crowns brown (dead)
        g[210:] = (190, 165, 95)                      # dry grass
        Image.fromarray(g).save(d / "ground.jpg", quality=95)
        r, _, _ = photo.analyse_photo(d / "ground.jpg")
        assert r["view"] == "ground" and 0.2 < r["foliage"]["brown_grey"] < 0.5 and r["dry_ground_share"] > 0.6


def test_scan_gets_storm_and_fire_simulation_and_highlights():
    """Every uploaded / --scan 3D scan: automatic storm + fire simulation, per-point 3D overlay, new finds highlighted."""
    with tempfile.TemporaryDirectory() as d:
        s0, f = _scan_without_map(d, size=300, seed=5)
        s = Site.from_las(f, crs="auto").run()
    sims = s.auto_simulations(hours=1)
    st, fr = sims["storm"], sims["fire"]
    assert st["fallen"]["mean"] > 0 and 0 <= st["p_line_outage"] <= 1
    assert fr["trees"]["reached"] > 0 and fr["trees"]["torching"] <= fr["trees"]["reached"]
    assert all(i["minutes"] is None or i["minutes"] > 0.5 for i in fr["first_reached"])   # not started on a house
    b = s.points_sim_bin()
    hl = int(np.frombuffer(b[:4], np.uint32)[0])
    head = json.loads(b[4:4 + hl])
    n = head["n"]
    fire_b = np.frombuffer(b, np.uint8, n, 4 + hl)
    storm_b = np.frombuffer(b, np.uint8, n, 4 + hl + n)
    assert head["fire"] and head["storm"] and (fire_b > 0).any() and (storm_b == 255).any()
    H = s.highlights()
    roads = [h for h in H if h["kind"] == "road"]
    assert roads and all(len(h["xyz"]) >= 2 and h["path"] for h in roads)
    assert any(h["kind"] == "power" for h in H)              # the unmapped line is highlighted too


def test_simulations_quantified_in_euros():
    """Storm and fire simulations report damage in EUR; stronger storms cost more; unit costs are editable."""
    from horus import costs
    s = site()
    lo = s.simulate_storm(18, 240, 80)["damage_eur"]
    hi = s.simulate_storm(32, 240, 80)["damage_eur"]
    assert hi["total_mean"] > 3 * max(lo["total_mean"], 1) and hi["total_p5"] <= hi["total_mean"] <= hi["total_p95"]
    assert abs(sum(hi["parts"].values()) - hi["total_mean"]) < 1e-6 * max(hi["total_mean"], 1)
    assert hi["trees_worth_felling"] > 0 and hi["damage_avoided_by_felling"] > hi["felling_cost"]
    lon, lat = s.frame.to_lonlat(400, 600)
    fr = s.simulate_fire(float(lon), float(lat), 1)["damage_eur"]
    assert fr["total"] > 0 and fr["parts"]["Timber value lost"] > 0 and fr["burned_ha"] > 0
    cfg = s.costs()
    old = cfg["outage_compensation_per_customer"]["value"]
    try:
        cfg["outage_compensation_per_customer"]["value"] = old * 2
        hi2 = s.simulate_storm(32, 240, 80)["damage_eur"]
        assert hi2["parts"]["Outage compensation"] > 1.9 * hi["parts"]["Outage compensation"]
    finally:
        cfg["outage_compensation_per_customer"]["value"] = old
    wo = s.workorder(5)
    assert all("expected_damage_eur" in r for r in wo)
    assert all(c["basis"] in ("derived", "assumption", "user") for c in costs.public(cfg))


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
