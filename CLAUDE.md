# Horus - notes for Claude Code

Forest risk & emergency intelligence (Forey hackathon). Python 3.10+, no web framework.

## Setup
pip install -r requirements.txt            # numpy, scipy, scikit-image, Pillow
pip install -r requirements-online.txt     # + laspy[lazrs] for .laz (NLS laser scans)

## Run
python -m horus serve                      # demo estate, opens http://localhost:8000
python -m horus serve --online             # + live open data (Sentinel-2, OSM, FMI, Metsakeskus, photos)
python -m horus fetch                      # check every online source (NLS key must show OK)
python -m horus serve --nls-laser --osm --online --lat 62.7471 --lon 27.2595 --radius-m 400   # real NLS scan
python -m horus serve --scan model.ply      # any 3D scan: las laz ply obj glb xyz txt csv pts e57 (horus/scan3d.py)
# web UI: orange "Upload 3D scan" button / drag a file onto the page -> road map, power lines, buildings, trees, risks
# every scan: Site.auto_simulations() (storm + fire), /api/site/points_sim.bin (3D overlay), /api/site/highlights (NEW roads/lines)
# a photo alone (.jpg/.png/.tif) is analysed too: horus/photo.py (crowns + health, dry grass, roads, line clearings)
# if the button is missing, an OLD server holds port 8000: Horus 1.2+ prints this and moves to 8001+
python -m horus analyze --out results/     # batch outputs (summary.json, work order CSV, GeoJSON, PNGs)
python -m horus dispatch --storm 2026-10-12   # drone dispatch on Forey's dataset

NLS API key: one line in nls_key.txt (or NLS_API_KEY env var, or --nls-key).

## Test
python tests/test_horus.py                 # all tests must PASS
HORUS_ONLINE_FIXTURES=tests/fixtures python -m horus serve --online   # offline test data (red banner in UI)

## Layout
horus/   lidar.py (trees), discover.py (road map from a scan with no map data, unmapped roads/buildings/lines), fire.py, wind.py, routing.py,
         dispatch.py, online.py (open-data connectors), render3d.py (3D pictures), site.py (pipeline), server.py (API)
web/     index.html, app.js, upload.js (Upload scan), hazard.js (Tree hazards: storm/fire sims; Emergency plan), views.js, live.js, map.js (canvas map), viewer3d.js (WebGL)
data/forey/  Forey challenge dataset

## Permissions
.claude/settings.json pre-approves only: python / pip (incl. .venv), venv activation, run.sh / run.bat,
and curl to http://localhost (health checks). Everything else still asks.

## Launch (what to do when asked to "run Horus")
1. python -m venv .venv  (if missing), then install: .venv python -m pip install -r requirements.txt -r requirements-online.txt
2. python tests/test_horus.py  -> all tests PASS
3. python -m horus serve  (run it in the background) -> report the URL it prints (8000, or 8001+ if busy)
4. check http://localhost:<port>/api/health shows version 1.5.0

## Rules for whoever edits this
- Keep ONE master copy. Do not fix bugs in a separate unzipped copy only: copy the fixed files back
  (horus/online.py, horus/routing.py, horus/__main__.py, ...) and rerun `python tests/test_horus.py`.
- nls_key.txt holds the working NLS key; never print it in logs.

## Money (v1.5)
horus/costs.py: unit costs (derived vs assumption), storm_run_cost/summarise, fire_cost, tree_expected_damage.
simulate_storm/simulate_fire return damage_eur; GET/POST /api/site/costs (saved in data/costs.json - keep out of zips).
