# HORUS: forest risk & emergency intelligence

*Forey Forest Intelligence Challenge. Main category: **3. Disruptions and resource coordination**, with contributions to 1 (wildfire), 2 (critical infrastructure) and 4 (safe routing).*

Horus turns the point cloud Forey already collects with a DJI M400 + Zenmuse L3 (+ multispectral) into decisions for the people who protect forests, power lines and residents:

| When | User | Decision Horus improves |
|---|---|---|
| **Before** | Grid operator (DSO) vegetation manager | *Which* trees along the 20 kV line to fell first, from a ranked, GPS-ready work order |
| **Before** | Rescue department / forest owner | Where fire risk is high **and** response is slow; where to reduce fuel |
| **During** | Rescue incident commander | How fast a fire spreads, when it reaches each house, the line and the roads; latest safe evacuation time |
| **After a storm** | DSO dispatcher, rescue, municipality | Which roads are passable for which vehicle; where the line most likely has faults |
| **After a storm** | Forey operations | Which drone operator flies where first, so the worst-hit areas are surveyed soonest |

---

## 1. Quick start (no internet needed)

Requirements: **Python 3.10+**, plus `numpy`, `scipy`, `scikit-image` and `Pillow`. No web framework, no GIS stack and no JavaScript build are needed.

```bash
cd horus
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m horus serve
```

Open **http://localhost:8000**. The header shows the version (**v1.5.0**) and an orange **⇪ Upload 3D scan** button. If you don't see them, an older Horus is still running on port 8000: close its window (or press Ctrl+C in it). If 8000 is busy, Horus now says so and starts on the next free port (8001, …); open the address it prints. The first start takes about 10 s: Horus generates the demo survey, processes about 4 million LiDAR points and runs every model.

Shortcuts: `./run.sh` (macOS/Linux) or `run.bat` (Windows) do the same, including creating the virtual environment.

**Docker:**
```bash
docker build -t horus .
docker run -p 8000:8000 horus
```

**Tests** (22 tests covering accuracy, physics sanity, API, security and the open-data parsers):
```bash
python tests/test_horus.py          # or: pip install pytest && python -m pytest -q
```

## 2. Use your own data

```bash
# any 3D scan straight from the command line: .las .laz .ply .obj .glb .xyz .txt .csv .pts .e57
python -m horus serve --scan forest_model.ply --lat 62.7471 --lon 27.2595   # lat/lon places un-georeferenced models

# real DJI Terra export (LAS 1.2–1.4; .laz needs: pip install "laspy[lazrs]")
python -m horus serve --las survey.las --crs EPSG:3067 --context context.geojson

# batch mode: writes summary.json, hazard_tree_workorder.csv, trees.geojson and PNG layers
python -m horus analyze --las survey.las --context context.geojson --out results/ --scenario live

# export the demo survey as a real LAS 1.4 file (TM35FIN) + context GeoJSON, e.g. to open in CloudCompare
python -m horus synth --out demo_estate.las

# drone dispatch on Forey's dataset, with a storm on 12 Oct around Kuopio
python -m horus dispatch --storm 2026-10-12 --lat 62.9 --lon 27.7 --radius 110 --gust 32
```

`context.geojson` (WGS84 lon/lat) holds what a drone cannot see: `kind: road` (LineString, `class: main|forest`), `kind: building` (Point, `type: house|care_home|cabin`, `weight`), `kind: depot` (Point, the rescue access point), and optionally `kind: powerline`. In production these come from Digiroad and the NLS topographic database. Power lines are **detected automatically** from LiDAR class-14 (wire) points.

Options: `--scenario live|normal|heatwave|storm`, `--host 0.0.0.0 --port 8000`. Set `HORUS_TOKEN=secret` to require a token (`http://host:8000/?token=secret`).


## 2a. National Land Survey laser scan, tree recognition and 3D pictures

The **NLS laser scan** tab: click a forest anywhere in Finland, then **Download & recognise trees**. Horus:
1. downloads the NLS open laser scanning (5 p, LAZ) for the square,
2. adds **NLS topographic-database power lines** (`sahkolinja`), buildings and roads (+ OpenStreetMap),
3. labels conductor points along the NLS power lines (national scans leave wires unclassified) and removes "trees" on roofs,
4. recognises every tree: **number, species (pine/spruce/birch) and health (healthy/stressed/dead from laser intensity)**,
5. runs the fire, storm, routing models and renders **geolocated 3D pictures** of the critical spots (hazard trees, the most exposed line span, dead-tree clusters, fire hot-spot, likely road blockage).

The Overview tab shows the species x health inventory, "3D" markers on the map where each picture was taken, a gallery (click: enlarge, download, fly the 3D viewer there), and a 3D picture of any spot you click.

Accuracy at NLS quality (reference forest degraded to 4.9 pts/m², intensity only): tree detection F1 0.73, height RMSE 0.6 m, species 73 %, health 96 % agreement (dead-tree flags are candidates for a field check).

The API key is read from `nls_key.txt` in this folder (included). Note: when this was built, every key-protected NLS request returned HTTP 401, so check the key is activated in your NLS account (omatili.maanmittauslaitos.fi) if downloads fail.

## 2a+. Discovering what the maps miss

Horus searches every point cloud for infrastructure that is **not in the map data** and adds it to the analysis:

| Finds | How (LiDAR evidence) | Reference-forest result |
|---|---|---|
| **Potential forest roads**: old tracks, skid trails | open (canopy < 1.5 m), smooth & gentle ground, 3-10 m wide, forest on both sides, straight-ish, ≥ 50 m; skeletonised and vectorised; ends snapped onto the road network | 2 of 2 hidden tracks found (97-99 % confidence), 0 false at the 90 % threshold |
| **Buildings in the forest**: cabins, cottages | LAS class 6 if classified, otherwise smooth, opaque roofs with no ground returns underneath | 2 of 2 hidden cabins found (drone and NLS quality), 0 false |
| **Unmapped power lines** | evenly strung points at constant height above a cleared corridor, sequential RANSAC | line found with no map data (645 trees in reach vs 641 with the real map line) |

Discovered roads become routable (the cottage is reached by fire engine via the discovered track), discovered buildings count in evacuation, fire exposure and strike risk, and each find gets a geolocated 3D picture.

### Structures at risk

Every structure (map data, OpenStreetMap / NLS buildings, and buildings **discovered in the LiDAR**) is ranked high / medium / low by **sensitivity** (hospital, care home, school, power substation, mast, house, cabin), **fire risk** within 30 m, **expected falling-tree strikes**, and **rescue access** (no road within 150 m, slow response). With `--online`, OSM substations, masts, towers, schools and hospitals are added automatically. Shown on the Overview tab as a ranked list with reasons and coloured rings on the map; it updates with the selected conditions (heatwave, storm, live weather).

## 2b. Live open data (Kuopio area or anywhere in Finland)

The **Live data** tab, `python -m horus fetch` and `serve --online` pull real, openly licensed data around the survey:

| Source | What Horus does with it | Key |
|---|---|---|
| **Sentinel-2 L2A** (Copernicus, via Microsoft Planetary Computer STAC + tiler) | Latest low-cloud true-colour and NDVI layers. **Year-over-year canopy-loss detection** (NDVI drop > 0.15 inside forest) flags clear-cuts, storm damage and bark-beetle kill: where to fly first. | none |
| **National Land Survey of Finland: laser scanning 5 p** (the national open 3D point clouds) | `--nls-laser` downloads the LAZ tiles around a point and runs the full Horus pipeline on the real 3D scan | `NLS_API_KEY` (free) |
| NLS orthophoto & topographic map WMTS | Basemaps, proxied by the server so the key never reaches the browser | `NLS_API_KEY` |
| **OpenStreetMap** (Overpass API) | Real roads (main/local/track), power lines with voltage, buildings (care homes / schools flagged), wetlands (= peat for the windthrow model), nearest fire station (= rescue entry point). `--osm` feeds them into routing, strike risk and evacuation. | none |
| **Finnish Forest Centre (Metsäkeskus)** forest stands WFS | Stand polygons with main species, height and age. On real surveys, an **independent check** of Horus' species detection (agreement per stand). | none |
| **FMI open data** | Nearest weather station (wind, gust, temperature) and **FMI warnings** (wind, forest-fire weather) for the area. "Run storm model with this gust" uses the observed gust. | none |
| **Wikimedia Commons** | Geotagged photos around the site (licence per image) | none |
| EOX Sentinel-2 cloudless | Extra satellite basemap | none |

```bash
pip install -r requirements-online.txt                  # adds laspy[lazrs] for .laz
python -m horus fetch --lat 62.747 --lon 27.26          # check every source and warm the cache
python -m horus serve --online                          # demo estate + live data tab populated

# Real national LiDAR around Kuopio (free key: https://omatili.maanmittauslaitos.fi)
# put the key in a one-line file nls_key.txt in this folder (git-ignored), or use --nls-key / NLS_API_KEY
python -m horus serve --nls-laser --osm --online --lat 62.7471 --lon 27.2595 --radius-m 400
# if the laser process asks for a map sheet: add --nls-sheet <code from MapSite>
```

Everything is cached in `data/cache/`, so once a site has been fetched the demo works without internet. Each source fails independently: a blocked API never breaks the app.

**If a source fails:** OpenStreetMap is queried in five small parts with four mirror servers (partial results are kept); FMI falls back from the nearest-station lookup to a bounding box, then to Kuopio; busy servers (HTTP 429 / 503) are retried with back-off. Wikimedia asks programs to identify themselves: set `HORUS_CONTACT=your@email` to include a contact address in Horus' user agent.

**Testing without internet:** `python tests/make_fixtures.py` writes canned responses in each API's real format. `HORUS_ONLINE_FIXTURES=tests/fixtures python -m horus serve --online` runs on them, and the UI shows a red **TEST FIXTURES** banner so they are never mistaken for real data.

## 2b. Upload a scan · tree hazards · emergency plan

**Upload scan tab.** Click the orange **⇪ Upload 3D scan** button (top right), or simply drag a file onto the page. Accepted 3D files:

| Format | What it is |
|---|---|
| `.las` `.laz` | drone / airborne LiDAR (DJI Terra, NLS laser scanning). `.laz` needs `pip install "laspy[lazrs]"` |
| `.stl` `.gltf` | 3D models (CAD / 3D-printing exports, glTF with its .bin) |
| `.ply` `.obj` `.glb` | 3D models and point clouds from photogrammetry (DJI Terra, Pix4D, Metashape, RealityCapture) or phone 3D-scanning apps; meshes are sampled into points, Y-up models are turned upright |
| `.xyz` `.txt` `.csv` `.pts` | point lists: x y z [intensity] [r g b] |
| `.e57` | terrestrial / mobile laser scanners (needs `pip install pye57`) |

You can upload several scans at once. Optionally add a photo or orthomosaic (`.jpg/.png/.tif` + `.jgw/.pgw/.tfw` world file) and a `.geojson` of known roads, lines and houses. Horus then:

1. reads the scan (large files are thinned evenly to the chosen point budget, so the whole area stays covered). Coordinates are detected automatically: ETRS-TM35FIN, or a local / un-georeferenced system that you can place with a lat/lon;
2. **makes a road map** from the 3D scan alone when no map data is given: open, smooth, gentle corridors through forest, kept as a network with junctions and branches; road ends that point at each other across an open clearing are joined as *inferred* links; the rescue access point is put where the main road enters the scan;
3. finds **power lines** (wires at constant height over a cleared strip) and **buildings** (solid, smooth roofs);
4. recognises **every tree**: species, height, DBH, health (multispectral NDVI, else laser intensity);
5. ranks the **danger to power lines, houses and roads** (structures at risk, hazard-tree work order) and runs the fire, storm and routing models.
   Tick *Land Survey (NLS)* to add NLS houses, critical infrastructure, roads and power lines (uses your key).

Files stay on your computer (`data/uploads/`). The API is `POST /api/upload/file?batch=..&name=..` (raw body), then `POST /api/upload/analyze`, then poll `GET /api/upload/status`. *Back to demo estate* restores the demo.

| Road map from the scan alone (3 synthetic estates, no map data) | Result |
|---|---|
| Drawn road that is real road (precision) | **97–100 %** |
| Share of the true road network found (recall) | 31–70 %. Misses are roads across open fields / clearings, where a 3D scan shows no corridor. The drone photo and the NLS road register fill these. |

**Every 3D scan is simulated automatically** (upload, or `--scan` on the command line). As soon as the analysis is done, Horus runs:
- a **storm** at the design gust (25 m/s, or today's gust if higher; 200 Monte-Carlo runs): which trees fall, P(power-line outage), trees across roads, buildings hit;
- a **fire** (2 h, today's conditions) started at the most dangerous spot, the highest fire risk 40–150 m from houses, roads and lines. It reports which trees burn and torch, and what the fire reaches first.

Click **▶ Fire in 3D** / **▶ Storm in 3D** in the results, or pick *Fire simulation* / *Storm simulation* under the 3D switch. In 3D, the point cloud is coloured by when the fire arrives (dark red = burned first, yellow = fire front) or by the trees that fall. Change *Conditions* (e.g. Drought) or the gust in *Tree hazards* and re-run as you like.

**New roads and power lines are always highlighted.** Anything found in the scan that is in no map data is marked everywhere:
- a header badge, e.g. **★ NEW: 1.1 km of road · 2 power lines** (click it for the map);
- a glowing orange (road) or magenta (power line) line with a **NEW** label on every 2D map;
- bright lines at road and wire height in the 3D view.

**Photos (no 3D scan needed).** Drop a `.jpg`, `.png` or `.tif` on its own and Horus analyses it straight away:

- **Drone / aerial photo taken from above.** Horus puts it on the map:
  - from a DJI photo's GPS, flight height, camera and heading (turned north-up automatically);
  - or from a world file or GeoTIFF;
  - or at the lat/lon you type, with the flight height you type.

  It counts every **tree crown** and rates it **healthy / stressed / dead** from its colour. It also finds **dry grass and bare ground** (fuel that burns easily), **roads** and **power-line clearings**, and lists the trees standing next to them.
- **Photo from the ground** (phone camera into the forest): share of **green, yellowing and brown/grey (dead or dry) foliage** and of **dry grass / litter**, with an overall verdict.
- One photo gives no tree heights, so storm-fall reach and fire spread still need a 3D scan. Upload the photo together with the scan to get both.

**2D / 3D switch.** Every map view has a **2D | 3D** switch in its top-left corner. 3D shows the point cloud of the current scan (also a scan you just uploaded), coloured to match the tab:
- danger to infrastructure on Storm, Tree hazards, Routes, Emergency plan and Upload;
- fire hazard on Wildfire and on Tree hazards → Fire;
- LAS class on Infrastructure;
- true colour on Overview.

Small chips change the colouring (true colour, elevation, LAS class, danger, tree health, fire hazard), and the camera frames the whole scan. Drag to rotate, right-drag to pan, wheel to zoom; switch back to 2D to click places on the map.

**Tree hazards tab.**
- *Storm: trees that fall.* Pick a gust and wind direction, then run a Monte-Carlo storm (300 runs). Every tree's failure is drawn from the windthrow model (species, H/D slenderness, health, exposed edge, peat soil), and fallen trees are laid down around the downwind direction. Outputs: trees falling (90 % range), P(power-line outage), trees across roads, buildings hit, a gust-damage curve, one simulated storm on the map (what each fallen tree hits), and the trees most likely to fall with the reasons.
- *Fire: trees that burn.* Per-tree fire hazard 0–100 combines:
  - dead / dry standing trees;
  - resinous spruce;
  - low crown base (ladder fuel);
  - dry grass or open ground around the tree;
  - today's fuel dryness (FFMC / BUI).

  The **dry grass + dead fuel** layer shows where fire starts and runs easily. Click the map to start a fire. Every tree it reaches is checked for **torching** (Van Wagner 1977 crown-fire criterion: crown base height and foliar moisture vs. the surface-fire intensity under the tree). Calibration: about 10 % of trees torch on a typical day (mostly dead trees and spruce), about 50 % in drought.

**Emergency plan tab.** Pick the emergency type (any / fire suppression / medical-evacuation / power-line repair) and a destination. Horus routes **every vehicle class** and scores each route's danger (0–100):
- the chance a fallen tree blocks it, and clearing time;
- storm-prone trees that can fall onto it;
- fire risk and torching trees beside it, and the simulated fire front;
- slope, narrow sections and branches over the road.

**Vehicle fit:** each vehicle has real dimensions:

| Vehicle | Width | Height | Weight | Minimum road width |
|---|---|---|---|---|
| Fire engine | 2.55 m | 3.3 m | 18 t | 3.0 m |
| Ambulance | 2.3 m | 2.9 m | 5 t | 2.8 m |
| 4x4 utility crew | 2.0 m | 1.9 m | 3.5 t | 2.6 m |
| ATV / quad | 1.2 m | 1.3 m | 0.5 t | 1.5 m |
| Forest machine | 2.9 m | 3.8 m | 20 t | 3.0 m |

Roads narrower than a vehicle needs are not used by it. The answer is a plain recommendation, for example *"send an ambulance - about 6 min, low danger"*. In a storm it may instead read *"first on scene is the rescue team on foot (19 min). Best vehicle: ATV …"*. Fastest, safest and fitting vehicles are listed too.

## 3. A 5-minute demo script

1. **Overview**: 22,459 trees detected individually on a 64 ha estate (Forey Flight Task 0001, Kuopio). 641 of them can reach the 20 kV line.
2. **Wildfire**: switch *Conditions → Drought / heatwave* (FWI 50, extreme). Click near House 6 to ignite. The care home is reached in about 18 min. Press *Evacuation route*: the last vehicle must leave within about 17 min.
3. **Storm & trees**: gust 31 m/s, then *Run storm model*. Felling **11 % of the trees in reach removes 50 % of the strike risk**. Download the CSV work order.
4. **Routes**: destination *Lake cabin 1*, then *Compare all vehicles*. The ambulance cannot get through; the fire engine needs 5 tree clearings; the ATV goes around through open forest.
5. **Drone dispatch**: *Plan storm response*. 154 emergency survey tasks, 118 with processed data within 24 h. Worst-hit areas are seen **21 % sooner** than with business-as-usual queuing.
6. **Live data**: Sentinel-2 canopy-loss map, real OSM power lines and roads, FMI gust and warnings, Metsäkeskus stands and photos for the Kuopio area.
7. **3D point cloud**: colour by *Hazard to assets* and *LAS class* (the detected conductors).
8. **Tree hazards**: *Simulate storm* at 24 m/s to see which trees fall and what they hit. Switch to *Fire: trees that burn* and click near House 6: about 3,000 trees are reached, about 1,000 of them torch.
9. **Emergency plan**: *Medical / evacuation* to Lake cabin 1 recommends the ambulance (about 6 min). Switch *Conditions → Autumn storm* and run it again: the ambulance is blocked, and Horus sends the rescue team on foot first and names the best vehicle.
10. **Upload 3D scan** (orange button, top right): `python -m horus synth --out demo.las`, then choose `demo.las`, or drag it onto the page. Roads, power line, buildings, trees and risks are found from the scan alone.
11. **Data quality**: detection F1 0.73, height RMSE 0.5 m, species 93 %, DTM error 2.5 cm.

---

## 4. What the data is and how the analysis works

### Data

| Input | Source in the demo | Source in production |
|---|---|---|
| LiDAR point cloud (XYZ, class, RGB, NIR) | Deterministic synthetic DJI L3 + multispectral survey (4.2 M points, 6.5 pts/m², LAS classes 2/3/5/6/9/14) with known ground truth | Forey survey, DJI Terra LAS export |
| Weather, 40-day history, 48 h gust forecast | Built-in scenarios | **Open-Meteo API** live (no key); FMI open data as an alternative |
| Roads, buildings, rescue access point | Vector layers in the synthetic estate | Digiroad, NLS topographic database, building register |
| Fleet: 1,000 flight tasks, 33 operators, travel matrix | **Forey's provided dataset** (`data/forey/`) | Forey's operations system |

The challenge zip contains no point cloud, only operational JSON. So Horus ships a synthetic estate generator **with ground truth**: every tree's position, height, species and health is known. That lets every step be *validated*, not just shown. The same pipeline runs unchanged on a real LAS file (`python -m horus synth` → `analyze --las` proves the path).

### Pipeline (`horus/`)

```
LAS ──► lidar.py   ground filter → DTM (0.5 m) → DSM → CHM (pit-filled)
                   → tree tops: variable-window local maxima → crowns: marker watershed
                   → features: crown shape, return-height distribution, NDVI, G/R
                   → species: Gaussian naive Bayes (field-plot calibrated) | health: NDVI
                   → DBH from Näslund height curves, slenderness H/D
                   → power line vectorised from class-14 conductor points
    ──► weather.py Canadian Fire Weather Index (FFMC, DMC, DC, ISI, BUI, FWI) from daily noon weather
    ──► fire.py    fuel types (FBP C-2/C-3/C-4/D-1/M-1/O-1a) from canopy cover, height, species, dead share
                   → rate of spread & Byram intensity → suppression-difficulty class
                   → risk index = hazard × ignition likelihood × exposure × response delay
                   → spread simulation: minimum travel time on 16-neighbour grid, elliptical wind
                     spread (length-to-breadth from wind speed), slope factor exp(3.533·tan^1.2)
    ──► wind.py    P(tree fails) = logistic(gust, H/D, spruce, upwind stand edge, emergence,
                   peat, dead/stressed, wet unfrozen soil)
                   → 36-direction ray casting (von Mises around downwind) over 1 m asset rasters
                   → P(hit) per line span / building / 25 m road piece (sizes weighted)
    ──► discover.py road map from the scan alone (junction-aware corridor network, gap closing,
                   inferred links across clearings), hidden tracks, buildings, power-line wires
    ──► fire.py    per-tree fire hazard + torching (Van Wagner crown-fire criterion), dry-fuel layer
    ──► site.py    Monte-Carlo storm (which trees fall, where, what they hit), emergency plan
                   (all vehicles, route danger, vehicle fit by width/height/weight)
    ──► routing.py 2 m travel-time graph per vehicle class (road speeds, LiDAR slope & canopy
                   density off-road, water/bog), storm blockages as clearing penalties or
                   barriers, fire front as barrier, foot legs where roads end; scipy Dijkstra
    ──► dispatch.py day-by-day insertion scheduler (availability windows, drone config,
                   multi-day tours, transit days) + storm mode with exposure-ranked emergency tasks
```

### Validation (synthetic estate, reference = trees visible from above, ≥ 10 m)

| Metric | Result |
|---|---|
| Tree detection recall / precision / F1 | 0.60 / 0.93 / **0.73** |
| Tree height RMSE / bias | **0.51 m** / −0.20 m |
| Species accuracy, field-plot calibrated (5 % of trees) | **93 %** (pre-trained default model, LiDAR only: 73 %) |
| Health accuracy / dead-tree recall | 98 % / 71 % |
| DTM mean absolute error | **2.5 cm** |
| FWI implementation vs. Van Wagner reference case | exact to 0.01 |
| TM35FIN projection | round-trip < 1e-8° |

Processing time: about 10 s for 64 ha and 4.2 M points on one laptop core.

### Honest limitations

- Fire spread assumes constant weather and models no spotting.
- The windthrow coefficients are literature-informed priors. **Calibrating them on the DSO's tree-fault records is the pilot's first job.**
- Routing ignores bridge and load limits.
- The probabilities rank risk for prioritisation; they are not guarantees.
- LiDAR-only surveys (70 % of Forey's tasks) get a weaker health signal from return density. Multispectral is the upsell.

## 5. Architecture & scaling

```
 Drone (L3 + MS) ─► Forey cloud upload ─► tile splitter (1 km², 30 m overlap)
                                              │  stateless workers (this Python package)
                                              ▼
              trees (GeoParquet/PostGIS) · rasters (COG) · work orders (CSV/GeoJSON)
                                              │
       weather (Open-Meteo/FMI) ─► scenario engine ─► API (horus.server) ─► web UI / DSO GIS / rescue CAD
```

- **Tiles are independent**, so throughput scales linearly with workers. At about 15 s per km² per core (after voxel-thinning dense L3 data to about 20 pts/m²), Forey's whole Sept–Oct dataset (181,709 ha ≈ 1,817 km²) takes about **8 core-hours**.
- Weather-dependent layers (fire, windthrow, routes) are recomputed in under 1 s per site, so they can refresh with every forecast run.
- Rasters are built on a nested grid: 0.5 m CHM, 1 m assets, 2 m routing, 5 m fire.
- **Security:** binds to localhost by default, with optional bearer token and path-traversal protection, and does no external calls except the weather request (site centre only). Power-line and care-home locations are critical-infrastructure data. Production needs role-based access, audit logs, EU data residency, and sharing rasters with rescue services without exposing raw clouds.

## 6. Commercial case

**Problem.** Trees falling on overhead lines cause most storm outages in Finland's forest-covered distribution networks. Storms such as *Asta* (2010) and *Tapani* (2011) left hundreds of thousands of customers without power. The Electricity Market Act requires DSOs to make networks weather-proof (a storm may not cause outages longer than 6 h in town-plan areas or 36 h elsewhere, with deadlines running to 2028/2036). Vegetation management today clears corridors at fixed intervals, blind to *which* trees are actually dangerous.

**Who pays.**
1. **Distribution system operators** (e.g. Elenia, Caruna, Järvi-Suomen Energia, Savon Voima). Hazard-tree work orders per km of line, plus storm rapid assessment on a retainer. This is the first buyer: a clear budget line (ROW clearing) and regulatory pressure.
2. **Regional rescue departments and municipalities.** Fire-risk and response-time maps, evacuation planning, passable-road maps after storms.
3. **Forest owners & insurers.** Post-storm change detection against Forey's pre-storm baseline for damage claims and salvage-harvest planning.
4. Later: road/rail authorities (corridor hazard trees), Defence Forces & preparedness (category 5: terrain passability).

**Why Forey.** Forey already flies these forests, owns the processing chain and holds *pre-event baselines*. Horus reuses the same flights for a second revenue stream, turns the operator network into an emergency service, and makes multispectral (health) an upsell.

**Pricing hypotheses to test in the pilot:** per-km corridor analysis (survey + work order), annual storm-response retainer per DSO region, and per-hectare risk layer for rescue/insurers.

**Pilot (3 months, one DSO, about 50 km of 20 kV forest line in Savo):**
- Fly LiDAR + MS and deliver the ranked work order and map.
- **Success criteria:**
  1. Of Horus's top-200 hazard trees, at least 70 % are confirmed as removal-worthy by the DSO's line inspectors.
  2. Back-test against the DSO's 2018–2025 tree-fault log: Horus's top-decile spans capture at least 40 % of historical tree faults.
  3. Cost per km is at or below the current corridor inspection cost.
- Option: one storm-response exercise with the rescue department.

**First step to market:** an introduction through Forey's existing forest-owner network to one Savo-region DSO's vegetation-management lead, using the back-test on their own fault data as the hook.

## 7. Answers to the challenge's nine questions

1. **Problem:** forests threaten infrastructure and people (fallen trees, wildfire), and the first hours after a disruption are decided blind.
2. **Users:** DSO vegetation managers and dispatchers, rescue incident commanders, and Forey's own operations.
3. **Data:** Forey LiDAR + multispectral point clouds; live weather and FWI; roads and buildings; Forey's fleet dataset.
4. **What we built:** the working Horus app (LiDAR pipeline, fire/storm/routing/dispatch models, web UI with 2D/3D views, CLI, tests).
5. **Logic:** per-tree physics-informed probabilities and FBP fire behaviour, combined with asset geometry and travel-time graphs (section 4).
6. **Decision improved:** which trees to cut first; when to evacuate and by which road; which drone flies where first after a storm.
7. **Forey's role:** data producer, processor and operator network. Horus is a product line on Forey's existing flights.
8. **Who pays:** DSOs first, then rescue services, insurers and forest owners.
9. **First pilot:** 50 km of 20 kV line with one DSO, validated against its own fault history (section 6).

## 8. Project layout

```
horus/            Python package (see module docstrings)
  models/         default species model (tools/train_species.py)
web/              single-page UI: canvas map engine, WebGL point-cloud viewer
data/forey/       Forey's provided dataset (flight tasks, operators, travel times)
tests/            test-suite
tools/            model training
```

Basemaps (OpenStreetMap / Esri imagery) load only when online; the app is fully usable offline.

## Damage in euros (v1.5)

Every storm and fire simulation now reports the expected damage in euros:

- **Storm:** a mean with a 90 % range across the Monte-Carlo runs, split into outage compensation, line repairs, road clearing, building repairs and timber lost. It also lists the trees whose felling cost is lower than the damage they are expected to cause.
- **Fire:** timber lost (torching versus surface fire), reforestation, expected building loss, line repair and outage compensation.
- **Work order:** a "€ at risk" column per tree.

Unit costs live in `horus/costs.py`. Each one is tagged **derived** or **assumption**:

- **Derived:** worked out from published Finnish figures (Elenia Storm Hannes, Fortum 2011 storms, Yle 2014 timber).
- **Assumption:** a placeholder to replace with the customer's own numbers.

To change them, use "Unit costs" in the web UI or `POST /api/site/costs {"values":{"customers_per_line":1000}}`. Changes are saved to `data/costs.json`; send `{"reset":true}` to restore the defaults.
