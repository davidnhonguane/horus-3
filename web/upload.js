// Horus - Upload scan: drop 3D point clouds (and optional photos) -> Horus finds roads, power lines,
// buildings, recognises every tree and ranks the danger to power lines, houses and roads.
"use strict";

const UP = { files: [], batch: null, crs: "auto", lat: "", lon: "", nls: true, osm: false, maxPts: 4000000, job: null, busy: false, results: null };
const UP_EXT = { scan: [".las", ".laz", ".ply", ".obj", ".glb", ".xyz", ".txt", ".csv", ".pts", ".e57"], image: [".jpg", ".jpeg", ".png", ".tif", ".tiff"], side: [".jgw", ".pgw", ".tfw", ".wld", ".geojson", ".json"] };
const upKind = (n) => { const e = n.slice(n.lastIndexOf(".")).toLowerCase(); return Object.keys(UP_EXT).find((k) => UP_EXT[k].includes(e)) || null; };
const mb = (b) => (b > 1e9 ? fmt(b / 1e9, 2) + " GB" : fmt(b / 1e6, 1) + " MB");
const newBatch = () => "u" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

function upSend(f) {
  return new Promise((res, rej) => {
    const x = new XMLHttpRequest();
    x.open("POST", tok(`/api/upload/file?batch=${UP.batch}&name=${encodeURIComponent(f.file.name)}`));
    if (TOKEN) x.setRequestHeader("Authorization", "Bearer " + TOKEN);
    x.setRequestHeader("Content-Type", "application/octet-stream");
    x.upload.onprogress = (e) => { if (e.lengthComputable) { f.progress = e.loaded / e.total; upFiles(); } };
    x.onload = () => { let j = {}; try { j = JSON.parse(x.responseText); } catch (e) { /* */ }
      if (x.status === 200) { f.state = "uploaded"; f.progress = 1; res(j); } else { f.state = "error"; f.error = j.error || x.statusText; rej(new Error(f.error)); } upFiles(); };
    x.onerror = () => { f.state = "error"; f.error = "network error"; upFiles(); rej(new Error("network error")); };
    f.state = "uploading"; upFiles();
    x.send(f.file);
  });
}

function upFiles() {
  const el = $("#upfiles");
  if (!el) return;
  el.innerHTML = UP.files.length ? UP.files.map((f, i) => `<div class="upf ${f.state}">
      <div class="n"><span class="k ${f.kind}">${f.kind === "scan" ? "3D scan" : f.kind === "image" ? "photo" : "extra"}</span> ${esc(f.file.name)} <span class="sub">${mb(f.file.size)}</span>
      ${f.state === "queued" && !UP.busy ? `<button class="x" data-rm="${i}" title="remove">×</button>` : ""}</div>
      <div class="bar2"><div style="width:${(f.progress || 0) * 100}%"></div></div>
      ${f.error ? `<div class="sub" style="color:var(--red)">${esc(f.error)}</div>` : ""}</div>`).join("") : "";
  el.querySelectorAll("[data-rm]").forEach((b) => b.addEventListener("click", () => { UP.files.splice(+b.dataset.rm, 1); window.tabUpload(); }));
}

function upAdd(list) {
  const bad = [];
  let newScan = false;
  if (UP.job && UP.job.state !== "running") { UP.files = []; UP.job = null; }   // new files after a finished run = a new upload
  for (const file of list) {
    const kind = upKind(file.name);
    if (!kind) { bad.push(file.name); continue; }
    if (!UP.files.some((f) => f.file.name === file.name && f.file.size === file.size)) {
      UP.files.push({ file, kind, state: "queued", progress: 0 });
      if (kind === "scan" || kind === "image") newScan = true;
    }
  }
  UP.error = bad.length ? `Horus can't read ${bad.join(", ")}. Upload a 3D scan (${UP_EXT.scan.join(" ")}). iPhone/iPad .usdz scans: export them as .glb, .obj or .ply from the scanning app.` : null;
  UP.results = null; UP.photos = null;
  window.tabUpload();
  // start straight away when a 3D scan or a photo was added (no extra click needed)
  if (newScan && !UP.busy) setTimeout(upRun, 250);
}

async function upRun() {
  if (!UP.files.some((f) => f.kind === "scan" || f.kind === "image")) { UP.error = "Add a 3D scan (" + UP_EXT.scan.join(" ") + ") or a photo (.jpg .png .tif)."; window.tabUpload(); return; }
  UP.busy = true; UP.batch = newBatch(); UP.job = null; UP.results = null; UP.photos = null; UP.error = null; UP.t0 = Date.now();
  window.tabUpload();
  try {
    for (const f of UP.files) { f.state = "queued"; f.progress = 0; f.error = null; }
    for (const f of UP.files) {
      try { await upSend(f); } catch (e) { if (f.kind !== "side") throw new Error(`${f.file.name}: ${e.message}`); }
    }
    UP.job = await api("/api/upload/analyze", { batch: UP.batch, crs: UP.crs, lat: UP.lat, lon: UP.lon, nls: UP.nls, osm: UP.osm,
      max_points: UP.maxPts, photo_view: UP.pview, photo_height_m: UP.pheight, scenario: $("#scenario").value === "live" ? "normal" : $("#scenario").value });
    upLog();
    while (UP.job.state === "running") {
      await new Promise((r) => setTimeout(r, 1200));
      UP.job = await api("/api/upload/status");
      upLog();
    }
    if (UP.job.state === "done" && UP.job.photos) {
      UP.photos = UP.job.photos;
    } else if (UP.job.state === "done") {
      await loadSite();
      const [inf, st, wo, ft] = await Promise.all([api("/api/site/infrastructure"), api("/api/site/structures"), api("/api/site/workorder?top=10"), api("/api/site/fire_trees?top=8")]);
      UP.results = { inf, st, wo, ft };
    } else if (UP.job.state === "failed") {
      UP.error = UP.job.error || "the analysis failed - see the log";
    }
  } catch (e) { UP.error = e.message; }
  UP.busy = false;
  window.tabUpload();
  if (UP.results || UP.photos) { const r = $("#upres"); if (r) r.scrollIntoView({ behavior: "smooth", block: "start" }); }
}

function upLog() {
  const el = $("#uplog");
  if (!el) return;
  const up = UP.files.filter((f) => f.state === "uploading");
  const secs = UP.t0 ? Math.round((Date.now() - UP.t0) / 1000) : 0;
  let h = (UP.job ? (UP.job.log || []).map((l) => `<div>${esc(l)}</div>`).join("") : "");
  if (UP.busy) h += `<div class="working"><span class="spin"></span>${up.length ? `Uploading ${esc(up[0].file.name)} … ${pct(up[0].progress || 0)}` : "Analysing the scan …"} <span class="sub">${secs} s</span></div>`;
  el.innerHTML = h;
}

function simBlock() {
  const F = S.fire, St = window.HZ && HZ.storm;
  if (!F || !F.trees || !St) return "";
  const fr = F.first_reached || [];
  return `<h3>Simulations on this scan</h3>
    <div class="kpis">
      ${kpi(fmt(St.fallen.mean), `trees fall in a ${fmt(St.gust_ms)} m/s storm`, "warn")}${kpi(pct(St.p_line_outage), "P(power-line outage)", St.p_line_outage > 0.5 ? "bad" : "")}
      ${kpi(fmt(St.trees_on_roads_mean, 1), "trees across roads")}${kpi(fmt(St.buildings_hit_mean, 2), "buildings hit (mean)")}
      ${kpi(fmt(F.trees.reached), `trees reached by a ${fmt(F.hours)} h fire`, "warn")}${kpi(fmt(F.trees.torching), "trees torching (crown fire)", "bad")}
      ${St.damage_eur ? kpi(eur(St.damage_eur.total_mean), `storm damage (90 %: ${eur(St.damage_eur.total_p5)}–${eur(St.damage_eur.total_p95)})`, "bad") : ""}
      ${F.damage_eur ? kpi(eur(F.damage_eur.total), "fire damage", "bad") : ""}
    </div>
    ${St.damage_eur && St.damage_eur.trees_worth_felling ? `<p class="note"><b>Felling ${fmt(St.damage_eur.trees_worth_felling)} trees (${eur(St.damage_eur.felling_cost)}) avoids ${eur(St.damage_eur.damage_avoided_by_felling)}</b> of expected storm damage. Details and unit costs in Tree hazards.</p>` : ""}
    ${fr.length ? `<p class="note">The fire was started at the most dangerous spot (highest fire risk near houses, roads and lines) in today's conditions. It reaches first: ${fr.slice(0, 3).map((i) => `${esc(i.asset)} in ${fmt(i.minutes)} min`).join(", ")}.</p>` : ""}
    <div style="margin:6px 0 4px"><button class="btn" data-sim="fire">▶ Fire in 3D</button> <button class="btn" data-sim="storm">▶ Storm in 3D</button> <button class="btn ghost" data-sim="fire2d">Fire on the map</button></div>`;
}

function upResults() {
  const s = S.sum, R = UP.results, I = R.inf, tr = s.trees, st = s.storm, f = s.fire;
  const nr = (a) => a.filter((x) => !x.mapped).length;
  const newRoads = nr(I.lidar_roads) + I.image_roads.filter((x) => !x.mapped && x.evidence === "image").length;
  const hi = R.st.filter((x) => x.level === "high");
  return `<h3>Results · ${esc(s.name)}</h3>
    ${s.meta.placement ? `<div class="callout">${esc(s.meta.placement)}</div>` : ""}
    <div class="kpis">
      ${kpi(fmt(tr.count), "trees recognised")}
      ${kpi(`${fmt(tr.health.dead)} / ${fmt(tr.health.stressed)}`, "dead / stressed trees", tr.health.dead > 0 ? "warn" : "")}
      ${kpi(fmt(S.vec.roads.length), `roads · ${fmt(newRoads)} new forest road(s) found`, newRoads ? "good" : "")}
      ${kpi(fmt(s.powerlines.length), `power line(s) · ${fmt(nr(I.wires))} found in the 3D scan`, s.powerlines.length ? "" : "")}
      ${kpi(fmt(nr(I.buildings)), "unmapped buildings found", nr(I.buildings) ? "warn" : "")}
      ${kpi(fmt(st.trees_reaching_line), "trees that can hit a power line", st.trees_reaching_line ? "warn" : "")}
      ${kpi(fmt(st.trees_easily_fall_design), "trees that fall easily (design storm)", "warn")}
      ${kpi(fmt(f.trees_very_high_hazard), "trees that burn very easily", "bad")}
    </div>
    ${simBlock()}
    <h3>Species &amp; health</h3>
    ${bars(Object.fromEntries(Object.entries(tr.species).map(([k, v]) => [k, v / Math.max(tr.count, 1)])), "var(--teal)")}
    <p class="note">${esc(s.classification.health_source)}.</p>
    <h3>Structures in danger</h3>
    ${R.st.length ? `<table><tr><th>Structure</th><th>Level</th><th>Why</th></tr>${R.st.slice(0, 8).map((b) => `<tr class="click" data-lon="${b.lon}" data-lat="${b.lat}"><td>${esc(b.name)}<br><span class="sub">${esc(b.kind.replace("_", " "))}</span></td><td>${lvlBadge(b.level)}</td><td class="sub">${esc(b.reasons.join("; ") || "-")}</td></tr>`).join("")}</table>` : `<p class="note">No buildings in or near the scan${UP.nls ? "" : " (tick NLS to load houses and critical infrastructure)"}.</p>`}
    <h3>Hazard trees threatening lines, houses &amp; roads</h3>
    ${R.wo.length ? `<table><tr><th>#</th><th>Tree</th><th>Threatens</th><th class="num">P(hit)</th><th>Action</th></tr>${R.wo.map((r) => `<tr class="click" data-lon="${r.lon}" data-lat="${r.lat}"><td>${r.rank}</td><td>${esc(r.species)} ${fmt(r.height_m, 1)} m<br><span class="sub">${esc(r.health)}</span></td><td>${esc(r.threatens)} <span class="sub">${fmt(r.distance_m, 1)} m</span></td><td class="num">${pct(Math.max(r.p_hit_power, r.p_hit_building, r.p_block_road), 0)}</td><td>${esc(r.action)}</td></tr>`).join("")}</table>` : `<p class="note">No tree can reach a power line, building or road.</p>`}
    <div style="margin-top:12px"><button class="btn" data-go="hazards">Tree hazards (storm / fire) →</button> <button class="btn ghost" data-go="plan">Emergency plan →</button> <button class="btn ghost" data-go="infra">Infrastructure map →</button></div>
    ${hi.length ? `<p class="note">${hi.length} structure(s) at high risk.</p>` : ""}`;
}

function photoMap() {
  const P = (UP.photos || []).filter((p) => p.bounds);
  if (!P.length) return false;
  map.clearLayers("");
  P.forEach((p, i) => map.setLayer("photo" + i, { type: "image", url: tok(p.overlay_url), bounds: p.bounds, z: 5 + i, opacity: 1 }));
  const b = P[0].bounds;
  if (UP._fitPhoto !== UP.batch) { map.fit(b, 40); UP._fitPhoto = UP.batch; }
  legend(lgRows([["#3ce66e", "healthy tree crown"], ["#ffbe28", "stressed / yellowing"], ["#ff3228", "dead / dry tree"], ["#ff9628", "road found in the photo"], ["#50d2ff", "power-line clearing"], ["#ffd23c", "dry grass / bare ground (burns easily)"]]));
  return true;
}

function photoResults() {
  return UP.photos.map((p, i) => {
    const head = `<h3>Photo ${UP.photos.length > 1 ? i + 1 + " · " : ""}${esc(p.name)}</h3><p class="note">${esc(p.placement)}${p.camera ? " · " + esc(p.camera) : ""}</p>
      <a href="${tok(p.overlay_url)}" target="_blank" title="open full size"><img class="phimg" src="${tok(p.overlay_url)}" alt="analysed photo"></a>`;
    if (p.view === "ground") {
      const f = p.foliage;
      return head + `<div class="kpis">
        ${kpi(pct(f.green), "green, healthy foliage", "good")}${kpi(pct(f.yellowing), "yellowing / stressed foliage", f.yellowing > 0.2 ? "warn" : "")}
        ${kpi(pct(f.brown_grey), "brown / grey: dead or dry", f.brown_grey > 0.25 ? "bad" : "")}${kpi(pct(p.dry_ground_share), "dry grass / litter on the ground", p.dry_ground_share > 0.4 ? "warn" : "")}
      </div>
      <div class="callout ${p.overall === "dead / dry" ? "red" : ""}"><b>Overall: ${esc(p.overall)}.</b> ${p.foliage.brown_grey > 0.25 || p.dry_ground_share > 0.4 ? "Lots of dry fuel: this stand burns easily in dry weather, and dead trees can fall in a storm." : "No large share of dead or dry fuel visible."}</div>
      <p class="note">A photo from the ground shows tree condition, not positions. For a map of every tree and the danger to lines, houses and roads, upload a 3D scan or a drone photo taken straight down.</p>`;
    }
    const t = p.trees, c = p.cover;
    return head + `<div class="kpis">
        ${kpi(fmt(t.count), `tree crowns (${fmt(t.per_ha)} per ha)`)}${kpi(`${fmt(t.dead)} / ${fmt(t.stressed)}`, "dead / stressed trees", t.dead ? "bad" : "")}
        ${kpi(pct(c.dry_grass_bare), "dry grass / bare ground (burns easily)", c.dry_grass_bare > 0.15 ? "warn" : "")}${kpi(pct(c.canopy), "tree canopy cover")}
        ${kpi(fmt(p.roads.length), "roads found in the photo")}${kpi(fmt(p.corridors.length), "power-line clearings")}
      </div>
      <p class="note">Area ${fmt(p.area_ha, 2)} ha · ${fmt(p.gsd_m * 100, 1)} cm per pixel${p.flight_height_m ? ` · flight height ${fmt(p.flight_height_m)} m` : ""}.</p>
      ${p.danger.length ? `<h3>Trees next to roads / power-line clearings</h3><table><tr><th>Tree</th><th>Next to</th><th class="num">Distance</th></tr>${p.danger.slice(0, 12).map((d) => `<tr><td>${esc(d.state)} · crown ${fmt(d.diameter_m, 1)} m</td><td>${esc(d.near)}</td><td class="num">${fmt(d.dist_m, 1)} m</td></tr>`).join("")}</table>` : ""}
      <p class="note">From one photo Horus sees crowns and their colour, not tree heights, so storm-fall reach and fire spread need a 3D scan of the same place. Upload it together with this photo.</p>`;
  }).join("");
}

window.tabUpload = function () {
  siteView(); siteBase({ hillOpacity: 0.6 });
  if (UP.photos && photoMap()) {
    /* photo shown on the map */
  } else if (UP.results) {
    const T = S.trees, mx = Math.max(1e-6, ...T.filter((t) => t.tg > 0).map((t) => t.rd).sort((a, b) => b - a).slice(0, 50));
    hzTrees((t) => (t.tg > 0 && t.rd / mx > 0.06 ? [riskCol(t.rd / mx), 2.4] : t.hl === 2 ? ["#ff4b3e", 1.8] : ["rgba(70,110,85,.5)", 0.9]));
    legend(lgRamp("Danger to infrastructure", RISK, "low", "high") + lgRows([["#ff4b3e", "dead tree"]]) + `<h4 style="margin-top:8px">Infrastructure</h4>${INFRA_LG}`);
  } else legend("");
  const scans = UP.files.filter((f) => f.kind === "scan");
  const demo = S.sum && S.sum.meta.kind === "synthetic";
  panel(`
    <h2>Upload a 3D scan</h2>
    <p class="lead">Add a 3D scan or a photo of a forest and Horus analyses it straight away. A photo alone gives tree crowns and their health, dry grass, roads and line clearings; a 3D scan adds tree heights and the full storm / fire / route analysis. Horus maps the roads and finds unmapped forest roads, detects power lines and buildings, recognises every tree (species, health) and ranks the danger to power lines, houses and roads, then runs the fire, storm and routing models on it.</p>
    <div class="drop" id="drop">
      <div class="big">⇪</div>
      <div><b>Drop files here</b> or <label class="lnk">browse<input type="file" id="upin" multiple hidden accept=".las,.laz,.ply,.obj,.glb,.xyz,.txt,.csv,.pts,.e57,.jpg,.jpeg,.png,.tif,.tiff,.jgw,.pgw,.tfw,.wld,.geojson,.json"></label></div>
      <div class="sub"><b>3D scans:</b> .las .laz (drone / airborne LiDAR, DJI Terra, NLS) · .ply .obj .glb (3D models from photogrammetry or phone 3D-scanning apps) · .xyz .txt .csv .pts (point lists) · .e57 (laser scanners)<br>optional photo / orthomosaic: .jpg .png .tif + world file (.jgw .pgw .tfw) · optional .geojson with known roads, lines and houses</div>
    </div>
    <div id="upfiles"></div>
    <h3>Options</h3>
    <div class="formrow"><label>Coordinates</label><select id="upcrs">
      <option value="auto" ${UP.crs === "auto" ? "selected" : ""}>Detect automatically</option>
      <option value="EPSG:3067" ${UP.crs === "EPSG:3067" ? "selected" : ""}>ETRS-TM35FIN (EPSG:3067) – Finnish standard</option>
      <option value="local" ${UP.crs === "local" ? "selected" : ""}>Local / not georeferenced</option></select></div>
    <div class="formrow"><label>Place at lat / lon</label><input type="text" id="uplat" placeholder="62.7471" value="${esc(UP.lat)}" style="width:86px"><input type="text" id="uplon" placeholder="27.2595" value="${esc(UP.lon)}" style="width:86px"></div>
    <p class="note" style="margin-top:-2px">Only needed for scans without real-world coordinates.</p>
    <div class="formrow"><label>Land Survey (NLS)</label><input type="checkbox" id="upnls" ${UP.nls ? "checked" : ""}> houses, critical infrastructure, roads, power lines + orthophoto (uses your NLS key)</div>
    <div class="formrow"><label>OpenStreetMap</label><input type="checkbox" id="uposm" ${UP.osm ? "checked" : ""}> roads, power lines, fire station</div>
    <div class="formrow"><label>Max points</label><select id="upmp">${[1e6, 2e6, 4e6, 8e6].map((v) => `<option value="${v}" ${v === UP.maxPts ? "selected" : ""}>${fmt(v / 1e6)} million (larger = slower)</option>`).join("")}</select></div>
    <div class="formrow"><label>Photo taken</label><select id="uppv">${[["auto", "detect automatically"], ["above", "from above (drone / aerial)"], ["ground", "from the ground"]].map(([k, v]) => `<option value="${k}" ${k === (UP.pview || "auto") ? "selected" : ""}>${v}</option>`).join("")}</select></div>
    <div class="formrow"><label>Flight height (m)</label><input type="text" id="upph" placeholder="from the photo" value="${esc(UP.pheight || "")}" style="width:90px"><span class="sub">only for photos without drone data</span></div>
    <button class="btn" id="upgo" ${UP.files.some((f) => f.kind !== "side") && !UP.busy ? "" : "disabled"}>${UP.busy ? "Working…" : UP.results ? "Analyse again" : "Upload &amp; analyse"}</button>
    ${!demo ? `<button class="btn ghost" id="updemo" ${UP.busy ? "disabled" : ""}>Back to demo estate</button>` : ""}
    ${UP.error ? `<div class="callout red"><b>Not analysed:</b> ${esc(UP.error)}</div>` : ""}
    <div class="log" id="uplog"></div>
    <div id="upres">${UP.photos ? photoResults() : UP.results ? upResults() : ""}</div>
    <p class="note">Files stay on this computer (saved under data/uploads). .las .ply .obj .glb .xyz work out of the box; .laz needs <code>pip install "laspy[lazrs]"</code>, .e57 needs <code>pip install pye57</code>. 3D models without real-world coordinates are placed at the lat/lon you give (or the Kuopio demo point).</p>
  `);
  upFiles(); upLog();
  if (UP.busy && !UP._tick) UP._tick = setInterval(() => { if (!UP.busy) { clearInterval(UP._tick); UP._tick = null; } upLog(); }, 1000);
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("on"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("on"); }));
  drop.addEventListener("drop", (e) => { if (!UP.busy) upAdd([...e.dataTransfer.files]); });
  $("#upin").onchange = (e) => upAdd([...e.target.files]);
  $("#upcrs").onchange = (e) => (UP.crs = e.target.value);
  $("#uplat").oninput = (e) => (UP.lat = e.target.value.trim());
  $("#uplon").oninput = (e) => (UP.lon = e.target.value.trim());
  $("#upnls").onchange = (e) => (UP.nls = e.target.checked);
  $("#uposm").onchange = (e) => (UP.osm = e.target.checked);
  $("#upmp").onchange = (e) => (UP.maxPts = +e.target.value);
  $("#uppv").onchange = (e) => (UP.pview = e.target.value);
  $("#upph").oninput = (e) => (UP.pheight = e.target.value.trim());
  $("#upgo").onclick = upRun;
  if ($("#updemo")) $("#updemo").onclick = async () => { busy(true); try { await api("/api/site/demo", {}); await loadSite(); UP.results = null; } catch (e) { alert(e.message); } busy(false); window.tabUpload(); };
  document.querySelectorAll("#panel [data-go]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); setTab(a.dataset.go); }));
  document.querySelectorAll("#panel [data-sim]").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.sim;
    HZ.mode = k === "storm" ? "storm" : "fire";
    S.mode3dUser = null;
    S.dim = k === "fire2d" ? "2d" : "3d";
    setTab("hazards");
  }));
  bindZoom();
  if (typeof dimUpdate === "function") dimUpdate();      // a new scan reloads the 3D view
};

// ---- always-visible entry points: top-bar button, drag & drop anywhere, version badge
(function () {
  const nb = $("#newBadge");
  if (nb) nb.addEventListener("click", () => { S.dim = "2d"; setTab("infra"); });
  const btn = $("#upTop");
  if (btn) btn.addEventListener("click", () => { setTab("upload"); const i = $("#upin"); if (i && !UP.busy) i.click(); });
  const ov = $("#dropall");
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; if (ov && S.tab !== "upload") ov.classList.remove("hidden"); });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth && ov) ov.classList.add("hidden"); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; if (ov) ov.classList.add("hidden");
    if (e.target.closest && e.target.closest("#drop")) return;      // the upload box handles its own drops
    if (UP.busy) return;
    setTab("upload"); upAdd([...e.dataTransfer.files]);
  });
  document.addEventListener("click", (e) => { const a = e.target.closest && e.target.closest(".upcall [data-go]"); if (a) { e.preventDefault(); setTab("upload"); } });
  fetch(tok("/api/health")).then((r) => r.json()).then((j) => { const v = $("#ver"); if (v && j.version) v.textContent = "v" + j.version; }).catch(() => {});
})();
