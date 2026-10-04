// Horus web app
"use strict";

const Q = new URLSearchParams(location.search);
const TOKEN = Q.get("token");
const $ = (s, el = document) => el.querySelector(s);
const fmt = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (v, d = 0) => (v == null ? "–" : fmt(v * 100, d) + "%");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const SPECIES = ["pine", "spruce", "birch"], HEALTH = ["healthy", "stressed", "dead"], TARGET = ["–", "road", "power line", "building"];
const SP_COL = ["#c9b458", "#2f8a5b", "#9ed06a"];
const VEH = {
  fire_engine: ["Fire engine", "#e5534b"], ambulance: ["Ambulance", "#f2f2f2"], pickup_4x4: ["4x4 utility crew", "#5aa9e6"],
  atv: ["ATV / quad", "#a685e2"], forwarder: ["Forest machine", "#e9a23b"], foot: ["On foot", "#6cc070"],
};

async function api(path, body) {
  const headers = { "Content-Type": "application/json" };
  if (TOKEN) headers.Authorization = "Bearer " + TOKEN;
  const r = await fetch(path, { method: body ? "POST" : "GET", headers, body: body ? JSON.stringify(body) : undefined });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const tok = (u) => (TOKEN ? u + (u.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(TOKEN) : u);

function busy(on) { $("#busy").classList.toggle("hidden", !on); }
function hint(text) { const h = $("#hint"); h.textContent = text || ""; h.classList.toggle("hidden", !text); }

// colour helpers
function lerpCol(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [t0, a] = stops[i - 1], [t1, b] = stops[i], u = (t - t0) / (t1 - t0 || 1);
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * u)).join(",")})`;
  }
  return `rgb(${stops[stops.length - 1][1].join(",")})`;
}
const RISK = [[0, [255, 236, 140]], [0.3, [250, 180, 50]], [0.6, [235, 100, 35]], [1, [200, 25, 45]]];
const riskCol = (t) => lerpCol(RISK, t);
const rampCss = (stops) => `linear-gradient(90deg, ${stops.map(([t, c]) => `rgb(${c.join(",")}) ${t * 100}%`).join(",")})`;

// ---------------------------------------------------------------- state
const S = { tab: "overview", dim: "2d", siteVer: 0, simVer: 0, hl: [], ver: 1, treeMode: "species", fireLayer: "fire_risk", route: [], click: null, workorder: [] };
const map = new MapView($("#map"));
let viewer = null;

map.onhover = (hit, e) => {
  const tip = $("#tip");
  if (!hit || !hit.layer.tip) { tip.classList.add("hidden"); return; }
  const r = $("#mapwrap").getBoundingClientRect();
  tip.innerHTML = hit.layer.tip(hit.index);
  tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 270) + "px";
  tip.style.top = e.clientY - r.top + 12 + "px";
  tip.classList.remove("hidden");
};
map.onclick = (ev) => { if (S.click) S.click(ev); };

// ---------------------------------------------------------------- loading
async function loadSite() {
  const [sum, vec, tr] = await Promise.all([api("/api/site/summary"), api("/api/site/vectors"), api("/api/site/trees")]);
  S.sum = sum; S.vec = vec;
  const n = tr.lon.length;
  S.trees = new Array(n);
  for (let i = 0; i < n; i++) S.trees[i] = { lon: tr.lon[i], lat: tr.lat[i], h: tr.h[i], sp: tr.species[i], hl: tr.health[i], pf: tr.p_fail_now[i], rn: tr.risk_now[i], rd: tr.risk_design[i], tg: tr.target[i], fh: tr.fire_hazard[i], pt: tr.p_torch[i] };
  $("#siteName").textContent = `${sum.name} · ${fmt(sum.area_ha, 0)} ha · ${sum.meta.kind === "synthetic" ? "synthetic DJI L3 survey" : "LiDAR survey"}`;
  S._fitted = false; S.fire = null; S.route = []; S.rt = null; S.ver++; S.siteVer++;
  if (typeof INF !== "undefined") INF.data = null;
  if (typeof V !== "undefined") V.list = null;
  if (window.HZ) { HZ.storm = null; HZ.fire = null; HZ.plan = null; }
  try { S.hl = await api("/api/site/highlights"); } catch (e) { S.hl = []; }
  try { applySims(await api("/api/site/sims")); } catch (e) { /* no simulations yet */ }
  simChanged();
  updateBadges();
}
function simChanged() { S.simVer++; }
// fire + storm simulations made automatically for a new scan -> Wildfire / Tree hazards / 3D
function applySims(sims) {
  if (!sims || !sims.fire) return;
  S.fire = sims.fire;
  if (window.HZ) { HZ.fire = Object.assign({}, sims.fire, { ver: S.ver }); HZ.storm = sims.storm || null; }
  simChanged();
}
async function refreshScenario(body) {
  busy(true);
  try {
    S.sum = await api("/api/site/scenario", body);
    const [vec, tr] = await Promise.all([api("/api/site/vectors"), api("/api/site/trees")]);
    S.vec = vec;
    for (let i = 0; i < S.trees.length; i++) { const t = S.trees[i]; t.pf = tr.p_fail_now[i]; t.rn = tr.risk_now[i]; t.rd = tr.risk_design[i]; t.tg = tr.target[i]; t.fh = tr.fire_hazard[i]; t.pt = tr.p_torch[i]; }
    if (window.HZ) { HZ.storm = null; HZ.fire = null; HZ.plan = null; }
    S.ver++; S.fire = null; S.route = []; simChanged();
    updateBadges(); render();
  } catch (e) { alert(e.message); }
  busy(false);
}
function updateBadges() {
  const w = S.sum.weather;
  const dc = { low: "#4cc3a6", moderate: "#9ed06a", high: "#e9a23b", "very high": "#ef7b3e", extreme: "#e5534b" }[w.danger] || "#ccc";
  $("#fwiBadge").innerHTML = `FWI <span style="color:${dc}">${fmt(w.fwi, 1)} ${esc(w.danger)}</span>`;
  $("#gustBadge").textContent = `Gust ${fmt(w.gust_ms, 0)} m/s from ${fmt(w.wind_dir_deg, 0)}°`;
  $("#scenario").value = w.scenario === "live" ? "live" : w.scenario;
  const nb = $("#newBadge");
  if (nb) {
    const r = S.hl.filter((h) => h.kind === "road"), pw = S.hl.filter((h) => h.kind === "power");
    const km = r.reduce((a, h) => a + h.length_m, 0) / 1000;
    nb.innerHTML = r.length || pw.length ? `★ NEW: ${r.length ? `${fmt(km, 1)} km of road` : ""}${r.length && pw.length ? " · " : ""}${pw.length ? `${pw.length} power line${pw.length > 1 ? "s" : ""}` : ""}` : "";
    nb.classList.toggle("hidden", !(r.length || pw.length));
  }
}

// ---------------------------------------------------------------- shared site layers
function siteBase(opts = {}) {
  map.clearLayers("");
  map.setLayer("hill", { type: "image", url: tok(`/api/site/layer/hillshade.png`), bounds: S.sum.bounds, z: 0, opacity: opts.hillOpacity ?? (map.basemap === "none" ? 1 : 0.55) });
  infraLayers(opts);
}
function infraLayers(opts = {}) {
  const v = S.vec;
  map.setLayer("roads", { type: "lines", z: 50, data: v.roads.map((r) => r.discovered
    ? { path: r.path, color: "#ff9a3c", width: 3, dash: [6, 4], casing: "rgba(0,0,0,.6)" }
    : { path: r.path, color: r.cls === "main" ? "#f4f1e8" : "#d9cfb3", width: r.cls === "main" ? 4 : 2.5, casing: "rgba(0,0,0,.55)" }) });
  map.setLayer("power", { type: "lines", z: 55, data: v.powerlines.map((p) => ({ path: p.path, color: p.discovered ? "#ff5fd2" : "#59d0ff", width: 2.2, dash: [7, 4], casing: "rgba(0,0,0,.6)" })) });
  map.setLayer("bld", {
    type: "points", z: 60, hover: true, data: v.buildings.map((b) => ({ lon: b.pos[0], lat: b.pos[1], r: b.discovered ? 7 : 6, color: b.discovered ? "#b07cff" : b.kind === "care_home" ? "#ff6b9a" : "#f0f0f0", stroke: b.discovered ? "#fff" : "#111", label: opts.labels === false ? null : b.discovered ? "Found: building (3D)" : b.name })),
    labelZoom: 15.5, tip: (i) => { const b = v.buildings[i]; return `<b>${esc(b.name)}</b><br>${esc(b.kind.replace("_", " "))}<br>P(tree strike, current storm): ${pct(b.p_hit, 1)}`; },
  });
  if (S.hl && S.hl.length) {                    // always highlight newly found roads / power lines
    map.setLayer("hlglow", { type: "lines", z: 49, data: S.hl.map((h) => ({ path: h.path, color: h.kind === "power" ? "rgba(255,64,230,.38)" : "rgba(255,150,40,.38)", width: 12 })) });
    map.setLayer("hlline", { type: "lines", z: 58, data: S.hl.map((h) => ({ path: h.path, color: h.kind === "power" ? "#ff40e6" : "#ff9628", width: 3.5, dash: [8, 4], casing: "rgba(0,0,0,.7)" })) });
    const big = S.hl.map((h) => h.length_m).sort((a, b) => b - a)[Math.min(5, S.hl.length - 1)] || 0;   // label the longest finds only
    map.setLayer("hllab", { type: "points", z: 64, labelZoom: 16, hover: true,
      data: S.hl.map((h) => { const m = h.path[Math.floor(h.path.length / 2)]; return { lon: m[0], lat: m[1], r: 4, color: h.kind === "power" ? "#ff40e6" : "#ff9628", stroke: "#fff", label: h.kind === "power" || h.length_m >= big ? (h.kind === "power" ? "NEW power line" : "NEW road") : null }; }),
      tip: (i) => { const h = S.hl[i]; return `<b>${h.kind === "power" ? "Power line" : "Road"} found in the scan</b><br>not in any map data<br>${fmt(h.length_m)} m${h.width_m ? ` · ~${fmt(h.width_m, 1)} m wide` : ""}${h.height_m ? ` · wires ${fmt(h.height_m, 1)} m up` : ""}`; } });
  }
  map.setLayer("depot", { type: "points", z: 61, data: [{ lon: v.depot[0], lat: v.depot[1], r: 7, color: "#e9a23b", stroke: "#111", label: "Rescue access point" }] });
}
function treeLayer(mode, filter) {
  S.treeMode = mode;
  const T = S.trees;
  const maxRd = Math.max(1e-6, ...T.filter((t) => t.tg > 0).map((t) => t.rd).sort((a, b) => b - a).slice(0, 50));
  const maxRn = Math.max(1e-6, ...T.filter((t) => t.tg > 0).map((t) => t.rn).sort((a, b) => b - a).slice(0, 50));
  for (const t of T) {
    if (mode === "species") t.color = SP_COL[t.sp];
    else if (mode === "health") t.color = t.hl === 2 ? "#ff4b3e" : t.hl === 1 ? "#f5b941" : "rgba(60,140,90,.75)";
    else if (mode === "risk_design") t.color = t.tg > 0 && t.rd / maxRd > 0.06 ? riskCol(t.rd / maxRd) : "rgba(70,110,85,.45)";
    else if (mode === "risk_now") t.color = t.tg > 0 && t.rn / maxRn > 0.06 ? riskCol(t.rn / maxRn) : t.pf > 0.25 ? "rgba(150,170,255,.85)" : "rgba(70,110,85,.45)";
    t.r = Math.max(0.9, t.h * 0.075);
    const hot = (mode === "risk_design" && t.tg > 0 && t.rd / maxRd > 0.06) || (mode === "risk_now" && (t.tg > 0 && t.rn / maxRn > 0.06)) || (mode === "health" && t.hl > 0);
    t.minPx = hot ? 2.4 : 0.9;
  }
  map.setLayer("trees", {
    type: "points", z: 30, metres: true, minRadiusPx: 0.9, n: T.length, get: (i) => (filter && !filter(T[i]) ? null : T[i]), hover: true,
    hoverFilter: (p) => map.zoom >= 16.2,
    tip: (i) => { const t = T[i]; return `<b>${SPECIES[t.sp]}</b> · ${fmt(t.h, 1)} m · ${HEALTH[t.hl]}<br>P(fail) now ${pct(t.pf, 1)}<br>threatens: ${TARGET[t.tg]}<br>design-storm risk ${fmt(t.rd, 3)}`; },
  });
}

// ---------------------------------------------------------------- legend
function legend(html) { $("#legend").innerHTML = html; }
const lgRows = (items) => items.map(([c, l]) => `<div class="row"><span class="sw" style="background:${c}"></span>${esc(l)}</div>`).join("");
const lgRamp = (title, stops, a, b) => `<h4>${title}</h4><div class="ramp" style="background:${rampCss(stops)}"></div><div class="ends"><span>${a}</span><span>${b}</span></div>`;
const INFRA_LG = lgRows([["#f4f1e8", "Road (map data)"], ["#ff9a3c", "Potential forest road (found in LiDAR)"], ["#59d0ff", "Power line"], ["#ff5fd2", "Power line found in LiDAR (unmapped)"], ["#b07cff", "Building found in forest (unmapped)"], ["#ff6b9a", "Care home"], ["#e9a23b", "Rescue access point"]]);

// ---------------------------------------------------------------- tabs
function setTab(t) {
  S.tab = t; S.click = null; hint("");
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  const three = t === "three" || (S.dim === "3d" && t !== "dispatch");
  $("#glwrap").classList.toggle("hidden", !three);
  $("#mapwrap").classList.toggle("hidden", three);
  render();
  dimUpdate();
}

// ---------------------------------------------------------------- 2D / 3D switch (over the map)
const MODES3D = { rgb: "True colour", height: "Elevation", class: "LAS class", hazard: "Danger to infrastructure", health: "Tree health", fire: "Fire hazard", firesim: "▶ Fire simulation", stormsim: "▶ Storm simulation" };
const LEG3D = {
  rgb: "", height: lgRamp("Elevation", [[0, [51, 77, 153]], [0.5, [204, 230, 102]], [1, [255, 140, 0]]], "low", "high"),
  class: lgRows([["#9e8561", "ground"], ["#26994d", "high vegetation"], ["#8ccc66", "low vegetation"], ["#ffe633", "power-line conductor"], ["#d94d4d", "building"], ["#4d8cd9", "water"]]),
  hazard: lgRamp("Design-storm danger to lines, houses, roads", [[0, [255, 217, 64]], [1, [255, 38, 13]]], "low", "high"),
  health: lgRows([["#338c4d", "healthy"], ["#f2b333", "stressed"], ["#e64033", "dead"]]),
  fire: lgRamp("Tree fire hazard", [[0, [237, 199, 77]], [1, [237, 41, 13]]], "high", "very high"),
  firesim: lgRamp("Simulated fire: when it arrives", [[0, [128, 10, 8]], [0.5, [191, 95, 20]], [1, [255, 214, 59]]], "burned first", "fire front"),
  stormsim: lgRows([["#ff2e1f", "tree falls in the simulated storm"], ["#e8b84a", "likely to fall (high P)"], ["#244d30", "stays standing"]]),
};
const HL3D = lgRows([["#ff9628", "NEW road found in the scan"], ["#ff40e6", "NEW power line found in the scan"]]);
function mode3dFor(t) {
  if (t === "hazards" && window.HZ && HZ.mode === "fire") return HZ.fire ? "firesim" : "fire";
  if (t === "hazards" && window.HZ && HZ.storm) return "stormsim";
  if (t === "fire") return S.fire ? "firesim" : "fire";
  if (t === "upload" && S.fire) return "firesim";
  if (["storm", "hazards", "plan", "routes", "upload"].includes(t)) return "hazard";
  if (t === "infra" || t === "quality") return "class";
  return "rgb";
}
async function ensureViewer() {
  if (!viewer) {
    viewer = new PointViewer($("#gl"));
    if (viewer.failed) return false;
  }
  if (viewer.siteVer !== S.siteVer) {
    busy(true);
    try {
      await viewer.load(tok("/api/site/points.bin")); viewer.siteVer = S.siteVer; viewer.simVer = -1;
      viewer.setHighlights(await api("/api/site/highlights"));
    } catch (e) { alert(e.message); }
    busy(false);
  }
  viewer.resize();
  return true;
}
async function dimUpdate() {
  const sw = $("#dimsw");
  if (!sw) return;
  const three = !$("#glwrap").classList.contains("hidden");
  sw.classList.toggle("hidden", S.tab === "dispatch");
  sw.querySelectorAll("[data-dim]").forEach((b) => b.classList.toggle("on", b.dataset.dim === (three ? "3d" : "2d")));
  const mm = $("#dimmodes");
  mm.classList.toggle("hidden", !three || S.tab === "three");
  if (!three || S.tab === "three") return;
  if (!(await ensureViewer())) { mm.innerHTML = `<span class="sub">WebGL is not available in this browser</span>`; return; }
  const m = S.mode3dUser && S.mode3dUser.tab === S.tab ? S.mode3dUser.m : mode3dFor(S.tab);
  if ((m === "firesim" || m === "stormsim") && viewer.simVer !== S.simVer) {
    try { await viewer.loadSim(tok("/api/site/points_sim.bin")); viewer.simVer = S.simVer; } catch (e) { /* keep last */ }
  }
  viewer.setMode(m);
  mm.innerHTML = Object.entries(MODES3D).map(([k, v]) => `<button class="chip ${k === m ? "on" : ""}" data-m3="${k}">${v}</button>`).join("")
    + (LEG3D[m] ? `<div class="m3leg">${LEG3D[m]}</div>` : "")
    + ((m === "firesim" && !(viewer.simHead && viewer.simHead.fire)) || (m === "stormsim" && !(viewer.simHead && viewer.simHead.storm)) ? `<div class="sub" style="margin-top:4px">No simulation yet: run one in Tree hazards.</div>` : "")
    + (viewer.hn ? `<div class="m3leg">${HL3D}</div>` : "")
    + (S.click ? `<div class="sub" style="margin-top:6px">Switch to 2D to click places on the map.</div>` : "");
  mm.querySelectorAll("[data-m3]").forEach((b) => b.addEventListener("click", () => { S.mode3dUser = { tab: S.tab, m: b.dataset.m3 }; dimUpdate(); }));
  legend("");
  hint("");
}
document.querySelectorAll("#dimsw [data-dim]").forEach((b) => b.addEventListener("click", () => {
  const want = b.dataset.dim;
  if (want === "2d" && S.tab === "three") { S.dim = "2d"; setTab("overview"); return; }
  S.dim = want;
  setTab(S.tab);
}));
function render() {
  $("#tip").classList.add("hidden");
  ({ overview: () => { tabOverview(); window.afterOverview && window.afterOverview(); }, fire: tabFire, storm: tabStorm, routes: tabRoutes,
     dispatch: tabDispatch, three: () => tabThree().then(() => window.afterThree && window.afterThree()), quality: tabQuality,
     infra: () => window.tabInfra(),
     upload: () => window.tabUpload(), hazards: () => window.tabHazards(), plan: () => window.tabPlan() }[S.tab])();
}
function panel(html) { $("#panel").innerHTML = html; $("#panel").scrollTop = 0; }
function kpi(v, l, cls = "") { return `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`; }
function bars(obj, color = "var(--amber)", d = 0) {
  return Object.entries(obj).map(([k, v]) => `<div class="bar"><div>${esc(k)}<div class="track"><div class="fill" style="width:${Math.min(100, v * 100)}%;background:${color}"></div></div></div><div class="num">${pct(v, d)}</div></div>`).join("");
}
function siteView() { if (!S._fitted || (S._lastView && S._lastView !== "site")) { map.fit(S.sum.bounds, 20); S._fitted = true; } S._lastView = "site"; }

// ---- Overview
function tabOverview() {
  siteView(); siteBase(); treeLayer("species");
  const s = S.sum, tr = s.trees, st = s.storm, f = s.fire, w = s.weather;
  const sp = tr.species, tot = tr.count;
  legend(`<h4>Trees (detected individually)</h4>${lgRows(SPECIES.map((n, i) => [SP_COL[i], `${n} · ${fmt(sp[n])}`]))}<h4 style="margin-top:8px">Infrastructure</h4>${INFRA_LG}`);
  panel(`
    <h2>${esc(s.name)}</h2>
    <p class="lead">Every tree in this ${fmt(s.area_ha, 0)} ha survey has been located from the point cloud and scored for fire, storm and infrastructure risk.</p>
    <div class="callout upcall"><b>Have your own 3D scan of a forest?</b> <a href="#" data-go="upload">Upload it →</a> (.las .laz .ply .obj .glb .xyz .e57): Horus maps the roads, power lines, houses and trees and ranks the dangers.</div>
    <div class="kpis">
      ${kpi(fmt(tr.count), "trees detected")}
      ${kpi(fmt(tr.health.dead), "dead / beetle-killed trees", tr.health.dead > 100 ? "warn" : "")}
      ${kpi(fmt(st.trees_reaching_line), "trees that can reach the 20 kV line", "warn")}
      ${kpi(pct(st.p_line_outage, 0), `P(line outage) at ${fmt(st.gust_ms)} m/s gust`, st.p_line_outage > 0.5 ? "bad" : "")}
      ${kpi(fmt(w.fwi, 1), `Fire Weather Index · ${esc(w.danger)}`, w.fwi > 20 ? "bad" : w.fwi > 10 ? "warn" : "good")}
      ${kpi(fmt(f.high_risk_ha, 1) + " ha", "high fire-risk area (index ≥ 60)", f.high_risk_ha > 5 ? "warn" : "")}
    </div>
    <h3>Decisions Horus supports</h3>
    <div class="card phase"><div class="tag">Before</div><div><div class="t">Where to cut, where to prepare</div><p>Ranked hazard-tree work order for the power line &amp; roads, and fuel-reduction zones near homes. <a href="#" data-go="storm">Storm &amp; trees →</a></p></div></div>
    <div class="card phase"><div class="tag">During</div><div><div class="t">How fast it spreads, what it reaches</div><p>Fire spread from any ignition point in today's weather, time until houses and the line are reached, safe evacuation routes. <a href="#" data-go="fire">Wildfire →</a></p></div></div>
    <div class="card phase"><div class="tag">After</div><div><div class="t">Which roads are passable, where to fly</div><p>Blocked-road probability per 25 m, routes per vehicle class, and drone dispatch to the worst-hit estates. <a href="#" data-go="routes">Routes →</a> <a href="#" data-go="dispatch">Dispatch →</a></p></div></div>
    <h3>Data flow</h3>
    <div class="card"><p>DJI M400 + Zenmuse L3 LiDAR (+ multispectral) → ground filter, DTM, canopy height model → individual trees, species, health → fused with weather (Open-Meteo, FWI), roads &amp; buildings (Digiroad / NLS) → fire, windthrow &amp; routing models → work orders, maps, dispatch plan.</p></div>
    <p class="note">Weather source: ${esc(w.source)}. ${esc(w.note || "")}</p>
  `);
  document.querySelectorAll("[data-go]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); setTab(a.dataset.go); }));
}

// ---- Fire
const FIRE_LAYERS = { fire_risk: "Fire-risk index", fuel: "Fuel types (FBP)", intensity: "Head-fire intensity", response: "Rescue response time" };
function fireLegend() {
  const L = S.fireLayer;
  let h = "";
  if (L === "fire_risk") h = lgRamp("Fire-risk index (0–100)", [[0, [255, 240, 140]], [0.35, [253, 190, 60]], [0.6, [240, 110, 30]], [0.8, [205, 30, 30]], [1, [110, 0, 40]]], "low", "very high");
  if (L === "fuel") h = `<h4>FBP fuel type</h4>` + lgRows([["rgb(30,90,50)", "C-2 boreal spruce"], ["rgb(120,140,40)", "C-3 mature pine"], ["rgb(60,160,70)", "C-4 immature conifer"], ["rgb(90,140,60)", "M-1 mixedwood"], ["rgb(170,210,90)", "D-1 deciduous"], ["rgb(230,215,140)", "O-1a open / clear-cut"], ["rgb(120,160,200)", "non-fuel"]]);
  if (L === "intensity") h = lgRamp("Head-fire intensity class", [[0, [255, 240, 140]], [0.35, [253, 190, 60]], [0.6, [240, 110, 30]], [0.8, [205, 30, 30]], [1, [110, 0, 40]]], "hand tools", "crown fire");
  if (L === "response") h = lgRamp("Engine drive + walk from road (min)", [[0, [40, 200, 120]], [0.35, [250, 220, 60]], [0.7, [230, 100, 40]], [1, [120, 30, 120]]], "0", "60+");
  if (S.fire) h += lgRamp("Fire arrival (30 min bands)", [[0, [255, 245, 120]], [0.25, [255, 170, 30]], [0.55, [230, 60, 20]], [1, [90, 10, 30]]], "0 h", `${S.fire.hours} h`);
  legend(h + `<h4 style="margin-top:8px">Infrastructure</h4>` + INFRA_LG);
}
function tabFire() {
  siteView(); siteBase();
  map.setLayer("firelayer", { type: "image", url: tok(`/api/site/layer/${S.fireLayer}.png?v=${S.ver}`), bounds: S.sum.bounds, z: 10, opacity: S.fire ? 0.25 : 0.8, pixelated: true });
  if (S.fire) map.setLayer("arrival", { type: "image", url: tok(S.fire.png_url + "&v=" + S.ver), bounds: S.sum.bounds, z: 20, opacity: 0.78, pixelated: true });
  if (S.fire) map.setLayer("ign", { type: "points", z: 70, data: [{ lon: S.fire.ignition[0], lat: S.fire.ignition[1], r: 7, color: "#ff2d2d", stroke: "#fff", label: "ignition" }] });
  drawRoutes();
  fireLegend();
  const s = S.sum, w = s.weather, f = s.fire;
  S.click = (ev) => igniteAt(ev.lon, ev.lat);
  hint("Click anywhere on the map to start a fire simulation");
  const codes = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"].map((k) => `<td class="num">${fmt(w[k], 1)}</td>`).join("");
  const fr = S.fire;
  panel(`
    <h2>Wildfire</h2>
    <p class="lead">Fuel types come from the LiDAR forest structure. Fire behaviour follows the Canadian FBP system, driven by today's fire weather.</p>
    <h3>Fire weather · ${esc(w.date)}</h3>
    <table><tr><th>FFMC</th><th>DMC</th><th>DC</th><th>ISI</th><th>BUI</th><th>FWI</th></tr><tr>${codes}</tr></table>
    <canvas class="spark" id="spark"></canvas>
    <p class="note">${fmt(w.temp_c, 0)} °C · RH ${fmt(w.rh, 0)} % · wind ${fmt(w.wind_kmh, 0)} km/h from ${fmt(w.wind_dir_deg, 0)}° · ${esc(w.source)}</p>
    <h3>Map layer</h3>
    <div class="toggles">${Object.entries(FIRE_LAYERS).map(([k, v]) => `<button class="chip ${k === S.fireLayer ? "on" : ""}" data-fl="${k}">${v}</button>`).join("")}</div>
    <div class="kpis" style="margin-top:12px">
      ${kpi(fmt(f.hfi_p90, 0), "kW/m head-fire intensity (p90)", f.hfi_p90 > 4000 ? "bad" : f.hfi_p90 > 2000 ? "warn" : "")}
      ${kpi(fmt(f.head_ros_m_min_p90, 1), "m/min spread rate (p90)")}
      ${kpi(fmt(f.high_risk_ha, 1) + " ha", "risk index ≥ 60")}
      ${kpi(fmt(f.response_p90_min, 0) + " min", "rescue response p90")}
    </div>
    <h3>Suppression difficulty (share of burnable area)</h3>
    ${bars(Object.fromEntries(Object.entries(f.intensity_share).filter(([k, v]) => k !== "no spread")), "var(--red)")}
    <h3>Fire-spread simulation</h3>
    <div class="formrow"><label>Duration</label><select id="fhours"><option>1</option><option>2</option><option selected>4</option><option>8</option></select> h</div>
    ${fr ? fireResultHtml(fr) : `<div class="callout">Click an ignition point on the map, for example a forest road near the village. The model shows where the fire is after each 30 minutes and when it reaches each asset.</div>`}
  `);
  document.querySelectorAll("[data-fl]").forEach((b) => b.addEventListener("click", () => { S.fireLayer = b.dataset.fl; tabFire(); }));
  $("#fhours").value = String(S.fhours || 4);
  $("#fhours").addEventListener("change", (e) => (S.fhours = +e.target.value));
  spark($("#spark"), w.fwi_history || []);
  const ev = $("#evac");
  if (ev) ev.addEventListener("click", evacuate);
}
function fireResultHtml(fr) {
  const rows = fr.impacts.filter((i) => i.minutes != null).slice(0, 12).map((i) => `<tr><td>${esc(i.asset)}</td><td>${esc(i.kind.replace("_", " "))}</td><td class="num">${i.minutes < 60 ? fmt(i.minutes, 0) + " min" : fmt(i.minutes / 60, 1) + " h"}</td></tr>`).join("");
  const care = fr.impacts.find((i) => i.kind === "care_home");
  return `
    <div class="kpis">${Object.entries(fr.area_ha).map(([k, v]) => kpi(fmt(v, 1) + " ha", `burned after ${k}`)).join("")}</div>
    <h3>Time until fire reaches</h3>
    <table><tr><th>Asset</th><th>Type</th><th class="num">Arrival</th></tr>${rows || `<tr><td colspan=3>No assets reached within ${fr.hours} h</td></tr>`}</table>
    ${care && care.minutes != null ? `<div class="callout red"><b>Care home reached in ${fmt(care.minutes, 0)} min.</b> Evacuation must start well before then.</div>` : ""}
    <button class="btn" id="evac">Evacuation route: care home → access point</button>
    <p class="note">Assumes the vehicle leaves now. The route avoids every cell the fire reaches within 15 min, plus a 10 m buffer.</p>`;
}
async function igniteAt(lon, lat) {
  busy(true);
  try { S.fire = await api("/api/site/fire", { lon, lat, hours: S.fhours || 4 }); S.ver++; S.route = []; simChanged(); tabFire(); }
  catch (e) { alert(e.message); }
  busy(false);
}
async function evacuate() {
  const care = S.vec.buildings.find((b) => b.kind === "care_home");
  busy(true);
  try {
    const r = await api("/api/site/route", { vehicle: "ambulance", start: care.pos, end: "depot", avoid_fire_min: 15, storm: false });
    S.route = r.ok ? [{ ...r, color: "#ffffff" }] : [];
    if (!r.ok) alert("No safe route: " + r.reason + ". Shelter in place or evacuate on foot / by air.");
    tabFire();
    if (r.ok) {
      $("#panel").insertAdjacentHTML("beforeend", `<div class="callout teal">Ambulance route: ${fmt(r.length_m)} m, about ${fmt(r.minutes, 1)} min to the access point, avoiding the fire front.${r.leave_within_min != null ? `<br><b>The last vehicle must leave within ${fmt(r.leave_within_min, 0)} min</b> (fire reaches the care home in ${fmt(r.fire_reaches_start_min, 0)} min).` : ""}</div>`);
      $("#panel").scrollTop = 1e6;
    }
  } catch (e) { alert(e.message); }
  busy(false);
}
function spark(c, hist) {
  if (!c || !hist.length) return;
  const dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const x = c.getContext("2d"); x.scale(dpr, dpr);
  const mx = Math.max(30, ...hist.map((d) => d.fwi));
  [[5, "#4cc3a6"], [10, "#9ed06a"], [20, "#e9a23b"], [30, "#e5534b"]].forEach(([v, col]) => { x.strokeStyle = col + "55"; x.beginPath(); const y = h - 4 - (v / mx) * (h - 8); x.moveTo(0, y); x.lineTo(w, y); x.stroke(); });
  x.strokeStyle = "#e9a23b"; x.lineWidth = 2; x.beginPath();
  hist.forEach((d, i) => { const px = (i / (hist.length - 1 || 1)) * (w - 4) + 2, py = h - 4 - (d.fwi / mx) * (h - 8); i ? x.lineTo(px, py) : x.moveTo(px, py); });
  x.stroke(); x.fillStyle = "#8ea39a"; x.font = "10px system-ui"; x.fillText("FWI, last 21 days", 4, 11);
}

// ---- Storm
function stormLayers() {
  const v = S.vec;
  map.setLayer("pieces", { type: "lines", z: 52, data: v.road_pieces.filter((p) => p.p_block > 0.02).map((p) => ({ path: p.path, color: riskCol(Math.min(1, p.p_block / 0.8)), width: 6 })) });
  map.setLayer("spans", { type: "lines", z: 56, data: v.power_spans.map((p) => ({ path: p.path, color: p.p_hit > 0.3 ? "#ff3d6e" : p.p_hit > 0.05 ? "#ffb347" : "#59d0ff", width: p.p_hit > 0.05 ? 4 : 2, casing: "rgba(0,0,0,.6)" })) });
}
function tabStorm() {
  siteView(); siteBase({ hillOpacity: 0.75 });
  treeLayer(S.stormMode || "risk_design");
  stormLayers();
  const s = S.sum, st = s.storm, w = s.weather;
  legend(lgRamp("Tree risk to infrastructure", RISK, "low", "highest") +
    (S.treeMode === "risk_now" ? `<div class="row" style="margin-top:6px"><span class="sw" style="background:rgba(150,170,255,.85)"></span>likely to fall (>25 %), no asset in reach</div>` : "") +
    lgRamp("Road piece: P(blocked)", RISK, "0", "≥ 80 %") +
    lgRows([["#ff3d6e", "line span P(hit) > 30 %"], ["#ffb347", "line span P(hit) 5–30 %"]]));
  const p50 = st.pareto_design["50%"], p80 = st.pareto_design["80%"];
  panel(`
    <h2>Storm &amp; hazard trees</h2>
    <p class="lead">Each tree gets a probability of falling, based on gust, slenderness, species, stand edge, peat soil and health. That is combined with the chance it hits the line, a road or a house.</p>
    <h3>Storm conditions</h3>
    <div class="formrow"><label>Max gust</label><input type="range" id="gust" min="8" max="40" step="1" value="${Math.round(w.gust_ms)}"><output id="gustv">${Math.round(w.gust_ms)} m/s</output></div>
    <div class="formrow"><label>Wind from</label><input type="range" id="wdir" min="0" max="350" step="10" value="${Math.round(w.wind_dir_deg / 10) * 10}"><output id="wdirv">${Math.round(w.wind_dir_deg)}°</output></div>
    <button class="btn" id="applyStorm">Run storm model</button> <button class="btn ghost" id="forecast">Use forecast</button>
    <div class="kpis" style="margin-top:12px">
      ${kpi(fmt(st.expected_fallen_trees, 0), "expected fallen trees", st.expected_fallen_trees > 200 ? "bad" : "")}
      ${kpi(pct(st.p_line_outage, 0), "P(≥1 tree on the 20 kV line)", st.p_line_outage > 0.5 ? "bad" : st.p_line_outage > 0.1 ? "warn" : "good")}
      ${kpi(fmt(st.expected_road_blockages, 1), "expected road blockages")}
      ${kpi(fmt(st.road_pieces_over_50), "road pieces ≥ 50 % blocked", st.road_pieces_over_50 > 0 ? "warn" : "good")}
    </div>
    <h3>Preventive work order · design storm ${st.design_gust_ms} m/s</h3>
    ${p50 ? `<div class="callout teal">Felling the top <b>${fmt(p50.trees)}</b> trees (${pct(p50.share_of_reaching)} of the ${fmt(st.trees_reaching_any)} trees within reach of an asset) removes <b>50 %</b> of the expected strike risk. The top ${fmt(p80.trees)} remove 80 %.</div>` : ""}
    <div class="toggles" style="margin-bottom:8px">
      <button class="chip ${S.treeMode === "risk_design" ? "on" : ""}" data-tm="risk_design">Design-storm risk</button>
      <button class="chip ${S.treeMode === "risk_now" ? "on" : ""}" data-tm="risk_now">Current conditions</button>
      <button class="chip ${S.treeMode === "health" ? "on" : ""}" data-tm="health">Health (NDVI)</button>
    </div>
    <table id="wo"><tr><th>#</th><th>Tree</th><th>Threatens</th><th class="num">Dist</th><th class="num">P(hit)</th><th class="num">€ at risk</th></tr><tr><td colspan=6>loading…</td></tr></table>
    <p style="margin-top:10px"><a class="btn ghost" href="${tok("/api/site/workorder.csv")}">Download full work order (CSV)</a></p>
    <p class="note">The CSV lists every tree within reach, with coordinates, species, health, height, DBH, slenderness, failure and hit probabilities and a suggested action. It is ready for a forestry contractor's GPS.</p>
  `);
  const g = $("#gust"), d = $("#wdir");
  g.oninput = () => ($("#gustv").textContent = g.value + " m/s");
  d.oninput = () => ($("#wdirv").textContent = d.value + "°");
  $("#applyStorm").onclick = () => refreshScenario({ scenario: $("#scenario").value === "live" ? "live" : "storm", gust: +g.value, wind_dir: +d.value }).then(() => { $("#scenario").value = S.sum.weather.scenario; });
  $("#forecast").onclick = () => refreshScenario({ scenario: "live" });
  document.querySelectorAll("[data-tm]").forEach((b) => b.addEventListener("click", () => { S.stormMode = b.dataset.tm; tabStorm(); }));
  api("/api/site/workorder?top=25").then((rows) => {
    S.workorder = rows;
    $("#wo").innerHTML = `<tr><th>#</th><th>Tree</th><th>Threatens</th><th class="num">Dist</th><th class="num">P(hit)</th><th class="num" title="expected damage this tree causes in the design storm (what felling it avoids)">€ at risk</th></tr>` + rows.map((r) =>
      `<tr class="click" data-lon="${r.lon}" data-lat="${r.lat}"><td>${r.rank}</td><td>${r.species} ${r.height_m} m${r.health !== "healthy" ? ` <span class="pill ${r.health === "dead" ? "p-high" : "p-med"}">${r.health}</span>` : ""}</td><td>${r.threatens === "bld" ? "building" : r.threatens}</td><td class="num">${r.distance_m} m</td><td class="num">${pct(Math.max(r.p_hit_power, r.p_hit_building, r.p_block_road), 0)}</td><td class="num">${r.expected_damage_eur != null ? "€" + fmt(r.expected_damage_eur) : "–"}</td></tr>`).join("");
    document.querySelectorAll("#wo tr.click").forEach((tr) => tr.addEventListener("click", () => { map.center = [+tr.dataset.lon, +tr.dataset.lat]; map.zoom = 18.3; map.setLayer("sel", { type: "circle", z: 80, lon: +tr.dataset.lon, lat: +tr.dataset.lat, radius_m: 4, color: "#fff", fill: "rgba(255,255,255,.15)", width: 2 }); }));
  });
}

// ---- Routes
function drawRoutes() {
  map.setLayer("routes", { type: "lines", z: 65, data: S.route.map((r) => ({ path: r.path, color: r.color, width: 4, casing: "rgba(0,0,0,.7)" })) });
}
function tabRoutes() {
  siteView(); siteBase();
  stormLayers();
  if (S.fire) map.setLayer("arrival", { type: "image", url: tok(S.fire.png_url + "&v=" + S.ver), bounds: S.sum.bounds, z: 20, opacity: 0.7, pixelated: true });
  drawRoutes();
  const R = (S.rt ||= { vehicle: "fire_engine", start: "depot", end: null, thr: 0.5, avoid: false });
  const blds = S.vec.buildings;
  if (R.end && R.end !== "depot") map.setLayer("dest", { type: "points", z: 75, data: [{ lon: R.end[0], lat: R.end[1], r: 7, color: "#e5534b", stroke: "#fff", label: "destination" }] });
  legend(lgRamp("Road piece: P(blocked by fallen tree)", RISK, "0", "≥ 80 %") + `<h4 style="margin-top:8px">Routes</h4>` + lgRows(Object.values(VEH).map(([l, c]) => [c, l])));
  S.click = (ev) => { R.end = [ev.lon, ev.lat]; tabRoutes(); };
  hint("Click the map to set a destination, or pick a building");
  panel(`
    <h2>Emergency routes</h2>
    <p class="lead">Travel times per vehicle class. Speeds come from road class, LiDAR slope and vegetation density. Roads likely blocked by fallen trees cost clearing time, or are impassable for vehicles without a saw.</p>
    <div class="formrow"><label>Vehicle</label><select id="veh">${Object.entries(VEH).map(([k, [l]]) => `<option value="${k}" ${k === R.vehicle ? "selected" : ""}>${l}</option>`).join("")}</select></div>
    <div class="formrow"><label>From</label><select id="from"><option value="depot">Rescue access point</option>${blds.map((b, i) => `<option value="b${i}">${esc(b.name)}</option>`).join("")}</select></div>
    <div class="formrow"><label>To</label><select id="to"><option value="">(click on map)</option><option value="depot">Rescue access point</option>${blds.map((b, i) => `<option value="b${i}">${esc(b.name)}</option>`).join("")}</select></div>
    <div class="formrow"><label>Treat road as blocked if P ≥</label><input type="range" id="thr" min="0.1" max="0.9" step="0.05" value="${R.thr}"><output id="thrv">${pct(R.thr)}</output></div>
    ${S.fire ? `<div class="formrow"><label>Avoid fire front</label><input type="checkbox" id="avoid" ${R.avoid ? "checked" : ""}> cells burning within 60 min</div>` : ""}
    <button class="btn" id="go">Compute route</button><button class="btn ghost" id="cmp">Compare all vehicles</button>
    <div id="rres"></div>
    <p class="note">Storm conditions come from the "Storm &amp; trees" tab (current gust ${fmt(S.sum.weather.gust_ms)} m/s). Run a fire simulation first to route around a fire.</p>
  `);
  const sel = (v) => (v === "depot" ? "depot" : v && v[0] === "b" ? blds[+v.slice(1)].pos : null);
  $("#from").value = typeof R.start === "string" ? R.start : R.startKey || "depot";
  if (R.endKey) $("#to").value = R.endKey;
  $("#veh").onchange = (e) => (R.vehicle = e.target.value);
  $("#from").onchange = (e) => { R.startKey = e.target.value; R.start = sel(e.target.value); };
  $("#to").onchange = (e) => { R.endKey = e.target.value; R.end = sel(e.target.value); tabRoutes(); };
  $("#thr").oninput = (e) => { R.thr = +e.target.value; $("#thrv").textContent = pct(R.thr); };
  if ($("#avoid")) $("#avoid").onchange = (e) => (R.avoid = e.target.checked);
  const one = async (veh) => api("/api/site/route", { vehicle: veh, start: R.start || "depot", end: R.end, block_threshold: R.thr, avoid_fire_min: R.avoid && S.fire ? 60 : null });
  $("#go").onclick = async () => {
    if (!R.end) return alert("Pick a destination first");
    busy(true);
    try {
      const r = await one(R.vehicle);
      S.route = r.ok ? [{ ...r, color: VEH[R.vehicle][1] }] : [];
      drawRoutes();
      $("#rres").innerHTML = routeHtml([[R.vehicle, r]]);
    } catch (e) { alert(e.message); }
    busy(false);
  };
  $("#cmp").onclick = async () => {
    if (!R.end) return alert("Pick a destination first");
    busy(true);
    const res = [];
    for (const k of Object.keys(VEH)) { try { res.push([k, await one(k)]); } catch (e) { res.push([k, { ok: false, reason: e.message }]); } }
    S.route = res.filter(([, r]) => r.ok).map(([k, r]) => ({ ...r, color: VEH[k][1] }));
    drawRoutes();
    $("#rres").innerHTML = routeHtml(res);
    busy(false);
  };
}
function routeHtml(res) {
  return `<h3>Result</h3><table><tr><th>Vehicle</th><th class="num">ETA</th><th class="num">Dist</th><th class="num">Clear</th><th class="num">Road</th></tr>` +
    res.map(([k, r]) => r.ok
      ? `<tr><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${VEH[k][1]}"></span> ${VEH[k][0]}${r.walk_m > 5 && k !== "foot" ? `<br><span style="color:var(--muted);font-size:11px">+ ${fmt(r.walk_m)} m on foot</span>` : ""}</td><td class="num">${fmt(r.minutes, 1)} min</td><td class="num">${fmt(r.length_m / 1000, 2)} km</td><td class="num">${r.clearings}</td><td class="num">${pct(r.road_share)}</td></tr>`
      : `<tr><td>${VEH[k][0]}</td><td colspan=4 style="color:var(--red)">${esc(r.reason)}</td></tr>`).join("") +
    `</table><p class="note">"Clear" counts the fallen-tree blockages the crew must cut through, already included in the ETA.</p>`;
}

// ---- Dispatch (Finland)
async function tabDispatch() {
  map.clearLayers("");
  legend("");
  S._lastView = "fleet";
  if (!S.fleet) { busy(true); try { S.fleet = await api("/api/fleet"); } catch (e) { panel(`<h2>Drone dispatch</h2><p>${esc(e.message)}</p>`); busy(false); return; } busy(false); }
  if (S.tab !== "dispatch") return;
  const F = S.fleet;
  if (!S._fleetFit) { map.fit([[59.7, 20.5], [70.1, 31.6]], 10); S._fleetFit = true; }
  map.setLayer("fin", { type: "polygon", z: 0, rings: [window.FINLAND], fill: "rgba(76,195,166,.05)", color: "#3c5a50", width: 1.2 });
  const pc = { high: "#e5534b", medium: "#e9a23b", low: "#4cc3a6" };
  map.setLayer("tasks", { type: "points", z: 20, hover: true, data: F.tasks.map((t) => ({ lon: t.lon, lat: t.lat, r: 2.6, color: pc[t.priority] + "cc" })),
    tip: (i) => { const t = F.tasks[i]; return `<b>${esc(t.name)}</b> · ${esc(t.municipality)}<br>${t.area} ha · ${t.priority}${t.ms ? " · multispectral" : ""}<br>due ${t.due} · baseline plan: ${t.on_time ? "on time" : t.done ? "late" : "not done"}`; } });
  map.setLayer("ops", { type: "points", z: 30, hover: true, data: F.operators.map((o) => ({ lon: o.lon, lat: o.lat, r: 5, color: o.multispectral ? "#a685e2" : "#f0f0f0", stroke: "#0b1210" })),
    tip: (i) => { const o = F.operators[i]; return `<b>${esc(o.name)}</b><br>home ${esc(o.home)} · ${o.multispectral ? "LiDAR + multispectral" : "LiDAR"}<br>${o.days_available} available days`; } });
  const D = (S.storm_in ||= { date: "2026-10-12", lat: 62.9, lon: 27.7, radius_km: 110, gust: 32 });
  map.setLayer("stormc", { type: "circle", z: 15, lon: D.lon, lat: D.lat, radius_m: D.radius_km * 1000, color: "#59d0ff", fill: "rgba(89,208,255,.07)", dash: [6, 5] });
  const R = S.stormRes;
  if (R) {
    const mx = Math.max(...R.emergency_tasks.map((t) => t.exposure), 1);
    map.setLayer("em", { type: "points", z: 40, hover: true, data: R.emergency_tasks.map((t) => ({ lon: t.lon, lat: t.lat, r: t.kind === "powerline_corridor" ? 6 : 4.5, color: riskCol(t.exposure / mx), stroke: t.kind === "powerline_corridor" ? "#59d0ff" : "#111" })),
      tip: (i) => { const t = R.emergency_tasks[i]; return `<b>${esc(t.name)}</b><br>gust ${t.gust_ms} m/s · exposure ${t.exposure}<br>flown: ${t.done || "–"}`; } });
    const day1 = R.assignments.filter((a) => a.data_ready_h_after_storm <= 24);
    map.setLayer("asg", { type: "lines", z: 35, data: day1.map((a) => ({ path: [[a.from_lon, a.from_lat], [a.lon, a.lat]], color: "rgba(233,162,59,.75)", width: 1.6, dash: [4, 3] })) });
  }
  legend(`<h4>Forey flight tasks (${F.tasks.length})</h4>` + lgRows([["#e5534b", "high priority"], ["#e9a23b", "medium"], ["#4cc3a6", "low"]]) +
    `<h4 style="margin-top:6px">Operators (${F.operators.length})</h4>` + lgRows([["#f0f0f0", "LiDAR drone"], ["#a685e2", "LiDAR + multispectral"]]) +
    (R ? `<h4 style="margin-top:6px">Storm response</h4>` + lgRamp("Emergency task exposure", RISK, "low", "high") + lgRows([["rgba(233,162,59,.9)", "first-24 h flights"]]) : ""));
  S.click = (ev) => { D.lat = +ev.lat.toFixed(3); D.lon = +ev.lon.toFixed(3); tabDispatch(); };
  hint("Click the map to move the storm centre");
  const b = F.baseline;
  panel(`
    <h2>Drone dispatch</h2>
    <p class="lead">Forey's fleet: ${F.operators.length} operators and ${F.tasks.length} flight tasks, ${esc(F.period.start)} to ${esc(F.period.end)}. In a storm, Horus turns the fleet into a rapid-assessment service.</p>
    <h3>Regular plan (Horus scheduler, baseline)</h3>
    <div class="kpis">
      ${kpi(pct(b.on_time_rate, 1), "tasks flown on time", "good")}
      ${kpi(pct(b.high_on_time_rate, 1), "high-priority on time")}
      ${kpi(fmt(b.hectares / 1000, 1) + "k ha", "hectares surveyed")}
      ${kpi(fmt(b.travel_hours, 0) + " h", "driving (multi-day tours)")}
    </div>
    <h3>Storm event</h3>
    <div class="formrow"><label>Date</label><input type="date" id="sdate" min="${F.period.start}" max="${F.period.end}" value="${D.date}"></div>
    <div class="formrow"><label>Centre</label><span class="num">${D.lat.toFixed(2)}° N, ${D.lon.toFixed(2)}° E</span></div>
    <div class="formrow"><label>Radius</label><input type="range" id="srad" min="30" max="250" step="10" value="${D.radius_km}"><output id="sradv">${D.radius_km} km</output></div>
    <div class="formrow"><label>Peak gust</label><input type="range" id="sgust" min="20" max="40" step="1" value="${D.gust}"><output id="sgustv">${D.gust} m/s</output></div>
    <button class="btn" id="runStorm">Plan storm response</button>
    <div id="sres">${R ? stormHtml(R) : `<p class="note">Horus creates re-survey tasks for estates Forey has already scanned inside the footprint, so a pre-storm point cloud exists for change detection. It adds 20 kV corridor inspections for the grid operator, ranks everything by storm exposure, and re-plans the fleet.</p>`}</div>
  `);
  $("#srad").oninput = (e) => { D.radius_km = +e.target.value; $("#sradv").textContent = D.radius_km + " km"; map.setLayer("stormc", { ...map.layer("stormc"), radius_m: D.radius_km * 1000 }); };
  $("#sgust").oninput = (e) => { D.gust = +e.target.value; $("#sgustv").textContent = D.gust + " m/s"; };
  $("#sdate").onchange = (e) => (D.date = e.target.value);
  $("#runStorm").onclick = async () => {
    busy(true);
    try { S.stormRes = await api("/api/fleet/storm", D); tabDispatch(); } catch (e) { alert(e.message); }
    busy(false);
  };
}
function stormHtml(R) {
  const k = R.kpi, bi = R.backlog_impact;
  const mx = Math.max(k.exposure_weighted_hours || 0, k.business_as_usual_exposure_weighted_hours || 0, 1);
  const gain = 1 - k.exposure_weighted_hours / k.business_as_usual_exposure_weighted_hours;
  const rows = R.assignments.slice(0, 14).map((a) => `<tr><td>${esc(a.operator)}</td><td>${esc(a.task.replace("Storm re-survey of ", "Re-survey "))}</td><td class="num">${a.date.slice(5)} ${a.start}</td><td class="num">${fmt(a.data_ready_h_after_storm, 1)} h</td></tr>`).join("");
  return `
    <div class="kpis" style="margin-top:12px">
      ${kpi(fmt(k.emergency_tasks), "emergency survey tasks")}
      ${kpi(`${k.data_within_24h}/${k.emergency_tasks}`, "processed data within 24 h", "good")}
      ${kpi(fmt(k.median_hours_to_data, 1) + " h", "median storm-to-data time")}
      ${kpi(fmt(bi.regular_tasks_delayed), "regular tasks pushed later", "warn")}
    </div>
    <h3>Exposure-weighted hours to data (lower is better)</h3>
    <div class="cmp"><span>Horus ranking</span><div class="track"><div class="fill" style="width:${(k.exposure_weighted_hours / mx) * 100}%;background:var(--teal)"></div></div><span class="num">${fmt(k.exposure_weighted_hours, 1)} h</span></div>
    <div class="cmp"><span>Business as usual</span><div class="track"><div class="fill" style="width:${(k.business_as_usual_exposure_weighted_hours / mx) * 100}%;background:var(--dim)"></div></div><span class="num">${fmt(k.business_as_usual_exposure_weighted_hours, 1)} h</span></div>
    ${gain > 0 ? `<div class="callout teal">The worst-hit areas are seen <b>${pct(gain)} sooner</b> than if storm requests were queued as ordinary high-priority tasks. The cost: ${fmt(bi.regular_tasks_delayed)} regular inventory flights move to a later day.</div>` : ""}
    <h3>Dispatch list</h3>
    <table><tr><th>Operator</th><th>Task</th><th class="num">Start</th><th class="num">Data</th></tr>${rows}</table>
    <p class="note">"Data" is the time from storm to processed point cloud: travel, flight, plus about 2 h of cloud processing. Dashed lines on the map are the first-24 h flights.</p>`;
}

// ---- 3D
async function tabThree() {
  legend("");
  const modes = MODES3D;
  panel(`
    <h2>3D point cloud</h2>
    <p class="lead">A decimated preview of the survey: 600k of ${fmt(S.sum.lidar.points)} points, ${fmt(S.sum.lidar.density_pts_m2, 1)} pts/m². Colour it by what Horus computed.</p>
    <div class="toggles">${Object.entries(modes).map(([k, v]) => `<button class="chip ${k === (viewer?.mode || "rgb") ? "on" : ""}" data-m="${k}">${v}</button>`).join("")}</div>
    <div id="m3legend" style="margin-top:12px"></div>
    <h3>What to look for</h3>
    <div class="card"><p>The 20 kV conductors are yellow in LAS-class mode. Horus vectorised them straight from the point cloud. Switch to "Hazard to assets" to see which crowns lean over the line and roads, and to "Tree health" to see the beetle-killed spruce stand next to the village.</p></div>
  `);
  const lg = { rgb: "", height: "", class: lgRows([["#9e8561", "ground"], ["#26994d", "high vegetation"], ["#8ccc66", "low vegetation"], ["#ffe633", "power-line conductor (14)"], ["#d94d4d", "building"], ["#4d8cd9", "water"]]), hazard: LEG3D.hazard, health: LEG3D.health, fire: LEG3D.fire };
  const setM = (m) => { viewer.setMode(m); $("#m3legend").innerHTML = lg[m]; document.querySelectorAll("[data-m]").forEach((b) => b.classList.toggle("on", b.dataset.m === m)); };
  document.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => setM(b.dataset.m)));
  if (!(await ensureViewer())) { panel("<h2>3D</h2><p>WebGL is not available in this browser.</p>"); return; }
  setM(viewer.mode);
}

// ---- Quality
function tabQuality() {
  siteView(); siteBase();
  map.setLayer("chm", { type: "image", url: tok(`/api/site/layer/chm.png`), bounds: S.sum.bounds, z: 5, opacity: 0.85 });
  legend(lgRamp("Canopy height model", [[0, [235, 240, 220]], [0.07, [200, 225, 170]], [0.4, [80, 160, 70]], [0.75, [25, 95, 40]], [1, [10, 50, 25]]], "0 m", "30 m"));
  const s = S.sum, v = s.validation, L = s.lidar, c = s.classification;
  const conf = v ? `<table><tr><th>true ↓ / pred →</th>${SPECIES.map((n) => `<th class="num">${n}</th>`).join("")}</tr>${v.species_confusion.map((r, i) => `<tr><td>${SPECIES[i]}</td>${r.map((x, j) => `<td class="num" style="${i === j ? "color:var(--teal);font-weight:600" : ""}">${fmt(x)}</td>`).join("")}</tr>`).join("")}</table>` : "";
  panel(`
    <h2>Data quality &amp; validation</h2>
    <p class="lead">Every number Horus shows comes from a measurable pipeline. Here is how good each step is.</p>
    <h3>Point cloud</h3>
    <table>
      <tr><td>Points</td><td class="num">${fmt(L.points)}</td></tr>
      <tr><td>Density</td><td class="num">${fmt(L.density_pts_m2, 1)} pts/m²</td></tr>
      <tr><td>Ground returns</td><td class="num">${fmt(L.ground_points)}</td></tr>
      <tr><td>Cells with ground hit (0.5 m)</td><td class="num">${pct(L.ground_cell_coverage)}</td></tr>
      <tr><td>Ground method</td><td class="num" style="font-family:var(--sans)">${esc(L.ground_method)}</td></tr>
    </table>
    ${v ? `
    <h3>Individual tree detection · vs reference trees ≥ ${v.min_reference_height_m} m</h3>
    <div class="kpis">
      ${kpi(pct(v.recall), "detection rate (recall)")}${kpi(pct(v.precision), "precision")}
      ${kpi(fmt(v.height_rmse_m, 2) + " m", "height RMSE")}${kpi(fmt(v.height_bias_m, 2) + " m", "height bias")}
      ${kpi(pct(v.species_accuracy, 1), "species accuracy", "good")}${kpi(pct(v.health_accuracy, 1), "health accuracy", "good")}
      ${kpi(pct(v.dead_tree_recall), "dead trees found")}${kpi(fmt(v.dtm_mae_m * 100, 1) + " cm", "DTM mean abs. error")}
    </div>
    <h3>Species confusion matrix</h3>${conf}
    <p class="note">Reference = dominant / co-dominant trees in the synthetic ground truth. Trees under taller crowns are invisible to an airborne sensor, which is typical for area-based vs. ITD inventories. Species model: ${esc(c.species_source)}, with calibration trees excluded from the accuracy figures.</p>` : `<div class="callout">No reference data for this survey. Add field-plot trees to calibrate species and report accuracy.</div>`}
    <h3>Health signal</h3><p class="note">${esc(c.health_source)}. ${c.multispectral ? "" : "Order the multispectral add-on for reliable dead-tree detection."}</p>
    <h3>Data lineage</h3>
    <table>
      <tr><th>Input</th><th>Source</th></tr>
      <tr><td>Point cloud, trees, terrain, power line</td><td>Forey drone survey (DJI M400 + L3)</td></tr>
      <tr><td>Tree health</td><td>Multispectral camera (NDVI)</td></tr>
      <tr><td>Weather, FWI, gusts</td><td>${esc(s.weather.source)}</td></tr>
      <tr><td>Roads, buildings</td><td>${esc(s.context_source)}</td></tr>
      <tr><td>Fire behaviour</td><td>Canadian FBP system (ST-X-3)</td></tr>
      <tr><td>Windthrow</td><td>Logistic model, to be calibrated on DSO outage logs</td></tr>
    </table>
    <h3>Known limitations</h3>
    <div class="card"><p>Fire spread uses constant weather and no spotting. The windthrow coefficients are literature-informed priors, not yet fitted to Finnish outage data, which is the pilot's job. Routing ignores road load limits and bridges. Probabilities are relative-risk tools for prioritisation, not guarantees.</p></div>
    <h3>Processing time</h3>
    <p class="note">${Object.entries(s.timings).map(([k, t]) => `${k}: ${t} s`).join(" · ")} (cumulative, ${fmt(s.area_ha, 0)} ha)</p>
  `);
}

// ---------------------------------------------------------------- boot
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
$("#scenario").addEventListener("change", (e) => refreshScenario({ scenario: e.target.value }));
map.tileUrlHook = (u) => (u.startsWith("/api/") ? tok(u) : u);
$("#basemap").addEventListener("change", (e) => { map.basemap = e.target.value; if (S.sum && S.tab !== "dispatch" && S.tab !== "three") render(); else map.redraw(); });
(async function boot() {
  // basemap: use OSM when the tile server is reachable, otherwise stay offline
  map.basemap = "none"; $("#basemap").value = "none";
  const probe = new Image();
  probe.onload = () => { map.basemap = "satellite"; $("#basemap").value = "satellite"; if (S.sum) render(); };
  probe.src = BASEMAPS.satellite.url.replace("{z}", 3).replace("{x}", 4).replace("{y}", 2);
  busy(true);
  try { await loadSite(); } catch (e) { panel(`<h2>Could not load</h2><p>${esc(e.message)}</p>`); busy(false); return; }
  busy(false);
  setTab("overview");
})();
