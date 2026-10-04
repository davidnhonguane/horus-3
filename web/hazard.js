// Horus - Tree hazards (which trees fall in a storm / burn in a fire) and Emergency plan
// (safest route + which vehicle fits, per mission).
"use strict";

const HZ = (window.HZ = { mode: "storm", gust: null, dir: null, runs: 300, storm: null, fireTrees: null, fireVer: -1, fire: null, hours: 2,
  plan: null, mission: "any", end: null, endKey: "" });
const FALL_COL = ["rgba(150,175,255,.85)", "#ff9a3c", "#ff3df0", "#ff3b30"];   // falls in forest / on road / on line / on building
const FLAM = [[0, [70, 120, 85]], [0.35, [233, 200, 80]], [0.6, [239, 123, 62]], [0.8, [229, 60, 50]], [1, [150, 0, 40]]];
const flamCol = (t) => lerpCol(FLAM, Math.max(0, Math.min(1, t)));
const lvlBadge = (lvl, v) => `<span class="lvl ${lvl}">${v != null ? v + " · " : ""}${lvl}</span>`;

function hzTrees(colorFn, rFn) {
  const T = S.trees;
  for (let i = 0; i < T.length; i++) { const c = colorFn(T[i], i); T[i].color = c[0]; T[i].minPx = c[1]; T[i].r = Math.max(0.9, T[i].h * 0.075) * (rFn ? rFn(T[i], i) : 1); }
  map.setLayer("trees", { type: "points", z: 30, metres: true, minRadiusPx: 0.9, n: T.length, get: (i) => T[i], hover: true, hoverFilter: () => map.zoom >= 16.2,
    tip: (i) => { const t = T[i]; const pf = HZ.storm ? HZ.storm.p_fall[i] : t.pf;
      return `<b>${SPECIES[t.sp]}</b> · ${fmt(t.h, 1)} m · ${HEALTH[t.hl]}<br>P(falls) ${HZ.storm ? `at ${fmt(HZ.storm.gust_ms)} m/s` : "now"}: ${pct(pf, 1)}<br>fire hazard ${fmt(t.fh)}/100 · P(torch if fire reaches it) ${pct(t.pt)}<br>threatens: ${TARGET[t.tg]}`; } });
}

function zoomTo(lon, lat) {
  map.center = [lon, lat]; map.zoom = 18.4;
  map.setLayer("sel", { type: "circle", z: 80, lon, lat, radius_m: 4, color: "#fff", fill: "rgba(255,255,255,.15)", width: 2 });
}
function bindZoom() {
  document.querySelectorAll("tr.click[data-lon]").forEach((tr) => tr.addEventListener("click", () => zoomTo(+tr.dataset.lon, +tr.dataset.lat)));
}

function curveSvg(curve, gust) {
  const W = 300, H = 120, p = 26, gx = (g) => p + ((g - 10) / 30) * (W - p - 8), mx = Math.max(...curve.map((c) => c.fallen), 1);
  const gy = (v) => H - 18 - (v / mx) * (H - 30), gy2 = (v) => H - 18 - v * (H - 30);
  const l1 = curve.map((c) => `${gx(c.gust)},${gy(c.fallen)}`).join(" "), l2 = curve.map((c) => `${gx(c.gust)},${gy2(c.p_line_outage)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" class="curve">
    <line x1="${p}" y1="${H - 18}" x2="${W - 8}" y2="${H - 18}" stroke="#2c3b35"/>
    ${[10, 20, 30, 40].map((g) => `<text x="${gx(g)}" y="${H - 5}" text-anchor="middle">${g}</text>`).join("")}
    <text x="${W - 8}" y="${H - 22}" text-anchor="end">gust m/s</text>
    <polyline points="${l1}" fill="none" stroke="#e9a23b" stroke-width="2"/>
    <polyline points="${l2}" fill="none" stroke="#59d0ff" stroke-width="2" stroke-dasharray="4 3"/>
    <line x1="${gx(gust)}" y1="8" x2="${gx(gust)}" y2="${H - 18}" stroke="#fff" stroke-dasharray="2 3"/>
    <text x="${p}" y="12" fill="#e9a23b">trees falling (max ${fmt(mx)})</text><text x="${p}" y="24" fill="#59d0ff">P(power-line outage)</text></svg>`;
}


// ---- money: damage of a simulation in euros
const eur = (v) => (v == null || !isFinite(v) ? "–" : Math.abs(v) >= 1e6 ? "€" + fmt(v / 1e6, 1) + "M" : Math.abs(v) >= 1e4 ? "€" + fmt(v / 1e3, 0) + "k" : "€" + fmt(v, 0));
function moneyParts(parts) {
  const tot = Object.values(parts).reduce((a, b) => a + b, 0) || 1;
  const cols = ["#e8772e", "#b8452a", "#f2b544", "#2e6b4a", "#7fa88f", "#5b6b63"];
  return Object.entries(parts).filter(([, v]) => v > 0.5).sort((a, b) => b[1] - a[1]).map(([k, v], i) =>
    `<div class="mrow"><span class="mlab">${esc(k)}</span><span class="mbar"><span style="width:${Math.max(2, (v / tot) * 100)}%;background:${cols[i % cols.length]}"></span></span><span class="mval">${eur(v)}</span></div>`).join("");
}
function stormMoney(D) {
  return `<h3>Damage in euros</h3>
    <div class="money"><div class="mbig">${eur(D.total_mean)}</div><div class="sub">expected cost of this storm · 90 % of storms: ${eur(D.total_p5)} – ${eur(D.total_p95)}</div></div>
    ${moneyParts(D.parts)}
    ${D.trees_worth_felling ? `<div class="callout teal" style="margin-top:10px"><b>Felling ${fmt(D.trees_worth_felling)} trees costs ${eur(D.felling_cost)} and avoids ${eur(D.damage_avoided_by_felling)} of expected damage in a storm like this</b> - every tree whose expected damage is above its felling cost.</div>` : ""}
    ${D.most_costly_trees && D.most_costly_trees.length ? `<table><tr><th>Most costly trees</th><th>Threatens</th><th class="num">Expected damage</th><th class="num">Net if felled</th></tr>${D.most_costly_trees.map((t) => `<tr class="click" data-lon="${t.lon}" data-lat="${t.lat}"><td>${esc(t.species)} ${fmt(t.height_m, 1)} m<br><span class="sub">P(fall) ${pct(t.p_fall, 1)}</span></td><td>${esc(t.threatens === "bld" ? "building" : t.threatens)}</td><td class="num">${eur(t.damage_eur)}</td><td class="num" style="color:${t.net_benefit_eur > 0 ? "var(--teal)" : "var(--muted)"}">${t.net_benefit_eur > 0 ? "+" : ""}${eur(t.net_benefit_eur)}</td></tr>`).join("")}</table>` : ""}
    ${costEditor()}`;
}
function fireMoney(D) {
  if (!D) return "";
  return `<h3>Damage in euros</h3>
    <div class="money"><div class="mbig">${eur(D.total)}</div><div class="sub">cost of this fire · ${fmt(D.burned_ha, 1)} ha burned · ${fmt(D.buildings_reached)} building(s) and ${fmt(D.lines_reached)} power line(s) reached</div></div>
    ${moneyParts(D.parts)}
    ${costEditor()}`;
}
function costEditor() {
  const C = window.COSTS;
  if (!C) { api("/api/site/costs").then((c) => { window.COSTS = c; }); return ""; }
  return `<details class="costs"><summary>Unit costs (€): edit with your own numbers</summary>
    <table>${C.map((c) => `<tr><td>${esc(c.label)}<br><span class="sub">${c.basis === "derived" ? "from published data: " : c.basis === "user" ? "your value · was: " : "assumption: "}${esc(c.source)}</span></td>
      <td class="num"><input type="text" data-cost="${c.key}" value="${c.value}" style="width:84px;text-align:right"><br><span class="sub">${esc(c.unit)}</span></td></tr>`).join("")}</table>
    <button class="btn" id="costSave">Save and re-run</button> <button class="btn ghost" id="costReset">Reset to defaults</button></details>`;
}
function bindCosts(rerun) {
  const save = async (body) => {
    busy(true);
    try { window.COSTS = await api("/api/site/costs", body); await rerun(); } catch (e) { alert(e.message); }
    busy(false);
  };
  const sv = $("#costSave");
  if (sv) sv.onclick = () => { const values = {}; document.querySelectorAll("[data-cost]").forEach((i) => (values[i.dataset.cost] = i.value)); save({ values }); };
  const rs = $("#costReset");
  if (rs) rs.onclick = () => save({ reset: true });
}
api("/api/site/costs").then((c) => { window.COSTS = c; }).catch(() => {});

// ------------------------------------------------------------------ Tree hazards tab
window.tabHazards = async function () {
  siteView(); siteBase({ hillOpacity: 0.6 });
  const chips = `<div class="toggles" style="margin-bottom:10px">
      <button class="chip ${HZ.mode === "storm" ? "on" : ""}" data-hm="storm">≋ Storm: trees that fall</button>
      <button class="chip ${HZ.mode === "fire" ? "on" : ""}" data-hm="fire">▲ Fire: trees that burn</button></div>`;
  if (HZ.mode === "storm") hzStorm(chips); else await hzFire(chips);
  document.querySelectorAll("[data-hm]").forEach((b) => b.addEventListener("click", () => { HZ.mode = b.dataset.hm; window.tabHazards(); }));
  if (typeof dimUpdate === "function") dimUpdate();      // keep the 3D view (if on) in step: storm vs fire colouring
};

function hzStorm(chips) {
  const w = S.sum.weather, R = HZ.storm;
  if (HZ.gust == null) HZ.gust = Math.max(24, Math.round(w.gust_ms));
  if (HZ.dir == null) HZ.dir = Math.round(w.wind_dir_deg);
  const pf = R ? R.p_fall : null;
  hzTrees((t, i) => { const p = pf ? pf[i] : t.pf; return p >= 0.05 ? [riskCol(Math.min(1, p / 0.6)), 2.2] : ["rgba(70,110,85,.45)", 0.9]; });
  if (R) {
    const L = R.fall_lines;
    map.setLayer("falls", { type: "lines", z: 40, data: L.lon0.map((_, i) => ({ path: [[L.lon0[i], L.lat0[i]], [L.lon1[i], L.lat1[i]]], color: FALL_COL[L.kind[i]], width: L.kind[i] ? 3 : 1.6 })) });
  }
  legend(lgRamp("P(tree falls)", RISK, "5 %", "≥ 60 %") + (R ? `<h4 style="margin-top:8px">Simulated fallen trees (one storm)</h4>` + lgRows([[FALL_COL[0], "falls in the forest"], [FALL_COL[1], "falls across a road"], [FALL_COL[2], "falls on the power line"], [FALL_COL[3], "falls on a building"]]) : "") + `<h4 style="margin-top:8px">Infrastructure</h4>${INFRA_LG}`);
  panel(`<h2>Tree hazards</h2>${chips}
    <p class="lead">Simulate a storm: every tree's chance of breaking or uprooting (species, height/diameter, health, stand edge, soil) is drawn ${HZ.runs} times. Fallen trees are laid down in the wind direction to see what they hit.</p>
    <div class="formrow"><label>Max gust</label><input type="range" id="hg" min="10" max="40" step="1" value="${HZ.gust}"><output id="hgv">${HZ.gust} m/s</output></div>
    <div class="formrow"><label>Wind from</label><input type="range" id="hd" min="0" max="359" step="5" value="${HZ.dir}"><output id="hdv">${HZ.dir}°</output></div>
    <button class="btn" id="hrun">Simulate storm</button>
    ${R ? `
    <h3>Result · ${fmt(R.gust_ms)} m/s from ${fmt(R.wind_from)}° · ${R.runs} runs</h3>
    <div class="kpis">
      ${kpi(`${fmt(R.fallen.mean)}`, `trees fall (90 %: ${fmt(R.fallen.p5)}–${fmt(R.fallen.p95)})`, "warn")}
      ${kpi(pct(R.p_line_outage, 0), "P(power-line outage)", R.p_line_outage > 0.5 ? "bad" : "")}
      ${kpi(fmt(R.trees_on_roads_mean, 1), "trees across roads (mean)", R.trees_on_roads_mean > 5 ? "warn" : "")}
      ${kpi(fmt(R.buildings_hit_mean, 2), `buildings hit (mean) · P(any) ${pct(R.p_any_building_hit)}`, R.p_any_building_hit > 0.3 ? "bad" : "")}
      ${kpi(fmt(R.easily_fall.over_50), "trees with P(fall) ≥ 50 %", "bad")}
      ${kpi(fmt(R.easily_fall.over_20), "trees with P(fall) ≥ 20 %", "warn")}
    </div>
    ${R.damage_eur ? stormMoney(R.damage_eur) : ""}
    <h3>How damage grows with the gust</h3>${curveSvg(R.curve, R.gust_ms)}
    <h3>Trees most likely to fall</h3>
    <table><tr><th>Tree</th><th class="num">P(fall)</th><th>Why</th></tr>${R.worst.map((t) => `<tr class="click" data-lon="${t.lon}" data-lat="${t.lat}"><td>${esc(t.species)} ${fmt(t.height_m, 1)} m<br><span class="sub">${esc(t.health)}${t.threatens !== "-" ? " · near " + esc(t.threatens) : ""}</span></td><td class="num">${pct(t.p_fall)}</td><td class="sub">${esc(t.why.join(", "))}</td></tr>`).join("")}</table>
    <p class="note">Click a row to zoom to the tree. The map shows one storm close to the median; numbers are over all runs.</p>` : `<div class="callout">Pick a gust and press <b>Simulate storm</b>. The current conditions are ${fmt(w.gust_ms)} m/s from ${fmt(w.wind_dir_deg)}°; ${fmt(S.sum.storm.trees_easily_fall_now)} trees have P(fall) ≥ 20 % today, ${fmt(S.sum.storm.trees_easily_fall_design)} in a ${fmt(S.sum.storm.design_gust_ms)} m/s design storm.</div>`}
  `);
  $("#hg").oninput = (e) => { HZ.gust = +e.target.value; $("#hgv").textContent = HZ.gust + " m/s"; };
  $("#hd").oninput = (e) => { HZ.dir = +e.target.value; $("#hdv").textContent = HZ.dir + "°"; };
  bindCosts(async () => { HZ.storm = await api("/api/site/storm_sim", { gust: HZ.gust, wind_dir: HZ.dir, runs: HZ.runs }); simChanged(); window.tabHazards(); });
  $("#hrun").onclick = async () => {
    busy(true);
    try { HZ.storm = await api("/api/site/storm_sim", { gust: HZ.gust, wind_dir: HZ.dir, runs: HZ.runs }); simChanged(); } catch (e) { alert(e.message); }
    busy(false); window.tabHazards();
  };
  bindZoom();
}

async function hzFire(chips) {
  if (!HZ.fireTrees || HZ.fireVer !== S.ver) {
    busy(true);
    try { HZ.fireTrees = await api("/api/site/fire_trees?top=20"); HZ.fireVer = S.ver; } catch (e) { busy(false); panel(`<h2>Tree hazards</h2>${chips}<p>${esc(e.message)}</p>`); return; }
    busy(false);
  }
  const F = HZ.fireTrees, run = HZ.fire && HZ.fire.ver === S.ver ? HZ.fire : null;
  map.setLayer("dry", { type: "image", url: tok(`/api/site/layer/dry_fuel.png?v=${S.ver}`), bounds: S.sum.bounds, z: 8, opacity: run ? 0.25 : 0.7, pixelated: true });
  if (run) {
    map.setLayer("arrival", { type: "image", url: tok(run.png_url + "&v=" + S.ver), bounds: S.sum.bounds, z: 20, opacity: 0.55, pixelated: true });
    map.setLayer("ign", { type: "points", z: 70, data: [{ lon: run.ignition[0], lat: run.ignition[1], r: 7, color: "#ff2d2d", stroke: "#fff", label: "ignition" }] });
    const reached = new Map(run.trees.ids.map((id, k) => [id, run.trees.p_torch[k]]));
    hzTrees((t, i) => (reached.has(i) ? (reached.get(i) >= 0.5 ? ["#ff2020", 2.6] : ["#ffa040", 1.6]) : ["rgba(70,110,85,.45)", 0.9]));
  } else {
    hzTrees((t) => (t.fh >= 50 || t.pt >= 0.5 ? [flamCol(t.fh / 100), t.fh >= 70 ? 2.6 : 1.8] : ["rgba(70,110,85,.45)", 0.9]));
  }
  legend((run ? `<h4>Simulated fire · ${fmt(run.hours)} h</h4>` + lgRows([["#ff2020", "tree torches (crown fire)"], ["#ffa040", "surface fire passes under it"]]) + `<div style="height:6px"></div>` : "")
    + lgRamp("Tree fire hazard", FLAM, "low", "very high") + `<div style="height:8px"></div>` + lgRamp("Dry grass + dead fuel today", RISK, "little", "lots"));
  const T = run ? run.trees : null;
  panel(`<h2>Tree hazards</h2>${chips}
    <p class="lead">Which trees burn easily: dead and dry trees, resinous spruce with branches to the ground (ladder fuel), and dry grass or open ground around them. Fuel dryness comes from today's Fire Weather Index (FFMC ${fmt(F.ffmc)}, BUI ${fmt(F.bui)}).</p>
    <div class="kpis">
      ${kpi(pct(F.dryness), "fuel dryness today", F.dryness > 0.6 ? "bad" : F.dryness > 0.35 ? "warn" : "good")}
      ${kpi(fmt(F.very_high), "trees with very high fire hazard (≥ 70)", "bad")}
      ${kpi(fmt(F.torch_if_reached), "trees that would torch if a fire reaches them", "warn")}
      ${kpi(fmt(F.dead), "dead / dry standing trees")}
      ${kpi(fmt(F.grass_ha, 1) + " ha", "open dry grass / clearings")}
      ${kpi(fmt(F.high), "trees with high hazard (50–69)")}
    </div>
    <h3>Simulate a fire</h3>
    <div class="formrow"><label>Duration</label><select id="fh">${[1, 2, 4, 6].map((h) => `<option value="${h}" ${h === HZ.hours ? "selected" : ""}>${h} h</option>`).join("")}</select></div>
    <p class="note">Click the map where the fire starts. It spreads with today's wind, slope and fuels; each tree it reaches is checked for torching (Van Wagner crown-fire criterion: crown base height vs. surface-fire intensity and foliage moisture).</p>
    ${T ? `
    <h3>Result · ${fmt(run.hours)} h fire</h3>
    <div class="kpis">
      ${kpi(fmt(T.reached), "trees reached by the fire", "warn")}${kpi(fmt(T.torching), "trees torching (crown fire)", "bad")}
      ${kpi(fmt(T.dead_reached), "dead trees burning")}${kpi(fmt(run.area_ha[run.hours + "h"] ?? Object.values(run.area_ha).pop(), 1) + " ha", "burned area")}
    </div>
    ${fireMoney(run.damage_eur)}
    <table><tr><th>First trees to torch</th><th class="num">min</th><th>Why</th></tr>${T.worst.map((t) => `<tr class="click" data-lon="${t.lon}" data-lat="${t.lat}"><td>${esc(t.species)} ${fmt(t.height_m, 1)} m<br><span class="sub">${esc(t.health)} · crown base ${fmt(t.crown_base_m, 1)} m</span></td><td class="num">${fmt(t.minutes)}</td><td class="sub">${esc(t.why.join(", "))}</td></tr>`).join("")}</table>` : ""}
    <h3>Trees that burn most easily today</h3>
    <table><tr><th>Tree</th><th class="num">Hazard</th><th>Why</th></tr>${F.trees.map((t) => `<tr class="click" data-lon="${t.lon}" data-lat="${t.lat}"><td>${esc(t.species)} ${fmt(t.height_m, 1)} m<br><span class="sub">${esc(t.health)}${t.near !== "-" ? " · near " + esc(t.near) : ""}</span></td><td class="num">${fmt(t.fire_hazard)}</td><td class="sub">${esc(t.why.join(", "))}</td></tr>`).join("")}</table>
    <p class="note">Change "Conditions" at the top (e.g. Drought / heatwave) to see how hazard grows as fuels dry out.</p>
  `);
  $("#fh").onchange = (e) => (HZ.hours = +e.target.value);
  if (run) bindCosts(async () => { const r = await api("/api/site/fire", { lon: run.ignition[0], lat: run.ignition[1], hours: run.hours }); r.ver = S.ver; HZ.fire = r; S.fire = r; simChanged(); window.tabHazards(); });
  S.click = async (ev) => {
    busy(true);
    try { const r = await api("/api/site/fire", { lon: ev.lon, lat: ev.lat, hours: HZ.hours }); r.ver = S.ver; HZ.fire = r; S.fire = r; simChanged(); } catch (e) { alert(e.message); }
    busy(false); window.tabHazards();
  };
  hint("Click the map to start a fire");
  bindZoom();
}

// ------------------------------------------------------------------ Emergency plan tab
window.tabPlan = function () {
  siteView(); siteBase({ hillOpacity: 0.6 });
  stormLayers();
  if (S.fire) map.setLayer("arrival", { type: "image", url: tok(S.fire.png_url + "&v=" + S.ver), bounds: S.sum.bounds, z: 20, opacity: 0.6, pixelated: true });
  const P = HZ.plan && HZ.plan.ver === S.ver ? HZ.plan : null, blds = S.vec.buildings;
  if (HZ.end) map.setLayer("dest", { type: "points", z: 75, data: [{ lon: HZ.end[0], lat: HZ.end[1], r: 7, color: "#e5534b", stroke: "#fff", label: "destination" }] });
  if (P) {
    const ok = P.options.filter((o) => o.ok);
    map.setLayer("routes", { type: "lines", z: 65, data: ok.sort((a, b) => (a.vehicle === P.best) - (b.vehicle === P.best)).map((o) => ({ path: o.path, color: VEH[o.vehicle][1], width: o.vehicle === P.best ? 6 : 2.2, dash: o.vehicle === P.best ? [] : [5, 4], casing: "rgba(0,0,0,.7)" })) });
  }
  legend(lgRamp("Road piece: P(blocked by fallen tree)", RISK, "0", "≥ 80 %") + `<h4 style="margin-top:8px">Routes (thick = recommended)</h4>` + lgRows(Object.values(VEH).map(([l, c]) => [c, l])));
  S.click = (ev) => { HZ.end = [ev.lon, ev.lat]; HZ.endKey = ""; HZ.plan = null; window.tabPlan(); };
  hint(HZ.end ? "" : "Click the map to set the emergency location, or pick a building");
  const w = S.sum.weather;
  panel(`
    <h2>Emergency plan</h2>
    <p class="lead">Finds the safest way to an emergency with every vehicle class, using everything Horus knows: road width vs. vehicle size, trees that can fall across the road in the current wind, roads already likely blocked, fire hazard and torching trees beside the route, the simulated fire front, slope and overhanging branches.</p>
    <div class="formrow"><label>Emergency</label><select id="pm">${[["any", "Any / first response"], ["fire", "Fire suppression"], ["medical", "Medical / evacuation"], ["power", "Power-line repair"]].map(([k, l]) => `<option value="${k}" ${k === HZ.mission ? "selected" : ""}>${l}</option>`).join("")}</select></div>
    <div class="formrow"><label>Destination</label><select id="pd"><option value="">(click on map)</option>${blds.map((b, i) => `<option value="b${i}" ${HZ.endKey === "b" + i ? "selected" : ""}>${esc(b.name)}</option>`).join("")}</select></div>
    ${S.fire ? `<div class="formrow"><label>Avoid fire front</label><input type="checkbox" id="pa" ${HZ.avoid ? "checked" : ""}> stay out of areas burning within 60 min</div>` : ""}
    <button class="btn" id="pgo" ${HZ.end ? "" : "disabled"}>Find safest route</button>
    <p class="note">Conditions: ${esc(w.scenario)} · gust ${fmt(w.gust_ms)} m/s from ${fmt(w.wind_dir_deg)}° · FWI ${fmt(w.fwi, 1)}${S.fire ? " · fire simulated" : ""}. Change them at the top, or simulate a fire in Tree hazards first.</p>
    ${P ? planHtml(P) : ""}
  `);
  $("#pm").onchange = (e) => { HZ.mission = e.target.value; HZ.plan = null; };
  $("#pd").onchange = (e) => { HZ.endKey = e.target.value; HZ.end = e.target.value ? blds[+e.target.value.slice(1)].pos : null; HZ.plan = null; window.tabPlan(); };
  if ($("#pa")) $("#pa").onchange = (e) => (HZ.avoid = e.target.checked);
  $("#pgo").onclick = async () => {
    busy(true);
    try { const r = await api("/api/site/plan", { end: HZ.end, mission: HZ.mission, avoid_fire_min: HZ.avoid && S.fire ? 60 : null }); r.ver = S.ver; HZ.plan = r; } catch (e) { alert(e.message); }
    busy(false); window.tabPlan();
  };
};

function planHtml(P) {
  const row = (o) => {
    if (!o.ok) return `<tr><td>${esc(o.label)}</td><td colspan="3" class="sub" style="color:var(--red)">${esc(o.reason)}</td></tr>`;
    const r = o.risk, star = o.vehicle === P.best ? "★ " : "";
    return `<tr class="${o.vehicle === P.best ? "best" : ""}"><td>${star}<span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${VEH[o.vehicle][1]}"></span> ${esc(o.label)}${o.suits_mission ? "" : ` <span class="sub">(not for this job)</span>`}
      <br><span class="sub">${fmt(o.fit.width_m, 1)} m wide · ${fmt(o.fit.height_m, 1)} m tall · ${fmt(o.fit.mass_t)} t · needs road ≥ ${fmt(o.fit.min_road_w, 1)} m</span></td>
      <td class="num">${fmt(o.expected_minutes)} min${r.expected_clearing_min > 0.5 ? `<br><span class="sub">incl. ${fmt(r.expected_clearing_min)} clearing</span>` : ""}</td>
      <td>${lvlBadge(o.level, o.danger)}<br><span class="sub">arrives ${pct(o.p_arrive)}</span></td>
      <td class="sub">${r.warnings.length ? esc(r.warnings.join("; ")) : "no hazards found on the route"}</td></tr>`;
  };
  return `<h3>Recommendation</h3><div class="callout rec">${esc(P.recommendation)}</div>
    <p class="note">Vehicles that fit and get through: ${P.vehicles_that_fit.length ? P.vehicles_that_fit.map((k) => esc(VEH[k][0])).join(", ") : "none"}.</p>
    <table class="plan"><tr><th>Vehicle</th><th class="num">ETA</th><th>Danger</th><th>On the route</th></tr>${P.options.map(row).join("")}</table>
    <p class="note">Danger 0–100 combines the chance of being stopped by a fallen tree, trees that can fall onto the route, fire risk and torching trees beside it, the fire front, and off-road distance. Roads narrower than a vehicle needs are not used by it.</p>`;
}
