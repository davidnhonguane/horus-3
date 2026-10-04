// Horus - Infrastructure tab: roads, power lines, houses and critical infrastructure
// found in the 3D scan, in images, and from the National Land Survey (NLS) topographic database.
"use strict";

const INF = { on: { ortho: true, lidar: true, image: true, power: true, bld: true, nls: true }, data: null, nls: null, ver: -1 };
const EVID_COL = { "LiDAR + image": "#ff6a00", LiDAR: "#ffa94d", image: "#ffe066" };
const NLS_BCOL = { house: "#f4f4f4", cabin: "#d9b38c", public: "#ff6b9a", industrial: "#9aa5b1", church: "#c9a0ff", other: "#bdbdbd" };
const CRIT_COL = { substation: "#ff3b30", transformer: "#ff7b54", mast: "#ffd60a", water_tower: "#4cc3ff" };

function infTabLegend() {
  const on = INF.on;
  let h = `<h4>Found in the forest</h4>` + lgRows([[EVID_COL["LiDAR + image"], "road: 3D scan + image (confirmed)"], [EVID_COL.LiDAR, "road: 3D scan only"], [EVID_COL.image, "road: image only"],
    ["#ff5fd2", "power line: wires in 3D scan"], ["rgba(120,255,140,.8)", "power-line clearing seen in image"], ["#b07cff", "building found in 3D scan"]]);
  if (on.nls && INF.nls?.ok) h += `<h4 style="margin-top:6px">National Land Survey</h4>` + lgRows([["#59d0ff", "power line"], ["#ffffff", "road"], ...Object.entries(NLS_BCOL).map(([k, c]) => [c, k === "public" ? "public / commercial" : k]), ["#ff3b30", "critical infrastructure"]]);
  legend(h);
}

function infTabLayers() {
  const D = INF.data, on = INF.on;
  map.clearLayers("inf_");
  if (on.ortho && D.image) map.setLayer("inf_ortho", { type: "image", z: 2, url: tok("/api/site/ortho.jpg"), bounds: D.image.bounds, opacity: 0.95 });
  if (on.lidar) map.setLayer("inf_lidar", { type: "lines", z: 52, data: D.lidar_roads.map((r) => ({ path: r.path, color: r.mapped ? "rgba(255,255,255,.35)" : EVID_COL[r.evidence], width: r.mapped ? 2 : 4.5, dash: r.mapped ? [2, 4] : [8, 4], casing: r.mapped ? null : "rgba(0,0,0,.65)" })) });
  if (on.image) map.setLayer("inf_image", { type: "lines", z: 51, data: D.image_roads.filter((r) => !r.mapped && r.evidence === "image").map((r) => ({ path: r.path, color: EVID_COL.image, width: 3.5, dash: [3, 4], casing: "rgba(0,0,0,.6)" })) });
  if (on.power) {
    map.setLayer("inf_corr", { type: "lines", z: 49, data: D.corridors.map((c) => ({ path: c.path, color: "rgba(120,255,140,.35)", width: 14 })) });
    map.setLayer("inf_wires", { type: "lines", z: 56, data: D.wires.map((w) => ({ path: w.path, color: "#ff5fd2", width: 2.5, dash: [7, 4], casing: "rgba(0,0,0,.6)" })) });
  }
  if (on.bld) map.setLayer("inf_bld", { type: "points", z: 62, hover: true, data: D.buildings.filter((b) => !b.mapped).map((b) => ({ lon: b.pos[0], lat: b.pos[1], r: 8, color: "#b07cff", stroke: "#fff", strokeWidth: 2, label: "found: building" })),
    tip: (i) => { const b = D.buildings.filter((x) => !x.mapped)[i]; return `<b>Building found in 3D scan</b><br>${b.area_m2} m² roof, ${b.height_m} m high<br>${esc(b.method)}`; } });
  const N = INF.nls;
  if (on.nls && N?.ok) {
    map.setLayer("inf_nls_roads", { type: "lines", z: 45, data: N.roads.map((r) => ({ path: r.path, color: r.cls === "main" ? "#ffffff" : r.cls === "path" ? "rgba(255,255,255,.5)" : "#e6d8b0", width: r.cls === "main" ? 3.5 : r.cls === "path" ? 1 : 2.2, dash: r.cls === "path" || /track|winter/.test(r.label) ? [4, 3] : [], casing: "rgba(0,0,0,.5)" })) });
    map.setLayer("inf_nls_power", { type: "lines", z: 47, data: N.power.map((p) => ({ path: p.path, color: "#59d0ff", width: 2.5, dash: [9, 4], casing: "rgba(0,0,0,.6)" })) });
    map.setLayer("inf_nls_bld", { type: "points", z: 60, hover: true, data: N.buildings.map((b) => ({ lon: b.lon, lat: b.lat, r: 4, color: NLS_BCOL[b.kind] || "#ddd", stroke: "#111" })),
      tip: (i) => `<b>${esc(N.buildings[i].label)}</b><br>NLS topographic database` });
    map.setLayer("inf_nls_crit", { type: "points", z: 63, hover: true, data: N.critical.map((c) => ({ lon: c.lon, lat: c.lat, r: 8, color: CRIT_COL[c.kind] || "#ff3b30", stroke: "#fff", strokeWidth: 2, label: c.label })),
      tip: (i) => `<b>${esc(N.critical[i].label)}</b><br>critical infrastructure (NLS)` });
  }
}

window.tabInfra = async function () {
  siteView(); siteBase({ hillOpacity: 0.6 });
  if (!INF.data || INF.ver !== S.ver) {
    busy(true);
    try { INF.data = await api("/api/site/infrastructure"); INF.ver = S.ver; INF.nls = INF.data.nls; } catch (e) { panel(`<h2>Infrastructure</h2><p>${esc(e.message)}</p>`); busy(false); return; }
    busy(false);
  }
  if (S.tab !== "infra") return;
  const D = INF.data, on = INF.on, u = D.unmapped || {};
  infTabLayers(); infTabLegend();
  const chip = (k, l) => `<button class="chip ${on[k] ? "on" : ""}" data-if="${k}">${l}</button>`;
  const roads = D.lidar_roads.filter((r) => !r.mapped).concat(D.image_roads.filter((r) => !r.mapped && r.evidence === "image"));
  const nb = D.buildings.filter((b) => !b.mapped);
  const N = INF.nls;
  const synthetic = S.sum.meta.kind === "synthetic";
  panel(`
    <h2>Infrastructure</h2>
    <p class="lead">Roads, power lines and buildings found in the forest from the <b>3D scan</b> and from <b>images</b>, plus the National Land Survey's register of houses, critical infrastructure, roads and power lines.</p>
    <div class="toggles">${chip("ortho", "Orthophoto")}${chip("lidar", "Roads from 3D")}${chip("image", "Roads from image")}${chip("power", "Power lines")}${chip("bld", "Buildings found")}${chip("nls", "NLS map data")}</div>
    <h3>Found in the forest (not in the map data)</h3>
    <div class="kpis">${kpi(fmt(u.roads ?? 0), "potential forest roads", u.roads ? "warn" : "")}${kpi(fmt(u.buildings ?? 0), "buildings in the forest", u.buildings ? "warn" : "")}
      ${kpi(fmt(u.power_lines ?? 0), "unmapped power lines (wires)", u.power_lines ? "warn" : "")}${kpi(fmt(u.line_corridors ?? 0), "unexplained line clearings (image)", u.line_corridors ? "warn" : "")}</div>
    <table style="margin-top:8px"><tr><th>Find</th><th>Evidence</th><th class="num">Size</th></tr>
      ${roads.map((r, i) => `<tr class="click" data-go="${r.path[Math.floor(r.path.length / 2)].join(",")}"><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${EVID_COL[r.evidence]}"></span> Forest road #${i + 1}</td><td>${esc(r.evidence)}</td><td class="num">${r.length_m} m · ${pct(r.confidence)}</td></tr>`).join("")}
      ${nb.map((b, i) => `<tr class="click" data-go="${b.pos.join(",")}"><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#b07cff"></span> Building #${i + 1}</td><td>3D scan</td><td class="num">${b.area_m2} m² · ${b.height_m} m</td></tr>`).join("")}
      ${D.wires.map((w, i) => `<tr class="click" data-go="${w.path[0].join(",")}"><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#ff5fd2"></span> Power line ${i + 1}${w.mapped ? "" : " (unmapped)"}</td><td>${esc(w.evidence)}</td><td class="num">${w.length_m} m · ${w.height_m} m high</td></tr>`).join("")}
      ${D.corridors.map((c, i) => `<tr class="click" data-go="${c.path[0].join(",")}"><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:rgba(120,255,140,.8)"></span> Line clearing ${i + 1}</td><td>${esc(c.evidence)}</td><td class="num">${c.length_m} m · ${c.width_m} m wide</td></tr>`).join("")}
    </table>
    <p class="note"><b>3D scan:</b> roads are open, smooth, flat strips with forest on both sides; power lines are wires hanging at constant height; buildings are smooth, solid roofs. <b>Image:</b> roads are bare gravel or soil corridors; power lines show as long, straight grassy clearings (the wires themselves are too thin to see). A find seen in both sources is the most reliable. Ground-check before use.</p>
    <h3>Image used</h3>
    <p class="note">${D.image ? `${esc(D.image.source)}${D.image.error ? ` - error: ${esc(D.image.error)}` : ""}` : "No image. Start with <code>--image photo.jpg</code> (plus a world file) or <code>--nls</code> to use the NLS orthophoto."}</p>
    <h3>National Land Survey register</h3>
    ${N ? (N.ok ? `<div class="kpis">${kpi(fmt(N.counts.buildings), "houses / buildings")}${kpi(fmt(N.counts.critical), "critical infrastructure", N.counts.critical ? "warn" : "")}${kpi(fmt(N.counts.roads), "road segments")}${kpi(fmt(N.counts.power_lines), "power-line segments")}</div>
        <table style="margin-top:8px">${Object.entries(N.counts.by_building_type).map(([k, n]) => `<tr><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${NLS_BCOL[k] || "#ddd"}"></span> ${esc(k === "public" ? "public / commercial" : k)}</td><td class="num">${n}</td></tr>`).join("")}
        ${N.critical.map((c) => `<tr class="click" data-go="${c.lon},${c.lat}"><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${CRIT_COL[c.kind]}"></span> ${esc(c.label)}</td><td class="num">${c.lat.toFixed(5)} N, ${c.lon.toFixed(5)} E</td></tr>`).join("")}</table>
        ${synthetic ? `<p class="note">These are the real features at this location in Kuopio. The demo forest is synthetic, so they are shown for reference only. On a real scan (<code>--las … --nls</code>) they feed routing, evacuation and storm risk.</p>` : ""}`
      : `<div class="callout red">${esc(N.error || "NLS data unavailable")}</div>`)
      : `<button class="btn" id="nlsload">Load NLS houses, critical infrastructure, roads and power lines</button><p class="note">Uses the key in <code>nls_key.txt</code>.</p>`}
  `);
  document.querySelectorAll("[data-if]").forEach((b) => b.addEventListener("click", () => { on[b.dataset.if] = !on[b.dataset.if]; window.tabInfra(); }));
  document.querySelectorAll("tr[data-go]").forEach((tr) => tr.addEventListener("click", () => { const [lo, la] = tr.dataset.go.split(",").map(Number); map.center = [lo, la]; map.zoom = 17.5; map.redraw(); }));
  const nl = $("#nlsload");
  if (nl) nl.onclick = async () => { busy(true); try { INF.nls = await api("/api/site/nls"); } catch (e) { INF.nls = { ok: false, error: e.message }; } busy(false); window.tabInfra(); };
};
