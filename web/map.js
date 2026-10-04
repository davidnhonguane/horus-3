// Horus MapView - tiny dependency-free slippy map (Web Mercator) on <canvas>.
// Layers: raster images with lat/lon bounds, polylines, points, circles, custom draw functions.
// Optional basemap tiles (OSM / Esri imagery) when online; works offline without them.
"use strict";

const TILE = 256;
const BASEMAPS = {
  none: null,
  osm: { url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", attr: "© OpenStreetMap contributors", max: 19 },
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr: "Imagery © Esri, Maxar, Earthstar Geographics", max: 19,
  },
  s2cloudless: {
    url: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg",
    attr: "Sentinel-2 cloudless 2021 by EOX IT Services GmbH (modified Copernicus Sentinel data)", max: 15,
  },
  // NLS layers go through the Horus server so the API key never reaches the browser
  nls_orto: { url: "/api/tiles/nls/ortokuva/{z}/{x}/{y}", attr: "Orthophoto © Maanmittauslaitos (NLS Finland), CC BY 4.0", max: 19 },
  nls_maasto: { url: "/api/tiles/nls/maastokartta/{z}/{x}/{y}", attr: "Topographic map © Maanmittauslaitos (NLS Finland), CC BY 4.0", max: 19 },
};

function lon2x(lon, z) { return ((lon + 180) / 360) * TILE * 2 ** z; }
function lat2y(lat, z) {
  const s = Math.sin((lat * Math.PI) / 180);
  return (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * TILE * 2 ** z;
}
function x2lon(x, z) { return (x / (TILE * 2 ** z)) * 360 - 180; }
function y2lat(y, z) {
  const n = Math.PI - (2 * Math.PI * y) / (TILE * 2 ** z);
  return (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
}

class MapView {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.center = [27.26, 62.75];
    this.zoom = 15;
    this.layers = [];
    this.basemap = "osm";
    this.tiles = new Map();
    this.imgCache = new Map();
    this.onclick = null;
    this.onhover = null;
    this._raf = 0;
    this._bind();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
  }
  resize() {
    const r = this.c.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    this.c.width = Math.round(r.width * dpr); this.c.height = Math.round(r.height * dpr);
    this.c.style.width = r.width + "px"; this.c.style.height = r.height + "px";
    if (this._pendingFit && r.width) { const f = this._pendingFit; this._pendingFit = null; this.fit(...f); }
    this.redraw();
  }
  project(lon, lat) {
    const z = this.zoom;
    return [lon2x(lon, z) - lon2x(this.center[0], z) + this.w / 2, lat2y(lat, z) - lat2y(this.center[1], z) + this.h / 2];
  }
  unproject(px, py) {
    const z = this.zoom;
    return [x2lon(px - this.w / 2 + lon2x(this.center[0], z), z), y2lat(py - this.h / 2 + lat2y(this.center[1], z), z)];
  }
  fit(bounds, pad = 30) { // [[lat0,lon0],[lat1,lon1]]
    const r = this.c.parentElement.getBoundingClientRect();
    if (!r.width || !r.height) { this._pendingFit = [bounds, pad]; return; }
    if (r.width !== this.w || r.height !== this.h) { this._pendingFit = null; this.w = r.width; this.h = r.height; this.resize(); }
    const [[a, b], [c, d]] = bounds;
    this.center = [(b + d) / 2, (a + c) / 2];
    for (let z = 19; z >= 2; z -= 0.25) {
      const w = Math.abs(lon2x(d, z) - lon2x(b, z)), h = Math.abs(lat2y(a, z) - lat2y(c, z));
      if (w < this.w - 2 * pad && h < this.h - 2 * pad) { this.zoom = z; break; }
    }
    this.redraw();
  }
  setLayer(id, layer) {
    const i = this.layers.findIndex((l) => l.id === id);
    layer.id = id;
    if (i >= 0) this.layers[i] = layer; else this.layers.push(layer);
    this.layers.sort((a, b) => (a.z || 0) - (b.z || 0));
    this.redraw();
  }
  removeLayer(id) { this.layers = this.layers.filter((l) => l.id !== id); this.redraw(); }
  clearLayers(prefix) { this.layers = this.layers.filter((l) => !l.id.startsWith(prefix)); this.redraw(); }
  layer(id) { return this.layers.find((l) => l.id === id); }
  redraw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }
  image(url) {
    let im = this.imgCache.get(url);
    if (!im) {
      im = new Image();
      im.onload = () => this.redraw();
      im.src = url;
      this.imgCache.set(url, im);
    }
    return im;
  }
  _tiles(ctx) {
    const bm = BASEMAPS[this.basemap];
    if (bm) this._drawTiles(ctx, bm.url, bm.max, this.basemap);
  }
  _drawTiles(ctx, url, max, cacheKey) {
    const z = Math.max(0, Math.min(max, Math.round(this.zoom)));
    const scale = 2 ** (this.zoom - z);
    const cx = lon2x(this.center[0], z), cy = lat2y(this.center[1], z);
    const x0 = Math.floor((cx - this.w / 2 / scale) / TILE), x1 = Math.floor((cx + this.w / 2 / scale) / TILE);
    const y0 = Math.floor((cy - this.h / 2 / scale) / TILE), y1 = Math.floor((cy + this.h / 2 / scale) / TILE);
    const n = 2 ** z;
    for (let tx = x0; tx <= x1; tx++) for (let ty = y0; ty <= y1; ty++) {
      if (ty < 0 || ty >= n) continue;
      const wx = ((tx % n) + n) % n;
      const key = `${cacheKey}/${z}/${wx}/${ty}`;
      let t = this.tiles.get(key);
      if (!t) {
        t = new Image();
        t.crossOrigin = "anonymous";
        t.onload = () => this.redraw();
        t.onerror = () => { t.failed = true; };
        t.src = (this.tileUrlHook ? this.tileUrlHook(url) : url).replace("{z}", z).replace("{x}", wx).replace("{y}", ty);
        this.tiles.set(key, t);
        if (this.tiles.size > 900) this.tiles.delete(this.tiles.keys().next().value);
      }
      if (t.complete && t.naturalWidth && !t.failed) {
        const px = (tx * TILE - cx) * scale + this.w / 2, py = (ty * TILE - cy) * scale + this.h / 2;
        ctx.drawImage(t, px, py, TILE * scale + 0.5, TILE * scale + 0.5);
      }
    }
  }
  draw() {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--map-bg") || "#1a2420";
    ctx.fillRect(0, 0, this.w, this.h);
    this._tiles(ctx);
    for (const L of this.layers) {
      if (L.visible === false) continue;
      ctx.save();
      ctx.globalAlpha = L.opacity ?? 1;
      try { this._drawLayer(ctx, L); } catch (e) { console.warn(e); }
      ctx.restore();
    }
    const bm = BASEMAPS[this.basemap];
    if (bm) {
      ctx.font = "10px system-ui, sans-serif";
      const tw = ctx.measureText(bm.attr).width;
      ctx.fillStyle = "rgba(10,16,14,.7)"; ctx.fillRect(this.w - tw - 10, this.h - 16, tw + 10, 16);
      ctx.fillStyle = "#cfd8d3"; ctx.fillText(bm.attr, this.w - tw - 5, this.h - 5);
    }
    this._scalebar(ctx);
  }
  _scalebar(ctx) {
    const mpp = (156543.03 * Math.cos((this.center[1] * Math.PI) / 180)) / 2 ** this.zoom;
    const target = 110 * mpp;
    const p = 10 ** Math.floor(Math.log10(target));
    const m = [1, 2, 5, 10].map((k) => k * p).filter((v) => v <= target).pop();
    const px = m / mpp;
    ctx.fillStyle = "rgba(10,16,14,.7)"; ctx.fillRect(10, this.h - 26, px + 16, 18);
    ctx.strokeStyle = "#e6ede9"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(18, this.h - 13); ctx.lineTo(18 + px, this.h - 13); ctx.stroke();
    ctx.fillStyle = "#e6ede9"; ctx.font = "10px system-ui"; ctx.fillText(m >= 1000 ? m / 1000 + " km" : m + " m", 20, this.h - 16);
  }
  _path(ctx, pts) {
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = this.project(pts[i][0], pts[i][1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
  }
  _drawLayer(ctx, L) {
    if (L.type === "tiles") {
      this._drawTiles(ctx, L.url, L.max || 18, L.id + "|" + L.url);
    } else if (L.type === "image") {
      const im = this.image(L.url);
      if (!im.complete || !im.naturalWidth) return;
      const [[a, b], [c, d]] = L.bounds;
      const [x0, y0] = this.project(b, c), [x1, y1] = this.project(d, a);
      ctx.imageSmoothingEnabled = !L.pixelated;
      ctx.drawImage(im, x0, y0, x1 - x0, y1 - y0);
    } else if (L.type === "lines") {
      for (const f of L.data) {
        this._path(ctx, f.path);
        if (f.casing) { ctx.strokeStyle = f.casing; ctx.lineWidth = (f.width || 2) + 3; ctx.setLineDash([]); ctx.stroke(); }
        ctx.strokeStyle = f.color || L.color || "#fff";
        ctx.lineWidth = f.width || L.width || 2;
        ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.setLineDash(f.dash || L.dash || []);
        ctx.stroke();
      }
    } else if (L.type === "points") {
      const r = typeof L.radius === "function" ? null : L.radius || 3;
      const n = L.n ?? L.data.length;
      const minPx = L.minRadiusPx ?? 1;
      for (let i = 0; i < n; i++) {
        const p = L.get ? L.get(i) : L.data[i];
        if (!p) continue;
        const [x, y] = this.project(p.lon, p.lat);
        if (x < -20 || y < -20 || x > this.w + 20 || y > this.h + 20) continue;
        let rr = p.r ?? r;
        if (L.metres) rr = Math.max(p.minPx ?? minPx, rr / ((156543.03 * Math.cos((p.lat * Math.PI) / 180)) / 2 ** this.zoom));
        ctx.fillStyle = p.color || L.color || "#fff";
        if (rr < 1.6) ctx.fillRect(x - rr, y - rr, rr * 2, rr * 2);
        else { ctx.beginPath(); ctx.arc(x, y, rr, 0, 6.2832); ctx.fill(); }
        if (p.stroke || L.stroke) { ctx.strokeStyle = p.stroke || L.stroke; ctx.lineWidth = p.strokeWidth || 1.5; ctx.stroke(); }
        if (p.label && this.zoom >= (L.labelZoom ?? 0)) {
          ctx.font = "600 11px system-ui"; ctx.lineWidth = 3; ctx.strokeStyle = "rgba(8,12,10,.85)";
          ctx.strokeText(p.label, x + rr + 4, y + 4); ctx.fillStyle = "#e9efe9"; ctx.fillText(p.label, x + rr + 4, y + 4);
        }
      }
    } else if (L.type === "circle") {
      const [x, y] = this.project(L.lon, L.lat);
      const mpp = (156543.03 * Math.cos((L.lat * Math.PI) / 180)) / 2 ** this.zoom;
      ctx.beginPath(); ctx.arc(x, y, L.radius_m / mpp, 0, 6.2832);
      ctx.fillStyle = L.fill || "rgba(255,255,255,.1)"; ctx.fill();
      ctx.strokeStyle = L.color || "#fff"; ctx.lineWidth = L.width || 2; ctx.setLineDash(L.dash || []); ctx.stroke();
    } else if (L.type === "polygon") {
      for (const ring of L.rings) {
        this._path(ctx, ring); ctx.closePath();
        ctx.fillStyle = L.fill || "rgba(255,255,255,.06)"; ctx.fill();
        ctx.strokeStyle = L.color || "#999"; ctx.lineWidth = L.width || 1; ctx.stroke();
      }
    } else if (L.type === "polys") {
      for (const f of L.data) {
        for (const ring of f.rings) { this._path(ctx, ring); ctx.closePath(); }
        ctx.fillStyle = f.fill || "rgba(255,255,255,.08)"; ctx.fill("evenodd");
        ctx.strokeStyle = f.color || "#ccc"; ctx.lineWidth = f.width || 1.2; ctx.setLineDash([]); ctx.stroke();
      }
    } else if (L.type === "custom") {
      L.draw(ctx, this);
    }
  }
  hitTest(px, py) {
    for (let k = this.layers.length - 1; k >= 0; k--) {
      const L = this.layers[k];
      if (L.visible === false || !L.hover || L.type !== "points") continue;
      const n = L.n ?? L.data.length;
      let best = -1, bd = 1e9;
      for (let i = 0; i < n; i++) {
        const p = L.get ? L.get(i) : L.data[i];
        if (!p || (L.hoverFilter && !L.hoverFilter(p, i))) continue;
        const [x, y] = this.project(p.lon, p.lat);
        const d = (x - px) ** 2 + (y - py) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      if (best >= 0 && bd < 100) return { layer: L, index: best };
    }
    return null;
  }
  _bind() {
    const c = this.c;
    let drag = null, moved = false;
    c.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY, ...this.center]; moved = false; c.setPointerCapture(e.pointerId); });
    c.addEventListener("pointermove", (e) => {
      const r = c.getBoundingClientRect();
      if (drag) {
        const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
        if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
        const z = this.zoom;
        this.center = [x2lon(lon2x(drag[2], z) - dx, z), y2lat(lat2y(drag[3], z) - dy, z)];
        this.redraw();
      } else if (this.onhover) {
        this.onhover(this.hitTest(e.clientX - r.left, e.clientY - r.top), e);
      }
    });
    c.addEventListener("pointerup", (e) => {
      const r = c.getBoundingClientRect();
      if (drag && !moved && this.onclick) {
        const [lon, lat] = this.unproject(e.clientX - r.left, e.clientY - r.top);
        this.onclick({ lon, lat, hit: this.hitTest(e.clientX - r.left, e.clientY - r.top), event: e });
      }
      drag = null;
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const [lon, lat] = this.unproject(px, py);
      this.zoom = Math.max(3, Math.min(20, this.zoom - Math.sign(e.deltaY) * 0.25));
      const [nx, ny] = this.project(lon, lat);
      this.center = this.unproject(this.w / 2 + (nx - px), this.h / 2 + (ny - py));
      this.redraw();
    }, { passive: false });
  }
}
window.MapView = MapView;
window.BASEMAPS = BASEMAPS;
