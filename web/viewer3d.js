// Horus 3D point-cloud viewer - raw WebGL, orbit camera, colour modes.
"use strict";

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function mat4LookAt(e, t, u) {
  let zx = e[0] - t[0], zy = e[1] - t[1], zz = e[2] - t[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = u[1] * zz - u[2] * zy, xy = u[2] * zx - u[0] * zz, xz = u[0] * zy - u[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
    -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
}
function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s;
  }
  return o;
}

const VS = `attribute vec3 p; attribute vec3 c; uniform mat4 m; uniform float s; uniform float mx; varying vec3 vc;
void main(){ vec4 q = m*vec4(p,1.0); gl_Position=q; gl_PointSize=clamp(s/q.w,1.0,mx); vc=c; }`;
const FS = `precision mediump float; varying vec3 vc; void main(){ gl_FragColor=vec4(vc,1.0); }`;

class PointViewer {
  constructor(canvas) {
    this.c = canvas;
    const gl = (this.gl = canvas.getContext("webgl", { antialias: true }));
    if (!gl) { this.failed = true; return; }
    const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const pr = (this.prog = gl.createProgram());
    gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(pr); gl.useProgram(pr);
    this.loc = { p: gl.getAttribLocation(pr, "p"), c: gl.getAttribLocation(pr, "c"), m: gl.getUniformLocation(pr, "m"), s: gl.getUniformLocation(pr, "s"), mx: gl.getUniformLocation(pr, "mx") };
    this.yaw = -0.7; this.pitch = 0.55; this.dist = 900; this.target = [0, 20, 0];
    this.mode = "rgb";
    this._bind();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }
  async load(url) {
    const buf = await (await fetch(url)).arrayBuffer();
    const hl = new DataView(buf).getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const n = head.n; let off = 4 + hl;
    this.xyz = new Float32Array(buf.slice(off, off + n * 12)); off += n * 12;
    this.cls = new Uint8Array(buf.slice(off, off + n)); off += n;
    this.haz = new Uint8Array(buf.slice(off, off + n)); off += n;
    this.health = new Uint8Array(buf.slice(off, off + n)); off += n;
    this.rgb = new Uint8Array(buf.slice(off, off + n * 3)); off += n * 3;
    this.fire = off + n <= buf.byteLength ? new Uint8Array(buf.slice(off, off + n)) : null;   // tree fire hazard (newer servers)
    this.n = n;
    let ymin = 1e9, ymax = -1e9;
    for (let i = 0; i < n; i++) { const y = this.xyz[i * 3 + 1]; if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
    this.ymin = ymin; this.ymax = ymax;
    // frame the whole scan, whatever its size (64 ha demo or a 1 ha uploaded stand)
    let xr = 0, zr = 0;
    for (let i = 0; i < n; i += 7) { xr = Math.max(xr, Math.abs(this.xyz[i * 3])); zr = Math.max(zr, Math.abs(this.xyz[i * 3 + 2])); }
    this.extent = Math.max(xr, zr, 20);
    this.fit();
    const gl = this.gl;
    this.pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.pb); gl.bufferData(gl.ARRAY_BUFFER, this.xyz, gl.STATIC_DRAW);
    this.cb = gl.createBuffer();
    this.setMode(this.mode);
    this.resize();
    return head;
  }
  async loadSim(url) {
    const buf = await (await fetch(url)).arrayBuffer();
    const hl = new DataView(buf).getUint32(0, true);
    this.simHead = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const n = this.simHead.n;
    if (n !== this.n) return;
    this.fsim = new Uint8Array(buf.slice(4 + hl, 4 + hl + n));
    this.ssim = new Uint8Array(buf.slice(4 + hl + n, 4 + hl + 2 * n));
    if (this.mode === "firesim" || this.mode === "stormsim") this.setMode(this.mode);
  }
  setHighlights(list) {
    // newly found roads (orange) and power lines (magenta): dense, larger points along each polyline
    const gl = this.gl, P = [], C = [];
    for (const h of list || []) {
      const col = h.kind === "power" ? [1.0, 0.25, 0.9] : [1.0, 0.6, 0.12];
      const q = h.xyz;
      for (let i = 0; i + 1 < q.length; i++) {
        const a = q[i], b = q[i + 1], L = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]), k = Math.max(1, Math.ceil(L / 0.35));
        for (let t = 0; t < k; t++) {
          const u = t / k;
          for (const dz of h.kind === "power" ? [0, 0.25] : [0, 0.3, 0.6]) {
            P.push(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u + dz, a[2] + (b[2] - a[2]) * u); C.push(...col);
          }
        }
      }
    }
    this.hn = P.length / 3;
    if (!this.hn) { this.draw(); return; }
    this.hb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.hb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(P), gl.STATIC_DRAW);
    this.hcb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.hcb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(C), gl.STATIC_DRAW);
    this.draw();
  }
  fit() {
    const e = this.extent || 450;
    this.dist = e * 2.0; this.target = [0, (this.ymax ?? 40) * 0.25, 0];
    this.yaw = -0.7; this.pitch = 0.55;
    this.draw();
  }
  setMode(mode) {
    this.mode = mode;
    if (!this.n) return;
    const n = this.n, col = new Float32Array(n * 3);
    // height above local ground is not stored; use elevation ramp within the scene
    const ramp = (t) => [Math.min(1, 0.2 + 1.6 * t), Math.min(1, 0.3 + 1.2 * Math.sin(t * 3.0)), Math.max(0, 0.6 - t)];
    for (let i = 0; i < n; i++) {
      let r, g, b; const k = this.cls[i];
      if (mode === "rgb") { r = this.rgb[i * 3] / 255; g = this.rgb[i * 3 + 1] / 255; b = this.rgb[i * 3 + 2] / 255; const s = 1.6; r = Math.min(1, r * s); g = Math.min(1, g * s); b = Math.min(1, b * s); }
      else if (mode === "height") { [r, g, b] = ramp((this.xyz[i * 3 + 1] - this.ymin) / (this.ymax - this.ymin + 1e-6)); }
      else if (mode === "class") {
        [r, g, b] = k === 2 ? [0.62, 0.52, 0.38] : k === 9 ? [0.3, 0.55, 0.85] : k === 6 ? [0.85, 0.3, 0.3] : k === 14 ? [1, 0.9, 0.2] : k === 3 ? [0.55, 0.8, 0.4] : [0.15, 0.6, 0.3];
      } else if (mode === "hazard") {
        const h = this.haz[i] / 255;
        if (k === 2 || k === 9) [r, g, b] = [0.35, 0.35, 0.33];
        else if (k === 14) [r, g, b] = [0.3, 0.8, 1];
        else if (k === 6) [r, g, b] = [0.8, 0.8, 0.85];
        else if (h < 0.12) [r, g, b] = [0.18, 0.32, 0.23];
        else { const u = Math.min(1, (h - 0.12) / 0.6); [r, g, b] = [1, 0.85 - 0.72 * u, 0.25 - 0.2 * u]; }
      } else if (mode === "firesim") {
        const f = this.fsim ? this.fsim[i] : 0;
        if (!f) [r, g, b] = k === 2 || k === 9 ? [0.2, 0.21, 0.2] : k === 6 ? [0.75, 0.75, 0.8] : k === 14 ? [0.3, 0.8, 1] : [0.13, 0.25, 0.17];
        else { const t = (f - 1) / 254; [r, g, b] = [0.5 + 0.5 * t, 0.04 + 0.8 * t * t, 0.03 + 0.2 * t * t * t]; }   // dark red = burned first, yellow = fire front
      } else if (mode === "stormsim") {
        const v = this.ssim ? this.ssim[i] : 0;
        if (v === 255) [r, g, b] = [1, 0.18, 0.12];
        else if (k === 2 || k === 9) [r, g, b] = [0.24, 0.24, 0.22];
        else if (k === 14) [r, g, b] = [0.3, 0.8, 1];
        else if (k === 6) [r, g, b] = [0.8, 0.8, 0.85];
        else if (v > 10) { const u = Math.min(1, v / 120); [r, g, b] = [0.3 + 0.7 * u, 0.42 + 0.3 * u, 0.25 - 0.1 * u]; }
        else [r, g, b] = [0.14, 0.3, 0.19];
      } else if (mode === "fire") {
        const f = this.fire ? this.fire[i] / 255 : 0;
        if (k === 2 || k === 9) [r, g, b] = [0.36, 0.33, 0.27];
        else if (k === 14) [r, g, b] = [0.3, 0.8, 1];
        else if (k === 6) [r, g, b] = [0.8, 0.8, 0.85];
        else if (f < 0.45) [r, g, b] = [0.2, 0.42, 0.28];
        else { const u = Math.min(1, (f - 0.45) / 0.4); [r, g, b] = [0.93, 0.78 - 0.62 * u, 0.3 - 0.25 * u]; }
      } else if (mode === "health") {
        const hl = this.health[i];
        if (k === 2 || k === 9) [r, g, b] = [0.35, 0.33, 0.3];
        else [r, g, b] = hl === 2 ? [0.9, 0.25, 0.2] : hl === 1 ? [0.95, 0.7, 0.2] : [0.2, 0.55, 0.3];
      }
      col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = b;
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cb); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    this.draw();
  }
  resize() {
    const r = this.c.parentElement.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    this.c.width = r.width * dpr; this.c.height = r.height * dpr;
    this.c.style.width = r.width + "px"; this.c.style.height = r.height + "px";
    this.draw();
  }
  draw() {
    if (!this.n || this.failed) return;
    const gl = this.gl;
    gl.viewport(0, 0, this.c.width, this.c.height);
    gl.clearColor(0.04, 0.06, 0.055, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const e = [this.target[0] + this.dist * Math.cos(this.pitch) * Math.sin(this.yaw), this.target[1] + this.dist * Math.sin(this.pitch),
      this.target[2] + this.dist * Math.cos(this.pitch) * Math.cos(this.yaw)];
    const P = mat4Perspective(0.8, this.c.width / this.c.height, Math.max(0.2, this.dist / 2000), Math.max(5000, this.dist * 6));
    const V = mat4LookAt(e, this.target, [0, 1, 0]);
    gl.uniformMatrix4fv(this.loc.m, false, mat4Mul(P, V));
    gl.uniform1f(this.loc.s, 900.0 * ((this.extent || 450) / 450) * (window.devicePixelRatio || 1));   // point size scales with the scene
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pb); gl.enableVertexAttribArray(this.loc.p); gl.vertexAttribPointer(this.loc.p, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cb); gl.enableVertexAttribArray(this.loc.c); gl.vertexAttribPointer(this.loc.c, 3, gl.FLOAT, false, 0, 0);
    gl.uniform1f(this.loc.mx, 6.0);
    gl.drawArrays(gl.POINTS, 0, this.n);
    if (this.hn) {                                  // highlights on top: newly found roads / power lines
      gl.uniform1f(this.loc.s, 2600.0 * ((this.extent || 450) / 450) * (window.devicePixelRatio || 1));
      gl.uniform1f(this.loc.mx, 9.0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.hb); gl.vertexAttribPointer(this.loc.p, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.hcb); gl.vertexAttribPointer(this.loc.c, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.POINTS, 0, this.hn);
    }
  }
  _bind() {
    const c = this.c; let d = null;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => { d = [e.clientX, e.clientY, e.button === 2 || e.shiftKey]; c.setPointerCapture(e.pointerId); });
    c.addEventListener("pointerup", () => (d = null));
    c.addEventListener("pointermove", (e) => {
      if (!d) return;
      const dx = e.clientX - d[0], dy = e.clientY - d[1]; d[0] = e.clientX; d[1] = e.clientY;
      if (d[2]) {
        const s = this.dist / 700;
        this.target[0] -= (Math.cos(this.yaw) * dx) * s; this.target[2] += (Math.sin(this.yaw) * dx) * s;
        this.target[0] -= (Math.sin(this.yaw) * dy) * s; this.target[2] -= (Math.cos(this.yaw) * dy) * s;
      } else {
        this.yaw -= dx * 0.005; this.pitch = Math.max(0.05, Math.min(1.5, this.pitch + dy * 0.005));
      }
      this.draw();
    });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.dist = Math.max(30, Math.min(3000, this.dist * (e.deltaY > 0 ? 1.12 : 0.89))); this.draw(); }, { passive: false });
  }
}
window.PointViewer = PointViewer;
