"""Read any 3D forest scan into Horus's point dict (same keys as lasio.read_las).

Supported:
  .las / .laz          LiDAR (DJI Terra, NLS laser scanning, ...)          -> lasio
  .ply                 point cloud or mesh, ASCII or binary (photogrammetry, phone LiDAR apps, CloudCompare)
  .obj                 mesh (photogrammetry / phone 3D scan exports); "v x y z [r g b]" colours supported
  .glb / .gltf         glTF 3D model (phone scanning apps, DJI / Pix4D web exports); .gltf with embedded or
                       side-by-side .bin buffers
  .stl                 mesh, ASCII or binary (3D-printing / CAD exports of a scanned model)
  .xyz .txt .csv .pts  text point lists: x y z [intensity] [r g b]
  .e57                 laser-scanner exchange format (needs: pip install pye57)

Meshes are turned into points by sampling their surfaces evenly (area-weighted), so a 3D model of a
forest behaves like a LiDAR first-return cloud. Models whose up axis is Y (glTF always; OBJ/PLY when
the vertical axis is clearly Y) are rotated to Z-up. Coordinates are kept as they are: georeferenced
files (TM35FIN) stay georeferenced, local models are placed on the map by Horus.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

SCAN_EXT = (".las", ".laz", ".ply", ".obj", ".glb", ".gltf", ".stl", ".xyz", ".txt", ".csv", ".pts", ".e57")
MESH_POINTS_PER_M2 = 25.0


def read_points(path, max_points: int | None = None) -> dict:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".las", ".laz"):
        from .lasio import read_las
        return read_las(path, max_points=max_points)
    if ext == ".ply":
        P = _read_ply(path)
    elif ext == ".obj":
        P = _read_obj(path)
    elif ext == ".glb":
        P = _read_glb(path)
    elif ext == ".gltf":
        P = _read_gltf(path)
    elif ext == ".stl":
        P = _read_stl(path)
    elif ext in (".xyz", ".txt", ".csv", ".pts"):
        P = _read_text(path)
    elif ext == ".e57":
        P = _read_e57(path)
    else:
        raise ValueError(f"{path.name}: unsupported 3D format {ext} (use {', '.join(SCAN_EXT)})")
    return _finish(P, path, max_points)


# ------------------------------------------------------------------------------------------- common
def _finish(P: dict, path: Path, max_points):
    xyz = np.asarray(P["xyz"], float)
    tris = P.get("tris")
    rgb = P.get("rgb")
    up = P.get("up")
    if up is None:
        up = _guess_up(xyz)
    if up == "y":                                   # Y-up model -> Z-up (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], 1)
    kind = "points"
    if tris is not None and len(tris):
        xyz, rgb = _sample_mesh(xyz, np.asarray(tris, np.int64), rgb, max_points or 4_000_000)
        kind = "mesh"
    ok = np.isfinite(xyz).all(1)
    xyz = xyz[ok]
    rgb = rgb[ok] if rgb is not None and len(rgb) == len(ok) else (None if rgb is None else rgb[: ok.sum()])
    inten = P.get("intensity")
    inten = inten[ok] if inten is not None and len(inten) == len(ok) else None
    n0 = xyz.shape[0]
    if n0 == 0:
        raise ValueError(f"{path.name}: no 3D points found")
    step = int(np.ceil(n0 / max_points)) if max_points and n0 > max_points else 1
    sel = slice(None, None, step)
    n = xyz[sel].shape[0]
    out = dict(x=xyz[sel, 0].copy(), y=xyz[sel, 1].copy(), z=xyz[sel, 2].copy(),
               classification=np.ones(n, np.uint8),
               intensity=(np.clip(inten[sel], 0, 65535).astype(np.uint16) if inten is not None else np.zeros(n, np.uint16)),
               return_number=np.ones(n, np.uint8), number_of_returns=np.ones(n, np.uint8))
    if rgb is not None:
        c = np.asarray(rgb[sel], float)
        if c.max() <= 1.0 + 1e-6:
            c = c * 255
        if c.max() <= 255.5:
            c = c * 256
        c = np.clip(c, 0, 65535).astype(np.uint16)
        out["red"], out["green"], out["blue"] = c[:, 0], c[:, 1], c[:, 2]
    mn, mx = xyz.min(0), xyz.max(0)
    out["header"] = {"version": ext_label(path), "point_format": -1, "points": int(n), "points_in_file": int(n0),
                     "decimation": step, "bounds": [*map(float, mn), *map(float, mx)], "wkt": P.get("wkt", ""),
                     "source_kind": kind, "up_axis": up}
    return out


def ext_label(path):
    return path.suffix.lower().lstrip(".").upper()


def _guess_up(xyz):
    """Up axis = the clearly shortest extent (forests are wider than tall); default Z."""
    if len(xyz) < 10:
        return "z"
    r = np.percentile(xyz, 99, 0) - np.percentile(xyz, 1, 0)
    # only clearly "lying" models: Y (height) far smaller than both horizontal extents, Z not flat.
    # Long narrow corridor scans (along a power line) stay Z-up.
    if r[1] < 0.35 * min(r[0], r[2]) and r[2] > 3 * r[1]:
        return "y"
    return "z"


def _sample_mesh(v, f, vcol, max_points):
    f = f[(f >= 0).all(1) & (f < len(v)).all(1)]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = float(area.sum())
    if tot <= 0:
        return v, vcol
    n = int(min(max_points, max(len(v), tot * MESH_POINTS_PER_M2)))
    rng = np.random.default_rng(0)
    k = rng.choice(len(f), size=n, p=area / tot)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    pts = a[k] * w0[:, None] + b[k] * w1[:, None] + c[k] * w2[:, None]
    col = None
    if vcol is not None and len(vcol) == len(v):
        vc = np.asarray(vcol, float)
        col = vc[f[k, 0]] * w0[:, None] + vc[f[k, 1]] * w1[:, None] + vc[f[k, 2]] * w2[:, None]
    return pts, col


# ------------------------------------------------------------------------------------------- PLY
_PLY_T = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2",
          "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4",
          "double": "f8", "float64": "f8"}


def _read_ply(path):
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path.name} is not a PLY file")
        fmt = None
        elements = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("PLY header not terminated")
            t = line.decode("ascii", "replace").split()
            if not t:
                continue
            if t[0] == "format":
                fmt = t[1]
            elif t[0] == "element":
                elements.append(dict(name=t[1], n=int(t[2]), props=[]))
            elif t[0] == "property":
                if t[1] == "list":
                    elements[-1]["props"].append((t[4], "list", _PLY_T[t[2]], _PLY_T[t[3]]))
                else:
                    elements[-1]["props"].append((t[2], _PLY_T[t[1]]))
            elif t[0] == "end_header":
                break
        data = fh.read()
    vert = faces = None
    if fmt == "ascii":
        rows = data.decode("ascii", "replace").split("\n")
        pos = 0
        for el in elements:
            if el["name"] == "vertex":
                names = [p[0] for p in el["props"]]
                arr = np.loadtxt(rows[pos:pos + el["n"]], ndmin=2)
                vert = {nm: arr[:, i] for i, nm in enumerate(names) if i < arr.shape[1]}
            elif el["name"] == "face":
                fl = [list(map(int, r.split()))[1:] for r in rows[pos:pos + el["n"]] if r.strip()]
                faces = _triangulate(fl)
            pos += el["n"]
    else:
        end = "<" if fmt == "binary_little_endian" else ">"
        off = 0
        for el in elements:
            if all(len(p) == 2 for p in el["props"]):
                dt = np.dtype([(p[0], end + p[1]) for p in el["props"]])
                arr = np.frombuffer(data, dtype=dt, count=el["n"], offset=off)
                off += dt.itemsize * el["n"]
                if el["name"] == "vertex":
                    vert = {nm: arr[nm].astype(float) for nm in dt.names}
            else:                                   # element with list properties (faces): walk it
                fl = []
                for _ in range(el["n"]):
                    row = []
                    for p in el["props"]:
                        if len(p) == 2:
                            off += np.dtype(p[1]).itemsize
                        else:
                            cnt_t, idx_t = np.dtype(end + p[2]), np.dtype(end + p[3])
                            cnt = int(np.frombuffer(data, cnt_t, 1, off)[0])
                            off += cnt_t.itemsize
                            row = np.frombuffer(data, idx_t, cnt, off).tolist()
                            off += idx_t.itemsize * cnt
                    fl.append(row)
                if el["name"] == "face":
                    faces = _triangulate(fl)
    if vert is None:
        raise ValueError(f"{path.name}: PLY has no vertex element")
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1)
    rgb = None
    for r, g, b in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if r in vert:
            rgb = np.stack([vert[r], vert[g], vert[b]], 1)
            break
    inten = next((vert[k] for k in ("intensity", "scalar_intensity", "scalar_Intensity") if k in vert), None)
    if inten is not None and inten.max() <= 1.0:
        inten = inten * 65535
    return dict(xyz=xyz, rgb=rgb, intensity=inten, tris=faces)


def _triangulate(fl):
    tris = []
    for f in fl:
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    return np.array(tris, np.int64) if tris else None


# ------------------------------------------------------------------------------------------- OBJ
def _read_obj(path):
    v, col, faces = [], [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                t = line.split()
                v.append(list(map(float, t[1:4])))
                if len(t) >= 7:
                    col.append(list(map(float, t[4:7])))
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) for p in line.split()[1:]]
                n = len(v)
                idx = [i - 1 if i > 0 else n + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    if not v:
        raise ValueError(f"{path.name}: OBJ has no vertices")
    xyz = np.array(v, float)
    rgb = np.array(col, float) if len(col) == len(v) else None
    return dict(xyz=xyz, rgb=rgb, tris=np.array(faces, np.int64) if faces else None)


# ------------------------------------------------------------------------------------------- GLB
_GL = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_glb(path):
    b = Path(path).read_bytes()
    if b[:4] != b"glTF":
        raise ValueError(f"{path.name} is not a binary glTF (.glb)")
    off = 12
    js, binc = None, b""
    while off < len(b):
        ln, typ = struct.unpack_from("<II", b, off)
        chunk = b[off + 8:off + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk)
        elif typ == 0x004E4942:
            binc = chunk
        off += 8 + ln
    if js is None:
        raise ValueError("glb without JSON chunk")
    return _gltf_meshes(js, [binc], path)


def _read_gltf(path):
    import base64
    js = json.loads(Path(path).read_text(errors="replace"))
    bufs = []
    for b in js.get("buffers", []):
        uri = b.get("uri", "")
        if uri.startswith("data:"):
            bufs.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            f = Path(path).parent / uri
            if not f.exists():
                raise ValueError(f"{path.name} needs its buffer file {uri}: upload it together with the .gltf "
                                 "(or export the model as a single .glb)")
            bufs.append(f.read_bytes())
    return _gltf_meshes(js, bufs, path)


def _gltf_meshes(js, bufs, path):
    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        binc = bufs[bv.get("buffer", 0)]
        dt = np.dtype("<" + _GL[a["componentType"]])
        nc = _NC[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0)
        if stride and stride != dt.itemsize * nc:
            raw = np.frombuffer(binc, np.uint8, stride * (a["count"] - 1) + dt.itemsize * nc, start)
            rows = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * nc), (stride, 1))
            arr = np.ascontiguousarray(rows).view(dt).reshape(a["count"], nc)
        else:
            arr = np.frombuffer(binc, dt, a["count"] * nc, start).reshape(a["count"], nc)
        if a.get("normalized") and dt.kind in "iu":
            arr = arr / float(np.iinfo(dt).max)
        return arr.astype(float)

    # node transforms (translation / scale / rotation / matrix), applied recursively
    def node_mat(n):
        if "matrix" in n:
            return np.array(n["matrix"], float).reshape(4, 4).T
        M = np.eye(4)
        if "scale" in n:
            M = np.diag([*n["scale"], 1.0]) @ M
        if "rotation" in n:
            x, y, z, w = n["rotation"]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            Rm = np.eye(4)
            Rm[:3, :3] = R
            M = Rm @ M
        if "translation" in n:
            T = np.eye(4)
            T[:3, 3] = n["translation"]
            M = T @ M
        return M

    xyz, rgb, tris = [], [], []
    nodes = js.get("nodes", [])
    roots = js["scenes"][js.get("scene", 0)]["nodes"] if js.get("scenes") else list(range(len(nodes)))

    def walk(i, parent):
        n = nodes[i]
        M = parent @ node_mat(n)
        if "mesh" in n:
            for prim in js["meshes"][n["mesh"]]["primitives"]:
                at = prim["attributes"]
                if "POSITION" not in at:
                    continue
                p = acc(at["POSITION"])
                p = (np.c_[p, np.ones(len(p))] @ M.T)[:, :3]
                base = sum(len(q) for q in xyz)
                xyz.append(p)
                c = acc(at["COLOR_0"])[:, :3] if "COLOR_0" in at else None
                rgb.append(c if c is not None else np.full((len(p), 3), np.nan))
                mode = prim.get("mode", 4)
                if "indices" in prim and mode == 4:
                    tris.append(acc(prim["indices"]).astype(np.int64).reshape(-1, 3) + base)
                elif mode == 4:
                    tris.append(np.arange(len(p)).reshape(-1, 3) + base)
        for ch in n.get("children", []):
            walk(ch, M)
    for r in roots:
        walk(r, np.eye(4))
    if not xyz:
        raise ValueError(f"{path.name}: no meshes / points in the 3D model")
    xyz = np.vstack(xyz)
    rgb = np.vstack(rgb)
    rgb = None if np.isnan(rgb).all() else np.nan_to_num(rgb, nan=0.5)
    return dict(xyz=xyz, rgb=rgb, tris=np.vstack(tris) if tris else None, up="y")


# ------------------------------------------------------------------------------------------- text
def _read_text(path):
    head = Path(path).read_text(errors="replace")[:4000].splitlines()
    delim = "," if sum(line.count(",") for line in head[:20]) > len(head[:20]) else None
    skip = 0
    for line in head:
        t = re.split(r"[,\s;]+", line.strip())
        try:
            [float(x) for x in t[:3]]
            break
        except ValueError:
            skip += 1
    if path.suffix.lower() == ".pts" and skip == 0 and len(re.split(r"\s+", head[0].strip())) == 1:
        skip = 1                                   # Leica .pts: first line = point count
    arr = np.loadtxt(path, delimiter=delim, skiprows=skip, ndmin=2, comments="#")
    if arr.shape[1] < 3:
        raise ValueError(f"{path.name}: need at least x y z columns")
    rgb = inten = None
    nc = arr.shape[1]
    if nc >= 7:                                    # x y z I r g b  (PTS / CloudCompare)
        inten, rgb = arr[:, 3], arr[:, 4:7]
    elif nc == 6:                                  # x y z r g b
        rgb = arr[:, 3:6]
    elif nc in (4, 5):                             # x y z I
        inten = arr[:, 3]
    if inten is not None and np.nanmin(inten) < 0:  # Leica PTS intensity -2048..2047
        inten = (inten + 2048) * 16
    return dict(xyz=arr[:, :3], rgb=rgb, intensity=inten)


# ------------------------------------------------------------------------------------------- E57
def _read_e57(path):
    try:
        import pye57  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading .e57 needs: pip install pye57  (or export the scan as .las/.ply)") from exc
    e = pye57.E57(str(path))
    xs, cs, ins = [], [], []
    for i in range(e.scan_count):
        d = e.read_scan(i, ignore_missing_fields=True, colors=True, intensity=True)
        xs.append(np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1))
        if "colorRed" in d:
            cs.append(np.stack([d["colorRed"], d["colorGreen"], d["colorBlue"]], 1))
        if "intensity" in d:
            ins.append(d["intensity"])
    return dict(xyz=np.vstack(xs), rgb=np.vstack(cs) if len(cs) == len(xs) else None,
                intensity=(np.concatenate(ins) * (65535 if np.concatenate(ins).max() <= 1 else 1)) if len(ins) == len(xs) else None,
                up="z")


# ------------------------------------------------------------------------------------------- STL
def _read_stl(path):
    b = Path(path).read_bytes()
    if len(b) >= 84:
        n = struct.unpack_from("<I", b, 80)[0]
        if 84 + n * 50 == len(b):                      # binary STL
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tri = np.frombuffer(b, dt, n, 84)["v"].astype(float)
            v = tri.reshape(-1, 3)
            return dict(xyz=v, tris=np.arange(len(v)).reshape(-1, 3))
    nums = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", b)
    if not nums:
        raise ValueError(f"{path.name}: no triangles found in STL")
    v = np.array(nums, dtype=float)
    v = v[: len(v) // 3 * 3]
    return dict(xyz=v, tris=np.arange(len(v)).reshape(-1, 3))
