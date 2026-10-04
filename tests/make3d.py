"""Writers for test 3D files (PLY / OBJ / GLB / XYZ / CSV) made from a synthetic forest, so every reader
in horus.scan3d can be tested without external tools."""
import json
import struct

import numpy as np


def write_ply(path, P, binary=True, n=None):
    k = slice(None, n)
    x, y, z = P["x"][k], P["y"][k], P["z"][k]
    r, g, b = (P[c][k] // 256 for c in ("red", "green", "blue")) if "red" in P else (np.full(x.size, 128),) * 3
    hdr = (f"ply\nformat {'binary_little_endian' if binary else 'ascii'} 1.0\nelement vertex {x.size}\n"
           "property double x\nproperty double y\nproperty double z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(hdr.encode())
        if binary:
            dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            a = np.zeros(x.size, dt)
            a["x"], a["y"], a["z"], a["r"], a["g"], a["b"] = x, y, z, r, g, b
            fh.write(a.tobytes())
        else:
            fh.write("\n".join(f"{a:.3f} {b_:.3f} {c:.3f} {d} {e} {f}" for a, b_, c, d, e, f in zip(x, y, z, r, g, b)).encode())


def write_xyz(path, P, header=False, sep=" "):
    a = np.stack([P["x"], P["y"], P["z"], P["intensity"].astype(float)], 1)
    np.savetxt(path, a, fmt="%.3f", delimiter=sep, header="x,y,z,intensity" if header else "", comments="")


def surface_mesh(P, cell=1.0):
    """Top-surface mesh (like a photogrammetry model): max z per cell, two triangles per cell."""
    x0, y0 = P["x"].min(), P["y"].min()
    ix = ((P["x"] - x0) / cell).astype(int)
    iy = ((P["y"] - y0) / cell).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    zt = np.full((ny, nx), -np.inf)
    np.maximum.at(zt, (iy, ix), P["z"])
    from scipy import ndimage as ndi
    bad = ~np.isfinite(zt)
    if bad.any():
        _, (jy, jx) = ndi.distance_transform_edt(bad, return_indices=True)
        zt = zt[jy, jx]
    yy, xx = np.mgrid[0:ny, 0:nx]
    v = np.stack([x0 + (xx + 0.5) * cell, y0 + (yy + 0.5) * cell, zt], -1).reshape(-1, 3)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return v, f


def write_obj(path, v, f, y_up=False):
    if y_up:
        v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    with open(path, "w") as fh:
        fh.write("# horus test mesh\n")
        fh.write("".join(f"v {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in v))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f))


def write_glb(path, v, f):
    """glTF 2.0 binary, Y-up as the spec requires, with a node translation."""
    vy = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1).astype(np.float32)
    off = vy.mean(0)
    vy = vy - off
    pos = vy.tobytes()
    ind = f.astype(np.uint32).ravel().tobytes()
    binc = pos + ind
    binc += b"\0" * ((4 - len(binc) % 4) % 4)
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0, "translation": [float(c) for c in off]}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
          "buffers": [{"byteLength": len(binc)}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
                          {"buffer": 0, "byteOffset": len(pos), "byteLength": len(ind)}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vy), "type": "VEC3",
                         "min": vy.min(0).tolist(), "max": vy.max(0).tolist()},
                        {"bufferView": 1, "componentType": 5125, "count": f.size, "type": "SCALAR"}]}
    jb = json.dumps(js).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binc)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(jb), b"JSON") + jb)
        fh.write(struct.pack("<I4s", len(binc), b"BIN\0") + binc)
