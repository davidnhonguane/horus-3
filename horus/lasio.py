"""Minimal, dependency-free LAS reader / writer (ASPRS LAS 1.2 - 1.4).

DJI Terra exports L2/L3 point clouds as LAS (point formats 2/3/7/8 typically).
Compressed LAZ is read through ``laspy`` when it is installed.

Returned dict keys: x, y, z (float64), classification (uint8), intensity (uint16),
return_number, number_of_returns (uint8), red, green, blue (uint16, optional),
nir (uint16, optional), plus 'header'.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_RGB_OFFSET = {2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30}
_NIR_OFFSET = {8: 36, 10: 36}


def read_las(path: str | Path, max_points: int | None = None) -> dict:
    path = Path(path)
    with open(path, "rb") as fh:
        raw = fh.read(375)
        if raw[:4] != b"LASF":
            raise ValueError(f"{path.name} is not a LAS/LAZ file")
        if path.suffix.lower() == ".laz" or raw[104] & 0x80:
            return _read_laz(path, max_points)
        vmaj, vmin = raw[24], raw[25]
        header_size = struct.unpack_from("<H", raw, 94)[0]
        offset_to_points = struct.unpack_from("<I", raw, 96)[0]
        fmt = raw[104] & 0x3F  # bits 6/7 flag compression in some writers
        rec_len = struct.unpack_from("<H", raw, 105)[0]
        n_legacy = struct.unpack_from("<I", raw, 107)[0]
        scale = struct.unpack_from("<3d", raw, 131)
        offset = struct.unpack_from("<3d", raw, 155)
        maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", raw, 179)
        n = n_legacy
        if vmin >= 4 and header_size >= 375:
            n = struct.unpack_from("<Q", raw, 247)[0] or n_legacy
        wkt = _wkt(fh, struct.unpack_from("<H", raw, 94)[0], struct.unpack_from("<I", raw, 100)[0])
    n_file = n
    n = min(n, (path.stat().st_size - offset_to_points) // rec_len)
    step = _stride(n, max_points)
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=offset_to_points, shape=(n, rec_len))
    rec = np.ascontiguousarray(mm[::step])       # even thinning keeps the whole area covered
    del mm
    n = rec.shape[0]

    def field(off, dt):
        return rec[:, off:off + np.dtype(dt).itemsize].copy().view(dt).ravel()

    X, Y, Z = field(0, "<i4"), field(4, "<i4"), field(8, "<i4")
    out = {
        "x": X * scale[0] + offset[0],
        "y": Y * scale[1] + offset[1],
        "z": Z * scale[2] + offset[2],
        "intensity": field(12, "<u2"),
    }
    if fmt <= 5:
        b = rec[:, 14]
        out["return_number"] = (b & 0x07).astype(np.uint8)
        out["number_of_returns"] = ((b >> 3) & 0x07).astype(np.uint8)
        out["classification"] = (rec[:, 15] & 0x1F).astype(np.uint8)
    else:
        b = rec[:, 14]
        out["return_number"] = (b & 0x0F).astype(np.uint8)
        out["number_of_returns"] = ((b >> 4) & 0x0F).astype(np.uint8)
        out["classification"] = rec[:, 16].astype(np.uint8)
    if fmt in _RGB_OFFSET:
        o = _RGB_OFFSET[fmt]
        out["red"], out["green"], out["blue"] = field(o, "<u2"), field(o + 2, "<u2"), field(o + 4, "<u2")
    if fmt in _NIR_OFFSET:
        out["nir"] = field(_NIR_OFFSET[fmt], "<u2")
    out["header"] = {
        "version": f"{vmaj}.{vmin}", "point_format": fmt, "points": int(n), "points_in_file": int(n_file),
        "decimation": step, "bounds": [minx, miny, minz, maxx, maxy, maxz], "wkt": wkt,
    }
    return out


def _stride(n, max_points):
    return int(np.ceil(n / max_points)) if max_points and n > max_points else 1


def _wkt(fh, header_size, n_vlr) -> str:
    """Coordinate system WKT from the LAS VLRs (record 2112), '' if none."""
    try:
        fh.seek(header_size)
        for _ in range(n_vlr):
            h = fh.read(54)
            rid, ln = struct.unpack_from("<HH", h, 18)
            body = fh.read(ln)
            if rid == 2112:
                return body.split(b"\0")[0].decode("utf-8", "replace")
    except (struct.error, OSError):
        pass
    return ""


def is_local_crs(wkt: str) -> bool:
    """True for engineering / local coordinate systems with no geodetic reference."""
    return bool(wkt) and ("LOCAL_CS" in wkt.upper() or "ENGCRS" in wkt.upper() or "NO GEODETIC" in wkt.upper())


def _read_laz(path: Path, max_points):
    """LAZ via laspy (pip install "laspy[lazrs]"), streamed in chunks and evenly thinned to max_points."""
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError('Reading .laz needs: pip install "laspy[lazrs]"  (or upload an uncompressed .las)') from exc
    with laspy.open(str(path)) as rd:
        hdr = rd.header
        n_file = int(hdr.point_count)
        step = _stride(n_file, max_points)
        dims = set(hdr.point_format.dimension_names)
        keys = ["x", "y", "z", "intensity", "classification", "return_number", "number_of_returns"]
        keys += [k for k in ("red", "green", "blue", "nir") if k in dims]
        cols = {k: [] for k in keys}
        seen = 0
        for ch in rd.chunk_iterator(2_000_000):
            idx = np.arange((-seen) % step, len(ch), step)
            seen += len(ch)
            for k in keys:
                cols[k].append(np.asarray(getattr(ch, k))[idx])
        wkt = ""
        for v in list(getattr(hdr, "vlrs", [])):
            if getattr(v, "record_id", None) == 2112:
                wkt = getattr(v, "string", "") or ""
    out = {k: np.concatenate(v) for k, v in cols.items()}
    for k in ("x", "y", "z"):
        out[k] = out[k].astype(float)
    for k in ("classification", "return_number", "number_of_returns"):
        out[k] = out[k].astype(np.uint8)
    out["header"] = {"version": str(hdr.version), "point_format": hdr.point_format.id, "points": int(out["x"].shape[0]),
                     "points_in_file": n_file, "decimation": step, "bounds": list(hdr.mins) + list(hdr.maxs), "wkt": wkt}
    return out


def write_las(path: str | Path, pts: dict, scale: float = 0.001) -> None:
    """Write LAS 1.4, point data record format 8 (XYZ, RGB, NIR) - what a fused
    DJI L3 + multispectral product looks like."""
    x, y, z = (np.asarray(pts[k], dtype=float) for k in ("x", "y", "z"))
    n = x.shape[0]
    offs = (np.floor(x.min()), np.floor(y.min()), np.floor(z.min()))
    rec_len = 38
    rec = np.zeros((n, rec_len), dtype=np.uint8)

    def put(off, arr, dt):
        rec[:, off:off + np.dtype(dt).itemsize] = np.ascontiguousarray(arr.astype(dt)).view(np.uint8).reshape(n, -1)

    put(0, np.round((x - offs[0]) / scale), "<i4")
    put(4, np.round((y - offs[1]) / scale), "<i4")
    put(8, np.round((z - offs[2]) / scale), "<i4")
    put(12, np.asarray(pts.get("intensity", np.zeros(n))), "<u2")
    rn = np.asarray(pts.get("return_number", np.ones(n)), dtype=np.uint8)
    nr = np.asarray(pts.get("number_of_returns", np.ones(n)), dtype=np.uint8)
    rec[:, 14] = (rn & 0x0F) | ((nr & 0x0F) << 4)
    rec[:, 16] = np.asarray(pts.get("classification", np.ones(n)), dtype=np.uint8)
    for i, k in enumerate(("red", "green", "blue", "nir")):
        put(30 + 2 * i, np.asarray(pts.get(k, np.zeros(n))), "<u2")

    hdr = bytearray(375)
    hdr[0:4] = b"LASF"
    hdr[24], hdr[25] = 1, 4
    sysid = b"HORUS".ljust(32, b"\0")
    gen = b"horus synthetic DJI-L3 like".ljust(32, b"\0")[:32]
    hdr[26:58] = sysid
    hdr[58:90] = gen
    struct.pack_into("<H", hdr, 94, 375)
    struct.pack_into("<I", hdr, 96, 375)
    struct.pack_into("<I", hdr, 100, 0)
    hdr[104] = 8
    struct.pack_into("<H", hdr, 105, rec_len)
    struct.pack_into("<I", hdr, 107, 0)
    struct.pack_into("<3d", hdr, 131, scale, scale, scale)
    struct.pack_into("<3d", hdr, 155, *offs)
    struct.pack_into("<6d", hdr, 179, x.max(), x.min(), y.max(), y.min(), z.max(), z.min())
    struct.pack_into("<Q", hdr, 247, n)
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(rec.tobytes())
