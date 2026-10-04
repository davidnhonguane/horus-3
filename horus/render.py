"""Raster -> PNG rendering (north-up) with small built-in colour maps."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _ramp(stops):
    stops = np.array(stops, dtype=float)
    pos, col = stops[:, 0], stops[:, 1:]

    def f(v):
        v = np.clip(v, 0, 1)
        out = np.empty(v.shape + (col.shape[1],))
        for c in range(col.shape[1]):
            out[..., c] = np.interp(v, pos, col[:, c])
        return out
    return f


RAMPS = {
    "chm": _ramp([(0, 235, 240, 220, 0), (0.07, 200, 225, 170, 150), (0.4, 80, 160, 70, 220),
                  (0.75, 25, 95, 40, 235), (1, 10, 50, 25, 245)]),
    "risk": _ramp([(0, 255, 255, 190, 0), (0.12, 255, 240, 140, 120), (0.35, 253, 190, 60, 185),
                   (0.6, 240, 110, 30, 215), (0.8, 205, 30, 30, 230), (1, 110, 0, 40, 240)]),
    "time": _ramp([(0, 40, 200, 120, 200), (0.35, 250, 220, 60, 200), (0.7, 230, 100, 40, 210),
                   (1, 120, 30, 120, 220)]),
    "fire": _ramp([(0, 255, 245, 120, 235), (0.25, 255, 170, 30, 225), (0.55, 230, 60, 20, 215),
                   (1, 90, 10, 30, 160)]),
    "ndvi": _ramp([(0, 150, 60, 40, 220), (0.35, 230, 200, 80, 220), (0.6, 120, 190, 80, 220),
                   (1, 20, 110, 40, 220)]),
}

FUEL_COLORS = {0: (120, 160, 200, 120), 1: (230, 215, 140, 170), 2: (30, 90, 50, 200), 3: (120, 140, 40, 200),
               4: (60, 160, 70, 200), 5: (170, 210, 90, 200), 6: (90, 140, 60, 200)}


def to_png(rgba: np.ndarray) -> bytes:
    img = Image.fromarray(np.flipud(rgba.astype(np.uint8)), "RGBA")
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def hillshade(dtm, res, az=315, alt=40, chm=None, water=None):
    gy, gx = np.gradient(dtm, res)
    slope = np.arctan(np.hypot(gx, gy) * 2.0)
    aspect = np.arctan2(-gx, gy)
    azr, altr = np.radians(360 - az + 90), np.radians(alt)
    hs = np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect)
    hs = np.clip(hs, 0, 1)
    base = np.stack([hs * 120 + 110, hs * 115 + 112, hs * 100 + 105], -1)
    if chm is not None:
        c = np.clip(chm / 28.0, 0, 1)[..., None]
        green = np.stack([hs * 50 + 30, hs * 90 + 70, hs * 50 + 35], -1)
        base = base * (1 - 0.85 * np.sqrt(c)) + green * 0.85 * np.sqrt(c)
    if water is not None:
        base[water] = (95, 140, 175)
    a = np.full(dtm.shape + (1,), 255)
    return np.concatenate([base, a], -1)


def scalar(v, ramp, vmin, vmax, mask=None):
    rgba = RAMPS[ramp]((v - vmin) / max(vmax - vmin, 1e-9))
    if mask is not None:
        rgba[~mask] = 0
    return rgba


def categorical(v, colors, mask=None):
    out = np.zeros(v.shape + (4,))
    for k, c in colors.items():
        out[v == k] = c
    if mask is not None:
        out[~mask] = 0
    return out


def upsample(a, f):
    return np.repeat(np.repeat(a, f, 0), f, 1)
