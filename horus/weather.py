"""Weather: live data from Open-Meteo (free, no API key) with offline scenarios,
plus the Canadian Forest Fire Weather Index (FWI) system (Van Wagner 1987), which
the Finnish Meteorological Institute also uses alongside its own forest-fire index.
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, asdict

import numpy as np

# day-length adjustment factors (N hemisphere, > 30N)
DMC_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
DC_LF = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


def ffmc_step(ffmc0, T, H, W, rain):
    mo = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)
    if rain > 0.5:
        rf = rain - 0.5
        inc = 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
        if mo > 150:
            inc += 0.0015 * (mo - 150) ** 2 * math.sqrt(rf)
        mo = min(mo + inc, 250)
    ed = 0.942 * H ** 0.679 + 11 * math.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - math.exp(-0.115 * H))
    if mo > ed:
        ko = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * math.sqrt(W) * (1 - (H / 100) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * T)
        m = ed + (mo - ed) * 10 ** (-kd)
    else:
        ew = 0.618 * H ** 0.753 + 10 * math.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - math.exp(-0.115 * H))
        if mo < ew:
            k1 = 0.424 * (1 - ((100 - H) / 100) ** 1.7) + 0.0694 * math.sqrt(W) * (1 - ((100 - H) / 100) ** 8)
            kw = k1 * 0.581 * math.exp(0.0365 * T)
            m = ew - (ew - mo) * 10 ** (-kw)
        else:
            m = mo
    return min(max(59.5 * (250 - m) / (147.2 + m), 0.0), 101.0)


def dmc_step(dmc0, T, H, rain, month):
    T = max(T, -1.1)
    rk = 1.894 * (T + 1.1) * (100 - H) * DMC_LE[month - 1] * 1e-4
    if rain > 1.5:
        rw = 0.92 * rain - 1.27
        wmi = 20 + math.exp(5.6348 - dmc0 / 43.43)
        if dmc0 <= 33:
            b = 100 / (0.5 + 0.3 * dmc0)
        elif dmc0 <= 65:
            b = 14 - 1.3 * math.log(dmc0)
        else:
            b = 6.2 * math.log(dmc0) - 17.2
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        pr = max(244.72 - 43.43 * math.log(wmr - 20), 0.0)
    else:
        pr = dmc0
    return max(pr + rk, 0.0)


def dc_step(dc0, T, rain, month):
    T = max(T, -2.8)
    pe = max((0.36 * (T + 2.8) + DC_LF[month - 1]) / 2, 0.0)
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        Qo = 800 * math.exp(-dc0 / 400)
        Qr = Qo + 3.937 * rd
        Dr = max(400 * math.log(800 / Qr), 0.0)
        return Dr + pe
    return dc0 + pe


def isi_calc(ffmc, W):
    fm = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    sf = 19.115 * math.exp(-0.1386 * fm) * (1 + fm ** 5.31 / 4.93e7)
    return sf * math.exp(0.05039 * W)


def bui_calc(dmc, dc):
    if dmc <= 0:
        return 0.0
    if dmc <= 0.4 * dc:
        b = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        b = dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(b, 0.0)


def fwi_calc(isi, bui):
    fD = 0.626 * bui ** 0.809 + 2 if bui <= 80 else 1000 / (25 + 108.64 * math.exp(-0.023 * bui))
    B = 0.1 * isi * fD
    return math.exp(2.72 * (0.434 * math.log(B)) ** 0.647) if B > 1 else B


def fwi_series(days):
    """days: list of dict(date 'YYYY-MM-DD', T, RH, W (km/h), rain (mm, 24 h)) -> list of codes."""
    ffmc, dmc, dc = 85.0, 6.0, 15.0
    out = []
    for d in days:
        month = int(d["date"][5:7])
        ffmc = ffmc_step(ffmc, d["T"], d["RH"], d["W"], d["rain"])
        dmc = dmc_step(dmc, d["T"], d["RH"], d["rain"], month)
        dc = dc_step(dc, d["T"], d["rain"], month)
        isi = isi_calc(ffmc, d["W"])
        bui = bui_calc(dmc, dc)
        out.append(dict(date=d["date"], ffmc=ffmc, dmc=dmc, dc=dc, isi=isi, bui=bui, fwi=fwi_calc(isi, bui)))
    return out


def danger_class(fwi):
    for lim, name in ((5, "low"), (10, "moderate"), (20, "high"), (30, "very high")):
        if fwi < lim:
            return name
    return "extreme"


@dataclass
class Conditions:
    source: str
    scenario: str
    date: str
    temp_c: float
    rh: float
    wind_kmh: float           # 10 m mean wind used for fire behaviour
    wind_dir_deg: float       # direction the wind blows FROM (meteorological)
    gust_ms: float            # max gust for windthrow
    rain_24h: float
    soil_wet: float           # 0..1 (wet, unfrozen soil -> weak anchorage)
    soil_frozen: bool
    ffmc: float = 0
    dmc: float = 0
    dc: float = 0
    isi: float = 0
    bui: float = 0
    fwi: float = 0
    danger: str = ""
    history: list = None
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        d["history"] = (self.history or [])[-21:]
        return d


SCENARIOS = {
    "normal": dict(label="Typical summer day", T=18, RH=55, W=12, rain_every=4, rain=6, gust=12, dir=240,
                   soil_wet=0.4, frozen=False, month=7),
    "heatwave": dict(label="Drought / heatwave (like July 2018)", T=28, RH=32, W=18, rain_every=0, rain=0,
                     gust=14, dir=200, soil_wet=0.1, frozen=False, month=7),
    "storm": dict(label="Autumn storm (like 'Aila' 2023 / 'Asta' 2010), unfrozen soil", T=9, RH=88, W=45,
                  rain_every=2, rain=9, gust=29, dir=225, soil_wet=0.9, frozen=False, month=10),
}


def scenario_conditions(name: str, date: str | None = None, gust: float | None = None,
                        wind_dir: float | None = None) -> Conditions:
    sc = SCENARIOS[name]
    rng = np.random.default_rng(len(name))
    month = sc["month"]
    days = []
    for i in range(45):
        rain = sc["rain"] if sc["rain_every"] and i % sc["rain_every"] == 1 else 0.0
        days.append(dict(date=f"2026-{month:02d}-{(i % 28) + 1:02d}", T=sc["T"] + rng.normal(0, 2),
                         RH=float(np.clip(sc["RH"] + rng.normal(0, 6), 15, 100)),
                         W=max(sc["W"] * 0.6 + rng.normal(0, 3), 0), rain=rain))
    days[-1].update(T=sc["T"], RH=sc["RH"], W=sc["W"], rain=days[-1]["rain"])
    hist = fwi_series(days)
    last = hist[-1]
    return Conditions(source="scenario", scenario=name, date=date or days[-1]["date"], temp_c=sc["T"], rh=sc["RH"],
                      wind_kmh=sc["W"], wind_dir_deg=wind_dir if wind_dir is not None else sc["dir"],
                      gust_ms=gust if gust is not None else sc["gust"], rain_24h=days[-1]["rain"],
                      soil_wet=sc["soil_wet"], soil_frozen=sc["frozen"],
                      ffmc=last["ffmc"], dmc=last["dmc"], dc=last["dc"], isi=last["isi"], bui=last["bui"],
                      fwi=last["fwi"], danger=danger_class(last["fwi"]), history=hist, note=sc["label"])


def fetch_open_meteo(lat: float, lon: float, timeout: float = 8.0) -> Conditions:
    """Live: 40 past days (FWI spin-up) + 2-day forecast for wind gusts."""
    url = ("https://api.open-meteo.com/v1/forecast?latitude={:.4f}&longitude={:.4f}"
           "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
           "precipitation,soil_moisture_3_to_9cm,soil_temperature_6cm"
           "&past_days=40&forecast_days=2&timezone=auto&wind_speed_unit=kmh").format(lat, lon)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        js = json.loads(r.read().decode())
    h = js["hourly"]
    times = h["time"]
    days = []
    for i, t in enumerate(times):
        if t.endswith("T12:00"):
            rain = float(np.nansum([v or 0 for v in h["precipitation"][max(0, i - 23):i + 1]]))
            days.append(dict(date=t[:10], T=h["temperature_2m"][i] or 0, RH=h["relative_humidity_2m"][i] or 50,
                             W=h["wind_speed_10m"][i] or 0, rain=rain, idx=i))
    # today = last day that is not in the future beyond tomorrow -> index -2 (today) when forecast_days=2
    today = days[-2] if len(days) >= 2 else days[-1]
    hist = fwi_series([d for d in days if d["date"] <= today["date"]])
    last = hist[-1]
    i0 = today["idx"] - 12
    gusts = [g for g in h["wind_gusts_10m"][i0:i0 + 48] if g is not None]
    dirs = [d for d in h["wind_direction_10m"][i0:i0 + 48] if d is not None]
    gi = int(np.argmax(gusts)) if gusts else 0
    sm = [v for v in h.get("soil_moisture_3_to_9cm", [])[i0:i0 + 24] if v is not None]
    st = [v for v in h.get("soil_temperature_6cm", [])[i0:i0 + 24] if v is not None]
    soil_wet = float(np.clip((np.mean(sm) - 0.15) / 0.30, 0, 1)) if sm else 0.5
    return Conditions(source="open-meteo.com (live)", scenario="live", date=today["date"], temp_c=today["T"],
                      rh=today["RH"], wind_kmh=today["W"], wind_dir_deg=float(dirs[gi] if dirs else 225),
                      gust_ms=float(gusts[gi] / 3.6 if gusts else 10), rain_24h=today["rain"], soil_wet=soil_wet,
                      soil_frozen=bool(st and np.mean(st) < 0), ffmc=last["ffmc"], dmc=last["dmc"], dc=last["dc"],
                      isi=last["isi"], bui=last["bui"], fwi=last["fwi"], danger=danger_class(last["fwi"]),
                      history=hist, note="Open-Meteo hourly reanalysis+forecast; FWI spun up over 40 days")


def get_conditions(scenario: str, lat: float, lon: float, gust=None, wind_dir=None) -> Conditions:
    if scenario == "live":
        try:
            c = fetch_open_meteo(lat, lon)
            if gust is not None:
                c.gust_ms = gust
            if wind_dir is not None:
                c.wind_dir_deg = wind_dir
            return c
        except Exception as exc:  # offline -> fall back, but say so
            c = scenario_conditions("normal", gust=gust, wind_dir=wind_dir)
            c.source = "fallback scenario (live weather unavailable)"
            c.note = f"Open-Meteo not reachable ({type(exc).__name__}); using typical summer day"
            return c
    return scenario_conditions(scenario, gust=gust, wind_dir=wind_dir)
