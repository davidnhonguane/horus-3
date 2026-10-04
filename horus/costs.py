"""Turn simulated storm and fire damage into euros.

Every unit cost is editable (web UI "Unit costs", or data/costs.json) and carries its basis:
  * "derived" - computed from a published Finnish figure (source given)
  * "assumption" - a placeholder to replace with the customer's own numbers in a pilot

Published anchors used for the derived values:
  * Elenia, Storm Hannes (2025): EUR 12-13 M total, over half customer compensation, up to 76,000
    customers without power, >2,800 fault repairs.
      -> compensation ~ EUR 6.5 M / 76,000 ~ EUR 85 per affected customer
      -> repair ~ (EUR 12.5 M - 6.5 M) / 2,800 ~ EUR 2,100 per fault
  * Fortum, 2011 Christmas storms: EUR 23.4 M compensation, >190,000 customers -> ~EUR 120 / customer
    (we use EUR 100 between the two)
  * Yle, July 2014 storm: 0.5 million m3 of timber worth ~EUR 20 M -> ~EUR 40 per m3
  * Electricity Market Act: standard compensation is capped at EUR 2,000 per customer per year
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

DEFAULTS = {
    "outage_compensation_per_customer": dict(value=100.0, unit="EUR / customer", basis="derived",
        label="Outage compensation per affected customer",
        source="Elenia Storm Hannes (~EUR 85) and Fortum 2011 storms (~EUR 120)"),
    "customers_per_line": dict(value=250.0, unit="customers", basis="assumption",
        label="Customers cut off when the scanned line fails",
        source="typical rural 20 kV feeder - replace with the grid company's own figure"),
    "line_fault_repair": dict(value=2100.0, unit="EUR / fault", basis="derived",
        label="Repair of one tree-on-line fault",
        source="Elenia Storm Hannes: (EUR 12.5 M - compensation) / 2,800 faults"),
    "road_tree_clearing": dict(value=300.0, unit="EUR / tree", basis="assumption",
        label="Clearing one tree off a road", source="crew with chainsaw ~1-2 h - replace with contractor price"),
    "building_tree_damage": dict(value=8000.0, unit="EUR / hit", basis="assumption",
        label="Repair after a tree falls on a building", source="roof / wall repair - replace with insurer claim data"),
    "timber_value_per_m3": dict(value=40.0, unit="EUR / m3", basis="derived",
        label="Timber value", source="Yle: 0.5 M m3 toppled in July 2014 worth ~EUR 20 M"),
    "storm_timber_loss_share": dict(value=0.3, unit="share", basis="assumption",
        label="Share of a fallen tree's value lost (breakage, decay, costlier harvest)",
        source="salvage recovers most of the value - replace with forest company data"),
    "fire_timber_loss_torching": dict(value=1.0, unit="share", basis="assumption",
        label="Timber value lost when a tree torches (crown fire)", source="crown fire kills the tree"),
    "fire_timber_loss_surface": dict(value=0.3, unit="share", basis="assumption",
        label="Timber value lost under a surface fire", source="scorch, later mortality, lower quality"),
    "reforestation_per_ha": dict(value=1200.0, unit="EUR / ha", basis="assumption",
        label="Restoring burned forest (site preparation + planting)", source="Finnish regeneration costs - check with Metsakeskus"),
    "house_value": dict(value=200000.0, unit="EUR", basis="assumption",
        label="Value of a house reached by fire", source="replace with insured values"),
    "cabin_value": dict(value=80000.0, unit="EUR", basis="assumption",
        label="Value of a cabin / small building reached by fire", source="replace with insured values"),
    "fire_building_loss_share": dict(value=0.25, unit="share", basis="assumption",
        label="Chance a building reached by the fire is lost", source="depends on defensible space and response"),
    "fire_line_repair": dict(value=50000.0, unit="EUR / line reached", basis="assumption",
        label="Power line repair after fire reaches it", source="poles + conductors - replace with grid company data"),
    "felling_cost_per_tree": dict(value=80.0, unit="EUR / tree", basis="assumption",
        label="Felling one hazard tree near a line", source="contractor price - used for the cost/benefit of each tree"),
}

CFG_FILE = Path(__file__).resolve().parent.parent / "data" / "costs.json"


def load(path=CFG_FILE) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    try:
        user = json.loads(Path(path).read_text())
        for k, v in user.items():
            if k in cfg:
                cfg[k]["value"] = float(v["value"] if isinstance(v, dict) else v)
                cfg[k]["basis"] = "user" if not isinstance(v, dict) or "basis" not in v else v["basis"]
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg, path=CFG_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({k: dict(value=v["value"], basis=v["basis"]) for k, v in cfg.items()}, indent=1))


def v(cfg, key) -> float:
    return float(cfg[key]["value"])


def tree_volume_m3(T) -> np.ndarray:
    """Stem volume from height and DBH: v = f * basal area * height, form factor 0.45."""
    d = np.asarray(T["dbh"], float) / 100.0
    return 0.45 * np.pi * (d / 2) ** 2 * np.asarray(T["h"], float)


def is_house(kind) -> bool:
    return kind not in ("cabin", "sauna", "shed", "barn")


# ------------------------------------------------------------------------------------------- storm
def storm_run_cost(cfg, line_hit_any, n_line_hits, n_road, n_bld, fallen_volume):
    """Per Monte-Carlo run (arrays) -> dict of EUR arrays by component."""
    out = {
        "Outage compensation": line_hit_any * v(cfg, "customers_per_line") * v(cfg, "outage_compensation_per_customer"),
        "Power line repairs": n_line_hits * v(cfg, "line_fault_repair"),
        "Road clearing": n_road * v(cfg, "road_tree_clearing"),
        "Building repairs": n_bld * v(cfg, "building_tree_damage"),
        "Timber value lost": fallen_volume * v(cfg, "timber_value_per_m3") * v(cfg, "storm_timber_loss_share"),
    }
    return out


def summarise(runs: dict) -> dict:
    total = sum(runs.values())
    return dict(total_mean=float(total.mean()), total_p5=float(np.percentile(total, 5)), total_p95=float(np.percentile(total, 95)),
                parts={k: float(a.mean()) for k, a in runs.items()})


def tree_expected_damage(cfg, T, p_fail, w_power, w_bld, w_road):
    """Expected EUR damage each tree causes in this storm (= what felling it avoids)."""
    line_cost = v(cfg, "line_fault_repair") + v(cfg, "customers_per_line") * v(cfg, "outage_compensation_per_customer")
    sizef = np.clip((np.asarray(T["h"]) - 8.0) / 10.0, 0.1, 1.0)
    timber = tree_volume_m3(T) * v(cfg, "timber_value_per_m3") * v(cfg, "storm_timber_loss_share")
    return p_fail * (w_power * line_cost + w_bld * v(cfg, "building_tree_damage")
                     + w_road * sizef * v(cfg, "road_tree_clearing") + timber)


# ------------------------------------------------------------------------------------------- fire
def fire_cost(cfg, T, reached_idx, p_torch, burned_ha, impacts, buildings) -> dict:
    vol = tree_volume_m3(T)[reached_idx]
    pt = np.asarray(p_torch, float)
    val = vol * v(cfg, "timber_value_per_m3")
    timber = float((val * (pt * v(cfg, "fire_timber_loss_torching") + (1 - pt) * v(cfg, "fire_timber_loss_surface"))).sum())
    kind_of = {b["name"]: b.get("kind", "house") for b in buildings}
    bld = 0.0
    n_bld = 0
    lines = 0
    for i in impacts:
        if i.get("minutes") is None:
            continue
        if i["kind"] == "power_line":
            lines += 1
        elif i["kind"] not in ("road_cut",):
            k = kind_of.get(i["asset"], i["kind"])
            bld += (v(cfg, "house_value") if is_house(k) else v(cfg, "cabin_value")) * v(cfg, "fire_building_loss_share")
            n_bld += 1
    parts = {
        "Timber value lost": timber,
        "Reforestation": float(burned_ha) * v(cfg, "reforestation_per_ha"),
        "Buildings (expected loss)": bld,
        "Power line repair": lines * v(cfg, "fire_line_repair"),
        "Outage compensation": (v(cfg, "customers_per_line") * v(cfg, "outage_compensation_per_customer")) if lines else 0.0,
    }
    return dict(total=float(sum(parts.values())), parts=parts, buildings_reached=n_bld, lines_reached=lines)


def public(cfg) -> list:
    return [dict(key=k, **c) for k, c in cfg.items()]
