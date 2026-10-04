"""Drone-operator dispatch on Forey's operational dataset.

Baseline: schedule 1 000 flight tasks (LiDAR / LiDAR+multispectral) for 33 operators
over the planning period, respecting availability windows, drone configuration,
task release / due dates, setup + flight minutes and the travel-time matrix.

Emergency mode (Horus): a storm hits a region on date D. Horus turns it into rapid
post-storm survey tasks - re-flights of estates Forey already holds a pre-storm
point cloud for (change detection baseline) and 20 kV corridor inspections for the
distribution system operator - and re-plans the fleet so the most exposed areas
are flown first. Output: who flies where, when the first data lands, and what the
storm response costs the regular backlog.

Heuristic: day-by-day global best-ratio insertion. Operators keep their position
overnight (multi-day tours); an idle operator starts a transit towards the most
valuable reachable task when nothing fits in today's window.
"""
from __future__ import annotations

import datetime as dt
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .geo import haversine_km

PRIO_W = {"high": 3.0, "medium": 2.0, "low": 1.0, "emergency": 12.0}
MS = "DJI_M400_L3_LIDAR_MULTISPECTRAL"


def _d(s):
    return dt.date.fromisoformat(s)


def _hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


class Dataset:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.ops = json.loads((folder / "operators.json").read_text())["operators"]
        ft = json.loads((folder / "flight-tasks.json").read_text())
        self.tasks = ft["flightTasks"]
        self.period = ft["planningPeriod"]
        tt = json.loads((folder / "travel-times.json").read_text())
        self.nodes = tt["nodes"]
        self.M = np.array(tt["durationsMinutes"], dtype=float)
        self.node_idx = {n["nodeId"]: i for i, n in enumerate(self.nodes)}
        self.lat = np.array([n["latitude"] for n in self.nodes])
        self.lon = np.array([n["longitude"] for n in self.nodes])
        # calibrate a distance -> minutes model for nodes outside the matrix
        D = haversine_km(self.lat[:, None], self.lon[:, None], self.lat[None], self.lon[None])
        m = D > 1
        A = np.c_[D[m], np.ones(m.sum())]
        self.fit = np.linalg.lstsq(A, self.M[m], rcond=None)[0]
        self.fit_mae = float(np.abs(A @ self.fit - self.M[m]).mean())
        self.summary = json.loads((folder / "dataset-summary.json").read_text()) if (folder / "dataset-summary.json").exists() else {}

    def minutes(self, lat1, lon1, lat2, lon2):
        return self.fit[0] * haversine_km(lat1, lon1, lat2, lon2) + self.fit[1]


class Scheduler:
    def __init__(self, ds: Dataset, extra_tasks=None):
        self.ds = ds
        tasks = [dict(t) for t in ds.tasks] + [dict(t) for t in (extra_tasks or [])]
        self.tasks = tasks
        n0 = len(ds.nodes)
        # node index per task; extra tasks get new nodes
        lat = list(ds.lat)
        lon = list(ds.lon)
        self.t_node = []
        for t in tasks:
            nid = t["location"]["id"]
            if nid in ds.node_idx:
                self.t_node.append(ds.node_idx[nid])
            else:
                self.t_node.append(len(lat))
                lat.append(t["location"]["latitude"])
                lon.append(t["location"]["longitude"])
        self.t_node = np.array(self.t_node)
        lat, lon = np.array(lat), np.array(lon)
        N = len(lat)
        M = np.empty((N, N))
        M[:n0, :n0] = ds.M
        if N > n0:
            M[n0:, :] = ds.minutes(lat[n0:, None], lon[n0:, None], lat[None], lon[None])
            M[:, n0:] = ds.minutes(lat[:, None], lon[:, None], lat[None, n0:], lon[None, n0:])
            np.fill_diagonal(M, 0)
        self.M = M
        self.lat, self.lon = lat, lon
        self.dur = np.array([t["setupMinutes"] + t["estimatedFlightMinutes"] for t in tasks], float)
        self.prio = np.array([PRIO_W[t["priority"]] for t in tasks])
        self.rel = np.array([_d(t["availableFrom"]).toordinal() for t in tasks])
        self.due = np.array([_d(t["dueDate"]).toordinal() for t in tasks])
        self.ms = np.array([t["requiredDroneConfiguration"] == MS for t in tasks])
        self.emerg = np.array([t.get("emergency", False) for t in tasks])
        self.done = np.full(len(tasks), -1)       # ordinal day completed
        self.done_by = np.full(len(tasks), -1)
        self.reserved = np.full(len(tasks), -1)
        ops = ds.ops
        self.op_pos = np.array([ds.node_idx[o["homeLocation"]["id"]] for o in ops])
        self.op_ms = np.array([MS in o["supportedDroneConfigurations"] for o in ops])
        self.op_transit = [None] * len(ops)      # (task, remaining minutes)
        self.avail = []
        for o in ops:
            a = {}
            for r in o["availability"]:
                if r["status"] == "available":
                    a[_d(r["date"]).toordinal()] = (_hm(r["start"]), _hm(r["end"]))
            self.avail.append(a)
        self.log = []   # dict(day, op, task, start_min, travel, dur)
        self.travel_total = 0.0
        self.work_total = 0.0
        self.window_total = 0.0

    # ------------------------------------------------------------------
    def value(self, day):
        slack = self.due - day
        urg = np.where(slack >= 0, 1 + 3.0 / (1 + np.maximum(slack, 0)), 4.0 + 0.5 * np.minimum(-slack, 10))
        return self.prio * urg

    def step(self, day: int):
        ops_today = [i for i in range(len(self.op_pos)) if day in self.avail[i]]
        if not ops_today:
            return
        val = self.value(day)
        open_ = (self.done < 0) & (self.rel <= day)
        ms_backlog = (open_ & self.ms).sum()
        tl = {}
        clock = {}
        for i in ops_today:
            s, e = self.avail[i][day]
            tl[i] = float(e - s)
            clock[i] = s
            self.window_total += e - s
            tr = self.op_transit[i]
            if tr is not None:
                t, rem = tr
                use = min(rem, tl[i])
                tl[i] -= use
                clock[i] += use
                self.travel_total += use
                if rem - use <= 1e-6:
                    self.op_pos[i] = self.t_node[t]
                    self.op_transit[i] = None
                else:
                    self.op_transit[i] = (t, rem - use)
                    tl[i] = 0
        active = [i for i in ops_today if self.op_transit[i] is None and tl[i] > 0]
        did = {i: False for i in ops_today}
        while active:
            A = np.array(active)
            open_ = (self.done < 0) & (self.rel <= day)
            cand = np.nonzero(open_)[0]
            if cand.size == 0:
                break
            trav = self.M[self.op_pos[A][:, None], self.t_node[cand][None, :]]
            need = trav + self.dur[cand][None, :]
            ok = need <= np.array([tl[i] for i in A])[:, None]
            ok &= (~self.ms[cand][None, :]) | self.op_ms[A][:, None]
            res = self.reserved[cand]
            ok &= (res[None, :] < 0) | (res[None, :] == A[:, None])
            if not ok.any():
                break
            score = val[cand][None, :] * 100 / (need + 30)
            # keep scarce multispectral drones on multispectral work
            score = np.where(self.op_ms[A][:, None] & ~self.ms[cand][None, :] & (ms_backlog > 0), score * 0.6, score)
            score = np.where(res[None, :] == A[:, None], score * 3, score)
            score = np.where(ok, score, -1)
            k = np.unravel_index(np.argmax(score), score.shape)
            i, t = int(A[k[0]]), int(cand[k[1]])
            travel = trav[k]
            self.log.append(dict(day=day, op=i, task=t, start=clock[i] + travel, travel=float(travel),
                                 from_node=int(self.op_pos[i]),
                                 dur=float(self.dur[t])))
            clock[i] += travel + self.dur[t]
            tl[i] -= travel + self.dur[t]
            self.travel_total += travel
            self.work_total += self.dur[t]
            self.done[t] = day
            self.done_by[t] = i
            self.reserved[t] = -1
            self.op_pos[i] = self.t_node[t]
            did[i] = True
            if self.ms[t]:
                ms_backlog -= 1
            active = [j for j in active if tl[j] > 20]
        # idle operators: start transit towards the most valuable reachable task
        open_ = (self.done < 0) & (self.reserved < 0) & (self.rel <= day + 2)
        for i in ops_today:
            if did[i] or self.op_transit[i] is not None or tl[i] <= 0:
                continue
            cand = np.nonzero(open_ & ((~self.ms) | self.op_ms[i]))[0]
            if cand.size == 0:
                continue
            trav = self.M[self.op_pos[i], self.t_node[cand]]
            sc = val[cand] / (trav + self.dur[cand] + 60)
            t = int(cand[np.argmax(sc)])
            self.reserved[t] = i
            self.op_transit[i] = (t, float(self.M[self.op_pos[i], self.t_node[t]]))
            open_[t] = False
            rem = self.op_transit[i][1]
            use = min(rem, tl[i])
            self.travel_total += use
            if rem - use <= 1e-6:
                self.op_pos[i] = self.t_node[t]
                self.op_transit[i] = None
            else:
                self.op_transit[i] = (t, rem - use)

    def run(self, start: int, end: int):
        for d in range(start, end + 1):
            self.step(d)
        return self

    def kpis(self, until: int | None = None, subset=None):
        sel = np.ones(len(self.tasks), bool) if subset is None else subset
        done = (self.done >= 0) & sel
        on_time = done & (self.done <= self.due)
        hi = sel & (self.prio == 3)
        return dict(
            tasks=int(sel.sum()), completed=int(done.sum()), completion_rate=float(done.sum() / max(sel.sum(), 1)),
            on_time=int(on_time.sum()), on_time_rate=float(on_time.sum() / max(sel.sum(), 1)),
            high_on_time_rate=float((on_time & hi).sum() / max(hi.sum(), 1)),
            mean_lateness_days=float(np.mean(np.maximum(self.done[done] - self.due[done], 0))) if done.any() else 0,
            hectares=float(sum(self.tasks[t]["areaHectares"] for t in np.nonzero(done)[0])),
            travel_hours=float(self.travel_total / 60), flight_hours=float(self.work_total / 60),
            utilisation=float(self.work_total / max(self.window_total, 1)),
        )


def period_days(ds: Dataset):
    s = _d(ds.period["start"]).toordinal()
    e = _d(ds.period["end"]).toordinal()
    return s, e


def baseline(ds: Dataset):
    s, e = period_days(ds)
    sch = Scheduler(ds).run(s, e)
    k = sch.kpis()
    by_day = {}
    for r in sch.log:
        by_day.setdefault(r["day"], 0)
        by_day[r["day"]] += 1
    return sch, k


def storm_tasks(ds: Dataset, date: str, lat: float, lon: float, radius_km: float, gust: float,
                n_corridors: int = 8, seed: int = 7):
    """Create emergency survey tasks for a storm footprint."""
    rng = np.random.default_rng(seed)
    out = []
    d0 = _d(date)
    for t in ds.tasks:
        r = float(haversine_km(lat, lon, t["location"]["latitude"], t["location"]["longitude"]))
        if r <= radius_km:
            g = gust * (1 - 0.45 * (r / radius_km) ** 2)
            area = max(20, int(t["areaHectares"] * 0.5))
            out.append(dict(
                id="E-" + t["id"][:8], type="emergency_survey", name=f"Storm re-survey of {t['name']}",
                emergency=True, kind="estate_change_detection", gust_ms=round(g, 1), distance_km=round(r, 1),
                location=dict(t["location"]), areaHectares=area, priority="emergency",
                availableFrom=date, dueDate=(d0 + dt.timedelta(days=2)).isoformat(),
                requiredDroneConfiguration="DJI_M400_L3_LIDAR", setupMinutes=10,
                flightMinutesPerHectare=0.6, estimatedFlightMinutes=int(area * 0.6),
                exposure=round(max(g - 15, 0) ** 2 * area / 100, 1),
                baseline_task=t["id"]))
    for k in range(n_corridors):
        a = rng.uniform(0, 2 * np.pi)
        rr = radius_km * np.sqrt(rng.uniform(0.05, 0.9))
        la = lat + rr / 111.0 * np.sin(a)
        lo = lon + rr / (111.0 * np.cos(np.radians(lat))) * np.cos(a)
        g = gust * (1 - 0.45 * (rr / radius_km) ** 2)
        out.append(dict(
            id=f"E-PL{k + 1:02d}", type="emergency_survey", name=f"20 kV corridor inspection #{k + 1}",
            emergency=True, kind="powerline_corridor", gust_ms=round(g, 1), distance_km=round(rr, 1),
            location=dict(id=f"pl-{k}", municipality="corridor", latitude=float(la), longitude=float(lo)),
            areaHectares=60, priority="emergency", availableFrom=date,
            dueDate=(d0 + dt.timedelta(days=1)).isoformat(), requiredDroneConfiguration="DJI_M400_L3_LIDAR",
            setupMinutes=10, flightMinutesPerHectare=0.6, estimatedFlightMinutes=36,
            exposure=round(max(g - 15, 0) ** 2 * 60 / 100 * 2, 1)))
    return out


def storm_response(ds: Dataset, date: str, lat: float, lon: float, radius_km: float = 60, gust: float = 30,
                   horizon_days: int = 10):
    s, e = period_days(ds)
    D = _d(date).toordinal()
    em = storm_tasks(ds, date, lat, lon, radius_km, gust)
    # warm-start both branches with the regular plan up to the storm
    base = Scheduler(ds, extra_tasks=em)
    base.rel[len(ds.tasks):] = 10 ** 9  # emergency tasks invisible before the storm
    base.run(s, D - 1)
    no_resp = deepcopy(base)
    resp = deepcopy(base)
    uniform = deepcopy(base)
    n0 = len(ds.tasks)
    for sch in (resp, uniform):
        sch.rel[n0:] = D
    # business as usual: storm requests are queued as ordinary high-priority tasks
    uniform.prio[n0:] = PRIO_W["high"]
    # Horus: emergency value scales with storm exposure at the site (gust x forest x assets)
    ex = np.array([t["exposure"] for t in em])
    resp.prio[n0:] = PRIO_W["emergency"] * (0.6 + 0.8 * ex / max(ex.max(), 1e-6))
    end = min(e, D + horizon_days)
    for sch in (no_resp, resp, uniform):
        sch.run(D, end)
    regular = np.zeros(len(resp.tasks), bool)
    regular[:n0] = True
    assign = _assignments(ds, resp, em, D, n0)
    assign_u = _assignments(ds, uniform, em, D, n0)
    done_em = resp.done[n0:]
    within = lambda h: int(np.sum([a["data_ready_h_after_storm"] <= h for a in assign]))
    reg_resp = resp.kpis(subset=regular)
    reg_no = no_resp.kpis(subset=regular)
    return dict(
        storm=dict(date=date, lat=lat, lon=lon, radius_km=radius_km, gust_ms=gust),
        emergency_tasks=[dict(name=t["name"], kind=t["kind"], lat=t["location"]["latitude"], lon=t["location"]["longitude"],
                              gust_ms=t["gust_ms"], exposure=t["exposure"], area_ha=t["areaHectares"],
                              done=dt.date.fromordinal(int(done_em[i])).isoformat() if done_em[i] >= 0 else None)
                         for i, t in enumerate(em)],
        assignments=assign,
        kpi=dict(emergency_tasks=len(em), completed=int((done_em >= 0).sum()),
                 data_within_24h=within(24), data_within_48h=within(48),
                 median_hours_to_data=float(np.median([a["data_ready_h_after_storm"] for a in assign])) if assign else None,
                 exposure_weighted_hours=_ewh(assign),
                 business_as_usual_exposure_weighted_hours=_ewh(assign_u),
                 business_as_usual_within_24h=int(np.sum([a['data_ready_h_after_storm'] <= 24 for a in assign_u])),
                 business_as_usual_completed=len(assign_u)),
        backlog_impact=dict(with_response=reg_resp, without_response=reg_no,
                            regular_tasks_delayed=int(np.sum((resp.done[:n0] > no_resp.done[:n0]) | ((resp.done[:n0] < 0) & (no_resp.done[:n0] >= 0)))),
                            high_priority_on_time_change=reg_resp["high_on_time_rate"] - reg_no["high_on_time_rate"]),
    )


def _ewh(assign):
    """Exposure-weighted mean hours from storm to delivered data (lower = the worst-hit
    areas are seen first)."""
    if not assign:
        return None
    w = np.array([a["exposure"] for a in assign])
    h = np.array([a["data_ready_h_after_storm"] for a in assign])
    return float((w * h).sum() / max(w.sum(), 1e-6))


def _assignments(ds, sch, em, D, n0):
    ops = ds.ops
    out = []
    for r in sch.log:
        if r["task"] < n0:
            continue
        t = sch.tasks[r["task"]]
        start_h = r["start"] / 60.0
        hours_after = (r["day"] - D) * 24 + start_h + r["dur"] / 60.0
        out.append(dict(operator=ops[r["op"]]["name"], operator_home=ops[r["op"]]["homeLocation"]["municipality"],
                        task=t["name"], kind=t["kind"], date=dt.date.fromordinal(r["day"]).isoformat(),
                        start=f"{int(start_h):02d}:{int(round((start_h % 1) * 60)) % 60:02d}",
                        travel_min=round(r["travel"]), flight_min=round(r["dur"]),
                        data_ready_h_after_storm=round(hours_after + 2.0, 1),  # + ~2 h cloud processing
                        gust_ms=t["gust_ms"], exposure=t["exposure"],
                        lat=t["location"]["latitude"], lon=t["location"]["longitude"],
                        from_lat=float(sch.lat[r["from_node"]]), from_lon=float(sch.lon[r["from_node"]])))
    out.sort(key=lambda a: a["data_ready_h_after_storm"])
    return out
