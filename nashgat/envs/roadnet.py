"""Parsing and validation of CityFlow road-network and flow files."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Road:
    id: str
    start: str
    end: str
    length: float
    n_lanes: int
    max_speed: float


@dataclass
class Movement:
    """One CityFlow roadLink of an intersection (startRoad -> endRoad)."""

    index: int
    kind: str
    start_road: str
    end_road: str
    n_lanes: int


@dataclass
class Intersection:
    id: str
    virtual: bool
    roads: list[str]
    movements: list[Movement] = field(default_factory=list)
    phases: list[list[int]] = field(default_factory=list)  # availableRoadLinks per light phase
    phase_times: list[int] = field(default_factory=list)


@dataclass
class RoadNet:
    roads: dict[str, Road]
    intersections: dict[str, Intersection]

    @property
    def agents(self) -> list[str]:
        """Signalised (non-virtual) intersections, sorted for a stable agent order."""
        return sorted(i.id for i in self.intersections.values() if not i.virtual)

    def neighbour_edges(self) -> list[tuple[int, int, str]]:
        """Directed agent edges (j -> i) induced by roads joining two signalised
        intersections, with the connecting road id."""
        idx = {a: k for k, a in enumerate(self.agents)}
        out = []
        for r in self.roads.values():
            if r.start in idx and r.end in idx:
                out.append((idx[r.start], idx[r.end], r.id))
        return sorted(out)


def _length(points: list[dict]) -> float:
    return sum(math.dist((points[k]["x"], points[k]["y"]), (points[k + 1]["x"], points[k + 1]["y"]))
               for k in range(len(points) - 1))


def load_roadnet(path: str | Path) -> RoadNet:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    roads = {}
    for r in raw["roads"]:
        roads[r["id"]] = Road(r["id"], r["startIntersection"], r["endIntersection"], _length(r["points"]),
                              len(r["lanes"]), float(r["lanes"][0]["maxSpeed"]))
    inters = {}
    for it in raw["intersections"]:
        inter = Intersection(it["id"], bool(it["virtual"]), list(it.get("roads", [])))
        for k, rl in enumerate(it.get("roadLinks", [])):
            lanes = {ll["startLaneIndex"] for ll in rl["laneLinks"]}
            inter.movements.append(Movement(k, rl["type"], rl["startRoad"], rl["endRoad"], max(1, len(lanes))))
        if not inter.virtual:
            tl = it["trafficLight"]
            inter.phases = [list(p["availableRoadLinks"]) for p in tl["lightphases"]]
            inter.phase_times = [int(p["time"]) for p in tl["lightphases"]]
        inters[it["id"]] = inter
    return RoadNet(roads, inters)


def load_flows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def expand_departures(flows: list[dict]) -> list[tuple[int, list[str]]]:
    """Expand CityFlow flow entries into (departure_time, route) per vehicle."""
    out = []
    for f in flows:
        t, end, iv = float(f["startTime"]), float(f["endTime"]), float(f.get("interval", 1.0))
        iv = max(iv, 1e-9)
        while t <= end + 1e-9:
            out.append((int(round(t)), list(f["route"])))
            if end == f["startTime"]:
                break
            t += iv
    out.sort(key=lambda z: z[0])
    return out


def validate(net: RoadNet, flows: list[dict]) -> dict:
    """Consistency checks reported in DATASET_CARD.md."""
    issues = []
    for r in net.roads.values():
        if r.start not in net.intersections or r.end not in net.intersections:
            issues.append(f"road {r.id} references unknown intersection")
        if r.length <= 0:
            issues.append(f"road {r.id} non-positive length")
    for it in net.intersections.values():
        for m in it.movements:
            if m.start_road not in net.roads or m.end_road not in net.roads:
                issues.append(f"{it.id}: movement {m.index} references unknown road")
            elif net.roads[m.start_road].end != it.id or net.roads[m.end_road].start != it.id:
                issues.append(f"{it.id}: movement {m.index} not incident to intersection")
        for p in it.phases:
            if any(k >= len(it.movements) for k in p):
                issues.append(f"{it.id}: phase references missing roadLink")
    bad_routes = 0
    for f in flows:
        rt = f["route"]
        if any(r not in net.roads for r in rt):
            bad_routes += 1
            continue
        for a, b in zip(rt[:-1], rt[1:]):
            if net.roads[a].end != net.roads[b].start:
                bad_routes += 1
                break
    deps = expand_departures(flows)
    return {
        "n_intersections": len(net.intersections),
        "n_signalised": len(net.agents),
        "n_virtual": sum(i.virtual for i in net.intersections.values()),
        "n_roads": len(net.roads),
        "n_phases": sorted({len(net.intersections[a].phases) for a in net.agents}),
        "n_movements": sorted({len(net.intersections[a].movements) for a in net.agents}),
        "road_length_range_m": (min(r.length for r in net.roads.values()), max(r.length for r in net.roads.values())),
        "n_flow_entries": len(flows),
        "n_vehicles": len(deps),
        "departure_range_s": (deps[0][0], deps[-1][0]) if deps else None,
        "route_length_hist": _hist([len(f["route"]) for f in flows]),
        "invalid_routes": bad_routes,
        "issues": issues,
    }


def _hist(xs: list[int]) -> dict[int, int]:
    h: dict[int, int] = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return dict(sorted(h.items()))
