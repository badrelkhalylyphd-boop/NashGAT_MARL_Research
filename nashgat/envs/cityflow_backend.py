"""CityFlow backend exposing the same interface as QueueSim.

Requires the `cityflow` Python package (built from https://github.com/cityflow-project/CityFlow).
It could not be built in the local development environment (Windows, no C++ toolchain),
so this module is exercised only in the Colab notebook. Queue = number of vehicles
with speed < 0.1 m/s on the movement's start lane(s), as in CoLight/PressLight.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .roadnet import RoadNet


class CityFlowBackend:
    def __init__(self, net: RoadNet, cfg):
        import cityflow  # noqa: F401  (ImportError is the signal that the backend is unavailable)

        self.net, self.cfg = net, cfg
        self.agents = net.agents
        self.road_ids = sorted(net.roads)
        self.ridx = {r: k for k, r in enumerate(self.road_ids)}
        self.cap = np.array([max(1, int(net.roads[r].length * net.roads[r].n_lanes // 7.5)) for r in self.road_ids])
        self.mov_key, self.mov_lanes, self.agent_movs, self.agent_phases = [], [], [], []
        self.start_lanes: list[list[str]] = []
        raw = json.load(open(cfg.roadnet, encoding="utf-8"))
        rl_lanes = {}
        for it in raw["intersections"]:
            for k, rl in enumerate(it.get("roadLinks", [])):
                rl_lanes[(it["id"], k)] = sorted({f"{rl['startRoad']}_{ll['startLaneIndex']}" for ll in rl["laneLinks"]})
        for a in self.agents:
            it = net.intersections[a]
            ids = []
            for m in it.movements:
                ids.append(len(self.mov_key))
                self.mov_key.append((self.ridx[m.start_road], self.ridx[m.end_road]))
                self.mov_lanes.append(m.n_lanes)
                self.start_lanes.append(rl_lanes[(a, m.index)])
            self.agent_movs.append(ids)
            self.agent_phases.append([[ids[k] for k in p] for p in it.phases])
        self.n_mov = len(self.mov_key)
        self.n_phases = [len(p) - 1 for p in self.agent_phases]
        self.failed_agents: set[int] = set()
        self._tmp = tempfile.mkdtemp(prefix="cityflow_")
        conf = {"interval": 1.0, "seed": int(cfg.seed), "dir": "", "roadnetFile": os.path.abspath(cfg.roadnet),
                "flowFile": os.path.abspath(cfg.flow), "rlTrafficLight": True, "saveReplay": False,
                "roadnetLogFile": "", "replayLogFile": "", "laneChange": False}
        self._conf_path = Path(self._tmp) / "config.json"
        self._conf_path.write_text(json.dumps(conf))
        self.reset()

    def reset(self) -> None:
        import cityflow

        self.eng = cityflow.Engine(str(self._conf_path), thread_num=1)
        nA = len(self.agents)
        self.t = 0
        self.phase = np.ones(nA, dtype=np.int64)
        self.pending_phase = np.ones(nA, dtype=np.int64)
        self.yellow_left = np.zeros(nA, dtype=np.int64)
        self._fixed_timer = np.zeros(nA, dtype=np.int64)
        self._wait_accum = 0.0
        self._prev_vehicles: set[str] = set()
        self._finished = 0
        for a, aid in enumerate(self.agents):
            self.eng.set_tl_phase(aid, 1)
        self._refresh()

    def set_phases(self, phases: np.ndarray) -> None:
        for a, p in enumerate(phases):
            if a in self.failed_agents:
                continue
            lp = int(p) + 1
            if lp != self.phase[a] and self.yellow_left[a] == 0:
                self.yellow_left[a] = self.cfg.yellow_time
                self.pending_phase[a] = lp
                self.eng.set_tl_phase(self.agents[a], 0)
            elif self.yellow_left[a] > 0:
                self.pending_phase[a] = lp

    def step_second(self) -> None:
        for a in self.failed_agents:
            self._fixed_timer[a] += 1
            if self._fixed_timer[a] >= 30 and self.yellow_left[a] == 0:
                self._fixed_timer[a] = 0
                self.yellow_left[a] = self.cfg.yellow_time
                self.pending_phase[a] = (self.phase[a] % self.n_phases[a]) + 1
                self.eng.set_tl_phase(self.agents[a], 0)
        self.eng.next_step()
        self.t += 1
        for a in range(len(self.agents)):
            if self.yellow_left[a] > 0:
                self.yellow_left[a] -= 1
                if self.yellow_left[a] == 0:
                    self.phase[a] = self.pending_phase[a]
                    self.eng.set_tl_phase(self.agents[a], int(self.phase[a]))
        self._refresh()
        self._wait_accum += float(sum(self._lane_wait.values()))

    def _refresh(self) -> None:
        self._lane_wait = self.eng.get_lane_waiting_vehicle_count()
        self._lane_cnt = self.eng.get_lane_vehicle_count()
        cur = set(self.eng.get_vehicles(include_waiting=True))
        self._finished += len(self._prev_vehicles - cur)  # vehicles that left the network this second
        self._prev_vehicles = cur
        self.count = np.zeros(len(self.road_ids), dtype=np.int64)
        for lane, c in self._lane_cnt.items():
            road = lane.rsplit("_", 1)[0]
            if road in self.ridx:
                self.count[self.ridx[road]] += c
        q = np.zeros(self.n_mov, dtype=np.int64)
        app = np.zeros(self.n_mov, dtype=np.int64)
        for m, lanes in enumerate(self.start_lanes):
            q[m] = sum(self._lane_wait.get(l, 0) for l in lanes)
            app[m] = sum(self._lane_cnt.get(l, 0) for l in lanes) - q[m]
        self._q = q
        self.approach = app

    def queue_lengths(self) -> np.ndarray:
        return self._q

    def metrics(self) -> dict:
        return {
            "time": self.t,
            "avg_travel_time": float(self.eng.get_average_travel_time()),
            "vehicles_in_network": int(self.eng.get_vehicle_count()),
            "avg_waiting_time": float("nan"),  # not exposed per vehicle by CityFlow; see total_waiting_vehicle_seconds
            "total_waiting_vehicle_seconds": self._wait_accum,
            "total_queue": int(self._q.sum()),
            "throughput": int(self._finished),
        }
