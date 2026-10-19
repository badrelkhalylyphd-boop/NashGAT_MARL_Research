"""Lightweight mesoscopic queue simulator for CityFlow road networks ("QueueSim").

This is NOT CityFlow. It is a separate, deliberately simple experimental environment
used where CityFlow cannot be built. It reads the original CityFlow roadnet/flow
files and models:

* free-flow travel on each road (time = length / maxSpeed, rounded up),
* point queues per movement (CityFlow roadLink) at the downstream intersection,
* saturation-flow discharge (one vehicle per `sat_headway` s per lane) during green,
* finite road storage (length * lanes / vehicle spacing) causing spill-back,
* a yellow interval on every phase change during which only the links of
  light phase 0 (right turns in the CoLight networks) may discharge.

There is no car-following, lane changing or vehicle-level kinematics, so absolute
travel times are not comparable with CityFlow numbers published elsewhere.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .roadnet import RoadNet


@dataclass
class SimConfig:
    yellow_time: int = 3
    sat_headway: float = 2.0  # s per vehicle per lane
    vehicle_spacing: float = 7.5  # m of storage per queued vehicle (length 5 + min gap 2.5)
    horizon: int = 3600


class QueueSim:
    def __init__(self, net: RoadNet, departures: list[tuple[int, list[str]]], cfg: SimConfig | None = None):
        self.net = net
        self.cfg = cfg or SimConfig()
        self.departures = departures
        self.agents = net.agents
        self.agent_index = {a: k for k, a in enumerate(self.agents)}
        self.road_ids = sorted(net.roads)
        self.ridx = {r: k for k, r in enumerate(self.road_ids)}
        R = len(self.road_ids)
        self.fft = np.array([max(1, math.ceil(net.roads[r].length / net.roads[r].max_speed)) for r in self.road_ids])
        self.cap = np.array([max(1, int(net.roads[r].length * net.roads[r].n_lanes // self.cfg.vehicle_spacing))
                             for r in self.road_ids])
        self.road_lanes = np.array([net.roads[r].n_lanes for r in self.road_ids])
        # movements: global index per (agent, link)
        self.mov_key: list[tuple[int, int]] = []  # (from road idx, to road idx)
        self.mov_lanes: list[int] = []
        self.mov_agent: list[int] = []
        self.agent_movs: list[list[int]] = []
        self.agent_phases: list[list[list[int]]] = []  # phases as global movement ids
        self.agent_yellow_links: list[list[int]] = []
        key_to_mov: dict[tuple[int, int], int] = {}
        for a in self.agents:
            it = net.intersections[a]
            ids = []
            for m in it.movements:
                g = len(self.mov_key)
                k = (self.ridx[m.start_road], self.ridx[m.end_road])
                self.mov_key.append(k)
                self.mov_lanes.append(m.n_lanes)
                self.mov_agent.append(self.agent_index[a])
                key_to_mov[k] = g
                ids.append(g)
            self.agent_movs.append(ids)
            self.agent_phases.append([[ids[k] for k in p] for p in it.phases])
            self.agent_yellow_links.append([ids[k] for k in it.phases[0]] if it.phases else [])
        self.key_to_mov = key_to_mov
        self.n_phases = [len(p) - 1 for p in self.agent_phases]  # controllable phases exclude phase 0
        self.n_mov = len(self.mov_key)
        self.mov_rate = np.array(self.mov_lanes, dtype=float) / self.cfg.sat_headway
        self.failed_agents: set[int] = set()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        R = len(self.road_ids)
        self.t = 0
        self.count = np.zeros(R, dtype=np.int64)  # vehicles on road (moving or queued)
        self.transit: list[deque] = [deque() for _ in range(R)]
        self.queues: list[deque] = [deque() for _ in range(self.n_mov)]
        self.approach = np.zeros(self.n_mov, dtype=np.int64)  # moving vehicles heading to each movement
        self.credit = np.zeros(self.n_mov)
        self.entry: dict[int, deque] = {}
        self.entry_credit = np.zeros(R)
        self.free_queues: dict[tuple[int, int], deque] = {}  # transitions with no roadLink (should not occur)
        n_veh = len(self.departures)
        self.route = [[self.ridx[r] for r in rt] for _, rt in self.departures]
        self.depart = np.array([d for d, _ in self.departures], dtype=np.int64)
        self.pos = np.zeros(n_veh, dtype=np.int64)
        self.finish = np.full(n_veh, -1, dtype=np.int64)
        self.wait = np.zeros(n_veh, dtype=np.int64)
        self.queue_join = np.zeros(n_veh, dtype=np.int64)
        self.next_dep = 0
        nA = len(self.agents)
        self.phase = np.ones(nA, dtype=np.int64)  # light phase index (1..P)
        self.yellow_left = np.zeros(nA, dtype=np.int64)
        self.pending_phase = np.ones(nA, dtype=np.int64)
        self.n_missing_links = 0
        self.green_time = np.zeros(nA, dtype=np.int64)
        self._fixed_timer = np.zeros(nA, dtype=np.int64)

    # --------------------------------------------------------------- control
    def set_phases(self, phases: np.ndarray) -> None:
        """phases: controllable phase per agent, 0-based (maps to light phase 1..P)."""
        for a, p in enumerate(phases):
            if a in self.failed_agents:
                continue  # failed controller follows its fixed-time fallback in _fixed_time_step
            lp = int(p) + 1
            if lp != self.phase[a] and self.yellow_left[a] == 0:
                self.yellow_left[a] = self.cfg.yellow_time
                self.pending_phase[a] = lp
            elif self.yellow_left[a] > 0:
                self.pending_phase[a] = lp

    def _fixed_time_step(self) -> None:
        for a in self.failed_agents:
            self._fixed_timer[a] += 1
            if self._fixed_timer[a] >= 30 and self.yellow_left[a] == 0:
                self._fixed_timer[a] = 0
                self.yellow_left[a] = self.cfg.yellow_time
                self.pending_phase[a] = (self.phase[a] % self.n_phases[a]) + 1

    # ------------------------------------------------------------- dynamics
    def _enter_road(self, v: int, r: int) -> None:
        self.count[r] += 1
        self.transit[r].append((self.t + int(self.fft[r]), v))
        rt = self.route[v]
        p = self.pos[v]
        if p + 1 < len(rt):
            m = self.key_to_mov.get((r, rt[p + 1]))
            if m is not None:
                self.approach[m] += 1

    def step_second(self) -> None:
        t = self.t
        # 1) new departures join the entry buffer of their first road
        while self.next_dep < len(self.depart) and self.depart[self.next_dep] <= t:
            v = self.next_dep
            self.entry.setdefault(self.route[v][0], deque()).append(v)
            self.next_dep += 1
        # 2) entry into the network, limited by road storage and lane capacity
        for r, buf in self.entry.items():
            if not buf:
                continue
            self.entry_credit[r] = min(self.entry_credit[r] + self.road_lanes[r] / self.cfg.sat_headway, self.road_lanes[r])
            while buf and self.entry_credit[r] >= 1.0 and self.count[r] < self.cap[r]:
                v = buf.popleft()
                self.wait[v] += t - self.depart[v]
                self._enter_road(v, r)
                self.entry_credit[r] -= 1.0
        # 3) vehicles reaching the end of a road join a movement queue or leave
        for r in range(len(self.transit)):
            tr = self.transit[r]
            while tr and tr[0][0] <= t:
                _, v = tr.popleft()
                rt = self.route[v]
                p = self.pos[v]
                if p + 1 >= len(rt):
                    self.count[r] -= 1
                    self.finish[v] = t
                    continue
                m = self.key_to_mov.get((r, rt[p + 1]))
                if m is None:
                    self.n_missing_links += 1
                    self.free_queues.setdefault((r, rt[p + 1]), deque()).append(v)
                    continue
                self.approach[m] -= 1
                self.queues[m].append(v)
                self.queue_join[v] = t
        # 4) discharge on green movements
        self._fixed_time_step()
        for a in range(len(self.agents)):
            if self.yellow_left[a] > 0:
                active = self.agent_yellow_links[a]
                self.yellow_left[a] -= 1
                if self.yellow_left[a] == 0:
                    self.phase[a] = self.pending_phase[a]
                    self.green_time[a] = 0
            else:
                active = self.agent_phases[a][self.phase[a]]
                self.green_time[a] += 1
            for m in active:
                q = self.queues[m]
                if not q:
                    self.credit[m] = 0.0
                    continue
                self.credit[m] = min(self.credit[m] + self.mov_rate[m], self.mov_lanes[m])
                r_in, r_out = self.mov_key[m]
                while q and self.credit[m] >= 1.0 and self.count[r_out] < self.cap[r_out]:
                    v = q.popleft()
                    self.wait[v] += t - self.queue_join[v]
                    self.count[r_in] -= 1
                    self.pos[v] += 1
                    self._enter_road(v, r_out)
                    self.credit[m] -= 1.0
        for (r_in, r_out), q in self.free_queues.items():  # unsignalised fallback
            while q and self.count[r_out] < self.cap[r_out]:
                v = q.popleft()
                self.count[r_in] -= 1
                self.pos[v] += 1
                self._enter_road(v, r_out)
        self.t += 1

    # ------------------------------------------------------------ read-outs
    def queue_lengths(self) -> np.ndarray:
        return np.fromiter((len(q) for q in self.queues), dtype=np.int64, count=self.n_mov)

    def metrics(self) -> dict:
        """CityFlow-style aggregate metrics at the current time.

        Travel time counts from the scheduled departure; vehicles still in the
        network (or waiting to enter) contribute their elapsed time so far.
        """
        started = self.depart[: self.next_dep]
        fin = self.finish[: self.next_dep]
        tt = np.where(fin >= 0, fin - started, self.t - started)
        q = self.queue_lengths()
        # waiting time of vehicles currently queued or waiting at entry is added on top of the accrued wait
        w = self.wait[: self.next_dep].astype(float).copy()
        for qq in self.queues:
            for v in qq:
                w[v] += self.t - self.queue_join[v]
        for buf in self.entry.values():
            for v in buf:
                w[v] += self.t - self.depart[v]
        return {
            "time": self.t,
            "vehicles_departed": int(self.next_dep),
            "throughput": int((fin >= 0).sum()),
            "avg_travel_time": float(tt.mean()) if len(tt) else 0.0,
            "avg_waiting_time": float(w.mean()) if len(w) else 0.0,
            "total_queue": int(q.sum()),
            "entry_backlog": int(sum(len(b) for b in self.entry.values())),
        }
