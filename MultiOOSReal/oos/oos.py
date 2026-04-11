from __future__ import annotations
import numpy as np

from physics.collision import collision_probability
from config import PC_THRESHOLD

EPSILON = 0.01
INTENT_EVERY = 1

DOCKED_M = 100
AVOID_DV_MS = 0.5


class OOS:

    def __init__(self, r, v, id, sensing_radius=5_000_000.0, wait_duration=300.0):
        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.id = id

        self.sensing_radius = sensing_radius
        self.wait_duration = wait_duration

        self.state = "IDLE"
        self.target = None
        self.target_score = 0.0

        self.inbox = []
        self.competitors = {}
        self.wait_start = 0.0
        self._intent_step = 0

        self._timeout_counter = 0
        self.failed_targets = {}

        self.metrics = None
        self.mission_start_time = None

        self.prev_dist = None
        self.divergence_counter = 0

    # ============================================================

    def step(self, env_state, current_time, network, fleet, completed_targets, active_collisons):

        self.process_inbox()

        if self.target is not None:
            if self.state in ["WAITING_DOMINANCE", "PLANNING"]:
                if not self._is_dominant():
                    print(f"OOS{self.id} YIELD on {self.target}")
                    self.failed_targets[self.target] = current_time
                    self._reset_mission()
                    return

        if self.target is not None:
            intent = self._make_intent(current_time)
            network.broadcast(intent, self.r, current_time, fleet)

        handler = {
            "IDLE": self._state_idle,
            "WAITING_DOMINANCE": self._state_waiting,
            "PLANNING": self._state_planning,
            "TRANSFER": self._state_transfer,
            "VEL_MATCH": self._state_vel_match,
            "DOCKING": self._state_docking,
            "MANEUVER": self._state_maneuver,
            "HOLD": self._state_hold,
        }.get(self.state)

        if handler:
            handler(env_state, current_time, network, fleet, completed_targets, active_collisons)

        self._intent_step += 1

    # ============================================================

    def _state_idle(self, env_state, current_time, network, fleet, completed_targets, active_collisions):
        visible = self._sense(env_state)
        target, score = self._select_target(visible, current_time, completed_targets, active_collisions)

        if target is None:
            return

        self.target = target
        self.target_score = score
        self.competitors = {}
        self.wait_start = current_time
        self.state = "WAITING_DOMINANCE"

        print(f"OOS{self.id} START MISSION → {target}")

        if self.metrics:
            self.metrics.log_mission_start()
            self.mission_start_time = current_time

        intent = self._make_intent(current_time)
        network.broadcast(intent, self.r, current_time, fleet)

    def _state_waiting(self, env_state, current_time, network, fleet, completed_targets, *_):

        if self._intent_step % INTENT_EVERY == 0:
            intent = self._make_intent(current_time)
            network.broadcast(intent, self.r, current_time, fleet)

        if current_time - self.wait_start < self.wait_duration:
            return

        if self._is_dominant():
            if self.metrics:
                self.metrics.log_wait_time(current_time - self.wait_start)

            print(f"OOS{self.id} PLANNING → {self.target}")
            self.state = "PLANNING"
        else:
            self._reset_mission()

    def _state_planning(self, env_state, *_):
        if self.target not in env_state:
            self._reset_mission()
            return

        self.state = "TRANSFER"
        self._timeout_counter = 0

    def _state_transfer(self, env_state, current_time, *_):
        target = env_state[self.target]

        rel_r = target["r"] - self.r
        rel_v = target["v"] - self.v

        dist = np.linalg.norm(rel_r)

        if self._check_divergence(dist):
            return

        kp = 5e-6
        kd = 1e-3

        dv = -kp * rel_r - kd * rel_v

        if np.linalg.norm(dv) > 0.3:
            dv = dv / np.linalg.norm(dv) * 0.3

        self.v += dv

        if self.metrics:
            self.metrics.log_dv(dv)

        if dist < 3000:
            self.state = "VEL_MATCH"

    def _state_vel_match(self, env_state, *_):
        self.state = "DOCKING"
        self._timeout_counter = 0

    def _state_docking(self, env_state, current_time, *_):
        target = env_state[self.target]

        rel_r = target["r"] - self.r
        dist = np.linalg.norm(rel_r)

        if self._check_divergence(dist):
            return

        if dist < DOCKED_M:
            self.state = "MANEUVER"
            return

        approach_speed = np.clip(dist * 5e-4, 0.05, 2.0)
        direction = rel_r / dist

        new_v = target["v"] + direction * approach_speed
        dv = new_v - self.v

        self.v = new_v

        if self.metrics:
            self.metrics.log_dv(dv)

        self._timeout_counter += 1
        if self._timeout_counter > 3000:
            print(f"OOS{self.id} DOCKING TIMEOUT → {self.target}")
            if self.metrics:
                self.metrics.log_failure()
            self._reset_mission()

    def _state_maneuver(self, env_state, *_):
        debris_key = self.target + "_DEBRIS"

        if debris_key in env_state:
            sat = env_state[self.target]
            deb = env_state[debris_key]

            diff = sat["r"] - deb["r"]
            direction = diff / (np.linalg.norm(diff) + 1e-9)

            dv = direction * AVOID_DV_MS
            sat["v"] += dv

            if self.metrics:
                self.metrics.log_dv(dv)
                self.metrics.log_collision_avoided()

        self.state = "HOLD"

    def _state_hold(self, env_state, current_time, _, __, completed_targets, active_collisions):

        debris_key = self.target + "_DEBRIS"

        if self.target not in env_state or debris_key not in env_state:
            self._reset_mission()
            return

        pc, _ = collision_probability(env_state[self.target], env_state[debris_key])

        if pc < PC_THRESHOLD:
            print(f"OOS{self.id} COMPLETE → {self.target}")

            if self.metrics and self.mission_start_time is not None:
                self.metrics.log_mission_complete(
                    current_time - self.mission_start_time
                )

            completed_targets.add(self.target)
            self._reset_mission()
        else:
            self.state = "MANEUVER"

    # ============================================================

    def _sense(self, env_state):
        return {
            k: v for k, v in env_state.items()
            if np.linalg.norm(v["r"] - self.r) <= self.sensing_radius
        }

    def _select_target(self, visible, current_time, completed_targets, active_collisions):
        best_score = -1
        best = None

        for sat in visible:
            if "_DEBRIS" in sat:
                continue

       

            if sat in completed_targets:
                continue

            if sat in self.failed_targets:
                if current_time - self.failed_targets[sat] < 2000:
                    continue

            debris = sat + "_DEBRIS"
            if debris not in visible:
                continue

            pc, _ = collision_probability(visible[sat], visible[debris])
            if pc < PC_THRESHOLD:
                continue

            dist = np.linalg.norm(visible[sat]["r"] - self.r)
            score = pc + 1.0 / (dist + 1e-6)

            if score > best_score:
                best_score = score
                best = sat

        return best, best_score

    def _make_intent(self, t):
        return {
            "type": "INTENT",
            "oos_id": self.id,
            "target": self.target,
            "score": self.target_score,
            "timestamp": t,
        }

    def process_inbox(self):
        for msg in self.inbox:
            if msg.get("type") != "INTENT":
                continue

            t = msg["target"]
            s = msg["score"]
            i = msg["oos_id"]

            if t not in self.competitors:
                self.competitors[t] = (s, i)
            else:
                if s > self.competitors[t][0]:
                    self.competitors[t] = (s, i)

        self.inbox.clear()

    def _is_dominant(self):
        rival = self.competitors.get(self.target)
        if rival is None:
            return True

        score, oid = rival

        if score > self.target_score + EPSILON:
            return False

        if abs(score - self.target_score) <= EPSILON:
            return self.id < oid

        return True

    def _check_divergence(self, dist):
        if self.prev_dist is None:
            self.prev_dist = dist
            return False

        if dist > self.prev_dist:
            self.divergence_counter += 1
        else:
            self.divergence_counter = 0

        self.prev_dist = dist

        if self.divergence_counter > 50:
            if self.metrics:
                self.metrics.log_failure()

            print(f"OOS{self.id} DIVERGENCE → {self.target}")
            self._reset_mission()
            return True

        return False

    def _reset_mission(self):
        self.target = None
        self.state = "IDLE"
        self._timeout_counter = 0
        self.prev_dist = None
        self.divergence_counter = 0
        self.mission_start_time = None