"""Bounded temporal labels for moving limit cycles during training/evaluation.

This observer is never exported or inserted into runtime action selection. It
identifies repeated travel in a small region without goal/combat progress; the
learner receives a recovery label on its existing 320-feature observation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np


PROGRESS_DEFAULTS = {
    "version": 1, "window": 120, "check_every": 5, "region_tiles": 3.,
    "min_path_tiles": 6., "cell_tiles": .5, "min_revisit_fraction": .5,
    "nav_progress_epsilon": .015625, "combat_fraction": .025,
}


def progress_config(value: Mapping | None = None) -> dict:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("progress_recovery must be a mapping")
    value = dict(value or {})
    unknown = set(value) - set(PROGRESS_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown progress_recovery keys: {sorted(unknown)}")
    result = {**PROGRESS_DEFAULTS, **value}
    if type(result["version"]) is not int or result["version"] != 1:
        raise ValueError("Unsupported progress_recovery version")
    if type(result["window"]) is not int or not 8 <= result["window"] <= 600:
        raise ValueError("Progress window must be an integer in 8..600")
    if type(result["check_every"]) is not int or not 1 <= result["check_every"] <= result["window"]:
        raise ValueError("check_every must be an integer in 1..window")
    for key in set(PROGRESS_DEFAULTS) - {"version", "window", "check_every"}:
        number = result[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"progress_recovery.{key} must be finite")
        result[key] = float(number)
    if not 0 < result["region_tiles"] <= 24 or not 0 < result["cell_tiles"] <= result["region_tiles"]:
        raise ValueError("Progress region/cell sizes are invalid")
    if result["min_path_tiles"] <= 0 or not 0 <= result["nav_progress_epsilon"] <= 1:
        raise ValueError("Progress path/nav thresholds are invalid")
    if not 0 < result["min_revisit_fraction"] <= 1 or not 0 <= result["combat_fraction"] <= 1:
        raise ValueError("Progress recurrence/combat fractions are invalid")
    return result


class ProgressTracker:
    """Fixed-size per-enemy history; identity/death/reset never leak labels.

    A recovery label requires: a full history, substantial movement, a confined
    bounding box, repeated visits to earlier cells, no improved navigation
    frontier, no actual health/terrain progress, and little meaningful combat.
    Broad travel, fresh navigation progress, brick clearing and active fighting
    are therefore not treated as loops. This is a conservative detector, not a
    proof that every unlabeled trajectory is productive.
    """
    def __init__(self, shape: tuple[int, int], config: Mapping | None = None):
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("ProgressTracker needs a positive [room,slot] shape")
        self.config = progress_config(config)
        self.shape = tuple(shape)
        window = self.config["window"]
        self.positions = np.zeros((window, *shape, 2), dtype=np.float32)
        self.navigation = np.ones((window, *shape, 2), dtype=np.float32)
        self.progress = np.zeros((window, *shape), dtype=bool)
        self.combat = np.zeros((window, *shape), dtype=bool)
        self.ids = np.zeros(shape, dtype=np.int64)
        self.count = np.zeros(shape, dtype=np.int32)
        self.hp = np.zeros(shape, dtype=np.float32)
        self.cached = np.zeros(shape, dtype=bool)
        self.cursor = 0
        self.calls = 0
        self.episode_keys: list[tuple | None] = [None] * shape[0]
        self.decision_steps = np.full(shape[0], -1, dtype=np.int64)
        self.room_health = np.full((shape[0], 3), np.nan)
        self.bricks = np.full(shape[0], np.nan)

    def reset_indices(self, indices: Sequence[int]) -> None:
        for index in indices:
            index = int(index)
            if not 0 <= index < self.shape[0]:
                raise ValueError("Invalid progress tracker room index")
            self.ids[index] = 0
            self.count[index] = 0
            self.cached[index] = False
            self.episode_keys[index] = None
            self.decision_steps[index] = -1
            self.room_health[index] = np.nan
            self.bricks[index] = np.nan

    def _room_progress(self, frame) -> np.ndarray:
        infos = getattr(frame, "infos", None) or [{} for _ in range(self.shape[0])]
        health = np.array([[info.get("base_hp", np.nan), info.get("lives", np.nan),
                            info.get("kills", np.nan)] for info in infos], dtype=np.float64)
        changed = (np.isfinite(health) & np.isfinite(self.room_health)
                   & (np.abs(health - self.room_health) > 1e-6)).any(axis=-1)
        self.room_health = health
        critic = getattr(frame, "critic_obs", None)
        if critic is not None and critic.shape[-1] >= 2304:
            # Global brick destruction is real progress. A scrolling local
            # terrain window changing as a tank moves is deliberately ignored.
            bricks = np.asarray(critic[:, :2304:4]).sum(axis=-1)
            changed |= np.isfinite(self.bricks) & (bricks < self.bricks - .5)
            self.bricks = bricks.copy()
        return changed

    def update(self, frame, complete: np.ndarray | None = None) -> np.ndarray:
        obs = np.asarray(frame.enemy_obs)
        ids = np.asarray(frame.enemy_ids)
        if ids.shape != self.shape or obs.shape != (*self.shape, 320):
            raise ValueError("Progress tracker frame shape changed")
        complete = np.zeros(self.shape[0], dtype=bool) if complete is None else np.asarray(complete, dtype=bool)
        if complete.shape != (self.shape[0],):
            raise ValueError("Progress complete mask has the wrong shape")
        infos = getattr(frame, "infos", None) or [{} for _ in range(self.shape[0])]
        reset_rooms = []
        for room, info in enumerate(infos):
            key = (info.get("level_id"), info.get("seed"))
            step = int(info.get("decision_step", -1))
            if (self.episode_keys[room] is not None and key != self.episode_keys[room]
                    or step >= 0 and self.decision_steps[room] >= 0 and step < self.decision_steps[room]):
                reset_rooms.append(room)
            self.episode_keys[room] = key
            self.decision_steps[room] = step
        if reset_rooms:
            self.reset_indices(reset_rooms)
            for room in reset_rooms:
                self.episode_keys[room] = (infos[room].get("level_id"), infos[room].get("seed"))
                self.decision_steps[room] = int(infos[room].get("decision_step", -1))
        active = np.asarray(frame.enemy_active, dtype=bool) & (ids != 0) & ~complete[:, None]
        same = active & (ids == self.ids)
        self.count = np.where(same, np.minimum(self.count + 1, self.config["window"]), active.astype(np.int32))
        self.cached &= same
        self.ids = np.where(active, ids, 0)
        room_progress = self._room_progress(frame)
        actual_progress = (room_progress[:, None] | (same & (obs[..., 6] < self.hp - 1e-5))) & active
        self.hp = obs[..., 6].copy()
        self.cached &= ~actual_progress

        role = np.rint(obs[..., 308] * 3).astype(np.int32)
        use_base = (role == 0) | (obs[..., 217] < .5) | ((role == 3) & (np.abs(obs[..., 208:210]).sum(axis=-1) > .32))
        base_nav = obs[..., 294:298].min(axis=-1)
        goal_nav = np.where(use_base, base_nav, obs[..., 290:294].min(axis=-1))
        facing = obs[..., 2:6].argmax(axis=-1)
        aim = np.take_along_axis(obs[..., 313:317], facing[..., None], axis=-1)[..., 0]
        purposeful_fire = (obs[..., 9] > .5) & (aim >= .1)
        bullet_present = (obs[..., 230:266:7] > .5).any(axis=-1)
        close_evasion = ((obs[..., 217] > .5) & (np.square(obs[..., 208:210]).sum(axis=-1) < .12**2)
                         & (bullet_present | (obs[..., 309:313] > .5).any(axis=-1)))
        combat = (purposeful_fire | close_evasion) & active
        self.positions[self.cursor] = obs[..., :2] * 24
        self.navigation[self.cursor, ..., 0] = goal_nav
        self.navigation[self.cursor, ..., 1] = base_nav
        self.progress[self.cursor] = actual_progress
        self.combat[self.cursor] = combat
        self.cursor = (self.cursor + 1) % self.config["window"]
        self.calls += 1
        # Newly observed genuine progress/combat clears cached labels promptly.
        self.cached &= ~combat
        if self.calls % self.config["check_every"] == 0:
            self._classify(active)
        return (self.cached & active).copy()

    def _classify(self, active: np.ndarray) -> None:
        window = self.config["window"]
        order = (np.arange(window) + self.cursor) % window
        positions = self.positions[order]
        navigation = self.navigation[order]
        span = np.ptp(positions, axis=0).max(axis=-1)
        path = np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum(axis=0)
        half = window // 2
        first_frontier = navigation[:half].min(axis=0)
        last_frontier = navigation[half:].min(axis=0)
        advanced = ((first_frontier - last_frontier) > self.config["nav_progress_epsilon"] + 1e-6).any(axis=-1)
        # Membership of late quantized positions in the earlier half captures
        # revisits without an unbounded dictionary or tank/device round-trips.
        cells = np.floor(positions / self.config["cell_tiles"]).astype(np.int32)
        width = int(math.ceil(24 / self.config["cell_tiles"])) + 2
        encoded = cells[..., 0] * width + cells[..., 1]
        # Tag cells by fixed slot column before membership: O(window*slots)
        # storage, rather than a quadratic pairwise comparison tensor.
        flat = encoded.reshape(window, -1)
        keys = flat.astype(np.int64) * flat.shape[1] + np.arange(flat.shape[1])
        revisited = np.isin(keys[half:], keys[:half].reshape(-1), kind="sort").mean(axis=0).reshape(self.shape)
        combat_fraction = self.combat.mean(axis=0)
        meaningful_combat = (combat_fraction > 0) & (combat_fraction >= self.config["combat_fraction"])
        self.cached = (active & (self.count >= window)
                       & (span <= self.config["region_tiles"])
                       & (path >= self.config["min_path_tiles"])
                       & (revisited >= self.config["min_revisit_fraction"])
                       & ~advanced & ~self.progress.any(axis=0) & ~meaningful_combat)
