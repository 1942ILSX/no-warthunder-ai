from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from nowarthunder_ai.evaluation import assess_report, calibration_metrics, evaluate, summarize_records
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.progress import progress_config


def records():
    rows = []
    for opponent in ("guard", "hunter", "wander"):
        for sample in range(50):
            rows.append({"opponent": opponent, "level_id": sample % 10 + 1,
                         "outcome": "defender_win" if sample < 40 else "enemy_win_base",
                         "enemy_alive_steps": 100, "enemy_stuck_steps": 0,
                         "enemy_observed_steps": 100, "enemy_unproductive_steps": 0,
                         "enemy_loop_observed_steps": 100, "enemy_loop_steps": 0,
                         "truncated": False})
    return rows


def config(**extra):
    return {"calibration_version": 3, "difficulty": "trainee", "target_defender_win_rate": [.7, .9], **extra}


def report(rows):
    return {**summarize_records(rows, version=3), "records": rows,
            "progress_recovery": progress_config()}


def test_v3_keeps_v2_bands_and_adds_only_measured_loop_quality():
    rows = records()
    for row in rows:
        if row["opponent"] == "wander":
            row["enemy_loop_steps"] = 100
    measured = report(rows)
    assessment = calibration_metrics(measured, config())
    assert assessment["eligible"] and assessment["calibration_version"] == 3
    assert measured["moving_no_progress_rate"] == pytest.approx(1 / 3)
    assert measured["primary"]["moving_no_progress_rate"] == 0
    assert measured["by_level_opponent"]["1"]["wander"]["moving_no_progress_rate"] == 1
    assert assessment["limits"]["primary"]["moving_no_progress_rate"] == .05


def test_v3_detects_one_problem_level_hidden_by_aggregate():
    rows = records()
    for row in rows:
        if row["opponent"] != "wander" and row["level_id"] == 9:
            row["enemy_loop_steps"] = 30
    measured = report(rows)
    assert measured["primary"]["moving_no_progress_rate"] == .03
    assessment = calibration_metrics(measured, config())
    assert not assessment["eligible"]
    assert assessment["worst_loop_level"] == "9"
    assert assessment["reasons"] == ["level:9.moving_no_progress_rate:quality_limit_exceeded"]


def test_v2_reassessment_never_fabricates_v3_loop_telemetry():
    rows = records()
    for row in rows:
        row.pop("enemy_loop_steps")
        row.pop("enemy_loop_observed_steps")
    original = {**summarize_records(rows, version=2), "records": rows}
    before = deepcopy(original)
    assessed = assess_report(original, config())
    assert original == before
    assert assessed["primary"]["moving_no_progress_rate"] is None
    assert not assessed["calibration"]["eligible"]
    assert "primary.moving_no_progress_rate:missing_moving_no_progress_telemetry" in assessed["calibration"]["reasons"]
    assert assessed["source_metrics_version"] == 2


@pytest.mark.parametrize("bad", [-1, 101, float("nan")])
def test_v3_rejects_invalid_loop_counts(bad):
    rows = records()
    rows[0]["enemy_loop_steps"] = bad
    assert not calibration_metrics(report(rows), config())["quality_passed"]


def test_v3_requires_the_declared_detector_settings():
    measured = report(records())
    measured["progress_recovery"]["window"] += 1
    assessment = calibration_metrics(measured, config())
    assert not assessment["eligible"]
    assert "primary.progress_recovery:missing_or_mismatched_detector_config" in assessment["reasons"]


class LoopEnv:
    def __init__(self, configs):
        self.steps = 0
        self.metadata = {}
        self.frame = self.make_frame()

    def make_frame(self):
        done = self.steps >= 25
        observation = np.zeros((1, 5, 320), dtype=np.float32)
        observation[0, 0, :2] = [(12 + self.steps % 2) / 24, 12 / 24]
        observation[..., 2] = 1
        active = np.zeros((1, 5), dtype=bool)
        active[0, 0] = not done
        ids = np.zeros((1, 5), dtype=np.int64)
        ids[0, 0] = 7 if self.steps < 12 else 8
        return SimpleNamespace(
            enemy_obs=observation, enemy_active=active, enemy_ids=ids,
            enemy_mask=np.ones((1, 5, 10), dtype=bool),
            defender_obs=np.zeros((1, 320), dtype=np.float32),
            defender_mask=np.ones((1, 10), dtype=bool), defender_active=np.array([not done]),
            enemy_reward=np.zeros(1), terminated=np.array([done]), truncated=np.array([False]),
            infos=[{"outcome": "defender_win" if done else "playing", "level_id": 1,
                    "decision_step": self.steps, "enemy_alive": 1, "enemy_stuck": 0}])

    def step(self, *_):
        self.steps += 1
        self.frame = self.make_frame()
        return self.frame

    def close(self):
        pass


def test_evaluate_uses_shared_tracker_and_keeps_per_enemy_counts_separate():
    actor = PolicyNetwork((8, 4))
    settings = {"window": 8, "check_every": 1, "min_path_tiles": 4}
    parameters = {"episodes": 1, "num_envs": 1, "max_steps": 25,
                  "opponents": ["guard"], "progress_recovery": settings}
    legacy = evaluate(actor, LoopEnv, "cpu", {**parameters, "calibration_version": 2})
    assert "enemy_loop_steps" not in legacy["records"][0]
    measured = evaluate(actor, LoopEnv, "cpu", {**parameters, "calibration_version": 3})
    row = measured["records"][0]
    assert row["enemy_loop_observed_steps"] == 25
    assert row["enemy_loop_steps"] == 11
    assert [(item["enemy_id"], item["observed_steps"], item["moving_no_progress_steps"])
            for item in row["enemy_progress"]] == [(7, 12, 5), (8, 13, 6)]
    assert measured["progress_recovery"] == progress_config(settings)
    assert measured["primary"]["moving_no_progress_rate"] == 11 / 25
