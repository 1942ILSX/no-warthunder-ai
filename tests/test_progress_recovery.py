from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nowarthunder_ai.difficulty import DifficultyController, control_config, difficulty_actor_loss
from nowarthunder_ai.learning import actor_samples, room_returns
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.progress import ProgressTracker, progress_config
from nowarthunder_ai.trainer import collect_rollout, resolved_config
from test_learning_core import rollout, identity_critic


def frame(step=0, enemy_id=147, seed=-3000019):
    obs = np.zeros((1, 5, 320), dtype=np.float32)
    phase = step % 20
    y = 10.5 + min(phase, 20 - phase) * .188
    obs[0, 0, :2] = np.array([7.5, y]) / 24
    obs[0, 0, 2] = 1
    obs[0, 0, 6] = 1
    obs[0, 0, 8] = .25
    obs[0, 0, 208] = 7.63 / 24
    obs[0, 0, 217] = 1
    obs[0, 0, 290:294] = [.1, .31, .12, .12]
    obs[0, 0, 294:298] = [.12, .3, .1, .12]
    obs[0, 0, 308] = 2 / 3
    active = np.zeros((1, 5), dtype=bool)
    active[0, 0] = True
    ids = np.zeros((1, 5), dtype=np.int32)
    ids[0, 0] = enemy_id
    return SimpleNamespace(enemy_obs=obs, enemy_active=active, enemy_ids=ids,
                           critic_obs=np.zeros((1, 2688), dtype=np.float32),
                           infos=[{"level_id": 9, "seed": seed, "decision_step": step,
                                   "base_hp": 3, "lives": 3, "kills": 13}])


def test_replay_shaped_moving_loop_is_detected_despite_no_stuck_or_stationary_signal():
    tracker = ProgressTracker((1, 5))
    found = []
    for step in range(180):
        item = frame(step)
        item.enemy_obs[0, 0, 9] = step % 12 == 0  # empty-lane firing is not progress
        # A scrolling local terrain view must not count as destroying terrain.
        item.enemy_obs[0, 0, 12] = step % 2
        result = tracker.update(item)
        found.append(bool(result[0, 0]))
        assert not result[0, 1:].any()
    assert not any(found[:119])
    assert all(found[119:])
    before = tracker.positions.nbytes + tracker.navigation.nbytes
    for step in range(180, 500):
        tracker.update(frame(step))
    assert tracker.positions.nbytes + tracker.navigation.nbytes == before


@pytest.mark.parametrize("kind", ["travel", "fire_at_target", "close_evasion", "navigation"])
def test_normal_travel_combat_and_new_goal_progress_are_not_loops(kind):
    tracker = ProgressTracker((1, 5))
    found = []
    for step in range(120):
        item = frame(step)
        if kind == "travel":
            item.enemy_obs[0, 0, 1] = (5 + step * .12) / 24
        elif kind == "fire_at_target":
            item.enemy_obs[0, 0, 313] = 1
            item.enemy_obs[0, 0, 9] = step % 10 == 0
        elif kind == "close_evasion":
            item.enemy_obs[0, 0, 208] = .08
            item.enemy_obs[0, 0, 230] = 1  # hostile projectile existence
        else:
            item.enemy_obs[0, 0, 290:294] = .3 if step < 60 else .15
        found.append(tracker.update(item)[0, 0])
    assert not any(found)


@pytest.mark.parametrize("kind", ["brick", "self_hp", "base_hp", "kills"])
def test_actual_terrain_and_health_progress_resets_no_progress_window(kind):
    tracker = ProgressTracker((1, 5))
    found = []
    for step in range(160):
        item = frame(step)
        if kind == "brick":
            item.critic_obs[0, 0] = step < 80
        elif kind == "self_hp":
            item.enemy_obs[0, 0, 6] = 1 if step < 80 else .5
        elif kind == "base_hp":
            item.infos[0]["base_hp"] = 3 if step < 80 else 2
        else:
            item.infos[0]["kills"] = 13 if step < 80 else 14
        found.append(tracker.update(item)[0, 0])
    assert not any(found)


def test_death_identity_and_episode_reset_clear_loop_labels():
    tracker = ProgressTracker((1, 5))
    for step in range(120):
        tracker.update(frame(step))
    assert tracker.update(frame(120))[0, 0]
    assert not tracker.update(frame(121, enemy_id=148))[0, 0]
    for step in range(122, 245):
        tracker.update(frame(step, enemy_id=148))
    assert tracker.update(frame(245, enemy_id=148))[0, 0]
    assert not tracker.update(frame(0, enemy_id=148, seed=99))[0, 0]
    for step in range(1, 125):
        tracker.update(frame(step, enemy_id=148, seed=99))
    assert tracker.update(frame(125, enemy_id=148, seed=99))[0, 0]
    item = frame(126, enemy_id=148, seed=99)
    item.enemy_active[:] = False
    assert not tracker.update(item).any()
    assert not tracker.update(frame(127, enemy_id=148, seed=99)).any()
    tracker.reset_indices([0])
    assert not tracker.update(frame(128, enemy_id=148, seed=99)).any()


def test_temporal_label_adds_recovery_gradient_without_changing_actor_input():
    obs = torch.from_numpy(frame().enemy_obs[0, :1].copy())
    obs[:, 8] = .75  # moving south, no static stuck/idle trigger
    obs[:, 316] = .3
    config = control_config({})
    budgets = {"fire_budget": .5, "turn_fire_budget": .2}
    logits = torch.zeros(1, 10, requires_grad=True)
    mask = torch.ones(1, 10, dtype=torch.bool)
    legacy, _ = difficulty_actor_loss(logits, obs, mask, config, budgets)
    assert legacy == 0
    loss, metrics = difficulty_actor_loss(logits, obs, mask, config, budgets,
                                          recovery_mask=torch.tensor([True]))
    loss.backward()
    assert metrics["difficulty_temporal_recovery_samples"] == 1
    assert logits.grad[0, 2] < 0 and logits.grad[0, 3] < 0
    assert logits.grad[0, 6] > 0
    assert obs.shape[-1] == 320


def test_actor_sample_filter_preserves_optional_temporal_labels():
    active = torch.zeros(1, 2, 5, dtype=torch.bool)
    active[0, 0, 0] = True
    active[0, 1, 1] = True
    data = rollout(active)
    data.recovery_mask = torch.ones_like(active)
    data.recovery_mask[0, 1, 1] = False
    returns = room_returns(data, identity_critic(), gamma=1, gae_lambda=1)
    assert actor_samples(data, returns)["recovery_mask"].tolist() == [True, False]
    data.recovery_mask = None
    assert "recovery_mask" not in actor_samples(data, returns).keys()


class LoopEnv:
    def __init__(self):
        self.step_number = 0
        self.frame = self.make_frame()

    def make_frame(self):
        item = frame(self.step_number)
        item.enemy_obs[0, 0, 1] = (10.5 + self.step_number % 2) / 24
        item.enemy_mask = np.ones((1, 5, 10), dtype=bool)
        item.defender_obs = np.zeros((1, 320), dtype=np.float32)
        item.defender_mask = np.ones((1, 10), dtype=bool)
        item.defender_active = np.ones(1, dtype=bool)
        item.enemy_reward = item.defender_reward = np.zeros(1, dtype=np.float32)
        item.terminated = item.truncated = np.zeros(1, dtype=bool)
        return item

    def step(self, *_):
        self.step_number += 1
        self.frame = self.make_frame()
        return self.frame


def test_collector_keeps_history_across_rollout_boundaries():
    settings = {"window": 8, "check_every": 1, "min_path_tiles": 3.}
    config = resolved_config({"horizon": 4, "progress_recovery": settings})
    env = LoopEnv()
    tracker = ProgressTracker((1, 5), settings)
    actor = PolicyNetwork((8, 4))
    first, current, _ = collect_rollout(env, env.frame, actor, None, role="enemy", config=config,
                                       iteration=0, device="cpu", progress_tracker=tracker)
    second, _, _ = collect_rollout(env, current, actor, None, role="enemy", config=config,
                                   iteration=1, device="cpu", progress_tracker=tracker)
    assert not first.recovery_mask.any()
    assert second.recovery_mask[0, -1, 0]
    assert not second.recovery_mask[0, :, 1:].any()


def test_saturated_effective_budgets_do_not_claim_updates_and_can_later_reduce():
    config = {"fire_budget": 1., "turn_fire_budget": 1.}
    assessment = {"quality_passed": True, "defender_win_rate": .9,
                  "target_defender_win_rate": [.1, .3]}
    controller = DifficultyController(config)
    for _ in range(30):
        controller.update(assessment)
    assert controller.updates == 0 and controller.budget_scale == 1.
    assert controller.last_status == "at_effective_bound"
    controller = DifficultyController(config, {"budget_scale": 2., "updates": 7})
    controller.update({**assessment, "defender_win_rate": 0.})
    assert controller.updates == 8 and controller.budget_scale < 1.
    assert all(value < 1 for value in controller.effective_budgets.values())


@pytest.mark.parametrize("bad", [{"window": 7}, {"window": 601}, {"check_every": 0},
                                   {"region_tiles": float("nan")}, {"min_revisit_fraction": 0},
                                   {"unknown": 1}])
def test_progress_config_rejects_invalid_or_unbounded_settings(bad):
    with pytest.raises(ValueError):
        progress_config(bad)
