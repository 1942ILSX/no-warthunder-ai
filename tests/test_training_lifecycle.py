from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nowarthunder_ai.checkpoint import load_actor_checkpoint, load_checkpoint, save_checkpoint
from nowarthunder_ai.distillation import distill
from nowarthunder_ai.evaluation import evaluate
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.trainer import collect_rollout, resolved_config, train


class FakeEnv:
    """Fast deterministic barrier simulator; reset zeroes whole-batch rewards."""
    instances = []

    def __init__(self, configs):
        self.configs = list(configs)
        self.n = len(configs)
        self.steps = np.zeros(self.n, dtype=np.int64)
        self.metadata = {"critic_size": 4, "observation_version": 2,
                         "normalization_version": 2, "defender_observation_version": 1}
        self.closed = False
        self.frame = self.make_frame()
        self.instances.append(self)

    def make_frame(self, reward=False):
        done = self.steps >= 2
        enemy_obs = np.zeros((self.n, 5, 320), dtype=np.float32)
        enemy_obs[..., 0] = self.steps[:, None]
        enemy_obs[..., 319] = .75  # distinct role/input feature must survive
        active = np.zeros((self.n, 5), dtype=bool)
        for i in range(self.n):
            active[i, int(self.steps[i] % 2)] = not done[i]
        masks = np.zeros((self.n, 5, 10), dtype=bool)
        masks[..., :3] = True
        return SimpleNamespace(
            enemy_obs=enemy_obs, enemy_mask=masks, enemy_active=active,
            enemy_ids=np.tile(np.arange(5), (self.n, 1)),
            defender_obs=np.zeros((self.n, 320), dtype=np.float32),
            defender_mask=np.ones((self.n, 10), dtype=bool), defender_active=~done,
            critic_obs=np.repeat(self.steps[:, None], 4, axis=1).astype(np.float32),
            enemy_reward=np.full(self.n, .1 if reward else 0, dtype=np.float32),
            defender_reward=np.full(self.n, -.1 if reward else 0, dtype=np.float32),
            terminated=np.zeros(self.n, dtype=bool), truncated=done,
            infos=[{"outcome": "step_limit" if done[i] else "playing", "decision_step": int(self.steps[i]),
                    "level_id": self.configs[i]["level_id"], "seed": self.configs[i]["seed"],
                    "enemy_stuck": 0, "enemy_alive": 1, "score": 0} for i in range(self.n)],
        )

    def reset(self, configs=None):
        if configs:
            self.configs = list(configs)
        self.steps[:] = 0
        self.frame = self.make_frame()
        return self.frame

    def step(self, enemy_actions, defender_actions):
        assert enemy_actions.shape == (self.n, 5)
        assert defender_actions.shape == (self.n,)
        self.steps += self.steps < 2
        self.frame = self.make_frame(reward=True)
        return self.frame

    def reset_indices(self, indices, configs):
        self.steps[indices] = 0
        for index, config in zip(indices, configs):
            self.configs[index] = config
        self.frame = self.make_frame()
        return self.frame

    def close(self):
        self.closed = True


def small_config(**overrides):
    return {"num_envs": 2, "horizon": 4, "iterations": 2, "actor_hidden": [8, 4],
            "critic_hidden": [8, 4], "ppo_epochs": 1, "minibatch_size": 8,
            "checkpoint_every": 1, "league_every": 1,
            "scripted_probability": 1., "latest_probability": 0.,
            "enemy_updates": 1, "defender_updates": 1, "smoke": True,
            "curriculum": [{"until_iteration": 5, "level_ids": [1], "max_steps": 2}],
            **overrides}


def test_collector_copies_final_observation_before_reset_and_only_active_samples():
    config = resolved_config(small_config())
    env = FakeEnv([{"level_id": 1, "seed": 1}] * 2)
    data, frame, episodes = collect_rollout(env, env.frame, PolicyNetwork((8, 4)), None,
                                          role="enemy", config=config, iteration=0, device="cpu")
    assert data.next_critic_observation[0, 1, 0] == 2
    assert data.critic_observation[0, 2, 0] == 0
    assert data.reward[0, 1] > 0  # reset response has zero reward
    assert data.active.sum() == 8
    assert len(episodes) == 4
    assert np.all(frame.critic_obs == 0)


def test_train_resume_checkpoint_league_and_output_protection(tmp_path):
    output = tmp_path / "training"
    result = train(small_config(), FakeEnv, "cpu", output)
    checkpoint = load_checkpoint(result["checkpoint"])
    assert checkpoint["iteration"] == 2
    assert checkpoint["trained"] and checkpoint["kind"] == "smoke"
    assert checkpoint["enemy_optimizer_updates"] > 0
    assert checkpoint["defender_optimizer_updates"] > 0
    assert len(checkpoint["league"]["entries"]["enemy"]) == 2
    before = checkpoint["enemy_optimizer_updates"]
    resumed = train(small_config(iterations=3), FakeEnv, "cpu", output, result["checkpoint"])
    state = load_checkpoint(resumed["checkpoint"])
    assert state["iteration"] == 3 and state["enemy_optimizer_updates"] > before
    assert all(env.closed for env in FakeEnv.instances[-2:])
    with pytest.raises(FileExistsError):
        train(small_config(), FakeEnv, "cpu", output)
    loaded = load_actor_checkpoint(resumed["checkpoint"])
    assert loaded.hidden_sizes == (8, 4)


def test_student_distillation_resume_and_ppo_finetune(tmp_path):
    teacher = train(small_config(iterations=1), FakeEnv, "cpu", tmp_path / "teacher")
    config = small_config(iterations=1, student_hidden=[6, 3], epochs=1, dataset_size=10)
    result = distill(config, FakeEnv, "cpu", tmp_path / "student", teacher["checkpoint"])
    state = load_checkpoint(result["checkpoint"])
    assert state["model_spec"]["enemy_hidden"] == (6, 3)
    assert state["trained"] and state["enemy_optimizer_updates"] > 0
    assert len(state["dataset"]["observations"]) == 8
    assert torch.all(state["dataset"]["observations"][:, 319] == .75)
    result = distill({**config, "iterations": 2}, FakeEnv, "cpu", tmp_path / "student",
                     tmp_path / "teacher-moved.pt", result["checkpoint"])
    assert load_checkpoint(result["checkpoint"])["dataset"]["observations"].shape[0] == 10
    tuned = train(small_config(iterations=1, training_role="enemy"), FakeEnv, "cpu",
                  tmp_path / "finetune", initial_enemy_checkpoint=result["checkpoint"])
    assert load_checkpoint(tuned["checkpoint"])["model_spec"]["enemy_hidden"] == (6, 3)


def test_bounded_evaluation_uses_disjoint_seeds_and_does_not_count_timeouts_as_wins():
    result = evaluate(PolicyNetwork((8, 4)), FakeEnv, "cpu",
                      {"episodes": 3, "num_envs": 2, "max_steps": 2, "opponents": ["guard"]})
    assert result["episodes"] == 3
    assert result["timeout_rate"] == result["truncation_rate"] == 1
    assert result["enemy_win_rate"] == result["defender_win_rate"] == 0
    assert all(record["seed"] < 0 for record in result["records"])


def test_best_candidate_keeps_explicit_ineligible_status(tmp_path):
    result = train(small_config(iterations=1, eval_every=1, eval_episodes=1, eval_max_steps=2,
                                difficulty="trainee", target_defender_win_rate=[.7, .9]),
                   FakeEnv, "cpu", tmp_path / "calibration")
    assert not result["selection"]["calibrated"]
    assert (tmp_path / "calibration" / "best.pt").exists()
    assert not (tmp_path / "calibration" / "best_eligible.pt").exists()


def test_resume_rejects_optimizer_and_calibration_campaign_changes(tmp_path):
    config = small_config(iterations=1, difficulty="expert", target_defender_win_rate=[.4, .6])
    result = train(config, FakeEnv, "cpu", tmp_path / "run")
    for change in ({"learning_rate": .001}, {"league_size": 100},
                   {"target_defender_win_rate": [.1, .3]}, {"eval_seed": 999}):
        with pytest.raises(ValueError, match="Resume config changed"):
            train({**config, **change, "iterations": 2}, FakeEnv, "cpu", tmp_path / "run", result["checkpoint"])


def test_distillation_rejects_random_enemy_teacher_and_profile_mismatch(tmp_path):
    teacher = train(small_config(iterations=1, training_role="defender"), FakeEnv, "cpu", tmp_path / "defender")
    state = load_checkpoint(teacher["checkpoint"])
    assert state["trained"] and state["enemy_optimizer_updates"] == 0
    with pytest.raises(ValueError, match="Teacher enemy actor"):
        distill(small_config(iterations=1), FakeEnv, "cpu", tmp_path / "bad-student", teacher["checkpoint"])
    good = train(small_config(iterations=1, difficulty="trainee"), FakeEnv, "cpu", tmp_path / "teacher")
    with pytest.raises(ValueError, match="must match the teacher"):
        distill(small_config(iterations=1, difficulty="ace"), FakeEnv, "cpu", tmp_path / "wrong-profile", good["checkpoint"])


def test_checkpoint_evaluation_rejects_stale_observation_normalization(tmp_path):
    trained = train(small_config(iterations=1), FakeEnv, "cpu", tmp_path / "teacher")
    state = load_checkpoint(trained["checkpoint"])
    state["normalization_version"] = 1
    old = save_checkpoint(tmp_path / "stale.pt", state)
    with pytest.raises(ValueError, match="normalization_version"):
        evaluate(old, FakeEnv, "cpu", {"episodes": 1, "max_steps": 2})
    assert FakeEnv.instances[-1].closed


def test_finetune_preserves_declared_difficulty_and_target_band(tmp_path):
    profile = {"difficulty": "trainee", "target_defender_win_rate": [.7, .9]}
    source = train(small_config(iterations=1, **profile), FakeEnv, "cpu", tmp_path / "source")
    incompatible = [
        ({**profile, "difficulty": "ace"}, "Fine-tuning difficulty"),
        ({"target_defender_win_rate": [.7, .9]}, "Fine-tuning difficulty"),
        ({**profile, "target_defender_win_rate": [.1, .3]}, "Fine-tuning target_defender_win_rate"),
        ({"difficulty": "trainee"}, "Fine-tuning target_defender_win_rate"),
    ]
    for index, (candidate, message) in enumerate(incompatible):
        output = tmp_path / f"rejected-{index}"
        with pytest.raises(ValueError, match=message):
            train(small_config(iterations=1, **candidate), FakeEnv, "cpu", output,
                  initial_enemy_checkpoint=source["checkpoint"])
        assert FakeEnv.instances[-1].closed
        assert not (output / "latest.pt").exists()

    tuned = train(small_config(iterations=1, training_role="enemy", **profile),
                  FakeEnv, "cpu", tmp_path / "matched",
                  initial_enemy_checkpoint=source["checkpoint"])
    state = load_checkpoint(tuned["checkpoint"])
    assert state["difficulty"] == profile["difficulty"]
    assert state["target_defender_win_rate"] == profile["target_defender_win_rate"]
    assert state["enemy_optimizer_updates"] > 0
