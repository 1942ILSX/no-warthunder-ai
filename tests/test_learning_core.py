import numpy as np
import pytest
import torch

from nowarthunder_ai.distillation import masked_distillation_loss
from nowarthunder_ai.evaluation import calibration_metrics
from nowarthunder_ai.learning import PPOLearner, Rollout, actor_samples, room_returns
from nowarthunder_ai.models import CentralCritic, PolicyNetwork, act


def rollout(active=None):
    n, t, slots = 1, 2, 5
    return Rollout(
        observation=torch.zeros(n, t, slots, 320),
        action_mask=torch.ones(n, t, slots, 10, dtype=torch.bool),
        active=torch.ones(n, t, slots, dtype=torch.bool) if active is None else active,
        action=torch.zeros(n, t, slots, dtype=torch.int64),
        log_prob=torch.full((n, t, slots), -np.log(10)),
        critic_observation=torch.tensor([[[0.], [1.]]]),
        next_critic_observation=torch.tensor([[[1.], [2.]]]),
        reward=torch.zeros(n, t), terminated=torch.zeros(n, t, dtype=torch.bool),
        truncated=torch.zeros(n, t, dtype=torch.bool),
    )


def identity_critic():
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1)
    return critic


def test_room_gae_continues_across_enemy_death_and_masks_actor_slots():
    active = torch.zeros(1, 2, 5, dtype=torch.bool)
    active[0, 0, 0] = True
    active[0, 1, 1] = True  # another tank, same team trajectory
    data = rollout(active)
    returns = room_returns(data, identity_critic(), gamma=1, gae_lambda=1)
    torch.testing.assert_close(returns["advantage"], torch.tensor([[[2.], [1.]]]))
    samples = actor_samples(data, returns)
    assert len(samples) == 2
    torch.testing.assert_close(samples["advantage"], torch.tensor([[2.], [1.]]))


def test_truncation_bootstraps_final_state_and_stops_before_reset_episode():
    data = rollout()
    data.critic_observation = torch.tensor([[[0.], [100.]]])
    data.next_critic_observation = torch.tensor([[[7.], [999.]]])
    data.truncated[0, 0] = True
    data.terminated[0, 1] = True
    returns = room_returns(data, identity_critic(), gamma=1, gae_lambda=1)
    torch.testing.assert_close(returns["value_target"], torch.tensor([[[7.], [0.]]]))


def test_all_inactive_still_updates_room_critic_without_actor_updates():
    torch.manual_seed(4)
    actor = PolicyNetwork((8, 4))
    critic = CentralCritic(1, (8, 4))
    learner = PPOLearner(actor, critic, {"ppo_epochs": 1})
    before_actor = [p.clone() for p in actor.parameters()]
    before_critic = [p.clone() for p in critic.parameters()]
    data = rollout(torch.zeros(1, 2, 5, dtype=torch.bool))
    data.reward.fill_(1)
    metrics = learner.update(data)
    assert metrics["actor_samples"] == 0
    assert metrics["room_samples"] == 2
    assert metrics["actor_optimizer_steps"] == 0
    assert all(torch.equal(before, after) for before, after in zip(before_actor, actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(before_critic, critic.parameters()))


def test_masked_batch_inference_excludes_inactive_and_preserves_features():
    policy = PolicyNetwork((8, 4))
    obs = np.random.default_rng(3).normal(size=(2, 5, 320)).astype(np.float32)
    saved = obs.copy()
    active = np.zeros((2, 5), dtype=bool)
    active[:, 0] = True
    mask = np.zeros((2, 5, 10), dtype=bool)
    mask[:, 0, 7] = True
    actions, _, logp = act(policy, obs, mask, active, "cpu")
    assert np.all(actions[:, 0] == 7)
    assert np.all(actions[:, 1:] == -1)
    assert torch.isfinite(logp).all()
    np.testing.assert_array_equal(obs, saved)
    mask[0, 0] = False
    with pytest.raises(ValueError, match="no legal action"):
        act(policy, obs, mask, active, "cpu")


def test_distillation_masks_illegal_logits_and_keeps_finite_gradients():
    student = torch.tensor([[1., 3., 1e7]], requires_grad=True)
    teacher = torch.tensor([[3., 1., -1e7]])
    loss = masked_distillation_loss(student, teacher, torch.tensor([[True, True, False]]), 2)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert student.grad[0, 2] == 0
    same = masked_distillation_loss(teacher, teacher, torch.tensor([[True, True, False]]))
    assert abs(float(same)) < 1e-6


def test_difficulty_selection_uses_band_and_quality_not_max_enemy_wins():
    config = {"difficulty": "trainee", "target_defender_win_rate": [.7, .9]}
    base = {"defender_win_rate": .8, "timeout_rate": 0., "stagnation_rate": .1,
            "episodes": 30, "enemy_alive_steps": 100}
    good = calibration_metrics(base, config)
    too_hard = calibration_metrics({**base, "defender_win_rate": .1}, config)
    stuck = calibration_metrics({**base, "stagnation_rate": .9}, config)
    tiny = calibration_metrics({**base, "episodes": 3}, config)
    assert good["eligible"]
    assert not too_hard["eligible"]
    assert good["rank"] < too_hard["rank"]
    assert not stuck["quality_passed"] and not tiny["eligible"]
