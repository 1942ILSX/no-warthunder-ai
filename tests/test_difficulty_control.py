import hashlib

import pytest
import torch

from nowarthunder_ai.checkpoint import actor_from_checkpoint, cpu_state, load_checkpoint
from nowarthunder_ai.difficulty import DifficultyController, control_config, difficulty_actor_loss
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.trainer import train
from test_training_lifecycle import FakeEnv, small_config


def observation(count=1):
    obs = torch.zeros(count, 320)
    obs[:, 2] = 1  # facing north
    obs[:, 8] = .25  # previous move north, so not idle
    obs[:, 294:298] = torch.tensor([.5, .2, .7, .8])
    return obs


def control(**kwargs):
    return control_config({"version": 1, "fire_budget": .1,
                           "turn_fire_budget": .1, **kwargs})


def calculate(logits, obs, config=None, mask=None):
    config = config or control()
    mask = torch.ones_like(logits, dtype=torch.bool) if mask is None else mask
    return difficulty_actor_loss(logits, obs, mask, config,
                                 {key: config[key] for key in ("fire_budget", "turn_fire_budget")})


def test_pressure_is_differentiable_and_only_costs_qualified_fire():
    obs = observation()
    obs[:, 313] = 1  # north target: idle fire and north fire qualify
    logits = torch.zeros(1, 10, requires_grad=True)
    loss, metrics = calculate(logits, obs)
    loss.backward()
    assert loss > 0 and metrics["difficulty_combat_opportunities"] == 1
    assert logits.grad[0, 1] > 0 and logits.grad[0, 3] > 0
    assert logits.grad[0, 2] < 0  # north movement without firing is not penalized
    assert torch.isfinite(logits.grad).all()


def test_mean_hinge_allows_individual_high_pressure_responses():
    obs = observation(2)
    obs[:, 313] = 1
    logits = torch.full((2, 10), -20.)
    logits[0, 1] = 20  # this individual state's fire probability is almost 1
    logits[1, 4] = 20  # the other state chooses a nonfire move
    logits.requires_grad_()
    loss, metrics = calculate(logits, obs, control(fire_budget=.6))
    loss.backward()
    assert metrics["difficulty_fire_pressure"].item() == pytest.approx(.5)
    assert loss.item() == 0 and torch.all(logits.grad == 0)


def test_turn_fire_pressure_excludes_already_facing_shots():
    obs = observation()
    obs[:, 314] = 1  # east target, actor faces north
    logits = torch.zeros(1, 10, requires_grad=True)
    loss, metrics = calculate(logits, obs, control(fire_budget=1., turn_fire_budget=.01))
    loss.backward()
    assert metrics["difficulty_turn_fire_pressure"] > .01
    assert logits.grad[0, 5] > 0
    obs[:, 2:6] = torch.tensor([0., 1., 0., 0.])
    _, metrics = calculate(logits.detach(), obs)
    assert metrics["difficulty_turn_fire_pressure"] == 0


@pytest.mark.parametrize("kind", ["brick", "emergency", "cooldown"])
def test_brick_self_defense_and_illegal_fire_are_exempt(kind):
    obs = observation()
    obs[:, 313] = .3 if kind == "brick" else 1.
    mask = torch.ones(1, 10, dtype=torch.bool)
    if kind == "emergency":
        obs[:, 217] = 1
        obs[:, 208:210] = .01
    if kind == "cooldown":
        mask[:, 1::2] = False
    logits = torch.zeros(1, 10, requires_grad=True)
    loss, metrics = calculate(logits, obs, mask=mask)
    loss.backward()
    assert loss == 0 and metrics["difficulty_combat_opportunities"] == 0
    assert torch.all(logits.grad == 0)


def test_recovery_ce_accepts_both_fire_variants_and_avoids_blocked_direction():
    obs = observation()
    obs[:, 8] = 0  # idle with no firing target, best feasible nav is east
    logits = torch.zeros(1, 10, requires_grad=True)
    loss, metrics = calculate(logits, obs)
    loss.backward()
    assert metrics["difficulty_recovery_samples"] == 1
    assert logits.grad[0, 4] < 0 and logits.grad[0, 5] < 0
    assert logits.grad[0, 4] == logits.grad[0, 5]
    obs[:, 8], obs[:, 317] = .5, .8  # was blocked moving east
    logits = torch.zeros(1, 10, requires_grad=True)
    loss, _ = calculate(logits, obs)
    loss.backward()
    assert logits.grad[0, 2] < 0 and logits.grad[0, 3] < 0  # next best north
    assert logits.grad[0, 4] > 0
    obs[:, 294:298] = 1  # no feasible recovery, do not invent an invalid move
    loss, metrics = calculate(torch.zeros(1, 10), obs)
    assert loss == 0 and metrics["difficulty_recovery_samples"] == 0


def test_masked_features_remain_zero_after_adam_and_export_as_plain_mlp():
    torch.manual_seed(2)
    policy = PolicyNetwork((8, 4))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    obs = torch.randn(10, 320)
    policy(obs).square().mean().backward()
    optimizer.step()  # populate momentum before introducing a handicap
    policy.set_masked_features([210, 211, 309])
    optimizer.zero_grad(set_to_none=True)
    policy(obs).square().mean().backward()
    optimizer.step()
    policy.enforce_input_gate()
    assert torch.all(policy.net[0].weight[:, [210, 211, 309]] == 0)
    assert "input_gate" not in policy.state_dict()
    plain = PolicyNetwork((8, 4))
    plain.load_state_dict(policy.state_dict())
    torch.testing.assert_close(policy(obs), plain.net(obs), rtol=0, atol=0)
    checkpoint = {"model_spec": {"enemy_hidden": [8, 4]}, "enemy_actor": cpu_state(policy),
                  "config": {"difficulty_control": {"masked_features": [210, 211, 309]}}}
    restored = actor_from_checkpoint(checkpoint)
    assert restored.masked_features == (210, 211, 309)
    with pytest.raises(ValueError, match="role"):
        policy.set_masked_features([308])


def test_controller_freezes_bad_quality_and_resumes_bounded_scale():
    controller = DifficultyController(control())
    assessment = {"quality_passed": False, "defender_win_rate": .1,
                  "target_defender_win_rate": [.4, .6]}
    controller.update(assessment)
    assert controller.budget_scale == 1
    assert controller.last_status == "frozen_quality_failed"
    assessment["quality_passed"] = True
    controller.update(assessment)
    assert controller.budget_scale < 1  # enemy too strong -> less pressure
    restored = DifficultyController(control(), controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
    for _ in range(100):
        controller.update(assessment)
        restored.update(assessment)
    assert restored.state_dict() == controller.state_dict()
    assert controller.budget_scale == controller.config["min_budget_scale"]
    assessment["defender_win_rate"] = .9
    for _ in range(100):
        controller.update(assessment)
    assert controller.budget_scale == controller.config["max_budget_scale"]
    old = controller.budget_scale
    controller.update({**assessment, "defender_win_rate": .5})
    assert controller.budget_scale == old and controller.last_status == "in_band"


@pytest.mark.parametrize("change", [
    {"pressure_coef": float("nan")}, {"fire_budget": float("inf")},
    {"recovery_coef": -1}, {"min_budget_scale": 0}, {"masked_features": [308]},
    {"masked_features": [320]}, {"version": 2}, {"enabled": True},
])
def test_invalid_control_config_fails_closed(change):
    with pytest.raises(ValueError):
        control_config(change)


def test_trainer_restores_controller_and_mask_and_passes_v2_contract(tmp_path, monkeypatch):
    import nowarthunder_ai.evaluation as module
    observed = []
    def fake_evaluate(actor, factory, device, config):
        observed.append(config)
        return {"defender_win_rate": .1}
    assessment = {"quality_passed": True, "eligible": False, "defender_win_rate": .1,
                  "target_defender_win_rate": [.4, .6], "rank": [0., .3, .4],
                  "reference": "guard/hunter primary", "calibration_version": 2}
    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(module, "calibration_metrics", lambda *_: dict(assessment))
    config = small_config(iterations=1, difficulty="expert", target_defender_win_rate=[.4, .6],
                          calibration_version=2, eval_every=1,
                          eval_opponents=["guard", "hunter", "wander"],
                          primary_opponents=["guard", "hunter"],
                          difficulty_control={"version": 1, "masked_features": [210]})
    first = train(config, FakeEnv, "cpu", tmp_path / "run")
    before = load_checkpoint(first["checkpoint"])
    assert before["difficulty_state"]["budget_scale"] < 1
    assert observed[0]["calibration_version"] == 2
    assert observed[0]["primary_opponents"] == ["guard", "hunter"]
    second = train({**config, "iterations": 2}, FakeEnv, "cpu", tmp_path / "run", first["checkpoint"])
    after = load_checkpoint(second["checkpoint"])
    assert after["difficulty_state"]["updates"] == before["difficulty_state"]["updates"] + 1
    assert torch.all(after["enemy_actor"]["net.0.weight"][:, 210] == 0)


def test_profile_transfer_requires_opt_in_and_records_source_identity(tmp_path):
    source = train(small_config(iterations=1, difficulty="ace", target_defender_win_rate=[.1, .3]),
                   FakeEnv, "cpu", tmp_path / "source")
    source_path = tmp_path / "source" / "latest.pt"
    original = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config = small_config(iterations=1, difficulty="trainee", target_defender_win_rate=[.7, .9],
                          calibration_version=2, difficulty_control={"masked_features": [210, 211]})
    with pytest.raises(ValueError, match="difficulty must match"):
        train(config, FakeEnv, "cpu", tmp_path / "guarded", initial_enemy_checkpoint=source["checkpoint"])
    result = train(config, FakeEnv, "cpu", tmp_path / "transferred",
                   initial_enemy_checkpoint=source["checkpoint"], allow_profile_transfer=True)
    checkpoint = load_checkpoint(result["checkpoint"])
    assert checkpoint["difficulty"] == "trainee"
    assert checkpoint["initial_source"]["difficulty"] == "ace"
    assert checkpoint["initial_source"]["sha256"] == original
    assert checkpoint["initial_source"]["profile_transfer"] is True
    assert any(item["seed"] == 1_000_000 for item in checkpoint["initial_source"]["calibration_ancestry"])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original
    with pytest.raises(ValueError, match="requires an initial checkpoint"):
        train(config, FakeEnv, "cpu", tmp_path / "invalid", allow_profile_transfer=True)
