"""Bounded distillation contract checks; no Go worker or long training needed."""
from copy import deepcopy

import pytest
import torch

from nowarthunder_ai.checkpoint import actor_from_checkpoint, load_checkpoint, save_checkpoint
from nowarthunder_ai.difficulty import control_config
from nowarthunder_ai.distillation import distill, inherit_teacher_contract
from nowarthunder_ai.trainer import train
from test_training_lifecycle import FakeEnv, small_config


def teacher_contract():
    return {
        "difficulty": "trainee",
        "target_defender_win_rate": [.7, .9],
        "config": {
            "difficulty_control": control_config({"version": 1, "masked_features": [0, 319]}),
            "calibration_version": 2,
            "eval_seed": 3_000_000,
            "eval_episodes": 50,
            "eval_max_steps": 9000,
            "eval_every": 25,
            "min_evaluation_episodes": 60,
            "max_truncation_rate": .1,
            "max_stagnation_rate": .25,
            "v2_min_primary_episodes": 100,
            "v2_min_opponent_episodes": 50,
            "v2_min_level_opponent_episodes": 5,
            "v2_max_primary_timeout_rate": .05,
            "consumed_campaigns": [
                {"seed": 1_000_000, "episodes": 20, "purpose": "legacy_teacher_selection"},
                {"seed": 2_000_000, "episodes": 100, "purpose": "reviewed_student_test"},
            ],
        },
    }


def test_teacher_contract_is_inherited_before_defaults_and_deep_copied():
    source = teacher_contract()
    result = inherit_teacher_contract({"eval_every": 0, "student_hidden": [6, 3]}, source)
    assert result["eval_seed"] == 3_000_000
    assert result["eval_every"] == 0
    assert result["calibration_version"] == 2
    assert result["difficulty"] == "trainee"
    for key, value in source["config"].items():
        if key != "eval_every":
            assert result[key] == value
    result["difficulty_control"]["masked_features"].append(210)
    result["consumed_campaigns"][0]["episodes"] = 1
    assert source["config"]["difficulty_control"]["masked_features"] == [0, 319]
    assert source["config"]["consumed_campaigns"][0]["episodes"] == 20


@pytest.mark.parametrize("override", [
    {"difficulty": "ace"},
    {"difficulty_control": {"version": 1, "masked_features": []}},
    {"difficulty_control": {"version": 1, "masked_features": [0, 319], "fire_budget": .9}},
    {"calibration_version": 1},
    {"v2_min_primary_episodes": 1},
    {"eval_seed": 2_000_000},
    {"consumed_campaigns": []},
])
def test_explicit_incompatible_contract_is_rejected(override):
    with pytest.raises(ValueError, match="must match the teacher profile contract"):
        inherit_teacher_contract(override, teacher_contract())


def test_equivalent_minimal_control_and_legacy_unlabeled_checkpoint():
    result = inherit_teacher_contract(
        {"difficulty_control": {"version": 1, "masked_features": [319, 0]}}, teacher_contract())
    assert control_config(result["difficulty_control"])["masked_features"] == [0, 319]
    legacy = inherit_teacher_contract({"iterations": 1}, {"difficulty": "unrated", "config": {}})
    assert legacy == {"iterations": 1}
    with pytest.raises(ValueError, match="difficulty_control must match"):
        inherit_teacher_contract({"difficulty_control": {"version": 1, "masked_features": [0]}},
                                 {"difficulty": "unrated", "config": {}})


def test_v3_detector_limits_and_opponents_survive_distillation_inheritance():
    from nowarthunder_ai.progress import progress_config
    source = teacher_contract()
    source['config'].update(calibration_version=3, progress_recovery=progress_config(),
                            v3_max_primary_moving_no_progress_rate=.04,
                            eval_opponents=['guard', 'hunter', 'wander'],
                            deployment_sampler={'mode': 'categorical', 'temperature': .3, 'rng': 'xorshift64-v1'})
    inherited = inherit_teacher_contract({}, source)
    assert inherited['calibration_version'] == 3
    assert inherited['progress_recovery'] == source['config']['progress_recovery']
    assert inherited['v3_max_primary_moving_no_progress_rate'] == .04
    assert inherited['eval_opponents'] == ['guard', 'hunter', 'wander']
    assert inherited['deployment_sampler'] == source['config']['deployment_sampler']
    inherit_teacher_contract({'deployment_sampler': {'mode': 'categorical', 'temperature': .3}}, source)
    for key, value in [('progress_recovery', {'window': 200}),
                       ('v3_max_primary_moving_no_progress_rate', .99),
                       ('deployment_sampler', {'mode': 'argmax'}),
                       ('eval_opponents', ['wander'])]:
        with pytest.raises(ValueError, match='must match the teacher profile contract'):
            inherit_teacher_contract({key: value}, source)
    inherited['progress_recovery']['window'] = 200
    assert source['config']['progress_recovery']['window'] == 120


def test_student_gate_and_teacher_provenance_survive_optimizer_and_resume(tmp_path):
    # Tiny FakeEnv training creates a valid test-only teacher. Its explicit
    # fixture contract has no calibrated gameplay claim.
    trained = train(small_config(iterations=1), FakeEnv, "cpu", tmp_path / "teacher")
    teacher = load_checkpoint(trained["checkpoint"])
    contract = teacher_contract()
    teacher.update(difficulty=contract["difficulty"],
                   target_defender_win_rate=contract["target_defender_win_rate"],
                   calibration_version=2,
                   difficulty_state={"budget_scale": .75, "updates": 4, "last_status": "in_band"},
                   checkpoint_identity="bounded-test-teacher",
                   selection={"calibrated": True})
    teacher["config"].update(deepcopy(contract["config"]))
    teacher_path = save_checkpoint(tmp_path / "controlled-teacher.pt", teacher)
    config = small_config(iterations=1, student_hidden=[6, 3], epochs=1, dataset_size=10,
                          eval_every=0)
    result = distill(config, FakeEnv, "cpu", tmp_path / "student", teacher_path)
    first = load_checkpoint(result["checkpoint"])
    assert first["calibration_version"] == 2
    assert first["config"]["eval_seed"] == 3_000_000
    assert first["config"]["consumed_campaigns"] == contract["config"]["consumed_campaigns"]
    assert first["config"]["v2_min_primary_episodes"] == 100
    assert first["source_difficulty_state"] == teacher["difficulty_state"]
    assert first["source_teacher_checkpoint"] == str(teacher_path)
    assert first["source_teacher_identity"] == "bounded-test-teacher"
    assert not first["calibration"]["eligible"] and not first["selection"]["calibrated"]
    assert first["enemy_optimizer_updates"] > 0
    assert torch.count_nonzero(first["enemy_actor"]["net.0.weight"][:, [0, 319]]) == 0
    actor = actor_from_checkpoint(first)
    assert actor.masked_features == (0, 319)
    probe = torch.randn(4, 320)
    changed = probe.clone()
    changed[:, [0, 319]] += 50
    torch.testing.assert_close(actor(probe), actor(changed), rtol=0, atol=0)

    # Loading a resumed tensor cannot reintroduce forbidden input columns, even
    # if a stale optimizer/checkpoint has nonzero weights in those columns.
    first["enemy_actor"]["net.0.weight"][:, [0, 319]] = 3
    resume_path = save_checkpoint(result["checkpoint"], first)
    resumed = distill({**config, "iterations": 2}, FakeEnv, "cpu", tmp_path / "student",
                      tmp_path / "moved-teacher.pt", resume_path)
    second = load_checkpoint(resumed["checkpoint"])
    assert second["source_teacher_checkpoint"] == str(teacher_path)
    assert second["config"]["consumed_campaigns"] == first["config"]["consumed_campaigns"]
    assert second["config"]["difficulty_control"] == first["config"]["difficulty_control"]
    assert second["enemy_optimizer_updates"] > first["enemy_optimizer_updates"]
    assert torch.count_nonzero(second["enemy_actor"]["net.0.weight"][:, [0, 319]]) == 0
    assert not second["calibration"]["eligible"]

    with pytest.raises(ValueError, match="difficulty_control must match"):
        distill({**config, "iterations": 3,
                 "difficulty_control": {"version": 1, "masked_features": []}},
                FakeEnv, "cpu", tmp_path / "student", teacher_path, resumed["checkpoint"])
