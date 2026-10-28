from copy import deepcopy
import hashlib
import json

import pytest
import torch

from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.provenance import calibration_campaigns, capture_checkpoint
from nowarthunder_ai.qualification import qualify_checkpoint


def fixture(tmp_path):
    source = tmp_path / "ace.pt"
    actor = PolicyNetwork((64, 32))
    state = {"format_version": 1, "kind": "training", "trained": True,
             "enemy_optimizer_updates": 80, "iteration": 125, "difficulty": "ace",
             "observation_version": 2, "normalization_version": 2,
             "model_spec": {"enemy_hidden": [64, 32], "observation_size": 320, "action_size": 10},
             "enemy_actor": actor.state_dict(), "enemy_critic": {"weight": torch.randn(3, 4)},
             "config": {"difficulty": "ace", "calibration_version": 2,
                        "target_defender_win_rate": [.1, .3], "eval_seed": 3_000_000, "eval_episodes": 50,
                        "consumed_campaigns": [{"seed": 2_000_000, "episodes": 100}]}}
    torch.save(state, source)
    rows = []
    for opponent in ("guard", "hunter", "wander"):
        for sample in range(100):
            rows.append({"opponent": opponent, "level_id": sample % 10 + 1,
                         "seed": -3_000_001 - sample, "outcome": "defender_win" if sample < 55 else "enemy_win_base",
                         "enemy_alive_steps": 100, "enemy_stuck_steps": 0, "enemy_observed_steps": 100,
                         "enemy_unproductive_steps": 0, "truncated": False})
    report = {"checkpoint_identity": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
              "difficulty": "ace", "metrics_version": 2, "deterministic_policy": True,
              "policy_sampler": {"mode": "argmax", "temperature": 1},
              "evaluation_config": {"eval_seed": 3_000_000, "episodes": 100}, "records": rows,
              "calibration": {"eligible": False}}  # source ace band intentionally fails
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(report))
    return source, path, report


def test_qualification_preserves_all_weights_and_consumes_selection_campaign(tmp_path):
    source, report_path, report = fixture(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "expert.pt"
    result = qualify_checkpoint(source, report_path, output)
    old, qualified = capture_checkpoint(source).state, capture_checkpoint(output).state
    assert source.read_bytes() == original
    assert qualified["difficulty"] == qualified["config"]["difficulty"] == "expert"
    assert qualified["kind"] == "qualified" and qualified["trained"]
    assert qualified["config"]["target_defender_win_rate"] == [.4, .6]
    assert qualified["enemy_optimizer_updates"] == old["enemy_optimizer_updates"]
    for group in ("enemy_actor", "enemy_critic"):
        for name, tensor in old[group].items():
            assert torch.equal(tensor, qualified[group][name])
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["qualification"]["assessment"]["defender_win_rate"] == .55
    assert result["qualification"]["release_evaluation_required"]
    assert result["qualification"]["weights_changed"] is False
    campaigns = {(abs(item["seed"]), item["episodes"]) for item in calibration_campaigns(qualified)}
    assert campaigns == {(3_000_000, 50), (3_000_000, 100), (2_000_000, 100)}
    assert "qualification" not in old and old["difficulty"] == "ace"


@pytest.mark.parametrize("change,message", [
    (lambda report: report["checkpoint_identity"].update(sha256="wrong"), "SHA256"),
    (lambda report: report.update(difficulty="expert"), "source difficulty"),
    (lambda report: report.update(retrospective_assessment=True), "fresh"),
    (lambda report: report.update(deterministic_policy=False), "argmax"),
    (lambda report: report["records"][0].update(seed=-99), "seed campaign"),
    (lambda report: report["records"][0].update(seed=report["records"][1]["seed"]), "exactly once"),
    (lambda report: report["records"][0].pop("enemy_unproductive_steps"), "qualification"),
])
def test_qualification_rejects_invalid_evidence_without_writing(tmp_path, change, message):
    source, report_path, report = fixture(tmp_path)
    change(report)
    report_path.write_text(json.dumps(report))
    output = tmp_path / "rejected" / "expert.pt"
    with pytest.raises(ValueError, match=message):
        qualify_checkpoint(source, report_path, output)
    assert not output.exists() and not output.parent.exists()


def test_claimed_eligibility_does_not_override_actual_out_of_band_behavior(tmp_path):
    source, report_path, report = fixture(tmp_path)
    for row in report["records"]:
        row["outcome"] = "enemy_win_base"
    report["calibration"] = {"eligible": True}
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="below_target_band"):
        qualify_checkpoint(source, report_path, tmp_path / "expert.pt")


def test_qualification_never_overwrites_any_existing_file(tmp_path):
    source, report_path, _ = fixture(tmp_path)
    for output in (source, report_path, tmp_path / "existing.pt"):
        if not output.exists():
            output.write_bytes(b"keep")
        original = output.read_bytes()
        with pytest.raises(FileExistsError):
            qualify_checkpoint(source, report_path, output)
        assert output.read_bytes() == original


def test_qualification_refuses_untrained_enemy_even_with_defender_updates(tmp_path):
    source, report_path, _ = fixture(tmp_path)
    state = capture_checkpoint(source).state
    state.update(enemy_optimizer_updates=0, defender_optimizer_updates=100)
    torch.save(state, source)
    with pytest.raises(ValueError, match="trained, non-smoke enemy"):
        qualify_checkpoint(source, report_path, tmp_path / "expert.pt")
