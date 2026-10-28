import hashlib
from pathlib import Path

import pytest
import torch

from nowarthunder_ai import cli, evaluation, export, provenance, trainer
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.provenance import calibration_campaigns, capture_checkpoint, freeze_checkpoint


def state(marker=1.):
    actor = PolicyNetwork((32, 16))
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.net[-1].bias[0] = marker
    return {'format_version': 1, 'kind': 'training', 'trained': True,
            'enemy_optimizer_updates': 1, 'difficulty': 'expert',
            'observation_version': 2, 'normalization_version': 2,
            'model_spec': {'enemy_hidden': [32, 16], 'observation_size': 320, 'action_size': 10},
            'enemy_actor': actor.state_dict(),
            'config': {'difficulty': 'expert', 'seed': 3, 'eval_seed': 1_000_000, 'eval_episodes': 20}}


def test_snapshot_hash_and_state_use_same_bytes_when_source_is_replaced(tmp_path, monkeypatch):
    source, replacement = tmp_path / 'latest.pt', tmp_path / 'replacement.pt'
    torch.save(state(1), source)
    torch.save(state(2), replacement)
    original = source.read_bytes()
    real_load = torch.load
    def replace_during_load(stream, **kwargs):
        source.write_bytes(replacement.read_bytes())
        return real_load(stream, **kwargs)
    monkeypatch.setattr(provenance.torch, 'load', replace_during_load)
    snapshot = capture_checkpoint(source)
    assert snapshot.data == original
    assert snapshot.sha256 == hashlib.sha256(original).hexdigest()
    assert snapshot.state['enemy_actor']['net.4.bias'][0] == 1
    assert source.read_bytes() != original


def test_cli_evaluate_certifies_captured_actor_not_replacement(tmp_path, monkeypatch):
    source = tmp_path / 'latest.pt'
    torch.save(state(1), source)
    original_hash = cli.file_sha256(source)
    reports = []
    monkeypatch.setattr(cli, '_factory', lambda args: None)
    monkeypatch.setattr(cli, 'write_report', lambda report, output=None: reports.append(report))
    def evaluate(actor, factory, device, config):
        assert isinstance(actor, PolicyNetwork)
        assert actor.net[-1].bias[0].item() == 1
        torch.save(state(2), source)
        return {'episodes': 100}
    monkeypatch.setattr(evaluation, 'evaluate', evaluate)
    assert cli.main(['evaluate', '--checkpoint', str(source), '--device', 'cpu']) == 0
    assert reports[0]['checkpoint_identity']['sha256'] == original_hash
    assert cli.file_sha256(source) != original_hash


def test_export_serializes_captured_weights_without_rereading_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / 'latest.pt'
    torch.save(state(1), source)
    real_capture = capture_checkpoint
    def replace_after_capture(path):
        snapshot = real_capture(path)
        torch.save(state(2), source)
        return snapshot
    monkeypatch.setattr(export, 'capture_checkpoint', replace_after_capture)
    destination = tmp_path / 'artifact'
    export.export_checkpoint(source, destination, policy_version='snapshot-test')
    actor = export.artifact_actor(destination)
    assert actor.net[-1].bias[0].item() == 1
    assert real_capture(source).state['enemy_actor']['net.4.bias'][0] == 2


def test_freeze_is_atomic_reusable_and_never_overwrites_different_source(tmp_path):
    source = tmp_path / 'source.pt'
    torch.save(state(), source)
    snapshot = capture_checkpoint(source)
    frozen = freeze_checkpoint(snapshot, tmp_path / 'runs')
    assert frozen.name == f'source-{snapshot.sha256}.pt'
    assert frozen.read_bytes() == snapshot.data
    assert freeze_checkpoint(snapshot, tmp_path / 'runs') == frozen
    frozen.write_bytes(b'keep these changed bytes')
    with pytest.raises(ValueError, match='differs'):
        freeze_checkpoint(snapshot, tmp_path / 'runs')
    assert frozen.read_bytes() == b'keep these changed bytes'
    assert source.read_bytes() == snapshot.data


@pytest.mark.parametrize('change', [
    {'observation_version': 1}, {'normalization_version': 1},
    {'model_spec': {'enemy_hidden': [32, 16], 'observation_size': 321, 'action_size': 10}},
    {'model_spec': {'enemy_hidden': [32, 16], 'observation_size': 320, 'action_size': 11}},
])
def test_preflight_rejects_stale_schema_without_creating_outputs(tmp_path, change):
    source = tmp_path / 'source.pt'
    torch.save({**state(), **change}, source)
    output = tmp_path / 'runs'
    with pytest.raises(ValueError):
        cli.difficulty_run_plan(str(source), str(output))
    assert not output.exists()


def test_preflight_loads_actor_and_rejects_nonfinite_weights(tmp_path):
    source = tmp_path / 'source.pt'
    bad = state()
    bad['enemy_actor']['net.0.weight'][0, 0] = float('nan')
    torch.save(bad, source)
    with pytest.raises(ValueError, match='non-finite'):
        cli.difficulty_run_plan(str(source), str(tmp_path / 'runs'))


def test_tune_freezes_source_once_across_profiles_and_plan_only_does_not_write(tmp_path, monkeypatch):
    source = tmp_path / 'latest.pt'
    torch.save(state(1), source)
    original = source.read_bytes()
    output = tmp_path / 'runs'
    reports, calls = [], []
    monkeypatch.setattr(cli, 'write_report', lambda report, output=None: reports.append(report))
    monkeypatch.setattr(cli, '_factory', lambda args: None)
    def train(config, factory, device, output, *, initial_enemy_checkpoint, allow_profile_transfer):
        frozen = capture_checkpoint(initial_enemy_checkpoint)
        calls.append((config['difficulty'], frozen.path, frozen.sha256))
        assert frozen.state['enemy_actor']['net.4.bias'][0] == 1
        assert allow_profile_transfer
        torch.save(state(99), source)
        return {'selection': {'calibrated': False}}
    monkeypatch.setattr(trainer, 'train', train)
    args = ['tune-difficulties', '--checkpoint', str(source), '--output', str(output),
            '--profiles', 'trainee', 'expert', '--iterations', '1', '--device', 'cpu']
    assert cli.main([*args, '--plan-only']) == 0
    assert not output.exists() and not calls
    assert cli.main(args) == 0
    assert [item[0] for item in calls] == ['trainee', 'expert']
    assert len({item[1] for item in calls}) == len({item[2] for item in calls}) == 1
    assert calls[0][1].read_bytes() == original


def test_tune_rechecks_frozen_hash_before_next_profile(tmp_path, monkeypatch):
    source = tmp_path / 'source.pt'
    torch.save(state(), source)
    calls = []
    monkeypatch.setattr(cli, '_factory', lambda args: None)
    monkeypatch.setattr(cli, 'write_report', lambda *args: None)
    def train(config, factory, device, output, *, initial_enemy_checkpoint, allow_profile_transfer):
        calls.append(config['difficulty'])
        Path(initial_enemy_checkpoint).write_bytes(b'changed')
        return {'selection': {}}
    monkeypatch.setattr(trainer, 'train', train)
    assert cli.main(['tune-difficulties', '--checkpoint', str(source), '--output', str(tmp_path / 'runs'),
                     '--profiles', 'trainee', 'expert', '--device', 'cpu']) == 1
    assert calls == ['trainee']


def test_flattened_transfer_and_teacher_campaigns_survive_repeated_transfer():
    source = {'config': {'eval_seed': 3_000_000, 'eval_episodes': 50},
              'initial_source': {'calibration_config': {'eval_seed': 1_000_000, 'eval_episodes': 20},
                                 'calibration_ancestry': [{'seed': 5_000_000, 'episodes': 70, 'purpose': 'older_transfer'}]},
              'teacher_checkpoint': {'config': {'eval_seed': 2_000_000, 'eval_episodes': 100}}}
    flattened = calibration_campaigns(source)
    assert {c['seed'] for c in flattened} == {1_000_000, 2_000_000, 3_000_000, 5_000_000}
    transferred = {'config': {'eval_seed': 6_000_000, 'eval_episodes': 50},
                   'initial_source': {'calibration_ancestry': flattened}}
    campaign = cli.evaluation_campaign(transferred, -5_000_000, 70)
    assert campaign['purpose'] == 'calibration_replay'
    assert campaign['overlapping_seeds_per_opponent'] == 70


def test_snapshot_rejects_unversioned_checkpoint(tmp_path):
    source = tmp_path / 'unversioned.pt'
    torch.save({'trained': True}, source)
    with pytest.raises(ValueError, match='Unsupported training checkpoint format'):
        capture_checkpoint(source)
