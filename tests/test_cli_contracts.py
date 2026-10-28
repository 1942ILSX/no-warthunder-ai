import numpy as np
from pathlib import Path
import pytest
import torch

from nowarthunder_ai import checkpoint, cli, distillation, evaluation
from nowarthunder_ai.export import parity_random_states, reference_action, verify_artifact, write_artifact
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.provenance import CheckpointSnapshot


def test_final_campaign_uses_checkpoint_seed_and_detects_aliases_and_partial_overlap():
    state = {'config': {'eval_seed': 2_000_000, 'eval_episodes': 20}}
    exact = cli.evaluation_campaign(state, -2_000_000, 20)
    assert exact['purpose'] == 'calibration_replay'
    assert exact['overlapping_seeds_per_opponent'] == 20
    overlap = cli.evaluation_campaign(state, 2_000_001, 100)
    assert overlap['purpose'] == 'calibration_overlap'
    assert overlap['overlapping_seeds_per_opponent'] == 19
    assert cli.evaluation_campaign(state, 2_000_020, 10)['purpose'] == 'final_test'
    student = {'config': {'eval_seed': 1_000_000, 'eval_episodes': 10}, 'teacher_checkpoint': state}
    inherited = cli.evaluation_campaign(student, 2_000_001, 100)
    assert inherited['purpose'] == 'calibration_overlap'
    assert inherited['overlapping_seeds_per_opponent'] == 19
    assert len(inherited['calibration_campaigns']) == 2
    with pytest.raises(ValueError, match='signed int64'):
        cli.evaluation_campaign(state, 2**63, 1)


def test_cli_evaluation_infers_profile_and_rejects_mismatch(monkeypatch):
    state = {'difficulty': 'trainee', 'config': {'eval_seed': 2_000_000, 'eval_episodes': 20}, 'model_spec': {}}
    monkeypatch.setattr(cli, 'capture_checkpoint', lambda path: CheckpointSnapshot(Path(path), b'', '0' * 64, state))
    monkeypatch.setattr(checkpoint, 'actor_from_checkpoint', lambda *args: object())
    monkeypatch.setattr(cli, '_factory', lambda args: None)
    reports, configs = [], []
    monkeypatch.setattr(cli, 'write_report', lambda report, output=None: reports.append(report))
    def evaluate(path, factory, device, config):
        configs.append(config)
        return {'episodes': 300}
    monkeypatch.setattr(evaluation, 'evaluate', evaluate)
    assert cli.main(['evaluate', '--checkpoint', 'saved.pt', '--device', 'cpu']) == 0
    assert configs[0]['difficulty'] == reports[0]['difficulty'] == 'trainee'
    assert reports[0]['campaign']['purpose'] == 'calibration_overlap'
    assert cli.main(['evaluate', '--checkpoint', 'saved.pt', '--profile', 'ace', '--device', 'cpu']) == 1
    assert len(configs) == 1


def test_cli_distillation_resume_does_not_read_moved_teacher(monkeypatch):
    reads, observed = [], []
    def load(path):
        reads.append(path)
        if path != 'student.pt':
            raise FileNotFoundError(path)
        return {'difficulty': 'trainee'}
    monkeypatch.setattr(checkpoint, 'load_checkpoint', load)
    monkeypatch.setattr(cli, '_factory', lambda args: None)
    monkeypatch.setattr(cli, 'write_report', lambda *args: None)
    def distill(config, factory, device, output, teacher, resume=None):
        observed.append((config['difficulty'], teacher, resume))
        return {'checkpoint': 'student.pt'}
    monkeypatch.setattr(distillation, 'distill', distill)
    assert cli.main(['distill', '--teacher', 'moved-teacher.pt', '--resume', 'student.pt',
                     '--output', 'unused', '--device', 'cpu']) == 0
    assert reads == ['student.pt']
    assert observed == [('trainee', 'moved-teacher.pt', 'student.pt')]


def test_categorical_parity_rng_corpus_exercises_entire_cdf(tmp_path):
    states = parity_random_states(np.random.default_rng(73), 128)
    expected = [reference_action([0.] * 10, [True] * 10, state, 'categorical')[0] for state in states]
    assert set(expected) == set(range(10))
    assert {0, 1, 2**64 - 1}.issubset(states)
    actor = PolicyNetwork((32, 16))
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    destination = tmp_path / 'uniform-policy'
    write_artifact(actor, destination, policy_version='cdf-test', training_seed=73,
                   trained=False, difficulty='trainee', sampler='categorical')
    observations = np.zeros((128, 320), dtype=np.float32)
    observations[:, 298:308] = 1
    report = verify_artifact(destination, observations=observations)
    assert all(count > 0 for count in report['samplers']['categorical']['action_counts'].values())
    assert report['samplers']['argmax']['action_counts']['0'] == 128
