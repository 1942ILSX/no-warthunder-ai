from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nowarthunder_ai import cli, evaluation
from nowarthunder_ai.checkpoint import actor_from_checkpoint, save_checkpoint
from nowarthunder_ai.env import BatchedEnv, enemy_model_identity, validate_enemy_policy_header
from nowarthunder_ai.export import (actor_weight_bytes, evaluation_artifact, export_checkpoint,
                                    write_artifact)
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.provenance import capture_checkpoint
from nowarthunder_ai.sampling import deployment_sampler
from nowarthunder_ai.qualification import qualify_checkpoint
from test_evaluation_v3 import LoopEnv
from test_v3_release import v3_checkpoint, v3_report


def categorical_source(path):
    state = v3_checkpoint(path)
    state['config']['deployment_sampler'] = {'mode': 'categorical', 'temperature': .3, 'rng': 'xorshift64-v1'}
    state['enemy_actor'] = actor_from_checkpoint(state).state_dict()
    save_checkpoint(path, state)
    return state


def test_actual_go_native_sampler_identity_replay_and_no_overrides(tmp_path):
    actor = PolicyNetwork((32, 16))
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    path = tmp_path / 'native'
    write_artifact(actor, path, policy_version='native-test', training_seed=73, trained=True,
                   difficulty='expert', sampler='categorical', temperature=.3)
    expected = enemy_model_identity(path)
    config = {'level_id': 1, 'seed': -12345, 'max_steps': 12, 'defender_bot': 'guard'}
    with BatchedEnv([config, config], enemy_model=path, workers=2) as env:
        assert env.metadata['enemy_policy'] == expected
        first = None
        for _ in range(12):
            frame = env.step(np.full((2, 5), -1, dtype=np.int64), np.full(2, -1, dtype=np.int64))
            np.testing.assert_array_equal(frame.critic_obs[0], frame.critic_obs[1])
            if first is None:
                first = frame.critic_obs[0].copy()
        env.reset_indices([0], [config])
        frame = env.step(np.full((2, 5), -1, dtype=np.int64), np.full(2, -1, dtype=np.int64))
        np.testing.assert_array_equal(frame.critic_obs[0], first)
        with pytest.raises(ValueError, match='cannot override'):
            env.step(np.zeros((2, 5), dtype=np.int64), np.full(2, -1, dtype=np.int64))
    altered = deepcopy(expected)
    altered['metadata']['sampler']['temperature'] = .4
    with pytest.raises(RuntimeError, match='identity differs'):
        validate_enemy_policy_header(altered, expected)
    with pytest.raises(RuntimeError, match='identity differs'):
        validate_enemy_policy_header(None, expected)


def test_native_evaluation_uses_actual_go_actions_and_records_rng():
    sampler = deployment_sampler({'mode': 'categorical', 'temperature': .3})
    identity = {'implementation': 'go-policymlp', 'metadata': {'sampler': sampler}}
    class NativeEnv(LoopEnv):
        def __init__(self, configs):
            super().__init__(configs)
            self.metadata = {'enemy_policy': deepcopy(identity)}
        def step(self, enemy, defender):
            assert np.all(enemy == -1)
            return super().step(enemy, defender)
    config = {'episodes': 1, 'num_envs': 1, 'max_steps': 25, 'opponents': ['guard'],
              'calibration_version': 3, 'deployment_sampler': sampler}
    report = evaluation.evaluate(None, NativeEnv, 'cpu', config, native_policy=identity)
    assert report['native_policy'] == identity
    assert report['policy_sampler'] == sampler
    assert report['deterministic_policy'] is False
    assert report['reproducible_seeded_policy'] is True
    assert report['moving_no_progress_telemetry_complete']
    with pytest.raises(ValueError, match='actual native Go'):
        evaluation.evaluate(PolicyNetwork((8, 4)), NativeEnv, 'cpu', config)
    with pytest.raises(ValueError, match='identity'):
        evaluation.evaluate(None, LoopEnv, 'cpu', config, native_policy=identity)


def test_categorical_release_requires_exact_native_weights_rng_and_temperature(tmp_path):
    path = tmp_path / 'source.pt'
    categorical_source(path)
    identity = evaluation_artifact(capture_checkpoint(path), tmp_path / 'candidate')
    report = v3_report(path)
    report.update(native_policy=identity, policy_sampler=identity['metadata']['sampler'],
                  deterministic_policy=False, reproducible_seeded_policy=True)
    output = tmp_path / 'release'
    report_path = tmp_path / 'report.json'
    changes = [
        (lambda r: r.pop('native_policy'), 'native Go'),
        (lambda r: r.update(deterministic_policy=True), 'native Go'),
        (lambda r: r.update(reproducible_seeded_policy=False), 'native Go'),
        (lambda r: r['policy_sampler'].update(temperature=.5), 'sampler/temperature'),
        (lambda r: r['native_policy']['metadata']['sampler'].update(rng='torch'), 'native Go'),
        (lambda r: r['native_policy']['metadata'].update(weightsSHA256='0' * 64), 'weights/schema'),
        (lambda r: r['native_policy']['metadata'].update(observationDelayDecisions=0), 'weights/schema'),
    ]
    for change, message in changes:
        invalid = deepcopy(report)
        change(invalid)
        report_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match=message):
            export_checkpoint(path, output, policy_version='native-release', evaluation=report_path)
        assert not output.exists()
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='deployment_sampler'):
        export_checkpoint(path, output, policy_version='native-release', evaluation=report_path, temperature=.4)
    result = export_checkpoint(path, output, policy_version='native-release', evaluation=report_path)
    assert result['metadata']['sampler'] == identity['metadata']['sampler']
    assert result['metadata']['weightsSHA256'] == identity['metadata']['weightsSHA256']
    assert json.loads((output / 'deployment-notes.json').read_text())['calibrationVerified']
    qualified = qualify_checkpoint(path, report_path, tmp_path / 'qualified.pt', target_profile='expert')
    assert qualified['qualification']['weights_changed'] is False


def test_cli_native_candidate_uses_captured_checkpoint_and_cleans_temporary_artifact(tmp_path, monkeypatch):
    path = tmp_path / 'source.pt'
    categorical_source(path)
    snapshot = capture_checkpoint(path)
    seen, reports = [], []
    monkeypatch.setattr(cli, '_factory', lambda args: seen.append(Path(args.enemy_model)) or None)
    monkeypatch.setattr(cli, 'write_report', lambda report, output=None: reports.append(report))
    def native_evaluate(actor, factory, device, config, *, native_policy):
        assert actor is None and config['deployment_sampler']['mode'] == 'categorical'
        metadata = json.loads((seen[0] / 'metadata.json').read_text())
        assert metadata == native_policy['metadata']
        assert not json.loads((seen[0] / 'evaluation-only.json').read_text())['releaseCertified']
        altered = deepcopy(snapshot.state)
        altered['enemy_actor']['net.4.bias'] += 1
        save_checkpoint(path, altered)
        return {'native_policy': native_policy, 'policy_sampler': metadata['sampler'],
                'deterministic_policy': False, 'reproducible_seeded_policy': True}
    monkeypatch.setattr(evaluation, 'evaluate', native_evaluate)
    assert cli.main(['evaluate', '--checkpoint', str(path), '--device', 'cpu']) == 0
    assert reports[0]['checkpoint_identity']['sha256'] == snapshot.sha256
    assert reports[0]['native_policy']['metadata']['policyVersion'] == 'eval-' + snapshot.sha256[:20]
    assert capture_checkpoint(path).sha256 != snapshot.sha256
    assert not seen[0].exists()


@pytest.mark.parametrize('invalid', [
    {'temperature': float('nan')}, {'temperature': float('inf')}, {'temperature': 0},
    {'temperature': True}, {'mode': 'sample'}, {'rng': 'torch'}, {'unexpected': 1},
])
def test_sampler_contract_rejects_unsupported_values(invalid):
    with pytest.raises(ValueError):
        deployment_sampler(invalid)


def test_current_actor_helper_runs_native_go_without_release_claim():
    actor = PolicyNetwork((32, 16))
    config = {'level_id': 1, 'seed': 42, 'max_steps': 3, 'defender_bot': 'guard'}
    settings = {'calibration_version': 3, 'episodes': 1, 'num_envs': 1, 'max_steps': 3,
                'opponents': ['guard'], 'difficulty': 'expert',
                'deployment_sampler': {'mode': 'categorical', 'temperature': .3}}
    with BatchedEnv([config], workers=2) as env:
        result = evaluation.evaluate_current_actor_native(actor, env, 'cpu', settings, optimizer_updates=1)
        assert result['native_policy']['metadata']['sampler']['rng'] == 'xorshift64-v1'
        assert result['policy_sampler']['mode'] == 'categorical'
        assert result['release_certified'] is False
        assert 'checkpoint_identity' not in result
        with pytest.raises(ValueError, match='recorded enemy optimizer'):
            evaluation.evaluate_current_actor_native(actor, env, 'cpu', settings, optimizer_updates=0)
        with pytest.raises(ValueError, match='only tiny'):
            evaluation.evaluate_current_actor_native(PolicyNetwork((128, 64)), env, 'cpu', settings, optimizer_updates=1)


def test_trainer_routes_categorical_calibration_through_native_helper(tmp_path, monkeypatch):
    from nowarthunder_ai.trainer import train
    from test_training_lifecycle import FakeEnv, small_config
    class NativeCapableFake(FakeEnv):
        def factory_with_enemy_model(self, path):
            raise AssertionError('Test helper is replaced; no Go run expected')
    seen = []
    def native_helper(actor, env, device, config, *, optimizer_updates):
        seen.append((actor.hidden_sizes, config['deployment_sampler'], optimizer_updates))
        return {'deterministic_policy': False, 'release_certified': False}
    monkeypatch.setattr(evaluation, 'evaluate_current_actor_native', native_helper)
    parameters = small_config(iterations=1, actor_hidden=[32, 16], difficulty='expert',
                              calibration_version=3, deployment_sampler={'mode': 'categorical', 'temperature': .3},
                              eval_every=1, eval_episodes=1)
    train(parameters, NativeCapableFake, 'cpu', tmp_path / 'train')
    assert seen[0][0] == (32, 16) and seen[0][1]['mode'] == 'categorical' and seen[0][2] > 0
    with pytest.raises(ValueError, match='only tiny'):
        train({**parameters, 'actor_hidden': [128, 64]}, NativeCapableFake, 'cpu', tmp_path / 'teacher')
