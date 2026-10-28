import json

import pytest
import torch

from nowarthunder_ai import cli
from nowarthunder_ai.checkpoint import save_checkpoint
from nowarthunder_ai.config import load_config
from nowarthunder_ai.evaluation import summarize_records
from nowarthunder_ai.export import export_checkpoint, verify_artifact
from nowarthunder_ai.models import PolicyNetwork


def source_checkpoint(path, *, smoke=False):
    actor = PolicyNetwork((64, 32))
    config = load_config('configs/difficulty/expert.yaml')
    state = {'format_version': 1, 'kind': 'smoke' if smoke else 'training',
             'trained': True, 'enemy_optimizer_updates': 10,
             'difficulty': 'expert', 'config': config,
             'observation_version': 2, 'normalization_version': 2,
             'defender_observation_version': 1,
             'model_spec': {'enemy_hidden': (64, 32), 'observation_size': 320, 'action_size': 10},
             'enemy_actor': actor.state_dict()}
    save_checkpoint(path, state)
    return state


def report_for(path, seed=4_000_000):
    records = []
    for name in ('guard', 'hunter', 'wander'):
        for index in range(50):
            records.append({'opponent': name, 'level_id': index % 10 + 1,
                            'seed': -(seed + index + 1),
                            'outcome': 'defender_win' if index % 2 else 'enemy_win_base',
                            'enemy_alive_steps': 100, 'enemy_stuck_steps': 0,
                            'enemy_observed_steps': 100, 'enemy_unproductive_steps': 0,
                            'truncated': False})
    return {**summarize_records(records, version=2), 'records': records, 'difficulty': 'expert',
            'checkpoint_identity': {'sha256': cli.file_sha256(path)},
            'deterministic_policy': True, 'policy_sampler': {'mode': 'argmax', 'temperature': 1.},
            'evaluation_config': {'eval_seed': seed, 'episodes': 50}}


def test_v2_release_requires_matching_independent_quality_evaluation(tmp_path):
    checkpoint = tmp_path / 'student.pt'
    source_checkpoint(checkpoint)
    destination = tmp_path / 'release'
    with pytest.raises(ValueError, match='requires --evaluation'):
        export_checkpoint(checkpoint, destination, policy_version='expert-v2')
    report_path = tmp_path / 'report.json'
    variants = []
    wrong = report_for(checkpoint)
    wrong['checkpoint_identity']['sha256'] = '0' * 64
    variants.append((wrong, 'SHA256'))
    variants.append((report_for(checkpoint, 3_000_000), 'overlaps'))
    failed = report_for(checkpoint)
    for record in failed['records']:
        if record['level_id'] == 8:
            record['enemy_stuck_steps'] = 100
    # Derived aggregates and any claimed eligibility must not override records.
    failed['calibration'] = {'eligible': True}
    variants.append((failed, 'quality gates'))
    for report, error in variants:
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match=error):
            export_checkpoint(checkpoint, destination, policy_version='expert-v2', evaluation=report_path)
        assert not destination.exists()
    report_path.write_text(json.dumps(report_for(checkpoint)))
    with pytest.raises(ValueError, match='argmax'):
        export_checkpoint(checkpoint, destination, policy_version='expert-v2', evaluation=report_path, sampler='categorical')
    exported = export_checkpoint(checkpoint, destination, policy_version='expert-v2', evaluation=report_path)
    assert exported['weights_bytes'] == 91816
    assert json.loads((destination / 'deployment-notes.json').read_text())['calibrationVerified']
    assert verify_artifact(destination, samples=32)['passed']


def test_consumed_final_campaign_is_not_reused():
    state = {'config': load_config('configs/difficulty/ace.yaml')}
    assert cli.evaluation_campaign(state, 2_000_000, 100)['overlapping_seeds_per_opponent'] == 100
    assert cli.evaluation_campaign(state, -3_000_001, 50)['overlapping_seeds_per_opponent'] == 49
    assert cli.evaluation_campaign(state, 4_000_000, 100)['purpose'] == 'final_test'


def test_tune_plan_validates_every_output_and_preserves_source(tmp_path):
    checkpoint = tmp_path / 'student.pt'
    source_checkpoint(checkpoint)
    before = cli.file_sha256(checkpoint)
    output = tmp_path / 'runs'
    plan = cli.difficulty_run_plan(str(checkpoint), str(output))
    assert [p['profile'] for p in plan] == ['trainee', 'expert', 'ace']
    assert len({p['config']['seed'] for p in plan}) == 3
    assert [p['config']['difficulty_control']['fire_budget'] for p in plan] == [.2, .45, 1.]
    assert not output.exists() and cli.file_sha256(checkpoint) == before
    (output / 'ace').mkdir(parents=True)
    (output / 'ace' / 'existing.txt').write_text('preserve')
    with pytest.raises(FileExistsError):
        cli.difficulty_run_plan(str(checkpoint), str(output))
    assert not (output / 'trainee').exists()
    smoke = tmp_path / 'smoke.pt'
    source_checkpoint(smoke, smoke=True)
    with pytest.raises(ValueError, match='non-smoke'):
        cli.difficulty_run_plan(str(smoke), str(tmp_path / 'smoke-run'))
