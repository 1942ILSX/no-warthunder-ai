import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nowarthunder_ai.config import load_config
from nowarthunder_ai.env import BatchedEnv
from nowarthunder_ai.export import write_artifact, verify_artifact, export_checkpoint
from nowarthunder_ai.models import PolicyNetwork


def test_real_go_bridge_determinism_cutoff_and_partial_reset():
    config = {'level_id': 1, 'seed': 42, 'max_steps': 20, 'defender_bot': 'mixed'}
    with BatchedEnv([config, config]) as env:
        for _ in range(20):
            frame = env.step(np.full((2, 5), -1, dtype=np.int64), np.full(2, -1, dtype=np.int64))
            np.testing.assert_array_equal(frame.critic_obs[0], frame.critic_obs[1])
            assert np.isfinite(frame.enemy_obs).all()
            assert np.max(np.abs(frame.enemy_obs)) <= 1
        assert frame.truncated.all() and not frame.terminated.any()
        assert np.all(frame.enemy_reward == 0)
        before = frame.critic_obs[1].copy()
        frame = env.reset_indices([0], [{**config, 'seed': 43}])
        np.testing.assert_array_equal(frame.critic_obs[1], before)
        assert not frame.truncated[0] and frame.truncated[1]
        with pytest.raises(ValueError):
            env.step(np.full((2, 5), 999, dtype=np.int64), np.full(2, -1, dtype=np.int64))
        env.step(np.full((2, 5), -1, dtype=np.int64), np.full(2, -1, dtype=np.int64))
        saved_configs = [dict(item) for item in env.configs]
        for indices in (None, [0]):
            with pytest.raises(RuntimeError, match='Go simulation'):
                invalid = {**config, 'level_id': 99}
                env.reset([invalid, config]) if indices is None else env.reset_indices(indices, [invalid])
            assert env.configs == saved_configs
        env.reset()
        process = env.process
    assert process.poll() is not None


@pytest.mark.parametrize('difficulty', ['trainee', 'expert', 'ace'])
def test_export_python_go_logits_masks_and_both_samplers(tmp_path, difficulty):
    torch.manual_seed(734)
    actor = PolicyNetwork((64, 32))
    destination = tmp_path / difficulty
    metadata = write_artifact(actor, destination, policy_version=f'{difficulty}-test',
                              training_seed=734, trained=False, difficulty=difficulty,
                              sampler='categorical', temperature=.7)
    assert (destination / 'weights.bin').stat().st_size == 91816
    assert metadata['difficulty'] == difficulty and not metadata['trained']
    report = verify_artifact(destination, samples=48)
    assert report['passed']
    assert all(mode['action_matches'] == 48 for mode in report['samplers'].values())


def test_profiles_are_independent_and_multipliers_are_metadata_only():
    profiles = [load_config(profile=name) for name in ('trainee', 'expert', 'ace')]
    assert len({c['seed'] for c in profiles}) == 3
    assert len({tuple(c['target_defender_win_rate']) for c in profiles}) == 3
    assert len({c['score_multiplier'] for c in profiles}) == 3
    assert all(c['gamma'] == .999 for c in profiles)
    assert load_config('configs/trainee.yaml')['difficulty'] == 'trainee'


def test_export_rejects_smoke_and_defender_only_provenance(tmp_path):
    for kind, updates in [('training', 0), ('smoke', 3), ('smoke_distillation', 3)]:
        checkpoint = tmp_path / (kind + '.pt')
        torch.save({'format_version': 1, 'kind': kind, 'trained': True, 'optimizer_updates': 3,
                    'enemy_optimizer_updates': updates}, checkpoint)
        with pytest.raises(ValueError, match='Untrained/smoke'):
            export_checkpoint(checkpoint, tmp_path / 'output', policy_version='test')
