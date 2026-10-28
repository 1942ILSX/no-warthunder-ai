from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nowarthunder_ai.checkpoint import load_checkpoint, save_checkpoint
from nowarthunder_ai.models import PolicyNetwork
from nowarthunder_ai.provenance import calibration_campaigns
from nowarthunder_ai.recovery import (ReplayReservoir, navigation_targets,
                                     recover, recovery_config, recovery_loss)


def source(path, *, smoke=False):
    actor = PolicyNetwork((32, 16))
    actor.set_masked_features([309])
    with torch.no_grad():
        actor.net[-1].bias[6] = 2
    state = {'format_version': 1, 'kind': 'smoke_training' if smoke else 'training',
             'trained': True, 'enemy_optimizer_updates': 10, 'difficulty': 'trainee',
             'observation_version': 2, 'normalization_version': 2,
             'model_spec': {'enemy_hidden': [32, 16], 'observation_size': 320, 'action_size': 10},
             'config': {'difficulty': 'trainee', 'calibration_version': 2,
                        'target_defender_win_rate': [.7, .9], 'eval_seed': 3_000_000, 'eval_episodes': 100,
                        'difficulty_control': {'version': 1, 'masked_features': [309]}},
             'enemy_actor': actor.state_dict()}
    save_checkpoint(path, state)
    return state


class RepairEnv:
    instances = []
    blocked = True

    def __init__(self, configs):
        self.configs = list(configs)
        self.all_configs = deepcopy(configs)
        self.n = len(configs)
        self.steps = np.zeros(self.n, dtype=int)
        self.actions = []
        self.closed = False
        self.metadata = {'observation_version': 2, 'normalization_version': 2}
        self.frame = self.make_frame()
        self.instances.append(self)

    def make_frame(self):
        obs = np.zeros((self.n, 5, 320), dtype=np.float32)
        obs[..., 0] = self.steps[:, None] / 24
        obs[..., 217] = 0
        obs[..., 294:298] = [.1, .2, .3, .4]
        obs[..., 317] = self.blocked * (self.steps[:, None] % 2)
        obs[..., 309] = 1
        mask = np.ones((self.n, 5, 10), dtype=bool)
        active = np.zeros((self.n, 5), dtype=bool)
        active[:, 0] = True
        ids = np.zeros((self.n, 5), dtype=int)
        ids[:, 0] = 7
        return SimpleNamespace(enemy_obs=obs, enemy_mask=mask, enemy_active=active, enemy_ids=ids,
                               terminated=self.steps >= 4, truncated=np.zeros(self.n, dtype=bool),
                               infos=[{'level_id': config['level_id'], 'seed': config['seed'],
                                       'decision_step': int(self.steps[i])} for i, config in enumerate(self.configs)])

    def step(self, actions, defender):
        self.actions.append(actions.copy())
        assert np.all(defender == -1)
        self.steps += 1
        self.frame = self.make_frame()
        return self.frame

    def reset_indices(self, indices, configs):
        self.steps[indices] = 0
        self.all_configs.extend(deepcopy(configs))
        for index, config in zip(indices, configs):
            self.configs[index] = config
        self.frame = self.make_frame()
        return self.frame

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True


def test_recovery_uses_positive_student_trajectories_and_preserves_contract(tmp_path):
    path = tmp_path / 'source.pt'
    before = source(path)
    original_bytes = path.read_bytes()
    RepairEnv.instances.clear()
    result = recover(path, RepairEnv, 'cpu', tmp_path / 'repair',
                     {'iterations': 2, 'num_envs': 20, 'collection_steps': 8, 'max_steps': 4,
                      'updates_per_iteration': 2, 'batch_size': 8, 'max_records': 30,
                      'max_recovery_records': 10, 'anchor_every': 1})
    saved = load_checkpoint(result['checkpoint'])
    assert path.read_bytes() == original_bytes
    assert saved['kind'] == 'recovery'
    assert saved['enemy_optimizer_updates'] == 14
    assert saved['calibration_version'] == saved['config']['calibration_version'] == 3
    assert saved['config']['target_defender_win_rate'] == [.7, .9]
    assert saved['config']['difficulty_control'] == before['config']['difficulty_control']
    assert saved['recovery_source']['sha256'] == result['source_sha256']
    assert saved['recovery_source']['source_calibration_version'] == 2
    assert not saved['selection']['calibrated'] and not saved['calibration']['eligible']
    assert saved['recovery_training']['collector_policy'] == 'student_argmax'
    assert not saved['recovery_training']['runtime_overrides']
    assert torch.count_nonzero(saved['enemy_actor']['net.0.weight'][:, 309]) == 0
    assert any(not torch.equal(saved['enemy_actor'][key], value) for key, value in before['enemy_actor'].items())
    assert all(env.closed for env in RepairEnv.instances)
    # At the first blocked state the actor still chooses South (action6),
    # while the recovery label is North. Collector must not apply that label.
    assert np.all(RepairEnv.instances[0].actions[1][:, 0] == 6)
    configurations = [config for env in RepairEnv.instances for config in env.all_configs]
    seeds = [config['seed'] for config in configurations]
    assert min(seeds) > 0 and len(seeds) == len(set(seeds))
    assert {(config['level_id'], config['defender_bot']) for config in configurations} == {
        (level, bot) for level in range(1, 11) for bot in ('guard', 'hunter')}
    assert any(item['seed'] == 3_000_000 and item['episodes'] == 100 for item in calibration_campaigns(saved))
    assert result['metrics']['anchor_records'] <= 20
    assert result['metrics']['recovery_records'] <= 10


def test_empty_recovery_replay_does_not_modify_weights_and_smoke_propagates(tmp_path):
    class NoRepair(RepairEnv):
        blocked = False
    path = tmp_path / 'smoke.pt'
    state = source(path, smoke=True)
    result = recover(path, NoRepair, 'cpu', tmp_path / 'repair',
                     {'iterations': 1, 'num_envs': 2, 'collection_steps': 3,
                      'updates_per_iteration': 2, 'max_records': 10, 'max_recovery_records': 5})
    saved = load_checkpoint(result['checkpoint'])
    assert saved['kind'] == 'smoke_recovery' and saved['config']['smoke']
    assert result['optimizer_updates'] == 0
    for key, value in state['enemy_actor'].items():
        assert torch.equal(saved['enemy_actor'][key], value)


def test_navigation_target_respects_stuck_alternatives_and_legality():
    observation = np.zeros((3, 320), dtype=np.float32)
    observation[:, 294:298] = [.1, .2, .3, .4]
    observation[0, 317] = 1
    observation[0, 8] = .25
    mask = np.ones((3, 10), dtype=bool)
    mask[1, 2:6] = False
    mask[2, 2:] = False
    target, feasible = navigation_targets(observation, mask)
    assert target[:2].tolist() == [2, 3]
    assert feasible.tolist() == [True, True, False]


def test_movement_loss_rewards_both_fire_variants_and_anchor_kl_is_zero_at_source():
    actor = PolicyNetwork((32, 16))
    obs = torch.zeros(2, 320)
    mask = torch.ones(2, 10, dtype=torch.bool)
    target = torch.tensor([1, 1])
    original = actor(obs).detach()
    batch = obs, mask, original, target
    loss, metrics = recovery_loss(actor, batch, batch, kl_coef=5., recovery_coef=1.)
    assert metrics['anchor_kl'].item() == pytest.approx(0., abs=1e-7)
    loss.backward()
    assert actor.net[-1].bias.grad[2] < 0 and actor.net[-1].bias.grad[3] < 0
    assert actor.net[-1].bias.grad[4] > 0


def test_replay_and_settings_are_bounded():
    reservoir = ReplayReservoir(3, np.random.default_rng(12))
    reservoir.add(np.zeros((20, 320), dtype=np.float32), np.ones((20, 10), dtype=bool),
                  np.zeros((20, 10), dtype=np.float32), np.ones(20, dtype=np.int64))
    assert reservoir.size == 3 and reservoir.seen == 20
    assert reservoir.sample(8, 'cpu')[0].shape == (8, 320)
    for settings in ({'seed': -3_000_000}, {'max_records': 200001}, {'learning_rate': float('nan')},
                     {'max_records': 5, 'max_recovery_records': 5}, {'runtime_override': True}):
        with pytest.raises(ValueError):
            recovery_config(settings)
