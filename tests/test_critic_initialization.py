from copy import deepcopy

import pytest
import torch

from nowarthunder_ai import trainer
from nowarthunder_ai.checkpoint import load_checkpoint, save_checkpoint
from test_training_lifecycle import FakeEnv, small_config


def make_source(tmp_path):
    result = trainer.train(small_config(iterations=2), FakeEnv, 'cpu', tmp_path / 'source')
    return result['checkpoint'], load_checkpoint(result['checkpoint'])


def test_optional_source_critics_are_loaded_before_fresh_optimizers(tmp_path, monkeypatch):
    source_path, source = make_source(tmp_path)
    original = open(source_path, 'rb').read()
    observed = []
    real_learner = trainer.PPOLearner
    def record_learner(actor, critic, config, controller):
        learner = real_learner(actor, critic, config, controller)
        observed.append({key: value.detach().clone() for key, value in critic.state_dict().items()})
        assert not learner.actor_optimizer.state and not learner.critic_optimizer.state
        return learner
    monkeypatch.setattr(trainer, 'PPOLearner', record_learner)
    parameters = small_config(iterations=1, initialize_critics_from_source=True)
    warmed = trainer.train(parameters, FakeEnv, 'cpu', tmp_path / 'warm', initial_enemy_checkpoint=source_path)
    for index, role in enumerate(('enemy', 'defender')):
        for key, value in source[f'{role}_critic'].items():
            torch.testing.assert_close(observed[index][key], value, rtol=0, atol=0)
    result = load_checkpoint(warmed['checkpoint'])
    assert result['initial_source']['critic_initialization'] == 'source'
    assert result['iteration'] == 1
    assert result['optimizer_updates'] < source['optimizer_updates']
    assert result['defender_optimizer_updates'] == 0 < source['defender_optimizer_updates']
    assert open(source_path, 'rb').read() == original

    fresh = trainer.train(small_config(iterations=1), FakeEnv, 'cpu', tmp_path / 'fresh',
                          initial_enemy_checkpoint=source_path)
    assert load_checkpoint(fresh['checkpoint'])['initial_source']['critic_initialization'] == 'fresh'
    assert any(not torch.equal(observed[2][key], value) for key, value in source['enemy_critic'].items())
    # The saved origin option is compatible with resuming the new run; resume
    # restores that run's optimizers instead of consulting its old source again.
    resumed = trainer.train({**parameters, 'iterations': 2}, FakeEnv, 'cpu', tmp_path / 'warm',
                             resume=warmed['checkpoint'])
    assert load_checkpoint(resumed['checkpoint'])['initial_source']['critic_initialization'] == 'source'


@pytest.mark.parametrize('change,message', [
    (lambda state: state['model_spec'].update(critic_size=999), 'schema'),
    (lambda state: state['model_spec'].update(critic_hidden=[999]), 'schema'),
    (lambda state: state.pop('defender_critic'), 'missing'),
    (lambda state: state['enemy_critic'].update({'net.0.weight': torch.zeros(1, 1)}), 'shapes'),
    (lambda state: state['enemy_critic']['net.0.weight'].fill_(float('nan')), 'nonfinite'),
])
def test_requested_critic_restore_rejects_incompatible_source(tmp_path, change, message):
    source_path, source = make_source(tmp_path)
    invalid = deepcopy(source)
    change(invalid)
    path = save_checkpoint(tmp_path / 'invalid.pt', invalid)
    with pytest.raises(ValueError, match=message):
        trainer.train(small_config(iterations=1, initialize_critics_from_source=True),
                      FakeEnv, 'cpu', tmp_path / 'rejected', initial_enemy_checkpoint=path)


def test_critic_initialization_requires_boolean_and_a_source(tmp_path):
    with pytest.raises(ValueError, match='boolean'):
        trainer.resolved_config(small_config(initialize_critics_from_source='yes'))
    with pytest.raises(ValueError, match='requires an initial checkpoint'):
        trainer.train(small_config(iterations=1, initialize_critics_from_source=True),
                      FakeEnv, 'cpu', tmp_path / 'invalid')
