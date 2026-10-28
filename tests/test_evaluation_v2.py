from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from nowarthunder_ai.evaluation import (StationaryTracker, assess_report,
                                        calibration_metrics, evaluate, summarize_records)
from nowarthunder_ai.models import PolicyNetwork


def records(telemetry=True):
    result = []
    for opponent in ('guard', 'hunter', 'wander'):
        for level in range(1, 11):
            for sample in range(5):
                item = {'opponent': opponent, 'level_id': level,
                        'outcome': 'defender_win' if opponent != 'wander' and sample < 4 else 'enemy_win_base',
                        'enemy_alive_steps': 100., 'enemy_stuck_steps': 0., 'enemy_return': 0.,
                        'truncated': False, 'decision_step': 100}
                if telemetry:
                    item.update(enemy_unproductive_steps=0., enemy_observed_steps=100.)
                result.append(item)
    return result


def configuration(**extra):
    return {'calibration_version': 2, 'difficulty': 'trainee',
            'target_defender_win_rate': [.7, .9], **extra}


def test_v2_band_excludes_diagnostic_wander_without_changing_aggregate():
    rows = records()
    report = summarize_records(rows, version=2)
    report['records'] = rows
    assert report['defender_win_rate'] == pytest.approx(80 / 150)
    assert report['primary']['defender_win_rate'] == .8
    assert report['primary']['episodes'] == 100
    assert report['by_level']['1']['episodes'] == 15
    assert report['by_level_opponent']['1']['wander']['defender_win_rate'] == 0
    assessment = calibration_metrics(report, configuration())
    assert assessment['eligible'] and assessment['coverage_passed']
    assert assessment['episodes'] == 100 and assessment['total_episodes'] == 150
    assert assessment['reasons'] == []


def test_v2_level_gates_catch_failures_hidden_by_healthy_aggregate():
    rows = records()
    for row in rows:
        if row['opponent'] != 'wander' and row['level_id'] == 8:
            row['enemy_stuck_steps'] = 30.
        if row['opponent'] != 'wander' and row['level_id'] == 9:
            row['enemy_unproductive_steps'] = 30.
    for row in [r for r in rows if r['level_id'] == 9 and r['opponent'] == 'guard'][:2]:
        row['outcome'] = 'timeout'
    report = summarize_records(rows, version=2)
    assert report['primary']['timeout_rate'] == .02
    assert report['primary']['stagnation_rate'] == .03
    assert report['primary']['unproductive_rate'] == .03
    assessment = calibration_metrics(report, configuration())
    assert not assessment['quality_passed'] and not assessment['eligible']
    reasons = assessment['reasons']
    assert 'level:8.stagnation_rate:quality_limit_exceeded' in reasons
    assert 'level:9.unproductive_rate:quality_limit_exceeded' in reasons
    assert 'level:9.timeout_rate:quality_limit_exceeded' in reasons
    assert not any(reason.startswith('primary.') and 'quality_limit_exceeded' in reason for reason in reasons)


def test_v2_requires_each_level_opponent_cell_and_sufficient_primary_games():
    rows = records()
    rows = [r for r in rows if not (r['opponent'] == 'hunter' and r['level_id'] == 10)]
    report = summarize_records(rows, version=2)
    assessment = calibration_metrics(report, configuration())
    assert not assessment['coverage_passed']
    assert 'level:10/opponent:hunter.episodes:insufficient_coverage' in assessment['reasons']
    assert 'primary.episodes:insufficient_coverage' in assessment['reasons']


def test_retrospective_v1_report_never_invents_stationary_telemetry():
    rows = records(telemetry=False)
    original = {**summarize_records(rows), 'records': rows,
                'calibration': {'eligible': True, 'legacy_marker': 1}}
    before = deepcopy(original)
    updated = assess_report(original, configuration())
    assert original == before
    assert updated['primary']['unproductive_rate'] is None
    assert updated['previous_calibration'] == original['calibration']
    assert not updated['calibration']['quality_passed']
    assert 'primary.unproductive_rate:missing_unproductive_telemetry' in updated['calibration']['reasons']
    assert updated['source_metrics_version'] == 1
    assert 'enemy_unproductive_steps' not in updated['records'][0]


def frame():
    observation = np.zeros((1, 5, 320), dtype=np.float32)
    observation[..., 2] = 1
    active = np.zeros((1, 5), dtype=bool)
    active[0, 0] = True
    ids = np.zeros((1, 5), dtype=np.int32)
    ids[0, 0] = 7
    return SimpleNamespace(enemy_obs=observation, enemy_active=active, enemy_ids=ids)


def test_stationary_tracker_waits_twenty_intervals_and_resets_identity_motion_and_activity():
    item = frame()
    tracker = StationaryTracker((1, 5))
    complete = np.array([False])
    for _ in range(20):
        assert tracker.update(item, complete)[0].tolist() == [0]
    unproductive, observed = tracker.update(item, complete)
    assert unproductive.tolist() == observed.tolist() == [1]
    item.enemy_ids[0, 0] = 8
    assert tracker.update(item, complete)[0].tolist() == [0]
    for _ in range(20):
        tracker.update(item, complete)
    item.enemy_obs[0, 0, 0] += .1 / 24
    assert tracker.update(item, complete)[0].tolist() == [0]
    item.enemy_active[0, 0] = False
    unproductive, observed = tracker.update(item, complete)
    assert unproductive.tolist() == observed.tolist() == [0]
    item.enemy_active[0, 0] = True
    assert tracker.update(item, complete)[0].tolist() == [0]


def test_stationary_tracker_requires_recent_fire_at_current_facing_bricks_or_targets():
    item = frame()
    tracker = StationaryTracker((1, 5))
    complete = np.array([False])
    for _ in range(21):
        tracker.update(item, complete)
    item.enemy_obs[0, 0, 314] = 1  # right-side target; tank faces up
    assert tracker.update(item, complete)[0].tolist() == [1]
    item.enemy_obs[0, 0, 313] = .1
    assert tracker.update(item, complete)[0].tolist() == [1]  # aim alone cannot excuse indefinite inactivity
    item.enemy_obs[0, 0, 9] = 1
    assert tracker.update(item, complete)[0].tolist() == [0]
    item.enemy_obs[0, 0, 313] = .8
    assert tracker.update(item, complete)[0].tolist() == [0]
    assert tracker.update(item, np.array([True]))[1].tolist() == [0]


def test_stationary_target_facing_camper_eventually_counts_without_fire():
    item = frame()
    item.enemy_obs[0, 0, 313] = 1
    tracker = StationaryTracker((1, 5))
    complete = np.array([False])
    for _ in range(20):
        assert tracker.update(item, complete)[0].tolist() == [0]
    for _ in range(40):
        assert tracker.update(item, complete)[0].tolist() == [1]


def test_stationary_firing_and_bounded_cooldown_waits_remain_productive():
    item = frame()
    item.enemy_obs[0, 0, 313] = .3
    item.enemy_obs[0, 0, 9] = 1
    tracker = StationaryTracker((1, 5))
    complete = np.array([False])
    for _ in range(60):
        assert tracker.update(item, complete)[0].tolist() == [0]
    item.enemy_obs[0, 0, 9] = 0
    for _ in range(19):
        assert tracker.update(item, complete)[0].tolist() == [0]
    assert tracker.update(item, complete)[0].tolist() == [1]
    item.enemy_obs[0, 0, 9] = 1
    assert tracker.update(item, complete)[0].tolist() == [0]
    item.enemy_obs[0, 0, 313] = 0  # meaningless firing cannot exempt stationary behavior
    assert tracker.update(item, complete)[0].tolist() == [1]


def test_stationary_recent_fire_does_not_leak_between_tank_identities():
    item = frame()
    item.enemy_obs[0, 0, 313] = 1
    item.enemy_obs[0, 0, 9] = 1
    tracker = StationaryTracker((1, 5), intervals=2)
    complete = np.array([False])
    tracker.update(item, complete)
    item.enemy_ids[0, 0] = 8
    item.enemy_obs[0, 0, 9] = 0
    tracker.update(item, complete)
    tracker.update(item, complete)
    assert tracker.update(item, complete)[0].tolist() == [1]


class StationaryEnv:
    def __init__(self, configs):
        self.steps = 0
        self.configs = configs
        self.metadata = {}
        self.frame = self.make_frame()

    def make_frame(self):
        item = frame()
        done = self.steps >= 25
        item.enemy_active[0, 0] = not done
        item.enemy_mask = np.ones((1, 5, 10), dtype=bool)
        item.defender_obs = np.zeros((1, 320), dtype=np.float32)
        item.defender_mask = np.ones((1, 10), dtype=bool)
        item.defender_active = np.array([not done])
        item.enemy_reward = np.zeros(1)
        item.terminated, item.truncated = np.array([done]), np.array([False])
        item.infos = [{'outcome': 'defender_win' if done else 'playing', 'level_id': 1,
                       'decision_step': self.steps, 'enemy_alive': 1, 'enemy_stuck': 0}]
        return item

    def step(self, *_):
        self.steps += 1
        self.frame = self.make_frame()
        return self.frame

    def close(self):
        pass


def test_evaluate_measures_v2_stationary_counts_but_preserves_v1_record_contract():
    actor = PolicyNetwork((8, 4))
    config = {'episodes': 1, 'num_envs': 1, 'max_steps': 25, 'opponents': ['guard']}
    legacy = evaluate(actor, StationaryEnv, 'cpu', config)
    assert 'metrics_version' not in legacy and 'primary' not in legacy
    assert 'enemy_unproductive_steps' not in legacy['records'][0]
    measured = evaluate(actor, StationaryEnv, 'cpu', {**config, 'calibration_version': 2})
    record = measured['records'][0]
    assert record['enemy_observed_steps'] == 25
    assert record['enemy_unproductive_steps'] == 5
    assert measured['primary']['unproductive_rate'] == .2
