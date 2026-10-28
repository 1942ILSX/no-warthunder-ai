"""Bounded deterministic-visitation repair for an existing tiny enemy actor.

The collector always executes the student's own argmax action. Temporal,
stationary and blocked labels are offline supervision only, never an action
override or an additional deployed input. All optimized episodes use a fresh
positive training-seed stream; negative calibration episodes are not accepted.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import (actor_from_checkpoint, cpu_state, log_metrics,
                         prepare_output, save_checkpoint, save_json,
                         validate_checkpoint_environment)
from .evaluation import StationaryTracker
from .progress import ProgressTracker, progress_config
from .provenance import calibration_campaigns, capture_checkpoint, freeze_checkpoint


DEFAULTS = {
    'iterations': 5, 'num_envs': 32, 'collection_steps': 1200, 'max_steps': 9000,
    'seed': 8_100_001, 'updates_per_iteration': 100, 'batch_size': 256,
    'learning_rate': 1e-5, 'kl_coef': 5., 'recovery_coef': 1.,
    'max_records': 60_000, 'max_recovery_records': 20_000, 'anchor_every': 4,
    'checkpoint_every': 1, 'max_grad_norm': .5,
}


def recovery_config(value: Mapping | None = None) -> dict:
    value = dict(value or {})
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown recovery settings: {sorted(unknown)}')
    result = {**DEFAULTS, **value}
    bounds = {'iterations': (1, 1000), 'num_envs': (1, 128),
              'collection_steps': (1, 9000), 'max_steps': (1, 9000),
              'seed': (1, 2**63 - 1), 'updates_per_iteration': (1, 10000),
              'batch_size': (1, 8192), 'max_records': (2, 200_000),
              'max_recovery_records': (1, 199_999), 'anchor_every': (1, 1000),
              'checkpoint_every': (1, 1000)}
    for key, (low, high) in bounds.items():
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'recovery.{key} must be an integer in {low}..{high}')
    if result['max_recovery_records'] >= result['max_records']:
        raise ValueError('Recovery replay must leave space for non-recovery anchors')
    for key in ('learning_rate', 'kl_coef', 'recovery_coef', 'max_grad_norm'):
        item = result[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            raise ValueError(f'recovery.{key} must be finite and positive')
    # At most one reset per room per step, plus the initial rooms each round.
    maximum_episodes = result['iterations'] * result['num_envs'] * (result['collection_steps'] + 1)
    if result['seed'] + maximum_episodes >= 2**63:
        raise ValueError('Positive training seed stream would exceed int64')
    return result


class ReplayReservoir:
    """Fixed allocation with uniform reservoir replacement over offered rows."""
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        self.size = self.seen = 0
        self.observations = np.empty((capacity, 320), dtype=np.float32)
        self.masks = np.empty((capacity, 10), dtype=bool)
        self.teacher_logits = np.empty((capacity, 10), dtype=np.float32)
        self.target_moves = np.empty(capacity, dtype=np.int64)

    def add(self, observations, masks, teacher_logits, target_moves):
        for observation, mask, logits, target in zip(observations, masks, teacher_logits, target_moves):
            self.seen += 1
            index = self.seen - 1 if self.size < self.capacity else int(self.rng.integers(self.seen))
            if index >= self.capacity:
                continue
            self.observations[index] = observation
            self.masks[index] = mask
            self.teacher_logits[index] = logits
            self.target_moves[index] = target
            self.size = min(self.capacity, self.size + 1)

    def sample(self, count: int, device):
        if not self.size:
            raise ValueError('Cannot sample empty recovery replay')
        indices = self.rng.integers(self.size, size=count)
        return tuple(torch.as_tensor(array[indices], device=device)
                     for array in (self.observations, self.masks, self.teacher_logits, self.target_moves))


def navigation_targets(observation: np.ndarray, action_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same movement target as difficulty_actor_loss, including stuck alternatives."""
    role = np.rint(observation[..., 308] * 3).astype(np.int64)
    use_base = ((role == 0) | (observation[..., 217] < .5)
                | ((role == 3) & (np.abs(observation[..., 208:210]).sum(-1) > .32)))
    navigation = np.where(use_base[..., None], observation[..., 294:298], observation[..., 290:294])
    feasible = action_mask[..., 2:].reshape(*action_mask.shape[:-1], 4, 2).any(-1) & (navigation < .999)
    previous = np.rint(observation[..., 8] * 4).astype(np.int64).clip(0, 4)
    alternatives = feasible & (np.arange(4) != previous[..., None] - 1)
    stuck = observation[..., 317] > .35
    feasible = np.where((stuck & alternatives.any(-1))[..., None], alternatives, feasible)
    target = np.where(feasible, navigation, 10.).argmin(-1) + 1
    return target, feasible.any(-1)


def recovery_flags(frame, progress: ProgressTracker, stationary: StationaryTracker):
    active = frame.enemy_active & (frame.enemy_ids != 0)
    temporal = progress.update(frame) & active
    stationary.update(frame, np.zeros(active.shape[0], dtype=bool))
    observation = frame.enemy_obs
    facing = observation[..., 2:6].argmax(-1)
    aim = np.take_along_axis(observation[..., 313:317], facing[..., None], axis=-1)[..., 0]
    engaged = (aim >= .1) & (stationary.fire_age < stationary.recent_fire_window)
    idle = active & (stationary.stationary >= stationary.intervals) & ~engaged
    blocked = active & (observation[..., 317] > .35)
    target, feasible = navigation_targets(observation, frame.enemy_mask)
    return (temporal | idle | blocked) & feasible, target, {
        'temporal': int(temporal.sum()), 'stationary': int(idle.sum()), 'blocked': int(blocked.sum())}


def recovery_loss(student, anchors, recovery, *, kl_coef: float, recovery_coef: float):
    """Teacher KL on ordinary states; movement-marginal CE only on repair states."""
    observation, mask, _, targets = recovery
    logp = F.log_softmax(student(observation).masked_fill(~mask, -1e9), dim=-1)
    movement_logp = torch.logsumexp(logp[:, 2:].reshape(-1, 4, 2), dim=-1)
    ce = -movement_logp.gather(1, (targets - 1).unsqueeze(-1)).mean()
    kl = ce.new_zeros(())
    if anchors is not None:
        observation, mask, original_logits, _ = anchors
        student_logp = F.log_softmax(student(observation).masked_fill(~mask, -1e9), dim=-1)
        teacher_logp = F.log_softmax(original_logits.masked_fill(~mask, -1e9), dim=-1)
        kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()
    return recovery_coef * ce + kl_coef * kl, {'recovery_ce': ce.detach(), 'anchor_kl': kl.detach()}


def recover(checkpoint: str | Path, env_factory: Callable, device: str | torch.device,
            output_dir: str | Path, config: Mapping | None = None) -> dict:
    """Collect deterministic student states and update a bounded tiny actor.

    This is a new repair run, not a PPO resume. Candidate checkpoints retain the
    source difficulty/quality contract but invalidate calibration eligibility.
    The caller separately evaluates candidates and chooses a fresh final test.
    """
    settings = recovery_config(config)
    source = capture_checkpoint(checkpoint)
    state = source.state
    if not state.get('trained') or int(state.get('enemy_optimizer_updates', 0)) <= 0:
        raise ValueError('Recovery requires an already trained source enemy actor')
    if state.get('observation_version') != 2 or state.get('normalization_version') != 2:
        raise ValueError('Recovery requires observation/normalization version 2')
    spec = state.get('model_spec', {})
    if (tuple(spec.get('enemy_hidden', ())) not in {(64, 32), (32, 16)}
            or spec.get('observation_size') != 320 or spec.get('action_size') != 10):
        raise ValueError('Recovery requires the supported tiny actor architecture')
    source_config = deepcopy(state.get('config', {}))
    if source_config.get('calibration_version') not in (2, 3):
        raise ValueError('Recovery source must have an explicit v2/v3 difficulty contract')
    source_calibration_version = source_config['calibration_version']
    # A new optimization run may strengthen v2 quality checks to v3. Preserve
    # every existing band/threshold, and require fresh loop telemetry afterward.
    source_config['calibration_version'] = 3
    detector = progress_config(source_config.get('progress_recovery'))
    source_config['progress_recovery'] = detector
    device = torch.device(device)
    teacher = actor_from_checkpoint(state, 'enemy', device).eval().requires_grad_(False)
    if any(not torch.isfinite(value).all() for value in teacher.parameters()):
        raise ValueError('Recovery source contains nonfinite weights')
    student = deepcopy(teacher).requires_grad_(True)
    output = prepare_output(output_dir, None)
    frozen_source = freeze_checkpoint(source, output)
    save_json(output / 'recovery-config.json', settings)
    rng = np.random.default_rng(settings['seed'])
    torch.manual_seed(settings['seed'])
    anchors = ReplayReservoir(settings['max_records'] - settings['max_recovery_records'], rng)
    recovery = ReplayReservoir(settings['max_recovery_records'], rng)
    optimizer = torch.optim.Adam(student.parameters(), lr=settings['learning_rate'])
    source_updates = int(state.get('enemy_optimizer_updates', 0))
    updates = episode_count = 0
    started = time.monotonic()
    candidates = []

    def next_episode():
        nonlocal episode_count
        index = episode_count
        episode_count += 1
        return {'level_id': index // 2 % 10 + 1, 'seed': settings['seed'] + index,
                'max_steps': settings['max_steps'], 'defender_bot': 'guard' if index % 2 == 0 else 'hunter'}

    for iteration in range(1, settings['iterations'] + 1):
        counts = {'temporal': 0, 'stationary': 0, 'blocked': 0, 'active': 0, 'recovery': 0}
        episodes_before = episode_count
        with env_factory([next_episode() for _ in range(settings['num_envs'])]) as env:
            validate_checkpoint_environment(state, env.metadata)
            progress = ProgressTracker(env.frame.enemy_active.shape, detector)
            stationary = StationaryTracker(env.frame.enemy_active.shape)
            for step in range(settings['collection_steps']):
                frame = env.frame
                flags, targets, labels = recovery_flags(frame, progress, stationary)
                active = frame.enemy_active & (frame.enemy_ids != 0)
                if np.any(active & ~frame.enemy_mask.any(-1)):
                    raise ValueError('Active recovery collector slot has no legal action')
                for key, value in labels.items():
                    counts[key] += value
                counts['active'] += int(active.sum())
                counts['recovery'] += int(flags.sum())
                ordinary = active & ~flags if step % settings['anchor_every'] == 0 else np.zeros_like(active)
                with torch.no_grad():
                    obs = torch.as_tensor(frame.enemy_obs, device=device)
                    logits = student(obs)
                    mask = torch.as_tensor(frame.enemy_mask, device=device)
                    actions = logits.masked_fill(~mask, -1e9).argmax(-1).cpu().numpy()
                    actions[~active] = -1
                    for selected, reservoir in ((flags, recovery), (ordinary, anchors)):
                        if selected.any():
                            observations = frame.enemy_obs[selected]
                            original_logits = teacher(torch.as_tensor(observations, device=device)).cpu().numpy()
                            reservoir.add(observations, frame.enemy_mask[selected], original_logits, targets[selected])
                frame = env.step(actions, np.full(settings['num_envs'], -1, dtype=np.int64))
                done = np.flatnonzero(frame.terminated | frame.truncated)
                if len(done):
                    env.reset_indices(done.tolist(), [next_episode() for _ in done])
                    progress.reset_indices(done)
                    stationary.ids[done] = 0
                    stationary.stationary[done] = 0
                    stationary.fire_age[done] = stationary.recent_fire_window
        losses = []
        # No synthetic labels: a round with no measured recovery examples
        # makes no optimizer update, preserving the source actor exactly.
        if recovery.size:
            for _ in range(settings['updates_per_iteration']):
                repair_batch = recovery.sample(settings['batch_size'], device)
                anchor_batch = anchors.sample(settings['batch_size'], device) if anchors.size else None
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = recovery_loss(student, anchor_batch, repair_batch,
                                               kl_coef=settings['kl_coef'], recovery_coef=settings['recovery_coef'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite recovery loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), settings['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                student.enforce_input_gate()
                updates += 1
                losses.append({key: float(value.cpu()) for key, value in metrics.items()})
        metrics = {'iteration': iteration, 'elapsed_seconds': round(time.monotonic() - started, 3),
                   'optimizer_updates': updates, 'episodes_started': episode_count - episodes_before,
                   'counts': counts, 'anchor_records': anchors.size, 'recovery_records': recovery.size,
                   'anchor_records_seen': anchors.seen, 'recovery_records_seen': recovery.seen,
                   **{key: float(np.mean([row[key] for row in losses])) if losses else 0.
                      for key in ('recovery_ce', 'anchor_kl')}}
        log_metrics(output / 'metrics.jsonl', metrics)
        if iteration % settings['checkpoint_every'] == 0 or iteration == settings['iterations']:
            smoke = str(state.get('kind', '')).startswith('smoke') or bool(source_config.get('smoke'))
            snapshot = {key: deepcopy(value) for key, value in state.items()
                        if key not in {'optimizers', 'league', 'rng', 'dataset', 'distillation_optimizer',
                                       'teacher_checkpoint', 'qualification'}}
            snapshot.update(kind='smoke_recovery' if smoke else 'recovery',
                            config={**deepcopy(source_config), 'seed': settings['seed']},
                            enemy_actor=cpu_state(student), iteration=iteration,
                            trained=True, calibration_version=3, enemy_optimizer_updates=source_updates + updates,
                            optimizer_updates=int(state.get('optimizer_updates', source_updates)) + updates,
                            recovery_optimizer=deepcopy(optimizer.state_dict()),
                            calibration={'eligible': False, 'reason': 'Recovery actor requires its own evaluation'},
                            selection={'calibrated': False, 'reason': 'Recovery actor requires its own evaluation'},
                            calibration_ancestry=calibration_campaigns(state),
                            recovery_source={'sha256': source.sha256, 'filename': source.path.name,
                                             'frozen_checkpoint': frozen_source.name,
                                             'kind': state.get('kind'), 'iteration': state.get('iteration'),
                                             'source_calibration_version': source_calibration_version,
                                             'source_enemy_optimizer_updates': source_updates},
                            recovery_training={**settings, 'optimizer_updates': updates,
                                               'positive_seed_interval': [settings['seed'], settings['seed'] + episode_count - 1],
                                               'collector_policy': 'student_argmax', 'runtime_overrides': False,
                                               'teacher': 'immutable_source_actor', 'last_metrics': metrics},
                            resume_semantics='Frozen recovery actor; initialize another repair or PPO fine-tuning run.')
            if smoke:
                snapshot['config']['smoke'] = True
            candidate = save_checkpoint(output / f'checkpoint-{iteration:06d}.pt', snapshot)
            save_checkpoint(output / 'latest.pt', snapshot)
            candidates.append(str(candidate))
        print(f"recovery [{iteration}/{settings['iterations']}] examples={recovery.size} anchors={anchors.size} "
              f"CE={metrics['recovery_ce']:.4f} KL={metrics['anchor_kl']:.6f} updates={updates}", flush=True)
    return {'checkpoint': str(output / 'latest.pt'), 'candidates': candidates, 'optimizer_updates': updates,
            'source_sha256': source.sha256, 'selection': {'calibrated': False}, 'metrics': metrics}
