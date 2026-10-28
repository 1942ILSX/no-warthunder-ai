"""Bounded held-out-seed evaluation against scripted and frozen opponents."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_actor_checkpoint, validate_checkpoint_environment
from .models import PolicyNetwork, act
from .sampling import deployment_sampler


PRIMARY_OPPONENTS = ("guard", "hunter")


def evaluate_current_actor_native(enemy_actor: PolicyNetwork, environment, device,
                                  config: Mapping, *, optimizer_updates: int) -> dict:
    """Native calibration during tiny-actor training, never release certification.

    Current weights are serialized while the learner is paused at evaluation.
    Independent release evaluation still needs a saved checkpoint SHA256.
    """
    import tempfile
    from .env import PROJECT_ROOT
    from .export import write_artifact
    factory_builder = getattr(environment, 'factory_with_enemy_model', None)
    if not callable(factory_builder):
        raise ValueError('Categorical training calibration requires a BatchedEnv with factory_with_enemy_model')
    if tuple(enemy_actor.hidden_sizes) not in {(64, 32), (32, 16)}:
        raise ValueError('Native categorical calibration supports only tiny 64,32 or32,16 actors; use argmax for larger teachers')
    if type(optimizer_updates) is not int or optimizer_updates < 1:
        raise ValueError('Native current-actor calibration requires recorded enemy optimizer updates')
    sampler = deployment_sampler(config.get('deployment_sampler'))
    temporary_root = PROJECT_ROOT / 'work' / 'native-evaluation'
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='training-', dir=temporary_root) as directory:
        artifact = Path(directory) / 'artifact'
        metadata = write_artifact(enemy_actor, artifact, policy_version='training-calibration',
                                  training_seed=int(config.get('seed', 0)), trained=True,
                                  difficulty=config['difficulty'], sampler=sampler['mode'], temperature=sampler['temperature'])
        identity = {'implementation': 'go-policymlp', 'metadata': metadata}
        result = evaluate(None, factory_builder(artifact), device, config, native_policy=identity)
    result['release_certified'] = False
    result['evaluation_purpose'] = 'current_actor_training_calibration'
    return result


class StationaryTracker:
    """Count prolonged, unproductive stationary decisions on actor observations.

    Positions are the same delayed observations supplied to the policy, not
    privileged instantaneous physics state. One count means one active enemy
    decision after 20 stationary intervals (about two seconds at 10 Hz).
    A current-facing target/brick line permits continued stationary engagement
    only with recent fire-action evidence from delayed observation feature 9.
    The initial stationary threshold is an aiming grace; the recent-fire window
    permits ordinary cooldown waits. This telemetry does not detect every loop
    or establish that every attempted shot was useful or physically fired.
    """
    def __init__(self, shape: tuple[int, ...], intervals: int = 20, epsilon_tiles: float = .02,
                 recent_fire_window: int = 20):
        if intervals < 1 or recent_fire_window < 1:
            raise ValueError("Stationary and fire windows must be positive")
        self.ids = np.zeros(shape, dtype=np.int64)
        self.positions = np.zeros((*shape, 2), dtype=np.float32)
        self.stationary = np.zeros(shape, dtype=np.int64)
        self.intervals = intervals
        self.epsilon = epsilon_tiles / 24
        self.recent_fire_window = recent_fire_window
        self.fire_age = np.full(shape, recent_fire_window, dtype=np.int64)

    def update(self, frame, complete: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        active = frame.enemy_active & ~complete[:, None] & (frame.enemy_ids != 0)
        positions = frame.enemy_obs[..., :2]
        same_id = active & (frame.enemy_ids == self.ids)
        moved = np.max(np.abs(positions - self.positions), axis=-1) > self.epsilon
        stationary = same_id & ~moved
        self.stationary = np.where(stationary, self.stationary + 1, 0)
        fire_evidence = active & (frame.enemy_obs[..., 9] > .5)
        self.fire_age = np.where(fire_evidence, 0, np.where(
            same_id, np.minimum(self.fire_age + 1, self.recent_fire_window), self.recent_fire_window))
        # Keep an anchor until the tank moves sufficiently; tiny repeated motion
        # cannot evade the threshold by moving less than epsilon each sample.
        reset = ~stationary
        self.positions[reset] = positions[reset]
        self.ids = np.where(active, frame.enemy_ids, 0)
        facing = frame.enemy_obs[..., 2:6].argmax(axis=-1)
        aim = np.take_along_axis(frame.enemy_obs[..., 313:317], facing[..., None], axis=-1)[..., 0]
        meaningful_line = aim >= .1  # brick, defender or base in the facing ray
        engaged = meaningful_line & (self.fire_age < self.recent_fire_window)
        unproductive = active & (self.stationary >= self.intervals) & ~engaged
        return unproductive.sum(axis=1), active.sum(axis=1)


def aggregate_records(items: list[dict], *, telemetry: bool = False, loop_telemetry: bool = False) -> dict:
    n = len(items)
    alive = sum(r.get("enemy_alive_steps", 0) for r in items)
    result = {"episodes": n,
              "enemy_win_rate": sum(str(r.get("outcome", "")).startswith("enemy_win_") for r in items) / max(n, 1),
              "defender_win_rate": sum(r.get("outcome") == "defender_win" for r in items) / max(n, 1),
              "truncation_rate": sum(bool(r.get("truncated")) for r in items) / max(n, 1),
              "timeout_rate": sum(bool(r.get("truncated")) or r.get("outcome") in {"timeout", "simulation_timeout"}
                                  for r in items) / max(n, 1),
              "stagnation_rate": sum(r.get("enemy_stuck_steps", 0) for r in items) / max(alive, 1),
              "enemy_alive_steps": alive,
              "mean_enemy_return": float(np.mean([r.get("enemy_return", 0) for r in items])) if items else 0.,
              **{f"mean_{key}": float(np.mean([r.get(key, 0) for r in items])) if items else 0.
                 for key in ("decision_step", "kills", "base_hp", "lives", "score", "enemy_stuck")}}
    if telemetry:
        present = bool(items) and all("enemy_unproductive_steps" in r and "enemy_observed_steps" in r for r in items)
        observed = sum(r["enemy_observed_steps"] for r in items) if present else None
        unproductive = sum(r["enemy_unproductive_steps"] for r in items) if present else None
        result.update(unproductive_telemetry_complete=present,
                      enemy_observed_steps=observed, enemy_unproductive_steps=unproductive,
                      unproductive_rate=unproductive / max(observed, 1) if present else None)
    if loop_telemetry:
        def valid(row):
            observed, loops = row.get("enemy_loop_observed_steps"), row.get("enemy_loop_steps")
            return (isinstance(observed, (int, float)) and isinstance(loops, (int, float))
                    and np.isfinite(observed) and np.isfinite(loops) and 0 <= loops <= observed)
        present = bool(items) and all(valid(row) for row in items)
        observed = sum(row["enemy_loop_observed_steps"] for row in items) if present else None
        loops = sum(row["enemy_loop_steps"] for row in items) if present else None
        result.update(moving_no_progress_telemetry_complete=present,
                      enemy_loop_observed_steps=observed, enemy_loop_steps=loops,
                      moving_no_progress_rate=loops / max(observed, 1) if present else None)
    return result


def summarize_records(records: list[dict], *, version: int = 1) -> dict:
    if version not in (1, 2, 3):
        raise ValueError("Unknown evaluation metrics version")
    telemetry = version >= 2
    def aggregate(rows):
        return aggregate_records(rows, telemetry=telemetry, loop_telemetry=version >= 3)
    def groups(rows: list[dict]) -> dict:
        opponents = sorted({r["opponent"] for r in rows})
        levels = sorted({int(r["level_id"]) for r in rows})
        return {"by_opponent": {name: aggregate([r for r in rows if r["opponent"] == name])
                                for name in opponents},
                "by_level": {str(level): aggregate([r for r in rows if int(r["level_id"]) == level])
                             for level in levels},
                "by_level_opponent": {str(level): {
                    name: aggregate([r for r in rows if int(r["level_id"]) == level and r["opponent"] == name])
                    for name in opponents} for level in levels}}
    all_groups = groups(records)
    result = {**aggregate(records),
              "by_opponent": all_groups["by_opponent"],
              "level_ids": sorted({int(r["level_id"]) for r in records})}
    if version >= 2:
        primary = [r for r in records if r["opponent"] in PRIMARY_OPPONENTS]
        result.update(metrics_version=version, calibration_version=version,
                      by_level=all_groups["by_level"], by_level_opponent=all_groups["by_level_opponent"],
                      primary={**aggregate(primary), **groups(primary)},
                      primary_opponents=list(PRIMARY_OPPONENTS),
                      diagnostic_opponents=sorted({r["opponent"] for r in records} - set(PRIMARY_OPPONENTS)))
    return result


def assess_report(report: Mapping, config: Mapping) -> dict:
    """Reassess saved records under v2/v3 without inventing absent telemetry.

    Returns a new mapping. Original reports/checkpoints remain unchanged. A v1
    report has no stationary telemetry and cannot receive v2 eligibility.
    """
    version = config.get("calibration_version")
    if version not in (2, 3):
        raise ValueError("Retrospective assessment requires calibration_version=2 or 3")
    if not isinstance(report.get("records"), list):
        raise ValueError("Retrospective assessment requires episode records")
    result = {**report, **summarize_records(report["records"], version=version),
              "retrospective_assessment": True, "source_metrics_version": report.get("metrics_version", 1)}
    if "calibration" in report:
        result["previous_calibration"] = report["calibration"]
    result["calibration"] = calibration_metrics(result, config)
    return result


def evaluate(enemy_actor: PolicyNetwork | str | Path | None, env_factory: Callable,
             device: str | torch.device, config: Mapping | None = None, *,
             defender_actor: PolicyNetwork | None = None,
             historical_defenders: list[PolicyNetwork] | None = None,
             native_policy: Mapping | None = None) -> dict:
    """Every listed opponent gets `episodes` held-out matches, bounded by max_steps.

    Enemy/defender wins and truncations are separate outcomes. A timeout is
    never silently counted as an attacker win. Learned opponents are frozen
    for the whole evaluation and receive their own defender observation schema.
    """
    config = dict(config or {})
    sampler = deployment_sampler(config.get('deployment_sampler'))
    if native_policy is not None:
        if enemy_actor is not None or native_policy.get('implementation') != 'go-policymlp':
            raise ValueError('Native evaluation requires the Go policy and no Python enemy actor')
        if native_policy.get('metadata', {}).get('sampler') != sampler:
            raise ValueError('Native policy sampler differs from checkpoint deployment_sampler')
    elif enemy_actor is None or sampler['mode'] != 'argmax':
        raise ValueError('Categorical evaluation requires the actual native Go policy')
    version = config.get("calibration_version", 1)
    if version not in (1, 2, 3):
        raise ValueError("Unknown calibration_version")
    progress_settings = None
    if version >= 3:
        from .progress import ProgressTracker, progress_config
        progress_settings = progress_config(config.get("progress_recovery"))
    episodes = int(config.get("episodes", 10))
    batch_size = int(config.get("num_envs", min(episodes, 16)))
    max_steps = int(config.get("max_steps", 9000))
    levels = config.get("level_ids", list(range(1, 11)))
    if episodes < 1 or batch_size < 1 or not 1 <= max_steps <= 9000 or not levels:
        raise ValueError("Evaluation needs positive episode/batch counts, levels and max_steps <= 9000")
    device = torch.device(device)
    if isinstance(enemy_actor, (str, Path)):
        enemy_actor = load_actor_checkpoint(enemy_actor, "enemy", device)
    opponents: list[tuple[str, PolicyNetwork | None]] = [
        (name, None) for name in config.get("opponents", ["guard", "hunter", "wander"])]
    if any(name not in {"guard", "hunter", "wander", "mixed"} for name, _ in opponents):
        raise ValueError("Evaluation scripted opponents must be guard, hunter, wander, or mixed")
    if defender_actor is not None:
        opponents.append(("learned", defender_actor))
    opponents.extend((f"historical:{i}", actor) for i, actor in enumerate(historical_defenders or []))
    if not opponents:
        raise ValueError("At least one evaluation opponent is required")
    records: list[dict] = []
    for label, defender in opponents:
        for start in range(0, episodes, batch_size):
            count = min(batch_size, episodes - start)
            configs = [{"level_id": int(levels[(start + i) % len(levels)]),
                        "seed": -(abs(int(config.get("eval_seed", config.get("seed", 1_000_000)))) + start + i + 1),
                        "max_steps": max_steps,
                        "defender_bot": label if defender is None else "mixed"}
                       for i in range(count)]
            env = env_factory(configs)
            try:
                if env.metadata.get('enemy_policy') != native_policy:
                    raise ValueError('Evaluation environment does not match the expected native policy identity')
                for actor, is_defender in ((enemy_actor, False), (defender, True)):
                    provenance = getattr(actor, "checkpoint_provenance", None)
                    if provenance is not None:
                        validate_checkpoint_environment(provenance, env.metadata, defender=is_defender)
                frame = getattr(env, "frame", None)
                if frame is None:
                    frame = env.reset()
                complete = np.zeros(count, dtype=bool)
                returns = np.zeros(count, dtype=np.float64)
                stuck_steps = np.zeros(count, dtype=np.float64)
                alive_steps = np.zeros(count, dtype=np.float64)
                unproductive_steps = np.zeros(count, dtype=np.float64)
                observed_steps = np.zeros(count, dtype=np.float64)
                stationary = StationaryTracker(frame.enemy_active.shape) if version >= 2 else None
                progress = ProgressTracker(frame.enemy_active.shape, progress_settings) if version >= 3 else None
                loop_steps = np.zeros(count, dtype=np.int64)
                loop_observed = np.zeros(count, dtype=np.int64)
                enemy_progress = [{} for _ in range(count)]
                for _ in range(max_steps):
                    if stationary is not None:
                        unproductive, observed = stationary.update(frame, complete)
                        unproductive_steps += unproductive
                        observed_steps += observed
                    if progress is not None:
                        looping = progress.update(frame, complete)
                        active = frame.enemy_active & (frame.enemy_ids != 0) & ~complete[:, None]
                        looping = looping & active
                        loop_steps += looping.sum(axis=1)
                        loop_observed += active.sum(axis=1)
                        for room, slot in np.argwhere(active):
                            identifier = int(frame.enemy_ids[room, slot])
                            item = enemy_progress[room].setdefault(identifier, {
                                "enemy_id": identifier, "observed_steps": 0, "moving_no_progress_steps": 0})
                            item["observed_steps"] += 1
                            item["moving_no_progress_steps"] += int(looping[room, slot])
                    if native_policy is not None:
                        enemy_actions = np.full(frame.enemy_active.shape, -1, dtype=np.int64)
                    else:
                        enemy_actions, _, _ = act(enemy_actor, frame.enemy_obs, frame.enemy_mask,
                                                  frame.enemy_active, device, deterministic=True)
                    defender_actions, _, _ = act(defender, frame.defender_obs, frame.defender_mask,
                                                 frame.defender_active, device, deterministic=True)
                    frame = env.step(enemy_actions, defender_actions)
                    returns += frame.enemy_reward * ~complete
                    stuck_steps += np.array([info.get("enemy_stuck", 0) for info in frame.infos]) * ~complete
                    alive_steps += np.array([info.get("enemy_alive", 0) for info in frame.infos]) * ~complete
                    newly_done = (frame.terminated | frame.truncated) & ~complete
                    for index in np.flatnonzero(newly_done):
                        record = {**frame.infos[int(index)], "opponent": label,
                                        "enemy_return": float(returns[index]),
                                        "enemy_stuck_steps": float(stuck_steps[index]),
                                        "enemy_alive_steps": float(alive_steps[index]),
                                  "truncated": bool(frame.truncated[index])}
                        if version >= 2:
                            record.update(enemy_unproductive_steps=float(unproductive_steps[index]),
                                          enemy_observed_steps=float(observed_steps[index]))
                        if version >= 3:
                            per_enemy = [dict(item, moving_no_progress_rate=item["moving_no_progress_steps"] / item["observed_steps"])
                                         for _, item in sorted(enemy_progress[index].items())]
                            record.update(enemy_loop_steps=int(loop_steps[index]),
                                          enemy_loop_observed_steps=int(loop_observed[index]),
                                          enemy_progress=per_enemy)
                        records.append(record)
                    complete |= newly_done
                    if complete.all():
                        break
                if not complete.all():
                    raise RuntimeError("Environment did not honor evaluation max_steps")
            finally:
                env.close()
    result = {**summarize_records(records, version=version),
            "records": records, "seed_domain": "negative (disjoint from training)",
            "deterministic_policy": sampler['mode'] == 'argmax',
            "policy_sampler": sampler if native_policy is not None else {key: sampler[key] for key in ('mode', 'temperature')}}
    if native_policy is not None:
        result.update(native_policy=dict(native_policy), reproducible_seeded_policy=True)
    if version >= 3:
        result["progress_recovery"] = progress_settings
    if "target_defender_win_rate" in config:
        result["calibration"] = calibration_metrics(result, config)
    return result


def calibration_metrics(evaluation: Mapping, config: Mapping) -> dict:
    """Rank candidates by safety of the proxy measurement, then target distance.

    These are fixed-bot calibration bands, not a claim about human win rates.
    An out-of-band or low-quality best candidate remains explicitly ineligible.
    """
    if config.get("calibration_version") == 3:
        return calibration_metrics_v3(evaluation, config)
    if config.get("calibration_version") == 2:
        return calibration_metrics_v2(evaluation, config)
    low, high = (float(n) for n in config["target_defender_win_rate"])
    if not 0 <= low <= high <= 1:
        raise ValueError("target_defender_win_rate must be an ordered band in [0,1]")
    rate = float(evaluation["defender_win_rate"])
    timeout = float(evaluation["timeout_rate"])
    stagnant = float(evaluation["stagnation_rate"])
    min_episodes = int(config.get("min_evaluation_episodes", 30))
    timeout_limit = float(config.get("max_truncation_rate", .1))
    stuck_limit = float(config.get("max_stagnation_rate", .25))
    coverage = (evaluation.get("level_ids", list(range(1, 11))) == list(range(1, 11))
                and set(evaluation.get("by_opponent", {"guard": 0, "hunter": 0, "wander": 0})) == {"guard", "hunter", "wander"})
    violations = (max(0.0, timeout - timeout_limit) + max(0.0, stagnant - stuck_limit)
                  + max(0, min_episodes - int(evaluation["episodes"])) / max(min_episodes, 1)
                  + (1.0 if evaluation.get("enemy_alive_steps", 0) <= 0 else 0.0)
                  + (0.0 if coverage else 1.0))
    distance = max(low - rate, rate - high, 0.0)
    quality = violations == 0
    return {"difficulty": config.get("difficulty", "unrated"),
            "reference": "fixed scripted bots; proxy only, not human win rate",
            "target_defender_win_rate": [low, high], "defender_win_rate": rate,
            "distance_to_band": distance, "quality_passed": quality,
            "coverage_passed": coverage,
            "eligible": quality and distance == 0, "timeout_rate": timeout,
            "stagnation_rate": stagnant, "episodes": int(evaluation["episodes"]),
            "rank": [violations, distance, abs(rate - (low + high) / 2)]}


def calibration_metrics_v2(evaluation: Mapping, config: Mapping) -> dict:
    """Apply primary-opponent quality gates before considering difficulty.

    Outcome bands are pooled across guard/hunter; quality gates additionally
    inspect each opponent and each level. Wander remains a diagnostic rather
    than lowering the reference defender win rate by one third.
    """
    low, high = (float(n) for n in config["target_defender_win_rate"])
    if not 0 <= low <= high <= 1:
        raise ValueError("target_defender_win_rate must be an ordered band in [0,1]")
    summary = evaluation if evaluation.get("metrics_version") == 2 and "primary" in evaluation else summarize_records(list(evaluation.get("records", [])), version=2)
    primary = summary["primary"]
    empty = aggregate_records([], telemetry=True)
    limits = {
        "primary": {"timeout_rate": .05, "stagnation_rate": .10, "unproductive_rate": .10},
        "opponent": {"timeout_rate": .10, "stagnation_rate": .15, "unproductive_rate": .15},
        "level": {"timeout_rate": .10, "stagnation_rate": .20, "unproductive_rate": .20},
    }
    for scope, metrics in limits.items():
        for metric in metrics:
            value = float(config.get(f"v2_max_{scope}_{metric}", metrics[metric]))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"v2_max_{scope}_{metric} must be in [0,1]")
            metrics[metric] = value
    minimums = {"primary": 100, "opponent": 50, "level_opponent": 5}
    for scope in minimums:
        value = config.get(f"v2_min_{scope}_episodes", minimums[scope])
        if type(value) is not int or value < 1:
            raise ValueError(f"v2_min_{scope}_episodes must be a positive integer")
        minimums[scope] = value
    failures: list[dict] = []
    coverage = True
    def fail(scope: str, metric: str, value, limit, reason: str) -> None:
        failures.append({"scope": scope, "metric": metric, "value": value,
                         "limit": limit, "reason": reason})
    def require_samples(scope: str, metrics: Mapping, required: int) -> None:
        nonlocal coverage
        if int(metrics["episodes"]) < required:
            coverage = False
            fail(scope, "episodes", int(metrics["episodes"]), required, "insufficient_coverage")
    def require_quality(scope: str, metrics: Mapping, kind: str) -> None:
        if metrics.get("enemy_alive_steps", 0) <= 0:
            fail(scope, "enemy_alive_steps", metrics.get("enemy_alive_steps", 0), 0, "no_live_enemy_observations")
        for metric, limit in limits[kind].items():
            value = metrics.get(metric)
            if metric == "unproductive_rate" and (
                    not metrics.get("unproductive_telemetry_complete") or metrics.get("enemy_observed_steps", 0) <= 0):
                fail(scope, metric, None, limit, "missing_unproductive_telemetry")
            elif value is None or not np.isfinite(value):
                fail(scope, metric, None, limit, "missing_or_nonfinite_metric")
            elif value > limit:
                fail(scope, metric, float(value), limit, "quality_limit_exceeded")
    require_samples("primary", primary, minimums["primary"])
    require_quality("primary", primary, "primary")
    for name in PRIMARY_OPPONENTS:
        metrics = primary["by_opponent"].get(name, empty)
        require_samples(f"opponent:{name}", metrics, minimums["opponent"])
        require_quality(f"opponent:{name}", metrics, "opponent")
    for level in range(1, 11):
        key = str(level)
        require_quality(f"level:{level}", primary["by_level"].get(key, empty), "level")
        for name in PRIMARY_OPPONENTS:
            metrics = primary["by_level_opponent"].get(key, {}).get(name, empty)
            require_samples(f"level:{level}/opponent:{name}", metrics, minimums["level_opponent"])
    def severity(metrics: Mapping, kind: str) -> float:
        values = [float(metrics.get(key) or 0) / max(limit, 1e-12) for key, limit in limits[kind].items()]
        return max(values)
    worst_level = max(primary["by_level"], key=lambda level: severity(primary["by_level"][level], "level"), default=None)
    worst_opponent = max(primary["by_opponent"], key=lambda name: severity(primary["by_opponent"][name], "opponent"), default=None)
    rate = float(primary["defender_win_rate"])
    distance = max(low - rate, rate - high, 0.0)
    quality = not failures
    reasons = sorted({f"{failure['scope']}.{failure['metric']}:{failure['reason']}" for failure in failures})
    if distance:
        reasons.append("primary.defender_win_rate:below_target_band" if rate < low else "primary.defender_win_rate:above_target_band")
    # Failure count is intentionally first: a broken actor must not outrank a
    # healthy actor merely by being closer to a target band.
    violations = float(len(failures))
    return {"calibration_version": 2, "difficulty": config.get("difficulty", "unrated"),
            "reference": "guard/hunter fixed bots; wander diagnostic; proxy only, not human win rate",
            "primary_opponents": list(PRIMARY_OPPONENTS),
            "target_defender_win_rate": [low, high], "defender_win_rate": rate,
            "distance_to_band": distance, "quality_passed": quality, "coverage_passed": coverage,
            "eligible": quality and distance == 0,
            "timeout_rate": float(primary["timeout_rate"]), "stagnation_rate": float(primary["stagnation_rate"]),
            "unproductive_rate": primary["unproductive_rate"],
            "unproductive_telemetry_complete": primary["unproductive_telemetry_complete"],
            "episodes": int(primary["episodes"]), "total_episodes": int(summary["episodes"]),
            "quality_failures": failures, "reasons": reasons, "limits": limits, "minimums": minimums,
            "worst_level": worst_level, "worst_opponent": worst_opponent,
            "rank": [violations, distance, abs(rate - (low + high) / 2)]}


def calibration_metrics_v3(evaluation: Mapping, config: Mapping) -> dict:
    """Keep every v2 gate and require independently measured moving-loop quality.

    This detector is a bounded behavioral proxy. It cannot prove that a policy
    makes progress in every possible state, and missing old telemetry is not a
    zero. Per-enemy records remain available for diagnosing the failing rooms.
    """
    from .progress import progress_config
    expected = progress_config(config.get("progress_recovery"))
    summary = evaluation if evaluation.get("metrics_version") == 3 and "primary" in evaluation else summarize_records(list(evaluation.get("records", [])), version=3)
    base = calibration_metrics_v2(evaluation, config)
    failures = list(base["quality_failures"])
    if evaluation.get("progress_recovery") != expected:
        failures.append({"scope": "primary", "metric": "progress_recovery", "value": evaluation.get("progress_recovery"),
                         "limit": expected, "reason": "missing_or_mismatched_detector_config"})
    limits = {"primary": .05, "opponent": .10, "level": .15}
    for kind, default in limits.items():
        key = f"v3_max_{kind}_moving_no_progress_rate"
        value = float(config.get(key, default))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0,1]")
        limits[kind] = value
    primary = summary["primary"]
    empty = aggregate_records([], telemetry=True, loop_telemetry=True)
    scopes = [("primary", primary, "primary")]
    scopes.extend((f"opponent:{name}", primary["by_opponent"].get(name, empty), "opponent") for name in PRIMARY_OPPONENTS)
    scopes.extend((f"level:{level}", primary["by_level"].get(str(level), empty), "level") for level in range(1, 11))
    for scope, metrics, kind in scopes:
        value = metrics.get("moving_no_progress_rate")
        reason = None
        if not metrics.get("moving_no_progress_telemetry_complete") or metrics.get("enemy_loop_observed_steps", 0) <= 0:
            reason = "missing_moving_no_progress_telemetry"
        elif value is None or not np.isfinite(value):
            reason = "missing_or_nonfinite_metric"
        elif value > limits[kind]:
            reason = "quality_limit_exceeded"
        if reason:
            failures.append({"scope": scope, "metric": "moving_no_progress_rate", "value": value,
                             "limit": limits[kind], "reason": reason})
    reasons = sorted({f"{item['scope']}.{item['metric']}:{item['reason']}" for item in failures})
    reasons.extend(reason for reason in base["reasons"] if reason.startswith("primary.defender_win_rate:"))
    return {**base, "calibration_version": 3, "quality_passed": not failures,
            "eligible": not failures and base["distance_to_band"] == 0,
            "quality_failures": failures, "reasons": reasons,
            "limits": {kind: {**base["limits"][kind], "moving_no_progress_rate": limit} for kind, limit in limits.items()},
            "moving_no_progress_rate": primary["moving_no_progress_rate"],
            "moving_no_progress_telemetry_complete": primary["moving_no_progress_telemetry_complete"],
            "progress_recovery": expected,
            "worst_loop_level": max(primary["by_level"], key=lambda level: primary["by_level"][level].get("moving_no_progress_rate") or 0, default=None),
            "worst_loop_opponent": max(primary["by_opponent"], key=lambda name: primary["by_opponent"][name].get("moving_no_progress_rate") or 0, default=None),
            "rank": [float(len(failures)), *base["rank"][1:]]}
