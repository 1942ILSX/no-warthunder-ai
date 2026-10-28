"""Offline alternating self-play; reusable entry point without CLI side effects."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import random
import time
import math

import numpy as np
import torch

from .checkpoint import (actor_from_checkpoint, capture_rng, cpu_state, load_checkpoint,
                         log_metrics, prepare_output, restore_rng, save_checkpoint, save_json,
                         validate_checkpoint_environment, validate_resume_config)
from .league import OpponentPool
from .learning import PPOLearner, Rollout
from .models import CentralCritic, PolicyNetwork, act
from .difficulty import DifficultyController, control_config
from .provenance import calibration_campaigns, capture_checkpoint
from .progress import ProgressTracker, progress_config


DEFAULTS = {
    "seed": 42, "num_envs": 32, "horizon": 128, "iterations": 2000,
    "actor_hidden": [256, 128], "critic_hidden": [256, 256],
    "learning_rate": 3e-4, "gamma": .999, "gae_lambda": .95,
    "clip_epsilon": .2, "entropy_coef": .01, "value_coef": .5,
    "ppo_epochs": 4, "minibatch_size": 1024, "max_grad_norm": .5,
    "training_role": "alternating", "enemy_updates": 4, "defender_updates": 1,
    "checkpoint_every": 25, "league_every": 25, "league_size": 8,
    "scripted_probability": .4, "latest_probability": .4,
    "eval_every": 0, "eval_episodes": 10, "eval_seed": 1_000_000,
    "log_every": 1,
    "curriculum": [
        {"until_iteration": 100, "level_ids": [1, 2], "max_steps": 600},
        {"until_iteration": 400, "level_ids": list(range(1, 7)), "max_steps": 1800},
        {"until_iteration": 2000, "level_ids": list(range(1, 11)), "max_steps": 9000},
    ],
}


def resolved_config(config: Mapping) -> dict:
    result = {**DEFAULTS, **dict(config)}
    if type(result.get('initialize_critics_from_source', False)) is not bool:
        raise ValueError('initialize_critics_from_source must be a boolean')
    if 'deployment_sampler' in result:
        from .sampling import deployment_sampler
        result['deployment_sampler'] = deployment_sampler(result['deployment_sampler'])
        if result['deployment_sampler']['mode'] == 'categorical' and result.get('calibration_version') not in (2, 3):
            raise ValueError('Categorical deployment requires a v2/v3 calibration contract')
    if result.get("difficulty_control") is not None:
        result["difficulty_control"] = control_config(result["difficulty_control"])
        if int(result.get("calibration_version", 1)) not in (2, 3):
            raise ValueError("difficulty_control requires calibration_version: 2 or 3")
        if "target_defender_win_rate" not in result:
            raise ValueError("difficulty_control requires target_defender_win_rate")
    if result.get("progress_recovery") is not None:
        result["progress_recovery"] = progress_config(result["progress_recovery"])
    for key in ("num_envs", "horizon", "iterations", "ppo_epochs", "minibatch_size",
                "checkpoint_every", "league_every", "league_size"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if result["num_envs"] > 256:
        raise ValueError("num_envs cannot exceed the Go bridge limit of 256")
    if type(result["log_every"]) is not int or result["log_every"] < 0:
        raise ValueError("log_every must be a nonnegative integer (0 disables progress output)")
    for key in ("learning_rate", "gamma", "gae_lambda", "clip_epsilon", "entropy_coef", "value_coef", "max_grad_norm"):
        if not math.isfinite(float(result[key])):
            raise ValueError(f"{key} must be finite")
    if result["entropy_coef"] < 0 or result["value_coef"] < 0 or result["clip_epsilon"] < 0:
        raise ValueError("Loss coefficients must be nonnegative")
    if result["training_role"] not in ("alternating", "enemy", "defender"):
        raise ValueError("training_role must be alternating, enemy, or defender")
    if "difficulty" in result and result["difficulty"] not in {"trainee", "expert", "ace"}:
        raise ValueError("difficulty must be trainee, expert, or ace")
    if "target_defender_win_rate" in result:
        low, high = result["target_defender_win_rate"]
        if not 0 <= low <= high <= 1:
            raise ValueError("target_defender_win_rate must be an ordered band in [0,1]")
        if result.get("eval_episodes", 10) < 10:
            # An evaluation campaign must include all ten levels at least once
            # per reference bot for an eligibility claim, enforced in metrics.
            result["min_evaluation_episodes"] = max(30, result.get("min_evaluation_episodes", 30))
    if result["enemy_updates"] < 1 or result["defender_updates"] < 1:
        raise ValueError("enemy_updates and defender_updates must be positive")
    scripted, latest = result["scripted_probability"], result["latest_probability"]
    if not (0 <= scripted <= 1 and 0 <= latest <= 1 and scripted + latest <= 1):
        raise ValueError("Opponent probabilities must be nonnegative and sum to at most one")
    if not (0 < result["gamma"] <= 1 and 0 <= result["gae_lambda"] <= 1):
        raise ValueError("Invalid gamma or gae_lambda")
    if result["learning_rate"] <= 0 or result["max_grad_norm"] <= 0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    previous = 0
    if not result["curriculum"]:
        raise ValueError("curriculum cannot be empty")
    for stage in result["curriculum"]:
        if stage["until_iteration"] <= previous:
            raise ValueError("curriculum boundaries must increase")
        previous = stage["until_iteration"]
        if not stage["level_ids"] or any(level not in range(1, 11) for level in stage["level_ids"]):
            raise ValueError("curriculum level_ids must be in 1..10")
        if not 1 <= stage.get("max_steps", 9000) <= 9000:
            raise ValueError("curriculum max_steps must be in 1..9000")
        bots = stage.get("defender_bots", ["mixed"])
        if not bots or any(bot not in {"guard", "hunter", "wander", "mixed"} for bot in bots):
            raise ValueError("Unknown defender bot in curriculum")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def curriculum_stage(config: Mapping, iteration: int) -> dict:
    for stage in config["curriculum"]:
        if iteration < stage["until_iteration"]:
            return stage
    return config["curriculum"][-1]


def episode_configs(config: Mapping, iteration: int, count: int) -> list[dict]:
    stage = curriculum_stage(config, iteration)
    return [{"level_id": int(random.choice(stage["level_ids"])),
             # Evaluation uses negative seeds, disjoint from this domain.
             "seed": int(np.random.randint(1, 1_000_000_000)),
             "max_steps": int(stage.get("max_steps", 9000)),
             "defender_bot": random.choice(stage.get("defender_bots", ["mixed"]))}
            for _ in range(count)]


def training_role(config: Mapping, iteration: int) -> str:
    if config["training_role"] != "alternating":
        return config["training_role"]
    cycle = config["enemy_updates"] + config["defender_updates"]
    return "enemy" if iteration % cycle < config["enemy_updates"] else "defender"


def collect_rollout(env, frame, actor: PolicyNetwork, opponent: PolicyNetwork | None,
                    *, role: str, config: Mapping, iteration: int, device,
                    progress_tracker: ProgressTracker | None = None) -> tuple[Rollout, object, list[dict]]:
    """Collect on-policy decisions; retain terminal critic state before resets."""
    arrays: dict[str, list] = {key: [] for key in Rollout.__dataclass_fields__ if key != "recovery_mask"}
    recovery_labels = []
    if progress_tracker is None and config.get("progress_recovery") is not None:
        progress_tracker = ProgressTracker(frame.enemy_active.shape, config["progress_recovery"])
    episodes: list[dict] = []
    for _ in range(int(config["horizon"])):
        if progress_tracker is not None:
            temporal_recovery = progress_tracker.update(frame)
            if role == "enemy":
                recovery_labels.append(temporal_recovery)
        enemy = actor if role == "enemy" else opponent
        defender = actor if role == "defender" else opponent
        enemy_actions, enemy_tensor, enemy_logp = act(
            enemy, frame.enemy_obs, frame.enemy_mask, frame.enemy_active, device)
        defender_actions, defender_tensor, defender_logp = act(
            defender, frame.defender_obs, frame.defender_mask, frame.defender_active, device)
        prefix = "enemy" if role == "enemy" else "defender"
        arrays["observation"].append(getattr(frame, prefix + "_obs").copy())
        arrays["action_mask"].append(getattr(frame, prefix + "_mask").copy())
        arrays["active"].append(getattr(frame, prefix + "_active").copy())
        arrays["action"].append(enemy_tensor if role == "enemy" else defender_tensor)
        arrays["log_prob"].append(enemy_logp if role == "enemy" else defender_logp)
        arrays["critic_observation"].append(frame.critic_obs.copy())
        transition = env.step(enemy_actions, defender_actions)
        arrays["next_critic_observation"].append(transition.critic_obs.copy())
        arrays["reward"].append(getattr(transition, prefix + "_reward").copy())
        arrays["terminated"].append(transition.terminated.copy())
        arrays["truncated"].append(transition.truncated.copy())
        done = np.flatnonzero(transition.terminated | transition.truncated)
        episodes.extend(dict(transition.infos[int(index)]) for index in done)
        if progress_tracker is not None and len(done):
            progress_tracker.reset_indices(done)
        # reset_indices replaces whole response rewards with zero. All learning
        # signals and final observations above have already been copied.
        frame = (env.reset_indices(done, episode_configs(config, iteration, len(done)))
                 if len(done) else transition)
    tensors = {}
    for key, sequence in arrays.items():
        tensors[key] = (torch.stack(sequence, dim=1) if torch.is_tensor(sequence[0])
                        else torch.as_tensor(np.stack(sequence, axis=1), device=device))
    if recovery_labels:
        tensors["recovery_mask"] = torch.as_tensor(np.stack(recovery_labels, axis=1), device=device)
    return Rollout(**tensors), frame, episodes


def _snapshot(config, iteration, actors, critics, learners, pool, metadata, actor_updates,
              selection=None, controller=None, initial_source=None) -> dict:
    optimizer_updates = sum(actor_updates.values())
    return {
        "format_version": 1, "kind": "smoke" if config.get("smoke") else "training",
        "trained": optimizer_updates > 0, "optimizer_updates": optimizer_updates,
        "enemy_optimizer_updates": actor_updates["enemy"],
        "defender_optimizer_updates": actor_updates["defender"],
        "difficulty": config.get("difficulty", "unrated"),
        "target_defender_win_rate": config.get("target_defender_win_rate"),
        "score_multiplier": config.get("score_multiplier"),
        "selection": selection or {},
        "calibration_version": config.get("calibration_version", 1),
        "difficulty_state": controller.state_dict() if controller is not None else None,
        "initial_source": initial_source,
        "observation_version": int(metadata.get("observation_version", 2)),
        "normalization_version": int(metadata.get("normalization_version", 2)),
        "defender_observation_version": int(metadata.get("defender_observation_version", 1)),
        "iteration": iteration, "config": dict(config), "metadata": dict(metadata),
        "model_spec": {"enemy_hidden": actors["enemy"].hidden_sizes,
                       "defender_hidden": actors["defender"].hidden_sizes,
                       "critic_hidden": critics["enemy"].hidden_sizes,
                       "critic_size": critics["enemy"].input_size,
                       "observation_size": 320, "action_size": 10},
        **{f"{role}_actor": cpu_state(actors[role]) for role in actors},
        **{f"{role}_critic": cpu_state(critics[role]) for role in critics},
        "optimizers": {role: learners[role].optimizer_state() for role in learners},
        "league": pool.state_dict(), "rng": capture_rng(),
        "resume_semantics": "Models, optimizers, league and RNG resume; live rooms restart as new episodes.",
    }


def train(config: Mapping, env_factory: Callable, device: str | torch.device,
          output_dir: str | Path, resume: str | Path | None = None,
          initial_enemy_checkpoint: str | Path | None = None, *,
          allow_profile_transfer: bool = False) -> dict:
    """Run a bounded number of updates; intended to be launched by the user.

    Resume restores optimizer/RNG/pool state, but deliberately starts fresh Go
    episodes because the bridge does not serialize live worlds. Initializing a
    tiny distilled actor is a new fine-tuning run, not optimizer resume.
    """
    config = resolved_config(config)
    if resume and initial_enemy_checkpoint:
        raise ValueError("Choose resume or initial_enemy_checkpoint, not both")
    if allow_profile_transfer and (resume or not initial_enemy_checkpoint):
        raise ValueError("allow_profile_transfer requires an initial checkpoint and no resume")
    if config.get('initialize_critics_from_source') and not (initial_enemy_checkpoint or resume):
        raise ValueError('initialize_critics_from_source requires an initial checkpoint for a new run')
    output = prepare_output(output_dir, resume)
    device = torch.device(device)
    seed_everything(int(config["seed"]))
    state = load_checkpoint(resume) if resume else None
    if state and state.get("kind") not in {"training", "smoke"}:
        raise ValueError("Use initial_enemy_checkpoint to fine-tune a distilled actor")
    if state and state.get("difficulty", "unrated") != config.get("difficulty", "unrated"):
        raise ValueError("Resume cannot change difficulty; initialize a new run instead")
    if state:
        validate_resume_config(config, state["config"])
    if state and state.get("kind") == "smoke":
        config["smoke"] = True
    control = config.get("difficulty_control")
    if control is not None and state and not isinstance(state.get("difficulty_state"), Mapping):
        raise ValueError("Controlled training checkpoint is missing difficulty_state")
    controller = DifficultyController(control, state.get("difficulty_state") if state else None) if control else None
    initial_source = state.get("initial_source") if state else None
    env = env_factory(episode_configs(config, 0, config["num_envs"]))
    started = time.monotonic()
    try:
        frame = getattr(env, "frame", None)
        if frame is None:
            frame = env.reset()
        critic_size = int(env.metadata["critic_size"])
        actors, critics = {}, {}
        if state and state["model_spec"]["critic_size"] != critic_size:
            raise ValueError("Checkpoint critic schema differs from the Go environment")
        if state:
            validate_checkpoint_environment(state, env.metadata, defender=True)
        for role in ("enemy", "defender"):
            hidden = (state["model_spec"][f"{role}_hidden"] if state else config["actor_hidden"])
            actors[role] = PolicyNetwork(hidden).to(device)
            critic_hidden = state["model_spec"]["critic_hidden"] if state else config["critic_hidden"]
            critics[role] = CentralCritic(critic_size, critic_hidden).to(device)
            if state:
                actors[role].load_state_dict(state[f"{role}_actor"])
                critics[role].load_state_dict(state[f"{role}_critic"])
        if initial_enemy_checkpoint:
            source_snapshot = capture_checkpoint(initial_enemy_checkpoint)
            initial, digest = source_snapshot.state, source_snapshot.sha256
            del source_snapshot  # avoid retaining a second copy of a large source file
            # Fine-tuning starts fresh optimizers, but must not silently relabel
            # a calibrated profile or change its reference difficulty band.
            # Unrated/unlabeled fixtures retain the lightweight test workflow.
            initial_difficulty = initial.get("difficulty")
            if not allow_profile_transfer and initial_difficulty in {"trainee", "expert", "ace"}:
                if config.get("difficulty") != initial_difficulty:
                    raise ValueError("Fine-tuning difficulty must match the initial checkpoint")
                initial_band = initial.get("target_defender_win_rate")
                target_band = config.get("target_defender_win_rate")
                if ((tuple(initial_band) if initial_band is not None else None)
                        != (tuple(target_band) if target_band is not None else None)):
                    raise ValueError("Fine-tuning target_defender_win_rate must match the initial checkpoint")
            validate_checkpoint_environment(initial, env.metadata, defender="defender_actor" in initial)
            if str(initial.get("kind", "")).startswith("smoke"):
                config["smoke"] = True
            actors["enemy"] = actor_from_checkpoint(initial, "enemy", device)
            if "defender_actor" in initial:
                actors["defender"] = actor_from_checkpoint(initial, "defender", device)
            if config.get('initialize_critics_from_source'):
                source_spec = initial.get('model_spec', {})
                if (source_spec.get('critic_size') != critic_size
                        or tuple(source_spec.get('critic_hidden', ())) != tuple(config['critic_hidden'])):
                    raise ValueError('Source critic schema must match the environment input and configured hidden sizes')
                # Validate both dictionaries before loading either. Optimizers,
                # counters and league state still start fresh in this new run.
                for role in ('enemy', 'defender'):
                    values = initial.get(f'{role}_critic')
                    expected = critics[role].state_dict()
                    if not isinstance(values, Mapping) or set(values) != set(expected):
                        raise ValueError(f'Source {role} critic weights are missing or have incompatible keys')
                    if any(not torch.is_tensor(values[key]) or values[key].shape != tensor.shape
                           or not torch.isfinite(values[key]).all() for key, tensor in expected.items()):
                        raise ValueError(f'Source {role} critic contains incompatible shapes or nonfinite weights')
                for role in ('enemy', 'defender'):
                    critics[role].load_state_dict(initial[f'{role}_critic'])
            source_config = initial.get("config", {})
            initial_source = {
                "sha256": digest, "filename": Path(initial_enemy_checkpoint).name,
                "difficulty": initial_difficulty, "kind": initial.get("kind"),
                "iteration": initial.get("iteration"), "model_spec": initial.get("model_spec"),
                "profile_transfer": bool(allow_profile_transfer),
                "critic_initialization": 'source' if config.get('initialize_critics_from_source') else 'fresh',
                "calibration_ancestry": calibration_campaigns(initial),
                "calibration_config": {key: source_config[key] for key in (
                    "calibration_version", "eval_seed", "eval_episodes", "eval_max_steps",
                    "eval_opponents", "primary_opponents", "diagnostic_opponents",
                    "target_defender_win_rate", "max_truncation_rate", "max_stagnation_rate",
                    "min_evaluation_episodes", "consumed_campaigns", "difficulty_control") if key in source_config},
            }
            if source_config.get("difficulty_control") is not None and control is None:
                raise ValueError("Initial policy has difficulty_control; explicitly preserve or replace it in the new config")
            del initial
        if control is not None:
            actors["enemy"].set_masked_features(control["masked_features"])
        if config.get('deployment_sampler', {}).get('mode') == 'categorical' and config.get('eval_every', 0):
            if tuple(actors['enemy'].hidden_sizes) not in {(64, 32), (32, 16)}:
                raise ValueError('Native categorical calibration supports only tiny actors; use argmax for larger teachers')
            if not callable(getattr(env, 'factory_with_enemy_model', None)):
                raise ValueError('Categorical training calibration requires a native-capable BatchedEnv')
        learners = {role: PPOLearner(actors[role], critics[role], config,
                                     controller if role == "enemy" else None) for role in actors}
        pool = OpponentPool(config["league_size"])
        start = int(state["iteration"]) if state else 0
        actor_updates = {role: int(state.get(f"{role}_optimizer_updates", 0)) if state else 0
                         for role in actors}
        if state:
            for role in actors:
                learners[role].load_optimizer_state(state["optimizers"][role])
            pool.load_state_dict(state["league"])
            restore_rng(state["rng"])
            frame = env.reset(episode_configs(config, start, config["num_envs"]))
        progress_tracker = (ProgressTracker(frame.enemy_active.shape, config["progress_recovery"])
                            if config.get("progress_recovery") is not None else None)
        last_metrics: dict = {}
        selection = dict(state.get("selection", {})) if state else {}
        last_checkpoint = Path(resume) if resume else output / "latest.pt"
        for iteration in range(start, config["iterations"]):
            role = training_role(config, iteration)
            other = "defender" if role == "enemy" else "enemy"
            opponent, opponent_label = pool.sample(
                other, actors[other], scripted_probability=config["scripted_probability"],
                latest_probability=config["latest_probability"], device=device)
            rollout, frame, episodes = collect_rollout(
                env, frame, actors[role], opponent, role=role, config=config,
                iteration=iteration, device=device, progress_tracker=progress_tracker)
            metrics = learners[role].update(rollout)
            actor_updates[role] += int(metrics["actor_optimizer_steps"])
            outcomes: dict[str, int] = {}
            for episode in episodes:
                name = episode.get("outcome", "unknown")
                outcomes[name] = outcomes.get(name, 0) + 1
            last_metrics = {"iteration": iteration + 1, "role": role, "opponent": opponent_label,
                            "room_transitions": (iteration + 1) * config["horizon"] * config["num_envs"],
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "episodes_finished": len(episodes), "outcomes": outcomes, **metrics}
            if controller is not None:
                last_metrics.update(controller.metrics())
            log_metrics(output / "metrics.jsonl", last_metrics)
            if config["log_every"] and ((iteration + 1) % config["log_every"] == 0
                                        or iteration + 1 == config["iterations"]):
                print(f"[{iteration + 1}/{config['iterations']}] {role} vs {opponent_label} "
                      f"actor_samples={int(metrics['actor_samples'])} "
                      f"policy_loss={metrics.get('loss_objective', 0):.5f} "
                      f"value_loss={metrics['loss_critic']:.5f} "
                      f"episodes={len(episodes)} elapsed={last_metrics['elapsed_seconds']:.1f}s", flush=True)
            if (iteration + 1) % config["league_every"] == 0:
                for member in actors:
                    pool.add(member, actors[member], iteration + 1)
            if config.get("eval_every", 0) and (iteration + 1) % config["eval_every"] == 0:
                from .evaluation import evaluate, calibration_metrics
                # Evaluation must not perturb training's policy sampling RNG.
                rng = capture_rng()
                try:
                    evaluation_config = {"episodes": config["eval_episodes"], "seed": config["eval_seed"],
                                         "max_steps": config.get("eval_max_steps", 9000),
                                         "level_ids": list(range(1, 11)),
                                         "opponents": ["guard", "hunter", "wander"]}
                    if int(config.get("calibration_version", 1)) in (2, 3):
                        # Carry the complete v2 contract (primary/diagnostic
                        # pools, quality gates and campaign identity).
                        evaluation_config = {**config, **evaluation_config,
                            "eval_seed": config["eval_seed"],
                            "level_ids": config.get("eval_level_ids", list(range(1, 11))),
                            "opponents": config.get("eval_opponents", ["guard", "hunter", "wander"])}
                    if config.get('deployment_sampler', {}).get('mode') == 'categorical':
                        from .evaluation import evaluate_current_actor_native
                        evaluation_config['deployment_sampler'] = config['deployment_sampler']
                        result = evaluate_current_actor_native(actors['enemy'], env, device, evaluation_config,
                                                                optimizer_updates=int(actor_updates['enemy']))
                    else:
                        result = evaluate(actors["enemy"], env_factory, device, evaluation_config)
                    log_metrics(output / "evaluation.jsonl", {"iteration": iteration + 1, **result})
                finally:
                    restore_rng(rng)
                if "target_defender_win_rate" in config:
                    assessment = calibration_metrics(result, config)
                    if actor_updates["enemy"] == 0:
                        assessment["eligible"] = False
                        assessment["untrained_enemy"] = True
                    if controller is not None:
                        if actor_updates["enemy"] > 0:
                            controller.update(assessment)
                        else:
                            controller.last_status = "frozen_untrained"
                        log_metrics(output / "difficulty.jsonl", {"iteration": iteration + 1,
                                    "calibration": assessment, **controller.metrics()})
                    candidate = {"iteration": iteration + 1, **assessment}
                    changed = (not selection.get("best_candidate") or
                               candidate["rank"] < selection["best_candidate"]["rank"])
                    eligible_changed = assessment["eligible"] and (
                        not selection.get("best_eligible") or
                        candidate["rank"] < selection["best_eligible"]["rank"])
                    if changed:
                        selection["best_candidate"] = candidate
                    if eligible_changed:
                        selection["best_eligible"] = candidate
                    selection["calibrated"] = bool(selection.get("best_eligible"))
                    selection["latest"] = candidate
                    selection["reference"] = assessment.get("reference", "Fixed guard/hunter/wander bots, negative held-out seeds; not human difficulty")
                    selection["calibration_version"] = config.get("calibration_version", 1)
                    evaluated = _snapshot(config, iteration + 1, actors, critics, learners,
                                          pool, env.metadata, actor_updates, selection, controller, initial_source)
                    evaluated["evaluation"] = result
                    evaluated["calibration"] = assessment
                    if changed:
                        save_checkpoint(output / "best.pt", evaluated)
                    if eligible_changed:
                        save_checkpoint(output / "best_eligible.pt", evaluated)
                    save_json(output / "selection.json", selection)
            if ((iteration + 1) % config["checkpoint_every"] == 0
                    or iteration + 1 == config["iterations"]):
                checkpoint = _snapshot(config, iteration + 1, actors, critics, learners,
                                       pool, env.metadata, actor_updates, selection, controller, initial_source)
                save_checkpoint(output / f"checkpoint-{iteration + 1:06d}.pt", checkpoint)
                last_checkpoint = save_checkpoint(output / "latest.pt", checkpoint)
        return {"checkpoint": str(last_checkpoint), "iteration": max(start, config["iterations"]),
                "metrics": last_metrics, "optimizer_updates": sum(actor_updates.values()),
                "selection": selection, "difficulty_state": controller.state_dict() if controller else None}
    finally:
        env.close()
