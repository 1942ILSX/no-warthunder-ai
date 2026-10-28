"""Online dataset aggregation: the teacher labels states visited by the student.

This is intentionally not a static imitation dataset of perfect teacher play.
The small policy controls the game, including its own mistakes, while a frozen
teacher supplies a masked action distribution for those same observations.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from copy import deepcopy
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import (actor_from_checkpoint, capture_rng, cpu_state, load_checkpoint,
                         log_metrics, prepare_output, restore_rng, save_checkpoint,
                         validate_checkpoint_environment, validate_resume_config)
from .difficulty import control_config
from .models import PolicyNetwork, act
from .trainer import episode_configs, resolved_config, seed_everything



# These describe teacher selection/control, not student optimization cadence.
# CLI defaults must omit these keys when the user did not explicitly provide a
# contract; inheritance happens before resolved_config adds unrelated defaults.
TEACHER_CONTRACT_KEYS = frozenset({
    "difficulty_control", "progress_recovery", "deployment_sampler", "calibration_version", "eval_seed", "eval_episodes",
    "eval_max_steps", "eval_opponents", "min_evaluation_episodes", "max_truncation_rate",
    "max_stagnation_rate", "consumed_campaigns",
})


def inherit_teacher_contract(config: Mapping, teacher_state: Mapping) -> dict:
    """Copy an omitted teacher contract; reject explicit incompatible values.

    Return independent nested containers so caller edits cannot mutate teacher
    provenance. eval_every remains the student's own logging/evaluation cadence.
    Legacy unlabeled checkpoints remain usable by the small unit-test workflow.
    """
    result = deepcopy(dict(config))
    source = teacher_state.get("config", {})
    expected = {key: deepcopy(value) for key, value in source.items()
                if key in TEACHER_CONTRACT_KEYS or key.startswith(("v2_", "v3_"))}
    # Current checkpoints keep these in config; support their explicit top-level
    # provenance too without treating missing legacy metadata as a new contract.
    for key in ("difficulty_control", "progress_recovery", "deployment_sampler", "calibration_version", "consumed_campaigns"):
        if key not in expected and key in teacher_state:
            expected[key] = deepcopy(teacher_state[key])
    if "difficulty_control" in result:
        # Missing legacy control means no gate; adding one is also a contract
        # change, rather than an implicit invitation to relabel the teacher.
        expected.setdefault("difficulty_control", None)
    difficulty = teacher_state.get("difficulty", source.get("difficulty", "unrated"))
    if difficulty in {"trainee", "expert", "ace"}:
        expected["difficulty"] = difficulty
    band = teacher_state.get("target_defender_win_rate", source.get("target_defender_win_rate"))
    if band is not None:
        expected["target_defender_win_rate"] = deepcopy(band)
    for key, value in expected.items():
        if key in result:
            if key == "difficulty_control":
                same = control_config(result[key]) == control_config(value)
            elif key == 'deployment_sampler':
                from .sampling import deployment_sampler
                same = deployment_sampler(result[key]) == deployment_sampler(value)
            else:
                same = (tuple(result[key]) == tuple(value) if key == "target_defender_win_rate"
                        and result[key] is not None else result[key] == value)
            if not same:
                label = "target band" if key == "target_defender_win_rate" else key
                raise ValueError(f"Distillation {label} must match the teacher profile contract")
        else:
            result[key] = deepcopy(value)
    return result

def masked_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                             masks: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Finite sentinel avoids 0 * -inf in KL for illegal actions. Their teacher
    # probabilities underflow to zero, so no target mass leaks to illegal moves.
    student = (student_logits / temperature).masked_fill(~masks, -1e9)
    teacher = (teacher_logits.detach() / temperature).masked_fill(~masks, -1e9)
    return F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1),
                    reduction="batchmean") * temperature**2


def distill(config: Mapping, env_factory: Callable, device: str | torch.device,
            output_dir: str | Path, teacher_checkpoint: str | Path,
            resume: str | Path | None = None) -> dict:
    """Bounded distillation, with checkpoint-compatible tiny actors for PPO."""
    state = load_checkpoint(resume) if resume else None
    if state and state.get("kind") not in {"distillation", "smoke_distillation"}:
        raise ValueError("Resume requires a distillation checkpoint")
    # A resumed student uses its embedded teacher even if the source moved.
    teacher_state = state["teacher_checkpoint"] if state else load_checkpoint(teacher_checkpoint)
    config = resolved_config({"learning_rate": 3e-4,
                              **inherit_teacher_contract(config, teacher_state)})
    config.setdefault("student_hidden", [64, 32])
    config.setdefault("temperature", 1.0)
    config.setdefault("dataset_size", 100_000)
    config.setdefault("epochs", 3)
    config.setdefault("learned_defender", False)
    if config["temperature"] <= 0 or config["dataset_size"] < 1 or config["epochs"] < 1:
        raise ValueError("Distillation needs positive temperature, dataset_size and epochs")
    output = prepare_output(output_dir, resume)
    device = torch.device(device)
    seed_everything(int(config["seed"]))
    if not teacher_state.get("trained") or teacher_state.get("enemy_optimizer_updates", 0) <= 0:
        raise ValueError("Teacher enemy actor has no recorded optimizer updates")
    if str(teacher_state.get("kind", "")).startswith("smoke") or (state and str(state.get("kind", "")).startswith("smoke")):
        config["smoke"] = True
    teacher_difficulty = teacher_state.get("difficulty", "unrated")
    if config.get("difficulty", teacher_difficulty) != teacher_difficulty:
        raise ValueError("Distillation difficulty must match the teacher profile")
    config.setdefault("difficulty", teacher_difficulty)
    teacher_band = teacher_state.get("target_defender_win_rate")
    if teacher_band is not None:
        if config.get("target_defender_win_rate", teacher_band) != teacher_band:
            raise ValueError("Distillation target band must match the teacher profile")
        config.setdefault("target_defender_win_rate", teacher_band)
    if state:
        validate_resume_config(config, state["config"])
    teacher = actor_from_checkpoint(teacher_state, "enemy", device).eval().requires_grad_(False)
    defender = (actor_from_checkpoint(teacher_state, "defender", device).eval().requires_grad_(False)
                if config["learned_defender"] and "defender_actor" in teacher_state else None)
    student = PolicyNetwork(state["model_spec"]["enemy_hidden"] if state else config["student_hidden"]).to(device)
    student_control = control_config(config.get("difficulty_control"))
    student.set_masked_features(student_control["masked_features"] if student_control else [])
    student.enforce_input_gate()
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config["learning_rate"]))
    # Store a rolling, bounded aggregate on CPU; GPU only sees minibatches.
    observations = torch.empty((0, 320), dtype=torch.float32)
    masks = torch.empty((0, 10), dtype=torch.bool)
    labels = torch.empty((0, 10), dtype=torch.float32)
    start, updates = 0, 0
    if state:
        student.load_state_dict(state["enemy_actor"])
        student.enforce_input_gate()
        optimizer.load_state_dict(state["distillation_optimizer"])
        observations, masks, labels = (state["dataset"][name] for name in ("observations", "masks", "labels"))
        start, updates = int(state["iteration"]), int(state["optimizer_updates"])
        restore_rng(state["rng"])
    env = env_factory(episode_configs(config, start, config["num_envs"]))
    started = time.monotonic()
    try:
        validate_checkpoint_environment(teacher_state, env.metadata, defender="defender_actor" in teacher_state)
        frame = getattr(env, "frame", None)
        if frame is None:
            frame = env.reset()
        last_metrics = {}
        last_checkpoint = Path(resume) if resume else output / "latest.pt"
        for iteration in range(start, config["iterations"]):
            seen_obs, seen_masks = [], []
            for _ in range(config["horizon"]):
                # Student actions, not teacher actions, generate all new states.
                enemy_actions, _, _ = act(student, frame.enemy_obs, frame.enemy_mask, frame.enemy_active, device)
                defender_actions, _, _ = act(defender, frame.defender_obs, frame.defender_mask,
                                              frame.defender_active, device)
                if frame.enemy_active.any():
                    seen_obs.append(frame.enemy_obs[frame.enemy_active].copy())
                    seen_masks.append(frame.enemy_mask[frame.enemy_active].copy())
                frame = env.step(enemy_actions, defender_actions)
                done = np.flatnonzero(frame.terminated | frame.truncated)
                if len(done):
                    frame = env.reset_indices(done, episode_configs(config, iteration, len(done)))
            fresh = 0
            if seen_obs:
                fresh_obs = torch.from_numpy(np.concatenate(seen_obs))
                fresh_masks = torch.from_numpy(np.concatenate(seen_masks))
                with torch.no_grad():
                    fresh_labels = torch.cat([teacher(batch.to(device)).cpu()
                                              for batch in fresh_obs.split(config["minibatch_size"])])
                fresh = len(fresh_obs)
                limit = int(config["dataset_size"])
                observations = torch.cat((observations, fresh_obs))[-limit:]
                masks = torch.cat((masks, fresh_masks))[-limit:]
                labels = torch.cat((labels, fresh_labels))[-limit:]
            losses = []
            for _ in range(int(config["epochs"])):
                for indices in torch.randperm(len(observations)).split(config["minibatch_size"]):
                    if not len(indices):
                        continue
                    loss = masked_distillation_loss(student(observations[indices].to(device)),
                                                    labels[indices].to(device), masks[indices].to(device),
                                                    float(config["temperature"]))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    student.enforce_input_gate()
                    losses.append(loss.detach())
                    updates += 1
            last_metrics = {"iteration": iteration + 1, "kind": "distillation",
                            "new_student_states": fresh, "dataset_states": len(observations),
                            "kl_loss": float(torch.stack(losses).mean().cpu()) if losses else 0.0,
                            "optimizer_updates": updates,
                            "elapsed_seconds": round(time.monotonic() - started, 3)}
            log_metrics(output / "metrics.jsonl", last_metrics)
            if config["log_every"] and ((iteration + 1) % config["log_every"] == 0
                                        or iteration + 1 == config["iterations"]):
                print(f"[{iteration + 1}/{config['iterations']}] distillation "
                      f"new_states={fresh} dataset={len(observations)} "
                      f"kl={last_metrics['kl_loss']:.5f} updates={updates} "
                      f"elapsed={last_metrics['elapsed_seconds']:.1f}s", flush=True)
            if ((iteration + 1) % config["checkpoint_every"] == 0
                    or iteration + 1 == config["iterations"]):
                spec = {**teacher_state["model_spec"], "enemy_hidden": student.hidden_sizes}
                snapshot = {"format_version": 1,
                            "kind": "smoke_distillation" if config.get("smoke") else "distillation",
                            "trained": updates > 0, "optimizer_updates": updates,
                            "enemy_optimizer_updates": updates,
                            "defender_optimizer_updates": teacher_state.get("defender_optimizer_updates", 0),
                            "difficulty": teacher_state.get("difficulty", "unrated"),
                            "target_defender_win_rate": teacher_state.get("target_defender_win_rate"),
                            "calibration_version": config.get("calibration_version", 1),
                            "calibration": {"eligible": False, "reason": "Student requires its own evaluation"},
                            "selection": {"calibrated": False, "reason": "Student requires its own evaluation"},
                            "source_difficulty_state": deepcopy(teacher_state.get("difficulty_state", {})),
                            "source_teacher_checkpoint": (state.get("source_teacher_checkpoint", str(teacher_checkpoint))
                                                          if state else str(teacher_checkpoint)),
                            "source_teacher_identity": deepcopy(teacher_state.get("checkpoint_identity")),
                            "observation_version": teacher_state["observation_version"],
                            "normalization_version": teacher_state.get("normalization_version", 2),
                            "defender_observation_version": teacher_state.get("defender_observation_version", 1),
                            "model_spec": spec, "config": dict(config), "iteration": iteration + 1,
                            "enemy_actor": cpu_state(student),
                            **{key: teacher_state[key] for key in ("defender_actor", "enemy_critic", "defender_critic")
                               if key in teacher_state},
                            "distillation_optimizer": optimizer.state_dict(), "rng": capture_rng(),
                            "teacher_checkpoint": teacher_state,
                            "dataset": {"observations": observations, "masks": masks, "labels": labels},
                            "resume_semantics": "Student, optimizer, teacher, dataset and RNG restored; live rooms restart."}
                save_checkpoint(output / f"checkpoint-{iteration + 1:06d}.pt", snapshot)
                last_checkpoint = save_checkpoint(output / "latest.pt", snapshot)
        return {"checkpoint": str(last_checkpoint), "iteration": max(start, config["iterations"]),
                "optimizer_updates": updates, "metrics": last_metrics}
    finally:
        env.close()
