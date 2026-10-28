"""Atomic, portable training checkpoints and append-only JSONL metrics."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import os
import random
import tempfile

import numpy as np
import torch

from .models import PolicyNetwork


def cpu_state(module: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def capture_rng() -> dict:
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path: str | Path, state: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    # Training checkpoints contain optimizer/RNG state. Only load checkpoints
    # produced by this project or otherwise trusted; pickle is not a sandbox.
    result = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(result, dict) or result.get("format_version") != 1:
        raise ValueError("Unsupported training checkpoint format")
    return result


def actor_from_checkpoint(state: Mapping, role: str = "enemy",
                          device: str | torch.device = "cpu") -> PolicyNetwork:
    if role not in ("enemy", "defender"):
        raise ValueError("role must be enemy or defender")
    spec = state["model_spec"]
    actor = PolicyNetwork(spec[f"{role}_hidden"], spec.get("observation_size", 320),
                          spec.get("action_size", 10)).to(device)
    actor.load_state_dict(state[f"{role}_actor"])
    if role == "enemy":
        control = state.get("config", {}).get("difficulty_control")
        if control is not None:
            from .difficulty import control_config
            actor.set_masked_features(control_config(control)["masked_features"])
    actor.checkpoint_provenance = {
        "observation_version": state.get("observation_version"),
        "normalization_version": state.get("normalization_version"),
        "defender_observation_version": state.get("defender_observation_version"),
        "role": role,
    }
    return actor


def load_actor_checkpoint(path: str | Path, role: str = "enemy",
                          device: str | torch.device = "cpu") -> PolicyNetwork:
    return actor_from_checkpoint(load_checkpoint(path), role, device)


def validate_checkpoint_environment(state: Mapping, metadata: Mapping, *, defender: bool = False) -> None:
    keys = ["observation_version", "normalization_version"]
    if defender:
        keys.append("defender_observation_version")
    for key in keys:
        if state.get(key) is None or state.get(key) != metadata.get(key):
            raise ValueError(f"Checkpoint {key} differs from the Go environment")


def validate_resume_config(config: Mapping, saved: Mapping) -> None:
    """A resume cannot silently change the optimizer, league or calibration."""
    allowed = {"iterations", "log_every", "checkpoint_every", "device", "smoke"}
    missing = object()
    for key in set(config) | set(saved):
        if key in allowed:
            continue
        left, right = config.get(key, missing), saved.get(key, missing)
        if left is missing or right is missing or json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
            raise ValueError(f"Resume config changed {key}; use the original config or start a new fine-tuning run")


def prepare_output(path: str | Path, resume: str | Path | None) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new run or resume")
    return output


def log_metrics(path: str | Path, metrics: Mapping) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(metrics), ensure_ascii=False, allow_nan=False) + "\n")


def save_json(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
