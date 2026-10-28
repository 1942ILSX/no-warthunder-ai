"""Explicit, immutable behavior-based profile qualification of a frozen actor.

Qualification changes metadata, never model parameters. It consumes the
selection campaign and does not replace a fresh independent release evaluation.
Only trusted local checkpoints should be passed to this pickle-based workflow.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import torch

from .checkpoint import actor_from_checkpoint
from .config import PROFILES, read_yaml
from .env import PROJECT_ROOT
from .evaluation import assess_report
from .provenance import calibration_campaigns, capture_checkpoint


def _tensor_digest(state: Mapping) -> str:
    """Fingerprint model tensor content independently of torch archive metadata."""
    digest = hashlib.sha256()
    for group in sorted(key for key in state if key.endswith(("_actor", "_critic"))):
        for name, tensor in sorted(state[group].items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([group, name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    """Publish a complete checkpoint atomically without overwriting any file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def qualify_checkpoint(source: str | Path, evaluation_path: str | Path,
                       output: str | Path, target_profile: str = "expert") -> dict:
    """Clone a measured actor into a target profile without training or deployment.

    The captured report must evaluate exactly the captured checkpoint with
    the saved sampler (native Go evidence for categorical). Actual episode records, rather than claimed eligibility,
    determine qualification under the target profile's fixed-bot band. Output
    is a new file; overwriting a source or prior qualification is forbidden.
    """
    if target_profile not in PROFILES:
        raise ValueError("Unknown target difficulty profile")
    snapshot = capture_checkpoint(source)
    state = snapshot.state
    destination = Path(output).expanduser().resolve()
    if destination == snapshot.path or destination.exists():
        raise FileExistsError(f"Qualification requires a new checkpoint path: {destination}")
    if (not state.get("trained") or int(state.get("enemy_optimizer_updates", 0)) <= 0
            or str(state.get("kind", "")).startswith("smoke") or state.get("config", {}).get("smoke")):
        raise ValueError("Qualification requires a trained, non-smoke enemy actor")
    if state.get("observation_version") != 2 or state.get("normalization_version") != 2:
        raise ValueError("Qualification requires observation/normalization version 2")
    spec = state.get("model_spec", {})
    if (tuple(spec.get("enemy_hidden", ())) not in {(64, 32), (32, 16)}
            or spec.get("observation_size") != 320 or spec.get("action_size") != 10):
        raise ValueError("Qualification requires the supported tiny deployment architecture")
    actor = actor_from_checkpoint(state)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Qualification cannot use nonfinite actor parameters")
    if any(not torch.equal(value, state["enemy_actor"][name]) for name, value in actor.state_dict().items()):
        raise ValueError("Stored actor differs from its effective input-gated parameters")
    report_bytes = Path(evaluation_path).read_bytes()
    report = json.loads(report_bytes)
    if not isinstance(report, dict) or report.get("checkpoint_identity", {}).get("sha256") != snapshot.sha256:
        raise ValueError("Qualification report checkpoint SHA256 does not match the source")
    if report.get("difficulty") != state.get("difficulty"):
        raise ValueError("Qualification report must identify the source difficulty")
    version = report.get("metrics_version")
    if version not in (2, 3) or report.get("retrospective_assessment"):
        raise ValueError("Qualification requires fresh v2/v3 behavioral telemetry")
    from .export import validate_evaluated_sampler
    from .sampling import deployment_sampler
    sampler = deployment_sampler(state.get('config', {}).get('deployment_sampler'))
    validate_evaluated_sampler(snapshot, report, sampler['mode'], sampler['temperature'])
    measured = report.get("evaluation_config", {})
    if type(measured.get("eval_seed")) is not int or type(measured.get("episodes")) is not int:
        raise ValueError("Qualification report must declare its seed campaign")
    seed, episodes = abs(measured["eval_seed"]), measured["episodes"]
    if episodes < 1 or seed + episodes > 2**63:
        raise ValueError("Invalid qualification seed campaign")
    rows = report.get("records")
    if not isinstance(rows, list):
        raise ValueError("Qualification requires actual episode records")
    expected_seeds = set(range(-seed - episodes, -seed))
    if any(type(row.get("seed")) is not int or row["seed"] not in expected_seeds for row in rows):
        raise ValueError("Qualification records do not match their declared seed campaign")
    for opponent in ("guard", "hunter"):
        seeds = [row["seed"] for row in rows if row.get("opponent") == opponent]
        if len(seeds) != episodes or set(seeds) != expected_seeds:
            raise ValueError(f"Qualification needs each declared seed exactly once for {opponent}")
    profile = read_yaml(PROJECT_ROOT / "configs" / f"{target_profile}.yaml")
    config = deepcopy(state["config"])
    config.update({key: deepcopy(profile[key]) for key in ("difficulty", "target_defender_win_rate", "score_multiplier")})
    config["calibration_version"] = version
    if version == 3:
        # Detector settings do not affect the policy; they describe telemetry.
        config["progress_recovery"] = deepcopy(report.get("progress_recovery"))
    assessment = assess_report(report, config)["calibration"]
    if not assessment["eligible"]:
        raise ValueError("Candidate failed target qualification: " + "; ".join(assessment["reasons"]))
    lineage = calibration_campaigns(state)
    consumed = {"seed": seed, "episodes": episodes, "purpose": "profile_qualification_selection"}
    config["consumed_campaigns"] = [*lineage, consumed]
    source_digest = _tensor_digest(state)
    qualification = {
        "version": 1, "method": "frozen_actor_behavioral_qualification",
        "source_checkpoint_sha256": snapshot.sha256, "source_difficulty": state.get("difficulty"),
        "source_kind": state.get("kind"), "source_iteration": state.get("iteration"),
        "evaluation_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "target_difficulty": target_profile, "assessment": assessment,
        "selection_campaign": consumed, "model_tensors_sha256": source_digest,
        "weights_changed": False, "release_evaluation_required": True,
    }
    result = {**state, "kind": "qualified", "difficulty": target_profile,
              "target_defender_win_rate": config["target_defender_win_rate"],
              "score_multiplier": config["score_multiplier"], "calibration_version": version,
              "config": config, "calibration_ancestry": [*lineage, consumed],
              "qualification": qualification,
              "selection": {"qualified": True, "calibrated": True, "assessment": assessment,
                            "release_evaluation_required": True},
              "resume_semantics": "Frozen qualified actor; initialize a new fine-tuning run to change weights."}
    if _tensor_digest(result) != source_digest:
        raise RuntimeError("Qualification unexpectedly changed model tensors")
    stream = io.BytesIO()
    torch.save(result, stream)
    output_bytes = stream.getvalue()
    _write_new(destination, output_bytes)
    return {"checkpoint": str(destination), "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "difficulty": target_profile, "qualification": qualification}
