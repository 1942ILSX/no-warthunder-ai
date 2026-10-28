"""Small, audited deployment format: plain JSON plus contiguous FP32 weights."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from .env import build_tool
from .provenance import CheckpointSnapshot, capture_checkpoint
from .sampling import deployment_sampler

ALLOWED_HIDDEN = {(64, 32), (32, 16)}
UINT64_MAX = (1 << 64) - 1
DEFAULT_RANDOM = 0x9e3779b97f4a7c15


def actor_weight_bytes(actor: torch.nn.Module) -> bytes:
    """Canonical production bytes, shared by candidate evaluation and release."""
    if tuple(actor.hidden_sizes) not in ALLOWED_HIDDEN:
        raise ValueError('Deployment accepts only 64,32 or 32,16 hidden sizes. Distill a teacher first.')
    layers = [module for module in actor.net if isinstance(module, torch.nn.Linear)]
    sizes = [320, *actor.hidden_sizes, 10]
    if len(layers) != 3 or any(layer.weight.shape != (sizes[i+1], sizes[i]) for i, layer in enumerate(layers)):
        raise ValueError('Actor does not match the production MLP shape')
    if [type(module) for module in actor.net] != [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear]:
        raise ValueError('Deployment requires exactly Linear/ReLU/Linear/ReLU/Linear')
    data = bytearray()
    for layer in layers:
        for tensor in (layer.weight, layer.bias):
            if tensor is None:
                raise ValueError('Every layer requires a bias')
            array = tensor.detach().cpu().numpy().astype('<f4', copy=False)
            if not np.isfinite(array).all():
                raise ValueError('Non-finite actor weights')
            data.extend(array.tobytes(order='C'))
    return bytes(data)


def write_artifact(actor: torch.nn.Module, output: str | Path, *, policy_version: str,
                   training_seed: int, trained: bool, difficulty: str, sampler: str = 'argmax',
                   temperature: float = 1.0) -> dict[str, Any]:
    if difficulty not in {'trainee', 'expert', 'ace'}:
        raise ValueError('difficulty must be trainee, expert, or ace')
    if re.fullmatch(r'[A-Za-z0-9._-]{1,64}', policy_version) is None:
        raise ValueError('Policy version must use 1–64 ASCII letters, digits, dot, underscore or hyphen')
    if type(training_seed) is not int or not 0 <= training_seed < 2**63:
        raise ValueError('training_seed must be a nonnegative int64')
    sampling = deployment_sampler({'mode': sampler, 'temperature': temperature})
    data = actor_weight_bytes(actor)
    metadata = {
        'schemaVersion': 1, 'modelName': 'no-warthunder-tiny', 'policyVersion': policy_version,
        'trainingSeed': training_seed, 'trained': bool(trained), 'difficulty': difficulty, 'observationVersion': 2,
        'normalizationVersion': 2, 'inputSize': 320, 'hiddenSizes': list(actor.hidden_sizes), 'outputSize': 10,
        'activation': 'relu', 'actionEncoding': 'move*2+fire', 'decisionHz': 10, 'observationDelayDecisions': 2,
        'weightsFile': 'weights.bin', 'weightsFormat': 'float32-le',
        'weightsLayout': 'out-in-row-major-weights-then-bias',
        'weightsSHA256': hashlib.sha256(data).hexdigest(),
        'sampler': sampling,
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'weights.bin').write_bytes(data)
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def export_checkpoint(checkpoint: str | Path, output: str | Path, *, policy_version: str,
                      sampler: str | None = None, temperature: float | None = None,
                      allow_untrained: bool = False, evaluation: str | Path | None = None) -> dict:
    # Checkpoints are local training artifacts with optimizer/RNG state; load only trusted files.
    from .checkpoint import actor_from_checkpoint
    snapshot = capture_checkpoint(checkpoint)
    payload = snapshot.state
    contract = deployment_sampler(payload.get('config', {}).get('deployment_sampler'))
    requested = deployment_sampler({**contract, **({'mode': sampler} if sampler is not None else {}),
                                    **({'temperature': temperature} if temperature is not None else {})})
    if requested != contract:
        raise ValueError(f'Requested export sampler differs from checkpoint deployment_sampler ({contract["mode"]})')
    sampler, temperature = contract['mode'], contract['temperature']
    trained = bool(payload.get('trained')) and int(payload.get('enemy_optimizer_updates', 0)) > 0 and not str(payload.get('kind', '')).startswith('smoke')
    if not trained and not allow_untrained:
        raise ValueError('Untrained/smoke checkpoint. Use a trained student; --allow-untrained is only for parity tests.')
    if payload.get('observation_version') != 2 or payload.get('normalization_version') != 2:
        raise ValueError('Checkpoint has an incompatible observation/normalization version')
    assessment = None
    version = payload.get('config', {}).get('calibration_version', 1)
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError('Unknown checkpoint calibration version')
    if trained and sampler == 'categorical' and version not in (2, 3):
        raise ValueError('Trained categorical export requires a calibrated v2/v3 checkpoint contract')
    if trained and version in (2, 3):
        if evaluation is None:
            raise ValueError(f'Trained v{version} export requires --evaluation from an independent evaluation of this exact checkpoint')
        assessment = validate_release_evaluation(snapshot, evaluation, sampler, temperature)
    actor = actor_from_checkpoint(payload, role='enemy', device='cpu')
    metadata = write_artifact(actor, output, policy_version=policy_version,
                              training_seed=int(payload.get('config', {}).get('seed', 0)), trained=trained,
                              difficulty=payload.get('difficulty', payload.get('config', {}).get('difficulty', 'expert')),
                              sampler=sampler, temperature=temperature)
    notes = {'difficulty': metadata['difficulty'], 'proposedScoreMultiplier': payload.get('config', {}).get('score_multiplier'),
             'appliedToTrainingReward': False, 'requiresGameIntegration': True,
             'calibrationVerified': assessment is not None,
             'calibration': assessment,
             'difficultyControl': payload.get('config', {}).get('difficulty_control')}
    if assessment is not None:
        notes['checkpointSHA256'] = snapshot.sha256
        notes['evaluationSHA256'] = assessment.pop('_evaluation_sha256')
    (Path(output) / 'deployment-notes.json').write_text(json.dumps(notes, indent=2) + '\n')
    return {'directory': str(Path(output).resolve()), 'parameter_count': sum(p.numel() for p in actor.parameters()),
            'weights_bytes': (Path(output) / 'weights.bin').stat().st_size, 'metadata': metadata}


def validate_evaluated_sampler(snapshot: CheckpointSnapshot, report: dict, sampler: str, temperature: float) -> None:
    """Categorical evidence must come from the actual seeded Go artifact."""
    from .checkpoint import actor_from_checkpoint
    actual = report.get('policy_sampler', {})
    if not isinstance(actual, dict) or actual.get('mode') != sampler or actual.get('temperature', 1.) != temperature:
        raise ValueError('Evaluation sampler/temperature differs from the exported policy')
    if sampler == 'argmax':
        if report.get('deterministic_policy') is not True:
            raise ValueError('Argmax export requires deterministic argmax evaluation')
        if 'native_policy' not in report:
            return
    native = report.get('native_policy', {})
    if not isinstance(native, dict) or not isinstance(native.get('metadata', {}), dict):
        raise ValueError('Export requires native Go policy evidence with valid metadata')
    metadata = native.get('metadata', {})
    expected = deployment_sampler({'mode': sampler, 'temperature': temperature})
    if (native.get('implementation') != 'go-policymlp' or metadata.get('sampler') != expected
            or actual != expected or report.get('reproducible_seeded_policy') is not True
            or report.get('deterministic_policy') is not (sampler == 'argmax')):
        raise ValueError('Export requires native Go policy evidence with the exact sampler and RNG')
    state = snapshot.state
    actor = actor_from_checkpoint(state, 'enemy', 'cpu')
    expected_metadata = {
        'schemaVersion': 1, 'modelName': 'no-warthunder-tiny', 'difficulty': state.get('difficulty'),
        'trainingSeed': int(state.get('config', {}).get('seed', 0)),
        'trained': True, 'observationVersion': 2, 'normalizationVersion': 2,
        'inputSize': 320, 'hiddenSizes': list(actor.hidden_sizes), 'outputSize': 10,
        'activation': 'relu', 'actionEncoding': 'move*2+fire', 'decisionHz': 10,
        'observationDelayDecisions': 2, 'weightsFile': 'weights.bin', 'weightsFormat': 'float32-le',
        'weightsLayout': 'out-in-row-major-weights-then-bias',
        'weightsSHA256': hashlib.sha256(actor_weight_bytes(actor)).hexdigest(),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError('Native Go evaluation artifact weights/schema do not match the exported checkpoint')


def evaluation_artifact(snapshot: CheckpointSnapshot, output: str | Path) -> dict:
    """Create only a temporary native-evaluation candidate, not a release certificate."""
    from .checkpoint import actor_from_checkpoint
    state = snapshot.state
    if (not state.get('trained') or int(state.get('enemy_optimizer_updates', 0)) <= 0
            or str(state.get('kind', '')).startswith('smoke') or state.get('config', {}).get('smoke')):
        raise ValueError('Native candidate evaluation requires a trained non-smoke checkpoint')
    if state.get('observation_version') != 2 or state.get('normalization_version') != 2:
        raise ValueError('Native candidate requires observation/normalization version2')
    contract = deployment_sampler(state.get('config', {}).get('deployment_sampler'))
    actor = actor_from_checkpoint(state, 'enemy', 'cpu')
    metadata = write_artifact(actor, output, policy_version='eval-' + snapshot.sha256[:20],
                              training_seed=int(state.get('config', {}).get('seed', 0)), trained=True,
                              difficulty=state['difficulty'], sampler=contract['mode'], temperature=contract['temperature'])
    (Path(output) / 'evaluation-only.json').write_text(json.dumps({
        'checkpointSHA256': snapshot.sha256, 'releaseCertified': False,
        'purpose': 'Temporary offline Go-native evaluation candidate; requires independent release gates.'}, indent=2) + '\n')
    return {'implementation': 'go-policymlp', 'metadata': metadata}


def validate_release_evaluation(snapshot: CheckpointSnapshot, evaluation, sampler, temperature=1.) -> dict:
    from .cli import evaluation_campaign
    from .evaluation import assess_report
    # The evaluation file also has one byte snapshot: its recorded hash must
    # describe the report actually assessed, not a later replacement.
    report_bytes = Path(evaluation).read_bytes()
    report = json.loads(report_bytes)
    if not isinstance(report, dict):
        raise ValueError('Release evaluation must be a JSON object')
    payload = snapshot.state
    if report.get('checkpoint_identity', {}).get('sha256') != snapshot.sha256:
        raise ValueError('Evaluation checkpoint SHA256 does not match the exported checkpoint')
    if report.get('difficulty') != payload.get('difficulty'):
        raise ValueError('Evaluation difficulty differs from the checkpoint')
    version = payload['config'].get('calibration_version')
    if version not in (2, 3) or report.get('metrics_version') != version or report.get('retrospective_assessment'):
        raise ValueError(f'v{version} export requires a fresh v{version} evaluation with matching behavioral telemetry')
    validate_evaluated_sampler(snapshot, report, sampler, temperature)
    config = payload['config']
    measured = report.get('evaluation_config', {})
    if type(measured.get('eval_seed')) is not int or type(measured.get('episodes')) is not int:
        raise ValueError('Evaluation campaign settings are missing')
    campaign = evaluation_campaign(payload, measured['eval_seed'], measured['episodes'])
    if campaign['overlapping_seeds_per_opponent']:
        raise ValueError('Release evaluation overlaps a consumed training/selection campaign')
    low, high = campaign['seed_interval']
    rows = report.get('records')
    if not isinstance(rows, list) or any(not isinstance(record, dict) or type(record.get('seed')) is not int
                                         or not low <= record['seed'] <= high for record in rows):
        raise ValueError('Evaluation records do not match their declared seed campaign')
    expected_seeds = set(range(low, high + 1))
    for opponent in ('guard', 'hunter'):
        seeds = [row['seed'] for row in rows if row.get('opponent') == opponent]
        if len(seeds) != measured['episodes'] or set(seeds) != expected_seeds:
            raise ValueError(f'Release evaluation needs each declared seed exactly once for {opponent}')
    assessment = assess_report(report, config)['calibration']
    if not assessment['eligible']:
        reasons = '; '.join(assessment.get('reasons', []))
        raise ValueError(f'v{version} evaluation failed difficulty/quality gates: ' + reasons)
    assessment['_evaluation_sha256'] = hashlib.sha256(report_bytes).hexdigest()
    return assessment


def artifact_actor(directory: str | Path) -> torch.nn.Module:
    from .models import PolicyNetwork
    root = Path(directory)
    metadata = json.loads((root / 'metadata.json').read_text())
    data = (root / 'weights.bin').read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata['weightsSHA256']:
        raise ValueError('Artifact checksum mismatch')
    if tuple(metadata['hiddenSizes']) not in ALLOWED_HIDDEN:
        raise ValueError('Unsupported deployed actor shape')
    actor = PolicyNetwork(tuple(metadata['hiddenSizes'])).cpu().eval()
    weights = np.frombuffer(data, dtype='<f4')
    offset = 0
    with torch.no_grad():
        for layer in actor.net:
            if not isinstance(layer, torch.nn.Linear):
                continue
            for tensor in (layer.weight, layer.bias):
                count = tensor.numel()
                tensor.copy_(torch.from_numpy(weights[offset:offset + count].copy().reshape(tensor.shape)))
                offset += count
    if offset != weights.size:
        raise ValueError('Incorrect artifact length')
    return actor


def verify_artifact(directory: str | Path, backend_path: str | Path | None = None,
                    observations: np.ndarray | None = None, seed: int = 73, samples: int = 128) -> dict:
    actor = artifact_actor(directory)
    supplied = observations is not None
    rng = np.random.default_rng(seed)
    if observations is None:
        observations = rng.uniform(-1, 1, (samples, 320)).astype(np.float32)
        masks = rng.integers(0, 2, (samples, 10), dtype=np.uint8).astype(bool)
        masks[:, 0] = True
        observations[:, 298:308] = masks.astype(np.float32)
    else:
        observations = np.asarray(observations, dtype=np.float32)
        masks = observations[:, 298:308] > .5
    with torch.no_grad():
        expected = actor(torch.from_numpy(observations)).numpy()
    # Small consecutive xorshift seeds produce tiny first uniforms and mostly
    # select action zero. Exercise the whole CDF with full-width states, plus
    # zero/extreme states and a chain of consecutive RNG transitions.
    random_states = parity_random_states(rng, len(observations))
    decisions = [17 * i for i in range(len(observations))]
    requests = [json.dumps({'version': 2, 'observation': obs.tolist(), 'mask': mask.tolist(),
                           'random': str(random_states[i]), 'decisions': decisions[i]})
                for i, (obs, mask) in enumerate(zip(observations, masks))]
    temperature = float(json.loads((Path(directory) / 'metadata.json').read_text())['sampler']['temperature'])
    checker = build_tool('policycheck', backend_path)
    mode_reports = {}
    for mode in ('argmax', 'categorical'):
        process = subprocess.run([str(checker), '-model', str(Path(directory).resolve()), '-sampler', mode, '-temperature', str(temperature)],
                                 input='\n'.join(requests) + '\n', capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise RuntimeError('Go parity check failed: ' + process.stderr[-4000:])
        responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        if len(responses) != len(requests) or any('error' in item for item in responses):
            raise RuntimeError('Unexpected Go parity responses: ' + process.stdout[:1000])
        actual = np.array([item['logits'] for item in responses], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        wanted = [reference_action(row, mask, random_states[i], mode, temperature)
                  for i, (row, mask) in enumerate(zip(expected, masks))]
        actual_actions = np.array([item['actionIndex'] for item in responses])
        if not np.array_equal(actual_actions, [item[0] for item in wanted]):
            raise AssertionError(f'Python/Go masked {mode} differs')
        if any(item['random'] != str(want[1]) or item['decisions'] != decisions[i] + 1
               for i, (item, want) in enumerate(zip(responses, wanted))):
            raise AssertionError('Python/Go RNG or decision counter differs')
        mode_reports[mode] = {'action_matches': len(wanted),
                             'action_counts': {str(action): int(np.count_nonzero(actual_actions == action)) for action in range(10)},
                             'max_absolute_logit_error': float(np.max(np.abs(actual - expected)))}
    report = {'samples': len(requests), 'samplers': mode_reports, 'temperature': temperature, 'rtol': 1e-4, 'atol': 1e-5,
              'rng_corpus': 'full-width uint64 states, zero/extremes, and consecutive xorshift states',
              'passed': True, 'corpus': 'supplied observations' if supplied else 'random valid normalized observations'}
    (Path(directory) / 'parity-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def next_random(random: int) -> int:
    random = random or DEFAULT_RANDOM
    random ^= (random << 13) & UINT64_MAX
    random ^= random >> 7
    random ^= (random << 17) & UINT64_MAX
    return random


def parity_random_states(rng: np.random.Generator, count: int) -> list[int]:
    if count < 1:
        raise ValueError('Parity corpus must contain at least one observation')
    states = [int(value) for value in rng.integers(0, UINT64_MAX, size=count, endpoint=True, dtype=np.uint64)]
    special = [0, 1, UINT64_MAX, DEFAULT_RANDOM]
    chain = next_random(DEFAULT_RANDOM)
    for _ in range(8):
        special.append(chain)
        chain = next_random(chain)
    states[:min(count, len(special))] = special[:count]
    return states


def reference_action(logits, mask, random: int, mode: str, temperature: float = 1.0) -> tuple[int, int]:
    random = next_random(random)
    best = max((i for i in range(10) if mask[i]), key=lambda i: float(logits[i]))
    if mode == 'categorical':
        weights = [math.exp((float(logits[i]) - float(logits[best])) / temperature) if mask[i] else 0.0 for i in range(10)]
        threshold = (random >> 11) / 9007199254740992.0 * sum(weights)
        total = 0.0
        for i, weight in enumerate(weights):
            total += weight
            if mask[i] and total > threshold:
                return i, random
    if mode == 'categorical':
        return max(i for i in range(10) if mask[i]), random
    return best, random
