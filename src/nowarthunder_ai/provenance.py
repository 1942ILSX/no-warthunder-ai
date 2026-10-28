"""Checkpoint snapshots and calibration lineage for offline workflows."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import torch


@dataclass(frozen=True)
class CheckpointSnapshot:
    path: Path
    data: bytes
    sha256: str
    state: dict


def capture_checkpoint(path: str | Path) -> CheckpointSnapshot:
    """Load and hash exactly the same bytes, even if a live path is replaced.

    Checkpoints use pickle for optimizer/RNG state. Read only trusted files.
    """
    source = Path(path).expanduser().resolve()
    data = source.read_bytes()
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or state.get('format_version') != 1:
        raise ValueError('Unsupported training checkpoint format')
    return CheckpointSnapshot(source, data, hashlib.sha256(data).hexdigest(), state)


def file_sha256(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def freeze_checkpoint(snapshot: CheckpointSnapshot, output: str | Path) -> Path:
    """Atomically create a hash-named source; never replace different bytes."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f'source-{snapshot.sha256}.pt'
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != snapshot.sha256:
            raise ValueError(f'Frozen source differs from its expected SHA256: {destination}')
        return destination
    descriptor, temporary = tempfile.mkstemp(prefix='.source-', suffix='.tmp', dir=root)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(snapshot.data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Linking makes the complete file visible atomically and refuses
            # to overwrite a destination created by another invocation.
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or file_sha256(destination) != snapshot.sha256:
                raise ValueError(f'Frozen source differs from its expected SHA256: {destination}')
    finally:
        os.unlink(temporary)
    return destination


def calibration_campaigns(state: Mapping) -> list[dict]:
    """Flatten teacher, transfer and consumed-campaign ancestry without weights.

    Transfer checkpoints store this output in
    initial_source.calibration_ancestry so repeated transfers retain lineage.
    """
    campaigns: list[dict] = []
    unique: set[tuple[int, int]] = set()
    visited: set[int] = set()

    def add(seed, episodes, purpose: str) -> None:
        seed, episodes = int(seed), int(episodes)
        if episodes < 1 or abs(seed) + episodes > 2**63:
            raise ValueError('Calibration campaign must have positive episodes and signed int64 seeds')
        key = abs(seed), episodes
        if key not in unique:
            unique.add(key)
            campaigns.append({'seed': seed, 'episodes': episodes, 'purpose': str(purpose)})

    def consume(items, purpose: str) -> None:
        for item in items or []:
            add(item['seed'], item['episodes'], item.get('purpose', purpose))

    def config(value: Mapping, purpose: str) -> None:
        add(value.get('eval_seed', 1000000), value.get('eval_episodes', 10), purpose)
        consume(value.get('consumed_campaigns'), 'previously_used_campaign')

    def visit(value: Mapping, purpose: str) -> None:
        if not isinstance(value, Mapping) or id(value) in visited:
            return
        visited.add(id(value))
        config(value.get('config', {}), purpose)
        consume(value.get('calibration_ancestry'), 'earlier_transfer')
        initial = value.get('initial_source')
        if isinstance(initial, Mapping):
            if isinstance(initial.get('calibration_config'), Mapping):
                config(initial['calibration_config'], 'initial_source')
            consume(initial.get('calibration_ancestry'), 'earlier_transfer')
        teacher = value.get('teacher_checkpoint')
        if isinstance(teacher, Mapping):
            visit(teacher, 'embedded_teacher')

    visit(state, 'checkpoint')
    return campaigns
