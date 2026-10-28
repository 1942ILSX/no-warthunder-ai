"""Batched, process-isolated Go simulation. No HTTP, WebSockets, or real-time sleeps.

Each response is a JSON descriptor followed by contiguous little-endian arrays.
A training decision barrier inside Game.Step preserves the production encoder,
perception queue, and action timing; Python never reimplements game physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
import hashlib
import json
import os
import select
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_FRAME_BYTES = 128 * 1024 * 1024


def enemy_model_identity(directory: str | Path) -> dict:
    """Read the exact local artifact identity the native Go bridge must load."""
    root = Path(directory).expanduser().resolve()
    metadata = json.loads((root / 'metadata.json').read_bytes())
    if not isinstance(metadata, dict) or metadata.get('weightsFile') != 'weights.bin':
        raise ValueError('Native evaluation requires weights.bin')
    digest = hashlib.sha256((root / 'weights.bin').read_bytes()).hexdigest()
    if metadata.get('weightsSHA256') != digest:
        raise ValueError('Native evaluation artifact checksum mismatch')
    return {'implementation': 'go-policymlp', 'metadata': metadata}


def validate_enemy_policy_header(actual, expected) -> None:
    if actual != expected:
        raise RuntimeError('Go bridge native enemy policy identity differs from the requested artifact')


def find_backend(value: str | Path | None = None) -> Path:
    candidate = value or os.getenv('NO_WARTHUNDER_BACKEND')
    candidates = [Path(candidate).expanduser()] if candidate else [
        PROJECT_ROOT.parent / 'no-warthunder-backend',
        PROJECT_ROOT.parent / 'BattleCityReforge',
        PROJECT_ROOT.parent.parent / 'GolandProjects' / 'BattleCityReforge',
    ]
    for path in candidates:
        if (path / 'go.mod').is_file() and (path / 'internal/game').is_dir():
            return path.resolve()
    raise FileNotFoundError('Go backend not found. Pass --backend PATH or set NO_WARTHUNDER_BACKEND.')


def go_executable() -> str:
    path = shutil.which('go')
    if path:
        return path
    if Path('/usr/local/go/bin/go').is_file():
        return '/usr/local/go/bin/go'
    raise FileNotFoundError('Go is required to build the simulation bridge. Add Go to PATH.')


def build_tool(name: str, backend: str | Path | None = None) -> Path:
    if name not in {'trainenv', 'policycheck'}:
        raise ValueError('Unknown Go tool')
    root = find_backend(backend)
    output = PROJECT_ROOT / 'build' / name
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(root).encode())
    for path in sorted([*root.glob('go.*'), *root.glob('internal/**/*.go'), *root.glob(f'cmd/{name}/**/*.go')]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    identity = digest.hexdigest()
    stamp = output.with_suffix('.sha256')
    if output.is_file() and stamp.is_file() and stamp.read_text() == identity:
        return output
    result = subprocess.run([go_executable(), 'build', '-trimpath', '-o', str(output), f'./cmd/{name}'], cwd=root, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'Go {name} build failed:\n{result.stderr[-8000:]}')
    stamp.write_text(identity)
    return output


@dataclass
class Frame:
    enemy_obs: np.ndarray
    enemy_mask: np.ndarray
    enemy_active: np.ndarray
    enemy_ids: np.ndarray
    defender_obs: np.ndarray
    defender_mask: np.ndarray
    defender_active: np.ndarray
    critic_obs: np.ndarray
    enemy_reward: np.ndarray
    defender_reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict[str, Any]]


class BatchedEnv:
    def __init__(self, configs: list[dict[str, Any]], backend_path: str | Path | None = None,
                 executable: str | Path | None = None, timeout: float = 60.0,
                 workers: int | None = None, enemy_model: str | Path | None = None):
        if not 1 <= len(configs) <= 256:
            raise ValueError('num_envs must be between 1 and 256')
        if workers is not None and (type(workers) is not int or workers < 1):
            raise ValueError('workers must be a positive integer')
        self.configs = [dict(config) for config in configs]
        self.num_envs = len(configs)
        self.timeout = timeout
        self._buffer = bytearray()
        self._closed = False
        self._errors: deque[str] = deque(maxlen=30)
        env = os.environ.copy()
        env['GOMAXPROCS'] = str(workers or min(12, os.cpu_count() or 1))
        binary = Path(executable) if executable else build_tool('trainenv', backend_path)
        self._binary, self._workers = binary, workers
        self.enemy_policy = enemy_model_identity(enemy_model) if enemy_model is not None else None
        command = [str(binary)]
        if enemy_model is not None:
            command.extend(['-enemy-model', str(Path(enemy_model).expanduser().resolve())])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, env=env)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        try:
            header, arrays = self._receive()
            self.metadata = {k: v for k, v in header.items() if k not in {'arrays', 'infos'}}
            required = {'protocol': 1, 'observation_version': 2, 'enemy_slots': 5,
                        'observation_size': 320, 'action_count': 10, 'normalization_version': 2, 'defender_observation_version': 1}
            if arrays or any(header.get(k) != v for k, v in required.items()):
                raise RuntimeError(f'Incompatible Go bridge metadata: {self.metadata}')
            validate_enemy_policy_header(header.get('enemy_policy'), self.enemy_policy)
            self.critic_size = int(header['critic_size'])
            self.frame = self.reset()
        except BaseException:
            self.close()
            raise

    def _read_stderr(self) -> None:
        assert self.process.stderr
        for line in iter(self.process.stderr.readline, b''):
            self._errors.append(line.decode(errors='replace').rstrip())

    def _fill(self, deadline: float) -> None:
        assert self.process.stdout
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
            raise TimeoutError('Go simulation bridge timed out. ' + '\n'.join(self._errors))
        chunk = os.read(self.process.stdout.fileno(), 65536)
        if not chunk:
            raise RuntimeError('Go simulation bridge closed unexpectedly. ' + '\n'.join(self._errors))
        self._buffer.extend(chunk)

    def _line(self, deadline: float) -> bytes:
        while b'\n' not in self._buffer:
            if len(self._buffer) > 4 * 1024 * 1024:
                raise RuntimeError('Oversized Go protocol header')
            self._fill(deadline)
        index = self._buffer.index(b'\n')
        value = bytes(self._buffer[:index])
        del self._buffer[:index + 1]
        return value

    def _exact(self, length: int, deadline: float) -> bytes:
        while len(self._buffer) < length:
            self._fill(deadline)
        value = bytes(self._buffer[:length])
        del self._buffer[:length]
        return value

    def _receive(self) -> tuple[dict, dict[str, np.ndarray]]:
        deadline = time.monotonic() + self.timeout
        header = json.loads(self._line(deadline))
        if header.get('error'):
            raise RuntimeError('Go simulation: ' + str(header['error']))
        arrays: dict[str, np.ndarray] = {}
        total = 0
        for descriptor in header.get('arrays', []):
            name, shape = descriptor['name'], descriptor['shape']
            dtype = np.dtype(descriptor['dtype'])
            if dtype.str not in {'<f4', '<i4', '|u1'} or any(type(n) is not int or n < 0 for n in shape):
                raise RuntimeError('Unsupported array descriptor from Go')
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            total += size
            if size != descriptor['nbytes'] or size < 0 or total > MAX_FRAME_BYTES or name in arrays:
                raise RuntimeError('Invalid or oversized Go binary frame')
            arrays[name] = np.frombuffer(self._exact(size, deadline), dtype=dtype).reshape(shape).copy()
        return header, arrays

    def _request(self, command: dict) -> Frame:
        if self._closed:
            raise RuntimeError('Environment is closed')
        assert self.process.stdin
        self.process.stdin.write(json.dumps(command, separators=(',', ':'), allow_nan=False).encode() + b'\n')
        self.process.stdin.flush()
        header, arrays = self._receive()
        n = self.num_envs
        shapes = {'enemy_obs': (n, 5, 320), 'enemy_mask': (n, 5, 10), 'enemy_active': (n, 5),
                  'enemy_ids': (n, 5), 'defender_obs': (n, 320), 'defender_mask': (n, 10),
                  'defender_active': (n,), 'critic_obs': (n, self.critic_size),
                  'enemy_reward': (n,), 'defender_reward': (n,), 'terminated': (n,), 'truncated': (n,)}
        if set(arrays) != set(shapes):
            raise RuntimeError('Incomplete Go frame: ' + str(set(arrays) ^ set(shapes)))
        for name, shape in shapes.items():
            if arrays[name].shape != shape:
                raise RuntimeError(f'Wrong shape for {name}: {arrays[name].shape}, expected {shape}')
            if arrays[name].dtype.kind == 'f' and not np.isfinite(arrays[name]).all():
                raise RuntimeError(f'Non-finite values in {name}')
        for name in ('enemy_mask', 'enemy_active', 'defender_mask', 'defender_active', 'terminated', 'truncated'):
            arrays[name] = arrays[name].astype(bool)
        infos = header.get('infos', [])
        if len(infos) != n:
            raise RuntimeError('Wrong number of info records')
        self.frame = Frame(**arrays, infos=infos)
        return self.frame

    def reset(self, configs: list[dict] | None = None) -> Frame:
        proposed = self.configs if configs is None else [dict(config) for config in configs]
        if len(proposed) != self.num_envs:
            raise ValueError('Reset cannot change batch size')
        frame = self._request({'cmd': 'reset', 'configs': proposed})
        # The Go bridge rejects invalid resets atomically. Keep our matching
        # configuration until it acknowledges the new worlds successfully.
        self.configs = proposed
        return frame

    def step(self, enemy_actions: np.ndarray, defender_actions: np.ndarray) -> Frame:
        enemy, defender = np.asarray(enemy_actions), np.asarray(defender_actions)
        for values, shape in ((enemy, (self.num_envs, 5)), (defender, (self.num_envs,))):
            if values.shape != shape or values.dtype.kind not in 'iu' or np.any((values < -1) | (values > 9)):
                raise ValueError('Actions must be integer -1 or 0..9 with the declared batch shape')
        if self.enemy_policy is not None and np.any(enemy != -1):
            raise ValueError('Native enemy policy evaluation cannot override Go actions')
        return self._request({'cmd': 'step', 'enemy_actions': enemy.tolist(), 'defender_actions': defender.tolist()})

    def reset_indices(self, indices: list[int] | np.ndarray, configs: list[dict]) -> Frame:
        ids = [int(i) for i in indices]
        if len(set(ids)) != len(ids) or len(ids) != len(configs) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError('Invalid reset indices/configurations')
        proposed = [dict(config) for config in configs]
        frame = self._request({'cmd': 'reset_indices', 'indices': ids, 'configs': proposed})
        for i, config in zip(ids, proposed):
            self.configs[i] = config
        return frame

    def factory_with_enemy_model(self, directory: str | Path):
        """Create isolated native-evaluation worlds with this bridge's settings."""
        model = Path(directory).expanduser().resolve()
        return lambda configs: BatchedEnv(configs, executable=self._binary, timeout=self.timeout,
                                          workers=self._workers, enemy_model=model)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            try:
                assert self.process.stdin
                self.process.stdin.write(b'{"cmd":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe:
                pipe.close()

    def __enter__(self) -> 'BatchedEnv':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
