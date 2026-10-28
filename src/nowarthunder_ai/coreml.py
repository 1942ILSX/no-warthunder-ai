"""Optional frozen-model inference experiment. Core ML does not train TorchRL.

CPU_AND_NE permits Neural Engine placement; Core ML may still choose CPU for this
small network. This module never claims that an ANE device was actually used.
"""
from __future__ import annotations
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch


def export_coreml(checkpoint: str, output: str, *, batch_size: int = 160, benchmark: bool = False) -> dict:
    if platform.system() != 'Darwin':
        raise RuntimeError('Core ML prediction requires macOS. Use MPS/CUDA/CPU training normally.')
    try:
        import coremltools as ct
    except ImportError as error:
        raise RuntimeError('Install the optional Core ML extra: pip install -e ".[coreml]"') from error
    from .checkpoint import actor_from_checkpoint
    from .provenance import capture_checkpoint
    from .sampling import deployment_sampler
    if not 1 <= batch_size <= 4096:
        raise ValueError('Core ML batch_size must be 1..4096')
    if Path(output).exists():
        raise FileExistsError(output)
    snapshot = capture_checkpoint(checkpoint)
    sampler = deployment_sampler(snapshot.state.get('config', {}).get('deployment_sampler'))
    actor = actor_from_checkpoint(snapshot.state, role='enemy', device='cpu').eval()
    example = torch.zeros(batch_size, 320)
    traced = torch.jit.trace(actor, example)
    model = ct.convert(traced, convert_to='mlprogram', inputs=[ct.TensorType(name='observation', shape=example.shape)],
                       outputs=[ct.TensorType(name='logits')], minimum_deployment_target=ct.target.macOS13,
                       compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16)
    model.save(output)
    report = {'path': str(Path(output).resolve()), 'batch_size': batch_size, 'compute_units': 'CPU_AND_NE',
              'ane_execution_confirmed': False, 'training_acceleration': False,
              'checkpoint_sha256': snapshot.sha256, 'deployment_sampler': sampler,
              'sampler_applied': False, 'release_certified': False,
              'note': 'Frozen inference only. Core ML chooses CPU/ANE placement; benchmark before using it.'}
    if benchmark:
        data = np.random.default_rng(42).uniform(-1, 1, (batch_size, 320)).astype(np.float32)
        with torch.no_grad():
            expected = actor(torch.from_numpy(data)).numpy()
        for _ in range(3):
            result = model.predict({'observation': data})['logits']
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=2e-3)
        start = time.perf_counter()
        for _ in range(20):
            model.predict({'observation': data})
        report['coreml_batch_ms'] = (time.perf_counter() - start) * 1000 / 20
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                actor(torch.from_numpy(data))
        report['torch_cpu_batch_ms'] = (time.perf_counter() - start) * 1000 / 20
        report['max_logit_error'] = float(np.max(np.abs(result - expected)))
    Path(output + '.report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
