"""Explicit accelerator selection; ANE is not a torch autograd device."""
import platform
import time
from importlib.metadata import version

import torch


def select_device(requested: str = 'auto') -> torch.device:
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this Python/PyTorch environment; select cpu instead.')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select auto or cpu.')
    if requested not in {'cpu', 'mps', 'cuda'}:
        raise ValueError('device must be auto, cpu, mps, or cuda. ANE is available only through optional Core ML inference.')
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def doctor(requested: str = 'auto') -> dict:
    device = select_device(requested)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = torch.nn.Sequential(torch.nn.Linear(320, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)).to(device)
    x = torch.randn(128, 320, device=device)
    loss = model(x).square().mean()
    loss.backward()
    synchronize(device)
    return {'python_architecture': platform.machine(), 'torch': version('torch'),
            'torchrl': version('torchrl'), 'tensordict': version('tensordict'),
            'selected_device': str(device), 'mps_available': torch.backends.mps.is_available(),
            'cuda_available': torch.cuda.is_available(), 'forward_backward': 'passed',
            'ane_training': False,
            'ane_note': 'TorchRL uses PyTorch MPS for Apple GPU training. Optional Core ML CPU_AND_NE is inference-only and does not guarantee ANE placement.'}


def benchmark_devices(batch_sizes=(1, 160, 1024), iterations: int = 30) -> list[dict]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = []
    available = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []) + (['cuda'] if torch.cuda.is_available() else [])
    for name in available:
        device = torch.device(name)
        model = torch.nn.Sequential(torch.nn.Linear(320, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)).to(device)
        for batch in batch_sizes:
            x = torch.randn(batch, 320, device=device)
            for _ in range(3):
                model.zero_grad(set_to_none=True); model(x).square().mean().backward()
            synchronize(device); started = time.perf_counter()
            for _ in range(iterations):
                model.zero_grad(set_to_none=True); model(x).square().mean().backward()
            synchronize(device)
            output.append({'device': name, 'batch_size': batch, 'forward_backward_ms': (time.perf_counter() - started) * 1000 / iterations})
    return output
