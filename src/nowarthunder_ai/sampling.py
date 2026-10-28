"""Explicit deployment sampler contract shared by evaluation and export."""
from collections.abc import Mapping
import math


def deployment_sampler(value: Mapping | None = None) -> dict:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError('deployment_sampler must be a mapping')
    value = dict(value or {})
    if set(value) - {'mode', 'temperature', 'rng'}:
        raise ValueError('Unknown deployment_sampler fields')
    result = {'mode': 'argmax', 'temperature': 1., 'rng': 'xorshift64-v1', **value}
    if result['mode'] not in ('argmax', 'categorical') or result['rng'] != 'xorshift64-v1':
        raise ValueError('Unsupported deployment sampler mode or RNG')
    temperature = result['temperature']
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Sampler temperature must be finite and positive')
    result['temperature'] = float(temperature)
    return result
