from pathlib import Path
import yaml
from .env import PROJECT_ROOT

PROFILES = ('trainee', 'expert', 'ace')


def read_yaml(path: str | Path, chain: tuple[Path, ...] = ()) -> dict:
    path = Path(path).expanduser().resolve()
    if path in chain or len(chain) >= 8:
        raise ValueError('Circular or excessively nested config inheritance')
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Config must be a YAML mapping')
    parent = value.pop('extends', None)
    base = read_yaml(path.parent / parent, (*chain, path)) if parent else {}
    return {**base, **value}


def load_config(path: str | None = None, profile: str | None = None, kind: str = 'train') -> dict:
    if profile and profile not in PROFILES:
        raise ValueError('Unknown difficulty profile')
    base = read_yaml(path or PROJECT_ROOT / 'configs' / (f'{profile}.yaml' if profile and kind == 'train' else f'{kind}.yaml'))
    if profile:
        choice = read_yaml(PROJECT_ROOT / 'configs' / f'{profile}.yaml')
        for key in ('difficulty', 'target_defender_win_rate', 'score_multiplier'):
            base[key] = choice[key]
        if kind == 'train' and path:
            base['seed'] = int(base.get('seed', 42)) + PROFILES.index(profile) * 1000003
        if kind != 'train':
            base['seed'] = choice['seed'] + (1000 if kind == 'distill' else 2000)
    return base
