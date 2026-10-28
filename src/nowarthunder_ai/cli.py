"""Explicit CLI entry points; importing modules or running doctor never trains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .config import PROFILES, load_config
from .devices import doctor, select_device, benchmark_devices
from .env import BatchedEnv, PROJECT_ROOT, build_tool, find_backend
from .provenance import (CheckpointSnapshot, calibration_campaigns, capture_checkpoint,
                         file_sha256, freeze_checkpoint)
from .sampling import deployment_sampler


def write_report(report: dict | list, output: str | None = None) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n')
        print(f'Report: {path.resolve()}', flush=True)
    else:
        print(text, flush=True)


def _factory(args):
    backend = find_backend(args.backend)
    executable = build_tool('trainenv', backend)
    return lambda configs: BatchedEnv(configs, executable=executable, workers=getattr(args, 'workers', None),
                                      enemy_model=getattr(args, 'enemy_model', None))


def checkpoint_profile(state: dict) -> str:
    profile = state.get('difficulty', state.get('config', {}).get('difficulty'))
    if profile not in PROFILES:
        raise ValueError('Checkpoint must declare a trainee, expert or ace difficulty profile')
    return profile


def evaluation_campaign(state: dict, seed: int, episodes: int) -> dict:
    """Describe actual seed intervals, including signed aliases and overlap.

    All scripted opponents replay the same interval. The saved checkpoint
    configuration is authoritative; a current YAML preset may have changed.
    """
    def interval(base: int, count: int) -> tuple[int, int]:
        if count < 1 or abs(base) + count > 2**63:
            raise ValueError('Evaluation episodes must be positive and seeds must fit signed int64')
        return -(abs(base) + count), -(abs(base) + 1)
    actual = interval(seed, episodes)
    campaigns = [{'source': campaign['purpose'], 'seed': campaign['seed'],
                  'seed_interval': list(interval(campaign['seed'], campaign['episodes']))}
                 for campaign in calibration_campaigns(state)]
    # A distilled student inherits teacher selection bias. Conservatively
    # include embedded teachers, and count overlapping seed ranges only once.
    intersections = sorted((max(actual[0], c['seed_interval'][0]), min(actual[1], c['seed_interval'][1]))
                           for c in campaigns if max(actual[0], c['seed_interval'][0]) <= min(actual[1], c['seed_interval'][1]))
    merged = []
    for low, high in intersections:
        if merged and low <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    overlap = sum(high - low + 1 for low, high in merged)
    exact = any(list(actual) == c['seed_interval'] for c in campaigns)
    purpose = 'calibration_replay' if exact else 'calibration_overlap' if overlap else 'final_test'
    return {'seed': seed, 'purpose': purpose, 'seed_interval': list(actual),
            'calibration_seed': campaigns[0]['seed'], 'calibration_seed_interval': campaigns[0]['seed_interval'],
            'calibration_campaigns': campaigns,
            'overlapping_seeds_per_opponent': overlap,
            'reference': 'Checkpoint, teacher, transfer ancestry and consumed campaigns; each scripted opponent shares the intervals'}


def difficulty_run_plan(checkpoint: str, output: str, profiles=PROFILES, iterations=None, *,
                        snapshot: CheckpointSnapshot | None = None) -> list[dict]:
    """Preflight all profiles before starting any long optimizer run."""
    import torch
    from .checkpoint import actor_from_checkpoint
    from .trainer import resolved_config
    snapshot = snapshot or capture_checkpoint(checkpoint)
    state = snapshot.state
    if (not state.get('trained') or state.get('enemy_optimizer_updates', 0) <= 0
            or str(state.get('kind', '')).startswith('smoke')):
        raise ValueError('Difficulty tuning requires a trained, non-smoke enemy checkpoint')
    spec = state.get('model_spec', {})
    if not isinstance(spec, dict) or not isinstance(spec.get('enemy_hidden'), (list, tuple)):
        raise ValueError('Difficulty source is missing its actor model specification')
    if tuple(spec['enemy_hidden']) not in {(64, 32), (32, 16)}:
        raise ValueError('Distill the source to a tiny actor before difficulty tuning')
    if state.get('observation_version') != 2 or state.get('normalization_version') != 2:
        raise ValueError('Difficulty source requires observation_version=2 and normalization_version=2')
    if spec.get('observation_size') != 320 or spec.get('action_size') != 10:
        raise ValueError('Difficulty source requires 320 observation features and 10 actions')
    if 'defender_actor' in state and state.get('defender_observation_version') != 1:
        raise ValueError('Difficulty source has an incompatible defender observation version')
    actor = actor_from_checkpoint(state, 'enemy', 'cpu')
    if any(not bool(torch.isfinite(parameter).all()) for parameter in actor.parameters()):
        raise ValueError('Difficulty source actor has non-finite weights')
    # Learned defender weights are copied by train as well; validate them before
    # any profile starts, rather than discovering a malformed source later.
    if 'defender_actor' in state:
        defender = actor_from_checkpoint(state, 'defender', 'cpu')
        if any(not bool(torch.isfinite(parameter).all()) for parameter in defender.parameters()):
            raise ValueError('Difficulty source defender has non-finite weights')
    source_profile = checkpoint_profile(state)
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in PROFILES for p in profiles):
        raise ValueError('Choose unique trainee/expert/ace profiles')
    plan = []
    root = Path(output)
    if root.exists() and not root.is_dir():
        raise FileExistsError(f'Output must be a directory: {root}')
    for profile in profiles:
        config_path = PROJECT_ROOT / 'configs' / 'difficulty' / f'{profile}.yaml'
        config = load_config(str(config_path))
        if iterations is not None:
            config['iterations'] = iterations
        config = resolved_config(config)
        destination = Path(output) / profile
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f'Output is not empty: {destination}; resume this profile with train instead')
        plan.append({'profile': profile, 'source_profile': source_profile, 'config_path': str(config_path),
                     'checkpoint': str(snapshot.path), 'source_sha256': snapshot.sha256,
                     'frozen_checkpoint': str(root / f'source-{snapshot.sha256}.pt'),
                     'output': str(destination), 'config': config})
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='NO WarThunder offline training. Production game integration is separate.')
    commands = parser.add_subparsers(dest='command')
    def execution(command, help_text):
        p = commands.add_parser(command, help=help_text)
        p.add_argument('--backend', help='Go backend checkout; also NO_WARTHUNDER_BACKEND')
        p.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
        p.add_argument('--workers', type=int, help='Go simulation CPU workers (default min(12, CPU count))')
        return p
    p = execution('doctor', 'Check dependencies, accelerator and Go bridge without training')
    p.add_argument('--skip-go', action='store_true')
    for name in ('train','train-all'):
        p = execution(name, 'Start training only when explicitly invoked')
        p.add_argument('--config')
        p.add_argument('--profile', choices=PROFILES)
        p.add_argument('--output', required=True)
        p.add_argument('--iterations', type=int)
        if name == 'train':
            p.add_argument('--resume')
            p.add_argument('--initial-enemy-checkpoint', help='Initialize an independent fine-tuning run from a distilled student')
            p.add_argument('--allow-profile-transfer', action='store_true', help='Explicitly retarget an initial checkpoint in a new run; never for resume')
    p = execution('tune-difficulties', 'Create independent tiny v2 difficulty runs from one trained student')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--profiles', choices=PROFILES, nargs='+', default=list(PROFILES))
    p.add_argument('--iterations', type=int)
    p.add_argument('--plan-only', action='store_true', help='Validate inputs and print the run plan without training')
    p = execution('distill', 'Train a tiny student on its own visited states')
    p.add_argument('--teacher', required=True); p.add_argument('--output', required=True)
    p.add_argument('--profile', choices=PROFILES); p.add_argument('--config')
    p.add_argument('--iterations', type=int); p.add_argument('--resume')
    p = execution('evaluate', 'Evaluate fixed held-out seeds; never update weights')
    p.add_argument('--checkpoint', required=True); p.add_argument('--profile', choices=PROFILES)
    p.add_argument('--episodes', type=int, default=100, help='Episodes per scripted opponent')
    p.add_argument('--seed', type=int, help='Default: 4000000 for v2/v3, 2000000 for legacy checkpoints')
    p.add_argument('--calibration-version', type=int, choices=(1, 2, 3), help='Default: saved checkpoint contract')
    p.add_argument('--native-go', action='store_true', help='Run the exact Go policy; automatic for categorical checkpoint contracts')
    p.add_argument('--max-steps', type=int, default=9000); p.add_argument('--output')
    p.add_argument('--levels', type=int, nargs='+', default=list(range(1,11)))
    p = commands.add_parser('export', help='Export FP32 tiny actor for later Go integration')
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--policy-version', required=True); p.add_argument('--allow-untrained', action='store_true')
    p.add_argument('--sampler', choices=('argmax','categorical'), help='Must match saved deployment_sampler; omitted uses the saved contract')
    p.add_argument('--temperature', type=float, help='Must match saved deployment_sampler; omitted uses the saved contract')
    p.add_argument('--evaluation', help='Passing independent evaluation JSON for this exact checkpoint (required for trained v2/v3 export)')
    p = commands.add_parser('qualify', help='Explicitly qualify a frozen actor for a measured difficulty; requires a new release evaluation')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--evaluation', required=True, help='Fresh calibration report for the exact source checkpoint')
    p.add_argument('--profile', choices=PROFILES, required=True)
    p.add_argument('--output', required=True, help='New checkpoint path; existing files are never replaced')
    for name in ('verify','benchmark-policy'):
        p = commands.add_parser(name)
        p.add_argument('--model', required=True); p.add_argument('--backend')
    commands.add_parser('benchmark-device', help='Compare CPU/GPU minibatch forward/backward cost')
    p = commands.add_parser('coreml', help='Optional frozen CPU/ANE inference export and comparison')
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=160); p.add_argument('--benchmark', action='store_true')
    args = parser.parse_args(argv if argv is not None else sys.argv[1:] or ['doctor'])
    try:
        if args.command == 'doctor':
            report = doctor(args.device)
            if not args.skip_go:
                factory = _factory(args)
                with factory([{'level_id':1,'seed':42,'max_steps':5,'defender_bot':'mixed'}]) as env:
                    report['go_bridge'] = env.metadata
                    report['backend_path'] = str(find_backend(args.backend))
            write_report(report)
        elif args.command == 'tune-difficulties':
            snapshot = capture_checkpoint(args.checkpoint)
            plan = difficulty_run_plan(args.checkpoint, args.output, args.profiles, args.iterations, snapshot=snapshot)
            if args.plan_only:
                write_report({'starts_training': False, 'runs': plan})
            else:
                import torch
                from .trainer import train
                torch.set_num_threads(min(4, torch.get_num_threads()))
                frozen = freeze_checkpoint(snapshot, args.output)
                factory = _factory(args)
                results = {}
                for item in plan:
                    if file_sha256(frozen) != snapshot.sha256:
                        raise ValueError(f'Frozen source changed before difficulty run: {frozen}')
                    config = item['config']
                    device = select_device(args.device if args.device != 'auto' else config.get('device', 'auto'))
                    print(f"difficulty v2: {item['profile']}, device={device}, output={item['output']}", flush=True)
                    results[item['profile']] = train(config, factory, device, item['output'],
                                                    initial_enemy_checkpoint=frozen, allow_profile_transfer=True)
                write_report({'runs': results, 'all_calibrated': all(r.get('selection', {}).get('calibrated', False)
                                                                    for r in results.values())})
        elif args.command in ('train','train-all','distill'):
            import torch
            torch.set_num_threads(min(4, torch.get_num_threads()))
            factory = _factory(args)
            inferred_profile = args.profile
            if args.command == 'distill' and inferred_profile is None:
                from .checkpoint import load_checkpoint
                # A resumed student embeds its original teacher. The old
                # external teacher path need not still exist.
                source = load_checkpoint(args.resume or args.teacher)
                inferred_profile = checkpoint_profile(source)
            profiles = PROFILES if args.command == 'train-all' else [inferred_profile]
            results = {}
            for profile in profiles:
                kind = 'distill' if args.command == 'distill' else 'train'
                config = load_config(args.config, profile, kind)
                if kind == 'distill' and args.config is None:
                    # Defaults control the optimizer/cadence, not the teacher's
                    # feature budget or calibration provenance.
                    inherited = {'difficulty_control', 'progress_recovery', 'deployment_sampler', 'calibration_version', 'consumed_campaigns',
                                 'eval_seed', 'eval_episodes', 'eval_max_steps', 'eval_opponents',
                                 'min_evaluation_episodes', 'max_truncation_rate', 'max_stagnation_rate'}
                    config = {k: v for k, v in config.items() if k not in inherited and not k.startswith(('v2_', 'v3_'))}
                if args.iterations is not None:
                    config['iterations'] = args.iterations
                device = select_device(args.device if args.device != 'auto' else config.get('device','auto'))
                output = str(Path(args.output) / profile) if args.command == 'train-all' else args.output
                print(f"{kind}: difficulty={config.get('difficulty', 'expert')}, device={device}, output={output}", flush=True)
                if kind == 'distill':
                    from .distillation import distill
                    result = distill(config,factory,device,output,args.teacher,resume=args.resume)
                else:
                    from .trainer import train
                    result = train(config,factory,device,output,resume=getattr(args,'resume',None),
                                   initial_enemy_checkpoint=getattr(args,'initial_enemy_checkpoint',None),
                                   allow_profile_transfer=getattr(args,'allow_profile_transfer', False))
                results[config.get('difficulty', 'expert')] = result
            write_report(results)
        elif args.command == 'evaluate':
            from .checkpoint import actor_from_checkpoint
            from .evaluation import evaluate
            snapshot = capture_checkpoint(args.checkpoint)
            state = snapshot.state
            profile = checkpoint_profile(state)
            if args.profile is not None and args.profile != profile:
                raise ValueError('Evaluation profile must match the checkpoint difficulty')
            config = {**load_config(profile=profile), **state.get('config', {})}
            if args.calibration_version is not None:
                config['calibration_version'] = args.calibration_version
            seed = args.seed if args.seed is not None else (4000000 if config.get('calibration_version') in (2, 3) else 2000000)
            campaign = evaluation_campaign(state, seed, args.episodes)
            config.update(episodes=args.episodes, max_steps=args.max_steps, level_ids=args.levels, eval_seed=seed)
            device = select_device(args.device)
            sampler = deployment_sampler(config.get('deployment_sampler'))
            if args.native_go or sampler['mode'] == 'categorical':
                from .export import evaluation_artifact
                temporary_root = PROJECT_ROOT / 'work' / 'native-evaluation'
                temporary_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='candidate-', dir=temporary_root) as directory:
                    args.enemy_model = str(Path(directory) / 'artifact')
                    native = evaluation_artifact(snapshot, args.enemy_model)
                    report = evaluate(None, _factory(args), device, config, native_policy=native)
            else:
                actor = actor_from_checkpoint(state, 'enemy', device)
                report = evaluate(actor, _factory(args), device, config)
            report['difficulty'] = profile
            report['campaign'] = campaign
            report['checkpoint_identity'] = {'sha256': snapshot.sha256,
                                             'filename': snapshot.path.name,
                                             'model_spec': state['model_spec']}
            report.setdefault('policy_sampler', {key: sampler[key] for key in ('mode', 'temperature')})
            report['evaluation_config'] = config
            write_report(report,args.output)
        elif args.command == 'qualify':
            from .qualification import qualify_checkpoint
            write_report(qualify_checkpoint(args.checkpoint, args.evaluation, args.output, args.profile))
        elif args.command == 'export':
            from .export import export_checkpoint
            write_report(export_checkpoint(args.checkpoint,args.output,policy_version=args.policy_version,
                                           sampler=args.sampler,temperature=args.temperature,allow_untrained=args.allow_untrained,
                                           evaluation=args.evaluation))
        elif args.command == 'verify':
            from .export import verify_artifact
            write_report(verify_artifact(args.model,args.backend))
        elif args.command == 'benchmark-policy':
            result = subprocess.run([str(build_tool('policycheck',args.backend)),'-model',str(Path(args.model).resolve()),
                                     '-benchmark','-iterations','100000'],check=True,text=True,capture_output=True,timeout=60)
            print(result.stdout, end='')
        elif args.command == 'benchmark-device':
            write_report(benchmark_devices())
        elif args.command == 'coreml':
            from .coreml import export_coreml
            write_report(export_coreml(args.checkpoint,args.output,batch_size=args.batch_size,benchmark=args.benchmark))
        else:
            parser.print_help()
        return 0
    except KeyboardInterrupt:
        print('\nInterrupted. Completed checkpoints remain in the run directory; use --resume to continue.',file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, TimeoutError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}',file=sys.stderr)
        return 1
