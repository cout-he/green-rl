"""Read-only audit of completed baseline artifacts; deserialize trusted local weights only.

The API never writes. The CLI writes a new analysis report after every check passes.
This checks retained evidence, not replay equivalence or paper performance claims.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ETA = {'lublin_256': .002, 'Cirne': .008, 'Jann': .003}


class VerificationError(ValueError):
    """Retained artifacts do not support a complete baseline audit."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained_file(base, path):
    path = Path(path)
    resolved = (path if path.is_absolute() else base / path).resolve()
    require(resolved.is_relative_to(base.resolve()), f'Path escapes artifact root: {path}')
    require(resolved.is_file(), f'Missing file: {resolved}')
    return resolved


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def option(command, flag, default=None):
    require(command.count(flag) <= 1, f'Duplicate command option: {flag}')
    if flag not in command:
        return default
    index = command.index(flag)
    require(index + 1 < len(command), f'Missing command option value: {flag}')
    return command[index + 1]


def verify_rng(rng, cuda_required):
    require(isinstance(rng, dict) and {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'environment'} <= rng.keys(), 'Incomplete RNG payload')
    # Restore into independent local generators, leaving global experiment RNG untouched.
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    cuda = rng['torch_cuda']
    require(isinstance(cuda, list) and (not cuda_required or len(cuda) > 0), 'Missing CUDA RNG state')
    require(all(isinstance(x, torch.Tensor) and x.dtype == torch.uint8 and x.ndim == 1 and x.numel() > 0 for x in cuda), 'Malformed CUDA RNG state')
    env = rng['environment']
    require(isinstance(env, dict) and {'kind', 'state'} <= env.keys(), 'Missing environment RNG state')
    if env['kind'] == 'RandomState':
        np.random.RandomState().set_state(env['state'])
    elif env['kind'] == 'Generator':
        bit_generator = env['state'].get('bit_generator')
        require(bit_generator in ('PCG64', 'PCG64DXSM', 'MT19937', 'Philox', 'SFC64'), 'Unknown environment bit generator')
        generator = getattr(np.random, bit_generator)()
        generator.state = env['state']
    else:
        raise VerificationError('Unknown environment RNG kind')


def verify_optimizer(optimizer, weights, label):
    require(isinstance(optimizer, dict) and {'param_groups', 'state'} <= optimizer.keys(), f'Incomplete {label}')
    groups, state = optimizer['param_groups'], optimizer['state']
    require(isinstance(groups, list) and bool(groups), f'Missing {label} parameter groups')
    parameters = [key for group in groups for key in group['params']]
    require(len(parameters) == len(set(parameters)) == len(weights) and set(parameters) == set(state), f'Incomplete {label} parameter state')
    for parameter, weight in zip(parameters, weights.values()):
        item = state[parameter]
        require({'step', 'exp_avg', 'exp_avg_sq'} <= item.keys(), f'Incomplete {label} Adam moments')
        step = float(item['step'])
        require(math.isfinite(step) and step > 0, f'Invalid {label} Adam step')
        for key in ('exp_avg', 'exp_avg_sq'):
            moment = item[key]
            require(isinstance(moment, torch.Tensor) and moment.shape == weight.shape and bool(torch.isfinite(moment).all()), f'Invalid {label} {key}')
        require(bool((item['exp_avg_sq'] >= 0).all()), f'Negative {label} second moment')


def verify_training(run, workload, seed, folder, step, epochs, trajectories):
    algorithm = 'MARL' if folder == 'MARL' else 'PPO'
    target = run / 'baseline' / workload / f'seed_{seed}' / 'models' / workload / folder
    csv_path = contained_file(run, target / f'{folder}_{workload}.csv')
    with csv_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    require(len(rows) == epochs and all(len(row) == 3 for row in rows), f'{target}: wrong CSV epoch count or shape')
    values = [[float(x) for x in row] for row in rows]
    require(all(all(math.isfinite(x) for x in row) and 0 <= row[1] <= 1 and row[2] <= -1 for row in values), f'{target}: invalid CSV metrics')
    require(all(math.isclose(row[0], row[1] + ETA[workload] * row[2], rel_tol=0, abs_tol=1e-9) for row in values), f'{target}: reward identity mismatch')
    log = contained_file(run, step['stdout'])
    events = [json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines() if line.startswith('{')]
    epoch_events = [event for event in events if event.get('event') == 'epoch']
    require([event['epoch'] for event in epoch_events] == list(range(1, epochs + 1)), f'{target}: nonconsecutive or missing log epochs')
    for row, event in zip(values, epoch_events):
        require(all(event[key] == value for key, value in {'algorithm': algorithm, 'seed': seed, 'epochs': epochs, 'traj_num': trajectories, 'backfill': 1}.items()), f'{target}: epoch metadata mismatch')
        require(row == [event['reward'], event['renewable_energy_utilization'], -event['average_bounded_slowdown']], f'{target}: CSV/log metric mismatch')
    checkpoint_path = contained_file(run, target / 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require(isinstance(checkpoint, dict) and checkpoint.get('format_version') == 1 and checkpoint.get('completed_epoch') == epochs, f'{target}: final checkpoint epoch or format mismatch')
    expected = dict(algorithm=algorithm, workload=workload, seed=seed, backfill=1, eta=ETA[workload], windows=[256,64,24], features=[8,4,2], traj_num=trajectories, semantics='author', delay_max_jobs=5, delay_times=[300,600,1200,1800,2400,3000,3600], gamma=1, lam=.97, clip=.2, updates=8, batch_size=256)
    require(checkpoint.get('metadata') == expected, f'{target}: checkpoint metadata mismatch')
    require(list(csv.reader(checkpoint['csv_history'].splitlines())) == rows, f'{target}: checkpoint CSV history mismatch')
    parameter_counts = {}
    for kind in ('actor', 'critic'):
        prefix = '_' if folder == 'MARL' else ''
        weights_path = contained_file(run, target / f'{prefix}{kind}.pkl')
        weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        saved = checkpoint[kind]
        require(isinstance(saved, dict) and bool(saved) and saved.keys() == weights.keys(), f'{target}: {kind} parameter keys mismatch')
        require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) and torch.equal(value, weights[key]) for key, value in saved.items()), f'{target}: {kind} parameters nonfinite or differ from standalone weights')
        parameter_counts[kind] = sum(value.numel() for value in saved.values())
        verify_optimizer(checkpoint[kind + '_optimizer'], saved, str(target) + '/' + kind)
    verify_rng(checkpoint['rng'], option(step['command'], '--device') == 'cuda')
    return dict(csv_rows=len(rows), consecutive_epochs=len(epoch_events), completed_epoch=checkpoint['completed_epoch'], metadata=checkpoint['metadata'], parameter_counts=parameter_counts, csv_equals_log=True, checkpoint_csv_equals_file=True, weights_equal_checkpoint=True, parameters_finite=True, adam_state_complete_and_finite=True, rng_state_present_and_parseable=True, cuda_rng_states=len(checkpoint['rng']['torch_cuda']), files_sha256={path.name: digest(path) for path in sorted(target.iterdir()) if path.is_file()})


def _verify_run(run, project_root, epochs, trajectories):
    run, project_root = Path(run).resolve(), Path(project_root).resolve()
    state = read_json(run / 'state.json')
    require(state['status'] == 'completed', 'Run is not completed')
    arguments = state['arguments']
    seeds, workloads = arguments['seeds'], arguments['workloads']
    require(arguments['stage'] == 'baseline', 'Full baseline run required')
    require(bool(seeds) and len(seeds) == len(set(seeds)) and all(type(seed) is int and seed >= 0 for seed in seeds), 'Invalid seed set')
    require(bool(workloads) and len(workloads) == len(set(workloads)) and set(workloads) <= ETA.keys(), 'Invalid workload set')
    steps = state['steps']
    expected_steps = {f'{stage}-{workload}-{seed}-{suffix}' for stage in ('smoke', 'baseline') for workload in workloads for seed in seeds for suffix in ('MARL-train','MaskablePPO-train','author-evaluate','paper-evaluate')}
    names = [step['name'] for step in steps]
    require(len(names) == len(set(names)) and set(names) == expected_steps, 'Missing, duplicate or unexpected pipeline steps')
    step_reports = []
    for stage in ('smoke', 'baseline'):
        for workload in workloads:
            for seed in seeds:
                for suffix in ('MARL-train','MaskablePPO-train','author-evaluate','paper-evaluate'):
                    name = f'{stage}-{workload}-{seed}-{suffix}'
                    step = next(item for item in steps if item['name'] == name)
                    require(step['status'] == 'completed' and step['returncode'] == 0, f'{name}: incomplete or nonzero exit')
                    stdout, stderr = contained_file(run, step['stdout']), contained_file(run, step['stderr'])
                    require(stderr.stat().st_size == 0, f'{name}: nonempty stderr')
                    command = step['command']
                    for flag, value in {'--seed':str(seed), '--workload':workload, '--backfill':'1'}.items():
                        require(option(command, flag) == value, f'{name}: command {flag} mismatch')
                    if suffix.endswith('-train'):
                        for flag, value in {'--epochs':str(epochs if stage == 'baseline' else 1), '--traj-num':str(trajectories if stage == 'baseline' else 2)}.items():
                            require(option(command, flag) == value, f'{name}: command {flag} mismatch')
                    else:
                        semantics = suffix.split('-')[0]
                        require(step['training_semantics'] == 'author' and step['evaluation_semantics'] == semantics, f'{name}: evaluation semantics mismatch')
                        for flag, value in {'--len':str(1024 if stage == 'baseline' else 64), '--iter':str(10 if stage == 'baseline' else 1)}.items():
                            require(option(command, flag) == value, f'{name}: command {flag} mismatch')
                        require(option(command, '--seed-mode', 'legacy') == arguments.get('evaluation_seed_mode', 'legacy'), f'{name}: evaluation seed mode mismatch')
                    step_reports.append(dict(name=name, returncode=0, stdout_sha256=digest(stdout), stderr_sha256=digest(stderr), stderr_bytes=0))
    manifest = read_json(run / 'source_manifest.json')
    require(isinstance(manifest, dict) and bool(manifest), 'Missing source manifest')
    for relative, expected in manifest.items():
        require(digest(contained_file(run / 'source', relative)) == expected, f'Source snapshot hash mismatch: {relative}')
    datasets = read_json(run / 'datasets.json')
    require(isinstance(datasets, list) and bool(datasets), 'Missing dataset manifest')
    require(len({item['path'] for item in datasets}) == len(datasets), 'Duplicate dataset manifest entry')
    for item in datasets:
        path = contained_file(project_root, item['path'])
        require(digest(path) == item['sha256'], f'Current input hash mismatch: {item["path"]}')
    training = {}
    for workload in workloads:
        for seed in seeds:
            for folder in ('MARL', 'MaskablePPO'):
                step = next(item for item in steps if item['name'] == f'baseline-{workload}-{seed}-{folder}-train')
                key = f'{workload}/seed_{seed}/' + ('MARL' if folder == 'MARL' else 'PPO')
                training[key] = verify_training(run, workload, seed, folder, step, epochs, trajectories)
    return dict(status='verified', verified_at_utc=datetime.now(timezone.utc).isoformat(), run_id=run.name, run_directory=str(run), source_commit=state['git_commit'], training_seeds=seeds, workloads=workloads, expected_epochs=epochs, expected_trajectories=trajectories, step_count=len(steps), exact_step_set_verified=True, all_step_exit_codes_zero=True, all_stderr_empty=True, steps=step_reports, source_snapshot_files_verified=len(manifest), source_snapshot_sha256=manifest, data_files_verified=len(datasets), current_input_data_sha256={item['path']:item['sha256'] for item in datasets}, original_state_sha256=digest(run / 'state.json'), source_manifest_sha256=digest(run / 'source_manifest.json'), dataset_manifest_sha256=digest(run / 'datasets.json'), verifier_sha256=digest(__file__), training=training, limits=['Audits retained artifacts and command metadata; does not rerun training or prove bitwise replay.', 'Does not establish paper performance reproduction or evaluation independence.', 'Source snapshot hashes are checked against each run manifest; current training sources may have changed after a run.', 'CUDA RNG byte states are present and structurally checked; no global or device RNG is restored.', 'Seed0 original evaluation is legacy; use separately verified paired evaluation for cross-seed statistics.'] if arguments.get('evaluation_seed_mode', 'legacy') == 'legacy' else ['Audits retained artifacts and command metadata; does not rerun training or prove bitwise replay.', 'Does not establish paper performance reproduction or evaluation independence.', 'Source snapshot hashes are checked against each run manifest; current training sources may have changed after a run.', 'CUDA RNG byte states are present and structurally checked; no global or device RNG is restored.'])


def verify_run(run, project_root=ROOT, epochs=300, trajectories=100):
    """Return audit evidence without changing artifacts or global random state."""
    try:
        require(type(epochs) is int and epochs > 0 and type(trajectories) is int and trajectories > 0, 'Invalid expected training configuration')
        return _verify_run(run, project_root, epochs, trajectories)
    except VerificationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AttributeError, EOFError) as error:
        raise VerificationError(f'Malformed or unreadable artifacts: {error}') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        report = verify_run(args.run_dir, args.project_root)
    except VerificationError as error:
        print(json.dumps({'status':'failed', 'error':str(error)}), file=sys.stderr)
        return 1
    target = args.run_dir.resolve() / 'analysis' / 'verification_multiseed.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status':'verified', 'report':str(target), 'steps':report['step_count'], 'training_seeds':report['training_seeds'], 'completed_epochs':{key:item['completed_epoch'] for key,item in report['training'].items()}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
