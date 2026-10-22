"""Audit real training/evaluation artifacts; load only this project's trusted weights."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    state = json.loads((run / 'state.json').read_text(encoding='utf-8'))
    assert state['status'] == 'completed' and len(state['steps']) == 8
    assert all(x['status'] == 'completed' and x['returncode'] == 0 for x in state['steps'])
    assert all(Path(x['stderr']).stat().st_size == 0 for x in state['steps'])
    manifest = json.loads((run / 'source_manifest.json').read_text())
    assert all(digest(run / 'source' / p) == value for p, value in manifest.items())
    datasets = json.loads((run / 'datasets.json').read_text())
    assert all(digest(ROOT / item['path']) == item['sha256'] for item in datasets)
    models = run / 'baseline/lublin_256/seed_0/models/lublin_256'
    results = {}
    for algorithm, folder, prefix in [('MARL', 'MARL', '_'), ('PPO', 'MaskablePPO', '')]:
        target = models / folder
        rows = list(csv.reader((target / f'{folder}_lublin_256.csv').open(newline='')))
        assert len(rows) == 300 and all(len(row) == 3 for row in rows)
        values = [[float(x) for x in row] for row in rows]
        assert all(all(math.isfinite(x) for x in row) and 0 <= row[1] <= 1 and row[2] <= -1 for row in values)
        assert all(math.isclose(row[0], row[1] + .002 * row[2], abs_tol=1e-9) for row in values)
        step = next(x for x in state['steps'] if x['name'] == f'baseline-lublin_256-0-{folder}-train')
        events = [json.loads(line) for line in Path(step['stdout']).read_text().splitlines() if line.startswith('{')]
        epochs = [x for x in events if x.get('event') == 'epoch']
        assert [x['epoch'] for x in epochs] == list(range(1, 301))
        assert all(row == [event['reward'], event['renewable_energy_utilization'], -event['average_bounded_slowdown']] for row, event in zip(values, epochs))
        checkpoint = torch.load(target / 'checkpoint.pt', map_location='cpu', weights_only=False)
        assert checkpoint['completed_epoch'] == 300
        assert checkpoint['metadata']['seed'] == 0 and checkpoint['metadata']['traj_num'] == 100
        assert checkpoint['metadata']['semantics'] == 'author' and checkpoint['metadata']['backfill'] == 1
        assert list(csv.reader(checkpoint['csv_history'].splitlines())) == rows
        for kind in ('actor', 'critic'):
            weights = torch.load(target / (prefix + kind + '.pkl'), map_location='cpu', weights_only=True)
            assert weights.keys() == checkpoint[kind].keys()
            assert all(torch.equal(weights[key], checkpoint[kind][key]) and torch.isfinite(weights[key]).all() for key in weights)
        assert all(checkpoint[key]['param_groups'] for key in ('actor_optimizer', 'critic_optimizer'))
        assert {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'environment'} <= checkpoint['rng'].keys()
        results[algorithm] = {'csv_rows': len(rows), 'consecutive_epochs': len(epochs), 'completed_epoch': 300,
                              'weights_equal_checkpoint': True, 'parameters_finite': True, 'csv_equals_log': True,
                              'full_state_present': True,
                              'files_sha256': {p.name: digest(p) for p in target.iterdir() if p.is_file()}}
    starts = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]
    overlap_pairs = [[a, b] for i, a in enumerate(starts) for b in starts[i + 1:] if max(a, b) < min(a + 1024, b + 1024)]
    report = {'status': 'verified', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'run_id': run.name, 'source_commit': state['git_commit'], 'source_snapshot_files_verified': len(manifest),
              'data_files_verified': len(datasets), 'all_step_exit_codes_zero': True, 'all_stderr_empty': True,
              'training': results, 'overlapping_evaluation_window_pairs': overlap_pairs,
              'original_state_sha256': digest(run / 'state.json'),
              'limits': ['One independent training seed.', 'Ten partly overlapping windows are descriptive evaluation slices, not independent training replicates.',
                         'Source snapshot verification does not claim current files remain identical after isolated evaluation fixes.']}
    target = run / 'analysis' / 'verification.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'verified', 'report': str(target), 'training': {key: value['completed_epoch'] for key, value in results.items()}}))


if __name__ == '__main__':
    main()
