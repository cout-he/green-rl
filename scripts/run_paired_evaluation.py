"""Re-evaluate fixed final weights with isolated matched RNG; never train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    baseline = args.baseline_run.resolve()
    state = json.loads((baseline / 'state.json').read_text(encoding='utf-8'))
    if state['status'] != 'completed':
        parser.error('The fixed-model baseline must be completed')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('run-id must be one directory name')
    output = ROOT / 'results' / args.run_id
    output.mkdir(exist_ok=False)
    target = baseline / 'baseline/lublin_256/seed_0'
    weights = sorted((target / 'models').rglob('*.pkl'))
    manifest = {str(p.relative_to(baseline)): hashlib.sha256(p.read_bytes()).hexdigest() for p in weights}
    record = {'status': 'running', 'baseline_run': str(baseline), 'model_sha256': manifest,
              'training_seed': 0, 'evaluation_seed': 0, 'seed_mode': 'paired',
              'started_at': datetime.now(timezone.utc).isoformat(), 'steps': []}
    (output / 'compare.py').write_bytes((ROOT / 'compare.py').read_bytes())
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        for semantics in ('author', 'paper'):
            env['GREEN_RL_DELAY_SEMANTICS'] = semantics
            command = [sys.executable, '-X', 'utf8', 'compare.py', '--workload', 'lublin_256',
                       '--backfill', '1', '--seed', '0', '--device', 'cuda', '--num-threads', '1',
                       '--len', '1024', '--iter', '10', '--algorithms', 'FCFS', 'F2', 'PPO', 'MARL',
                       '--seed-mode', 'paired', '--model-root', str(target / 'models'),
                       '--output-dir', str(output / semantics)]
            with (output / (semantics + '.stdout.log')).open('wb') as out, (output / (semantics + '.stderr.log')).open('wb') as err:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, timeout=600)
            record['steps'].append({'semantics': semantics, 'command': command, 'returncode': process.returncode})
            if process.returncode:
                raise RuntimeError(semantics + ' evaluation failed')
        after = {str(p.relative_to(baseline)): hashlib.sha256(p.read_bytes()).hexdigest() for p in weights}
        if after != manifest:
            raise RuntimeError('Frozen model weights changed during evaluation')
        rows = {mode: json.loads((output / mode / 'rows.json').read_text()) for mode in ('author', 'paper')}
        for algorithm in ('FCFS', 'F2', 'PPO'):
            a = [row for row in rows['author'] if row['algorithm'] == algorithm]
            b = [row for row in rows['paper'] if row['algorithm'] == algorithm]
            if a != b:
                raise RuntimeError('Zero-effect control differs: ' + algorithm)
        record.update(status='completed', control_equality=True)
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        record['ended_at'] = datetime.now(timezone.utc).isoformat()
        (output / 'state.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'completed', 'output': str(output), 'controls_identical': True}))


if __name__ == '__main__':
    main()
