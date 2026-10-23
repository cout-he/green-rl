"""Use the tested process-tree watchdog for exactly one bounded pilot command."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_research import capture_source, execute_step, git_output, now, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        parser.error('run-id must be a single safe directory name')
    output = ROOT / 'results' / args.run_id
    supervision = ROOT / 'results' / (args.run_id + '-supervisor')
    if output.exists() or supervision.exists() or (ROOT / '.research.lock').exists():
        parser.error('Existing output or research lock; preserve evidence and inspect actual processes')
    supervision.mkdir(parents=True, exist_ok=False)
    state = {'status': 'preparing', 'pid': os.getpid(), 'started_at': now(),
             'run_id': args.run_id, 'git_commit': git_output('rev-parse', 'HEAD'),
             'git_status': git_output('status', '--short'),
             'hard_process_timeout_seconds': 600, 'soft_matrix_timeout_seconds': 550,
             'episode_cap': 18, 'pilot_output': str(output)}

    def publish(step=None):
        if step is not None:
            state['step'] = step
        write_json(supervision / 'state.json', state)

    publish()
    try:
        capture_source(supervision)
        command = [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8',
                   str(ROOT / 'scripts/run_forecast_pilot.py'), '--output-dir', str(output),
                   '--timeout-seconds', '550', '--device', 'cuda', '--num-threads', '1']
        state['status'] = 'running'
        publish()
        step = execute_step('frozen-forecast-pilot', command, ROOT, supervision / 'logs',
                            600, expected=[output / 'summary.json', output / 'rows.json'],
                            env=dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1',
                                     OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), on_update=publish)
        state['step'] = step
        state['status'] = step['status']
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        state['ended_at'] = now()
        publish()
    print(json.dumps({'status': state['status'], 'supervision': str(supervision)}), flush=True)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
