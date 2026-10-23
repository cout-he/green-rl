"""One bounded frozen-policy H1 diagnostic. No training or shared-file mutation.

Outputs: state.json (written before episodes), rows.jsonl/rows.json, summary.json,
and traces/<episode>.json.gz. Each trace contains exact zero-gate categories,
physical intervals, forecast counters, job records, actions and event snapshots.
Run only after review; --output-dir must never have existed.
"""
import argparse
from collections import Counter
import configparser
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EPISODE_CAP = 18
GATE_KEYS = ('actions', 'events', 'schedule', 'jobs', 'metrics', 'rng')
BASELINE = 'gas-marl-20261021-baseline-02'
MULTISEED = 'gas-marl-20261022-multiseed-01'


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _bytes(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def select_windows(starts, length=1024, count=2):
    """Select half-open job-index windows in original order, retaining iteration."""
    if length <= 0 or count <= 0:
        raise ValueError('Window length and count must be positive')
    selected = []
    for iteration, start in enumerate(starts, 1):
        if not any(start < w['start'] + length and w['start'] < start + length for w in selected):
            selected.append({'start': int(start), 'iteration': iteration, 'sequence_length': length})
            if len(selected) == count:
                break
    return selected


def evaluation_seed(seed, iteration):
    # compare.py's fixed available.index('MARL') is 5; iteration is one-based.
    return (seed + 1000003 * iteration + 10007 * 5) % (2 ** 32)


def zero_gate(original, zero):
    hashes = {name: {'original': digest(original[name]), 'zero': digest(zero[name])} for name in GATE_KEYS}
    equal = {name: hashes[name]['original'] == hashes[name]['zero'] for name in GATE_KEYS}
    return {'passed': all(equal.values()), 'equal': equal, 'sha256': hashes, 'tolerance': 'exact'}


def summarize_h1(rows, seeds=(0, 1, 2), starts=(6567, 919), protocol_valid=True):
    selected = {(r['seed'], r['start'], r['condition']): r for r in rows
                if r['condition'] in ('zero', 'plus20')}
    per_seed = []
    for seed in seeds:
        differences = []
        for start in starts:
            zero, stress = selected.get((seed, start, 'zero')), selected.get((seed, start, 'plus20'))
            if zero is not None and stress is not None:
                value = stress['renewable_energy_utilization'] - zero['renewable_energy_utilization']
                differences.append({'start': start, 'difference': value,
                                    'relative_change': value / zero['renewable_energy_utilization']
                                    if zero['renewable_energy_utilization'] else None})
        mean = sum(d['difference'] for d in differences) / len(differences) if differences else None
        per_seed.append({'seed': seed, 'pairs': differences, 'window_count': len(differences),
                         'mean_difference': mean, 'strictly_negative': mean is not None and mean < 0})
    complete = all(r['window_count'] == len(starts) for r in per_seed)
    means = [r['mean_difference'] for r in per_seed if r['mean_difference'] is not None]
    descriptive_negative = complete and all(r['strictly_negative'] for r in per_seed)
    return {'complete': complete, 'protocol_valid': bool(protocol_valid),
            'supported': bool(protocol_valid) and descriptive_negative,
            'descriptive_all_seed_means_negative': descriptive_negative,
            'per_seed': per_seed, 'seed_mean_difference_range': [min(means), max(means)] if means else None,
            'interpretation': 'Controlled sensitivity diagnostic only; no significance, generalization or novelty claim.'}


def quantiles(values, level='jobs_within_episode'):
    import numpy as np
    if not values:
        return {'method': 'linear', 'level': level, 'count': 0,
                'p50': None, 'p95': None, 'p99': None, 'maximum': None}
    q = np.quantile(values, [.5, .95, .99], method='linear')
    return {'method': 'linear', 'level': level, 'count': len(values),
            'p50': float(q[0]), 'p95': float(q[1]), 'p99': float(q[2]), 'maximum': float(max(values))}


def physical_energy(slots, physical, jobs, static_power, first_arrival):
    """Keep the author's first-schedule to final-completion log denominator."""
    intervals, total, green = [], 0.0, 0.0
    for before, after in zip(slots, slots[1:]):
        start, end, power = before['timeSlot'], after['timeSlot'], before['power']
        energy = (end - start) * power
        renewable = physical.getGreenUtili(power, start, end)
        total += energy
        green += renewable
        intervals.append({'start': start, 'end': end, 'power_w': power, 'total_energy_j': energy,
                          'green_energy_j': renewable, 'brown_energy_j': energy - renewable})
    begin, end = (slots[0]['timeSlot'], slots[-1]['timeSlot']) if slots else (first_arrival, first_arrival)
    independent = sum(j['power'] * j['run_time'] for j in jobs) + static_power * (end - begin)
    return {'total_energy_j': total, 'green_energy_j': green, 'brown_energy_j': total - green,
            'log_start': begin, 'completion_timestamp': end, 'log_horizon_seconds': end - begin,
            'makespan_seconds': end - first_arrival, 'initial_idle_gap_seconds': begin - first_arrival,
            'initial_idle_gap_energy_j': static_power * (begin - first_arrival),
            'energy_identity_expected_j': independent,
            'energy_identity_passed': math.isclose(total, independent, rel_tol=1e-12, abs_tol=1e-6),
            'independent_renewable_energy_utilization': green / total if total else 1,
            'physical_intervals': intervals,
            'denominator': 'Original PowerStruc first scheduled time through last actual completion; initial idle gap separate.'}


def validate_config(path):
    config = configparser.ConfigParser()
    config.read(path, encoding='utf-8')
    expected = {'MAX_QUEUE_SIZE': 256, 'run_win': 64, 'green_win': 24, 'eta': .002}
    actual = {key: config.getfloat('GAS-MARL setting', key) for key in expected}
    if actual != expected:
        raise ValueError('Current config must match frozen 256/64/24/.002: ' + str(actual))
    return actual


def validate_evaluation_evidence(root, audit):
    """Authenticate the copies actually consumed and pin each model to its seed."""
    root = Path(root).resolve()
    evidence = root / 'research/evidence/multiseed'
    hashes = json.loads((evidence / 'evidence_sha256.json').read_text(encoding='utf-8'))
    verified, metadata = {}, []
    audited_models = {k.replace('\\', '/'): v for k, v in audit['checksum_list'].items()}
    for seed in (0, 1, 2):
        for suffix in ('metadata', 'rows'):
            name = 'seed%d_author_%s.json' % (seed, suffix)
            if file_hash(evidence / name) != hashes[name]:
                raise ValueError('Consumed evaluation evidence hash mismatch: ' + name)
            verified['research/evidence/multiseed/' + name] = hashes[name]
        item = json.loads((evidence / ('seed%d_author_metadata.json' % seed)).read_text(encoding='utf-8'))
        run = BASELINE if seed == 0 else MULTISEED
        fixed_root = 'results/%s/baseline/lublin_256/seed_%d/models' % (run, seed)
        resolved = (root / item['model_root']).resolve()
        if resolved != (root / fixed_root).resolve():
            raise ValueError('Evaluation model root does not identify the audited final seed model')
        for name in ('_actor.pkl', '_critic.pkl'):
            model_path = (resolved / 'lublin_256/MARL' / name).relative_to(root).as_posix()
            if model_path not in audited_models:
                raise ValueError('Evaluation model weight is absent from audited final checksums: ' + model_path)
        metadata.append(item)
    return metadata, verified


def validate_frozen(root=ROOT):
    """Check audited inputs/weights and current evaluator source before evaluation."""
    root = Path(root)
    evidence = root / 'research/evidence/multiseed'
    audit = json.loads((evidence / 'training_verification.json').read_text(encoding='utf-8'))
    delivery = json.loads((evidence / 'delivery_validation.json').read_text(encoding='utf-8'))
    if audit['status'] != 'verified' or delivery['status'] != 'verified_delivery':
        raise ValueError('Completed audited three-seed baseline required')
    inputs = json.loads((evidence / 'input_sha256.json').read_text(encoding='utf-8'))
    expected = {**inputs, **audit['checksum_list']}
    current_sources = json.loads((root / 'results' / MULTISEED / 'source_manifest.json').read_text(encoding='utf-8'))
    source_names = ('HPCSimPickJobs.py', 'cluster.py', 'greenPower.py', 'PowerStruc.py', 'job.py',
                    'MARL.py', 'MaskablePPO.py', 'training_checkpoint.py', 'compare.py', 'configFile/config.ini')
    normalized_source = {k.replace('\\', '/'): v for k, v in current_sources.items()}
    for name in source_names:
        expected[name] = normalized_source[name]
    datasets = json.loads((root / 'research/evidence/baseline02/datasets.json').read_text(encoding='utf-8'))
    expected.update({d['path']: d['sha256'] for d in datasets})
    metadata, consumed_evidence = validate_evaluation_evidence(root, audit)
    expected.update(consumed_evidence)
    for name, wanted in expected.items():
        path = root / name.replace('\\', '/')
        if not path.is_file() or file_hash(path) != wanted:
            raise ValueError('Frozen input/source mismatch: ' + name)
    configs = validate_config(root / 'configFile/config.ini')
    if any(m['starts'] != metadata[0]['starts'] or m['sequence_length'] != 1024 or
           m['evaluation_seed_mode'] != 'paired' or m['delay_semantics'] != 'author' or m['backfill'] != 1
           for m in metadata):
        raise ValueError('Paired author protocol metadata mismatch')
    for seed in (0, 1, 2):
        model_audit = audit['per_seed'][str(seed)]['MARL']
        if model_audit['completed_epoch'] != 300 or not model_audit['weights_equal_checkpoint']:
            raise ValueError('Audited final epoch-300 MARL weights required')
    for name in ('scripts/run_forecast_pilot.py', 'scripts/forecast_pilot_env.py',
                 'research/innovation_screening.md', 'research/evidence/multiseed/training_verification.json',
                 'research/evidence/multiseed/delivery_validation.json',
                 'research/evidence/multiseed/input_sha256.json',
                 'research/evidence/multiseed/evidence_sha256.json'):
        expected[name] = file_hash(root / name)
    return {'schema_version': '1.0', 'hypothesis': 'All three seed means of ReUtil_plus20-ReUtil_zero are strictly negative.',
            'seeds': [0, 1, 2], 'windows': select_windows(metadata[0]['starts']), 'config': configs,
            'backfill': 1, 'delay_semantics': 'author', 'eta': .002, 'evaluation_policy': 'sample',
            'seed_derivation': '(training_seed + 1000003 * original_iteration + 10007 * 5) mod 2**32',
            'frozen_sha256': {k.replace('\\', '/'): v for k, v in expected.items()},
            'model_roots': {str(seed): metadata[seed]['model_root'] for seed in (0, 1, 2)},
            'reference_rows': {str(seed): json.loads((evidence / ('seed%d_author_rows.json' % seed)).read_text(encoding='utf-8'))
                               for seed in (0, 1, 2)},
            'identity_tolerances': {'energy_rel': 1e-12, 'energy_abs_j': 1e-6, 'bsld_rel': 1e-12, 'bsld_abs': 1e-10},
            'zero_gate_tolerance': 'exact across six categories; cost/forecast metadata excluded'}


def _check_frozen(manifest, root=ROOT):
    for name, wanted in manifest['frozen_sha256'].items():
        if file_hash(Path(root) / name) != wanted:
            raise ValueError('Frozen input mutated during pilot: ' + name)


def execute_matrix(output, manifest, episode, timeout_seconds=600, max_actions=4096,
                   lock_path=None, clock=time.monotonic, final_check=None):
    """Execute all zero gates before stress; persist failures without retrying."""
    output = Path(output)
    planned = len(manifest['seeds']) * len(manifest['windows']) * 3
    if planned > EPISODE_CAP or planned < 1 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or max_actions < 1:
        raise ValueError('Invalid bounded pilot budget (maximum 18 episode starts)')
    # Share the research pipeline's lock so neither process can start over the other.
    lock_path = Path(lock_path) if lock_path else ROOT / '.research.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(lock_fd, _bytes({'pid': os.getpid(), 'output': str(output.resolve()), 'created_utc': datetime.now(timezone.utc).isoformat()}))
        output.mkdir(parents=True, exist_ok=False)
        (output / 'traces').mkdir()
        begin = clock()
        deadline = begin + timeout_seconds
        cpu_begin = time.process_time()
        state = {'schema_version': '1.0', 'status': 'running', 'manifest': manifest,
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
                 'budget': {'planned_episode_starts': planned, 'episode_cap': EPISODE_CAP,
                            'timeout_seconds': timeout_seconds, 'max_actions_per_episode': max_actions},
                 'episodes_started': 0, 'episodes_completed': 0, 'zero_gates': [], 'rows': [],
                 'frozen_inputs_verified_after': False}
        _save(output / 'state.json', state)
        def one(seed, window, condition):
            if clock() >= deadline:
                raise TimeoutError('Pilot total wall-clock timeout')
            if state['episodes_started'] >= EPISODE_CAP:
                raise RuntimeError('Absolute 18 episode-start cap reached')
            identity = {'seed': seed, **window, 'condition': condition,
                        'bias': {'original': None, 'zero': 0.0, 'plus20': .2}[condition],
                        'policy_seed': evaluation_seed(seed, window['iteration'])}
            state['episodes_started'] += 1
            state['active_episode'] = identity
            _save(output / 'state.json', state)
            episode_begin = clock()
            episode_cpu_begin = time.process_time()
            try:
                result = episode(identity, deadline, max_actions)
            except BaseException as exc:
                partial = getattr(exc, 'pilot_partial', None)
                if partial is not None:
                    trace_path = output / 'traces' / ('partial_%02d.json.gz' % state['episodes_started'])
                    with gzip.open(trace_path, 'wb') as stream:
                        stream.write(_bytes(partial))
                    state['partial_trace'] = str(trace_path.relative_to(output))
                raise
            overran_episode = clock() >= deadline
            trace_path = output / 'traces' / ('%02d_seed%d_start%d_%s.json.gz' %
                          (state['episodes_started'], seed, window['start'], condition))
            with gzip.open(trace_path, 'wb') as stream:
                stream.write(_bytes(result['trace']))
            row = {**identity, **result['row'], 'trace': str(trace_path.relative_to(output)),
                   'trace_content_sha256': digest(result['trace']), 'trace_bytes': trace_path.stat().st_size,
                   'gate_sha256': {name: digest(result['trace'][name]) for name in GATE_KEYS}}
            row['controller_wall_seconds'] = clock() - episode_begin
            row['controller_process_cpu_seconds'] = time.process_time() - episode_cpu_begin
            state['rows'].append(row)
            state['episodes_completed'] += 1
            state.pop('active_episode', None)
            with (output / 'rows.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(_plain(row), allow_nan=False) + '\n')
                stream.flush()
            _save(output / 'state.json', state)
            if overran_episode or clock() >= deadline:
                raise TimeoutError('Pilot total wall-clock timeout after episode or trace serialization; completed data retained')
            print(json.dumps({'event': 'pilot_episode', **{k: row[k] for k in identity},
                              'episodes_completed': state['episodes_completed'],
                              'wall_seconds': row.get('wall_seconds')}), flush=True)
            return result
        try:
            for seed in manifest['seeds']:
                for window in manifest['windows']:
                    pair_begin = clock()
                    original = one(seed, window, 'original')
                    zero = one(seed, window, 'zero')
                    gate = {'seed': seed, 'start': window['start'], **zero_gate(original['trace'], zero['trace'])}
                    state['zero_gates'].append(gate)
                    if len(state['zero_gates']) == 1:
                        cost = sum(r['row'].get('wall_seconds', 0) for r in (original, zero))
                        pair_cost = clock() - pair_begin
                        state['first_pair_cost'] = {'episodes': 2, 'simulation_wall_seconds_sum': cost,
                                                   'pair_wall_seconds': pair_cost,
                                                   'projected_18_episode_seconds': pair_cost * 9,
                                                   'measurement': 'First original/zero pair including model/env setup, traces, gates and instrumentation.'}
                    _save(output / 'state.json', state)
                    if not gate['passed']:
                        raise RuntimeError('Zero-error equivalence gate failed; stress episodes forbidden')
            state['all_zero_gates_passed'] = True
            _save(output / 'state.json', state)
            for seed in manifest['seeds']:
                for window in manifest['windows']:
                    one(seed, window, 'plus20')
            if final_check:
                final_check()
            state['frozen_inputs_verified_after'] = True
            if clock() >= deadline:
                raise TimeoutError('Pilot total wall-clock timeout after final frozen-input check')
            state['status'] = 'completed'
        except BaseException as exc:
            state.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                         traceback=traceback.format_exc())
        finally:
            state['ended_utc'] = datetime.now(timezone.utc).isoformat()
            state['total_wall_seconds'] = clock() - begin
            state['total_process_cpu_seconds'] = time.process_time() - cpu_begin
            protocol_valid = (state['status'] == 'completed' and state.get('all_zero_gates_passed', False)
                              and state['frozen_inputs_verified_after'])
            state['h1'] = summarize_h1(state['rows'], manifest['seeds'], [w['start'] for w in manifest['windows']], protocol_valid)
            state['log_bytes_before_final_summary'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            _save(output / 'rows.json', state['rows'])
            _save(output / 'state.json', state)
            summary = {k: v for k, v in state.items() if k not in ('manifest', 'rows', 'traceback')}
            summary['manifest_file'] = 'state.json'
            summary['rows_file'] = 'rows.json'
            _save(output / 'summary.json', summary)
        return summary
    finally:
        os.close(lock_fd)
        lock_path.unlink()


def rng_snapshot(env):
    import numpy as np
    import torch
    env_rng = getattr(env, 'np_random', None)
    if hasattr(env_rng, 'bit_generator'):
        environment = env_rng.bit_generator.state
    elif hasattr(env_rng, 'get_state'):
        environment = env_rng.get_state()
    else:
        environment = None
    return {'python': digest(random.getstate()), 'numpy': digest(np.random.get_state()),
            'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'torch_cuda': [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_available() else [],
            'environment': digest(environment), 'encoding': 'SHA256 of exact state bytes or canonical full state JSON'}


def job_record(job):
    fields = ('job_id', 'submit_time', 'run_time', 'request_time', 'request_number_of_processors',
              'number_of_allocated_processors', 'request_number_of_nodes', 'power', 'scheduled_time')
    record = {name: getattr(job, name) for name in fields}
    record['allocated_machine_ids'] = [m.id for m in (job.allocated_machines or [])]
    scheduled = job.scheduled_time >= 0
    record.update(wait_seconds=job.scheduled_time - job.submit_time if scheduled else None,
                  completion_timestamp=job.scheduled_time + job.run_time if scheduled else None,
                  bounded_slowdown=max(1., (job.scheduled_time - job.submit_time + job.run_time) /
                                       max(job.run_time, 10)) if scheduled else None)
    return record


class EpisodeTrace:
    """Instance wrappers only; every wrapped original is invoked exactly once."""
    def __init__(self, env, jobs, deadline):
        self.env, self.jobs, self.deadline = env, jobs, deadline
        self.events, self.schedule, self.actions = [], [], []
        self.backfill = {'candidate_checks': 0, 'accepted_checks': 0, 'actual_scheduled': 0,
                         'candidate_definition': 'Calls reaching cluster.backfill_check after queue/time predicates.',
                         'all_queue_predicate_evaluations': None,
                         'missing_reason': 'Short-circuit queue predicates are not exposed by baseline methods.'}
        self.clipping = {'slot_values': 0, 'future_slot_values': 0, 'power_clipped': 0,
                         'future_power_clipped': 0, 'max_raw_power_ratio': 0.,
                         'threshold': 1 - 1e-5,
                         'job_green_feature': 'getGreenJobState has no clipping; returned features logged in events'}
        self.selected = None
        self.restores = []
        for name in ('schedule', 'schedule_backfill', 'moveforward_for_job', 'skip_for_resources',
                     'moveforward_for_resources_backfill', 'skip1', 'skip2',
                     'moveforward_green_backfilling_delay_action1', 'moveforward_green_backfilling_delay_action2'):
            self.wrap(env, name, self.event_wrapper(name))
        self.wrap(env.cluster.PowerStruc, 'updateCurrentTime', self.event_wrapper('power.updateCurrentTime'))
        self.wrap(env.cluster.PowerStruc, 'update', self.event_wrapper('power.update'))
        self.wrap(env.cluster, 'allocate', self.allocate_wrapper)
        self.wrap(env.cluster, 'release', self.event_wrapper('cluster.release'))
        self.wrap(env.cluster, 'backfill_check', self.backfill_wrapper)
        self.wrap(env.cluster, 'getGreenJobState', self.job_green_wrapper)
        self.wrap(env.cluster.greenPower, 'getGreenPowerSlot', self.slot_wrapper)

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Pilot deadline during environment transition')

    def state(self):
        return {'time': self.env.current_timestamp, 'next_arrival_index': self.env.next_arriving_job_idx,
                'queue': [j.job_id for j in self.env.job_queue],
                'running': [[j.job_id, j.scheduled_time, j.run_time, j.request_time] for j in self.env.running_jobs],
                'free_nodes': self.env.cluster.free_node, 'used_nodes': self.env.cluster.used_node}

    def wrap(self, obj, name, factory):
        original = getattr(obj, name)
        had_instance = name in vars(obj)
        self.restores.append((obj, name, original, had_instance))
        setattr(obj, name, factory(original))

    def event_wrapper(self, name):
        def factory(original):
            def call(*args, **kwargs):
                self.check_time()
                self.events.append({'method': name, 'phase': 'before', 'state': self.state()})
                result = original(*args, **kwargs)
                self.events.append({'method': name, 'phase': 'after', 'state': self.state()})
                return result
            return call
        return factory

    def allocate_wrapper(self, original):
        def call(job_id, processors):
            self.check_time()
            result = original(job_id, processors)
            backfilled = self.selected is not None and job_id != self.selected
            record = {'job_id': job_id, 'time': self.env.current_timestamp, 'processors': processors,
                      'machine_ids': [m.id for m in result], 'backfilled': backfilled}
            self.schedule.append(record)
            if backfilled and result:
                self.backfill['actual_scheduled'] += 1
            self.events.append({'method': 'cluster.allocate', **record, 'state': self.state()})
            return result
        return call

    def backfill_wrapper(self, original):
        def call(running, job, current_time, backfill=1):
            self.check_time()
            result = original(running, job, current_time, backfill)
            self.backfill['candidate_checks'] += 1
            self.backfill['accepted_checks'] += int(bool(result))
            self.events.append({'method': 'cluster.backfill_check', 'job_id': job.job_id,
                                'time': current_time, 'accepted': bool(result), 'state': self.state()})
            return result
        return call

    def job_green_wrapper(self, original):
        def call(job, current_time, current_slot):
            result = original(job, current_time, current_slot)
            self.events.append({'method': 'cluster.getGreenJobState', 'job_id': job.job_id,
                                'time': current_time, 'features': list(result)})
            return result
        return call

    def slot_wrapper(self, original):
        def call(current_time):
            slots = original(current_time)
            for index, slot in enumerate(slots):
                ratio = slot['power'] / 19000.
                self.clipping['slot_values'] += 1
                self.clipping['future_slot_values'] += int(index > 0)
                self.clipping['power_clipped'] += int(ratio > self.clipping['threshold'])
                self.clipping['future_power_clipped'] += int(index > 0 and ratio > self.clipping['threshold'])
                self.clipping['max_raw_power_ratio'] = max(self.clipping['max_raw_power_ratio'], ratio)
            self.events.append({'method': 'greenPower.getGreenPowerSlot', 'time': current_time, 'raw_slots': slots})
            return slots
        return call

    def restore(self):
        for obj, name, original, had_instance in reversed(self.restores):
            if had_instance:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def process_memory():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {'rss_bytes': counters.WorkingSetSize, 'peak_rss_bytes': counters.PeakWorkingSetSize,
                    'source': 'Windows GetProcessMemoryInfo; peak is process lifetime, not episode reset',
                    'peak_missing_reason': None}
    try:
        import psutil
        memory = psutil.Process().memory_info()
        return {'rss_bytes': memory.rss, 'peak_rss_bytes': getattr(memory, 'peak_wset', None),
                'source': 'psutil OS process memory',
                'peak_missing_reason': None if hasattr(memory, 'peak_wset') else 'OS peak field unavailable'}
    except ImportError:
        return {'rss_bytes': None, 'peak_rss_bytes': None, 'source': None,
                'peak_missing_reason': 'psutil unavailable; no memory value inferred'}


def gpu_cost():
    import torch
    if not torch.cuda.is_available():
        return {'available': False, 'utilization': None, 'utilization_missing_reason': 'CUDA unavailable'}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {'available': True, 'device': device, 'name': props.name, 'total_memory_bytes': props.total_memory,
            'allocated_bytes': torch.cuda.memory_allocated(device), 'reserved_bytes': torch.cuda.memory_reserved(device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'utilization': None,
            'utilization_missing_reason': 'No sampled GPU utilization service; allocator values are not utilization.'}


def run_episode(spec, env, model, deadline, max_actions=4096, reference=None):
    """Run baseline action logic with transparent traces; small fixtures supported."""
    import numpy as np
    import torch
    from HPCSimPickJobs import MAX_QUEUE_SIZE, JOB_FEATURES, action2_num, delayMaxJobNum, delayTimeList, eta
    from MARL import configure_runtime
    from scripts.forecast_pilot_env import install_forecast
    begin, cpu_begin = time.monotonic(), time.process_time()
    configure_runtime(spec['policy_seed'], str(model.device), 1)
    env.seed(spec['policy_seed'])
    env.reset_for_test(spec['sequence_length'], spec['start'])
    jobs = [env.loads[i] for i in range(spec['start'], spec['start'] + spec['sequence_length'])]
    physical = env.cluster.greenPower
    physical_before = digest(physical.greenPowerList)
    proxy = install_forecast(env, spec['bias']) if spec['bias'] is not None else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trace = {'actions': [], 'events': [], 'schedule': [], 'jobs': [], 'metrics': {},
             'rng': {'before': rng_snapshot(env)}, 'observations': []}
    instrumentation = EpisodeTrace(env, jobs, deadline)
    trace.update(actions=instrumentation.actions, events=instrumentation.events, schedule=instrumentation.schedule)
    try:
        observation = env.build_observation()
        running_num, reward, green_reward = 0, 0., 0.
        while True:
            instrumentation.check_time()
            if len(trace['actions']) >= max_actions:
                raise RuntimeError('Maximum actions per episode reached')
            invalid = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                value = observation[i:i + JOB_FEATURES]
                invalid.append(1 if all(value == [0] + [1] * (JOB_FEATURES - 2) + [0]) or
                               all(value == [1] * JOB_FEATURES) else 0)
            mask2 = np.zeros(action2_num, dtype=int)
            if running_num < delayMaxJobNum:
                mask2[running_num + 1:delayMaxJobNum + 1] = 1
            trace['observations'].append({'time': env.current_timestamp, 'sha256': digest(observation),
                                          'mask1': invalid, 'mask2': mask2.tolist()})
            with torch.no_grad():
                a1, a2 = model.eval_action(observation, invalid, mask2)
            a1, a2 = int(a1.item()), int(a2.item())
            selected = env.pairs[a1][0]
            if selected is None:
                raise RuntimeError('Policy selected an empty queue position')
            instrumentation.selected = selected.job_id
            before = env.current_timestamp
            requested = 0
            if 0 < a2 <= delayMaxJobNum:
                # sorted() preserves the original running list's mutation semantics.
                running = sorted(env.running_jobs, key=lambda j: j.scheduled_time + j.request_time)
                requested = min(running[a2 - 1].scheduled_time + running[a2 - 1].request_time, before + 3600) - before
            elif a2 > delayMaxJobNum:
                requested = delayTimeList[a2 - delayMaxJobNum - 1]
            action = {'a1': a1, 'a2': a2, 'job_id': selected.job_id, 'decision_time': before,
                      'requested_delay_seconds': requested}
            trace['actions'].append(action)
            result = env.step(a1, a2)
            observation, r, done, _, _, _, running_num, green = result
            reward += r
            green_reward += green
            action.update(selected_scheduled_time=selected.scheduled_time,
                          actual_delay_seconds=selected.scheduled_time - before,
                          post_step_timestamp=env.current_timestamp)
            if done:
                break
        trace['jobs'] = [job_record(job) for job in jobs]
        if any(job['scheduled_time'] < 0 for job in trace['jobs']):
            raise RuntimeError('Episode done with unscheduled jobs')
        energy = physical_energy(env.cluster.PowerStruc.powerSlotLog, physical, trace['jobs'],
                                 env.cluster.statPower, min(j.submit_time for j in jobs))
        trace['physical_intervals'] = energy.pop('physical_intervals')
        metrics = {'average_bounded_slowdown': -reward, 'renewable_energy_utilization': green_reward,
                   'reward': green_reward + eta * reward, **energy}
        independent_bsld = sum(j['bounded_slowdown'] for j in trace['jobs']) / len(jobs)
        metrics['independent_average_bounded_slowdown'] = independent_bsld
        metrics['bsld_identity_passed'] = math.isclose(independent_bsld, -reward, rel_tol=1e-12, abs_tol=1e-10)
        metrics['reutil_identity_passed'] = green_reward == energy['independent_renewable_energy_utilization']
        metrics['physical_supply_unchanged'] = digest(physical.greenPowerList) == physical_before
        trace['metrics'] = metrics
        if not all(metrics[k] for k in ('energy_identity_passed', 'bsld_identity_passed',
                                       'reutil_identity_passed', 'physical_supply_unchanged')):
            raise RuntimeError('Independent physical/metric identity failed')
        trace['rng']['after'] = rng_snapshot(env)
        trace['forecast'] = proxy.metadata if proxy else {'condition': 'unwrapped physical'}
        trace['forecast_counters'] = proxy.counters if proxy else None
        trace['clipping'] = instrumentation.clipping
        row = {**metrics, 'wait_quantiles': quantiles([j['wait_seconds'] for j in trace['jobs']]),
               'bsld_quantiles': quantiles([j['bounded_slowdown'] for j in trace['jobs']]),
               'actions_count': len(trace['actions']),
               'action1_distribution': dict(Counter(str(a['a1']) for a in trace['actions'])),
               'action2_distribution': dict(Counter(str(a['a2']) for a in trace['actions'])),
               'requested_delay_quantiles': quantiles([a['requested_delay_seconds'] for a in trace['actions']], 'actions_within_episode'),
               'actual_delay_quantiles': quantiles([a['actual_delay_seconds'] for a in trace['actions']], 'actions_within_episode'),
               'backfill': instrumentation.backfill, 'forecast_clipping': instrumentation.clipping,
               'forecast_counters': proxy.counters if proxy else None,
               'scheduled_jobs': len(jobs), 'unscheduled_jobs': 0,
               'physically_incomplete_jobs_at_final_decision': sum(j['completion_timestamp'] > env.current_timestamp for j in trace['jobs']),
               'final_decision_timestamp': env.current_timestamp,
               'completion_definition': 'Baseline done means all jobs scheduled; remaining running jobs complete at recorded actual times.',
               'wall_seconds': time.monotonic() - begin, 'process_cpu_seconds': time.process_time() - cpu_begin,
               'process_memory': process_memory(), 'gpu': gpu_cost()}
        if reference:
            keys = ('average_bounded_slowdown', 'renewable_energy_utilization', 'reward')
            row['saved_baseline_metrics'] = {k: reference[k] for k in keys}
            row['saved_baseline_exact_match'] = all(row[k] == reference[k] for k in keys)
            if spec['condition'] == 'original' and not row['saved_baseline_exact_match']:
                raise RuntimeError('Unwrapped original does not reproduce audited paired baseline metrics exactly')
        return {'row': row, 'trace': trace}
    except BaseException as exc:
        trace['jobs'] = [job_record(job) for job in jobs]
        trace['rng']['after_failure'] = rng_snapshot(env)
        trace['failure'] = {'type': type(exc).__name__, 'message': str(exc),
                            'wall_seconds': time.monotonic() - begin, 'process_cpu_seconds': time.process_time() - cpu_begin,
                            'process_memory': process_memory(), 'gpu': gpu_cost()}
        exc.pilot_partial = trace
        raise
    finally:
        instrumentation.restore()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=600)
    parser.add_argument('--max-actions', type=int, default=4096)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--num-threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.num_threads != 1:
        parser.error('Frozen paired protocol uses one Torch thread')
    if Path.cwd().resolve() != ROOT.resolve():
        parser.error('Run from project root so author relative config/data paths are fixed')
    manifest = validate_frozen()
    sys.path.insert(0, str(ROOT))
    import torch
    from HPCSimPickJobs import HPCEnv
    from compare import load_policy
    from MARL import configure_runtime
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('Frozen CUDA evaluation requested but CUDA is unavailable')
    manifest['environment'] = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                               'cuda': torch.version.cuda, 'device': args.device, 'num_threads': args.num_threads,
                               'process_memory': process_memory(), 'gpu': gpu_cost()}
    models = {}
    def episode(spec, deadline, max_actions):
        seed = spec['seed']
        if seed not in models:
            configure_runtime(seed, args.device, args.num_threads)
            path = ROOT / manifest['model_roots'][str(seed)] / 'lublin_256/MARL'
            models[seed] = load_policy('MARL', str(path), args.device)
        env = HPCEnv(backfill=1, delay_semantics='author')
        env.my_init(str(ROOT / 'data/lublin_256.swf'))
        reference = next(r for r in manifest['reference_rows'][str(seed)]
                         if r['algorithm'] == 'MARL' and r['start'] == spec['start'])
        return run_episode(spec, env, models[seed], deadline, max_actions, reference)
    summary = execute_matrix(args.output_dir, manifest, episode, args.timeout_seconds, args.max_actions,
                             final_check=lambda: _check_frozen(manifest))
    print(json.dumps({'event': 'pilot_finished', 'status': summary['status'], 'output': str(args.output_dir),
                      'episodes_completed': summary['episodes_completed'], 'h1': summary['h1']}), flush=True)
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
