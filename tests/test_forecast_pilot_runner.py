"""Bounded runner checks; fixtures contain three jobs, never a scientific window."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import shutil

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'scripts/run_forecast_pilot.py'
if PATH.exists():
    spec = importlib.util.spec_from_file_location('forecast_runner', PATH)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
else:
    pilot = None


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(pilot, 'The forecast runner must exist')

    def manifest(self):
        return {'seeds': [0, 1, 2], 'windows': [
            {'start': 6567, 'iteration': 1, 'sequence_length': 1024},
            {'start': 919, 'iteration': 3, 'sequence_length': 1024}]}

    def result(self, spec, deadline, max_actions):
        trace = {key: [1, 2] for key in ('actions', 'events', 'schedule', 'jobs')}
        trace.update(metrics={'renewable_energy_utilization': .7}, rng={'before': 'a', 'after': 'b'})
        row = {**spec, 'renewable_energy_utilization': .7, 'wall_seconds': .1,
               'process_cpu_seconds': .05}
        return {'row': row, 'trace': trace}

    def test_selection_skips_overlap_without_renumbering_original_iteration(self):
        windows = pilot.select_windows([6567, 7146, 919, 4498], 1024)
        self.assertEqual([(w['start'], w['iteration']) for w in windows], [(6567, 1), (919, 3)])
        self.assertEqual(pilot.evaluation_seed(2, 3), 3050046)

    def test_selection_accepts_touching_half_open_windows(self):
        self.assertEqual([w['start'] for w in pilot.select_windows([10, 15, 20], 10)], [10, 20])

    def test_zero_gate_rejects_rng_consumption_and_each_other_category(self):
        import copy
        original = self.result({}, None, None)['trace']
        self.assertTrue(pilot.zero_gate(original, copy.deepcopy(original))['passed'])
        for key in ('actions', 'events', 'schedule', 'jobs', 'metrics', 'rng'):
            changed = copy.deepcopy(original)
            changed[key] = 'different'
            gate = pilot.zero_gate(original, changed)
            self.assertFalse(gate['passed'], key)
            self.assertFalse(gate['equal'][key])

    def test_all_six_gates_precede_any_stress_and_zero_is_reused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp) / 'pilot'
            calls = []
            def episode(spec, deadline, max_actions):
                calls.append(spec['condition'])
                self.assertTrue((output / 'state.json').exists())
                state = json.loads((output / 'state.json').read_text())
                self.assertEqual(state['episodes_started'], len(calls))
                return self.result(spec, deadline, max_actions)
            summary = pilot.execute_matrix(output, self.manifest(), episode, lock_path=Path(temp) / 'lock')
            self.assertEqual(calls[:12], ['original', 'zero'] * 6)
            self.assertEqual(calls[12:], ['plus20'] * 6)
            self.assertEqual(summary['episodes_completed'], 18)
            self.assertEqual(len(summary['zero_gates']), 6)
            self.assertEqual(summary['first_pair_cost']['episodes'], 2)
            self.assertEqual(len(json.loads((output / 'rows.json').read_text())), 18)

    def test_zero_failure_preserves_both_traces_and_never_runs_stress(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp) / 'pilot'
            def episode(spec, deadline, max_actions):
                result = self.result(spec, deadline, max_actions)
                if spec['condition'] == 'zero':
                    result['trace']['rng'] = 'consumed'
                return result
            summary = pilot.execute_matrix(output, self.manifest(), episode, lock_path=Path(temp) / 'lock')
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['episodes_started'], 2)
            self.assertEqual(len(list((output / 'traces').glob('*.gz'))), 2)
            self.assertFalse(summary['zero_gates'][0]['passed'])

    def test_budget_refuses_over_cap_before_starting_any_episode(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            manifest = self.manifest()
            manifest['seeds'].append(3)
            with self.assertRaises(ValueError):
                pilot.execute_matrix(Path(temp) / 'pilot', manifest, self.result, lock_path=Path(temp) / 'lock')
            self.assertFalse((Path(temp) / 'pilot').exists())

    def test_existing_partial_directory_and_active_lock_are_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp) / 'pilot'
            output.mkdir()
            (output / 'state.json').write_text('{"status":"failed"}')
            with self.assertRaises(FileExistsError):
                pilot.execute_matrix(output, self.manifest(), self.result, lock_path=Path(temp) / 'lock')
            (Path(temp) / 'lock').write_text('active')
            with self.assertRaises(FileExistsError):
                pilot.execute_matrix(Path(temp) / 'other', self.manifest(), self.result, lock_path=Path(temp) / 'lock')
            self.assertFalse((Path(temp) / 'other').exists())

    def test_default_lock_excludes_existing_baseline_research_process(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            (Path(temp) / '.research.lock').write_text('baseline running')
            with patch.object(pilot, 'ROOT', Path(temp)):
                with self.assertRaises(FileExistsError):
                    pilot.execute_matrix(Path(temp) / 'pilot', self.manifest(), self.result)
            self.assertFalse((Path(temp) / 'pilot').exists())

    def test_timeout_keeps_failure_record_without_starting_episode(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            ticks = iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            summary = pilot.execute_matrix(Path(temp) / 'pilot', self.manifest(), self.result,
                                           timeout_seconds=.5, clock=lambda: next(ticks),
                                           lock_path=Path(temp) / 'lock')
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['episodes_started'], 0)
            self.assertIn('Timeout', summary['error_type'])

    def test_nonfinite_timeout_is_rejected_without_starting(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            for value in (float('nan'), float('inf'), -float('inf')):
                output = Path(temp) / str(value)
                with self.assertRaises(ValueError):
                    pilot.execute_matrix(output, self.manifest(), self.result,
                                         timeout_seconds=value, lock_path=Path(temp) / 'lock')
                self.assertFalse(output.exists())

    def test_final_episode_overrun_is_saved_and_never_marked_completed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            current = [0.]
            calls = [0]
            def episode(spec, deadline, max_actions):
                calls[0] += 1
                if calls[0] == 18:
                    current[0] = 11.
                return self.result(spec, deadline, max_actions)
            summary = pilot.execute_matrix(Path(temp) / 'pilot', self.manifest(), episode,
                                           timeout_seconds=10, clock=lambda: current[0],
                                           lock_path=Path(temp) / 'lock')
            self.assertEqual(summary['episodes_completed'], 18)
            self.assertEqual(summary['status'], 'failed')
            self.assertFalse(summary['h1']['protocol_valid'])
            self.assertIn('Timeout', summary['error_type'])

    def test_failed_final_freeze_invalidates_h1_but_retains_descriptive_negative_signs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            def episode(spec, deadline, max_actions):
                result = self.result(spec, deadline, max_actions)
                if spec['condition'] == 'plus20':
                    result['row']['renewable_energy_utilization'] = .6
                return result
            def final_check():
                raise ValueError('Frozen weight changed')
            summary = pilot.execute_matrix(Path(temp) / 'pilot', self.manifest(), episode,
                                           final_check=final_check, lock_path=Path(temp) / 'lock')
            self.assertEqual(summary['status'], 'failed')
            self.assertTrue(summary['h1']['complete'])
            self.assertFalse(summary['h1']['supported'])
            self.assertFalse(summary['h1']['protocol_valid'])
            self.assertTrue(summary['h1']['descriptive_all_seed_means_negative'])

    def test_h1_requires_every_training_seed_negative_and_all_pairs_present(self):
        rows = []
        for seed, stressed in [(0, .6), (1, .8), (2, .6)]:
            for start in (6567, 919):
                for condition, value in [('zero', .7), ('plus20', stressed)]:
                    rows.append({'seed': seed, 'start': start, 'condition': condition,
                                 'renewable_energy_utilization': value})
        result = pilot.summarize_h1(rows)
        self.assertTrue(result['complete'])
        self.assertFalse(result['supported'])
        self.assertGreater(result['per_seed'][1]['mean_difference'], 0)
        self.assertFalse(pilot.summarize_h1(rows[:-1])['complete'])

    def test_quantiles_are_linear_within_episode_job_level(self):
        values = pilot.quantiles([0, 10, 20, 30])
        self.assertEqual(values['p50'], 15)
        self.assertAlmostEqual(values['p95'], 28.5)
        self.assertEqual(values['method'], 'linear')
        self.assertEqual(values['level'], 'jobs_within_episode')

    def test_action_quantiles_identify_actions_instead_of_jobs(self):
        self.assertEqual(pilot.quantiles([1, 3], level='actions_within_episode')['level'],
                         'actions_within_episode')

    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows process counters')
    def test_memory_cost_reports_os_peak_without_optional_psutil(self):
        memory = pilot.process_memory()
        self.assertGreater(memory['rss_bytes'], 0)
        self.assertGreaterEqual(memory['peak_rss_bytes'], memory['rss_bytes'])

    def test_first_pair_accounts_for_setup_using_controller_clock(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            clock_now = [10.0]
            def episode(spec, deadline, max_actions):
                # Advance the runner's public clock through separate setup and
                # simulation phases. A 4ms sleep can fit entirely within one
                # Windows GetTickCount64 tick (15.625ms), so it cannot establish
                # this accounting contract reliably.
                clock_now[0] += .75
                result = self.result(spec, deadline, max_actions)
                clock_now[0] += .125
                result['row']['wall_seconds'] = .125
                return result
            summary = pilot.execute_matrix(Path(temp) / 'pilot', self.manifest(), episode,
                                           clock=lambda: clock_now[0], lock_path=Path(temp) / 'lock')
            cost = summary['first_pair_cost']
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(cost['episodes'], 2)
            self.assertEqual(cost['pair_wall_seconds'], 1.75)
            self.assertEqual(cost['simulation_wall_seconds_sum'], .25)
            self.assertEqual(cost['projected_18_episode_seconds'], 15.75)

    def test_small_real_environment_zero_path_has_identical_actions_events_jobs_and_rng(self):
        from HPCSimPickJobs import HPCEnv
        from cluster import Cluster
        from job import Job, Workloads
        import torch

        class FirstAvailablePolicy:
            device = torch.device('cpu')
            def eval_action(self, observation, mask1, mask2):
                # Consume the same genuine Torch RNG in every condition.
                torch.rand(1)
                return torch.tensor(mask1.index(0)), torch.tensor(0)

        def environment():
            env = HPCEnv(backfill=1, delay_semantics='author')
            env.loads = Workloads.__new__(Workloads)
            env.loads.all_jobs = []
            for job_id, submit, runtime, processors in [(1, 0, 20, 1), (2, 1, 10, 2), (3, 2, 5, 1)]:
                job = Job(' '.join(str(v) for v in [job_id, submit, 0, runtime, processors, 0, 0,
                                                   processors, runtime, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
                job.power = 20
                env.loads.all_jobs.append(job)
            env.loads.max_exec_time = 20
            env.loads.max_procs = 2
            env.cluster = Cluster('fixture', 2, 1, 8, 50, 24, 200, 7200)
            return env

        results = []
        for condition, bias in [('original', None), ('zero', 0.), ('plus20', .2)]:
            spec = {'seed': 0, 'policy_seed': 123, 'start': 0, 'iteration': 1,
                    'sequence_length': 3, 'condition': condition, 'bias': bias}
            results.append(pilot.run_episode(spec, environment(), FirstAvailablePolicy(), time.monotonic() + 30, 10))
        self.assertTrue(pilot.zero_gate(results[0]['trace'], results[1]['trace'])['passed'])
        for result in results:
            self.assertEqual(result['row']['scheduled_jobs'], 3)
            self.assertEqual(result['row']['unscheduled_jobs'], 0)
            self.assertTrue(result['row']['energy_identity_passed'])
            self.assertTrue(result['row']['physical_supply_unchanged'])
            self.assertEqual(result['row']['actual_delay_quantiles']['level'], 'actions_within_episode')
            self.assertGreater(result['row']['physically_incomplete_jobs_at_final_decision'], 0)
            self.assertEqual(result['row']['backfill']['actual_scheduled'], 1)
            self.assertGreaterEqual(result['row']['backfill']['candidate_checks'], 1)
            self.assertEqual([(j['job_id'], j['scheduled_time']) for j in result['trace']['jobs']],
                             [(1, 0), (2, 20), (3, 2)])
        self.assertGreater(results[2]['row']['forecast_counters']['biased_future_segments'], 0)
        with self.assertRaises(RuntimeError) as failure:
            pilot.run_episode(spec, environment(), FirstAvailablePolicy(), time.monotonic() + 30, 1)
        self.assertIn('Maximum actions', str(failure.exception))
        self.assertEqual(len(failure.exception.pilot_partial['actions']), 1)

    def test_physical_energy_keeps_baseline_log_horizon_and_initial_gap_separate(self):
        from greenPower import green_power
        green = green_power.__new__(green_power)
        green.greenPowerList = [20, 5]
        slots = [{'timeSlot': 1800, 'power': 30}, {'timeSlot': 3600, 'power': 10},
                 {'timeSlot': 5400, 'power': 10}]
        jobs = [{'power': 20, 'run_time': 1800}]
        result = pilot.physical_energy(slots, green, jobs, 10, first_arrival=1000)
        self.assertEqual(result['total_energy_j'], 72000)
        self.assertEqual(result['green_energy_j'], 45000)
        self.assertEqual(result['brown_energy_j'], 27000)
        self.assertEqual(result['initial_idle_gap_energy_j'], 8000)
        self.assertEqual(result['makespan_seconds'], 4400)
        self.assertTrue(result['energy_identity_passed'])

    def test_config_mismatch_is_refused_before_importing_simulator(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            config = Path(temp) / 'config.ini'
            config.write_text('[GAS-MARL setting]\nMAX_QUEUE_SIZE=16\nrun_win=64\ngreen_win=24\neta=.002\n')
            with self.assertRaises(ValueError):
                pilot.validate_config(config)

    def test_consumed_evidence_copy_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            target = Path(temp) / 'research/evidence/multiseed'
            target.mkdir(parents=True)
            for path in (ROOT / 'research/evidence/multiseed').glob('seed*_author_*.json'):
                shutil.copyfile(path, target / path.name)
            shutil.copyfile(ROOT / 'research/evidence/multiseed/evidence_sha256.json', target / 'evidence_sha256.json')
            audit = json.loads((ROOT / 'research/evidence/multiseed/training_verification.json').read_text())
            (target / 'seed0_author_rows.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'evidence'):
                pilot.validate_evaluation_evidence(Path(temp), audit)

    def test_even_hash_consistent_metadata_cannot_redirect_final_audited_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            target = Path(temp) / 'research/evidence/multiseed'
            target.mkdir(parents=True)
            for path in (ROOT / 'research/evidence/multiseed').glob('seed*_author_*.json'):
                shutil.copyfile(path, target / path.name)
            hashes = json.loads((ROOT / 'research/evidence/multiseed/evidence_sha256.json').read_text())
            path = target / 'seed1_author_metadata.json'
            metadata = json.loads(path.read_text())
            metadata['model_root'] = 'arbitrary_models'
            path.write_text(json.dumps(metadata))
            hashes[path.name] = __import__('hashlib').sha256(path.read_bytes()).hexdigest()
            (target / 'evidence_sha256.json').write_text(json.dumps(hashes))
            audit = json.loads((ROOT / 'research/evidence/multiseed/training_verification.json').read_text())
            with self.assertRaisesRegex(ValueError, 'model'):
                pilot.validate_evaluation_evidence(Path(temp), audit)


if __name__ == '__main__':
    unittest.main()
