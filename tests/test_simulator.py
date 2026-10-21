"""Behavior checks for the author's simulator and the opt-in deadline fix."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from HPCSimPickJobs import HPCEnv
from cluster import Cluster
from greenPower import green_power
from job import Job
from PowerStruc import power_struc


def make_job(job_id, submit_time=0, run_time=10, request_time=10,
             processors=1, power=20):
    fields = [job_id, submit_time, 0, run_time, processors, 0, 0,
              processors, request_time, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    job = Job(" ".join(str(value) for value in fields))
    job.power = power
    return job


def delay_fixture(semantics):
    # Real resource and power classes; the running job completes at 6000 s.
    env = HPCEnv(backfill=1, delay_semantics=semantics)
    env.cluster = Cluster("test", 2, 1, 8, 50, 24, 200, 7200)
    running = make_job(1, run_time=6000, request_time=6000)
    running.scheduled_time = 0
    running.allocated_machines = env.cluster.allocate(1, 1)
    env.cluster.PowerStruc.update(0, 6000, running.power)
    target = make_job(2, submit_time=100)
    env.current_timestamp = 100
    env.running_jobs = [running]
    env.job_queue = [target]
    env.loads = []
    return env, target


class Type2DeadlineTests(unittest.TestCase):
    def test_author_preserves_type2_event_overshoot(self):
        # The original event list is 3600, 6000; its 3700 deadline is skipped.
        env, target = delay_fixture("author")
        env.moveforward_green_backfilling_delay_action1(target, 1)
        self.assertEqual(env.current_timestamp, 6000)

    def test_paper_stops_at_type2_deadline_before_next_completion(self):
        env, target = delay_fixture("paper")
        env.moveforward_green_backfilling_delay_action1(target, 1)
        self.assertEqual(env.current_timestamp, 3700)
        self.assertTrue(env.cluster.can_allocated(target))

    def test_type3_fixed_deadline_is_unchanged(self):
        for semantics in ("author", "paper"):
            with self.subTest(semantics=semantics):
                env, target = delay_fixture(semantics)
                env.moveforward_green_backfilling_delay_action2(target, 300)
                self.assertEqual(env.current_timestamp, 400)

    def test_paper_keeps_waiting_for_resources_after_deadline(self):
        env, target = delay_fixture("paper")
        target.request_number_of_processors = 2
        self.assertFalse(env.cluster.can_allocated(target))
        env.moveforward_green_backfilling_delay_action1(target, 1)
        self.assertEqual(env.current_timestamp, 6000)
        self.assertTrue(env.cluster.can_allocated(target))

    def test_paper_processes_arrival_at_deadline(self):
        env, target = delay_fixture("paper")
        arriving = make_job(3, submit_time=3700)
        env.loads = [arriving]
        env.last_job_in_batch = 1
        env.moveforward_green_backfilling_delay_action1(target, 1)
        self.assertEqual(env.current_timestamp, 3700)
        self.assertIn(arriving, env.job_queue)
        self.assertEqual(env.next_arriving_job_idx, 1)


class DelayModeSelectionTests(unittest.TestCase):
    def test_environment_variable_selects_paper_behavior(self):
        with patch.dict(os.environ, {"GREEN_RL_DELAY_SEMANTICS": "paper"}):
            env = HPCEnv(backfill=1)
        self.assertEqual(env.delay_semantics, "paper")

    def test_explicit_author_overrides_paper_environment(self):
        with patch.dict(os.environ, {"GREEN_RL_DELAY_SEMANTICS": "paper"}):
            env = HPCEnv(backfill=1, delay_semantics="author")
        self.assertEqual(env.delay_semantics, "author")

    def test_unset_environment_defaults_to_author(self):
        with patch.dict(os.environ, {}, clear=True):
            env = HPCEnv(backfill=1)
        self.assertEqual(env.delay_semantics, "author")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            HPCEnv(delay_semantics="papre")


class SimulatorMetricTests(unittest.TestCase):
    def test_slowdown_uses_actual_runtime_and_ten_second_bound(self):
        env = HPCEnv()
        long_job = make_job(1, run_time=20, request_time=200)
        long_job.scheduled_time = 20
        short_job = make_job(2, run_time=2, request_time=200)
        short_job.scheduled_time = 0
        self.assertEqual(env.job_score(long_job), 2.0)
        self.assertEqual(env.job_score(short_job), 1.0)
        scores = {1: env.job_score(long_job), 2: env.job_score(short_job)}
        env.num_job_in_batch = 2
        env.post_process_score(scores)
        self.assertEqual(sum(scores.values()), 1.5)

    def test_resources_are_exclusive_until_release(self):
        cluster = Cluster("test", 2, 1, 8, 50, 24, 200, 7200)
        job = make_job(1, processors=2)
        self.assertTrue(cluster.can_allocated(job))
        allocated = cluster.allocate(1, 2)
        self.assertEqual(len(allocated), 2)
        self.assertFalse(cluster.can_allocated(make_job(2)))
        self.assertEqual(cluster.allocate(2, 1), [])
        self.assertEqual(cluster.free_node, 0)
        cluster.release(allocated)
        self.assertEqual(cluster.free_node, 2)
        self.assertTrue(cluster.can_allocated(job))

    def test_energy_integrates_hour_boundary_and_idle_gaps(self):
        # 0-1800: 30 W; 1800-3600: 10 W idle; 3600-5400: 30 W.
        # Renewable supply is 20 W then 5 W. Green/total = 63000/126000.
        power = power_struc(10)
        power.update(0, 1800, 20)
        power.update(3600, 5400, 20)
        green = green_power.__new__(green_power)
        green.greenPowerList = [20, 5]
        self.assertAlmostEqual(green.getGreenPowerUtilization(power.powerSlotLog), 0.5)
        # A constant 30 W load across the hour consumes 20*100+5*100 J green.
        self.assertEqual(green.getGreenUtili(30, 3500, 3700), 2500)


if __name__ == "__main__":
    unittest.main()
