"""Forecast-only perturbations must preserve the author's physical energy path."""

from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from HPCSimPickJobs import HPCEnv, JOB_FEATURES, MAX_QUEUE_SIZE, MAX_GREEN, run_win
from cluster import Cluster, Machine
from greenPower import green_power
from job import Job
from PowerStruc import power_struc
from scripts.forecast_pilot_env import ForecastPowerProxy, install_forecast


def physical_supply(values=(20, 5, 30), window=3):
    physical = green_power.__new__(green_power)
    physical.greenPowerList = list(values)
    physical.greenWin = window
    return physical


def minimal_env(values=None, current_time=3500):
    """Use real author methods with a small, hand-set physical trace."""
    physical = physical_supply(values or [10] * 30, window=24)
    cluster = Cluster.__new__(Cluster)
    cluster.total_node = cluster.free_node = 2
    cluster.used_node = 0
    cluster.num_procs_per_node = 1
    cluster.all_nodes = [Machine(0), Machine(1)]
    cluster.statPower = 0
    cluster.PowerStruc = power_struc(0)
    cluster.greenPower = physical
    cluster.green_win = 24
    env = HPCEnv.__new__(HPCEnv)
    env.cluster = cluster
    env.current_timestamp = current_time
    env.job_queue = []
    env.running_jobs = []
    env.loads = SimpleNamespace(max_exec_time=7200, max_procs=2)
    return env


def make_job():
    fields = [1, 3500, 0, 200, 1, 0, 0, 1, 200, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    job = Job(" ".join(str(value) for value in fields))
    job.power = 12
    return job


class ForecastPowerProxyTests(unittest.TestCase):
    def test_zero_bias_matches_original_outputs_exactly(self):
        physical = physical_supply()
        proxy = ForecastPowerProxy(physical, 0, lambda: 3500)
        for time in (0, 3500):
            self.assertEqual(proxy.getGreenPowerSlot(time), physical.getGreenPowerSlot(time))
        for args in ((30, 3500, 3700, 0, 2), (30, 3500, 14400, 0, 2)):
            self.assertEqual(proxy.getGreenUtiliEstimate(*args), physical.getGreenUtiliEstimate(*args))
        self.assertEqual(proxy.counters["slot_calls"], 2)
        self.assertEqual(proxy.counters["estimate_calls"], 2)
        self.assertEqual(proxy.counters["biased_future_segments"], 0)

    def test_zero_bias_delegates_even_overridden_methods_without_reading_clock(self):
        class Original(green_power):
            def getGreenPowerSlot(self, currentTime):
                return slot_result

            def getGreenUtiliEstimate(self, *args):
                return estimate_result

        def unread_clock():
            raise AssertionError("zero bias must not inspect an extra clock")

        slot_result, estimate_result = object(), object()
        proxy = ForecastPowerProxy(Original.__new__(Original), 0, unread_clock)
        self.assertIs(proxy.getGreenPowerSlot(10), slot_result)
        self.assertIs(proxy.getGreenUtiliEstimate(1, 2, 3, 4, 5), estimate_result)

    def test_current_partial_hour_stays_exact_and_future_hours_are_biased(self):
        proxy = ForecastPowerProxy(physical_supply(), .2, lambda: 3500)
        self.assertEqual(proxy.getGreenPowerSlot(3500), [
            {"lastTime": 100, "power": 20},
            {"lastTime": 3600, "power": 6},
            {"lastTime": 3600, "power": 36},
        ])
        self.assertEqual(proxy.getGreenUtiliEstimate(30, 3500, 3700, 0, 2), 2600)
        self.assertEqual(proxy.counters["biased_slot_segments"], 2)
        self.assertEqual(proxy.counters["biased_estimate_segments"], 1)

    def test_future_target_is_biased_when_window_wraps_to_current_source_hour(self):
        proxy = ForecastPowerProxy(physical_supply((10, 20, 999), window=2), .2, lambda: 3500)
        # Target hour 2 wraps to source hour 0, but remains a future forecast.
        self.assertEqual(proxy.getGreenUtiliEstimate(100, 7200, 7300, 0, 1), 1200)
        self.assertEqual(proxy.counters["biased_estimate_segments"], 1)

    def test_clock_callback_tracks_the_observed_hour_as_environment_advances(self):
        clock = SimpleNamespace(now=3500)
        proxy = ForecastPowerProxy(physical_supply(), .2, lambda: clock.now)
        self.assertEqual(proxy.getGreenUtiliEstimate(30, 3600, 3700, 0, 2), 600)
        clock.now = 3600
        self.assertEqual(proxy.getGreenUtiliEstimate(30, 3600, 3700, 1, 2), 500)

    def test_fixed_schedule_physical_energy_remains_unchanged(self):
        physical = physical_supply((20, 5))
        proxy = ForecastPowerProxy(physical, .2, lambda: 0)
        power = power_struc(10)
        power.update(0, 1800, 20)
        power.update(3600, 5400, 20)
        self.assertEqual(proxy.getGreenUtili(30, 3500, 3700), 2500)
        self.assertEqual(proxy.getGreenPowerUtilization(power.powerSlotLog), .5)
        self.assertEqual(proxy.counters["estimate_calls"], 0)

    def test_forecast_calls_never_mutate_or_replace_physical_supply_list(self):
        physical = physical_supply()
        original_list = physical.greenPowerList
        proxy = ForecastPowerProxy(physical, .2, lambda: 3500)
        slots = proxy.getGreenPowerSlot(3500)
        slots[1]["power"] = 12345
        proxy.getGreenUtiliEstimate(100, 3500, 14400, 0, 2)
        self.assertIs(proxy.greenPowerList, original_list)
        self.assertEqual(original_list, [20, 5, 30])
        self.assertEqual(proxy.greenWin, 3)

    def test_only_nonempty_changed_future_segments_are_counted(self):
        proxy = ForecastPowerProxy(physical_supply((0, 0, 10)), .2, lambda: 0)
        proxy.getGreenPowerSlot(0)
        proxy.getGreenUtiliEstimate(100, 3600, 7200, 0, 2)
        self.assertEqual(proxy.counters["biased_slot_segments"], 1)
        self.assertEqual(proxy.counters["biased_estimate_segments"], 0)
        self.assertEqual(proxy.counters["biased_future_segments"], 1)
        counters = proxy.counters
        counters["slot_calls"] = 999
        self.assertEqual(proxy.counters["slot_calls"], 1)

    def test_bias_smaller_than_float_precision_does_not_count_unchanged_supply(self):
        proxy = ForecastPowerProxy(physical_supply(), 1e-20, lambda: 0)
        self.assertEqual(proxy.getGreenPowerSlot(0)[1]["power"], 5)
        self.assertEqual(proxy.getGreenUtiliEstimate(100, 3600, 3700, 0, 2), 500)
        self.assertEqual(proxy.counters["biased_future_segments"], 0)

    def test_invalid_biases_are_rejected_and_finite_decreases_are_supported(self):
        for bias in (-1, -2, float("nan"), float("inf"), -float("inf")):
            with self.subTest(bias=bias), self.assertRaises(ValueError):
                ForecastPowerProxy(physical_supply(), bias, lambda: 0)
        proxy = ForecastPowerProxy(physical_supply(), -.5, lambda: 0)
        self.assertEqual(proxy.getGreenPowerSlot(0)[1]["power"], 2.5)
        self.assertEqual(proxy.metadata["future_multiplier"], .5)

    def test_install_rejects_nested_proxy_without_changing_existing_forecast(self):
        env = minimal_env()
        proxy = install_forecast(env, .2)
        with self.assertRaises(ValueError):
            install_forecast(env, .2)
        self.assertIs(env.cluster.greenPower, proxy)
        self.assertEqual(proxy.getGreenPowerSlot(3500)[1]["power"], 12)
        self.assertEqual(proxy.metadata["future_multiplier"], 1.2)

    def test_install_routes_job_feature_backfill_and_24_slot_observation(self):
        env = minimal_env()
        job = make_job()
        physical = env.cluster.greenPower
        self.assertEqual(env.cluster.getGreenJobState(job, 3500, []), (1, 1 / 6))
        # A longer job is rejected at 10 W but accepted with the future 12 W forecast.
        long_job = make_job()
        long_job.request_time = 30000
        self.assertFalse(env.cluster.backfill_check([], long_job, 3500, 1))
        proxy = install_forecast(env, .2)
        self.assertIs(env.cluster.greenPower, proxy)
        self.assertIs(proxy.physical, physical)
        self.assertEqual(env.cluster.getGreenJobState(job, 3500, []), (1, 1 / 12))
        self.assertTrue(env.cluster.backfill_check([], long_job, 3500, 1))
        env.job_queue = [job]
        observation = env.build_observation()
        green_start = (MAX_QUEUE_SIZE + run_win) * JOB_FEATURES
        self.assertEqual(observation[green_start + 1], 10 / MAX_GREEN)
        self.assertEqual(observation[green_start + JOB_FEATURES + 1], 12 / MAX_GREEN)
        self.assertEqual(proxy.counters["slot_calls"], 1)
        self.assertEqual(proxy.counters["estimate_calls"], 6)
        self.assertGreater(proxy.counters["biased_future_segments"], 0)

    def test_install_and_forecast_do_not_consume_global_random_streams(self):
        for bias in (0, .2):
            with self.subTest(bias=bias):
                env = minimal_env()
                py_state = random.getstate()
                numpy_state = np.random.get_state()
                torch_state = torch.get_rng_state().clone()
                proxy = install_forecast(env, bias)
                proxy.getGreenPowerSlot(3500)
                proxy.getGreenUtiliEstimate(20, 3500, 10000, 0, 23)
                proxy.getGreenUtili(20, 3500, 10000)
                proxy.getGreenPowerUtilization([
                    {"timeSlot": 3500, "power": 20}, {"timeSlot": 10000, "power": 0}])
                self.assertEqual(random.getstate(), py_state)
                actual_numpy = np.random.get_state()
                self.assertEqual(actual_numpy[0], numpy_state[0])
                np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
                self.assertEqual(actual_numpy[2:], numpy_state[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))


if __name__ == "__main__":
    unittest.main()
