"""Failure fixtures for the read-only frozen-pilot evidence auditor."""

import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from greenPower import green_power
from scripts.analyze_forecast_pilot import (
    AuditError, audit_episode, audit_run, canonical_hash, file_hash,
    read_trace, summarize_h1, validate_frozen, validate_protocol,
)


GATE_KEYS = ("actions", "events", "schedule", "jobs", "metrics", "rng")


def protocol_fixture():
    manifest = {"seeds": [0, 1, 2], "windows": [
        {"start": 6567, "iteration": 1, "sequence_length": 1024},
        {"start": 919, "iteration": 3, "sequence_length": 1024}],
        "eta": .002, "backfill": 1, "delay_semantics": "author"}
    rows, gates, stress_rows = [], [], []
    for seed in range(3):
        for window in manifest["windows"]:
            hashes = {name: "a" * 64 for name in GATE_KEYS}
            gates.append({"seed": seed, "start": window["start"], "passed": True,
                          "equal": {name: True for name in GATE_KEYS}, "tolerance": "exact",
                          "sha256": {name: {"original": value, "zero": value}
                                     for name, value in hashes.items()}})
            for condition, bias in (("original", None), ("zero", 0), ("plus20", .2)):
                target = stress_rows if condition == "plus20" else rows
                target.append({"seed": seed, **window, "condition": condition, "bias": bias,
                               "policy_seed": (seed + 1000003 * window["iteration"] + 10007 * 5) % 2**32,
                               "gate_sha256": hashes,
                               "renewable_energy_utilization": .8 if condition != "plus20" else .7})
    rows.extend(stress_rows)
    state = {"status": "completed", "manifest": manifest, "rows": rows,
             "episodes_started": 18, "episodes_completed": 18,
             "budget": {"planned_episode_starts": 18, "episode_cap": 18},
             "all_zero_gates_passed": True, "frozen_inputs_verified_after": True,
             "zero_gates": gates}
    return state, rows


def episode_fixture():
    physical = green_power.__new__(green_power)
    physical.greenPowerList, physical.greenWin = [5] * 24, 24
    jobs = []
    for job_id, submit, scheduled in ((1, 0, 0), (2, 5, 10)):
        jobs.append({"job_id": job_id, "submit_time": submit, "scheduled_time": scheduled,
                     "run_time": 10, "request_time": 10, "power": 10,
                     "request_number_of_processors": 1, "allocated_machine_ids": [0],
                     "completion_timestamp": scheduled + 10, "wait_seconds": scheduled - submit,
                     "bounded_slowdown": 1 if job_id == 1 else 1.5})
    metrics = {"average_bounded_slowdown": 1.25, "renewable_energy_utilization": .5,
               "reward": .4975, "total_energy_j": 200, "green_energy_j": 100,
               "brown_energy_j": 100, "log_start": 0, "completion_timestamp": 20,
               "log_horizon_seconds": 20, "makespan_seconds": 20,
               "initial_idle_gap_seconds": 0, "initial_idle_gap_energy_j": 0,
               "energy_identity_expected_j": 200, "independent_average_bounded_slowdown": 1.25,
               "independent_renewable_energy_utilization": .5,
               "energy_identity_passed": True, "bsld_identity_passed": True,
               "reutil_identity_passed": True, "physical_supply_unchanged": True}
    schedule = [{"job_id": j["job_id"], "time": j["scheduled_time"], "processors": 1,
                 "machine_ids": [0], "backfilled": False} for j in jobs]
    trace = {"jobs": jobs, "schedule": schedule, "actions": [], "rng": {},
             "events": [{"method": "greenPower.getGreenPowerSlot", "time": 0,
                         "raw_slots": physical.getGreenPowerSlot(0)}],
             "metrics": metrics, "forecast": {"condition": "unwrapped physical"},
             "physical_intervals": [{"start": 0, "end": 20, "power_w": 10,
                 "total_energy_j": 200, "green_energy_j": 100, "brown_energy_j": 100}]}
    row = {**metrics, "condition": "original", "bias": None, "scheduled_jobs": 2,
           "unscheduled_jobs": 0, "actions_count": 0,
           "wait_quantiles": {"method": "linear", "level": "jobs_within_episode", "count": 2,
               "p50": 2.5, "p95": 4.75, "p99": 4.95, "maximum": 5},
           "bsld_quantiles": {"method": "linear", "level": "jobs_within_episode", "count": 2,
               "p50": 1.25, "p95": 1.475, "p99": 1.495, "maximum": 1.5}}
    return trace, row, physical


class ForecastPilotAnalysisTests(unittest.TestCase):
    def test_protocol_requires_completed_unique_eighteen_episode_matrix(self):
        state, rows = protocol_fixture()
        validate_protocol(state, rows)
        for mutation in ("failed", "missing", "duplicate", "unverified_inputs"):
            altered = copy.deepcopy(state)
            selected = altered["rows"]
            if mutation == "failed":
                altered["status"] = "failed"
            elif mutation == "missing":
                selected.pop()
            elif mutation == "duplicate":
                selected[-1] = copy.deepcopy(selected[0])
            else:
                altered["frozen_inputs_verified_after"] = False
            with self.subTest(mutation=mutation), self.assertRaises(AuditError):
                validate_protocol(altered, selected)

    def test_protocol_checks_each_exact_gate_against_row_category_hashes(self):
        state, rows = protocol_fixture()
        state["zero_gates"][0]["sha256"]["actions"]["zero"] = "b" * 64
        with self.assertRaises(AuditError):
            validate_protocol(state, rows)

    def test_protocol_rejects_self_consistent_unregistered_window_or_iteration(self):
        for mutation in ("window", "iteration"):
            state, rows = protocol_fixture()
            window = state["manifest"]["windows"][0]
            if mutation == "window":
                window["start"] = 6568  # Still disjoint from the second window.
                for gate in state["zero_gates"]:
                    if gate["start"] == 6567:
                        gate["start"] = 6568
            else:
                window["iteration"] = 2
            for row in rows:
                if row["start"] == 6567:
                    row.update(window)
                    row["policy_seed"] = (row["seed"] + 1000003 * row["iteration"] + 10007 * 5) % 2**32
            with self.subTest(mutation=mutation), self.assertRaises(AuditError):
                validate_protocol(state, rows)

    def test_protocol_rejects_stress_before_all_gates_or_any_runner_order_change(self):
        for mutation in ("early_stress", "reversed_pair", "gate_window_order", "stress_window_order"):
            state, rows = protocol_fixture()
            if mutation == "early_stress":
                rows[1], rows[12] = rows[12], rows[1]
            elif mutation == "reversed_pair":
                rows[0], rows[1] = rows[1], rows[0]
            elif mutation == "gate_window_order":
                rows[:4] = rows[2:4] + rows[:2]
            else:
                rows[12], rows[13] = rows[13], rows[12]
            with self.subTest(mutation=mutation), self.assertRaises(AuditError):
                validate_protocol(state, rows)

    def test_frozen_file_validation_rejects_changed_bytes_and_path_escape(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / "source.py").write_text("original", encoding="utf-8")
            manifest = {"frozen_sha256": {"source.py": file_hash(root / "source.py")}}
            self.assertEqual(validate_frozen(root, manifest)["source.py"], manifest["frozen_sha256"]["source.py"])
            (root / "source.py").write_text("changed", encoding="utf-8")
            with self.assertRaises(AuditError):
                validate_frozen(root, manifest)
            with self.assertRaises(AuditError):
                validate_frozen(root, {"frozen_sha256": {"../outside": "a" * 64}})

    def test_trace_rejects_canonical_content_or_category_hash_mismatch(self):
        trace, row, _ = episode_fixture()
        row["trace_content_sha256"] = canonical_hash(trace)
        row["gate_sha256"] = {key: canonical_hash(trace[key]) for key in GATE_KEYS}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(trace, stream)
            row["trace_bytes"] = path.stat().st_size
            self.assertEqual(read_trace(path, row), trace)
            row["gate_sha256"]["jobs"] = "b" * 64
            with self.assertRaises(AuditError):
                read_trace(path, row)

    def test_independent_episode_energy_jobs_and_quantiles_are_verified(self):
        trace, row, physical = episode_fixture()
        result = audit_episode(trace, row, physical, expected_jobs=2, static_power=0)
        self.assertEqual(result["reconstructed_green_energy_j"], 100)
        self.assertEqual(result["jobs_verified"], 2)
        self.assertEqual(result["forecast_slot_values_verified"], 24)

    def test_matching_row_and_trace_cannot_hide_forecast_energy_leak(self):
        trace, row, physical = episode_fixture()
        trace["physical_intervals"][0].update(green_energy_j=120, brown_energy_j=80)
        for metrics in (trace["metrics"], row):
            metrics.update(green_energy_j=120, brown_energy_j=80,
                           renewable_energy_utilization=.6,
                           independent_renewable_energy_utilization=.6, reward=.5975)
        with self.assertRaises(AuditError):
            audit_episode(trace, row, physical, expected_jobs=2, static_power=0)

    def test_negative_wait_duplicate_schedule_and_bad_quantiles_are_rejected(self):
        for mutation in ("wait", "schedule", "quantile", "slot"):
            trace, row, physical = episode_fixture()
            if mutation == "wait":
                trace["jobs"][0]["scheduled_time"] = -1
            elif mutation == "schedule":
                trace["schedule"][1] = trace["schedule"][0].copy()
            elif mutation == "quantile":
                row["wait_quantiles"]["p95"] = 999
            else:
                trace["events"][0]["raw_slots"][1]["power"] = 6
            with self.subTest(mutation=mutation), self.assertRaises(AuditError):
                audit_episode(trace, row, physical, expected_jobs=2, static_power=0)

    def test_h1_requires_two_pairs_per_seed_and_all_three_strict_negative_means(self):
        _, rows = protocol_fixture()
        self.assertTrue(summarize_h1(rows)["supported"])
        next(r for r in rows if r["seed"] == 2 and r["start"] == 919 and r["condition"] == "plus20")["renewable_energy_utilization"] = .95
        self.assertFalse(summarize_h1(rows)["supported"])
        rows.pop()
        with self.assertRaises(AuditError):
            summarize_h1(rows)

    def test_partial_run_remains_read_only_without_effect_claim(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            run = root / "results" / "partial"
            run.mkdir(parents=True)
            state = {"status": "running", "manifest": {}, "rows": []}
            path = run / "state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            before = path.read_bytes()
            result = audit_run(run, root=root)
            self.assertFalse(result["proof"]["protocol_valid"])
            self.assertIsNone(result["summary"]["h1"])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
