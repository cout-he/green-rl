"""Read-only audit and compact evidence export for the frozen H1 pilot.

No policies are loaded and no episodes are run. Large gzip traces stay local;
each is read, authenticated, checked, and released before opening the next.
"""

import argparse
from collections import Counter
import configparser
import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
GATE_KEYS = ("actions", "events", "schedule", "jobs", "metrics", "rng")
CONDITIONS = {"original": None, "zero": 0, "plus20": .2}
REGISTERED_WINDOWS = ((6567, 1), (919, 3))


class AuditError(ValueError):
    """Evidence cannot support the registered protocol or reported identity."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _inside(root, name):
    root = Path(root).resolve()
    path = (root / str(name).replace("\\", "/")).resolve()
    require(path.is_relative_to(root), "Path escapes its evidence root: " + str(name))
    return path


def _close(actual, expected, label, absolute=1e-6):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=absolute),
            "%s: %r != independently derived %r" % (label, actual, expected))


def validate_frozen(root, manifest):
    checked = {}
    for name, expected in manifest["frozen_sha256"].items():
        path = _inside(root, name)
        require(path.is_file(), "Missing frozen file: " + name)
        actual = file_hash(path)
        require(actual == expected, "Frozen file changed: " + name)
        checked[path.relative_to(Path(root).resolve()).as_posix()] = actual
    require(bool(checked), "Frozen checksum manifest is empty")
    return checked


def validate_protocol(state, rows):
    require(state.get("status") == "completed", "Pilot is partial or failed; no effect claim is permitted")
    manifest = state["manifest"]
    require(manifest["seeds"] == [0, 1, 2], "Three fixed training seeds are required")
    windows = manifest["windows"]
    require(len(windows) == 2 and all(w["sequence_length"] == 1024 for w in windows),
            "Exactly two 1024-job windows are required")
    require([(w["start"], w["iteration"]) for w in windows] == list(REGISTERED_WINDOWS),
            "Windows or original iterations differ from the registered [(6567, 1), (919, 3)] design")
    require(manifest["eta"] == .002 and manifest["backfill"] == 1
            and manifest["delay_semantics"] == "author", "Frozen protocol parameters differ")
    require(state["episodes_started"] == state["episodes_completed"] == len(rows) == 18,
            "The completed episode matrix must contain exactly 18 rows and starts")
    require(state["budget"]["episode_cap"] == state["budget"]["planned_episode_starts"] == 18,
            "Episode budget differs from the registered cap")
    require(state.get("all_zero_gates_passed") is True
            and state.get("frozen_inputs_verified_after") is True,
            "Zero gates or final frozen-input check did not pass")
    expected_order = [(s, start, c) for s in manifest["seeds"] for start, _ in REGISTERED_WINDOWS
                      for c in ("original", "zero")]
    expected_order += [(s, start, "plus20") for s in manifest["seeds"] for start, _ in REGISTERED_WINDOWS]
    expected = set(expected_order)
    selected = {}
    window_map = {w["start"]: w for w in windows}
    for position, row in enumerate(rows):
        identity = (row["seed"], row["start"], row["condition"])
        require(identity in expected and identity not in selected, "Missing, unexpected, or duplicate episode identity")
        require(identity == expected_order[position],
                "Episode row order differs from the frozen runner: six original/zero pairs must precede six plus20 rows")
        window = window_map[row["start"]]
        require(row["sequence_length"] == 1024 and row["iteration"] == window["iteration"],
                "Row window identity differs from manifest")
        require(row["bias"] == CONDITIONS[row["condition"]], "Condition multiplier differs from H1")
        policy_seed = (row["seed"] + 1000003 * row["iteration"] + 10007 * 5) % 2**32
        require(row["policy_seed"] == policy_seed, "Paired action sampling seed differs")
        selected[identity] = row
    require(set(selected) == expected, "Incomplete episode matrix")
    gates = state["zero_gates"]
    require(len(gates) == 6, "Exactly six zero-error gates are required")
    gate_identities = set()
    for gate in gates:
        identity = (gate["seed"], gate["start"])
        require(identity not in gate_identities and (*identity, "original") in selected,
                "Duplicate or unexpected zero gate")
        gate_identities.add(identity)
        require(gate["passed"] is True and gate["tolerance"] == "exact", "Zero gate is not an exact pass")
        for key in GATE_KEYS:
            original = selected[(*identity, "original")]["gate_sha256"][key]
            zero = selected[(*identity, "zero")]["gate_sha256"][key]
            require(gate["equal"][key] is True and original == zero
                    and gate["sha256"][key] == {"original": original, "zero": zero},
                    "Zero gate category/hash mismatch: " + key)
    return selected


def read_trace(path, row):
    require(path.stat().st_size == row["trace_bytes"], "Compressed trace size mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    require(canonical_hash(trace) == row["trace_content_sha256"], "Canonical trace content hash mismatch")
    for key in GATE_KEYS:
        require(canonical_hash(trace[key]) == row["gate_sha256"][key], "Trace category hash mismatch: " + key)
    return trace


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _check_quantiles(saved, values, label, level="jobs_within_episode"):
    require(saved["method"] == "linear" and saved["level"] == level
            and saved["count"] == len(values), label + " quantile metadata mismatch")
    for key, probability in (("p50", .5), ("p95", .95), ("p99", .99)):
        _close(saved[key], _quantile(values, probability), label + " " + key, absolute=1e-10)
    _close(saved["maximum"], max(values), label + " maximum", absolute=1e-10)


def audit_episode(trace, row, physical, expected_jobs=1024, static_power=1600, eta=.002, input_jobs=None):
    jobs, schedule = trace["jobs"], trace["schedule"]
    require(len(jobs) == len(schedule) == expected_jobs and row["scheduled_jobs"] == expected_jobs
            and row["unscheduled_jobs"] == 0, "Unscheduled jobs or incorrect schedule count")
    by_id = {j["job_id"]: j for j in jobs}
    allocated = {s["job_id"]: s for s in schedule}
    require(len(by_id) == len(allocated) == expected_jobs and set(by_id) == set(allocated),
            "Duplicate or missing job/schedule identities")
    if input_jobs is not None:
        require(set(by_id) == set(input_jobs), "Job set differs from the original fixed input window")
    waits, slowdowns, transitions = [], [], []
    machines = {}
    for job in jobs:
        job_id = job["job_id"]
        start, run, submit = job["scheduled_time"], job["run_time"], job["submit_time"]
        require(start >= submit and start >= 0 and run > 0 and job["power"] >= 0,
                "Negative job wait, invalid runtime, or invalid physical power")
        wait, complete = start - submit, start + run
        slowdown = max(1., (wait + run) / max(run, 10))
        _close(job["wait_seconds"], wait, "Job wait", 1e-10)
        _close(job["completion_timestamp"], complete, "Job completion", 1e-10)
        _close(job["bounded_slowdown"], slowdown, "Job BSLD", 1e-10)
        record = allocated[job_id]
        require(record["time"] == start and record["processors"] == job["request_number_of_processors"]
                and record["machine_ids"] == job["allocated_machine_ids"] and record["machine_ids"]
                and len(set(record["machine_ids"])) == len(record["machine_ids"]),
                "Schedule time/resource record disagrees with job")
        for machine in record["machine_ids"]:
            machines.setdefault(machine, []).append((start, complete))
        if input_jobs is not None:
            for field, expected in input_jobs[job_id].items():
                require(job[field] == expected, "Physical job input changed: %s/%s" % (job_id, field))
        waits.append(wait)
        slowdowns.append(slowdown)
        transitions.extend(((start, job["power"]), (complete, -job["power"])))
    for intervals in machines.values():
        ordered = sorted(intervals)
        require(all(previous[1] <= after[0] for previous, after in zip(ordered, ordered[1:])),
                "A physical machine is allocated to overlapping jobs")
    metrics = trace["metrics"]
    for key, value in metrics.items():
        require(key in row and row[key] == value, "Row/trace metric mismatch: " + key)
    for key in ("energy_identity_passed", "bsld_identity_passed", "reutil_identity_passed", "physical_supply_unchanged"):
        require(metrics[key] is True, "Runner recorded failed physical identity: " + key)
    mean_bsld = sum(slowdowns) / expected_jobs
    _close(metrics["average_bounded_slowdown"], mean_bsld, "Independent mean BSLD", 1e-10)
    _close(metrics["independent_average_bounded_slowdown"], mean_bsld, "Saved independent BSLD", 1e-10)
    _check_quantiles(row["wait_quantiles"], waits, "Job waits")
    _check_quantiles(row["bsld_quantiles"], slowdowns, "Job BSLD")
    physical_intervals = trace["physical_intervals"]
    require(bool(physical_intervals), "Physical energy intervals are missing")
    changes = sorted(transitions)
    change_index, load = 0, static_power
    total, green, interval_brown = 0., 0., 0.
    previous_end = physical_intervals[0]["start"]
    for interval in physical_intervals:
        start, end, power = interval["start"], interval["end"], interval["power_w"]
        require(start == previous_end and end >= start and power >= 0,
                "Physical intervals have a gap, reversal, or negative power")
        while change_index < len(changes) and changes[change_index][0] <= start:
            load += changes[change_index][1]
            change_index += 1
        _close(power, load, "Interval power reconstructed from fixed jobs")
        energy = (end - start) * power
        used_green = physical.getGreenUtili(power, start, end)
        _close(interval["total_energy_j"], energy, "Interval total energy")
        _close(interval["green_energy_j"], used_green, "Original physical supply green energy")
        _close(interval["brown_energy_j"], energy - used_green, "Interval brown energy")
        total += energy
        green += used_green
        interval_brown += interval["brown_energy_j"]
        previous_end = end
    begin, end = physical_intervals[0]["start"], physical_intervals[-1]["end"]
    first_arrival = min(j["submit_time"] for j in jobs)
    require(begin == min(j["scheduled_time"] for j in jobs)
            and end == max(j["completion_timestamp"] for j in jobs), "Energy horizon differs from actual schedule")
    independent_total = sum(j["power"] * j["run_time"] for j in jobs) + static_power * (end - begin)
    _close(total, independent_total, "Job/static physical energy identity")
    reutil = green / total if total else 1
    expected_metrics = {"total_energy_j": total, "green_energy_j": green, "brown_energy_j": total - green,
                        "log_start": begin, "completion_timestamp": end, "log_horizon_seconds": end - begin,
                        "makespan_seconds": end - first_arrival, "initial_idle_gap_seconds": begin - first_arrival,
                        "initial_idle_gap_energy_j": static_power * (begin - first_arrival),
                        "energy_identity_expected_j": independent_total,
                        "renewable_energy_utilization": reutil,
                        "independent_renewable_energy_utilization": reutil,
                        "reward": reutil - eta * mean_bsld}
    for key, expected in expected_metrics.items():
        _close(metrics[key], expected, "Physical metric " + key)
    _close(interval_brown, total - green, "Summed interval brown energy")
    require(row["actions_count"] == len(trace["actions"]), "Action count mismatch")
    if trace["actions"]:
        for action_key, distribution_key in (("a1", "action1_distribution"), ("a2", "action2_distribution")):
            require(dict(Counter(str(a[action_key]) for a in trace["actions"])) == row[distribution_key],
                    "Action distribution mismatch")
        for field in ("requested", "actual"):
            _check_quantiles(row[field + "_delay_quantiles"],
                             [a[field + "_delay_seconds"] for a in trace["actions"]],
                             field + " delay", "actions_within_episode")
    slot_count = 0
    for event in trace["events"]:
        if event["method"] != "greenPower.getGreenPowerSlot":
            continue
        expected = physical.getGreenPowerSlot(event["time"])
        if row["condition"] == "plus20":
            expected = [dict(slot, power=slot["power"] * (1.2 if index else 1))
                        for index, slot in enumerate(expected)]
        require(event["raw_slots"] == expected, "Observable forecast slots differ from original physical source/bias")
        slot_count += len(expected)
    require(slot_count > 0, "No observable forecast supply values were recorded")
    if row["condition"] != "original":
        require(trace["forecast"]["bias"] == row["bias"]
                and trace["forecast"]["future_multiplier"] == 1 + row["bias"]
                and trace["forecast"]["physical_supply_mutated"] is False,
                "Forecast identity metadata differs from condition")
    return {"jobs_verified": len(jobs), "schedule_allocations_verified": len(schedule),
            "physical_intervals_verified": len(physical_intervals),
            "forecast_slot_values_verified": slot_count,
            "reconstructed_total_energy_j": total, "reconstructed_green_energy_j": green,
            "independent_mean_bsld": mean_bsld, "nonnegative_waits": True,
            "machine_allocations_exclusive": True, "physical_boundary_passed": True}


def summarize_h1(rows):
    selected = {(r["seed"], r["start"], r["condition"]): r for r in rows if r["condition"] in ("zero", "plus20")}
    require(len(selected) == 12, "H1 requires twelve unique scientific rows")
    starts = sorted({r["start"] for r in rows})
    require(len(starts) == 2, "H1 requires two fixed paired windows")
    per_seed = []
    for seed in (0, 1, 2):
        differences = []
        for start in starts:
            require((seed, start, "zero") in selected and (seed, start, "plus20") in selected,
                    "H1 scientific pair is missing")
            zero, stress = selected[(seed, start, "zero")], selected[(seed, start, "plus20")]
            difference = stress["renewable_energy_utilization"] - zero["renewable_energy_utilization"]
            differences.append({"start": start, "difference": difference,
                                "relative_change": difference / zero["renewable_energy_utilization"]
                                if zero["renewable_energy_utilization"] else None})
        mean = sum(d["difference"] for d in differences) / 2
        per_seed.append({"seed": seed, "pairs": differences, "window_count": 2,
                         "mean_difference": mean, "strictly_negative": mean < 0})
    means = [s["mean_difference"] for s in per_seed]
    supported = all(s["strictly_negative"] for s in per_seed)
    return {"complete": True, "protocol_valid": True, "supported": supported,
            "descriptive_all_seed_means_negative": supported, "per_seed": per_seed,
            "seed_mean_difference_range": [min(means), max(means)],
            "interpretation": "Controlled sensitivity diagnostic only; no significance, generalization or novelty claim."}


def _original_inputs(root):
    sys.path.insert(0, str(root))
    from greenPower import green_power
    from job import Workloads
    config = configparser.ConfigParser()
    config.read(root / "configFile/config.ini", encoding="utf-8")
    original_cwd = Path.cwd()
    try:
        os.chdir(root)
        physical = green_power(24, config.getfloat("general setting", "numberPv"),
                               config.getfloat("general setting", "turbinePowerNominal"))
        with contextlib.redirect_stdout(io.StringIO()):
            workloads = Workloads(str(root / "data/lublin_256.swf"))
    finally:
        os.chdir(original_cwd)
    static_power = (config.getfloat("general setting", "idlePower")
                    * math.ceil(workloads.max_nodes / config.getint("general setting", "processor_per_machine")))
    return physical, workloads, static_power


def _paired_metrics(rows):
    keys = ("renewable_energy_utilization", "average_bounded_slowdown", "reward", "total_energy_j",
            "green_energy_j", "brown_energy_j", "makespan_seconds", "wall_seconds", "controller_wall_seconds")
    selected = {(r["seed"], r["start"], r["condition"]): r for r in rows}
    result = []
    for seed, start, condition in selected:
        if condition != "zero":
            continue
        zero, stress = selected[(seed, start, "zero")], selected[(seed, start, "plus20")]
        comparisons = {}
        for key in keys:
            difference = stress[key] - zero[key]
            comparisons[key] = {"zero": zero[key], "plus20": stress[key], "difference": difference,
                                "relative_change": difference / zero[key] if zero[key] else None}
        result.append({"seed": seed, "start": start, "metrics": comparisons})
    return result


def audit_run(run_dir, root=ROOT):
    root, run = Path(root).resolve(), Path(run_dir).resolve()
    require(run.is_relative_to(root), "Pilot result directory must be inside project root")
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    rows_path = run / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.is_file() else state.get("rows", [])
    result = {"raw_rows": rows, "manifest": state.get("manifest", {}),
              "proof": {"schema_version": "1.0", "run_id": run.name, "run_path": run.relative_to(root).as_posix(),
                        "status": "invalid", "protocol_valid": False, "errors": [], "episodes": []},
              "summary": {"status": state.get("status"), "h1": None}, "checksums": {}}
    try:
        validate_protocol(state, rows)
        require(rows == state["rows"], "Final rows file differs from final state rows")
        rows_jsonl = [json.loads(line) for line in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
        require(rows_jsonl == rows, "Append-only rows log differs from final rows")
        runner_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        for key in ("status", "episodes_started", "episodes_completed", "zero_gates", "h1", "frozen_inputs_verified_after"):
            require(runner_summary[key] == state[key], "Runner summary differs from state: " + key)
        manifest = state["manifest"]
        result["checksums"].update(validate_frozen(root, manifest))
        physical, workloads, static_power = _original_inputs(root)
        supply_hash = canonical_hash(physical.greenPowerList)
        for row in rows:
            trace_path = _inside(run, row["trace"])
            trace = read_trace(trace_path, row)
            input_fields = ("job_id", "submit_time", "run_time", "request_time", "request_number_of_processors", "power")
            input_jobs = {workloads[i].job_id: {field: getattr(workloads[i], field) for field in input_fields}
                          for i in range(row["start"], row["start"] + 1024)}
            proof = audit_episode(trace, row, physical, static_power=static_power, eta=manifest["eta"], input_jobs=input_jobs)
            result["proof"]["episodes"].append({"seed": row["seed"], "start": row["start"],
                "condition": row["condition"], "trace_path": trace_path.relative_to(root).as_posix(),
                "compressed_sha256": file_hash(trace_path), "trace_content_sha256": row["trace_content_sha256"], **proof})
            del trace
        require(canonical_hash(physical.greenPowerList) == supply_hash, "Auditor's original physical array changed")
        h1 = summarize_h1(rows)
        saved_h1 = runner_summary["h1"]
        require(all(h1[key] == saved_h1[key] for key in ("complete", "protocol_valid", "supported", "descriptive_all_seed_means_negative", "seed_mean_difference_range")),
                "Independent H1 conclusion differs from runner summary")
        for seed in h1["per_seed"]:
            saved = next(s for s in saved_h1["per_seed"] if s["seed"] == seed["seed"])
            require(seed["mean_difference"] == saved["mean_difference"], "Independent per-seed H1 mean differs")
        supervision = root / "results" / (run.name + "-supervisor")
        supervisor_state = json.loads((supervision / "state.json").read_text(encoding="utf-8"))
        require(supervisor_state["status"] == "completed", "Process supervisor did not complete successfully")
        costs = {key: state.get(key) for key in ("started_utc", "ended_utc", "total_wall_seconds",
            "total_process_cpu_seconds", "first_pair_cost", "log_bytes_before_final_summary", "budget", "command")}
        costs.update(simulation_wall_seconds_sum=sum(r["wall_seconds"] for r in rows),
                     controller_wall_seconds_sum=sum(r["controller_wall_seconds"] for r in rows),
                     simulation_process_cpu_seconds_sum=sum(r["process_cpu_seconds"] for r in rows),
                     controller_process_cpu_seconds_sum=sum(r["controller_process_cpu_seconds"] for r in rows),
                     process_peak_rss_bytes=max(r["process_memory"]["peak_rss_bytes"] or 0 for r in rows),
                     gpu_peak_allocated_bytes=max(r["gpu"].get("peak_allocated_bytes", 0) for r in rows),
                     gpu_peak_reserved_bytes=max(r["gpu"].get("peak_reserved_bytes", 0) for r in rows),
                     memory_scope="OS process lifetime peak; per-episode allocator peak reset by runner",
                     gpu_utilization=None, gpu_utilization_missing_reason="No sampled utilization service; allocator bytes are not utilization",
                     supervisor=supervisor_state)
        result["proof"].update(status="verified", protocol_valid=True, h1=h1,
            checks={"complete_unique_matrix": True, "six_exact_zero_gates": True,
                    "all_trace_hashes": True, "all_input_source_model_hashes": True,
                    "all_fixed_jobs_and_schedules": True, "physical_energy_reconstruction": True,
                    "observable_forecast_values": True, "runner_h1_agreement": True},
            evidence_limits=["No saved before/after full-array hash strings; equality flags and independent original-source reconstruction are distinct evidence.",
                             "Reused baseline windows; no unseen-test, significance, generalization, novelty, or robustness-method claim."],
            frozen_files_verified=len(result["checksums"]),
            exact_zero_gates_verified=6, original_supply_values=len(physical.greenPowerList),
            original_supply_sha256=supply_hash, static_power_w=static_power,
            full_supply_evidence={"runner_full_array_unchanged_assertions": 18,
                "persisted_before_after_array_hashes": None,
                "limitation": "Runner persisted equality booleans, not before/after full-array hash strings.",
                "independent_proof": "Frozen source/data hashes, full original array reconstruction, every physical interval and every observable forecast slot verified."})
        result["summary"].update(h1=h1, paired_metrics=_paired_metrics(rows), costs=costs,
            interpretation="Six paired windows from reused baseline windows; three seed-level means. Individual windows may improve. No significance, generalization, robustness-method, or novelty claim.")
        # Recheck frozen files after the complete sequential audit.
        require(validate_frozen(root, manifest) == result["checksums"], "Frozen files changed during audit")
    except (AuditError, KeyError, OSError, ValueError, TypeError, IndexError) as error:
        result["proof"]["status"] = "invalid"
        result["proof"]["protocol_valid"] = False
        result["proof"]["h1"] = None
        result["proof"]["errors"].append(type(error).__name__ + ": " + str(error))
        result["summary"]["h1"] = None
    for folder in (run, root / "results" / (run.name + "-supervisor")):
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    result["checksums"][path.relative_to(root).as_posix()] = file_hash(path)
    result["summary"]["protocol_valid"] = result["proof"]["protocol_valid"]
    result["summary"]["cost_rows_preserved"] = len(rows)
    return result


def _portable(value, root):
    if isinstance(value, dict):
        return {k: _portable(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v, root) for v in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        prefix = root.as_posix().rstrip("/") + "/"
        if normalized.lower().startswith(prefix.lower()):
            return normalized[len(prefix):]
        return normalized if "\\" in value else value
    return value


def export_evidence(result, output_dir, root=ROOT):
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    require(output.is_relative_to(root), "Evidence export must stay inside project root")
    require(output != root and not output.exists(), "Evidence directory must be new")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"proof.json": result["proof"], "raw_rows.json": result["raw_rows"],
                 "summary.json": result["summary"], "frozen_manifest.json": result["manifest"]}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(_portable(value, root), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = _portable(result["raw_rows"], root)
    fields = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    checksums = dict(result["checksums"])
    for path in output.iterdir():
        checksums[path.relative_to(root).as_posix()] = file_hash(path)
    ledger = {"algorithm": "sha256", "path_base": "project_root", "files": checksums,
              "large_traces": "Compressed traces remain at their local result paths; only hashes are exported.",
              "self_hash": "Excluded to avoid a circular checksum; all other evidence files are included."}
    (output / "checksum_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_run(args.run_dir)
    export_evidence(result, args.output_dir)
    print(json.dumps({"protocol_valid": result["proof"]["protocol_valid"], "errors": result["proof"]["errors"],
                      "episodes_audited": len(result["proof"]["episodes"]),
                      "h1": result["summary"]["h1"], "output": str(args.output_dir)}, allow_nan=False))
    return 0 if result["proof"]["protocol_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
