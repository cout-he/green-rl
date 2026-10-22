"""Validate a completed baseline and write descriptive, evidence-linked summaries.

Uses only the standard library. It never trains, edits configuration, or treats
smoke evidence as a baseline. Ten evaluation slices are not ten training seeds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
STARTS = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]
ALGORITHMS = ["FCFS", "F2", "PPO", "MARL"]
ANCHORS = {
    "lublin_256": {"MARL": (31.738, .8154), "PPO": (56.358, .6767)},
    "Cirne": {"MARL": (17.895, .7545)},
    "Jann": {"MARL": (12.534, .7424)},
}
ETA = {"lublin_256": .002, "Cirne": .008, "Jann": .003}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric(value, name):
    require(not isinstance(value, bool), name + " must be numeric")
    number = float(value)
    require(math.isfinite(number), name + " must be finite")
    if name == "average_bounded_slowdown":
        require(number >= 1, "BSLD must be >= 1")
    elif name == "renewable_energy_utilization":
        require(0 <= number <= 1, "ReUtil must be in [0,1]")
    return number


def difference(bsld, reutil, reference_bsld, reference_reutil):
    return {"bsld_relative_percent": 100 * (bsld / reference_bsld - 1),
            "reutil_percentage_points": 100 * (reutil - reference_reutil)}


def validate_evaluation(root, workload, seed, semantics):
    directory = root / "baseline" / workload / f"seed_{seed}" / "evaluation" / semantics
    metadata = read_json(directory / "metadata.json")
    rows = read_json(directory / "rows.json")
    require(metadata.get("workload") == workload and metadata.get("seed") == seed,
            f"{directory}: metadata workload/seed mismatch")
    require(metadata.get("backfill") == 1, "baseline requires Green-Backfilling (backfill=1)")
    require(metadata.get("sequence_length") == 1024 and metadata.get("starts") == STARTS,
            "baseline requires the published ten starts and sequence length 1024")
    require(metadata.get("evaluation_policy") == "sample", "expected sampled evaluation policy")
    require(set(metadata.get("algorithms", [])) == set(ALGORITHMS)
            and len(metadata["algorithms"]) == len(ALGORITHMS), "baseline algorithm set mismatch")
    require(isinstance(rows, list) and len(rows) == 40, "expected 10 sequences for each of 4 algorithms")
    require(all(row.get("algorithm") in ALGORITHMS for row in rows), "unknown row algorithm")
    with (directory / "result.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 4 and {row["algorithm"] for row in csv_rows} == set(ALGORITHMS),
            "result.csv must contain each baseline algorithm exactly once")
    results = []
    for algorithm in ALGORITHMS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        require(len(selected) == 10 and {row.get("iteration") for row in selected} == set(range(1, 11)),
                f"{algorithm}: expected ten unique evaluation iterations")
        selected.sort(key=lambda row: row["iteration"])
        require([row.get("start") for row in selected] == STARTS,
                f"{algorithm}: evaluation starts mismatch")
        require(all(row.get("sequence_length") == 1024 and row.get("seed") == seed for row in selected),
                f"{algorithm}: row length/seed mismatch")
        bslds = [metric(row["average_bounded_slowdown"], "average_bounded_slowdown") for row in selected]
        reutils = [metric(row["renewable_energy_utilization"], "renewable_energy_utilization") for row in selected]
        for row, bsld, reutil in zip(selected, bslds, reutils):
            reward = metric(row["reward"], "reward")
            require(math.isclose(reward, reutil - ETA[workload] * bsld, rel_tol=1e-9, abs_tol=1e-9),
                    f"{algorithm}: reward is inconsistent with eta, BSLD and ReUtil")
        bsld_mean, reutil_mean = statistics.mean(bslds), statistics.mean(reutils)
        aggregate = next(row for row in csv_rows if row["algorithm"] == algorithm)
        for value, column, key in [(bsld_mean, "average bounded slowdown", "average_bounded_slowdown"),
                                   (reutil_mean, "renewable energy utilization", "renewable_energy_utilization")]:
            require(math.isclose(value, metric(aggregate[column], key), rel_tol=1e-9, abs_tol=1e-9),
                    f"{algorithm}: CSV mean differs from per-sequence {key}")
        result = {"workload": workload, "training_seed": seed, "semantics": semantics,
                  "evaluation_seed_mode": metadata.get("evaluation_seed_mode", "legacy"),
                  "training_semantics": "author", "algorithm": algorithm,
                  "evaluation_policy": metadata["evaluation_policy"], "sequence_count": 10,
                  "sequence_length": 1024, "bsld_mean": bsld_mean,
                  "bsld_sequence_sample_stddev": statistics.stdev(bslds),
                  "reutil_mean": reutil_mean, "reutil_sequence_sample_stddev": statistics.stdev(reutils),
                  "evidence": {name: str(directory / name) for name in ("result.csv", "rows.json", "metadata.json")}}
        if algorithm in ANCHORS.get(workload, {}):
            bsld_anchor, reutil_anchor = ANCHORS[workload][algorithm]
            result["published_anchor"] = {"bsld": bsld_anchor, "reutil": reutil_anchor,
                                          "source": "research/PROTOCOL.md, Table 3/4"}
            # Only the author mode is a comparison against published author behavior.
            if semantics == "author":
                result["published_difference"] = difference(bsld_mean, reutil_mean, bsld_anchor, reutil_anchor)
        results.append(result)
    return results


def evidence(root, state):
    manifest_path = root / "source_manifest.json"
    source = {"path": str(root / "source"), "manifest": str(manifest_path), "status": "missing"}
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        require(isinstance(manifest, dict) and bool(manifest), "source manifest must be nonempty")
        for relative, digest in manifest.items():
            path = (root / "source" / relative).resolve()
            require(path.is_relative_to((root / "source").resolve()), "source manifest path escapes snapshot")
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    "source snapshot hash mismatch: " + relative)
        source.update(status="verified", file_count=len(manifest))
    environment = {"status": "missing", "run_scoped": False,
                   "note": "No software environment evidence found; run environment cannot be inferred."}
    for path in [root / "environment.json", root / "research/environment.json", PROJECT / "research/environment.json"]:
        if path.is_file():
            report = read_json(path)
            scoped = path.is_relative_to(root)
            environment = {"status": "recorded" if scoped else "external_report_only", "run_scoped": scoped,
                           "path": str(path), "recorded_at_utc": report.get("recorded_at_utc"),
                           "python": report.get("python"), "dependencies": report.get("dependencies"),
                           "torch": report.get("torch"),
                           "deviations": report.get("installation", {}).get("deviations", []),
                           "note": "Run-scoped report." if scoped else
                           "External environment report; not an immutable capture of this run's environment."}
            break
    return {"state": str(root / "state.json"), "git_commit_at_run_start": state.get("git_commit"),
            "git_status_at_run_start": state.get("git_status"), "source_snapshot": source,
            "source_patch": {"path": str(root / "source.patch"),
                             "status": "present" if (root / "source.patch").is_file() else "missing"},
            "software_environment": environment, "protocol": str(PROJECT / "research/PROTOCOL.md")}


def build_summary(root):
    state = read_json(root / "state.json")
    require(state.get("status") == "completed", "run state must be completed; no partial summary is allowed")
    arguments = state.get("arguments", {})
    workloads, seeds = arguments.get("workloads", []), arguments.get("seeds", [])
    require(bool(workloads) and bool(seeds), "run must record workloads and training seeds")
    require(len(set(workloads)) == len(workloads) and len(set(seeds)) == len(seeds), "duplicate workloads/seeds")
    require(all(workload in ETA for workload in workloads), "unknown workload")
    require(all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds), "invalid training seed")
    steps = state.get("steps", [])
    results = []
    for workload in workloads:
        for seed in seeds:
            for suffix in ("MARL-train", "MaskablePPO-train", "author-evaluate", "paper-evaluate"):
                name = f"baseline-{workload}-{seed}-{suffix}"
                records = [step for step in steps if step.get("name") == name]
                require(len(records) == 1 and records[0].get("status") == "completed"
                        and records[0].get("returncode") == 0, "missing/incomplete baseline step: " + name)
                if suffix.endswith("evaluate"):
                    require(records[0].get("training_semantics") == "author"
                            and records[0].get("evaluation_semantics") == suffix.split("-")[0],
                            "evaluation/training semantics mismatch: " + name)
            author = validate_evaluation(root, workload, seed, "author")
            paper = validate_evaluation(root, workload, seed, "paper")
            for row in paper:
                reference = next(value for value in author if value["algorithm"] == row["algorithm"])
                row["author_sensitivity"] = difference(row["bsld_mean"], row["reutil_mean"],
                                                       reference["bsld_mean"], reference["reutil_mean"])
            results.extend(author + paper)
    return {"status": "completed_baseline_validated", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(root), "independent_training_seeds": {workload: list(seeds) for workload in workloads},
            "evaluations": results, "evidence": evidence(root, state),
            "interpretation": ["Ten slices are not independent training seeds; standard deviations describe evaluation sequences only.",
                               "Author and paper evaluations both use author-trained models. Paper mode is sensitivity analysis, not a new algorithm.",
                               "Legacy evaluations share one RNG stream across algorithms and sequences: paper-author differences may include sampling-stream changes and are not isolated causal effects.",
                               "No significance test or reproduction-success claim is inferred.",
                               "Positive BSLD differences mean slower; positive ReUtil differences mean more renewable utilization."]}


def markdown(summary):
    lines = ["# Completed baseline descriptive summary", "", "Status: " + summary["status"], ""]
    lines.extend(summary["interpretation"])
    lines.extend(["", "| Workload | Train seed | Eval mode | Algorithm | BSLD mean ± sequence SD | ReUtil mean ± sequence SD | BSLD Δ (%) | ReUtil Δ (pp) | Reference |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in summary["evaluations"]:
        delta = row.get("published_difference", row.get("author_sensitivity"))
        bsld_delta = f"{delta['bsld_relative_percent']:+.4f}" if delta else "—"
        reutil_delta = f"{delta['reutil_percentage_points']:+.4f}" if delta else "—"
        reference = "same author-trained model, author evaluation" if row["semantics"] == "paper" else ("published Table 3/4" if delta else "no supplied anchor")
        lines.append(f"| {row['workload']} | {row['training_seed']} | {row['semantics']} | {row['algorithm']} | "
                     f"{row['bsld_mean']:.6f} ± {row['bsld_sequence_sample_stddev']:.6f} | "
                     f"{row['reutil_mean']:.6f} ± {row['reutil_sequence_sample_stddev']:.6f} | {bsld_delta} | {reutil_delta} | {reference} |")
    ev = summary["evidence"]
    lines.extend(["", "Evaluation policy: sample. Each algorithm uses the same ten published starts, 1024 jobs per sequence.",
                  "", f"Source commit at run start: `{ev['git_commit_at_run_start']}`; recorded dirty status: `{ev['git_status_at_run_start']}`.",
                  f"Source snapshot: [{ev['source_snapshot']['status']}]({Path(ev['source_snapshot']['manifest']).as_posix()}).",
                  f"State: [state.json]({Path(ev['state']).as_posix()}).",
                  "", "Software environment: " + ev["software_environment"]["status"] + ". " + ev["software_environment"]["note"]])
    environment = ev["software_environment"]
    if environment.get("path"):
        lines.append(f"Environment evidence: [JSON]({Path(environment['path']).as_posix()}).")
    lines.extend("- " + item for item in environment.get("deviations", []))
    for row in summary["evaluations"]:
        if row["algorithm"] == "MARL":
            path = row["evidence"]["rows.json"]
            lines.append(f"\n{row['workload']} seed {row['training_seed']} {row['semantics']}: [per-sequence evidence]({Path(path).as_posix()}).")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.run_dir.resolve(strict=True)
        summary = build_summary(root)
        text = markdown(summary)
        output = root / "analysis"
        output.mkdir(exist_ok=True)
        for filename, content in [("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                                  ("summary.md", text)]:
            temporary = output / (filename + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, output / filename)
        print(json.dumps({"status": summary["status"], "analysis": str(output)}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
