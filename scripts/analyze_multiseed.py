"""Validate matched evaluation evidence and summarize seed-level variation.

Offline and standard-library only: no simulator imports, training or weight loading.
One statistical observation is a seed's mean over the ten fixed evaluation windows.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
STARTS = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]
ALGORITHMS = ("FCFS", "F2", "PPO", "MARL")
ALGORITHM_INDEX = {"FCFS": 0, "F2": 1, "PPO": 4, "MARL": 5}
METRICS = ("bsld", "reutil", "reward")
MODES = ("author", "paper")
SEED_DERIVATION = "(seed + 1000003 * iteration + 10007 * fixed_algorithm_index) mod 2**32"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def number(value, name):
    require(not isinstance(value, bool), name + " must be numeric")
    result = float(value)
    require(math.isfinite(result), name + " must be finite")
    if name == "bsld":
        require(result >= 1, "BSLD must be >= 1")
    if name == "reutil":
        require(0 <= result <= 1, "ReUtil must be in [0,1]")
    return result


def validate_evaluation(directory, seed, mode):
    """Return normalized per-window records and each algorithm's ten-window mean."""
    metadata = read_json(directory / "metadata.json")
    require(metadata.get("workload") == "lublin_256" and metadata.get("seed") == seed,
            "evaluation workload/seed mismatch")
    require(metadata.get("backfill") == 1 and metadata.get("sequence_length") == 1024,
            "expected Green-Backfilling and 1024-job evaluation windows")
    require(metadata.get("starts") == STARTS, "evaluation starts mismatch")
    require(metadata.get("evaluation_seed_mode") == "paired", "only paired evaluation is comparable")
    require(metadata.get("evaluation_policy") == "sample", "expected sampled evaluation")
    require(metadata.get("delay_semantics") == mode, "evaluation metric mode mismatch")
    require(metadata.get("seed_derivation") == SEED_DERIVATION, "unknown policy seed derivation")
    require(len(metadata.get("algorithms", [])) == 4 and set(metadata["algorithms"]) == set(ALGORITHMS),
            "evaluation algorithm set mismatch")
    rows = read_json(directory / "rows.json")
    require(isinstance(rows, list) and len(rows) == 40, "expected 40 evaluation rows")
    require(all(row.get("algorithm") in ALGORITHMS for row in rows), "unknown evaluation algorithm")
    with (directory / "result.csv").open(encoding="utf-8-sig", newline="") as handle:
        totals = list(csv.DictReader(handle))
    require(len(totals) == 4 and {row["algorithm"] for row in totals} == set(ALGORITHMS),
            "result CSV must contain each algorithm exactly once")
    windows, per_seed = [], []
    for algorithm in ALGORITHMS:
        selected = sorted((row for row in rows if row["algorithm"] == algorithm), key=lambda row: row["iteration"])
        require([row.get("iteration") for row in selected] == list(range(1, 11)),
                "expected ten distinct iterations per algorithm")
        require([row.get("start") for row in selected] == STARTS, "row starts mismatch")
        normalized = []
        for row in selected:
            require(row.get("seed") == seed and row.get("sequence_length") == 1024,
                    "row seed/length mismatch")
            policy_seed = (seed + 1000003 * row["iteration"] + 10007 * ALGORITHM_INDEX[algorithm]) % 2**32
            require(row.get("policy_seed") == policy_seed, "policy seed mismatch")
            bsld = number(row["average_bounded_slowdown"], "bsld")
            reutil = number(row["renewable_energy_utilization"], "reutil")
            reward = number(row["reward"], "reward")
            require(math.isclose(reward, reutil - .002 * bsld, rel_tol=1e-9, abs_tol=1e-9),
                    "reward differs from ReUtil - 0.002 * BSLD")
            normalized.append(dict(seed=seed, metric_mode=mode, algorithm=algorithm,
                window_index=row["iteration"], start_index=row["start"], sequence_length=1024,
                policy_seed=policy_seed, bsld=bsld, reutil=reutil, reward=reward))
        summary = dict(seed=seed, metric_mode=mode, algorithm=algorithm, window_count=10)
        for metric in METRICS:
            values = [row[metric] for row in normalized]
            summary[metric + "_mean"] = statistics.mean(values)
            summary[metric + "_window_sd"] = statistics.stdev(values)
        total = next(row for row in totals if row["algorithm"] == algorithm)
        for metric, column in (("bsld", "average bounded slowdown"), ("reutil", "renewable energy utilization")):
            require(math.isclose(summary[metric + "_mean"], number(total[column], metric),
                                 rel_tol=1e-9, abs_tol=1e-9), "CSV mean differs from evaluation windows")
        windows.extend(normalized)
        per_seed.append(summary)
    return windows, per_seed


def validate_controls(windows):
    """Count exact author/paper matches for FCFS, F2 and PPO, including RNG seeds."""
    matched = 0
    lookup = {(row["seed"], row["metric_mode"], row["algorithm"], row["window_index"]): row for row in windows}
    require(len(lookup) == len(windows), "duplicate evaluation window")
    for row in windows:
        if row["metric_mode"] != "author" or row["algorithm"] not in ("FCFS", "F2", "PPO"):
            continue
        paper = lookup.get((row["seed"], "paper", row["algorithm"], row["window_index"]))
        require(paper is not None and all(row[key] == paper[key] for key in row if key != "metric_mode"),
                "author/paper zero-effect control differs")
        matched += 1
    return matched


def aggregate_seed_means(per_seed):
    result = []
    keys = [(row["seed"], row["metric_mode"], row["algorithm"]) for row in per_seed]
    require(len(set(keys)) == len(keys), "duplicate seed/mode/algorithm")
    seeds = {row["seed"] for row in per_seed}
    require(len(seeds) >= 2, "seed variation requires at least two seeds")
    for mode in MODES:
        for algorithm in ALGORITHMS:
            selected = [row for row in per_seed if row["metric_mode"] == mode and row["algorithm"] == algorithm]
            if not selected:
                continue
            require({row["seed"] for row in selected} == seeds, "unbalanced seed coverage")
            summary = dict(metric_mode=mode, algorithm=algorithm, seed_count=len(selected))
            for metric in METRICS:
                values = [number(row[metric + "_mean"], metric) for row in selected]
                summary[metric + "_mean"] = statistics.mean(values)
                summary[metric + "_seed_sd"] = statistics.stdev(values)
                summary[metric + "_min_seed_mean"] = min(values)
                summary[metric + "_max_seed_mean"] = max(values)
            result.append(summary)
    return result


def difference_row(value, reference):
    result = {metric + "_difference": value[metric + "_mean"] - reference[metric + "_mean"] for metric in METRICS}
    result["bsld_relative_percent"] = 100 * (value["bsld_mean"] / reference["bsld_mean"] - 1)
    result["reutil_percentage_points"] = 100 * result["reutil_difference"]
    return result


def summarize_differences(rows):
    result = {"seed_count": len(rows), "per_seed": rows}
    for key in ("bsld_difference", "reutil_difference", "reward_difference", "bsld_relative_percent", "reutil_percentage_points"):
        values = [row[key] for row in rows]
        result[key + "_mean"] = statistics.mean(values)
        result[key + "_seed_sd"] = statistics.stdev(values)
    result["both_metrics_better_seed_count"] = sum(row["bsld_difference"] < 0 and row["reutil_difference"] > 0 for row in rows)
    return result


def paired_comparisons(per_seed):
    aggregate_seed_means(per_seed)  # Validate one balanced observation per seed/mode/algorithm.
    lookup = {(row["seed"], row["metric_mode"], row["algorithm"]): row for row in per_seed}
    result = []
    for mode in MODES:
        rows = [dict(seed=seed, **difference_row(lookup[seed, mode, "MARL"], lookup[seed, mode, "PPO"]))
                for seed in sorted({row["seed"] for row in per_seed})]
        result.append(dict(metric_mode=mode, algorithm="MARL", reference_algorithm="PPO", **summarize_differences(rows)))
    return result


def mode_sensitivity(per_seed):
    aggregate_seed_means(per_seed)
    lookup = {(row["seed"], row["metric_mode"], row["algorithm"]): row for row in per_seed}
    result = []
    for algorithm in ALGORITHMS:
        if not any(row["algorithm"] == algorithm for row in per_seed):
            continue
        rows = [dict(seed=seed, **difference_row(lookup[seed, "paper", algorithm], lookup[seed, "author", algorithm]))
                for seed in sorted({row["seed"] for row in per_seed})]
        result.append(dict(algorithm=algorithm, comparison="paper_minus_author", **summarize_differences(rows)))
    return result


def analyze(project, seed0_training_run, seed0_evaluation_run, multiseed_run):
    """Build outputs in memory; refuse all invalid evidence before writing files."""
    inputs, copies, runs, training, windows, per_seed = {}, {}, [], [], [], []

    def relative(path):
        return path.resolve().relative_to(project.resolve()).as_posix()

    def capture(path):
        data = path.read_bytes()
        inputs[relative(path)] = hashlib.sha256(data).hexdigest()
        return data

    def json_input(path):
        return json.loads(capture(path).decode("utf-8-sig"))

    for run, seeds in ((seed0_training_run, [0]), (multiseed_run, [1, 2])):
        state = json_input(run / "state.json")
        require(state.get("status") == "completed", "training run is incomplete")
        require(state.get("arguments", {}).get("seeds") == seeds
                and state["arguments"].get("workloads") == ["lublin_256"], "training workload/seed set mismatch")
        require(all(step.get("status") == "completed" and step.get("returncode") == 0
                    for step in state.get("steps", [])), "incomplete or failed training run step")
        for seed in seeds:
            for suffix in ("MARL-train", "MaskablePPO-train", "author-evaluate", "paper-evaluate"):
                steps = [step for step in state.get("steps", []) if step.get("name") == f"baseline-lublin_256-{seed}-{suffix}"]
                require(len(steps) == 1, "missing or duplicate completed baseline step")
                if suffix.endswith("evaluate"):
                    require(steps[0].get("training_semantics") == "author"
                            and steps[0].get("evaluation_semantics") == suffix.split("-")[0],
                            "baseline training/evaluation semantics mismatch")
        if seeds == [1, 2]:
            require(state["arguments"].get("evaluation_seed_mode") == "paired", "multiseed run must use paired evaluation")
        runs.append(dict(run_dir=relative(run), training_seeds=seeds, status=state["status"],
                         git_commit=state.get("git_commit"), git_status=state.get("git_status"),
                         started_at=state.get("started_at"), ended_at=state.get("ended_at"),
                         completed_step_count=len(state["steps"])))
        for filename in ("source_manifest.json", "environment.json", "datasets.json"):
            if (run / filename).is_file():
                capture(run / filename)
        for seed in seeds:
            for algorithm, folder in (("MARL", "MARL"), ("PPO", "MaskablePPO")):
                source = run / f"baseline/lublin_256/seed_{seed}/models/lublin_256/{folder}/{folder}_lublin_256.csv"
                content = capture(source)
                rows = list(csv.reader(content.decode("utf-8-sig").splitlines()))
                require(len(rows) == 300 and all(len(row) == 3 for row in rows), "training CSV must have 300 three-value epochs")
                values = [[number(row[0], "reward"), number(row[1], "reutil"), number(-float(row[2]), "bsld")] for row in rows]
                require(all(math.isclose(reward, reutil - .002 * bsld, rel_tol=1e-9, abs_tol=1e-9)
                            for reward, reutil, bsld in values), "training CSV reward inconsistent with metrics")
                filename = f"training_seed{seed}_{algorithm}.csv"
                copies[filename] = content
                training.append(dict(seed=seed, algorithm=algorithm, completed_epochs=300,
                    source=relative(source), evidence_file=filename,
                    columns=["reward", "reutil", "negative_bsld"],
                    final_epoch=dict(reward=values[-1][0], reutil=values[-1][1], bsld=values[-1][2])))
    state = json_input(seed0_evaluation_run / "state.json")
    require(state.get("status") == "completed" and state.get("seed_mode") == "paired"
            and state.get("training_seed") == 0 and state.get("evaluation_seed") == 0,
            "seed-zero paired reevaluation must be completed with matching base seed")
    require(Path(state.get("baseline_run", "")).resolve() == seed0_training_run.resolve(),
            "paired seed-zero evaluation refers to a different training run")
    require(len(state.get("steps", [])) == 2 and {step.get("semantics") for step in state["steps"]} == set(MODES)
            and all(step.get("returncode") == 0 for step in state["steps"]), "seed-zero paired evaluation steps incomplete")
    evaluations = []
    for seed in (0, 1, 2):
        train_run = seed0_training_run if seed == 0 else multiseed_run
        for mode in MODES:
            directory = (seed0_evaluation_run / mode if seed == 0
                         else multiseed_run / f"baseline/lublin_256/seed_{seed}/evaluation/{mode}")
            for filename in ("rows.json", "result.csv", "metadata.json"):
                capture(directory / filename)
            metadata = read_json(directory / "metadata.json")
            require(Path(metadata.get("model_root", "")).resolve() ==
                    (train_run / f"baseline/lublin_256/seed_{seed}/models").resolve(), "evaluation model source mismatch")
            eval_windows, eval_means = validate_evaluation(directory, seed, mode)
            windows.extend(eval_windows)
            per_seed.extend(eval_means)
            prefix = f"seed{seed}_{mode}"
            copies[prefix + "_rows.json"] = (directory / "rows.json").read_bytes()
            copies[prefix + "_result.csv"] = (directory / "result.csv").read_bytes()
            metadata["model_root"] = relative(Path(metadata["model_root"]))
            copies[prefix + "_metadata.json"] = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
            evaluations.append(dict(seed=seed, metric_mode=mode, source_dir=relative(directory), evidence_prefix=prefix))
    controls = validate_controls(windows)
    require(controls == 90, "expected 90 matched zero-effect control window pairs")
    aggregate = aggregate_seed_means(per_seed)
    require(len(aggregate) == 8 and len(per_seed) == 24 and len(windows) == 240, "incomplete three-seed evaluation matrix")
    overlaps = [dict(first_window_index=i + 1, second_window_index=j + 1,
                     first_start=a, second_start=b, overlapping_jobs=min(a + 1024, b + 1024) - max(a, b))
                for i, a in enumerate(STARTS) for j, b in enumerate(STARTS) if i < j and abs(a - b) < 1024]
    anchors = {}
    for algorithm, bsld, reutil in (("MARL", 31.738, .8154), ("PPO", 56.358, .6767)):
        author = next(row for row in aggregate if row["metric_mode"] == "author" and row["algorithm"] == algorithm)
        anchors[algorithm] = dict(bsld=bsld, reutil=reutil, source="research/PROTOCOL.md, Table 3/4",
            author_bsld_difference=author["bsld_mean"] - bsld,
            author_bsld_relative_percent=100 * (author["bsld_mean"] / bsld - 1),
            author_reutil_percentage_points=100 * (author["reutil_mean"] - reutil))
    if (project / "research/PROTOCOL.md").is_file():
        capture(project / "research/PROTOCOL.md")
    if (project / "scripts/analyze_multiseed.py").is_file():
        capture(project / "scripts/analyze_multiseed.py")
    summary = dict(schema_version="1.0", status="validated_three_seed_paired_baseline",
        generated_at_utc=datetime.now(timezone.utc).isoformat(), workload="lublin_256", backfill=1,
        training_semantics="author", evaluation_policy="sample", evaluation_seed_mode="paired",
        seeds=[0, 1, 2], eta=.002, per_seed=per_seed, aggregate=aggregate,
        paired_seed_comparisons=paired_comparisons(per_seed), metric_mode_sensitivity=mode_sensitivity(per_seed),
        paper_anchors=anchors, training=training,
        provenance=dict(training_runs=runs, evaluation_sources=evaluations,
            seed0_recorded_model_sha256={key.replace("\\", "/"): value for key, value in state.get("model_sha256", {}).items()},
            input_sha256_file="input_sha256.json", evidence_sha256_file="evidence_sha256.json",
            path_convention="Original input paths relative to repository root; evidence filenames relative to this directory.",
            metadata_copies="model_root normalized to repository-relative paths; input hashes identify the unmodified originals."),
        validation=dict(per_window_reward_equation=True, result_csv_equals_window_means=True,
            per_window_policy_seed_derivation=True, all_training_csv_epochs=300,
            author_paper_exact_control_window_pairs=controls, evaluation_row_count=len(windows),
            training_csv_count=len(training), model_tensor_validation="separate independent audit; this tool does not load weights"),
        window_dependence=dict(starts=STARTS, sequence_length=1024, overlap_pair_count=len(overlaps), overlap_pairs=overlaps),
        statistical_unit="One seed's mean over the same ten fixed windows; across-seed sample SD uses ddof=1 and n=3.",
        caveats=["Training seed and evaluation base seed share the same value. Across-seed variation is end-to-end random variation, not isolated training variance.",
            "Ten fixed evaluation windows include 11 overlapping pairs; they are descriptive slices, not 30 independent replicates.",
            "FCFS and F2 are deterministic controls; their repeated seed records are not independently trained policies.",
            "Paper mode evaluates author-trained fixed models with corrected delay semantics; it is sensitivity analysis, not a retrained algorithm or an innovation.",
            "Seed-level MARL-minus-PPO and paper-minus-author differences are paired descriptively. Three seeds do not establish significance or universal improvement.",
            "Only paired seed-zero reevaluation is included. Legacy seed-zero evaluation and best training epochs are excluded.",
            "Positive BSLD differences mean slower; positive ReUtil differences mean higher renewable utilization."])
    return summary, windows, copies, dict(sorted(inputs.items()))


def write_evidence(output, summary, windows, copies, inputs):
    output.mkdir(parents=True, exist_ok=True)
    written = []

    def write(filename, data):
        (output / filename).write_bytes(data)
        written.append(filename)

    def write_json(filename, data):
        write(filename, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))

    def write_csv(filename, rows):
        import io
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        write(filename, handle.getvalue().encode("utf-8"))

    for filename, data in copies.items():
        write(filename, data)
    write_json("multiseed_summary.json", summary)
    write_json("input_sha256.json", inputs)
    write_csv("evaluation_windows.csv", windows)
    write_csv("seed_summary.csv", summary["per_seed"])
    write_csv("aggregate_summary.csv", summary["aggregate"])
    write_csv("paired_seed_comparisons.csv", [dict(metric_mode=entry["metric_mode"], **row)
        for entry in summary["paired_seed_comparisons"] for row in entry["per_seed"]])
    write_csv("metric_mode_sensitivity.csv", [dict(algorithm=entry["algorithm"], **row)
        for entry in summary["metric_mode_sensitivity"] for row in entry["per_seed"]])
    write_json("evidence_sha256.json", {filename: hashlib.sha256((output / filename).read_bytes()).hexdigest()
               for filename in sorted(written)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--seed0-training-run", type=Path, default=Path("results/gas-marl-20261021-baseline-02"))
    parser.add_argument("--seed0-evaluation-run", type=Path, default=Path("results/gas-marl-20261022-paired-01"))
    parser.add_argument("--multiseed-run", type=Path, default=Path("results/gas-marl-20261022-multiseed-01"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/evidence/multiseed"))
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    def resolve(path):
        return (path if path.is_absolute() else project / path).resolve()
    try:
        result = analyze(project, resolve(args.seed0_training_run), resolve(args.seed0_evaluation_run), resolve(args.multiseed_run))
        output = resolve(args.output_dir)
        write_evidence(output, *result)
        print(json.dumps(dict(status=result[0]["status"], output_dir=str(output), seed_count=3)))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps(dict(status="refused", error=str(exc))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
