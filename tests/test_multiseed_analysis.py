"""Hand-checkable synthetic inputs; none of these numbers are research results."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_multiseed.py"
STARTS = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]
INDEX = {"FCFS": 0, "F2": 1, "PPO": 4, "MARL": 5}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def evaluation(directory, seed=0, mode="author"):
    rows, totals = [], []
    for algorithm, base in [("FCFS", 70), ("F2", 60), ("PPO", 40), ("MARL", 10)]:
        for index, start in enumerate(STARTS):
            bsld, reutil = base + index, .8
            rows.append(dict(algorithm=algorithm, iteration=index + 1, start=start,
                             sequence_length=1024, seed=seed,
                             policy_seed=seed + 1000003 * (index + 1) + 10007 * INDEX[algorithm],
                             average_bounded_slowdown=bsld,
                             renewable_energy_utilization=reutil, reward=reutil - .002 * bsld))
        totals.append([algorithm, base + 4.5, .8])
    write_json(directory / "rows.json", rows)
    write_json(directory / "metadata.json", dict(workload="lublin_256", seed=seed,
               backfill=1, algorithms=list(INDEX), starts=STARTS, sequence_length=1024,
               evaluation_policy="sample", evaluation_seed_mode="paired", delay_semantics=mode,
               seed_derivation="(seed + 1000003 * iteration + 10007 * fixed_algorithm_index) mod 2**32"))
    with (directory / "result.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "average bounded slowdown", "renewable energy utilization"])
        writer.writerows(totals)


def completed_runs(root):
    train0, paired, multi = (root / name for name in ("train0", "paired0", "multi12"))
    for run, seeds in ((train0, [0]), (multi, [1, 2])):
        steps = []
        for seed in seeds:
            for suffix in ("MARL-train", "MaskablePPO-train", "author-evaluate", "paper-evaluate"):
                step = dict(name=f"baseline-lublin_256-{seed}-{suffix}", status="completed", returncode=0)
                if suffix.endswith("evaluate"):
                    step.update(training_semantics="author", evaluation_semantics=suffix.split("-")[0])
                steps.append(step)
            for folder in ("MARL", "MaskablePPO"):
                path = run / f"baseline/lublin_256/seed_{seed}/models/lublin_256/{folder}/{folder}_lublin_256.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("0.78,0.8,-10\n" * 300)
            for mode in ("author", "paper"):
                directory = (paired / mode if seed == 0 else run / f"baseline/lublin_256/seed_{seed}/evaluation/{mode}")
                evaluation(directory, seed, mode)
                data = json.loads((directory / "metadata.json").read_text())
                data["model_root"] = str(run / f"baseline/lublin_256/seed_{seed}/models")
                write_json(directory / "metadata.json", data)
        write_json(run / "state.json", dict(status="completed", git_commit="fixture", git_status="",
                   arguments=dict(workloads=["lublin_256"], seeds=seeds, evaluation_seed_mode="paired"), steps=steps))
    write_json(paired / "state.json", dict(status="completed", baseline_run=str(train0), training_seed=0,
               evaluation_seed=0, seed_mode="paired", control_equality=True,
               steps=[dict(semantics=mode, returncode=0) for mode in ("author", "paper")]))
    return train0, paired, multi


class MultiseedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def module(self):
        self.assertTrue(SCRIPT.is_file(), "three-seed analyzer is missing")
        spec = importlib.util.spec_from_file_location("multiseed_analysis", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def change_json(self, path, change):
        data = json.loads(path.read_text())
        change(data)
        write_json(path, data)

    def summaries(self):
        return [dict(seed=seed, metric_mode=mode, algorithm=algorithm, window_count=10,
                     bsld_mean=bsld, reutil_mean=reutil, reward_mean=reutil - .002 * bsld,
                     bsld_window_sd=99, reutil_window_sd=.1, reward_window_sd=.2)
                for seed, offset in enumerate((0, 10, 20))
                for mode in ("author", "paper")
                for algorithm, bsld, reutil in (("PPO", 40 + offset, .6),
                    ("MARL", 10 + offset + (5 if mode == "paper" else 0), .8))]

    def test_seed_sd_uses_three_seed_means_and_rejects_duplicate_seed(self):
        module = self.module()
        result = module.aggregate_seed_means(self.summaries())
        marl = next(row for row in result if row["metric_mode"] == "author" and row["algorithm"] == "MARL")
        self.assertEqual(marl["seed_count"], 3)
        self.assertEqual(marl["bsld_mean"], 20)
        self.assertEqual(marl["bsld_seed_sd"], 10)
        self.assertEqual(marl["reutil_seed_sd"], 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.aggregate_seed_means(self.summaries() + [self.summaries()[0]])

    def test_paired_seed_differences_and_semantic_sensitivity(self):
        module = self.module()
        paired = module.paired_comparisons(self.summaries())
        author = next(row for row in paired if row["metric_mode"] == "author")
        self.assertEqual(author["bsld_difference_mean"], -30)
        self.assertEqual(author["bsld_difference_seed_sd"], 0)
        self.assertAlmostEqual(author["reutil_percentage_points_mean"], 20)
        self.assertEqual(author["both_metrics_better_seed_count"], 3)
        sensitivity = module.mode_sensitivity(self.summaries())
        marl = next(row for row in sensitivity if row["algorithm"] == "MARL")
        self.assertEqual(marl["bsld_difference_mean"], 5)
        self.assertEqual(marl["bsld_difference_seed_sd"], 0)
        self.assertEqual(marl["reutil_percentage_points_mean"], 0)

    def test_evaluation_checks_policy_seed_reward_mean_and_window_identity(self):
        module = self.module()
        directory = self.root / "eval"
        evaluation(directory)
        windows, seeds = module.validate_evaluation(directory, 0, "author")
        self.assertEqual(len(windows), 40)
        self.assertEqual(next(row["bsld_mean"] for row in seeds if row["algorithm"] == "MARL"), 14.5)
        for change, message in [
                (lambda rows: rows[0].update(policy_seed=0), "policy seed"),
                (lambda rows: rows[0].update(reward=.1), "reward"),
                (lambda rows: rows[0].update(start=7146), "starts"),
                (lambda rows: rows[0].update(renewable_energy_utilization=float("nan")), "finite")]:
            with self.subTest(message=message):
                evaluation(directory)
                self.change_json(directory / "rows.json", change)
                with self.assertRaisesRegex(ValueError, message):
                    module.validate_evaluation(directory, 0, "author")
        evaluation(directory)
        data = (directory / "result.csv").read_text().replace("74.5", "75.5")
        (directory / "result.csv").write_text(data)
        with self.assertRaisesRegex(ValueError, "CSV mean"):
            module.validate_evaluation(directory, 0, "author")

    def test_legacy_metadata_and_nonzero_control_effect_are_rejected(self):
        module = self.module()
        evaluation(self.root / "author", 0, "author")
        evaluation(self.root / "paper", 0, "paper")
        author, _ = module.validate_evaluation(self.root / "author", 0, "author")
        paper, _ = module.validate_evaluation(self.root / "paper", 0, "paper")
        self.assertEqual(module.validate_controls(author + paper), 30)
        paper[0]["bsld"] += 1
        with self.assertRaisesRegex(ValueError, "control"):
            module.validate_controls(author + paper)
        self.change_json(self.root / "author/metadata.json", lambda data: data.update(evaluation_seed_mode="legacy"))
        with self.assertRaisesRegex(ValueError, "paired"):
            module.validate_evaluation(self.root / "author", 0, "author")

    def test_cli_copies_only_paired_evidence_and_hashes_relative_input_paths(self):
        module = self.module()
        self.assertTrue(hasattr(module, "main"), "evidence-export CLI is missing")
        train0, paired, multi = completed_runs(self.root)
        # Deliberately invalid legacy values must never be consumed for seed zero.
        write_json(train0 / "baseline/lublin_256/seed_0/evaluation/author/rows.json", [{"legacy": "do not read"}])
        output = self.root / "evidence"
        args = ["--project-root", str(self.root), "--seed0-training-run", str(train0),
                "--seed0-evaluation-run", str(paired), "--multiseed-run", str(multi), "--output-dir", str(output)]
        self.assertEqual(module.main(args), 0)
        summary = json.loads((output / "multiseed_summary.json").read_text())
        self.assertEqual(summary["status"], "validated_three_seed_paired_baseline")
        self.assertEqual(summary["validation"]["author_paper_exact_control_window_pairs"], 90)
        self.assertEqual(summary["window_dependence"]["overlap_pair_count"], 11)
        self.assertEqual(len(summary["per_seed"]), 24)
        with (output / "evaluation_windows.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 240)
        hashes = json.loads((output / "input_sha256.json").read_text())
        self.assertIn("paired0/author/rows.json", hashes)
        self.assertNotIn("train0/baseline/lublin_256/seed_0/evaluation/author/rows.json", hashes)
        self.assertTrue(all(not Path(path).is_absolute() and "\\" not in path for path in hashes))
        self.assertEqual((output / "training_seed0_MARL.csv").read_text().splitlines()[0], "0.78,0.8,-10")
        self.assertFalse(list(output.rglob("*.pkl")))
        with (output / "seed_summary.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 24)

    def test_cli_refuses_partial_run_before_writing_evidence(self):
        module = self.module()
        self.assertTrue(hasattr(module, "main"), "evidence-export CLI is missing")
        train0, paired, multi = completed_runs(self.root)
        self.change_json(multi / "state.json", lambda state: state["steps"][-1].update(returncode=1))
        output = self.root / "evidence"
        self.assertNotEqual(module.main(["--project-root", str(self.root), "--seed0-training-run", str(train0),
            "--seed0-evaluation-run", str(paired), "--multiseed-run", str(multi), "--output-dir", str(output)]), 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
