"""Synthetic fixtures only: these values are not research results."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_baseline.py"
STARTS = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    steps = [{"name": "baseline-lublin_256-0-" + name, "status": "completed",
              "returncode": 0} for name in
             ["MARL-train", "MaskablePPO-train", "author-evaluate", "paper-evaluate"]]
    for step in steps[2:]:
        step.update(training_semantics="author", evaluation_semantics=step["name"].split("-")[-2])
    write_json(root / "state.json", {"status": "completed", "git_commit": "fixture-only",
               "git_status": "", "arguments": {"workloads": ["lublin_256"], "seeds": [0]}, "steps": steps})
    for mode in ("author", "paper"):
        target = root / "baseline/lublin_256/seed_0/evaluation" / mode
        rows = []
        csv_rows = []
        for algorithm, bsld, reutil in [("FCFS", 90., .4), ("F2", 80., .5),
                                      ("PPO", 56.358, .6767), ("MARL", 31.738 * 1.1, .8254)]:
            if mode == "paper" and algorithm == "MARL":
                bsld *= .9
                reutil += .01
            for iteration, start in enumerate(STARTS, 1):
                rows.append(dict(algorithm=algorithm, iteration=iteration, start=start,
                                 sequence_length=1024, seed=0, average_bounded_slowdown=bsld,
                                 renewable_energy_utilization=reutil, reward=reutil - .002 * bsld))
            csv_rows.append([algorithm, bsld, reutil])
        write_json(target / "rows.json", rows)
        write_json(target / "metadata.json", dict(workload="lublin_256", seed=0, backfill=1,
                   algorithms=["FCFS", "F2", "PPO", "MARL"], starts=STARTS,
                   sequence_length=1024, evaluation_policy="sample"))
        with (target / "result.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["algorithm", "average bounded slowdown", "renewable energy utilization"])
            writer.writerows(csv_rows)


class BaselineSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def module(self):
        self.assertTrue(SCRIPT.is_file(), "offline baseline summary tool is missing")
        spec = importlib.util.spec_from_file_location("baseline_summary", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def change_json(self, relative, change):
        path = self.root / relative
        data = json.loads(path.read_text())
        change(data)
        write_json(path, data)

    def test_running_run_is_rejected_without_output(self):
        self.change_json("state.json", lambda data: data.update(status="running"))
        self.assertNotEqual(self.module().main(["--run-dir", str(self.root)]), 0)
        self.assertFalse((self.root / "analysis").exists())

    def test_smoke_only_is_rejected(self):
        self.change_json("state.json", lambda data: data.update(steps=[dict(step, name=step["name"].replace("baseline-", "smoke-")) for step in data["steps"]]))
        self.assertNotEqual(self.module().main(["--run-dir", str(self.root)]), 0)

    def test_incomplete_baseline_step_is_rejected(self):
        self.change_json("state.json", lambda data: data["steps"][-1].update(status="failed"))
        self.assertNotEqual(self.module().main(["--run-dir", str(self.root)]), 0)

    def test_bad_metrics_and_wrong_means_are_rejected(self):
        module = self.module()
        for metric, value in [("average_bounded_slowdown", .9), ("average_bounded_slowdown", float("nan")),
                              ("renewable_energy_utilization", 1.1), ("renewable_energy_utilization", float("inf")),
                              ("average_bounded_slowdown", 100.)]:
            with self.subTest(metric=metric, value=value):
                fixture(self.root)
                self.change_json("baseline/lublin_256/seed_0/evaluation/author/rows.json",
                                 lambda rows: rows[0].update({metric: value}))
                self.assertNotEqual(module.main(["--run-dir", str(self.root)]), 0)
                self.assertFalse((self.root / "analysis").exists())

    def test_wrong_sequence_length_or_duplicate_start_is_rejected(self):
        module = self.module()
        for change in [lambda rows: rows[0].update(sequence_length=64),
                       lambda rows: rows[0].update(start=STARTS[1]), lambda rows: rows.pop()]:
            fixture(self.root)
            self.change_json("baseline/lublin_256/seed_0/evaluation/author/rows.json", change)
            self.assertNotEqual(module.main(["--run-dir", str(self.root)]), 0)

    def test_complete_baseline_and_difference_arithmetic(self):
        self.assertEqual(self.module().main(["--run-dir", str(self.root)]), 0)
        summary = json.loads((self.root / "analysis/summary.json").read_text())
        author = next(row for row in summary["evaluations"] if row["semantics"] == "author" and row["algorithm"] == "MARL")
        paper = next(row for row in summary["evaluations"] if row["semantics"] == "paper" and row["algorithm"] == "MARL")
        self.assertAlmostEqual(author["published_difference"]["bsld_relative_percent"], 10.)
        self.assertAlmostEqual(author["published_difference"]["reutil_percentage_points"], 1.)
        self.assertAlmostEqual(paper["author_sensitivity"]["bsld_relative_percent"], -10.)
        self.assertAlmostEqual(paper["author_sensitivity"]["reutil_percentage_points"], 1.)
        self.assertEqual(author["sequence_count"], 10)
        self.assertEqual(summary["independent_training_seeds"], {"lublin_256": [0]})
        self.assertIn("not independent training seeds", (self.root / "analysis/summary.md").read_text(encoding="utf-8"))
        self.assertEqual(author["evaluation_policy"], "sample")
        self.assertEqual(summary["evidence"]["source_snapshot"]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
