"""Regression tests for experiment entry points; no training dependencies required."""
import ast
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def isolated_function(filename, name, namespace):
    """Execute the actual function body without importing unavailable ML packages."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / filename), "exec"), namespace)
    return namespace[name]


def checkpoint_agent(filename, directory):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPO")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("save_using_model_name", "load_using_model_name")]
    # The filesystem effects are real; only tensor serialization is substituted.
    def save(state, path):
        Path(path).write_text(state, encoding="utf-8")
    def load(path, **kwargs):
        return Path(path).read_text(encoding="utf-8")
    namespace = {"os": os, "torch": SimpleNamespace(save=save, load=load)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="CheckpointAgent", bases=[], keywords=[],
                                             body=methods, decorator_list=[])], type_ignores=[])),
                 str(ROOT / filename), "exec"), namespace)
    agent = namespace["CheckpointAgent"]()
    agent.device = "cpu"
    agent.actor_net = SimpleNamespace(state_dict=lambda: "actor-state", load_state_dict=lambda state: None)
    agent.critic_net = SimpleNamespace(state_dict=lambda: "critic-state", load_state_dict=lambda state: None)
    return agent


def execute_main(filename, arguments, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare))
    namespace["__name__"] = "__main__"
    with patch("sys.argv", [filename] + arguments):
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(ROOT / filename), "exec"), namespace)


class TrainingEntrypointTests(unittest.TestCase):
    def test_ppo_training_honors_requested_backfill(self):
        class EnvironmentReached(Exception):
            pass
        def environment(backfill):
            self.assertEqual(backfill, 0, "PPO silently overrides --backfill 0")
            raise EnvironmentReached
        train = isolated_function("MaskablePPO.py", "train", {"HPCEnv": environment,
                                  "configure_runtime": lambda *args: "cpu"})
        with self.assertRaises(EnvironmentReached):
            train("lublin_256", 0)

    def test_ppo_checkpoint_files_are_inside_requested_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            directory = Path(temporary) / "MaskablePPO"
            agent = checkpoint_agent("MaskablePPO.py", directory)
            agent.save_using_model_name(str(directory))
            self.assertTrue((directory / "actor.pkl").is_file())
            self.assertTrue((directory / "critic.pkl").is_file())
            agent.load_using_model_name(str(directory))

    def test_marl_checkpoint_directory_accepts_path_without_trailing_separator(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            directory = Path(temporary) / "MARL"
            agent = checkpoint_agent("MARL.py", directory)
            agent.save_using_model_name(str(directory))
            self.assertTrue((directory / "_actor.pkl").is_file())
            self.assertTrue((directory / "_critic.pkl").is_file())
            agent.load_using_model_name(str(directory))

    def test_more_than_ten_evaluation_starts_fail_with_clear_validation(self):
        env = SimpleNamespace(reset_for_test=lambda *args: None,
                              schedule_curr_sequence_reset=lambda score: ({1: 2.0}, .5),
                              schedule_LPTPN_sequence_reset=lambda: ({1: 2.0}, .5),
                              fcfs_score=None, f2_score=None)
        namespace = {"workload_name": "lublin_256", "random": __import__("random"),
                     "eta": .002, "GA_policy": lambda *a, **kw: (-2., .5),
                     "load_policy": lambda *a, **kw: None,
                     "RL_OneAction": lambda *a: (-2., .5),
                     "RL_MultiAction": lambda *a: (-2., .5)}
        run_policy = isolated_function("compare.py", "run_policy", namespace)
        with self.assertRaisesRegex(ValueError, "10|ten"):
            run_policy(env, 64, 11)

    def test_training_rejects_zero_epochs_before_constructing_environment(self):
        for filename in ("MARL.py", "MaskablePPO.py"):
            with self.subTest(filename=filename):
                train = isolated_function(filename, "train", {})
                with self.assertRaisesRegex(ValueError, "epochs"):
                    train("lublin_256", 0, epochs=0, traj_num=1)

    def test_evaluation_subset_resets_each_algorithm_and_records_per_sequence_metrics(self):
        resets = []
        env = SimpleNamespace(loads=SimpleNamespace(size=lambda: 10000), backfill=1,
                              reset_for_test=lambda *args: resets.append(args),
                              schedule_curr_sequence_reset=lambda score: ({1: 2.0}, .5),
                              fcfs_score=None, f2_score=None)
        namespace = {"os": os, "csv": csv, "json": json, "eta": .002,
                     "GA_policy": lambda *args, **kwargs: (-2., .5),
                     "configure_runtime": lambda *args: "cpu"}
        run_policy = isolated_function("compare.py", "run_policy", namespace)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            with redirect_stdout(io.StringIO()):
                run_policy(env, 64, 1, workload="lublin_256", model_root="models/seed_3",
                           output_dir=temporary, seed=3, device="cpu", num_threads=1,
                           algorithms=["FCFS", "F2", "GA"])
            self.assertEqual(resets, [(64, 6567), (64, 6567), (64, 6567)])
            with open(Path(temporary) / "result.csv", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["algorithm"] for row in rows], ["FCFS", "F2", "GA"])
            self.assertEqual(float(rows[0]["average bounded slowdown"]), 2.)
            self.assertEqual(float(rows[0]["renewable energy utilization"]), .5)
            self.assertEqual(float(rows[2]["average bounded slowdown"]), 2.)
            details = json.loads((Path(temporary) / "rows.json").read_text(encoding="utf-8"))
            self.assertEqual(details[0]["start"], 6567)
            self.assertEqual(details[0]["seed"], 3)
            self.assertEqual(details[0]["sequence_length"], 64)
            self.assertAlmostEqual(details[0]["reward"], .496)
            metadata = json.loads((Path(temporary) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["evaluation_policy"], "sample")
            self.assertEqual(metadata["seed"], 3)

    def test_training_cli_accepts_reproducible_run_controls(self):
        for filename in ("MARL.py", "MaskablePPO.py"):
            with self.subTest(filename=filename):
                calls = []
                execute_main(filename, ["--workload", "cirne", "--backfill", "2",
                                        "--epochs", "3", "--traj-num", "4", "--seed", "9",
                                        "--output-dir", "models/seed_9/cirne", "--device", "cpu",
                                        "--num-threads", "2"], {"train": lambda *args: calls.append(args)})
                self.assertEqual(calls, [("cirne", 2, 3, 4, 9, "models/seed_9/cirne", "cpu", 2)])

    def test_evaluation_cli_accepts_space_separated_algorithm_subset(self):
        calls = []
        env = SimpleNamespace(my_init=lambda **kwargs: None)
        execute_main("compare.py", ["--len", "64", "--iter", "1", "--seed", "7",
                                    "--model-root", "models/seed_7", "--output-dir", "evaluation",
                                    "--device", "cpu", "--num-threads", "2",
                                    "--algorithms", "FCFS", "F2", "PPO", "MARL"],
                     {"os": os, "HPCEnv": lambda **kwargs: env,
                      "run_policy": lambda *args, **kwargs: calls.append((args, kwargs))})
        self.assertEqual(calls[0][0], (env, 64, 1))
        self.assertEqual(calls[0][1]["algorithms"], ["FCFS", "F2", "PPO", "MARL"])
        self.assertEqual(calls[0][1]["model_root"], "models/seed_7")
        self.assertEqual(calls[0][1]["seed"], 7)
        self.assertEqual(calls[0][1]["device"], "cpu")
        self.assertEqual(calls[0][1]["num_threads"], 2)


if __name__ == "__main__":
    unittest.main()
