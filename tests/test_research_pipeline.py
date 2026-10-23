"""Check process failures, artifacts, and timeouts through real subprocesses."""
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_research.py"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = SCRIPT.parents[1] / "results" / "test_tmp"
        cls.scratch.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location("run_research", SCRIPT)
        cls.pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.pipeline)

    def checkpoint_fixture(self, path, epoch=300):
        import torch
        from training_checkpoint import save_checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        actor, critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        agent = SimpleNamespace(actor_net=actor, critic_net=critic,
                                actor_optimizer=torch.optim.Adam(actor.parameters()),
                                critic_net_optimizer=torch.optim.Adam(critic.parameters()))
        save_checkpoint(path, agent, epoch, {"algorithm": path.parent.name})

    def test_completed_checkpoints_survive_two_successive_resume_runs(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            (root / "configFile").mkdir()
            (root / "configFile/config.ini").write_text("[GAS-MARL setting]\neta=0.002\nMAX_QUEUE_SIZE=256\n")
            (root / "data").mkdir()
            (root / "MARL.py").write_text("# unchanged training implementation\n")
            old = root / "results/A"
            suffix = Path("baseline/lublin_256/seed_0/models/lublin_256")
            for algorithm in ("MARL", "MaskablePPO"):
                self.checkpoint_fixture(old / suffix / algorithm / "checkpoint.pt")
            (old / "datasets.json").write_text("[]")
            (old / "source_manifest.json").write_text(json.dumps({"MARL.py": hashlib.sha256((root / "MARL.py").read_bytes()).hexdigest()}))
            trained = []
            def process_boundary(name, command, cwd, log_dir, *args):
                import torch
                if name.startswith("baseline-") and name.endswith("-train"):
                    if "--resume" not in command:
                        trained.append(name)
                    else:
                        source = Path(command[command.index("--resume") + 1])
                        if torch.load(source, map_location="cpu", weights_only=False)["completed_epoch"] != 300:
                            trained.append(name)
                    # A completed resume exports final weights, but has no new epoch to checkpoint.
                for artifact in args[1]:
                    Path(artifact).parent.mkdir(parents=True, exist_ok=True)
                    Path(artifact).write_bytes(b"successful process artifact")
                interrupted = "B" in Path(log_dir).parts and name == "baseline-lublin_256-0-author-evaluate"
                return {"name": name, "status": "failed" if interrupted else "completed", "command": command}
            with patch.object(self.pipeline, "ROOT", root), patch.object(self.pipeline, "git_output", return_value="test"), patch.object(self.pipeline, "execute_step", side_effect=process_boundary):
                self.assertEqual(self.pipeline.main(["--stage", "baseline", "--run-id", "B", "--resume-from", str(old)]), 1)
                previous = root / "results/B"
                for algorithm in ("MARL", "MaskablePPO"):
                    self.assertEqual((previous / suffix / algorithm / "checkpoint.pt").read_bytes(),
                                     (old / suffix / algorithm / "checkpoint.pt").read_bytes())
                self.assertEqual(self.pipeline.main(["--stage", "baseline", "--run-id", "C", "--resume-from", str(previous)]), 0)
                with patch.object(self.pipeline.shutil, "copyfileobj", side_effect=OSError("disk full")):
                    self.assertEqual(self.pipeline.main(["--stage", "baseline", "--run-id", "copy-failed", "--resume-from", str(previous)]), 1)
            self.assertEqual(trained, [], "Completed algorithms must not repeat the 300-epoch training")
            failed = json.loads((root / "results/copy-failed/state.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["steps"][-1]["status"], "failed")
            self.assertIn("disk full", failed["steps"][-1]["error"])
            self.assertIn("disk full", failed["error"])
            self.assertFalse((root / "results/copy-failed" / suffix / "MARL/checkpoint.pt").exists())
            steps = json.loads((root / "results/C/state.json").read_text())["steps"]
            for index, algorithm in ((4, "MARL"), (5, "MaskablePPO")):
                source = previous / suffix / algorithm / "checkpoint.pt"
                self.assertEqual(steps[index]["resume_checkpoint"], str(source))
                self.assertEqual((root / "results/C" / suffix / algorithm / "checkpoint.pt").read_bytes(), source.read_bytes())

    def test_checkpoint_inheritance_preserves_new_file_and_rejects_actor_only(self):
        import torch
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            source, target = root / "old/checkpoint.pt", root / "new/checkpoint.pt"
            self.checkpoint_fixture(source)
            target.parent.mkdir()
            target.write_bytes(b"new checkpoint from child")
            self.assertFalse(self.pipeline.inherit_completed_checkpoint(source, target, 300))
            self.assertEqual(target.read_bytes(), b"new checkpoint from child")
            target.unlink()
            original_copy = self.pipeline.shutil.copyfileobj
            def child_publishes_during_copy(original, output):
                original_copy(original, output)
                target.write_bytes(b"new checkpoint published during inheritance")
            with patch.object(self.pipeline.shutil, "copyfileobj", side_effect=child_publishes_during_copy):
                self.assertFalse(self.pipeline.inherit_completed_checkpoint(source, target, 300))
            self.assertEqual(target.read_bytes(), b"new checkpoint published during inheritance")
            self.assertEqual(list(target.parent.iterdir()), [target])
            target.unlink()
            self.checkpoint_fixture(source, epoch=127)
            self.assertFalse(self.pipeline.inherit_completed_checkpoint(source, target, 300))
            self.assertFalse(target.exists())
            torch.save({"weight": torch.ones(1)}, source)
            with self.assertRaisesRegex(ValueError, "complete.*checkpoint|checkpoint.*complete"):
                self.pipeline.inherit_completed_checkpoint(source, target, 300)
            self.assertFalse(target.exists())

    def test_checkpoint_inheritance_copy_failure_is_reported_without_partial_file(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            source, target = root / "old/checkpoint.pt", root / "new/checkpoint.pt"
            self.checkpoint_fixture(source)
            with patch.object(self.pipeline.shutil, "copyfileobj", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    self.pipeline.inherit_completed_checkpoint(source, target, 300)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])

    def test_success_records_output_and_exit_code(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            artifact = root / "artifact.txt"
            result = self.pipeline.execute_step(
                "ok", [sys.executable, "-c", "from pathlib import Path; Path('artifact.txt').write_text('evidence')"],
                root, root / "logs", timeout=10, expected=[artifact])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(artifact.read_text(), "evidence")

    def test_nonzero_exit_is_failed_and_keeps_stderr(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            result = self.pipeline.execute_step(
                "bad", [sys.executable, "-c", "import sys; print('root cause', file=sys.stderr); sys.exit(7)"],
                root, root / "logs", timeout=10)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["returncode"], 7)
            self.assertIn("root cause", Path(result["stderr"]).read_text())

    def test_missing_expected_artifact_is_not_success(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            result = self.pipeline.execute_step(
                "empty", [sys.executable, "-c", "pass"], root, root / "logs",
                timeout=10, expected=[root / "missing.csv"])
            self.assertEqual(result["status"], "failed")
            self.assertIn("missing.csv", result["error"])

    def test_timeout_terminates_process_and_records_failure(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            result = self.pipeline.execute_step(
                "slow", [sys.executable, "-c", "import time; time.sleep(20)"],
                root, root / "logs", timeout=0.2)
            self.assertEqual(result["status"], "timed_out")
            self.assertLess(result["elapsed_seconds"], 10)

    def test_state_replacement_is_valid_json(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            target = Path(tmp) / "nested" / "state.json"
            self.pipeline.write_json(target, {"status": "running", "step": 1})
            self.pipeline.write_json(target, {"status": "completed", "step": 2})
            self.assertEqual(json.loads(target.read_text()), {"status": "completed", "step": 2})

    def test_configuration_is_restored_if_evidence_copy_fails(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            config = root / "configFile" / "config.ini"
            config.parent.mkdir()
            original = b"[GAS-MARL setting]\neta=0.002\nMAX_QUEUE_SIZE=256\n"
            config.write_bytes(original)
            with patch.object(self.pipeline, "ROOT", root):
                with self.assertRaises(OSError):
                    with self.pipeline.experiment_config("Cirne", root / "absent" / "copy.ini"):
                        pass
            self.assertEqual(config.read_bytes(), original)

    def test_source_snapshot_preserves_executed_uncommitted_code(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            script = root / "MARL.py"
            script.write_text("original evidence", encoding="utf-8")
            run_dir = root / "results" / "run"
            run_dir.mkdir(parents=True)
            with patch.object(self.pipeline, "ROOT", root), patch.object(self.pipeline, "git_output", return_value="dirty diff"):
                self.pipeline.capture_source(run_dir)
            script.write_text("later changed", encoding="utf-8")
            self.assertEqual((run_dir / "source" / "MARL.py").read_text(), "original evidence")
            self.assertIn("MARL.py", json.loads((run_dir / "source_manifest.json").read_text()))

    def test_baseline_execution_runs_smoke_before_full_training(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            (root / "configFile").mkdir()
            (root / "configFile" / "config.ini").write_text("[GAS-MARL setting]\neta=0.002\nMAX_QUEUE_SIZE=256\n")
            (root / "data").mkdir()
            def successful_process(name, command, cwd, log_dir, *args):
                return {"name": name, "status": "completed", "command": command}
            with patch.object(self.pipeline, "ROOT", root), patch.object(self.pipeline, "git_output", return_value="test"), patch.object(self.pipeline, "execute_step", side_effect=successful_process):
                self.assertEqual(self.pipeline.main(["--stage", "baseline", "--run-id", "gate-test", "--evaluation-seed-mode", "paired"]), 0)
            state = json.loads((root / "results" / "gate-test" / "state.json").read_text())
            names = [step["name"] for step in state["steps"]]
            self.assertEqual(len(names), 8)
            self.assertTrue(all(name.startswith("smoke-") for name in names[:4]))
            self.assertTrue(all(name.startswith("baseline-") for name in names[4:]))
            for step in state["steps"]:
                if step["name"].endswith("evaluate"):
                    command = step["command"]
                    self.assertEqual(command[command.index("--seed-mode") + 1], "paired")

    def test_resume_uses_matching_baseline_checkpoint_without_resuming_smoke(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            (root / "configFile").mkdir()
            (root / "configFile" / "config.ini").write_text("[GAS-MARL setting]\neta=0.002\nMAX_QUEUE_SIZE=256\n")
            (root / "data").mkdir()
            old = root / "results" / "interrupted"
            checkpoint = old / "baseline/lublin_256/seed_0/models/lublin_256/MARL/checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            self.checkpoint_fixture(checkpoint, epoch=127)
            (root / "MARL.py").write_text("# training source fixture\n")
            (old / "datasets.json").write_text("[]")
            (old / "source_manifest.json").write_text(json.dumps({"MARL.py": hashlib.sha256((root / "MARL.py").read_bytes()).hexdigest()}))
            def successful_process(name, command, cwd, log_dir, *args):
                return {"name": name, "status": "completed", "command": command}
            with patch.object(self.pipeline, "ROOT", root), patch.object(self.pipeline, "git_output", return_value="test"), patch.object(self.pipeline, "execute_step", side_effect=successful_process):
                result = self.pipeline.main(["--stage", "baseline", "--run-id", "resumed", "--resume-from", str(old), "--checkpoint-every", "3"])
            self.assertEqual(result, 0)
            steps = json.loads((root / "results/resumed/state.json").read_text())["steps"]
            self.assertNotIn("--resume", steps[0]["command"])
            baseline_marl = steps[4]["command"]
            self.assertEqual(baseline_marl[baseline_marl.index("--resume") + 1], str(checkpoint))
            self.assertEqual(baseline_marl[baseline_marl.index("--checkpoint-every") + 1], "3")
            self.assertNotIn("--resume", steps[5]["command"])

    def test_resume_rejects_changed_data_or_training_implementation(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            data = root / "data" / "trace.swf"
            data.write_bytes(b"original data")
            module = root / "MARL.py"
            module.write_bytes(b"original implementation")
            old = root / "previous"
            old.mkdir()
            with patch.object(self.pipeline, "ROOT", root):
                (old / "datasets.json").write_text(json.dumps(self.pipeline.dataset_manifest()))
                (old / "source_manifest.json").write_text(json.dumps({"MARL.py": hashlib.sha256(module.read_bytes()).hexdigest()}))
                self.pipeline.verify_resume_provenance(old)
                data.write_bytes(b"modified data")
                with self.assertRaisesRegex(ValueError, "data"):
                    self.pipeline.verify_resume_provenance(old)
                data.write_bytes(b"original data")
                module.write_bytes(b"modified implementation")
                with self.assertRaisesRegex(ValueError, "source"):
                    self.pipeline.verify_resume_provenance(old)

    def test_timeout_stops_descendant_process_before_it_writes(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; time.sleep(1.2); Path('orphan.txt').write_text('still running')\"]); print('spawned', flush=True); time.sleep(20)"
            result = self.pipeline.execute_step("descendant", [sys.executable, "-c", code], root, root / "logs", timeout=0.6)
            import time
            time.sleep(1.5)
            self.assertEqual(result["status"], "timed_out")
            self.assertFalse((root / "orphan.txt").exists(), "timed-out child process survived its parent")


if __name__ == "__main__":
    unittest.main()
