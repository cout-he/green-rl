"""Checkpoint equivalence tests using tiny networks, without simulator training."""
import ast
import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training_checkpoint import load_checkpoint, save_checkpoint


HERE = Path(__file__).resolve().parent
METADATA = {"algorithm": "MARL", "workload": "synthetic", "backfill": 1,
            "seed": 43, "eta": .002, "windows": [256, 64, 24],
            "traj_num": 100, "semantics": "author"}


def seed_everything():
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(43)


def make_agent(device):
    actor = torch.nn.Linear(3, 2).to(device)
    critic = torch.nn.Linear(3, 1).to(device)
    return SimpleNamespace(actor_net=actor, critic_net=critic,
                           actor_optimizer=torch.optim.Adam(actor.parameters(), lr=.001),
                           critic_net_optimizer=torch.optim.Adam(critic.parameters(), lr=.002))


def update(agent, env, device):
    x = torch.randn(4, 3, device=device)
    scalar = random.random() + np.random.random() + env.np_random.random_sample()
    y = torch.randn(4, 2, device=device) + scalar
    agent.actor_optimizer.zero_grad()
    ((agent.actor_net(x) - y) ** 2).mean().backward()
    agent.actor_optimizer.step()
    agent.critic_net_optimizer.zero_grad()
    ((agent.critic_net(x) - y[:, :1]) ** 2).mean().backward()
    agent.critic_net_optimizer.step()


def snapshot(agent):
    return copy.deepcopy({"actor": agent.actor_net.state_dict(),
                          "critic": agent.critic_net.state_dict(),
                          "actor_adam": agent.actor_optimizer.state_dict(),
                          "critic_adam": agent.critic_net_optimizer.state_dict()})


def subsequent_draws(env):
    draws = [random.random(), np.random.random(4), torch.rand(4),
             env.np_random.random_sample(4)]
    if torch.cuda.is_available():
        draws.append(torch.rand(4, device="cuda"))
    return draws


class TrainingCheckpointTests(unittest.TestCase):
    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def check_resume_equivalence(self, device):
        seed_everything()
        agent = make_agent(device)
        env = SimpleNamespace(np_random=np.random.RandomState(19))
        for _ in range(5):
            update(agent, env, device)
        expected = snapshot(agent)
        expected_draws = subsequent_draws(env)

        seed_everything()
        interrupted = make_agent(device)
        env = SimpleNamespace(np_random=np.random.RandomState(19))
        for _ in range(2):
            update(interrupted, env, device)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, interrupted, 2, METADATA, env=env)
            resumed = make_agent(device)  # Consumes RNG; load must restore it.
            resumed_env = SimpleNamespace(np_random=np.random.RandomState(999))
            epoch = load_checkpoint(path, resumed, METADATA, env=resumed_env)
            self.assertEqual(epoch, 2)
            for _ in range(epoch, 5):
                update(resumed, resumed_env, device)
            self.assert_nested_equal(snapshot(resumed), expected)
            self.assert_nested_equal(subsequent_draws(resumed_env), expected_draws)

    def test_cpu_resume_matches_uninterrupted_parameters_adam_and_random_streams(self):
        self.check_resume_equivalence("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_resume_matches_uninterrupted_parameters_adam_and_random_streams(self):
        self.check_resume_equivalence("cuda")

    def test_incompatible_metadata_and_plain_actor_weights_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "latest.pt"
            agent = make_agent("cpu")
            save_checkpoint(path, agent, 2, METADATA)
            incompatible = {**METADATA, "backfill": 2}
            with self.assertRaises(ValueError):
                load_checkpoint(path, agent, incompatible)
            torch.save(agent.actor_net.state_dict(), path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, agent, METADATA)

    def test_resume_restores_csv_history_and_removes_unsaved_tail(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "latest.pt"
            csv_path = Path(directory) / "history.csv"
            csv_path.write_text("1,0.2\n2,0.3\n", encoding="utf-8")
            agent = make_agent("cpu")
            save_checkpoint(path, agent, 2, METADATA, csv_path=csv_path)
            csv_path.write_text("1,0.2\n2,0.3\n3,0.4\n", encoding="utf-8")
            load_checkpoint(path, agent, METADATA, csv_path=csv_path)
            self.assertEqual(csv_path.read_text(encoding="utf-8"), "1,0.2\n2,0.3\n")

    def test_failed_atomic_save_preserves_previous_checkpoint_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "latest.pt"
            agent = make_agent("cpu")
            save_checkpoint(path, agent, 2, METADATA)
            original = path.read_bytes()
            with patch("training_checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, agent, 3, METADATA)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(load_checkpoint(path, agent, METADATA), 2)
            self.assertEqual([entry.name for entry in Path(directory).iterdir()], ["latest.pt"])

    def test_incompatible_checkpoint_rejection_does_not_consume_rng(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "latest.pt"
            agent = make_agent("cpu")
            save_checkpoint(path, agent, 2, METADATA)
            seed_everything()
            expected = subsequent_draws(SimpleNamespace(np_random=np.random.RandomState(19)))
            seed_everything()
            env = SimpleNamespace(np_random=np.random.RandomState(19))
            with self.assertRaises(ValueError):
                load_checkpoint(path, agent, {**METADATA, "semantics": "paper"}, env=env)
            self.assert_nested_equal(subsequent_draws(env), expected)

    def test_training_cli_accepts_appended_checkpoint_controls(self):
        for filename in ("MARL.py", "MaskablePPO.py"):
            with self.subTest(filename=filename):
                calls = []
                def train(workload, backfill, epochs=300, traj_num=100, seed=0,
                          output_dir=None, device='auto', num_threads=1,
                          checkpoint_every=5, resume=None):
                    calls.append((workload, backfill, epochs, traj_num, seed, output_dir,
                                  device, num_threads, checkpoint_every, resume))
                source = ast.parse((HERE.parent / filename).read_text(encoding="utf-8"))
                main = next(node for node in source.body if isinstance(node, ast.If)
                            and isinstance(node.test, ast.Compare))
                namespace = {"__name__": "__main__", "train": train}
                with patch("sys.argv", [filename, "--workload", "cirne", "--epochs", "8",
                                        "--checkpoint-every", "2", "--resume", "latest.pt"]):
                    exec(compile(ast.Module(body=[main], type_ignores=[]), filename, "exec"), namespace)
                self.assertEqual(calls, [("cirne", 0, 8, 100, 0, None, 'auto', 1, 2, "latest.pt")])


if __name__ == "__main__":
    unittest.main()
