"""Real small checkpoint fixtures; synthetic values are not research results."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_multiseed.py'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    project = root / 'project'
    run = project / 'results' / 'fixture'
    (run / 'source').mkdir(parents=True)
    (run / 'source/MARL.py').write_text('# synthetic snapshot\n')
    (project / 'data').mkdir()
    (project / 'data/input.csv').write_text('synthetic\n')
    write_json(run / 'source_manifest.json', {'MARL.py': sha(run / 'source/MARL.py')})
    write_json(run / 'datasets.json', [{'path': 'data/input.csv', 'sha256': sha(project / 'data/input.csv'), 'bytes': 10}])
    steps = []
    for stage in ('smoke', 'baseline'):
        for seed in (1, 2):
            for folder in ('MARL', 'MaskablePPO', 'author', 'paper'):
                training = folder in ('MARL', 'MaskablePPO')
                name = f'{stage}-lublin_256-{seed}-{folder}-' + ('train' if training else 'evaluate')
                logs = run / stage / 'lublin_256' / f'seed_{seed}' / 'logs'
                logs.mkdir(parents=True, exist_ok=True)
                stdout, stderr = logs / f'{name}.stdout.log', logs / f'{name}.stderr.log'
                stdout.write_text('')
                stderr.write_text('')
                command = ['python', folder + '.py' if training else 'compare.py', '--workload', 'lublin_256', '--seed', str(seed), '--backfill', '1', '--device', 'cpu']
                if training:
                    command += ['--epochs', str(3 if stage == 'baseline' else 1), '--traj-num', str(100 if stage == 'baseline' else 2)]
                else:
                    command += ['--len', str(1024 if stage == 'baseline' else 64), '--iter', str(10 if stage == 'baseline' else 1), '--seed-mode', 'paired']
                step = dict(name=name, status='completed', returncode=0, stdout=str(stdout), stderr=str(stderr), command=command)
                if not training:
                    step.update(training_semantics='author', evaluation_semantics=folder)
                steps.append(step)
                if stage != 'baseline' or not training:
                    continue
                algorithm = 'MARL' if folder == 'MARL' else 'PPO'
                target = run / 'baseline/lublin_256' / f'seed_{seed}/models/lublin_256' / folder
                target.mkdir(parents=True)
                rows = [[.496, .5, -2.], [.497, .503, -3.], [.498, .506, -4.]]
                csv_path = target / f'{folder}_lublin_256.csv'
                with csv_path.open('w', newline='') as stream:
                    csv.writer(stream).writerows(rows)
                events = [dict(event='epoch', algorithm=algorithm, epoch=i, epochs=3, traj_num=100, seed=seed, backfill=1, reward=r, renewable_energy_utilization=u, average_bounded_slowdown=-b) for i, (r,u,b) in enumerate(rows, 1)]
                stdout.write_text('\n'.join(json.dumps(x) for x in events))
                actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
                optimizers = [torch.optim.Adam(actor.parameters()), torch.optim.Adam(critic.parameters())]
                for net, opt in zip((actor, critic), optimizers):
                    net(torch.tensor([[1., 2.]])).sum().backward()
                    opt.step()
                metadata = dict(algorithm=algorithm, workload='lublin_256', seed=seed, backfill=1, eta=.002, traj_num=100, semantics='author', windows=[256,64,24], features=[8,4,2], delay_max_jobs=5, delay_times=[300,600,1200,1800,2400,3000,3600], gamma=1, lam=.97, clip=.2, updates=8, batch_size=256)
                checkpoint = dict(format_version=1, completed_epoch=3, metadata=metadata, actor=actor.state_dict(), critic=critic.state_dict(), actor_optimizer=optimizers[0].state_dict(), critic_optimizer=optimizers[1].state_dict(), csv_history=csv_path.read_text(), rng=dict(python=random.Random(5).getstate(), numpy=np.random.RandomState(6).get_state(), torch_cpu=torch.Generator().manual_seed(7).get_state(), torch_cuda=[], environment=dict(kind='RandomState', state=np.random.RandomState(8).get_state())))
                torch.save(checkpoint, target / 'checkpoint.pt')
                prefix = '_' if folder == 'MARL' else ''
                torch.save(checkpoint['actor'], target / f'{prefix}actor.pkl')
                torch.save(checkpoint['critic'], target / f'{prefix}critic.pkl')
    write_json(run / 'state.json', dict(status='completed', git_commit='fixture-only', arguments=dict(stage='baseline', workloads=['lublin_256'], seeds=[1,2], evaluation_seed_mode='paired', device='cpu'), steps=steps))
    return project, run


class MultiseedVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.project, self.run = fixture(Path(temporary.name))

    def module(self):
        self.assertTrue(SCRIPT.is_file(), 'reusable multiseed artifact verifier is missing')
        spec = importlib.util.spec_from_file_location('verify_multiseed', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def audit(self):
        return self.module().verify_run(self.run, self.project, epochs=3)

    def change_json(self, relative, change):
        path = self.run / relative
        value = json.loads(path.read_text())
        change(value)
        write_json(path, value)

    def change_checkpoint(self, change):
        path = self.run / 'baseline/lublin_256/seed_1/models/lublin_256/MARL/checkpoint.pt'
        value = torch.load(path, weights_only=False)
        change(value)
        torch.save(value, path)

    def rejected(self):
        module = self.module()
        with self.assertRaises(module.VerificationError):
            module.verify_run(self.run, self.project, epochs=3)
        self.assertFalse((self.run / 'analysis').exists())

    def test_two_seeds_verify_with_exact_steps_and_leave_inputs_unchanged(self):
        before = {str(p): sha(p) for p in self.run.rglob('*') if p.is_file()}
        report = self.audit()
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['step_count'], 16)
        self.assertEqual(set(report['training']), {'lublin_256/seed_1/MARL', 'lublin_256/seed_1/PPO', 'lublin_256/seed_2/MARL', 'lublin_256/seed_2/PPO'})
        self.assertTrue(all(v['completed_epoch'] == 3 for v in report['training'].values()))
        self.assertEqual(before, {str(p): sha(p) for p in self.run.rglob('*') if p.is_file()})

    def test_running_run_is_rejected(self):
        self.change_json('state.json', lambda x: x.update(status='running'))
        self.rejected()

    def test_missing_step_is_rejected(self):
        self.change_json('state.json', lambda x: x['steps'].pop())
        self.rejected()

    def test_duplicate_step_is_rejected(self):
        self.change_json('state.json', lambda x: x['steps'].__setitem__(-1, x['steps'][0]))
        self.rejected()

    def test_nonzero_exit_is_rejected(self):
        self.change_json('state.json', lambda x: x['steps'][0].update(returncode=1))
        self.rejected()

    def test_nonempty_stderr_is_rejected(self):
        state = json.loads((self.run / 'state.json').read_text())
        Path(state['steps'][0]['stderr']).write_text('failure')
        self.rejected()

    def test_wrong_command_seed_is_rejected(self):
        self.change_json('state.json', lambda x: x['steps'][0]['command'].__setitem__(5, '999'))
        self.rejected()

    def test_source_snapshot_tamper_is_rejected(self):
        (self.run / 'source/MARL.py').write_text('tampered')
        self.rejected()

    def test_dataset_tamper_is_rejected(self):
        (self.project / 'data/input.csv').write_text('tampered')
        self.rejected()

    def test_missing_csv_epoch_is_rejected(self):
        path = self.run / 'baseline/lublin_256/seed_1/models/lublin_256/MARL/MARL_lublin_256.csv'
        path.write_text('.496,.5,-2\n.498,.506,-4\n')
        self.rejected()

    def test_repeated_log_epoch_is_rejected(self):
        state = json.loads((self.run / 'state.json').read_text())
        path = Path(next(x['stdout'] for x in state['steps'] if x['name'] == 'baseline-lublin_256-1-MARL-train'))
        events = [json.loads(x) for x in path.read_text().splitlines()]
        events[-1]['epoch'] = 2
        path.write_text('\n'.join(json.dumps(x) for x in events))
        self.rejected()

    def test_wrong_log_seed_is_rejected(self):
        state = json.loads((self.run / 'state.json').read_text())
        path = Path(next(x['stdout'] for x in state['steps'] if x['name'] == 'baseline-lublin_256-1-MARL-train'))
        path.write_text(path.read_text().replace('"seed": 1', '"seed": 0'))
        self.rejected()

    def test_checkpoint_epoch_mismatch_is_rejected(self):
        self.change_checkpoint(lambda x: x.update(completed_epoch=2))
        self.rejected()

    def test_checkpoint_seed_mismatch_is_rejected(self):
        self.change_checkpoint(lambda x: x['metadata'].update(seed=0))
        self.rejected()

    def test_checkpoint_semantics_mismatch_is_rejected(self):
        self.change_checkpoint(lambda x: x['metadata'].update(semantics='paper'))
        self.rejected()

    def test_checkpoint_csv_mismatch_is_rejected(self):
        self.change_checkpoint(lambda x: x.update(csv_history=''))
        self.rejected()

    def test_missing_adam_parameter_state_is_rejected(self):
        self.change_checkpoint(lambda x: x['actor_optimizer']['state'].pop(next(iter(x['actor_optimizer']['state']))))
        self.rejected()

    def test_nonfinite_adam_moment_is_rejected(self):
        self.change_checkpoint(lambda x: next(iter(x['actor_optimizer']['state'].values()))['exp_avg'].fill_(float('nan')))
        self.rejected()

    def test_invalid_rng_state_is_rejected(self):
        self.change_checkpoint(lambda x: x['rng'].update(torch_cpu=torch.tensor([1], dtype=torch.uint8)))
        self.rejected()

    def test_standalone_weight_mismatch_is_rejected(self):
        path = self.run / 'baseline/lublin_256/seed_1/models/lublin_256/MARL/_actor.pkl'
        weights = torch.load(path, weights_only=True)
        weights['weight'] += 1
        torch.save(weights, path)
        self.rejected()

    def test_nonfinite_weights_even_when_equal_are_rejected(self):
        self.change_checkpoint(lambda x: x['actor']['weight'].fill_(float('inf')))
        target = self.run / 'baseline/lublin_256/seed_1/models/lublin_256/MARL'
        torch.save(torch.load(target / 'checkpoint.pt', weights_only=False)['actor'], target / '_actor.pkl')
        self.rejected()


if __name__ == '__main__':
    unittest.main()
