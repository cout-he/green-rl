"""Atomic, epoch-boundary training snapshots (load only trusted local snapshots)."""
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _environment_rng(env):
    if env is None:
        return None
    generator = env.np_random
    if hasattr(generator, 'get_state'):
        return {'kind': 'RandomState', 'state': generator.get_state()}
    return {'kind': 'Generator', 'state': generator.bit_generator.state}


def save_checkpoint(path, agent, completed_epoch, metadata, env=None, csv_path=None):
    """Save after a complete update and cleared rollout buffer, without drawing RNG."""
    payload = {
        'format_version': 1,
        'completed_epoch': completed_epoch,
        'metadata': dict(metadata),
        'actor': agent.actor_net.state_dict(),
        'critic': agent.critic_net.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_net_optimizer.state_dict(),
        'rng': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'environment': _environment_rng(env),
        },
        'csv_history': Path(csv_path).read_text(encoding='utf-8') if csv_path is not None else None,
    }
    _atomic_write(path, lambda stream: torch.save(payload, stream))


def load_checkpoint(path, agent, expected_metadata, env=None, csv_path=None):
    """Restore full training state, returning the number of completed epochs."""
    # NumPy/Python RNG states require full deserialization; actor-only files are not resumable.
    # Always read RNG byte tensors on CPU even when the model trains on CUDA.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    required = {'format_version', 'completed_epoch', 'metadata', 'actor', 'critic',
                'actor_optimizer', 'critic_optimizer', 'rng', 'csv_history'}
    if not isinstance(payload, dict) or not required.issubset(payload) or payload['format_version'] != 1:
        raise ValueError('Not a complete training checkpoint; actor weights alone cannot resume training')
    if payload['metadata'] != expected_metadata:
        differing = sorted(key for key in set(payload['metadata']) | set(expected_metadata)
                           if payload['metadata'].get(key) != expected_metadata.get(key))
        raise ValueError('Incompatible checkpoint metadata: ' + ', '.join(differing))
    epoch = payload['completed_epoch']
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError('Invalid completed epoch in checkpoint')
    rng = payload['rng']
    environment = rng['environment']
    if (environment is None) != (env is None):
        raise ValueError('Checkpoint/environment RNG context does not match')
    if rng['torch_cuda'] and (not torch.cuda.is_available()
                              or len(rng['torch_cuda']) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG device count differs from checkpoint')
    if csv_path is not None and payload['csv_history'] is None:
        raise ValueError('Checkpoint has no CSV history to restore')
    agent.actor_net.load_state_dict(payload['actor'])
    agent.critic_net.load_state_dict(payload['critic'])
    # PyTorch moves Adam moment tensors to each parameter's device on load.
    agent.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    agent.critic_net_optimizer.load_state_dict(payload['critic_optimizer'])
    if csv_path is not None:
        history = payload['csv_history'].encode('utf-8')
        _atomic_write(csv_path, lambda stream: stream.write(history))
    if env is not None:
        if environment['kind'] == 'RandomState':
            env.np_random.set_state(environment['state'])
        else:
            env.np_random.bit_generator.state = environment['state']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'].cpu())
    if rng['torch_cuda']:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng['torch_cuda']])
    return epoch
