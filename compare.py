import csv
import json
import os
import random
import torch
from HPCSimPickJobs import *


from MaskablePPO import PPO
from GA import GA
from MARL import PPO as MARL, configure_runtime


def column_averages(matrix):
    transposed = list(zip(*matrix))

    averages = [sum(column) / len(column) for column in transposed]

    return averages

def safe_get(lst, index, default=None):
    return lst[index] if len(lst) > index else default

def load_policy(modelName, model_path, device='auto'):
    # handle which epoch to load from

    inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
    featureNum_size = [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE]
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if modelName=='MARL':
        model = MARL(batch_size=256, inputNum_size=inputNum_size,
                  featureNum_size=featureNum_size, device=device)
        model.load_using_model_name(model_path)

    elif modelName == 'PPO':
        model = PPO(batch_size=256, inputNum_size=inputNum_size,
                  featureNum_size=featureNum_size, device=device)
        model.load_using_model_name(model_path)
    else:
        raise ValueError("Unknown policy: " + modelName)

    model.actor_net.eval()
    model.critic_net.eval()
    return model


def RL_MultiAction(model,env):
    o = env.build_observation()
    running_num = 0

    reward = 0
    green_reward=0

    while True:
        lst = []
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                lst.append(1)
            elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                lst.append(1)
            else:
                lst.append(0)
        mask2 = np.zeros(action2_num, dtype=int)
        if running_num < delayMaxJobNum:
            mask2[running_num + 1:delayMaxJobNum + 1] = 1

        a1,a2=model.eval_action(o,lst,mask2)

        o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = env.step(a1.item(), a2.item())

        reward += r
        green_reward +=greenRwd
        if d:
            break
    return reward,green_reward

def RL_OneAction(model,env):
    o = env.build_observation()
    green_reward=0
    reward = 0
    while True:
        lst = []
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                lst.append(0)
            elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                lst.append(0)
            else:
                lst.append(1)
        for i in range(run_win):
            lst.append(0)
        for i in range(green_win):
            lst.append(0)

        a=model.eval_action(o,lst)
        o, r, d, r2, sjf_t, f1_t,running_num,greenRwd = env.step(a.item(), 0)

        reward += r
        green_reward += greenRwd
        if d:
            break

    return reward,green_reward

def GA_policy(env,eta):
    ga=GA(eta=eta)
    reward = 0
    green_reward=0

    while True:
        task_list = copy.deepcopy(env.job_queue)
        if len(task_list) == 1:
            exec_seq = [0]
        else:
            ga.init(task_list)
            exec_seq, _ = ga.run(env)
        for i in range(len(task_list)):
            selected_job = task_list[exec_seq[i]]
            if selected_job not in env.job_queue:
                continue
            ind = env.job_queue.index(selected_job)
            o, r, d, running_num, greenRwd = env.step_for_ga(ind, 0)

            reward += r
            green_reward += greenRwd

        if d:
            break

    return reward, green_reward

# @profile
def run_policy(env, nums, iters, workload='lublin_256', model_root=None,
               output_dir='.', seed=0, device='auto', num_threads=1, algorithms=None,
               seed_mode='legacy'):
    if seed_mode not in ('legacy', 'paired'):
        raise ValueError("seed_mode must be legacy or paired")
    if not 1 <= iters <= 10:
        raise ValueError("iters must be between 1 and 10 for the published evaluation starts")
    if nums < 1:
        raise ValueError("sequence length must be positive")
    available = ('FCFS', 'F2', 'LPTPN', 'GA', 'PPO', 'MARL')
    if algorithms is None:
        algorithms = list(available)
    elif isinstance(algorithms, str):
        algorithms = algorithms.split(',')
    algorithms = list(dict.fromkeys(part.strip().upper() for token in algorithms
                                   for part in token.split(',') if part.strip()))
    if not algorithms or any(name not in available for name in algorithms):
        raise ValueError("algorithms must be selected from " + ', '.join(available))
    start_list = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]  # List of randomly generated scheduling start points
    starts = start_list[:iters]
    if any(start + nums > env.loads.size() for start in starts):
        raise ValueError("sequence length extends beyond the workload at a published start")
    device = configure_runtime(seed, device, num_threads)
    model_root = os.fspath(model_root) if model_root is not None else '.'
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    models = {}
    for algorithm, subdirectory in (('PPO', 'MaskablePPO'), ('MARL', 'MARL')):
        if algorithm in algorithms:
            models[algorithm] = load_policy(algorithm, os.path.join(model_root, workload, subdirectory), device)
    rows = []
    for iter_num, start in enumerate(starts):
        for algorithm in algorithms:
            policy_seed = None
            if seed_mode == 'paired':
                # Fixed identities keep seed assignments independent of selection/order.
                policy_seed = (seed + 1000003 * (iter_num + 1) + 10007 * available.index(algorithm)) % (2 ** 32)
                device = configure_runtime(policy_seed, device, num_threads)
                if hasattr(env, 'seed'):
                    env.seed(policy_seed)
            env.reset_for_test(nums, start)
            if algorithm in ('FCFS', 'F2'):
                score = env.fcfs_score if algorithm == 'FCFS' else env.f2_score
                logs, green_reward = env.schedule_curr_sequence_reset(score)
                slowdown = sum(logs.values())
            elif algorithm == 'LPTPN':
                logs, green_reward = env.schedule_LPTPN_sequence_reset()
                slowdown = sum(logs.values())
            elif algorithm == 'GA':
                reward, green_reward = GA_policy(env, eta=eta)
                slowdown = -reward
            elif algorithm == 'PPO':
                reward, green_reward = RL_OneAction(models[algorithm], env)
                slowdown = -reward
            else:
                reward, green_reward = RL_MultiAction(models[algorithm], env)
                slowdown = -reward
            row = {"algorithm": algorithm, "iteration": iter_num + 1, "start": start,
                   "sequence_length": nums, "seed": seed,
                   "policy_seed": policy_seed,
                   "average_bounded_slowdown": float(slowdown),
                   "renewable_energy_utilization": float(green_reward),
                   "reward": float(green_reward - eta * slowdown)}
            rows.append(row)
            print(json.dumps({"event": "evaluation", **row}), flush=True)
    with open(os.path.join(output_dir, 'rows.json'), 'w', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)
    fields = ['algorithm', 'average bounded slowdown', 'renewable energy utilization']
    with open(os.path.join(output_dir, 'result.csv'), 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for algorithm in algorithms:
            selected = [row for row in rows if row['algorithm'] == algorithm]
            writer.writerow({'algorithm': algorithm,
                             'average bounded slowdown': sum(row['average_bounded_slowdown'] for row in selected) / len(selected),
                             'renewable energy utilization': sum(row['renewable_energy_utilization'] for row in selected) / len(selected)})
    metadata = {'workload': workload, 'seed': seed, 'device': str(device),
                'num_threads': num_threads, 'backfill': env.backfill,
                'model_root': os.path.abspath(model_root), 'algorithms': algorithms,
                'starts': starts, 'sequence_length': nums, 'evaluation_policy': 'sample'}
    metadata.update(evaluation_seed_mode=seed_mode, delay_semantics=getattr(env, 'delay_semantics', 'unknown'),
                    seed_derivation='(seed + 1000003 * iteration + 10007 * fixed_algorithm_index) mod 2**32'
                    if seed_mode == 'paired' else 'one shared stream for the complete evaluation')
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2)
    return rows


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='lublin_256')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--backfill', type=int, default=1)
    parser.add_argument('--model-root', default=None)
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--algorithms', nargs='+', default=None)
    parser.add_argument('--seed-mode', choices=['legacy', 'paired'], default='legacy')
    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_name=args.workload
    workload_file = os.path.join(current_dir, "./data/"+workload_name+".swf")
    # initialize the environment from scratch
    env = HPCEnv(backfill=args.backfill)
    env.my_init(workload_file=workload_file)


    start = time.time()
    run_policy(env, args.len, args.iter, workload=args.workload, model_root=args.model_root,
               output_dir=args.output_dir, seed=args.seed, device=args.device,
               num_threads=args.num_threads, algorithms=args.algorithms, seed_mode=args.seed_mode)
