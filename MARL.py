
import os

import csv
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
from HPCSimPickJobs import *
from training_checkpoint import save_checkpoint, load_checkpoint


def configure_runtime(seed=0, device='auto', num_threads=1):
    """Seed all sampling sources and explicitly select the execution device."""
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

class Buffer():
    def __init__(self):
        self.buffer_num = 0
        self.states = []
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.Returns = []
        self.advantages = []
        self.job_inputs = []

    def clear_buffer(self):
        self.buffer_num = 0
        self.states = []
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.Returns = []
        self.advantages = []
        self.job_inputs = []

    def store_buffer(self, state, mask1, mask2, action1, action2, log_prob1, log_prob2, Return, advantage, job_input,
                     nums):
        self.buffer_num = self.buffer_num + nums
        self.states.extend(state)
        self.masks1.extend(mask1)
        self.masks2.extend(mask2)
        self.actions1.extend(action1)
        self.actions2.extend(action2)
        self.log_probs1.extend(log_prob1)
        self.log_probs2.extend(log_prob2)
        self.Returns.extend(Return)
        self.advantages.extend(advantage)
        self.job_inputs.extend(job_input)


class ActorNet(nn.Module):

    def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3):
        super(ActorNet, self).__init__()
        self.d_model = 128

        self.num_inputs1 = num_inputs1
        self.featureNum1 = featureNum1
        self.num_inputs2 = num_inputs2
        self.featureNum2 = featureNum2
        self.num_inputs3 = num_inputs3
        self.featureNum3 = featureNum3

        self.embedding = nn.Linear(in_features=JOB_FEATURES, out_features=self.d_model)
        self.JobEncoder = nn.Sequential(
            nn.Linear(self.featureNum1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.GreenEncoder = nn.Sequential(
            nn.Linear(self.featureNum3, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(self.featureNum2, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.hidden = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, JOB_FEATURES),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.decoder2 = nn.Sequential(
            nn.Linear((self.num_inputs1 + self.num_inputs2 + self.num_inputs3 + 1) * JOB_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, action2_num),
        )

    def forward(self, x):
        job = x[:, :self.num_inputs1, :self.featureNum1]
        run = x[:, self.num_inputs1:self.num_inputs1 + self.num_inputs2, :self.featureNum2]
        green = x[:, self.num_inputs1 + self.num_inputs2:self.num_inputs1 + self.num_inputs2 + self.num_inputs3,
                :self.featureNum3]
        job = self.JobEncoder(job)
        run = self.RunningJobEncoder(run)
        green = self.GreenEncoder(green)

        return job, run, green

    def getActionn1(self, x, mask):
        encoder_out, _, _ = self.forward(x)
        logits = self.decoder1(encoder_out)

        logits = logits.squeeze(dim=-1)

        logits = logits - mask * 1e9
        probs = F.softmax(logits, dim=-1)

        return probs

    def getAction2(self, x, mask, job_input):
        job, run, green = self.forward(x)
        job_input = self.embedding(job_input)
        encoder_out = torch.cat([job, run, green, job_input], dim=1)
        encoder_out = self.hidden(encoder_out)
        encoder_out = self.flatten(encoder_out)
        logits = self.decoder2(encoder_out)
        logits = logits - mask * 1e9
        probs = F.softmax(logits, dim=-1)

        return probs


class CriticNet(nn.Module):

    def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3):
        super(CriticNet, self).__init__()
        self.d_model = 128

        self.num_inputs1 = num_inputs1
        self.featureNum1 = featureNum1
        self.num_inputs2 = num_inputs2
        self.featureNum2 = featureNum2
        self.num_inputs3 = num_inputs3
        self.featureNum3 = featureNum3

        self.JobEncoder = nn.Sequential(
            nn.Linear(self.featureNum1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.GreenEncoder = nn.Sequential(
            nn.Linear(self.featureNum3, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(self.featureNum2, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.hidden = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, JOB_FEATURES),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear((self.num_inputs1 + self.num_inputs2 + self.num_inputs3) * JOB_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        job = x[:, :self.num_inputs1, :self.featureNum1]
        run = x[:, self.num_inputs1:self.num_inputs1 + self.num_inputs2, :self.featureNum2]
        green = x[:, self.num_inputs1 + self.num_inputs2:self.num_inputs1 + self.num_inputs2 + self.num_inputs3,
                :self.featureNum3]
        green = self.GreenEncoder(green)
        job = self.JobEncoder(job)
        run = self.RunningJobEncoder(run)

        con = torch.cat([job, run, green], dim=1)

        con = self.hidden(con)
        con = self.flatten(con)
        value = self.out(con)
        return value


class PPO():
    def __init__(self, batch_size=10, inputNum_size=[], featureNum_size=[],
                 device='cpu'):
        super(PPO, self).__init__()
        self.num_inputs1 = inputNum_size[0]
        self.num_inputs2 = inputNum_size[1]
        self.num_inputs3 = inputNum_size[2]

        self.featureNum1 = featureNum_size[0]
        self.featureNum2 = featureNum_size[1]
        self.featureNum3 = featureNum_size[2]

        self.device = device
        self.actor_net = ActorNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3).to(self.device)
        self.critic_net = CriticNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3).to(self.device)
        self.batch_size = batch_size
        self.gamma = 1
        self.lam = 0.97

        self.states = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.rewards_seq = []
        self.actions1 = []
        self.actions2 = []
        self.values = []
        self.masks1 = []
        self.masks2 = []
        self.job_inputs = []
        self.buffer = Buffer()

        self.ppo_update_time = 8
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        self.entropy_coefficient = 0

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=0.0001, eps=1e-6)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=0.0005, eps=1e-6)

    def choose_action(self, state, mask1, mask2):
        with torch.no_grad():
            probs1 = self.actor_net.getActionn1(state, mask1)
        dist_bin1 = Categorical(probs=probs1)
        ac1 = dist_bin1.sample()
        log_prob1 = dist_bin1.log_prob(ac1)
        job_input = state[:, ac1]
        with torch.no_grad():
            probs2 = self.actor_net.getAction2(state, mask2, job_input)
        dist_bin2 = Categorical(probs=probs2)
        ac2 = dist_bin2.sample()
        log_prob2 = dist_bin2.log_prob(ac2)

        value = self.critic_net(state)
        return ac1, log_prob1, ac2, log_prob2, value, job_input

    def act_job(self, state, mask1, ac1):
        probs1 = self.actor_net.getActionn1(state, mask1)
        dist_bin1 = Categorical(probs=probs1)
        log_prob1 = dist_bin1.log_prob(ac1)
        entropy1 = dist_bin1.entropy()

        return log_prob1, entropy1

    def act_exc(self, state, mask2, job_input, ac2):
        probs2 = self.actor_net.getAction2(state, mask2, job_input)
        dist_bin2 = Categorical(probs=probs2)
        log_prob2 = dist_bin2.log_prob(ac2)
        entropy2 = dist_bin2.entropy()
        return log_prob2, entropy2

    def normalize(self, advantages):
        nor_advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-9)
        return nor_advantages

    def remember(self, state, value, log_prob1, log_prob2, action1, action2, reward, mask1, mask2, device, job_input):
        self.rewards_seq.append(reward)
        self.states.append(state.to("cpu"))
        self.log_probs1.append(log_prob1.to("cpu"))
        self.log_probs2.append(log_prob2.to("cpu"))
        self.values.append(value.to("cpu"))
        self.actions1.append(action1.to("cpu"))
        self.actions2.append(action2.to("cpu"))
        self.masks1.append(mask1.to("cpu"))
        self.masks2.append(mask2.to("cpu"))
        self.job_inputs.append(job_input.to("cpu"))

    def clear_memory(self):
        self.rewards_seq = []
        self.states = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.values = []
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []
        self.job_inputs = []

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        rews = np.append(np.array(self.rewards_seq), last_val)
        values = torch.cat(self.values, dim=0)
        values = values.squeeze(dim=-1)
        vals = np.append(np.array(values.cpu()), last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = self.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        ret = self.discount_cumsum(rews, self.gamma)[:-1]
        # ret=adv+vals[:-1]
        return adv, ret

    def storeIntoBuffter(self, reward):
        advantages, returns = self.finish_path(reward)
        returns = returns.tolist()
        advantages = advantages.tolist()

        self.buffer.store_buffer(self.states, self.masks1, self.masks2, self.actions1, self.actions2, self.log_probs1,
                                 self.log_probs2,
                                 returns, advantages, self.job_inputs, len(self.states))

    def compute_value_loss(self, states, returns):
        state_values = self.critic_net(states)
        state_values = torch.squeeze(state_values, dim=1)

        # Calculate value loss using F.mse_loss
        value_loss = F.mse_loss(state_values, returns)
        return value_loss

    def compute_actor_loss(self,
                           states,
                           masks1,
                           masks2,
                           actions1,
                           actions2,
                           advantages,
                           old_log_probs1,
                           old_log_probs2,
                           job_input
                           ):

        log_probs1, entropy1 = self.act_job(states, masks1, actions1)
        log_probs2, entropy2 = self.act_exc(states, masks2, job_input, actions2)
        # Compute the policy loss
        log1 = old_log_probs1 + old_log_probs2
        log2 = log_probs1 + log_probs2
        logratio = log2 - log1
        ratio = torch.exp(logratio)

        surr1 = ratio * advantages
        clip_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
        surr2 = clip_ratio * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))  # MAX->MIN descent
        entropy = (entropy1 + entropy2) / 2
        entropy_loss = torch.mean(entropy)

        total_loss = policy_loss - self.entropy_coefficient * entropy_loss

        return total_loss, policy_loss, entropy_loss

    def train(self):
        states = torch.cat(self.buffer.states, dim=0)
        masks1 = torch.cat(self.buffer.masks1, dim=0)
        masks2 = torch.cat(self.buffer.masks2, dim=0)
        actions1 = torch.cat(self.buffer.actions1, dim=0)
        log_probs1 = torch.cat(self.buffer.log_probs1, dim=0)
        actions2 = torch.cat(self.buffer.actions2, dim=0)
        log_probs2 = torch.cat(self.buffer.log_probs2, dim=0)
        job_inputs = torch.cat(self.buffer.job_inputs, dim=0)
        returns = torch.tensor(self.buffer.Returns)
        advantages = torch.tensor(self.buffer.advantages)
        advantages = self.normalize(advantages)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.states))), self.batch_size, False):
                index_tensor = torch.tensor(index)
                sampled_states = torch.index_select(states, dim=0, index=index_tensor).to(self.device)
                sampled_masks1 = torch.index_select(masks1, dim=0, index=index_tensor).to(self.device)
                sampled_masks2 = torch.index_select(masks2, dim=0, index=index_tensor).to(self.device)
                sampled_actions1 = torch.index_select(actions1, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs1 = torch.index_select(log_probs1, dim=0, index=index_tensor).to(self.device)
                sampled_actions2 = torch.index_select(actions2, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs2 = torch.index_select(log_probs2, dim=0, index=index_tensor).to(self.device)
                sampled_returns = torch.index_select(returns, dim=0, index=index_tensor).to(self.device)
                sampled_advantages = torch.index_select(advantages, dim=0, index=index_tensor).to(self.device)
                sampled_job_inputs = torch.index_select(job_inputs, dim=0, index=index_tensor).to(self.device)

                self.actor_optimizer.zero_grad()
                action_loss, polic_loss, entropy_loss = self.compute_actor_loss(sampled_states, sampled_masks1,
                                                                                sampled_masks2,
                                                                                sampled_actions1, sampled_actions2,
                                                                                sampled_advantages,
                                                                                sampled_log_probs1, sampled_log_probs2,
                                                                                sampled_job_inputs)
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_net_optimizer.zero_grad()
                value_loss = self.compute_value_loss(sampled_states, sampled_returns)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

    def save_using_model_name(self, model_name_path):
        os.makedirs(model_name_path, exist_ok=True)
        torch.save(self.actor_net.state_dict(), os.path.join(model_name_path, "_actor.pkl"))
        torch.save(self.critic_net.state_dict(),
                   os.path.join(model_name_path, "_critic.pkl"))

    def load_using_model_name(self, model_name_path):
        self.actor_net.load_state_dict(
            torch.load(os.path.join(model_name_path, "_actor.pkl"), map_location=self.device))
        self.critic_net.load_state_dict(
            torch.load(os.path.join(model_name_path, "_critic.pkl"), map_location=self.device))

    def eval_action(self,o,mask1,mask2):
        with torch.no_grad():
            o = o.reshape(1, MAX_QUEUE_SIZE + run_win + green_win, JOB_FEATURES)
            state = torch.FloatTensor(o).to(self.device)
            mask1 = np.array(mask1).reshape(1, MAX_QUEUE_SIZE)
            mask1 = torch.FloatTensor(mask1).to(self.device)
            mask2 = mask2.reshape(1, action2_num)
            mask2 = torch.FloatTensor(mask2).to(self.device)

            probs1 = self.actor_net.getActionn1(state,mask1)
            dist_bin1 = Categorical(probs=probs1)
            ac1 = dist_bin1.sample()
            job_input = state[:, ac1]
            probs2 = self.actor_net.getAction2(state, mask2, job_input)
            dist_bin2 = Categorical(probs=probs2)
            ac2 = dist_bin2.sample()

        return ac1, ac2

def train(workload, backfill, epochs=300, traj_num=100, seed=0, output_dir=None,
          device='auto', num_threads=1, checkpoint_every=5, resume=None):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if traj_num < 1:
        raise ValueError("traj_num must be positive")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be zero or positive")
    device = configure_runtime(seed, device, num_threads)
    env = HPCEnv(backfill=backfill)
    env.seed(seed)
    current_dir = os.getcwd()
    workload_name = workload
    workload_file = os.path.join(current_dir, "./data/" + workload_name + ".swf")
    env.my_init(workload_file=workload_file)

    output_dir = os.fspath(output_dir) if output_dir is not None else os.path.join(workload_name, 'MARL')
    os.makedirs(output_dir, exist_ok=True)

    inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
    featureNum_size = [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE]
    ppo = PPO(batch_size=256, inputNum_size=inputNum_size,
              featureNum_size=featureNum_size, device=device)
    csv_path = os.path.join(output_dir, 'MARL_' + workload_name + '.csv')
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
    metadata = {'algorithm': 'MARL', 'workload': workload, 'backfill': backfill,
                'seed': seed, 'eta': eta, 'windows': inputNum_size,
                'features': featureNum_size, 'traj_num': traj_num,
                'semantics': env.delay_semantics, 'delay_max_jobs': delayMaxJobNum,
                'delay_times': list(delayTimeList), 'gamma': ppo.gamma, 'lam': ppo.lam,
                'clip': ppo.clip_param, 'updates': ppo.ppo_update_time,
                'batch_size': ppo.batch_size}
    start_epoch = load_checkpoint(resume, ppo, metadata, env=env, csv_path=csv_path) if resume else 0
    if start_epoch > epochs:
        raise ValueError('Checkpoint has more completed epochs than requested total epochs')
    if resume:
        print(json.dumps({'event': 'resume', 'algorithm': 'MARL',
                          'completed_epoch': start_epoch, 'epochs': epochs,
                          'checkpoint': os.fspath(resume)}), flush=True)
    for epoch in range(start_epoch, epochs):
        o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
        running_num = 0
        t = 0
        epoch_reward = 0
        green_reward = 0
        wait_reward = 0
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
            with torch.no_grad():
                o = o.reshape(1, MAX_QUEUE_SIZE + run_win + green_win, JOB_FEATURES)
                state = torch.FloatTensor(o).to(device)
                mask1 = np.array(lst).reshape(1, MAX_QUEUE_SIZE)
                mask1 = torch.FloatTensor(mask1).to(device)
                mask2 = mask2.reshape(1, action2_num)
                mask2 = torch.FloatTensor(mask2).to(device)
                action1, log_prob1, action2, log_prob2, value, job_input = ppo.choose_action(state, mask1, mask2)
            ppo.remember(state, value, log_prob1, log_prob2, action1, action2, greenRwd, mask1, mask2, device,
                         job_input)
            o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = env.step(action1.item(), action2.item())
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t

            green_reward += greenRwd
            wait_reward += r
            epoch_reward += eta * r + greenRwd

            if d:
                t += 1
                ppo.storeIntoBuffter(eta * r + greenRwd)
                ppo.clear_memory()
                o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
                running_num = 0
                if t >= traj_num:
                    break

        ppo.train()
        with open(csv_path, mode='a',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [float(epoch_reward / traj_num), float(green_reward / traj_num), float(wait_reward / traj_num)])
        ppo.buffer.clear_buffer()
        if checkpoint_every and ((epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs):
            save_checkpoint(checkpoint_path, ppo, epoch + 1, metadata, env=env, csv_path=csv_path)
        print(json.dumps({"event": "epoch", "algorithm": "MARL", "epoch": epoch + 1,
                          "epochs": epochs, "traj_num": traj_num, "seed": seed,
                          "backfill": backfill, "device": str(device),
                          "reward": float(epoch_reward / traj_num),
                          "renewable_energy_utilization": float(green_reward / traj_num),
                          "average_bounded_slowdown": float(-wait_reward / traj_num)}), flush=True)

    ppo.save_using_model_name(output_dir)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='lublin_256')
    parser.add_argument('--backfill', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--traj-num', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=5,
                        help='Save complete training state every N epochs; 0 disables checkpoints')
    parser.add_argument('--resume', default=None, help='Resume a complete trusted training checkpoint')
    args = parser.parse_args()
    train_args = (args.workload, args.backfill, args.epochs, args.traj_num, args.seed,
                  args.output_dir, args.device, args.num_threads)
    if args.checkpoint_every != 5 or args.resume is not None:
        train_args += (args.checkpoint_every, args.resume)
    train(*train_args)
