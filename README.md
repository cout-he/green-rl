# GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning

## Reproduction project

This repository preserves the authors' source at tag `upstream-import` and develops a reproducible Python/PyTorch workflow on branch `research/gas-marl-reproduction`. The original source is from https://github.com/sioncr/green-rl-sched. Please cite Chen et al. as shown below.

The research setup uses Python 3.11 and an isolated `.venv`. Install the checked dependency set with `pip install -r requirements-research.txt`; the original `requirements.txt` is retained for provenance and contains conflicting NumPy pins. PyTorch is updated for the local RTX 5060, with the software/hardware deviation recorded in `research/environment.json`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_environment.py --require-cuda
.\.venv\Scripts\python.exe scripts/run_research.py --stage smoke --device cuda
.\.venv\Scripts\python.exe scripts/run_research.py --stage baseline --device cuda
```

`baseline` always runs the short smoke workflow first. It then trains GAS-MARL and PPO for the original 300 iterations with 100 trajectories of 256 jobs, initially on Lublin-256 with Green-Backfilling. Evaluation uses FCFS, F2, PPO, and GAS-MARL on common test starts. GA/LPTPN and other configurations remain a later expansion; this is not yet the entire paper experiment matrix.

Each experiment writes isolated models, logs, source snapshots, per-sequence results, and `state.json` under `results/<run-id>/`. The latest state is linked by `research/LATEST_RUN.json`. Training uses the authors' delay semantics. Evaluation separately records `author` and `paper` modes; the latter corrects only the tested Type 2 deadline event. Evaluating an author-trained model under `paper` semantics is a behavior audit, not an independently retrained improvement claim.

See `research/PROTOCOL.md` for the scientific targets, numerical anchors, and evidence rules. Local weights, environment files, and large logs are excluded from Git.

The bounded three-seed Lublin-256 + Green-Backfilling baseline is now complete. The [three-seed report](reports/三种子基线验证报告_GAS_MARL.md) records all six 300-epoch trainings and comparable paired evaluations. Author-mode GAS-MARL outperforms PPO in both metrics for each seed, but differs from the published numerical anchors; this does not reproduce the full paper experiment matrix or establish an innovation. Small source data and audit evidence are in `research/evidence/multiseed/`; the earlier single-seed report is retained.

Read-only checks and plots for the retained local runs:

```powershell
.\.venv\Scripts\python.exe scripts/verify_multiseed.py --run-dir results/gas-marl-20261022-multiseed-01
.\.venv\Scripts\python.exe scripts/analyze_multiseed.py
.\.venv\Scripts\python.exe scripts/plot_multiseed.py
```

These commands use local artifacts; trained weights are not part of the Git repository. `verify_multiseed.py` regenerates a timestamped audit report, so the packaged ledger's derived-report hashes describe the delivery-time audit, while its model/input hashes identify immutable experimental artifacts.

## Original project description

GAS-MARL is a green-aware job scheduling algorithm for HPC clusters based on multi-action deep reinforcement learning, which optimizes both renewable energy utilization and average bounded slowdown. This repository contains the source code of GAS-MARL and the datasets used.

## Install

All necessary packages can be installed with

```
pip install -r requirements.txt
```

## Set the configuration file

Between training and testing, we need to set the parameters of the experimental environment in the `configFile/config.ini`. Below is an example.

```
[GAS-MARL setting]
eta = 0.002
MAX_QUEUE_SIZE = 256
MAX_perNodePower=50
run_win = 64
green_win = 24
delayMaxJobNum=5
delayTimeList=[300,600,1200,1800,2400,3000,3600]

[general setting]
turbinePowerNominal = 7200
numberPv = 200
processor_per_machine = 8
idlePower = 50
```

Here’s the description for each parameter in the updated configuration:

### [training setting]

- **`eta`**: Penalty factors for model training
- **`MAX_QUEUE_SIZE`**: The maximum number of jobs in the waiting queue information.
- **`run_win`**: The maximum number of running jobs in the running jobs information.
- **`green_win`**: The number of slots in renewable energy information.
- **`delayMaxJobNum`**: The maximum number of `RN` in delay decision action type 2.
- **`delayTimeList`**: The delay time candidate list `DS` in delay decision action type 3.

### [general setting]

- **`turbinePowerNominal`**: The rated power of the wind turbine.
- **`numberPv`**: The effective irradiated area of the photovoltaic panel.
- **`processor_per_machine`**: The number of processors available per machine.
- **`idlePower`**: The power consumption (in watts) of a machine when it is idle.
- **`MAX_perNodePower`**: The maximum power consumption (in watts) allowed per processor.

## Training

Both GAS-MARL and PPO need to be trained first. Below are the command-line instructions for training the two algorithms.

The options for training GAS-MARL:

```
python MARL.py --workload [str] --backfill [int]
```

The options for training PPO:

```
python MaskablePPO.py --workload [str] --backfill [int]
```

Here are the descriptions for each option:

- **--workload:** the name of the job trace (lublin_256, Cirne, Jann)
- **--backfill:** the backfill policy, 0 = No backfilling, 1 = Greenn-backfilling, 2 = EASY-Backfilling.

## Testing

The options for testing all algorithm:

```
python compare.py --workload [str] --len [int] --iter [int] --backfill [int]
```

Here are the descriptions for each option:

- **--workload:** the name of the job trace (lublin_256, Cirne, Jann)
- **--len:** the length of the scheduling sequence
- **--iter:** the number of job sequences sampled
- **--backfill:** the backfill policy, 0 = No backfilling, 1 = Greenn-backfilling, 2 = EASY-Backfilling.

## An Example

Here is a complete example of training and testing:

First, set the `configFile/config.ini` as shown in the "Set the configuration file" section, and then start training GAS-MARL and PPO.

Train GAS-MARL with Green-Backfilling enabled:

```
python MARL.py --workload lublin_256 --backfill 1

```

Train PPO with Green-Backfilling enabled:

```
python MaskablePPO.py --workload lublin_256 --backfill 1
```

Test the performance of each algorithm with Green-Backfilling enabled:

```
python compare.py --workload lublin_256 --len 1024 --iter 10  --backfill 1 
```

## Output

The performance monitoring data of the model training process for GAS-MARL and PPO will be output separately in `MARL_[workload].csv` and `Maskable)[worklaod].csv`.

After running tests with `compare.py`, the results will be printed to the console and saved in `result.csv`.

## Citing GAS-MARL

```
If you reference or use GAS-MARL in your research, please cite:
@article{CHEN2025107760,
title = {GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning},
journal = {Future Generation Computer Systems},
volume = {167},
pages = {107760},
year = {2025},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2025.107760},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X2500055X},
author = {Rui Chen and Weiwei Lin and Huikang Huang and Xiaoying Ye and Zhiping Peng},
keywords = {Job scheduling, High-performance computing, Deep Reinforcement Learning, Renewable energy, Green computing},
}
```

### Acknowledgment

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases:

https://github.com/DIR-LAB/deep-batch-scheduler

http://github.com/laurentphilippe/greenpower
