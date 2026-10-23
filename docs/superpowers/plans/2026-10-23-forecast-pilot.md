# Frozen model forecast pilot implementation plan

> For agentic workers: use subagent-driven development for the two isolated implementation units, then review and verify integration before any scientific episode. User explicitly authorized autonomous execution of this existing research design.

**Goal:** Test the preregistered H1 without training or altering real renewable supply.

**Architecture:** A proxy separates prediction reads from physical energy integration. A standalone runner instruments existing environments and enforces source/model/input identity, zero-error equivalence and a fixed episode budget. Original modules and shared configuration remain unchanged.

**Tech stack:** Existing Python 3.11 PyTorch CUDA environment; standard-library manifests and traces; existing simulator; Word report if useful.

**Spec:** research/innovation_screening.md, research/PROTOCOL.md.

## Global constraints

- Three final MARL weights, author semantics, Lublin-256 and Green-Backfilling; no training or new hyperparameters.
- Select starts 6567 then 919 by original list order and non-overlap; use original iteration indices 1 and 3 for policy seeding.
- Current observed hour remains accurate; all future target intervals multiply the baseline prediction by 1.20, including repeated-window target intervals.
- Zero bias delegates exactly. Physical supply, jobs and physical energy integrators remain untouched.
- Run at most eighteen full episodes: six original controls, six zero-bias evaluations and six positive-bias evaluations; reuse zero data. All six zero gates pass before positive bias.
- Save manifest before execution. Fresh output, exclusive execution, action and total wall-clock limits. Keep partial failures, costs and negative outcomes.

## Review focus

1. Future intervals wrapping to the currently observed source slot must still be treated as future predictions.
2. Proxy/instrumentation must not consume RNG or change zero-bias actions, event/schedule traces, job records or physical metrics.
3. Job green features and backfill estimates must use predictions; actual green integration must never call perturbed forecasts.
4. Subset selection must retain original policy-seed iteration indices, rather than renumbering the selected windows.
5. No failure, resume or existing output can silently restart episodes or exceed the fixed full-episode budget.

## Task 1 Prediction isolation

Files: scripts/forecast_pilot_env.py, tests/test_forecast_pilot_env.py.

Interface: ForecastPowerProxy(physical, bias, current_time); install_forecast(env,bias) returns proxy. Metadata and counters are observational only. Methods retain original simulator signatures.

- [x] Write failing real-fixture tests for all channels, current/future/wrapped intervals, exact zero behavior, unchanged physical supply and energy, and invalid bias.
- [x] Implement the minimal proxy without modifying original modules.
- [x] Run focused tests and review scientific information boundaries.

## Task 2 Bounded execution and measurement

Files: scripts/run_forecast_pilot.py, tests/test_forecast_pilot_runner.py.

Interfaces: select_windows, evaluation_seed, zero_gate, summarize_h1 and a CLI with mandatory fresh output directory. Consume Task 1 only after unit API is stable.

- [x] Write failing tests for original window indices, duplicate output/limits, zero-gate categories, mixed-sign H1 and finite cost/energy summaries.
- [x] Implement frozen input/model checks, state/lock, transparent traces, exact gate, real-energy integration and all required metrics with missingness explicit.
- [x] Run focused tests; root runs the integrated suite under normal Windows permissions for taskkill tests.

## Task 3 Scientific gate and one execution

- [x] Independently review both units and confirm no baseline/config changes.
- [x] Freeze manifests and runtime limits before the first original/zero pair; inspect pair cost and stop on failed equivalence.
- [x] Complete remaining predeclared episodes only if gates and budget permit. Preserve every result and per-seed sign, with no retry grid or extra training.

## Task 4 Evidence and decision

- [x] Audit raw rows against summary, real energy, zero gates, all eighteen episode slots and hashes.
- [x] Create a concise report with negative results and costs, visually verify any Word output and prepare small evidence for the existing draft PR.
- [x] Update research state. Identify the two missing lawful direct-neighbor fulltexts and delete repeated wakeups; include this handoff in the user notification.
