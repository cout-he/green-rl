# Baseline figure contract

Claim: One author-trained seed performs better than PPO in this local evaluation;
this is neither numerical reproduction of the paper nor a new algorithm.

Archetype: quantitative grids. Python/matplotlib exclusively; no external
template or image assets. Final width 183 mm, body/ticks/legends 7 pt,
panel letters 8 pt, editable PDF/SVG text, PNG previews at 600 dpi.

Figure 1: three panels show reward, AvgBSLD and ReUtil over all 300 epochs
for PPO and GAS-MARL. Every raw epoch is included as a faint line with points;
solid curves are strictly backward-looking ten-epoch means (available epochs
at the first nine positions). No extreme values are clipped. There is no
training-seed variability estimate because n=1 independently trained seed.

Figure 2: author semantics only, both policies trained with author semantics.
Two panels show all ten workload-window observations for each of FCFS, F2,
PPO and GAS-MARL, with mean and sample SD across windows. Gray connecting
lines identify matched windows. Open squares show supplied Table 3/4 anchors
for PPO and GAS-MARL; these anchors are reference means without fabricated
uncertainty. All ten windows have 1024 jobs; eleven pairs of windows overlap.
Window SD is descriptive, not independent-seed uncertainty. No CI, statistical
test, p-value or significance claim. The paper-semantics results are excluded
from these author-mode comparison panels because the original evaluation
also contains random-stream contamination.

Source: the completed gas-marl-20261021-baseline-02 run. Preserve original
CSV/JSON inputs and export every plotted observation, summary and anchor to
separate source-data CSVs. Record hashes, counts, numerical transformations,
automatic preflight and panel-by-panel visual QA alongside the figures.
