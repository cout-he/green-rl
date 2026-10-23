# Three seed baseline figure contract

Claim: describe whether the author-trained GAS-MARL advantage over PPO recurs in the three paired end-to-end seed runs; do not infer exact paper reproduction or a novel improvement.

Archetype: quantitative grid. Python matplotlib only, following the established project workflow and saved backend.

Evaluation figure: panel a shows each seed's ten-window mean BSLD for every algorithm, panel b shows corresponding ReUtil. Panels c and d show within-seed GAS-MARL versus PPO BSLD relative difference and ReUtil percentage-point difference. The lower panels answer the paired-effect question and are not replacements for absolute values. All 240 evaluation rows (three seeds, two modes, four algorithms, ten windows) remain in source data; the author panels aggregate the 120 author rows, while a separate sensitivity table reports all three paper-mode comparisons.

Training figure: reward, BSLD and ReUtil panels show all six raw 300-epoch traces, three per algorithm, plus algorithm-level epoch means and sample SD. No epoch/seed selection. BSLD is the negation of the third stored CSV column. No best-checkpoint selection or smoothing is needed for the aggregate trace.

Replicate: three end-to-end seeds, each using the same training and base evaluation seed. Each seed mean uses ten fixed 1024-job windows, including eleven overlapping pairs. Across-seed sample SD uses ddof=1 on three seed means; it is descriptive and is not a confidence interval, significance test or pure training variance.

Source mapping: canonical per_seed.seed -> display marker; algorithm -> stable method color; bsld_mean/reutil_mean -> absolute axes; aggregate.bsld_seed_sd/reutil_seed_sd -> descriptive error bars; within-seed paired calculations -> lower panels. Training copies preserve original headerless three-column CSVs. Seed0 paired evaluation is used exclusively; legacy seed0 evaluation is excluded by protocol because it is not comparable.

Export: fixed 183 mm width; 7 pt minimum normal/tick text, 8 pt panel labels; editable PDF and SVG plus 600 dpi PNG. TIFF is not requested. Source CSV/JSON, input SHA256, scripts and this contract accompany the exports. No raster manipulation or external visual template. Reuse is style only from the prior project figures; all statistics are recomputed for the new replicate unit.

QA gate: source preflight, actual PDF text-size audit, inspect each panel and each complete figure, then inspect the Word report pages. Record exclusions and any remaining warnings. Preserve the earlier single-seed report and figures.
