# Three seed figure and report QA

All figures use Python matplotlib 3.8.4, editable PDF/SVG, fixed 183 mm width and 600 dpi PNG previews. Source validator: 19 PASS, 1 WARN, 0 FAIL. The warning is the absence of unrequested TIFF; PDF/SVG and PNG meet this report's export contract. Both actual PDF text audits find a minimum of 7 pt and zero glyphs below 5 pt.

| Panel | Scientific role | Center and spread | Replicate | Visual inspection |
| --- | --- | --- | --- | --- |
| Evaluation a | Absolute author BSLD versus paper anchors | Each seed's 10-window mean; overall mean ± sample seed SD | 3 end-to-end seeds | Clear labels, every seed shown, anchor and SD remain distinct |
| Evaluation b | Absolute author ReUtil versus paper anchors | Same as panel a | Same as panel a | Seed1 green deficit is visible; no obscured error bars |
| Evaluation c | Paired BSLD relative change from PPO | Three ratios and mean ± sample SD of the ratios | Within-seed comparison | Shortened axis label removes initial collision with panel letter; zero reference shown |
| Evaluation d | Paired ReUtil difference from PPO | Three percentage-point differences and mean ± sample SD | Within-seed comparison | Same uncertainty definition as c; labels and error range clear |
| Training a | Reward learning trajectory | All raw epochs plus per-epoch mean ± sample seed SD | 3 seeds per algorithm | Two method families distinguishable by line style and color |
| Training b | Slowdown trajectory | Same as training a; stored negative BSLD sign reversed | Same as training a | All epochs retained, no clipped data or label collisions |
| Training c | Renewable utilization trajectory | Same as training a | Same as training a | Both uncertainty bands visible; no selected epochs |

No significance test, CI, synthetic observation, smoothing, best checkpoint selection or data exclusion is used. Seed0's legacy evaluation is excluded by protocol and replaced by its complete paired reevaluation. All 240 paired-mode rows and all 1800 training rows are retained in source data. The ten windows overlap in eleven pairs. Train and evaluation base seeds are coupled, so seed SD is end-to-end variability. FCFS/F2 are deterministic controls, not independently trained policies.

The four-page Word report was exported read-only in a separate hidden Word instance because LibreOffice is unavailable. Each page and all seven figure panels were visually inspected. Tables, headings, captions, fonts, page numbers and bottom margins are readable, with no clipped content, overlap, broken tables or blank pages. The Markdown header uses explicit line breaks within cells rather than physical newline-separated table rows. Earlier single-seed artifacts are preserved.
