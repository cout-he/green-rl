# Baseline figure QA

Completed 2026-10-22. Python/matplotlib 3.8.4 exclusively; no R, external
templates, generated observations, significance tests or CI. Original inputs
remain unchanged. Input SHA256 values and source commit are recorded in
gas_marl_baseline_provenance.json.

## Data and statistics

- Figure 1 includes every epoch: 300 per algorithm, 600 source rows. The third
  training CSV column is negative slowdown and is negated for positive AvgBSLD.
  Both raw values and strictly trailing ten-epoch means are included in source
  CSV. Epochs 1–9 use all available preceding history. No exclusions or clipping
  of extremes, no seed variability estimate, and no invented uncertainty band.
- Figure 2 includes every original-author observation: four algorithms × ten
  windows = 40 source rows. Center is arithmetic mean; error bars are sample
  SD with denominator n−1 across workload windows. All algorithms use identical
  starts and 1024 jobs. Eleven pairs of windows overlap. There is only one
  independently trained seed; these are not ten independent training repeats.
  Gray connections preserve matched-window identity.
- Open squares represent the supplied published mean anchors: PPO
  56.358/0.6767; GAS-MARL 31.738/0.8154. No reference uncertainty is invented.
- Figure 2 uses the original author evaluation, not the separate paired
  re-evaluation. The original paper-semantics results are not displayed.
  The claim is limited to the single-seed local comparison; neither numerical
  paper reproduction nor a new algorithm is established.

## Automated checks

- Source preflight: 19 PASS, 1 WARN, 0 FAIL; exit 0. The sole warning is that
  PNG previews are provided without a TIFF submission raster. This is accepted
  because the requested bundle is PNG/PDF/SVG and editable vector exports are
  the publication masters. No submission-format claim is made for PNG.
- Both exported PDFs: text audit exit 0, minimum actual text run 7 pt, zero
  glyphs below 5 pt, and no warnings. PDF uses embedded TrueType text; SVG
  retains text elements. Body, axis, tick and legend text is 7 pt; panel labels
  are 8 pt.
- Fixed canvas width is exactly 183 mm. Training height is 79 mm; evaluation
  height is 94 mm. PNG previews are 600 dpi; PDF/SVG remain vector.
- Geometry QA checks visible text bounding boxes against the complete canvas
  and records all axis ranges. No text lies outside the canvas.

## Visual inspection

Both final PNGs were opened after the final render. All five panels and both
complete assemblies were inspected, including raw extremes, uncertainty
extents, axis labels, legends, panel letters and footer statements.

| Panel | Unique evidence | Center/spread | Replicate unit | Visual result |
| --- | --- | --- | --- | --- |
| Training a | Reward trajectory over 300 epochs | Raw points and trailing mean; no uncertainty band | One training seed, epoch trajectory | All values visible; curves and labels clear |
| Training b | Positive AvgBSLD trajectory | Same smoothing rule; raw extremes retained | Same single training seed | Full slowdown range visible; no clipping |
| Training c | Renewable utilization trajectory | Same smoothing rule; raw values retained | Same single training seed | Both method curves readable; palette consistent |
| Evaluation a | Per-window slowdown and reference means | All ten points per algorithm; mean ± sample window SD | Ten matched, overlapping windows | Points, error bars and open references separated |
| Evaluation b | Per-window renewable utilization | Same descriptive SD definition | Same overlapping windows | All data and uncertainty extents visible |

Neutral baselines are subordinate to the colored policy comparison; PPO is
ochre/dashed and GAS-MARL blue/solid in training panels. Legends have no opaque
mask or border. No labels collide with marks, uncertainty bars or adjacent
panels. Footnotes remain inside the canvas. No image-processing adjustments,
photographic assets or microscopy elements are used.
