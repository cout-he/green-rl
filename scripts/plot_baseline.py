"""Plot completed, single-seed baseline evidence; never train or alter inputs.

All 300 training epochs and all 10 author-evaluation windows are retained.
Window sample SD is descriptive; the windows are not independent training seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

# Keep the font cache inside the writable project when the home directory is restricted.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"FCFS": "#7F858B", "F2": "#708B7B", "PPO": "#B47B43", "MARL": "#326E92"}
LABELS = {"FCFS": "FCFS", "F2": "F2", "PPO": "PPO", "MARL": "GAS-MARL"}
ALGORITHMS = ["FCFS", "F2", "PPO", "MARL"]
ANCHORS = {"PPO": (56.358, .6767), "MARL": (31.738, .8154)}
STARTS = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trailing_mean(values, width=10):
    """Backward-looking arithmetic mean; early epochs use available history."""
    return [statistics.mean(values[max(0, index - width + 1):index + 1]) for index in range(len(values))]


def load_data(run):
    state = load_json(run / "state.json")
    if state.get("status") != "completed":
        raise ValueError("Only completed baselines may be plotted")
    for suffix in ("MARL-train", "MaskablePPO-train", "author-evaluate", "paper-evaluate"):
        steps = [s for s in state["steps"] if s["name"] == "baseline-lublin_256-0-" + suffix]
        if len(steps) != 1 or steps[0]["status"] != "completed" or steps[0]["returncode"] != 0:
            raise ValueError("Missing completed full-baseline step: " + suffix)
    source = run / "baseline/lublin_256/seed_0"
    training, paths = {}, [run / "state.json", source / "config.ini"]
    for algorithm, directory in [("MARL", "MARL"), ("PPO", "MaskablePPO")]:
        path = source / "models/lublin_256" / directory / (directory + "_lublin_256.csv")
        paths.append(path)
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [[float(value) for value in row] for row in csv.reader(stream)]
        if len(rows) != 300 or any(len(row) != 3 or not all(math.isfinite(v) for v in row) for row in rows):
            raise ValueError("Expected exactly 300 finite three-metric training rows")
        training[algorithm] = {"reward": [r[0] for r in rows], "bsld": [-r[2] for r in rows],
                               "reutil": [r[1] for r in rows]}
    evaluation = source / "evaluation/author"
    paths.extend(evaluation / name for name in ("rows.json", "result.csv", "metadata.json"))
    rows = load_json(evaluation / "rows.json")
    metadata = load_json(evaluation / "metadata.json")
    if metadata["starts"] != STARTS or metadata["sequence_length"] != 1024 or metadata["evaluation_policy"] != "sample":
        raise ValueError("Unexpected evaluation protocol")
    if len(rows) != 40 or {r["algorithm"] for r in rows} != set(ALGORITHMS):
        raise ValueError("Expected every observation from four algorithms and ten windows")
    with (evaluation / "result.csv").open(encoding="utf-8", newline="") as stream:
        aggregates = {r["algorithm"]: r for r in csv.DictReader(stream)}
    for algorithm in ALGORITHMS:
        selected = [r for r in rows if r["algorithm"] == algorithm]
        if len(selected) != 10 or [r["start"] for r in selected] != STARTS:
            raise ValueError("Window identity/count mismatch")
        for row in selected:
            bsld, reutil = row["average_bounded_slowdown"], row["renewable_energy_utilization"]
            if not (math.isfinite(bsld) and bsld >= 1 and math.isfinite(reutil) and 0 <= reutil <= 1):
                raise ValueError("Invalid metric")
            if row["seed"] != 0 or row["sequence_length"] != 1024:
                raise ValueError("Unexpected training seed or sequence length")
        for key, column in [("average_bounded_slowdown", "average bounded slowdown"),
                            ("renewable_energy_utilization", "renewable energy utilization")]:
            if not math.isclose(statistics.mean(r[key] for r in selected), float(aggregates[algorithm][column]), abs_tol=1e-10):
                raise ValueError("Aggregate disagrees with source observations")
    return training, rows, paths, state


def export(fig, target):
    # Fixed 183 mm canvas; avoid tight-crop resizing of the physical figure.
    fig.canvas.draw()
    text = [item for item in fig.findobj(Text) if item.get_visible() and item.get_text()]
    minimum = min(item.get_fontsize() for item in text)
    if minimum < 5:
        raise ValueError("Text below the 5 pt glyph floor")
    for extension in (".png", ".pdf", ".svg"):
        fig.savefig(target.with_suffix(extension), dpi=600, facecolor="white")
        if extension == ".svg":
            path = target.with_suffix(extension)
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    return {"width_mm": fig.get_figwidth() * 25.4, "height_mm": fig.get_figheight() * 25.4,
            "minimum_source_text_pt": minimum, "visible_text_count": len(text)}


def training_figure(training, output):
    width_mm = 183
    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, 79 / 25.4))
    fig.subplots_adjust(left=.08, right=.985, bottom=.25, top=.73, wspace=.48)
    labels = {"reward": "Training reward", "bsld": "Average bounded slowdown", "reutil": "Renewable energy utilization"}
    for panel, (ax, metric) in enumerate(zip(axes, labels)):
        for algorithm in ("PPO", "MARL"):
            values = training[algorithm][metric]
            epochs = range(1, 301)
            ax.plot(epochs, values, color=COLORS[algorithm], alpha=.24, linewidth=.5,
                    marker=".", markersize=1.2, zorder=1)
            ax.plot(epochs, trailing_mean(values), color=COLORS[algorithm], linewidth=1.4,
                    linestyle="--" if algorithm == "PPO" else "-", zorder=2)
        ax.set(xlabel="Training epoch", ylabel=labels[metric], xlim=(1, 300))
        ax.set_xticks([1, 100, 200, 300])
        ax.margins(y=.08)
        ax.text(-.23, 1.08, chr(97 + panel), transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")
    handles = [Line2D([], [], color=COLORS[a], linewidth=1.4,
                      linestyle="--" if a == "PPO" else "-", label=LABELS[a]) for a in ("PPO", "MARL")]
    handles += [Line2D([], [], color="#858585", alpha=.35, linewidth=.6, marker=".", markersize=2,
                       label="Raw epochs"), Line2D([], [], color="#555555", linewidth=1.4,
                                                   label="Trailing 10-epoch mean")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .90), ncol=4,
               columnspacing=1.5, handlelength=2)
    fig.text(.08, .965, "Training under author semantics", fontsize=8, fontweight="bold", va="top")
    fig.text(.08, .09, "One independently trained seed per algorithm; all 300 raw epochs retained.", fontsize=7)
    fig.text(.08, .035, "Trailing mean uses available history for epochs 1–9; no uncertainty bands.", fontsize=7)
    result = export(fig, output / "gas_marl_baseline_training")
    plt.close(fig)
    return result


def evaluation_figure(rows, output):
    width_mm = 183
    fig, axes = plt.subplots(1, 2, figsize=(width_mm / 25.4, 94 / 25.4))
    fig.subplots_adjust(left=.08, right=.985, bottom=.27, top=.76, wspace=.30)
    keys = ["average_bounded_slowdown", "renewable_energy_utilization"]
    ylabels = ["Average bounded slowdown", "Renewable energy utilization"]
    # Deterministic display offsets identify ten observations; they do not alter y values.
    offsets = [-.16 + .28 * index / 9 for index in range(10)]
    for panel, (ax, key, ylabel) in enumerate(zip(axes, keys, ylabels)):
        for index in range(10):
            matched = [next(r[key] for r in rows if r["algorithm"] == a and r["iteration"] == index + 1) for a in ALGORITHMS]
            ax.plot([x + offsets[index] for x in range(4)], matched, color="#C3C7CB", alpha=.55,
                    linewidth=.4, zorder=1)
        for x, algorithm in enumerate(ALGORITHMS):
            values = [r[key] for r in rows if r["algorithm"] == algorithm]
            ax.scatter([x + offset for offset in offsets], values, s=13, color=COLORS[algorithm],
                       alpha=.8, edgecolors="none", zorder=3)
            ax.errorbar(x + .23, statistics.mean(values), yerr=statistics.stdev(values),
                        fmt="D", markersize=3.5, color=COLORS[algorithm], elinewidth=1.15,
                        capsize=2.6, capthick=1, zorder=4)
            if algorithm in ANCHORS:
                ax.scatter(x + .40, ANCHORS[algorithm][panel], s=25, marker="s",
                           facecolors="none", edgecolors="#262626", linewidths=1, zorder=5)
        ax.set_xticks(range(4), [LABELS[a] for a in ALGORITHMS])
        ax.set(ylabel=ylabel, xlim=(-.4, 3.65))
        ax.margins(y=.10)
        ax.text(-.12, 1.07, chr(97 + panel), transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")
    handles = [Line2D([], [], color="#777777", marker="o", linestyle="none", markersize=3, label="Workload window"),
               Line2D([], [], color="#555555", marker="D", linestyle="none", markersize=3.5, label="Mean ± window SD"),
               Line2D([], [], color="#262626", marker="s", markerfacecolor="none", linestyle="none", markersize=4,
                      label="Published Table 3/4 mean")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .91), ncol=3,
               columnspacing=1.8, handlelength=1)
    fig.text(.08, .97, "Author-mode evaluation of author-trained policies", fontsize=8, fontweight="bold", va="top")
    fig.text(.08, .145, "n = 1 training seed; 10 matched windows × 1024 jobs; gray lines connect the same window.", fontsize=7)
    fig.text(.08, .09, "11 window pairs overlap. Sample SD describes windows; no CI or significance test.", fontsize=7)
    fig.text(.08, .035, "Published means are references only; these results establish neither a new algorithm nor numerical reproduction.", fontsize=7)
    result = export(fig, output / "gas_marl_baseline_author_evaluation")
    plt.close(fig)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/gas-marl-20261021-baseline-02")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args(argv)
    training, rows, paths, state = load_data(args.run_dir.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "text.usetex": False})
    training_source = []
    for algorithm, values in training.items():
        smoothed = {key: trailing_mean(series) for key, series in values.items()}
        for index in range(300):
            training_source.append({"algorithm": LABELS[algorithm], "training_seed": 0, "epoch": index + 1,
                                    **{key: series[index] for key, series in values.items()},
                                    **{key + "_trailing_10_mean": series[index] for key, series in smoothed.items()}})
    write_csv(output / "gas_marl_baseline_training_source.csv", training_source)
    write_csv(output / "gas_marl_baseline_author_windows_source.csv", rows)
    summaries = []
    for algorithm in ALGORITHMS:
        selected = [r for r in rows if r["algorithm"] == algorithm]
        summaries.append({"algorithm": algorithm, "training_seed": 0, "window_count": len(selected),
                          **{key + suffix: function(r[key] for r in selected)
                             for key in ("average_bounded_slowdown", "renewable_energy_utilization")
                             for suffix, function in [("_mean", statistics.mean), ("_sample_sd", statistics.stdev)]},
                          "published_bsld_mean": ANCHORS.get(algorithm, (None, None))[0],
                          "published_reutil_mean": ANCHORS.get(algorithm, (None, None))[1]})
    write_csv(output / "gas_marl_baseline_author_summary_source.csv", summaries)
    geometry = {"training": training_figure(training, output), "author_evaluation": evaluation_figure(rows, output)}
    provenance = {"source_run": str(args.run_dir.resolve()), "git_commit_at_run_start": state["git_commit"],
                  "matplotlib_version": matplotlib.__version__, "figures": geometry,
                  "source_rows": {"training": len(training_source), "evaluation": len(rows)},
                  "exclusions": "None: all 600 epoch rows and 40 evaluation rows included.",
                  "transformations": "Training third CSV column negated to obtain positive BSLD; trailing mean width 10, backward only; fixed horizontal offsets for evaluation points.",
                  "uncertainty": "Sample SD (ddof=1) across ten overlapping windows, descriptive only; no seed SD, CI or significance test.",
                  "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (output / "gas_marl_baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "output_dir": str(output), "figures": geometry}))


if __name__ == "__main__":
    main()
