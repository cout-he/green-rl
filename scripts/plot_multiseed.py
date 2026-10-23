"""Plot every verified seed using descriptive SD; no training or input edits."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

EVIDENCE = ROOT / 'research/evidence/multiseed'
OUT = ROOT / 'reports/figures'
COLORS = {'FCFS': '#7F858B', 'F2': '#708B7B', 'PPO': '#B47B43', 'MARL': '#326E92'}
LABELS = {'FCFS': 'FCFS', 'F2': 'F2', 'PPO': 'PPO', 'MARL': 'GAS-MARL'}
ALGORITHMS = list(COLORS)
ANCHORS = {'PPO': (56.358, .6767), 'MARL': (31.738, .8154)}
MARKERS = ['o', '^', 's']


def export(fig, name):
    fig.canvas.draw()
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in visible) >= 7
    target = OUT / name
    fig.savefig(target.with_suffix('.png'), dpi=600, facecolor='white')
    fig.savefig(target.with_suffix('.pdf'), facecolor='white')
    fig.savefig(target.with_suffix('.svg'), facecolor='white')
    svg = target.with_suffix('.svg')
    svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    info = {'width_mm': fig.get_figwidth() * 25.4, 'height_mm': fig.get_figheight() * 25.4,
            'minimum_text_pt': min(t.get_fontsize() for t in visible)}
    plt.close(fig)
    return info


def evaluation(summary):
    seeds = [0, 1, 2]
    rows = [r for r in summary['per_seed'] if r['metric_mode'] == 'author']
    assert len(rows) == 12
    width_mm = 183
    fig, axes = plt.subplots(2, 2, figsize=(width_mm / 25.4, 150 / 25.4))
    fig.subplots_adjust(left=.09, right=.975, bottom=.20, top=.84, wspace=.34, hspace=.76)
    for panel, (key, ylabel) in enumerate([('bsld_mean', 'Average bounded slowdown'), ('reutil_mean', 'Renewable energy utilization')]):
        ax = axes[0, panel]
        for seed in seeds:
            values = [next(r[key] for r in rows if r['seed'] == seed and r['algorithm'] == a) for a in ALGORITHMS]
            ax.plot([x - .16 + .12 * seed for x in range(4)], values, color='#C8CBCD', linewidth=.6, zorder=1)
        for x, algorithm in enumerate(ALGORITHMS):
            values = [next(r[key] for r in rows if r['seed'] == seed and r['algorithm'] == algorithm) for seed in seeds]
            for seed, value in zip(seeds, values):
                ax.scatter(x - .16 + .12 * seed, value, color=COLORS[algorithm], marker=MARKERS[seed], s=22, zorder=3)
            ax.errorbar(x + .23, statistics.mean(values), yerr=statistics.stdev(values), fmt='D', markersize=3.3,
                        color=COLORS[algorithm], capsize=2.5, elinewidth=1.1, zorder=4)
            if algorithm in ANCHORS:
                ax.scatter(x + .41, ANCHORS[algorithm][panel], marker='s', facecolors='none', edgecolors='#222222', s=26, zorder=4)
        ax.set_xticks(range(4), [LABELS[a] for a in ALGORITHMS])
        ax.set(ylabel=ylabel, xlim=(-.4, 3.65))
        ax.margins(y=.12)
    for panel, metric in enumerate(('bsld_mean', 'reutil_mean')):
        ax = axes[1, panel]
        values = []
        for seed in seeds:
            marl = next(r[metric] for r in rows if r['seed'] == seed and r['algorithm'] == 'MARL')
            ppo = next(r[metric] for r in rows if r['seed'] == seed and r['algorithm'] == 'PPO')
            value = 100 * (marl / ppo - 1) if panel == 0 else 100 * (marl - ppo)
            values.append(value)
            ax.scatter(seed, value, color=COLORS['MARL'], marker=MARKERS[seed], s=28, zorder=3)
        ax.axhline(0, color='#A9A9A9', linewidth=.7, linestyle='--')
        ax.errorbar(3.4, statistics.mean(values), yerr=statistics.stdev(values), fmt='D', color=COLORS['MARL'], capsize=3, markersize=4)
        ax.set_xticks([0, 1, 2, 3.4], ['Seed 0', 'Seed 1', 'Seed 2', 'Mean ± SD'])
        ax.set(ylabel='BSLD difference (%)' if panel == 0 else 'ReUtil difference (pp)', xlim=(-.5, 4))
        ax.margins(y=.18)
    for panel, ax in enumerate(axes.flat):
        ax.text(-.16, 1.08, chr(97 + panel), transform=ax.transAxes, fontsize=8, fontweight='bold')
    handles = [Line2D([], [], color='#555555', marker=MARKERS[s], linestyle='none', markersize=4, label='Seed ' + str(s)) for s in seeds]
    handles += [Line2D([], [], color='#555555', marker='D', linestyle='none', markersize=4, label='Mean ± seed SD'),
                Line2D([], [], color='#222222', marker='s', markerfacecolor='none', linestyle='none', markersize=4, label='Paper Table 3/4')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .95), ncol=5, columnspacing=1.1, handlelength=1)
    fig.text(.09, .987, 'Three-seed paired evaluation under author semantics', fontsize=8, fontweight='bold', va='top')
    fig.text(.09, .11, 'Lower panels: GAS-MARL vs PPO within seed. Seed points average 10 fixed windows × 1024 jobs.', fontsize=7)
    fig.text(.09, .065, 'SD is descriptive across 3 end-to-end seeds; training seed equals base evaluation seed.', fontsize=7)
    fig.text(.09, .02, '11 window pairs overlap; no CI, significance test, or claim of exact numerical reproduction.', fontsize=7)
    return export(fig, 'gas_marl_multiseed_evaluation')


def training():
    data = {}
    source = []
    for algorithm in ('PPO', 'MARL'):
        for seed in (0, 1, 2):
            path = EVIDENCE / f'training_seed{seed}_{algorithm}.csv'
            with path.open(newline='', encoding='utf-8-sig') as stream:
                rows = [[float(v) for v in r] for r in csv.reader(stream)]
            assert len(rows) == 300 and all(len(r) == 3 and all(math.isfinite(v) for v in r) for r in rows)
            data[algorithm, seed] = {'reward': [r[0] for r in rows], 'reutil': [r[1] for r in rows], 'bsld': [-r[2] for r in rows]}
            for epoch, r in enumerate(rows, 1):
                source.append({'seed': seed, 'algorithm': algorithm, 'epoch': epoch, 'reward': r[0], 'reutil': r[1], 'bsld': -r[2]})
    with (OUT / 'gas_marl_multiseed_training_source.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source[0]))
        writer.writeheader()
        writer.writerows(source)
    width_mm = 183
    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, 85 / 25.4))
    fig.subplots_adjust(left=.08, right=.985, bottom=.28, top=.73, wspace=.49)
    for panel, (ax, key, label) in enumerate(zip(axes, ['reward', 'bsld', 'reutil'], ['Training reward', 'Average bounded slowdown', 'Renewable energy utilization'])):
        for algorithm in ('PPO', 'MARL'):
            values = [data[algorithm, seed][key] for seed in (0, 1, 2)]
            mean = [statistics.mean(v) for v in zip(*values)]
            sd = [statistics.stdev(v) for v in zip(*values)]
            epochs = range(1, 301)
            for series in values:
                ax.plot(epochs, series, linewidth=.4, color=COLORS[algorithm], alpha=.18, marker='.', markersize=.6)
            ax.fill_between(epochs, [m-s for m, s in zip(mean, sd)], [m+s for m, s in zip(mean, sd)], alpha=.13, color=COLORS[algorithm], linewidth=0)
            ax.plot(epochs, mean, color=COLORS[algorithm], linewidth=1, linestyle='--' if algorithm == 'PPO' else '-')
        ax.set(xlabel='Training epoch', ylabel=label, xlim=(1, 300))
        ax.set_xticks([1, 100, 200, 300])
        ax.text(-.23, 1.09, chr(97 + panel), transform=ax.transAxes, fontsize=8, fontweight='bold')
    handles = [Line2D([], [], color=COLORS[a], linestyle='--' if a == 'PPO' else '-', label=LABELS[a]) for a in ('PPO', 'MARL')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .91), ncol=2)
    fig.text(.08, .98, 'Author-mode training across all three seeds', fontsize=8, fontweight='bold', va='top')
    fig.text(.08, .13, 'All 1800 raw epoch rows retained; bold lines are epoch-wise means across 3 seeds.', fontsize=7)
    fig.text(.08, .065, 'Bands show ±1 sample seed SD; no smoothing or checkpoint selection.', fontsize=7)
    return export(fig, 'gas_marl_multiseed_training')


def main():
    summary_path = EVIDENCE / 'multiseed_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    geometry = {'evaluation': evaluation(summary), 'training': training()}
    inputs = [summary_path, *sorted(EVIDENCE.glob('training_seed*.csv'))]
    report = {'figures': geometry, 'matplotlib': matplotlib.__version__, 'input_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'exclusions': 'None within the three paired seeds. Legacy seed0 evaluation is intentionally excluded by protocol.',
              'transformations': 'BSLD sign reversed from stored training CSV; ten-window arithmetic seed means; ddof=1 seed SD; ratios and percentage-point differences shown explicitly.'}
    (OUT / 'gas_marl_multiseed_provenance.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(geometry))


if __name__ == '__main__':
    main()
