"""Plot cooperation, network structure, and full degree distributions."""

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 11, 'savefig.dpi': 180})


def plot_condition(folder):
    folder = Path(folder)
    settings = json.loads((folder / 'settings.json').read_text())
    with np.load(folder / 'averages.npz') as saved:
        metrics = saved['metrics']
        histograms = saved['degree_histogram']
        times = saved['sample_times']
    title = (f"{settings['network']}, r={settings['r']:g}, "
             f"pn={settings['pn']:g} ({settings['repeats']} runs)")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for column, color, label in zip(
            [2, 3, 4, 0], ['coral', 'dodgerblue', 'red', '#00dd99'],
            ['C-C', 'D-D', 'C-D', r'$f_C$']):
        ax.plot(metrics[:, column], color=color, label=label, lw=1.8)
    ax.set(xlabel='Time', ylabel='Proportion', ylim=(-0.03, 1.03), title=title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / 'evolution.png')
    fig.savefig(folder / 'evolution.pdf')
    plt.close(fig)

    selected = np.flatnonzero(times > 0)
    if len(selected) == 0:
        selected = np.array([0])
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 3.7), squeeze=False)
    for ax, sample_index in zip(axes[0], selected):
        histogram = histograms[sample_index]
        upper = int(np.flatnonzero(histogram.sum(axis=0) > 0)[-1])
        degree = np.arange(upper + 1)
        ax.bar(degree - 0.18, histogram[0, :upper + 1], width=0.36,
               color='tab:blue', label='Cooperator')
        ax.bar(degree + 0.18, histogram[1, :upper + 1], width=0.36,
               color='tab:orange', label='Defector')
        ax.set(xlabel='Degree', ylabel='Fraction of all nodes',
               title=f't={times[sample_index]}', xlim=(-0.8, upper + 0.8))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / 'degree_distribution.png')
    fig.savefig(folder / 'degree_distribution.pdf')
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', default=str(Path(__file__).resolve().parent / 'results'))
    args = parser.parse_args()
    folders = [path.parent for path in Path(args.folder).rglob('averages.npz')]
    if not folders:
        raise SystemExit('No averages.npz found')
    for folder in folders:
        plot_condition(folder)
