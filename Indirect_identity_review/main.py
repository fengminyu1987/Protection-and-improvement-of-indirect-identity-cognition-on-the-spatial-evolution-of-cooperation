"""Run the experiments, save individual results, and generate figures."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import json
import multiprocessing
import numpy as np
import config
import model
from plot import plot_condition


def run_one(job):
    result = model.simulate(
        N=job['N'], network=job['network'], r=job['r'], pn=job['pn'],
        T=job['T'], seed=job['seed'], sample_times=job['sample_times'])
    path = Path(job['folder']) / f"rep_{job['rep']:03d}.npz"
    np.savez_compressed(path, **result)
    return path


def aggregate(folder, repeats):
    runs = []
    for rep in range(repeats):
        with np.load(folder / f'rep_{rep:03d}.npz') as saved:
            runs.append({key: saved[key] for key in saved.files})
    metrics = np.array([run['metrics'] for run in runs])
    
    count = np.isfinite(metrics).sum(axis=0)
    averages = np.full(count.shape, np.nan)
    np.divide(np.nansum(metrics, axis=0), count, out=averages, where=count > 0)
    np.savez_compressed(
        folder / 'averages.npz', metrics=averages,
        degree_histogram=np.mean([run['degree_histogram'] for run in runs], axis=0),
        sample_times=runs[0]['sample_times'], metric_names=np.array(model.METRICS))
    with (folder / 'time_series.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['round', *model.METRICS])
        for t, row in enumerate(averages):
            writer.writerow([t, *row])


def run_experiments(experiment, output):
    if experiment['N'] < 5 or experiment['repeats'] < 1 or experiment['workers'] < 1:
        raise ValueError('Invalid network size, repeat count, or worker count')
    for name in ['networks', 'r_values', 'pn_values']:
        if not experiment[name] or len(set(experiment[name])) != len(experiment[name]):
            raise ValueError(f'{name} must contain distinct values')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output folder is not empty. Choose a new --output folder.')
    folders, jobs = [], []
    for network in experiment['networks']:
        if network not in ['Regular', 'BA']:
            raise ValueError('Network must be Regular or BA')
        for r in experiment['r_values']:
            for pn in experiment['pn_values']:
                rounds = experiment['rounds'][network]
                if not 0 <= pn <= 1 or not np.isfinite(r) or r < 0 or rounds < 0:
                    raise ValueError('Invalid r, pn, or round count')
                folder = output / f'{network}_r{r}_pn{pn}'
                settings = dict(N=experiment['N'], network=network, r=r, pn=pn,
                                T=rounds, repeats=experiment['repeats'],
                                base_seed=experiment['seed'],
                                sample_times=experiment['sample_times'])
                folders.append((folder, settings))
                for rep in range(experiment['repeats']):
                    seed = int(experiment['seed'] + 1000000 * (network == 'BA')
                               + round(pn * 1000) * 100 + rep) % (2**32 - 97531)
                    jobs.append({**settings, 'folder': str(folder), 'rep': rep, 'seed': seed})
    for folder, settings in folders:
        folder.mkdir(parents=True)
        (folder / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    
    model.simulate(N=10, T=1, seed=123)
    if experiment['workers'] == 1:
        for i, job in enumerate(jobs, 1):
            print(f'{i}/{len(jobs)}: {run_one(job)}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=experiment['workers'],
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                print(f'{i}/{len(jobs)}: {future.result()}', flush=True)
    for folder, _ in folders:
        aggregate(folder, experiment['repeats'])
        plot_condition(folder)
    print(f'Results and figures saved to {output}', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(__file__).resolve().parent / 'results'))
    args = parser.parse_args()
    run_experiments(config.EXPERIMENT, args.output)
