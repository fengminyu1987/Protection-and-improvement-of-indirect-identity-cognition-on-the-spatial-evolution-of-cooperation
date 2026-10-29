"""Experiment parameters. Edit these values before running main.py."""

EXPERIMENT = {
    'N': 1500,
    'networks': ['Regular', 'BA'],
    'r_values': [5.4],
    'pn_values': [0.3, 0.5, 0.75],
    'rounds': {'Regular': 500, 'BA': 1000},
    'repeats': 10,
    'seed': 20261023,
    'workers': 2,
    'sample_times': [0, 100, 300, 500, 1000],
}
