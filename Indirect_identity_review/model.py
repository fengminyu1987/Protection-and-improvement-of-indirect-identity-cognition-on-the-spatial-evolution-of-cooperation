"""Public-goods game with neighborhood-based rewiring and two-hop learning.

Strategies are encoded as C=0 and D=1.
"""

import random
import networkx as nx
import numpy as np
from numba import njit

METRICS = ['fc', 'edges', 'CC', 'DD', 'CD']

def initialize(n, network, seed):
    if n < 5 or network not in ['Regular', 'BA']:
        raise ValueError('Use N >= 5 and network Regular or BA')
    random.seed(int(seed))
    if network == 'BA':
        graph = nx.barabasi_albert_graph(n, 2)
    else:
        graph = nx.random_regular_graph(4, n)
    for node in graph:
        graph.nodes[node]['strategy'] = 0 if random.random() > 0.5 else 1
    labels = list(graph)
    index = {node: i for i, node in enumerate(labels)}
    adjacency = np.zeros((n, n), np.int32)
    degree = np.array([graph.degree(node) for node in labels], np.int64)
    strategy = np.array([graph.nodes[node]['strategy'] for node in labels], np.int64)
    for i, node in enumerate(labels):
        adjacency[i, :degree[i]] = [index[neighbor] for neighbor in graph[node]]
    return adjacency, degree, strategy, np.array(labels)


@njit(cache=True)
def payoffs(adjacency, degree, strategy, r):
    result = -(1 - strategy) * (degree + 1.0)
    for center in range(len(degree)):
        count = 1 - strategy[center]
        for k in range(degree[center]):
            count += 1 - strategy[adjacency[center, k]]
        benefit = r * count / (degree[center] + 1)
        result[center] += benefit
        for k in range(degree[center]):
            result[adjacency[center, k]] += benefit
    return result


@njit(cache=True)
def neighborhood_scores(adjacency, degree, payoff):
    result = np.zeros(len(degree))
    for i in range(len(degree)):
        high = payoff[i]
        correction = 0.0
        for k in range(degree[i]):
            value = payoff[adjacency[i, k]]
            total = high + value
            # Magnitudes choose the rounding correction; the sum remains signed.
            if abs(high) >= abs(value):
                correction += (high - total) + value
            else:
                correction += (value - total) + high
            high = total
        denominator = degree[i] + 1
        if denominator:
            result[i] = (high + correction) / denominator
    return result


@njit(cache=True)
def learn(adjacency, degree, strategy, payoff, i, draws, kappa):
    if degree[i] == 0:
        return strategy[i]
    candidates = np.zeros(len(degree), np.bool_)
    for k in range(degree[i]):
        neighbor = adjacency[i, k]
        for j in range(degree[neighbor]):
            candidates[adjacency[neighbor, j]] = True
    count = candidates.sum()
    if count == 0:
        return strategy[i]
    rank = int(draws[1] * count)
    model_node = -1
    for node in range(len(degree)):
        if candidates[node]:
            if rank == 0:
                model_node = node
                break
            rank -= 1
    difference = (payoff[i] - payoff[model_node]) / kappa
    if difference >= 0:
        probability = np.exp(-difference) / (1 + np.exp(-difference))
    else:
        probability = 1 / (1 + np.exp(difference))
    return strategy[model_node] if draws[2] < probability else strategy[i]


@njit(cache=True)
def remove_neighbor(adjacency, degree, i, j):
    position = -1
    for k in range(degree[i]):
        if adjacency[i, k] == j:
            position = k
            break
    if position < 0:
        raise ValueError('Missing edge')
    for k in range(position, degree[i] - 1):
        adjacency[i, k] = adjacency[i, k + 1]
    degree[i] -= 1


@njit(cache=True)
def add_edge(adjacency, degree, i, j):
    if i == j:
        raise ValueError('Self loop')
    for k in range(degree[i]):
        if adjacency[i, k] == j:
            raise ValueError('Duplicate insertion')
    adjacency[i, degree[i]] = j
    degree[i] += 1
    adjacency[j, degree[j]] = i
    degree[j] += 1


@njit(cache=True)
def rewire(adjacency, degree, score, i, pn, draws):
    if degree[i] == 0:
        return
    lowest = np.inf
    ties = 0
    for k in range(degree[i]):
        value = score[adjacency[i, k]]
        if value < lowest:
            lowest, ties = value, 1
        elif value == lowest:
            ties += 1
    rank = int(draws[3] * ties)
    old_neighbor = -1
    for k in range(degree[i]):
        node = adjacency[i, k]
        if score[node] == lowest:
            if rank == 0:
                old_neighbor = node
                break
            rank -= 1
    remove_neighbor(adjacency, degree, i, old_neighbor)
    remove_neighbor(adjacency, degree, old_neighbor, i)

    forbidden = np.zeros(len(degree), np.bool_)
    forbidden[i] = True
    weights = score - score.min()
    total = weights.sum()
    mass = 0.0
    for node in range(len(degree)):
        if not forbidden[node]:
            mass += weights[node]
    prefer = draws[4] < pn
    weighted = prefer and total > 0 and mass > 1e-12 * max(1.0, total)
    if not weighted:
        mass = len(degree) - forbidden.sum()
    draw = draws[5]
    threshold = draw * mass
    cumulative = 0.0
    target = -1
    for node in range(len(degree)):
        if not forbidden[node]:
            weight = weights[node] if weighted else 1.0
            if weight > 0:
                target = node
            cumulative += weight
            if cumulative > threshold:
                target = node
                break
    if target < 0:
        raise ValueError('No target')
    duplicate = False
    for k in range(degree[i]):
        if adjacency[i, k] == target:
            duplicate = True
            break
    if not duplicate:
        add_edge(adjacency, degree, i, target)


@njit(cache=True)
def measure(adjacency, degree, strategy):
    n = len(degree)
    metrics = np.zeros(5)
    histogram = np.zeros((2, n))
    links = np.zeros(3)
    metrics[0] = np.mean(strategy == 0)
    metrics[1] = degree.sum() / 2
    for i in range(n):
        histogram[strategy[i], degree[i]] += 1 / n
        for k in range(degree[i]):
            j = adjacency[i, k]
            if j > i:
                pair = strategy[i] + strategy[j]
                links[0 if pair == 0 else 1 if pair == 2 else 2] += 1
    for k in range(3):
        metrics[2 + k] = links[k] / metrics[1] if metrics[1] > 0 else np.nan
    return metrics, histogram


@njit(cache=True)
def kernel(adjacency, degree, strategy, r, pn, rounds, seed, times):
    np.random.seed(seed)
    n = len(degree)
    metrics = np.empty((rounds + 1, 5))
    histograms = np.empty((len(times), 2, n))
    for t in range(rounds + 1):
        metrics[t], histogram = measure(adjacency, degree, strategy)
        for k in range(len(times)):
            if times[k] == t:
                histograms[k] = histogram
        if t == rounds or metrics[t, 1] == 0:
            continue

        payoff = payoffs(adjacency, degree, strategy, r)
        score = neighborhood_scores(adjacency, degree, payoff)
        homogeneous = metrics[t, 0] == 0 or metrics[t, 0] == 1

        uniforms = np.random.random((n, 6))
        for i in range(n):
            draws = uniforms[i]
            rewire(adjacency, degree, score, i, pn, draws)
            if not homogeneous:
                strategy[i] = learn(adjacency, degree, strategy, payoff, i, draws, 0.1)
        if not np.isfinite(payoff).all() or not np.isfinite(score).all():
            raise ValueError('Non-finite payoff or neighborhood score')
    return metrics, histograms


def simulate(N=1500, network='Regular', r=5.4, pn=0.75, T=500,
             seed=12345, sample_times=(0, 100, 300, 500, 1000)):
    if not 0 <= pn <= 1 or not np.isfinite(r) or r < 0 or T < 0:
        raise ValueError('Invalid experiment parameter')
    adjacency, degree, strategy, _ = initialize(N, network, seed)
    times = np.array(sorted({0, T} | {int(t) for t in sample_times if 0 <= t <= T}),
                     np.int64)
    metrics, histograms = kernel(adjacency, degree, strategy, r, pn, T,
                                 int(seed + 97531) % (2**32), times)
    return dict(metrics=metrics, degree_histogram=histograms,
                sample_times=times, seed=seed, metric_names=np.array(METRICS))
