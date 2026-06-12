from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Individual:
    """
    Individual representation for edge-based overlapping complex detection.

    chromosome:
        1D array of length E.
        chromosome[e] stores the index of an adjacent edge selected by edge e.

    complexes:
        decoded node communities / protein complexes.

    obj:
        objective vector [conductance_loss, go_coherence_loss].
    """

    chromosome: np.ndarray
    complexes: List[set[int]] = field(default_factory=list)

    obj: Optional[np.ndarray] = None

    obj1_conductance: float = 0.0
    obj2_go_coherence_loss: float = 0.0

    kv: int = 0
    avg_go: float = 0.0


def build_incident_edges(edge_pairs: np.ndarray, num_nodes: Optional[int] = None) -> List[List[int]]:
    """
    Build incident edge list for each node.

    MATLAB equivalent:
        incidentEdges{node} = list of edge indices touching that node

    Python indexing:
        nodes are assumed 0-based
        edges are 0-based

    Parameters
    ----------
    edge_pairs:
        E x 2 numpy array of integer node indices.
    num_nodes:
        optional number of nodes. If None, inferred from edge_pairs.

    Returns
    -------
    incident_edges:
        list where incident_edges[u] contains edge indices incident to node u.
    """

    edge_pairs = np.asarray(edge_pairs, dtype=int)

    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise ValueError("edge_pairs must be an E x 2 array.")

    if edge_pairs.size == 0:
        if num_nodes is None:
            num_nodes = 0
        return [[] for _ in range(num_nodes)]

    if num_nodes is None:
        num_nodes = int(edge_pairs.max()) + 1

    incident_edges: List[List[int]] = [[] for _ in range(num_nodes)]

    for e, (u, v) in enumerate(edge_pairs):
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            raise ValueError(f"Invalid node index in edge {e}: ({u}, {v})")

        incident_edges[u].append(e)
        incident_edges[v].append(e)

    return incident_edges


def build_adjacent_edge_list(edge_pairs: np.ndarray, num_nodes: Optional[int] = None) -> List[np.ndarray]:
    """
    Precompute adjacent edge candidates for every edge.

    Two edges are adjacent if they share at least one endpoint.

    MATLAB equivalent:
        candidates = [incidentEdges{u}, incidentEdges{v}]
        candidates = unique(candidates)
        candidates(candidates == e) = []

    Returns
    -------
    adjacent_edge_list:
        list of length E.
        adjacent_edge_list[e] is an array of candidate adjacent edge indices.
    """

    edge_pairs = np.asarray(edge_pairs, dtype=int)
    num_edges = edge_pairs.shape[0]

    incident_edges = build_incident_edges(edge_pairs, num_nodes=num_nodes)

    adjacent_edge_list: List[np.ndarray] = []

    for e, (u, v) in enumerate(edge_pairs):
        candidates = set(incident_edges[u])
        candidates.update(incident_edges[v])
        candidates.discard(e)

        adjacent_edge_list.append(np.array(sorted(candidates), dtype=int))

    if len(adjacent_edge_list) != num_edges:
        raise RuntimeError("Adjacent edge list size mismatch.")

    return adjacent_edge_list


def create_population_overlapping(
    edge_pairs: np.ndarray,
    population_size: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Individual]:
    """
    Create initial edge-based population.

    MATLAB equivalent:
        CreatePopulationOverlapping(proteinPairs, PopulationSize)

    Difference from MATLAB:
        MATLAB uses:
            edge indices: 1..E
            0 means no adjacent edge

        Python uses:
            edge indices: 0..E-1
            -1 means no adjacent edge
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive.")

    edge_pairs = np.asarray(edge_pairs, dtype=int)

    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise ValueError("edge_pairs must be an E x 2 array.")

    if rng is None:
        rng = np.random.default_rng()

    num_edges = edge_pairs.shape[0]
    adjacent_edge_list = build_adjacent_edge_list(edge_pairs)

    population: List[Individual] = []

    for _ in range(population_size):
        chromosome = np.full(num_edges, -1, dtype=int)

        for e in range(num_edges):
            candidates = adjacent_edge_list[e]

            if candidates.size > 0:
                chromosome[e] = int(rng.choice(candidates))
            else:
                chromosome[e] = -1

        population.append(Individual(chromosome=chromosome))

    return population


def copy_individual(ind: Individual) -> Individual:
    """
    Deep-ish copy of an Individual.

    Useful for MOEA/D replacement and child creation.
    """

    return Individual(
        chromosome=ind.chromosome.copy(),
        complexes=[set(c) for c in ind.complexes],
        obj=None if ind.obj is None else ind.obj.copy(),
        obj1_conductance=float(ind.obj1_conductance),
        obj2_go_coherence_loss=float(ind.obj2_go_coherence_loss),
        kv=int(ind.kv),
        avg_go=float(ind.avg_go),
    )