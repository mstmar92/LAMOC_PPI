from __future__ import annotations

from typing import List

import numpy as np

from encoding import Individual


def _connected_components_from_edge_links(
    src: np.ndarray,
    dst: np.ndarray,
    num_edges: int,
) -> List[List[int]]:
    """
    Find connected components in the edge-level graph.

    The edge-level graph has:
        nodes = original PPI edge indices
        links = chromosome links between adjacent edges

    This matches MATLAB graph(src, dst, [], numEdges), which is undirected.
    """

    parent = list(range(num_edges))
    rank = [0] * num_edges

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)

        if ra == rb:
            return

        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in zip(src, dst):
        union(int(a), int(b))

    groups: dict[int, List[int]] = {}

    for e in range(num_edges):
        root = find(e)
        groups.setdefault(root, []).append(e)

    return list(groups.values())


def compute_complex_decoding_overlapping_fast(
    chromosome: np.ndarray,
    edge_pairs: np.ndarray,
) -> List[set[int]]:
    """
    Decode one edge-based chromosome into overlapping node complexes.

    MATLAB equivalent:
        ComputeCmplxDecodingOverlappingFast(chromosome, ProteinLabelPairs)

    Python indexing:
        chromosome[e] = adjacent edge index in 0..E-1
        chromosome[e] = -1 means no valid link

    Returns
    -------
    complexes:
        list of sets of node indices.
    """

    chromosome = np.asarray(chromosome, dtype=int)
    edge_pairs = np.asarray(edge_pairs, dtype=int)

    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise ValueError("edge_pairs must be an E x 2 array.")

    num_edges = chromosome.shape[0]

    if edge_pairs.shape[0] != num_edges:
        raise ValueError(
            f"chromosome length ({num_edges}) does not match number of edges "
            f"({edge_pairs.shape[0]})."
        )

    # MATLAB valid = chromosome > 0
    # Python equivalent: valid = chromosome >= 0
    valid = chromosome >= 0

    src = np.where(valid)[0]
    dst = chromosome[valid]

    # If no valid links, each edge becomes its own node complex.
    if src.size == 0:
        return [set(map(int, edge_pairs[e])) for e in range(num_edges)]

    # Remove invalid destination indices.
    keep_valid_dst = (dst >= 0) & (dst < num_edges)
    src = src[keep_valid_dst]
    dst = dst[keep_valid_dst]

    # Remove self-links.
    keep = src != dst
    src = src[keep]
    dst = dst[keep]

    # If only invalid/self links existed, each edge becomes its own complex.
    if src.size == 0:
        return [set(map(int, edge_pairs[e])) for e in range(num_edges)]

    # Connected components over edge indices.
    edge_components = _connected_components_from_edge_links(src, dst, num_edges)

    # Convert edge communities to node communities.
    complexes: List[set[int]] = []

    for comp_edges in edge_components:
        nodes: set[int] = set()

        for e in comp_edges:
            u, v = edge_pairs[e]
            nodes.add(int(u))
            nodes.add(int(v))

        complexes.append(nodes)

    return complexes


def decode_population(
    edge_pairs: np.ndarray,
    population: List[Individual],
    population_size: int | None = None,
) -> List[Individual]:
    """
    Decode all individuals.

    MATLAB equivalent:
        Individual2CmplxDecodingOverlapping(ProteinLabelPairs, Population, PopulationSize)
    """

    if population_size is None:
        population_size = len(population)

    if population_size > len(population):
        raise ValueError("population_size is larger than the actual population size.")

    for i in range(population_size):
        ind = population[i]

        if ind.chromosome is None:
            raise ValueError(f"population[{i}] does not contain chromosome.")

        ind.complexes = compute_complex_decoding_overlapping_fast(
            chromosome=ind.chromosome,
            edge_pairs=edge_pairs,
        )

    return population