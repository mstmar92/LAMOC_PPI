from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import sparse

from encoding import Individual


MIN_COMPLEX_SIZE = 3
EPS = 1e-12


def prepare_adjacency_matrix(A) -> sparse.csr_matrix:
    """
    Convert adjacency matrix to CSR sparse matrix and remove diagonal.
    """

    A = sparse.csr_matrix(A, dtype=float)
    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def compute_conductance(
    members: set[int],
    A_ppi: sparse.csr_matrix,
    degrees: np.ndarray,
    total_volume: float,
) -> float:
    """
    Conductance of one complex.

    MATLAB equivalent:
        m_in = nnz(triu(A_sub_ppi, 1))
        volC = sum(deg(members))
        cutC = volC - 2*m_in
        denom = min(volC, totalVol - volC)
        phi = cutC / denom
    """

    if len(members) < MIN_COMPLEX_SIZE:
        return 1.0

    idx = np.array(sorted(members), dtype=int)

    A_sub = A_ppi[idx[:, None], idx]
    m_in = sparse.triu(A_sub, k=1).nnz

    vol_c = float(np.sum(degrees[idx]))
    cut_c = vol_c - 2.0 * float(m_in)

    denom = min(vol_c, float(total_volume) - vol_c)

    if denom <= EPS:
        phi = 1.0
    else:
        phi = cut_c / (denom + EPS)

    return float(np.clip(phi, 0.0, 1.0))


def compute_go_coherence(
    members: set[int],
    A_go: sparse.csr_matrix,
) -> float:
    """
    Average pairwise GO similarity inside one complex.

    MATLAB equivalent:
        goCoh = sum(triu(A_sub_go,1)) / pairCount
    """

    s = len(members)

    if s < MIN_COMPLEX_SIZE:
        return 0.0

    idx = np.array(sorted(members), dtype=int)

    A_sub = A_go[idx[:, None], idx]

    pair_count = s * (s - 1) / 2.0

    if pair_count <= 0:
        return 0.0

    go_sum = float(sparse.triu(A_sub, k=1).sum())
    go_coh = go_sum / (pair_count + EPS)

    return float(np.clip(go_coh, 0.0, 1.0))


def compute_individual_fitness(
    individual: Individual,
    A_ppi: sparse.csr_matrix,
    A_go: Optional[sparse.csr_matrix],
    degrees: np.ndarray,
    total_volume: float,
    use_go: bool,
    min_size: int = MIN_COMPLEX_SIZE,
) -> Individual:
    """
    Compute objectives for one individual.

    Equivalent to:
        ComputeOurFitnessCollectionOverlappingFast
    """

    if individual.complexes is None:
        raise ValueError("Individual must contain decoded complexes before fitness.")

    valid_complexes: List[set[int]] = []

    n = A_ppi.shape[0]

    for c in individual.complexes:
        members = {int(x) for x in c if 0 <= int(x) < n}

        if len(members) >= min_size:
            valid_complexes.append(members)

    kv = len(valid_complexes)

    if kv == 0:
        obj1 = 1.0
        avg_go = 0.0
        obj2 = 1.0
    else:
        cond_values = [
            compute_conductance(c, A_ppi, degrees, total_volume)
            for c in valid_complexes
        ]

        obj1 = float(np.mean(cond_values))

        if use_go and A_go is not None and A_go.nnz > 0:
            go_values = [
                compute_go_coherence(c, A_go)
                for c in valid_complexes
            ]
            avg_go = float(np.mean(go_values))
        else:
            avg_go = 0.0

        obj2 = 1.0 - avg_go

        obj1 = float(np.clip(obj1, 0.0, 1.0))
        avg_go = float(np.clip(avg_go, 0.0, 1.0))
        obj2 = float(np.clip(obj2, 0.0, 1.0))

    individual.obj = np.array([obj1, obj2], dtype=float)
    individual.obj1_conductance = obj1
    individual.obj2_go_coherence_loss = obj2
    individual.kv = kv
    individual.avg_go = avg_go

    return individual


def compute_population_fitness(
    population: List[Individual],
    A_ppi,
    A_go=None,
    population_size: int | None = None,
    min_size: int = MIN_COMPLEX_SIZE,
) -> List[Individual]:
    """
    Compute fitness for all individuals.

    MATLAB equivalent:
        ComputeFitnessOverlapping
    """

    if population_size is None:
        population_size = len(population)

    if population_size > len(population):
        raise ValueError("population_size is larger than actual population size.")

    A_ppi = prepare_adjacency_matrix(A_ppi)

    n = A_ppi.shape[0]

    degrees = np.asarray(A_ppi.sum(axis=1)).ravel()
    total_volume = float(np.sum(degrees))

    if A_go is None:
        A_go_prepared = sparse.csr_matrix((n, n), dtype=float)
        use_go = False
    else:
        A_go_prepared = prepare_adjacency_matrix(A_go)

        if A_go_prepared.shape != A_ppi.shape:
            raise ValueError("A_go size does not match A_ppi size.")

        A_go_prepared.data[A_go_prepared.data < 0] = 0.0
        A_go_prepared.eliminate_zeros()

        use_go = A_go_prepared.nnz > 0

    for i in range(population_size):
        compute_individual_fitness(
            individual=population[i],
            A_ppi=A_ppi,
            A_go=A_go_prepared,
            degrees=degrees,
            total_volume=total_volume,
            use_go=use_go,
            min_size=min_size,
        )

    return population