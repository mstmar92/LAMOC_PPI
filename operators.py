from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import sparse

from encoding import Individual, copy_individual


@dataclass
class MutationParams:
    min_size: int = 3
    p_add_triangle: float = 0.6
    p_prune: float = 0.4
    max_complex_adds_per_node: int = 2
    max_moves_per_node: int = 1

    go_score_weight: float = 1.0
    go_add_threshold: float = 0.15
    go_prune_threshold: float = 0.10


@dataclass
class RepairParams:
    min_size: int = 3

    core_threshold: float = 0.60
    remove_threshold: float = 0.25

    alpha_topo: float = 0.50
    beta_tri: float = 0.30
    gamma_go: float = 0.20

    add_threshold: float = 0.45
    go_add_threshold: float = 0.10
    min_internal_neighbors: int = 2
    max_boundary_adds: int = 5
    max_candidate_pool: int = 20


def prepare_sparse(A) -> sparse.csr_matrix:
    A = sparse.csr_matrix(A, dtype=float)
    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def build_neighbors_from_adjacency(A_ppi: sparse.csr_matrix) -> List[np.ndarray]:
    """
    Build neighbour list equivalent to:
        IndicesInteractionProtein
        NumInteractionProtein

    Returns list where neighbors[u] is numpy array of adjacent node indices.
    """

    A_ppi = prepare_sparse(A_ppi)
    n = A_ppi.shape[0]

    neighbors: List[np.ndarray] = []

    for u in range(n):
        start = A_ppi.indptr[u]
        end = A_ppi.indptr[u + 1]
        neighbors.append(A_ppi.indices[start:end].astype(int))

    return neighbors


def crossover(
    parent1: Individual,
    parent2: Individual,
    rng: Optional[np.random.Generator] = None,
    pc: float = 0.8,
) -> Individual:
    """
    Uniform crossover for edge-based chromosome.

    MATLAB equivalent:
        Crossover(Parent1, Parent2, N, Child)

    pc = 0.8 means about 80% genes from parent1 and 20% from parent2.
    """

    if rng is None:
        rng = np.random.default_rng()

    if parent1.chromosome.shape != parent2.chromosome.shape:
        raise ValueError("Parent chromosomes have different lengths.")

    num_edges = parent1.chromosome.shape[0]

    mask = rng.random(num_edges) <= pc

    child_chromosome = parent1.chromosome.copy()
    child_chromosome[~mask] = parent2.chromosome[~mask]

    return Individual(chromosome=child_chromosome)


def _membership_map(
    complexes: List[set[int]],
    n: int,
) -> List[set[int]]:
    protein_complex_map: List[set[int]] = [set() for _ in range(n)]

    for c_idx, members in enumerate(complexes):
        clean_members = {int(u) for u in members if 0 <= int(u) < n}
        complexes[c_idx] = clean_members

        for u in clean_members:
            protein_complex_map[u].add(c_idx)

    return protein_complex_map


def _go_mean(A_go: sparse.csr_matrix, u: int, members: set[int]) -> float:
    if not members:
        return 0.0

    idx = np.array(sorted(members), dtype=int)
    val = float(A_go[u, idx].mean())

    if np.isnan(val) or np.isinf(val):
        return 0.0

    return float(np.clip(val, 0.0, 1.0))


def mutation_ctgm_overlapping(
    child: Individual,
    A_ppi,
    A_go,
    neighbors: Optional[List[np.ndarray]] = None,
    pm: float = 0.2,
    params: Optional[MutationParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> Individual:
    """
    Heuristic-guided overlapping mutation using:
        - PPI topology
        - GO semantic similarity

    Operates on child.complexes, not child.chromosome.

    MATLAB equivalent:
        Mutation_CTGM_Overlapping
    """

    if rng is None:
        rng = np.random.default_rng()

    if params is None:
        params = MutationParams()

    A_ppi = prepare_sparse(A_ppi)
    A_go = prepare_sparse(A_go)

    n = A_ppi.shape[0]

    if neighbors is None:
        neighbors = build_neighbors_from_adjacency(A_ppi)

    if child.complexes is None:
        raise ValueError("child must contain decoded complexes before mutation.")

    complexes = [set(c) for c in child.complexes]
    k = len(complexes)

    if k == 0:
        return child

    protein_complex_map = _membership_map(complexes, n)

    for u in range(n):
        neigh = neighbors[u]

        if neigh.size == 0 or rng.random() > pm:
            continue

        curr_cs = set(protein_complex_map[u])

        if not curr_cs:
            continue

        # ======================================================
        # A) Boundary-aware add/move
        # ======================================================
        move_count = 0

        for c in list(curr_cs):
            if move_count >= params.max_moves_per_node:
                break

            if c < 0 or c >= k:
                continue

            members = complexes[c]

            kin = sum(1 for v in neigh if int(v) in members)
            kout = len(neigh) - kin

            if kin <= kout:
                candidate_complexes: set[int] = set()

                for v in neigh:
                    candidate_complexes.update(protein_complex_map[int(v)])

                candidate_complexes.discard(c)
                candidate_complexes = {cc for cc in candidate_complexes if 0 <= cc < k}

                best_c = None
                best_score = -float("inf")

                for cc in candidate_complexes:
                    if u in complexes[cc]:
                        continue

                    members2 = complexes[cc]

                    kin2 = sum(1 for v in neigh if int(v) in members2)
                    kout2 = len(neigh) - kin2

                    tri_gain = 0

                    for v in neigh:
                        v = int(v)

                        if v not in members2:
                            continue

                        tri_gain += sum(1 for x in neighbors[v] if int(x) in members2)

                    go_gain = _go_mean(A_go, u, members2)

                    score = (
                        float(kin2 - kout2)
                        + 0.1 * float(tri_gain)
                        + params.go_score_weight * go_gain
                    )

                    if score > best_score:
                        best_score = score
                        best_c = cc

                if best_c is not None:
                    complexes[best_c].add(u)
                    protein_complex_map[u].add(best_c)
                    move_count += 1

        # ======================================================
        # B) Triangle-closure add with GO support
        # ======================================================
        if rng.random() <= params.p_add_triangle:
            adds = 0

            candidate_complexes: set[int] = set()

            for v in neigh:
                candidate_complexes.update(protein_complex_map[int(v)])

            candidate_complexes = {cc for cc in candidate_complexes if 0 <= cc < k}

            for cc in candidate_complexes:
                if adds >= params.max_complex_adds_per_node:
                    break

                if u in complexes[cc]:
                    continue

                members_cc = complexes[cc]

                nin = 0

                for v in neigh:
                    if int(v) in members_cc:
                        nin += 1

                        if nin >= 2:
                            break

                go_gain = _go_mean(A_go, u, members_cc)

                if nin >= 2 and go_gain >= params.go_add_threshold:
                    complexes[cc].add(u)
                    protein_complex_map[u].add(cc)
                    adds += 1

        # ======================================================
        # C) Prune weak boundary nodes
        # ======================================================
        if rng.random() <= params.p_prune:
            curr_cs2 = list(protein_complex_map[u])

            for c in curr_cs2:
                if c < 0 or c >= k:
                    continue

                members = complexes[c]

                if len(members) <= params.min_size:
                    continue

                kin = 0
                kout = 0
                tri_support = 0

                for v in neigh:
                    v = int(v)

                    if v in members:
                        kin += 1
                        tri_support += sum(1 for x in neighbors[v] if int(x) in members)
                    else:
                        kout += 1

                members_wo_u = set(members)
                members_wo_u.discard(u)

                go_support = _go_mean(A_go, u, members_wo_u)

                if kin <= kout and tri_support <= 1 and go_support < params.go_prune_threshold:
                    complexes[c].discard(u)
                    protein_complex_map[u].discard(c)

    # Final clean
    child.complexes = [
        {int(u) for u in c if 0 <= int(u) < n}
        for c in complexes
    ]

    return child


def local_member_affinity(
    members: set[int],
    A_ppi,
    A_go,
    neighbors: List[np.ndarray],
    params: RepairParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Equivalent to MATLAB LocalMemberAffinity.
    """

    A_go = prepare_sparse(A_go)

    members = {int(u) for u in members}
    member_list = sorted(members)
    member_set = set(member_list)

    m = len(member_list)

    affinity = np.zeros(m, dtype=float)
    topo_score = np.zeros(m, dtype=float)
    tri_score = np.zeros(m, dtype=float)
    go_score = np.zeros(m, dtype=float)

    complex_size = len(member_set)

    for i, u in enumerate(member_list):
        neigh = neighbors[u]

        if neigh.size == 0:
            continue

        internal_neigh = [int(v) for v in neigh if int(v) in member_set]

        kin = len(internal_neigh)
        kout = len(neigh) - kin

        topo_score[i] = kin / (kin + kout + 1e-6)

        tri_support = 0

        for v in internal_neigh:
            tri_support += sum(1 for x in neighbors[v] if int(x) in member_set)

        tri_score[i] = tri_support / (max(1, kin) * max(1, complex_size))
        tri_score[i] = min(tri_score[i], 1.0)

        others = set(member_set)
        others.discard(u)

        go_score[i] = _go_mean(A_go, u, others)

        affinity[i] = (
            params.alpha_topo * topo_score[i]
            + params.beta_tri * tri_score[i]
            + params.gamma_go * go_score[i]
        )

        affinity[i] = float(np.clip(affinity[i], 0.0, 1.0))

    return affinity, topo_score, tri_score, go_score


def local_candidate_affinity(
    u: int,
    complex_members: set[int],
    A_ppi,
    A_go,
    neighbors: List[np.ndarray],
    params: RepairParams,
) -> tuple[float, float, int]:
    """
    Equivalent to MATLAB LocalCandidateAffinity.
    """

    A_go = prepare_sparse(A_go)

    member_set = {int(x) for x in complex_members}
    neigh = neighbors[u]

    if neigh.size == 0:
        return 0.0, 0.0, 0

    internal_neigh = [int(v) for v in neigh if int(v) in member_set]

    nin = len(internal_neigh)
    nout = len(neigh) - nin

    topo_score = nin / (nin + nout + 1e-6)

    tri_support = 0

    for v in internal_neigh:
        tri_support += sum(1 for x in neighbors[v] if int(x) in member_set)

    tri_score = tri_support / (max(1, nin) * max(1, len(member_set)))
    tri_score = min(tri_score, 1.0)

    go_score = _go_mean(A_go, u, member_set)

    affinity = (
        params.alpha_topo * topo_score
        + params.beta_tri * tri_score
        + params.gamma_go * go_score
    )

    affinity = float(np.clip(affinity, 0.0, 1.0))

    return affinity, go_score, nin


def core_boundary_repair(
    child: Individual,
    A_ppi,
    A_go,
    neighbors: Optional[List[np.ndarray]] = None,
    params: Optional[RepairParams] = None,
) -> Individual:
    """
    Core-boundary repair for overlapping complexes.

    MATLAB equivalent:
        CoreBoundaryRepair
    """

    if params is None:
        params = RepairParams()

    A_ppi = prepare_sparse(A_ppi)
    A_go = prepare_sparse(A_go)

    n = A_ppi.shape[0]

    if neighbors is None:
        neighbors = build_neighbors_from_adjacency(A_ppi)

    if child.complexes is None:
        raise ValueError("child must contain decoded complexes before repair.")

    repaired_complexes: List[set[int]] = []

    for members in child.complexes:
        members = {int(u) for u in members if 0 <= int(u) < n}

        if len(members) < params.min_size:
            repaired_complexes.append(members)
            continue

        member_list = sorted(members)

        affinity, topo_score, tri_score, go_score = local_member_affinity(
            members=members,
            A_ppi=A_ppi,
            A_go=A_go,
            neighbors=neighbors,
            params=params,
        )

        core_nodes = {
            member_list[i]
            for i, score in enumerate(affinity)
            if score >= params.core_threshold
        }

        if len(core_nodes) < params.min_size:
            order = np.argsort(-affinity)
            core_nodes = {
                member_list[i]
                for i in order[: min(params.min_size, len(member_list))]
            }

        boundary_nodes = [u for u in member_list if u not in core_nodes]

        kept_boundary: set[int] = set()

        index_of = {u: idx for idx, u in enumerate(member_list)}

        for u in boundary_nodes:
            idx = index_of[u]

            if affinity[idx] >= params.remove_threshold:
                kept_boundary.add(u)

        repaired_members = set(core_nodes)
        repaired_members.update(kept_boundary)

        # Preserve minimum size using best original members
        if len(repaired_members) < params.min_size:
            order = np.argsort(-affinity)

            for idx in order:
                u = member_list[int(idx)]

                repaired_members.add(u)

                if len(repaired_members) >= params.min_size:
                    break

        # Boundary expansion
        if len(core_nodes) >= params.min_size and params.max_boundary_adds > 0:
            candidate_pool: set[int] = set()

            for u in core_nodes:
                candidate_pool.update(int(v) for v in neighbors[u])

            candidate_pool.difference_update(repaired_members)
            candidate_pool = {u for u in candidate_pool if 0 <= u < n}

            if candidate_pool:
                nin_quick = []

                for u in candidate_pool:
                    nin = sum(1 for v in neighbors[u] if int(v) in repaired_members)
                    nin_quick.append((u, nin))

                nin_quick = [
                    (u, nin)
                    for u, nin in nin_quick
                    if nin >= params.min_internal_neighbors
                ]

                if nin_quick:
                    nin_quick.sort(key=lambda x: x[1], reverse=True)
                    limited_candidates = [
                        u for u, _ in nin_quick[: params.max_candidate_pool]
                    ]

                    scored_candidates = []

                    for u in limited_candidates:
                        score_u, go_u, nin_u = local_candidate_affinity(
                            u=u,
                            complex_members=repaired_members,
                            A_ppi=A_ppi,
                            A_go=A_go,
                            neighbors=neighbors,
                            params=params,
                        )

                        if (
                            score_u >= params.add_threshold
                            and go_u >= params.go_add_threshold
                            and nin_u >= params.min_internal_neighbors
                        ):
                            scored_candidates.append((u, score_u))

                    if scored_candidates:
                        scored_candidates.sort(key=lambda x: x[1], reverse=True)
                        add_nodes = [
                            u for u, _ in scored_candidates[: params.max_boundary_adds]
                        ]

                        repaired_members.update(add_nodes)

        repaired_complexes.append({int(u) for u in repaired_members if 0 <= int(u) < n})

    child.complexes = repaired_complexes

    return child