from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import sparse
from tqdm import tqdm

from decoding import decode_population
from encoding import Individual, copy_individual
from moead import (
    ConvergenceLog,
    ParetoEntry,
    init_weights,
    update_convergence_log,
    update_near_pareto_set,
    update_problem,
    update_reference,
)
from objectives import compute_population_fitness, prepare_adjacency_matrix
from operators import (
    MutationParams,
    RepairParams,
    build_neighbors_from_adjacency,
    core_boundary_repair,
    crossover,
    mutation_ctgm_overlapping,
)


@dataclass
class AlgorithmConfig:
    population_size: int = 100
    generations: int = 100

    neighbour: int = 5
    objective_dimension: int = 2

    pm: float = 0.2
    crossover_probability: float = 0.8

    use_go_in_fitness: bool = True
    use_heuristic_mutation: bool = True
    use_repair: bool = True

    random_seed: int = 42

    mutation_params: MutationParams = field(default_factory=MutationParams)
    repair_params: RepairParams = field(default_factory=RepairParams)

@dataclass
class AlgorithmResults:
    final_near_pareto_optimal_set: List[ParetoEntry]
    final_population: List[Individual]

    final_ideal_point: np.ndarray
    final_indiv_point: List[Optional[Individual]]

    convergence: ConvergenceLog
    meta: Dict[str, Any]


def _select_parents_from_neighbourhood(
    population: List[Individual],
    subproblems,
    neighbour: int,
    rng: np.random.Generator,
) -> tuple[List[Individual], List[Individual]]:
    """
    MOEA/D neighbourhood selection.

    MATLAB equivalent:
        rand1 = randi(Params.Neighbour)
        rand2 = randi(Params.Neighbour)
        P1 = SubProblem(i).Neighbour(rand1)
        P2 = SubProblem(i).Neighbour(rand2)
    """

    pop_size = len(subproblems)

    parent1: List[Individual] = []
    parent2: List[Individual] = []

    for i in range(pop_size):
        neigh = subproblems[i].neighbours

        if len(neigh) == 0:
            raise ValueError(f"Subproblem {i} has empty neighbourhood.")

        r1 = int(rng.integers(0, neighbour))
        r2 = int(rng.integers(0, neighbour))

        p1_idx = int(neigh[r1])
        p2_idx = int(neigh[r2])

        parent1.append(population[p1_idx])
        parent2.append(population[p2_idx])

    return parent1, parent2


def run_moead_overlapping(
    A_ppi,
    A_go,
    edge_pairs: np.ndarray,
    initial_population: List[Individual],
    config: AlgorithmConfig,
) -> AlgorithmResults:
    """
    Python equivalent of MOEAD_Overlapping_TNSE.

    Assumptions:
        - edge_pairs are 0-based node indices.
        - initial_population already has chromosomes.
        - this function will decode and compute fitness if needed.
    """

    rng = np.random.default_rng(config.random_seed)

    A_ppi = prepare_adjacency_matrix(A_ppi)

    n = A_ppi.shape[0]

    if config.use_go_in_fitness or config.use_heuristic_mutation or config.use_repair:
        if A_go is None:
            A_go_prepared = sparse.csr_matrix((n, n), dtype=float)
        else:
            A_go_prepared = prepare_adjacency_matrix(A_go)
    else:
        A_go_prepared = sparse.csr_matrix((n, n), dtype=float)

    if A_go_prepared.shape != A_ppi.shape:
        raise ValueError("A_go size does not match A_ppi size.")

    if config.use_go_in_fitness:
        A_go_fitness = A_go_prepared
    else:
        A_go_fitness = sparse.csr_matrix((n, n), dtype=float)

    pop_size = config.population_size

    if len(initial_population) < pop_size:
        raise ValueError("initial_population is smaller than config.population_size.")

    population = [copy_individual(ind) for ind in initial_population[:pop_size]]

    # Make sure initial population is decoded and evaluated.
    population = decode_population(edge_pairs, population, pop_size)
    population = compute_population_fitness(
        population=population,
        A_ppi=A_ppi,
        A_go=A_go_fitness,
        population_size=pop_size,
    )

    neighbour = min(config.neighbour, pop_size)

    subproblems = init_weights(
        pop_size=pop_size,
        neighbour=neighbour,
        objective_dimension=config.objective_dimension,
    )

    # If objective_dimension == 3, init_weights changes the number of subproblems.
    # For this MATLAB-matching version, we enforce 2 objectives and pop_size.
    if len(subproblems) != pop_size:
        raise ValueError("Number of subproblems must equal population size for this runner.")

    ideal_point = np.ones(config.objective_dimension, dtype=float) * 1e7
    indiv_point: List[Optional[Individual]] = [None for _ in range(config.objective_dimension)]

    archive: List[ParetoEntry] = []
    convergence = ConvergenceLog()

    # Initial reference and archive update.
    ideal_point, indiv_point = update_reference(
        ideal_point=ideal_point,
        indiv_point=indiv_point,
        population=population,
        population_size=pop_size,
    )

    archive = update_near_pareto_set(
        archive=archive,
        population=population,
        pop_size=pop_size,
        objective_dimension=config.objective_dimension,
    )

    convergence = update_convergence_log(
        log=convergence,
        generation=1,
        population=population,
        archive=archive,
    )

    neighbors = build_neighbors_from_adjacency(A_ppi)

    generation_times: List[float] = []

    for generation in tqdm(
        range(2, config.generations + 1),
        desc="MOEA/D generations",
    ):
        # Selection
        parent1, parent2 = _select_parents_from_neighbourhood(
            population=population,
            subproblems=subproblems,
            neighbour=neighbour,
            rng=rng,
        )

        # Crossover
        children: List[Individual] = []

        for i in range(pop_size):
            child = crossover(
                parent1=parent1[i],
                parent2=parent2[i],
                rng=rng,
                pc=config.crossover_probability,
            )
            children.append(child)

        # Decode after crossover
        children = decode_population(edge_pairs, children, pop_size)

        # Heuristic mutation
        if config.use_heuristic_mutation:
            for i in range(pop_size):
                children[i] = mutation_ctgm_overlapping(
                    child=children[i],
                    A_ppi=A_ppi,
                    A_go=A_go_prepared,
                    neighbors=neighbors,
                    pm=config.pm,
                    params=config.mutation_params,
                    rng=rng,
                )

        # Core-boundary repair
        if config.use_repair:
            for i in range(pop_size):
                children[i] = core_boundary_repair(
                    child=children[i],
                    A_ppi=A_ppi,
                    A_go=A_go_prepared,
                    neighbors=neighbors,
                    params=config.repair_params,
                )

        # Fitness
        children = compute_population_fitness(
            population=children,
            A_ppi=A_ppi,
            A_go=A_go_fitness,
            population_size=pop_size,
        )

        # Update reference
        ideal_point, indiv_point = update_reference(
            ideal_point=ideal_point,
            indiv_point=indiv_point,
            population=children,
            population_size=pop_size,
        )

        # Update population
        population = update_problem(
            population=population,
            children=children,
            subproblems=subproblems,
            neighbour=neighbour,
            ideal_point=ideal_point,
            objective_dimension=config.objective_dimension,
        )

        # Update Pareto archive
        archive = update_near_pareto_set(
            archive=archive,
            population=children,
            pop_size=pop_size,
            objective_dimension=config.objective_dimension,
        )

        convergence = update_convergence_log(
            log=convergence,
            generation=generation,
            population=population,
            archive=archive,
        )

        if generation == 2 or generation % 10 == 0 or generation == config.generations:
            best_obj1 = convergence.best_obj1[-1]
            best_obj2 = convergence.best_obj2[-1]
            print(
                f"Generation {generation:03d} | "
                f"Pareto size={len(archive)} | "
                f"Best Obj=[{best_obj1:.4f}, {best_obj2:.4f}]"
            )

    meta = {
        "population_size": config.population_size,
        "generations": config.generations,
        "neighbour": neighbour,
        "objective_dimension": config.objective_dimension,
        "pm": config.pm,
        "crossover_probability": config.crossover_probability,
        "use_go_in_fitness": config.use_go_in_fitness,
        "use_heuristic_mutation": config.use_heuristic_mutation,
        "use_repair": config.use_repair,
        "random_seed": config.random_seed,
    }

    return AlgorithmResults(
        final_near_pareto_optimal_set=archive,
        final_population=population,
        final_ideal_point=ideal_point,
        final_indiv_point=indiv_point,
        convergence=convergence,
        meta=meta,
    )