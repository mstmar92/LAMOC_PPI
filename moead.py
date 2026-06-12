from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from encoding import Individual, copy_individual


EPS_WEIGHT = 1e-6


@dataclass
class SubProblem:
    weight: np.ndarray
    neighbours: np.ndarray
    optimal: float = float("inf")
    opt_point: Optional[np.ndarray] = None
    cur_point: Optional[np.ndarray] = None


@dataclass
class ParetoEntry:
    chromosome: np.ndarray
    complexes: List[set[int]]
    obj: np.ndarray

    obj1_conductance: float
    obj2_go_coherence_loss: float

    kv: int = 0
    avg_go: float = 0.0
    dominated: bool = False


@dataclass
class ConvergenceLog:
    best_obj1: List[float] = field(default_factory=list)
    best_obj2: List[float] = field(default_factory=list)

    mean_obj1: List[float] = field(default_factory=list)
    mean_obj2: List[float] = field(default_factory=list)

    pareto_size: List[int] = field(default_factory=list)
    mean_kv: List[float] = field(default_factory=list)
    mean_avg_go: List[float] = field(default_factory=list)


def init_weights(
    pop_size: int,
    neighbour: int,
    objective_dimension: int,
) -> List[SubProblem]:
    """
    MATLAB equivalent:
        InitWeights(PopSize, Neighbour, ObjectiveDimension)

    Current implementation supports 2 or 3 objectives, matching MATLAB.
    """

    if pop_size < 2:
        raise ValueError("pop_size must be at least 2.")

    neighbour = min(neighbour, pop_size)

    subproblems: List[SubProblem] = []

    if objective_dimension == 2:
        for i in range(pop_size):
            w1 = i / (pop_size - 1)
            w2 = 1.0 - w1

            subproblems.append(
                SubProblem(
                    weight=np.array([w1, w2], dtype=float),
                    neighbours=np.array([], dtype=int),
                )
            )

    elif objective_dimension == 3:
        for i in range(pop_size + 1):
            for j in range(pop_size - i + 1):
                k = pop_size - i - j
                weight = np.array([i, j, k], dtype=float) / pop_size

                subproblems.append(
                    SubProblem(
                        weight=weight,
                        neighbours=np.array([], dtype=int),
                    )
                )
    else:
        raise ValueError("Only objective_dimension = 2 or 3 is supported.")

    length = len(subproblems)
    neighbour = min(neighbour, length)

    weight_matrix = np.vstack([sp.weight for sp in subproblems])

    # squared Euclidean distance between weights
    diff = weight_matrix[:, None, :] - weight_matrix[None, :, :]
    dist_matrix = np.sum(diff * diff, axis=2)

    for i in range(length):
        order = np.argsort(dist_matrix[i])
        subproblems[i].neighbours = order[:neighbour].astype(int)

    return subproblems


def update_reference(
    ideal_point: np.ndarray,
    indiv_point: List[Optional[Individual]],
    population: List[Individual],
    population_size: Optional[int] = None,
) -> tuple[np.ndarray, List[Optional[Individual]]]:
    """
    MATLAB equivalent:
        UpdateReferenceOverlapping

    Updates ideal point under minimization.
    """

    if population_size is None:
        population_size = len(population)

    if population_size > len(population):
        raise ValueError("population_size is larger than actual population size.")

    objective_dimension = len(ideal_point)

    if len(indiv_point) < objective_dimension:
        indiv_point = [None for _ in range(objective_dimension)]

    for i in range(population_size):
        ind = population[i]

        if ind.obj is None or len(ind.obj) < objective_dimension:
            raise ValueError(f"population[{i}] does not contain a valid obj field.")

        obj = ind.obj

        if np.any(np.isnan(obj)) or np.any(np.isinf(obj)):
            continue

        for d in range(objective_dimension):
            if obj[d] < ideal_point[d]:
                ideal_point[d] = float(obj[d])
                indiv_point[d] = copy_individual(ind)

    return ideal_point, indiv_point


def scalar_function(
    individual: Individual,
    weight: np.ndarray,
    ideal_point: np.ndarray,
    objective_dimension: int,
) -> float:
    """
    Tchebycheff scalarization for minimization.

    MATLAB equivalent:
        ScalarFunction(Individual, Weight, IndivPoint, ObjectiveDimension)

    Difference:
        Python passes ideal_point directly instead of IndivPoint(d).Ideal.
    """

    if individual.obj is None or len(individual.obj) < objective_dimension:
        raise ValueError("individual does not contain a valid obj field.")

    if len(weight) < objective_dimension:
        raise ValueError("weight vector dimension is smaller than objective_dimension.")

    values = []

    for d in range(objective_dimension):
        f_d = float(individual.obj[d])
        z_d = float(ideal_point[d])

        if np.isnan(f_d) or np.isinf(f_d):
            return float("inf")

        w_d = max(float(weight[d]), EPS_WEIGHT)
        values.append(w_d * abs(f_d - z_d))

    return float(max(values))


def update_problem(
    population: List[Individual],
    children: List[Individual],
    subproblems: List[SubProblem],
    neighbour: int,
    ideal_point: np.ndarray,
    objective_dimension: int,
) -> List[Individual]:
    """
    MATLAB equivalent:
        UpdateProblemOverlapping

    Uses Tchebycheff scalarization to update MOEA/D neighbourhoods.
    """

    if objective_dimension != 2:
        raise ValueError("update_problem currently supports 2 objectives only.")

    pop_size = len(subproblems)

    for problem_counter in range(pop_size):
        child = children[problem_counter]

        for neighbour_counter in range(neighbour):
            neighbour_index = int(subproblems[problem_counter].neighbours[neighbour_counter])

            current = population[neighbour_index]
            weight = subproblems[neighbour_index].weight

            f1 = scalar_function(
                individual=current,
                weight=weight,
                ideal_point=ideal_point,
                objective_dimension=objective_dimension,
            )

            f2 = scalar_function(
                individual=child,
                weight=weight,
                ideal_point=ideal_point,
                objective_dimension=objective_dimension,
            )

            if f2 < f1:
                population[neighbour_index] = copy_individual(child)

    return population


def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """
    Minimization dominance.

    a dominates b if:
        a <= b for all objectives
        and a < b for at least one objective
    """

    obj_a = np.asarray(obj_a, dtype=float)
    obj_b = np.asarray(obj_b, dtype=float)

    if np.any(np.isnan(obj_a)) or np.any(np.isinf(obj_a)):
        return False

    if np.any(np.isnan(obj_b)) or np.any(np.isinf(obj_b)):
        return False

    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def _canonical_complex_structure(complexes: List[set[int]]) -> tuple[tuple[int, ...], ...]:
    """
    Canonical form for duplicate checking.

    MATLAB uses isequal(CmplxID1, CmplxID2), which is order-sensitive.
    Here we use a stricter biological duplicate check:
        sort nodes within each complex
        sort complexes
    """

    cleaned = [tuple(sorted(int(x) for x in c)) for c in complexes if len(c) > 0]
    cleaned.sort()
    return tuple(cleaned)


def _pareto_entry_from_individual(ind: Individual) -> ParetoEntry:
    if ind.obj is None:
        raise ValueError("Cannot create ParetoEntry from individual without obj.")

    return ParetoEntry(
        chromosome=ind.chromosome.copy(),
        complexes=[set(c) for c in ind.complexes],
        obj=ind.obj.copy(),
        obj1_conductance=float(ind.obj1_conductance),
        obj2_go_coherence_loss=float(ind.obj2_go_coherence_loss),
        kv=int(ind.kv),
        avg_go=float(ind.avg_go),
        dominated=False,
    )


def update_near_pareto_set(
    archive: List[ParetoEntry],
    population: List[Individual],
    pop_size: Optional[int] = None,
    objective_dimension: int = 2,
) -> List[ParetoEntry]:
    """
    MATLAB equivalent:
        UpdateNearParetoOptimalSetOverlapping
    """

    if objective_dimension != 2:
        raise ValueError("update_near_pareto_set currently supports 2 objectives only.")

    if pop_size is None:
        pop_size = len(population)

    if pop_size > len(population):
        raise ValueError("pop_size is larger than actual population size.")

    for i in range(pop_size):
        ind = population[i]

        if ind.obj is None or len(ind.obj) < objective_dimension:
            raise ValueError(f"population[{i}] does not contain valid obj.")

        if np.any(np.isnan(ind.obj)) or np.any(np.isinf(ind.obj)):
            continue

        new_struct = _canonical_complex_structure(ind.complexes)
        should_add = True

        for entry in archive:
            old_struct = _canonical_complex_structure(entry.complexes)

            # Duplicate community structure
            if old_struct == new_struct:
                entry.dominated = False
                should_add = False
                break

            # Existing archive entry dominates new individual
            if dominates(entry.obj, ind.obj):
                entry.dominated = False
                should_add = False
                break

            # New individual dominates archive entry
            if dominates(ind.obj, entry.obj):
                entry.dominated = True
            else:
                entry.dominated = False

        if should_add:
            archive = erase_dominated_from_archive(archive)
            archive.append(_pareto_entry_from_individual(ind))

    return archive


def erase_dominated_from_archive(archive: List[ParetoEntry]) -> List[ParetoEntry]:
    """
    MATLAB equivalent:
        EraseFromNearParetoOptimalSetOverlapping
    """

    archive = [entry for entry in archive if not entry.dominated]

    for entry in archive:
        entry.dominated = False

    return archive


def update_convergence_log(
    log: ConvergenceLog,
    generation: int,
    population: List[Individual],
    archive: List[ParetoEntry],
) -> ConvergenceLog:
    """
    Python equivalent of UpdateConvergenceLog helper.
    """

    obj_values = [ind.obj for ind in population if ind.obj is not None]

    if obj_values:
        obj_mat = np.vstack(obj_values)

        log.best_obj1.append(float(np.min(obj_mat[:, 0])))
        log.best_obj2.append(float(np.min(obj_mat[:, 1])))

        log.mean_obj1.append(float(np.mean(obj_mat[:, 0])))
        log.mean_obj2.append(float(np.mean(obj_mat[:, 1])))
    else:
        log.best_obj1.append(float("nan"))
        log.best_obj2.append(float("nan"))
        log.mean_obj1.append(float("nan"))
        log.mean_obj2.append(float("nan"))

    log.pareto_size.append(len(archive))

    kv_vals = [ind.kv for ind in population]
    avg_go_vals = [ind.avg_go for ind in population]

    log.mean_kv.append(float(np.mean(kv_vals)) if kv_vals else float("nan"))
    log.mean_avg_go.append(float(np.mean(avg_go_vals)) if avg_go_vals else float("nan"))

    return log