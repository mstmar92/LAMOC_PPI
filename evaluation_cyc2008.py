from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
from scipy import sparse


Complex = Set[int]


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    mmr: float
    num_predicted: int
    avg_predicted_size: float


def read_cyc2008_complexes_txt(path: str) -> List[set[str]]:
    """
    Read CYC2008 complexes from TXT file.

    Each non-empty, non-comment line is treated as one complex.
    Protein names are normalized to uppercase.
    """

    complexes: List[set[str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            tokens = [x.strip().upper() for x in line.split() if x.strip()]

            if tokens:
                complexes.append(set(tokens))

    return complexes


def convert_benchmark_to_indices(
    benchmark_complexes_names: List[set[str]],
    protein_labels: Sequence[str],
) -> List[Complex]:
    """
    Convert benchmark protein names to integer indices.

    protein_labels must correspond to the internal node ordering.
    """

    protein_map: Dict[str, int] = {
        str(label).strip().upper(): i
        for i, label in enumerate(protein_labels)
        if str(label).strip()
    }

    benchmark_idx: List[Complex] = []

    for comp in benchmark_complexes_names:
        idx = {
            protein_map[name.strip().upper()]
            for name in comp
            if name.strip().upper() in protein_map
        }
        benchmark_idx.append(idx)

    return benchmark_idx


def filter_complexes_by_size(
    complexes: Iterable[Iterable[int]],
    min_size: int = 3,
) -> List[Complex]:
    """
    Keep only complexes with size >= min_size.
    """

    out: List[Complex] = []

    for comp in complexes:
        c = {int(x) for x in comp if int(x) >= 0}

        if len(c) >= min_size:
            out.append(c)

    return out


def complexes_to_sparse_matrix(
    complexes: List[Complex],
    num_nodes: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Convert complexes to sparse membership matrix.

    Rows = complexes
    Columns = proteins/nodes
    M[i, j] = 1 if node j belongs to complex i
    """

    rows = []
    cols = []
    sizes = np.zeros(len(complexes), dtype=float)

    for i, comp in enumerate(complexes):
        clean = sorted({int(x) for x in comp if 0 <= int(x) < num_nodes})
        sizes[i] = len(clean)

        for u in clean:
            rows.append(i)
            cols.append(u)

    vals = np.ones(len(rows), dtype=bool)

    mat = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(complexes), num_nodes),
        dtype=bool,
    )

    return mat, sizes


def greedy_mmr(os_matrix: np.ndarray, num_benchmark: int) -> float:
    """
    Greedy maximum matching ratio.

    MATLAB equivalent:
        GreedyMMR(OS, B)
    """

    if os_matrix.size == 0 or num_benchmark == 0:
        return 0.0

    os_work = os_matrix.copy()

    p, b = os_work.shape

    used_p = np.zeros(p, dtype=bool)
    used_b = np.zeros(b, dtype=bool)

    scores: List[float] = []

    while True:
        temp = os_work.copy()
        temp[used_p, :] = -np.inf
        temp[:, used_b] = -np.inf

        max_val = np.max(temp)

        if not np.isfinite(max_val) or max_val <= 0:
            break

        i, j = np.unravel_index(np.argmax(temp), temp.shape)

        used_p[i] = True
        used_b[j] = True

        scores.append(float(max_val))

    return float(np.sum(scores) / num_benchmark)


def evaluate_complex_set_cyc2008_fast(
    predicted_complexes: List[Complex],
    benchmark_matrix: sparse.csr_matrix,
    benchmark_sizes: np.ndarray,
    num_nodes: int,
    overlap_threshold: float = 0.20,
    min_size: int = 3,
) -> EvaluationMetrics:
    """
    Evaluate predicted complexes against benchmark complexes.

    Overlap score:
        OS(P,B) = |P ∩ B|^2 / (|P| * |B|)
    """

    predicted_complexes = filter_complexes_by_size(predicted_complexes, min_size=min_size)

    p = len(predicted_complexes)
    b = benchmark_matrix.shape[0]

    if p == 0 or b == 0:
        return EvaluationMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mmr=0.0,
            num_predicted=p,
            avg_predicted_size=0.0,
        )

    pred_matrix, pred_sizes = complexes_to_sparse_matrix(predicted_complexes, num_nodes)

    inter_counts = pred_matrix.astype(float) @ benchmark_matrix.astype(float).T
    inter_counts = inter_counts.toarray()

    denom = pred_sizes[:, None] * benchmark_sizes[None, :]

    os_matrix = np.zeros((p, b), dtype=float)
    valid = denom > 0

    os_matrix[valid] = (inter_counts[valid] ** 2) / denom[valid]

    pred_matched = np.max(os_matrix, axis=1) >= overlap_threshold
    precision = float(np.sum(pred_matched) / p)

    bench_matched = np.max(os_matrix, axis=0) >= overlap_threshold
    recall = float(np.sum(bench_matched) / b)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    mmr = greedy_mmr(os_matrix, b)

    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mmr=mmr,
        num_predicted=p,
        avg_predicted_size=float(np.mean(pred_sizes)) if len(pred_sizes) else 0.0,
    )


def choose_best_pareto_by_f1(
    pareto_entries,
    benchmark_matrix: sparse.csr_matrix,
    benchmark_sizes: np.ndarray,
    num_nodes: int,
    overlap_threshold: float = 0.20,
    min_size: int = 3,
):
    """
    Choose best Pareto entry using benchmark F1.

    This is for offline evaluation only.
    """

    best_entry = None
    best_metrics = None
    best_index = -1
    best_f1 = -float("inf")

    for i, entry in enumerate(pareto_entries):
        metrics = evaluate_complex_set_cyc2008_fast(
            predicted_complexes=entry.complexes,
            benchmark_matrix=benchmark_matrix,
            benchmark_sizes=benchmark_sizes,
            num_nodes=num_nodes,
            overlap_threshold=overlap_threshold,
            min_size=min_size,
        )

        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_entry = entry
            best_metrics = metrics
            best_index = i

    return best_index, best_entry, best_metrics