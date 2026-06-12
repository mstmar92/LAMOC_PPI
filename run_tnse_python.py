from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from algorithm import AlgorithmConfig, run_moead_overlapping
from encoding import create_population_overlapping
from evaluation_cyc2008 import (
    choose_best_pareto_by_f1,
    complexes_to_sparse_matrix,
    convert_benchmark_to_indices,
    filter_complexes_by_size,
    read_cyc2008_complexes_txt,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tnse_python_parallel"

NETWORK_FILES = {
    "Gavin": DATA_RAW / "Gavin_BioGRID_21817_clean_edges.tsv",
    "Collins": DATA_RAW / "Collins_clean_edges.tsv",
    "Krogan": DATA_RAW / "Krogan_BioGRID_21816_clean_edges.tsv",
}

CYC2008_FILE = DATA_RAW / "CYC2008_complexes.txt"


# ============================================================
# Experiment modes
# ============================================================

METHODS = {
    "TOPO": {
        "use_go_in_fitness": False,
        "use_heuristic_mutation": False,
        "use_repair": False,
    },
    "PPI_GO": {
        "use_go_in_fitness": True,
        "use_heuristic_mutation": False,
        "use_repair": False,
    },
    "FULL": {
        "use_go_in_fitness": True,
        "use_heuristic_mutation": True,
        "use_repair": True,
    },
}


# ============================================================
# Utility functions
# ============================================================

def normalize_label(x: str) -> str:
    return str(x).strip().upper()


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    a = normalize_label(a)
    b = normalize_label(b)
    return (a, b) if a <= b else (b, a)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_edge_list_symbols(path: Path) -> list[tuple[str, str]]:
    """
    Load clean edge list TSV.

    Expected format:
        proteinA proteinB

    Header is tolerated; duplicated undirected edges are removed.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing edge file: {path}")

    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")

    if df.shape[1] < 2:
        df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")

    edges: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        u = normalize_label(row.iloc[0])
        v = normalize_label(row.iloc[1])

        if not u or not v:
            continue

        # Skip possible header rows.
        if u in {"PROTEINA", "PROTEIN_A", "SOURCE", "FROM", "NODE1"}:
            continue
        if v in {"PROTEINB", "PROTEIN_B", "TARGET", "TO", "NODE2"}:
            continue

        if u == v:
            continue

        a, b = sorted((u, v))
        edges.append((a, b))

    seen = set()
    clean_edges: list[tuple[str, str]] = []

    for u, v in edges:
        key = (u, v)

        if key not in seen:
            seen.add(key)
            clean_edges.append(key)

    if not clean_edges:
        raise ValueError(f"No valid edges loaded from {path}")

    return clean_edges


def build_indexed_network(
    symbol_edges: list[tuple[str, str]],
) -> tuple[np.ndarray, list[str], Dict[str, int], sparse.csr_matrix]:
    """
    Convert symbol edge list to:
        edge_pairs: E x 2 0-based integer array
        protein_labels: index -> symbol
        protein_to_idx: symbol -> index
        A_ppi: sparse adjacency
    """

    proteins = sorted(set([u for u, _ in symbol_edges] + [v for _, v in symbol_edges]))
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    edge_pairs = np.array(
        [(protein_to_idx[u], protein_to_idx[v]) for u, v in symbol_edges],
        dtype=int,
    )

    n = len(proteins)

    rows = edge_pairs[:, 0]
    cols = edge_pairs[:, 1]

    A = sparse.csr_matrix(
        (np.ones(len(edge_pairs), dtype=float), (rows, cols)),
        shape=(n, n),
    )

    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()

    return edge_pairs, proteins, protein_to_idx, A


def build_go_matrix_from_domain_caches(
    dataset_name: str,
    symbol_edges: list[tuple[str, str]],
    protein_to_idx: Dict[str, int],
) -> sparse.csr_matrix:
    """
    Build one GO matrix for the MATLAB-matching 2-objective version.

    We currently have BP/CC/MF caches separately, so we combine them as:
        A_go = mean(BP, CC, MF)

    Later, for the new method, BP/CC/MF can remain separate layers.
    """

    bp_path = DATA_PROCESSED / f"{dataset_name}_BP_wang_cache.pkl"
    cc_path = DATA_PROCESSED / f"{dataset_name}_CC_wang_cache.pkl"
    mf_path = DATA_PROCESSED / f"{dataset_name}_MF_wang_cache.pkl"

    if not bp_path.exists() or not cc_path.exists() or not mf_path.exists():
        raise FileNotFoundError(
            f"Missing GO domain cache for {dataset_name}. Expected:\n"
            f"{bp_path}\n{cc_path}\n{mf_path}"
        )

    bp_cache = load_pickle(bp_path)
    cc_cache = load_pickle(cc_path)
    mf_cache = load_pickle(mf_path)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for u, v in symbol_edges:
        pair = canonical_pair(u, v)

        bp = float(bp_cache.get(pair, 0.0))
        cc = float(cc_cache.get(pair, 0.0))
        mf = float(mf_cache.get(pair, 0.0))

        sim = (bp + cc + mf) / 3.0

        if sim <= 0:
            continue

        i = protein_to_idx[u]
        j = protein_to_idx[v]

        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([sim, sim])

    n = len(protein_to_idx)

    A_go = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n, n),
        dtype=float,
    )

    A_go.setdiag(0)
    A_go.eliminate_zeros()

    return A_go


def save_predicted_complexes(
    complexes: List[set[int]],
    protein_labels: list[str],
    out_path: Path,
    min_size: int = 3,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    complexes = filter_complexes_by_size(complexes, min_size=min_size)

    with out_path.open("w", encoding="utf-8") as f:
        for comp in complexes:
            names = [protein_labels[i] for i in sorted(comp)]
            f.write(" ".join(names) + "\n")


# ============================================================
# Worker task
# ============================================================

def run_single_task(task: dict) -> dict:
    """
    One independent Dataset × Method × Run task.

    Important:
        This function must stay at top level for Windows multiprocessing.
    """

    dataset_name = task["dataset_name"]
    method_name = task["method_name"]
    flags = task["flags"]
    run_id = task["run_id"]

    edge_pairs = task["edge_pairs"]
    symbol_edges = task["symbol_edges"]
    protein_labels = task["protein_labels"]
    protein_to_idx = task["protein_to_idx"]
    benchmark_names = task["benchmark_names"]

    population_size = task["population_size"]
    generations = task["generations"]
    overlap_threshold = task["overlap_threshold"]
    min_complex_size = task["min_complex_size"]

    n = len(protein_labels)

    # Rebuild PPI matrix inside the worker.
    rows = edge_pairs[:, 0]
    cols = edge_pairs[:, 1]

    A_ppi = sparse.csr_matrix(
        (np.ones(len(edge_pairs), dtype=float), (rows, cols)),
        shape=(n, n),
    )

    A_ppi = A_ppi + A_ppi.T
    A_ppi.setdiag(0)
    A_ppi.eliminate_zeros()

    # Rebuild GO matrix inside the worker.
    A_go = build_go_matrix_from_domain_caches(
        dataset_name=dataset_name,
        symbol_edges=symbol_edges,
        protein_to_idx=protein_to_idx,
    )

    # Benchmark mapping inside the worker.
    benchmark_idx = convert_benchmark_to_indices(
        benchmark_complexes_names=benchmark_names,
        protein_labels=protein_labels,
    )

    benchmark_idx = filter_complexes_by_size(
        benchmark_idx,
        min_size=min_complex_size,
    )

    Bmat, Bsizes = complexes_to_sparse_matrix(benchmark_idx, n)

    # Seeds follow the MATLAB logic.
    init_seed = 1000 + run_id
    run_seed = 100000 + run_id

    rng_init = np.random.default_rng(init_seed)

    initial_population = create_population_overlapping(
        edge_pairs=edge_pairs,
        population_size=population_size,
        rng=rng_init,
    )

    config = AlgorithmConfig(
        population_size=population_size,
        generations=generations,
        neighbour=min(5, population_size),
        objective_dimension=2,
        pm=0.2,
        crossover_probability=0.8,
        use_go_in_fitness=flags["use_go_in_fitness"],
        use_heuristic_mutation=flags["use_heuristic_mutation"],
        use_repair=flags["use_repair"],
        random_seed=run_seed,
    )

    results = run_moead_overlapping(
        A_ppi=A_ppi,
        A_go=A_go,
        edge_pairs=edge_pairs,
        initial_population=initial_population,
        config=config,
    )

    best_idx, best_entry, best_metrics = choose_best_pareto_by_f1(
        pareto_entries=results.final_near_pareto_optimal_set,
        benchmark_matrix=Bmat,
        benchmark_sizes=Bsizes,
        num_nodes=n,
        overlap_threshold=overlap_threshold,
        min_size=min_complex_size,
    )

    pred_dir = RESULTS_DIR / dataset_name / method_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    if best_entry is None:
        row = {
            "Dataset": dataset_name,
            "Method": method_name,
            "Run": run_id,
            "BestParetoIndex": np.nan,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "MMR": 0.0,
            "NumPredicted": 0,
            "AvgPredictedSize": 0.0,
            "Obj1": np.nan,
            "Obj2": np.nan,
            "Kv": np.nan,
            "avgGO": np.nan,
        }
    else:
        pred_path = pred_dir / f"run_{run_id:02d}_best_predicted_complexes.txt"

        save_predicted_complexes(
            complexes=best_entry.complexes,
            protein_labels=protein_labels,
            out_path=pred_path,
            min_size=min_complex_size,
        )

        row = {
            "Dataset": dataset_name,
            "Method": method_name,
            "Run": run_id,
            "BestParetoIndex": best_idx,
            "Precision": best_metrics.precision,
            "Recall": best_metrics.recall,
            "F1": best_metrics.f1,
            "MMR": best_metrics.mmr,
            "NumPredicted": best_metrics.num_predicted,
            "AvgPredictedSize": best_metrics.avg_predicted_size,
            "Obj1": float(best_entry.obj[0]),
            "Obj2": float(best_entry.obj[1]),
            "Kv": best_entry.kv,
            "avgGO": best_entry.avg_go,
        }

    return row


# ============================================================
# Parallel runner
# ============================================================

def run_all_parallel(
    max_run: int = 30,
    population_size: int = 100,
    generations: int = 100,
    overlap_threshold: float = 0.20,
    min_complex_size: int = 3,
    n_workers: int = 40,
) -> None:
    """
    Parallel runner.

    Parallelization level:
        each Dataset × Method × Run is one independent task.

    For 44 cores:
        start with n_workers=40.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CYC2008 benchmark...")
    benchmark_names = read_cyc2008_complexes_txt(str(CYC2008_FILE))
    print(f"Raw benchmark complexes: {len(benchmark_names)}")

    print(f"Using parallel workers: {n_workers}")

    tasks: list[dict] = []

    for dataset_name, edge_path in NETWORK_FILES.items():
        print("\nPreparing dataset:", dataset_name)

        symbol_edges = load_edge_list_symbols(edge_path)
        edge_pairs, protein_labels, protein_to_idx, A_ppi = build_indexed_network(symbol_edges)

        print(f"Proteins: {len(protein_labels)}")
        print(f"Edges: {edge_pairs.shape[0]}")

        for method_name, flags in METHODS.items():
            for run_id in range(1, max_run + 1):
                tasks.append(
                    {
                        "dataset_name": dataset_name,
                        "method_name": method_name,
                        "flags": flags,
                        "run_id": run_id,
                        "edge_pairs": edge_pairs,
                        "symbol_edges": symbol_edges,
                        "protein_labels": protein_labels,
                        "protein_to_idx": protein_to_idx,
                        "benchmark_names": benchmark_names,
                        "population_size": population_size,
                        "generations": generations,
                        "overlap_threshold": overlap_threshold,
                        "min_complex_size": min_complex_size,
                    }
                )

    print(f"\nTotal tasks: {len(tasks)}")
    print("Starting parallel execution...\n")

    all_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {
            executor.submit(run_single_task, task): task
            for task in tasks
        }

        completed = 0

        for future in as_completed(future_to_task):
            task = future_to_task[future]

            dataset_name = task["dataset_name"]
            method_name = task["method_name"]
            run_id = task["run_id"]

            try:
                row = future.result()
                all_rows.append(row)

                completed += 1

                print(
                    f"[{completed}/{len(tasks)}] "
                    f"{dataset_name} | {method_name} | Run {run_id} | "
                    f"F1={row['F1']:.4f}, "
                    f"P={row['Precision']:.4f}, "
                    f"R={row['Recall']:.4f}, "
                    f"MMR={row['MMR']:.4f}, "
                    f"Pred={row['NumPredicted']}"
                )

                partial_df = pd.DataFrame(all_rows)
                partial_df.to_csv(
                    RESULTS_DIR / "TNSE_Python_AllRuns_parallel_partial.csv",
                    index=False,
                )

            except Exception as e:
                completed += 1

                print(
                    f"[{completed}/{len(tasks)}] ERROR in "
                    f"{dataset_name} | {method_name} | Run {run_id}: {repr(e)}"
                )

    all_df = pd.DataFrame(all_rows)

    if all_df.empty:
        print("No successful runs.")
        return

    all_df = all_df.sort_values(["Dataset", "Method", "Run"]).reset_index(drop=True)

    summary = (
        all_df
        .groupby(["Dataset", "Method"], as_index=False)
        .agg(
            Precision_Mean=("Precision", "mean"),
            Precision_Std=("Precision", "std"),
            Recall_Mean=("Recall", "mean"),
            Recall_Std=("Recall", "std"),
            F1_Mean=("F1", "mean"),
            F1_Std=("F1", "std"),
            MMR_Mean=("MMR", "mean"),
            MMR_Std=("MMR", "std"),
            NumPredicted_Mean=("NumPredicted", "mean"),
            NumPredicted_Std=("NumPredicted", "std"),
            AvgPredictedSize_Mean=("AvgPredictedSize", "mean"),
            AvgPredictedSize_Std=("AvgPredictedSize", "std"),
            Obj1_Mean=("Obj1", "mean"),
            Obj1_Std=("Obj1", "std"),
            Obj2_Mean=("Obj2", "mean"),
            Obj2_Std=("Obj2", "std"),
        )
    )

    out_all = RESULTS_DIR / "TNSE_Python_AllRuns_parallel.csv"
    out_summary = RESULTS_DIR / "TNSE_Python_Summary_parallel.csv"

    all_df.to_csv(out_all, index=False)
    summary.to_csv(out_summary, index=False)

    print("\nFinal summary:")
    print(summary)

    print("\nSaved:")
    print(out_all)
    print(out_summary)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ----------------------------
    # Smoke test first
    # ----------------------------
    run_all_parallel(
        max_run=30,
        population_size=100,
        generations=100,
        n_workers=40,
    )

