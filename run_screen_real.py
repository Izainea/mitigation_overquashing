"""
Experiment 2 (screening) — Psi severity index on real directed networks.

For each real network we compute, WITHOUT training anything:

  mass_k(j)  = sum_i (A^k)_{ij}   (total length-k walk mass into j)
  Psi_k(j;w) = log2(mass_k(j)) - log2(w)

via the row-vector iteration  v_k = v_{k-1} A  (v_0 = 1^T), which is
O(k * nnz(A)) — the sparse-precomputation argument of the paper, in action.

We report, per range k: the maximum bits of incoming walk entropy
(log2 max_j mass_k(j)), the fraction of nodes above capacity for widths
w in {32, 64, 128}, and — on the smaller graphs, where full sparse powers
fit — the count of vertex pairs joined by PARALLEL length-k walks
((A^k)_{ij} > 1), i.e. dim e_j kQ_k e_i > 1: the substrate the quotient
algebra kQ/I acts on.

Directions:
  'as-is'    arrows as stored (citing -> cited): chains of citations
             accumulate INTO foundational papers.
  'reversed' influence direction (cited -> citing): how far a paper's
             influence propagates forward.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
QA_DATA = os.path.normpath(os.path.join(HERE, "..", "quivers_analysis", "data"))
TABLES = os.path.join(HERE, "results", "tables")

WIDTHS = (32, 64, 128)
MAX_K = 8
PAIR_POWER_NNZ_LIMIT = 3e8   # abort full-power multiplicity count beyond this


# ---------------------------------------------------------------------------
# Loaders -> sparse adjacency (directed, multiplicity-preserving)
# ---------------------------------------------------------------------------

def load_cit_hepph() -> sp.csr_matrix:
    path = os.path.join(QA_DATA, "cit-hepph", "cit-HepPh.txt")
    e = np.loadtxt(path, dtype=np.int64, comments="#")
    ids = np.unique(e)
    remap = {int(v): i for i, v in enumerate(ids)}
    rows = np.fromiter((remap[int(u)] for u in e[:, 0]), dtype=np.int64)
    cols = np.fromiter((remap[int(v)] for v in e[:, 1]), dtype=np.int64)
    n = len(ids)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def load_cora() -> sp.csr_matrix:
    path = os.path.join(QA_DATA, "cora", "cora.cites")
    e = np.loadtxt(path, dtype=np.int64)
    # cora.cites lines are <cited> <citing>; flip to citing -> cited
    ids = np.unique(e)
    remap = {int(v): i for i, v in enumerate(ids)}
    rows = np.fromiter((remap[int(u)] for u in e[:, 1]), dtype=np.int64)
    cols = np.fromiter((remap[int(v)] for v in e[:, 0]), dtype=np.int64)
    n = len(ids)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def load_canadas() -> sp.csr_matrix:
    path = os.path.join(QA_DATA, "canadas_citation_network.json")
    d = json.load(open(path, encoding="utf-8"))
    papers = d["papers"]
    names = [p["id"] for p in papers] + list(d.get("reference_pool", {}).keys())
    remap = {v: i for i, v in enumerate(names)}
    rows, cols = [], []
    for p in papers:
        for ref in p.get("references", p.get("cites", [])):
            if ref in remap:
                rows.append(remap[p["id"]]); cols.append(remap[ref])
    n = len(names)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def screen(name: str, A: sp.csr_matrix, direction: str, max_k: int = MAX_K):
    if direction == "reversed":
        A = A.T.tocsr()
    n = A.shape[0]
    rows = []
    v = np.ones(n)                      # v_k = 1^T A^k  (mass INTO each node)
    Ak = None
    do_pairs = A.nnz * 1.0
    Ak = sp.identity(n, format="csr", dtype=np.float64)
    pairs_ok = True
    for k in range(1, max_k + 1):
        v = A.T @ v                     # (A^T v)_j = sum_i v_i A_{ij}
        pos = v > 0
        if not pos.any():
            break
        bits = np.log2(v[pos])
        row = dict(network=name, direction=direction, k=k,
                   reachable_targets=int(pos.sum()),
                   max_bits=round(float(bits.max()), 2),
                   mean_bits=round(float(bits.mean()), 2))
        for w in WIDTHS:
            row[f"frac_over_w{w}"] = round(float((bits > np.log2(w)).mean() * pos.mean()), 4)
        # parallel-walk multiplicity via full power, while it stays feasible
        if pairs_ok:
            try:
                Ak = (Ak @ A).tocsr()
                if Ak.nnz > PAIR_POWER_NNZ_LIMIT:
                    raise MemoryError
                row["pairs_with_parallel_walks"] = int((Ak.data > 1).sum())
                row["max_multiplicity"] = int(Ak.data.max()) if Ak.nnz else 0
            except MemoryError:
                pairs_ok = False
                row["pairs_with_parallel_walks"] = None
                row["max_multiplicity"] = None
        else:
            row["pairs_with_parallel_walks"] = None
            row["max_multiplicity"] = None
        rows.append(row)
    return rows


def main():
    out = []
    for name, loader in (("cora", load_cora),
                         ("canadas", load_canadas),
                         ("cit-HepPh", load_cit_hepph)):
        A = loader()
        print(f"[{name}] n={A.shape[0]}  arrows={A.nnz}")
        for direction in ("as-is", "reversed"):
            out += screen(name, A, direction)
    df = pd.DataFrame(out)
    dest = os.path.join(TABLES, "real_network_severity_screen.csv")
    df.to_csv(dest, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
