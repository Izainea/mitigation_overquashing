"""
Experiment 1 — Algebraic over-squashing severity index vs. measured advantage.

The index, per target j at range k, is the excess of incoming walk entropy over
representation capacity:

    Psi_k(j) = log2( sum_i (A^k)_{ij} ) - log2(w)

where sum_i (A^k)_{ij} = sum_i dim(e_j kQ_k e_i) is the total number of
length-k walks into j (graded path algebra, Prop. count in the paper), and w is
the hidden width used in the experiments. Over-squashing at j is predicted when
Psi_k(j) > 0: more walk-information flows in than the embedding can hold
(cf. aiq.gnn.over_squashing_diagnostic, which uses the per-pair version
H_k(i,j) = log2 (A^k)_{ij}).

This script computes Psi on every benchmark already run for the paper
(bottleneck chains, RingTransfer, Peptides-func) at the decision-relevant
target and range, and joins it with the measured advantage of Walk Attention
over the best one-hop baseline from results/tables/*.csv. No training — pure
sparse linear algebra on A.

Peptides requires torch (to read the processed dataset); the synthetic part
runs with numpy/scipy alone and is emitted first.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "results", "tables")


# ---------------------------------------------------------------------------
# Graph constructions (exact replicas of src/oversquash/data_bottleneck.py
# and data_ring.py, numpy-only)
# ---------------------------------------------------------------------------

def bottleneck_adjacency(K: int, M: int, depth: int):
    """A for the bottleneck chain; returns (A, target, n)."""
    layers = []
    nid = K
    for _ in range(depth):
        layers.append(list(range(nid, nid + M)))
        nid += M
    target = nid
    n = nid + 1
    rows, cols = [], []
    for s in range(K):
        for v in layers[0]:
            rows.append(s); cols.append(v)
    for d in range(depth - 1):
        for u in layers[d]:
            for v in layers[d + 1]:
                rows.append(u); cols.append(v)
    for u in layers[-1]:
        rows.append(u); cols.append(target)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return A, target, n


def ring_adjacency(n: int):
    """A for the undirected cycle (both orientations); target = n//2."""
    rows, cols = [], []
    for i in range(n):
        j = (i + 1) % n
        rows += [i, j]; cols += [j, i]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return A, n // 2


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

def psi_at_target(A: sp.spmatrix, target: int, k: int, width: int) -> float:
    """Psi_k(target) = log2(total length-k walk mass into target) - log2(w)."""
    Ak = sp.identity(A.shape[0], format="csr")
    for _ in range(k):
        Ak = Ak @ A
    mass = float(np.asarray(Ak[:, target].todense()).sum())
    return float(np.log2(mass) - np.log2(width)) if mass > 0 else -np.inf


def psi_graph_stats(A: sp.spmatrix, ks, width: int):
    """max/mean over targets of Psi_k, maximized over the used ranges ks."""
    n = A.shape[0]
    best = np.full(n, -np.inf)
    Ak = sp.identity(n, format="csr")
    for k in range(1, max(ks) + 1):
        Ak = Ak @ A
        if k in ks:
            mass = np.asarray(Ak.sum(axis=0)).ravel()  # column sums: into j
            with np.errstate(divide="ignore"):
                psi = np.log2(np.maximum(mass, 1e-300)) - np.log2(width)
            best = np.maximum(best, psi)
    return float(best.max()), float(best.mean())


# ---------------------------------------------------------------------------
# Measured advantages (from the already-published CSVs)
# ---------------------------------------------------------------------------

def advantage_bottleneck(depth: int) -> tuple[float, float]:
    """WA mean - best one-hop baseline mean (GCN/GIN/GAT), on val accuracy."""
    main = pd.read_csv(os.path.join(TABLES, "bottleneck_main_5seeds.csv"))
    gg = pd.read_csv(os.path.join(TABLES, "bottleneck_gcn_gin_5seeds.csv"))
    df = pd.concat([main, gg])
    df = df[df.depth == depth]
    m = df.groupby("model")["val_acc"].mean()
    onehop = m[[i for i in m.index if i in ("gcn", "gin", "gat")]].max()
    return float(m["walkattn"]), float(m["walkattn"] - onehop)


def advantage_ring(n: int) -> tuple[float, float]:
    df = pd.read_csv(os.path.join(TABLES, "ring_transfer_5seeds.csv"))
    df = df[df.ring == n]
    m = df.groupby("model")["acc"].mean()
    onehop = m[[i for i in m.index if i in ("gcn", "gat")]].max()
    return float(m["walkattn"]), float(m["walkattn"] - onehop)


def advantage_peptides() -> tuple[float, float]:
    df = pd.read_csv(os.path.join(TABLES, "lrgb_peptides_func_4seeds.csv"))
    m = df.groupby("model")["test_ap"].mean()
    onehop = m[[i for i in m.index if i in ("gcn", "gat")]].max()
    return float(m["walkattn"]), float(m["walkattn"] - onehop)


# ---------------------------------------------------------------------------
# Peptides (needs torch; skipped gracefully if unavailable)
# ---------------------------------------------------------------------------

def peptides_psi(width: int = 80, ks=(1, 2, 3, 4), split: str = "train",
                 max_graphs: int | None = None):
    """Mean over molecules of the per-graph max/mean Psi at the used ranges."""
    import torch
    from torch_geometric.datasets import LRGBDataset
    root = os.path.join(HERE, "data", "lrgb")
    ds = LRGBDataset(root=root, name="Peptides-func", split=split)
    maxes, means = [], []
    idx = range(len(ds)) if max_graphs is None else range(min(len(ds), max_graphs))
    for i in idx:
        d = ds[i]
        ei = d.edge_index.numpy()
        n = int(d.num_nodes)
        A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
        mx, mn = psi_graph_stats(A, set(ks), width)
        maxes.append(mx); means.append(mn)
    return float(np.mean(maxes)), float(np.mean(means)), len(maxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = []

    # Bottleneck chains: K=5, M=4, w=16, decision range k=d+1, decision target
    for d in (2, 3):
        A, target, n = bottleneck_adjacency(K=5, M=4, depth=d)
        psi = psi_at_target(A, target, k=d + 1, width=16)
        wa, adv = advantage_bottleneck(d)
        rows.append(dict(benchmark=f"bottleneck_d{d}", n_nodes=n, width=16,
                         range_k=d + 1, psi_target=round(psi, 3),
                         walkattn=round(wa, 3), advantage=round(adv, 3)))

    # RingTransfer: w=32, decision range k=n/2, target antipodal
    for n in (6, 10, 14, 18):
        A, target = ring_adjacency(n)
        psi = psi_at_target(A, target, k=n // 2, width=32)
        wa, adv = advantage_ring(n)
        rows.append(dict(benchmark=f"ring_n{n}", n_nodes=n, width=32,
                         range_k=n // 2, psi_target=round(psi, 3),
                         walkattn=round(wa, 3), advantage=round(adv, 3)))

    # Peptides-func: w=80, ranges 1..4; per-graph max over targets, then mean
    try:
        mx, mn, ngraphs = peptides_psi()
        wa, adv = advantage_peptides()
        rows.append(dict(benchmark="peptides_func", n_nodes=f"{ngraphs} graphs",
                         width=80, range_k="1-4", psi_target=round(mx, 3),
                         walkattn=round(wa, 3), advantage=round(adv, 3)))
    except Exception as e:  # torch missing or dataset unreadable
        print(f"[peptides skipped: {type(e).__name__}: {e}]")

    df = pd.DataFrame(rows)
    out = os.path.join(TABLES, "severity_index.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten: {out}")

    # Rank correlation between index and advantage (the headline number)
    dfn = df[pd.to_numeric(df.psi_target, errors="coerce").notna()]
    if len(dfn) >= 3:
        from scipy.stats import spearmanr, pearsonr
        rho, p = spearmanr(dfn.psi_target, dfn.advantage)
        r, pr = pearsonr(dfn.psi_target, dfn.advantage)
        print(f"\nSpearman rho(Psi, advantage) = {rho:.3f} (p={p:.4f})  "
              f"| Pearson r = {r:.3f} (p={pr:.4f})  | n={len(dfn)}")


if __name__ == "__main__":
    main()
