"""
cit-HepPh temporal benchmark — model comparison (Frente B).

Transductive node classification on the context graph built by
run_hepph_dataset.py. All models share: the directed graph (citer -> cited,
so messages flow from citers into the papers they cite), the same minimal
features (pub_year, log1p in/out degree at the cut, z-scored), hidden width,
optimiser recipe (AdamW + warmup + cosine + grad clip), 5 seeds, and a
stratified 60/20/20 split of the cohort. Metric: balanced accuracy on test.

Models
------
mlp       features only (no graph) — the floor.
gcn       4 x GCNConv.
gat       4 x GATConv (4 heads).
walkattn  4 x WalkAttention, layer k attending over the top-T sources per
          target by length-k walk multiplicity (truncated support — the
          'Scaling the support' mechanism, operational).

Usage:  python run_hepph_models.py [t1 t3 t5 ...]   (default: all five)
Writes: results/tables/hepph_models_5seeds.csv (per-seed rows, appended)
"""

from __future__ import annotations

import os, sys, csv, math, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

from oversquash.attention import WalkAttention

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "hepph")
OUT = os.path.join(HERE, "results", "tables", "hepph_models_5seeds.csv")

HIDDEN, HEADS, LAYERS, DROP = 64, 4, 4, 0.2
EPOCHS, LR, WARMUP, PATIENCE = 200, 2e-3, 10, 30
TOP_T = 32
SEEDS = [0, 1, 2, 3, 4]
TARGETS = {
    "t1": "t1_impact_strat", "t2": "t2_early_riser", "t3": "t3_breadth",
    "t4": "t4_persistence", "t5": "t5_reach_growth",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_graph():
    e = np.load(os.path.join(DATA, "context_edges.npy"))
    n = int(e.max()) + 1
    A = sp.csr_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n))
    coh = pd.read_csv(os.path.join(DATA, "cohort.csv"))
    # features for ALL context nodes: in/out degree; pub_year only known for
    # cohort? -> recompute in/out degree from A; year needs the full map, so
    # rebuild it from cohort + degrees (non-cohort nodes get year = median).
    in_d = np.asarray(A.sum(axis=0)).ravel()
    out_d = np.asarray(A.sum(axis=1)).ravel()
    year = np.full(n, np.nan)
    year[coh.node.values] = coh.pub_year.values
    year[np.isnan(year)] = np.nanmedian(year)
    X = np.column_stack([year, np.log1p(in_d), np.log1p(out_d)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return A, torch.tensor(X, dtype=torch.float32), coh, n


def walk_masks(A: sp.csr_matrix, n_layers: int, top_t: int):
    """mask_k[j, i] = 1 iff i -> j by a length-k walk, truncated to the top-T
    sources per target by multiplicity. Returns sparse torch tensors."""
    n = A.shape[0]
    masks = []
    Ak = None
    for k in range(1, n_layers + 1):
        Ak = A if Ak is None else (Ak @ A).tocsc()
        Ac = Ak.tocsc()
        rows_t, cols_t = [], []       # (target j, source i)
        indptr, idx, dat = Ac.indptr, Ac.indices, Ac.data
        for j in range(n):
            lo, hi = indptr[j], indptr[j + 1]
            if lo == hi:
                continue
            src, mult = idx[lo:hi], dat[lo:hi]
            if len(src) > top_t:
                keep = np.argpartition(mult, -top_t)[-top_t:]
                src = src[keep]
            rows_t.append(np.full(len(src), j)); cols_t.append(src)
        tgt = np.concatenate(rows_t); srcs = np.concatenate(cols_t)
        m = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([tgt, srcs]), dtype=torch.long),
            torch.ones(len(tgt)), (n, n)).coalesce()
        masks.append(m)
        print(f"  mask k={k}: nnz={m._nnz()}")
    return masks


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, d_in, hidden, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ELU(),
                                 nn.Dropout(DROP), nn.Linear(hidden, hidden),
                                 nn.ELU(), nn.Linear(hidden, n_cls))
    def forward(self, x, *_): return self.net(x)


class GNN(nn.Module):
    def __init__(self, d_in, hidden, n_cls, kind, residual=False):
        super().__init__()
        self.residual = residual
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for l in range(LAYERS):
            i = d_in if l == 0 else hidden
            if kind == "gcn":
                self.convs.append(GCNConv(i, hidden, add_self_loops=True))
            else:
                self.convs.append(GATConv(i, hidden // HEADS, heads=HEADS,
                                          add_self_loops=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, n_cls)
    def forward(self, x, edge_index, _masks=None):
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.elu(norm(conv(x, edge_index))), DROP, self.training)
            x = h + x if (self.residual and h.shape == x.shape) else h
        return self.head(x)


class WANet(nn.Module):
    """One WalkAttention layer per range k = 1..LAYERS (paper architecture)."""
    def __init__(self, d_in, hidden, n_cls):
        super().__init__()
        self.layers, self.norms = nn.ModuleList(), nn.ModuleList()
        for l in range(LAYERS):
            i = d_in if l == 0 else hidden
            self.layers.append(WalkAttention(i, hidden // HEADS, n_heads=HEADS,
                                             concat=True, dropout=DROP))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, n_cls)
    def forward(self, x, _edge_index, masks):
        for layer, norm, m in zip(self.layers, self.norms, masks):
            x = F.dropout(F.elu(norm(layer(x, m))), DROP, self.training)
        return self.head(x)


def build(name, d_in, n_cls):
    if name == "mlp": return MLP(d_in, HIDDEN, n_cls)
    if name in ("gcn", "gat"): return GNN(d_in, HIDDEN, n_cls, name)
    if name in ("gcn_res", "gat_res"):
        return GNN(d_in, HIDDEN, n_cls, name[:3], residual=True)
    if name == "walkattn": return WANet(d_in, HIDDEN, n_cls)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def balanced_acc(logits, y):
    pred = logits.argmax(-1)
    accs = []
    for c in y.unique():
        m = y == c
        accs.append((pred[m] == c).float().mean().item())
    return float(np.mean(accs))


def run_one(model_name, target_col, seed, A, X, coh, edge_index, masks):
    torch.manual_seed(seed); np.random.seed(seed)
    lab = coh[target_col].values
    nodes = coh.node.values[~np.isnan(lab)]
    y_def = lab[~np.isnan(lab)].astype(int)

    # stratified 60/20/20 split
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y_def):
        idx = np.where(y_def == c)[0]; rng.shuffle(idx)
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    tr, va, te = map(np.array, (tr, va, te))

    y = torch.tensor(y_def, dtype=torch.long)
    node_t = torch.tensor(nodes, dtype=torch.long)
    n_cls = int(y.max()) + 1
    w = torch.tensor([1.0 / max((y_def == c).mean(), 1e-9)
                      for c in range(n_cls)], dtype=torch.float32)

    model = build(model_name, X.size(1), n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / WARMUP if ep < WARMUP
        else 0.5 * (1 + math.cos(math.pi * (ep - WARMUP) / max(EPOCHS - WARMUP, 1))))

    best_va, best_te, patience = -1, -1, 0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        logits = model(X, edge_index, masks)[node_t]
        loss = F.cross_entropy(logits[torch.tensor(tr)], y[torch.tensor(tr)], weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(X, edge_index, masks)[node_t]
            va_acc = balanced_acc(logits[torch.tensor(va)], y[torch.tensor(va)])
            te_acc = balanced_acc(logits[torch.tensor(te)], y[torch.tensor(te)])
        if va_acc > best_va:
            best_va, best_te, patience = va_acc, te_acc, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    return best_va, best_te


def main():
    targets = [t for t in sys.argv[1:] if t in TARGETS] or list(TARGETS)
    A, X, coh, n = load_graph()
    e = np.load(os.path.join(DATA, "context_edges.npy"))
    edge_index = torch.tensor(e, dtype=torch.long)
    print(f"graph: {n} nodes, {e.shape[1]} edges; targets: {targets}")
    print("building walk masks (top-%d)..." % TOP_T)
    masks = walk_masks(A, LAYERS, TOP_T)

    exists = os.path.exists(OUT)
    f = open(OUT, "a", newline=""); wcsv = csv.writer(f)
    if not exists:
        wcsv.writerow(["target", "seed", "model", "val_bacc", "test_bacc"])
    t0 = time.time()
    for tkey in targets:
        col = TARGETS[tkey]
        import os as _os
        model_list = tuple(_os.environ.get(
            "HEPPH_MODELS", "mlp,gcn,gat,walkattn").split(","))
        for model_name in model_list:
            accs = []
            for seed in SEEDS:
                va, te = run_one(model_name, col, seed, A, X, coh,
                                 edge_index, masks)
                accs.append(te)
                wcsv.writerow([tkey, seed, model_name,
                               round(va, 4), round(te, 4)])
                f.flush()
                print(f"[{tkey} s={seed}] {model_name:9s} test={te:.3f}  "
                      f"({(time.time()-t0)/60:.0f} min)")
            print(f"  {tkey} {model_name:9s} {np.mean(accs):.3f} "
                  f"+- {np.std(accs):.3f}")
    f.close()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
