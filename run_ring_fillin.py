"""RingTransfer rerun: GCN / GAT / WalkAttention across ring sizes, 5 seeds
(the published table was single-seed, which produced a spurious non-monotone
GCN column). One shared optimiser recipe for all models (AdamW + 5-epoch
warmup + cosine decay + grad clip). Run: python run_ring_multiseed.py
Writes results/tables/ring_transfer_5seeds.csv (per-seed rows)."""
import time, math, warnings, csv
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGLoader
from oversquash.data_ring import ring_dataset
from oversquash.lrgb import AttachLRGBMasks, collate_lrgb
from oversquash.models import build_model

K = 5                       # classes; chance = 0.20
RINGS = [12, 16, 20]     # fill-in ring sizes for the severity-index curve
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN, N_VAL = 1000, 400
EPOCHS, LR, BS, HIDDEN = 60, 5e-3, 64, 32
OUT = 'results/tables/ring_transfer_fillin_5seeds.csv'

def evaluate(m, loader, fwd):
    m.eval(); acc = []
    with torch.no_grad():
        for b in loader:
            lg = fwd(m, b); mm = b.root_mask
            acc.append((lg[mm].argmax(-1) == b.y[mm]).float().mean().item())
    return float(np.mean(acc))

def run_one(name, n, nl, seed):
    torch.manual_seed(seed)
    tr_raw = ring_dataset(n, N_TRAIN, seed=2 * seed, num_classes=K)
    va_raw = ring_dataset(n, N_VAL, seed=2 * seed + 1, num_classes=K)
    in_dim = tr_raw[0].x.size(-1)
    if name in ('gcn', 'gat'):
        trl = PyGLoader(tr_raw, batch_size=BS, shuffle=True)
        val = PyGLoader(va_raw, batch_size=BS)
        fwd = lambda m, b: m(b.x, b.edge_index, getattr(b, 'batch', None))[0]
    else:  # walkattn consumes the precomputed walk masks
        tf = AttachLRGBMasks(n_layers=nl)
        tr = [tf(d) for d in tr_raw]; va = [tf(d) for d in va_raw]
        trl = DataLoader(tr, batch_size=BS, shuffle=True, collate_fn=collate_lrgb)
        val = DataLoader(va, batch_size=BS, collate_fn=collate_lrgb)
        fwd = lambda m, b: m(b.x, b.edge_index, getattr(b, 'batch', None),
                             walk_masks=b.walk_masks)[0]
    m = build_model(name, in_dim, HIDDEN, K, nl, heads=4)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1, (e + 1) / 5) *
        (0.5 * (1 + math.cos(math.pi * max(0, e - 5) / max(1, EPOCHS - 5)))))
    best = 0.0
    for e in range(EPOCHS):
        m.train()
        for b in trl:
            opt.zero_grad()
            F.cross_entropy(fwd(m, b), b.y, ignore_index=-100).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        best = max(best, evaluate(m, val, fwd))
    return best

rows, t00 = [], time.time()
for n in RINGS:
    nl = n // 2
    for seed in SEEDS:
        for name in ['gcn', 'gat', 'walkattn']:
            acc = run_one(name, n, nl, seed)
            rows.append({'ring': n, 'distance': nl, 'seed': seed,
                         'model': name, 'acc': round(acc, 4)})
            print(f'[n={n} s={seed}] {name:9s} acc={acc:.3f}  '
                  f'({(time.time()-t00)/60:.0f} min elapsed)', flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['ring', 'distance', 'seed', 'model', 'acc'])
    w.writeheader()
    for r in rows: w.writerow(r)

print('\n=== SUMMARY (mean +- std over 5 seeds) ===')
import collections
agg = collections.defaultdict(list)
for r in rows: agg[(r['ring'], r['model'])].append(r['acc'])
for (n, mname), vs in sorted(agg.items()):
    print(f'  n={n:2d} {mname:9s} {np.mean(vs):.3f} +- {np.std(vs, ddof=1):.3f}')
print('saved ' + OUT)
