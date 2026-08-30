"""
cit-HepPh temporal benchmark — dataset builder (Frente B).

Setup
-----
Context graph : papers dated <= CUT (1997-12-31) and citations among them.
Cohort        : papers dated in [1996-01-01, CUT] (they have citation history
                by the cut AND room to grow afterwards).
Future window : citations arriving from papers dated in [1998-01-01,
                2000-12-31].

Node features at the cut (same for every model; deliberately minimal):
    pub_year (fractional), in_degree, out_degree  — all at the cut.

Targets (per cohort paper)
--------------------------
T1  impact_stratified : top-quartile of future citations WITHIN its stratum
                        of in-degree at cut {0, 1-2, 3-9, 10+}. Binary.
T2  early_riser       : top-quartile of the residual of log(1+c_fut) on
                        [log(1+in_deg), age]. Binary.
T3  breadth           : among future citers, fraction of INDEPENDENT pairs
                        (no citation between them, no shared reference other
                        than the paper itself); above-median. Binary; defined
                        when the paper has >= 2 future citers.
T4  persistence       : fraction of future citations arriving late
                        (1999-07-01 onward); above-median. Binary; defined
                        when the paper has >= 3 future citations.
T5  reach_growth      : growth of |{j : j -> ... -> c, length <= 2}| (the
                        impact 2-neighbourhood, reversed direction) between
                        the cut graph and the 2000 graph; top-quartile. Binary.

Cross-listed SNAP ids (11<true_id>) are normalised by subtracting 1.1e8.

Outputs
-------
data/hepph/context_edges.npy   (2, E) int64, citing -> cited, remapped ids
data/hepph/cohort.csv          node, features, strata, all targets
data/hepph/meta.json           counts, class balances, config
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "quivers_analysis", "data", "cit-hepph"))
OUT = os.path.join(HERE, "data", "hepph")

CUT = 1998.0            # context: date < CUT
COHORT_FROM = 1996.0
FUT_FROM, FUT_TO = 1998.0, 2001.0   # future window [1998, 2001)
LATE_FROM = 1999.5                  # T4: "late" citations
MAX_CITERS_PAIRS = 100              # T3: sample cap for pairwise independence
SEED = 0


def normalize(ids: np.ndarray) -> np.ndarray:
    ids = ids.copy()
    ids[ids >= 110_000_000] -= 110_000_000
    return ids


def load_raw():
    e = np.loadtxt(os.path.join(RAW, "cit-HepPh.txt"), dtype=np.int64, comments="#")
    d = np.genfromtxt(os.path.join(RAW, "cit-HepPh-dates.txt"), dtype=None,
                      encoding="utf-8", comments="#",
                      names=("id", "date"))
    ids = normalize(d["id"].astype(np.int64))
    dates = pd.to_datetime(d["date"])
    fyear = dates.year + (dates.dayofyear - 1) / 365.25
    date_of = {}
    for i, y in zip(ids, fyear):          # keep the earliest date on duplicates
        if i not in date_of or y < date_of[i]:
            date_of[i] = float(y)
    e = normalize(e.ravel()).reshape(-1, 2)
    return e, date_of


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)
    edges, date_of = load_raw()

    # keep only edges whose two endpoints are dated
    dated = np.array([[date_of.get(u, np.nan), date_of.get(v, np.nan)]
                      for u, v in edges])
    ok = ~np.isnan(dated).any(axis=1)
    edges, dated = edges[ok], dated[ok]
    print(f"dated edges: {len(edges)}")

    du, dv = dated[:, 0], dated[:, 1]      # date(citing), date(cited)

    # ---- graphs -----------------------------------------------------------
    ctx_mask = (du < CUT) & (dv < CUT)
    fut_mask = (du >= FUT_FROM) & (du < FUT_TO)       # future citing acts
    e_ctx = edges[ctx_mask]
    e_fut = edges[fut_mask]                            # citing -> cited

    ctx_nodes = np.unique(e_ctx)
    # cohort: dated papers in the context graph published in [1996, CUT)
    cohort = np.array(sorted(i for i in ctx_nodes
                             if COHORT_FROM <= date_of[i] < CUT))
    print(f"context nodes: {len(ctx_nodes)}  cohort: {len(cohort)}")

    remap = {int(v): i for i, v in enumerate(ctx_nodes)}
    n = len(ctx_nodes)
    rows = np.fromiter((remap[int(u)] for u in e_ctx[:, 0]), np.int64)
    cols = np.fromiter((remap[int(v)] for v in e_ctx[:, 1]), np.int64)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    in_deg = np.asarray(A.sum(axis=0)).ravel()     # citations received at cut
    out_deg = np.asarray(A.sum(axis=1)).ravel()    # references made
    years = np.array([date_of[int(v)] for v in ctx_nodes])

    # ---- future citations per cohort paper --------------------------------
    fut_by_paper: dict[int, list[tuple[int, float]]] = {int(c): [] for c in cohort}
    for (u, v), yu in zip(e_fut, du[fut_mask]):
        if int(v) in fut_by_paper:
            fut_by_paper[int(v)].append((int(u), float(yu)))

    c_fut = np.array([len(fut_by_paper[int(c)]) for c in cohort])
    print(f"future citations: total={c_fut.sum()}  mean={c_fut.mean():.2f}  "
          f"median={np.median(c_fut):.0f}  max={c_fut.max()}")

    cidx = np.array([remap[int(c)] for c in cohort])
    cin, cage = in_deg[cidx], CUT - years[cidx]

    # ---- T1: stratified top-quartile --------------------------------------
    strata = np.digitize(cin, [1, 3, 10])          # 0:0, 1:1-2, 2:3-9, 3:10+
    t1 = np.zeros(len(cohort), dtype=int)
    for s in range(4):
        m = strata == s
        if m.sum() < 8:
            continue
        thr = np.quantile(c_fut[m], 0.75)
        t1[m] = (c_fut[m] > thr).astype(int)       # strict: ties -> 0

    # ---- T2: early-riser residual -----------------------------------------
    X = np.column_stack([np.log1p(cin), cage, np.ones(len(cohort))])
    y = np.log1p(c_fut)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    t2 = (resid > np.quantile(resid, 0.75)).astype(int)

    # ---- T3: breadth (independent future citers) --------------------------
    # reference sets over the FULL dated graph up to 2001 (citing -> cited)
    all_mask = du < FUT_TO
    refs: dict[int, set] = {}
    for u, v in edges[all_mask]:
        refs.setdefault(int(u), set()).add(int(v))
    cites_pair = {(int(u), int(v)) for u, v in edges[all_mask]}

    t3_frac = np.full(len(cohort), np.nan)
    for ci, c in enumerate(cohort):
        citers = [u for u, _ in fut_by_paper[int(c)]]
        if len(citers) < 2:
            continue
        if len(citers) > MAX_CITERS_PAIRS:
            citers = list(rng.choice(citers, MAX_CITERS_PAIRS, replace=False))
        indep = tot = 0
        for a in range(len(citers)):
            for b in range(a + 1, len(citers)):
                u, v = citers[a], citers[b]
                tot += 1
                if (u, v) in cites_pair or (v, u) in cites_pair:
                    continue
                shared = (refs.get(u, set()) & refs.get(v, set())) - {int(c)}
                if not shared:
                    indep += 1
        t3_frac[ci] = indep / tot if tot else np.nan
    t3_def = ~np.isnan(t3_frac)
    med3 = np.nanmedian(t3_frac)
    t3 = np.where(t3_def, (t3_frac > med3).astype(float), np.nan)

    # ---- T4: persistence ---------------------------------------------------
    t4_frac = np.full(len(cohort), np.nan)
    for ci, c in enumerate(cohort):
        ys = [y for _, y in fut_by_paper[int(c)]]
        if len(ys) >= 3:
            t4_frac[ci] = np.mean(np.array(ys) >= LATE_FROM)
    t4_def = ~np.isnan(t4_frac)
    med4 = np.nanmedian(t4_frac)
    t4 = np.where(t4_def, (t4_frac > med4).astype(float), np.nan)

    # ---- T5: reach growth (impact 2-neighbourhood, reversed) --------------
    def reach2_sizes(edge_arr, targets_set):
        """|{j : j -> ... -> c, length <= 2}| for each cohort paper c."""
        citers_of: dict[int, set] = {}
        for u, v in edge_arr:
            citers_of.setdefault(int(v), set()).add(int(u))
        out = {}
        for c in targets_set:
            l1 = citers_of.get(int(c), set())
            l2 = set()
            for u in l1:
                l2 |= citers_of.get(u, set())
            out[int(c)] = len((l1 | l2) - {int(c)})
        return out

    ch = set(int(c) for c in cohort)
    r_cut = reach2_sizes(edges[ctx_mask], ch)
    r_fut = reach2_sizes(edges[du < FUT_TO], ch)
    growth = np.array([r_fut[int(c)] - r_cut[int(c)] for c in cohort])
    t5 = (growth > np.quantile(growth, 0.75)).astype(int)

    # ---- write -------------------------------------------------------------
    np.save(os.path.join(OUT, "context_edges.npy"),
            np.vstack([rows, cols]).astype(np.int64))
    df = pd.DataFrame({
        "node": cidx,                       # index into context graph
        "snap_id": cohort,
        "pub_year": years[cidx].round(3),
        "in_deg_cut": cin.astype(int),
        "out_deg_cut": out_deg[cidx].astype(int),
        "stratum": strata,
        "c_future": c_fut,
        "t1_impact_strat": t1,
        "t2_early_riser": t2,
        "t3_breadth": t3,
        "t3_frac_indep": np.round(t3_frac, 4),
        "t4_persistence": t4,
        "t4_frac_late": np.round(t4_frac, 4),
        "t5_reach_growth": t5,
        "reach2_growth": growth,
    })
    df.to_csv(os.path.join(OUT, "cohort.csv"), index=False)

    meta = dict(
        cut=CUT, cohort_from=COHORT_FROM, future=[FUT_FROM, FUT_TO],
        n_context=int(n), n_context_edges=int(A.nnz),
        n_cohort=int(len(cohort)),
        strata_sizes={int(s): int((strata == s).sum()) for s in range(4)},
        balance=dict(
            t1=float(t1.mean()), t2=float(t2.mean()),
            t3=float(np.nanmean(t3)), t3_defined=int(t3_def.sum()),
            t4=float(np.nanmean(t4)), t4_defined=int(t4_def.sum()),
            t5=float(t5.mean()),
        ),
    )
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\nwritten: {OUT}\\cohort.csv, context_edges.npy, meta.json")


if __name__ == "__main__":
    main()
