"""Recover the tuned genotype/expression kernel weight (w) for G+E models.

The main benchmark stores predictions but not the selected w; this script
re-tunes the same folds and records w and alpha, so we can test whether the
model leans more on the transcriptome when predicting unseen populations.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import Panel, make_splits, tune_krr, OUT, PROC

ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0])
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    P = Panel()
    traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    rows = []
    for r in traits.itertuples():
        t0 = time.time()
        ids, y, cl = P.trait(r.study_id, r.trait)
        cl, _ = P.merge_rare(ids, cl)
        ig = [P.gidx[a] for a in ids]; ie = [P.eidx[a] for a in ids]
        KG = P.KG[np.ix_(ig, ig)]; KE = P.KE[np.ix_(ie, ie)]
        for cv in ["random", "lopo"]:
            reps = [0, 1] if cv == "random" else [0]
            for rep in reps:
                splits = make_splits(cl, cv, seed=rep + 1)
                for k, (tr, te) in enumerate(splits):
                    inner = make_splits(cl[tr], cv, seed=100 + rep, inner=True)
                    w, a, _ = tune_krr(y, tr, inner, ALPHAS, KG=KG, KE=KE, wgrid=WGRID)
                    rows.append({"trait_id": r.trait_id, "group": r.group, "h2": r.h2,
                                 "cv": cv, "rep": rep, "fold": k, "n_train": len(tr),
                                 "w": w, "alpha": a})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "results", "kernel_weights.csv"), index=False)
        print(f"[{r.trait_id}] {time.time()-t0:.0f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
