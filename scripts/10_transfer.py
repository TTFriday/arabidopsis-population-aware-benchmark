"""Cross-environment / cross-study transfer (train on A, evaluate on B, held-out accessions)."""
import os, sys, time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import (Panel, make_splits, tune_krr, krr_fit_predict,
                           scale_blocks, OUT)

ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0])
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PAIRS = [
    ("study12", "FT16", "study38", "DTF2", "related endpoints, cross-study"),
    ("study38", "DTF2", "study12", "FT16", "related endpoints, cross-study"),
    ("study12", "FT16", "study12", "FT10", "same study, 16C -> 10C"),
    ("study43", "stomata density 261", "study43", "stomata density", "same conditions, paired records"),
    ("study43", "Stomatasize 261", "study43", "stomatasize", "same conditions, paired records"),
    ("study127", "Raffinose 22C", "study127", "Raffinose 32C", "22C -> 32C"),
    ("study127", "Raffinose 32C", "study127", "Raffinose 22C", "32C -> 22C"),
]


def wz(v):
    lo, hi = np.percentile(v, [1, 99]); v = np.clip(v, lo, hi)
    return (v - v.mean()) / v.std()


def fit_lasso(Xtr, ytr, Xte, inner):
    m = LassoCV(cv=inner, alphas=30, eps=1e-3, max_iter=3000, tol=1e-3,
                n_jobs=1, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def perm_p(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs = pearsonr(y, pred)[0]
    null = np.array([pearsonr(rng.permutation(y), pred)[0] for _ in range(n)])
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1)


def _score(oy, op):
    m = ~np.isnan(oy)
    r, p = perm_p(oy[m], op[m])
    return r, p, int(np.sum(m))


def run_krr(K, y_fit, y_eval, splits, cl):
    n = len(y_fit); oy = np.full(n, np.nan); op = np.full(n, np.nan)
    for tr, te in splits:
        inner = make_splits(cl[tr], "random", seed=10, inner=True)
        _, a, _ = tune_krr(y_fit, tr, inner, ALPHAS, K=K)
        op[te] = krr_fit_predict(K, y_fit, tr, te, a)
        oy[te] = y_eval[te]
    return _score(oy, op)


def run_ge_krr(KG, KE, y_fit, y_eval, splits, cl):
    n = len(y_fit); oy = np.full(n, np.nan); op = np.full(n, np.nan)
    for tr, te in splits:
        inner = make_splits(cl[tr], "random", seed=10, inner=True)
        w, a, _ = tune_krr(y_fit, tr, inner, ALPHAS, KG=KG, KE=KE, wgrid=WGRID)
        Kw = w * KG + (1 - w) * KE
        op[te] = krr_fit_predict(Kw, y_fit, tr, te, a)
        oy[te] = y_eval[te]
    return _score(oy, op)


def run_lasso(Xg, Xe, pg, pe, y_fit, y_eval, splits, cl, feat):
    n = len(y_fit); oy = np.full(n, np.nan); op = np.full(n, np.nan)
    for tr, te in splits:
        inner = make_splits(cl[tr], "random", seed=10, inner=True)
        if feat == "G":
            Xtr, Xte = scale_blocks(Xg, None, tr, te, pg, 1)
        elif feat == "E":
            Xtr, Xte = scale_blocks(None, Xe, tr, te, 1, pe)
        else:
            Xtr, Xte = scale_blocks(Xg, Xe, tr, te, pg, pe)
        op[te] = fit_lasso(Xtr, y_fit[tr], Xte, inner)
        oy[te] = y_eval[te]
    return _score(oy, op)


def main():
    P = Panel(); rows = []
    for sidA, trA, sidB, trB, note in PAIRS:
        gA = P.long[(P.long.study_id == int(sidA[5:])) & (P.long.trait == trA)].groupby("accession_id")["value"].mean()
        gB = P.long[(P.long.study_id == int(sidB[5:])) & (P.long.trait == trB)].groupby("accession_id")["value"].mean()
        ids = sorted(set(gA.index) & set(gB.index) & P.gset & P.eset, key=lambda x: int(x))
        if len(ids) < 40:
            continue
        yA, yB = wz(gA.loc[ids].values), wz(gB.loc[ids].values)
        cl = np.array([P.pop[a] for a in ids])
        ig = [P.gidx[a] for a in ids]; ie = [P.eidx[a] for a in ids]
        KG = P.KG[np.ix_(ig, ig)]; KE = P.KE[np.ix_(ie, ie)]
        Xg = P.G[ig]; Xe = P.E[ie]; pg, pe = Xg.shape[1], Xe.shape[1]
        splits = make_splits(cl, "random", seed=2)
        print(f"\n{trA} -> {trB}: n={len(ids)} pheno r={np.corrcoef(yA,yB)[0,1]:.3f}", flush=True)

        for model in ["kridge", "lasso"]:
            for feat in ["G", "E", "GE"]:
                t0 = time.time()
                if model == "kridge":
                    if feat == "G": r, p, nn = run_krr(KG, yA, yB, splits, cl)
                    elif feat == "E": r, p, nn = run_krr(KE, yA, yB, splits, cl)
                    else: r, p, nn = run_ge_krr(KG, KE, yA, yB, splits, cl)
                else:
                    r, p, nn = run_lasso(Xg, Xe, pg, pe, yA, yB, splits, cl, feat)
                rows.append({"source": trA, "target": trB, "note": note, "model": model,
                             "features": feat, "n": nn, "pheno_r": np.corrcoef(yA, yB)[0, 1],
                             "r": r, "perm_p": p, "sec": time.time() - t0})
                print(f"  {model:6s} {feat:2s} r={r: .3f} p={p:.4f} ({time.time()-t0:.0f}s)", flush=True)

        for feat in ["G", "GE"]:
            t0 = time.time()
            if feat == "G": r, _, _ = run_krr(KG, yB, yB, splits, cl)
            else: r, _, _ = run_ge_krr(KG, KE, yB, yB, splits, cl)
            rows.append({"source": "within", "target": trB, "note": "within-target random CV",
                         "model": "kridge", "features": feat, "n": len(ids), "pheno_r": np.nan,
                         "r": r, "perm_p": np.nan, "sec": time.time() - t0})
            print(f"  within {feat}: r={r:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "results", "transfer.csv"), index=False)
    print("\nDONE")


if __name__ == "__main__":
    main()
