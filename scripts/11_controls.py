"""Negative controls: label permutation and kernel-alignment permutation.

Hyperparameters are tuned inside each outer training fold, exactly as in the
main benchmark, so the observed r here matches the main results.
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import (Panel, make_splits, tune_krr, krr_fit_predict,
                           center_kernel, OUT)

ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0])
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAITS = [(12, "FT16"), (38, "DTF2"), (126, "Trichome density"),
          (43, "stomata density 261"), (34, "rSeeds thp"), (127, "Raffinose 22C")]
NLAB = 300; NKER = 100


def tuned_folds(KG, KE, y, cl, splits, feat):
    folds = []
    for tr, te in splits:
        inner = make_splits(cl[tr], "random", seed=100, inner=True)
        if feat == "GE":
            w, a, _ = tune_krr(y, tr, inner, ALPHAS, KG=KG, KE=KE, wgrid=WGRID)
            K = w * KG + (1 - w) * KE
        else:
            K = KG if feat == "G" else KE
            _, a, _ = tune_krr(y, tr, inner, ALPHAS, K=K)
            w = np.nan
        folds.append({"tr": tr, "te": te, "K": K, "alpha": a, "w": w})
    return folds


def real_r(folds, y):
    oy = np.full(len(y), np.nan); op = np.full(len(y), np.nan)
    for f in folds:
        op[f["te"]] = krr_fit_predict(f["K"], y, f["tr"], f["te"], f["alpha"])
        oy[f["te"]] = y[f["te"]]
    return pearsonr(oy, op)[0]


def null_label(folds, y, nperm=NLAB, seed=0):
    cache = []
    for f in folds:
        Kc, Kvc = center_kernel(f["K"][np.ix_(f["tr"], f["tr"])],
                                f["K"][np.ix_(f["te"], f["tr"])])
        vals, U = eigh(Kc); vals = np.clip(vals, 1e-8, None)
        cache.append((f["tr"], f["te"], U, vals, Kvc, f["alpha"]))
    rng = np.random.default_rng(seed)
    out = np.empty(nperm)
    for p in range(nperm):
        yp = rng.permutation(y)
        oy = np.full(len(y), np.nan); op = np.full(len(y), np.nan)
        for tr, te, U, vals, Kvc, a in cache:
            yc = yp[tr] - yp[tr].mean()
            coef = U @ ((U.T @ yc) / (vals + a))
            op[te] = Kvc @ coef + yp[tr].mean(); oy[te] = yp[te]
        out[p] = pearsonr(oy, op)[0]
    return out


def null_kernel(folds, y, nperm=NKER, seed=1):
    rng = np.random.default_rng(seed); n = len(y); out = np.empty(nperm)
    for p in range(nperm):
        P = rng.permutation(n)
        oy = np.full(n, np.nan); op = np.full(n, np.nan)
        for f in folds:
            Kp = f["K"][np.ix_(P, P)]
            op[f["te"]] = krr_fit_predict(Kp, y, f["tr"], f["te"], f["alpha"])
            oy[f["te"]] = y[f["te"]]
        out[p] = pearsonr(oy, op)[0]
    return out


def main():
    P = Panel(); rows = []
    for sid, tr in TRAITS:
        ids, y, cl = P.trait(sid, tr); cl, _ = P.merge_rare(ids, cl)
        ig = [P.gidx[a] for a in ids]; ie = [P.eidx[a] for a in ids]
        KG = P.KG[np.ix_(ig, ig)]; KE = P.KE[np.ix_(ie, ie)]
        splits = make_splits(cl, "random", seed=1)
        print(f"\n[{tr}] n={len(y)}", flush=True)
        for feat in ["G", "E", "GE"]:
            t0 = time.time()
            folds = tuned_folds(KG, KE, y, cl, splits, feat)
            r = real_r(folds, y)
            nl = null_label(folds, y)
            nk = null_kernel(folds, y)
            p_lab = (1 + np.sum(np.abs(nl) >= abs(r))) / (len(nl) + 1)
            p_ker = (1 + np.sum(np.abs(nk) >= abs(r))) / (len(nk) + 1)
            alphas = [f["alpha"] for f in folds]; ws = [f["w"] for f in folds]
            rows.append({"trait": tr, "study_id": sid, "features": feat, "n": len(y),
                         "real_r": r, "alpha": float(np.median(alphas)),
                         "w": float(np.nanmedian(ws)) if feat == "GE" else np.nan,
                         "null_label_mean": nl.mean(), "null_label_sd": nl.std(),
                         "null_label_q95": np.percentile(np.abs(nl), 95), "p_label": p_lab,
                         "null_kernel_mean": nk.mean(), "null_kernel_sd": nk.std(),
                         "null_kernel_q95": np.percentile(np.abs(nk), 95), "p_kernel": p_ker,
                         "sec": time.time() - t0})
            print(f"  {feat:2s}: r={r: .3f} label p={p_lab:.4f} (null sd {nl.std():.3f}) "
                  f"kernel p={p_ker:.4f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "results", "controls.csv"), index=False)
    print("\nDONE")


if __name__ == "__main__":
    main()
