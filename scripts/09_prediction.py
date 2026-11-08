"""Run G/E/G+E prediction under random vs leave-population-out CV."""
import os, sys, time, json, argparse, gc
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import (Panel, make_splits, tune_krr, krr_fit_predict,
                           scale_blocks, metrics, OUT, PROC)

ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0])
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]
LASSO_ALPHAS = np.logspace(-4, 0.5, 12)


def fit_lasso(Xtr, ytr, Xte, inner):
    m = LassoCV(cv=inner, alphas=30, eps=1e-3, max_iter=3000, tol=1e-3,
                n_jobs=1, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte), m.alpha_


def fit_hgb(Xtr, ytr, Xte, n):
    m = HistGradientBoostingRegressor(
        max_iter=300 if n >= 200 else 200, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=10 if n >= 200 else 5,
        l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def scale_pcs(PCg, PCe, tr, te):
    parts_tr, parts_te = [], []
    if PCg is not None:
        mu, sd = PCg[tr].mean(0), PCg[tr].std(0); sd[sd == 0] = 1
        parts_tr.append((PCg[tr] - mu) / sd); parts_te.append((PCg[te] - mu) / sd)
    if PCe is not None:
        mu, sd = PCe[tr].mean(0), PCe[tr].std(0); sd[sd == 0] = 1
        parts_tr.append((PCe[tr] - mu) / sd); parts_te.append((PCe[te] - mu) / sd)
    return np.hstack(parts_tr), np.hstack(parts_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", default="")
    ap.add_argument("--models", default="kridge,lasso,hgb")
    ap.add_argument("--features", default="G,E,GE")
    ap.add_argument("--cvs", default="random,lopo")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--fullpanel", type=int, default=0)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    models = args.models.split(","); features = args.features.split(",")
    cvs = args.cvs.split(",")
    odir = os.path.join(OUT, "results", args.tag)
    os.makedirs(odir, exist_ok=True)
    os.makedirs(os.path.join(odir, "oof"), exist_ok=True)

    P = Panel()
    traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    if args.traits:
        want = set(args.traits.split(","))
        traits = traits[traits.trait_id.isin(want)]
    print(f"running {len(traits)} traits, models={models}, features={features}, cvs={cvs}, fullpanel={args.fullpanel}", flush=True)

    for row in traits.itertuples():
        t0 = time.time()
        ids, y, cl = P.trait(row.study_id, row.trait, require_expr=not args.fullpanel)
        cl, mapping = P.merge_rare(ids, cl, min_n=10)
        n = len(ids)
        ig = [P.gidx[a] for a in ids]
        KG = P.KG[np.ix_(ig, ig)]
        PCg = P.PCG[ig]
        if args.fullpanel:
            Xg, Xe, KE, PCe = P.G[ig], None, None, None
            pg = Xg.shape[1]; pe = 0
        else:
            ie = [P.eidx[a] for a in ids]
            Xg, Xe = P.G[ig], P.E[ie]
            KE = P.KE[np.ix_(ie, ie)]; PCe = P.PCE[ie]
            pg = Xg.shape[1]; pe = Xe.shape[1]
        print(f"\n[{row.trait_id}] n={n} clusters={np.bincount(cl).tolist()} h2={row.h2:.2f} expr={not args.fullpanel}", flush=True)

        for cv in cvs:
            reps = args.repeats if cv == "random" else 1
            for rep in range(reps):
                splits = make_splits(cl, cv, seed=rep + 1)
                for model in models:
                    for feat in features:
                        if args.fullpanel and (feat != "G" or model != "kridge"):
                            continue
                        oy = np.full(n, np.nan); op = np.full(n, np.nan)
                        fold_rows = []
                        for k, (tr, te) in enumerate(splits):
                            if len(te) < 5:
                                continue
                            inner = make_splits(cl[tr], cv, seed=100 + rep, inner=True)
                            f0 = time.time()
                            if model == "kridge":
                                if feat == "G":
                                    _, a, _ = tune_krr(y, tr, inner, ALPHAS, K=KG)
                                    pred = krr_fit_predict(KG, y, tr, te, a)
                                elif feat == "E":
                                    _, a, _ = tune_krr(y, tr, inner, ALPHAS, K=KE)
                                    pred = krr_fit_predict(KE, y, tr, te, a)
                                else:
                                    w, a, _ = tune_krr(y, tr, inner, ALPHAS, KG=KG, KE=KE, wgrid=WGRID)
                                    Kw = w * KG + (1 - w) * KE
                                    pred = krr_fit_predict(Kw, y, tr, te, a)
                            elif model == "lasso":
                                if feat == "G":
                                    Xtr, Xte = scale_blocks(Xg, None, tr, te, pg, 1)
                                elif feat == "E":
                                    Xtr, Xte = scale_blocks(None, Xe, tr, te, 1, pe)
                                else:
                                    Xtr, Xte = scale_blocks(Xg, Xe, tr, te, pg, pe)
                                pred, a = fit_lasso(Xtr, y[tr], Xte, inner)
                            else:
                                if feat == "G":
                                    Xtr, Xte = scale_pcs(PCg, None, tr, te)
                                elif feat == "E":
                                    Xtr, Xte = scale_pcs(None, PCe, tr, te)
                                else:
                                    Xtr, Xte = scale_pcs(PCg, PCe, tr, te)
                                pred = fit_hgb(Xtr, y[tr], Xte, len(tr))
                            oy[te] = y[te]; op[te] = pred
                            r2f = metrics(y[te], pred)
                            fold_rows.append({"trait_id": row.trait_id, "model": model, "features": feat,
                                              "cv": cv, "rep": rep, "fold": k, "n_test": len(te),
                                              "r2_fold": r2f[0], "sec": time.time() - f0})
                        m = ~np.isnan(oy)
                        r2, pr, sr, mae = metrics(oy[m], op[m])
                        pd.DataFrame(fold_rows).to_csv(os.path.join(odir, "fold_metrics.csv"),
                                                       mode="a", header=not os.path.exists(os.path.join(odir, "fold_metrics.csv")), index=False)
                        rec = {"study_id": row.study_id, "trait_id": row.trait_id, "group": row.group,
                               "h2": row.h2, "n": n, "model": model, "features": feat, "cv": cv,
                               "rep": rep, "r2": r2, "pearson": pr, "spearman": sr, "mae": mae,
                               "r2_fold_mean": np.nanmean([x["r2_fold"] for x in fold_rows]),
                               "r2_fold_sd": np.nanstd([x["r2_fold"] for x in fold_rows])}
                        pd.DataFrame([rec]).to_csv(os.path.join(odir, "pooled_metrics.csv"), mode="a",
                                                   header=not os.path.exists(os.path.join(odir, "pooled_metrics.csv")), index=False)
                        np.savez_compressed(os.path.join(odir, "oof", f"{row.trait_id}__{model}__{feat}__{cv}__r{rep}.npz"),
                                            ids=np.array(ids), y=oy, yhat=op)
                        print(f"  {cv} r{rep} {model:6s} {feat:2s}: R2={r2: .3f}  ({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
