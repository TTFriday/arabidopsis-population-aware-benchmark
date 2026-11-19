"""Strict fold-internal preprocessing sensitivity for KRR (G, E, G+E), random and LOPO CV.

Inside every outer fold, using training accessions only:
  marker standardization and genomic kernel;
  expression gene filter (detection >= 0.2, mean > 1, variance > 0.01), z-scoring and kernel,
  starting from raw log2(normalized count + 1) values of GSE80744;
  phenotype winsorization (1st/99th percentiles) and standardization.
Test accessions are projected with training statistics; predictions are back-transformed to the
training phenotype scale and evaluated against raw replicate means.
LOPO inner tuning is grouped whenever the training set contains at least two genetic groups.
The MAF/LD marker set is phenotype-independent and kept as in the main analysis.
"""
import os, sys, time
import grce_common as C
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.model_selection import GroupKFold, KFold

OUT = os.path.join(C.RES, "strict_pipeline")
CACHE = os.path.join(C.WORK, "expr_log2_raw.npy")
_P = None
_R = None


def build_cache():
    acc_e = pd.read_csv(os.path.join(C.ORIG, "data_processed", "expr_accessions.csv"))["accession_id"].astype(str).values
    p = os.path.join(C.DATA, "03_transcriptome_1001T",
                     "GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz")
    df = pd.read_csv(p, sep="\t", index_col=0)
    df.columns = [c[1:] if c.startswith("X") else c for c in df.columns]
    df = df[list(acc_e)]
    X = np.log2(df.values.astype(np.float32) + 1.0).T
    np.save(CACHE, np.ascontiguousarray(X))
    print("cached raw expression", X.shape, flush=True)


def init():
    global _P, _R
    from benchmark_lib import Panel
    _P = Panel()
    _R = np.load(CACHE, mmap_mode="r")


def geno_kernel(X, tr):
    mu = X[tr].mean(0); sd = X[tr].std(0)
    keep = sd > 0
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    return (Z @ Z.T).astype(np.float64) / keep.sum(), int(keep.sum())


def expr_kernel(R, tr):
    Rt = R[tr]
    det = (Rt > 0).mean(0); mu = Rt.mean(0); var = Rt.var(0)
    keep = (det >= 0.2) & (mu > 1.0) & (var > 0.01)
    Z = (R[:, keep] - mu[keep]) / (np.sqrt(var[keep]) + 1e-6)
    return (Z @ Z.T).astype(np.float64) / keep.sum(), int(keep.sum())


def inner_splits(cl_tr, cv, seed):
    from benchmark_lib import make_splits
    if cv == "random":
        return make_splits(cl_tr, "random", seed=seed, inner=True), "random"
    u = np.unique(cl_tr)
    if len(u) >= 2:
        return list(GroupKFold(n_splits=min(3, len(u))).split(np.zeros(len(cl_tr)), groups=cl_tr)), "grouped"
    return list(KFold(3, shuffle=True, random_state=seed).split(np.zeros(len(cl_tr)))), "random"


def run_trait(args):
    from benchmark_lib import make_splits, tune_krr, krr_fit_predict
    sid, trait, trait_id, grp = args
    P, R = _P, _R
    ids, _, cl = P.trait(sid, trait)
    cl, _ = P.merge_rare(ids, cl, min_n=10)
    g = P.long[(P.long.study_id == sid) & (P.long.trait == trait)].groupby("accession_id")["value"].mean()
    yraw = g.loc[ids].values.astype(np.float64)
    ig = [P.gidx[a] for a in ids]; ie = [P.eidx[a] for a in ids]
    Xg = P.G[ig]; Re = np.asarray(R[ie], dtype=np.float32)
    rows, frows = [], []
    for cv in ("random", "lopo"):
        for rep in range(2 if cv == "random" else 1):
            splits = make_splits(cl, cv, seed=rep + 1)
            op = {f: np.full(len(ids), np.nan) for f in ("G", "E", "GE")}
            for k, (tr, te) in enumerate(splits):
                if len(te) < 5:
                    continue
                lo, hi = np.percentile(yraw[tr], [1, 99])
                yc = np.clip(yraw, lo, hi)
                mu, sd = yc[tr].mean(), yc[tr].std()
                ys = (yc - mu) / sd
                KG, pg = geno_kernel(Xg, tr)
                KE, pe = expr_kernel(Re, tr)
                inner, itype = inner_splits(cl[tr], cv, 100 + rep)
                _, ag, _ = tune_krr(ys, tr, inner, C.ALPHAS, K=KG)
                op["G"][te] = mu + sd * krr_fit_predict(KG, ys, tr, te, ag)
                _, ae, _ = tune_krr(ys, tr, inner, C.ALPHAS, K=KE)
                op["E"][te] = mu + sd * krr_fit_predict(KE, ys, tr, te, ae)
                w, a, _ = tune_krr(ys, tr, inner, C.ALPHAS, KG=KG, KE=KE, wgrid=C.WGRID)
                op["GE"][te] = mu + sd * krr_fit_predict(w * KG + (1 - w) * KE, ys, tr, te, a)
                frows.append({"trait_id": trait_id, "cv": cv, "rep": rep, "fold": k, "n_train": len(tr),
                              "n_test": len(te), "markers_used": pg, "genes_used": pe,
                              "inner_tuning": itype, "w_GE": w})
            for f, pred in op.items():
                m = np.isfinite(pred)
                rows.append({"trait_id": trait_id, "group": grp, "features": f, "cv": cv, "rep": rep,
                             "n": int(m.sum()),
                             "r": float(pearsonr(yraw[m], pred[m])[0]) if np.std(pred[m]) > 1e-8 else np.nan,
                             "spearman": float(spearmanr(yraw[m], pred[m])[0]) if np.std(pred[m]) > 1e-8 else np.nan})
    return rows, frows


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(CACHE):
        build_cache()
    from benchmark_lib import PROC
    traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    jobs = [(r.study_id, r.trait, r.trait_id, r.group) for r in traits.itertuples()]
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time(); rows, frows = [], []
    with Pool(nproc, initializer=init) as pool:
        for i, (r, f) in enumerate(pool.imap_unordered(run_trait, jobs)):
            rows.extend(r); frows.extend(f)
            print(f"  {i+1}/{len(jobs)} {r[0]['trait_id']} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); fd = pd.DataFrame(frows)
    df.to_csv(os.path.join(OUT, "strict_per_trait_raw.csv"), index=False)
    fd.to_csv(os.path.join(OUT, "strict_folds.csv"), index=False)

    avg = df.groupby(["trait_id", "group", "features", "cv"], as_index=False)[["r", "spearman"]].mean()
    avg.to_csv(os.path.join(OUT, "strict_per_trait.csv"), index=False)
    base = pd.read_csv(os.path.join(C.ORIG, "results", "analysis", "per_trait.csv"))
    rows = []
    for (f, cv), sub in avg.groupby(["features", "cv"]):
        b = base.set_index("trait_id").loc[sub.trait_id, f"kridge__{f}__{cv}"].values
        d = sub.r.values - b
        rows.append({"features": f, "cv": cv, "n_traits": len(sub),
                     "median_r_strict": float(np.median(sub.r)), "median_r_main": float(np.median(b)),
                     "median_diff": float(np.median(d)),
                     "spearman_strict_vs_main": float(spearmanr(sub.r, b)[0]),
                     "wilcoxon_p_diff": float(wilcoxon(d).pvalue)})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(OUT, "strict_vs_main.csv"), index=False)
    print(comp.round(3).to_string(), flush=True)

    wide = avg.pivot_table(index="trait_id", columns=["features", "cv"], values="r")
    tests = []
    for cv in ("random", "lopo"):
        for a, b in (("GE", "G"), ("E", "G")):
            d = (wide[(a, cv)] - wide[(b, cv)]).dropna()
            tests.append({"comparison": f"{a}-{b}", "cv": cv, "n_traits": len(d),
                          "median_diff": float(d.median()), "wilcoxon_p": float(wilcoxon(d).pvalue)})
    for f in ("G", "E", "GE"):
        d = (wide[(f, "random")] - wide[(f, "lopo")]).dropna()
        tests.append({"comparison": f"{f} random-LOPO gap", "cv": "both", "n_traits": len(d),
                      "median_diff": float(d.median()), "wilcoxon_p": float(wilcoxon(d).pvalue)})
    t = pd.DataFrame(tests)
    t["holm_p"] = C.holm(t.wilcoxon_p.values)
    t.to_csv(os.path.join(OUT, "strict_tests.csv"), index=False)
    print(t.round(4).to_string(), flush=True)
    lf = fd[fd.cv == "lopo"]
    print("LOPO folds with grouped inner tuning:", int((lf.inner_tuning == "grouped").sum()), "/", len(lf),
          "; traits all grouped:", int(lf.groupby("trait_id").inner_tuning.apply(lambda s: (s == "grouped").all()).sum()),
          flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
