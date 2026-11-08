"""Sensitivity analyses for the population-aware benchmark.

A. Metric robustness: Spearman/R2/MAE summaries; GE-vs-G tests on Spearman (Holm, 6 tests).
B. Domain-stratified accuracy and portability gaps (KRR).
C. LOPO cluster-definition sensitivity: k = 4, 5, 8 vs the primary k = 6 (KRR G/E/G+E).
D. Expression residualized on the top 10 genotype PCs (KRR E and G+E), random and LOPO.
   The original expression kernel restricted to the same accessions is the exact sham,
   because residualization keeps the original per-gene z-scores and kernel divisor.
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import (Panel, make_splits, tune_krr, krr_fit_predict, OUT, PROC)

RES = os.path.join(OUT, "results")
SENS = os.path.join(RES, "sensitivity")
os.makedirs(SENS, exist_ok=True)
ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0])
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, p[idx] * (len(p) - rank))
        adj[idx] = min(run, 1.0)
    return adj


def krr_lopo_or_random(y, cl, KG, KE, feat, cv, rep):
    if cv == "lopo" and len(np.unique(cl)) < 2:
        raise ValueError("Outer LOPO requires at least two clusters after rare-cluster merging")
    splits = make_splits(cl, cv, seed=rep + 1)
    oy = np.full(len(y), np.nan); op = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(te) < 5:
            continue
        inner = make_splits(cl[tr], cv, seed=100 + rep, inner=True)
        if feat == "G":
            _, a, _ = tune_krr(y, tr, inner, ALPHAS, K=KG)
            pred = krr_fit_predict(KG, y, tr, te, a)
        elif feat == "E":
            _, a, _ = tune_krr(y, tr, inner, ALPHAS, K=KE)
            pred = krr_fit_predict(KE, y, tr, te, a)
        else:
            w, a, _ = tune_krr(y, tr, inner, ALPHAS, KG=KG, KE=KE, wgrid=WGRID)
            pred = krr_fit_predict(w * KG + (1 - w) * KE, y, tr, te, a)
        oy[te] = y[te]; op[te] = pred
    m = ~np.isnan(oy)
    return float(pearsonr(oy[m], op[m])[0]) if np.std(op[m]) > 1e-8 else np.nan


def part_a():
    pm = pd.read_csv(os.path.join(RES, "full", "pooled_metrics.csv"))
    pm = pm.drop_duplicates(["trait_id", "model", "features", "cv", "rep"], keep="last")
    g = pm.groupby(["trait_id", "group", "model", "features", "cv"], as_index=False)[
        ["pearson", "spearman", "r2", "mae"]].mean()
    rows = []
    for (model, feat, cv), sub in g.groupby(["model", "features", "cv"]):
        rows.append({"Model": model, "Features": feat, "CV": cv, "n_traits": len(sub),
                     "Median Pearson r": sub.pearson.median(), "Median Spearman rho": sub.spearman.median(),
                     "Median R2": sub.r2.median(), "Median MAE": sub.mae.median(),
                     "Mean Spearman rho": sub.spearman.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(SENS, "metric_summary.csv"), index=False)

    wide = g.pivot_table(index=["trait_id", "model", "cv"], columns="features", values="spearman")
    tests = []
    for (model, cv), sub in wide.groupby(["model", "cv"]):
        d = (sub["GE"] - sub["G"]).dropna()
        w = wilcoxon(d)
        tests.append({"Model": model, "CV": cv, "n_traits": len(d),
                      "Median delta Spearman": float(d.median()), "Wilcoxon P": float(w.pvalue)})
    t = pd.DataFrame(tests)
    t["Holm P"] = holm(t["Wilcoxon P"].values)
    t.to_csv(os.path.join(SENS, "metric_spearman_tests.csv"), index=False)
    print("A done", flush=True)


def part_b():
    p = pd.read_csv(os.path.join(RES, "analysis", "per_trait.csv"))
    rows = []
    for dom, sub in p.groupby("group"):
        rows.append({"Domain": dom, "n_traits": len(sub),
                     "KRR G random": sub.kridge__G__random.median(),
                     "KRR G+E random": sub.kridge__GE__random.median(),
                     "KRR E random": sub.kridge__E__random.median(),
                     "KRR G LOPO": sub.kridge__G__lopo.median(),
                     "KRR G+E LOPO": sub.kridge__GE__lopo.median(),
                     "KRR E LOPO": sub.kridge__E__lopo.median(),
                     "Median gap G": (sub.kridge__G__random - sub.kridge__G__lopo).median(),
                     "Median gap G+E": (sub.kridge__GE__random - sub.kridge__GE__lopo).median()})
    pd.DataFrame(rows).to_csv(os.path.join(SENS, "domain_summary.csv"), index=False)
    print("B done", flush=True)


def summarize_clusters(df, baseline):
    """Summarize valid outer LOPO fits against the same traits' random-CV scores."""
    if df.duplicated(["k", "trait_id", "Features"]).any():
        raise ValueError("Cluster sensitivity requires one row per k/trait/feature")
    baseline = baseline.set_index("trait_id", verify_integrity=True)
    rows = []
    for (k, feat), sub in df.groupby(["k", "Features"]):
        valid = sub.loc[(sub.n_clusters >= 2) & np.isfinite(sub["LOPO r"])]
        lopo = valid["LOPO r"].to_numpy()
        random = baseline.loc[valid.trait_id, f"kridge__{feat}__random"].to_numpy()
        if not np.isfinite(random).all():
            raise ValueError(f"Non-finite matched random baseline for k={k}, feature={feat}")
        nvalid = len(valid)
        rows.append({"k": k, "Features": feat, "n traits": nvalid,
                     "Excluded traits": 27 - nvalid,
                     "Median LOPO r": float(np.median(lopo)) if nvalid else np.nan,
                     "Matched random r": float(np.median(random)) if nvalid else np.nan,
                     "Median paired gap": float(np.median(random - lopo)) if nvalid else np.nan})
    return pd.DataFrame(rows, columns=["k", "Features", "n traits", "Excluded traits",
                                       "Median LOPO r", "Matched random r", "Median paired gap"])


def part_c(P):
    pcs = pd.read_csv(os.path.join(PROC, "PCs.csv"), dtype={"accession_id": str})
    Xs = StandardScaler().fit_transform(pcs[[f"PC{i+1}" for i in range(10)]])
    accs = pcs.accession_id.values
    clus = {}
    for k in (4, 5):
        lab = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(Xs)
        clus[k] = dict(zip(accs, lab))
        print(f"k={k} full-panel sizes {np.bincount(lab).tolist()}", flush=True)
    pc8 = pd.read_csv(os.path.join(PROC, "pop_clusters.csv"), dtype={"accession_id": str})
    clus[8] = dict(zip(pc8.accession_id, pc8.cluster8))

    traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    rows = []
    t0 = time.time()
    for k in (4, 5, 8):
        for row in traits.itertuples():
            ids, y, _ = P.trait(row.study_id, row.trait)
            cl = np.array([clus[k][a] for a in ids])
            cl, _ = P.merge_rare(ids, cl, min_n=10)
            n_clusters = len(np.unique(cl))
            if n_clusters >= 2:
                ig = [P.gidx[a] for a in ids]; ie = [P.eidx[a] for a in ids]
                KG = P.KG[np.ix_(ig, ig)]; KE = P.KE[np.ix_(ie, ie)]
            for feat in ("G", "E", "GE"):
                r = (krr_lopo_or_random(y, cl, KG, KE, feat, "lopo", 0)
                     if n_clusters >= 2 else np.nan)
                rows.append({"k": k, "trait_id": row.trait_id, "group": row.group,
                             "Features": feat, "n": len(ids), "n_clusters": n_clusters,
                             "LOPO r": r})
            pd.DataFrame(rows).to_csv(os.path.join(SENS, "cluster_per_trait.csv"), index=False)
        print(f"k={k} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    baseline = pd.read_csv(os.path.join(RES, "analysis", "per_trait.csv"), float_precision="round_trip")
    summ = summarize_clusters(df, baseline)
    summ.to_csv(os.path.join(SENS, "cluster_summary.csv"), index=False)
    print("C done", flush=True)


def part_d(P):
    shared = [a for a in P.acc_e if a in P.gset]
    ie = [P.eidx[a] for a in shared]; ig = [P.gidx[a] for a in shared]
    E = P.E[ie].astype(np.float64)
    PC = np.column_stack([np.ones(len(shared)), P.PCG[ig][:, :10].astype(np.float64)])
    coef, _, _, _ = np.linalg.lstsq(PC, E, rcond=None)
    Eres = E - PC @ coef
    KEres = (Eres @ Eres.T) / E.shape[1]
    var_removed = 1.0 - float((Eres ** 2).sum() / (E ** 2).sum())
    per_gene = 1.0 - (Eres ** 2).sum(0) / np.maximum((E ** 2).sum(0), 1e-12)
    json.dump({"n_shared": len(shared), "variance_removed_overall": var_removed,
               "variance_removed_median_gene": float(np.median(per_gene))},
              open(os.path.join(SENS, "residualization_info.json"), "w"), indent=2)
    print(f"residualized {len(shared)} accessions; variance removed overall {var_removed:.3f}, "
          f"median gene {np.median(per_gene):.3f}", flush=True)
    sidx = {a: i for i, a in enumerate(shared)}

    traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    rows = []
    t0 = time.time()
    for row in traits.itertuples():
        ids, y, cl = P.trait(row.study_id, row.trait)
        cl, _ = P.merge_rare(ids, cl, min_n=10)
        ig = [P.gidx[a] for a in ids]; ii = [sidx[a] for a in ids]
        KG = P.KG[np.ix_(ig, ig)]; KR = KEres[np.ix_(ii, ii)]
        for cv in ("random", "lopo"):
            reps = 2 if cv == "random" else 1
            for rep in range(reps):
                for feat in ("E", "GE"):
                    r = krr_lopo_or_random(y, cl, KG, KR, feat, cv, rep)
                    rows.append({"trait_id": row.trait_id, "group": row.group, "Features": feat,
                                 "CV": cv, "rep": rep, "n": len(ids), "r": r})
        pd.DataFrame(rows).to_csv(os.path.join(SENS, "expr_residual_per_trait.csv"), index=False)
        print(f"  [{row.trait_id}] done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    avg = df.groupby(["trait_id", "Features", "CV"], as_index=False)["r"].mean()
    avg.to_csv(os.path.join(SENS, "expr_residual_averaged.csv"), index=False)
    summ = avg.groupby(["Features", "CV"])["r"].median().reset_index()
    summ.to_csv(os.path.join(SENS, "expr_residual_summary.csv"), index=False)
    print("D done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B,C,D")
    args = ap.parse_args()
    parts = args.parts.split(",")
    P = Panel() if ("C" in parts or "D" in parts) else None
    if "A" in parts: part_a()
    if "B" in parts: part_b()
    if "C" in parts: part_c(P)
    if "D" in parts: part_d(P)


if __name__ == "__main__":
    main()