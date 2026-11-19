"""Genetic-resource description of the six groups and relatedness of test to training sets.

A1  Group composition, gene diversity, polymorphism and pairwise Hudson Fst (1,135 accessions).
A2  For every trait x held-out group (LOPO): Fst and genomic relationship between the held-out
    group and its training set, joined to within-group KRR accuracy; random-CV folds for contrast.
A3  Accession-level accuracy by tertile of maximum genomic relationship to the training set.
"""
import os, json
import grce_common as C
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from benchmark_lib import Panel, make_splits, PROC

OUT = os.path.join(C.RES, "diversity")
os.makedirs(OUT, exist_ok=True)
ORES = os.path.join(C.ORIG, "results")


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    from scipy.stats import rankdata, t as tdist
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = len(rx) - 3
    tt = r * np.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * tdist.sf(abs(tt), df))


def group_freqs(G, idx):
    return G[idx].mean(0).astype(np.float64)


def main():
    P = Panel()
    G = P.G
    KG = P.KG
    lab = pd.read_csv(os.path.join(ORES, "analysis", "pop_labels.csv"))
    lab = {r.cluster6: r.label.split(" (")[0] for r in lab.itertuples()}
    meta = pd.read_csv(os.path.join(PROC, "pop_clusters.csv"), dtype={"accession_id": str})
    meta = meta.set_index("accession_id").loc[P.acc_g]
    cl_all = np.array([P.pop[a] for a in P.acc_g])
    traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))

    panel = set()
    for row in traits.itertuples():
        ids, _, _ = P.trait(row.study_id, row.trait)
        panel.update(ids)
    in_panel = np.array([a in panel for a in P.acc_g])
    print("analysis panel", in_panel.sum(), flush=True)

    # ---- A1 ----
    groups = sorted(np.unique(cl_all))
    rows = []
    for g in groups:
        idx = np.flatnonzero(cl_all == g)
        n = len(idx)
        p = group_freqs(G, idx)
        he = float(np.mean(n / (n - 1) * 2 * p * (1 - p)))
        maf = np.minimum(p, 1 - p)
        rest = np.flatnonzero(cl_all != g)
        fst_rest = C.hudson_fst(p, group_freqs(G, rest), n, len(rest))
        sub = KG[np.ix_(idx, idx)]
        offd = sub[~np.eye(n, dtype=bool)]
        cc = meta.country.iloc[idx].value_counts()
        top = "; ".join(f"{c} {v}" for c, v in cc.head(3).items())
        rows.append({"cluster6": g, "Group": lab[g], "n (1,135)": n,
                     "n (analysis panel)": int(in_panel[idx].sum()),
                     "Countries (top 3)": top, "n countries": int(cc.size),
                     "Median latitude": float(np.nanmedian(meta.lat.iloc[idx])),
                     "Median longitude": float(np.nanmedian(meta.lon.iloc[idx])),
                     "Gene diversity": he, "Polymorphic markers (MAF>=0.05)": float(np.mean(maf >= 0.05)),
                     "Mean within-group relationship": float(offd.mean()),
                     "Fst vs rest": fst_rest})
    gtab = pd.DataFrame(rows)
    gtab.to_csv(os.path.join(OUT, "group_diversity.csv"), index=False)
    print(gtab.to_string(), flush=True)

    fst = np.zeros((len(groups), len(groups)))
    for i, a in enumerate(groups):
        for j, b in enumerate(groups):
            if j <= i:
                continue
            ia, ib = np.flatnonzero(cl_all == a), np.flatnonzero(cl_all == b)
            fst[i, j] = fst[j, i] = C.hudson_fst(group_freqs(G, ia), group_freqs(G, ib), len(ia), len(ib))
    names = [lab[g] for g in groups]
    pd.DataFrame(fst, index=names, columns=names).to_csv(os.path.join(OUT, "pairwise_fst.csv"))
    print(pd.DataFrame(fst, index=names, columns=names).round(3).to_string(), flush=True)

    # global differentiation: share of standardized-genotype variance between groups
    Zs = G - G.mean(0)
    sd = G.std(0); sd[sd == 0] = 1
    Zs = Zs / sd
    tot = float((Zs ** 2).sum())
    between = 0.0
    for g in groups:
        idx = cl_all == g
        between += idx.sum() * float((Zs[idx].mean(0) ** 2).sum())
    json.dump({"n_accessions": int(len(cl_all)), "n_markers": int(G.shape[1]),
               "analysis_panel": int(in_panel.sum()),
               "between_group_share_of_genotypic_variance": between / tot,
               "pairwise_fst_min": float(fst[np.triu_indices(len(groups), 1)].min()),
               "pairwise_fst_max": float(fst[np.triu_indices(len(groups), 1)].max()),
               "panel_gene_diversity": float(np.mean(2 * G[in_panel].mean(0) * (1 - G[in_panel].mean(0)))),
               "full_gene_diversity": float(np.mean(2 * G.mean(0) * (1 - G.mean(0))))},
              open(os.path.join(OUT, "diversity_summary.json"), "w"), indent=2)

    # ---- A2 + A3 ----
    unit_rows, fold_rows, acc_rows = [], [], []
    for row in traits.itertuples():
        ids, y, cl = P.trait(row.study_id, row.trait)
        cl, _ = P.merge_rare(ids, cl, min_n=10)
        ig = np.array([P.gidx[a] for a in ids])
        K = KG[np.ix_(ig, ig)]
        Gt = G[ig]
        for cv in ("random", "lopo"):
            splits = make_splits(cl, cv, seed=1)
            maxk = np.full(len(ids), np.nan)
            for k, (tr, te) in enumerate(splits):
                if len(te) < 5:
                    continue
                Kt = K[np.ix_(te, tr)]
                mk = Kt.max(1)
                maxk[te] = mk
                pte, ptr = Gt[te].mean(0).astype(np.float64), Gt[tr].mean(0).astype(np.float64)
                f = C.hudson_fst(pte, ptr, len(te), len(tr))
                fold_rows.append({"trait_id": row.trait_id, "cv": cv, "fold": k, "n_test": len(te),
                                  "n_train": len(tr), "Fst test-train": f,
                                  "Mean max relationship": float(mk.mean()),
                                  "Mean relationship": float(Kt.mean())})
                if cv == "lopo":
                    unit_rows.append({"trait_id": row.trait_id, "group": row.group, "h2": row.h2,
                                      "cluster6": int(np.unique(cl[te])[0]), "n_test": len(te),
                                      "Fst test-train": f, "Mean max relationship": float(mk.mean()),
                                      "Mean relationship": float(Kt.mean())})
            for feat in ("G", "GE"):
                d = np.load(os.path.join(ORES, "full", "oof", f"{row.trait_id}__kridge__{feat}__{cv}__r0.npz"))
                assert list(d["ids"]) == list(ids), row.trait_id
                m = np.isfinite(d["yhat"]) & np.isfinite(maxk)
                if m.sum() < 30:
                    continue
                q = pd.qcut(maxk[m], 3, labels=False, duplicates="drop")
                for t in np.unique(q):
                    s = q == t
                    yy, pp = d["y"][m][s], d["yhat"][m][s]
                    r = pearsonr(yy, pp)[0] if np.std(pp) > 1e-8 and s.sum() >= 10 else np.nan
                    acc_rows.append({"trait_id": row.trait_id, "cv": cv, "features": feat,
                                     "tertile": int(t) + 1, "n": int(s.sum()),
                                     "median max relationship": float(np.median(maxk[m][s])), "r": r})
        print(f"  {row.trait_id} done", flush=True)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(os.path.join(OUT, "fold_relatedness.csv"), index=False)
    units = pd.DataFrame(unit_rows)
    byp = pd.read_csv(os.path.join(ORES, "analysis", "lopo_by_population.csv"))
    wide = byp.pivot_table(index=["trait_id", "cluster6"], columns="features", values="r").reset_index()
    wide.columns = ["trait_id", "cluster6"] + [f"r within ({c})" for c in wide.columns[2:]]
    units = units.merge(wide, on=["trait_id", "cluster6"], how="left")
    units["Held-out group"] = units.cluster6.map(lab)
    units.to_csv(os.path.join(OUT, "lopo_units_relatedness.csv"), index=False)
    accs = pd.DataFrame(acc_rows)
    accs.to_csv(os.path.join(OUT, "accuracy_by_relationship_tertile.csv"), index=False)

    # summaries
    tr_mean = folds.groupby(["trait_id", "cv"]).apply(
        lambda s: np.average(s["Mean max relationship"], weights=s.n_test), include_groups=False).unstack()
    fst_mean = folds.groupby(["trait_id", "cv"]).apply(
        lambda s: np.average(s["Fst test-train"], weights=s.n_test), include_groups=False).unstack()
    summ = {"median_trait_max_relationship_random": float(tr_mean["random"].median()),
            "median_trait_max_relationship_lopo": float(tr_mean["lopo"].median()),
            "traits_lopo_lower_relationship": int((tr_mean["lopo"] < tr_mean["random"]).sum()),
            "median_trait_fst_random": float(fst_mean["random"].median()),
            "median_trait_fst_lopo": float(fst_mean["lopo"].median()),
            "n_traits": int(len(tr_mean))}
    cors = []
    for feat in ("G", "E", "GE"):
        col = f"r within ({feat})"
        u = units.dropna(subset=[col]).copy()
        u["rc"] = u[col] - u.groupby("trait_id")[col].transform("mean")
        for x in ("Mean max relationship", "Fst test-train", "n_test"):
            u["xc"] = u[x] - u.groupby("trait_id")[x].transform("mean")
            s1 = spearmanr(u[x], u[col])
            multi = u.groupby("trait_id").trait_id.transform("size") > 1
            s2 = spearmanr(u.loc[multi, "xc"], u.loc[multi, "rc"])
            pr = partial_spearman(u.loc[multi, "xc"], u.loc[multi, "rc"],
                                  np.log(u.loc[multi, "n_test"]) - u.loc[multi].groupby("trait_id").n_test.transform(lambda v: np.log(v).mean()))
            cors.append({"features": feat, "predictor": x, "n_units": len(u),
                         "spearman_rho": float(s1.statistic), "p": float(s1.pvalue),
                         "n_units_within_trait": int(multi.sum()),
                         "within_trait_rho": float(s2.statistic), "within_trait_p": float(s2.pvalue),
                         "within_trait_partial_rho_given_size": pr[0], "partial_p": pr[1]})
    cors = pd.DataFrame(cors)
    cors.to_csv(os.path.join(OUT, "relatedness_accuracy_correlations.csv"), index=False)
    print(cors.round(3).to_string(), flush=True)
    tert = accs.groupby(["cv", "features", "tertile"]).agg(
        n_traits=("r", "count"), median_r=("r", "median"),
        median_max_relationship=("median max relationship", "median")).reset_index()
    tert.to_csv(os.path.join(OUT, "tertile_summary.csv"), index=False)
    print(tert.round(3).to_string(), flush=True)
    json.dump(summ, open(os.path.join(OUT, "relatedness_summary.json"), "w"), indent=2)
    print(summ, flush=True)


if __name__ == "__main__":
    main()
