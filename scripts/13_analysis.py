"""Statistical synthesis of the prediction benchmark.

Reads results/<tag>/pooled_metrics.csv and OOF predictions, produces paired
tests, bootstrap CIs, population-level portability, and summary tables used by
the figures and manuscript.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import Panel, OUT, PROC

RES = os.path.join(OUT, "results")
ANA = os.path.join(RES, "analysis")
os.makedirs(ANA, exist_ok=True)
RNG = np.random.default_rng(20261027)


def load_pooled(tag):
    p = os.path.join(RES, tag, "pooled_metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df.drop_duplicates(["trait_id", "model", "features", "cv", "rep"], keep="last")


def boot_ci(x, nboot=20000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(nboot, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, label, group):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) < 5:
        return {"comparison": label, "subset": group, "n_traits": int(len(d)),
                "median_diff": np.nan, "mean_diff": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p_wilcoxon": np.nan, "frac_positive": np.nan}
    try:
        p = float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        p = np.nan
    lo, hi = boot_ci(d)
    return {"comparison": label, "subset": group, "n_traits": int(len(d)),
            "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
            "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p,
            "frac_positive": float((d > 0).mean())}


def wide(pool):
    agg = (pool.groupby(["trait_id", "model", "features", "cv"])
           .agg(r=("pearson", "mean"), r_sd=("pearson", "std"),
                r2=("r2", "mean"), rho=("spearman", "mean"), mae=("mae", "mean"),
                n_rep=("pearson", "size")).reset_index())
    agg["col"] = agg.model + "|" + agg.features + "|" + agg.cv
    w = agg.pivot_table(index="trait_id", columns="col", values="r")
    w.columns = [c.replace("|", "__") for c in w.columns]
    return w.reset_index(), agg


def pop_labels(P):
    pop = pd.read_csv(os.path.join(PROC, "pop_clusters.csv"), dtype={"accession_id": str})
    abbr = {"GER": "GER", "Czech Republic": "CZE", "Spain": "ESP", "Sweden": "SWE",
            "Russia": "RUS", "Armenia": "ARM", "Finland": "FIN", "UK": "UK",
            "US": "US", "Italy": "ITA", "France": "FRA", "Kyrgyzstan": "KGZ",
            "Uzbekistan": "UZB", "Azerbaijan": "AZE", "Greece": "GRC"}
    lab = {}
    for c, g in pop.groupby("cluster6"):
        top = g.country.fillna("NA").value_counts().head(2)
        codes = [abbr.get(k, k) for k in top.index]
        lab[int(c)] = "/".join(codes) + f" (n={len(g)})"
    out = pd.DataFrame({"cluster6": sorted(lab), "label": [lab[k] for k in sorted(lab)]})
    out.to_csv(os.path.join(ANA, "pop_labels.csv"), index=False)
    return lab


def lopo_by_population(pool, P):
    """Within-population predictive accuracy of each held-out population."""
    meta = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    rows = []
    for row in meta.itertuples():
        ids, y, cl = P.trait(row.study_id, row.trait, require_expr=True)
        cl, _ = P.merge_rare(ids, cl, min_n=10)
        for feat in ["G", "E", "GE"]:
            f = os.path.join(RES, "full", "oof", f"{row.trait_id}__kridge__{feat}__lopo__r0.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            assert list(d["ids"]) == list(ids), row.trait_id
            yhat = d["yhat"]
            for c in np.unique(cl):
                m = (cl == c) & np.isfinite(yhat)
                if m.sum() < 15:
                    continue
                r = np.corrcoef(y[m], yhat[m])[0, 1] if np.std(yhat[m]) > 1e-8 else np.nan
                rows.append({"trait_id": row.trait_id, "group": row.group, "features": feat,
                             "cluster6": int(c), "n_pop": int(m.sum()), "r": r})
    return pd.DataFrame(rows)


def main():
    pool = load_pooled("full")
    if pool is None:
        raise SystemExit("results/full/pooled_metrics.csv not found - finish 09_prediction first")
    w, agg = wide(pool)
    meta = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))[
        ["trait_id", "group", "h2", "n", "n_expr"]]
    per = meta.merge(w, on="trait_id", how="left")
    per.to_csv(os.path.join(ANA, "per_trait.csv"), index=False)

    combos = sorted(agg.col.unique())
    summ = []
    for col in combos:
        r = per[col.replace("|", "__")].dropna().values
        if len(r) == 0:
            continue
        model, feat, cv = col.split("|")
        summ.append({"model": model, "features": feat, "cv": cv, "n_traits": len(r),
                     "mean_r": r.mean(), "median_r": np.median(r), "sd_r": r.std(ddof=1) if len(r) > 1 else np.nan,
                     "q25": np.percentile(r, 25), "q75": np.percentile(r, 75),
                     "min_r": r.min(), "max_r": r.max(),
                     "frac_r_gt_0.1": (r > 0.1).mean(), "frac_r_gt_0.3": (r > 0.3).mean(),
                     "frac_r_gt_0.5": (r > 0.5).mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(ANA, "summary_by_combo.csv"), index=False)

    def col(model, feat, cv):
        return per[f"{model}__{feat}__{cv}"].values

    tests = []
    for cv in ["random", "lopo"]:
        tests.append(paired(col("kridge", "GE", cv), col("kridge", "G", cv), "GE_vs_G (kridge)", cv))
        tests.append(paired(col("kridge", "E", cv), col("kridge", "G", cv), "E_vs_G (kridge)", cv))
        tests.append(paired(col("lasso", "GE", cv), col("lasso", "G", cv), "GE_vs_G (lasso)", cv))
        tests.append(paired(col("lasso", "E", cv), col("lasso", "G", cv), "E_vs_G (lasso)", cv))
        tests.append(paired(col("hgb", "GE", cv), col("hgb", "G", cv), "GE_vs_G (hgb)", cv))
        tests.append(paired(col("hgb", "E", cv), col("hgb", "G", cv), "E_vs_G (hgb)", cv))
        tests.append(paired(col("kridge", "GE", cv), col("lasso", "GE", cv), "kridge_vs_lasso (GE)", cv))
        tests.append(paired(col("kridge", "GE", cv), col("hgb", "GE", cv), "kridge_vs_hgb (GE)", cv))
        tests.append(paired(col("lasso", "GE", cv), col("hgb", "GE", cv), "lasso_vs_hgb (GE)", cv))
    for feat in ["G", "E", "GE"]:
        for model in ["kridge", "lasso", "hgb"]:
            tests.append(paired(col(model, feat, "random"), col(model, feat, "lopo"),
                                f"random_minus_lopo ({model} {feat})", "portability_gap"))
    for model in ["kridge", "lasso", "hgb"]:
        gap_G = col(model, "G", "random") - col(model, "G", "lopo")
        gap_E = col(model, "E", "random") - col(model, "E", "lopo")
        gap_GE = col(model, "GE", "random") - col(model, "GE", "lopo")
        tests.append(paired(gap_GE, gap_G, f"gap_GE_minus_gap_G ({model})", "portability_gap"))
        tests.append(paired(gap_E, gap_G, f"gap_E_minus_gap_G ({model})", "portability_gap"))
    tests = pd.DataFrame(tests)
    tests["family"] = np.where(tests.subset.eq("portability_gap"), "portability", "within_cv")
    tests["p_holm"] = np.nan
    for fam, idx in tests.groupby("family").groups.items():
        sub = tests.loc[idx]
        ok = sub.p_wilcoxon.notna()
        order = sub.loc[ok].p_wilcoxon.sort_values().index
        m = int(ok.sum()); running = 0.0
        for rank, i in enumerate(order):
            val = min(1.0, (m - rank) * tests.loc[i, "p_wilcoxon"])
            running = max(running, val)
            tests.loc[i, "p_holm"] = running
    tests.to_csv(os.path.join(ANA, "tests.csv"), index=False)

    h2rows = []
    for colname in combos:
        r = per[colname.replace("|", "__")].values
        m = np.isfinite(r) & np.isfinite(per.h2.values)
        rho, p = spearmanr(per.h2.values[m], r[m])
        model, feat, cv = colname.split("|")
        h2rows.append({"model": model, "features": feat, "cv": cv, "n_traits": int(m.sum()),
                       "spearman_h2_r": rho, "p": p})
    pd.DataFrame(h2rows).to_csv(os.path.join(ANA, "h2_correlation.csv"), index=False)

    # kernel weight: does the model lean on expression when populations are held out?
    weight_test = None
    p = os.path.join(RES, "kernel_weights.csv")
    if os.path.exists(p):
        kw = pd.read_csv(p)
        kwm = kw.groupby(["trait_id", "cv"]).w.median().unstack()
        kwm["delta_w"] = kwm["random"] - kwm["lopo"]
        kwm.to_csv(os.path.join(ANA, "kernel_weights_by_trait.csv"))
        weight_test = paired(kwm["random"].values, kwm["lopo"].values,
                             "kernel_weight (random $-$ LOPO)", "expression_weight")
        pd.DataFrame([weight_test]).to_csv(os.path.join(ANA, "kernel_weight_test.csv"), index=False)

    group_rows = []
    for g, sub in per.groupby("group"):
        rec = {"group": g, "n_traits": len(sub), "median_h2": sub.h2.median()}
        for model, feat, cv in [("kridge", "G", "random"), ("kridge", "E", "random"),
                                ("kridge", "GE", "random"), ("kridge", "G", "lopo"),
                                ("kridge", "E", "lopo"), ("kridge", "GE", "lopo")]:
            rec[f"{model}_{feat}_{cv}"] = sub[f"{model}__{feat}__{cv}"].median()
        group_rows.append(rec)
    pd.DataFrame(group_rows).to_csv(os.path.join(ANA, "group_summary.csv"), index=False)

    # best feature set per trait under each CV
    best_rows = []
    for row in per.itertuples():
        rec = {"trait_id": row.trait_id}
        for cv in ["random", "lopo"]:
            vals = {f: getattr(row, f"kridge__{f}__{cv}") for f in ["G", "E", "GE"]}
            finite = {k: v for k, v in vals.items() if np.isfinite(v)}
            rec[f"best_{cv}"] = max(finite, key=finite.get) if finite else None
            rec[f"best_r_{cv}"] = max(finite.values()) if finite else np.nan
        best_rows.append(rec)
    pd.DataFrame(best_rows).to_csv(os.path.join(ANA, "best_feature_per_trait.csv"), index=False)

    # leave-population-out accuracy measured within each held-out population
    P = Panel()
    lab = pop_labels(P)
    bypop = lopo_by_population(pool, P)
    if len(bypop):
        bypop["population"] = bypop.cluster6.map(lab)
        bypop.to_csv(os.path.join(ANA, "lopo_by_population.csv"), index=False)
        popsum = (bypop.groupby(["features", "population"])
                  .agg(n_traits=("r", "size"), median_r=("r", "median"),
                       mean_r=("r", "mean")).reset_index())
        popsum.to_csv(os.path.join(ANA, "lopo_population_summary.csv"), index=False)

    # full-panel genotype sensitivity (expression panel vs all genotyped accessions)
    fp = load_pooled("fullpanel")
    fp_rows = []
    if fp is not None:
        fpa = (fp.groupby(["trait_id", "features", "cv"]).pearson.mean().reset_index())
        fpw = fpa[fpa.cv.isin(["random", "lopo"])].pivot_table(
            index="trait_id", columns="cv", values="pearson")
        fpw.columns = [f"fullpanel_{c}" for c in fpw.columns]
        fpw = fpw.reset_index()
        cmp = meta.merge(fpw, on="trait_id")
        cmp = cmp.merge(per[["trait_id", "kridge__G__random", "kridge__G__lopo"]], on="trait_id")
        cmp["delta_r_random"] = cmp["kridge__G__random"] - cmp["fullpanel_random"]
        cmp["delta_r_lopo"] = cmp["kridge__G__lopo"] - cmp["fullpanel_lopo"]
        cmp.to_csv(os.path.join(ANA, "fullpanel_vs_exprpanel.csv"), index=False)
        for c, lab_ in [("delta_r_random", "expr_panel_minus_fullpanel (G random)"),
                        ("delta_r_lopo", "expr_panel_minus_fullpanel (G lopo)")]:
            fp_rows.append(paired(cmp["kridge__G__random" if "random" in c else "kridge__G__lopo"],
                                  cmp["fullpanel_random" if "random" in c else "fullpanel_lopo"],
                                  lab_, "panel_sensitivity"))
        pd.DataFrame(fp_rows).to_csv(os.path.join(ANA, "fullpanel_tests.csv"), index=False)

    # variance decomposition: mixed model with trait random intercept
    try:
        import statsmodels.formula.api as smf
        long = []
        for row in per.itertuples():
            for feat in ["G", "E", "GE"]:
                for cv in ["random", "lopo"]:
                    long.append({"trait_id": row.trait_id, "group": row.group, "h2": row.h2,
                                 "features": feat, "cv": cv,
                                 "r": getattr(row, f"kridge__{feat}__{cv}")})
        ld = pd.DataFrame(long).dropna(subset=["r"])
        m = smf.mixedlm("r ~ C(features, Treatment('G')) * C(cv, Treatment('random')) + h2",
                        ld, groups=ld["trait_id"]).fit(reml=True)
        coef = pd.DataFrame({"term": m.params.index, "estimate": m.params.values,
                             "se": m.bse.values, "p": m.pvalues.values})
        coef.to_csv(os.path.join(ANA, "mixed_model.csv"), index=False)
        with open(os.path.join(ANA, "mixed_model_summary.txt"), "w") as f:
            f.write(m.summary().as_text())
    except Exception as e:
        print("mixed model skipped:", e)

    # cross-references made by the other scripts
    for src, name, keys in [
        ("controls.csv", "controls_summary", ["metric", "mean", "median", "min", "max", "n"]),
        ("transfer.csv", "transfer_summary", ["model", "features", "mean_r", "median_r", "n"]),
        ("enrichment.csv", "enrichment_summary", ["features", "score", "mean_fold", "min_p", "n"]),
    ]:
        p = os.path.join(RES, src)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if src == "controls.csv":
            out = df.groupby("features").agg(
                n=("trait", "size"), mean_real_r=("real_r", "mean"),
                mean_p_label=("p_label", "mean"), mean_p_kernel=("p_kernel", "mean"),
                frac_p_label_lt_005=("p_label", lambda s: (s < 0.05).mean())).reset_index()
        elif src == "transfer.csv":
            out = (df[df.source != "within"].groupby(["model", "features"])
                   .agg(n=("r", "size"), mean_r=("r", "mean"), median_r=("r", "median"),
                        min_p=("perm_p", "min")).reset_index())
        else:
            out = (df.groupby(["features", "score"]).agg(
                n=("p", "size"), mean_fold=("fold_enrich", "mean"), min_p=("p", "min"))
                .reset_index())
        out.to_csv(os.path.join(ANA, f"{name}.csv"), index=False)

    # does random CV rank traits/models the same way as LOPO?
    rc_rows = []
    for model, feat in [("kridge", "G"), ("kridge", "E"), ("kridge", "GE"),
                        ("lasso", "G"), ("lasso", "E"), ("lasso", "GE"),
                        ("hgb", "G"), ("hgb", "E"), ("hgb", "GE")]:
        a = per[f"{model}__{feat}__random"].values
        b = per[f"{model}__{feat}__lopo"].values
        m = np.isfinite(a) & np.isfinite(b)
        rho, p = spearmanr(a[m], b[m])
        rc_rows.append({"model": model, "features": feat, "n_traits": int(m.sum()),
                        "spearman_random_vs_lopo": rho, "p": p})
    pd.DataFrame(rc_rows).to_csv(os.path.join(ANA, "rank_consistency.csv"), index=False)

    # agreement of the best model (G+E) between CV schemes
    agree_rows = []
    for cv in ["random", "lopo"]:
        sub = per[[f"{m}__GE__{cv}" for m in ["kridge", "lasso", "hgb"]]]
        best = sub.idxmax(axis=1).str.split("__").str[0]
        per[f"best_model_{cv}"] = best
        agree_rows.append({"cv": cv, "n": int(best.notna().sum()),
                           "kridge": int((best == "kridge").sum()),
                           "lasso": int((best == "lasso").sum()),
                           "hgb": int((best == "hgb").sum())})
    both = per.dropna(subset=["best_model_random", "best_model_lopo"])
    pd.DataFrame(agree_rows).to_csv(os.path.join(ANA, "best_model_counts.csv"), index=False)
    model_agreement = float((both.best_model_random == both.best_model_lopo).mean()) if len(both) else np.nan

    headline = {
        "n_traits": int(per.shape[0]),
        "best_model_agreement_random_vs_lopo": model_agreement,
        "median_r": {f"{m}|{f}|{cv}": v for (m, f, cv), v in
                     summ.set_index(["model", "features", "cv"]).median_r.items()},
        "frac_best_feature_random": pd.read_csv(os.path.join(ANA, "best_feature_per_trait.csv"))
            .best_random.value_counts(normalize=True).to_dict(),
        "frac_best_feature_lopo": pd.read_csv(os.path.join(ANA, "best_feature_per_trait.csv"))
            .best_lopo.value_counts(normalize=True).to_dict(),
        "tests": {f"{r.comparison} | {r.subset}": {k: (None if pd.isna(v) else v)
                                                    for k, v in r._asdict().items() if k not in ("Index",)}
                  for r in tests.itertuples()},
        "h2_spearman": {f"{m}|{f}|{cv}": v for (m, f, cv), v in
                        pd.DataFrame(h2rows).set_index(["model", "features", "cv"]).spearman_h2_r.items()},
        "kernel_weight_test": weight_test,
    }
    with open(os.path.join(ANA, "main_stats.json"), "w") as f:
        json.dump(headline, f, indent=2, default=float)
    print(summ.to_string(index=False))
    print("\n", tests.round(4).to_string(index=False))
    print("\nsaved to", ANA)


if __name__ == "__main__":
    main()
