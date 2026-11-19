"""Target-group augmentation: how many phenotyped accessions of a new genetic group are needed?

For each trait and each (merged) genetic group with >= 20 accessions, the group is split at random
into an evaluation half and a candidate pool. Training uses all other groups plus 0, 10, 25 or 50%
of the group (nested draws from the pool). A size-matched design removes the same number of
other-group accessions so that training size stays constant; units unable to replace the full
candidate pool are excluded from that design. Training-only phenotype clipping is used, and
predictions are back-transformed for scoring against raw evaluation outcomes. Absolute split
bias and RMSE are expressed relative to the fixed full-trait raw standard deviation.
"""
import os, sys, time, zlib
import grce_common as C
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import pearsonr, wilcoxon
from sklearn.model_selection import KFold

OUT = os.path.join(C.RES, "augmentation")
FRACS = [0.0, 0.10, 0.25, 0.50]
REPS = 10
MIN_GROUP = 20
_P = None


def init():
    global _P
    from benchmark_lib import Panel
    _P = Panel()


def run_unit(args):
    from benchmark_lib import tune_krr, krr_fit_predict
    sid, trait, trait_id, grp, g = args
    P = _P
    ids, _, cl = P.trait(sid, trait)
    cl, _ = P.merge_rare(ids, cl, min_n=10)
    raw = P.long[(P.long.study_id == sid) & (P.long.trait == trait)].groupby("accession_id")["value"].mean()
    yraw = raw.loc[ids].values.astype(np.float64)
    score_sd = yraw.std()
    ig = [P.gidx[a] for a in ids]; ie = [P.eidx[a] for a in ids]
    KG = P.KG[np.ix_(ig, ig)]; KE = P.KE[np.ix_(ie, ie)]
    gidx = np.flatnonzero(cl == g); others = np.flatnonzero(cl != g)
    ng = len(gidx)
    designs = ("added", "size-matched") if ng // 2 <= len(others) else ("added",)
    rows = []
    for rep in range(REPS):
        rng = np.random.default_rng(zlib.crc32(f"{trait_id}|{g}|{rep}".encode()))
        perm = rng.permutation(gidx)
        n_eval = (ng + 1) // 2
        ev, pool = np.sort(perm[:n_eval]), perm[n_eval:]
        drop_order = rng.permutation(others)
        for f in FRACS:
            m = min(int(round(f * ng)), len(pool))
            for design in designs:
                if f == 0 and design == "size-matched":
                    continue
                kept = others if design == "added" else np.setdiff1d(others, drop_order[:m])
                tr = np.sort(np.concatenate([kept, pool[:m]]))
                assert design != "size-matched" or len(tr) == len(others)
                assert not np.intersect1d(tr, ev).size
                lo, hi = np.percentile(yraw[tr], [1, 99])
                clipped = np.clip(yraw, lo, hi)
                mu, sd = clipped[tr].mean(), clipped[tr].std()
                y = (clipped - mu) / sd
                inner = list(KFold(3, shuffle=True, random_state=100 + rep).split(tr))
                for feat in ("G", "E", "GE"):
                    if feat == "G":
                        _, a, _ = tune_krr(y, tr, inner, C.ALPHAS, K=KG)
                        pred = krr_fit_predict(KG, y, tr, ev, a); w = 1.0
                    elif feat == "E":
                        _, a, _ = tune_krr(y, tr, inner, C.ALPHAS, K=KE)
                        pred = krr_fit_predict(KE, y, tr, ev, a); w = 0.0
                    else:
                        w, a, _ = tune_krr(y, tr, inner, C.ALPHAS, KG=KG, KE=KE, wgrid=C.WGRID)
                        pred = krr_fit_predict(w * KG + (1 - w) * KE, y, tr, ev, a)
                    predraw = mu + sd * pred
                    residual = (predraw - yraw[ev]) / score_sd
                    r = pearsonr(yraw[ev], predraw)[0] if np.std(predraw) > 1e-8 else np.nan
                    for d in (designs if f == 0 else [design]):
                        rows.append({"trait_id": trait_id, "group": grp, "cluster6": int(g), "n_group": ng,
                                     "n_eval": len(ev), "rep": rep, "fraction": f, "n_added": m,
                                     "design": d, "n_train": len(tr), "features": feat, "r": r,
                                     "bias": float(np.mean(residual)),
                                     "abs_bias": float(np.abs(np.mean(residual))),
                                     "rmse": float(np.sqrt(np.mean(residual ** 2))),
                                     "alpha": float(a), "w": float(w)})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    from benchmark_lib import Panel, PROC
    P = Panel()
    traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    units = []
    for row in traits.itertuples():
        ids, y, cl = P.trait(row.study_id, row.trait)
        cl, _ = P.merge_rare(ids, cl, min_n=10)
        u, cnt = np.unique(cl, return_counts=True)
        if len(u) < 2:
            continue
        for g, c in zip(u, cnt):
            if c >= MIN_GROUP:
                units.append((row.study_id, row.trait, row.trait_id, row.group, int(g)))
    del P
    print(f"{len(units)} trait x group units", flush=True)
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time(); allrows = []
    with Pool(nproc, initializer=init) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_unit, units)):
            allrows.extend(rows)
            pd.DataFrame(allrows).to_csv(os.path.join(OUT, "augmentation_raw.csv"), index=False)
            print(f"  {i+1}/{len(units)} {rows[0]['trait_id']} g{rows[0]['cluster6']} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT, "augmentation_raw.csv"), index=False)

    unit = df.groupby(["trait_id", "cluster6", "design", "features", "fraction"], as_index=False).agg(
        r=("r", "mean"), bias=("bias", "mean"), mean_abs_bias=("abs_bias", "mean"), rmse=("rmse", "mean"),
        n_added=("n_added", "mean"), n_group=("n_group", "first"))
    unit.to_csv(os.path.join(OUT, "augmentation_units.csv"), index=False)
    summ = unit.groupby(["design", "features", "fraction"]).agg(
        n_units=("r", "count"), median_r=("r", "median"), mean_r=("r", "mean"),
        median_abs_bias=("mean_abs_bias", "median"),
        median_n_added=("n_added", "median")).reset_index()
    summ.to_csv(os.path.join(OUT, "augmentation_summary.csv"), index=False)
    print(summ.round(3).to_string(), flush=True)

    tests = []
    for (design, feat), sub in unit.groupby(["design", "features"]):
        wide = sub.pivot_table(index=["trait_id", "cluster6"], columns="fraction", values="r")
        for f in FRACS[1:]:
            d = (wide[f] - wide[0.0]).dropna()
            w = wilcoxon(d)
            tests.append({"design": design, "features": feat, "fraction": f, "n_units": len(d),
                          "median_delta_r": float(d.median()), "frac_positive": float((d > 0).mean()),
                          "wilcoxon_p": float(w.pvalue)})
    for f in FRACS:
        sub = unit[(unit.design == "added") & (unit.fraction == f)]
        wide = sub.pivot_table(index=["trait_id", "cluster6"], columns="features", values="r")
        d = (wide["GE"] - wide["G"]).dropna()
        tests.append({"design": "added", "features": "GE-G", "fraction": f, "n_units": len(d),
                      "median_delta_r": float(d.median()), "frac_positive": float((d > 0).mean()),
                      "wilcoxon_p": float(wilcoxon(d).pvalue)})
    t = pd.DataFrame(tests)
    t["holm_p"] = C.holm(t.wilcoxon_p.values)
    t.to_csv(os.path.join(OUT, "augmentation_tests.csv"), index=False)
    print(t.round(4).to_string(), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
