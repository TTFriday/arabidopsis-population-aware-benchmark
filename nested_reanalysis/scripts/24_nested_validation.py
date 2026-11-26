import argparse
import hashlib
import json
import time
from multiprocessing import Pool

import biosystems_lib as B
import numpy as np
import pandas as pd

_P = None


def init():
    global _P
    _P = B.Panel()


def run(trait_id):
    ids, y, cl, X, R, study = _P.trait(trait_id)
    out = B.NEW / "results" / "nested"
    out.mkdir(parents=True, exist_ok=True)
    rows, folds, groups, decomposition = [], [], [], []
    scale = float(y.std())
    for cv in ("random", "lopo"):
        for repeat in range(2 if cv == "random" else 1):
            partition, _ = B.splits(cl, cv, repeat + 1)
            predictions = {feature: np.full(len(ids), np.nan) for feature in ("G", "E", "GE", "group_mean")}
            membership = np.full(len(ids), -1)
            for fold, (train, test) in enumerate(partition):
                pred, parameters, details = B.nested_predict(X, R, y, cl, train, test, cv, 100 + repeat)
                membership[test] = fold
                for feature in ("G", "E", "GE"):
                    predictions[feature][test] = pred[feature]
                    folds.append({"trait_id": trait_id, "study_id": study, "cv": cv, "repeat": repeat,
                                  "fold": fold, "features": feature, "n_train": len(train), "n_test": len(test),
                                  "fst": B.fst(X, train, test), **parameters[feature], **details,
                                  "train_ids_sha256": hashlib.sha256("|".join(ids[train]).encode()).hexdigest(),
                                  **B.metrics(y[test], pred[feature], scale)})
                means = {c: y[train][cl[train] == c].mean() for c in np.unique(cl[train])}
                predictions["group_mean"][test] = np.array([means.get(c, y[train].mean()) for c in cl[test]])
            assert np.all(membership >= 0)
            np.savez_compressed(out / f"{trait_id}__{cv}__r{repeat}.npz", ids=ids, y=y, clusters=cl,
                                fold=membership, **predictions)
            for feature, pred in predictions.items():
                rows.append({"trait_id": trait_id, "study_id": study, "cv": cv, "repeat": repeat,
                             "features": feature, "n": len(y), **B.metrics(y, pred, scale)})
                yy, pp = y.copy(), pred.copy()
                between_variance = 0.0
                for c in np.unique(cl):
                    mask = cl == c
                    between_variance += mask.sum() * (y[mask].mean() - y.mean()) ** 2
                    yy[mask] -= y[mask].mean()
                    pp[mask] -= pred[mask].mean()
                    if cv == "lopo" and mask.sum() >= 15 and feature != "group_mean":
                        fold = int(membership[mask][0])
                        train, test = partition[fold]
                        groups.append({"trait_id": trait_id, "study_id": study, "cluster6": int(c),
                                       "features": feature, "n_test": int(mask.sum()), "n_train": len(train),
                                       "fst": B.fst(X, train, test), **B.metrics(y[mask], pred[mask], scale)})
                decomposition.append({"trait_id": trait_id, "study_id": study, "features": feature,
                                      "cv": cv, "repeat": repeat, "r_pooled": B.metrics(y, pred, scale)["r"],
                                      "r_group_centered": B.metrics(yy, pp, scale)["r"],
                                      "phenotype_between_group_share": between_variance / np.sum((y - y.mean()) ** 2)})
    for name, data in (("traits", rows), ("folds", folds), ("groups", groups), ("decomposition", decomposition)):
        pd.DataFrame(data).to_csv(out / f"{trait_id}__{name}.csv", index=False)
    return trait_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--traits", nargs="*")
    args = parser.parse_args()
    out = B.NEW / "results" / "nested"
    out.mkdir(parents=True, exist_ok=True)
    traits = pd.read_csv(B.PROC / "benchmark_traits.csv").trait_id.tolist()
    if args.traits:
        traits = args.traits
    start = time.perf_counter()
    with Pool(args.workers, initializer=init) as pool:
        for count, trait in enumerate(pool.imap_unordered(run, traits), 1):
            print(f"Nested validation {count}/{len(traits)} {trait}: {time.perf_counter()-start:.1f}s", flush=True)
    for name in ("traits", "folds", "groups", "decomposition"):
        pd.concat([pd.read_csv(out / f"{trait}__{name}.csv") for trait in traits], ignore_index=True).to_csv(out / f"{name}_raw.csv", index=False)
    raw = pd.read_csv(out / "traits_raw.csv")
    averaged = raw.groupby(["trait_id", "study_id", "cv", "features"], as_index=False)[["r", "spearman", "r2", "bias", "abs_bias", "rmse"]].mean()
    averaged.to_csv(out / "per_trait.csv", index=False)
    summary = averaged.groupby(["features", "cv"], as_index=False).agg(n_traits=("r", "count"), median_r=("r", "median"), median_r2=("r2", "median"), median_rmse=("rmse", "median"))
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    (out / "run.json").write_text(json.dumps({"traits": traits, "workers": args.workers, "elapsed_seconds": time.perf_counter()-start,
                                              "outer_random_repeats": 2, "outer_lopo_repeats": 1,
                                              "preprocessing": "Refitted within each inner-training fold and outer-training set; fixed upstream marker ascertainment, clustering, imputation and normalized-count resource"}, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
