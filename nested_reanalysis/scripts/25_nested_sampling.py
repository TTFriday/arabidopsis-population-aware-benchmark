import argparse
import hashlib
import json
import time
import zlib
from multiprocessing import Pool

import biosystems_lib as B
import numpy as np
import pandas as pd

_P = None


def init():
    global _P
    _P = B.Panel()


def run(job):
    trait_id, group = job
    ids, y, cl, X, R, study = _P.trait(trait_id)
    gidx, others = np.flatnonzero(cl == group), np.flatnonzero(cl != group)
    size_matched = len(gidx) // 2 <= len(others)
    out = B.NEW / "results" / "sampling"
    out.mkdir(parents=True, exist_ok=True)
    rows, assignments = [], []
    scale = float(y.std())
    for repeat in range(10):
        rng = np.random.default_rng(zlib.crc32(f"{trait_id}|{group}|{repeat}".encode()))
        perm = rng.permutation(gidx)
        n_eval = (len(gidx) + 1) // 2
        test, pool = np.sort(perm[:n_eval]), perm[n_eval:]
        removal = rng.permutation(others)
        assignments.append({"repeat": repeat, "evaluation": ids[test].tolist(), "candidate_order": ids[pool].tolist(),
                            "other_group_removal_order": ids[removal].tolist()})
        arms = [("added", 0.0, others), ("added", 0.5, np.sort(np.concatenate([others, pool])))]
        if size_matched:
            arms.append(("size-matched", 0.5, np.sort(np.concatenate([np.setdiff1d(others, removal[:len(pool)]), pool]))))
        for design, fraction, train in arms:
            if design == "size-matched":
                assert len(train) == len(others)
            assert not np.intersect1d(train, test).size
            prediction, parameters, details = B.nested_predict(X, R, y, cl, train, test, "unstratified", 100+repeat, expression=False)
            result = {"trait_id": trait_id, "study_id": study, "cluster6": int(group), "repeat": repeat,
                      "design": design, "fraction": fraction, "features": "G", "n_group": len(gidx),
                      "n_test": len(test), "n_train": len(train), "n_added": 0 if fraction == 0 else len(pool),
                      "train_ids_sha256": hashlib.sha256("|".join(ids[train]).encode()).hexdigest(),
                      "test_ids_sha256": hashlib.sha256("|".join(ids[test]).encode()).hexdigest(),
                      **parameters["G"], **details, **B.metrics(y[test], prediction["G"], scale)}
            rows.append(result)
            if fraction == 0 and size_matched:
                rows.append({**result, "design": "size-matched"})
    pd.DataFrame(rows).to_csv(out / f"{trait_id}__g{group}.csv", index=False)
    (out / f"{trait_id}__g{group}_splits.json").write_text(json.dumps(assignments), encoding="utf8")
    return trait_id, group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    panel = B.Panel()
    jobs = []
    for trait_id in panel.traits.trait_id:
        _, _, cl, _, _, _ = panel.trait(trait_id)
        group, count = np.unique(cl, return_counts=True)
        if len(group) >= 2:
            jobs.extend((trait_id, int(g)) for g, n in zip(group, count) if n >= 20)
    del panel
    start = time.perf_counter()
    with Pool(args.workers, initializer=init) as pool:
        for i, job in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"Nested target sampling {i}/{len(jobs)} {job}: {time.perf_counter()-start:.1f}s", flush=True)
    out = B.NEW / "results" / "sampling"
    raw = pd.concat([pd.read_csv(out / f"{trait}__g{g}.csv") for trait, g in jobs], ignore_index=True)
    raw.to_csv(out / "sampling_raw.csv", index=False)
    units = raw.groupby(["trait_id", "study_id", "cluster6", "features", "design", "fraction"], as_index=False).agg(
        r=("r", "mean"), mean_abs_bias=("abs_bias", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"),
        n_added=("n_added", "mean"), n_group=("n_group", "first"), n_train=("n_train", "mean"))
    units.to_csv(out / "sampling_units.csv", index=False)
    summary = units.groupby(["design", "features", "fraction"], as_index=False).agg(
        n_units=("r", "count"), median_r=("r", "median"), median_abs_bias=("mean_abs_bias", "median"), median_rmse=("rmse", "median"),
        median_n_added=("n_added", "median"))
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    (out / "run.json").write_text(json.dumps({"elapsed_seconds": time.perf_counter()-start, "workers": args.workers,
                                              "units": len(jobs), "repeats": 10, "feature": "G",
                                              "fractions": [0, 0.5], "fully_nested_prediction_preprocessing": True,
                                              "sampling_seed": "CRC32(trait_id|group|repeat)",
                                              "scope": "Confirmatory reanalysis of archived endpoint design, not independent external validation"}, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
