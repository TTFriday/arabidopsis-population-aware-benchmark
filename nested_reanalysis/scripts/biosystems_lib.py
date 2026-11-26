import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
NEW = Path(__file__).resolve().parents[1]
PROC = Path(os.environ.get("ARABIDOPSIS_PROCESSED", NEW / "inputs" / "data_processed"))
EXPR = Path(os.environ.get("ARABIDOPSIS_EXPRESSION_LOG", NEW / "inputs" / "expr_log2_raw.npy"))
ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0])
WEIGHTS = np.array([0, 0.25, 0.5, 0.75, 1.0])


class Panel:
    def __init__(self):
        self.gids = pd.read_csv(PROC / "accessions.csv", dtype=str).accession_id.to_numpy()
        self.eids = pd.read_csv(PROC / "expr_accessions.csv", dtype=str).accession_id.to_numpy()
        self.gidx = {a: i for i, a in enumerate(self.gids)}
        self.eidx = {a: i for i, a in enumerate(self.eids)}
        self.G = np.load(PROC / "geno_matrix.npy", mmap_mode="r").T
        self.R = np.load(EXPR, mmap_mode="r")
        self.long = pd.read_csv(PROC / "phenotypes_long.csv", dtype={"accession_id": str})
        clusters = pd.read_csv(PROC / "pop6.csv", dtype={"accession_id": str})
        self.pop = dict(zip(clusters.accession_id, clusters.cluster))
        self.pcs = pd.read_csv(PROC / "PCs.csv", dtype={"accession_id": str}).set_index("accession_id")
        self.traits = pd.read_csv(PROC / "benchmark_traits.csv")

    def trait(self, trait_id):
        row = self.traits.set_index("trait_id").loc[trait_id]
        means = self.long[(self.long.study_id == row.study_id) & (self.long.trait == row.trait)].groupby("accession_id").value.mean()
        ids = np.array(sorted((a for a in means.index if a in self.gidx and a in self.eidx), key=int))
        y = means.loc[ids].to_numpy(dtype=float)
        cl = np.array([self.pop[a] for a in ids])
        pcs = self.pcs.loc[ids].iloc[:, :5].to_numpy()
        groups, counts = np.unique(cl, return_counts=True)
        big = groups[counts >= 10]
        if not len(big):
            big = groups[[np.argmax(counts)]]
        centroids = {c: pcs[cl == c].mean(0) for c in groups}
        mapping = {c: c if c in big else min(big, key=lambda b: np.linalg.norm(centroids[c] - centroids[b])) for c in groups}
        cl = np.array([mapping[c] for c in cl])
        X = np.asarray(self.G[[self.gidx[a] for a in ids]], dtype=np.float32)
        R = np.asarray(self.R[[self.eidx[a] for a in ids]], dtype=np.float32)
        return ids, y, cl, X, R, int(row.study_id)


def splits(cl, cv, seed, inner=False):
    groups = np.unique(cl)
    if cv == "unstratified":
        return list(KFold(3, shuffle=True, random_state=seed).split(np.zeros(len(cl)))), "random"
    if cv == "random":
        count = np.min(np.bincount(pd.factorize(cl)[0]))
        k = max(2, min(3 if inner else 5, int(count)))
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(cl)), cl)), "stratified"
    if len(groups) >= 2:
        k = min(3, len(groups)) if inner else len(groups)
        return list(GroupKFold(k).split(np.zeros(len(cl)), groups=cl)), "grouped"
    return list(KFold(3 if inner else 2, shuffle=True, random_state=seed).split(np.zeros(len(cl)))), "one-group-random"


def kernels(X, R, train, expression=True):
    mean, sd = X[train].mean(0), X[train].std(0)
    keep = sd > 0
    Z = (X[:, keep] - mean[keep]) / sd[keep]
    KG = np.asarray(Z @ Z.T / keep.sum(), dtype=float)
    KE, genes = None, 0
    if expression:
        rt = R[train]
        mean, sd = rt.mean(0), rt.std(0)
        keep = ((rt > 0).mean(0) >= 0.2) & (mean > 1) & (sd ** 2 > 0.01)
        genes = int(keep.sum())
        Z = (R[:, keep] - mean[keep]) / sd[keep]
        KE = np.asarray(Z @ Z.T / genes, dtype=float)
    return KG, KE, int((X[train].std(0) > 0).sum()), genes


def response(raw, train):
    y = np.clip(raw[train], *np.percentile(raw[train], [1, 99]))
    mean, sd = y.mean(), y.std()
    if sd == 0:
        sd = 1.0
    return (y - mean) / sd, mean, sd


def ridge_grid(K, raw, train, test, alphas=ALPHAS):
    Kt, Kv = K[np.ix_(train, train)], K[np.ix_(test, train)]
    column = Kt.mean(0)
    Kt = Kt - column[None, :] - Kt.mean(1)[:, None] + column.mean()
    Kv = Kv - column[None, :] - Kv.mean(1)[:, None] + column.mean()
    values, U = eigh(Kt, check_finite=False)
    values = np.maximum(values, 0)
    y, mean, sd = response(raw, train)
    coefficients = (U.T @ (y - y.mean()))[:, None] / (values[:, None] + alphas[None, :])
    return mean + sd * ((Kv @ U) @ coefficients + y.mean())


def nested_predict(X, R, raw, cl, train, test, cv, seed, expression=True):
    assert not np.intersect1d(train, test).size
    inner, inner_type = splits(cl[train], cv, seed, inner=True)
    feature_weights = {"G": np.array([1.0])}
    if expression:
        feature_weights.update({"E": np.array([0.0]), "GE": WEIGHTS})
    losses = {f: np.zeros((len(w), len(ALPHAS))) for f, w in feature_weights.items()}
    for it, iv in inner:
        ti, vi = train[it], train[iv]
        KG, KE, _, _ = kernels(X, R, ti, expression)
        for feature, weights in feature_weights.items():
            for k, weight in enumerate(weights):
                K = KG if weight == 1 else KE if weight == 0 else weight * KG + (1 - weight) * KE
                pred = ridge_grid(K, raw, ti, vi)
                losses[feature][k] += np.mean((raw[vi, None] - pred) ** 2) / len(inner)
    KG, KE, markers, genes = kernels(X, R, train, expression)
    output, parameters = {}, {}
    for feature, weights in feature_weights.items():
        index = np.unravel_index(np.argmin(losses[feature]), losses[feature].shape)
        weight, alpha = float(weights[index[0]]), float(ALPHAS[index[1]])
        K = KG if weight == 1 else KE if weight == 0 else weight * KG + (1 - weight) * KE
        output[feature] = ridge_grid(K, raw, train, test, np.array([alpha]))[:, 0]
        parameters[feature] = {"alpha": alpha, "weight": weight, "inner_mse": float(losses[feature][index])}
    return output, parameters, {"inner_type": inner_type, "markers": markers, "genes": genes}


def metrics(y, prediction, scale):
    residual = (prediction - y) / scale
    variable = np.std(prediction) > 1e-8 and np.std(y) > 1e-8
    return {"r": float(pearsonr(y, prediction).statistic) if variable else np.nan,
            "spearman": float(spearmanr(y, prediction).statistic) if variable else np.nan,
            "r2": float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)) if np.std(y) > 0 else np.nan,
            "bias": float(residual.mean()), "abs_bias": float(abs(residual.mean())),
            "rmse": float(np.sqrt(np.mean(residual ** 2)))}


def fst(X, train, test):
    p1, p2 = X[train].mean(0).astype(float), X[test].mean(0).astype(float)
    numerator = (p1 - p2) ** 2 - p1 * (1 - p1) / (len(train) - 1) - p2 * (1 - p2) / (len(test) - 1)
    denominator = p1 * (1 - p2) + p2 * (1 - p1)
    return float(numerator.sum() / denominator.sum())
