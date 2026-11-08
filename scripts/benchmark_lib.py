"""Shared data/model utilities for the population-aware prediction benchmark."""
import os, json, gc
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import LassoCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy.stats import pearsonr, spearmanr

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(OUT, "data_processed")


class Panel:
    def __init__(self):
        self.acc_g = pd.read_csv(os.path.join(PROC, "accessions.csv"))["accession_id"].astype(str).values
        self.G = np.load(os.path.join(PROC, "geno_matrix.npy")).T.astype(np.float32)  # n_g x p
        self.KG = np.load(os.path.join(PROC, "GRM.npy")).astype(np.float64)
        pcg = pd.read_csv(os.path.join(PROC, "PCs.csv"), dtype={"accession_id": str})
        pcg = pcg.set_index("accession_id").loc[self.acc_g]
        self.PCG = pcg.values.astype(np.float32)

        self.acc_e = pd.read_csv(os.path.join(PROC, "expr_accessions.csv"))["accession_id"].astype(str).values
        self.E = np.load(os.path.join(PROC, "expr_matrix.npy")).T.astype(np.float32)  # n_e x p
        ke = np.load(os.path.join(PROC, "expr_kernel.npy")).astype(np.float64)
        self.KE = ke
        pce = pd.read_csv(os.path.join(PROC, "expr_PCs.csv"), dtype={"accession_id": str})
        pce = pce.set_index("accession_id").loc[self.acc_e]
        self.PCE = pce.values.astype(np.float32)

        self.long = pd.read_csv(os.path.join(PROC, "phenotypes_long.csv"), dtype={"accession_id": str})
        pop = pd.read_csv(os.path.join(PROC, "pop6.csv"), dtype={"accession_id": str})
        self.pop = dict(zip(pop.accession_id, pop.cluster))
        self.gidx = {a: i for i, a in enumerate(self.acc_g)}
        self.eidx = {a: i for i, a in enumerate(self.acc_e)}
        self.gset, self.eset = set(self.acc_g), set(self.acc_e)

    def trait(self, sid, trait, require_expr=True):
        g = self.long[(self.long.study_id == sid) & (self.long.trait == trait)]
        g = g.groupby("accession_id")["value"].mean()
        ids = [a for a in g.index if a in self.gset and (a in self.eset if require_expr else True)]
        ids = sorted(ids, key=lambda x: int(x))
        y = g.loc[ids].values.astype(np.float64)
        lo, hi = np.percentile(y, [1, 99])
        y = np.clip(y, lo, hi)
        y = (y - y.mean()) / y.std()
        cl = np.array([self.pop[a] for a in ids])
        return ids, y, cl

    def merge_rare(self, ids, cl, min_n=10):
        """Merge clusters with <min_n panel samples into the nearest larger cluster (genotype PC space)."""
        ig = [self.gidx[a] for a in ids]
        P = self.PCG[ig][:, :5]
        u, cnt = np.unique(cl, return_counts=True)
        cent = {c: P[cl == c].mean(0) for c in u}
        big = [c for c in u if cnt[list(u).index(c)] >= min_n]
        if not big:
            big = [u[int(np.argmax(cnt))]]
        mapping = {c: c for c in u}
        for c in u:
            if c not in big:
                mapping[c] = min(big, key=lambda b: np.linalg.norm(cent[c] - cent[b]))
        return np.array([mapping[c] for c in cl]), mapping


def make_splits(cl, scheme, seed=0, inner=False):
    u = np.unique(cl)
    if scheme == "random" or (inner and len(u) < 3):
        n_splits = 3 if inner else 5
        n_splits = max(2, min(n_splits, int(np.min(np.bincount(pd.factorize(cl)[0])))))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(cl)), cl))
    n_groups = min(3 if inner else len(u), len(u))
    if n_groups < 2:
        from sklearn.model_selection import KFold
        return list(KFold(n_splits=2, shuffle=True, random_state=seed).split(np.zeros(len(cl))))
    gkf = GroupKFold(n_splits=n_groups)
    return list(gkf.split(np.zeros(len(cl)), groups=cl))


def center_kernel(K, Kcross=None):
    n = K.shape[0]
    rm = K.mean(axis=1, keepdims=True)
    gm = K.mean()
    Kc = K - rm - rm.T + gm
    if Kcross is None:
        return Kc
    Kcc = Kcross - rm.T - Kcross.mean(axis=1, keepdims=True) + gm
    return Kc, Kcc


def krr_cv_alpha(K, y, tr, inner, alphas):
    errs = np.zeros(len(alphas))
    for itr, iva in inner:
        tri, vai = tr[itr], tr[iva]
        Ksub = K[np.ix_(tri, tri)]
        Kval = K[np.ix_(vai, tri)]
        Kc, Kvc = center_kernel(Ksub, Kval)
        vals, U = eigh(Kc)
        vals = np.clip(vals, 1e-8, None)
        yc = y[tri] - y[tri].mean()
        Uty = U.T @ yc
        for i, a in enumerate(alphas):
            coef = U @ (Uty / (vals + a))
            pred = Kvc @ coef + y[tri].mean()
            errs[i] += np.mean((y[vai] - pred) ** 2)
    return alphas[int(np.argmin(errs))], float(errs.min() / len(inner))


def krr_fit_predict(K, y, tr, te, alpha):
    Ksub = K[np.ix_(tr, tr)]
    Kval = K[np.ix_(te, tr)]
    Kc, Kvc = center_kernel(Ksub, Kval)
    vals, U = eigh(Kc)
    vals = np.clip(vals, 1e-8, None)
    yc = y[tr] - y[tr].mean()
    coef = U @ ((U.T @ yc) / (vals + alpha))
    return Kvc @ coef + y[tr].mean()


def tune_krr(y, tr, inner, alphas, K=None, KG=None, KE=None, wgrid=None):
    best = (np.inf, None, None)
    if wgrid is None:
        a, e = krr_cv_alpha(K, y, tr, inner, alphas)
        return None, a, e
    for w in wgrid:
        Kw = w * KG + (1 - w) * KE
        a, e = krr_cv_alpha(Kw, y, tr, inner, alphas)
        if e < best[0]:
            best = (e, w, a)
    return best[1], best[2], best[0]


def scale_blocks(Xg, Xe, tr, te, pg, pe):
    parts_tr, parts_te = [], []
    if Xg is not None:
        mu = Xg[tr].mean(0); sd = Xg[tr].std(0); sd[sd == 0] = 1
        parts_tr.append((Xg[tr] - mu) / sd / np.sqrt(pg))
        parts_te.append((Xg[te] - mu) / sd / np.sqrt(pg))
    if Xe is not None:
        mu = Xe[tr].mean(0); sd = Xe[tr].std(0); sd[sd == 0] = 1
        parts_tr.append((Xe[tr] - mu) / sd / np.sqrt(pe))
        parts_te.append((Xe[te] - mu) / sd / np.sqrt(pe))
    return np.hstack(parts_tr), np.hstack(parts_te)


def metrics(y, yp):
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - yp) ** 2) / ss if ss > 0 else np.nan
    pr = pearsonr(y, yp)[0] if np.std(yp) > 1e-8 else np.nan
    sr = spearmanr(y, yp)[0] if np.std(yp) > 1e-8 else np.nan
    return r2, pr, sr, float(np.mean(np.abs(y - yp)))
