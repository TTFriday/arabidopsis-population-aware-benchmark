"""Estimate genomic heritability (REML) for all traits with n>=200."""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(OUT, "data_processed")

K = np.load(os.path.join(PROC, "GRM.npy")).astype(np.float64)
acc = pd.read_csv(os.path.join(PROC, "accessions.csv"))["accession_id"].astype(str).values
idx = {a: i for i, a in enumerate(acc)}
vals, U = np.linalg.eigh(K)
vals = np.clip(vals, 1e-6, None)
n_all = len(acc)
Xt1 = U.T @ np.ones((n_all, 1))

long = pd.read_csv(os.path.join(PROC, "phenotypes_long.csv"), dtype={"accession_id": str})
long = long[long.accession_id.isin(idx)]

def reml_h2(y):
    yt = U.T @ y
    n = len(y)
    def negll(loglam):
        lam = np.exp(loglam)
        d = vals + lam
        w = 1.0 / d
        Xw = Xt1 * w[:, None]
        A = Xt1.T @ Xw
        b = Xt1.T @ (w * yt)
        beta = np.linalg.solve(A, b)
        r = yt - Xt1 @ beta
        s2 = float(np.sum(w * r * r) / (n - 1))
        if s2 <= 0:
            return 1e12
        logdetA = np.log(A[0, 0])
        return 0.5 * ((n - 1) * np.log(2 * np.pi) + (n - 1) * np.log(s2)
                      + np.sum(np.log(d)) + logdetA + (n - 1))
    res = minimize_scalar(negll, bounds=(-9, 4), method="bounded",
                          options={"xatol": 1e-4})
    lam = np.exp(res.x)
    return 1.0 / (1.0 + lam), res.fun

rows = []
for (sid, tr), g in long.groupby(["study_id", "trait"]):
    g = g.drop_duplicates("accession_id")
    if len(g) < 200:
        continue
    y = np.full(n_all, np.nan)
    y[[idx[a] for a in g.accession_id]] = g.value.values
    m = ~np.isnan(y)
    # subset K to observed
    if m.sum() < 200:
        continue
    Ks = K[np.ix_(m, m)]
    v, Us = np.linalg.eigh(Ks)
    v = np.clip(v, 1e-6, None)
    yy = y[m]; yy = (yy - yy.mean()) / yy.std()
    yt = Us.T @ yy; Xs = Us.T @ np.ones((len(yy), 1)); n = len(yy)
    def negll(loglam, vals=v, yt=yt, Xs=Xs, n=n):
        lam = np.exp(loglam); d = vals + lam; w = 1.0 / d
        Xw = Xs * w[:, None]; A = Xs.T @ Xw; b = Xs.T @ (w * yt)
        beta = np.linalg.solve(A, b); r = yt - Xs @ beta
        s2 = float(np.sum(w * r * r) / (n - 1))
        if s2 <= 0: return 1e12
        return 0.5 * ((n - 1) * np.log(2 * np.pi) + (n - 1) * np.log(s2)
                      + np.sum(np.log(d)) + np.log(A[0, 0]) + (n - 1))
    res = minimize_scalar(negll, bounds=(-9, 4), method="bounded", options={"xatol": 1e-4})
    h2 = 1.0 / (1.0 + np.exp(res.x))
    rows.append({"study_id": sid, "trait": tr, "n": int(m.sum()), "h2": h2,
                 "mean": float(g.value.mean()), "sd": float(g.value.std()),
                 "skew": float(g.value.skew()), "kurt": float(g.value.kurt())})
    print(f"{sid:>3} {tr[:45]:<45} n={m.sum():4d} h2={h2:.2f}", flush=True)

h = pd.DataFrame(rows).sort_values("h2", ascending=False)
h.to_csv(os.path.join(OUT, "results", "heritability.csv"), index=False)
print("\ntraits:", len(h), "h2>=0.2:", int((h.h2 >= 0.2).sum()), "h2>=0.3:", int((h.h2 >= 0.3).sum()))
print(h.head(30).to_string(index=False))
