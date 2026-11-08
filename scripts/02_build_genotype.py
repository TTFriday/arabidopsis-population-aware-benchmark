"""Stream SNP HDF5 -> MAF filter -> 5 kb thinning -> LD prune -> GRM/PCA/populations."""
import os, json, time
import numpy as np
import pandas as pd

DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(OUT, "data_processed")
os.makedirs(PROC, exist_ok=True)
SNP = os.path.join(DATA, "01_genotype_1001G/1001_SNP_MATRIX/imputed_snps_binary.hdf5")

import h5py
f = h5py.File(SNP, "r")
snps = f["snps"]; positions = f["positions"]
acc = np.array([x.decode() for x in f["accessions"][:]])
chr_regions = positions.attrs["chr_regions"]
chrs = [c.decode() for c in positions.attrs["chrs"]]
n_snp, n_acc = snps.shape
print("matrix", n_snp, n_acc, flush=True)

MAF_MIN = 0.05
BIN = 5000

# ---- pass 1: per-chromosome MAF + peak-per-bin ----
sel_idx, sel_chr, sel_pos, sel_maf = [], [], [], []
t0 = time.time()
for ci, (a, b) in enumerate(chr_regions):
    pos_c = positions[a:b]
    maf_c = np.empty(b - a, dtype=np.float32)
    for s in range(a, b, 50000):
        e = min(s + 50000, b)
        x = snps[s:e]
        s1 = x.sum(axis=1, dtype=np.int32)
        p = s1 / n_acc
        maf_c[s - a:e - a] = np.minimum(p, 1 - p)
    keep = maf_c >= MAF_MIN
    idx_k = np.flatnonzero(keep)
    maf_k = maf_c[idx_k]; pos_k = pos_c[idx_k]
    bins = pos_k // BIN
    order = np.lexsort((maf_k, bins))
    sb = bins[order]
    last = np.r_[sb[1:] != sb[:-1], True]
    pick = order[last]
    sel_idx.append(idx_k[pick] + a)
    sel_chr.append(np.full(len(pick), ci + 1, dtype=np.int8))
    sel_pos.append(pos_k[pick]); sel_maf.append(maf_k[pick])
    print(f"chr{ci+1}: {b-a} SNPs, MAF>=0.05: {keep.sum()}, binned: {len(pick)}, {time.time()-t0:.0f}s", flush=True)

sel_idx = np.concatenate(sel_idx); sel_chr = np.concatenate(sel_chr)
sel_pos = np.concatenate(sel_pos); sel_maf = np.concatenate(sel_maf)
print("selected after thinning:", len(sel_idx), flush=True)

# ---- pass 2: extract genotypes (sequential blocks, no random reads) ----
X = np.empty((len(sel_idx), n_acc), dtype=np.int8)
ptr = 0
for s in range(0, n_snp, 100000):
    e = min(s + 100000, n_snp)
    lo = np.searchsorted(sel_idx, s, side="left")
    hi = np.searchsorted(sel_idx, e, side="left")
    if hi > lo:
        block = snps[s:e]
        X[lo:hi] = block[sel_idx[lo:hi] - s]
f.close()

# ---- LD prune (window 200, r2 > 0.5) ----
def ld_prune(X, maf, win=200, r2max=0.5):
    keep = np.zeros(X.shape[0], dtype=bool)
    n = X.shape[0]
    for w0 in range(0, n, win):
        w1 = min(w0 + win, n)
        Xw = X[w0:w1].astype(np.float32)
        Xw -= Xw.mean(axis=1, keepdims=True)
        sd = Xw.std(axis=1); sd[sd == 0] = 1
        Xw /= sd[:, None]
        C = (Xw @ Xw.T) / Xw.shape[1]
        order = np.argsort(-maf[w0:w1])
        kept_local = []
        for j in order:
            if not kept_local:
                kept_local.append(j); continue
            if np.max(np.abs(C[j, kept_local])) <= r2max:
                kept_local.append(j)
        keep[w0 + np.array(kept_local)] = True
    return keep

keep = ld_prune(X, sel_maf)
print("after LD prune:", int(keep.sum()), flush=True)
X = X[keep]; sel_idx = sel_idx[keep]; sel_chr = sel_chr[keep]
sel_pos = sel_pos[keep]; sel_maf = sel_maf[keep]

np.save(os.path.join(PROC, "geno_matrix.npy"), X)
pd.DataFrame({"chr": sel_chr, "pos": sel_pos, "maf": sel_maf, "hdf5_index": sel_idx}).to_csv(
    os.path.join(PROC, "geno_meta.csv"), index=False)
pd.DataFrame({"accession_id": acc}).to_csv(os.path.join(PROC, "accessions.csv"), index=False)

# ---- GRM (standardized) ----
Xf = X.astype(np.float32)
mu = Xf.mean(axis=1, keepdims=True)
sd = Xf.std(axis=1, keepdims=True); sd[sd == 0] = 1
Z = (Xf - mu) / sd
G = (Z.T @ Z) / Z.shape[0]
G = G.astype(np.float32)
np.save(os.path.join(PROC, "GRM.npy"), G)
print("GRM", G.shape, "diag", np.diag(G).mean().round(3), flush=True)

# ---- PCA ----
from sklearn.decomposition import PCA
pca = PCA(n_components=20, svd_solver="randomized", random_state=42)
pcs = pca.fit_transform(Z.T)
pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(20)])
pcs_df.insert(0, "accession_id", acc)
pcs_df.to_csv(os.path.join(PROC, "PCs.csv"), index=False)
print("PC var explained:", np.round(pca.explained_variance_ratio_[:5], 3), flush=True)

# ---- population clusters (k=8) ----
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
pcs_s = StandardScaler().fit_transform(pcs[:, :10])
km = KMeans(n_clusters=8, n_init=20, random_state=42).fit(pcs_s)
cl = pd.DataFrame({"accession_id": acc, "cluster": km.labels_})
cl.to_csv(os.path.join(PROC, "pop_clusters.csv"), index=False)
print("cluster sizes:", np.bincount(km.labels_), flush=True)

json.dump({"n_markers": int(X.shape[0]), "maf_min": MAF_MIN, "bin_bp": BIN,
           "n_pcs": 20, "n_clusters": 8},
          open(os.path.join(PROC, "geno_meta.json"), "w"), indent=2)
print("done", flush=True)
