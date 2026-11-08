"""Process 1001T expression: log2, gene filter, standardization, PCs, kernel."""
import os, gzip
import numpy as np
import pandas as pd

DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(OUT, "data_processed")
p = os.path.join(DATA, "03_transcriptome_1001T/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz")

df = pd.read_csv(p, sep="\t", index_col=0)
print("raw", df.shape, flush=True)
df.columns = [c[1:] if c.startswith("X") else c for c in df.columns]
X = np.log2(df.values.astype(np.float32) + 1.0)
genes = df.index.astype(str).values
accs = df.columns.astype(str).values
print("log2 range", X.min(), X.max(), flush=True)

det = (X > 0).mean(axis=1)
mu = X.mean(axis=1); var = X.var(axis=1)
keep = (det >= 0.2) & (mu > 1.0) & (var > 0.01)
print(f"genes kept {keep.sum()} / {len(keep)}", flush=True)
X = X[keep]; genes = genes[keep]

Z = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
Z = Z.astype(np.float32)
np.save(os.path.join(PROC, "expr_matrix.npy"), Z)
pd.DataFrame({"gene_id": genes}).to_csv(os.path.join(PROC, "expr_genes.csv"), index=False)
pd.DataFrame({"accession_id": accs}).to_csv(os.path.join(PROC, "expr_accessions.csv"), index=False)

# expression kernel over all 727
K = (Z.T @ Z) / Z.shape[0]
np.save(os.path.join(PROC, "expr_kernel.npy"), K.astype(np.float32))
print("expr kernel", K.shape, "diag", np.diag(K).mean().round(3), flush=True)

# PCs
from sklearn.decomposition import PCA
pca = PCA(n_components=50, svd_solver="randomized", random_state=42)
pcs = pca.fit_transform(Z.T)
pcdf = pd.DataFrame(pcs, columns=[f"EPC{i+1}" for i in range(50)])
pcdf.insert(0, "accession_id", accs)
pcdf.to_csv(os.path.join(PROC, "expr_PCs.csv"), index=False)
print("expr PC var", np.round(pca.explained_variance_ratio_[:5], 3), flush=True)
