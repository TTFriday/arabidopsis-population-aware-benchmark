"""Define population clusters (k=6) on genotype PCs; validate against geography."""
import os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(OUT, "data_processed")

pcs = pd.read_csv(os.path.join(PROC, "PCs.csv"), dtype={"accession_id": str})
meta = json.load(open(os.path.join(DATA, "02_phenotype_AraPheno/accession_list.json"), encoding="utf-8"))
m = pd.DataFrame([{"accession_id": str(a["pk"]), "name": a["name"], "country": a["country"],
                   "lat": a["latitude"], "lon": a["longitude"]} for a in meta])
df = pcs.merge(m, on="accession_id", how="left")

for k in (6, 8):
    Xs = StandardScaler().fit_transform(df[[f"PC{i+1}" for i in range(10)]])
    lab = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(Xs)
    df[f"cluster{k}"] = lab
    sizes = np.bincount(lab)
    print(k, sizes)
    for c in range(k):
        sub = df[df[f"cluster{k}"] == c]
        top = sub.country.value_counts().head(3).to_dict()
        print(f"  c{c}: n={len(sub):4d} lat={sub.lat.mean():6.1f} lon={sub.lon.mean():7.1f} {top}")

df[["accession_id", "name", "country", "lat", "lon", "cluster6", "cluster8"]].to_csv(
    os.path.join(PROC, "pop_clusters.csv"), index=False)
df[["accession_id", "cluster6"]].rename(columns={"cluster6": "cluster"}).to_csv(
    os.path.join(PROC, "pop6.csv"), index=False)
