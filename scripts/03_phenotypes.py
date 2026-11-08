"""Curate AraPheno traits: drop climate covariates, aggregate replicates per accession."""
import os, json
import numpy as np
import pandas as pd

DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AP = os.path.join(DATA, "02_phenotype_AraPheno")
PROC = os.path.join(OUT, "data_processed")
os.makedirs(PROC, exist_ok=True)

studies = json.load(open(os.path.join(AP, "study_list.json"), encoding="utf-8"))
snpset = set(pd.read_csv(os.path.join(PROC, "accessions.csv"))["accession_id"].astype(str))

long_rows, qc_rows = [], []
for st in studies:
    p = os.path.join(AP, "study_values", f"study_{st['id']:03d}.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p, encoding="utf-8", low_memory=False)
    df["accession_id"] = df["accession_id"].astype(str).str.strip()
    df = df[df["accession_id"].isin(snpset)]
    traits = [c for c in df.columns if c not in ("accession_id", "replicate_id") and not c.startswith("clim-")]
    for tr in traits:
        v = pd.to_numeric(df[tr], errors="coerce")
        tmp = pd.DataFrame({"accession_id": df["accession_id"], "value": v}).dropna()
        if tmp.empty:
            continue
        agg = tmp.groupby("accession_id")["value"].agg(["mean", "count", "std"])
        agg = agg.reset_index()
        for _, r in agg.iterrows():
            long_rows.append((st["id"], st["name"], tr, r["accession_id"], r["mean"], int(r["count"])))
        qc_rows.append({
            "study_id": st["id"], "study_name": st["name"], "trait": tr,
            "n_acc": len(agg), "n_rep_mean": agg["count"].mean(),
            "mean": agg["mean"].mean(), "sd": agg["mean"].std(),
            "min": agg["mean"].min(), "max": agg["mean"].max(),
            "cv": agg["mean"].std() / (abs(agg["mean"].mean()) + 1e-12),
        })

long = pd.DataFrame(long_rows, columns=["study_id", "study_name", "trait", "accession_id", "value", "n_rep"])
long.to_csv(os.path.join(PROC, "phenotypes_long.csv"), index=False)
qc = pd.DataFrame(qc_rows).sort_values(["n_acc"], ascending=False)
qc.to_csv(os.path.join(OUT, "results", "trait_qc.csv"), index=False)
print("traits:", len(qc), "usable (n>=200):", int((qc.n_acc >= 200).sum()),
      "rows:", len(long))
print(qc[qc.n_acc >= 250].head(40)[["study_id", "trait", "n_acc", "n_rep_mean", "mean", "sd"]].to_string(index=False))
