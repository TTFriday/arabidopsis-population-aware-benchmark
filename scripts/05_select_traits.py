"""Select benchmark traits: n>=250, 0.2<=h2<=0.95, expression overlap>=100, LD-independent."""
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(OUT, "data_processed")

h = pd.read_csv(os.path.join(OUT, "results", "heritability.csv"))
long = pd.read_csv(os.path.join(PROC, "phenotypes_long.csv"), dtype={"accession_id": str})
long = long[long.accession_id.isin(set(long.accession_id))]

# expression accessions
import gzip
with gzip.open(os.path.join(os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(OUT, "data")), "03_transcriptome_1001T/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz"), "rt") as f:
    hdr = f.readline().rstrip("\n").split("\t")
expr = set(x[1:] for x in hdr[1:])

cand = h[(h.n >= 250) & (h.h2 >= 0.2) & (h.h2 <= 0.95)].copy()
name_map = long.drop_duplicates(["study_id", "trait"]).set_index(["study_id", "trait"])["study_name"]
cand["study_name"] = [name_map.get((r.study_id, r.trait), "") for r in cand.itertuples()]

# expression overlap per trait
piv = long.pivot_table(index="accession_id", columns=["study_id", "trait"], values="value", aggfunc="mean")
cand["n_expr"] = [len(set(piv.index) & expr & set(long[(long.study_id == r.study_id) & (long.trait == r.trait)].accession_id)) for r in cand.itertuples()]
cand = cand[cand.n_expr >= 100].sort_values("h2", ascending=False)

# greedy within-study redundancy pruning at |r|>0.8
keep = []
for r in cand.itertuples():
    g = long[(long.study_id == r.study_id) & (long.trait == r.trait)].set_index("accession_id")["value"]
    redundant = False
    for k in keep:
        if k.study_id != r.study_id:
            continue
        gk = long[(long.study_id == k.study_id) & (long.trait == k.trait)].set_index("accession_id")["value"]
        common = g.index.intersection(gk.index)
        if len(common) > 50 and abs(np.corrcoef(g.loc[common], gk.loc[common])[0, 1]) > 0.8:
            redundant = True; break
    if not redundant:
        keep.append(r)

sel = pd.DataFrame([{ "study_id": r.study_id, "study_name": r.study_name, "trait": r.trait,
                      "n": r.n, "n_expr": r.n_expr, "h2": r.h2, "mean": r.mean, "sd": r.sd,
                      "skew": r.skew} for r in keep])
sel.to_csv(os.path.join(OUT, "results", "selected_traits.csv"), index=False)

# trait group annotation
groups = {12: "flowering", 38: "flowering/architecture", 34: "fitness", 31: "growth allometry",
          43: "stomata/gas exchange", 126: "trichome/metal", 127: "seed chemistry", 37: "pollination",
          2: "flowering/size", 4: "defense", 36: "root architecture", 1: "classic GWAS"}
sel["group"] = sel.study_id.map(groups).fillna("other")
print(sel.to_string(index=False))
print("\nselected:", len(sel), "studies:", sel.study_id.nunique(), "h2 range:", sel.h2.min().round(2), sel.h2.max().round(2))
print(sel.group.value_counts().to_dict())
