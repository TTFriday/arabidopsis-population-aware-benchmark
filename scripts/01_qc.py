"""Data QC: align 1001G genotypes, AraPheno phenotypes, 1001T expression."""
import json, os, glob, gzip
import numpy as np
import pandas as pd

DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)

# ---- SNP accessions ----
import h5py
snp_path = os.path.join(DATA, "01_genotype_1001G/1001_SNP_MATRIX/imputed_snps_binary.hdf5")
with h5py.File(snp_path, "r") as f:
    snp_acc = [x.decode() for x in f["accessions"][:]]
print("SNP accessions:", len(snp_acc), "unique:", len(set(snp_acc)))

# ---- expression accessions ----
expr_path = os.path.join(DATA, "03_transcriptome_1001T/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz")
with gzip.open(expr_path, "rt") as f:
    header = f.readline().rstrip("\n").split("\t")
expr_acc = [h[1:] for h in header[1:] if h.startswith("X")]
print("expression columns:", len(header) - 1, "parsed IDs:", len(expr_acc))

# ---- AraPheno ----
ap = os.path.join(DATA, "02_phenotype_AraPheno")
studies = json.load(open(os.path.join(ap, "study_list.json"), encoding="utf-8"))
phenos = json.load(open(os.path.join(ap, "phenotype_list.json"), encoding="utf-8"))
acc_meta = json.load(open(os.path.join(ap, "accession_list.json"), encoding="utf-8"))
acc2name = {str(a["pk"]): a["name"] for a in acc_meta}

snp_set, expr_set = set(snp_acc), set(expr_acc)
rows = []
for st in studies:
    sid = st["id"]
    path = os.path.join(ap, "study_values", f"study_{sid:03d}.csv")
    if not os.path.exists(path):
        print("missing", path); continue
    df = pd.read_csv(path, encoding="utf-8")
    traits = [c for c in df.columns if c not in ("accession_id", "replicate_id")]
    clim = [c for c in traits if c.startswith("clim-")]
    real = [c for c in traits if not c.startswith("clim-")]
    accs = set(df["accession_id"].astype(str))
    rows.append({
        "study_id": sid,
        "study_name": st["name"],
        "n_rows": len(df),
        "n_accessions": len(accs),
        "n_traits_total": len(traits),
        "n_traits_clim": len(clim),
        "n_traits_real": len(real),
        "n_acc_snp": len(accs & snp_set),
        "n_acc_snp_expr": len(accs & snp_set & expr_set),
        "traits": ";".join(real[:60]),
    })
summ = pd.DataFrame(rows).sort_values("n_acc_snp_expr", ascending=False)
summ.to_csv(os.path.join(OUT, "results", "study_overview.csv"), index=False, encoding="utf-8")
print(summ[["study_id", "n_accessions", "n_traits_real", "n_acc_snp", "n_acc_snp_expr"]].to_string(index=False))

# ---- master ID table ----
all_acc = sorted(set(snp_acc) | expr_set, key=lambda x: int(x) if x.isdigit() else 10**9)
idtab = pd.DataFrame({"accession_id": all_acc})
idtab["in_snp"] = idtab["accession_id"].isin(snp_set)
idtab["in_expr"] = idtab["accession_id"].isin(expr_set)
idtab["name"] = idtab["accession_id"].map(acc2name)
idtab.to_csv(os.path.join(OUT, "results", "id_table.csv"), index=False, encoding="utf-8")
print("ID table:", len(idtab), "SNP+expr:", int((idtab.in_snp & idtab.in_expr).sum()))

meta = {
    "n_snp_acc": len(snp_acc),
    "n_expr_acc": len(expr_acc),
    "n_snp_expr": len(snp_set & expr_set),
    "n_studies": len(studies),
    "n_phenotypes": len(phenos),
}
json.dump(meta, open(os.path.join(OUT, "results", "qc_meta.json"), "w"), indent=2)
print(json.dumps(meta))
