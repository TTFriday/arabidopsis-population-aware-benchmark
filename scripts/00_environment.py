"""Capture software, hardware, input-file and parameter provenance for the run."""
import os, sys, json, platform, subprocess, datetime
import numpy as np
import scipy, pandas, sklearn, matplotlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import OUT, PROC

DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
KEYS = [
    "01_genotype_1001G/1001_SNP_MATRIX/imputed_snps_binary.hdf5",
    "01_genotype_1001G/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5",
    "02_phenotype_AraPheno/study_list.json",
    "02_phenotype_AraPheno/phenotype_list.json",
    "03_transcriptome_1001T/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz",
    "05_annotation_GO_KEGG/tair.gaf.gz",
    "05_annotation_GO_KEGG/go-basic.obo",
    "06_genome_TAIR10/Arabidopsis_thaliana.TAIR10.63.gff3.gz",
]

out = []
out.append(f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
out.append(f"platform: {platform.platform()}")
out.append(f"machine: {platform.machine()}  processor: {platform.processor()}")
out.append(f"python: {sys.version.split()[0]}  executable: {sys.executable}")
try:
    import psutil
    out.append(f"logical CPUs: {psutil.cpu_count()}  RAM: {psutil.virtual_memory().total/1e9:.1f} GB")
except Exception:
    pass

out.append("\n[packages]")
for name, mod in [("numpy", np), ("scipy", scipy), ("pandas", pandas),
                  ("scikit-learn", sklearn), ("matplotlib", matplotlib)]:
    out.append(f"{name}=={mod.__version__}")
for name in ["statsmodels", "h5py", "psutil"]:
    try:
        out.append(f"{name}=={__import__(name).__version__}")
    except Exception:
        out.append(f"{name}: not installed")

out.append("\n[input files]")
for k in KEYS:
    p = os.path.join(DATA, k)
    if os.path.exists(p):
        st = os.stat(p)
        out.append(f"{k}\t{st.st_size/1e6:.1f} MB\tmtime={datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}")
    else:
        out.append(f"{k}\tMISSING")

out.append("\n[analysis parameters]")
params = {
    "MAF_min": 0.05, "thinning_bp": 5000, "LD_window_snps": 200, "LD_r2_max": 0.5,
    "n_markers": 12846, "n_PCs": 50, "n_population_clusters": 6,
    "trait_min_n": 250, "trait_h2_min": 0.2, "trait_h2_max": 0.95,
    "expression_transform": "log2(count+1), gene-wise z-score", "n_genes": 23008,
    "KRR_alphas": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    "G_E_weight_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    "random_cv": "StratifiedKFold(5), population-stratified, seeds 1-2",
    "lopo_cv": "GroupKFold by population cluster, rare clusters merged at n<10",
    "inner_cv": "3-fold (grouped when outer CV is LOPO)",
    "primary_metric": "Pearson r of out-of-fold predictions",
    "label_permutations": 300, "kernel_permutations": 100,
}
out.append(json.dumps(params, indent=2))

commands = [
    "python scripts/00_environment.py",
    "python scripts/01_qc.py",
    "python scripts/02_build_genotype.py",
    "python scripts/03_phenotypes.py",
    "python scripts/04_heritability.py",
    "python scripts/05_select_traits.py",
    "python scripts/06_expression.py",
    "python scripts/07_populations.py",
    "python scripts/08_finalize_traits.py",
    "python scripts/09_prediction.py --repeats 2 --tag full",
    "python scripts/09_prediction.py --fullpanel 1 --models kridge --features G --cvs random,lopo --repeats 2 --tag fullpanel",
    "python scripts/10_transfer.py",
    "python scripts/11_controls.py",
    "python scripts/12_enrichment.py",
    "python scripts/15_export_splits.py",
    "python scripts/13_analysis.py",
    "python scripts/14_figures.py",
    "python scripts/16_tables.py",
]
out.append("\n[commands]")
out.extend(commands)

os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
p = os.path.join(OUT, "results", "environment.txt")
with open(p, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
print("\nwrote", p)
